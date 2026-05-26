"""
Candidate Set Builder for routing loss.

Builds candidate sets per example containing:
- 1 positive model (at index 0)
- K_semantic semantic negatives from same domain
- K_far far negatives from other domains
- K_hard hard negatives from mining cache

Ensures deterministic fallback handling for sparse domains.
"""

from typing import List, Dict, Tuple, Optional, Set
import random
from .model_registry import ModelRegistry
from ..utils_carve.router_constants import (
    DEFAULT_K_TOTAL,
    DEFAULT_K_SEMANTIC,
    DEFAULT_K_FAR,
    DEFAULT_K_HARD,
    SEMANTIC_POOL_MODE_PARENT_GROUP,
)
from ..utils_carve.router_exceptions import CandidateValidationError


class CandidateSetBuilder:
    """
    Builds candidate sets for routing loss with positive + semantic + far + hard negatives.
    
    Guarantees:
    - Output always has exactly K_total candidates
    - Positive model is always at index 0
    - All candidates are unique
    - Deterministic fallback for sparse domains
    
    Usage:
        builder = CandidateSetBuilder(
            registry=registry,
            K_total=64,
            K_semantic=48,
            K_far=8,
            K_hard=7,
        )
        candidates = builder.build(
            y_idx=model_idx,
            domain="cv",
            hard_negative_cache=cache
        )
    """
    
    def __init__(
        self,
        registry: ModelRegistry,
        K_total: int = DEFAULT_K_TOTAL,
        K_semantic: int = DEFAULT_K_SEMANTIC,
        K_far: int = DEFAULT_K_FAR,
        K_hard: int = DEFAULT_K_HARD,
        random_seed: Optional[int] = None,
        use_legacy_batching: bool = False,
        semantic_pool_mode: str = SEMANTIC_POOL_MODE_PARENT_GROUP,
        semantic_pool_max_domains: Optional[int] = None,
        semantic_pool_depth: int = 1,
    ):
        """
        Initialize candidate set builder.
        
        Args:
            registry: ModelRegistry instance
            K_total: Total number of candidates (including positive)
            K_semantic: Target number of semantic negatives (same domain)
            K_far: Target number of far negatives (other domains)
            K_hard: Target number of hard negatives from cache
            random_seed: Optional seed for reproducibility
            semantic_pool_mode: Mode for semantic pool expansion (Option B)
                - "domain_only": Only exact domain (original behavior)
                - "parent_group": Expand to related domains via parent group
                - "taxonomy_graph": Use explicit taxonomy graph (future)
            semantic_pool_max_domains: Maximum number of related domains to include
            semantic_pool_depth: Graph traversal depth (for taxonomy_graph mode)
        """
        if K_total <= 0:
            raise ValueError(f"K_total must be positive, got {K_total}")
        if K_semantic + K_far + K_hard + 1 > K_total * 1.5:
            raise ValueError(
                f"K_semantic ({K_semantic}) + K_far ({K_far}) + K_hard ({K_hard}) + 1 "
                f"should be approximately K_total ({K_total}), but sum exceeds 1.5 * K_total. "
                f"This may cause candidate set construction issues."
            )
        
        self.registry = registry
        self.K_total = K_total
        self.K_semantic = K_semantic
        self.K_far = K_far
        self.K_hard = K_hard
        self.use_legacy_batching = use_legacy_batching
        self.semantic_pool_mode = semantic_pool_mode
        self.semantic_pool_max_domains = semantic_pool_max_domains
        self.semantic_pool_depth = semantic_pool_depth
        self._semantic_pool_cache: Dict[str, List[int]] = {}
        self._far_pool_cache: Dict[str, List[int]] = {}
        self._all_model_indices: List[int] = list(range(len(self.registry)))
        
        if random_seed is not None:
            random.seed(random_seed)
    
    def build(
        self,
        y_idx: int,
        domain: str,
        hard_negative_cache: Optional[Dict[Tuple[str, int], List[int]]] = None,
    ) -> List[int]:
        """
        Build candidate set for a single example.
        
        Args:
            y_idx: Positive model index
            domain: Sample domain (will be canonicalized)
            hard_negative_cache: Dict[(domain, y_idx)] -> List[hard_neg_idx]
        
        Returns:
            List of up to K_total unique model indices with y_idx at index 0.
            If registry has fewer than K_total models, returns all available models.
        """
        if y_idx < 0 or y_idx >= len(self.registry):
            raise CandidateValidationError(
                f"Invalid model index {y_idx}. Valid range: [0, {len(self.registry)}). "
                f"This indicates the gold model is not registered in the model registry."
            )

        if self.use_legacy_batching:
            return self._build_legacy_exact(y_idx=y_idx, domain=domain, hard_negative_cache=hard_negative_cache)
        
        # Canonicalize domain
        domain = domain.strip().lower() if isinstance(domain, str) else "unknown"
        
        chosen: Set[int] = {y_idx}
        candidates: List[int] = [y_idx]  # Positive always at index 0
        
        # ============================================================
        # Step 1: Add hard negatives from cache
        # ============================================================
        hard_negs_added = 0
        if hard_negative_cache:
            cache_key = (domain, y_idx)
            hard_negs = hard_negative_cache.get(cache_key, [])
            
            for hard_idx in hard_negs:
                if hard_idx not in chosen and hard_negs_added < self.K_hard:
                    candidates.append(hard_idx)
                    chosen.add(hard_idx)
                    hard_negs_added += 1
        
        # ============================================================
        # Step 2: Add semantic negatives from expanded semantic pool
        # ============================================================
        semantic_negs_added = 0
        semantic_target = self.K_semantic
        
        if self.use_legacy_batching:
            semantic_pool = self.registry.get_semantic_pool(
                domain=domain,
                mode=self.semantic_pool_mode,
                depth=self.semantic_pool_depth,
                max_domains=self.semantic_pool_max_domains,
                exclude=chosen,
            )
        else:
            # Optimized mode: cache unfiltered pool and apply per-example excludes locally.
            if domain not in self._semantic_pool_cache:
                self._semantic_pool_cache[domain] = self.registry.get_semantic_pool(
                    domain=domain,
                    mode=self.semantic_pool_mode,
                    depth=self.semantic_pool_depth,
                    max_domains=self.semantic_pool_max_domains,
                    exclude=set(),
                )
            semantic_pool = [idx for idx in self._semantic_pool_cache[domain] if idx not in chosen]
        
        # Handle sparse pools: if not enough models in pool, reduce target
        available_semantic = len(semantic_pool)
        if available_semantic < semantic_target:
            semantic_target = available_semantic
        
        # Sample semantic negatives
        if semantic_pool:
            if self.use_legacy_batching:
                random.shuffle(semantic_pool)
                selected_semantic = semantic_pool[:semantic_target]
            else:
                selected_semantic = (
                    random.sample(semantic_pool, semantic_target)
                    if len(semantic_pool) > semantic_target
                    else semantic_pool
                )
            for idx in selected_semantic:
                if idx not in chosen:
                    candidates.append(idx)
                    chosen.add(idx)
                    semantic_negs_added += 1
        
        # ============================================================
        # Step 3: Add far negatives from other domains
        # ============================================================
        far_negs_added = 0
        far_target = self.K_far
        semantic_shortfall = self.K_semantic - semantic_negs_added
        far_target += semantic_shortfall
        
        if self.use_legacy_batching:
            far_pool: List[int] = []
            other_domains = self.registry.get_other_domains(domain)
            for other_domain in other_domains:
                far_pool.extend(self.registry.get_domain_models(other_domain, exclude=chosen))
        else:
            if domain not in self._far_pool_cache:
                other_domains = self.registry.get_other_domains(domain)
                far_pool_unfiltered: List[int] = []
                for other_domain in other_domains:
                    far_pool_unfiltered.extend(self.registry.get_domain_models(other_domain, exclude=set()))
                self._far_pool_cache[domain] = far_pool_unfiltered
            far_pool = [idx for idx in self._far_pool_cache[domain] if idx not in chosen]
        
        if far_pool:
            if self.use_legacy_batching:
                random.shuffle(far_pool)
                selected_far = far_pool[:far_target]
            else:
                selected_far = (
                    random.sample(far_pool, far_target)
                    if len(far_pool) > far_target
                    else far_pool
                )
            for idx in selected_far:
                candidates.append(idx)
                chosen.add(idx)
                far_negs_added += 1
        
        # ============================================================
        # Step 4: Fill remaining slots with random models
        # ============================================================
        remaining_needed = self.K_total - len(candidates)
        
        if remaining_needed > 0:
            remaining_pool = [idx for idx in self._all_model_indices if idx not in chosen]
            if remaining_pool:
                if self.use_legacy_batching:
                    random.shuffle(remaining_pool)
                    selected_remaining = remaining_pool[:remaining_needed]
                else:
                    selected_remaining = (
                        random.sample(remaining_pool, remaining_needed)
                        if len(remaining_pool) > remaining_needed
                        else remaining_pool
                    )
                for idx in selected_remaining:
                    candidates.append(idx)
                    chosen.add(idx)
                    if len(candidates) >= self.K_total:
                        break
        
        # ============================================================
        # Step 5: Guarantee exactly K_total candidates (or fewer if registry too small)
        # ============================================================
        max_possible = len(self.registry)
        effective_K = min(self.K_total, max_possible)
        
        if len(candidates) > effective_K:
            candidates = candidates[:effective_K]
        elif len(candidates) < effective_K:
            remaining = [idx for idx in self._all_model_indices if idx not in chosen]
            needed = effective_K - len(candidates)
            if remaining:
                extra = random.sample(remaining, needed) if len(remaining) > needed else remaining
                candidates.extend(extra)
        
        # ============================================================
        # Final validation
        # ============================================================
        if len(candidates) > self.K_total:
            raise CandidateValidationError(
                f"Expected at most {self.K_total} candidates, got {len(candidates)}. "
                f"This indicates a bug in candidate set construction."
            )
        if len(candidates) != len(set(candidates)):
            duplicates = [c for c in candidates if candidates.count(c) > 1]
            raise CandidateValidationError(
                f"Candidates should be unique, got {len(candidates)} total but {len(set(candidates))} unique. "
                f"Duplicates: {duplicates}"
            )
        if candidates[0] != y_idx:
            raise CandidateValidationError(
                f"Positive model should be at index 0, got {candidates[0]} instead of {y_idx}. "
                f"This is a critical error - the positive model must be at position 0."
            )
        
        return candidates

    def _build_legacy_exact(
        self,
        y_idx: int,
        domain: str,
        hard_negative_cache: Optional[Dict[Tuple[str, int], List[int]]] = None,
    ) -> List[int]:
        """Exact legacy candidate construction path from commit 31e0b417."""
        # Canonicalize domain
        domain = domain.strip().lower() if isinstance(domain, str) else "unknown"

        chosen: Set[int] = {y_idx}
        candidates: List[int] = [y_idx]  # Positive always at index 0

        hard_negs_added = 0
        if hard_negative_cache:
            cache_key = (domain, y_idx)
            hard_negs = hard_negative_cache.get(cache_key, [])
            for hard_idx in hard_negs:
                if hard_idx not in chosen and hard_negs_added < self.K_hard:
                    candidates.append(hard_idx)
                    chosen.add(hard_idx)
                    hard_negs_added += 1

        # Step 2: Add semantic negatives from expanded semantic pool
        semantic_negs_added = 0
        semantic_target = self.K_semantic
        semantic_pool = self.registry.get_semantic_pool(
            domain=domain,
            mode=self.semantic_pool_mode,
            depth=self.semantic_pool_depth,
            max_domains=self.semantic_pool_max_domains,
            exclude=chosen,
        )
        available_semantic = len(semantic_pool)
        if available_semantic < semantic_target:
            semantic_target = available_semantic
        if semantic_pool:
            random.shuffle(semantic_pool)
            for idx in semantic_pool[:semantic_target]:
                if idx not in chosen:
                    candidates.append(idx)
                    chosen.add(idx)
                    semantic_negs_added += 1

        # Step 3: Add far negatives from other domains
        far_negs_added = 0
        far_target = self.K_far
        semantic_shortfall = self.K_semantic - semantic_negs_added
        far_target += semantic_shortfall
        other_domains = self.registry.get_other_domains(domain)
        far_pool: List[int] = []
        for other_domain in other_domains:
            far_pool.extend(self.registry.get_domain_models(other_domain, exclude=chosen))
        if far_pool:
            random.shuffle(far_pool)
            for idx in far_pool:
                if idx not in chosen and far_negs_added < far_target:
                    candidates.append(idx)
                    chosen.add(idx)
                    far_negs_added += 1

        # Step 4: Fill remaining slots with random models
        remaining_needed = self.K_total - len(candidates)
        if remaining_needed > 0:
            all_models = list(range(len(self.registry)))
            random.shuffle(all_models)
            for idx in all_models:
                if idx not in chosen:
                    candidates.append(idx)
                    chosen.add(idx)
                    if len(candidates) >= self.K_total:
                        break

        # Step 5: Guarantee exactly K_total candidates (or fewer if registry too small)
        max_possible = len(self.registry)
        effective_K = min(self.K_total, max_possible)
        if len(candidates) > effective_K:
            candidates = candidates[:effective_K]
        elif len(candidates) < effective_K:
            remaining = [idx for idx in range(max_possible) if idx not in chosen]
            random.shuffle(remaining)
            candidates.extend(remaining[:effective_K - len(candidates)])

        if len(candidates) > self.K_total:
            raise CandidateValidationError(
                f"Expected at most {self.K_total} candidates, got {len(candidates)}. "
                f"This indicates a bug in candidate set construction."
            )
        if len(candidates) != len(set(candidates)):
            duplicates = [c for c in candidates if candidates.count(c) > 1]
            raise CandidateValidationError(
                f"Candidates should be unique, got {len(candidates)} total but {len(set(candidates))} unique. "
                f"Duplicates: {duplicates}"
            )
        if candidates[0] != y_idx:
            raise CandidateValidationError(
                f"Positive model should be at index 0, got {candidates[0]} instead of {y_idx}. "
                f"This is a critical error - the positive model must be at position 0."
            )
        return candidates
    
    def build_batch(
        self,
        y_indices: List[int],
        domains: List[str],
        hard_negative_cache: Optional[Dict[Tuple[str, int], List[int]]] = None,
    ) -> List[List[int]]:
        """
        Build candidate sets for a batch of examples.
        
        Args:
            y_indices: List of positive model indices
            domains: List of domains (same length as y_indices)
            hard_negative_cache: Optional hard negative cache
        
        Returns:
            List of candidate lists, one per example
        """
        if len(y_indices) != len(domains):
            raise ValueError(
                f"y_indices and domains must have same length, "
                f"got {len(y_indices)} and {len(domains)} respectively"
            )
        
        batch_candidates = []
        for y_idx, domain in zip(y_indices, domains):
            candidates = self.build(y_idx, domain, hard_negative_cache)
            batch_candidates.append(candidates)
        
        return batch_candidates
    
    def get_composition_stats(
        self,
        candidates: List[int],
        y_idx: int,
        domain: str,
        hard_negative_cache: Optional[Dict[Tuple[str, int], List[int]]] = None,
    ) -> Dict[str, int]:
        """
        Analyze composition of a candidate set (for logging/debugging).
        
        Args:
            candidates: Candidate list to analyze
            y_idx: Positive model index
            domain: Domain of the example
            hard_negative_cache: Hard negative cache
        
        Returns:
            Dict with counts: {
                "positive": 1,
                "hard": num_hard,
                "semantic": num_semantic,
                "far": num_far,
                "other": num_other
            }
        """
        domain = domain.strip().lower() if isinstance(domain, str) else "unknown"
        
        hard_negs = set()
        if hard_negative_cache:
            cache_key = (domain, y_idx)
            hard_negs = set(hard_negative_cache.get(cache_key, []))
        
        semantic_pool = set(self.registry.get_semantic_pool(
            domain=domain,
            mode=self.semantic_pool_mode,
            depth=self.semantic_pool_depth,
            max_domains=self.semantic_pool_max_domains,
            exclude={y_idx},
        ))
        
        stats = {
            "positive": 0,
            "hard": 0,
            "semantic": 0,
            "far": 0,
            "other": 0,
        }
        
        for idx in candidates:
            if idx == y_idx:
                stats["positive"] += 1
            elif idx in hard_negs:
                stats["hard"] += 1
            elif idx in semantic_pool:
                stats["semantic"] += 1
            else:
                model_domain = self.registry.metadata.get(idx, {}).get("domain", "unknown")
                if model_domain != domain:
                    stats["far"] += 1
                else:
                    stats["other"] += 1
        
        return stats

