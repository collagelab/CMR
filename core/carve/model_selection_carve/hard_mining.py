"""
Hard Negative Mining for routing loss.

Periodically mines confusable models within semantic pools (same domain)
to populate a cache of hard negatives for candidate sampling.

Key features:
- Runs under torch.no_grad() (no gradient retention)
- Scores over semantic pool larger than K_total (256-1024 models)
- Uses cached model embeddings (no full forward pass)
- Stores top K_hard_pool confusable models per (domain, model_idx)
"""

from typing import Dict, List, Tuple, Optional, Any
import random
import torch
import torch.nn as nn
from .model_registry import ModelRegistry
from ..utils_carve.router_constants import (
    DEFAULT_K_HARD_POOL,
    DEFAULT_SEMANTIC_POOL_SIZE,
    DEFAULT_MAX_POOL_SIZE,
    DEFAULT_MAX_EXAMPLES_PER_UPDATE,
    SEMANTIC_POOL_MODE_PARENT_GROUP,
)
from ..utils_carve.router_exceptions import HardNegativeMiningError


class HardNegativeMiner:
    """
    Mines hard negatives by scoring confusable models within semantic pools.
    
    Usage:
        miner = HardNegativeMiner(
            registry=registry,
            K_hard_pool=20,
            semantic_pool_size=512,
        )
        
        # Periodically (e.g., every 200 steps):
        miner.update_cache(
            batch_examples=examples,
            router_model=router,
            max_examples=128,
        )
    """
    
    def __init__(
        self,
        registry: ModelRegistry,
        K_hard_pool: int = DEFAULT_K_HARD_POOL,
        semantic_pool_size: int = DEFAULT_SEMANTIC_POOL_SIZE,
        max_pool_size: int = DEFAULT_MAX_POOL_SIZE,
        semantic_pool_mode: str = SEMANTIC_POOL_MODE_PARENT_GROUP,
        semantic_pool_max_domains: Optional[int] = None,
        semantic_pool_depth: int = 1,
    ):
        """
        Initialize hard negative miner.
        
        Args:
            registry: ModelRegistry instance
            K_hard_pool: Number of hard negatives to store per (domain, model_idx)
            semantic_pool_size: Target size for semantic pool sampling
            max_pool_size: Maximum semantic pool size (cap for large domains)
            semantic_pool_mode: Mode for semantic pool expansion (Option B)
                - "domain_only": Only exact domain (original behavior)
                - "parent_group": Expand to related domains via parent group
                - "taxonomy_graph": Use explicit taxonomy graph (future)
            semantic_pool_max_domains: Maximum number of related domains to include
            semantic_pool_depth: Graph traversal depth (for taxonomy_graph mode)
        
        Raises:
            ValueError: If parameters are invalid
        """
        if K_hard_pool <= 0:
            raise ValueError(f"K_hard_pool must be positive, got {K_hard_pool}")
        if semantic_pool_size <= 0:
            raise ValueError(f"semantic_pool_size must be positive, got {semantic_pool_size}")
        if max_pool_size <= 0:
            raise ValueError(f"max_pool_size must be positive, got {max_pool_size}")
        if K_hard_pool > max_pool_size:
            raise ValueError(
                f"K_hard_pool ({K_hard_pool}) cannot exceed max_pool_size ({max_pool_size})"
            )
        
        self.registry = registry
        self.K_hard_pool = K_hard_pool
        self.semantic_pool_size = semantic_pool_size
        self.max_pool_size = max_pool_size
        self.semantic_pool_mode = semantic_pool_mode
        self.semantic_pool_max_domains = semantic_pool_max_domains
        self.semantic_pool_depth = semantic_pool_depth
        self.cache: Dict[Tuple[str, int], List[int]] = {}
        self.num_updates = 0
        self.num_examples_processed = 0
    
    @torch.no_grad()
    def update_cache(
        self,
        batch_examples: List[Dict[str, Any]],
        router_model: nn.Module,
        prompt_embeddings: Optional[torch.Tensor] = None,
        max_examples: int = DEFAULT_MAX_EXAMPLES_PER_UPDATE,
    ):
        """
        Update hard negative cache for a batch of examples.
        
        Runs under torch.no_grad() to avoid gradient retention.
        
        Args:
            batch_examples: List of examples with 'model_idx', 'domain', 'prompt_embedding'
            router_model: Router model with model_embeddings attribute
            prompt_embeddings: Optional pre-computed prompt embeddings [B, D]
            max_examples: Maximum examples to process per update (for efficiency)
        """
        if not batch_examples:
            return
        
        if max_examples <= 0:
            raise ValueError(f"max_examples must be positive, got {max_examples}")
        examples_to_process = batch_examples[:max_examples]
        
        try:
            device = next(router_model.parameters()).device
        except StopIteration:
            raise HardNegativeMiningError(
                "Router model has no parameters. Cannot determine device for mining."
            )
        
        for i, example in enumerate(examples_to_process):
            y_idx = example.get('model_idx')
            domain = example.get('domain', 'unknown')
            
            if y_idx is None:
                continue
            
            domain = domain.strip().lower() if isinstance(domain, str) else "unknown"
            semantic_pool = self.registry.get_semantic_pool(
                domain=domain,
                mode=self.semantic_pool_mode,
                depth=self.semantic_pool_depth,
                max_domains=self.semantic_pool_max_domains,
                exclude={y_idx},
            )
            
            if len(semantic_pool) < 2:
                continue
            
            if len(semantic_pool) > self.max_pool_size:
                semantic_pool = random.sample(semantic_pool, self.max_pool_size)
            
            if prompt_embeddings is not None and i < len(prompt_embeddings):
                prompt_emb = prompt_embeddings[i]  # [D]
            elif 'prompt_embedding' in example:
                prompt_emb = example['prompt_embedding']
                if isinstance(prompt_emb, torch.Tensor):
                    prompt_emb = prompt_emb.to(device)
                else:
                    continue
            else:
                continue
            
            if prompt_emb.dim() > 1:
                prompt_emb = prompt_emb.squeeze()
            
            try:
                pool_indices = torch.tensor(semantic_pool, dtype=torch.long, device=device)
                pool_embeddings = router_model.model_embeddings(pool_indices)  # [N, D]
            except (AttributeError, KeyError) as e:
                raise HardNegativeMiningError(
                    f"Failed to get model embeddings: {e}. "
                    f"Router model must have 'model_embeddings' attribute."
                )
            
            if prompt_emb.shape[0] != pool_embeddings.shape[1]:
                raise HardNegativeMiningError(
                    f"Dimension mismatch: prompt_emb has dim {prompt_emb.shape[0]}, "
                    f"but model embeddings have dim {pool_embeddings.shape[1]}"
                )
            
            scores = torch.matmul(prompt_emb, pool_embeddings.T)  # [N]
            
            if len(scores) <= self.K_hard_pool:
                hard_neg_indices = list(range(len(semantic_pool)))
            else:
                _, top_indices = torch.topk(scores, k=self.K_hard_pool, largest=True)
                hard_neg_indices = top_indices.cpu().tolist()
            
            hard_negs = [semantic_pool[idx] for idx in hard_neg_indices]
            
            if len(hard_negs) != min(self.K_hard_pool, len(semantic_pool)):
                raise HardNegativeMiningError(
                    f"Expected {min(self.K_hard_pool, len(semantic_pool))} hard negatives, "
                    f"but got {len(hard_negs)}"
                )
            
            cache_key = (domain, y_idx)
            self.cache[cache_key] = hard_negs
            
            self.num_examples_processed += 1
        
        self.num_updates += 1
    
    def get_cache_size(self) -> int:
        """Get number of entries in cache."""
        return len(self.cache)
    
    def get_hit_rate(
        self,
        batch_examples: List[Dict[str, Any]],
    ) -> float:
        """
        Compute cache hit rate for a batch of examples.
        
        Args:
            batch_examples: List of examples with 'model_idx' and 'domain'
        
        Returns:
            Fraction of examples with cache hits (0.0 to 1.0)
        """
        if not batch_examples:
            return 0.0
        
        hits = 0
        for example in batch_examples:
            y_idx = example.get('model_idx')
            domain = example.get('domain', 'unknown')
            
            if y_idx is None:
                continue
            
            domain = domain.strip().lower() if isinstance(domain, str) else "unknown"
            
            cache_key = (domain, y_idx)
            if cache_key in self.cache:
                hits += 1
        
        return hits / len(batch_examples)
    
    def clear_cache(self):
        """Clear the hard negative cache."""
        self.cache.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get mining statistics."""
        return {
            "num_updates": self.num_updates,
            "num_examples_processed": self.num_examples_processed,
            "cache_size": len(self.cache),
        }

