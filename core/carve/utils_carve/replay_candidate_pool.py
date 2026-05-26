

import random
from typing import List, Dict, Optional, Tuple, Any, Set
from collections import defaultdict
from dataclasses import dataclass
import re
from .router_similarity_utils import canonicalize_domain
from .retrieval_replay import normalize_text

try:
    from .prompt_similarity_adapter import PromptSimilarityRetrieverAdapter, RetrievedItem
except ImportError:
    PromptSimilarityRetrieverAdapter = None
    RetrievedItem = None


def extract_instruction_only(text: str) -> str:
    """Extract only the ###Instruction part from prompts that may contain ###Input."""
    if not text:
        return ""
    text = str(text).strip()
    
    # If it contains ###Instruction, extract just that part
    if "###Instruction:" in text or "###Instruction " in text:
        # Find the instruction marker (case insensitive)
        import re
        # Match ###Instruction: or ###Instruction followed by content
        match = re.search(r'###Instruction:?\s*(.+?)(?=###Input:|###|$)', text, re.IGNORECASE | re.DOTALL)
        if match:
            instruction = match.group(1).strip()
            return instruction
        # Fallback: split on ###Input if present
        if "###Input:" in text:
            parts = text.split("###Input:", 1)
            return parts[0].replace("###Instruction:", "").replace("###Instruction", "").strip()
        # If no ###Input, return everything after ###Instruction
        if "###Instruction:" in text:
            return text.split("###Instruction:", 1)[1].strip()
        if "###Instruction " in text:
            return text.split("###Instruction ", 1)[1].strip()
    
    # If no ###Instruction marker, return as-is (might be plain instruction)
    return text


@dataclass
class CandidateItem:
    """Represents a candidate item in the pool with metadata."""
    idx: int                    # Unique index in the pool
    prompt_text: str           # The prompt text
    domain: str                # Domain identifier
    model_name: str            # Model name/ID
    unique_id: Optional[str] = None  # Optional unique identifier
    
    def __hash__(self):
        return hash(self.idx)
    
    def __eq__(self, other):
        if isinstance(other, CandidateItem):
            return self.idx == other.idx
        return False


class ReplayCandidatePool:
    """
    A pool of replay candidates for X-CLR that enables sampling with
    guaranteed same-domain positives.
    
    This pool is built once per experience from the replay buffer examples
    and provides efficient sampling of candidates for X-CLR loss computation.
    
    Key features:
    - Stores candidates with domain/model metadata
    - Provides domain-aware sampling to guarantee positives
    - Falls back gracefully when same-domain pool is too small
    
    Usage:
        pool = ReplayCandidatePool.from_examples(replay_examples)
        candidates = pool.sample_candidates(
            anchor_domain="cv",
            anchor_model="resnet",
            n_candidates=31,
            min_pos=2,
            exclude_idx=5  # exclude the anchor itself
        )
    """
    
    def __init__(self):
        """Initialize an empty candidate pool."""
        self._candidates: List[CandidateItem] = []
        self._by_domain: Dict[str, List[CandidateItem]] = defaultdict(list)
        self._by_model: Dict[str, List[CandidateItem]] = defaultdict(list)
        self._by_domain_model: Dict[Tuple[str, str], List[CandidateItem]] = defaultdict(list)
        self._all_domains: Set[str] = set()
        self._all_models: Set[str] = set()
        self._similarity_retriever: Optional[PromptSimilarityRetrieverAdapter] = None
        
    def __len__(self) -> int:
        return len(self._candidates)
    
    @classmethod
    def from_examples(
        cls,
        examples: List[Dict[str, Any]],
        prompt_key: str = "prompt",
        domain_key: str = "domain", 
        model_key: str = "model_name",
        id_key: Optional[str] = None,
    ) -> "ReplayCandidatePool":
        """
        Build a candidate pool from a list of example dictionaries.
        
        Args:
            examples: List of example dicts containing prompt, domain, model_name
            prompt_key: Key for prompt text in examples
            domain_key: Key for domain in examples
            model_key: Key for model_name in examples
            id_key: Optional key for unique ID (to detect and exclude anchor)
            
        Returns:
            A populated ReplayCandidatePool instance
        """
        pool = cls()
        
        for idx, example in enumerate(examples):
            # Extract fields with fallbacks
            # Try multiple keys for prompt_text (it's critical - retrieval depends on it)
            raw_prompt = example.get(prompt_key)
            if not raw_prompt:
                # Try common alternative keys
                raw_prompt = example.get("instruction") or example.get("input") or example.get("query") or ""
            if not raw_prompt or not isinstance(raw_prompt, str):
                raw_prompt = str(raw_prompt) if raw_prompt else ""
            
            # Extract only the ###Instruction part
            prompt_text = extract_instruction_only(raw_prompt)
            
            if not prompt_text.strip():
                # If still empty, this is a problem - log warning but continue
                import warnings
                warnings.warn(
                    f"Example {idx} has no prompt_text after extraction (tried keys: {prompt_key}, 'instruction', 'input', 'query'). "
                    f"Raw prompt: {raw_prompt[:100] if raw_prompt else 'EMPTY'}. "
                    f"Available keys: {list(example.keys())[:10]}"
                )
            
            domain = example.get(domain_key, "unknown")
            model_name = example.get(model_key, "unknown")
            
            # Extract unique_id - try multiple keys and generate if missing
            unique_id = None
            if id_key:
                unique_id = example.get(id_key)
            # Try common alternative keys
            if not unique_id:
                unique_id = example.get("unique_id") or example.get("id") or example.get("example_id")
            # Generate stable unique_id from index if still missing
            if not unique_id:
                unique_id = f"pool_idx_{idx}"
            
            # Handle alternative key names
            if not domain and "domain_name" in example:
                domain = example["domain_name"]
            if not model_name and "model_id" in example:
                model_name = example["model_id"]
            
            # Create candidate item with canonicalized domain
            item = CandidateItem(
                idx=idx,
                prompt_text=prompt_text,
                domain=canonicalize_domain(domain),
                model_name=str(model_name) if model_name else "unknown",
                unique_id=str(unique_id),
            )
            
            pool.add_candidate(item)
        
        return pool
    
    def add_candidate(self, item: CandidateItem):
        """Add a candidate item to the pool."""
        self._candidates.append(item)
        self._by_domain[item.domain].append(item)
        self._by_model[item.model_name].append(item)
        self._by_domain_model[(item.domain, item.model_name)].append(item)
        self._all_domains.add(item.domain)
        self._all_models.add(item.model_name)
    
    def get_domain_counts(self) -> Dict[str, int]:
        """Get counts of candidates per domain."""
        return {d: len(items) for d, items in self._by_domain.items()}
    
    def get_model_counts(self) -> Dict[str, int]:
        """Get counts of candidates per model."""
        return {m: len(items) for m, items in self._by_model.items()}
    
    def initialize_similarity_retriever(
        self,
        retriever_type: str = "sentence_transformer",
        device: Optional[str] = None,
        all_prompts_corpus: Optional[List[Dict[str, Any]]] = None,
    ):
        """
        Initialize the prompt similarity retriever for prompt_similarity sampling mode.
        
        Args:
            retriever_type: Type of retriever ("bm25", "sentence_transformer", "splade", "flagembedding")
            device: Device for GPU-based retrievers
            all_prompts_corpus: Optional list of all training examples to retrieve over.
                               If provided, retrieval will search over all prompts, not just replay pool.
                               Each dict should have: prompt/instruction, domain, model_name, unique_id
                               If None, only retrieves over replay pool candidates.
        """
        if PromptSimilarityRetrieverAdapter is None:
            raise ImportError("PromptSimilarityRetrieverAdapter not available. Install required dependencies.")
        
        # Build lists for retriever
        # Ensure all candidates have non-empty prompt_text (required for retrieval)
        prompt_texts = []
        unique_ids = []
        domains = []
        model_names = []
        
        # Track both unique_ids and normalized prompt texts to avoid duplicates
        added_unique_ids = set()
        added_normalized_texts = set()
        
        for c in self._candidates:
            prompt = c.prompt_text or ""
            if not prompt.strip():
                # Skip candidates without prompt_text - they can't be retrieved
                import warnings
                warnings.warn(
                    f"Skipping candidate {c.idx} (unique_id={c.unique_id}) in retriever initialization: "
                    f"no prompt_text available"
                )
                continue
            
            # Check for duplicate normalized prompt text (text-level deduplication)
            normalized_prompt = normalize_text(prompt)
            if normalized_prompt in added_normalized_texts:
                # Skip if we've already added a candidate with the same normalized prompt text
                continue
            
            prompt_texts.append(prompt)
            unique_id = c.unique_id if c.unique_id else f"idx_{c.idx}"
            unique_ids.append(unique_id)
            domains.append(c.domain)
            model_names.append(c.model_name)
            added_unique_ids.add(unique_id)
            added_normalized_texts.add(normalized_prompt)
        
        if all_prompts_corpus:
            # Extract prompts from all_prompts_corpus, skipping duplicates
            for example in all_prompts_corpus:
                # Try multiple keys for prompt_text
                raw_prompt = example.get("prompt") or example.get("instruction") or example.get("input") or ""
                if not raw_prompt or not isinstance(raw_prompt, str):
                    raw_prompt = str(raw_prompt) if raw_prompt else ""
                
                # Extract only the ###Instruction part
                prompt = extract_instruction_only(raw_prompt)
                
                if not prompt.strip():
                    continue
                
                # Extract metadata
                uid = example.get("unique_id") or example.get("id") or example.get("example_id")
                if not uid:
                    # Generate a unique ID if missing
                    uid = f"all_corpus_{len(prompt_texts)}"
                
                uid_str = str(uid)
                
                # Skip if this unique_id is already in the pool (prefer pool version)
                if uid_str in added_unique_ids:
                    continue
                
                # Skip if this normalized prompt text is already in the index (text-level deduplication)
                normalized_prompt = normalize_text(prompt)
                if normalized_prompt in added_normalized_texts:
                    continue
                
                domain = example.get("domain", "unknown")
                model_name = example.get("model_name") or example.get("model_id", "unknown")
                
                # Add to lists (now deduplicated by both ID and text)
                prompt_texts.append(prompt)
                unique_ids.append(uid_str)
                domains.append(domain)
                model_names.append(str(model_name))
                added_unique_ids.add(uid_str)
                added_normalized_texts.add(normalized_prompt)
        
        if not prompt_texts:
            raise ValueError(
                "No candidates with valid prompt_text found. Cannot initialize similarity retriever. "
                "Ensure candidates have prompt_text when building the pool."
            )
        
        # Initialize and fit retriever
        self._similarity_retriever = PromptSimilarityRetrieverAdapter(
            retriever_type=retriever_type,
            device=device,
        )
        self._similarity_retriever.fit(
            prompt_texts=prompt_texts,
            unique_ids=unique_ids,
            domains=domains,
            model_names=model_names,
        )
    
    def sample_candidates(
        self,
        anchor_domain: str,
        anchor_model: str,
        n_candidates: int,
        min_pos: int = 1,
        exclude_idx: Optional[int] = None,
        exclude_unique_id: Optional[str] = None,
        rng: Optional[random.Random] = None,
        sampling_strategy: str = "domain",
        anchor_prompt_text: Optional[str] = None,
    ) -> Tuple[List[CandidateItem], Dict[str, Any]]:
        """
        Sample candidates for an anchor, trying to guarantee min_pos same-domain positives.
        
        The sampling strategy depends on `sampling_strategy`:
        - "domain": Domain-based sampling (existing behavior)
            1. Sample min_pos candidates from same domain (excluding anchor)
            2. Fill remaining slots with random candidates from other domains/models
            3. If same-domain pool is too small, use whatever is available
        - "prompt_similarity": Prompt similarity-based sampling
            1. Query similarity retriever with anchor prompt
            2. Select top n_candidates candidates (excluding anchor)
            3. Return candidates sorted by similarity
        
        Args:
            anchor_domain: Domain of the anchor example
            anchor_model: Model name of the anchor example
            n_candidates: Total number of candidates to sample
            min_pos: Minimum number of same-domain candidates to include (for domain strategy)
            exclude_idx: Optional index to exclude (the anchor itself)
            exclude_unique_id: Optional unique ID to exclude (the anchor itself)
            rng: Optional random.Random instance for reproducibility
            sampling_strategy: "domain" or "prompt_similarity"
            anchor_prompt_text: Prompt text of the anchor (required for prompt_similarity strategy)
            
        Returns:
            Tuple of:
            - List of sampled CandidateItem objects
            - Dict with sampling statistics (num_same_domain, num_same_model, etc.)
        """
        if sampling_strategy == "prompt_similarity":
            return self._sample_candidates_by_similarity(
                anchor_prompt_text=anchor_prompt_text,
                anchor_domain=anchor_domain,
                anchor_model=anchor_model,
                n_candidates=n_candidates,
                exclude_unique_id=exclude_unique_id,
            )
        else:
            # Original domain-based sampling
            return self._sample_candidates_by_domain(
                anchor_domain=anchor_domain,
                anchor_model=anchor_model,
                n_candidates=n_candidates,
                min_pos=min_pos,
                exclude_idx=exclude_idx,
                exclude_unique_id=exclude_unique_id,
                rng=rng,
            )
    
    def _sample_candidates_by_similarity(
        self,
        anchor_prompt_text: str,
        anchor_domain: str,
        anchor_model: str,
        n_candidates: int,
        exclude_unique_id: Optional[str] = None,
    ) -> Tuple[List[CandidateItem], Dict[str, Any]]:
        """
        Sample candidates using prompt similarity retrieval.
        
        Args:
            anchor_prompt_text: Prompt text of the anchor
            anchor_domain: Domain of the anchor (for stats)
            anchor_model: Model name of the anchor (for stats)
            n_candidates: Number of candidates to return
            exclude_unique_id: Optional unique ID to exclude
            
        Returns:
            Tuple of (candidates, stats)
        """
        if self._similarity_retriever is None:
            raise ValueError(
                "Similarity retriever not initialized. Call initialize_similarity_retriever() first."
            )
        
        if not anchor_prompt_text:
            raise ValueError("anchor_prompt_text is required for prompt_similarity sampling")
        
        # Retrieve similar prompts
        # IMPORTANT: exclude_unique_id must be set to prevent anchor from being retrieved
        # Retrieve more candidates to account for text-level deduplication (same prompt text with different IDs)
        retrieve_k = n_candidates * 2  # Retrieve extra to account for text-level deduplication
        retrieved_items = self._similarity_retriever.retrieve(
            prompt_text=anchor_prompt_text,
            k=retrieve_k,
            exclude_unique_id=exclude_unique_id,
        )
        
        # Double-check: filter out anchor if it somehow got through (by ID)
        if exclude_unique_id:
            retrieved_items = [item for item in retrieved_items if item.unique_id != exclude_unique_id]
        
        # Double-check: filter out anchor if it somehow got through (by normalized text)
        normalized_anchor = normalize_text(anchor_prompt_text)
        retrieved_items = [
            item for item in retrieved_items 
            if normalize_text(item.prompt_text or "") != normalized_anchor
        ]
        
        # Note: We don't trim here - let the deduplication loop handle it
        
        # Convert RetrievedItem to CandidateItem
        # Use RetrievedItem's prompt_text directly (it's already retrieved based on prompt similarity)
        # NOTE: When retrieve_over_all=True, retrieved_items may include items NOT in self._candidates
        # (e.g., from training dataset). We should include ALL retrieved items, not just those in the pool.
        candidates = []
        unique_id_to_candidate = {c.unique_id if c.unique_id else f"idx_{c.idx}": c for c in self._candidates}
        
        # Track which retrieved items come from pool vs external corpus
        from_pool_count = 0
        from_external_count = 0
        
        # Track unique_ids we've already added to avoid duplicates (ID-level)
        added_candidate_ids = set()
        # Track normalized prompt texts we've already added to avoid duplicates (text-level)
        added_normalized_texts = set()
        
        for retrieved in retrieved_items:
            # Skip if we've already added this unique_id (deduplicate by ID)
            if retrieved.unique_id in added_candidate_ids:
                continue
            
            # Always use RetrievedItem's prompt_text (it's what was actually retrieved)
            # Ensure prompt_text is a non-empty string - this should NEVER be empty if retrieval worked
            prompt_text = retrieved.prompt_text or ""
            if not isinstance(prompt_text, str):
                prompt_text = str(prompt_text) if prompt_text else ""
            
            # Validate that we have prompt_text - if not, this is a critical error
            if not prompt_text.strip():
                import warnings
                warnings.warn(
                    f"Retrieved item with unique_id={retrieved.unique_id} has empty prompt_text! "
                    f"This should not happen - the corpus should contain prompt_text. "
                    f"Skipping this candidate."
                )
                continue
            
            # Skip if we've already added a candidate with the same normalized prompt text (deduplicate by text)
            normalized_prompt = normalize_text(prompt_text)
            if normalized_prompt in added_normalized_texts:
                continue
            
            # Try to find existing candidate in pool first (to preserve idx and other metadata)
            if retrieved.unique_id in unique_id_to_candidate:
                pool_candidate = unique_id_to_candidate[retrieved.unique_id]
                # Always create new CandidateItem with prompt_text from RetrievedItem
                # This ensures prompt_text is always present (retrieved items always have it)
                candidates.append(CandidateItem(
                    idx=pool_candidate.idx,
                    prompt_text=prompt_text,  # Always use retrieved prompt_text (validated above)
                    domain=pool_candidate.domain if pool_candidate.domain else retrieved.domain,
                    model_name=pool_candidate.model_name if pool_candidate.model_name else retrieved.model_name,
                    unique_id=pool_candidate.unique_id,
                ))
                added_candidate_ids.add(retrieved.unique_id)
                added_normalized_texts.add(normalized_prompt)
                from_pool_count += 1
            else:
                # If not found in pool, this is from external corpus (training dataset)
                # Create new CandidateItem from RetrievedItem - this is valid when retrieve_over_all=True
                # Find a suitable idx (use max idx + 1 or find unused idx)
                max_idx = max([c.idx for c in self._candidates], default=-1) + len(candidates) + 1
                candidates.append(CandidateItem(
                    idx=max_idx,
                    prompt_text=prompt_text,  # RetrievedItem prompt_text (validated above)
                    domain=retrieved.domain,
                    model_name=retrieved.model_name,
                    unique_id=retrieved.unique_id,
                ))
                added_candidate_ids.add(retrieved.unique_id)
                added_normalized_texts.add(normalized_prompt)
                from_external_count += 1
        
        # Compute statistics
        anchor_domain_canonical = canonicalize_domain(anchor_domain)
        num_same_domain = sum(1 for c in candidates if canonicalize_domain(c.domain) == anchor_domain_canonical)
        num_same_model = sum(1 for c in candidates if c.model_name == anchor_model)
        
        stats = {
            "num_same_domain": num_same_domain,
            "num_same_model": num_same_model,
            "num_total": len(candidates),
            "sampling_strategy": "prompt_similarity",
            "retrieved_from_pool": from_pool_count,
            "retrieved_from_external": from_external_count,
        }
        
        return candidates, stats
    
    def _sample_candidates_by_domain(
        self,
        anchor_domain: str,
        anchor_model: str,
        n_candidates: int,
        min_pos: int = 1,
        exclude_idx: Optional[int] = None,
        exclude_unique_id: Optional[str] = None,
        rng: Optional[random.Random] = None,
    ) -> Tuple[List[CandidateItem], Dict[str, Any]]:
        """
        Original domain-based sampling strategy.
        
        This is the existing implementation extracted into a separate method.
        """
        if rng is None:
            rng = random.Random()
        
        if n_candidates <= 0 or len(self._candidates) == 0:
            return [], {"num_same_domain": 0, "num_same_model": 0, "num_total": 0}
        
        # Canonicalize anchor domain for consistent comparison
        anchor_domain_canonical = canonicalize_domain(anchor_domain)
        
        # Get candidates from same domain (excluding anchor)
        same_domain_candidates = [
            c for c in self._by_domain.get(anchor_domain_canonical, [])
            if (exclude_idx is None or c.idx != exclude_idx)
            and (exclude_unique_id is None or c.unique_id != exclude_unique_id)
        ]
        
        # Get candidates from other domains
        other_domain_candidates = [
            c for c in self._candidates
            if canonicalize_domain(c.domain) != anchor_domain_canonical
            and (exclude_idx is None or c.idx != exclude_idx)
            and (exclude_unique_id is None or c.unique_id != exclude_unique_id)
        ]
        
        selected: List[CandidateItem] = []
        selected_indices: Set[int] = set()
        selected_normalized_texts: Set[str] = set()  # Track normalized prompt texts to avoid duplicates
        
        # Step 1: Sample same-domain candidates (up to min_pos)
        n_same_domain_available = len(same_domain_candidates)
        n_same_domain_to_sample = min(min_pos, n_same_domain_available, n_candidates)
        
        if n_same_domain_to_sample > 0:
            sampled_same = rng.sample(same_domain_candidates, n_same_domain_to_sample)
            for c in sampled_same:
                # Skip if we've already added a candidate with the same normalized prompt text
                normalized_text = normalize_text(c.prompt_text or "")
                if normalized_text in selected_normalized_texts:
                    continue
                selected.append(c)
                selected_indices.add(c.idx)
                selected_normalized_texts.add(normalized_text)
        
        # Step 2: Fill remaining slots with other candidates
        remaining_slots = n_candidates - len(selected)
        
        if remaining_slots > 0:
            # First try other domains
            other_available = [
                c for c in other_domain_candidates 
                if c.idx not in selected_indices 
                and normalize_text(c.prompt_text or "") not in selected_normalized_texts
            ]
            n_from_other = min(remaining_slots, len(other_available))
            
            if n_from_other > 0:
                sampled_other = rng.sample(other_available, n_from_other)
                for c in sampled_other:
                    selected.append(c)
                    selected_indices.add(c.idx)
                    selected_normalized_texts.add(normalize_text(c.prompt_text or ""))
                remaining_slots -= n_from_other
            
            # If still need more, sample additional from same domain (if available)
            if remaining_slots > 0:
                same_available = [
                    c for c in same_domain_candidates 
                    if c.idx not in selected_indices 
                    and normalize_text(c.prompt_text or "") not in selected_normalized_texts
                ]
                n_from_same = min(remaining_slots, len(same_available))
                if n_from_same > 0:
                    sampled_same_extra = rng.sample(same_available, n_from_same)
                    for c in sampled_same_extra:
                        selected.append(c)
                        selected_indices.add(c.idx)
                        selected_normalized_texts.add(normalize_text(c.prompt_text or ""))
        
        # Compute statistics (using canonicalized domain)
        num_same_domain = sum(1 for c in selected if canonicalize_domain(c.domain) == anchor_domain_canonical)
        num_same_model = sum(1 for c in selected if c.model_name == anchor_model)
        
        # Task C: Self-exclusion check (log-only, behind debug flag)
        self_exclusion_violations = []
        if exclude_idx is not None:
            for c in selected:
                if c.idx == exclude_idx:
                    self_exclusion_violations.append(("idx", exclude_idx, c.idx))
        if exclude_unique_id is not None:
            for c in selected:
                if c.unique_id and str(c.unique_id) == str(exclude_unique_id):
                    self_exclusion_violations.append(("unique_id", exclude_unique_id, c.unique_id))
        
        stats = {
            "num_same_domain": num_same_domain,
            "num_same_model": num_same_model,
            "num_total": len(selected),
            "same_domain_available": n_same_domain_available,
            "other_domain_available": len(other_domain_candidates),
            "requested_min_pos": min_pos,
            "sampling_strategy": "domain",
        }
        
        # Shuffle to avoid positional bias (same-domain always first)
        rng.shuffle(selected)
        
        return selected, stats
    
    def sample_batch_candidates(
        self,
        anchor_domains: List[str],
        anchor_models: List[str],
        n_candidates: int,
        min_pos: int = 1,
        exclude_indices: Optional[List[int]] = None,
        rng: Optional[random.Random] = None,
    ) -> Tuple[List[List[CandidateItem]], Dict[str, Any]]:
        """
        Sample candidates for a batch of anchors.
        
        Args:
            anchor_domains: List of domains for each anchor
            anchor_models: List of model names for each anchor
            n_candidates: Number of candidates per anchor
            min_pos: Minimum same-domain candidates per anchor
            exclude_indices: Optional list of indices to exclude (one per anchor)
            rng: Optional random.Random instance
            
        Returns:
            Tuple of:
            - List of candidate lists (one per anchor)
            - Aggregated statistics dict
        """
        if rng is None:
            rng = random.Random()
        
        all_candidates = []
        total_same_domain = 0
        total_same_model = 0
        total_candidates = 0
        anchors_with_positives = 0
        
        for i, (domain, model) in enumerate(zip(anchor_domains, anchor_models)):
            exclude_idx = exclude_indices[i] if exclude_indices else None
            
            candidates, stats = self.sample_candidates(
                anchor_domain=domain,
                anchor_model=model,
                n_candidates=n_candidates,
                min_pos=min_pos,
                exclude_idx=exclude_idx,
                rng=rng,
            )
            
            all_candidates.append(candidates)
            total_same_domain += stats["num_same_domain"]
            total_same_model += stats["num_same_model"]
            total_candidates += stats["num_total"]
            
            if stats["num_same_domain"] > 0:
                anchors_with_positives += 1
        
        n_anchors = len(anchor_domains)
        agg_stats = {
            "avg_same_domain": total_same_domain / max(1, n_anchors),
            "avg_same_model": total_same_model / max(1, n_anchors),
            "avg_candidates_total": total_candidates / max(1, n_anchors),
            "pct_anchors_with_positives": 100.0 * anchors_with_positives / max(1, n_anchors),
            "num_anchors": n_anchors,
        }
        
        return all_candidates, agg_stats
    
    def get_summary(self) -> str:
        """Get a human-readable summary of the pool."""
        domain_counts = self.get_domain_counts()
        top_domains = sorted(domain_counts.items(), key=lambda x: -x[1])[:5]
        
        lines = [
            f"ReplayCandidatePool Summary:",
            f"  Total candidates: {len(self._candidates)}",
            f"  Unique domains: {len(self._all_domains)}",
            f"  Unique models: {len(self._all_models)}",
            f"  Top domains: {top_domains}",
        ]
        return "\n".join(lines)


def build_replay_candidate_pool(
    replay_examples: List[Dict[str, Any]],
    verbose: bool = True,
) -> ReplayCandidatePool:
    """
    Build a ReplayCandidatePool from replay buffer examples.
    
    This is a convenience function to be called from the trainer.
    
    Args:
        replay_examples: List of raw replay examples with prompt, domain, model_name
        verbose: Whether to print summary
        
    Returns:
        Populated ReplayCandidatePool
    """
    pool = ReplayCandidatePool.from_examples(
        examples=replay_examples,
        prompt_key="instruction",  # Use "instruction" (raw prompt), not "prompt" (formatted with system prompt)
        domain_key="domain",
        model_key="model_name",
        id_key="unique_id",  # Try unique_id first, will fallback to other keys or generate
    )
    
    if verbose and len(pool) > 0:
        print(f"  [ReplayCandidatePool] {pool.get_summary()}")
    
    return pool

