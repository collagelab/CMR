"""
Domain + Model-Aware Coreset Replay Sampler

This module implements a diversity-preserving replay buffer selection strategy
for continual learning. Instead of naive random replay, it uses:

1. Domain-aware proportional allocation with floor/cap constraints
2. Per-model diversity limits to avoid over-representing frequent models  
3. Farthest-point sampling in embedding space for maximum diversity

CL Motivation:
- Random replay can undersample long-tail domains/models, leading to 
  catastrophic forgetting on those subpopulations.
- Coreset selection ensures balanced coverage across domains and models,
  while maximizing embedding-space diversity within those constraints.
"""

import os
import json
import hashlib
import random
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
from collections import defaultdict

import numpy as np
from sentence_transformers import SentenceTransformer
import torch
import torch.nn.functional as F

from ..model_selection_carve.model_registry import ModelRegistry, normalize_model_name


# ============================================================================
# Embedding Cache Utilities
# ============================================================================

def get_cache_dir(cache_dir: Optional[str] = None) -> Path:
    """Get the embedding cache directory, creating if necessary."""
    if cache_dir is None:
        # Default to cco/cache/embeddings
        package_root = Path(__file__).parent.parent
        cache_dir = package_root / "cache" / "embeddings"
    else:
        cache_dir = Path(cache_dir)
    
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def get_prompt_cache_dir(cache_dir: Optional[str] = None) -> Path:
    """Get the CARvE prompt-embedding cache directory, creating if necessary."""
    if cache_dir is None:
        package_root = Path(__file__).parent.parent
        cache_dir = package_root / "cache" / "prompt_embeddings"
    else:
        cache_dir = Path(cache_dir)

    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def compute_cache_key(examples: List[Dict[str, Any]], embedding_source: str) -> str:
    """Compute a stable cache key for a set of examples."""
    # Use instruction texts and model names to create a deterministic hash
    key_data = []
    for ex in examples:
        instruction = ex.get("instruction", "")
        model_name = ex.get("model_name", "")
        key_data.append(f"{instruction[:100]}|{model_name}")
    
    combined = f"{embedding_source}:" + "|||".join(sorted(key_data))
    return hashlib.md5(combined.encode()).hexdigest()


def compute_prompt_embed_cache_key(
    examples: List[Dict[str, Any]],
    router_registry_base_path: str,
    repo_id: str,
    max_length: int,
    seed: Optional[int],
    selector_name: str = "carve_prompt",
) -> str:
    """
    Cache key for prompt embeddings computed from the CARvE LM.

    IMPORTANT: This is order-sensitive because embeddings are returned in the same order
    as `examples`.
    """
    key_parts: List[str] = []
    for ex in examples:
        instruction = ex.get("instruction", "") or ""
        model_name = ex.get("model_name", "") or ""
        domain = ex.get("domain", "") or ""
        key_parts.append(f"{instruction[:120]}|{model_name}|{domain}")

    # Include the router checkpoint identity + model identity + tokenization length.
    meta = f"{selector_name}|{router_registry_base_path}|{repo_id}|L{max_length}|seed{seed}"
    combined = meta + "|||" + "|||".join(key_parts)
    return hashlib.md5(combined.encode()).hexdigest()


def load_cached_prompt_embeddings(
    cache_dir: Path,
    cache_key: str,
) -> Optional[np.ndarray]:
    """Load cached CARvE prompt embeddings if present."""
    emb_path = cache_dir / f"{cache_key}_prompt_emb.npy"
    if not emb_path.exists():
        return None
    try:
        return np.load(emb_path)
    except Exception as e:
        print(f"Warning: Failed to load cached prompt embeddings: {e}")
        return None


def save_cached_prompt_embeddings(
    cache_dir: Path,
    cache_key: str,
    embeddings: np.ndarray,
):
    """Save CARvE prompt embeddings."""
    emb_path = cache_dir / f"{cache_key}_prompt_emb.npy"
    try:
        np.save(emb_path, embeddings)
    except Exception as e:
        print(f"Warning: Failed to cache prompt embeddings: {e}")


def _get_domain(ex: Dict[str, Any]) -> str:
    domain = ex.get("domain", "unknown")
    if not domain:
        api_data = ex.get("api_data", {})
        if isinstance(api_data, dict):
            domain = api_data.get("domain", "unknown")
    return domain or "unknown"


def _build_training_prompt_text(
    ex: Dict[str, Any],
    train_config: Any,
) -> str:
    """
    Build the same prompt string used in training conversion (up to '###Response:').

    We intentionally exclude the completion (model_name) because the router prompt
    embedding pools only over prompt tokens.
    """
    instruction = (ex.get("instruction", "") or "").replace("\r\n", "\n").strip()

    # System prompt: default gorilla prompt unless overridden.
    # Keep this minimal to avoid importing prompt templates here.
    system_prompt = getattr(train_config, "system_prompt", "") or ""
    if system_prompt == "":
        system_prompt = (
            "You are Gorilla, an expert API model router. "
            "Read the ###Instruction and ###Input below and return ONLY a single model name. "
            "Do not invent model name. Do not return anything else.\n\n"
        )

    # Optional retriever card snippet (if configured and present in raw data).
    model_card = ""
    retriever = getattr(train_config, "retriever", None)
    if retriever:
        key_map = {
            "bm25": "bm25_retrieved_info",
            "sentence_transformer": "sentence_transformer_retrieved_info",
            "splade": "splade_retrieved_info",
            "flagembedding": "flagembedding_retrieved_info",
        }
        retrieved_key = key_map.get(retriever)
        if retrieved_key:
            retrieved_info = ex.get(retrieved_key, "")
            if retrieved_info:
                model_card = "\n<Reference API>: " + str(retrieved_info).replace("\r\n", "\n").strip()

    return system_prompt + instruction + model_card + "\n###Response:"


def _load_router_weights_from_base_path(router_registry_base_path: str) -> Tuple[ModelRegistry, torch.Tensor, torch.Tensor]:
    """
    Load ModelRegistry + router weights from a base checkpoint directory.

    Expects:
      - {ckpt_dir}/model_registry.json (router_registry_base_path)
      - {ckpt_dir}/router_model.pt (RouterModel state_dict)
    """
    registry = ModelRegistry.load(router_registry_base_path)
    ckpt_dir = Path(router_registry_base_path).parent
    router_path = ckpt_dir / "router_model.pt"
    if not router_path.exists():
        raise FileNotFoundError(f"router_model.pt not found next to registry: {router_path}")

    state = torch.load(router_path, map_location="cpu")
    # Saved state_dict is RouterModel only in current codebase.
    # Be defensive about key prefixes in case wrappers change.
    def _pick(d: dict, key: str) -> torch.Tensor:
        if key in d:
            return d[key]
        # Common prefix variants
        for prefix in ("router.", "module.router.", "model.router."):
            k2 = prefix + key
            if k2 in d:
                return d[k2]
        raise KeyError(f"Key '{key}' not found in router checkpoint. Keys: {list(d.keys())[:10]}...")

    prompt_proj_w = _pick(state, "prompt_projection.weight")  # [D_router, D_lm]
    model_emb_w = _pick(state, "model_embeddings.weight")     # [num_models, D_router]
    return registry, prompt_proj_w, model_emb_w


def _herding_order(
    X: np.ndarray,
    k: int,
    mu_unit: Optional[np.ndarray] = None,
) -> List[int]:
    """
    Greedy herding order to approximate cluster mean (iCaRL-style).

    Args:
        X: [n, d] feature vectors (recommended: unit-normalized)
        k: number to select (k <= n)
        mu_unit: optional unit-normalized target mean [d]
    Returns:
        List of selected indices into X, length k.
    """
    n = X.shape[0]
    if k <= 0 or n == 0:
        return []
    if k >= n:
        return list(range(n))

    if mu_unit is None:
        mu = X.mean(axis=0, keepdims=False)
        mu_norm = np.linalg.norm(mu) + 1e-8
        mu_unit = mu / mu_norm

    selected: List[int] = []
    remaining = np.ones(n, dtype=bool)
    s = np.zeros((X.shape[1],), dtype=np.float32)

    for t in range(1, k + 1):
        idxs = np.where(remaining)[0]
        # Candidate new means: (s + X[i]) / t
        # Score by cosine to mu_unit.
        cand = (s[None, :] + X[idxs]) / float(t)
        cand_norm = np.linalg.norm(cand, axis=1) + 1e-8
        cand_unit = cand / cand_norm[:, None]
        scores = cand_unit @ mu_unit
        best_local = int(idxs[int(np.argmax(scores))])
        selected.append(best_local)
        remaining[best_local] = False
        s += X[best_local].astype(np.float32)

    return selected


def load_cached_embeddings(
    cache_dir: Path, 
    cache_key: str
) -> Optional[Tuple[np.ndarray, Dict[int, int]]]:
    """
    Load cached embeddings if they exist.
    
    Returns:
        Tuple of (embeddings array, index mapping) or None if not cached
    """
    embeddings_path = cache_dir / f"{cache_key}_embeddings.npy"
    mapping_path = cache_dir / f"{cache_key}_mapping.json"
    
    if embeddings_path.exists() and mapping_path.exists():
        try:
            embeddings = np.load(embeddings_path)
            with open(mapping_path, 'r') as f:
                mapping = {int(k): v for k, v in json.load(f).items()}
            return embeddings, mapping
        except Exception as e:
            print(f"Warning: Failed to load cached embeddings: {e}")
            return None
    return None


def save_cached_embeddings(
    cache_dir: Path,
    cache_key: str,
    embeddings: np.ndarray,
    index_mapping: Dict[int, int]
):
    """Save embeddings to cache."""
    embeddings_path = cache_dir / f"{cache_key}_embeddings.npy"
    mapping_path = cache_dir / f"{cache_key}_mapping.json"
    
    try:
        np.save(embeddings_path, embeddings)
        with open(mapping_path, 'w') as f:
            json.dump(index_mapping, f)
    except Exception as e:
        print(f"Warning: Failed to cache embeddings: {e}")


# ============================================================================
# Embedding Computation
# ============================================================================

def compute_embeddings(
    examples: List[Dict[str, Any]],
    embedding_source: str = "sentence_transformer",
    cache_dir: Optional[str] = None,
    batch_size: int = 64,
    device: Optional[str] = None
) -> Tuple[np.ndarray, Dict[int, int]]:
    """
    Compute or load cached embeddings for examples.
    
    Args:
        examples: List of training examples with 'instruction' field
        embedding_source: Which embedding model to use
        cache_dir: Optional cache directory path
        batch_size: Batch size for embedding computation
        device: Device for computation ('cuda' or 'cpu')
    
    Returns:
        Tuple of:
        - embeddings: np.ndarray of shape [num_examples, embedding_dim]
        - index_mapping: Dict mapping original index -> embedding index
    """
    cache_path = get_cache_dir(cache_dir)
    cache_key = compute_cache_key(examples, embedding_source)
    
    # Try loading from cache
    cached = load_cached_embeddings(cache_path, cache_key)
    if cached is not None:
        print(f"  Loaded cached embeddings for {len(cached[0])} examples")
        return cached
    
    # Compute embeddings
    print(f"  Computing embeddings with {embedding_source} for {len(examples)} examples...")
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Extract prompts
    prompts = []
    index_mapping = {}
    for i, ex in enumerate(examples):
        prompt = ex.get("instruction", "").strip()
        if prompt:
            index_mapping[i] = len(prompts)
            prompts.append(prompt)
    
    if not prompts:
        return np.array([]), {}
    
    # Compute based on source
    if embedding_source == "sentence_transformer":
        model = SentenceTransformer(
            "all-mpnet-base-v2",
            device=device,
            cache_folder=os.environ.get("HF_HOME", None)
        )
        embeddings = model.encode(
            prompts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )
    elif embedding_source == "flagembedding":
        from .retrievers_carve.bgem3 import BGEM3Retriever
        retriever = BGEM3Retriever(device=device)
        embeddings_dict = retriever.model.encode(
            prompts,
            batch_size=batch_size,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False
        )
        embeddings = embeddings_dict["dense_vecs"]
    else:
        # Default to sentence transformer
        model = SentenceTransformer(
            "all-mpnet-base-v2",
            device=device,
            cache_folder=os.environ.get("HF_HOME", None)
        )
        embeddings = model.encode(
            prompts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )
    
    # Cache results
    save_cached_embeddings(cache_path, cache_key, embeddings, index_mapping)
    print(f"  Cached embeddings for future use")
    
    return embeddings, index_mapping


# ============================================================================
# Farthest-Point Sampling
# ============================================================================

def farthest_point_sampling(
    embeddings: np.ndarray,
    indices: List[int],
    k: int,
    seed: Optional[int] = None,
    existing_selected: Optional[List[int]] = None
) -> List[int]:
    """
    Select k points using farthest-point sampling for maximum diversity.
    
    This greedily selects points that are farthest from the already-selected set,
    ensuring good coverage of the embedding space.
    
    Optimized version that:
    - Pre-normalizes candidate embeddings once
    - Uses incremental distance updates instead of full matrix multiplication
    - Maintains selected embeddings as numpy array for efficiency
    
    Args:
        embeddings: Full embeddings array
        indices: Indices of candidate points to select from
        k: Number of points to select
        seed: Random seed for initial point selection
        existing_selected: Optional list of already-selected indices to consider
                          when computing distances (for filling quotas)
    
    Returns:
        List of selected indices
    """
    if len(indices) <= k:
        return list(indices)
    
    if k == 0:
        return []
    
    # Get embeddings for candidates and normalize ONCE (optimization)
    candidate_embeddings = embeddings[indices]
    candidate_norms = np.linalg.norm(candidate_embeddings, axis=1, keepdims=True)
    candidate_norm = candidate_embeddings / (candidate_norms + 1e-8)
    
    # Use set for O(1) lookups (optimization)
    selected_set = set()
    selected = []
    selected_norm_list = []
    
    # If we have existing selections, normalize them once
    if existing_selected and len(existing_selected) > 0:
        existing_emb = embeddings[existing_selected]
        existing_norms = np.linalg.norm(existing_emb, axis=1, keepdims=True)
        selected_norm_list = (existing_emb / (existing_norms + 1e-8)).tolist()
        # Note: existing_selected are embedding indices, not original indices
        # We'll track them separately for distance computation
    
    # Pick random starting point if no existing selections
    if not selected_norm_list:
        if seed is not None:
            random.seed(seed)
        start_idx = random.randint(0, len(indices) - 1)
        selected.append(indices[start_idx])
        selected_set.add(indices[start_idx])
        # Normalize and store
        start_emb = embeddings[indices[start_idx]]
        start_norm = start_emb / (np.linalg.norm(start_emb) + 1e-8)
        selected_norm_list.append(start_norm)
    
    # Maintain max similarities to selected set for each candidate (optimization)
    # This avoids recomputing full similarity matrix each iteration
    # We track max similarity (closest point) - farthest point has minimum max similarity
    n_candidates = len(indices)
    
    # Initialize distances based on current selected set
    if len(selected_norm_list) > 0:
        # Compute similarities to all currently selected points
        selected_norm_array = np.array(selected_norm_list)
        similarities = np.dot(candidate_norm, selected_norm_array.T)  # [n_candidates, n_selected]
        max_similarities = np.max(similarities, axis=1)  # Max similarity to any selected point
    else:
        # Should not happen, but initialize to -inf if no selections yet
        max_similarities = np.full(n_candidates, -np.inf)
    
    # Greedily add farthest points
    while len(selected) < k:
        # Mask already selected candidates
        for i, idx in enumerate(indices):
            if idx in selected_set:
                max_similarities[i] = float('inf')  # Don't re-select
        
        # Find farthest point (minimum max-similarity)
        farthest_candidate_idx = np.argmin(max_similarities)
        farthest_original_idx = indices[farthest_candidate_idx]
        
        selected.append(farthest_original_idx)
        selected_set.add(farthest_original_idx)
        
        # Get normalized embedding of newly selected point
        new_emb = embeddings[farthest_original_idx]
        new_norm = new_emb / (np.linalg.norm(new_emb) + 1e-8)
        selected_norm_list.append(new_norm)
        
        # Incrementally update max similarities: compute similarity to newly added point
        # and update to be max of current and new similarity
        new_similarities = np.dot(candidate_norm, new_norm)  # [n_candidates]
        max_similarities = np.maximum(max_similarities, new_similarities)
        
        # Mask the newly selected point for next iteration
        max_similarities[farthest_candidate_idx] = float('inf')
    
    return selected


# ============================================================================
# Main Coreset Builder
# ============================================================================

def compute_coreset_cache_key(
    examples: List[Dict[str, Any]],
    replay_ratio: float,
    min_per_domain: int,
    max_per_domain: Optional[int],
    max_per_model: int,
    embedding_source: str,
    seed: Optional[int],
    selector_name: str = "coreset"
) -> str:
    """Compute a stable cache key for coreset selection."""
    # Base key from examples (same as embedding cache)
    example_key = compute_cache_key(examples, embedding_source)
    
    # Add coreset-specific parameters
    params = (
        f"{selector_name}_{replay_ratio}_{min_per_domain}_"
        f"{max_per_domain}_{max_per_model}_{seed}"
    )
    combined = f"coreset_{example_key}_{params}"
    return hashlib.md5(combined.encode()).hexdigest()


def load_cached_coreset(
    cache_dir: Path,
    cache_key: str
) -> Optional[List[int]]:
    """Load cached coreset indices if they exist."""
    coreset_path = cache_dir / f"{cache_key}_coreset.json"
    
    if coreset_path.exists():
        try:
            with open(coreset_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Failed to load cached coreset: {e}")
            return None
    return None


def save_cached_coreset(
    cache_dir: Path,
    cache_key: str,
    indices: List[int]
):
    """Save coreset indices to cache."""
    coreset_path = cache_dir / f"{cache_key}_coreset.json"
    
    try:
        with open(coreset_path, 'w') as f:
            json.dump(indices, f)
    except Exception as e:
        print(f"Warning: Failed to cache coreset: {e}")


def build_domain_model_coreset_replay(
    apibench_examples: List[Dict[str, Any]],
    replay_ratio: float,
    min_per_domain: int,
    max_per_domain: Optional[int],
    max_per_model: int,
    embedding_source: str = "sentence_transformer",
    cache_dir: Optional[str] = None,
    seed: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Build a domain + model-aware coreset replay buffer.
    
    Algorithm:
    1. Compute total replay budget B = replay_ratio * len(examples)
    2. Group examples by domain, compute proportional quotas with floor/cap
    3. Within each domain, group by model and select up to max_per_model diverse examples
    4. Use farthest-point sampling in embedding space for diversity
    5. Fill or trim domain quotas as needed
    
    Args:
        apibench_examples: Full list of APIBench training examples
        replay_ratio: Fraction of examples to include in replay (e.g., 0.1)
        min_per_domain: Minimum examples per domain (floor)
        max_per_domain: Optional maximum per domain (cap)
        max_per_model: Maximum examples per model within a domain
        embedding_source: Embedding model to use for diversity sampling
        cache_dir: Optional embedding cache directory
        seed: Random seed for reproducibility
    
    Returns:
        List of selected examples for replay
    """
    if not apibench_examples:
        return []
    
    # Step 0: Check for cached coreset
    cache_path = get_cache_dir(cache_dir)
    coreset_cache_key = compute_coreset_cache_key(
        apibench_examples, replay_ratio, min_per_domain, max_per_domain,
        max_per_model, embedding_source, seed
    )
    
    cached_indices = load_cached_coreset(cache_path, coreset_cache_key)
    if cached_indices is not None:
        print(f"\n=== Loading Cached Coreset Replay ===")
        print(f"  Loaded {len(cached_indices)} cached coreset indices")
        result = [apibench_examples[i] for i in cached_indices if i < len(apibench_examples)]
        print(f"  Final replay size: {len(result)}")
        return result
    
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    # Step 1: Compute total budget
    total_budget = int(replay_ratio * len(apibench_examples))
    print(f"\n=== Building Domain+Model Coreset Replay ===")
    print(f"  Total examples: {len(apibench_examples)}")
    print(f"  Replay ratio: {replay_ratio}")
    print(f"  Target budget: {total_budget}")
    
    # Step 2: Compute embeddings (with caching)
    embeddings, index_mapping = compute_embeddings(
        apibench_examples,
        embedding_source=embedding_source,
        cache_dir=cache_dir
    )
    
    # Create reverse mapping: embedding_idx -> original_idx
    reverse_mapping = {v: k for k, v in index_mapping.items()}
    
    # Step 3: Group by domain
    by_domain: Dict[str, List[int]] = defaultdict(list)
    for i, ex in enumerate(apibench_examples):
        domain = ex.get("domain", "unknown")
        if not domain:
            # Try to get from api_data
            api_data = ex.get("api_data", {})
            if isinstance(api_data, dict):
                domain = api_data.get("domain", "unknown")
        if not domain:
            domain = "unknown"
        by_domain[domain].append(i)
    
    print(f"  Domains found: {len(by_domain)}")
    
    # Step 4: Compute domain quotas
    domain_quotas = {}
    for domain, indices in by_domain.items():
        # Proportional allocation
        raw_quota = total_budget * len(indices) / len(apibench_examples)
        quota = max(min_per_domain, int(round(raw_quota)))
        
        # Apply cap
        if max_per_domain is not None:
            quota = min(quota, max_per_domain, len(indices))
        else:
            quota = min(quota, len(indices))
        
        domain_quotas[domain] = quota
    
    # Step 5: Build coreset per domain
    all_selected = []
    domain_stats = {}
    
    for domain, domain_indices in by_domain.items():
        quota = domain_quotas[domain]
        
        # Group by model within domain
        by_model: Dict[str, List[int]] = defaultdict(list)
        for idx in domain_indices:
            model_id = apibench_examples[idx].get("model_name", "unknown")
            by_model[model_id].append(idx)
        
        # First pass: select up to max_per_model per model using FPS
        domain_coreset = []
        for model_id, model_indices in by_model.items():
            # Get embedding indices for this model's examples
            model_emb_indices = [
                index_mapping[i] for i in model_indices 
                if i in index_mapping
            ]
            
            if not model_emb_indices:
                # No embeddings, just take random sample
                n_select = min(max_per_model, len(model_indices))
                domain_coreset.extend(random.sample(model_indices, n_select))
            else:
                # Use FPS to select diverse examples
                n_select = min(max_per_model, len(model_emb_indices))
                selected_emb_indices = farthest_point_sampling(
                    embeddings,
                    model_emb_indices,
                    n_select,
                    seed=seed
                )
                # Map back to original indices
                for emb_idx in selected_emb_indices:
                    domain_coreset.append(reverse_mapping[emb_idx])
        
        # Adjust to quota
        if len(domain_coreset) > quota:
            # Downsample using FPS on the coreset
            coreset_emb_indices = [
                index_mapping[i] for i in domain_coreset 
                if i in index_mapping
            ]
            if len(coreset_emb_indices) >= quota:
                selected_emb_indices = farthest_point_sampling(
                    embeddings,
                    coreset_emb_indices,
                    quota,
                    seed=seed
                )
                domain_coreset = [reverse_mapping[i] for i in selected_emb_indices]
            else:
                domain_coreset = random.sample(domain_coreset, quota)
        
        elif len(domain_coreset) < quota:
            # Fill remaining slots from unused examples in domain
            used_set = set(domain_coreset)
            remaining = [i for i in domain_indices if i not in used_set]
            
            if remaining:
                needed = quota - len(domain_coreset)
                remaining_emb_indices = [
                    index_mapping[i] for i in remaining 
                    if i in index_mapping
                ]
                
                if len(remaining_emb_indices) >= needed:
                    # Use FPS with existing selections as reference
                    existing_emb = [index_mapping[i] for i in domain_coreset if i in index_mapping]
                    selected_emb_indices = farthest_point_sampling(
                        embeddings,
                        remaining_emb_indices,
                        needed,
                        seed=seed,
                        existing_selected=existing_emb
                    )
                    for emb_idx in selected_emb_indices:
                        domain_coreset.append(reverse_mapping[emb_idx])
                else:
                    # Add all remaining
                    domain_coreset.extend(remaining)
        
        all_selected.extend(domain_coreset)
        
        # Collect model stats for this domain
        model_counts = defaultdict(int)
        for idx in domain_coreset:
            model_id = apibench_examples[idx].get("model_name", "unknown")
            model_counts[model_id] += 1
        
        domain_stats[domain] = {
            "count": len(domain_coreset),
            "quota": quota,
            "total_available": len(domain_indices),
            "unique_models": len(model_counts)
        }
    
    # Step 6: Global adjustment if needed
    if len(all_selected) > total_budget:
        # Trim globally using FPS
        print(f"  Trimming {len(all_selected)} -> {total_budget} globally")
        all_emb_indices = [
            index_mapping[i] for i in all_selected 
            if i in index_mapping
        ]
        if len(all_emb_indices) >= total_budget:
            selected_emb_indices = farthest_point_sampling(
                embeddings,
                all_emb_indices,
                total_budget,
                seed=seed
            )
            all_selected = [reverse_mapping[i] for i in selected_emb_indices]
        else:
            all_selected = random.sample(all_selected, total_budget)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_selected = []
    for idx in all_selected:
        if idx not in seen:
            seen.add(idx)
            unique_selected.append(idx)
    
    # Build final result
    result = [apibench_examples[i] for i in unique_selected]
    
    # Cache the coreset indices for future runs
    save_cached_coreset(cache_path, coreset_cache_key, unique_selected)
    print(f"  Cached coreset selection for future use")
    
    # Log statistics
    print(f"\n  Final replay size: {len(result)}")
    print(f"\n  Per-domain statistics:")
    for domain, stats in sorted(domain_stats.items(), key=lambda x: -x[1]["count"]):
        print(f"    {domain}: {stats['count']} selected (quota={stats['quota']}, "
              f"available={stats['total_available']}, unique_models={stats['unique_models']})")
    
    # Log top models
    model_counts = defaultdict(int)
    for ex in result:
        model_counts[ex.get("model_name", "unknown")] += 1
    
    print(f"\n  Top-10 models by replay frequency:")
    for model, count in sorted(model_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"    {model}: {count}")
    
    return result


def representative_sampling(
    embeddings: np.ndarray,
    indices: List[int],
    k: int,
    boundary_fraction: float = 0.25
) -> List[int]:
    """
    Select representative samples with center-first + boundary strategy.

    Strategy:
    - Pick most central points (closest to cluster centroid) for prototypical replay.
    - Optionally reserve a small fraction for boundary points (farthest from centroid)
      to preserve decision boundaries.
    """
    if len(indices) <= k:
        return list(indices)
    if k <= 0:
        return []

    candidate_embeddings = embeddings[indices]
    candidate_norms = np.linalg.norm(candidate_embeddings, axis=1, keepdims=True)
    candidate_norm = candidate_embeddings / (candidate_norms + 1e-8)

    centroid = np.mean(candidate_norm, axis=0, keepdims=True)
    centroid = centroid / (np.linalg.norm(centroid, axis=1, keepdims=True) + 1e-8)

    # Higher cosine similarity => closer to representative center.
    center_similarity = np.dot(candidate_norm, centroid.T).reshape(-1)
    sorted_center = np.argsort(-center_similarity)

    boundary_fraction = max(0.0, min(1.0, boundary_fraction))
    n_boundary = int(round(k * boundary_fraction))
    n_boundary = min(n_boundary, max(0, k - 1))
    n_center = k - n_boundary

    selected_local: List[int] = []
    used = set()

    # Center-first: prototype prompts.
    for local_idx in sorted_center:
        if len(selected_local) >= n_center:
            break
        selected_local.append(local_idx)
        used.add(local_idx)

    # Boundary points: retain some near-edge coverage.
    if n_boundary > 0:
        sorted_boundary = np.argsort(center_similarity)  # Low similarity => farther away
        for local_idx in sorted_boundary:
            if len(selected_local) >= k:
                break
            if local_idx in used:
                continue
            selected_local.append(local_idx)
            used.add(local_idx)

    # Fill if needed (rare, mostly due rounding/degenerate shapes).
    if len(selected_local) < k:
        for local_idx in sorted_center:
            if len(selected_local) >= k:
                break
            if local_idx in used:
                continue
            selected_local.append(local_idx)
            used.add(local_idx)

    return [indices[i] for i in selected_local[:k]]


def build_domain_model_representative_replay(
    apibench_examples: List[Dict[str, Any]],
    replay_ratio: float,
    min_per_domain: int,
    max_per_domain: Optional[int],
    max_per_model: int,
    embedding_source: str = "sentence_transformer",
    cache_dir: Optional[str] = None,
    seed: Optional[int] = None,
    boundary_fraction: float = 0.25,
) -> List[Dict[str, Any]]:
    """
    Build replay buffer from representative prompts per domain/model.

    This selector is designed for continual-learning memory where replay examples
    are chosen once per experience, without access to future data.
    """
    if not apibench_examples:
        return []

    cache_path = get_cache_dir(cache_dir)
    cache_key = compute_coreset_cache_key(
        apibench_examples,
        replay_ratio,
        min_per_domain,
        max_per_domain,
        max_per_model,
        embedding_source,
        seed,
        selector_name=f"representative_bf{boundary_fraction}",
    )
    cached_indices = load_cached_coreset(cache_path, cache_key)
    if cached_indices is not None:
        print(f"\n=== Loading Cached Representative Replay ===")
        print(f"  Loaded {len(cached_indices)} cached representative indices")
        result = [apibench_examples[i] for i in cached_indices if i < len(apibench_examples)]
        print(f"  Final replay size: {len(result)}")
        return result

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    total_budget = int(replay_ratio * len(apibench_examples))
    print(f"\n=== Building Domain+Model Representative Replay ===")
    print(f"  Total examples: {len(apibench_examples)}")
    print(f"  Replay ratio: {replay_ratio}")
    print(f"  Target budget: {total_budget}")
    print(f"  Boundary fraction: {boundary_fraction}")

    embeddings, index_mapping = compute_embeddings(
        apibench_examples,
        embedding_source=embedding_source,
        cache_dir=cache_dir,
    )
    reverse_mapping = {v: k for k, v in index_mapping.items()}

    by_domain: Dict[str, List[int]] = defaultdict(list)
    for i, ex in enumerate(apibench_examples):
        domain = ex.get("domain", "unknown")
        if not domain:
            api_data = ex.get("api_data", {})
            if isinstance(api_data, dict):
                domain = api_data.get("domain", "unknown")
        if not domain:
            domain = "unknown"
        by_domain[domain].append(i)

    print(f"  Domains found: {len(by_domain)}")

    domain_quotas: Dict[str, int] = {}
    for domain, indices in by_domain.items():
        raw_quota = total_budget * len(indices) / len(apibench_examples)
        quota = max(min_per_domain, int(round(raw_quota)))
        if max_per_domain is not None:
            quota = min(quota, max_per_domain, len(indices))
        else:
            quota = min(quota, len(indices))
        domain_quotas[domain] = quota

    all_selected: List[int] = []
    domain_stats: Dict[str, Dict[str, int]] = {}

    for domain, domain_indices in by_domain.items():
        quota = domain_quotas[domain]

        by_model: Dict[str, List[int]] = defaultdict(list)
        for idx in domain_indices:
            model_id = apibench_examples[idx].get("model_name", "unknown")
            by_model[model_id].append(idx)

        domain_selected: List[int] = []
        for _, model_indices in by_model.items():
            model_emb_indices = [index_mapping[i] for i in model_indices if i in index_mapping]
            if not model_emb_indices:
                n_select = min(max_per_model, len(model_indices))
                domain_selected.extend(random.sample(model_indices, n_select))
                continue

            n_select = min(max_per_model, len(model_emb_indices))
            selected_emb = representative_sampling(
                embeddings,
                model_emb_indices,
                n_select,
                boundary_fraction=boundary_fraction,
            )
            domain_selected.extend(reverse_mapping[e] for e in selected_emb)

        if len(domain_selected) > quota:
            selected_emb_indices = [index_mapping[i] for i in domain_selected if i in index_mapping]
            if len(selected_emb_indices) >= quota:
                trimmed_emb = representative_sampling(
                    embeddings,
                    selected_emb_indices,
                    quota,
                    boundary_fraction=boundary_fraction,
                )
                domain_selected = [reverse_mapping[e] for e in trimmed_emb]
            else:
                domain_selected = random.sample(domain_selected, quota)
        elif len(domain_selected) < quota:
            used_set = set(domain_selected)
            remaining = [i for i in domain_indices if i not in used_set]
            needed = quota - len(domain_selected)
            remaining_emb = [index_mapping[i] for i in remaining if i in index_mapping]
            if len(remaining_emb) >= needed and needed > 0:
                fill_emb = representative_sampling(
                    embeddings,
                    remaining_emb,
                    needed,
                    boundary_fraction=boundary_fraction,
                )
                domain_selected.extend(reverse_mapping[e] for e in fill_emb)
            elif needed > 0 and remaining:
                domain_selected.extend(random.sample(remaining, min(needed, len(remaining))))

        all_selected.extend(domain_selected)

        model_counts = defaultdict(int)
        for idx in domain_selected:
            model_id = apibench_examples[idx].get("model_name", "unknown")
            model_counts[model_id] += 1
        domain_stats[domain] = {
            "count": len(domain_selected),
            "quota": quota,
            "total_available": len(domain_indices),
            "unique_models": len(model_counts),
        }

    if len(all_selected) > total_budget:
        print(f"  Trimming {len(all_selected)} -> {total_budget} globally")
        all_emb_indices = [index_mapping[i] for i in all_selected if i in index_mapping]
        if len(all_emb_indices) >= total_budget:
            selected_emb_indices = representative_sampling(
                embeddings,
                all_emb_indices,
                total_budget,
                boundary_fraction=boundary_fraction,
            )
            all_selected = [reverse_mapping[i] for i in selected_emb_indices]
        else:
            all_selected = random.sample(all_selected, total_budget)

    seen = set()
    unique_selected = []
    for idx in all_selected:
        if idx not in seen:
            seen.add(idx)
            unique_selected.append(idx)

    result = [apibench_examples[i] for i in unique_selected]
    save_cached_coreset(cache_path, cache_key, unique_selected)
    print(f"  Cached representative selection for future use")

    print(f"\n  Final replay size: {len(result)}")
    print(f"\n  Per-domain statistics:")
    for domain, stats in sorted(domain_stats.items(), key=lambda x: -x[1]["count"]):
        print(
            f"    {domain}: {stats['count']} selected (quota={stats['quota']}, "
            f"available={stats['total_available']}, unique_models={stats['unique_models']})"
        )

    model_counts = defaultdict(int)
    for ex in result:
        model_counts[ex.get("model_name", "unknown")] += 1
    print(f"\n  Top-10 models by replay frequency:")
    for model, count in sorted(model_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"    {model}: {count}")

    return result


def build_domain_model_herding_replay(
    apibench_examples: List[Dict[str, Any]],
    replay_ratio: float,
    min_per_domain: int,
    max_per_domain: Optional[int],
    max_per_model: int,
    train_config: Any,
    model_manager: Any,
    router_registry_base_path: str,
    cache_dir: Optional[str] = None,
    seed: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Domain-aware herding replay:

    Stage A (model space): within each domain, order models by cosine similarity to
    the domain centroid in router `model_embeddings` space.

    Stage B (prompt space): for each (domain, model), compute CARvE prompt embeddings
    (last prompt token hidden state -> router prompt_projection -> L2 normalize), then
    order prompts by greedy herding to best match the (domain, model) centroid.

    Selection: iterate models in Stage-A order and take prompts in Stage-B order until
    the domain quota is filled. If the quota is still not filled, continue taking next
    prompts per model in Stage-A order (soft max_per_model, consistent with existing replay).
    """
    if not apibench_examples:
        return []
    if replay_ratio <= 0:
        return []
    if not router_registry_base_path:
        raise ValueError("router_registry_base_path must be set for domain_model_herding replay.")

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    total_budget = int(replay_ratio * len(apibench_examples))
    if total_budget <= 0:
        return []

    print(f"\n=== Building Domain+Model Herding Replay ===")
    print(f"  Total examples: {len(apibench_examples)}")
    print(f"  Replay ratio: {replay_ratio}")
    print(f"  Target budget: {total_budget}")
    print(f"  Router registry base: {router_registry_base_path}")

    # Group by domain and by (domain, model).
    by_domain: Dict[str, List[int]] = defaultdict(list)
    by_domain_model: Dict[Tuple[str, str], List[int]] = defaultdict(list)
    for i, ex in enumerate(apibench_examples):
        domain = _get_domain(ex)
        model_name = ex.get("model_name", "unknown") or "unknown"
        by_domain[domain].append(i)
        by_domain_model[(domain, model_name)].append(i)

    domains = list(by_domain.keys())
    n_domains = len(domains)
    print(f"  Domains found: {n_domains}")

    # Load router components from previous checkpoint.
    registry, prompt_proj_w_cpu, model_emb_w_cpu = _load_router_weights_from_base_path(router_registry_base_path)

    # Build normalized lookup from model name -> registry idx (case-insensitive).
    norm2idx: Dict[str, int] = {}
    for name, idx in registry.model2idx.items():
        norm2idx[normalize_model_name(name)] = idx

    # Normalize model embeddings (router space).
    model_emb = model_emb_w_cpu.float()
    model_emb_unit = F.normalize(model_emb, p=2, dim=-1).cpu().numpy().astype(np.float32)

    # Compute domain quotas with exact-sum adjustment (avoid global trimming artifacts).
    # Adjust min_per_domain downward if budget is too small to satisfy the floor.
    effective_min = min_per_domain
    if n_domains > 0 and total_budget < n_domains * min_per_domain:
        effective_min = max(1, total_budget // n_domains)
        print(f"  [Quota] Adjusted min_per_domain {min_per_domain} -> {effective_min} (budget too small for floor)")

    raw_quotas: Dict[str, float] = {
        d: total_budget * len(by_domain[d]) / len(apibench_examples)
        for d in domains
    }
    quotas: Dict[str, int] = {}
    for d in domains:
        q = max(effective_min, int(round(raw_quotas[d])))
        if max_per_domain is not None:
            q = min(q, max_per_domain, len(by_domain[d]))
        else:
            q = min(q, len(by_domain[d]))
        quotas[d] = q

    # Adjust to sum exactly to total_budget (within availability / floor constraints).
    def _total_q() -> int:
        return sum(quotas.values())

    # If over, decrement from largest quotas first.
    while _total_q() > total_budget:
        # Pick a domain that can be decremented without violating floor.
        candidates = [d for d in domains if quotas[d] > effective_min]
        if not candidates:
            break
        d = max(candidates, key=lambda x: quotas[x])
        quotas[d] -= 1

    # If under, increment where possible by residual.
    while _total_q() < total_budget:
        candidates = [d for d in domains if quotas[d] < len(by_domain[d]) and (max_per_domain is None or quotas[d] < max_per_domain)]
        if not candidates:
            break
        # Prefer domains with largest positive residual.
        d = max(candidates, key=lambda x: raw_quotas[x] - quotas[x])
        quotas[d] += 1

    # Precompute CARvE prompt embeddings for all examples.
    # Intentionally recompute every run by default so replay reflects current model state.
    prompts = [_build_training_prompt_text(ex, train_config) for ex in apibench_examples]
    tokenizer = model_manager.tokenizer
    model = model_manager.model
    device = model.device
    model.eval()

    prompt_proj_w = prompt_proj_w_cpu.to(device=device, dtype=torch.float32)  # [D_router, D_lm]
    max_length = int(getattr(train_config, "max_length", 1024))
    batch_size = int(getattr(train_config, "batch_size", 4))
    # Prompt embedding pass is inference-only; use a larger batch if possible.
    batch_size = max(batch_size, 8)

    embs: List[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(prompts), batch_size):
            batch_prompts = prompts[start:start + batch_size]
            tok = tokenizer(
                batch_prompts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            tok = {k: v.to(device) for k, v in tok.items()}
            out = model(**tok, output_hidden_states=True, return_dict=True)
            hs = out.hidden_states[-1]  # [B, L, D_lm]
            attn = tok["attention_mask"].long()  # [B, L]
            B, L = attn.shape
            pos = torch.arange(L, device=device).unsqueeze(0).expand(B, L)
            last_idx = (attn * pos).max(dim=1).values  # [B]
            batch_idx = torch.arange(B, device=device)
            last_h = hs[batch_idx, last_idx]  # [B, D_lm]
            q = last_h @ prompt_proj_w.T  # [B, D_router]
            q = F.normalize(q, p=2, dim=-1)
            embs.append(q.detach().cpu().float().numpy().astype(np.float32))

    prompt_emb_unit = np.concatenate(embs, axis=0)
    print(f"  Computed CARvE prompt embeddings (fresh): {prompt_emb_unit.shape}")

    # Stage A: order models within each domain by closeness to domain centroid in model-embedding space.
    domain_model_order: Dict[str, List[str]] = {}
    for d in domains:
        # Collect unique models in this domain that exist in registry.
        models_in_domain = []
        seen = set()
        for idx_ex in by_domain[d]:
            mn = apibench_examples[idx_ex].get("model_name", "unknown") or "unknown"
            if mn in seen:
                continue
            seen.add(mn)
            m_idx = norm2idx.get(normalize_model_name(mn))
            if m_idx is None:
                continue
            models_in_domain.append(mn)

        if not models_in_domain:
            domain_model_order[d] = []
            continue

        m_indices = [norm2idx[normalize_model_name(mn)] for mn in models_in_domain]
        V = model_emb_unit[m_indices]  # [m, D]
        c = V.mean(axis=0)
        c = c / (np.linalg.norm(c) + 1e-8)
        scores = V @ c
        order = [mn for mn, _ in sorted(zip(models_in_domain, scores.tolist()), key=lambda x: -x[1])]
        domain_model_order[d] = order

    # Stage B + selection per domain.
    selected_indices: List[int] = []
    for d in domains:
        quota = quotas.get(d, 0)
        if quota <= 0:
            continue

        models_order = domain_model_order.get(d, [])
        if not models_order:
            # Fallback: random within domain.
            picked = random.sample(by_domain[d], min(quota, len(by_domain[d])))
            selected_indices.extend(picked)
            continue

        # Precompute per (domain, model) ordered lists (herding) lazily.
        per_pair_order: Dict[str, List[int]] = {}

        def _get_pair_order(model_name: str) -> List[int]:
            if model_name in per_pair_order:
                return per_pair_order[model_name]
            ex_ids = by_domain_model.get((d, model_name), [])
            if not ex_ids:
                per_pair_order[model_name] = []
                return []
            X = prompt_emb_unit[ex_ids]  # [n, D]
            mu = X.mean(axis=0)
            mu_unit = mu / (np.linalg.norm(mu) + 1e-8)
            # Only compute as much of the herding order as we could possibly need in this domain.
            # This avoids quadratic-time full ordering for large (domain, model) clusters.
            k_need = min(len(ex_ids), quota)
            order_local = _herding_order(X, k=k_need, mu_unit=mu_unit)
            # Store as global example indices in selected order.
            ordered_global = [ex_ids[i] for i in order_local]
            per_pair_order[model_name] = ordered_global
            return ordered_global

        domain_selected: List[int] = []
        used = set()
        # Pointer into each model's ordered examples.
        ptrs = {mn: 0 for mn in models_order}

        # Pass 1: round-robin one prompt/model/sweep up to max_per_model.
        # This preserves broad model coverage before concentrating on top models.
        made_progress = True
        while len(domain_selected) < quota and made_progress:
            made_progress = False
            for mn in models_order:
                if len(domain_selected) >= quota:
                    break
                ordered = _get_pair_order(mn)
                if not ordered:
                    continue
                p = ptrs.get(mn, 0)
                cap = min(max_per_model, len(ordered))
                if p >= cap:
                    continue
                idx_ex = ordered[p]
                ptrs[mn] = p + 1
                if idx_ex in used:
                    continue
                used.add(idx_ex)
                domain_selected.append(idx_ex)
                made_progress = True

        # Pass 2: if still short, continue taking next prompts per model (soft cap).
        if len(domain_selected) < quota:
            made_progress = True
            while len(domain_selected) < quota and made_progress:
                made_progress = False
                for mn in models_order:
                    if len(domain_selected) >= quota:
                        break
                    ordered = _get_pair_order(mn)
                    p = ptrs.get(mn, 0)
                    if p >= len(ordered):
                        continue
                    idx_ex = ordered[p]
                    ptrs[mn] = p + 1
                    if idx_ex in used:
                        continue
                    used.add(idx_ex)
                    domain_selected.append(idx_ex)
                    made_progress = True

        # Final fallback: if still short (e.g., missing registry entries), fill randomly.
        if len(domain_selected) < quota:
            remaining = [i for i in by_domain[d] if i not in used]
            if remaining:
                needed = quota - len(domain_selected)
                domain_selected.extend(random.sample(remaining, min(needed, len(remaining))))

        selected_indices.extend(domain_selected[:quota])

        # Lightweight diagnostics for concentration / coverage by domain.
        by_model_count: Dict[str, int] = defaultdict(int)
        for idx_ex in domain_selected[:quota]:
            m = apibench_examples[idx_ex].get("model_name", "unknown") or "unknown"
            by_model_count[m] += 1
        selected_unique = len(by_model_count)
        top_share = 0.0
        if by_model_count:
            top_share = max(by_model_count.values()) / float(len(domain_selected[:quota]))
        print(
            f"  [Herding][{d}] selected={len(domain_selected[:quota])}/{quota} "
            f"unique_models={selected_unique}/{len(set(models_order))} "
            f"top1_share={top_share:.2f}"
        )

    # Deduplicate globally while preserving order.
    seen = set()
    unique_selected: List[int] = []
    for idx in selected_indices:
        if idx not in seen:
            seen.add(idx)
            unique_selected.append(idx)

    # Final trim to exact budget if needed (should be rare after quota adjustment).
    if len(unique_selected) > total_budget:
        unique_selected = unique_selected[:total_budget]

    result = [apibench_examples[i] for i in unique_selected]
    print(f"\n  Final replay size: {len(result)}")
    return result

