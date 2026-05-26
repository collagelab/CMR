"""
Router training helper functions for NeighborConsistencySFTTrainer.

This module provides the integration logic for semantic batching + candidate-set
routing loss training.
"""

from typing import Dict, List, Optional, Any, Tuple, Union
import torch
import torch.nn.functional as F
import random

from ..model_selection_carve import (
    ModelRegistry,
    CandidateSetBuilder,
    HardNegativeMiner,
    RouterModel,
)
from ..model_selection_carve.router import extract_prompt_mask
from ..model_selection_carve.model_registry import normalize_model_name
from .router_similarity_utils import build_taxonomy_soft_graph
from .router_constants import (
    DEFAULT_ROUTER_EPS,
    DEFAULT_ROUTER_TAU,
    DEFAULT_GRAPH_TAU,
    DEFAULT_GRAPH_TAU_TARGET,
    DEFAULT_GRAPH_ALPHA_DOMAIN,
    DEFAULT_MAX_GRAPH_MODELS,
)
from .router_exceptions import (
    CandidateValidationError,
    LabelAlignmentError,
)

def filter_and_validate_candidates(
    candidates_list: List[List[int]],
    gold_model_names: List[str],
    gold_indices: List[int],
    registry: ModelRegistry,
    K_total: int,
    debug: bool = False,
) -> Tuple[List[List[int]], List[int], Dict[str, Any]]:
    """
    Filter and validate candidate sets with strict hygiene checks.
    
    Performs:
    1. Remove invalid model IDs (not in registry)
    2. Remove malformed entries (empty strings, None, "huggingface.co/" patterns)
    3. Remove duplicates while preserving order
    4. Ensure gold model is present (re-insert at position 0 if needed)
    5. Enforce K_total size (pad or truncate)
    
    Args:
        candidates_list: List of candidate lists (one per example) [B, variable_K]
        gold_model_names: List of gold model names (strings) [B]
        gold_indices: List of gold model indices [B]
        registry: ModelRegistry instance
        K_total: Target candidate set size
        debug: Whether to print debug info
    
    Returns:
        Tuple of:
        - filtered_candidates: List[List[int]] of filtered candidates [B, K_total]
        - updated_gold_indices: List[int] updated gold indices (may change if gold was re-inserted)
        - stats: Dict with filtering statistics
    """
    batch_size = len(candidates_list)
    # Valid universe is index range [0, registry_size)
    registry_size = len(registry)
    valid_universe = set(range(registry_size))
    stats = {
        "total_candidates_before": 0,
        "invalid_removed": 0,
        "duplicates_removed": 0,
        "gold_reinserted": 0,
        "padded": 0,
        "truncated": 0,
    }
    
    filtered_candidates = []
    updated_gold_indices = []
    

    valid_index_mask = [False] * registry_size
    all_valid_indices = []
    for idx in range(registry_size):
        model_name = registry.idx2model.get(idx, None)
        is_valid = (
            isinstance(model_name, str)
            and len(model_name) > 0
            and "huggingface.co/" not in model_name.lower()
        )
        valid_index_mask[idx] = is_valid
        if is_valid:
            all_valid_indices.append(idx)
    for i in range(batch_size):
        candidates_i = candidates_list[i]
        gold_model = gold_model_names[i]
        gold_idx = gold_indices[i]
        
        stats["total_candidates_before"] += len(candidates_i)
        
        # Step 1: Filter to valid indices only
        valid_candidates = []
        for c_idx in candidates_i:
            # Check if index is valid
            if c_idx < 0 or c_idx >= registry_size:
                stats["invalid_removed"] += 1
                continue

            if not valid_index_mask[c_idx]:
                stats["invalid_removed"] += 1
                continue
            
            valid_candidates.append(c_idx)
        
        # Step 2: Remove duplicates while preserving order
        seen = set()
        deduped = []
        for c_idx in valid_candidates:
            if c_idx not in seen:
                deduped.append(c_idx)
                seen.add(c_idx)
            else:
                stats["duplicates_removed"] += 1
        
        # Step 3: Ensure gold model is present
        gold_present = gold_idx in deduped
        if not gold_present:
            # Re-insert gold at position 0.
            # Gold validity is index-based only; name-shape hygiene checks apply to negatives.
            if 0 <= gold_idx < registry_size:
                deduped.insert(0, gold_idx)
                stats["gold_reinserted"] += 1
                if debug:
                    print(f"  [Candidate Hygiene] Example {i}: Re-inserted gold model {gold_model} (idx={gold_idx}) at position 0")
            else:
                # Gold model itself is invalid - this is a critical error
                raise ValueError(
                    f"CRITICAL: Gold model '{gold_model}' (idx={gold_idx}) is not in valid universe! "
                    f"Valid range: [0, {registry_size}), model in registry: {gold_model in registry.model2idx}. "
                    f"This indicates a data integrity issue - the gold model must exist in the registry."
                )
        else:
            # Move gold to position 0 if not already there
            if deduped[0] != gold_idx:
                deduped.remove(gold_idx)
                deduped.insert(0, gold_idx)
        
        # Step 4: Enforce K_total size
        if len(deduped) < K_total:
            # Pad with valid negatives not already in candidates.
            # Fast path: rejection-sample for small deficits to avoid rebuilding
            # large availability lists for every example.
            deduped_set = set(deduped)
            needed = K_total - len(deduped)

            if needed > 0:
                max_unique_available = len(all_valid_indices) - len(deduped_set)
                if max_unique_available < needed:
                    available = [idx for idx in all_valid_indices if idx not in deduped_set]
                    random.shuffle(available)
                    deduped.extend(available[:needed])
                    stats["padded"] += min(needed, len(available))
                elif needed <= 8:
                    added = 0
                    attempts = 0
                    max_attempts = max(64, needed * 32)
                    while added < needed and attempts < max_attempts:
                        attempts += 1
                        pick = random.choice(all_valid_indices)
                        if pick not in deduped_set:
                            deduped.append(pick)
                            deduped_set.add(pick)
                            added += 1
                    if added < needed:
                        available = [idx for idx in all_valid_indices if idx not in deduped_set]
                        random.shuffle(available)
                        extra = available[:needed - added]
                        deduped.extend(extra)
                        added += len(extra)
                    stats["padded"] += added
                else:
                    available = [idx for idx in all_valid_indices if idx not in deduped_set]
                    random.shuffle(available)
                    deduped.extend(available[:needed])
                    stats["padded"] += needed
        elif len(deduped) > K_total:
            # Truncate but never drop gold (at position 0)
            deduped = [deduped[0]] + deduped[1:K_total]
            stats["truncated"] += len(candidates_i) - K_total
        
        # Final validation with proper error messages
        if len(deduped) != K_total:
            raise CandidateValidationError(
                f"Example {i}: Expected {K_total} candidates after filtering, got {len(deduped)}. "
                f"Gold model: {gold_model} (idx={gold_idx})"
            )
        if deduped[0] != gold_idx:
            raise CandidateValidationError(
                f"Example {i}: Gold model must be at index 0, got {deduped[0]} instead of {gold_idx}. "
                f"Gold model: {gold_model}"
            )
        if len(set(deduped)) != K_total:
            duplicates = [c for c in deduped if deduped.count(c) > 1]
            raise CandidateValidationError(
                f"Example {i}: Duplicates found in final candidate set! Duplicates: {duplicates}"
            )
        if not all(c in valid_universe for c in deduped):
            invalid = [c for c in deduped if c not in valid_universe]
            raise CandidateValidationError(
                f"Example {i}: Invalid candidates in final set: {invalid}. "
                f"Valid range: [0, {registry_size})"
            )
        
        filtered_candidates.append(deduped)
        updated_gold_indices.append(gold_idx)

    return filtered_candidates, updated_gold_indices, stats


def check_label_candidate_alignment(
    candidates_list: List[List[int]],
    gold_model_names: List[str],
    gold_indices: List[int],
    registry: ModelRegistry,
    debug: bool = False,
) -> None:
    """
    Strict assertion: For each example i, candidates_i[gold_index_i] == gold_model.
    
    This is a critical check - if misaligned, the router cannot learn correctly.
    
    Args:
        candidates_list: List of candidate lists [B, K]
        gold_model_names: List of gold model names (strings) [B]
        gold_indices: List of gold indices (should be 0 for all, since positive is at index 0)
        registry: ModelRegistry instance
        debug: Whether to print detailed error messages
    
    Raises:
        AssertionError: If alignment fails
    """
    batch_size = len(candidates_list)
    
    for i in range(batch_size):
        gold_model = gold_model_names[i]
        gold_index_i = gold_indices[i]  # Should be 0 (positive at index 0)
        candidates_i = candidates_list[i]
        
        # Check: candidates_i[gold_index_i] should map to gold_model
        # Note: gold_index_i should always be 0 (positive at index 0)
        if gold_index_i >= len(candidates_i):
            error_msg = (
                f"Example {i}: gold_index_i ({gold_index_i}) >= len(candidates_i) ({len(candidates_i)}). "
                f"Gold model: {gold_model}. This indicates a critical alignment issue."
            )
            raise LabelAlignmentError(error_msg)
        
        candidate_at_gold_idx = candidates_i[gold_index_i]
        candidate_model_name = registry.idx2model.get(candidate_at_gold_idx, f"unknown_idx_{candidate_at_gold_idx}")
        
        # Get gold model index from registry with normalized lookup (case-insensitive)
        gold_model_idx = None
        if gold_model in registry.model2idx:
            # Fast path: direct lookup
            gold_model_idx = registry.model2idx[gold_model]
        else:
            # Normalized lookup (handles case/normalization mismatches)
            normalized_gold = normalize_model_name(gold_model)
            for existing_name, idx in registry.model2idx.items():
                if normalize_model_name(existing_name) == normalized_gold:
                    gold_model_idx = idx
                    break
        
        # Critical check: candidate at gold_index must match gold_model
        if candidate_at_gold_idx != gold_model_idx:
            # Check if gold_model is present elsewhere in candidates
            gold_present = gold_model_idx in candidates_i if gold_model_idx is not None else False
            gold_present_at = candidates_i.index(gold_model_idx) if gold_present else None
            
            error_msg = (
                f"Example {i}: Label-candidate misalignment. "
                f"candidates_i[{gold_index_i}] = {candidate_at_gold_idx} ('{candidate_model_name}'), "
                f"but gold_model = '{gold_model}' (idx={gold_model_idx}). "
            )
            
            if gold_present:
                error_msg += f"Gold model IS present in candidates at index {gold_present_at}, but not at expected position {gold_index_i}."
            else:
                error_msg += f"Gold model is NOT present in candidates_i. First 10 candidates: {candidates_i[:10]}"
            
            raise LabelAlignmentError(error_msg)
def compute_routing_loss(
    router_model: RouterModel,
    hidden_states: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: torch.Tensor,
    candidate_indices: torch.Tensor,
    prompt_len: torch.Tensor,
    use_soft_targets: bool = False,
    soft_target_eps: float = 0.1,
    neighbor_indices: Optional[List[List[int]]] = None,
    neighbor_positions: Optional[List[List[int]]] = None,
    device: Optional[torch.device] = None,
    return_accuracy: bool = False,
    debug: bool = False,
    global_step: int = 0,
    return_per_example: bool = False,  # If True, return per-example losses [B] instead of scalar
) -> Union[Tuple[torch.Tensor, Optional[Dict[str, float]]], Tuple[torch.Tensor, torch.Tensor, Optional[Dict[str, float]]]]:
    """
    Compute routing loss with hard or soft targets.
    
    Args:
        router_model: Router module
        hidden_states: LM hidden states [B, seq_len, D]
        labels: Labels tensor [B, seq_len] (used for verification only)
        attention_mask: Attention mask [B, seq_len]
        candidate_indices: Candidate model indices [B, K]
        prompt_len: Per-example prompt length [B] - boundary between prompt and completion
        use_soft_targets: Whether to use soft targets with graph neighbors
        soft_target_eps: Mass to distribute to neighbors (1-eps on positive)
        neighbor_indices: List of neighbor model indices per example (for soft targets)
        device: Device for tensors
        return_accuracy: If True, also return accuracy metrics
        debug: Whether to enable detailed debugging
        global_step: Current global step (for debug filtering)
    
    Returns:
        Tuple of (loss, metrics_dict) where metrics_dict contains accuracy if return_accuracy=True
    """
    if device is None:
        device = hidden_states.device
    
    batch_size = hidden_states.size(0)
    K = candidate_indices.size(1)
    # Extract prompt mask using explicit prompt_len boundary
    prompt_mask = extract_prompt_mask(
        prompt_len, 
        attention_mask, 
        labels=labels if debug else None,  # Pass labels for verification in debug mode
        debug=debug, 
        global_step=global_step
    )
    
    # Compute logits (with compute tracking)
    result = router_model(
        hidden_states, 
        prompt_mask, 
        candidate_indices, 
        return_compute_metrics=True
    )
    if isinstance(result, tuple):
        logits, compute_metrics = result
    else:
        logits = result
        compute_metrics = None
    
    # Compute loss (hard or soft targets)
    # Use 'none' reduction if per-example losses are needed, else 'mean'
    reduction = 'none' if return_per_example else 'mean'
    loss = _compute_routing_loss_core(
        logits=logits,
        candidate_indices=candidate_indices,
        use_soft_targets=use_soft_targets,
        soft_target_eps=soft_target_eps,
        neighbor_indices=neighbor_indices,
        neighbor_positions=neighbor_positions,
        batch_size=batch_size,
        K=K,
        device=device,
        reduction=reduction,
    )
    
    # If per-example losses requested, return them separately
    if return_per_example:
        loss_per_example = loss  # [B]
        loss_mean = loss_per_example.mean()  # Scalar for metrics
    else:
        loss_mean = loss  # Scalar
        loss_per_example = None
    

    # Compute accuracy metrics if requested
    metrics = None
    if return_accuracy:
        with torch.no_grad():
            # Compute top-1 accuracy (positive model is at index 0)
            predictions = torch.argmax(logits, dim=-1)  # [B]
            correct = (predictions == 0).float()
            top1_accuracy = correct.mean().item()
            
            # Compute top-k accuracies
            top3_predictions = torch.topk(logits, k=min(3, K), dim=-1).indices  # [B, 3]
            top3_accuracy = (top3_predictions == 0).any(dim=-1).float().mean().item()
            
            top5_predictions = torch.topk(logits, k=min(5, K), dim=-1).indices  # [B, 5]
            top5_accuracy = (top5_predictions == 0).any(dim=-1).float().mean().item()
            
            # Average rank of positive model
            sorted_indices = torch.argsort(logits, dim=-1, descending=True)  # [B, K]
            positive_ranks = (sorted_indices == 0).nonzero(as_tuple=True)[1].float()  # Rank of index 0
            avg_rank = positive_ranks.mean().item() + 1  # +1 for 1-indexed rank
            
            # Average score of positive vs negative
            positive_scores = logits[:, 0]  # [B]
            negative_scores = logits[:, 1:].mean(dim=-1)  # [B]
            avg_positive_score = positive_scores.mean().item()
            avg_negative_score = negative_scores.mean().item()
            score_margin = avg_positive_score - avg_negative_score
            
            metrics = {
                "top1_accuracy": top1_accuracy,
                "top3_accuracy": top3_accuracy,
                "top5_accuracy": top5_accuracy,
                "avg_positive_rank": avg_rank,
                "avg_positive_score": avg_positive_score,
                "avg_negative_score": avg_negative_score,
                "score_margin": score_margin,
            }
    
    # Add compute metrics if available
    if compute_metrics is not None and metrics is not None:
        metrics["compute"] = compute_metrics
    elif compute_metrics is not None:
        metrics = {"compute": compute_metrics}

    if return_per_example:
        return loss_mean, loss_per_example, metrics
    else:
        return loss_mean, metrics


def _compute_routing_loss_core(
    logits: torch.Tensor,
    candidate_indices: torch.Tensor,
    use_soft_targets: bool,
    soft_target_eps: float,
    neighbor_indices: Optional[List[List[int]]],
    neighbor_positions: Optional[List[List[int]]],
    batch_size: int,
    K: int,
    device: torch.device,
    reduction: str = 'mean',
) -> torch.Tensor:
    """
    Core routing loss computation (hard or soft targets).
    
    Args:
        logits: Router logits [B, K]
        candidate_indices: Candidate model indices [B, K]
        use_soft_targets: Whether to use soft targets
        soft_target_eps: Epsilon for soft targets
        neighbor_indices: List of neighbor indices per example
        batch_size: Batch size
        K: Candidate set size
        device: Device for tensors
        reduction: 'mean' or 'none' (for per-example losses)
    
    Returns:
        Loss tensor (scalar if reduction='mean', [B] if reduction='none')
    """
    if use_soft_targets and neighbor_indices is not None:
        # Soft targets: distribute eps mass to graph neighbors
        soft_targets = torch.zeros(batch_size, K, dtype=torch.float32, device=device)
        soft_targets[:, 0] = 1 - soft_target_eps  # Mass on positive (always at index 0)
        
        # Distribute eps to neighbors present in candidates
        for i in range(batch_size):
            if neighbor_positions is not None and i < len(neighbor_positions):
                positions_i = neighbor_positions[i]
                if positions_i:
                    soft_targets[i, positions_i] = soft_target_eps / len(positions_i)
                continue

            if i >= len(neighbor_indices):
                continue

            neighbors = neighbor_indices[i]
            if not neighbors:
                continue

            # Backward-compatible fallback path when precomputed positions are unavailable.
            neighbors_tensor = torch.tensor(neighbors, dtype=candidate_indices.dtype, device=device)
            candidates_i = candidate_indices[i, 1:]  # Exclude positive
            neighbor_mask = torch.isin(candidates_i, neighbors_tensor)

            num_neighbors = neighbor_mask.sum().item()
            if num_neighbors > 0:
                positions_i = torch.where(neighbor_mask)[0] + 1
                soft_targets[i, positions_i] = soft_target_eps / num_neighbors
        
        # Compute soft-target cross-entropy
        log_probs = F.log_softmax(logits, dim=-1)  # [B, K]
        loss_per_example = -(soft_targets * log_probs).sum(dim=-1)  # [B]
        if reduction == 'mean':
            return loss_per_example.mean()
        else:
            return loss_per_example
    else:
        # Hard targets: positive always at index 0
        targets = torch.zeros(batch_size, dtype=torch.long, device=device)
        if reduction == 'none':
            return F.cross_entropy(logits, targets, reduction='none')
        else:
            return F.cross_entropy(logits, targets)


def compute_label_graph_regularizer(
    router_model: RouterModel,
    candidate_indices: torch.Tensor,
    registry: ModelRegistry,
    tau: float = DEFAULT_GRAPH_TAU,
    tau_target: float = DEFAULT_GRAPH_TAU_TARGET,
    alpha_domain: float = DEFAULT_GRAPH_ALPHA_DOMAIN,
    max_models: int = DEFAULT_MAX_GRAPH_MODELS,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Compute label-side graph regularizer (X-CLR style on model embeddings).
    
    Aligns learned model embeddings with taxonomy structure by matching
    predicted similarities with graph-derived target similarities.
    
    Args:
        router_model: Router module
        candidate_indices: Candidate model indices [B, K]
        registry: ModelRegistry instance
        tau: Temperature for predicted similarities
        tau_target: Temperature for target similarities
        alpha_domain: Similarity weight for same-domain pairs
        max_models: Maximum models for graph (subsample if exceeds)
        device: Device for tensors
    
    Returns:
        Graph regularization loss
    """
    if device is None:
        device = candidate_indices.device
    
    # Get union of candidates across batch
    all_candidate_ids = torch.unique(candidate_indices)  # [M]
    
    # Subsample if too large (avoids O(M^2) explosion)
    if len(all_candidate_ids) > max_models:
        indices = torch.randperm(len(all_candidate_ids), device=device)[:max_models]
        all_candidate_ids = all_candidate_ids[indices]
    
    M = len(all_candidate_ids)
    if M < 2:
        # Need at least 2 models for graph
        return torch.tensor(0.0, device=device)
    
    # Build taxonomy graph among candidates
    candidate_models = [registry.idx2model[idx.item()] for idx in all_candidate_ids]
    candidate_domains = [registry.metadata[idx.item()]['domain'] for idx in all_candidate_ids]
    
    G_target = build_taxonomy_soft_graph(
        batch_model_ids=candidate_models,
        batch_domains=candidate_domains,
        alpha_domain=alpha_domain,
        device=device,
    )  # [M, M]
    
    # Compute similarities among model embeddings
    M_emb = router_model.get_model_embeddings(all_candidate_ids)  # [M, D]
    M_norm = F.normalize(M_emb, p=2, dim=-1)
    S_pred = torch.mm(M_norm, M_norm.T) / tau  # [M, M]
    
    # Target logits
    T_logits = G_target / tau_target  # [M, M]
    
    # Mask diagonal to -inf to exclude self-similarity from distributions
    # This prevents diagonal dominance (self-sim=1 dominates softmax)
    mask = torch.eye(M, device=device, dtype=torch.bool)
    S_pred = S_pred.masked_fill(mask, float("-inf"))
    T_logits = T_logits.masked_fill(mask, float("-inf"))
    
    # Match distributions (row-wise KL divergence over other models only)
    loss_graph = F.kl_div(
        F.log_softmax(S_pred, dim=-1),
        F.softmax(T_logits, dim=-1),
        reduction='batchmean'
    )
    
    return loss_graph


def compute_router_metrics(
    logits: torch.Tensor,
    candidate_indices: torch.Tensor,
    candidate_builder: Optional[CandidateSetBuilder] = None,
    y_indices: Optional[List[int]] = None,
    domains: Optional[List[str]] = None,
    hard_negative_cache: Optional[Dict] = None,
) -> Dict[str, Any]:
    """
    Compute router metrics for logging.
    
    Args:
        logits: Router logits [B, K]
        candidate_indices: Candidate model indices [B, K]
        candidate_builder: Optional CandidateSetBuilder for composition stats
        y_indices: Positive model indices (for composition stats)
        domains: Domains (for composition stats)
        hard_negative_cache: Hard negative cache (for composition stats)
    
    Returns:
        Dictionary of metrics
    """
    batch_size, K = logits.shape
    device = logits.device
    
    # Positive is always at index 0
    targets = torch.zeros(batch_size, dtype=torch.long, device=device)
    
    # Top-1 accuracy
    preds = logits.argmax(dim=-1)
    top1_acc = (preds == targets).float().mean().item()
    
    # Top-5 accuracy
    top5_preds = logits.topk(k=min(5, K), dim=-1)[1]
    top5_acc = (top5_preds == targets.unsqueeze(1)).any(dim=-1).float().mean().item()
    
    metrics = {
        "top1_in_candidates": top1_acc,
        "top5_in_candidates": top5_acc,
        "num_anchors": batch_size,  # Number of examples in batch
    }
    
    # Candidate composition stats (if builder available)
    if candidate_builder and y_indices and domains:
        total_positive = 0
        total_hard = 0
        total_semantic = 0
        total_far = 0
        total_other = 0
        
        for i in range(min(batch_size, len(y_indices), len(domains))):
            candidates = candidate_indices[i].cpu().tolist()
            stats = candidate_builder.get_composition_stats(
                candidates=candidates,
                y_idx=y_indices[i],
                domain=domains[i],
                hard_negative_cache=hard_negative_cache,
            )
            total_positive += stats.get("positive", 0)
            total_hard += stats.get("hard", 0)
            total_semantic += stats.get("semantic", 0)
            total_far += stats.get("far", 0)
            total_other += stats.get("other", 0)
        
        total = total_positive + total_hard + total_semantic + total_far + total_other
        if total > 0:
            metrics["candidate_comp_positive"] = total_positive / total
            metrics["candidate_comp_hard"] = total_hard / total
            metrics["candidate_comp_semantic"] = total_semantic / total
            metrics["candidate_comp_far"] = total_far / total
            metrics["candidate_comp_other"] = total_other / total
            # Also store counts for detailed logging
            metrics["candidate_count_positive"] = total_positive
            metrics["candidate_count_hard"] = total_hard
            metrics["candidate_count_semantic"] = total_semantic
            metrics["candidate_count_far"] = total_far
            metrics["candidate_count_other"] = total_other
            metrics["candidate_count_total"] = total
    
    # Hard negative hit rate (if cache available)
    if hard_negative_cache and y_indices and domains:
        hits = 0
        for i in range(min(batch_size, len(y_indices), len(domains))):
            y_idx = y_indices[i]
            domain = domains[i]
            cache_key = (domain.strip().lower() if isinstance(domain, str) else "unknown", y_idx)
            if cache_key in hard_negative_cache:
                hits += 1
        metrics["hard_negative_hit_rate"] = hits / batch_size if batch_size > 0 else 0.0
    
    return metrics

