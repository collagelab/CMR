"""
Router evaluation: predict model selections using the trained router.

This module evaluates the router's ability to select the correct model for each test prompt.
"""

import torch
import torch.nn.functional as F
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import json
from collections import defaultdict
import numpy as np
import hashlib

from .model_selection_carve import ModelRegistry, RouterModel
from .model_selection_carve.candidates import CandidateSetBuilder
from .model_selection_carve.model_registry import normalize_domain
from .openmodel_carve import LoRAModelManager


def tensor_sha(t: torch.Tensor) -> str:
    """
    Compute SHA1 hash of tensor bytes (float32 CPU representation).
    
    Args:
        t: Input tensor
    
    Returns:
        First 8 hex characters of SHA1 hash
    """
    # Convert to float32 CPU and get bytes
    t_bytes = t.float().cpu().numpy().tobytes()
    return hashlib.sha1(t_bytes).hexdigest()[:8]


def format_eval_prompt(
    ex: Dict[str, Any],
    tokenizer: Any,
    eval_use_chat_template: bool = False,
    system_prompt: str = "",
    model_card: str = "",
) -> str:
    """
    Format evaluation prompt to match training format.
    
    Args:
        ex: Example dict with 'prompt_text' or 'instruction'
        tokenizer: Tokenizer (for chat template if enabled)
        eval_use_chat_template: If True, use tokenizer.apply_chat_template
        system_prompt: System prompt to prepend (if not using chat template)
        model_card: Model card/retriever info to append (if not using chat template)
    
    Returns:
        Formatted prompt string
    """
    # Extract prompt text
    prompt_text = ex.get("prompt_text") or ex.get("instruction", "")
    
    if eval_use_chat_template and hasattr(tokenizer, "apply_chat_template"):
        # Use chat template (if available and enabled)
        # This matches training if training used chat templates
        messages = [
            {"role": "system", "content": system_prompt} if system_prompt else None,
            {"role": "user", "content": prompt_text + model_card}
        ]
        messages = [m for m in messages if m is not None]
        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return formatted
    else:
        # Default: use string concatenation (matches training format)
        # Training format: system_prompt + prompt + model_card + "\n###Response:"
        full_prompt = system_prompt + prompt_text + model_card + "\n###Response:"
        return full_prompt


def compute_registry_fingerprint(model_registry: ModelRegistry) -> str:
    """
    Compute stable fingerprint of registry ordering.
    
    Args:
        model_registry: ModelRegistry instance
    
    Returns:
        First 12 hex characters of SHA1 hash
    """
    # Build ordered list of model names by index
    num_models = len(model_registry)
    idx2model_list = [model_registry.idx2model[i] for i in range(num_models)]
    registry_str = "\n".join(idx2model_list)
    return hashlib.sha1(registry_str.encode()).hexdigest()[:12]


def load_trained_router(
    checkpoint_dir: Path,
    device: str = "cuda",
    strict: bool = True,
    num_models_override: Optional[int] = None,
) -> RouterModel:
    """
    Load a trained RouterModel robustly.

    Expected files:
      - router_config.json   (saved at train time)
      - router_model.pt      (state_dict)

    Supports registry expansion via num_models_override by resizing model_embeddings.
    
    Args:
        checkpoint_dir: Directory containing the router checkpoint
        device: Device to load the router on
        strict: If True, use strict=True for load_state_dict (default: True for evaluation)
        num_models_override: Optional override for num_models if registry evolved since training
    
    Returns:
        Loaded RouterModel with exact training-time architecture parameters
    
    Raises:
        FileNotFoundError: If router checkpoint or config files are missing
        KeyError: If checkpoint is missing required keys
        ValueError: If embedding dimensions don't match
    """
    router_path = checkpoint_dir / "router_model.pt"
    cfg_path = checkpoint_dir / "router_config.json"

    if not router_path.exists():
        raise FileNotFoundError(f"No router checkpoint found at {router_path}")
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"Missing router_config.json at {cfg_path}. "
            f"Save config during training to ensure evaluation parity."
        )

    cfg = json.loads(cfg_path.read_text())

    # Use trained config; optionally override num_models if registry evolved
    num_models_ckpt = int(cfg["num_models"])
    num_models = int(num_models_override) if num_models_override is not None else num_models_ckpt

    # Load all architecture parameters from config
    # Handle backward compatibility: lm_hidden_size may be missing in older checkpoints
    lm_hidden_size = int(cfg.get("lm_hidden_size", 4096))  # Default to 4096 if missing
    
    router = RouterModel(
        num_models=num_models,
        embedding_dim=int(cfg["embedding_dim"]),
        lm_hidden_size=lm_hidden_size,
        tau=float(cfg["tau"]),
        pooling=cfg["pooling"],
    ).to(device)

    # Load state dict with safer options (weights_only if available)
    try:
        # PyTorch >= 2.0 supports weights_only for security
        state = torch.load(router_path, map_location=device, weights_only=True)
    except TypeError:
        # Fallback for older PyTorch versions
        state = torch.load(router_path, map_location=device)

    # Handle model_embeddings resize if num_models differs.
    if num_models != num_models_ckpt:
        if "model_embeddings.weight" not in state:
            raise KeyError("Checkpoint missing model_embeddings.weight; cannot resize safely.")

        old_w = state["model_embeddings.weight"]  # [num_models_ckpt, D]
        new_w = router.model_embeddings.weight    # [num_models, D]
        if old_w.shape[1] != new_w.shape[1]:
            raise ValueError(
                f"Embedding dim mismatch: ckpt {old_w.shape} vs model {new_w.shape}"
            )

        overlap = min(num_models, num_models_ckpt)
        with torch.no_grad():
            new_w[:overlap].copy_(old_w[:overlap])

        # Exclude resized embedding tensor from state_dict load.
        state = {k: v for k, v in state.items() if k != "model_embeddings.weight"}

    router.load_state_dict(state, strict=strict)
    del state

    router.eval()
    return router


def compute_domain_scores(
    logits_all: torch.Tensor,
    domain_to_indices_tensor: Dict[str, torch.Tensor],
    mode: str = "logsumexp",
    topk: int = 10,
    alpha: float = 0.5,
) -> Dict[str, torch.Tensor]:
    """
    Compute domain scores by aggregating model logits within each domain.
    
    Supports multiple aggregation strategies:
    - "logsumexp": logsumexp over all models in domain (default, stable aggregation)
    - "max": maximum logit in domain
    - "topk_logsumexp": logsumexp over top-k models in domain
    - "hybrid": alpha * max + (1-alpha) * logsumexp(topk)
    
    Args:
        logits_all: Router logits for all models [num_models]
        domain_to_indices_tensor: Mapping from domain name to tensor of model indices (on same device as logits_all)
        mode: Scoring mode ("logsumexp", "max", "topk_logsumexp", "hybrid")
        topk: Number of top models to use for topk_logsumexp or hybrid (default: 10)
        alpha: Weight for max in hybrid mode (default: 0.5)
    
    Returns:
        Dictionary mapping domain name to aggregated score (tensor)
    """
    domain_scores = {}
    for domain, idx in domain_to_indices_tensor.items():
        if idx.numel() == 0:
            continue
        dom_logits = logits_all.index_select(0, idx)

        if mode == "max":
            score = dom_logits.max()

        elif mode == "logsumexp":
            score = torch.logsumexp(dom_logits, dim=0)

        elif mode == "topk_logsumexp":
            k = min(topk, dom_logits.numel())
            topk_vals = torch.topk(dom_logits, k=k, largest=True).values
            score = torch.logsumexp(topk_vals, dim=0)

        elif mode == "hybrid":
            k = min(topk, dom_logits.numel())
            topk_vals = torch.topk(dom_logits, k=k, largest=True).values
            lse_topk = torch.logsumexp(topk_vals, dim=0)
            mx = dom_logits.max()
            score = alpha * mx + (1.0 - alpha) * lse_topk

        else:
            raise ValueError(f"Unknown hier domain score mode: {mode}")

        domain_scores[domain] = score
    return domain_scores


def hierarchical_rerank_topN(
    logits_all: torch.Tensor,
    restricted_indices: List[int],
    gold_model_idx: int,
    device: torch.device,
) -> Tuple[bool, bool, bool]:
    """
    Re-rank models within a restricted set and compute accuracy metrics.
    
    Args:
        logits_all: Router logits for all models [num_models]
        restricted_indices: List of model indices in the restricted set (already computed)
        gold_model_idx: Index of the gold model
        device: Device for tensors
    
    Returns:
        Tuple of (top1_correct, top5_correct, top10_correct)
    """
    # Check if gold model is in restricted set
    if gold_model_idx not in restricted_indices:
        # Gold not in restricted set - all metrics are False
        return False, False, False
    
    # Re-rank models by logits restricted to union
    restricted_indices_tensor = torch.tensor(restricted_indices, dtype=torch.long, device=device)
    restricted_logits = logits_all[restricted_indices_tensor]  # [restricted_size]
    
    # Find gold model's position in restricted set
    gold_pos_in_restricted = restricted_indices.index(gold_model_idx)
    
    # Compute ranks
    sorted_restricted = torch.argsort(restricted_logits, descending=True)
    gold_rank_restricted = (sorted_restricted == gold_pos_in_restricted).nonzero(as_tuple=True)[0].item() + 1
    
    # Compute top-k accuracies
    top1_correct = (gold_rank_restricted == 1)
    top5_correct = (gold_rank_restricted <= 5)
    top10_correct = (gold_rank_restricted <= 10)
    
    return top1_correct, top5_correct, top10_correct


def build_restricted_set(
    top_domains: List[str],
    domain_to_indices: Dict[str, List[int]],
    N: int,
    num_models: int,
) -> List[int]:
    """
    Build restricted set as union of models in top-N domains.
    If union is empty, fall back to all models.
    
    Args:
        top_domains: List of domain names (sorted by score, descending)
        domain_to_indices: Mapping from domain name to list of model indices
        N: Number of top domains to consider
        num_models: Total number of models (for fallback)
    
    Returns:
        Sorted list of model indices in restricted set
    """
    restricted_indices_set = set()
    for domain in top_domains[:N]:
        if domain in domain_to_indices:
            restricted_indices_set.update(domain_to_indices[domain])
    
    # Handle edge case: if union is empty, fall back to all models
    if not restricted_indices_set:
        restricted_indices = list(range(num_models))
    else:
        restricted_indices = sorted(list(restricted_indices_set))
    
    return restricted_indices


def evaluate_router(
    router_model: RouterModel,
    model_registry: ModelRegistry,
    lm_model: LoRAModelManager,
    test_data: List[Dict[str, Any]],
    k_values: List[int] = [1, 3, 5, 10],
    batch_size: int = 32,
    device: str = "cuda",
    debug: bool = False,
    eval_use_chat_template: bool = False,
    system_prompt: str = "",
    checkpoint_dir: Optional[Path] = None,
    max_length: int = 512,
    candidate_K: Optional[int] = None,
    router_config: Optional[Dict[str, Any]] = None,
    hierarchical_eval: bool = False,
    hierarchy_level: str = "domain",
    hierarchical_topk: int = 1,
    hier_domain_score_mode: str = "logsumexp",
    hier_domain_topk: int = 10,
    hier_domain_hybrid_alpha: float = 0.5,
    model_family_lookup: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    Evaluate router model on test data.
    
    Args:
        router_model: Trained RouterModel
        model_registry: ModelRegistry with model name → ID mappings
        lm_model: LoRAModelManager to encode prompts
        test_data: List of test examples with 'prompt_text' and 'model_name'
        k_values: List of k values for top-k accuracy computation
        batch_size: Batch size for evaluation
        device: Device for computation
        debug: Whether to enable detailed debug output
        eval_use_chat_template: If True, use tokenizer.apply_chat_template for formatting
        system_prompt: System prompt to prepend (if not using chat template)
        checkpoint_dir: Optional checkpoint directory for registry fingerprint validation
        hierarchical_eval: If True, enable two-stage hierarchical evaluation
        hierarchy_level: Level for hierarchical grouping ("domain" or "parent_group")
        hierarchical_topk: Number of top groups to consider (default: 1)
        hier_domain_score_mode: Domain scoring strategy ("logsumexp", "max", "topk_logsumexp", "hybrid", default: "logsumexp")
        hier_domain_topk: Number of top models for topk_logsumexp/hybrid modes (default: 10)
        hier_domain_hybrid_alpha: Weight for max in hybrid mode (default: 0.5)
    
    Returns:
        Dictionary of evaluation metrics (and diagnostics if debug=True)
        If hierarchical_eval=True, includes hierarchical metrics prefixed with "hier_"
    """
    # A) Enforce eval mode for both router and LM
    router_model.eval()
    if hasattr(lm_model, 'model') and hasattr(lm_model.model, 'eval'):
        lm_model.model.eval()
    elif hasattr(lm_model, 'eval'):
        lm_model.eval()
    
    # Validate registry ↔ embedding alignment (C)
    num_models = len(model_registry)
    embedding_dim = router_model.embedding_dim
    
    # Check embedding table shape matches registry
    assert router_model.model_embeddings.weight.shape[0] == num_models, \
        f"Registry has {num_models} models but embedding table has {router_model.model_embeddings.weight.shape[0]} rows"
    assert router_model.model_embeddings.weight.shape[1] == embedding_dim, \
        f"Expected embedding_dim={embedding_dim} but got {router_model.model_embeddings.weight.shape[1]}"
    
    # Compute registry fingerprint
    registry_fingerprint = compute_registry_fingerprint(model_registry)
    
    
    # Check against saved fingerprint if checkpoint_dir provided
    if checkpoint_dir is not None:
        fingerprint_path = checkpoint_dir / "registry_fingerprint.txt"
        router_config_path = checkpoint_dir / "router_config.json"
        
        saved_fingerprint = None
        if fingerprint_path.exists():
            saved_fingerprint = fingerprint_path.read_text().strip()
        elif router_config_path.exists():
            # Try to load from router_config.json if it has fingerprint
            try:
                with open(router_config_path, 'r') as f:
                    config_data = json.load(f)
                    saved_fingerprint = config_data.get("registry_fingerprint")
            except:
                pass
        
        if saved_fingerprint:
            if saved_fingerprint != registry_fingerprint:
                raise RuntimeError(
                    f"Registry fingerprint mismatch!\n"
                    f"  Saved: {saved_fingerprint}\n"
                    f"  Current: {registry_fingerprint}\n"
                    f"This indicates idx2model ordering changed. Check registry loading."
                )
    
    # Track diagnostics for return
    diagnostics = {
        "registry_fingerprint": registry_fingerprint,
        "ids_hashes": [],
        "prompt_emb_hashes": [],
        "score_vec_hashes": [],
    } if debug else None
    
    all_predictions: List[int] = []
    all_labels: List[int] = []
    all_domains: List[str] = []
    all_scores: List[List[float]] = []  # Store all scores for analysis
    
    # Track per-domain metrics
    domain_correct = defaultdict(int)
    domain_total = defaultdict(int)
    
    # Track model family accuracy
    family_correct = defaultdict(int)
    family_total = defaultdict(int)
    
    
    # Track top-k accuracies (acc_all over all models)
    topk_correct = {k: 0 for k in k_values}

    # Track score margins for comparison with training
    candidate_score_margins = []  # For candidate-set evaluation
    all_models_score_margins = []  # For all-models evaluation

    # Track candidate-set accuracies (acc_candidate over K candidates)
    cand_top1_correct = 0
    cand_total = 0
    
    # Track gold_in_registry for diagnostic
    gold_in_registry_count = 0
    total_examples_processed = 0
    
    # Track entropy for diagnostic
    all_entropies = []
    
    # Track example IDs for debug output
    example_counter = 0
    
    # Track hierarchical evaluation metrics (if enabled)
    hier_group_correct = 0 if hierarchical_eval else 0
    hier_group_total = 0 if hierarchical_eval else 0
    hier_model_topk_correct = {k: 0 for k in k_values} if hierarchical_eval else {}
    hier_model_total = 0 if hierarchical_eval else 0
    hier_e2e_top1_correct = 0 if hierarchical_eval else 0
    hier_restricted_sizes = [] if hierarchical_eval else []
    
    # Track Top-N Domain Hierarchical Rerank metrics (N=1,2,3)
    # Legacy: conditional metrics (only when gold domain in top-N)
    hier_model_top1_atN = {1: 0, 2: 0, 3: 0} if hierarchical_eval else {}
    hier_model_top5_atN = {1: 0, 2: 0, 3: 0} if hierarchical_eval else {}
    hier_model_top10_atN = {1: 0, 2: 0, 3: 0} if hierarchical_eval else {}
    hier_model_total_atN = {1: 0, 2: 0, 3: 0} if hierarchical_eval else {}  # Count of examples where gold in restricted set
    
    # New: E2E metrics (all examples, correct denominator)
    hier_model_top1_e2e_atN = {1: 0, 2: 0, 3: 0} if hierarchical_eval else {}  # E2E correct count
    hier_model_top1_cond_atN = {1: 0, 2: 0, 3: 0} if hierarchical_eval else {}  # Conditional correct count (same as hier_model_top1_atN, but renamed for clarity)
    hier_domain_included_atN = {1: 0, 2: 0, 3: 0} if hierarchical_eval else {}  # Count where gold domain ∈ predicted top-N
    
    hier_restricted_sizes_atN = {1: [], 2: [], 3: []} if hierarchical_eval else {}
    
    # Track missing gold examples (gold model not in registry)
    hier_missing_gold_count = 0 if hierarchical_eval else 0
    
    # Debug: Track why examples are filtered out (kept for backward compatibility)
    hier_gold_not_in_predicted_domain = {1: 0, 2: 0, 3: 0} if hierarchical_eval and debug else {}
    hier_gold_domain_missing_from_registry = 0 if hierarchical_eval and debug else 0
    
    # Precompute group mappings for hierarchical evaluation
    group_to_model_indices: Dict[str, List[int]] = {}
    group_to_model_indices_tensor: Dict[str, torch.Tensor] = {}
    model_idx_to_group: Dict[int, str] = {}
    if hierarchical_eval:
        if hierarchy_level == "domain":
            # Use domain2models mapping
            for domain, model_indices in model_registry.domain2models.items():
                group_to_model_indices[domain] = model_indices
                for model_idx in model_indices:
                    model_idx_to_group[model_idx] = domain
        elif hierarchy_level == "parent_group":
            # Use parent_group2models mapping
            for parent_group, model_indices in model_registry.parent_group2models.items():
                group_to_model_indices[parent_group] = model_indices
                for model_idx in model_indices:
                    model_idx_to_group[model_idx] = parent_group
        else:
            raise ValueError(f"Invalid hierarchy_level: {hierarchy_level}. Must be 'domain' or 'parent_group'")
        
        # Ensure all models have a group (map missing to "unknown")
        for model_idx in range(num_models):
            if model_idx not in model_idx_to_group:
                if "unknown" not in group_to_model_indices:
                    group_to_model_indices["unknown"] = []
                group_to_model_indices["unknown"].append(model_idx)
                model_idx_to_group[model_idx] = "unknown"
        
        # Precompute tensor-based indices for efficient domain scoring
        device_tensor = torch.device(device)
        for domain, model_indices in group_to_model_indices.items():
            if model_indices:
                group_to_model_indices_tensor[domain] = torch.tensor(
                    model_indices, dtype=torch.long, device=device_tensor
                )
            else:
                group_to_model_indices_tensor[domain] = torch.tensor(
                    [], dtype=torch.long, device=device_tensor
                )

    
    num_examples = len(test_data)
    print(f"\nEvaluating router on {num_examples} test examples...")
    
    # Check if model_family_lookup is available
    if model_family_lookup is None:
        print(f"      WARNING: model_family_lookup is None - family accuracy may not be computed correctly!")
        print(f"      Make sure to pass model_family_lookup when calling evaluate_router.")
    else:
        print(f"      model_family_lookup available with {len(model_family_lookup)} entries")
    
    # Track first batch for diagnostics
    first_batch_processed = False
    
    with torch.no_grad():
        for start_idx in range(0, num_examples, batch_size):
            end_idx = min(start_idx + batch_size, num_examples)
            batch = test_data[start_idx:end_idx]
            
            # B) Format prompts to match training format
            tokenizer = lm_model.tokenizer
            formatted_prompts = []
            for ex in batch:
                # Extract model_card if available (for retriever info)
                model_card = ex.get("model_card", "") or ex.get("reference_api", "")
                if model_card and not model_card.startswith("\n<Reference API>:"):
                    model_card = "\n<Reference API>: " + model_card
                
                formatted = format_eval_prompt(
                    ex=ex,
                    tokenizer=tokenizer,
                    eval_use_chat_template=eval_use_chat_template,
                    system_prompt=system_prompt,
                    model_card=model_card,
                )
                formatted_prompts.append(formatted)
            
            prompts = formatted_prompts
            labels = [ex['model_name'] for ex in batch]
            domains = [ex.get('domain', 'unknown') for ex in batch]
            
            # Convert labels to indices
            label_indices = []
            valid_mask = []
            gold_in_registry_flags = []
            for label in labels:
                if label in model_registry.model2idx:
                    label_indices.append(model_registry.model2idx[label])
                    valid_mask.append(True)
                    gold_in_registry_flags.append(True)
                else:
                    # Unknown model in test set
                    label_indices.append(-1)
                    valid_mask.append(False)
                    gold_in_registry_flags.append(False)
            
            inputs = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            ).to(device)
                       
            # Get model outputs with hidden states
            lm_outputs = lm_model.model.model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                output_hidden_states=True,
                return_dict=True,
            )
            
            hidden_states = lm_outputs.hidden_states[-1]  # [B, L, D]
            router_model = router_model.to(device=hidden_states.device, dtype=hidden_states.dtype)
            prompt_mask = inputs["attention_mask"].to(dtype=torch.bool)  # [B, L]

            prompt_embeddings = router_model.encode_prompt(
                hidden_states=hidden_states,
                prompt_mask=prompt_mask,
                )  # [B, embedding_dim] - already projected to router embedding space
            
            prompt_embs = prompt_embeddings  # Alias for clarity
            batch_size_actual = prompt_embs.shape[0]
            
            B = batch_size_actual
            result = router_model.score_all(
                hidden_states=hidden_states,
                prompt_mask=prompt_mask,
                return_compute_metrics=False,
            )
            if isinstance(result, tuple):
                scores = result[0]
            else:
                scores = result
            

            
            # ====================================================================
            # DIAGNOSTICS: After computing logits
            # ====================================================================
            logits = scores  # Alias for clarity

            
            
            top_scores, top_indices = scores.topk(k=max(k_values), dim=-1)
            
            cand_logits: Optional[torch.Tensor] = None
            cand_indices_tensor: Optional[torch.Tensor] = None
            if candidate_K is not None and candidate_K > 0:
                # Build candidate sets using the same strategy as training
                # Defaults match router_config.json if available
                K_total = candidate_K
                K_semantic = router_config.get("K_semantic", K_total - 1) if router_config else min(48, K_total - 1)
                K_far = router_config.get("K_far", 0) if router_config else 0
                K_hard = router_config.get("K_hard", 0) if router_config else 0

                cand_builder = CandidateSetBuilder(
                    registry=model_registry,
                    K_total=K_total,
                    K_semantic=K_semantic,
                    K_far=K_far,
                    K_hard=K_hard,
                    )

                cand_y_indices: List[int] = []
                cand_domains: List[str] = []
                cand_valid_indices: List[int] = []
                for i, (label_idx, domain, valid) in enumerate(zip(label_indices, domains, valid_mask)):
                    if not valid or label_idx < 0:
                        continue
                    cand_y_indices.append(label_idx)
                    cand_domains.append(domain)
                    cand_valid_indices.append(i)

                if cand_y_indices:
                    cand_list = cand_builder.build_batch(
                        y_indices=cand_y_indices,
                        domains=cand_domains,
                        hard_negative_cache=None,
                    )
                    cand_indices_tensor = torch.tensor(
                        cand_list, dtype=torch.long, device=device
                    )  # [B_cand, K]

                    # Sub-select hidden_states / prompt_mask for candidate examples
                    hs_cand = hidden_states[cand_valid_indices]
                    pm_cand = prompt_mask[cand_valid_indices]

                    result = router_model(
                        hidden_states=hs_cand,
                        prompt_mask=pm_cand,
                        candidate_indices=cand_indices_tensor,
                        return_compute_metrics=False,
                    )
                    if isinstance(result, tuple):
                        cand_logits = result[0]
                    else:
                        cand_logits = result

                    # Candidate-set top1 accuracy: gold always at index 0 by construction
                    cand_pred = cand_logits.argmax(dim=-1)  # [B_cand]
                    cand_top1_correct += (cand_pred == 0).sum().item()
                    cand_total += cand_logits.shape[0]
                    
                    # Track score margins for candidate sets (comparable to training)
                    if cand_logits.shape[0] > 0:
                        positive_scores = cand_logits[:, 0]  # [B_cand] - gold at index 0
                        negative_scores = cand_logits[:, 1:].mean(dim=-1)  # [B_cand] - average of negatives
                        margins = (positive_scores - negative_scores).cpu().tolist()
                        candidate_score_margins.extend(margins)

            # Compute metrics for valid examples (global acc_all)
            for i, (label_idx, pred_indices, domain, valid, gold_in_reg) in enumerate(
                zip(label_indices, top_indices, domains, valid_mask, gold_in_registry_flags)
            ):
                example_id = start_idx + i
                total_examples_processed += 1
                
                # ====================================================================
                # PER-EXAMPLE DIAGNOSTICS: Use p_i and logits_i, print hashes
                # ====================================================================
                p_i = prompt_embs[i]  # [embedding_dim]
                logits_i = logits[i]  # [num_models]
                
                # Compute hashes (using tensor_sha helper)
                p_hash = tensor_sha(p_i)
                l_hash = tensor_sha(logits_i)
                if debug:
                    print(f"[DIAG] i={i} p_hash={p_hash} l_hash={l_hash}")
                
                if gold_in_reg:
                    gold_in_registry_count += 1
                
                if not valid:
                    continue
                
                # Get score vector for this example
                # Convert to float32 before numpy conversion (handles BFloat16)
                score_vec = logits_i.cpu().float().numpy()  # [N]
                
                # Track score margins for all-models evaluation
                gold_score = logits_i[label_idx].item()
                # Get average score of all other models
                other_scores = torch.cat([logits_i[:label_idx], logits_i[label_idx+1:]]).mean().item()
                all_models_score_margins.append(gold_score - other_scores)
                
                # Compute entropy of softmax distribution
                # Convert scores to probabilities using softmax
                probs = F.softmax(logits_i, dim=-1).cpu().float().numpy()
                # Compute entropy: -sum(p * log(p))
                entropy = -np.sum(probs * np.log(probs + 1e-10))
                all_entropies.append(entropy)
                
                # Get top-10 predictions for debug output
                top10_scores, top10_indices = logits_i.topk(k=min(10, len(score_vec)), dim=-1)
                top10_predictions = [
                    (rank + 1, model_registry.idx2model[idx.item()], score.item())
                    for rank, (idx, score) in enumerate(zip(top10_indices, top10_scores))
                ]
                
                # Find gold rank among ALL models
                # Sort all scores to find rank
                sorted_indices = torch.argsort(logits_i, descending=True)
                gold_rank = (sorted_indices == label_idx).nonzero(as_tuple=True)[0].item() + 1
                gold_score = logits_i[label_idx].item()
                
                
                # Top-1 prediction
                pred_idx = pred_indices[0].item()
                all_predictions.append(pred_idx)
                all_labels.append(label_idx)
                all_domains.append(domain)
                all_scores.append(logits_i.cpu().tolist())
                
                # Top-k accuracy (over all models)
                for k in k_values:
                    if label_idx in pred_indices[:k].tolist():
                        topk_correct[k] += 1
                
                # Hierarchical evaluation: two-stage inference with Top-N Domain Hierarchical Rerank
                if hierarchical_eval and valid:
                
                    # Step 1: Compute domain scores using selected aggregation strategy
                    domain_scores = compute_domain_scores(
                        logits_all=logits_i,
                        domain_to_indices_tensor=group_to_model_indices_tensor,
                        mode=hier_domain_score_mode,
                        topk=hier_domain_topk,
                        alpha=hier_domain_hybrid_alpha,
                    )
                    
                    # Step 2: Identify top-N domains (N=1,2,3) by domain_score
                    if domain_scores:
                        # Sort domains by score (descending)
                        # Performance: Use torch.topk on tensor instead of sorting with .item() to avoid CPU sync
                        domain_names = list(domain_scores.keys())
                        domain_scores_tensor = torch.stack([domain_scores[name] for name in domain_names])
                        # Get top-3 domains (for N=1,2,3)
                        k = min(3, len(domain_scores_tensor))
                        topk_scores, topk_indices = torch.topk(domain_scores_tensor, k=k, largest=True)
                        top_domains = [domain_names[idx.item()] for idx in topk_indices]
                        
                        # Step 3: Check if gold model is in registry (for denominator tracking)
                        # Note: gold_in_reg is True when model is in registry (label_idx >= 0 is implied)
                        gold_in_registry = (label_idx >= 0 and gold_in_reg)
                        if not gold_in_registry:
                            hier_missing_gold_count += 1
                        
                        # Step 4: Get gold group and predicted group (for legacy metrics)
                        # Normalize top_domains to ensure consistency (group_to_model_indices keys are normalized)
                        top_domains_normalized = [normalize_domain(d) for d in top_domains] if top_domains else []
                        predicted_group = top_domains_normalized[0] if top_domains_normalized else "unknown"
                        
                        # Compute legacy group accuracy (top-1 domain) - only when gold in registry
                        if gold_in_registry:
                            gold_group = model_idx_to_group.get(label_idx, "unknown")
                            gold_group_normalized = normalize_domain(gold_group) if gold_group != "unknown" else "unknown"
                            hier_group_total += 1
                            if predicted_group == gold_group_normalized:
                                hier_group_correct += 1
                        
                        # Step 5: For each N in {1,2,3}, compute hierarchical rerank metrics
                        # Build restricted sets once and reuse (fixes issue #2: avoid computing twice)
                        restricted_indices_atN = {}
                        for N in [1, 2, 3]:
                            restricted_indices = build_restricted_set(
                                top_domains=top_domains_normalized,
                                domain_to_indices=group_to_model_indices,
                                N=N,
                                num_models=len(logits_i),
                            )
                            restricted_indices_atN[N] = restricted_indices
                        
                        for N in [1, 2, 3]:
                            restricted_indices = restricted_indices_atN[N]
                            # Track restricted set size (fixes issue #1: use same fallback logic)
                            restricted_size = len(restricted_indices)
                            hier_restricted_sizes_atN[N].append(restricted_size)
                            
                            # Check if gold domain is in predicted top-N domains (for domain recall)
                            # Note: Both model_idx_to_group and top_domains_normalized use normalized domains
                            gold_domain_in_topN = False
                            if gold_in_registry:
                                gold_group = model_idx_to_group.get(label_idx, "unknown")
                                gold_group_normalized = normalize_domain(gold_group) if gold_group != "unknown" else "unknown"
                                # top_domains_normalized are already normalized
                                gold_domain_in_topN = (gold_group_normalized in top_domains_normalized[:N])
                                if gold_domain_in_topN:
                                    hier_domain_included_atN[N] += 1
                            
                            # Compute hierarchical rerank metrics (using pre-built restricted set)
                            top1_correct, top5_correct, top10_correct = hierarchical_rerank_topN(
                                logits_all=logits_i,
                                restricted_indices=restricted_indices,
                                gold_model_idx=label_idx,
                                device=device,
                            )
                            
                            # Check if gold model is in restricted set
                            gold_in_restricted = (label_idx in restricted_indices) if gold_in_registry else False
                            
    
                            if gold_in_registry:
                                # E2E is correct only when domain is selected AND model is top-1
                                if gold_domain_in_topN and gold_in_restricted and top1_correct:
                                    hier_model_top1_e2e_atN[N] += 1
                                # Note: If gold domain not in top-N OR gold not top-1, E2E count stays 0 (correctly counts as incorrect)
                            
                            if gold_domain_in_topN and gold_in_registry:
                                hier_model_total_atN[N] += 1  # Count of examples where gold domain in top-N
                                if gold_in_restricted and top1_correct:
                                    hier_model_top1_atN[N] += 1  # Legacy: conditional top1
                                    hier_model_top1_cond_atN[N] += 1  # New: explicit conditional top1
                                if gold_in_restricted and top5_correct:
                                    hier_model_top5_atN[N] += 1
                                if gold_in_restricted and top10_correct:
                                    hier_model_top10_atN[N] += 1
                            
                            # Debug: Track why gold model is not in restricted set
                            if debug and gold_in_registry and not gold_in_restricted:
                                gold_group = model_idx_to_group.get(label_idx, "unknown")
                                gold_group_normalized = normalize_domain(gold_group) if gold_group != "unknown" else "unknown"
                                if gold_group_normalized not in group_to_model_indices:
                                    hier_gold_domain_missing_from_registry += 1
                                elif gold_group_normalized not in top_domains_normalized[:N]:
                                    hier_gold_not_in_predicted_domain[N] += 1
                        

                        legacy_restricted_indices = group_to_model_indices.get(predicted_group, [])
                        if legacy_restricted_indices:
                            # Convert to tensor for indexing
                            legacy_restricted_indices_tensor = torch.tensor(legacy_restricted_indices, dtype=torch.long, device=device)
                            # Get logits for models in restricted set
                            legacy_restricted_logits = logits_i[legacy_restricted_indices_tensor]  # [num_models_in_restricted]
                            
                            # Track restricted set size (legacy)
                            hier_restricted_sizes.append(len(legacy_restricted_indices))
                            
                            if label_idx in legacy_restricted_indices and gold_in_registry:
                                # Get index of gold model in restricted_indices list
                                gold_pos_in_restricted = legacy_restricted_indices.index(label_idx)
                                
                                # Compute top-k accuracy within restricted set (legacy - CONDITIONAL)
                                hier_model_total += 1
                                sorted_restricted = torch.argsort(legacy_restricted_logits, descending=True)
                                gold_rank_restricted = (sorted_restricted == gold_pos_in_restricted).nonzero(as_tuple=True)[0].item() + 1
                                
                                # Check top-k accuracy (legacy - CONDITIONAL)
                                for k in k_values:
                                    if gold_rank_restricted <= k:
                                        hier_model_topk_correct[k] += 1
                                
                                if gold_in_registry:
                                    gold_group = model_idx_to_group.get(label_idx, "unknown")
                                    gold_group_normalized = normalize_domain(gold_group) if gold_group != "unknown" else "unknown"
                                    if predicted_group == gold_group_normalized and gold_rank_restricted == 1:
                                        hier_e2e_top1_correct += 1
                        else:
                            # Predicted group is empty (shouldn't happen, but handle gracefully)
                            hier_restricted_sizes.append(0)
                
                # Domain accuracy (is predicted model in same domain as ground truth?)
                pred_model_name = model_registry.idx2model[pred_idx]
                true_model_name = model_registry.idx2model[label_idx]
                # Metadata is keyed by index, not model name
                pred_domain = model_registry.metadata.get(pred_idx, {}).get('domain', 'unknown')
                true_domain = model_registry.metadata.get(label_idx, {}).get('domain', 'unknown')
                
                # Normalize domains before comparison (consistent with other domain comparisons in this file)
                pred_domain_normalized = normalize_domain(pred_domain)
                true_domain_normalized = normalize_domain(true_domain)
                
                # Use normalized domain as key for consistent counting
                domain_total[true_domain_normalized] += 1
                if pred_domain_normalized == true_domain_normalized:
                    domain_correct[true_domain_normalized] += 1
                
                # Model family accuracy (is predicted model in same family as ground truth?)
                # Get model names first
                pred_model_name = model_registry.idx2model[pred_idx]
                

                pred_family = None
                true_family = None
                
                if model_family_lookup is not None:
                    # Try exact match first
                    pred_family = model_family_lookup.get(pred_model_name)
                    true_family = model_family_lookup.get(true_model_name)
                    
                    # Also try normalized names (case-insensitive lookup) if exact match failed
                    if pred_family is None:
                        from .model_selection_carve.model_registry import normalize_model_name
                        pred_family = model_family_lookup.get(normalize_model_name(pred_model_name))
                    if true_family is None:
                        from .model_selection_carve.model_registry import normalize_model_name
                        true_family = model_family_lookup.get(normalize_model_name(true_model_name))
                
                # Fallback: try registry metadata (might not have family info if created without family_key)
                if pred_family is None:
                    pred_metadata = model_registry.metadata.get(pred_idx, {})
                    pred_family = pred_metadata.get('family') or pred_metadata.get('model_family')
                
                if true_family is None:
                    true_metadata = model_registry.metadata.get(label_idx, {})
                    true_family = true_metadata.get('family') or true_metadata.get('model_family')
                
                # Fallback: try to get from batch/test_data if not found yet
                if pred_family is None:
                    # Look up predicted model's family from test data by matching model name
                    for test_ex in test_data:
                        if test_ex.get('model_name') == pred_model_name:
                            pred_family = test_ex.get('model_family')
                            break
                
                if true_family is None:
                    # Get true family from current batch example (more efficient than searching all test_data)
                    if i < len(batch):
                        true_family = batch[i].get('model_family')
                    # Fallback: search test_data if not in batch
                    if true_family is None:
                        # Use example_id to index into test_data directly (more efficient)
                        if example_id < len(test_data):
                            true_family = test_data[example_id].get('model_family')
                        # Last resort: search by model name
                        if true_family is None:
                            for test_ex in test_data:
                                if test_ex.get('model_name') == true_model_name:
                                    true_family = test_ex.get('model_family')
                                    break
                
                # Only count if both families are available (not None and not empty)
                if pred_family and true_family and pred_family.strip() and true_family.strip():
                    family_total[true_family] += 1
                    if pred_family == true_family:
                        family_correct[true_family] += 1
            
            if (end_idx) % 100 == 0 or end_idx == num_examples:
                print(f"  Processed {end_idx}/{num_examples} examples...")
    
    # Compute overall metrics
    num_valid = len(all_predictions)
    
    topk_accuracy = {}
    for k in k_values:
        topk_accuracy[f"top{k}_accuracy"] = topk_correct[k] / num_valid if num_valid > 0 else 0.0
    
    # Overall domain accuracy
    total_domain_correct = sum(domain_correct.values())
    total_domain_count = sum(domain_total.values())
    overall_domain_accuracy = total_domain_correct / total_domain_count if total_domain_count > 0 else 0.0
    
    # Per-domain accuracy
    per_domain_accuracy = {}
    for domain in domain_total:
        acc = domain_correct[domain] / domain_total[domain] if domain_total[domain] > 0 else 0.0
        per_domain_accuracy[f"accuracy_domain_{domain}"] = acc
    
    total_family_correct = sum(family_correct.values())
    total_family_count = sum(family_total.values())
    overall_family_accuracy = total_family_correct / total_family_count if total_family_count > 0 else 0.0
    family_accuracy_all_examples = total_family_correct / num_valid if num_valid > 0 else 0.0
    
    # Per-family accuracy
    per_family_accuracy = {}
    for family in family_total:
        acc = family_correct[family] / family_total[family] if family_total[family] > 0 else 0.0
        per_family_accuracy[f"accuracy_family_{family}"] = acc
    
    # Initialize forgetting metrics (will be computed in exp2 diagnostics section if M_old is available)
    model_forgetting = None
    domain_forgetting = None
    model_family_forgetting = None
    
    # Candidate-set accuracy (acc_candidate)
    if cand_total > 0:
        acc_candidate_top1 = cand_top1_correct / cand_total
    else:
        acc_candidate_top1 = 0.0
    
    # Score margin statistics (for comparison with training)
    import numpy as _np_margin
    candidate_margin_mean = float(_np_margin.mean(candidate_score_margins)) if candidate_score_margins else 0.0
    all_models_margin_mean = float(_np_margin.mean(all_models_score_margins)) if all_models_score_margins else 0.0
    
    # Gold rank diagnostics over ALL models (geometry / global separation)
    gold_ranks: List[int] = []
    gold_mrr_values: List[float] = []
    gold_margin_top1: List[float] = []
    gold_margin_topK: List[float] = []

    if num_valid > 0 and all_scores:
        K_global = max(k_values) if k_values else 1
        import numpy as _np
        for score_vec, label_idx in zip(all_scores, all_labels):
            if label_idx < 0 or label_idx >= len(score_vec):
                continue
            scores_arr = _np.asarray(score_vec, dtype=_np.float32)
            gold_score = float(scores_arr[label_idx])
            # Rank: 1 + number of models with strictly higher score
            rank = int((scores_arr > gold_score).sum()) + 1
            gold_ranks.append(rank)
            gold_mrr_values.append(1.0 / rank)
            # Margin vs top-1
            top1_score = float(scores_arr.max())
            gold_margin_top1.append(gold_score - top1_score)
            # Margin vs top-K boundary
            if len(scores_arr) >= K_global:
                kth_idx = _np.argpartition(-scores_arr, K_global - 1)[K_global - 1]
                kth_score = float(scores_arr[kth_idx])
            else:
                kth_score = float(scores_arr.min())
            gold_margin_topK.append(gold_score - kth_score)

    def _safe_stat(xs, fn, default=0.0):
        return float(fn(xs)) if xs else float(default)

    import numpy as _np2
    gold_rank_median = _safe_stat(gold_ranks, lambda v: _np2.median(v))
    gold_rank_p90 = _safe_stat(gold_ranks, lambda v: _np2.percentile(v, 90))
    gold_rank_mean = _safe_stat(gold_ranks, lambda v: _np2.mean(v))
    gold_mrr = _safe_stat(gold_mrr_values, lambda v: _np2.mean(v))
    gold_margin_top1_mean = _safe_stat(gold_margin_top1, lambda v: _np2.mean(v))
    gold_margin_topK_mean = _safe_stat(gold_margin_topK, lambda v: _np2.mean(v))

    # Diagnostic: percent of examples where gold_in_registry==True
    gold_in_registry_percent = (gold_in_registry_count / total_examples_processed * 100.0) if total_examples_processed > 0 else 0.0
    
    # Diagnostic: entropy statistics
    # Convert numpy types to Python native types for JSON serialization
    mean_entropy = float(np.mean(all_entropies)) if all_entropies else 0.0
    std_entropy = float(np.std(all_entropies)) if all_entropies else 0.0
    min_entropy = float(np.min(all_entropies)) if all_entropies else 0.0
    max_entropy = float(np.max(all_entropies)) if all_entropies else 0.0
    
    # Expected entropy for uniform distribution (log(N))
    # F) Fix: uniform has HIGH entropy, low entropy means peaked/collapsed
    expected_entropy_uniform = float(np.log(len(model_registry))) if len(model_registry) > 0 else 0.0
    
    # Compute hierarchical metrics (if enabled)
    hier_metrics = {}
    if hierarchical_eval:
        # Group accuracy
        hier_metrics["hier_group_accuracy"] = hier_group_correct / hier_group_total if hier_group_total > 0 else 0.0
        
        # Model top-k accuracy within restricted set (legacy - top-1 domain only)
        for k in k_values:
            hier_metrics[f"hier_model_top{k}"] = hier_model_topk_correct[k] / hier_model_total if hier_model_total > 0 else 0.0
        
        # End-to-end accuracy (group correct AND model correct) (legacy)
        hier_metrics["hier_e2e_top1"] = hier_e2e_top1_correct / hier_model_total if hier_model_total > 0 else 0.0
        
        # Restricted set size statistics (legacy - top-1 domain only)
        if hier_restricted_sizes:
            hier_metrics["hier_restricted_size_mean"] = float(np.mean(hier_restricted_sizes))
            hier_metrics["hier_restricted_size_median"] = float(np.median(hier_restricted_sizes))
            hier_metrics["hier_restricted_size_p90"] = float(np.percentile(hier_restricted_sizes, 90))
        else:
            hier_metrics["hier_restricted_size_mean"] = 0.0
            hier_metrics["hier_restricted_size_median"] = 0.0
            hier_metrics["hier_restricted_size_p90"] = 0.0
        
        # Top-N Domain Hierarchical Rerank metrics (N=1,2,3)
        # Use correct denominators: gold_in_registry_count for primary, total_examples_processed for strict
        denom_in_registry = gold_in_registry_count  # Examples where gold model is in registry
        denom_total = total_examples_processed  # All examples (strict)
        
        for N in [1, 2, 3]:
            # Legacy conditional metrics (only when gold domain in top-N)
            total_N = hier_model_total_atN[N]  # Count where gold in restricted set
            if total_N > 0:
                hier_metrics[f"hier_model_top1_at{N}"] = hier_model_top1_atN[N] / total_N
                hier_metrics[f"hier_model_top5_at{N}"] = hier_model_top5_atN[N] / total_N
                hier_metrics[f"hier_model_top10_at{N}"] = hier_model_top10_atN[N] / total_N
            else:
                hier_metrics[f"hier_model_top1_at{N}"] = 0.0
                hier_metrics[f"hier_model_top5_at{N}"] = 0.0
                hier_metrics[f"hier_model_top10_at{N}"] = 0.0
            
            # New: Conditional metrics (explicit naming)
            # Conditional denominator: examples where gold domain in top-N (should match hier_domain_included_atN[N])
            included_count = hier_domain_included_atN[N]  # Count where gold domain in top-N
            # Validation: hier_model_total_atN[N] should equal hier_domain_included_atN[N] after our fix
            if hier_model_total_atN[N] != included_count:
                # This should not happen - warn if there's a mismatch
                pass  # Will be caught in validation below
            if included_count > 0:
                hier_metrics[f"hier_model_top1_cond_at{N}"] = hier_model_top1_cond_atN[N] / included_count
            else:
                hier_metrics[f"hier_model_top1_cond_at{N}"] = 0.0
            
            # New: E2E metrics (primary - all examples where gold in registry)
            # E2E numerator should equal conditional numerator (both count: gold_domain_in_topN AND gold_in_restricted AND top1_correct)
            # Therefore: E2E = domain_recall * conditional_accuracy
            if denom_in_registry > 0:
                hier_metrics[f"hier_model_top1_e2e_at{N}"] = hier_model_top1_e2e_atN[N] / denom_in_registry
            else:
                hier_metrics[f"hier_model_top1_e2e_at{N}"] = 0.0
            
            # New: E2E strict metrics (all examples, missing gold counts as incorrect)
            if denom_total > 0:
                hier_metrics[f"hier_model_top1_e2e_at{N}_strict"] = hier_model_top1_e2e_atN[N] / denom_total
            else:
                hier_metrics[f"hier_model_top1_e2e_at{N}_strict"] = 0.0
            
            # New: Domain recall@N (fraction where gold domain ∈ predicted top-N)
            if denom_in_registry > 0:
                hier_metrics[f"hier_domain_recall_at{N}"] = hier_domain_included_atN[N] / denom_in_registry
            else:
                hier_metrics[f"hier_domain_recall_at{N}"] = 0.0
            
            # New: Domain recall@N strict (all examples)
            if denom_total > 0:
                hier_metrics[f"hier_domain_recall_at{N}_strict"] = hier_domain_included_atN[N] / denom_total
            else:
                hier_metrics[f"hier_domain_recall_at{N}_strict"] = 0.0
            
            # Restricted set size statistics for each N
            sizes_N = hier_restricted_sizes_atN[N]
            if sizes_N:
                hier_metrics[f"hier_restricted_size_mean_at{N}"] = float(np.mean(sizes_N))
                hier_metrics[f"hier_restricted_size_median_at{N}"] = float(np.median(sizes_N))
                hier_metrics[f"hier_restricted_size_p90_at{N}"] = float(np.percentile(sizes_N, 90))
            else:
                hier_metrics[f"hier_restricted_size_mean_at{N}"] = 0.0
                hier_metrics[f"hier_restricted_size_median_at{N}"] = 0.0
                hier_metrics[f"hier_restricted_size_p90_at{N}"] = 0.0
        
        # Add diagnostic counts
        hier_metrics["hier_missing_gold_count"] = hier_missing_gold_count
        hier_metrics["hier_gold_in_registry_count"] = denom_in_registry
        hier_metrics["hier_num_examples_total"] = denom_total
        hier_metrics["hier_domain_score_mode"] = hier_domain_score_mode
        hier_metrics["hier_domain_topk"] = hier_domain_topk
        hier_metrics["hier_domain_hybrid_alpha"] = hier_domain_hybrid_alpha
    
    # =====================================================================
    # Exp2 Collapse Diagnostics: Separate new-model interference vs exp1 drift
    # =====================================================================
    exp2_diagnostics = {}
    M_old = None
    print(f"  [Exp2 Diagnostics] Starting M_old detection...")
    print(f"  [Exp2 Diagnostics] checkpoint_dir: {checkpoint_dir}")
    print(f"  [Exp2 Diagnostics] router_config is None: {router_config is None}")
    if checkpoint_dir is not None:
        
        # Try 1: Check router_config for explicit M_old or base registry path
        if router_config is not None:
            # Always print what we're checking (not just in debug mode)
            print(f"  [Exp2 Diagnostics] Checking router_config for M_old...")
            print(f"  [Exp2 Diagnostics] router_config keys: {list(router_config.keys())}")
            
            # Check for explicit M_old
            if "router_exp1_preservation_M_old" in router_config:
                M_old_value = router_config.get("router_exp1_preservation_M_old")
                print(f"  [Exp2 Diagnostics] Found router_exp1_preservation_M_old = {M_old_value}")
                if M_old_value is not None:
                    M_old = int(M_old_value)
                    print(f"  [Exp2 Diagnostics] ✓ M_old={M_old} (from router_config.router_exp1_preservation_M_old)")
                else:
                    print(f"  [Exp2 Diagnostics] ⚠️  router_exp1_preservation_M_old is None")
            else:
                print(f"  [Exp2 Diagnostics] router_exp1_preservation_M_old not found in router_config")
            
            # Check for base registry path
            if M_old is None and "router_registry_base_path" in router_config:
                base_registry_path_str = router_config.get("router_registry_base_path")
                print(f"  [Exp2 Diagnostics] Found router_registry_base_path = {base_registry_path_str}")
                if base_registry_path_str:
                    base_registry_path = Path(base_registry_path_str)
                    # Try to resolve relative path: check multiple possible locations
                    if not base_registry_path.is_absolute():
                        # Strategy 1: Try relative to current working directory (project root)
                        resolved_paths = [Path.cwd() / base_registry_path]
                        # Strategy 2: Try relative to checkpoint directory's parent (if checkpoint is in results/ or core/experiments/)
                        if "results" in str(checkpoint_dir) or "experiments" in str(checkpoint_dir):
                            # Go up to project root: results/apibench/... -> results/ -> project root
                            # or core/experiments/... -> core/experiments/ -> cmr/ -> project root
                            project_root = checkpoint_dir
                            while project_root.name not in ["results", "cmr", "CMR"] and len(project_root.parts) > 1:
                                project_root = project_root.parent
                            if project_root.parent.exists():
                                resolved_paths.append(project_root.parent / base_registry_path)
                        # Strategy 3: Try relative to checkpoint directory itself
                        resolved_paths.append(checkpoint_dir / base_registry_path)
                        
                        # Find first existing path
                        for candidate_path in resolved_paths:
                            if candidate_path.exists():
                                base_registry_path = candidate_path
                                break
                    
                    if base_registry_path.exists():
                        try:
                            with open(base_registry_path, 'r') as f:
                                base_registry_data = json.load(f)
                                if "num_models" in base_registry_data:
                                    M_old = int(base_registry_data["num_models"])
                                elif "model2idx" in base_registry_data:
                                    M_old = len(base_registry_data["model2idx"])
                            if M_old is not None:
                                print(f"  [Exp2 Diagnostics] M_old={M_old} (from router_registry_base_path: {base_registry_path})")
                        except Exception as e:
                            print(f"  ⚠️  [Exp2 Diagnostics] Could not read M_old from base registry {base_registry_path}: {e}")
                    else:
                        print(f"  ⚠️  [Exp2 Diagnostics] Base registry path does not exist: {base_registry_path_str}")
                        print(f"      Resolved to: {base_registry_path.resolve()}")
                        print(f"      Current working directory: {Path.cwd()}")
                        print(f"      Checkpoint directory: {checkpoint_dir}")
        
        # Try 2: Check current checkpoint's model_registry.json (if it's exp1-sized)
        if M_old is None:
            model_registry_path = checkpoint_dir / "model_registry.json"
            if model_registry_path.exists():
                try:
                    with open(model_registry_path, 'r') as f:
                        registry_data = json.load(f)
                        # Get num_models from saved registry
                        if "num_models" in registry_data:
                            candidate_M_old = int(registry_data["num_models"])
                        elif "model2idx" in registry_data:
                            candidate_M_old = len(registry_data["model2idx"])
                        else:
                            candidate_M_old = None
                        
                        # Only use if it's smaller than current registry (indicates exp1)
                        if candidate_M_old is not None and candidate_M_old < len(model_registry):
                            M_old = candidate_M_old
                            print(f"  [Exp2 Diagnostics] M_old={M_old} (from checkpoint model_registry.json, current registry: {len(model_registry)})")
                        elif candidate_M_old is not None:
                            print(f"  [Exp2 Diagnostics] Checkpoint registry size ({candidate_M_old}) >= current registry ({len(model_registry)}), skipping")
                except Exception as e:
                    print(f"  ⚠️  [Exp2 Diagnostics] Could not read M_old from {model_registry_path}: {e}")
            else:
                print(f"  ⚠️  [Exp2 Diagnostics] Checkpoint model_registry.json not found: {model_registry_path}")
        
        # Warn if M_old still not found
        if M_old is None:
            print(f"  ⚠️  [Exp2 Diagnostics] M_old not found. Cannot compute exp2 collapse diagnostics.")
            print(f"      Tried: router_config, router_registry_base_path, checkpoint model_registry.json")
            if router_config is None:
                print(f"      router_config is None - check if router_config.json exists in checkpoint_dir")
            elif "router_exp1_preservation_M_old" not in router_config and "router_registry_base_path" not in router_config:
                print(f"      router_config exists but missing both router_exp1_preservation_M_old and router_registry_base_path")
                print(f"      router_config keys: {list(router_config.keys())}")
            else:
                print(f"      router_config has fields but path resolution may have failed")
            print(f"      To enable exp2 diagnostics, ensure router_config contains router_exp1_preservation_M_old or router_registry_base_path")
        
        if M_old is not None and len(all_predictions) > 0 and len(all_labels) > 0:
            M_new = len(model_registry)
            
            # Convert to numpy for easier computation
            pred_indices = np.array(all_predictions)
            gold_indices = np.array(all_labels)
            
            # A) pred_new_rate = mean(pred_idx >= M_old) - % predictions with pred_idx >= M_old
            pred_new_mask = pred_indices >= M_old
            pred_new_rate = pred_new_mask.mean()
            
            # B) pred_new_rate_on_old_gold = mean((pred_idx >= M_old) & (gold_idx < M_old))
            # to measure new-model interference on old-gold examples
            old_gold_mask = gold_indices < M_old
            pred_new_on_old_gold = (pred_new_mask & old_gold_mask).mean()
            num_old_gold = old_gold_mask.sum()
            
            # C) old_only_top1_accuracy: compute top1 using argmax over logits[:M_old] only
            # (for examples with gold_idx < M_old)
            # Note: We need to recompute predictions with restricted logits
            # For now, we'll approximate by checking if pred_idx < M_old when gold_idx < M_old
            old_only_correct = ((pred_indices < M_old) & (pred_indices == gold_indices) & old_gold_mask).sum()
            old_only_total = old_gold_mask.sum()
            old_only_top1_accuracy = old_only_correct / old_only_total if old_only_total > 0 else 0.0
            
            # D) full_top1_accuracy (existing - already computed as top1_accuracy)
            full_top1_accuracy = topk_accuracy.get("top1_accuracy", 0.0)
            
            # E) top1_accuracy_exp1_slice: compute top-1 accuracy restricted to exp1 slice (logits[:M_old])
            # for ALL examples (not just old-gold ones)
            # This requires recomputing predictions using only logits[:M_old]
            top1_accuracy_exp1_slice = None
            if len(all_scores) > 0 and len(all_scores) == len(all_labels):
                # Recompute predictions using only exp1 slice
                exp1_slice_correct = 0
                exp1_slice_total = 0
                for scores_list, gold_idx in zip(all_scores, all_labels):
                    if gold_idx < 0:  # Skip invalid examples
                        continue
                    # Convert to tensor and restrict to exp1 slice
                    scores_tensor = torch.tensor(scores_list[:M_old], dtype=torch.float32)  # [M_old]
                    if len(scores_tensor) > 0:
                        pred_idx_exp1 = scores_tensor.argmax().item()
                        if pred_idx_exp1 == gold_idx:
                            exp1_slice_correct += 1
                        exp1_slice_total += 1
                
                top1_accuracy_exp1_slice = exp1_slice_correct / exp1_slice_total if exp1_slice_total > 0 else 0.0
            else:
                # Fallback: approximate using existing predictions (less accurate)
                # Only count as correct if pred_idx < M_old and matches gold
                exp1_slice_correct = ((pred_indices < M_old) & (pred_indices == gold_indices)).sum()
                exp1_slice_total = len(pred_indices)
                top1_accuracy_exp1_slice = exp1_slice_correct / exp1_slice_total if exp1_slice_total > 0 else 0.0
            
            # Compute forgetting metrics for old models/domains/families
            # Forgetting = 1 - accuracy on old items (measures performance degradation)
            
            # Identify old models, domains, and families
            old_model_indices = set(range(M_old))
            old_domains = set()
            old_families = set()
            for idx in old_model_indices:
                metadata = model_registry.metadata.get(idx, {})
                domain = metadata.get('domain')
                family = metadata.get('family') or metadata.get('model_family')
                if domain:
                    old_domains.add(domain)
                if family:
                    old_families.add(family)
            
            # Compute accuracy on old models (already computed as old_only_top1_accuracy)
            model_forgetting = 1.0 - old_only_top1_accuracy if old_only_total > 0 else None
            
            # Compute accuracy on old domains
            old_domain_correct_count = 0
            old_domain_total_count = 0
            for domain in old_domains:
                if domain in domain_total:
                    old_domain_total_count += domain_total[domain]
                    old_domain_correct_count += domain_correct[domain]
            domain_forgetting = 1.0 - (old_domain_correct_count / old_domain_total_count) if old_domain_total_count > 0 else None
            
            # Compute accuracy on old families
            old_family_correct_count = 0
            old_family_total_count = 0
            for family in old_families:
                if family in family_total:
                    old_family_total_count += family_total[family]
                    old_family_correct_count += family_correct[family]
            model_family_forgetting = 1.0 - (old_family_correct_count / old_family_total_count) if old_family_total_count > 0 else None
            
            exp2_diagnostics = {
                "exp2_M_old": M_old,
                "exp2_M_new": M_new,
                "exp2_pred_new_rate": float(pred_new_rate),  # % predictions with pred_idx >= M_old
                "exp2_pred_new_rate_on_old_gold": float(pred_new_on_old_gold),
                "exp2_old_only_top1_accuracy": float(old_only_top1_accuracy),
                "exp2_full_top1_accuracy": float(full_top1_accuracy),  # Top-1 accuracy over full registry
                "exp2_top1_accuracy_exp1_slice": float(top1_accuracy_exp1_slice) if top1_accuracy_exp1_slice is not None else 0.0,  # Top-1 accuracy restricted to exp1 slice
                "exp2_num_old_gold_examples": int(num_old_gold),
                "exp2_old_only_correct": int(old_only_correct),
            }
            
            # Update forgetting metrics (will be added to main metrics dict)
            if model_forgetting is not None:
                model_forgetting = float(model_forgetting)
            if domain_forgetting is not None:
                domain_forgetting = float(domain_forgetting)
            if model_family_forgetting is not None:
                model_family_forgetting = float(model_family_forgetting)
            

    
    # Add forgetting metrics to metrics dict (if computed)
    forgetting_metrics = {}
    if model_forgetting is not None:
        forgetting_metrics["model_forgetting"] = model_forgetting
    if domain_forgetting is not None:
        forgetting_metrics["domain_forgetting"] = domain_forgetting
    if model_family_forgetting is not None:
        forgetting_metrics["model_family_forgetting"] = model_family_forgetting
    
    metrics = {
        **topk_accuracy,
        "domain_accuracy": overall_domain_accuracy,
        **per_domain_accuracy,
        "model_family_accuracy": overall_family_accuracy,  # Uses total_family_count as denominator (correct calculation)
        "model_family_accuracy_conditional": overall_family_accuracy,  # Same as model_family_accuracy (for backward compatibility)
        "model_family_accuracy_all_examples": family_accuracy_all_examples,  # Uses num_valid as denominator (for comparison)
        "model_family_num_examples_with_family_info": total_family_count,  # Diagnostic: how many examples had family info
        **per_family_accuracy,
        **forgetting_metrics,
        "num_examples_evaluated": num_valid,
        "num_examples_total": num_examples,
        "num_models": len(model_registry),
        "gold_in_registry_percent": gold_in_registry_percent,
        # Dual-metric reporting
        "top1_accuracy_candidate": acc_candidate_top1,
        # Global rank diagnostics over all models
        "gold_rank_median": gold_rank_median,
        "gold_rank_p90": gold_rank_p90,
        "gold_rank_mean": gold_rank_mean,
        "gold_mrr": gold_mrr,
        "gold_margin_top1_mean": gold_margin_top1_mean,
        "gold_margin_topK_mean": gold_margin_topK_mean,
        "entropy_mean": mean_entropy,
        "entropy_std": std_entropy,
        "entropy_min": min_entropy,
        "entropy_max": max_entropy,
        "entropy_expected_uniform": expected_entropy_uniform,
        # Score margins (for comparison with training)
        "candidate_score_margin_mean": candidate_margin_mean,
        "all_models_score_margin_mean": all_models_margin_mean,
        # Hierarchical evaluation metrics (if enabled)
        **hier_metrics,
        # Exp2 collapse diagnostics (if checkpoint_dir provided)
        **exp2_diagnostics,
    }
    
    
    
   
    
    # Print score margin comparison with training
    print(f"\n{'='*80}")
    print(f"[SCORE MARGIN COMPARISON]")
    print(f"{'='*80}")
    print(f"  Candidate-set score margin (comparable to training): {candidate_margin_mean:.4f}")
    print(f"  All-models score margin: {all_models_margin_mean:.4f}")
    print(f"  Training avg_score_margin (from logs): ~5.88")
    print(f"  {'✓' if candidate_margin_mean > 0 else '✗'} Candidate margin is {'positive' if candidate_margin_mean > 0 else 'negative'}")
    print(f"{'='*80}\n")
    
    return metrics


def save_router_predictions(
    predictions: List[int],
    scores: List[List[float]],
    test_data: List[Dict[str, Any]],
    model_registry: ModelRegistry,
    output_path: Path,
):
    """
    Save router predictions to a JSON file for analysis.
    
    Args:
        predictions: List of predicted model indices
        scores: List of score vectors (one per example)
        test_data: Original test examples
        model_registry: ModelRegistry for name lookups
        output_path: Path to save predictions
    """
    output = []
    for pred_idx, score_vec, example in zip(predictions, scores, test_data):
        pred_name = model_registry.idx2model[pred_idx]
        true_name = example['model_name']
        
        # Get top-5 predictions for this example
        top5_scores, top5_indices = torch.tensor(score_vec).topk(k=5)
        top5_predictions = [
            {
                "model_name": model_registry.idx2model[idx],
                "score": score.item(),
            }
            for idx, score in zip(top5_indices.tolist(), top5_scores.tolist())
        ]
        
        output.append({
            "prompt": example['prompt_text'],
            "ground_truth": true_name,
            "prediction": pred_name,
            "correct": pred_name == true_name,
            "domain_ground_truth": example.get('domain', 'unknown'),
            "top5_predictions": top5_predictions,
        })
    
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"✓ Saved router predictions to {output_path}")


