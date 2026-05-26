"""
Card-Guided Cold-Start Initialisation for New Router Embedding Rows.

When the router embedding table expands (M_old → M_new models), new rows are
initialised as a card-similarity-weighted mixture of the already-learned old
embeddings rather than random Xavier noise.

Formula:
    v_init(m_new) = sum_{m in old} softmax(cos(phi(card(m_new)), phi(card(m))) / tau)[m] * e(m)

where:
    phi  = frozen sentence transformer (same as used in coreset_replay.py)
    e(m) = learned routing embedding for old model m
    tau  = temperature (card_init_tau, default 0.07)

This gives new models a warm start biased toward the routing geometry of
semantically similar old models, reducing the number of steps needed for the
new embeddings to reach useful representations.
"""

import hashlib
import os
import re
from typing import Dict, List, Optional, TYPE_CHECKING

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer
    from ..model_selection_carve.model_registry import ModelRegistry


# ============================================================================
# Card Embedding Cache
# ============================================================================

def build_card_embedding_cache(
    model_ids: List[str],
    registry: "ModelRegistry",
    card_encoder: Optional["SentenceTransformer"] = None,
) -> Dict[str, torch.Tensor]:
    """
    Encode the card text for each model and return L2-normalised embeddings.

    The card text is sourced from registry metadata in priority order:
      1. ``card_text``  – full model card / description (if stored)
      2. ``model_text`` – typically the model name with minor augmentation
      3. ``model_id``   – raw ID as last resort

    Args:
        model_ids:    List of model-name strings to encode.
        registry:     ModelRegistry whose ``model2idx`` and ``metadata`` are
                      already populated.
        card_encoder: Optional loaded SentenceTransformer instance (frozen).
            If None, uses a deterministic hashed text embedding built from
            local card text metadata only (fully offline).

    Returns:
        Dict mapping model_id -> L2-normalised card embedding tensor [C] on CPU.
    """
    texts: List[str] = []
    valid_ids: List[str] = []

    for mid in model_ids:
        idx = registry.model2idx.get(mid)
        if idx is None:
            texts.append(mid)
        else:
            meta = registry.metadata.get(idx, {})
            text = meta.get("card_text") or meta.get("model_text") or mid
            texts.append(str(text))
        valid_ids.append(mid)

    if not texts:
        return {}

    result: Dict[str, torch.Tensor] = {}
    if card_encoder is not None:
        embeddings = card_encoder.encode(
            texts,
            batch_size=64,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,   # L2-normalise at encode time
        )
        for mid, emb in zip(valid_ids, embeddings):
            t = torch.from_numpy(emb).float()
            t = F.normalize(t, p=2, dim=0)
            result[mid] = t
        return result

    dim = 768
    token_re = re.compile(r"[a-z0-9_]+")
    for mid, text in zip(valid_ids, texts):
        vec = torch.zeros(dim, dtype=torch.float32)
        for tok in token_re.findall(str(text).lower()):
            digest = hashlib.md5(tok.encode("utf-8")).hexdigest()
            idx = int(digest[:8], 16) % dim
            sign = -1.0 if (int(digest[8:10], 16) % 2) else 1.0
            vec[idx] += sign
        if torch.count_nonzero(vec) == 0:
            digest = hashlib.md5(mid.encode("utf-8")).hexdigest()
            vec[int(digest[:8], 16) % dim] = 1.0
        result[mid] = F.normalize(vec, p=2, dim=0)

    return result


# ============================================================================
# Card-Guided Initialisation
# ============================================================================

def card_guided_init(
    new_model_ids: List[str],
    old_model_ids: List[str],
    old_emb: torch.Tensor,
    card_cache: Dict[str, torch.Tensor],
    tau: float = 0.07,
    topk: int = 30,
    scope: str = "global",
    min_sim_threshold: float = 0.2,
    fallback_domain: bool = True,
    registry: Optional["ModelRegistry"] = None,
) -> torch.Tensor:
    """
    Compute warm-start embeddings for new models via card-similarity weighting.

    For each new model m_new:
      1. Retrieve its card embedding phi(card(m_new)).
      2. Compute cosine similarity with every old model's card embedding.
      3. Restrict to top-``topk`` most-similar old models.
      4. Apply softmax with temperature ``tau``.
      5. Return weighted sum of the corresponding ``old_emb`` rows.

    Edge case – low-similarity fallback:
      If the maximum cosine similarity for a new model is below
      ``min_sim_threshold`` **and** ``fallback_domain=True``, the initialisation
      falls back to the mean of old embeddings that share the same domain (from
      registry metadata).  If the domain has no old-model overlap, or if
      ``fallback_domain=False``, falls back to the global mean of old embeddings.

    Args:
        new_model_ids:      Ordered list of new model IDs (rows M_old:M_new).
        old_model_ids:      Ordered list of old model IDs (rows 0:M_old), must
                            align with ``old_emb`` row order.
        old_emb:            Tensor [M_old, D] of already-learned embeddings.
        card_cache:         Dict model_id -> L2-normalised card emb [C] (CPU).
        tau:                Softmax temperature for similarity weighting.
        topk:               Number of nearest old models to consider.
        scope:              Similarity scope:
                            - "global": compare against all old models
                            - "within_domain": compare only against old models
                              with the same normalized domain as the new model.
                            Falls back to "global" if registry/domain metadata is
                            unavailable or no same-domain old models exist.
        min_sim_threshold:  Minimum max-similarity to use weighted init;
                            below this threshold the domain-mean fallback fires.
        fallback_domain:    Whether to use domain-mean fallback when similarity
                            is too low (True) or global mean (False).
        registry:           ModelRegistry used for domain-based fallback.
                            If None, domain fallback is skipped.

    Returns:
        Tensor [len(new_model_ids), D] on the same device / dtype as ``old_emb``.
    """
    device = old_emb.device
    dtype = old_emb.dtype
    M_old, D = old_emb.shape
    N_new = len(new_model_ids)

    old_card_tensors = []
    for mid in old_model_ids:
        if mid in card_cache:
            old_card_tensors.append(card_cache[mid])
        else:
            c = old_card_tensors[0].shape[0] if old_card_tensors else 768
            old_card_tensors.append(torch.zeros(c))

    old_card_mat = torch.stack(old_card_tensors, dim=0)  # [M_old, C]  (CPU, float32)

    old_emb_cpu = old_emb.detach().float().cpu()  # [M_old, D]
    global_mean = old_emb_cpu.mean(dim=0)          # [D]

    domain_mean_cache: Dict[str, torch.Tensor] = {}
    old_domain_to_indices: Dict[str, List[int]] = {}

    if registry is not None:
        from ..model_selection_carve.model_registry import normalize_domain

        for i, mid in enumerate(old_model_ids):
            idx = registry.model2idx.get(mid)
            if idx is None:
                continue
            meta = registry.metadata.get(idx, {})
            d = normalize_domain(meta.get("domain", "unknown"))
            old_domain_to_indices.setdefault(d, []).append(i)

    def _get_domain_mean(domain: str) -> torch.Tensor:
        if domain in domain_mean_cache:
            return domain_mean_cache[domain]
        if registry is not None:
            norm_domain = normalize_domain(domain)
            domain_indices = old_domain_to_indices.get(norm_domain, [])
            if domain_indices:
                mean = old_emb_cpu[domain_indices].mean(dim=0)
                domain_mean_cache[domain] = mean
                return mean
        domain_mean_cache[domain] = global_mean
        return global_mean

    actual_topk = min(topk, M_old)
    result_rows: List[torch.Tensor] = []

    for new_mid in new_model_ids:
        if new_mid not in card_cache:
            result_rows.append(global_mean.clone())
            continue

        new_card = card_cache[new_mid]  # [C]

        # Cosine similarities: new_card is already L2-normalised, old_card_mat
        # may have edge-case rows that weren't normalised -> re-normalise
        old_norm = F.normalize(old_card_mat, p=2, dim=1)  # [M_old, C]
        sims = (old_norm @ new_card).float()              # [M_old]

        candidate_indices = torch.arange(M_old, dtype=torch.long)
        if scope == "within_domain" and registry is not None:
            idx = registry.model2idx.get(new_mid)
            if idx is not None:
                from ..model_selection_carve.model_registry import normalize_domain
                meta = registry.metadata.get(idx, {})
                new_domain = normalize_domain(meta.get("domain", "unknown"))
                domain_candidates = old_domain_to_indices.get(new_domain, [])
                if domain_candidates:
                    candidate_indices = torch.tensor(domain_candidates, dtype=torch.long)

        sims_candidates = sims[candidate_indices]
        max_sim = sims_candidates.max().item()

        if max_sim < min_sim_threshold and fallback_domain:
            domain = "unknown"
            if registry is not None:
                idx = registry.model2idx.get(new_mid)
                if idx is not None:
                    domain = registry.metadata.get(idx, {}).get("domain", "unknown")
            result_rows.append(_get_domain_mean(domain).clone())
            continue

        k = min(actual_topk, int(candidate_indices.numel()))
        topk_sims, topk_local_indices = sims_candidates.topk(k)  # [k]
        topk_indices = candidate_indices[topk_local_indices]     # [k]

        weights = torch.softmax(topk_sims / tau, dim=0)   # [k]

        selected = old_emb_cpu[topk_indices.long()]        # [k, D]
        weighted = (weights.unsqueeze(1) * selected).sum(dim=0)  # [D]

        result_rows.append(weighted)

    result = torch.stack(result_rows, dim=0)  # [N_new, D]

    result = result.to(device=device, dtype=dtype)
    return result
