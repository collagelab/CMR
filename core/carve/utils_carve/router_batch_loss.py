from typing import Any, Dict, Optional

import torch
from torch import nn

from .router_similarity_utils import enable_hidden_states, get_last_hidden_states
from .router_training import (
    check_label_candidate_alignment,
    compute_router_metrics,
    compute_routing_loss,
    filter_and_validate_candidates,
)
from ..model_selection_carve import normalize_domain
from ..model_selection_carve.router import extract_prompt_mask


def compute_routing_loss_for_batch(
    trainer,
    model: nn.Module,
    inputs: Dict[str, torch.Tensor],
    outputs: Any,
    apply_replay_multiplier: bool = False,
    replay_loss_multiplier: float = 1.0,
    use_soft_targets: Optional[bool] = None,
    soft_target_eps: Optional[float] = None,
) -> Optional[torch.Tensor]:
    """Compute routing loss for the current batch with full trainer side effects."""
    if trainer._router_model is None or trainer._router_registry is None:
        return None

    batch_size = inputs["input_ids"].shape[0]
    device = inputs["input_ids"].device

    hidden_states = get_last_hidden_states(outputs)
    if hidden_states is None:
        enable_hidden_states(model)
        return None

    labels = inputs.get("labels")
    if labels is None:
        return None
    attention_mask = inputs.get("attention_mask")

    if "model_name" not in inputs or "domain" not in inputs:
        return None

    model_names = inputs["model_name"]
    domains = inputs["domain"]

    if not isinstance(model_names, list):
        model_names = list(model_names) if hasattr(model_names, "__iter__") else [str(model_names)] * batch_size
    if not isinstance(domains, list):
        domains = list(domains) if hasattr(domains, "__iter__") else [str(domains)] * batch_size

    is_replay_list = None
    if "is_replay" in inputs:
        is_replay_raw = inputs["is_replay"]
        if isinstance(is_replay_raw, list):
            is_replay_list = is_replay_raw
        elif hasattr(is_replay_raw, "__iter__"):
            is_replay_list = list(is_replay_raw)
        else:
            is_replay_list = [bool(is_replay_raw)] * batch_size

    unknown_before_norm = []
    unknown_replay_before = []
    unknown_nonreplay_before = []
    y_indices = []
    valid_indices = []
    unknown_models = []
    unknown_replay = []
    unknown_nonreplay = []

    for i, model_name in enumerate(model_names):
        if model_name not in trainer._router_registry.model2idx:
            unknown_before_norm.append(model_name)
            if is_replay_list is not None and i < len(is_replay_list) and is_replay_list[i]:
                unknown_replay_before.append(model_name)
            else:
                unknown_nonreplay_before.append(model_name)

        model_idx = trainer._get_model_idx(model_name)
        if model_idx is not None:
            y_indices.append(model_idx)
            valid_indices.append(i)
        else:
            unknown_models.append(model_name)
            if is_replay_list is not None and i < len(is_replay_list) and is_replay_list[i]:
                unknown_replay.append(model_name)
            else:
                unknown_nonreplay.append(model_name)

    num_filtered = len(model_names) - len(y_indices)
    num_unknown_before_norm = len(unknown_before_norm)
    num_unknown_after_norm = len(unknown_models)
    num_fixed_by_norm = num_unknown_before_norm - num_unknown_after_norm

    if not hasattr(trainer, "_router_filter_stats"):
        trainer._router_filter_stats = {
            "total_examples": 0,
            "total_filtered": 0,
            "total_unknown": 0,
            "unknown_before_norm": 0,
            "unknown_after_norm": 0,
            "fixed_by_norm": 0,
            "unknown_replay_before": 0,
            "unknown_nonreplay_before": 0,
            "unknown_replay_after": 0,
            "unknown_nonreplay_after": 0,
        }
    trainer._router_filter_stats["total_examples"] += len(model_names)
    trainer._router_filter_stats["total_filtered"] += num_filtered
    trainer._router_filter_stats["total_unknown"] += len(unknown_models)
    trainer._router_filter_stats["unknown_before_norm"] += num_unknown_before_norm
    trainer._router_filter_stats["unknown_after_norm"] += num_unknown_after_norm
    trainer._router_filter_stats["fixed_by_norm"] += num_fixed_by_norm
    trainer._router_filter_stats["unknown_replay_before"] += len(unknown_replay_before)
    trainer._router_filter_stats["unknown_nonreplay_before"] += len(unknown_nonreplay_before)
    trainer._router_filter_stats["unknown_replay_after"] += len(unknown_replay)
    trainer._router_filter_stats["unknown_nonreplay_after"] += len(unknown_nonreplay)

    if trainer.state.global_step % 100 == 0 and trainer._router_filter_stats["total_examples"] > 0:
        total = trainer._router_filter_stats["total_examples"]
        unknown_before = trainer._router_filter_stats["unknown_before_norm"]
        unknown_after = trainer._router_filter_stats["unknown_after_norm"]
        fixed = trainer._router_filter_stats["fixed_by_norm"]
        replay_before = trainer._router_filter_stats["unknown_replay_before"]
        replay_after = trainer._router_filter_stats["unknown_replay_after"]
        nonreplay_before = trainer._router_filter_stats["unknown_nonreplay_before"]
        nonreplay_after = trainer._router_filter_stats["unknown_nonreplay_after"]
        if unknown_before > 0:
            print(f"  [Model Name Normalization] Step {trainer.state.global_step}:")
            print(f"    Unknown before norm: {unknown_before} ({100 * unknown_before / total:.2f}%)")
            print(f"    Unknown after norm: {unknown_after} ({100 * unknown_after / total:.2f}%)")
            print(f"    Fixed by normalization: {fixed} ({100 * fixed / max(1, unknown_before):.1f}% of unknowns)")
            if replay_before > 0 or replay_after > 0:
                print(f"    Replay unknowns: {replay_before} → {replay_after} (fixed: {replay_before - replay_after})")
            if nonreplay_before > 0 or nonreplay_after > 0:
                print(
                    f"    Non-replay unknowns: {nonreplay_before} → {nonreplay_after} "
                    f"(fixed: {nonreplay_before - nonreplay_after})"
                )

    if len(unknown_replay) > 0 and trainer.state.global_step % 50 == 0:
        print(
            f"  ⚠️  [Replay Drop Warning] {len(unknown_replay)} replay examples dropped "
            f"due to unknown models (step {trainer.state.global_step})"
        )

    if not y_indices:
        return None

    normalized_domains = [normalize_domain(domains[i]) for i in valid_indices]
    candidate_list = trainer._router_candidate_builder.build_batch(
        y_indices=y_indices,
        domains=normalized_domains,
        hard_negative_cache=trainer._router_hard_negative_cache,
    )

    gold_model_names = [trainer._router_registry.idx2model.get(y_idx, f"unknown_idx_{y_idx}") for y_idx in y_indices]
    filtered_candidates, updated_y_indices, _filter_stats = filter_and_validate_candidates(
        candidates_list=candidate_list,
        gold_model_names=gold_model_names,
        gold_indices=y_indices,
        registry=trainer._router_registry,
        K_total=trainer._router_candidate_builder.K_total,
        debug=(trainer.state.global_step < 3),
    )

    y_indices = updated_y_indices
    candidate_list = filtered_candidates
    candidate_indices = torch.tensor(candidate_list, dtype=torch.long, device=device)

    if trainer._router_anchor_enable and trainer._router_anchor_scope == "touched":
        trainer._router_anchor_candidate_indices = candidate_indices
        trainer._router_anchor_y_indices = y_indices

    check_label_candidate_alignment(
        candidates_list=candidate_list,
        gold_model_names=gold_model_names,
        gold_indices=[0] * len(y_indices),
        registry=trainer._router_registry,
        debug=(trainer.state.global_step < 3),
    )

    hidden_states_valid = hidden_states[valid_indices]
    labels_valid = labels[valid_indices]
    attention_mask_valid = attention_mask[valid_indices] if attention_mask is not None else None

    prompt_len = inputs.get("prompt_len")
    if prompt_len is None:
        print("  ⚠️ WARNING: prompt_len not found in batch! Falling back to label inference.")
        print("             This indicates RouterDataCollator is not being used.")
        prompt_len = torch.zeros(batch_size, dtype=torch.long, device=device)
        for i in range(batch_size):
            for j in range(labels.shape[1]):
                if labels[i, j] != -100:
                    prompt_len[i] = j
                    break
            else:
                prompt_len[i] = max(1, labels.shape[1] // 2)

    prompt_len_valid = prompt_len[valid_indices]
    effective_use_soft_targets_local = use_soft_targets if use_soft_targets is not None else trainer._router_use_soft_targets
    neighbor_indices = None
    neighbor_positions = None
    if effective_use_soft_targets_local:
        neighbor_indices = []
        neighbor_positions = []
        for y_idx in y_indices:
            neighbors = trainer._router_registry.get_neighbors(y_idx, k=trainer._router_soft_target_k_neighbors)
            neighbor_indices.append(neighbors)
        for i, neighbors in enumerate(neighbor_indices):
            if i >= len(candidate_list) or not neighbors:
                neighbor_positions.append([])
                continue
            positions_i = []
            neighbor_set = set(neighbors)
            for pos, cand_idx in enumerate(candidate_list[i][1:], start=1):
                if cand_idx in neighbor_set:
                    positions_i.append(pos)
            neighbor_positions.append(positions_i)

    prompt_mask = extract_prompt_mask(
        prompt_len_valid,
        attention_mask_valid if attention_mask_valid is not None else torch.ones_like(labels_valid),
        labels=labels_valid if (trainer.state.global_step < 3) else None,
        debug=False,
        global_step=trainer.state.global_step,
    )

    effective_soft_target_eps_local = soft_target_eps if soft_target_eps is not None else trainer._router_soft_target_eps
    need_per_example = apply_replay_multiplier and replay_loss_multiplier > 1.0

    if need_per_example:
        loss_mean, loss_per_example, accuracy_metrics = compute_routing_loss(
            router_model=trainer._router_model,
            hidden_states=hidden_states_valid,
            labels=labels_valid,
            attention_mask=attention_mask_valid,
            candidate_indices=candidate_indices,
            prompt_len=prompt_len_valid,
            use_soft_targets=effective_use_soft_targets_local,
            soft_target_eps=effective_soft_target_eps_local,
            neighbor_indices=neighbor_indices,
            neighbor_positions=neighbor_positions,
            device=device,
            return_accuracy=True,
            return_per_example=True,
            debug=False,
            global_step=trainer.state.global_step,
        )
        is_replay = inputs.get("is_replay", None)
        if is_replay is not None:
            if not isinstance(is_replay, list):
                is_replay = list(is_replay) if hasattr(is_replay, "__iter__") else [bool(is_replay)] * batch_size

            is_replay_valid = torch.tensor([is_replay[i] for i in valid_indices], dtype=torch.float32, device=device)
            num_replay_valid = is_replay_valid.sum().item()
            num_total_valid = len(is_replay_valid)
            replay_valid_fraction = num_replay_valid / num_total_valid if num_total_valid > 0 else 0.0
            num_replay_original = sum(1 for r in is_replay if r)
            num_total_original = len(is_replay)
            replay_ratio_original = num_replay_original / num_total_original if num_total_original > 0 else 0.0
            weights = 1.0 + (replay_loss_multiplier - 1.0) * is_replay_valid
            weighted_loss_per_example = loss_per_example * weights
            loss = weighted_loss_per_example.mean()
            if replay_ratio_original > 0.1 and replay_valid_fraction < 0.01:
                if not hasattr(trainer, "_replay_drop_warning_count"):
                    trainer._replay_drop_warning_count = 0
                trainer._replay_drop_warning_count += 1
                if trainer._replay_drop_warning_count <= 5 or trainer.state.global_step % 200 == 0:
                    print(
                        f"  ⚠️  [Replay Drop Warning] Step {trainer.state.global_step}: "
                        f"Original batch has {replay_ratio_original:.1%} replay, "
                        f"but only {replay_valid_fraction:.1%} in valid examples. "
                        f"Replay may be dropped by unknown model filtering!"
                    )
        else:
            loss = loss_mean
    else:
        loss, accuracy_metrics = compute_routing_loss(
            router_model=trainer._router_model,
            hidden_states=hidden_states_valid,
            labels=labels_valid,
            attention_mask=attention_mask_valid,
            candidate_indices=candidate_indices,
            prompt_len=prompt_len_valid,
            use_soft_targets=effective_use_soft_targets_local,
            soft_target_eps=effective_soft_target_eps_local,
            neighbor_indices=neighbor_indices,
            neighbor_positions=neighbor_positions,
            device=device,
            return_accuracy=True,
            return_per_example=False,
            debug=False,
            global_step=trainer.state.global_step,
        )

    if accuracy_metrics:
        accuracy_only = {k: v for k, v in accuracy_metrics.items() if k != "compute"}
        if accuracy_only:
            if not hasattr(trainer, "_router_accuracy_sum"):
                trainer._router_accuracy_sum = {k: 0.0 for k in accuracy_only.keys()}
                trainer._router_accuracy_count = 0
            for key, value in accuracy_only.items():
                if isinstance(value, (int, float)):
                    trainer._router_accuracy_sum[key] += value
            trainer._router_accuracy_count += 1

    step = int(getattr(trainer.state, "global_step", -1))
    should_log = (step < 10) or (step % 100 == 0)
    if should_log:
        with torch.no_grad():
            metrics_logits = trainer._router_model(
                hidden_states_valid,
                extract_prompt_mask(
                    prompt_len_valid,
                    attention_mask_valid if attention_mask_valid is not None else torch.ones_like(labels_valid),
                    debug=False,
                    global_step=0,
                ),
                candidate_indices,
            )
            _metrics = compute_router_metrics(
                logits=metrics_logits,
                candidate_indices=candidate_indices,
                candidate_builder=trainer._router_candidate_builder,
                y_indices=y_indices,
                domains=normalized_domains,
                hard_negative_cache=trainer._router_hard_negative_cache,
            )

    if trainer.state.global_step % trainer._router_mine_every_steps == 0 and trainer.state.global_step > 0:
        batch_examples = []
        for i, idx in enumerate(valid_indices):
            prompt_mask = extract_prompt_mask(
                prompt_len[idx : idx + 1],
                attention_mask[idx : idx + 1] if attention_mask is not None else torch.ones(1, labels.shape[1], device=device),
                debug=False,
                global_step=0,
            )
            prompt_emb = trainer._router_model.encode_prompt(hidden_states[idx : idx + 1], prompt_mask).squeeze(0)
            batch_examples.append(
                {
                    "model_idx": y_indices[i],
                    "domain": domains[idx],
                    "prompt_embedding": prompt_emb.detach(),
                }
            )

        trainer._router_hard_miner.update_cache(
            batch_examples=batch_examples,
            router_model=trainer._router_model,
            max_examples=128,
        )
        trainer._router_hard_negative_cache.update(trainer._router_hard_miner.cache)
        stats = trainer._router_hard_miner.get_stats()
        print(
            f"  [Hard Mining @ step {trainer.state.global_step}] "
            f"Updates: {stats['num_updates']}, "
            f"Examples processed: {stats['num_examples_processed']}, "
            f"Cache size: {stats['cache_size']} (trainer cache: {len(trainer._router_hard_negative_cache)})"
        )

    return loss
