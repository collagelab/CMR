from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def should_apply_in_phase(apply_phase: str, is_phase1: bool, two_phase_enabled: bool) -> bool:
    """Resolve whether a regularizer should run in current phase."""
    if apply_phase == "both":
        return True
    if apply_phase == "phase1":
        if two_phase_enabled:
            return is_phase1
        return True
    if apply_phase == "phase2":
        return not is_phase1
    return False


def compute_router_anchor_loss(
    *,
    router_model,
    router_anchor_ref_cpu: torch.Tensor,
    router_anchor_ref: Optional[torch.Tensor],
    router_anchor_mode: str,
    router_anchor_scope: str,
    router_anchor_M_old: Optional[int],
    router_anchor_candidate_indices=None,
    router_anchor_y_indices=None,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Compute embedding anchor loss and return loss + possibly-updated device ref."""
    if router_model is None or router_anchor_ref_cpu is None:
        return None, router_anchor_ref
    if router_anchor_M_old is None or router_anchor_M_old <= 0:
        return None, router_anchor_ref

    device = router_model.model_embeddings.weight.device
    dtype = router_model.model_embeddings.weight.dtype
    m_old = min(int(router_anchor_M_old), router_model.model_embeddings.weight.shape[0])
    if m_old <= 0:
        return None, router_anchor_ref

    if (
        router_anchor_ref is None
        or router_anchor_ref.device != device
        or router_anchor_ref.dtype != dtype
    ):
        router_anchor_ref = router_anchor_ref_cpu.to(device=device, dtype=dtype)

    e_old = router_model.model_embeddings.weight[:m_old]
    ref_rows = router_anchor_ref[:m_old]

    if router_anchor_scope == "touched":
        if router_anchor_candidate_indices is None or router_anchor_y_indices is None:
            row_indices = torch.arange(m_old, device=device)
        else:
            all_indices = set()
            for y_idx in router_anchor_y_indices:
                if isinstance(y_idx, torch.Tensor):
                    y_idx = y_idx.item()
                if y_idx < m_old:
                    all_indices.add(int(y_idx))

            if isinstance(router_anchor_candidate_indices, torch.Tensor):
                candidate_flat = router_anchor_candidate_indices.flatten().detach().cpu().numpy()
            else:
                candidate_flat = []
                for cand_list in router_anchor_candidate_indices:
                    if isinstance(cand_list, (list, tuple)):
                        candidate_flat.extend(cand_list)
                    else:
                        candidate_flat.append(cand_list)
                candidate_flat = np.array(candidate_flat)

            for idx in candidate_flat:
                idx_int = int(idx)
                if idx_int < m_old:
                    all_indices.add(idx_int)

            if not all_indices:
                zero = torch.tensor(0.0, device=device, dtype=dtype, requires_grad=True)
                return zero, router_anchor_ref

            row_indices = torch.tensor(sorted(all_indices), dtype=torch.long, device=device)

        e_old = e_old[row_indices]
        ref_rows = ref_rows[row_indices]

    e_old_float = e_old.float()
    ref_rows_float = ref_rows.float()

    if router_anchor_mode == "normalized":
        e = F.normalize(e_old_float, p=2, dim=-1)
        r = F.normalize(ref_rows_float, p=2, dim=-1)
        cos = (e * r).sum(dim=-1).clamp(-1, 1)
        anchor_loss = (1 - cos).mean()
    else:
        diff = e_old_float - ref_rows_float
        anchor_loss = diff.pow(2).mean()
    return anchor_loss, router_anchor_ref


def compute_router_proj_anchor_loss(
    *,
    router_model,
    router_proj_anchor_ref_cpu,
    router_proj_anchor_ref,
) -> Tuple[Optional[torch.Tensor], Optional[dict]]:
    """Compute projection anchor loss and return loss + possibly-updated ref cache."""
    if router_model is None or router_proj_anchor_ref_cpu is None:
        return None, router_proj_anchor_ref

    proj = router_model.prompt_projection
    device = next(proj.parameters()).device
    dtype = next(proj.parameters()).dtype

    if (
        router_proj_anchor_ref is None
        or next(iter(router_proj_anchor_ref.values())).device != device
        or next(iter(router_proj_anchor_ref.values())).dtype != dtype
    ):
        router_proj_anchor_ref = {
            k: v.to(device=device, dtype=dtype) for k, v in router_proj_anchor_ref_cpu.items()
        }

    losses = []
    for name, p in proj.named_parameters():
        if name not in router_proj_anchor_ref:
            continue
        ref = router_proj_anchor_ref[name]
        losses.append((p.float() - ref.float()).pow(2).mean())

    if not losses:
        return None, router_proj_anchor_ref
    return torch.stack(losses).mean(), router_proj_anchor_ref
