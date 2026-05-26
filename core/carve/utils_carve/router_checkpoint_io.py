import json
import os
from pathlib import Path
from typing import Optional

import torch


def normalize_registry_path(path: str) -> str:
    """If path is a directory, append model_registry.json."""
    return os.path.join(path, "model_registry.json") if os.path.isdir(path) else path


def find_latest_checkpoint(directory: Path) -> Optional[Path]:
    """Return checkpoint-N subdirectory with the highest N, or None."""
    candidates = [
        d for d in directory.iterdir()
        if d.is_dir() and d.name.startswith("checkpoint-")
    ]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda d: int(d.name.split("-")[1]) if d.name.split("-")[1].isdigit() else -1,
    )


def save_router_checkpoint(
    *,
    output_dir: str,
    router_model,
    router_registry,
    router_candidate_builder,
    router_registry_base_path: Optional[str] = None,
    router_exp1_preservation_M_old: Optional[int] = None,
    router_anchor_enable: bool = False,
    router_anchor_M_old: Optional[int] = None,
    router_anchor_mode: Optional[str] = None,
    router_anchor_ref_cpu=None,
) -> None:
    """Save router model, registry, and router config for evaluation."""
    if router_model is None or router_registry is None:
        return

    os.makedirs(output_dir, exist_ok=True)

    router_path = os.path.join(output_dir, "router_model.pt")
    torch.save(router_model.state_dict(), router_path)
    print(f"✓ Saved router model to {router_path}")

    registry_path = os.path.join(output_dir, "model_registry.json")
    router_registry.save(registry_path)

    router_config = {
        "num_models": len(router_registry),
        "embedding_dim": router_model.embedding_dim,
        "lm_hidden_size": router_model.lm_hidden_size,
        "tau": router_model.tau,
        "pooling": router_model.pooling,
        "K_total": router_candidate_builder.K_total,
        "K_semantic": router_candidate_builder.K_semantic,
        "K_far": router_candidate_builder.K_far,
        "K_hard": router_candidate_builder.K_hard,
    }
    if router_registry_base_path:
        router_config["router_registry_base_path"] = router_registry_base_path
    if router_exp1_preservation_M_old is not None:
        router_config["router_exp1_preservation_M_old"] = router_exp1_preservation_M_old
    if router_anchor_enable and router_anchor_M_old is not None:
        router_config["router_anchor_M_old"] = router_anchor_M_old
        router_config["router_anchor_mode"] = router_anchor_mode

    config_path = os.path.join(output_dir, "router_config.json")
    with open(config_path, "w") as f:
        json.dump(router_config, f, indent=2)

    if router_anchor_enable and router_anchor_ref_cpu is not None:
        anchor_ref_path = os.path.join(output_dir, "router_anchor_ref.pt")
        torch.save(router_anchor_ref_cpu, anchor_ref_path)
