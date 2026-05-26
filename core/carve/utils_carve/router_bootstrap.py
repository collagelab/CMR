import os
import traceback
from pathlib import Path
from typing import Any, Optional

import torch
from torch import nn

from .card_init import build_card_embedding_cache, card_guided_init
from ..model_selection_carve import ModelRegistry, RouterModel


DEFAULT_ROUTER_EMBEDDING_DIM = 4096
MODEL_REGISTRY_FILENAME = "model_registry.json"
ROUTER_MODEL_FILENAME = "router_model.pt"
ROUTER_ANCHOR_REF_FILENAME = "router_anchor_ref.pt"
SENTENCE_TRANSFORMER_MODEL = "all-mpnet-base-v2"


def _safe_dataset_len(dataset: Any) -> Optional[int]:
    """Returns dataset length when available, otherwise None."""
    return len(dataset) if hasattr(dataset, "__len__") else None


def _collect_dict_examples(dataset: Any, *, dataset_name: str) -> list[dict]:
    """Collects dictionary examples from an iterable dataset."""
    if dataset is None or not hasattr(dataset, "__iter__"):
        return []

    examples: list[dict] = []
    try:
        for example in dataset:
            if isinstance(example, dict):
                examples.append(example)
    except Exception as exc:
        print(f"  Warning: Could not iterate {dataset_name} for registry: {exc}")
    return examples


def _checkpoint_sort_key(checkpoint_dir: Path) -> int:
    """Extracts numeric checkpoint step for sorting, defaulting to 0."""
    parts = checkpoint_dir.name.split("-")
    if len(parts) < 2:
        return 0
    return int(parts[1]) if parts[1].isdigit() else 0


def _sorted_checkpoint_dirs(exp_root: Path) -> list[Path]:
    """Returns checkpoint directories sorted from newest to oldest step."""
    return sorted(
        [d for d in exp_root.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")],
        key=_checkpoint_sort_key,
        reverse=True,
    )


def _find_latest_registry_in_experiment(exp_root: Path) -> Optional[str]:
    """Finds the most recent model registry file in checkpoint directories."""
    for checkpoint_dir in _sorted_checkpoint_dirs(exp_root):
        potential_registry = checkpoint_dir / MODEL_REGISTRY_FILENAME
        if potential_registry.exists():
            return str(potential_registry)
    return None


def _derive_base_registry_path_from_trainer_args(trainer) -> Optional[str]:
    """Attempts to discover a base registry path for extend mode."""
    resume_path = getattr(trainer.args, "resume_from_checkpoint", None)
    if resume_path and os.path.isdir(resume_path):
        candidate = os.path.join(resume_path, MODEL_REGISTRY_FILENAME)
        if os.path.exists(candidate):
            print(f"  Derived base registry path from resume_from_checkpoint: {candidate}")
            return candidate

    output_dir = getattr(trainer.args, "output_dir", None)
    if not output_dir:
        return None

    output_path = Path(output_dir)
    exp_root = output_path.parent if output_path.name.startswith("checkpoint-") else output_path

    found = _find_latest_registry_in_experiment(exp_root)
    if found:
        print(f"  Auto-found base registry from latest checkpoint: {found}")
        return found

    if not exp_root.parent.exists():
        return None

    parent_dir = exp_root.parent
    current_exp_name = exp_root.name.split("-")[0] if "-" in exp_root.name else None
    if not current_exp_name:
        return None

    other_exp_dirs = [
        d for d in parent_dir.iterdir() if d.is_dir() and not d.name.startswith(current_exp_name) and "-" in d.name
    ]
    other_exp_dirs.sort(key=lambda path: path.stat().st_mtime, reverse=True)

    for other_exp_dir in other_exp_dirs:
        found = _find_latest_registry_in_experiment(other_exp_dir)
        if found:
            print(f"  Auto-found base registry from previous experience ({other_exp_dir.name}): {found}")
            return found
    return None


def _validate_registry_id_stability(base_registry: ModelRegistry, extended_registry: ModelRegistry, m_old: int) -> None:
    """Ensures model IDs from the base registry are unchanged after extension."""
    violations: list[str] = []

    for idx in range(m_old):
        if idx not in extended_registry.idx2model:
            violations.append(f"Base model ID {idx} missing after extension")
            continue

        current_model = extended_registry.idx2model[idx]
        old_model = base_registry.idx2model.get(idx)
        if old_model is None:
            violations.append(f"ID {idx} exists in extended registry but not in base registry")
        elif current_model != old_model:
            violations.append(f"ID {idx} changed: base='{old_model}' -> extended='{current_model}'")

    for model_name, old_idx in base_registry.model2idx.items():
        if model_name not in extended_registry.model2idx:
            violations.append(f"Model '{model_name}' (base ID {old_idx}) missing in extended registry")
            continue

        new_idx = extended_registry.model2idx[model_name]
        if new_idx != old_idx:
            violations.append(f"Model '{model_name}' ID changed: base={old_idx} -> extended={new_idx}")

    if violations:
        error_msg = "CRITICAL: Registry ID stability violations detected:\n" + "\n".join(
            f"  - {violation}" for violation in violations[:10]
        )
        if len(violations) > 10:
            error_msg += f"\n  ... and {len(violations) - 10} more violations"
        raise ValueError(error_msg)


def _infer_router_embedding_dim(model, requested_dim: Optional[int]) -> int:
    """Resolves router embedding dimension from overrides or model configs."""
    if requested_dim is not None:
        return requested_dim
    if hasattr(model, "config") and hasattr(model.config, "hidden_size"):
        return model.config.hidden_size
    if hasattr(model, "model") and hasattr(model.model, "config"):
        return model.model.config.hidden_size
    if hasattr(model, "base_model") and hasattr(model.base_model, "config"):
        return model.base_model.config.hidden_size
    return DEFAULT_ROUTER_EMBEDDING_DIM


def _infer_resume_path(trainer, *, router_registry_init_mode: str, router_registry_base_path: Optional[str]) -> Optional[str]:
    """Infers checkpoint directory to load router weights from."""
    explicit_resume = getattr(trainer.args, "resume_from_checkpoint", None)
    if explicit_resume:
        print(f"  [Router Checkpoint] Using explicit resume_from_checkpoint: {explicit_resume}")
        return explicit_resume

    if router_registry_init_mode != "extend" or not router_registry_base_path:
        return None

    if os.path.isdir(router_registry_base_path):
        exp_dir = Path(router_registry_base_path)
        checkpoint_dir = trainer._find_latest_checkpoint(exp_dir)
        if checkpoint_dir is not None:
            potential_router = checkpoint_dir / ROUTER_MODEL_FILENAME
            if potential_router.exists():
                resolved_path = str(checkpoint_dir)
                print(
                    "  [Router Checkpoint] Inferred resume_from_checkpoint from "
                    f"router_registry_base_path (latest checkpoint): {resolved_path}"
                )
                return resolved_path

        print(f"  ⚠️  [Router Checkpoint] router_registry_base_path is directory {router_registry_base_path}")
        print("     But no checkpoint directories with router_model.pt found")
        return None

    base_registry_dir = os.path.dirname(router_registry_base_path)
    if not os.path.isdir(base_registry_dir):
        return None

    potential_router = os.path.join(base_registry_dir, ROUTER_MODEL_FILENAME)
    if os.path.exists(potential_router):
        print(f"  [Router Checkpoint] Inferred resume_from_checkpoint from router_registry_base_path: {base_registry_dir}")
        return base_registry_dir

    print(f"  ⚠️  [Router Checkpoint] router_registry_base_path points to {base_registry_dir}")
    print(f"     But router_model.pt not found at {potential_router}")
    return None


def _resolve_router_checkpoint_path(resume_path: Optional[str]) -> Optional[str]:
    """Resolves router checkpoint file path from a resume directory."""
    if not resume_path:
        return None
    if not os.path.isdir(resume_path):
        print(f"  ⚠️  resume_from_checkpoint is not a directory: {resume_path}")
        return None

    candidate = os.path.join(resume_path, ROUTER_MODEL_FILENAME)
    if os.path.exists(candidate):
        print(f"  ✓ Found router checkpoint at {candidate}")
        return candidate

    print(f"  ⚠️  Router checkpoint NOT found at {candidate}")
    print("     Router will be randomly initialized (not loading exp1 weights)")
    return None


def _maybe_init_card_encoder(trainer) -> None:
    """Best-effort local initialization of sentence-transformer encoder."""
    if trainer._card_encoder is not None:
        return

    try:
        from sentence_transformers import SentenceTransformer

        trainer._card_encoder = SentenceTransformer(
            SENTENCE_TRANSFORMER_MODEL,
            device="cpu",
            cache_folder=os.environ.get("HF_HOME", None),
            local_files_only=True,
        )
        print(f"  [Card-Init] Loaded local {SENTENCE_TRANSFORMER_MODEL} encoder.")
    except Exception as exc:
        trainer._card_encoder = None
        print(
            "  [Card-Init] SentenceTransformer unavailable locally "
            f"({exc}); using offline hashed card-text embeddings."
        )


def _apply_card_guided_embedding_init(trainer, *, embedding_table, m_old: int, m_new: int) -> None:
    """Applies card-guided init for newly added model embedding rows."""
    if not trainer._card_guided_init_enable or trainer._router_registry is None:
        return

    _maybe_init_card_encoder(trainer)
    old_model_ids = [
        trainer._router_registry.idx2model[i] for i in range(m_old) if i in trainer._router_registry.idx2model
    ]
    new_model_ids = [
        trainer._router_registry.idx2model[i] for i in range(m_old, m_new) if i in trainer._router_registry.idx2model
    ]
    if not new_model_ids or not old_model_ids:
        return

    all_ids_needed = set(old_model_ids) | set(new_model_ids)
    if trainer._card_embedding_cache is None:
        trainer._card_embedding_cache = {}

    missing = [model_id for model_id in all_ids_needed if model_id not in trainer._card_embedding_cache]
    if missing:
        new_entries = build_card_embedding_cache(missing, trainer._router_registry, trainer._card_encoder)
        trainer._card_embedding_cache.update(new_entries)

    init_rows = card_guided_init(
        new_model_ids=new_model_ids,
        old_model_ids=old_model_ids,
        old_emb=embedding_table[:m_old],
        card_cache=trainer._card_embedding_cache,
        tau=trainer._card_init_tau,
        topk=trainer._card_init_topk,
        scope=trainer._card_init_scope,
        min_sim_threshold=trainer._card_init_min_sim_threshold,
        fallback_domain=trainer._card_init_fallback_domain,
        registry=trainer._router_registry,
    )
    embedding_table[m_old : m_old + len(new_model_ids)].copy_(init_rows)
    print(
        f"  [Card-Init] Card-guided init: {m_new - m_old} new models "
        f"initialised from top-{trainer._card_init_topk} card neighbours "
        f"(tau={trainer._card_init_tau}, scope={trainer._card_init_scope})"
    )


def _load_router_checkpoint(trainer, *, router_checkpoint_path: str, device: torch.device) -> None:
    """Loads router checkpoint with robust embedding resize handling."""
    print(f"\n  [Router Checkpoint Loading] Loading router weights from {router_checkpoint_path}")
    try:
        router_state = torch.load(router_checkpoint_path, map_location=device)
        print(
            f"  [Router Checkpoint Loading] Checkpoint loaded, keys: {list(router_state.keys())[:5]}... "
            f"(total: {len(router_state)} keys)"
        )

        embeddings_manually_handled = False
        if "model_embeddings.weight" in router_state:
            old_emb_shape = router_state["model_embeddings.weight"].shape
            new_emb_shape = trainer._router_model.model_embeddings.weight.shape
            m_old, d_old = old_emb_shape
            m_new, d_new = new_emb_shape
            print(
                f"  [Router Checkpoint Loading] Embedding shapes: checkpoint={old_emb_shape}, current={new_emb_shape}"
            )

            if trainer._router_exp1_preservation_enable and trainer._router_exp1_preservation_M_old is None:
                trainer._router_exp1_preservation_M_old = m_old
                print(f"  [Exp1-Preservation] Auto-detected M_old={m_old} from checkpoint")

            if trainer._router_anchor_enable and trainer._router_anchor_M_old is None:
                trainer._router_anchor_M_old = m_old
                print(f"  [Router Anchor] Auto-detected M_old={m_old} from checkpoint")

            if m_new != m_old or d_new != d_old:
                print(f"  Router embedding size changed: {old_emb_shape} -> {new_emb_shape}")
                if d_new != d_old:
                    raise ValueError(
                        f"Embedding dimension mismatch: checkpoint has {d_old}, but model requires {d_new}. Cannot resize."
                    )

                old_emb = router_state["model_embeddings.weight"]
                new_emb = trainer._router_model.model_embeddings.weight
                overlap = min(m_old, m_new)

                with torch.no_grad():
                    new_emb[:overlap].copy_(old_emb[:overlap])

                    if m_new > m_old:
                        nn.init.xavier_uniform_(new_emb[m_old:])
                        try:
                            _apply_card_guided_embedding_init(trainer, embedding_table=new_emb, m_old=m_old, m_new=m_new)
                        except Exception as exc:
                            print(f"  ⚠ Card-guided init failed ({exc}), keeping Xavier init")
                        print(f"  ✓ Copied {overlap} embedding rows, initialized {m_new - m_old} new rows")
                    else:
                        print(f"  ✓ Copied {overlap} embedding rows (registry shrunk)")

                del old_emb
                embeddings_manually_handled = True
                router_state = {k: v for k, v in router_state.items() if k != "model_embeddings.weight"}

        missing, _unexpected = trainer._router_model.load_state_dict(router_state, strict=False)
        _ = [
            key for key in missing if not (key == "model_embeddings.weight" and embeddings_manually_handled)
        ]

        del router_state
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as exc:
        print(f"  ❌ Error loading router checkpoint: {exc}")
        traceback.print_exc()
        print("  ⚠️  Router will be randomly initialized (checkpoint loading failed)")


def _infer_anchor_m_old(trainer, *, router_registry_init_mode: str, base_registry_path: Optional[str]) -> Optional[int]:
    """Infers M_old for anchoring based on available experiment metadata."""
    if trainer._router_anchor_M_old is not None:
        return trainer._router_anchor_M_old

    if trainer._router_exp1_preservation_M_old is not None:
        anchor_m_old = trainer._router_exp1_preservation_M_old
        print(f"  [Router Anchor] Using M_old from exp1-preservation: {anchor_m_old}")
        return anchor_m_old

    if router_registry_init_mode != "extend" or not base_registry_path:
        return None

    try:
        normalized_path = trainer._normalize_registry_path(base_registry_path)
        if os.path.exists(normalized_path):
            base_registry = ModelRegistry.load(normalized_path)
            return len(base_registry)
    except Exception as exc:
        print(f"  ⚠️  [Router Anchor] Could not infer M_old from base registry: {exc}")
    return None


def _configure_router_anchor_reference(
    trainer,
    *,
    router_registry_init_mode: str,
    base_registry_path: Optional[str],
    router_checkpoint_path: Optional[str],
) -> None:
    """Configures router anchor snapshot or disables anchoring when unavailable."""
    if not trainer._router_anchor_enable:
        return

    anchor_m_old = _infer_anchor_m_old(
        trainer,
        router_registry_init_mode=router_registry_init_mode,
        base_registry_path=base_registry_path,
    )
    if anchor_m_old is None or anchor_m_old <= 0:
        print("  ⚠️  [Router Anchor] Could not determine M_old, disabling anchoring")
        trainer._router_anchor_enable = False
        return

    m_new = len(trainer._router_registry)
    if m_new <= anchor_m_old:
        print(f"  ⚠️  [Router Anchor] M_new ({m_new}) <= M_old ({anchor_m_old}), disabling anchoring")
        trainer._router_anchor_enable = False
        return

    anchor_loaded = False
    if router_checkpoint_path:
        anchor_ref_path = os.path.join(os.path.dirname(router_checkpoint_path), ROUTER_ANCHOR_REF_FILENAME)
        if os.path.exists(anchor_ref_path):
            try:
                trainer._router_anchor_ref_cpu = torch.load(anchor_ref_path, map_location="cpu")
                if trainer._router_anchor_ref_cpu.shape[0] == anchor_m_old:
                    anchor_loaded = True
            except Exception as exc:
                print(f"  ⚠️  [Router Anchor] Could not load reference from checkpoint: {exc}")

    if not anchor_loaded:
        anchor_ref_path_exists = (
            router_checkpoint_path
            and os.path.exists(os.path.join(os.path.dirname(router_checkpoint_path), ROUTER_ANCHOR_REF_FILENAME))
        )
        is_starting_exp2_from_exp1 = (
            router_registry_init_mode == "extend"
            and base_registry_path
            and os.path.exists(base_registry_path)
            and not anchor_ref_path_exists
        )

        if is_starting_exp2_from_exp1:
            with torch.no_grad():
                e_old = trainer._router_model.model_embeddings.weight[:anchor_m_old]
                trainer._router_anchor_ref_cpu = e_old.detach().float().cpu().clone()
                trainer._router_anchor_ref_cpu.requires_grad_(False)
            print(f"  [Router Anchor] Captured exp1 reference snapshot: M_old={anchor_m_old}, M_new={m_new}")
        else:
            print("  ⚠️  [Router Anchor] Not starting exp2 from exp1 (or anchor ref missing), disabling anchoring")
            trainer._router_anchor_enable = False
            return

    trainer._router_anchor_M_old = anchor_m_old


def _capture_router_projection_anchor(trainer) -> None:
    """Captures projection anchor reference for router prompt projection."""
    if not trainer._router_proj_anchor_enable or trainer._router_model is None:
        return

    with torch.no_grad():
        trainer._router_proj_anchor_ref_cpu = {
            key: value.detach().clone().cpu().float()
            for key, value in trainer._router_model.prompt_projection.state_dict().items()
        }
        for value in trainer._router_proj_anchor_ref_cpu.values():
            value.requires_grad_(False)
        trainer._router_proj_anchor_ref = None
        print("  [Router Proj Anchor] Captured projection reference snapshot (CPU FP32)")

        proj_anchor_loss_init = trainer._compute_router_proj_anchor_loss()
        if proj_anchor_loss_init is None:
            return

        lambda_val = (
            float(trainer._router_proj_anchor_lambda)
            if not isinstance(trainer._router_proj_anchor_lambda, (int, float))
            else trainer._router_proj_anchor_lambda
        )
        print(f"  [Router Proj Anchor] Initial proj anchor loss (should be ~0): {proj_anchor_loss_init.item():.8e}")
        print(
            f"  [Router Proj Anchor] Initial weighted loss (λ={lambda_val:.1e}): "
            f"{lambda_val * proj_anchor_loss_init.item():.8e}"
        )


def bootstrap_router_state(
    trainer,
    *,
    model,
    train_dataset,
    eval_dataset,
    router_registry_init_mode: str,
    router_registry_base_path: Optional[str],
    router_registry_path: Optional[str],
    router_embedding_dim: Optional[int],
    router_tau: float,
    router_pooling: str,
) -> None:
    """Build/extend registry, initialize router, load checkpoints, and capture anchor refs."""
    print("\n[Router] Building model registry...")
    train_dataset_size = _safe_dataset_len(train_dataset)
    eval_dataset_size = _safe_dataset_len(eval_dataset) if eval_dataset else None

    if train_dataset_size:
        print(f"  Processing ALL {train_dataset_size} training examples")
    if eval_dataset_size:
        print(f"  Processing ALL {eval_dataset_size} eval examples")

    train_examples_for_registry = _collect_dict_examples(train_dataset, dataset_name="train_dataset")

    if trainer._xclr_replay_source_examples:
        train_examples_for_registry.extend(trainer._xclr_replay_source_examples)

    eval_examples_for_registry = _collect_dict_examples(eval_dataset, dataset_name="eval_dataset")

    all_examples_for_registry = train_examples_for_registry + eval_examples_for_registry
    base_registry_path = router_registry_base_path
    if router_registry_init_mode == "extend" and not base_registry_path:
        base_registry_path = _derive_base_registry_path_from_trainer_args(trainer)

        if not base_registry_path:
            print("  ⚠️  To use extend mode, provide router_registry_base_path or ensure previous checkpoint exists\n")
            router_registry_init_mode = "fresh"

    if router_registry_init_mode == "extend" and base_registry_path:
        base_registry_path = trainer._normalize_registry_path(base_registry_path)

        if os.path.exists(base_registry_path):
            print(f"  [Extend Mode] Loading base registry from {base_registry_path}")
            trainer._router_registry = ModelRegistry.load(base_registry_path)
            m_old = len(trainer._router_registry)
        else:
            print(f"  ⚠️  Base registry not found at {base_registry_path}; starting fresh registry")
            trainer._router_registry = ModelRegistry()
            m_old = 0

        print("  Extending registry with models from current dataset...")
        num_added = trainer._router_registry.extend_from_examples(
            examples=all_examples_for_registry,
            model_name_key="model_name",
            domain_key="domain",
            family_key=None,
        )
        m_new = len(trainer._router_registry)

        print(f"  Loaded base registry: {m_old} models; extended registry: {m_new} models; added: {num_added}")

        base_registry = None
        normalized_path = trainer._normalize_registry_path(base_registry_path)
        if os.path.exists(normalized_path):
            base_registry = ModelRegistry.load(normalized_path)

        if base_registry is not None:
            _validate_registry_id_stability(base_registry, trainer._router_registry, m_old)

        if base_registry is not None:
            print("  ✓ Registry ID stability check passed:")
            print(f"    - All {m_old} prior IDs unchanged (idx2model equality verified)")
            print(
                f"    - All {len(base_registry.model2idx)} prior model names have same IDs (model2idx equality verified)"
            )

    elif router_registry_path and os.path.exists(router_registry_path):
        print(f"  Loading registry from {router_registry_path}")
        trainer._router_registry = ModelRegistry.load(router_registry_path)
    else:
        print("  [Fresh Mode] Building registry from scratch")
        trainer._router_registry = ModelRegistry.from_examples(
            train_examples=train_examples_for_registry,
            replay_examples=None,
            raw_prompts=None,
        )
        if router_registry_path:
            print(f"  Saving registry to {router_registry_path}")
            os.makedirs(os.path.dirname(router_registry_path), exist_ok=True)
            trainer._router_registry.save(router_registry_path)

    print(
        f"  Registry: {len(trainer._router_registry)} unique models, "
        f"{len(trainer._router_registry.get_all_domains())} domains"
    )

    router_embedding_dim = _infer_router_embedding_dim(model, router_embedding_dim)
    first_param = next(model.parameters())
    device = first_param.device
    dtype = first_param.dtype
    lm_hidden_size = model.config.hidden_size if hasattr(model.config, "hidden_size") else DEFAULT_ROUTER_EMBEDDING_DIM

    resume_path = _infer_resume_path(
        trainer,
        router_registry_init_mode=router_registry_init_mode,
        router_registry_base_path=router_registry_base_path,
    )
    router_checkpoint_path = _resolve_router_checkpoint_path(resume_path)

    if not resume_path:
        print("  ⚠️  No resume_from_checkpoint specified")
        if router_registry_init_mode == "extend":
            print("     WARNING: Registry init_mode='extend' but no checkpoint to load from!")

    print(
        f"  Creating router model: num_models={len(trainer._router_registry)}, embedding_dim={router_embedding_dim}"
    )
    trainer._router_model = RouterModel(
        num_models=len(trainer._router_registry),
        embedding_dim=router_embedding_dim,
        lm_hidden_size=lm_hidden_size,
        tau=router_tau,
        pooling=router_pooling,
    ).to(device=device, dtype=dtype)

    if router_checkpoint_path:
        _load_router_checkpoint(trainer, router_checkpoint_path=router_checkpoint_path, device=device)

    _configure_router_anchor_reference(
        trainer,
        router_registry_init_mode=router_registry_init_mode,
        base_registry_path=base_registry_path,
        router_checkpoint_path=router_checkpoint_path,
    )
    _capture_router_projection_anchor(trainer)
