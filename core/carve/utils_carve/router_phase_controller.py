import gc

import torch
from torch import nn


class RouterPhaseController:
    """Owns phase-specific LM freeze/unfreeze and router LR synchronization."""

    def __init__(self, trainer):
        self.trainer = trainer
        self._lm_param_requires_grad_original = None
        self._lm_param_groups_original_lr = None
        self._exp1_preservation_hooks = []
        self._exp1_preservation_applied = False

    def snapshot_lm_requires_grad_state(self, model: nn.Module):
        if self._lm_param_requires_grad_original is not None:
            return
        snapshot = {}
        if hasattr(model, "base_model"):
            for param in model.base_model.parameters():
                snapshot[id(param)] = bool(param.requires_grad)
        else:
            for param in model.parameters():
                snapshot[id(param)] = bool(param.requires_grad)
        self._lm_param_requires_grad_original = snapshot

    def restore_lm_requires_grad_state(self, model: nn.Module):
        if self._lm_param_requires_grad_original is None:
            return
        if hasattr(model, "base_model"):
            for param in model.base_model.parameters():
                param.requires_grad = self._lm_param_requires_grad_original.get(id(param), bool(param.requires_grad))
        else:
            for param in model.parameters():
                param.requires_grad = self._lm_param_requires_grad_original.get(id(param), bool(param.requires_grad))
        self._lm_param_requires_grad_original = None

    def phase_transition_memory_cleanup(self):
        if self.trainer.optimizer is not None:
            try:
                self.trainer.optimizer.zero_grad(set_to_none=True)
            except Exception:
                pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def freeze_lm_parameters(self, model: nn.Module):
        self.snapshot_lm_requires_grad_state(model)

        if self.trainer._router_model is not None:
            for param in self.trainer._router_model.parameters():
                param.requires_grad = True

        if self.trainer.optimizer is None:
            if hasattr(model, "base_model"):
                for param in model.base_model.parameters():
                    param.requires_grad = False
                if hasattr(model, "router"):
                    for param in model.router.parameters():
                        param.requires_grad = True
            else:
                for param in model.parameters():
                    param.requires_grad = False
            print("  [Phase 1] Set requires_grad=False for LM parameters (optimizer not available)")
            return

        if self._lm_param_groups_original_lr is None:
            self._lm_param_groups_original_lr = {}

        lm_param_group_indices = []
        router_param_group_indices = []

        for i, param_group in enumerate(self.trainer.optimizer.param_groups):
            param_ids_in_group = {id(p) for p in param_group["params"]}
            is_router_group = False
            if self.trainer._router_model is not None:
                router_param_ids = {id(p) for p in self.trainer._router_model.parameters()}
                if param_ids_in_group & router_param_ids:
                    is_router_group = True
                    router_param_group_indices.append(i)

            group_name = param_group.get("name", "").lower()
            if not is_router_group and ("router" in group_name):
                is_router_group = True
                router_param_group_indices.append(i)

            if not is_router_group:
                lm_param_group_indices.append(i)
                if i not in self._lm_param_groups_original_lr:
                    self._lm_param_groups_original_lr[i] = param_group["lr"]

        for i in lm_param_group_indices:
            self.trainer.optimizer.param_groups[i]["lr"] = 0.0

        for i in router_param_group_indices:
            if self.trainer.optimizer.param_groups[i]["lr"] == 0.0:
                if hasattr(self.trainer, "_router_phase1_proj_lr"):
                    if "proj" in self.trainer.optimizer.param_groups[i].get("name", "").lower():
                        self.trainer.optimizer.param_groups[i]["lr"] = self.trainer._router_phase1_proj_lr
                    else:
                        self.trainer.optimizer.param_groups[i]["lr"] = self.trainer._router_phase1_embedding_lr
                else:
                    self.trainer.optimizer.param_groups[i]["lr"] = self.trainer._router_proj_lr

        if hasattr(model, "base_model"):
            for param in model.base_model.parameters():
                param.requires_grad = False
            if hasattr(model, "router"):
                for param in model.router.parameters():
                    param.requires_grad = True
        else:
            for param in model.parameters():
                param.requires_grad = False

        if self.trainer._router_model is not None:
            for param in self.trainer._router_model.parameters():
                param.requires_grad = True

        if self.trainer._router_model is not None:
            router_trainable_count = sum(1 for p in self.trainer._router_model.parameters() if p.requires_grad)
            router_total_count = sum(1 for _ in self.trainer._router_model.parameters())
            if router_total_count > 0 and router_trainable_count == 0:
                raise RuntimeError("CRITICAL: All router parameters are frozen! Router must be trainable in Phase 1.")

    def apply_exp1_preservation(self):
        if not self.trainer._router_exp1_preservation_enable:
            return
        if self.trainer._router_model is None:
            return
        m_old = self.trainer._router_exp1_preservation_M_old
        if m_old is None:
            print("  ⚠️  [Exp1-Preservation] M_old not set, skipping preservation mode")
            return
        m_new = len(self.trainer._router_registry)
        if m_new <= m_old:
            print(f"  ⚠️  [Exp1-Preservation] M_new ({m_new}) <= M_old ({m_old}), skipping preservation mode")
            return
        emb_weight = self.trainer._router_model.model_embeddings.weight

        def zero_old_grad_hook(grad):
            if grad is not None:
                grad_clone = grad.clone()
                grad_clone[:m_old] = 0.0
                return grad_clone
            return grad

        hook_handle = emb_weight.register_hook(zero_old_grad_hook)
        self._exp1_preservation_hooks.append(hook_handle)
        self._exp1_preservation_applied = True

    def remove_exp1_preservation(self):
        if not self.trainer._router_exp1_preservation_enable or not self._exp1_preservation_applied:
            return
        if self.trainer._router_model is None:
            return
        for hook_handle in self._exp1_preservation_hooks:
            hook_handle.remove()
        self._exp1_preservation_hooks = []
        self._exp1_preservation_applied = False

    def unfreeze_lm_parameters(self, model: nn.Module):
        if self.trainer._router_model is not None:
            for param in self.trainer._router_model.parameters():
                param.requires_grad = True

        if self.trainer.optimizer is None:
            self.restore_lm_requires_grad_state(model)
            if hasattr(model, "router"):
                for param in model.router.parameters():
                    param.requires_grad = True
            print("  [Phase 2] Set requires_grad=True for LM parameters (optimizer not available)")
            return

        if self._lm_param_groups_original_lr is not None:
            num_restored = 0
            for i, original_lr in self._lm_param_groups_original_lr.items():
                if i < len(self.trainer.optimizer.param_groups):
                    self.trainer.optimizer.param_groups[i]["lr"] = original_lr
                    num_restored += 1
            print(f"  [Phase 2] Restored original LR for {num_restored} LM param groups")
            self._lm_param_groups_original_lr = None

        self.restore_lm_requires_grad_state(model)
        if hasattr(model, "router"):
            for param in model.router.parameters():
                param.requires_grad = True
        if self.trainer._router_model is not None:
            for param in self.trainer._router_model.parameters():
                param.requires_grad = True

    def update_router_learning_rates(self, proj_lr: float, embedding_lr: float):
        if self.trainer.optimizer is None:
            return
        updated_proj = False
        updated_embedding = False
        for param_group in self.trainer.optimizer.param_groups:
            name_lower = param_group.get("name", "").lower()
            if "router" in name_lower or "proj" in name_lower:
                if "embedding" not in name_lower:
                    param_group["lr"] = proj_lr
                    updated_proj = True
            elif "embedding" in name_lower or "model_embedding" in name_lower:
                param_group["lr"] = embedding_lr
                updated_embedding = True
        if not updated_proj or not updated_embedding:
            for param_group in self.trainer.optimizer.param_groups:
                if len(param_group["params"]) > 0 and self.trainer._router_model is not None:
                    router_param_ids = {id(p) for p in self.trainer._router_model.parameters()}
                    if any(id(p) in router_param_ids for p in param_group["params"]):
                        if (
                            hasattr(self.trainer._router_model, "projection")
                            and any(id(p) in router_param_ids for p in self.trainer._router_model.projection.parameters())
                        ):
                            param_group["lr"] = proj_lr
                            updated_proj = True
                        elif (
                            hasattr(self.trainer._router_model, "model_embedding")
                            and any(id(p) in router_param_ids for p in self.trainer._router_model.model_embedding.parameters())
                        ):
                            param_group["lr"] = embedding_lr
                            updated_embedding = True
        if updated_proj or updated_embedding:
            print(f"  [LR Update] Router proj_lr: {proj_lr:.2e}, embedding_lr: {embedding_lr:.2e}")

    def on_lr_scheduler_step(self, global_step: int):
        if self.trainer.optimizer is None:
            return
        if self.trainer._router_two_phase_enable:
            step_in_experience = global_step - (self.trainer._experience_start_global_step or 0)
            if self.trainer._phase1_steps is not None and step_in_experience < self.trainer._phase1_steps:
                self.update_router_learning_rates(
                    self.trainer._router_phase1_proj_lr,
                    self.trainer._router_phase1_embedding_lr,
                )
            else:
                self.update_router_learning_rates(
                    self.trainer._original_router_proj_lr,
                    self.trainer._original_router_embedding_lr,
                )
        else:
            self.update_router_learning_rates(
                self.trainer._original_router_proj_lr,
                self.trainer._original_router_embedding_lr,
            )
