import traceback
from typing import Any, Dict, Optional, Tuple, Union

import torch
from torch import nn

from .router_batch_loss import compute_routing_loss_for_batch as _compute_routing_loss_for_batch_impl
from .router_regularizers import should_apply_in_phase


class RouterOutputs:
    """Minimal model-output container used inside compute_loss."""

    __slots__ = ("loss", "hidden_states")

    def __init__(self, loss_val, last_hidden):
        self.loss = loss_val
        self.hidden_states = (last_hidden,)


def compute_loss_pipeline(
    trainer,
    model: nn.Module,
    inputs: Dict[str, torch.Tensor],
    return_outputs: bool = False,
    num_items_in_batch: Optional[int] = None,
    *,
    router_modes,
    pure_router_modes,
) -> Union[torch.Tensor, Tuple[torch.Tensor, Any]]:
    """Full trainer loss pipeline extracted from NeighborConsistencySFTTrainer.compute_loss."""
    global_step = trainer._global_step
    if global_step == 0:
        if trainer._router_freeze_lm:
            for name, param in model.named_parameters():
                if "router" in name.lower() or "prompt_projection" in name.lower() or "model_embeddings" in name.lower():
                    continue
                if param.requires_grad:
                    param.requires_grad = False
            if hasattr(model, "base_model"):
                model.base_model.eval()
            elif hasattr(model, "model"):
                model.model.eval()

    if trainer._router_two_phase_enable:
        if trainer._experience_start_global_step is None:
            trainer._experience_start_global_step = global_step
            print(f"\n  [Two-Phase Training] Experience started at global_step={global_step}")

        step_in_experience = global_step - trainer._experience_start_global_step

        if trainer._phase1_steps is None:
            if hasattr(trainer, "train_dataloader") and trainer.train_dataloader is not None:
                steps_per_epoch = len(trainer.train_dataloader) // trainer.args.gradient_accumulation_steps
                total_optimizer_steps = int(steps_per_epoch * trainer.args.num_train_epochs)
                if hasattr(trainer.args, "max_steps") and trainer.args.max_steps is not None and trainer.args.max_steps > 0:
                    total_optimizer_steps = trainer.args.max_steps
                trainer._phase1_steps = int(trainer._router_phase1_frac * total_optimizer_steps)
                print(f"  [Two-Phase Training] Computed phase1_steps={trainer._phase1_steps} from dataloader")

        is_phase1 = trainer._phase1_steps is not None and step_in_experience < trainer._phase1_steps
        current_phase = "Phase 1" if is_phase1 else "Phase 2"

        if current_phase != trainer._current_phase:
            if trainer._current_phase is not None:
                print(f"[PHASE TRANSITION] Switching from {trainer._current_phase} to {current_phase}")
                trainer._log_router_param_groups(step=global_step)
            else:
                print(f"[TWO-PHASE TRAINING] Starting in {current_phase}")
                if is_phase1:
                    if trainer._router_exp1_preservation_enable:
                        trainer._phase_controller.apply_exp1_preservation()
                    if trainer._router_anchor_enable:
                        lambda_val = (
                            float(trainer._router_anchor_lambda)
                            if not isinstance(trainer._router_anchor_lambda, (int, float))
                            else trainer._router_anchor_lambda
                        )
                        print(f"      lambda: {lambda_val:.1e}")
                else:
                    print("  Phase 2 settings (using original config):")

            if trainer._current_phase is not None and trainer._current_phase != current_phase:
                if current_phase == "Phase 2":
                    if trainer._router_freeze_lm:
                        print("  [Phase 2] router_freeze_lm=True, keeping LM frozen")
                    else:
                        trainer._phase_controller.unfreeze_lm_parameters(model)
                    trainer._phase_controller.remove_exp1_preservation()
                    trainer._phase_controller.phase_transition_memory_cleanup()

            trainer._current_phase = current_phase
            trainer._phase_transition_logged = True

        if is_phase1:
            effective_loss_mode = trainer._router_phase1_loss_mode
            effective_router_loss_weight = trainer._router_phase1_router_loss_weight
            effective_lm_loss_weight = trainer._router_phase1_lm_loss_weight
            effective_use_soft_targets = trainer._router_phase1_use_soft_targets
            effective_soft_target_eps = trainer._router_phase1_soft_target_eps

            if not trainer._lm_params_frozen:
                trainer._phase_controller.freeze_lm_parameters(model)
                trainer._lm_params_frozen = True

            if trainer.optimizer is not None:
                trainer._phase_controller.update_router_learning_rates(
                    trainer._router_phase1_proj_lr,
                    trainer._router_phase1_embedding_lr,
                )
        else:
            effective_loss_mode = trainer._original_loss_mode
            effective_router_loss_weight = trainer._original_router_loss_weight
            effective_lm_loss_weight = 0.0 if effective_loss_mode in pure_router_modes else trainer._original_lm_loss_weight
            effective_use_soft_targets = trainer._original_router_use_soft_targets
            effective_soft_target_eps = trainer._original_router_soft_target_eps

            if trainer._lm_params_frozen:
                if trainer._router_freeze_lm:
                    trainer._lm_params_frozen = True
                else:
                    trainer._phase_controller.unfreeze_lm_parameters(model)
                    trainer._lm_params_frozen = False

            if trainer.optimizer is not None:
                trainer._phase_controller.update_router_learning_rates(
                    trainer._original_router_proj_lr,
                    trainer._original_router_embedding_lr,
                )
    else:
        is_phase1 = False
        effective_loss_mode = trainer._loss_mode
        effective_router_loss_weight = trainer._router_loss_weight
        effective_lm_loss_weight = 0.0 if effective_loss_mode in pure_router_modes else trainer._lm_loss_weight
        effective_use_soft_targets = trainer._router_use_soft_targets
        effective_soft_target_eps = trainer._router_soft_target_eps

    if (
        effective_loss_mode in pure_router_modes
        or (trainer._router_two_phase_enable and is_phase1)
        or trainer._router_freeze_lm
    ):
        trainer._ensure_frozen_lm_in_eval_mode()

    if effective_loss_mode in router_modes:
        if (trainer._router_two_phase_enable and is_phase1) or trainer._router_freeze_lm:
            with torch.no_grad():
                outputs_no_grad = model(
                    input_ids=inputs.get("input_ids"),
                    attention_mask=inputs.get("attention_mask"),
                    labels=inputs.get("labels"),
                    output_hidden_states=True,
                    return_dict=True,
                )
            last_hidden_state = outputs_no_grad.hidden_states[-1].clone()
            loss_value = outputs_no_grad.loss
            del outputs_no_grad
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            last_hidden_state = last_hidden_state.requires_grad_()
            outputs = RouterOutputs(loss_value, last_hidden_state)
            loss_supervised = loss_value
        else:
            outputs_full = model(
                input_ids=inputs.get("input_ids"),
                attention_mask=inputs.get("attention_mask"),
                labels=inputs.get("labels"),
                output_hidden_states=True,
                return_dict=True,
            )
            last_hidden_state = outputs_full.hidden_states[-1].clone()
            loss_value = outputs_full.loss
            del outputs_full
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            last_hidden_state = last_hidden_state.requires_grad_()
            outputs = RouterOutputs(loss_value, last_hidden_state)
            loss_supervised = loss_value
    else:
        if return_outputs:
            loss_supervised, outputs = super(type(trainer), trainer).compute_loss(
                model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch
            )
        else:
            loss_supervised = super(type(trainer), trainer).compute_loss(
                model, inputs, return_outputs=False, num_items_in_batch=num_items_in_batch
            )
            outputs = None

    total_loss = effective_lm_loss_weight * loss_supervised

    if effective_loss_mode in router_modes:
        try:
            apply_replay_multiplier = (
                trainer._router_two_phase_enable and is_phase1 and trainer._router_replay_loss_multiplier > 1.0
            )

            router_loss = _compute_routing_loss_for_batch_impl(
                trainer,
                model=model,
                inputs=inputs,
                outputs=outputs,
                apply_replay_multiplier=apply_replay_multiplier,
                replay_loss_multiplier=trainer._router_replay_loss_multiplier if apply_replay_multiplier else 1.0,
                use_soft_targets=effective_use_soft_targets,
                soft_target_eps=effective_soft_target_eps,
            )

            if router_loss is not None and router_loss > 0:
                trainer._router_accum.update(router_loss.item())

                if effective_loss_mode in pure_router_modes:
                    total_loss = effective_router_loss_weight * router_loss
                else:
                    total_loss = (effective_lm_loss_weight * loss_supervised) + (effective_router_loss_weight * router_loss)

                if effective_loss_mode in ["router+graph", "supervised+router+graph"]:
                    graph_loss = trainer._compute_router_graph_regularizer_for_batch(inputs)
                    if graph_loss is not None and graph_loss > 0:
                        trainer._router_graph_accum.update(graph_loss.item())
                        total_loss = total_loss + trainer._router_label_graph_lambda * graph_loss

                if trainer._router_ewc_enable and trainer._router_ewc_regularizer is not None and trainer._router_model is not None:
                    ewc_should_apply = should_apply_in_phase(
                        trainer._router_ewc_apply_phase,
                        is_phase1=is_phase1,
                        two_phase_enabled=trainer._router_two_phase_enable,
                    )
                    if ewc_should_apply:
                        ewc_loss = trainer._router_ewc_regularizer.compute_ewc_loss(trainer._router_model)
                        if ewc_loss > 0:
                            trainer._router_ewc_accum.update(ewc_loss.item(), ewc_loss.item())
                            total_loss = total_loss + ewc_loss

                raw_anchor_loss_val = None
                raw_proj_anchor_loss_val = None

                if not trainer._router_ewc_enable and trainer._router_anchor_enable and trainer._router_anchor_ref_cpu is not None:
                    should_apply = should_apply_in_phase(
                        trainer._router_anchor_apply_phase,
                        is_phase1=is_phase1,
                        two_phase_enabled=trainer._router_two_phase_enable,
                    )
                    if should_apply:
                        anchor_loss = trainer._compute_router_anchor_loss()
                        if anchor_loss is not None and anchor_loss > 0:
                            raw_anchor_loss_val = float(anchor_loss.detach())
                            weighted_anchor_loss = trainer._router_anchor_lambda * anchor_loss
                            trainer._router_anchor_accum.update(anchor_loss.item(), weighted_anchor_loss.item())
                            total_loss = total_loss + weighted_anchor_loss

                if (
                    not trainer._router_ewc_enable
                    and trainer._router_proj_anchor_enable
                    and trainer._router_proj_anchor_ref_cpu is not None
                ):
                    proj_should_apply = should_apply_in_phase(
                        trainer._router_proj_anchor_apply_phase,
                        is_phase1=is_phase1,
                        two_phase_enabled=trainer._router_two_phase_enable,
                    )
                    if proj_should_apply:
                        proj_anchor_loss = trainer._compute_router_proj_anchor_loss()
                        if proj_anchor_loss is not None and proj_anchor_loss > 0:
                            raw_proj_anchor_loss_val = float(proj_anchor_loss.detach())
                            weighted_proj_anchor_loss = trainer._router_proj_anchor_lambda * proj_anchor_loss
                            trainer._router_proj_anchor_accum.update(
                                proj_anchor_loss.item(), weighted_proj_anchor_loss.item()
                            )
                            total_loss = total_loss + weighted_proj_anchor_loss

                if trainer._router_anchor_adaptive_enable or trainer._router_proj_anchor_adaptive_enable:
                    trainer._update_adaptive_anchor_lambdas(
                        router_loss=float(router_loss.detach()),
                        emb_loss=raw_anchor_loss_val,
                        proj_loss=raw_proj_anchor_loss_val,
                        is_phase1=is_phase1,
                        global_step=trainer.state.global_step,
                    )
        except Exception as e:
            if trainer.state.global_step % 100 == 0:
                print(f"  Warning: Router loss computation failed: {e}")
                traceback.print_exc()

    if return_outputs:
        return total_loss, outputs
    return total_loss
