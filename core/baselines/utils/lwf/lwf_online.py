import os

import torch
import torch.nn.functional as F
from trl.trainer.sft_trainer import SFTTrainer
from typing import Any


class LwFSFTTrainerOnline(SFTTrainer):
    def __init__(
        self,
        alpha: float = 1.0,
        temperature: float = 2.0,
        kd_on_new: bool = False,
        teacher_adapter_name: str = "teacher",
        student_adapter_name: str = "student",
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.alpha = float(alpha)
        self.temperature = float(temperature)
        self.kd_on_new = bool(kd_on_new)
        self.teacher_adapter_name = teacher_adapter_name
        self.student_adapter_name = student_adapter_name
        self._debug_call_count = 0
        self._debug_warmup_prints = 3
        self._debug_print_every = max(1, int(os.getenv("LWF_DEBUG_EVERY", "200")))
        self._printed_debug_header = False

    def _should_print_debug(self) -> bool:
        self._debug_call_count += 1
        return (
            self._debug_call_count <= self._debug_warmup_prints
            or self._debug_call_count % self._debug_print_every == 0
        )

    def _extract_logits(self, outputs: Any):
        if outputs is None:
            return None
        if hasattr(outputs, "logits"):
            return outputs.logits
        if isinstance(outputs, dict) and "logits" in outputs:
            return outputs["logits"]
        raise RuntimeError("Could not find 'logits' in model outputs.")

    def _set_adapter_safe(self, model, adapter_name: str):
        if hasattr(model, "set_adapter"):
            model.set_adapter(adapter_name)
        else:
            raise RuntimeError("Model does not support set_adapter().")

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        should_print_debug = self._should_print_debug()
        if not self._printed_debug_header:
            print(
                "[LwF][KD] enabled with "
                f"alpha={self.alpha} temperature={self.temperature}"
                f"kd_on_new={self.kd_on_new} "
                f"teacher={self.teacher_adapter_name} student={self.student_adapter_name} "
                f"debug_every={self._debug_print_every}"
            )
            self._printed_debug_header = True

        if should_print_debug:
            print(f"[LwF][KD] input keys: {list(inputs.keys())}")
        self._set_adapter_safe(model, self.student_adapter_name)

        try:
            base_loss, outputs = super().compute_loss(
                model,
                inputs,
                return_outputs=True,
                num_items_in_batch=num_items_in_batch,
            )
        except TypeError:
            base_loss, outputs = super().compute_loss(
                model,
                inputs,
                return_outputs=True,
            )

        logits_student = self._extract_logits(outputs)
        if logits_student is None:
            if return_outputs:
                return base_loss, outputs
            return base_loss

        peft_cfg = getattr(model, "peft_config", {})
        if self.teacher_adapter_name not in peft_cfg:
            if should_print_debug:
                print(
                    f"[LwF][KD] skipped step={self._debug_call_count}: "
                    f"teacher adapter '{self.teacher_adapter_name}' missing"
                )
            if return_outputs:
                return base_loss, outputs
            return base_loss

        teacher_inputs = {}
        allowed_keys = {"input_ids", "attention_mask", "position_ids"}
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor) and k in allowed_keys:
                teacher_inputs[k] = v.to(logits_student.device)

        current_training_mode = model.training

        with torch.no_grad():
            self._set_adapter_safe(model, self.teacher_adapter_name)
            model.eval()

            # Avoid extra memory from accelerate output wrapping during teacher pass.
            teacher_model = model
            if hasattr(self, "accelerator") and self.accelerator is not None:
                teacher_model = self.accelerator.unwrap_model(model)
            if should_print_debug:
                print(f"[LwF][KD] step={self._debug_call_count} running teacher forward pass")
            teacher_outputs = teacher_model(
                **teacher_inputs,
                output_hidden_states=False,
                use_cache=False,
            )
            logits_teacher = self._extract_logits(teacher_outputs)

        self._set_adapter_safe(model, self.student_adapter_name)
        if current_training_mode:
            model.train()

        if logits_teacher is None:
            if return_outputs:
                return base_loss, outputs
            return base_loss

        if logits_teacher.device != logits_student.device:
            logits_teacher = logits_teacher.to(logits_student.device)

        # Align KD positions with CausalLM next-token loss:
        # logits at t predict labels at t+1.
        if logits_student.shape[1] <= 1:
            if should_print_debug:
                print(
                    f"[LwF][KD] skipped step={self._debug_call_count}: sequence too short for shifted KD"
                )
            if return_outputs:
                return base_loss, outputs
            return base_loss

        logits_student_kd = logits_student[:, :-1, :]
        logits_teacher_kd = logits_teacher[:, :-1, :]

        labels = inputs.get("labels", None)
        is_old = inputs.get("is_old", None)
        device = logits_student.device

        if self.kd_on_new:
            if should_print_debug:
                print(
                    f"[LwF][KD] step={self._debug_call_count}: kd_on_new is True and 'is_old' missing, applying KD to all examples"
                )
            apply_kd_example_mask = torch.ones(
                logits_student.shape[0], dtype=torch.bool, device=device
            )
        else:
            if isinstance(is_old, torch.Tensor):
                apply_kd_example_mask = is_old.to(device).bool()
                if should_print_debug:
                    num_kd_examples = int(apply_kd_example_mask.sum().item())
                    total_examples = apply_kd_example_mask.shape[0]
                    print(
                        f"[LwF][KD] step={self._debug_call_count} KD examples: {num_kd_examples}/{total_examples}"
                    )
            else:
                if should_print_debug:
                    print(
                        f"[LwF][KD] skipped step={self._debug_call_count}: kd_on_new is False and 'is_old' missing, using base loss only"
                    )
                if return_outputs:
                    return base_loss, outputs
                return base_loss
        if not apply_kd_example_mask.any():
            if should_print_debug:
                print(
                    f"[LwF][KD] skipped step={self._debug_call_count}: no examples selected for KD"
                )
            if return_outputs:
                return base_loss, outputs
            return base_loss

        if labels is not None:
            labels = labels.to(device)
            token_mask = labels[:, 1:] != -100
        else:
            token_mask = torch.ones_like(
                logits_student_kd[..., 0], dtype=torch.bool, device=device
            )

        example_mask_exp = apply_kd_example_mask.view(-1, 1).expand_as(token_mask)
        final_mask = token_mask & example_mask_exp

        if not final_mask.any():
            if should_print_debug:
                print(
                    f"[LwF][KD] skipped step={self._debug_call_count}: no valid tokens after masking"
                )
            if return_outputs:
                return base_loss, outputs
            return base_loss

        T = float(self.temperature)
        log_p_student = F.log_softmax(logits_student_kd / T, dim=-1)
        p_teacher = F.softmax(logits_teacher_kd / T, dim=-1)

        kl_per_element = F.kl_div(log_p_student, p_teacher, reduction="none")
        kl_per_token = kl_per_element.sum(dim=-1)
        kd_loss = kl_per_token[final_mask].mean()
        kd_loss = (T * T) * kd_loss

        total_loss = base_loss + (self.alpha * kd_loss)

        if should_print_debug:
            kd_examples = int(apply_kd_example_mask.sum().item())
            kd_tokens = int(final_mask.sum().item())
            print(
                f"[LwF][KD] step={self._debug_call_count} kd_examples={kd_examples} kd_tokens={kd_tokens} "
                f"base_loss={float(base_loss.detach().item()):.4f} kd_loss={float(kd_loss.detach().item()):.4f} "
                f"total_loss={float(total_loss.detach().item()):.4f}"
            )

        if return_outputs:
            return total_loss, outputs
        return total_loss