import os
from typing import Optional

import torch
from peft import (  # NOQA
    LoraConfig,
    PeftModel,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import AutoModelForCausalLM

from ...openmodel import LoRAModelManager
try:
    import flash_attn  # noqa: F401
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False

HF_AUTH_KW = {}

class DualLoRAModelManager(LoRAModelManager):
    """
    Single-backbone LoRA manager with two adapters:
      - student: trainable
      - teacher: frozen

    Typical use:
      - exp 0: create/load only student
      - exp > 0:
          * load previous checkpoint as student
          * load same previous checkpoint as teacher
          * train only student
          * switch adapter at runtime with set_adapter(...)
    """

    STUDENT_ADAPTER_NAME = "student"
    TEACHER_ADAPTER_NAME = "teacher"

    def __init__(
        self,
        config,
        device_map: str = "auto",
        pad_token: Optional[str] = None,
        student_lora_path: Optional[str] = None,
        teacher_lora_path: Optional[str] = None,
        init_model=False,
    ):
        self.student_lora_path = student_lora_path
        self.teacher_lora_path = teacher_lora_path

        super().__init__(
            config=config,
            device_map=device_map,
            pad_token=pad_token,
            lora_paths=None,
        )

        self.model = self._build_dual_model()

    def _build_dual_model(self):
        """
        Build a single base model and attach:
          - student adapter
          - teacher adapter (optional)
        """
        # Case 1: no previous checkpoint -> create fresh student adapter
        if self.student_lora_path is None:
            model = self._build_model(self.config)
            try:
                model.set_adapter(self.STUDENT_ADAPTER_NAME)
            except Exception:
                pass

            self._freeze_non_student_params()
            self._set_trainable_adapter(self.STUDENT_ADAPTER_NAME)
            model.train()
            print("[DualLoRA] Created fresh student adapter.")
            return model

        # Case 2: load previous checkpoint as student
        base = self._load_base_for_dual_adapter_training()
        model = PeftModel.from_pretrained(
            base,
            self.student_lora_path,
            adapter_name=self.STUDENT_ADAPTER_NAME,
            is_trainable=True,
            device_map=self.device_map,
        )
        print(f"[DualLoRA] Loaded student adapter from {self.student_lora_path}")

        # Optional teacher adapter
        if self.teacher_lora_path is not None:
            model.load_adapter(
                self.teacher_lora_path,
                adapter_name=self.TEACHER_ADAPTER_NAME,
                is_trainable=False,
            )
            print(f"[DualLoRA] Loaded teacher adapter from {self.teacher_lora_path}")

        model.set_adapter(self.STUDENT_ADAPTER_NAME)

        self.model = model
        self._freeze_non_student_params()
        self._set_trainable_adapter(self.STUDENT_ADAPTER_NAME)
        model.train()
        return model

    def _load_base_for_dual_adapter_training(self):
        """
        Load base model once, exactly like training mode in parent logic.
        """
        quantization_config = self._get_quantization_config(self.config)

        use_flash_attn = (
            getattr(self.config, "low_memory_mode", False) and "FLASH_ATTN_AVAILABLE" in globals() and FLASH_ATTN_AVAILABLE
        )
        print(
            f"[DualLoRA][base] repo_id={self.repo_id} "
            f"quantized={quantization_config is not None} flash_attn={use_flash_attn}"
        )

        base = AutoModelForCausalLM.from_pretrained(
            self.repo_id,
            cache_dir=self.cache_dir,
            device_map=self.device_map,
            attn_implementation="flash_attention_2" if use_flash_attn else None,
            quantization_config=quantization_config,
            **HF_AUTH_KW,
        )

        if quantization_config is not None:
            base = prepare_model_for_kbit_training(
                base,
                use_gradient_checkpointing=getattr(
                    self.config, "activation_checkpointing", True
                ),
            )

        try:
            base.config.output_hidden_states = False
            base.config.output_attentions = False
            base.config.use_cache = False
        except Exception:
            pass

        return base

    def has_teacher(self) -> bool:
        return (
            hasattr(self.model, "peft_config")
            and self.TEACHER_ADAPTER_NAME in self.model.peft_config
        )

    def has_student(self) -> bool:
        return (
            hasattr(self.model, "peft_config")
            and self.STUDENT_ADAPTER_NAME in self.model.peft_config
        )

    def set_student(self):
        if not self.has_student():
            raise RuntimeError("Student adapter not loaded.")
        self.model.set_adapter(self.STUDENT_ADAPTER_NAME)

    def set_teacher(self):
        if not self.has_teacher():
            raise RuntimeError("Teacher adapter not loaded.")
        self.model.set_adapter(self.TEACHER_ADAPTER_NAME)

    def _freeze_non_student_params(self):
        """
        Freeze everything first.
        Then _set_trainable_adapter will re-enable only student params.
        """
        for _, p in self.model.named_parameters():
            p.requires_grad = False

    def _set_trainable_adapter(self, adapter_name: str):
        """
        Make only the specified adapter trainable.
        Works by matching parameter names that include the adapter name.
        """
        found = 0
        for name, p in self.model.named_parameters():
            if adapter_name in name:
                p.requires_grad = True
                found += 1

        if found == 0:
            print(
                f"[DualLoRA][WARN] No parameters matched adapter name '{adapter_name}'. "
                "Check actual parameter names in named_parameters()."
            )

    def freeze_teacher(self):
        if not self.has_teacher():
            return

        for name, p in self.model.named_parameters():
            if self.TEACHER_ADAPTER_NAME in name:
                p.requires_grad = False

    def train_student_only(self):
        self._freeze_non_student_params()
        self._set_trainable_adapter(self.STUDENT_ADAPTER_NAME)
        self.freeze_teacher()
        self.set_student()
        self.model.train()
        print("[DualLoRA] Training mode set to student-only.")

    @torch.no_grad()
    def forward_teacher(self, **inputs):
        if not self.has_teacher():
            raise RuntimeError("Teacher adapter not loaded.")

        current_mode = self.model.training
        self.set_teacher()
        self.model.eval()
        outputs = self.model(**inputs)
        self.set_student()
        if current_mode:
            self.model.train()
        return outputs

    def forward_student(self, **inputs):
        self.set_student()
        return self.model(**inputs)

    def print_trainable_parameters_summary(self):
        total = 0
        trainable = 0
        for _, p in self.model.named_parameters():
            n = p.numel()
            total += n
            if p.requires_grad:
                trainable += n

        pct = 100.0 * trainable / total if total > 0 else 0.0
        print(
            f"[DualLoRA] trainable params: {trainable:,} / {total:,} ({pct:.4f}%)"
        )

    def save_student_adapter(self, output_dir: str):
        """
        Save only the student adapter.
        """
        os.makedirs(output_dir, exist_ok=True)
        self.set_student()
        self.model.save_pretrained(output_dir)
        print(f"[DualLoRA] Saved student adapter to {output_dir}")