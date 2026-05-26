import logging
import os
import warnings
from contextlib import nullcontext
from typing import List, Optional

import torch
from peft import (  # NOQA
    LoraConfig,
    PeftModel,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.utils import logging as transformers_logging

from .utils.configs import EvalConfig, TrainConfig

# Check if flash-attn is available
try:
    import flash_attn  # noqa: F401

    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False

transformers_logging.set_verbosity_error()


# Suppress transformers warnings and info logs
logging.getLogger("transformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

# Variabili globali configurabili
HF_AUTH_KW = {}


class LoRAModelManager:
    def __init__(
        self,
        config: TrainConfig | EvalConfig,  # lora parameters
        device_map: str = "auto",
        pad_token: Optional[str] = None,
        lora_paths: Optional[List[str]] = None,
        init_model=True,
    ):
        self.repo_id = config.repo_id
        self.device_map = device_map
        self.model = None
        self.tokenizer = None
        self.pad_token = pad_token
        self.lora_paths = lora_paths
        self.config = config

        # Get cache directory with fallback and save as instance attribute
        self.cache_dir = os.environ.get("HF_HOME", None)

        self.tokenizer = AutoTokenizer.from_pretrained(
            config.repo_id, cache_dir=self.cache_dir, **HF_AUTH_KW
        )

        if pad_token:
            self.tokenizer.pad_token = pad_token
        elif self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if isinstance(config, EvalConfig):
            inference_mode = True
        else:
            inference_mode = False

        if self.lora_paths is not None and len(self.lora_paths) > 0:
            # automatically find the latest checkpoint in each provided lora path and update the paths to point to those checkpoints
            
            updated_lora_paths = []
            for path in self.lora_paths:
                checkpoint_dirs = [d for d in os.listdir(path) if d.startswith("checkpoint-")]
                if checkpoint_dirs:
                    updated_lora_paths.append(os.path.join(path, checkpoint_dirs[0]))
                else:
                    raise ValueError(f"No checkpoint directory found in {path}. Please ensure that the path contains a checkpoint folder.")
            
            self.lora_paths = updated_lora_paths
            
            self.model = self._load_with_lora(
                self.lora_paths, inference_mode=inference_mode
            )
        elif init_model:  # create
            self.model = self._build_model(self.config)
        else:
            self.model = None

    def _get_quantization_config(self, config):
        """Create quantization config based on settings"""
        # Check if quantization is enabled
        quantization = getattr(config, "use_quantization", False)

        if quantization:
            return BitsAndBytesConfig(
                load_in_4bit=True,
            )

        return None

    def _build_model(
        self,
        config: TrainConfig | EvalConfig,
    ):
        # Determine if Flash Attention 2 should be used (low_memory_mode enables it)
        use_flash_attn = (
            getattr(config, "low_memory_mode", False) and FLASH_ATTN_AVAILABLE
        )

        if getattr(config, "low_memory_mode", False) and not FLASH_ATTN_AVAILABLE:
            print(
                "Warning: low_memory_mode is enabled but flash-attn is not available. "
                'Install flash-attn for GPU acceleration: uv pip install -e ".[gpu]" '
                "or set low_memory_mode=False in configuration."
            )

        quantization_config = self._get_quantization_config(config)

        # Load base model with appropriate settings for quantization and flash attention
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
                    config, "activation_checkpointing", True
                ),
            )

        # Keep memory footprint low during training by disabling optional outputs/cache.
        try:
            base.config.output_hidden_states = False  # type: ignore
            base.config.output_attentions = False  # type: ignore
            base.config.use_cache = False  # type: ignore
        except Exception:
            pass

        # Config LoRA
        peft_cfg = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            # bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=config.target_modules,
            inference_mode=False,
        )

        return get_peft_model(base, peft_cfg)

    def _load_with_lora(self, lora_paths: List[str], inference_mode: bool = True):
        """Carica modello base + adapter LoRA esistente"""
        quantization_config = (
            None if inference_mode else self._get_quantization_config(self.config)
        )

        base = AutoModelForCausalLM.from_pretrained(
            self.repo_id,
            cache_dir=self.cache_dir,
            device_map=self.device_map,
            attn_implementation="flash_attention_2"
            if (inference_mode and FLASH_ATTN_AVAILABLE)
            else None,
            dtype=torch.bfloat16 if inference_mode else None,
            quantization_config=quantization_config,
            **HF_AUTH_KW,
        )

        # Prepare base model for training if not in inference mode
        if not inference_mode and quantization_config is not None:
            base = prepare_model_for_kbit_training(
                base,
                use_gradient_checkpointing=getattr(
                    self.config, "activation_checkpointing", True
                ),
            )

        if not inference_mode:
            try:
                base.config.output_hidden_states = False  # type: ignore
                base.config.output_attentions = False  # type: ignore
                base.config.use_cache = False  # type: ignore
            except Exception:
                pass

        if len(lora_paths) == 1:
            model = PeftModel.from_pretrained(
                base,
                lora_paths[0],
                device_map=self.device_map,
                is_trainable=not inference_mode,
            )
            print(f"Loaded LoRA adapter from {lora_paths[0]}")
            if inference_mode:
                model.eval()
            else:
                model.train()
            return model
        else:
            if self.config.lora_merging_strategy in ["ties", "dare_linear"]:
                assert len(lora_paths) == len(self.config.weights), (
                    "When using multiple adapters, please provide a weight for each adapter in config.weights"
                )

            adapters = [lora_paths[0].split("/experiments")[-1]]
            model = PeftModel.from_pretrained(
                base,
                lora_paths[0],
                device_map=self.device_map,
                adapter_name=lora_paths[0].split("/experiments")[-1],
                is_trainable=not inference_mode,
            )

            for path in lora_paths[1:]:
                adapter_name = path.split("/experiments")[-1]
                model.load_adapter(
                    path, adapter_name=adapter_name, is_trainable=not inference_mode
                )
                adapters.append(adapter_name)

            if (
                self.config.lora_merging_strategy == "ties"
                or self.config.lora_merging_strategy == "dare_linear"
            ):
                model.add_weighted_adapter(
                    adapters=adapters,
                    weights=self.config.weights,
                    adapter_name=self.config.lora_merging_strategy,  # new adapter name
                    combination_type=self.config.lora_merging_strategy,
                    density=self.config.density,
                )
                model.set_adapter(self.config.lora_merging_strategy)
                print(
                    f"Merged adapters {adapters} into new adapter '{self.config.lora_merging_strategy}'"
                )

            elif self.config.lora_merging_strategy == "arithmetic_mean":
                num_adapters = len(adapters)
                equal_weights = [1.0 / num_adapters] * num_adapters

                model.add_weighted_adapter(
                    adapters=adapters,
                    weights=equal_weights,
                    adapter_name=self.config.lora_merging_strategy,
                    combination_type="linear",
                )
                model.set_adapter(self.config.lora_merging_strategy)
                print(
                    f"Merged adapters {adapters} into new adapter '{self.config.lora_merging_strategy}'"
                )
            else:
                raise ValueError(
                    f"Unknown lora_merging_strategy: {self.config.lora_merging_strategy}. Supported strategies are 'ties', 'dare_linear', and 'arithmetic_mean'."
                )

            if inference_mode:
                model.eval()
            else:
                model.train()
            return model

    def _generate_batch(
        self,
        prompts: List[str],
        max_new_tokens: int = 64,
        skip_special_tokens: bool = True,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int | None = None,
        penalty_alpha: float | None = None,
    ):
        # Debug: check prompts type
        if not isinstance(prompts, list):
            raise TypeError(f"prompts must be a list, got {type(prompts)}")
        for i, p in enumerate(prompts):
            if not isinstance(p, str):
                raise TypeError(f"prompts[{i}] must be a str, got {type(p)}: {p}")
        
        # Force left-padding for inference — critical for decoder-only models
        original_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"
        
        tokenized_input = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            add_special_tokens=True,
        ).to(self.model.device)
        
        # Restore original padding side after tokenization
        self.tokenizer.padding_side = original_padding_side

        ctx = (
            torch.autocast(device_type=self.model.device.type, dtype=self.model.dtype)
            if self.model.dtype in [torch.float16, torch.bfloat16]
            else nullcontext()
        )

        with torch.no_grad(), ctx:
            gen_kwargs = dict(
                **tokenized_input,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,  # ADDED: Explicitly set pad token
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
            )
            if top_k is not None:
                gen_kwargs["top_k"] = top_k
            # If penalty_alpha is provided, use contrastive search settings
            # (Transformers uses penalty_alpha together with top_k)
            if penalty_alpha is not None:
                gen_kwargs["penalty_alpha"] = penalty_alpha
                # Ensure deterministic decoding for contrastive search
                gen_kwargs["do_sample"] = False
                # top_k is required for contrastive search; default to 10 if not set
                gen_kwargs["top_k"] = gen_kwargs.get("top_k", 10)

            # Non-standard args kept (as previously present)
            gen_kwargs["stop_strings"] = [self.tokenizer.eos_token]
            gen_kwargs["tokenizer"] = self.tokenizer

            gen_output = self.model.generate(**gen_kwargs)

        # Only decode the generated tokens, not the input
        prompt_length = tokenized_input["input_ids"].shape[1]
        generated_tokens = gen_output[:, prompt_length:]

        outputs = self.tokenizer.batch_decode(
            generated_tokens, skip_special_tokens=skip_special_tokens
        )

        return outputs

    def generate_batch_safe(self, prompts, batch_size=64, **gen_kwargs):
        all_outputs = []
        for i in tqdm(range(0, len(prompts), batch_size), desc="Processing"):
            batch = prompts[i : i + batch_size]
            outputs = self._generate_batch(batch, **gen_kwargs)
            all_outputs.extend(outputs)
        return all_outputs
