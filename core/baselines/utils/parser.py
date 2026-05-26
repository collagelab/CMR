import argparse
from pathlib import Path
from typing import Any, Dict

import yaml

from .configs import EvalConfig, TrainConfig


def create_config_from_yaml(
    config_class: TrainConfig | EvalConfig, overrides: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Create TrainConfig instance from YAML file with optional overrides.

    Args:
        config_path: Path to YAML configuration file
        **overrides: Additional parameters to override YAML values

    Returns:
        TrainConfig instance
    """
    # get and delete config from overrides to avoid issues
    config_path = Path(overrides.pop("config", ""))
    print(f"Loading configuration from: {config_path}")
    if config_path == "" or not config_path.exists():
        raise ValueError("Configuration file path must be provided and exist.")

    with open(config_path, "r") as f:
        config_data = yaml.safe_load(f)

    # delete overrides that are None
    overrides = {k: v for k, v in overrides.items() if v is not None}

    unknown_params = set(overrides.keys()) - set(
        config_class.__dataclass_fields__.keys()
    )
    if unknown_params:
        raise ValueError(f"Unknown parameters in overrides: {unknown_params}")

    config_data.update(overrides)

    return config_data


# general class
class Parser:
    def __init__(self):
        self.parser = argparse.ArgumentParser(
            description="Train or Evaluate a LoRA fine-tuned model on a specific dataset",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="",
        )
        
        self.parser.add_argument(
            "--experience_names",  # train and eval
            type=str,
            required=False,
            nargs="+",
            help="Specify which experience(s) (dataset) to evaluate. Can pass multiple: --experiences_name apibench mllm",
        )
        
        self.parser.add_argument(
            "--variant_name",
            type=str,
            required=False,
            help="Specify the variant name for the experiment. It can't be empty.",
        )
        self.parser.add_argument(
            "--experiences_sequence",
            type=str,
            nargs="+",
            required=False,
            choices=["apibench", "mllm", "hugging-bench-1", "hugging-bench-2"],
            help="Sequence of experiences to train on sequentially (e.g., --experiences_sequence apibench mllm hugging-bench-1)",
        )

        self.parser.add_argument(
            "--model_indices",  # train and eval
            type=str,
            required=False,
            nargs="+",
            help="Specify which model indices to use for retrieval, Can pass multiple in order: --model_indices e1 e1_e2",
        )


        self.parser.add_argument(
            "--lora_adapters",
            type=str,
            nargs="+",
            required=False,
            help="Training: specify which lora adapter to use to start training\n Evaluation specify one or more lora adapters to use (e.g., --lora_adapters adapter1 adapter2 adapter3). ",
        )
        
        self.parser.add_argument(
            "--repo_id",
            type=str,
            required=False,
            help="Base model repository id to evaluate (e.g., --repo_id Qwen/Qwen2.5-7B)",
            choices=["huggyllama/llama-7b", "Qwen/Qwen2.5-7B","Qwen/Qwen3-4B"]
        )


# child classes
class TrainParser(Parser):
    def __init__(self) -> TrainConfig:
        super().__init__()
        
        self.parser.add_argument(
            "--retriever",
            type=str,
            required=False,
            help="Specify which retriever to use",
            choices=["bm25", "sentence_transformer", "splade", "flagembedding"],
        )
        
        self.parser.add_argument(
            "--mode",
            type=str,
            required=False,
            choices=["replay", "sequential-finetuning", "joint-training", "merging", "ewc", "lwf"],
            help="Specify which training mode to use: 'replay', 'sequential-finetuing', 'joint-training', 'merging', 'ewc', or 'lwf'",
        )

        self.parser.add_argument(
            "--config",
            type=str,
            default="configurations/train_config.yaml",
            help="Path to YAML configuration file",
        )

        self.parser.add_argument(
            "--epochs",
            type=int,
            required=False,
            help="Number of training epochs",
        )

        self.parser.add_argument(
            "--batch_size",
            type=int,
            required=False,
            help="Training batch size",
        )
        
        self.parser.add_argument(
            "--max_length",
            type=int,
            required=False,
            help="Maximum sequence length for inputs",
        )

        self.parser.add_argument(
            "--lr",
            type=float,
            required=False,
            help="Learning rate",
        )

        self.parser.add_argument(
            "--seed",
            type=int,
            required=False,
            help="Random seed for reproducibility",
        )

        self.parser.add_argument(
            "--replay_percentage",
            type=float,
            required=False,
            help="Percentage of previous experience samples to replay (e.g., 0.1 for 10%). If both replay_percentage and replay_num_samples are provided, replay_num_samples takes precedence.",
        )
        self.parser.add_argument(
            "--replay_num_samples",
            type=int,
            required=False,
            help="Fixed number of samples to replay from each previous experience. If both replay_percentage and replay_num_samples are provided, replay_num_samples takes precedence.",
        )
        
        self.parser.add_argument(
            "--output_path",
            type=str,
            required=False,
            help="[OPTIONAL] For training: specify the path directory for adapters \n For evaluation: specify the path directory for evaluation results",
        )
        
        self.parser.add_argument(
            "--grad_accum",
            type=int,
            required=False,
            help="Gradient accumulation steps default is 2 in train_config.yaml",
        )
        
        self.parser.add_argument(
            "--ewc_fisher_samples",
            type=int,
            required=False,
            help="Number of samples to use for computing Fisher Information Matrix in EWC",
        )
        
        self.parser.add_argument(
            "--ewc_lambda",
            type=float,
            required=False,
            help="Regularization strength for EWC",
        )

        self.parser.add_argument(
            "--alpha",
            type=float,
            required=False,
            default=1.0,
            help="Weight of KD loss in LwF (default: 1.0).",
        )
        self.parser.add_argument(
            "--temperature",
            type=float,
            required=False,
            default=2.0,
            help="Distillation temperature in LwF (default: 2.0).",
        )
        self.parser.add_argument(
            "--kd_on_new",
            action="store_true",
            default=None,
            help="Apply KD also on new examples.",
        )

    def parse_args(self) -> TrainConfig:
        args = self.parser.parse_args()
        params = create_config_from_yaml(config_class=TrainConfig, overrides=vars(args))
        
        if params.get("variant_name", "").strip() == "":
            raise ValueError("The 'variant_name' parameter cannot be empty.")
        
        train_config = TrainConfig(**params)
        
        if (train_config.retriever is None) != (train_config.model_indices is None):
            raise ValueError(
                "Both 'retriever' and 'model_indices' must be provided together or both set to None."
            )
        
        if train_config.model_indices is not None:
            if train_config.mode != "joint-training" and len(train_config.model_indices) != len(train_config.experience_names):
                raise ValueError(
                    f"The number of model indices ({len(train_config.model_indices)}) must match the number of experiences ({len(train_config.experience_names)})."
                )
        
        return train_config

class EvalParser(Parser):
    def __init__(self) -> EvalConfig:
        super().__init__()
        
        self.parser.add_argument(
            "--retrievers",
            type=str,
            nargs="+",
            required=False,
            help="Specify a list of retrievers to use for evaluation. Can pass multiple in order: --retrievers retriever1 retriever2",
        )

        self.parser.add_argument(
            "--config",
            type=str,
            default="configurations/eval_config.yaml",
            help="Path to YAML configuration file",
        )
        
        self.parser.add_argument(
            "--input_max_length",
            type=int,
            required=False,
            help="Maximum sequence length for evaluation inputs",
        )
        
        self.parser.add_argument(
            "--lora_merging_strategy",
            type=str,
            required=False,
            choices=["ties", "dare_linear", "arithmetic_mean"],
            # dest="lora_merging_strategy",  # Map to config field name
            help="Specify which merging strategy to use (e.g., --lora_merging_strategy ties)",
        )

        self.parser.add_argument(
            "--weights",
            type=float,
            nargs="+",
            required=False,
            # dest="ties_or_dare_weights",  # Map to config field name
            help="Adapter weights for merging strategy (e.g., --weights 1.0 1.0)",
        )

        self.parser.add_argument(
            "--density",
            type=float,
            required=False,
            # dest="ties_or_dare_density",  # Map to config field name
            help="Adapter density for merging strategy (e.g., --density 0.3)",
        )

        self.parser.add_argument(
            "--eval_on_train",
            action=argparse.BooleanOptionalAction,
            default=None,
            help="Also run evaluation on the training set (default: off)",
        )
        self.parser.add_argument(
            "--eval_batch_size",
            type=int,
            required=False,
            help="Evaluation batch size",
        )

        self.parser.add_argument(
            "--oracle",
            action=argparse.BooleanOptionalAction,
            default=None,
            help="Enable oracle retrieval (filter by ground-truth domain before retrieval).",
        )

    def parse_args(self) -> EvalConfig:
        args = self.parser.parse_args()
        params = create_config_from_yaml(EvalConfig, overrides=vars(args))
        
        # Create the config instance
        eval_config = EvalConfig(**params)
        
        if eval_config.variant_name.strip() == "":
            raise ValueError("The 'variant_name' parameter cannot be empty.")
        
        if (eval_config.retrievers is None) != (eval_config.model_indices is None):
            raise ValueError(
                "Both 'retrievers' and 'model_indices' must be provided together or both set to None."
            )
        
        if eval_config.model_indices is not None and len(eval_config.model_indices) != len(eval_config.experience_names):
            raise ValueError(
                f"The number of model indices ({len(eval_config.model_indices)}) must match the number of experiences ({len(eval_config.experience_names)})."
            )
        
        
        if eval_config.lora_merging_strategy is not None and len(eval_config.lora_adapters) < 2:
            raise ValueError(
                f"Merging strategy is specified ({eval_config.lora_merging_strategy}), but less than two LoRA adapters provided for merging."
            )

        return eval_config
