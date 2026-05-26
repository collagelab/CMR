from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
import os

# Get the project root directory
# Handle both source directory and installed package scenarios
_current_file = Path(__file__).resolve()

# Check if we're in site-packages (installed package)
if "site-packages" in str(_current_file) or "dist-packages" in str(_current_file):
    # When installed, use CMR_PROJECT_ROOT env var if set, otherwise use cwd
    PROJECT_ROOT = Path(os.environ.get("CMR_PROJECT_ROOT", Path.cwd()))
else:
    # When running from source, use relative path
    PROJECT_ROOT = _current_file.parent.parent.parent.parent


@dataclass
class ApibenchDataConfig:
    train_set: str = str(PROJECT_ROOT / "data" / "processed" / "cleaned-apibench-hf-train.json")
    val_set: str = str(PROJECT_ROOT / "data" / "processed" / "cleaned-apibench-hf-val.json")
    test_set: str = str(PROJECT_ROOT / "data" / "processed" / "cleaned-apibench-hf-eval.json")
    model_date_cutoff: Optional[str] = "Jun 2023"  # Date cutoff for model selection (e.g., "Jun 2023")

@dataclass
class MLLMDataConfig:
    train_set: str = str(PROJECT_ROOT / "data" / "processed" / "cleaned-mllm-train.json")
    val_set: str = str(PROJECT_ROOT / "data" / "processed" / "cleaned-mllm-val.json")
    test_set: str = str(PROJECT_ROOT / "data" / "processed" / "cleaned-mllm-eval.json")
    model_date_cutoff: Optional[str] = "Oct 2024"  # Date cutoff for model selection (e.g., "Dec 2023")
    
@dataclass
class HuggingBench1DataConfig:
    train_set: str = str(PROJECT_ROOT / "data" / "processed" / "cleaned-hugging-bench-1-train.json")
    val_set: str = str(PROJECT_ROOT / "data" / "processed" / "cleaned-hugging-bench-1-val.json")
    test_set: str = str(PROJECT_ROOT / "data" / "processed" / "cleaned-hugging-bench-1-eval.json")
    model_date_cutoff: Optional[str] = None  # Date cutoff for model selection

@dataclass
class HuggingBench2DataConfig:
    train_set: str = str(PROJECT_ROOT / "data" / "processed" / "cleaned-hugging-bench-2-train.json")
    val_set: str = str(PROJECT_ROOT / "data" / "processed" / "cleaned-hugging-bench-2-val.json")
    test_set: str = str(PROJECT_ROOT / "data" / "processed" / "cleaned-hugging-bench-2-eval.json")
    model_date_cutoff: Optional[str] = None  # Date cutoff for model selection
    


class ModelIndicesDataConfig:
    def __init__(self):
        self.e1: str = str(PROJECT_ROOT / "data" / "model_indices" / "e1.json")
        self.e1_e2: str = str(PROJECT_ROOT / "data" / "model_indices" / "e1_e2.json")
        self.e1_e2_e3: str = str(PROJECT_ROOT / "data" / "model_indices" / "e1_e2_e3.json")
        self.e1_e2_e3_e4: str = str(PROJECT_ROOT / "data" / "model_indices" / "e1_e2_e3_e4.json")
    
    def get_model_index_path(self, model_index: str) -> str:
        if model_index == "e1":
            return self.e1
        elif model_index == "e1_e2":
            return self.e1_e2
        elif model_index == "e1_e2_e3":
            return self.e1_e2_e3
        elif model_index == "e1_e2_e3_e4":
            return self.e1_e2_e3_e4
        else:
            raise ValueError(f"Unknown model index: {model_index}")


@dataclass
class TrainConfig:
    # Model and experiment identifiers
    experience_names: List[str]  # name of the experience(s), e.g., ["apibench", "mllm", "hugging-bench-1", "hugging-bench-2"]
    model_indices: Optional[List[str]]  # list of paths to the model indices files for retrieval
    output_path: Path  # root directory for output
    variant_name: str  # variant name for the experiment
    extra_info: str  # any extra info to append to the output directory name
    mode: str  # training mode, choices: "replay", "sequential-finetuing", "joint-training"

    repo_id: str  # base model to use
    retriever: Optional[str]  # specify retriever if needed, e.g., "bm25", "sentence_transformer", "splade", "flagembedding"

    # Training hyperparameters
    epochs: int
    batch_size: int
    grad_accum: int
    lr: float
    max_length: int
    max_grad_norm: float
    packing: bool
    group_by_length: bool
    completion_only_loss: bool
    label_smoothing: float

    # LoRA parameters
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    target_modules: List[str]

    # Checkpoint and evaluation options
    resume_from: Optional[str]
    lora_adapters: Optional[List[str]]  # list of LoRA adapters to use
    early_stopping_patience: int  
    early_stopping_threshold: float
    no_validation: bool
    hyperparameters_search: bool  # whether to perform hyperparameter search

    # Optimizer and scheduler
    weight_decay: float
    warmup_steps: int
    lr_scheduler_type: str
    optim: str
    logging_steps: int
    save_strategy: str
    save_total_limit: int
    metric_for_best_model: str
    greater_is_better: bool

    activation_checkpointing: bool

    # Memory optimization options
    low_memory_mode: bool
    use_quantization: bool

    # Replay configuration for continual learning
    replay_percentage: Optional[float]  # Percentage of previous experience samples to replay (e.g., 0.1 for 10%)
    replay_num_samples: Optional[int]  # Fixed number of samples to replay from each previous experience
    # Note: If both replay_percentage and replay_num_samples are None, no replay is performed
    # If both are provided, replay_num_samples takes precedence

    # LwF-specific parameters (used by online/offline LwF trainers)
    alpha: Optional[float] # LwF loss weight (higher = more emphasis on old tasks)
    temperature: Optional[float] # LwF temperature for softening probabilities (higher = softer probabilities)
    kd_on_new: Optional[bool] # whether to apply KD loss on new task samples (default: False)
    
    # EWC (Elastic Weight Consolidation) configuration for continual learning
    ewc_lambda: float  # EWC regularization strength (higher = more preservation of old tasks)
    ewc_fisher_samples: Optional[int] # Number of samples to use for Fisher computation (None = use all)
    ewc_fisher_batch_size: Optional[int]  # Batch size for Fisher computation (None = use training batch_size)
    ewc_mode: str
    ewc_decay_factor: float
    
    # Reproducibility
    seed: int = None  # random seed for reproducibility
 
    
@dataclass
class EvalConfig:
    # Model and experiment identifiers
    experience_names: (
        List[str]
    )  # name of the experience(s), e.g., ["apibench", "mllm"]
    model_indices: Optional[List[str]]  # list of paths to the model_indices files for retrieval
    lora_adapters: List[str]  # list of LoRA adapters to use
    repo_id: str  # base model to use
    eval_on_train: bool  # optionally evaluate on train set as well (default: off)
    
    variant_name: str  # variant name for the output directory
    
    retrievers: Optional[
        List[str]
    ]  # specify retriever if needed, e.g., "bm25", "sentence_transformer", "splade", "flagembedding"


    # Input/Output settings
    input_max_length: int
    max_new_tokens: int
    temperature: float
    do_sample: bool
    # Decoding controls (optional; only applied if provided)
    top_p: float
    top_k: Optional[int]
    # Not currently used in generation, but accepted for experimentation/logging

    # Evaluation settings
    eval_batch_size: int

    # LoRA merging strategy settings
    lora_merging_strategy: Optional[str]  # ties, dare_linear, arithmetic_mean, or null
    weights: List[
        float
    ]  # use only when lora_merging_strategy is "ties" or "dare_linear"
    density: float
    oracle: bool = False

   