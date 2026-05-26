from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

# Get the project root directory by searching for pyproject.toml
def _find_project_root() -> Path:
    """Find project root by looking for pyproject.toml marker file."""
    current = Path(__file__).resolve()
    for parent in [current] + list(current.parents):
        if (parent / "pyproject.toml").exists():
            return parent
    # Fallback to assuming standard structure
    return Path(__file__).parent.parent.parent

PROJECT_ROOT = _find_project_root()


@dataclass
class ApibenchDataConfig:
    train_set: str = str(PROJECT_ROOT / "data" / "processed" / "cleaned-apibench-hf-train.json")
    val_set: str = str(PROJECT_ROOT / "data" / "processed" / "cleaned-apibench-hf-val.json")
    test_set: str = str(PROJECT_ROOT / "data" / "processed" / "cleaned-apibench-hf-eval.json")

@dataclass
class MLLMDataConfig:
    train_set: str = str(PROJECT_ROOT / "data" / "processed" / "cleaned-mllm-train.json")
    val_set: str = str(PROJECT_ROOT / "data" / "processed" / "cleaned-mllm-val.json")
    test_set: str = str(PROJECT_ROOT / "data" / "processed" / "cleaned-mllm-eval.json")


@dataclass
class HuggingBench1DataConfig:
    train_set: str = str(PROJECT_ROOT / "data" / "processed" / "cleaned-hugging-bench-1-train.json")
    val_set: str = str(PROJECT_ROOT / "data" / "processed" / "cleaned-hugging-bench-1-val.json")
    test_set: str = str(PROJECT_ROOT / "data" / "processed" / "cleaned-hugging-bench-1-eval.json")

@dataclass
class HuggingBench2DataConfig:
    train_set: str = str(PROJECT_ROOT / "data" / "processed" / "cleaned-hugging-bench-2-train.json")
    val_set: str = str(PROJECT_ROOT / "data" / "processed" / "cleaned-hugging-bench-2-val.json")
    test_set: str = str(PROJECT_ROOT / "data" / "processed" / "cleaned-hugging-bench-2-eval.json")

@dataclass
class TrainConfig:
    # Model and experiment identifiers
    experience_name: str  # name of the experience, e.g., "apibench" "mllm"
    output_root: Path
    variant_name: str = "" # variant name for the experiment
    extra_info: str = ""  # any extra info to append to the output directory name
    

    repo_id: str = "huggyllama/llama-7b" # base model to use
    retriever: Optional[str] = None  # specify retriever if needed, e.g., "bm25", "sentence_transformer", "splade", "flagembedding"
    
    # System prompt configuration
    system_prompt: Optional[str] = ""  # custom system prompt, if None uses default gorilla_prompt
    system_prompt_format: Optional[str] = None  # specify system prompt format if needed, e.g., "gorilla_prompt", "gorilla_prompt_explanation", "gorilla_prompt_explanation_json"
    
    # Reproducibility
    seed: int = None  # random seed for reproducibility
    
    # Training hyperparameters
    epochs: int = 15
    batch_size: int = 4
    grad_accum: int = 2
    lr: float = 0.0005
    max_length: int = 1024
    max_grad_norm: float = 1.0
    packing: bool = False
    group_by_length: bool = True
    completion_only_loss: bool = True
    label_smoothing: float = 0.0

    # LoRA parameters
    lora_r: int = 32  # try 64
    lora_alpha: int = 32 # typically 2 * lora_r (128)
    lora_dropout: float = 0.1  # try 0.05
    target_modules: List[str] = field(
        default_factory=lambda: ['q_proj', 'k_proj', 'v_proj', 'o_proj'])

    # Checkpoint and evaluation options
    resume_from: Optional[str] = None
    lora_adapters: Optional[List[str]] = None  # list of LoRA adapters to use
    early_stopping_patience: int = 3  # check we are not overfitting
    early_stopping_threshold: float = 0.0
    no_validation: bool = False
    hyperparameters_search: bool = False  # whether to perform hyperparameter search
    eval_at_step0: bool = False  # Run evaluation immediately after loading checkpoint but before training (global_step==0)
    

    # Optimizer and scheduler
    weight_decay: float = 0.001
    warmup_steps: int = 10
    lr_scheduler_type: str = "linear"  # "warmup_stable_decay"
    optim: str = "adamw_torch"
    logging_steps: int = 1
    save_strategy: str = "epoch"
    save_total_limit: int = 3
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False


    activation_checkpointing: bool = True
    
    # Memory optimization options
    low_memory_mode: bool = False  # Enable memory optimizations: Flash Attention 2 (if available) and activation offloading
    use_quantization: bool = False

    # Replay configuration for continual learning
    experiences_sequence: Optional[List[str]] = None  # List of experience names to train on sequentially, e.g., ["apibench", "mllm"]
    joint_training: bool = False  # If True, combine all experiences into one dataset for joint training (upper bound baseline)
    replay_percentage: Optional[float] = None  # Percentage of previous experience samples to replay (e.g., 0.1 for 10%)
    replay_num_samples: Optional[int] = None  # Fixed number of samples to replay from each previous experience
    
    # Few-shot retrieval replay configuration
    fewshot_top_k: int = 3  # Number of retrieved neighbors for few-shot augmentation
    fewshot_replay_ratio: float = 0.1  # Replay buffer ratio (default 10%)
    fewshot_max_card_tokens: int = 200  # Max tokens for model card snippets
    fewshot_dropout_prob: float = 0.0  # Probability of using original prompt without examples (0.0 = always use examples, 1.0 = never use examples)
    
    replay_strategy: str = "random"  # "random", "domain_model_coreset", "domain_model_representative", or "domain_model_herding"
    replay_min_per_domain: int = 5  # Minimum examples to include per domain (floor)
    replay_max_per_domain: Optional[int] = None  # Optional cap per domain (None = no cap)
    replay_max_per_model: int = 3  # Max examples per model within each domain
    replay_embedding_source: str = "sentence_transformer"  # Embedding model for coreset: "sentence_transformer", "flagembedding", etc.
    replay_embedding_cache_dir: Optional[str] = None  # Directory to cache embeddings (None = cmr/cache/embeddings)
    replay_representative_boundary_fraction: float = 0.25  # Fraction of representative replay reserved for boundary samples

    loss_mode: str = "supervised"  # "supervised", "router", "router+graph"
    router_loss_weight: float = 1.0  # Weight for routing loss
    lm_loss_weight: float = 1.0  # Weight for LM supervised loss
    
    # Semantic batching
    semantic_batching: bool = False  # Enable domain-homogeneous batching
    domains_per_batch: int = 1  # Number of domains per batch (1 = pure, >1 = mixed)
    mix_replay_in_semantic_batches: bool = True  # Mix replay examples into semantic batches
    
    # Router architecture
    router_embedding_dim: Optional[int] = None  # Defaults to hidden_size of base model
    router_tau: float = 0.07  # Temperature for scaling logits
    router_pooling: str = "last_token"  # Pooling strategy: "last_token" or "mean"
    # Router learning rates (split by parameter group for stability)
    router_proj_lr: Optional[float] = None  # Learning rate for projection head (None = use args.learning_rate)
    router_embedding_lr: Optional[float] = None  # Learning rate for embedding table (None = use args.learning_rate)
    # Recommended: router_proj_lr=3e-4 to 5e-4, router_embedding_lr=1e-4 to 3e-4
    # This helps stabilize training by reducing spiky updates from the embedding table
    
    # Candidate sampling
    router_K_total: int = 64  # Total candidates per example (including positive)
    router_K_semantic: int = 48  # Target semantic negatives (same domain)
    router_K_far: int = 8  # Target far negatives (other domains)
    router_K_hard: int = 7  # Target hard negatives from cache (7 + 1 positive = 64 - 8)
    router_use_legacy_batching: bool = False  # If true, use pre-optimization candidate batching (slower, no caching)
    
    # Hard negative mining
    router_mine_every_steps: int = 200  # Mine hard negatives every N steps
    router_K_hard_pool: int = 20  # Store top K confusable models in cache
    router_semantic_pool_size: int = 512  # Semantic pool size for mining (per domain)
    router_max_pool_size: int = 1024  # Maximum pool size (cap for large domains)
    
    # Semantic Pool Expansion
    router_semantic_pool_mode: str = "parent_group"  # "domain_only", "parent_group", "taxonomy_graph"
    router_semantic_pool_max_domains: Optional[int] = None  # Max related domains to include (None = all)
    router_semantic_pool_depth: int = 1  # Graph traversal depth (for taxonomy_graph mode)
    
    # Soft targets (graph-smoothed supervision)
    router_use_soft_targets: bool = False  # Distribute small mass to graph neighbors
    router_soft_target_eps: float = 0.1  # Mass to distribute to neighbors (1-eps on positive)
    router_soft_target_k_neighbors: int = 5  # Number of neighbors to consider
    
    # Label-side graph regularizer
    router_use_label_graph_reg: bool = False  # Enable label-side graph alignment
    router_label_graph_lambda: float = 0.1  # Weight for graph regularizer
    router_label_graph_tau: float = 0.07  # Temperature for predicted similarities
    router_label_graph_tau_target: float = 0.1  # Temperature for target similarities
    router_label_graph_max_models: int = 256  # Max models for graph regularizer (subsample if exceeds)
    router_label_graph_alpha_domain: float = 0.3  # Similarity for same-domain pairs in taxonomy
    
    # Model registry persistence
    router_registry_path: Optional[str] = None  # Path to save/load registry (for continual learning)
    router_registry_init_mode: str = "extend"  # "fresh" or "extend" - how to initialize registry
    router_registry_base_path: Optional[str] = None  # Path to previous registry JSON (for extend mode)
    
    # Two-phase training schedule
    router_two_phase_enable: bool = False  # Enable two-phase schedule (Phase 1: stability warmup, Phase 2: main training)
    router_phase1_frac: float = 0.2  # Fraction of total steps for Phase 1 (stability warmup)
    router_phase1_loss_mode: str = "router"  # Loss mode for Phase 1 (typically "router" for router-only)
    router_phase1_replay_ratio: Optional[float] = None  # NOTE: Not implemented - replay ratio must be set in sampler, not here. Use router_replay_loss_multiplier for replay pressure.
    router_phase1_router_loss_weight: float = 1.0  # Router loss weight for Phase 1
    router_phase1_lm_loss_weight: float = 0.0  # LM loss weight for Phase 1 (0.0 ensures LM is frozen)
    router_phase1_proj_lr: Optional[float] = None  # Router projection LR for Phase 1 (None = use router_proj_lr)
    router_phase1_embedding_lr: Optional[float] = None  # Router embedding LR for Phase 1 (None = use router_embedding_lr)
    router_phase1_use_soft_targets: bool = False  # Soft targets for Phase 1
    router_phase1_soft_target_eps: float = 0.02  # Soft target epsilon for Phase 1
    router_replay_loss_multiplier: float = 1.0  # Multiplier for router loss on replay examples (applied in Phase 1 if >1.0)
    
    # Exp1-preservation training mode
    router_exp1_preservation_enable: bool = False  # Enable exp1-preservation mode (freeze old embeddings during Phase 1, keep projection trainable to prevent accuracy drop)
    router_exp1_preservation_M_old: Optional[int] = None  # Base registry size from exp1 (auto-detected from checkpoint if None)
    
    # Router embedding anchoring regularizer
    router_anchor_enable: bool = False  # Enable embedding anchoring to preserve exp1 routing when registry is extended
    router_anchor_lambda: float = 1e-3  # Weight for anchor loss (lambda in total_loss += lambda * anchor_loss)
    router_anchor_mode: str = "normalized"  # Anchor mode: "raw" (L2) or "normalized" (cosine, preferred for router scoring)
    router_anchor_apply_phase: str = "phase1"  # When to apply anchoring: "phase1", "phase2", or "both"
    router_anchor_scope: str = "all_old"  # Which rows to anchor: "all_old" (all rows < M_old) or "touched" (only rows in current step's candidates + gold)
    router_anchor_M_old: Optional[int] = None  # Base registry size M_old (auto-detected from checkpoint if None)
    
    # Router projection anchoring regularizer (for exp2+ to reduce projection drift)
    router_proj_anchor_enable: bool = False  # Enable projection anchoring to preserve exp1 projection when registry is extended
    router_proj_anchor_lambda: float = 1e-2  # Weight for projection anchor loss (lambda in total_loss += lambda * proj_anchor_loss)
    router_proj_anchor_apply_phase: str = "phase1"  # When to apply projection anchoring: "phase1", "phase2", or "both"

    # Adaptive Lambda Scaling – Embedding Anchor (router_anchor_lambda)
    router_anchor_adaptive_enable: bool = False             # Enable adaptive lambda scaling
    router_anchor_adaptive_beta: float = 0.99               # EMA decay for loss tracking
    router_anchor_adaptive_update_every: int = 20           # Steps between lambda updates
    router_anchor_adaptive_target_ratio_phase1: float = 0.3 # Target anchor/router ratio in phase1
    router_anchor_adaptive_target_ratio_phase2: float = 0.1 # Target anchor/router ratio in phase2
    router_anchor_lambda_min: float = 1e2                   # Minimum lambda clamp
    router_anchor_lambda_max: float = 5e4                   # Maximum lambda clamp
    router_anchor_adaptive_reg_ema_min: float = 1e-8        # Min EMA anchor loss to gate update

    # Adaptive Lambda Scaling – Projection Anchor (router_proj_anchor_lambda)
    router_proj_anchor_adaptive_enable: bool = False             # Enable adaptive lambda scaling
    router_proj_anchor_adaptive_beta: float = 0.99               # EMA decay for loss tracking
    router_proj_anchor_adaptive_update_every: int = 20           # Steps between lambda updates
    router_proj_anchor_adaptive_target_ratio_phase1: float = 0.5 # Target proj_anchor/router ratio in phase1
    router_proj_anchor_adaptive_target_ratio_phase2: float = 0.1 # Target proj_anchor/router ratio in phase2
    router_proj_anchor_lambda_min: float = 1e2                   # Minimum lambda clamp
    router_proj_anchor_lambda_max: float = 2e4                   # Maximum lambda clamp
    router_proj_anchor_adaptive_reg_ema_min: float = 1e-10       # Min EMA proj anchor loss to gate update

    # Router EWC regularizer
    router_ewc_enable: bool = False  # Enable EWC on router projection + embedding layers
    router_ewc_lambda: float = 10000.0  # Regularization strength (ewc_loss = ewc_lambda * Σ F*(θ-θ*)²)
    router_ewc_mode: str = "online"  # "online" (recommended, constant memory) or "separate" (one snapshot per task)
    router_ewc_decay_factor: float = 0.9  # Exponential decay for online consolidation (0 < decay ≤ 1)
    router_ewc_apply_phase: str = "phase1"  # When to apply EWC: "phase1", "phase2", or "both"
    router_ewc_fisher_samples: Optional[int] = None  # Samples used for Fisher computation (None = all)
    router_ewc_fisher_batch_size: int = 4  # Batch size for Fisher computation dataloader

    # Router freeze LM option (for router-only runs to prevent unintentional LM updates)
    router_freeze_lm: bool = False  # If True, force LM requires_grad=False in router-only mode (router still trainable)

    # Label Noise Configuration
    label_noise_prob: float = 0.0        # Fraction of examples to corrupt (0.0 = off, 0.2 = 20%)
    label_noise_target: str = "model"    # What to corrupt: "model", "domain", or "both"
    label_noise_mode: str = "random"     # How to sample noisy label:
                                         #   "random"      – uniform draw from entire pool
                                         #   "same_domain" – draw from same domain (harder noise for model)
    label_noise_replay: bool = False     # Whether to also apply noise to replay examples (default: False)

    # Card-guided cold-start initialisation for new embedding rows
    card_guided_init_enable: bool = False  # Replace Xavier init of new rows with card-similarity-weighted mixture of old embeddings (opt-in)
    card_init_tau: float = 0.07  # Softmax temperature for card-similarity weighting
    card_init_topk: int = 30  # Number of nearest old models to mix per new model
    card_init_scope: str = "global"  # Similarity scope: "global" (all old models) or "within_domain" (only old models in the same domain)
    card_init_min_sim_threshold: float = 0.2  # Min max-similarity to use weighted init; below → domain-mean fallback
    card_init_fallback_domain: bool = True  # Fall back to domain-mean (vs. global mean) when similarity is below threshold


@dataclass
class EvalConfig:
    # Model and experiment identifiers   
    experience_name: str = "apibench"  # name of the experience, e.g., "apibench" "mllm"
    lora_adapters: List[str] = field(default_factory=list)  # list of LoRA adapters to use
    repo_id: str = "huggyllama/llama-7b"  # base model to use
    eval_on_train: bool = False  # optionally evaluate on train set as well (default: off)
    
    # Input/Output settings
    input_max_length: int = 1024
    max_new_tokens: int = 64
    temperature: float = 0.4
    do_sample: bool = True
    # Decoding controls (optional; only applied if provided)
    top_p: float = 1.0
    top_k: Optional[int] = None
    penalty_alpha: Optional[float] = None
    # Not currently used in generation, but accepted for experimentation/logging
    random_prefix_len: Optional[int] = None
    sample_num: Optional[int] = None
    
    output_name: Optional[str] = None  # Name of the directory to save the evaluation results
    
    # Evaluation settings
    eval_batch_size: int = 4
    
    # LoRA merging strategy settings
    lora_merging_strategy: Optional[str] = None  # ties, dare_linear, arithmetic_mean, or null
    ties_or_dare_weights: List[float] = field(default_factory=lambda: [1.0, 1.0])  # use only when lora_merging_strategy is "ties" or "dare_linear"
    ties_or_dare_density: float = 0.3
    
    retriever: Optional[str] = None  # specify retriever if needed, e.g., "bm25", "sentence_transformer", "splade", "flagembedding"
    system_prompt_format: Optional[str] = None  # specify system prompt format if needed, e.g., "gorilla_prompt", "gorilla_prompt_explanation", "gorilla_prompt_explanation_json"
    use_router: bool = False  # use router evaluation instead of text generation
    debug_router_eval: bool = False  # enable detailed debugging output for router evaluation
    strict_router_load: bool = False  # use strict=True when loading router weights
    eval_on_train_samples: bool = False  # load 50 examples from training split and run router evaluation on them
    hierarchical_eval: bool = False  # enable hierarchical (two-stage) evaluation: predict group then model within group
    hierarchy_level: str = "domain"  # hierarchy level for hierarchical evaluation: "domain" or "parent_group"
    hierarchical_topk: int = 1  # number of top groups to consider in hierarchical evaluation
    hier_domain_score_mode: str = "logsumexp"  # domain scoring strategy: "logsumexp", "max", "topk_logsumexp", "hybrid"
    hier_domain_topk: int = 10  # number of top models for topk_logsumexp/hybrid domain scoring modes
    hier_domain_hybrid_alpha: float = 0.5  # weight for max in hybrid domain scoring mode (0.0 = pure logsumexp, 1.0 = pure max)
    
    # Few-shot retrieval configuration
    fewshot_top_k: int = 3  # Number of retrieved neighbors for few-shot augmentation (should match TrainConfig.fewshot_top_k)
    fewshot_max_card_tokens: int = 200  # Max tokens for model card snippets (should match TrainConfig.fewshot_max_card_tokens)
    fewshot_replay_seed: Optional[int] = 42  # Seed for replay buffer sampling (should match TrainConfig.seed used during training)
    fewshot_replay_ratio: float = 0.1  # Replay buffer ratio (should match TrainConfig.fewshot_replay_ratio, default 10%)
    fewshot_dropout_prob: float = 0.0  # Probability of using original prompt without examples during evaluation (0.0 = always use examples, typically 0.0 for eval)
    
