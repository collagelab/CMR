import argparse
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Any
import torch
from .utils_carve.configs import ApibenchDataConfig, MLLMDataConfig, TrainConfig, HuggingBench1DataConfig, HuggingBench2DataConfig
from .utils_carve.config_loader import create_train_config_from_yaml
from .utils_carve.utility import set_seed
from .openmodel_carve import LoRAModelManager
from .utils_carve.prepareDataset import convert_to_conversational, convert_to_retrieval_replay_fewshot, load_dataset_json
from .utils_carve.retrieval_replay import PromptReplayBuffer
from .utils_carve.coreset_replay import (
    build_domain_model_coreset_replay,
    build_domain_model_representative_replay,
    build_domain_model_herding_replay,
)
from .utils_carve.wandb import WandbLogger
from datasets import concatenate_datasets, Dataset
from dotenv import load_dotenv


from .train_loop_carve import train
from ..baselines.utils.ewc_utils import RouterEWCRegularizer

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent   
load_dotenv(PROJECT_ROOT / ".env")


def get_dataset_config(experience_name: str):
    """Get the dataset configuration for a given experience name."""
    if experience_name == "apibench":
        return ApibenchDataConfig()
    elif experience_name == "mllm":
        return MLLMDataConfig()
    elif experience_name == "hugging-bench-1":
        return HuggingBench1DataConfig()
    elif experience_name == "hugging-bench-2":
        return HuggingBench2DataConfig()
    else:
        raise ValueError(f"Unknown experience name: {experience_name}")


def load_and_convert_dataset(
    experience_name: str,
    train_config: TrainConfig,
    model: LoRAModelManager,
    dataset_config,
    is_validation: bool = False,
    experience_idx: int = 0,
    replay_buffer: Optional[PromptReplayBuffer] = None,
    experience_index: Optional[object] = None
):
    """Load and convert a dataset for a given experience."""
    if is_validation:
        dataset_json = load_dataset_json(dataset_config.val_set)
    else:
        dataset_json = load_dataset_json(dataset_config.train_set)
    
    # Check if using retrieval_replay_fewshot baseline
    use_fewshot = "retrieval_replay_fewshot" in train_config.variant_name.lower()
    

    if use_fewshot and not is_validation:
        # Use few-shot augmentation for training (not validation)
        dataset = convert_to_retrieval_replay_fewshot(
            raw_data=dataset_json,
            config=train_config,
            tokenizer=model.tokenizer,
            dataset_config=dataset_config,
            experience_idx=experience_idx,
            experience_name=experience_name,
            replay_buffer=replay_buffer,
            experience_index=experience_index
        )
    else:
        dataset = convert_to_conversational(
            raw_data=dataset_json,
            config=train_config,
            tokenizer=model.tokenizer,
            dataset_config=dataset_config
        )
    
    return dataset, dataset_json  # Return both converted dataset and raw data


def sample_replay_data(
    previous_datasets: Dict[str, Dataset],
    previous_raw_data: Dict[str, List[Dict[str, Any]]],
    replay_percentage: Optional[float] = None,
    replay_num_samples: Optional[int] = None,
    seed: Optional[int] = None
) -> tuple[List[Dataset], List[Dict[str, Any]]]:
    """
    Sample replay data from previous experiences using random sampling.
    This is the baseline replay strategy (replay_strategy="random").
    
    Args:
        previous_datasets: Dictionary mapping experience names to their datasets
        previous_raw_data: Dictionary mapping experience names to their raw JSON data
        replay_percentage: Percentage of samples to replay (e.g., 0.1 for 10%)
        replay_num_samples: Fixed number of samples to replay
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of:
        - List of sampled datasets from previous experiences
        - List of raw examples that were selected (for neighbor consistency)
    """
    if not previous_datasets:
        return [], []
    
    replay_datasets = []
    replay_raw_examples = []
    
    for exp_name, dataset in previous_datasets.items():
        dataset_size = len(dataset)
        
        if dataset_size == 0:
            continue
        
        # Determine number of samples to replay
        if replay_num_samples is not None:
            num_samples = min(replay_num_samples, dataset_size)
        elif replay_percentage is not None:
            num_samples = max(1, int(dataset_size * replay_percentage))
        else:
            # No replay configured
            continue
        
        # Sample from the dataset
        if seed is not None:
            random.seed(seed)
        
        indices = random.sample(range(dataset_size), num_samples)
        sampled_dataset = dataset.select(indices)
        
        # Mark examples as replay for neighbor-contrastive loss
        # Update is_replay field to True for all sampled examples
        if "is_replay" in sampled_dataset.column_names:
            sampled_dataset = sampled_dataset.map(
                lambda x: {**x, "is_replay": True},
                desc=f"Marking replay from {exp_name}"
            )
        else:
            # Add is_replay column if not present
            sampled_dataset = sampled_dataset.add_column(
                "is_replay", [True] * len(sampled_dataset)
            )
        
        replay_datasets.append(sampled_dataset)
        
        # Also collect the raw examples for neighbor consistency
        if exp_name in previous_raw_data:
            raw_data = previous_raw_data[exp_name]
            for idx in indices:
                if idx < len(raw_data):
                    replay_raw_examples.append(raw_data[idx])
        
        print(f"  Replaying {num_samples} samples from {exp_name} (out of {dataset_size} total)")
    
    return replay_datasets, replay_raw_examples


def sample_coreset_replay_data(
    previous_raw_data: Dict[str, List[Dict[str, Any]]],
    train_config: TrainConfig,
    model,
    dataset_configs: Dict[str, Any]
) -> tuple[List[Dataset], List[Dict[str, Any]]]:
    """
    Sample replay data using domain-aware replay selection.

    Supported strategies:
    - domain_model_coreset: diversity-preserving FPS replay
    - domain_model_representative: center-first representative replay
    - domain_model_herding: model-centroid ordering + prompt herding in router prompt space
    
    Args:
        previous_raw_data: Dict mapping experience names to their raw JSON data
        train_config: Training config with coreset hyperparameters
        model: The model manager (for tokenizer access)
        dataset_configs: Dict mapping experience names to their dataset configs
    
    Returns:
        Tuple of:
        - List of converted datasets for replay
        - List of raw examples that were selected (for neighbor consistency)
    """
    if not previous_raw_data:
        return [], []
    
    replay_datasets = []
    replay_raw_examples = []
    
    for exp_name, raw_data in previous_raw_data.items():
        if not raw_data:
            continue
        
        # Compute replay ratio
        if train_config.replay_num_samples is not None:
            replay_ratio = train_config.replay_num_samples / len(raw_data)
        elif train_config.replay_percentage is not None:
            replay_ratio = train_config.replay_percentage
        else:
            continue
        
        replay_ratio = min(replay_ratio, 1.0)
        
        if train_config.replay_strategy == "domain_model_representative":
            print(f"\n  Building representative replay for {exp_name}:")
            coreset_examples = build_domain_model_representative_replay(
                apibench_examples=raw_data,
                replay_ratio=replay_ratio,
                min_per_domain=train_config.replay_min_per_domain,
                max_per_domain=train_config.replay_max_per_domain,
                max_per_model=train_config.replay_max_per_model,
                embedding_source=train_config.replay_embedding_source,
                cache_dir=train_config.replay_embedding_cache_dir,
                seed=train_config.seed,
                boundary_fraction=train_config.replay_representative_boundary_fraction,
            )
        elif train_config.replay_strategy == "domain_model_herding":
            print(f"\n  Building herding replay for {exp_name}:")
            coreset_examples = build_domain_model_herding_replay(
                apibench_examples=raw_data,
                replay_ratio=replay_ratio,
                min_per_domain=train_config.replay_min_per_domain,
                max_per_domain=train_config.replay_max_per_domain,
                max_per_model=train_config.replay_max_per_model,
                train_config=train_config,
                model_manager=model,
                router_registry_base_path=train_config.router_registry_base_path,
                cache_dir=train_config.replay_embedding_cache_dir,
                seed=train_config.seed,
            )
        else:
            # Default domain-aware strategy: coreset diversity replay.
            print(f"\n  Building coreset replay for {exp_name}:")
            coreset_examples = build_domain_model_coreset_replay(
                apibench_examples=raw_data,
                replay_ratio=replay_ratio,
                min_per_domain=train_config.replay_min_per_domain,
                max_per_domain=train_config.replay_max_per_domain,
                max_per_model=train_config.replay_max_per_model,
                embedding_source=train_config.replay_embedding_source,
                cache_dir=train_config.replay_embedding_cache_dir,
                seed=train_config.seed
            )
        
        if not coreset_examples:
            continue
        
        # Store raw examples for neighbor consistency (this is our replay buffer!)
        replay_raw_examples.extend(coreset_examples)
        
        # Convert to training format (mark as replay examples)
        dataset_config = dataset_configs.get(exp_name)
        converted_dataset = convert_to_conversational(
            raw_data=coreset_examples,
            config=train_config,
            tokenizer=model.tokenizer,
            dataset_config=dataset_config,
            is_replay=True,  # Mark replay examples for contrastive loss
        )
        
        replay_datasets.append(converted_dataset)
        print(f"  Coreset replay from {exp_name}: {len(coreset_examples)} examples")
    
    return replay_datasets, replay_raw_examples


def parse_args() -> tuple[TrainConfig, List[str]]:
    parser = argparse.ArgumentParser(
        description="Train a LoRA fine-tuned model on a specific dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configurations/train_config.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--retriever",
        type=str,
        required=False,
        choices=["bm25", "sentence_transformer", "splade", "flagembedding"],
        help="Specify which retriever to use",
    )
    parser.add_argument(
        "--experience_name",
        type=str,
        required=False,
        choices=["apibench", "mllm", "hugging-bench-1", "hugging-bench-2"],
        help="Specify which experience (dataset) to use",
    )
    parser.add_argument(
        "--variant_name",
        type=str,
        required=False,
        help="Specify the variant name for the experiment",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        required=False,
        help="Specify the root directory for output",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        required=False,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        required=False,
        help="Training batch size",
    )
    parser.add_argument(
        "--lr",
        type=float,
        required=False,
        help="Learning rate",
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=False,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--system_prompt",
        type=str,
        required=False,
        help="Custom system prompt to use instead of default gorilla_prompt",
    )
    
    parser.add_argument(
        "--system_prompt_format",
        type=str,
        required=False,
        choices=["gorilla_prompt", "gorilla_prompt_explanation", "gorilla_prompt_explanation_json"],
        help="Specify the system prompt format to use. gorilla_prompt: standard gorilla prompt with no explanation, predict only the model_name; gorilla_prompt_explanation: gorilla prompt with explanation using gorilla format <<<model_name>>>:my_model <<<explanation>>>:my_explanation; gorilla_prompt_explanation_json: gorilla prompt with explanation in json format {\"model_name\": \"my_model\", \"explanation\": \"my_explanation\"}",
    )
    parser.add_argument(
        "--experiences_sequence",
        type=str,
        nargs="+",
        required=False,
        choices=["apibench", "mllm", "hugging-bench-1", "hugging-bench-2"],
        help="Sequence of experiences to train on sequentially (e.g., --experiences_sequence apibench mllm hugging-bench-1)",
    )
    parser.add_argument(
        "--joint_training",
        action="store_true",
        help="Enable joint training mode: combine all 4 experiences (apibench, mllm, hugging-bench-1, hugging-bench-2) into one dataset for upper bound baseline",
    )
    parser.add_argument(
        "--skip_training_experiences",
        type=str,
        nargs="+",
        required=False,
        choices=["apibench", "mllm", "hugging-bench-1", "hugging-bench-2"],
        help="Experiences to skip training on but still load data for replay (e.g., --skip_training_experiences apibench). "
             "Use with --lora_adapters to load a checkpoint from the skipped experience.",
    )
    parser.add_argument(
        "--replay_percentage",
        type=float,
        required=False,
        help="Percentage of previous experience samples to replay (e.g., 0.1 for 10%%). If both replay_percentage and replay_num_samples are provided, replay_num_samples takes precedence.",
    )
    parser.add_argument(
        "--replay_num_samples",
        type=int,
        required=False,
        help="Fixed number of samples to replay from each previous experience. If both replay_percentage and replay_num_samples are provided, replay_num_samples takes precedence.",
    )
    parser.add_argument(
        "--replay_strategy",
        type=str,
        required=False,
        choices=["random", "domain_model_coreset", "domain_model_representative", "domain_model_herding"],
        help=(
            "Replay strategy: 'random' (baseline), "
            "'domain_model_coreset' (diversity-preserving), or "
            "'domain_model_representative' (center-first representative replay), or "
            "'domain_model_herding' (model-centroid ordering + prompt herding)"
        ),
    )
    parser.add_argument(
        "--replay_min_per_domain",
        type=int,
        required=False,
        help="Minimum examples per domain for coreset replay (default: 5)",
    )
    parser.add_argument(
        "--replay_max_per_domain",
        type=int,
        required=False,
        help="Maximum examples per domain for coreset replay (default: None = no cap)",
    )
    parser.add_argument(
        "--replay_max_per_model",
        type=int,
        required=False,
        help="Maximum examples per model within each domain for coreset replay (default: 3)",
    )
    # Router (semantic batching + candidate-set routing loss) arguments
    parser.add_argument(
        "--loss_mode",
        type=str,
        required=False,
        choices=["supervised", "router", "router+graph", "supervised+router", "supervised+router+graph"],
        help="Training loss mode: supervised (LM only), router, router+graph, or combined",
    )
    parser.add_argument(
        "--router_loss_weight",
        type=float,
        required=False,
        help="Weight for router loss (default: 1.0)",
    )
    parser.add_argument(
        "--lm_loss_weight",
        type=float,
        required=False,
        help="Weight for LM supervised loss (default: 1.0)",
    )
    parser.add_argument(
        "--router_embedding_dim",
        type=int,
        required=False,
        help="Dimension of router model embeddings (default: 256)",
    )
    parser.add_argument(
        "--router_tau",
        type=float,
        required=False,
        help="Temperature for router softmax (default: 0.07)",
    )
    parser.add_argument(
        "--router_pooling",
        type=str,
        required=False,
        choices=["last_token", "mean"],
        help="Pooling strategy for router: 'last_token' or 'mean' (default: last_token)",
    )
    parser.add_argument(
        "--router_proj_lr",
        type=float,
        required=False,
        help="Learning rate for router projection head (default: use --lr). Recommended: 3e-4 to 5e-4",
    )
    parser.add_argument(
        "--router_embedding_lr",
        type=float,
        required=False,
        help="Learning rate for router embedding table (default: use --lr). Recommended: 1e-4 to 3e-4",
    )
    parser.add_argument(
        "--router_K_total",
        type=int,
        required=False,
        help="Total candidate set size for routing (default: 64)",
    )
    parser.add_argument(
        "--router_K_semantic",
        type=int,
        required=False,
        help="Number of semantic (in-domain) negatives (default: 48)",
    )
    parser.add_argument(
        "--router_K_far",
        type=int,
        required=False,
        help="Number of far (out-of-domain) negatives (default: 8)",
    )
    parser.add_argument(
        "--router_K_hard",
        type=int,
        required=False,
        help="Number of hard negatives from mining (default: 7)",
    )
    parser.add_argument(
        "--router_use_legacy_batching",
        action="store_true",
        help="Use legacy (slower) candidate batching without caching",
    )
    parser.add_argument(
        "--router_mine_every_steps",
        type=int,
        required=False,
        help="Mine hard negatives every N steps (default: 200)",
    )
    parser.add_argument(
        "--router_use_soft_targets",
        action="store_true",
        help="Use soft targets (distribute mass to graph neighbors)",
    )
    parser.add_argument(
        "--router_eps",
        type=float,
        required=False,
        help="Epsilon for soft targets (default: 0.1)",
    )
    parser.add_argument(
        "--router_k_neighbors",
        type=int,
        required=False,
        help="Number of neighbors for soft targets (default: 3)",
    )
    parser.add_argument(
        "--router_use_label_graph_reg",
        action="store_true",
        help="Use label-side graph regularization",
    )
    parser.add_argument(
        "--router_lambda_graph",
        type=float,
        required=False,
        help="Weight for label-side graph regularizer (default: 0.1)",
    )
    parser.add_argument(
        "--router_max_graph_models",
        type=int,
        required=False,
        help="Max models for graph regularizer computation (default: 256)",
    )
    parser.add_argument(
        "--semantic_batching",
        action="store_true",
        help="Use domain-based semantic batching (for router training)",
    )
    parser.add_argument(
        "--domains_per_batch",
        type=int,
        required=False,
        help="Number of domains per batch for semantic batching (default: 1)",
    )
    parser.add_argument(
        "--router_registry_init_mode",
        type=str,
        required=False,
        choices=["fresh", "extend"],
        help="How to initialize registry: 'fresh' (build from scratch) or 'extend' (load and extend from base, default: extend)",
    )
    parser.add_argument(
        "--router_registry_base_path",
        type=str,
        required=False,
        help="Path to previous registry JSON for extend mode (e.g., core/experiments/apibench-D_tau008_seed40/checkpoint-310/model_registry.json). If not provided, will try to derive from resume_from_checkpoint.",
    )
    
    parser.add_argument(
        "--lora_adapters",
        type=str,
        nargs='+',
        required=False,
        help="Specify which adapters to use (e.g., --lora_adapters adapter1 adapter2 adapter3)",
    )
    # Router embedding anchoring regularizer arguments
    parser.add_argument(
        "--router_anchor_enable",
        action="store_true",
        help="Enable router embedding anchoring to preserve exp1 routing when registry is extended",
    )
    parser.add_argument(
        "--router_anchor_lambda",
        type=float,
        required=False,
        help="Weight for anchor loss (default: 1e-3)",
    )
    parser.add_argument(
        "--router_anchor_mode",
        type=str,
        required=False,
        choices=["raw", "normalized"],
        help="Anchor mode: 'raw' (L2) or 'normalized' (cosine, preferred, default: normalized)",
    )
    parser.add_argument(
        "--router_anchor_apply_phase",
        type=str,
        required=False,
        choices=["phase1", "phase2", "both"],
        help="When to apply anchoring: 'phase1', 'phase2', or 'both' (default: phase1)",
    )
    parser.add_argument(
        "--router_anchor_scope",
        type=str,
        required=False,
        choices=["all_old", "touched"],
        help="Which rows to anchor: 'all_old' (all rows < M_old) or 'touched' (only rows in current step, default: all_old)",
    )
    parser.add_argument(
        "--router_anchor_M_old",
        type=int,
        required=False,
        help="Base registry size M_old (auto-detected from checkpoint if not provided)",
    )

    args = parser.parse_args()
    
    # Create train_config from YAML with command line overrides
    config_overrides = {}
    
    # Only add command line args if they are provided (not None)
    if args.experience_name is not None:
        config_overrides['experience_name'] = args.experience_name
    if args.variant_name is not None:
        config_overrides['variant_name'] = args.variant_name
    if args.output_root is not None:
        config_overrides['output_root'] = Path(args.output_root)
    if args.epochs is not None:
        config_overrides['epochs'] = args.epochs
    if args.batch_size is not None:
        config_overrides['batch_size'] = args.batch_size
    if args.lr is not None:
        config_overrides['lr'] = args.lr
    if args.retriever is not None:
        config_overrides['retriever'] = args.retriever
    if args.seed is not None:
        config_overrides['seed'] = args.seed
    if args.system_prompt is not None:
        config_overrides['system_prompt'] = args.system_prompt
    if args.system_prompt_format is not None:
        config_overrides['system_prompt_format'] = args.system_prompt_format
    if args.experiences_sequence is not None:
        config_overrides['experiences_sequence'] = args.experiences_sequence
    if args.joint_training:
        config_overrides['joint_training'] = True
    if args.replay_percentage is not None:
        config_overrides['replay_percentage'] = args.replay_percentage
    if args.replay_num_samples is not None:
        config_overrides['replay_num_samples'] = args.replay_num_samples
    if args.replay_strategy is not None:
        config_overrides['replay_strategy'] = args.replay_strategy
    if args.replay_min_per_domain is not None:
        config_overrides['replay_min_per_domain'] = args.replay_min_per_domain
    if args.replay_max_per_domain is not None:
        config_overrides['replay_max_per_domain'] = args.replay_max_per_domain
    if args.replay_max_per_model is not None:
        config_overrides['replay_max_per_model'] = args.replay_max_per_model
    if args.loss_mode is not None:
        config_overrides['loss_mode'] = args.loss_mode
    if args.router_loss_weight is not None:
        config_overrides['router_loss_weight'] = args.router_loss_weight
    if args.lm_loss_weight is not None:
        config_overrides['lm_loss_weight'] = args.lm_loss_weight
    if args.router_embedding_dim is not None:
        config_overrides['router_embedding_dim'] = args.router_embedding_dim
    if args.router_tau is not None:
        config_overrides['router_tau'] = args.router_tau
    if args.router_pooling is not None:
        config_overrides['router_pooling'] = args.router_pooling
    if args.router_proj_lr is not None:
        config_overrides['router_proj_lr'] = args.router_proj_lr
    if args.router_embedding_lr is not None:
        config_overrides['router_embedding_lr'] = args.router_embedding_lr
    if args.router_K_total is not None:
        config_overrides['router_K_total'] = args.router_K_total
    if args.router_K_semantic is not None:
        config_overrides['router_K_semantic'] = args.router_K_semantic
    if args.router_K_far is not None:
        config_overrides['router_K_far'] = args.router_K_far
    if args.router_K_hard is not None:
        config_overrides['router_K_hard'] = args.router_K_hard
    if args.router_use_legacy_batching:
        config_overrides['router_use_legacy_batching'] = True
    if args.router_mine_every_steps is not None:
        config_overrides['router_mine_every_steps'] = args.router_mine_every_steps
    if args.router_use_soft_targets:
        config_overrides['router_use_soft_targets'] = True
    if args.router_eps is not None:
        config_overrides['router_eps'] = args.router_eps
    if args.router_k_neighbors is not None:
        config_overrides['router_k_neighbors'] = args.router_k_neighbors
    if args.router_use_label_graph_reg:
        config_overrides['router_use_label_graph_reg'] = True
    if args.router_lambda_graph is not None:
        config_overrides['router_lambda_graph'] = args.router_lambda_graph
    if args.router_max_graph_models is not None:
        config_overrides['router_max_graph_models'] = args.router_max_graph_models
    if args.semantic_batching:
        config_overrides['semantic_batching'] = True
    if args.domains_per_batch is not None:
        config_overrides['domains_per_batch'] = args.domains_per_batch
    if args.router_registry_init_mode is not None:
        config_overrides['router_registry_init_mode'] = args.router_registry_init_mode
    if args.router_registry_base_path is not None:
        config_overrides['router_registry_base_path'] = args.router_registry_base_path
    if args.lora_adapters is not None:
        config_overrides['lora_adapters'] = args.lora_adapters
    # Router anchor regularizer overrides
    if args.router_anchor_enable:
        config_overrides['router_anchor_enable'] = True
    if args.router_anchor_lambda is not None:
        config_overrides['router_anchor_lambda'] = args.router_anchor_lambda
    if args.router_anchor_mode is not None:
        config_overrides['router_anchor_mode'] = args.router_anchor_mode
    if args.router_anchor_apply_phase is not None:
        config_overrides['router_anchor_apply_phase'] = args.router_anchor_apply_phase
    if args.router_anchor_scope is not None:
        config_overrides['router_anchor_scope'] = args.router_anchor_scope
    if args.router_anchor_M_old is not None:
        config_overrides['router_anchor_M_old'] = args.router_anchor_M_old

    train_config = create_train_config_from_yaml(args.config, **config_overrides)
    
    # Return both config and skip_training_experiences (not part of config dataclass)
    skip_training_experiences = args.skip_training_experiences or []
    return train_config, skip_training_experiences


def print_startup_validation(train_config: TrainConfig, experiences: List[str]):
    """
    Print startup validation and sanity checks for ablation runs.
    
    This helps verify configuration correctness, especially for mllm-only ablations.
    """
    print("\n" + "=" * 80)
    print("TRAINING CONFIGURATION SUMMARY")
    print("=" * 80)
    print(f"Experience(s):              {experiences}")
    print(f"Replay strategy:            {train_config.replay_strategy}")
    print(f"Replay percentage:          {train_config.replay_percentage}")
    print(f"Replay num samples:         {train_config.replay_num_samples}")
    print(f"Two-phase enabled:          {train_config.router_two_phase_enable}")
    print(f"Embedding anchor enabled:   {train_config.router_anchor_enable}")
    print(f"Projection anchor enabled:  {train_config.router_proj_anchor_enable}")
    print(f"Loss mode:                  {train_config.loss_mode}")
    print(f"Seed:                       {train_config.seed}")
    print(f"Variant name:               {train_config.variant_name}")
    print(f"Extra info:                 {train_config.extra_info}")
    
    # Special warnings for single-experience runs
    if len(experiences) == 1:
        print(f"\n[SINGLE EXPERIENCE MODE]")
        print(f"  Running on single experience: {experiences[0]}")
        print(f"  Replay will be EMPTY (no prior experiences)")
        if train_config.replay_percentage or train_config.replay_num_samples:
            print(f"  ⚠ Replay is configured but will be skipped (no prior data)")
    
    print("=" * 80 + "\n")


def main():

    train_config, skip_training_experiences = parse_args()
    
    if train_config.seed is not None:
        # Set seed for reproducibility
        set_seed(train_config.seed)
    
    # Initialize WandB logger
    wandb_key = os.getenv("WANDB_API_KEY")
    if wandb_key:
        wandb_logger = WandbLogger(wandb_key, train_config, mode="train")
    else:
        wandb_logger = None
        print("Warning: WANDB_API_KEY not found in environment variables. Skipping WandB logging.")
    
    lora_paths = [f"./core/experiments/{adapter}" for adapter in train_config.lora_adapters]
    model = LoRAModelManager(config=train_config, lora_paths=lora_paths)

    # Determine which experiences to train on
    if train_config.joint_training:
        # Joint training mode: combine experiences into one dataset
        # If experiences_sequence is set, use that (for cumulative training)
        # Otherwise, use all 4 experiences (standard joint training)
        if train_config.experiences_sequence is not None:
            experiences = train_config.experiences_sequence
            print(f"CUMULATIVE JOINT TRAINING MODE: Combining {len(experiences)} experiences into one dataset")
        else:
            experiences = ["apibench", "mllm", "hugging-bench-1", "hugging-bench-2"]
            print(f"JOINT TRAINING MODE: Combining all 4 experiences into one dataset")
        print(f"Experiences: {experiences}")
        # Disable replay for joint training (all data is available)
        train_config.replay_percentage = None
        train_config.replay_num_samples = None
    elif train_config.experiences_sequence is not None:
        experiences = train_config.experiences_sequence
        print(f"Training on sequence of experiences: {experiences}")
    else:
        # Single experience mode (backward compatibility)
        experiences = [train_config.experience_name]
        print(f"Training on single experience: {train_config.experience_name}")
    
    # Print startup validation (especially useful for ablation runs)
    print_startup_validation(train_config, experiences)
    
    # Store datasets from previous experiences for replay
    previous_experience_datasets: Dict[str, Dataset] = {}
    
    # Store raw data and configs for coreset replay (needed for domain/model-aware sampling)
    previous_experience_raw_data: Dict[str, List[Dict[str, Any]]] = {}
    previous_experience_configs: Dict[str, Any] = {}
    
    # Store the ACTUAL replay examples selected (for neighbor consistency)
    # This is the replay buffer we have access to, not the full previous experience
    replay_buffer_raw_data: Dict[str, List[Dict[str, Any]]] = {}
    
    # Check if using retrieval_replay_fewshot baseline
    use_fewshot = "retrieval_replay_fewshot" in train_config.variant_name.lower()
    
    # Check replay strategy
    use_domain_aware_replay = train_config.replay_strategy in {
        "domain_model_coreset",
        "domain_model_representative",
        "domain_model_herding",
    }
    if use_domain_aware_replay:
        print(f"Using domain-aware replay strategy: {train_config.replay_strategy}")
        print(f"  min_per_domain={train_config.replay_min_per_domain}, "
              f"max_per_domain={train_config.replay_max_per_domain}, "
              f"max_per_model={train_config.replay_max_per_model}")
        if train_config.replay_strategy == "domain_model_representative":
            print(
                f"  representative_boundary_fraction="
                f"{train_config.replay_representative_boundary_fraction}"
            )
    else:
        print(f"Using random replay strategy (baseline)")
    
    # Initialize replay buffer for few-shot baseline
    replay_buffer = None
    if use_fewshot:
        replay_buffer = PromptReplayBuffer(
            replay_ratio=train_config.fewshot_replay_ratio,
            seed=train_config.seed
        )
        print("Using retrieval_replay_fewshot baseline with few-shot augmentation")
    
    # Get list of experiences to skip training on (but still load data for replay)
    skip_training_set = set(skip_training_experiences)
    if skip_training_set:
        print(f"Will skip training on: {skip_training_set} (data will be loaded for replay)")
    
    # Handle joint training mode
    if train_config.joint_training:
        print(f"\n{'='*80}")
        print(f"JOINT TRAINING: Loading and combining all {len(experiences)} experiences")
        print(f"{'='*80}\n")
        
        # Load and convert training datasets from all experiences
        all_train_datasets = []
        all_val_datasets = []
        
        # Save the original experience_name before the loop (for cumulative training)
        # This preserves the experience_name set in config (e.g., "mllm" for step 2)
        original_experience_name = train_config.experience_name
        
        for exp_idx, experience_name in enumerate(experiences):
            print(f"Loading {experience_name} (experience {exp_idx + 1}/{len(experiences)})...")
            
            # Get dataset configuration for current experience
            dataset_config = get_dataset_config(experience_name)
            
            # Temporarily set experience_name for dataset loading
            train_config.experience_name = experience_name
            
            # Load and convert training dataset
            dataset_train, _ = load_and_convert_dataset(
                experience_name=experience_name,
                train_config=train_config,
                model=model,
                dataset_config=dataset_config,
                is_validation=False,
                experience_idx=exp_idx,
                replay_buffer=None,
                experience_index=None
            )
            all_train_datasets.append(dataset_train)
            print(f"  Loaded {len(dataset_train)} training examples from {experience_name}")
            
            # Load and convert validation dataset
            dataset_val, _ = load_and_convert_dataset(
                experience_name=experience_name,
                train_config=train_config,
                model=model,
                dataset_config=dataset_config,
                is_validation=True,
                experience_idx=exp_idx,
                replay_buffer=None,
                experience_index=None
            )
            all_val_datasets.append(dataset_val)
            print(f"  Loaded {len(dataset_val)} validation examples from {experience_name}")
            
            # Restore original experience_name
            train_config.experience_name = original_experience_name
        
        # Combine all training datasets
        print(f"\nCombining training datasets...")
        combined_train_dataset = concatenate_datasets(all_train_datasets)
        print(f"  Combined training dataset size: {len(combined_train_dataset)} examples")
        print(f"  Breakdown: {', '.join([f'{exp}: {len(ds)}' for exp, ds in zip(experiences, all_train_datasets)])}")
        
        # Combine all validation datasets
        print(f"\nCombining validation datasets...")
        combined_val_dataset = concatenate_datasets(all_val_datasets)
        print(f"  Combined validation dataset size: {len(combined_val_dataset)} examples")
        print(f"  Breakdown: {', '.join([f'{exp}: {len(ds)}' for exp, ds in zip(experiences, all_val_datasets)])}")
        
        # Set experience_name to a joint identifier for output directory
        # Unless it was already set to one of the experiences (for cumulative training)
        # In cumulative training, we want to save to the last experience name, not "joint"
        if train_config.experience_name not in experiences:
            train_config.experience_name = "joint"
        
        print(f"\n{'='*80}")
        print(f"STARTING JOINT TRAINING on combined dataset")
        print(f"{'='*80}\n")
        
        # Train on combined dataset (no replay, no neighbor loss needed)
        train(
            trainConfig=train_config,
            model=model,
            dataset_train=combined_train_dataset,
            dataset_val=combined_val_dataset,
            wandb_logger=wandb_logger,
            replay_source_examples=None)  # No replay needed
        
        print(f"Completed joint training on all {len(experiences)} experiences\n")
        
        # Finish WandB logging
        if wandb_logger:
            wandb_logger.finish()
        
        return
    
    # =========================================================================
    # Router EWC regularizer lifecycle management
    # =========================================================================
    router_ewc_regularizer = None
    router_ewc_save_dirs: List[Path] = []
    if getattr(train_config, 'router_ewc_enable', False):
        legacy_router_ewc_dir = Path(train_config.output_root) / "router_ewc_state"
        router_ewc_regularizer = RouterEWCRegularizer(
            ewc_lambda=getattr(train_config, 'router_ewc_lambda', 10000.0),
            mode=getattr(train_config, 'router_ewc_mode', 'online'),
            decay_factor=getattr(train_config, 'router_ewc_decay_factor', 0.9),
        )
        loaded_state = False
        has_previous_adapter = bool(getattr(train_config, "lora_adapters", []) or [])
        # Prefer checkpoint-scoped state from the previous adapter in continual runs.
        for adapter in (getattr(train_config, "lora_adapters", []) or []):
            adapter_ckpt_dir = Path("./core/experiments") / adapter
            candidate = adapter_ckpt_dir / "router_ewc_state"
            if candidate.exists():
                try:
                    router_ewc_regularizer.load(candidate)
                    print(f"[RouterEWC] Loaded EWC state from previous adapter checkpoint: {candidate}")
                    loaded_state = True
                    break
                except Exception as e:
                    print(f"[RouterEWC] Could not load checkpoint EWC state from {candidate} ({e})")
        # Backward-compatible fallback to legacy global path. Guard it behind
        # presence of a previous adapter so exp1 runs do not accidentally load
        # stale cross-ablation state.
        if not loaded_state and has_previous_adapter and legacy_router_ewc_dir.exists():
            try:
                router_ewc_regularizer.load(legacy_router_ewc_dir)
                print(f"[RouterEWC] Loaded existing EWC state from legacy path: {legacy_router_ewc_dir}")
                loaded_state = True
            except Exception as e:
                print(f"[RouterEWC] Could not load legacy EWC state ({e}); starting fresh.")
        if not loaded_state:
            print("[RouterEWC] No existing EWC state found; will compute after first experience.")
        # Keep legacy save for backward compatibility.
        router_ewc_save_dirs = [legacy_router_ewc_dir]

    # Train on each experience sequentially
    for exp_idx, experience_name in enumerate(experiences):
        skip_training = experience_name in skip_training_set
        
        if skip_training:
            print(f"\n{'='*80}")
            print(f"Experience {exp_idx + 1}/{len(experiences)}: {experience_name} [SKIPPING TRAINING - loading data only]")
            print(f"{'='*80}\n")
        else:
            print(f"\n{'='*80}")
            print(f"Training on Experience {exp_idx + 1}/{len(experiences)}: {experience_name}")
            print(f"{'='*80}\n")
        
        # Update train_config with current experience name
        train_config.experience_name = experience_name
        
        # Get dataset configuration for current experience
        dataset_config = get_dataset_config(experience_name)
        
        # For few-shot baseline: sample replay at beginning of experience (E2+)
        if use_fewshot and exp_idx > 0 and replay_buffer:
            print(f"\nSampling replay data for few-shot baseline:")
            # Get raw data from previous experience
            prev_experience_name = experiences[exp_idx - 1]
            if prev_experience_name in previous_experience_raw_data:
                replay_buffer.add_experience(
                    previous_experience_raw_data[prev_experience_name],
                    prev_experience_name
                )
        
        # Load and convert training dataset
        dataset_train, raw_train_data = load_and_convert_dataset(
            experience_name=experience_name,
            train_config=train_config,
            model=model,
            dataset_config=dataset_config,
            is_validation=False,
            experience_idx=exp_idx,
            replay_buffer=replay_buffer if use_fewshot else None,
            experience_index=None  # Will be built inside conversion function
        )
        
        # Load and convert validation dataset
        dataset_val, raw_val_data = load_and_convert_dataset(
            experience_name=experience_name,
            train_config=train_config,
            model=model,
            dataset_config=dataset_config,
            is_validation=True,
            experience_idx=exp_idx,
            replay_buffer=None,  # No replay for validation
            experience_index=None
        )
        
        # If no_validation is True, combine train and val sets into a single training set
        if train_config.no_validation:
            print("no_validation is True: Combining train and val sets into a single training set")
            dataset_train = concatenate_datasets([dataset_train, dataset_val])
            dataset_val = None  # Set to None so it's not used for evaluation
            # Also combine raw data for replay
            raw_train_data = raw_train_data + raw_val_data
        
        # Store the original training dataset (before replay) for future replay
        # This needs to be done before we add replay data
        original_dataset_for_replay = dataset_train
        
        # Store raw data for replay (both few-shot and coreset strategies need this)
        previous_experience_raw_data[experience_name] = raw_train_data
        previous_experience_configs[experience_name] = dataset_config
        
        # Sample replay data from previous experiences if configured
        # Also store the raw replay examples for neighbor consistency
        current_replay_raw_examples = []
        
        if exp_idx > 0 and (train_config.replay_percentage is not None or train_config.replay_num_samples is not None):
            print(f"\n[REPLAY] Sampling replay data from previous experiences for {experience_name} (experience {exp_idx+1}/{len(experiences)})")
            print(f"[REPLAY] Strategy: {train_config.replay_strategy}")
            print(f"[REPLAY] Percentage: {train_config.replay_percentage}")
            print(f"[REPLAY] Num samples: {train_config.replay_num_samples}")
            
            if use_domain_aware_replay:
                # Use domain + model-aware coreset sampling
                # Only pass previous experiences (not current one)
                prev_raw_data = {k: v for k, v in previous_experience_raw_data.items() if k != experience_name}
                prev_configs = {k: v for k, v in previous_experience_configs.items() if k != experience_name}
                
                replay_datasets, current_replay_raw_examples = sample_coreset_replay_data(
                    previous_raw_data=prev_raw_data,
                    train_config=train_config,
                    model=model,
                    dataset_configs=prev_configs
                )
            else:
                # Use random sampling (baseline)
                prev_raw_data = {k: v for k, v in previous_experience_raw_data.items() if k != experience_name}
                replay_datasets, current_replay_raw_examples = sample_replay_data(
                    previous_datasets=previous_experience_datasets,
                    previous_raw_data=prev_raw_data,
                    replay_percentage=train_config.replay_percentage,
                    replay_num_samples=train_config.replay_num_samples,
                    seed=train_config.seed
                )
            
            # Log actual replay size
            total_replay_size = sum(len(d) for d in replay_datasets) if replay_datasets else 0
            print(f"[REPLAY] Total replay dataset size: {total_replay_size} examples")
            if total_replay_size == 0:
                print(f"[REPLAY] ⚠ WARNING: Replay dataset is EMPTY (no prior experience data available)")
            
            if replay_datasets:
                # Concatenate replay data with current training data
                all_datasets = [dataset_train] + replay_datasets
                dataset_train = concatenate_datasets(all_datasets)
                current_size = len(original_dataset_for_replay)
                replay_size = sum(len(d) for d in replay_datasets)
                print(f"  Combined dataset size: {len(dataset_train)} (current: {current_size}, replay: {replay_size})")
            
            # Store replay examples for neighbor consistency
            # This is our actual "memory" of previous experiences
            if current_replay_raw_examples:
                replay_buffer_raw_data[experience_name] = current_replay_raw_examples
                print(f"  [Neighbor Consistency] Stored {len(current_replay_raw_examples)} replay examples for consistency index")
        else:
            print(f"\n[REPLAY] No replay configured (exp_idx={exp_idx}, replay_percentage={train_config.replay_percentage}, replay_num_samples={train_config.replay_num_samples})")
            replay_datasets = []
        
        # Store the original training dataset (before replay) for future replay
        # This needs to happen even when skipping training
        previous_experience_datasets[experience_name] = original_dataset_for_replay
        
        # Skip actual training if requested
        if skip_training:
            print(f"[SKIP] Loaded {len(raw_train_data)} examples from {experience_name} for replay")
            print(f"[SKIP] Skipping training on {experience_name}\n")
            continue
        
        print(f'\nSTART TRAINING on {experience_name}')

        # Prepare neighbor source data for consistency/contrastive loss (if enabled)
        trainer = train(
            trainConfig=train_config,
            model=model,
            dataset_train=dataset_train,
            dataset_val=dataset_val,
            wandb_logger=wandb_logger,
            replay_source_examples=current_replay_raw_examples if current_replay_raw_examples else None,
            router_ewc_regularizer=router_ewc_regularizer,
        )

        print(f"Completed training on {experience_name}\n")

        # ---------------------------------------------------------------
        # Compute Fisher Information for router EWC after each experience
        # ---------------------------------------------------------------
        if (
            router_ewc_regularizer is not None
            and trainer is not None
            and hasattr(trainer, 'compute_router_fisher')
            and hasattr(trainer, '_router_model')
            and trainer._router_model is not None
        ):
            print(f"\n[RouterEWC] Computing Fisher Information after experience: {experience_name}")
            from torch.utils.data import DataLoader
            fisher_batch_size = getattr(train_config, 'router_ewc_fisher_batch_size', 4)
            fisher_num_samples = getattr(train_config, 'router_ewc_fisher_samples', None)
            fisher_dataloader = DataLoader(
                trainer.train_dataset,
                batch_size=fisher_batch_size,
                shuffle=False,
                collate_fn=trainer.data_collator,
                num_workers=0,
            )
            fisher_dict, optimal_params = trainer.compute_router_fisher(
                fisher_dataloader,
                num_samples=fisher_num_samples,
            )
            if fisher_dict:
                router_ewc_regularizer.add_experience(fisher_dict, optimal_params)
                # Save under the latest checkpoint so the next run can discover
                # Fisher from the previous adapter directly.
                if hasattr(trainer, "_find_latest_checkpoint") and hasattr(trainer, "args"):
                    latest_ckpt = trainer._find_latest_checkpoint(Path(trainer.args.output_dir))
                    if latest_ckpt is not None:
                        ckpt_state_dir = Path(latest_ckpt) / "router_ewc_state"
                        router_ewc_save_dirs.insert(0, ckpt_state_dir)
                # Save to all unique targets (checkpoint-scoped + legacy fallback).
                seen_save_dirs = set()
                for save_dir in router_ewc_save_dirs:
                    key = str(save_dir.resolve()) if save_dir.exists() else str(save_dir)
                    if key in seen_save_dirs:
                        continue
                    seen_save_dirs.add(key)
                    router_ewc_regularizer.save(save_dir)
                    print(f"[RouterEWC] EWC state saved to {save_dir}")
            else:
                print("[RouterEWC] WARNING: Fisher computation returned empty dict; skipping save.")

    # Finish WandB logging
    if wandb_logger:
        wandb_logger.finish()

if __name__ == "__main__":
    main()
