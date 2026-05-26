import os
import json
import random
from pathlib import Path
from typing import Dict, List, Optional
import torch
import gc
from torch.utils.data import DataLoader


from datasets import Dataset, concatenate_datasets
from dotenv import load_dotenv

from .openmodel import LoRAModelManager
from .train_loop import train
from .utils.ewc_utils import EWCRegularizer
from .utils.configs import (  # NOQA
    ApibenchDataConfig,
    MLLMDataConfig,
    HuggingBench1DataConfig,
    HuggingBench2DataConfig,
    ModelIndicesDataConfig,
)
from .utils.parser import TrainParser
from .utils.prepareDataset import convert_to_conversational, load_dataset_json
from .utils.utility import set_seed
from .utils.lwf.openmodelLwF import DualLoRAModelManager
                
PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent.parent
load_dotenv(PROJECT_ROOT / ".env")

cache_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.hf_cache"))

os.makedirs(cache_dir, exist_ok=True)
os.environ["HF_HOME"] = cache_dir
os.environ["HF_HUB_CACHE"] = cache_dir
os.environ["TRANSFORMERS_CACHE"] = cache_dir
os.environ["TOKENIZERS_CACHE"] = cache_dir
os.environ["SENTENCE_TRANSFORMERS_HOME"] = cache_dir


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


def load_old_model_ids_from_index(index_path: Optional[str], key: str = "model_name") -> Optional[set[str]]:
    if index_path is None:
        return None

    old_model_ids: set[str] = set()
    with open(index_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            value = record.get(key, None)
            if value is not None and str(value).strip() != "":
                old_model_ids.add(str(value).strip())

    return old_model_ids


def sample_replay_data(
    previous_datasets: Dict[str, Dataset],
    replay_percentage: Optional[float] = None,
    replay_num_samples: Optional[int] = None,
    seed: Optional[int] = None,
) -> List[Dataset]:
    """
    Sample replay data from previous experiences.

    Args:
        previous_datasets: Dictionary mapping experience names to their datasets
        replay_percentage: Percentage of samples to replay (e.g., 0.1 for 10%)
        replay_num_samples: Fixed number of samples to replay
        seed: Random seed for reproducibility

    Returns:
        List of sampled datasets from previous experiences
    """
    if not previous_datasets:
        return []

    replay_datasets = []

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
        replay_datasets.append(sampled_dataset)

        print(
            f"  Replaying {num_samples} samples from {exp_name} (out of {dataset_size} total)"
        )

    return replay_datasets


def main():
    parser = TrainParser()
    train_config = parser.parse_args()

    print(train_config)

    if train_config.seed is not None:
        # Set seed for reproducibility
        set_seed(train_config.seed)
 
    lora_paths = [
        f"./core/experiments/{adapter}" for adapter in train_config.lora_adapters
    ]
    model = LoRAModelManager(config=train_config, lora_paths=lora_paths)
    
    experiences = train_config.experience_names
    
    # Store datasets from previous experiences for replay
    previous_experience_datasets: Dict[str, Dataset] = {}
    
    # Initialize EWC regularizer if mode is "ewc"
    ewc_regularizer = None
    if train_config.mode == "ewc":
        ewc_regularizer = EWCRegularizer(
        ewc_lambda=train_config.ewc_lambda,
        mode=train_config.ewc_mode,           # "online"
        decay_factor=train_config.ewc_decay_factor,  # 0.9
    )

    # Train on each experience sequentially for sequential finetuning or replay
    if train_config.mode in ["sequential-finetuning", "replay", "merging", "ewc", "lwf"]:
        for exp_idx, experience_name in enumerate(experiences):
                
            print(f"\n{'=' * 80}")
            print(
                f"Training on Experience {exp_idx + 1}/{len(experiences)}: {experience_name}"
            )
            print(f"{'=' * 80}\n")

            
            # Get dataset configuration for current experience
            dataset_config = get_dataset_config(experience_name)
            model_index_name = train_config.model_indices[exp_idx] if train_config.model_indices is not None else None
            retriever_name = train_config.retriever if train_config.retriever is not None else None  
            
            # Load and convert training dataset
            old_model_ids = None
            if train_config.mode == "lwf":
                old_index_name_by_exp = {
                    1: "e1",
                    2: "e1_e2",
                    3: "e1_e2_e3",
                }
                old_index_name = old_index_name_by_exp.get(exp_idx, None)
                model_index_path = (
                    ModelIndicesDataConfig().get_model_index_path(old_index_name)
                    if old_index_name is not None
                    else None
                )
                old_model_ids = load_old_model_ids_from_index(model_index_path)

            dataset_train = convert_to_conversational(
                raw_data=load_dataset_json(dataset_config.train_set),
                tokenizer=model.tokenizer,
                model_index_name=model_index_name,
                retriever_name=retriever_name,
                old_model_ids=old_model_ids,
            )

            # Load and convert validation dataset
            dataset_val = convert_to_conversational(
                raw_data=load_dataset_json(dataset_config.val_set),
                tokenizer=model.tokenizer,
                model_index_name=model_index_name,
                retriever_name=retriever_name,
                old_model_ids=old_model_ids,
            )

            # If no_validation is True, combine train and val sets into a single training set
            if train_config.no_validation:
                print(
                    "no_validation is True: Combining train and val sets into a single training set"
                )
                dataset_train = concatenate_datasets([dataset_train, dataset_val])
                dataset_val = None  # Set to None so it's not used for evaluation

            if train_config.mode == "replay":
                # Store the original training dataset (before replay) for future replay
                # This needs to be done before we add replay data
                original_dataset_for_replay = dataset_train

                # Sample replay data from previous experiences if configured
                if exp_idx > 0 and (
                    train_config.replay_percentage is not None
                    or train_config.replay_num_samples is not None
                ):
                    print("\nSampling replay data from previous experiences:")
                    replay_datasets = sample_replay_data(
                        previous_datasets=previous_experience_datasets,
                        replay_percentage=train_config.replay_percentage,
                        replay_num_samples=train_config.replay_num_samples,
                        seed=train_config.seed,
                    )

                    if replay_datasets:
                        # Concatenate replay data with current training data
                        all_datasets = [dataset_train] + replay_datasets
                        dataset_train = concatenate_datasets(all_datasets)
                        current_size = len(original_dataset_for_replay)
                        replay_size = sum(len(d) for d in replay_datasets)
                        print(
                            f"  Combined dataset size: {len(dataset_train)} (current: {current_size}, replay: {replay_size})"
                        )
            elif train_config.mode == "sequential-finetuning":
                pass
            
            elif train_config.mode == "merging":
                del model
                torch.cuda.empty_cache()
                gc.collect()
                model = LoRAModelManager(config=train_config, lora_paths=None)
            
            elif train_config.mode == "ewc":
                # For EWC, continue training on the same model with EWC regularization
                # No need to reload or reset the model
                pass

            elif train_config.mode == "lwf":
                if exp_idx == 0:
                    pass
                else:
                    base_path = f"./core/baselines/experiments/{experiences[exp_idx - 1]}-{train_config.variant_name}{f'-{train_config.extra_info}' if train_config.extra_info != '' else ''}"
                    dir_names = os.listdir(base_path)
                    dir_name = [d for d in dir_names if d.startswith("checkpoint-")][0]
                    checkpoint_path = os.path.join(base_path, dir_name)
                    print(
                        f"Loading LoRA adapter from {checkpoint_path} for LwF student/teacher initialization."
                    )
                    del model
                    torch.cuda.empty_cache()
                    gc.collect()

                    model = DualLoRAModelManager(
                        config=train_config,
                        device_map="cuda" if torch.cuda.is_available() else "cpu",
                        student_lora_path=checkpoint_path,
                        teacher_lora_path=checkpoint_path,
                    )
                    model.train_student_only()

            print(f"\nSTART TRAINING on {experience_name}")
            trainer = train(
                trainConfig=train_config,
                model=model,
                dataset_train=dataset_train,
                dataset_val=dataset_val,
                experience_name=experience_name,
                ewc_regularizer=ewc_regularizer,
            )
            
            if train_config.mode == "replay":
                # Store the original training dataset (before replay) for future replay
                previous_experience_datasets[experience_name] = original_dataset_for_replay
            
            elif train_config.mode == "ewc":
                # After training, compute Fisher information and store optimal parameters
                print(f"\nComputing Fisher Information for {experience_name}...")
                
                # Get the dataset from the trainer
                fisher_dataset = trainer.train_dataset
                # Create a new dataloader with smaller batch size
                fisher_dataloader = DataLoader(
                    fisher_dataset,
                    batch_size=train_config.ewc_fisher_batch_size,
                    shuffle=False,
                    collate_fn=trainer.data_collator,
                    num_workers=0,  # Use 0 to avoid multiprocessing issues
                )
                
                # Add experience to EWC regularizer
                ewc_regularizer.add_experience(
                    model=model.model,
                    dataloader=fisher_dataloader,
                    num_samples=train_config.ewc_fisher_samples,
                )
                
                # Free Fisher computation memory
                del fisher_dataloader
                torch.cuda.empty_cache()
                gc.collect()
                
                # Save EWC state
                ewc_save_dir = Path(train_config.output_path) / f"{train_config.variant_name}{f'-{train_config.extra_info}' if train_config.extra_info != '' else ''}"
                ewc_regularizer.save(ewc_save_dir)

            # Release the trainer so its model/optimizer state can be freed before the next experience.
            del trainer
            torch.cuda.empty_cache()
            gc.collect()

            print(f"Completed training on {experience_name}\n")
    
    if train_config.mode == "joint-training":
        # concatenate all datasets for joint training
        all_train_datasets = []
        all_val_datasets = []
        for exp_idx, experience_name in enumerate(experiences):
            print(f"\nLoading dataset for experience: {experience_name}")
            dataset_config = get_dataset_config(experience_name)
            
            model_index_name = train_config.model_indices[0] if train_config.model_indices is not None else None
            retriever_name = train_config.retriever if train_config.retriever is not None else None  
            
            dataset_train = convert_to_conversational(
                raw_data=load_dataset_json(dataset_config.train_set),
                tokenizer=model.tokenizer,
                model_index_name=model_index_name,
                retriever_name=retriever_name,
            )
            dataset_val = convert_to_conversational(
                raw_data=load_dataset_json(dataset_config.val_set),
                tokenizer=model.tokenizer,
                model_index_name=model_index_name,
                retriever_name=retriever_name,
            )
            all_train_datasets.append(dataset_train)
            all_val_datasets.append(dataset_val)

        # Concatenate all training and validation datasets
        dataset_train = concatenate_datasets(all_train_datasets)
        dataset_val = concatenate_datasets(all_val_datasets)
        
        print(f"\nSTART JOINT TRAINING: {experiences}")
        train(
            trainConfig=train_config,
            model=model,
            dataset_train=dataset_train,
            dataset_val=dataset_val,
            experience_name=experience_name,
        )  
    
if __name__ == "__main__":
    main()