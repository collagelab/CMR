import gc
from pathlib import Path

import torch
from transformers import EarlyStoppingCallback
from trl import SFTConfig, SFTTrainer

from .utils.configs import TrainConfig
from .utils.utility import MemoryCleanupCallback
from .utils.wandb import WandbTrainingCallback
from ..baselines.utils.wandb import WandbLogger
from .utils.ewc_trainer import EWCSFTTrainer
from .utils.lwf.train_loop_lwf_online import train_lwf
import os

def train(
    trainConfig: TrainConfig, model, dataset_train, dataset_val, experience_name, ewc_regularizer=None
):
    if trainConfig.mode == "lwf":
        return train_lwf(
            trainConfig=trainConfig,
            model=model,
            dataset_train=dataset_train,
            dataset_val=dataset_val,
            experience_name=experience_name,
            alpha=trainConfig.alpha,
            temperature=trainConfig.temperature,
            kd_on_new=trainConfig.kd_on_new,
        )

    # Initialize WandB logger
    wandb_key = os.getenv("WANDB_API_KEY")
    if wandb_key:
        wandb_logger = WandbLogger(wandb_key, trainConfig, mode="train")
    else:
        wandb_logger = None
        print(
            "Warning: WANDB_API_KEY not found in environment variables. Skipping WandB logging."
        )
        
        
    # create dir where experiments are saved
    adapter_dir = (
        Path(trainConfig.output_path)
        / f"{experience_name}-{trainConfig.variant_name}{f'-{trainConfig.extra_info}' if trainConfig.extra_info != '' else ''}"
    )  # add timestamp if needed
    if trainConfig.retriever is not None:
        # Append retriever to the final path component in a Path-safe way
        adapter_dir = adapter_dir.with_name(
            adapter_dir.name + "-" + trainConfig.retriever
        )
    adapter_dir.mkdir(parents=True, exist_ok=True)

    tok = model.tokenizer
    llm = model.model

    tok.padding_side = "right"
    tok.add_eos_token = False

    cfg_kwargs = dict(
        # Reproducibility
        **(
            {"seed": trainConfig.seed, "data_seed": trainConfig.seed}
            if trainConfig.seed is not None
            else {}
        ),
        # Gradient checkpointing
        gradient_checkpointing=trainConfig.activation_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        # Gradient accumulation and batch size
        gradient_accumulation_steps=trainConfig.grad_accum,
        per_device_train_batch_size=trainConfig.batch_size,
        max_length=trainConfig.max_length,
        # Efficiency
        group_by_length=trainConfig.group_by_length,
        packing=trainConfig.packing,
        # Optimizer and scheduler
        weight_decay=trainConfig.weight_decay,
        learning_rate=trainConfig.lr,
        lr_scheduler_type=trainConfig.lr_scheduler_type,
        optim=trainConfig.optim,
        warmup_steps=trainConfig.warmup_steps,
        max_grad_norm=trainConfig.max_grad_norm,
        label_smoothing_factor=trainConfig.label_smoothing,
        # Training control
        num_train_epochs=trainConfig.epochs,
        logging_steps=trainConfig.logging_steps,
        logging_dir=str(adapter_dir / "logs"),
        output_dir=str(adapter_dir),
        # Reporting
        report_to="wandb" if wandb_logger else "none",
        disable_tqdm=False,
        # Loss settings
        completion_only_loss=trainConfig.completion_only_loss,
        # Checkpointing
        save_strategy=trainConfig.save_strategy,
        save_total_limit=trainConfig.save_total_limit,
        eval_strategy="no" if trainConfig.no_validation else "epoch",
        load_best_model_at_end=trainConfig.hyperparameters_search,
        metric_for_best_model=trainConfig.metric_for_best_model
        if not trainConfig.no_validation
        else None,
        greater_is_better=trainConfig.greater_is_better,
        dataloader_pin_memory=False,  # Reduce memory overhead
        eval_accumulation_steps=4,  # Don't accumulate all eval outputs
        save_safetensors=True,  # More efficient checkpoint format
    )

    # Add activation offloading if low_memory_mode is enabled
    if trainConfig.low_memory_mode:
        cfg_kwargs["gradient_checkpointing_kwargs"] = {
            "use_reentrant": False,
            "offload_to_cpu": True,  # Offload activations to CPU
        }
        # Only enable bf16 if the system supports it
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            cfg_kwargs["bf16"] = True  # Use BF16 half precision
        else:
            cfg_kwargs["fp16"] = True  # Fallback to FP16 if bf16 not supported

    sft_cfg = SFTConfig(**cfg_kwargs)

    # Set up callbacks
    callbacks = [MemoryCleanupCallback()]
    if wandb_logger:
        callbacks.append(WandbTrainingCallback(wandb_logger))

    # Add early stopping callback if validation is enabled
    if not trainConfig.no_validation:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=trainConfig.early_stopping_patience,
                early_stopping_threshold=trainConfig.early_stopping_threshold,
            )
        )

    # Only pass eval_dataset if validation is enabled
    trainer_kwargs = {
        "model": llm,
        "processing_class": tok,
        "args": sft_cfg,
        "train_dataset": dataset_train,
        "callbacks": callbacks,
    }

    # Only include eval_dataset if validation is enabled
    if not trainConfig.no_validation and dataset_val is not None:
        trainer_kwargs["eval_dataset"] = dataset_val

    # Use EWC trainer if EWC is enabled
    if trainConfig.mode == "ewc" and ewc_regularizer is not None:
        trainer = EWCSFTTrainer(ewc_regularizer=ewc_regularizer, **trainer_kwargs)
    else:
        trainer = SFTTrainer(**trainer_kwargs)

    torch.cuda.empty_cache()
    gc.collect()

    # Train
    if trainConfig.resume_from:
        base_path = "./core/experiments/"
        trainer.train(resume_from_checkpoint=base_path + f"{trainConfig.resume_from}/")
    else:
        trainer.train()

    try:
        torch.save(trainer.args, str(adapter_dir / "training_args.bin"))
    except Exception:
        pass

    if trainConfig.hyperparameters_search:
        if trainConfig.no_validation:
            raise ValueError(
                "Cannot perform hyperparameter search with no_validation=True. Evaluation is required for hyperparameter search."
            )
        eval_results = trainer.evaluate()
        return eval_results["eval_loss"]
    
    # Finish WandB logging
    if wandb_logger:
        wandb_logger.finish()

    # Return trainer for EWC Fisher computation or None for hyperparameter search
    if trainConfig.hyperparameters_search:
        return eval_results["eval_loss"]
    else:
        return trainer