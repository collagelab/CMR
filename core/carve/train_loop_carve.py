import torch
from pathlib import Path
from typing import Optional, List, Dict, Any
from .utils_carve.configs import TrainConfig
from trl import SFTConfig, SFTTrainer
from transformers import EarlyStoppingCallback
from .utils_carve.wandb import WandbTrainingCallback
from .utils_carve.utility import MemoryCleanupCallback
from .utils_carve.consistency_trainer import NeighborConsistencySFTTrainer
from .utils_carve.router_collator import RouterDataCollator
import gc

def train(
    trainConfig: TrainConfig,
    model,
    dataset_train,
    dataset_val,
    wandb_logger=None,
    replay_source_examples: Optional[List[Dict[str, Any]]] = None,
    router_ewc_regularizer=None,
):
    uses_router_or_similarity_losses = (
        hasattr(trainConfig, "loss_mode")
        and trainConfig.loss_mode in ["router", "router+graph", "supervised+router", "supervised+router+graph"]
    )

    """
    Train a model using SFT with optional neighbor-consistency regularisation.
    
    Args:
        trainConfig: Training configuration
        model: LoRAModelManager instance
        dataset_train: Training dataset
        dataset_val: Validation dataset (can be None if no_validation=True)
        wandb_logger: Optional WandB logger
        replay_source_examples: Raw replay examples for X-CLR candidate pool
                               Used when xclr_apply_to="replay_only" to build candidate pool
    """
    # create dir where experiments are saved
    adapter_dir = Path(trainConfig.output_root) / \
        f"{trainConfig.experience_name}-{trainConfig.variant_name}{f'-{trainConfig.extra_info}' if trainConfig.extra_info != '' else ''}"  # add timestamp if needed
    if trainConfig.retriever is not None:
        # Append retriever to the final path component in a Path-safe way
        adapter_dir = adapter_dir.with_name(adapter_dir.name + "-" + trainConfig.retriever)
    adapter_dir.mkdir(parents=True, exist_ok=True)

    tok = model.tokenizer
    llm = model.model
    
    tok.padding_side = "right"   
    tok.add_eos_token = False

    cfg_kwargs = dict(
        # Reproducibility
        **({"seed": trainConfig.seed, "data_seed": trainConfig.seed} if trainConfig.seed is not None else {}),
        
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
        metric_for_best_model=trainConfig.metric_for_best_model if not trainConfig.no_validation else None,
        greater_is_better=trainConfig.greater_is_better,

        dataloader_pin_memory=False,  # Reduce memory overhead
        eval_accumulation_steps=4,  # Don't accumulate all eval outputs
        save_safetensors=True,  # More efficient checkpoint format
        
        # Preserve extra columns only when router/similarity objectives require them.
        remove_unused_columns=not uses_router_or_similarity_losses,
    )
    
    # For plain supervised parity with classic train-baseline, avoid low-memory/offload path.
    effective_low_memory_mode = trainConfig.low_memory_mode and uses_router_or_similarity_losses
    # Add activation offloading only when low-memory mode is enabled for router/similarity objectives.
    if effective_low_memory_mode:
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
        callbacks.append(EarlyStoppingCallback(
            early_stopping_patience=trainConfig.early_stopping_patience,
            early_stopping_threshold=trainConfig.early_stopping_threshold
        ))

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
    
    # ==========================================================================
    # Choose between standard SFTTrainer and NeighborConsistencySFTTrainer
    # ==========================================================================

    if hasattr(trainConfig, 'loss_mode') and trainConfig.loss_mode in ["router", "router+graph", "supervised+router", "supervised+router+graph"]:
        # ======================================================================
        # Router Mode (Semantic Batching + Candidate-Set Routing Loss)
        # ======================================================================
        print("\n[Router] Initializing router training...")
        
        # Use custom collator that preserves metadata (model_name, domain)
        router_collator = RouterDataCollator(
            tokenizer=tok,
            mlm=False,  # Causal language modeling, not masked LM
            max_length=trainConfig.max_length,
        )
        trainer_kwargs["data_collator"] = router_collator
        print("[Router] Using RouterDataCollator to preserve metadata and compute prompt_len")
        
        # Add router-specific parameters to trainer_kwargs
        trainer_kwargs.update({
            "loss_mode": trainConfig.loss_mode,
            "router_loss_weight": getattr(trainConfig, 'router_loss_weight', 1.0),
            "lm_loss_weight": getattr(trainConfig, 'lm_loss_weight', 1.0),
            # Semantic batching (for router training)
            "semantic_batching": getattr(trainConfig, 'semantic_batching', False),
            "domains_per_batch": getattr(trainConfig, 'domains_per_batch', 1),
            "mix_replay_in_semantic_batches": getattr(trainConfig, 'mix_replay_in_semantic_batches', True),
            # Router architecture
            "router_embedding_dim": getattr(trainConfig, 'router_embedding_dim', None),
            "router_tau": getattr(trainConfig, 'router_tau', 0.07),
            "router_pooling": getattr(trainConfig, 'router_pooling', 'last_token'),
            # Router learning rates
            "router_proj_lr": getattr(trainConfig, 'router_proj_lr', None),
            "router_embedding_lr": getattr(trainConfig, 'router_embedding_lr', None),
            # Candidate sampling
            "router_K_total": getattr(trainConfig, 'router_K_total', 64),
            "router_K_semantic": getattr(trainConfig, 'router_K_semantic', 48),
            "router_K_far": getattr(trainConfig, 'router_K_far', 8),
            "router_K_hard": getattr(trainConfig, 'router_K_hard', 7),
            "router_use_legacy_batching": getattr(trainConfig, 'router_use_legacy_batching', False),
            # Hard negative mining
            "router_mine_every_steps": getattr(trainConfig, 'router_mine_every_steps', 200),
            "router_K_hard_pool": getattr(trainConfig, 'router_K_hard_pool', 20),
            "router_semantic_pool_size": getattr(trainConfig, 'router_semantic_pool_size', 512),
            "router_max_pool_size": getattr(trainConfig, 'router_max_pool_size', 1024),
            # Soft targets
            "router_use_soft_targets": getattr(trainConfig, 'router_use_soft_targets', False),
            "router_soft_target_eps": getattr(trainConfig, 'router_soft_target_eps', 0.1),
            "router_soft_target_k_neighbors": getattr(trainConfig, 'router_soft_target_k_neighbors', 3),
            # Label-side graph regularization
            "router_use_label_graph_reg": getattr(trainConfig, 'router_use_label_graph_reg', False),
            "router_label_graph_lambda": getattr(trainConfig, 'router_label_graph_lambda', 0.1),
            "router_label_graph_tau": getattr(trainConfig, 'router_label_graph_tau', 0.07),
            "router_label_graph_tau_target": getattr(trainConfig, 'router_label_graph_tau_target', 0.1),
            "router_label_graph_max_models": getattr(trainConfig, 'router_label_graph_max_models', 256),
            "router_label_graph_alpha_domain": getattr(trainConfig, 'router_label_graph_alpha_domain', 0.3),
            # Model registry
            "router_registry_path": getattr(trainConfig, 'router_registry_path', None),
            "router_registry_init_mode": getattr(trainConfig, 'router_registry_init_mode', 'extend'),
            "router_registry_base_path": getattr(trainConfig, 'router_registry_base_path', None),
            # Two-phase training schedule
            "router_two_phase_enable": getattr(trainConfig, 'router_two_phase_enable', False),
            "router_phase1_frac": getattr(trainConfig, 'router_phase1_frac', 0.2),
            "router_phase1_loss_mode": getattr(trainConfig, 'router_phase1_loss_mode', 'router'),
            "router_phase1_replay_ratio": getattr(trainConfig, 'router_phase1_replay_ratio', None),
            "router_phase1_router_loss_weight": getattr(trainConfig, 'router_phase1_router_loss_weight', 1.0),
            "router_phase1_lm_loss_weight": getattr(trainConfig, 'router_phase1_lm_loss_weight', 0.0),
            "router_phase1_proj_lr": getattr(trainConfig, 'router_phase1_proj_lr', None),
            "router_phase1_embedding_lr": getattr(trainConfig, 'router_phase1_embedding_lr', None),
            "router_phase1_use_soft_targets": getattr(trainConfig, 'router_phase1_use_soft_targets', False),
            "router_phase1_soft_target_eps": getattr(trainConfig, 'router_phase1_soft_target_eps', 0.02),
            "router_replay_loss_multiplier": getattr(trainConfig, 'router_replay_loss_multiplier', 1.0),
            # Exp1-preservation training mode
            "router_exp1_preservation_enable": getattr(trainConfig, 'router_exp1_preservation_enable', False),
            "router_exp1_preservation_M_old": getattr(trainConfig, 'router_exp1_preservation_M_old', None),
            # Router embedding anchoring regularizer
            "router_anchor_enable": getattr(trainConfig, 'router_anchor_enable', False),
            "router_anchor_lambda": getattr(trainConfig, 'router_anchor_lambda', 1e-3),
            "router_anchor_mode": getattr(trainConfig, 'router_anchor_mode', "normalized"),
            "router_anchor_apply_phase": getattr(trainConfig, 'router_anchor_apply_phase', "phase1"),
            "router_anchor_scope": getattr(trainConfig, 'router_anchor_scope', "all_old"),
            "router_anchor_M_old": getattr(trainConfig, 'router_anchor_M_old', None),
            # Router projection anchoring
            "router_proj_anchor_enable": getattr(trainConfig, 'router_proj_anchor_enable', False),
            "router_proj_anchor_lambda": getattr(trainConfig, 'router_proj_anchor_lambda', 1e-2),
            "router_proj_anchor_apply_phase": getattr(trainConfig, 'router_proj_anchor_apply_phase', "phase1"),
            # Adaptive lambda scaling – embedding anchor
            "router_anchor_adaptive_enable": getattr(trainConfig, 'router_anchor_adaptive_enable', False),
            "router_anchor_adaptive_beta": getattr(trainConfig, 'router_anchor_adaptive_beta', 0.99),
            "router_anchor_adaptive_update_every": getattr(trainConfig, 'router_anchor_adaptive_update_every', 20),
            "router_anchor_adaptive_target_ratio_phase1": getattr(trainConfig, 'router_anchor_adaptive_target_ratio_phase1', 0.3),
            "router_anchor_adaptive_target_ratio_phase2": getattr(trainConfig, 'router_anchor_adaptive_target_ratio_phase2', 0.1),
            "router_anchor_lambda_min": getattr(trainConfig, 'router_anchor_lambda_min', 1e2),
            "router_anchor_lambda_max": getattr(trainConfig, 'router_anchor_lambda_max', 5e4),
            "router_anchor_adaptive_reg_ema_min": getattr(trainConfig, 'router_anchor_adaptive_reg_ema_min', 1e-8),
            # Adaptive lambda scaling – projection anchor
            "router_proj_anchor_adaptive_enable": getattr(trainConfig, 'router_proj_anchor_adaptive_enable', False),
            "router_proj_anchor_adaptive_beta": getattr(trainConfig, 'router_proj_anchor_adaptive_beta', 0.99),
            "router_proj_anchor_adaptive_update_every": getattr(trainConfig, 'router_proj_anchor_adaptive_update_every', 20),
            "router_proj_anchor_adaptive_target_ratio_phase1": getattr(trainConfig, 'router_proj_anchor_adaptive_target_ratio_phase1', 0.5),
            "router_proj_anchor_adaptive_target_ratio_phase2": getattr(trainConfig, 'router_proj_anchor_adaptive_target_ratio_phase2', 0.1),
            "router_proj_anchor_lambda_min": getattr(trainConfig, 'router_proj_anchor_lambda_min', 1e2),
            "router_proj_anchor_lambda_max": getattr(trainConfig, 'router_proj_anchor_lambda_max', 2e4),
            "router_proj_anchor_adaptive_reg_ema_min": getattr(trainConfig, 'router_proj_anchor_adaptive_reg_ema_min', 1e-10),

            # Router EWC regularizer (replaces anchor losses when enabled)
            "router_ewc_regularizer": router_ewc_regularizer,
            "router_ewc_enable": getattr(trainConfig, 'router_ewc_enable', False),
            "router_ewc_apply_phase": getattr(trainConfig, 'router_ewc_apply_phase', "phase1"),
            # Router freeze LM option
            "router_freeze_lm": getattr(trainConfig, 'router_freeze_lm', False),
            # Card-guided cold-start initialisation for new embedding rows
            "card_guided_init_enable": getattr(trainConfig, 'card_guided_init_enable', False),
            "card_init_tau": getattr(trainConfig, 'card_init_tau', 0.07),
            "card_init_topk": getattr(trainConfig, 'card_init_topk', 30),
            "card_init_scope": getattr(trainConfig, 'card_init_scope', "global"),
            "card_init_min_sim_threshold": getattr(trainConfig, 'card_init_min_sim_threshold', 0.2),
            "card_init_fallback_domain": getattr(trainConfig, 'card_init_fallback_domain', True),
            # Pass replay source examples for registry building
            "replay_source_examples": replay_source_examples,
        })
        
        trainer = NeighborConsistencySFTTrainer(**trainer_kwargs)
        print("[Router] Using NeighborConsistencySFTTrainer with router training")
    else:
        # Standard SFTTrainer (baseline)
        trainer = SFTTrainer(**trainer_kwargs)
    torch.cuda.empty_cache()
    gc.collect()
    
    # Step-0 evaluation (sanity check: confirm expansion-only behavior)
    if trainConfig.eval_at_step0 and hasattr(trainer, 'evaluate') and dataset_val is not None:
        print("\n" + "="*80)
        print("[STEP-0 EVALUATION] Running evaluation before training begins (global_step==0)")
        print("="*80)
        try:
            # Ensure trainer is in eval mode
            trainer.model.eval()
            if hasattr(trainer, '_router_model') and trainer._router_model is not None:
                trainer._router_model.eval()
            
            # Run evaluation
            eval_results = trainer.evaluate(eval_dataset=dataset_val)
            
            print(f"\n[STEP-0 EVALUATION RESULTS]")
            for key, value in eval_results.items():
                if isinstance(value, (int, float)):
                    print(f"  {key}: {value:.6f}")
                else:
                    print(f"  {key}: {value}")
            print("="*80 + "\n")
        except Exception as e:
            print(f"  ⚠️  Warning: Step-0 evaluation failed: {e}")
            print("  Continuing with training...\n")
    
    # Train
    if trainConfig.resume_from:
        base_path = "./core/experiments/"
        trainer.train(resume_from_checkpoint=base_path+f"{trainConfig.resume_from}/")
    else:
        trainer.train()
    
    try:
        torch.save(trainer.args, str(adapter_dir / "training_args.bin"))
    except Exception:
        pass
    
    if trainConfig.hyperparameters_search:
        if trainConfig.no_validation:
            raise ValueError("Cannot perform hyperparameter search with no_validation=True. Evaluation is required for hyperparameter search.")
        eval_results = trainer.evaluate()
        return eval_results["eval_loss"]

    return trainer


