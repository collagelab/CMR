"""
Custom SFTTrainer with Neighbour-Consistency and Neighbour-Contrastive Regularisation




"""

import logging
import os
from pathlib import Path
from typing import Dict, Optional, Any, Union, Tuple
from collections import defaultdict

logger = logging.getLogger(__name__)

import torch
from torch import nn
from torch.utils.data import DataLoader

from trl import SFTTrainer, SFTConfig
from transformers import PreTrainedModel, PreTrainedTokenizer
from transformers.trainer_callback import TrainerCallback


from .router_similarity_utils import (
     enable_hidden_states,
 )
from .router_training import (
    compute_label_graph_regularizer,
)
from ..model_selection_carve import (
    normalize_model_name,
    CandidateSetBuilder,
    HardNegativeMiner,
)
from .router_trainer_metrics import (
    LossAccum as _LossAccum,
)
from .router_regularizers import (
    compute_router_anchor_loss as _compute_router_anchor_loss_helper,
    compute_router_proj_anchor_loss as _compute_router_proj_anchor_loss_helper,
)
from .router_bootstrap import bootstrap_router_state
from .router_fisher import compute_router_fisher_information
from .router_phase_controller import RouterPhaseController
from .router_batch_loss import compute_routing_loss_for_batch as _compute_routing_loss_for_batch_impl
from .router_loss_pipeline import compute_loss_pipeline

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

_ROUTER_MODES = frozenset({"router", "router+graph", "supervised+router", "supervised+router+graph"})
_PURE_ROUTER_MODES = frozenset({"router", "router+graph"})
class ConsistencyLoggingCallback(TrainerCallback):
    """Callback to log consistency, contrastive, and X-CLR loss metrics at epoch end."""
    
    def __init__(self, trainer: "NeighborConsistencySFTTrainer"):
        self.trainer = trainer
    
    def on_epoch_end(self, args, state, control, **kwargs):
        """Log and reset metrics at epoch end."""

        
        # Log router metrics if enabled
        if hasattr(self.trainer, "_loss_mode") and self.trainer._loss_mode in _ROUTER_MODES:
            metrics = self.trainer.get_router_metrics()
            if metrics:
                logger.info(f"\n  [Router Metrics @ Epoch {state.epoch:.0f}]")
                for key, value in metrics.items():
                    if isinstance(value, float):
                        logger.info(f"    {key}: {value:.4f}")
                    else:
                        logger.info(f"    {key}: {value}")
        
        self.trainer.reset_consistency_metrics()
    
    def on_save(self, args, state, control, **kwargs):
        """Save router checkpoint when a checkpoint is saved during training."""
        # Save router checkpoint if router training was enabled
        if hasattr(self.trainer, "_loss_mode") and self.trainer._loss_mode in _ROUTER_MODES:
            if hasattr(self.trainer, "_router_model") and self.trainer._router_model is not None:
                output_dir = Path(args.output_dir)
                checkpoint_dir = None
                possible_names = [
                    f"checkpoint-{state.global_step}",  # Step-based
                    f"checkpoint-{int(state.epoch)}",    # Epoch-based
                ]

                for name in possible_names:
                    candidate = output_dir / name
                    if candidate.exists():
                        checkpoint_dir = candidate
                        break

                if checkpoint_dir is None:
                    checkpoint_dirs = [d for d in output_dir.iterdir()
                                     if d.is_dir() and d.name.startswith("checkpoint-")]
                    if checkpoint_dirs:
                        checkpoint_dirs.sort(key=lambda x: os.path.getmtime(x), reverse=True)
                        checkpoint_dir = checkpoint_dirs[0]

                if checkpoint_dir and checkpoint_dir.exists():
                    logger.info(f"\n[Router] Saving router checkpoint to {checkpoint_dir}")
                    self.trainer.save_router_checkpoint(str(checkpoint_dir))
    
    def on_train_end(self, args, state, control, **kwargs):
        """Save router checkpoint at end of training (fallback to parent directory for backwards compatibility)."""
        if hasattr(self.trainer, "_loss_mode") and self.trainer._loss_mode in _ROUTER_MODES:
            if hasattr(self.trainer, "_router_model") and self.trainer._router_model is not None:
                output_dir = Path(args.output_dir)
                latest_checkpoint = self.trainer._find_latest_checkpoint(output_dir)

                if latest_checkpoint is not None:
                    logger.info(f"\n[Router] Saving router checkpoint to latest checkpoint: {latest_checkpoint}")
                    self.trainer.save_router_checkpoint(str(latest_checkpoint))
                else:
                    logger.info(f"\n[Router] Saving router checkpoint to {args.output_dir}")
                    self.trainer.save_router_checkpoint(args.output_dir)


class NeighborConsistencySFTTrainer(SFTTrainer):
    """
    SFTTrainer with optional neighbour-consistency regularisation.
    
    When use_neighbor_consistency=True, this trainer:
    1. Retrieves similar prompts for each training example
    2. Computes the standard supervised loss
    3. Adds a KL-divergence consistency loss between anchor and neighbor predictions
    4. Returns the combined loss: L_supervised + weight * L_consistency
    
    The neighbor index should be built from previous experience data (e.g., APIBench)
    and passed during initialization.
    


    """
    
    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module],
        args: SFTConfig,
        train_dataset,
        processing_class: Optional[PreTrainedTokenizer] = None,
        eval_dataset=None,
        callbacks=None,

        loss_mode: str = "supervised",  # "supervised", "router", "router+graph"
        router_loss_weight: float = 1.0,  # Weight for routing loss
        lm_loss_weight: float = 1.0,  # Weight for LM supervised loss
        # Semantic batching
        semantic_batching: bool = False,  # Enable domain-based semantic batching
        domains_per_batch: int = 1,  # Number of domains per batch
        mix_replay_in_semantic_batches: bool = True,  # Mix replay into semantic batches
        # Router architecture
        router_embedding_dim: Optional[int] = None,  # Defaults to hidden_size
        router_tau: float = 0.07,  # Temperature for scaling logits
        router_pooling: str = "last_token",  # "last_token" or "mean"
        # Router learning rates (split by parameter group)
        router_proj_lr: Optional[float] = None,  # Learning rate for projection head (None = use args.learning_rate)
        router_embedding_lr: Optional[float] = None,  # Learning rate for embedding table (None = use args.learning_rate)
        # Candidate sampling
        router_K_total: int = 64,
        router_K_semantic: int = 48,
        router_K_far: int = 8,
        router_K_hard: int = 7,
        router_use_legacy_batching: bool = False,
        # Hard negative mining
        router_mine_every_steps: int = 200,
        router_K_hard_pool: int = 20,
        router_semantic_pool_size: int = 512,
        router_max_pool_size: int = 1024,
        # Semantic pool expansion (Option B)
        router_semantic_pool_mode: str = "parent_group",
        router_semantic_pool_max_domains: Optional[int] = None,
        router_semantic_pool_depth: int = 1,
        # Soft targets
        router_use_soft_targets: bool = False,
        router_soft_target_eps: float = 0.1,
        router_soft_target_k_neighbors: int = 5,
        # Label-side graph regularizer
        router_use_label_graph_reg: bool = False,
        router_label_graph_lambda: float = 0.1,
        router_label_graph_tau: float = 0.07,
        router_label_graph_tau_target: float = 0.1,
        router_label_graph_max_models: int = 256,
        router_label_graph_alpha_domain: float = 0.3,
        # Model registry
        router_registry_path: Optional[str] = None,
        router_registry_init_mode: str = "extend",
        router_registry_base_path: Optional[str] = None,
        # Two-phase training schedule (for Experience 2+ to reduce forgetting)
        router_two_phase_enable: bool = False,
        router_phase1_frac: float = 0.2,
        router_phase1_loss_mode: str = "router",
        router_phase1_replay_ratio: Optional[float] = None,
        router_phase1_router_loss_weight: float = 1.0,
        router_phase1_lm_loss_weight: float = 0.0,
        router_phase1_proj_lr: Optional[float] = None,
        router_phase1_embedding_lr: Optional[float] = None,
        router_phase1_use_soft_targets: bool = False,
        router_phase1_soft_target_eps: float = 0.02,
        router_replay_loss_multiplier: float = 1.0,
        # Exp1-preservation training mode
        router_exp1_preservation_enable: bool = False,
        router_exp1_preservation_M_old: Optional[int] = None,
        # Router embedding anchoring regularizer
        router_anchor_enable: bool = False,
        router_anchor_lambda: float = 1e-3,
        router_anchor_mode: str = "normalized",
        router_anchor_apply_phase: str = "phase1",
        router_anchor_scope: str = "all_old",
        router_anchor_M_old: Optional[int] = None,
        # Router projection anchoring regularizer
        router_proj_anchor_enable: bool = False,
        router_proj_anchor_lambda: float = 1e-2,
        router_proj_anchor_apply_phase: str = "phase1",
        # Adaptive lambda scaling – embedding anchor
        router_anchor_adaptive_enable: bool = False,
        router_anchor_adaptive_beta: float = 0.99,
        router_anchor_adaptive_update_every: int = 20,
        router_anchor_adaptive_target_ratio_phase1: float = 0.3,
        router_anchor_adaptive_target_ratio_phase2: float = 0.1,
        router_anchor_lambda_min: float = 1e2,
        router_anchor_lambda_max: float = 5e4,
        router_anchor_adaptive_reg_ema_min: float = 1e-8,
        # Adaptive lambda scaling – projection anchor
        router_proj_anchor_adaptive_enable: bool = False,
        router_proj_anchor_adaptive_beta: float = 0.99,
        router_proj_anchor_adaptive_update_every: int = 20,
        router_proj_anchor_adaptive_target_ratio_phase1: float = 0.5,
        router_proj_anchor_adaptive_target_ratio_phase2: float = 0.1,
        router_proj_anchor_lambda_min: float = 1e2,
        router_proj_anchor_lambda_max: float = 2e4,
        router_proj_anchor_adaptive_reg_ema_min: float = 1e-10,

        # Router EWC regularizer (replaces anchor losses when enabled)
        router_ewc_regularizer=None,
        router_ewc_enable: bool = False,
        router_ewc_apply_phase: str = "phase1",
        # Router freeze LM option (for router-only runs)
        router_freeze_lm: bool = False,
        # Card-guided cold-start initialisation for new embedding rows
        card_guided_init_enable: bool = False,
        card_init_tau: float = 0.07,
        card_init_topk: int = 30,
        card_init_scope: str = "global",
        card_init_min_sim_threshold: float = 0.2,
        card_init_fallback_domain: bool = True,
        **kwargs
    ):
        """
        Initialize the trainer.
        
        Args:
            model: The model to train
            args: SFTConfig training arguments
            train_dataset: Training dataset
            processing_class: Tokenizer
            eval_dataset: Optional evaluation dataset
            callbacks: Optional list of callbacks
            

            

            
            **kwargs: Additional arguments for SFTTrainer
            

        """
        # Initialize callbacks list
        if callbacks is None:
            callbacks = []
        
        # Initialize tracking metrics
        self._consistency_accum = _LossAccum()
        self._supervised_accum = _LossAccum()
        self._contrastive_accum = _LossAccum()
        self._contrastive_anchors_used = 0
        self._contrastive_negatives_used = 0
    
        replay_source_examples = kwargs.pop('replay_source_examples', None)
        
        super().__init__(
            model=model,
            args=args,
            train_dataset=train_dataset,
            processing_class=processing_class,
            eval_dataset=eval_dataset,
            callbacks=callbacks,
            **kwargs
        )
        
        self._xclr_replay_source_examples = replay_source_examples
        
        if loss_mode in _ROUTER_MODES:
            self.add_callback(ConsistencyLoggingCallback(self))
        
  
        self._loss_mode = loss_mode
        self._router_loss_weight = router_loss_weight
        self._lm_loss_weight = lm_loss_weight
        # Semantic batching
        self._semantic_batching = semantic_batching
        self._domains_per_batch = domains_per_batch
        self._mix_replay_in_semantic_batches = mix_replay_in_semantic_batches
        # Router components
        self._router_model = None
        self._router_registry = None
        self._router_candidate_builder = None
        self._router_hard_miner = None
        self._router_hard_negative_cache = {}
        self._router_use_soft_targets = router_use_soft_targets
        self._router_soft_target_eps = router_soft_target_eps
        self._router_soft_target_k_neighbors = router_soft_target_k_neighbors
        self._router_use_label_graph_reg = router_use_label_graph_reg
        self._router_label_graph_lambda = router_label_graph_lambda
        self._router_label_graph_tau = router_label_graph_tau
        self._router_label_graph_tau_target = router_label_graph_tau_target
        self._router_label_graph_max_models = router_label_graph_max_models
        self._router_label_graph_alpha_domain = router_label_graph_alpha_domain
        self._router_mine_every_steps = router_mine_every_steps
        self._router_use_legacy_batching = router_use_legacy_batching
        # Router learning rates (ensure they are floats)
        self._router_proj_lr = float(router_proj_lr) if router_proj_lr is not None else float(args.learning_rate)
        self._router_embedding_lr = float(router_embedding_lr) if router_embedding_lr is not None else float(args.learning_rate)
        # Two-phase training schedule (for Experience 2+ to reduce forgetting)
        self._router_two_phase_enable = router_two_phase_enable
        self._router_phase1_frac = router_phase1_frac
        self._router_phase1_loss_mode = router_phase1_loss_mode
        self._router_phase1_replay_ratio = router_phase1_replay_ratio
        self._router_phase1_router_loss_weight = router_phase1_router_loss_weight
        self._router_phase1_lm_loss_weight = router_phase1_lm_loss_weight
        self._router_phase1_proj_lr = float(router_phase1_proj_lr) if router_phase1_proj_lr is not None else self._router_proj_lr
        self._router_phase1_embedding_lr = float(router_phase1_embedding_lr) if router_phase1_embedding_lr is not None else self._router_embedding_lr
        self._router_phase1_use_soft_targets = router_phase1_use_soft_targets
        self._router_phase1_soft_target_eps = router_phase1_soft_target_eps
        self._router_replay_loss_multiplier = router_replay_loss_multiplier
        
        # Exp1-preservation training mode
        self._router_exp1_preservation_enable = router_exp1_preservation_enable
        self._router_exp1_preservation_M_old = router_exp1_preservation_M_old
        # Router embedding anchoring regularizer
        self._router_anchor_enable = router_anchor_enable
        self._router_anchor_lambda = float(router_anchor_lambda) if router_anchor_lambda is not None else 1e-3
        self._router_anchor_mode = router_anchor_mode
        self._router_anchor_apply_phase = router_anchor_apply_phase
        self._router_anchor_scope = router_anchor_scope
        self._router_anchor_M_old = router_anchor_M_old
        self._router_anchor_ref_cpu = None  # Reference snapshot on CPU (FP32, compact)
        self._router_anchor_ref = None  # Cached device copy (materialized on-demand)
        # Projection anchoring
        self._router_proj_anchor_enable = router_proj_anchor_enable
        self._router_proj_anchor_lambda = float(router_proj_anchor_lambda) if router_proj_anchor_lambda is not None else 1e-2
        self._router_proj_anchor_apply_phase = router_proj_anchor_apply_phase
        self._router_proj_anchor_ref_cpu = None  # Reference snapshot of projection weights (CPU, FP32)
        self._router_proj_anchor_ref = None  # Cached device copy (materialized on-demand)
        # Adaptive lambda scaling – embedding anchor
        self._router_anchor_adaptive_enable = router_anchor_adaptive_enable
        self._router_anchor_adaptive_beta = router_anchor_adaptive_beta
        self._router_anchor_adaptive_update_every = router_anchor_adaptive_update_every
        self._router_anchor_adaptive_target_ratio_phase1 = router_anchor_adaptive_target_ratio_phase1
        self._router_anchor_adaptive_target_ratio_phase2 = router_anchor_adaptive_target_ratio_phase2
        self._router_anchor_lambda_min = float(router_anchor_lambda_min)
        self._router_anchor_lambda_max = float(router_anchor_lambda_max)
        self._router_anchor_adaptive_reg_ema_min = float(router_anchor_adaptive_reg_ema_min)
        # Adaptive lambda scaling – projection anchor
        self._router_proj_anchor_adaptive_enable = router_proj_anchor_adaptive_enable
        self._router_proj_anchor_adaptive_beta = router_proj_anchor_adaptive_beta
        self._router_proj_anchor_adaptive_update_every = router_proj_anchor_adaptive_update_every
        self._router_proj_anchor_adaptive_target_ratio_phase1 = router_proj_anchor_adaptive_target_ratio_phase1
        self._router_proj_anchor_adaptive_target_ratio_phase2 = router_proj_anchor_adaptive_target_ratio_phase2
        self._router_proj_anchor_lambda_min = float(router_proj_anchor_lambda_min)
        self._router_proj_anchor_lambda_max = float(router_proj_anchor_lambda_max)
        self._router_proj_anchor_adaptive_reg_ema_min = float(router_proj_anchor_adaptive_reg_ema_min)
        # EMA state for adaptive lambda controller
        self._ema_router_loss: float = 0.0
        self._ema_emb_anchor_loss: float = 0.0
        self._ema_proj_anchor_loss: float = 0.0

        # Router EWC regularizer
        self._router_ewc_regularizer = router_ewc_regularizer
        self._router_ewc_enable = router_ewc_enable
        self._router_ewc_apply_phase = router_ewc_apply_phase
        # Router freeze LM option
        self._router_freeze_lm = router_freeze_lm
        self._exp1_preservation_applied = False  # Track if freezing has been applied
        # Card-guided cold-start initialisation
        self._card_guided_init_enable = card_guided_init_enable
        self._card_init_tau = card_init_tau
        self._card_init_topk = card_init_topk
        self._card_init_scope = card_init_scope
        self._card_init_min_sim_threshold = card_init_min_sim_threshold
        self._card_init_fallback_domain = card_init_fallback_domain
        self._card_encoder = None          # Lazy-loaded SentenceTransformer (shared across calls)
        self._card_embedding_cache: Optional[Dict[str, Any]] = None  # Per-expansion cache
        self._exp1_preservation_hooks = []  # Store gradient hook handles for cleanup
        # Store base registry path for saving to router_config.json (for evaluation diagnostics)
        self._router_registry_base_path = router_registry_base_path
        
        # Phase tracking
        self._phase1_steps = None  # Will be computed once during training setup
        self._current_phase = None  # Will be set in compute_loss
        self._phase_transition_logged = False  # Track if we've logged phase transitions
        self._experience_start_global_step = None  # Track when this experience started (for per-experience phase switching)
        self._lm_param_groups_original_lr = None  # Store original LM param group LRs for restoration
        self._lm_param_requires_grad_original = None  # Snapshot of LM requires_grad before Phase 1 freeze
        
        # Store original config values
        self._original_loss_mode = loss_mode
        self._original_router_loss_weight = router_loss_weight
        self._original_lm_loss_weight = lm_loss_weight
        self._original_router_proj_lr = self._router_proj_lr
        self._original_router_embedding_lr = self._router_embedding_lr
        self._original_router_use_soft_targets = router_use_soft_targets
        self._original_router_soft_target_eps = router_soft_target_eps
        
        # Track if LM params are frozen
        self._lm_params_frozen = False
        self._lm_param_groups_original = None  # Store original optimizer param groups
        
        # Router metrics tracking
        self._router_accum = _LossAccum()
        self._router_graph_accum = _LossAccum()
        self._router_anchor_accum = _LossAccum()
        self._router_proj_anchor_accum = _LossAccum()
        self._router_ewc_accum = _LossAccum()
        
        # Semantic batching validation
        self._semantic_batch_validation_count = 0
        self._semantic_batch_validation_max = 10  # Validate first 10 batches
        self._frozen_lm_eval_mode_logged = False
        self._phase_controller = RouterPhaseController(self)
        
        if loss_mode in _ROUTER_MODES:
            print(f"\n[NeighborConsistencySFTTrainer] Router training mode initialized:")
            print(f"  loss_mode: {loss_mode}")
            print(f"  router_loss_weight: {router_loss_weight}")
            print(f"  lm_loss_weight: {lm_loss_weight}")
            print(f"  tau: {router_tau}")
            print(f"  pooling: {router_pooling}")
            print(f"  K_total: {router_K_total} (K_semantic={router_K_semantic}, K_far={router_K_far}, K_hard={router_K_hard})")
            print(f"  mine_every_steps: {router_mine_every_steps}")
            print(f"  use_soft_targets: {router_use_soft_targets}")
            if router_use_soft_targets:
                print(f"  soft_target_eps: {router_soft_target_eps}")
                print(f"  soft_target_k_neighbors: {router_soft_target_k_neighbors}")
            print(f"  semantic_batching: {semantic_batching}")
            if semantic_batching:
                print(f"    domains_per_batch: {domains_per_batch}")
                print(f"    mix_replay: {mix_replay_in_semantic_batches}")
            if loss_mode in ["router+graph", "supervised+router+graph"]:
                print(f"  label_graph_lambda: {router_label_graph_lambda}")
                print(f"  label_graph_alpha_domain: {router_label_graph_alpha_domain}")
            
            bootstrap_router_state(
                self,
                model=model,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                router_registry_init_mode=router_registry_init_mode,
                router_registry_base_path=router_registry_base_path,
                router_registry_path=router_registry_path,
                router_embedding_dim=router_embedding_dim,
                router_tau=router_tau,
                router_pooling=router_pooling,
            )
            

            
            # Initialize candidate builder
            self._router_candidate_builder = CandidateSetBuilder(
                registry=self._router_registry,
                K_total=router_K_total,
                K_semantic=router_K_semantic,
                K_far=router_K_far,
                K_hard=router_K_hard,
                use_legacy_batching=router_use_legacy_batching,
                semantic_pool_mode=router_semantic_pool_mode,
                semantic_pool_max_domains=router_semantic_pool_max_domains,
                semantic_pool_depth=router_semantic_pool_depth,
            )
            
            # Initialize hard negative miner
            self._router_hard_miner = HardNegativeMiner(
                registry=self._router_registry,
                K_hard_pool=router_K_hard_pool,
                semantic_pool_size=router_semantic_pool_size,
                max_pool_size=router_max_pool_size,
                semantic_pool_mode=router_semantic_pool_mode,
                semantic_pool_max_domains=router_semantic_pool_max_domains,
                semantic_pool_depth=router_semantic_pool_depth,
            )
            
            # Enable hidden states output for prompt embedding extraction
            enable_hidden_states(model)
            
            # If the LM is frozen (router-only training), force it to eval mode
            self._ensure_frozen_lm_in_eval_mode()
    
    @property
    def _global_step(self) -> int:
        """Current global optimiser step (0 before training starts)."""
        return self.state.global_step if self.state is not None else 0

    @property
    def _base_lm(self) -> nn.Module:
        """The underlying LM (unwrapped from PEFT/composite wrappers)."""
        if hasattr(self.model, 'base_model'):
            return self.model.base_model
        if hasattr(self.model, 'model'):
            return self.model.model
        return self.model

    @staticmethod
    def _normalize_registry_path(path: str) -> str:
        """If path is a directory, append model_registry.json."""
        return os.path.join(path, "model_registry.json") if os.path.isdir(path) else path

    @staticmethod
    def _find_latest_checkpoint(directory: Path) -> Optional[Path]:
        """Return the checkpoint-N subdirectory with the highest N, or None."""
        candidates = [
            d for d in directory.iterdir()
            if d.is_dir() and d.name.startswith("checkpoint-")
        ]
        if not candidates:
            return None
        return max(
            candidates,
            key=lambda d: int(d.name.split("-")[1]) if d.name.split("-")[1].isdigit() else -1,
        )

    @staticmethod
    def _disable_dropout(module: nn.Module) -> None:
        """Recursively set all Dropout sub-modules to eval mode."""
        for m in module.modules():
            if isinstance(m, nn.Dropout):
                m.eval()
            elif hasattr(m, 'dropout') and isinstance(m.dropout, nn.Dropout):
                m.dropout.eval()
            elif hasattr(m, 'attention_dropout') and isinstance(m.attention_dropout, nn.Dropout):
                m.attention_dropout.eval()

    def _ensure_frozen_lm_in_eval_mode(self):
        """
        If the base LM is frozen (all parameters have requires_grad=False),
        force it to eval mode to disable dropout and ensure deterministic behavior.
        
        This is critical for router-only training modes where the LM is just used
        as a feature extractor and should not be updated.
        
        NOTE: Only applies to "router" and "router+graph" modes. For "supervised+router"
        and "supervised+router+graph", the LM should be trained, so it stays in train mode.
        """
        # Only apply this logic for pure router modes (not supervised+router modes)
        if self._loss_mode not in _PURE_ROUTER_MODES:
            return
        
        # Get the base model (could be model, model.model, or model.base_model)
        base_model = self._base_lm
        
        base_params = list(base_model.parameters())
        if not base_params:
            return
        
        all_frozen = all(not p.requires_grad for p in base_params)
        
        if all_frozen:
            base_model.eval()
            if not self._frozen_lm_eval_mode_logged:
                self._frozen_lm_eval_mode_logged = True
            
            self._disable_dropout(base_model)

            if self._router_model is not None:
                self._router_model.train()
        else:
            trainable_base_params = [p for p in base_params if p.requires_grad]
            router_params = list(self._router_model.parameters()) if self._router_model is not None else []
            trainable_router_params = [p for p in router_params if p.requires_grad] if router_params else []
            
            if trainable_router_params and not trainable_base_params:
                base_model.eval()

                self._disable_dropout(base_model)

                # Ensure router is in train mode
                if self._router_model is not None:
                    self._router_model.train()

    def create_optimizer(self):
        """
        Override to include X-CLR projection head and router parameters in the optimizer.
        """
        # Call parent to create the base optimizer
        super().create_optimizer()
        
        # Add router parameters to optimizer if router mode is enabled
        if self._loss_mode in _ROUTER_MODES and self._router_model is not None:
            # Split router parameters into projection head and embedding table
            proj_params = list(self._router_model.prompt_projection.parameters())
            embedding_params = list(self._router_model.model_embeddings.parameters())
            
            # Add projection head parameters with separate LR
            if proj_params:
                self.optimizer.add_param_group({
                    'params': proj_params,
                    'lr': self._router_proj_lr,
                    'weight_decay': self.args.weight_decay,
                })
                num_proj_params = sum(p.numel() for p in proj_params)
                print(f"[Router] Added projection head parameters to optimizer "
                      f"({num_proj_params} params, lr={self._router_proj_lr})")
            
            # Add embedding table parameters with separate LR
            if embedding_params:
                self.optimizer.add_param_group({
                    'params': embedding_params,
                    'lr': self._router_embedding_lr,
                    'weight_decay': self.args.weight_decay,
                })
                num_embedding_params = sum(p.numel() for p in embedding_params)
                print(f"[Router] Added embedding table parameters to optimizer "
                      f"({num_embedding_params} params, lr={self._router_embedding_lr})")
                

        
        return self.optimizer
    
    def save_router_checkpoint(self, output_dir: str):
        """
        Save router model and registry for evaluation.
        
        Args:
            output_dir: Directory to save router checkpoints
        """
        if self._router_model is None or self._router_registry is None:
            return

        os.makedirs(output_dir, exist_ok=True)

        router_path = os.path.join(output_dir, "router_model.pt")
        torch.save(self._router_model.state_dict(), router_path)
        print(f"✓ Saved router model to {router_path}")

        registry_path = os.path.join(output_dir, "model_registry.json")
        self._router_registry.save(registry_path)

        router_config = {
            "num_models": len(self._router_registry),
            "embedding_dim": self._router_model.embedding_dim,
            "lm_hidden_size": self._router_model.lm_hidden_size,
            "tau": self._router_model.tau,
            "pooling": self._router_model.pooling,
            "K_total": self._router_candidate_builder.K_total,
            "K_semantic": self._router_candidate_builder.K_semantic,
            "K_far": self._router_candidate_builder.K_far,
            "K_hard": self._router_candidate_builder.K_hard,
        }
        if self._router_registry_base_path:
            router_config["router_registry_base_path"] = self._router_registry_base_path
        if self._router_exp1_preservation_M_old is not None:
            router_config["router_exp1_preservation_M_old"] = self._router_exp1_preservation_M_old
        if self._router_anchor_enable and self._router_anchor_M_old is not None:
            router_config["router_anchor_M_old"] = self._router_anchor_M_old
            router_config["router_anchor_mode"] = self._router_anchor_mode
        import json

        config_path = os.path.join(output_dir, "router_config.json")
        with open(config_path, "w") as f:
            json.dump(router_config, f, indent=2)

        if self._router_anchor_enable and self._router_anchor_ref_cpu is not None:
            anchor_ref_path = os.path.join(output_dir, "router_anchor_ref.pt")
            torch.save(self._router_anchor_ref_cpu, anchor_ref_path)

    
    def get_train_dataloader(self) -> DataLoader:
        """
        Override to use custom batch samplers:

        - DomainBatchSampler when router semantic batching is enabled
        """
        if (self._semantic_batching and 
            self._loss_mode in _ROUTER_MODES):
            
            from .samplers import DomainBatchSampler
            
            dataset = self.train_dataset
            batch_size = self.args.per_device_train_batch_size
            
            seed = self.args.seed if self.args.seed is not None else 42
            if hasattr(self, 'state') and hasattr(self.state, 'epoch') and self.state.epoch is not None:
                seed = seed + int(self.state.epoch)
            
            batch_sampler = DomainBatchSampler(
                dataset=dataset,
                batch_size=batch_size,
                domains_per_batch=self._domains_per_batch,
                shuffle=True,
                seed=seed,
                drop_last=self.args.dataloader_drop_last,
            )
            
            sampler_length = len(batch_sampler)
            optimizer_steps_per_epoch = sampler_length // self.args.gradient_accumulation_steps
            total_optimizer_steps = optimizer_steps_per_epoch * self.args.num_train_epochs
            
            if self._router_two_phase_enable:
                self._phase1_steps = int(self._router_phase1_frac * total_optimizer_steps)
                print(f"    Phase 1 (stability warmup): steps 0-{self._phase1_steps-1} ({self._router_phase1_frac*100:.1f}% of training)")
                print(f"    Phase 2 (main training): steps {self._phase1_steps}-{total_optimizer_steps-1} ({(1-self._router_phase1_frac)*100:.1f}% of training)")
            

            dataloader = DataLoader(
                dataset,
                batch_sampler=batch_sampler,
                collate_fn=self.data_collator,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
            )
        
            
            return dataloader
        
        else:
            # Use default parent implementation
            dataloader = super().get_train_dataloader()
            
            if self._router_two_phase_enable and self._phase1_steps is None:
                steps_per_epoch = len(dataloader) // self.args.gradient_accumulation_steps
                total_optimizer_steps = int(steps_per_epoch * self.args.num_train_epochs)
                
                if hasattr(self.args, 'max_steps') and self.args.max_steps is not None and self.args.max_steps > 0:
                    total_optimizer_steps = self.args.max_steps
                
                self._phase1_steps = int(self._router_phase1_frac * total_optimizer_steps)
                print(f"    Phase 1 (stability warmup): {self._phase1_steps} steps ({self._router_phase1_frac*100:.1f}% of training)")
                print(f"    Phase 2 (main training): {total_optimizer_steps - self._phase1_steps} steps ({(1-self._router_phase1_frac)*100:.1f}% of training)")
            
            return dataloader
    

    def _log_router_param_groups(self, step: Optional[int] = None):
        """
        Log router parameter groups with their LRs for diagnostics.
        """
        if self.optimizer is None:
            return
        
        step_str = f" @ step {step}" if step is not None else ""
        print(f"\n  [Router Param Groups{step_str}]:")
        
        router_proj_lr = None
        router_embedding_lr = None
        
        for i, param_group in enumerate(self.optimizer.param_groups):
            name = param_group.get('name', 'unnamed')
            lr = param_group['lr']
            num_params = sum(p.numel() for p in param_group['params'])
            
            is_router = False
            is_proj = False
            is_embedding = False
            
            name_lower = name.lower()
            if 'router' in name_lower:
                is_router = True
                if 'proj' in name_lower and 'embedding' not in name_lower:
                    is_proj = True
                    router_proj_lr = lr
                elif 'embedding' in name_lower or 'model_embedding' in name_lower:
                    is_embedding = True
                    router_embedding_lr = lr
            
            if not is_router and self._router_model is not None:
                router_param_ids = {id(p) for p in self._router_model.parameters()}
                if any(id(p) in router_param_ids for p in param_group['params']):
                    is_router = True
                    if hasattr(self._router_model, 'projection') and any(id(p) in router_param_ids for p in self._router_model.projection.parameters()):
                        is_proj = True
                        router_proj_lr = lr
                    elif hasattr(self._router_model, 'model_embedding') and any(id(p) in router_param_ids for p in self._router_model.model_embedding.parameters()):
                        is_embedding = True
                        router_embedding_lr = lr
            
            group_type = "ROUTER"
            if is_proj:
                group_type = "ROUTER_PROJ"
            elif is_embedding:
                group_type = "ROUTER_EMB"
            elif not is_router:
                group_type = "LM"
            
            print(f"    Group {i}: {name} | LR={lr:.2e} | {num_params:,} params | {group_type}")
        
        if router_proj_lr is not None or router_embedding_lr is not None:
            print(f"  [Router LRs{step_str}]: proj={router_proj_lr:.2e if router_proj_lr is not None else 'N/A'}, "
                  f"embedding={router_embedding_lr:.2e if router_embedding_lr is not None else 'N/A'}")
    
    def lr_scheduler_step(self, scheduler, metric):
        """
        Override to restore router LRs after scheduler step.
        
        The scheduler may modify all param group LRs, including router groups.
        We restore router LRs to their intended values after the scheduler step.
        """
        super().lr_scheduler_step(scheduler, metric)
        self._phase_controller.on_lr_scheduler_step(self._global_step)

    
    def compute_loss(
        self,
        model: nn.Module,
        inputs: Dict[str, torch.Tensor],
        return_outputs: bool = False,
        num_items_in_batch: Optional[int] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Any]]:
        return compute_loss_pipeline(
            self,
            model=model,
            inputs=inputs,
            return_outputs=return_outputs,
            num_items_in_batch=num_items_in_batch,
            router_modes=_ROUTER_MODES,
            pure_router_modes=_PURE_ROUTER_MODES,
        )
    
    
    def _get_model_idx(self, model_name: str) -> Optional[int]:
        """Return registry index for model_name, using normalised fallback lookup."""
        if model_name in self._router_registry.model2idx:
            return self._router_registry.model2idx[model_name]
        normalized = normalize_model_name(model_name)
        for existing_name, idx in self._router_registry.model2idx.items():
            if normalize_model_name(existing_name) == normalized:
                return idx
        return None

    def _update_adaptive_anchor_lambdas(
        self,
        router_loss: float,
        emb_loss: Optional[float],
        proj_loss: Optional[float],
        is_phase1: bool,
        global_step: int,
    ) -> None:
        """Update router_anchor_lambda / router_proj_anchor_lambda via a loss-ratio EMA controller.

        Every ``update_every`` steps the controller sets
            lambda = target_ratio * ema_router / (ema_reg + eps)
        and clamps to [lambda_min, lambda_max].  The EMA is only updated for a
        loss term when that term was actually computed this step (non-None).
        When adaptive mode is disabled the fixed lambdas are left unchanged.
        """
        phase_str = "phase1" if is_phase1 else "phase2"

        # --- Update router EMA (always, when adaptive is enabled for either term) ---
        if self._router_anchor_adaptive_enable:
            b = self._router_anchor_adaptive_beta
            self._ema_router_loss = b * self._ema_router_loss + (1.0 - b) * router_loss
        elif self._router_proj_anchor_adaptive_enable:
            b = self._router_proj_anchor_adaptive_beta
            self._ema_router_loss = b * self._ema_router_loss + (1.0 - b) * router_loss

        # --- Update embedding anchor EMA (only when loss was computed this step) ---
        if self._router_anchor_adaptive_enable and emb_loss is not None:
            b = self._router_anchor_adaptive_beta
            self._ema_emb_anchor_loss = b * self._ema_emb_anchor_loss + (1.0 - b) * emb_loss

        # --- Update projection anchor EMA (only when loss was computed this step) ---
        if self._router_proj_anchor_adaptive_enable and proj_loss is not None:
            b = self._router_proj_anchor_adaptive_beta
            self._ema_proj_anchor_loss = b * self._ema_proj_anchor_loss + (1.0 - b) * proj_loss

        # --- Periodic lambda update: embedding anchor ---
        if self._router_anchor_adaptive_enable:
            n = self._router_anchor_adaptive_update_every
            if global_step > 0 and global_step % n == 0:
                target_ratio = (
                    self._router_anchor_adaptive_target_ratio_phase1 if is_phase1
                    else self._router_anchor_adaptive_target_ratio_phase2
                )
                ema_reg = self._ema_emb_anchor_loss
                if ema_reg >= self._router_anchor_adaptive_reg_ema_min:
                    new_lambda = target_ratio * self._ema_router_loss / (ema_reg + 1e-30)
                    new_lambda = max(self._router_anchor_lambda_min,
                                     min(self._router_anchor_lambda_max, new_lambda))
                    self._router_anchor_lambda = new_lambda
                    print(
                        f"  [Adaptive Emb Anchor @ step {global_step}]"
                        f"  lambda={new_lambda:.3e}"
                        f"  ema_router={self._ema_router_loss:.4e}"
                        f"  ema_emb_reg={ema_reg:.4e}"
                        f"  target_ratio={target_ratio}"
                        f"  phase={phase_str}"
                    )
                else:
                    print(
                        f"  [Adaptive Emb Anchor @ step {global_step}]"
                        f"  skipped (ema_emb_reg={ema_reg:.2e} < reg_ema_min={self._router_anchor_adaptive_reg_ema_min:.2e})"
                        f"  lambda unchanged={self._router_anchor_lambda:.3e}"
                    )

        # --- Periodic lambda update: projection anchor ---
        if self._router_proj_anchor_adaptive_enable:
            n = self._router_proj_anchor_adaptive_update_every
            if global_step > 0 and global_step % n == 0:
                target_ratio = (
                    self._router_proj_anchor_adaptive_target_ratio_phase1 if is_phase1
                    else self._router_proj_anchor_adaptive_target_ratio_phase2
                )
                ema_reg = self._ema_proj_anchor_loss
                if ema_reg >= self._router_proj_anchor_adaptive_reg_ema_min:
                    new_lambda = target_ratio * self._ema_router_loss / (ema_reg + 1e-30)
                    new_lambda = max(self._router_proj_anchor_lambda_min,
                                     min(self._router_proj_anchor_lambda_max, new_lambda))
                    self._router_proj_anchor_lambda = new_lambda
                    print(
                        f"  [Adaptive Proj Anchor @ step {global_step}]"
                        f"  lambda={new_lambda:.3e}"
                        f"  ema_router={self._ema_router_loss:.4e}"
                        f"  ema_proj_reg={ema_reg:.4e}"
                        f"  target_ratio={target_ratio}"
                        f"  phase={phase_str}"
                    )
                else:
                    print(
                        f"  [Adaptive Proj Anchor @ step {global_step}]"
                        f"  skipped (ema_proj_reg={ema_reg:.2e} < reg_ema_min={self._router_proj_anchor_adaptive_reg_ema_min:.2e})"
                        f"  lambda unchanged={self._router_proj_anchor_lambda:.3e}"
                    )

    def _compute_router_anchor_loss(self) -> Optional[torch.Tensor]:
        """
        Compute router embedding anchor loss to preserve exp1 routing.
        
        This loss penalizes drift of old embedding rows (indices < M_old) away from
        a reference snapshot taken immediately after loading/resizing from exp1 checkpoint.
        
        Returns:
            Anchor loss tensor (scalar), or None if computation fails
        """
        anchor_loss, self._router_anchor_ref = _compute_router_anchor_loss_helper(
            router_model=self._router_model,
            router_anchor_ref_cpu=self._router_anchor_ref_cpu,
            router_anchor_ref=self._router_anchor_ref,
            router_anchor_mode=self._router_anchor_mode,
            router_anchor_scope=self._router_anchor_scope,
            router_anchor_M_old=self._router_anchor_M_old,
            router_anchor_candidate_indices=getattr(self, "_router_anchor_candidate_indices", None),
            router_anchor_y_indices=getattr(self, "_router_anchor_y_indices", None),
        )
        return anchor_loss
    
    def _compute_router_proj_anchor_loss(self) -> Optional[torch.Tensor]:
        """
        Compute router projection anchor loss to preserve exp1 projection.
        
        This loss penalizes drift of projection weights away from a reference
        snapshot taken immediately after loading/resizing from exp1 checkpoint.
        
        CRITICAL: Uses named_parameters() (not state_dict()) to ensure gradients
        flow through the computation graph.
        
        Returns:
            Projection anchor loss tensor (scalar), or None if computation fails
        """
        proj_anchor_loss, self._router_proj_anchor_ref = _compute_router_proj_anchor_loss_helper(
            router_model=self._router_model,
            router_proj_anchor_ref_cpu=self._router_proj_anchor_ref_cpu,
            router_proj_anchor_ref=self._router_proj_anchor_ref,
        )
        return proj_anchor_loss
    
    def compute_router_fisher(
        self,
        dataloader: torch.utils.data.DataLoader,
        num_samples: Optional[int] = None,
    ):
        """
        Compute the Fisher Information Matrix for all router parameters using the
        router (contrastive/InfoNCE) loss.

        Called by main_carve.py after each experience. Returns (fisher_dict,
        optimal_params) which the caller then passes to
        RouterEWCRegularizer.add_experience().

        Args:
            dataloader: DataLoader built from RouterDataCollator batches
                        (must contain 'model_name', 'domain', 'prompt_len' metadata).
            num_samples: Cap on samples used. None = use all data.

        Returns:
            fisher_dict: dict mapping router param name → Fisher tensor (CPU).
            optimal_params: dict mapping router param name → current value (CPU).
        """
        return compute_router_fisher_information(
            lm_model=self.model,
            router_model=self._router_model,
            dataloader=dataloader,
            routing_loss_fn=lambda m, inp, out: _compute_routing_loss_for_batch_impl(self, m, inp, out),
            enable_hidden_states_fn=enable_hidden_states,
            num_samples=num_samples,
        )

    def _compute_router_graph_regularizer_for_batch(
        self,
        inputs: Dict[str, torch.Tensor],
    ) -> Optional[torch.Tensor]:
        """
        Compute label-side graph regularizer for router embeddings.

        Aligns learned model embeddings with taxonomy structure.
        """
        if self._router_model is None or self._router_registry is None:
            return None
        
        
        batch_size = inputs["input_ids"].shape[0]
        device = inputs["input_ids"].device
        
        # Extract metadata
        if "model_name" not in inputs or "domain" not in inputs:
            return None
        
        model_names = inputs["model_name"]
        domains = inputs["domain"]
        
        # Convert to lists
        if not isinstance(model_names, list):
            model_names = list(model_names) if hasattr(model_names, '__iter__') else [str(model_names)] * batch_size
        if not isinstance(domains, list):
            domains = list(domains) if hasattr(domains, '__iter__') else [str(domains)] * batch_size
        
        y_indices = []
        for model_name in model_names:
            model_idx = self._get_model_idx(model_name)
            if model_idx is not None:
                y_indices.append(model_idx)
        
        if not y_indices:
            return None
        
        # Build candidates (simplified - just use model indices from batch)
        candidate_indices = torch.tensor([y_indices], dtype=torch.long, device=device)  # [1, B]
        
        loss = compute_label_graph_regularizer(
            router_model=self._router_model,
            candidate_indices=candidate_indices,
            registry=self._router_registry,
            tau=self._router_label_graph_tau,
            tau_target=self._router_label_graph_tau_target,
            alpha_domain=self._router_label_graph_alpha_domain,
            max_models=self._router_label_graph_max_models,
            device=device,
        )
        
        return loss
    
    def get_router_metrics(self) -> Dict[str, float]:
        """Get average router metrics for logging."""
        metrics = {}
        if self._router_accum.count > 0:
            avg_router = self._router_accum.total / self._router_accum.count
            avg_supervised = self._supervised_accum.total / max(1, self._supervised_accum.count)
            metrics["avg_router_loss"] = avg_router
            metrics["avg_supervised_loss"] = avg_supervised
            if avg_supervised > 0:
                metrics["router_to_supervised_ratio"] = avg_router / avg_supervised

        if self._router_graph_accum.count > 0:
            avg_graph = self._router_graph_accum.total / self._router_graph_accum.count
            metrics["avg_graph_loss"] = avg_graph

        if self._router_ewc_accum.count > 0:
            avg_ewc = self._router_ewc_accum.total / self._router_ewc_accum.count
            metrics["avg_router_ewc_loss"] = avg_ewc

        if hasattr(self, "_router_accuracy_sum") and self._router_accuracy_count > 0:
            for key, value_sum in self._router_accuracy_sum.items():
                avg_value = value_sum / self._router_accuracy_count
                metrics[f"avg_{key}"] = avg_value

        if self._router_hard_miner:
            stats = self._router_hard_miner.get_stats()
            metrics.update(
                {
                    "hard_mining_updates": stats["num_updates"],
                    "hard_mining_examples_processed": stats["num_examples_processed"],
                    "hard_mining_cache_size": stats["cache_size"],
                }
            )

        return metrics
    
    
    def reset_consistency_metrics(self):
        """Reset tracking metrics (call at epoch start)."""
        self._consistency_accum.reset()
        self._supervised_accum.reset()
        self._contrastive_accum.reset()
        self._router_accum.reset()
        self._router_graph_accum.reset()
        self._router_anchor_accum.reset()
        self._router_proj_anchor_accum.reset()
        self._router_ewc_accum.reset()
        self._contrastive_anchors_used = 0
        self._contrastive_negatives_used = 0
        if hasattr(self, "_router_accuracy_sum"):
            for key in self._router_accuracy_sum.keys():
                self._router_accuracy_sum[key] = 0.0
            self._router_accuracy_count = 0
        self._neighbor_domain_stats = defaultdict(lambda: {"same": 0, "different": 0})
        