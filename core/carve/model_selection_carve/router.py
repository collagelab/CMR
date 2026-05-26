"""
Router Model for prompt → model selection.

The router consists of:
1. Prompt encoder: Pools over prompt tokens (labels == -100) from LM hidden states
2. Model embeddings: Learned ID embeddings for each model
3. Scoring: Dot product between prompt and model embeddings, scaled by temperature

CRITICAL: Router parameters must be included in optimizer to train.
This is handled in the trainer by creating a composite model wrapper.
"""

from typing import Optional, Literal, Dict, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils_carve.router_constants import (
    DEFAULT_ROUTER_TAU,
    POOLING_LAST_TOKEN,
    POOLING_MEAN,
)
from ..utils_carve.router_exceptions import RouterTrainingError, ModelRegistryError


class RouterModel(nn.Module):
    """
    Neural router for model selection.
    
    Architecture:
    - encode_prompt: Pools over prompt region of hidden states
    - model_embeddings: Learned embedding table [num_models, embedding_dim]
    - forward: Computes logits = (prompt @ model_embeddings) / tau
    
    Usage:
        router = RouterModel(
            num_models=1000,
            embedding_dim=4096,
            tau=0.07,
            pooling='last_token'
        )
        
        # In training loop:
        logits = router(
            hidden_states=lm_hidden_states,  # [B, seq_len, D]
            prompt_mask=prompt_mask,          # [B, seq_len]
            candidate_indices=candidates      # [B, K]
        )
    
    """
    
    def __init__(
        self,
        num_models: int,
        embedding_dim: int,
        lm_hidden_size: int = 4096,
        tau: float = DEFAULT_ROUTER_TAU,
        pooling: Literal['last_token', 'mean'] = POOLING_LAST_TOKEN,
    ):
        """
        Initialize router model.
        
        Args:
            num_models: Number of unique models in registry
            embedding_dim: Dimension of model embeddings (router space)
            lm_hidden_size: Dimension of LM hidden states (will be projected to embedding_dim)
            tau: Temperature for scaling logits
            pooling: Pooling strategy over prompt tokens
        """
        super().__init__()
        
        # Validate inputs
        if num_models <= 0:
            raise ValueError(f"num_models must be positive, got {num_models}")
        if embedding_dim <= 0:
            raise ValueError(f"embedding_dim must be positive, got {embedding_dim}")
        if lm_hidden_size <= 0:
            raise ValueError(f"lm_hidden_size must be positive, got {lm_hidden_size}")
        if tau <= 0:
            raise ValueError(f"tau (temperature) must be positive, got {tau}")
        if pooling not in [POOLING_LAST_TOKEN, POOLING_MEAN]:
            raise ValueError(f"pooling must be '{POOLING_LAST_TOKEN}' or '{POOLING_MEAN}', got '{pooling}'")
        
        self.num_models = num_models
        self.embedding_dim = embedding_dim
        self.lm_hidden_size = lm_hidden_size
        self.tau = tau
        self.pooling = pooling
        self.prompt_projection = nn.Linear(lm_hidden_size, embedding_dim, bias=False)
        self.model_embeddings = nn.Embedding(num_models, embedding_dim)
        nn.init.xavier_uniform_(self.model_embeddings.weight)
        nn.init.xavier_uniform_(self.prompt_projection.weight)
    
    def encode_prompt(
        self,
        hidden_states: torch.Tensor,
        prompt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode prompt by pooling over prompt tokens.
        
        Args:
            hidden_states: LM hidden states [B, seq_len, D]
            prompt_mask: Binary mask for prompt tokens [B, seq_len]
                         1 = prompt token, 0 = completion/padding token
        
        Returns:
            Prompt embeddings [B, D]
        """
        if self.pooling == POOLING_LAST_TOKEN:

            if prompt_mask.dtype == torch.bool:
                prompt_mask_long = prompt_mask.long()
            elif prompt_mask.dtype == torch.long:
                prompt_mask_long = prompt_mask
            else:
                MASK_THRESHOLD = 0.5
                prompt_mask_bool = prompt_mask > MASK_THRESHOLD
                prompt_mask_long = prompt_mask_bool.long()
            
            B, L, D = hidden_states.shape
        
            positions = torch.arange(L, device=hidden_states.device).unsqueeze(0).expand(B, L)  # [B, L]
            masked_positions = prompt_mask_long * positions  # [B, L] - positions where mask==1, 0 elsewhere
            last_idx = masked_positions.max(dim=1).values  # [B] - max position where mask==1
            
            batch_indices = torch.arange(B, device=hidden_states.device)
            mask_at_last = prompt_mask_long[batch_indices, last_idx]  # [B]
            
            if not (mask_at_last == 1).all():
                invalid_count = (mask_at_last == 0).sum().item()
                raise RouterTrainingError(
                    f"{invalid_count} examples have last_idx pointing to invalid positions (mask==0). "
                    f"This indicates a bug in last_idx computation."
                )
            
            q_lm = hidden_states[batch_indices, last_idx]  # [B, lm_hidden_size]
        
        elif self.pooling == POOLING_MEAN:

            mask_float = prompt_mask.float()
            
            mask_expanded = mask_float.unsqueeze(-1)  # [B, seq_len, 1]
            
            masked_hidden = hidden_states * mask_expanded  # [B, seq_len, lm_hidden_size]
            sum_hidden = masked_hidden.sum(dim=1)  # [B, lm_hidden_size]
            
            MIN_TOKENS = 1.0
            num_tokens_raw = mask_float.sum(dim=1, keepdim=True)  # [B, 1]
            
            num_tokens = num_tokens_raw.clamp(min=MIN_TOKENS)  # [B, 1]
            
            q_lm = sum_hidden / num_tokens  # [B, lm_hidden_size]
            
        else:
            raise ValueError(
                f"Unknown pooling strategy: '{self.pooling}'. "
                f"Must be '{POOLING_LAST_TOKEN}' or '{POOLING_MEAN}'."
            )
    
        q = self.prompt_projection(q_lm)  # [B, embedding_dim]
        
        return q
    
    def _calculate_flops(
        self,
        batch_size: int,
        seq_len: int,
        K: int,
        prompt_mask: torch.Tensor,
    ) -> Dict[str, int]:
        """
        Calculate FLOPs for router forward pass.
        
        Args:
            batch_size: Batch size B
            seq_len: Sequence length
            K: Number of candidate models
            prompt_mask: Prompt mask [B, seq_len] for computing prompt token count
        
        Returns:
            Dictionary with FLOPs breakdown
        """
        # Count prompt tokens per example (for mean pooling FLOPs)
        if self.pooling == POOLING_MEAN:
            prompt_token_counts = prompt_mask.float().sum(dim=1)  # [B]
            avg_prompt_tokens = prompt_token_counts.mean().item()
        else:   
            avg_prompt_tokens = 1.0
        
        if self.pooling == POOLING_MEAN:
            pooling_flops = int(batch_size * seq_len * self.lm_hidden_size)
        else:
            pooling_flops = int(batch_size * self.lm_hidden_size)
        
        #    - Linear projection: B * lm_hidden_size * embedding_dim
        projection_flops = batch_size * self.lm_hidden_size * self.embedding_dim
        
        prompt_encoding_flops = pooling_flops + projection_flops
        
        # 2. Embedding lookup: O(1) per candidate (just indexing), negligible
        embedding_lookup_flops = 0
        
        # 3. Normalization FLOPs
        #    - Prompt normalization: B * embedding_dim (L2 norm)
        prompt_norm_flops = batch_size * self.embedding_dim
        candidate_norm_flops = batch_size * K * self.embedding_dim
        normalization_flops = prompt_norm_flops + candidate_norm_flops
        
        # 4. Batch matrix multiplication: B * embedding_dim * K
        bmm_flops = batch_size * self.embedding_dim * K
        
        # 5. Temperature scaling: B * K (negligible, but count it)
        temp_scaling_flops = batch_size * K
        
        total_flops = (
            prompt_encoding_flops +
            embedding_lookup_flops +
            normalization_flops +
            bmm_flops +
            temp_scaling_flops
        )
        
        return {
            "total_flops": total_flops,
            "prompt_encoding_flops": prompt_encoding_flops,
            "pooling_flops": pooling_flops,
            "projection_flops": projection_flops,
            "normalization_flops": normalization_flops,
            "bmm_flops": bmm_flops,
            "temp_scaling_flops": temp_scaling_flops,
            "batch_size": batch_size,
            "seq_len": seq_len,
            "K": K,
            "avg_prompt_tokens": avg_prompt_tokens,
        }
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        prompt_mask: torch.Tensor,
        candidate_indices: torch.Tensor,
        print_pipeline_flags: bool = False,
        return_compute_metrics: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, float]]]:
        """
        Compute routing logits for candidate sets.
        
        Args:
            hidden_states: LM hidden states [B, seq_len, D]
            prompt_mask: Binary mask for prompt tokens [B, seq_len]
            candidate_indices: Candidate model indices [B, K]
            print_pipeline_flags: Whether to print pipeline parity flags (for eval diagnostics)
            return_compute_metrics: If True, also return compute metrics (FLOPs, etc.)
        
        Returns:
            If return_compute_metrics=False: Logits [B, K]
            If return_compute_metrics=True: (Logits [B, K], compute_metrics dict)
        """
        # Pipeline parity flags (for eval diagnostics)
        if print_pipeline_flags:
            print(f"\n[Router Pipeline Flags]")
            print(f"  normalize_prompt_emb: True (L2 normalization)")
            print(f"  normalize_model_emb: True (L2 normalization)")
            print(f"  temperature_scaling: True (tau={self.tau})")
            print(f"  prompt_projection: True (Linear({self.lm_hidden_size} -> {self.embedding_dim}))")
            print(f"  pooling: {self.pooling}")
        
        B, seq_len, _ = hidden_states.shape
        K = candidate_indices.shape[1]
        
        q = self.encode_prompt(hidden_states, prompt_mask)  # [B, D]
        
        M = self.model_embeddings(candidate_indices)  # [B, K, D]
        
        q = F.normalize(q, p=2, dim=-1)  # [B, D] with ||q|| = 1
        M = F.normalize(M, p=2, dim=-1)  # [B, K, D] with ||M[i,j]|| = 1
        
        logits = torch.bmm(q.unsqueeze(1), M.transpose(1, 2)).squeeze(1)  # [B, K]
        
        logits = logits / self.tau
        
        compute_metrics = None
        if return_compute_metrics:
            flops_dict = self._calculate_flops(B, seq_len, K, prompt_mask)
            compute_metrics = {
                "flops": float(flops_dict["total_flops"]),
                "flops_per_example": float(flops_dict["total_flops"]) / B,
                "prompt_encoding_flops": float(flops_dict["prompt_encoding_flops"]),
                "normalization_flops": float(flops_dict["normalization_flops"]),
                "bmm_flops": float(flops_dict["bmm_flops"]),
                "batch_size": B,
                "seq_len": seq_len,
                "K": K,
                "avg_prompt_tokens": flops_dict["avg_prompt_tokens"],
            }
            return logits, compute_metrics
        
        return logits
    
    def get_model_embedding(self, model_idx: int) -> torch.Tensor:
        """
        Get embedding for a single model.
        
        Args:
            model_idx: Model index
        
        Returns:
            Model embedding [D]
        
        Raises:
            IndexError: If model_idx is out of range
        """
        if model_idx < 0 or model_idx >= self.num_models:
            raise IndexError(
                f"model_idx {model_idx} is out of range [0, {self.num_models}). "
                f"Valid model indices are 0 to {self.num_models - 1}."
            )
        return self.model_embeddings.weight[model_idx]
    
    def get_model_embeddings(self, model_indices: torch.Tensor) -> torch.Tensor:
        """
        Get embeddings for multiple models.
        
        Args:
            model_indices: Model indices [N] or [B, K]
        
        Returns:
            Model embeddings [N, D] or [B, K, D]
        """
        return self.model_embeddings(model_indices)
    
    def score_all(
        self,
        hidden_states: torch.Tensor,
        prompt_mask: torch.Tensor,
        return_compute_metrics: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, float]]]:
        """
        Score all models using the same internal pipeline as forward().
        
        This is equivalent to calling forward() with all models as candidates,
        but provides a cleaner API for evaluation.
        
        Args:
            hidden_states: LM hidden states [B, seq_len, D]
            prompt_mask: Binary mask for prompt tokens [B, seq_len]
            return_compute_metrics: If True, also return compute metrics (FLOPs, etc.)
        
        Returns:
            If return_compute_metrics=False: Logits [B, num_models]
            If return_compute_metrics=True: (Logits [B, num_models], compute_metrics dict)
        """
        B = hidden_states.shape[0]
        num_models = self.num_models
        device = hidden_states.device
        
        candidate_indices = torch.arange(num_models, device=device).unsqueeze(0).expand(B, -1)  # [B, num_models]
        
        return self.forward(
            hidden_states=hidden_states,
            prompt_mask=prompt_mask,
            candidate_indices=candidate_indices,
            print_pipeline_flags=False,
            return_compute_metrics=return_compute_metrics,
        )  # [B, num_models]


class CompositeModelWithRouter(nn.Module):
    """
    Wrapper that combines base LM and router into a single module.
    
    This ensures router parameters are included in model.parameters()
    and will be optimized by the trainer's optimizer.
    
    CRITICAL: This is the recommended way to integrate the router.
    Without this, router.model_embeddings will NOT be updated during training.
    
    Usage:
        # In trainer __init__:
        base_model = ...  # HF model
        router = RouterModel(...)
        
        # Wrap them together
        composite_model = CompositeModelWithRouter(base_model, router)
        
        # Now pass composite_model to SFTTrainer
        # trainer.model.base_model is the LM
        # trainer.model.router is the router
    """
    
    def __init__(self, base_model: nn.Module, router: RouterModel):
        """
        Initialize composite model.
        
        Args:
            base_model: Base language model (e.g., LlamaForCausalLM)
            router: Router model
        """
        super().__init__()
        self.base_model = base_model
        self.router = router
    
    def forward(self, *args, **kwargs):
        """Forward pass delegates to base model."""
        return self.base_model(*args, **kwargs)
    
    def __getattr__(self, name: str):
        """
        Delegate attribute access to base_model for compatibility.
        
        This allows trainer.model.config, trainer.model.generate(), etc.
        to work as expected.
        """
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.base_model, name)


def extract_prompt_mask(
    prompt_len: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: Optional[torch.Tensor] = None,
    debug: bool = False,
    global_step: int = 0,
) -> torch.Tensor:
    """
    Extract prompt mask from prompt_len and attention mask.
    
    CORRECT IMPLEMENTATION: Uses explicit prompt_len boundary, not label inference.
    This matches inference-time behavior where we only have access to prompt tokens.
    
    Args:
        prompt_len: Per-example prompt length [B] - boundary between prompt and completion
        attention_mask: Attention mask [B, seq_len] - 1 for real tokens, 0 for padding
        labels: (Optional) Label tensor [B, seq_len] for verification only
        debug: Whether to print detailed debugging info
        global_step: Current global step (for debug filtering)
    
    Returns:
        Prompt mask [B, seq_len]
        1 = prompt token (position < prompt_len AND attention_mask==1)
        0 = completion token or padding
    """

    B, L = attention_mask.shape
    device = attention_mask.device

    token_pos = attention_mask.long().cumsum(dim=1) - 1          # pads become -1
    prompt_len_expanded = prompt_len.unsqueeze(1)                # [B, 1]

    prompt_mask = (attention_mask == 1) & (token_pos < prompt_len_expanded)
    return prompt_mask.to(torch.float32)
