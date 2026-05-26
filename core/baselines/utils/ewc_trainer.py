"""
Custom SFTTrainer with EWC (Elastic Weight Consolidation) support.
"""

import torch
from trl import SFTTrainer
from typing import Optional, Dict, Any
from .ewc_utils import EWCRegularizer


class EWCSFTTrainer(SFTTrainer):
    """
    Extended SFTTrainer that adds EWC regularization loss during training.
    """
    
    def __init__(
        self,
        ewc_regularizer: Optional[EWCRegularizer] = None,
        *args,
        **kwargs
    ):
        """
        Initialize EWC-aware trainer.
        
        Args:
            ewc_regularizer: EWC regularizer instance (None = no EWC)
            *args, **kwargs: Arguments passed to SFTTrainer
        """
        super().__init__(*args, **kwargs)
        self.ewc_regularizer = ewc_regularizer
        self._last_ewc_loss: Optional[float] = None
        self._last_task_loss: Optional[float] = None
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Compute loss with EWC regularization.
        
        The total loss is: L_task + L_EWC
        where L_task is the standard task loss and L_EWC is the EWC penalty.
        """
        # Compute standard SFT task loss via super(), which handles completion_only_loss,
        # num_items_in_batch averaging, and any other SFTTrainer-specific logic.
        if return_outputs:
            loss, outputs = super().compute_loss(model, inputs, return_outputs=True, **kwargs)
        else:
            loss = super().compute_loss(model, inputs, return_outputs=False, **kwargs)
        
        # Add EWC regularization if available
        fisher = False
        if len(self.ewc_regularizer.fisher_dict_list) > 0 or len(self.ewc_regularizer.consolidated_fisher) > 0:
            fisher = True

        if self.ewc_regularizer is not None and fisher:
            ewc_loss = self.ewc_regularizer.compute_ewc_loss(model)
            # Store for injection into the next standard trainer log entry
            self._last_ewc_loss = ewc_loss.item()
            self._last_task_loss = loss.item()
            loss = loss + ewc_loss
        
        return (loss, outputs) if return_outputs else loss

    def log(self, logs: Dict[str, Any], *args, **kwargs):
        """
        Inject EWC and task loss into the standard trainer log so they appear
        in the same entry as 'loss', 'grad_norm', etc., without extra repetition.
        """
        if self._last_ewc_loss is not None and "loss" in logs:
            logs["ewc_loss"] = round(self._last_ewc_loss, 6)
            logs["task_loss"] = round(self._last_task_loss, 6)
            self._last_ewc_loss = None
            self._last_task_loss = None
        super().log(logs, *args, **kwargs)
