import random
import numpy as np
import torch
from transformers import TrainerCallback
import gc

def set_seed(seed: int):
    """Set seed for reproducibility across all libraries"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # For deterministic behavior (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to: {seed}")


class MemoryCleanupCallback(TrainerCallback):
    """Callback to clean up memory between epochs."""
    
    def on_epoch_end(self, args, state, control, **kwargs):
        """Clear cache at the end of each epoch."""
        torch.cuda.empty_cache()
        gc.collect()
        
    def on_evaluate(self, args, state, control, **kwargs):
        """Clear cache before evaluation."""
        torch.cuda.empty_cache()
        gc.collect()