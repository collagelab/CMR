import wandb
import os
from pathlib import Path
from typing import Union, Optional
from .configs import TrainConfig, EvalConfig

# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent
from transformers import TrainerCallback
from datetime import datetime

class WandbLogger:
    def __init__(self, key: str, config: Union[TrainConfig, EvalConfig], mode: str, wandb_dir: Optional[str] = None):
        """
        Initialize WandB logger.
        
        Args:
            key (str): WandB API key.
            config (TrainConfig): Configuration object containing training parameters.
            mode (str): Mode of operation, either "train" or "eval".
            wandb_dir (str, optional): Custom directory where wandb will save its files.
                                     If None, uses wandb's default location.
        """
        # Set custom wandb directory if provided, otherwise use project root
        if wandb_dir:
            wandb_dir = Path(wandb_dir).resolve()
        else:
            wandb_dir = PROJECT_ROOT / "wandb_logs"

        wandb_dir.mkdir(parents=True, exist_ok=True)
        os.environ["WANDB_DIR"] = str(wandb_dir)
        print(f"WandB files will be saved to: {wandb_dir}")
        
        wandb.login(key=key)
        
        
        
        # Create wandb config dictionary from config dataclass
        config_dict = {field: getattr(config, field) for field in config.__dataclass_fields__}
        
        # Build safe tags: exclude verbose fields and enforce WandB limits (1-64 chars)
        excluded_keys = {"system_prompt", "lora_adapters", "target_modules", "experience_names"}
        
        def sanitize_tag(s: str) -> str:
            # Replace unsafe separators and whitespace
            s = str(s).replace("/", "-").replace("\\", "-").replace(":", "-").replace(",", "-")
            s = " ".join(s.split())  # collapse whitespace
            s = s.replace(" ", "_")
            # WandB requires 1..64 chars; truncate if necessary
            return s[:64] if len(s) > 64 else s
        
        tags = []
        for k, v in config_dict.items():
            if v is None or k in excluded_keys:
                continue
        
        
            tags.append(f"{k}:{v}".replace("/", "-"))  # Replace '/' with '-' in tags
        tags.append(f"mode:{mode}")
        
        # Deduplicate while preserving order
        seen = set()
        tags = [t for t in tags if not (t in seen or seen.add(t))]

        # Additional wandb.init parameters
        init_kwargs = {
            "project": "cmr",
            "config": config_dict,
            "tags": tags,
            "mode": "offline"
        }
        
        # Add directory if specified
        if wandb_dir:
            init_kwargs["dir"] = str(wandb_dir)
        
        wandb.init(**init_kwargs)

    def log(self, metrics, step=None):
        wandb.log(metrics, step=step)

    def finish(self):
        wandb.finish()


class WandbTrainingCallback(TrainerCallback):
    """Custom WandB callback for enhanced training logging."""
    
    def __init__(self, wandb_logger: WandbLogger):
        self.wandb_logger = wandb_logger
    
    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        """Log training metrics to WandB."""
        if logs:
            # Filter out non-numeric values and format logs
            filtered_logs = {}
            for key, value in logs.items():
                if isinstance(value, (int, float)):
                    filtered_logs[key] = value
            
            if filtered_logs:
                # Let WandB handle step incrementing automatically to avoid conflicts
                self.wandb_logger.log(filtered_logs)
    
    def on_evaluate(self, args, state, control, model=None, logs=None, **kwargs):
        """Log evaluation metrics to WandB."""
        if logs:
            eval_logs = {}
            for key, value in logs.items():
                if isinstance(value, (int, float)) and key.startswith('eval_'):
                    eval_logs[key] = value
            
            if eval_logs:
                # Let WandB handle step incrementing automatically to avoid conflicts
                self.wandb_logger.log(eval_logs)