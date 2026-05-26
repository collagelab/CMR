"""
Configuration loader utility for YAML-based configs.
"""
import yaml
from pathlib import Path
from typing import Dict, Any
from .configs import EvalConfig, TrainConfig
from .prepareDataset import dict_retriever


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config_data = yaml.safe_load(f)
    
    return config_data


def create_eval_config_from_yaml(config_path: str, **overrides) -> EvalConfig:
    """
    Create EvalConfig instance from YAML file with optional overrides.
    
    Args:
        config_path: Path to YAML configuration file
        **overrides: Additional parameters to override YAML values
    
    Returns:
        EvalConfig instance
    """
    config_data = load_yaml_config(config_path)
    
    config_data.update(overrides)
    
    valid_params = {}
    for field in EvalConfig.__dataclass_fields__:
        if field in config_data and config_data[field] is not None:
            valid_params[field] = config_data[field]
    
    eval_config = EvalConfig(**valid_params)
    
    if eval_config.retriever is not None:
        retrievers = list(dict_retriever.keys())

        if not any(ret in adapter for adapter in eval_config.lora_adapters for ret in retrievers):
            raise ValueError(
                f"Not valid adapter '{eval_config.lora_adapters}' for retriever '{eval_config.retriever}'.")
    
    if len(eval_config.lora_adapters) > 1:
        eval_config.lora_adapters = sorted(eval_config.lora_adapters)
    
    return eval_config


def create_train_config_from_yaml(config_path: str, **overrides) -> TrainConfig:
    """
    Create TrainConfig instance from YAML file with optional overrides.
    
    Args:
        config_path: Path to YAML configuration file
        **overrides: Additional parameters to override YAML values
    
    Returns:
        TrainConfig instance
    """
    config_data = load_yaml_config(config_path)
    
    config_data.update(overrides)
    
    valid_params = {}
    for field in TrainConfig.__dataclass_fields__:
        if field in config_data and config_data[field] is not None:
            valid_params[field] = config_data[field]
    
    return TrainConfig(**valid_params)