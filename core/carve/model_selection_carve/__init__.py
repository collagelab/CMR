"""
Model selection components for continual learning router.

This package provides:
- ModelRegistry: Centralized model name → ID mapping with metadata
- CandidateSetBuilder: Builds candidate sets for routing loss
- HardNegativeMiner: Mines confusable models for hard negative sampling
- RouterModel: Neural router for model selection
- CompositeModelWithRouter: Wrapper for model + router (ensures optimizer includes router params)
"""

from .model_registry import ModelRegistry, normalize_domain, normalize_model_name
from .candidates import CandidateSetBuilder
from .hard_mining import HardNegativeMiner
from .router import RouterModel, CompositeModelWithRouter

__all__ = [
    "ModelRegistry",
    "CandidateSetBuilder",
    "HardNegativeMiner",
    "RouterModel",
    "CompositeModelWithRouter",
]

