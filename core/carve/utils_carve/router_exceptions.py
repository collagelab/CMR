"""
Custom exceptions for router training.

This module defines domain-specific exceptions for better error handling
and debugging in router training code.
"""


class RouterTrainingError(Exception):
    """Base exception for router training errors."""
    pass


class CandidateValidationError(RouterTrainingError):
    """Raised when candidate set validation fails."""
    pass


class ModelRegistryError(RouterTrainingError):
    """Raised when model registry operations fail."""
    pass


class LabelAlignmentError(RouterTrainingError):
    """Raised when label-candidate alignment check fails."""
    pass


class HardNegativeMiningError(RouterTrainingError):
    """Raised when hard negative mining fails."""
    pass

