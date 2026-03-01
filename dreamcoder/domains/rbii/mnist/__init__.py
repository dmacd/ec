"""
MNIST-specific RBII prototype implementation.

This package is intentionally isolated from the existing simple RBII character
prototype in sibling files.
"""

from .types import MNISTPrediction, tmnist_pred, tmnist_state
from .state import MNISTState, MNISTStateView
from .loop import MNISTRBIIConfig, MNISTRBIILoop

__all__ = [
    "MNISTPrediction",
    "MNISTRBIIConfig",
    "MNISTRBIILoop",
    "MNISTState",
    "MNISTStateView",
    "tmnist_pred",
    "tmnist_state",
]
