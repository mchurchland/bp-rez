"""Two-layer reservoir computing research prototype."""

from .data import NARMASplit, make_narma10_splits
from .models import MODEL_NAMES, build_model, count_trainable_parameters

__all__ = [
    "MODEL_NAMES",
    "NARMASplit",
    "build_model",
    "count_trainable_parameters",
    "make_narma10_splits",
]
