"""Publication-oriented NARMA reservoir benchmark."""

from .data import NARMASplit, NARMATask, TASKS, make_paired_splits
from .models import MODEL_NAMES, ReservoirConfig, build_model

__all__ = [
    "MODEL_NAMES",
    "NARMASplit",
    "NARMATask",
    "ReservoirConfig",
    "TASKS",
    "build_model",
    "make_paired_splits",
]
