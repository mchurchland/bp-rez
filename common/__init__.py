"""Utilities shared by the independent NARMA and solar experiments."""

from .reservoir import (
    count_all_tensor_scalars,
    count_trainable_parameters,
    make_projection,
    make_recurrent_matrix,
)
from .runtime import resolve_device, seed_everything

__all__ = [
    "count_all_tensor_scalars",
    "count_trainable_parameters",
    "make_projection",
    "make_recurrent_matrix",
    "resolve_device",
    "seed_everything",
]
