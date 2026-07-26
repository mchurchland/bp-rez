"""Small reservoir-construction helpers shared by both research tracks."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


def spectral_radius(matrix: torch.Tensor) -> float:
    """Return the largest absolute eigenvalue of a square matrix."""

    values = np.linalg.eigvals(matrix.detach().cpu().numpy().astype(np.float64))
    return float(np.max(np.abs(values)))


def make_recurrent_matrix(
    size: int,
    radius: float,
    density: float,
    generator: torch.Generator,
) -> torch.Tensor:
    """Create a sparse random matrix with an exact requested spectral radius."""

    if size < 1:
        raise ValueError("reservoir size must be positive")
    if not 0.0 < density <= 1.0:
        raise ValueError("reservoir density must be in (0, 1]")
    if radius < 0.0:
        raise ValueError("spectral radius must be nonnegative")
    for _ in range(100):
        matrix = 2.0 * torch.rand((size, size), generator=generator) - 1.0
        matrix *= torch.rand((size, size), generator=generator) < density
        observed = spectral_radius(matrix)
        if observed >= 1e-12:
            return (matrix * (radius / observed)).to(torch.float32)
    raise RuntimeError(
        "could not sample a recurrent matrix with nonzero spectral radius "
        "after 100 deterministic attempts"
    )


def make_projection(
    out_features: int,
    in_features: int,
    scale: float,
    generator: torch.Generator,
) -> torch.Tensor:
    """Create a fixed, fan-in-scaled uniform random projection."""

    if out_features < 1 or in_features < 1:
        raise ValueError("projection dimensions must be positive")
    bound = scale / np.sqrt(in_features)
    return torch.empty(out_features, in_features).uniform_(
        -bound, bound, generator=generator
    )


def count_trainable_parameters(model: nn.Module) -> int:
    """Count gradient-trainable parameters."""

    return sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )


def count_all_tensor_scalars(model: nn.Module) -> int:
    """Count all unique parameter and persistent-buffer scalar values."""

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    buffer_count = sum(buffer.numel() for buffer in model.buffers())
    return parameter_count + buffer_count
