"""Validation-tuned, mean-scaled ridge readouts with an unpenalized intercept."""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class RidgeFit:
    alpha: float
    validation_mse: float
    weight: torch.Tensor
    bias: torch.Tensor
    feature_mean: torch.Tensor
    feature_scale: torch.Tensor
    fit_seconds: float


@dataclass(frozen=True)
class _RidgeSystem:
    singular_values: torch.Tensor
    right_vectors: torch.Tensor
    projected_target: torch.Tensor
    sample_count: int
    feature_mean: torch.Tensor
    feature_scale: torch.Tensor
    target_mean: torch.Tensor


def _prepare_system(
    features: torch.Tensor,
    target: torch.Tensor,
) -> _RidgeSystem:
    """Compute one standardized float64 SVD without forming ``X.T @ X``."""

    x = features.detach().cpu().to(torch.float64)
    y = target.detach().cpu().to(torch.float64).reshape(-1, 1)
    if len(x) != len(y):
        raise ValueError("feature and target lengths differ")
    if len(x) < 2:
        raise ValueError("ridge fitting requires at least two rows")
    mean = x.mean(dim=0)
    scale = x.std(dim=0, unbiased=False).clamp_min(1e-12)
    normalized = (x - mean) / scale
    y_mean = y.mean(dim=0)
    centered_y = y - y_mean
    left, singular_values, right_transpose = torch.linalg.svd(
        normalized, full_matrices=False
    )
    projected_target = left.T @ centered_y
    return _RidgeSystem(
        singular_values=singular_values,
        right_vectors=right_transpose.T,
        projected_target=projected_target,
        sample_count=len(normalized),
        feature_mean=mean,
        feature_scale=scale,
        target_mean=y_mean,
    )


def _solve(system: _RidgeSystem, alpha: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Solve one ridge value and convert back to raw feature coordinates."""

    singular = system.singular_values
    denominator = singular.square() + alpha * system.sample_count
    tolerance = (
        torch.finfo(singular.dtype).eps
        * max(system.sample_count, system.right_vectors.shape[0])
        * singular.max()
    )
    factors = torch.where(
        singular > tolerance,
        singular / denominator.clamp_min(torch.finfo(singular.dtype).tiny),
        torch.zeros_like(singular),
    )
    standardized_weight = system.right_vectors @ (
        factors.unsqueeze(-1) * system.projected_target
    )
    raw_weight = (standardized_weight.squeeze(-1) / system.feature_scale).unsqueeze(0)
    bias = system.target_mean - system.feature_mean @ raw_weight.squeeze(0)
    return raw_weight, bias


def tune_and_refit_ridge(
    train_features: torch.Tensor,
    train_target: torch.Tensor,
    validation_features: torch.Tensor,
    validation_target: torch.Tensor,
    alphas: tuple[float, ...],
) -> RidgeFit:
    """Choose alpha on validation, then refit on train plus validation.

    The final test set is deliberately absent from this interface.
    """

    if not alphas:
        raise ValueError("at least one ridge alpha is required")
    if any(alpha < 0.0 for alpha in alphas):
        raise ValueError("ridge alphas must be nonnegative")
    start = time.perf_counter()
    val_x = validation_features.detach().cpu().to(torch.float64)
    val_y = validation_target.detach().cpu().to(torch.float64).reshape(-1, 1)
    development_system = _prepare_system(train_features, train_target)
    best_alpha = -1.0
    best_validation = float("inf")
    for alpha in alphas:
        weight, bias = _solve(development_system, alpha)
        prediction = val_x @ weight.T + bias
        mse = torch.mean((prediction - val_y) ** 2).item()
        if mse < best_validation:
            best_validation = mse
            best_alpha = float(alpha)
    combined_features = torch.cat(
        (train_features.detach().cpu(), validation_features.detach().cpu())
    )
    combined_target = torch.cat(
        (train_target.detach().cpu(), validation_target.detach().cpu())
    )
    final_system = _prepare_system(combined_features, combined_target)
    weight, bias = _solve(final_system, best_alpha)
    return RidgeFit(
        alpha=best_alpha,
        validation_mse=best_validation,
        weight=weight.to(torch.float32),
        bias=bias.to(torch.float32),
        feature_mean=final_system.feature_mean,
        feature_scale=final_system.feature_scale,
        fit_seconds=time.perf_counter() - start,
    )
