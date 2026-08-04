"""Streaming multi-output ridge regression for the solar readout."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class RidgeStatistics:
    """Sufficient statistics for a linear model with an intercept."""

    feature_gram: torch.Tensor
    feature_target: torch.Tensor
    feature_sum: torch.Tensor
    target_sum: torch.Tensor
    target_square_sum: torch.Tensor
    rows: int = 0

    @classmethod
    def empty(cls, feature_size: int, output_size: int) -> "RidgeStatistics":
        if feature_size < 1 or output_size < 1:
            raise ValueError("ridge dimensions must be positive")
        return cls(
            feature_gram=torch.zeros((feature_size, feature_size), dtype=torch.float64),
            feature_target=torch.zeros((feature_size, output_size), dtype=torch.float64),
            feature_sum=torch.zeros(feature_size, dtype=torch.float64),
            target_sum=torch.zeros(output_size, dtype=torch.float64),
            target_square_sum=torch.zeros((), dtype=torch.float64),
        )

    @property
    def feature_size(self) -> int:
        return self.feature_gram.shape[0]

    @property
    def output_size(self) -> int:
        return self.feature_target.shape[1]

    def update(self, features: torch.Tensor, target: torch.Tensor) -> None:
        x = features.detach().reshape(-1, self.feature_size).cpu().to(torch.float64)
        y = target.detach().reshape(-1, self.output_size).cpu().to(torch.float64)
        if len(x) != len(y):
            raise ValueError("ridge feature and target row counts differ")
        self.feature_gram += x.T @ x
        self.feature_target += x.T @ y
        self.feature_sum += x.sum(dim=0)
        self.target_sum += y.sum(dim=0)
        self.target_square_sum += y.square().sum()
        self.rows += len(x)

    def augmented_system(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.rows < 1:
            raise ValueError("cannot solve ridge regression without rows")
        gram = torch.empty(
            (self.feature_size + 1, self.feature_size + 1), dtype=torch.float64
        )
        gram[:-1, :-1] = self.feature_gram
        gram[:-1, -1] = self.feature_sum
        gram[-1, :-1] = self.feature_sum
        gram[-1, -1] = self.rows
        cross = torch.cat((self.feature_target, self.target_sum[None, :]), dim=0)
        return gram, cross

    def mse(self, coefficients: torch.Tensor) -> float:
        """Evaluate coefficients shaped ``[features + intercept, outputs]``."""

        gram, cross = self.augmented_system()
        if coefficients.shape != cross.shape:
            raise ValueError("ridge coefficient dimensions do not match statistics")
        squared_error = (
            self.target_square_sum
            - 2.0 * torch.sum(coefficients * cross)
            + torch.sum(coefficients * (gram @ coefficients))
        )
        denominator = self.rows * self.output_size
        return float(torch.clamp(squared_error / denominator, min=0.0))


def solve_ridge(statistics: RidgeStatistics, alpha: float) -> torch.Tensor:
    """Solve mean-scaled ridge while leaving the intercept unpenalized."""

    if alpha < 0.0:
        raise ValueError("ridge alpha must be nonnegative")
    gram, cross = statistics.augmented_system()
    penalty = torch.eye(len(gram), dtype=gram.dtype) * (alpha * statistics.rows)
    penalty[-1, -1] = 0.0
    system = gram + penalty
    try:
        return torch.linalg.solve(system, cross)
    except torch.linalg.LinAlgError:
        return torch.linalg.lstsq(system, cross).solution


def tune_ridge(
    training: RidgeStatistics,
    validation: RidgeStatistics,
    alphas: tuple[float, ...],
) -> tuple[float, float, torch.Tensor, torch.Tensor]:
    """Select alpha on validation and return a training-only fitted readout."""

    if not alphas:
        raise ValueError("at least one ridge alpha is required")
    if any(alpha < 0.0 for alpha in alphas):
        raise ValueError("ridge alphas must be nonnegative")
    best_alpha = -1.0
    best_mse = float("inf")
    best_coefficients = None
    for alpha in alphas:
        coefficients = solve_ridge(training, alpha)
        validation_mse = validation.mse(coefficients)
        if validation_mse < best_mse:
            best_alpha = float(alpha)
            best_mse = validation_mse
            best_coefficients = coefficients
    if best_coefficients is None:
        raise RuntimeError("ridge selection did not produce coefficients")
    weight = best_coefficients[:-1].T.to(torch.float32)
    bias = best_coefficients[-1].to(torch.float32)
    return best_alpha, best_mse, weight, bias
