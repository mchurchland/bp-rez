"""Convergent gradient training for learned NARMA encoders and the GRU control."""

from __future__ import annotations

import copy
import math
import time
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn

from .data import NARMASplit
from .models import BenchmarkModel


@dataclass(frozen=True)
class OptimizerConfig:
    learning_rate: float = 1.0e-3
    weight_decay: float = 0.0
    max_epochs: int = 1_000
    early_stopping_patience: int = 100
    min_delta: float = 1.0e-8
    gradient_clip: float = 1.0
    scheduler_patience: int = 20
    scheduler_factor: float = 0.5
    minimum_learning_rate: float = 1.0e-6

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _mse_after_washout(
    prediction: torch.Tensor, target: torch.Tensor, washout: int
) -> torch.Tensor:
    if washout >= len(target):
        raise ValueError("washout must be shorter than the sequence")
    return nn.functional.mse_loss(prediction[washout:], target[washout:])


def train_gradient_model(
    model: BenchmarkModel,
    train: NARMASplit,
    validation: NARMASplit,
    *,
    washout: int,
    optimizer_config: OptimizerConfig,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, list[float]]]:
    """Train, schedule, early-stop and restore the best validation checkpoint."""

    if not model.gradient_model:
        return (
            {
                "best_epoch": 0,
                "epochs_ran": 0,
                "best_validation_mse_joint_head": None,
                "hit_epoch_cap": False,
                "preprocessor_fit_seconds": 0.0,
                "gradient_training_seconds": 0.0,
                "training_seconds": 0.0,
            },
            {
                "train_mse": [],
                "validation_mse": [],
                "gradient_norm_before_clip": [],
                "gradient_norm_after_clip": [],
                "learning_rate": [],
                "gradient_was_clipped": [],
            },
        )
    model.to(device)
    train_u = train.u.to(device)
    train_y = train.y.to(device)
    validation_u = validation.u.to(device)
    validation_y = validation.y.to(device)
    _synchronize(device)
    prepare_started = time.perf_counter()
    model.prepare(train_u, washout)
    _synchronize(device)
    preprocessor_seconds = time.perf_counter() - prepare_started
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=optimizer_config.learning_rate,
        weight_decay=optimizer_config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=optimizer_config.scheduler_factor,
        patience=optimizer_config.scheduler_patience,
        min_lr=optimizer_config.minimum_learning_rate,
    )
    history: dict[str, list[float]] = {
        "train_mse": [],
        "validation_mse": [],
        "gradient_norm_before_clip": [],
        "gradient_norm_after_clip": [],
        "learning_rate": [],
        "gradient_was_clipped": [],
    }
    best_loss = float("inf")
    best_epoch = -1
    best_state: dict[str, torch.Tensor] | None = None
    stale = 0
    _synchronize(device)
    started = time.perf_counter()
    for epoch in range(optimizer_config.max_epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        prediction = model(train_u)
        loss = _mse_after_washout(prediction, train_y, washout)
        loss.backward()
        squared = sum(
            float(torch.sum(parameter.grad.detach() ** 2).cpu())
            for parameter in parameters
            if parameter.grad is not None
        )
        before_clip = math.sqrt(squared)
        if optimizer_config.gradient_clip > 0.0:
            nn.utils.clip_grad_norm_(parameters, optimizer_config.gradient_clip)
        squared_after = sum(
            float(torch.sum(parameter.grad.detach() ** 2).cpu())
            for parameter in parameters
            if parameter.grad is not None
        )
        after_clip = math.sqrt(squared_after)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            validation_prediction = model(validation_u)
            validation_loss = float(
                _mse_after_washout(validation_prediction, validation_y, washout).cpu()
            )
        scheduler.step(validation_loss)
        learning_rate = float(optimizer.param_groups[0]["lr"])
        history["train_mse"].append(float(loss.detach().cpu()))
        history["validation_mse"].append(validation_loss)
        history["gradient_norm_before_clip"].append(before_clip)
        history["gradient_norm_after_clip"].append(after_clip)
        history["learning_rate"].append(learning_rate)
        history["gradient_was_clipped"].append(
            float(
                optimizer_config.gradient_clip > 0.0
                and before_clip > optimizer_config.gradient_clip
            )
        )

        if validation_loss < best_loss - optimizer_config.min_delta:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = copy.deepcopy(
                {
                    name: value.detach().cpu()
                    for name, value in model.state_dict().items()
                }
            )
            stale = 0
        else:
            stale += 1
            if stale >= optimizer_config.early_stopping_patience:
                break
    _synchronize(device)
    elapsed = time.perf_counter() - started
    if best_state is None:
        raise RuntimeError("gradient training did not produce a valid checkpoint")
    model.load_state_dict(best_state)
    model.to(device)
    epochs_ran = len(history["train_mse"])
    return (
        {
            "best_epoch": best_epoch + 1,
            "epochs_ran": epochs_ran,
            "best_validation_mse_joint_head": best_loss,
            "hit_epoch_cap": epochs_ran >= optimizer_config.max_epochs,
            "preprocessor_fit_seconds": preprocessor_seconds,
            "gradient_training_seconds": elapsed,
            "training_seconds": preprocessor_seconds + elapsed,
        },
        history,
    )
