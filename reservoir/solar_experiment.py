"""Training and analysis for the secondary Copernicus experiment."""

from __future__ import annotations

import csv
import json
import math
import os
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "bp_reservoir_mpl")
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from .experiment import resolve_device, seed_everything
from .models import count_trainable_parameters
from .solar_data import (
    COPERNICUS_LIFETIME_DAYS,
    EARTH_PERIOD_DAYS,
    MARS_PERIOD_DAYS,
    SOLAR_SAMPLING_MODES,
    SolarDataset,
    earth_view_angles,
    make_solar_splits,
)
from .solar_models import SOLAR_MODEL_NAMES, SolarModelBase, build_solar_model


@dataclass
class SolarExperimentConfig:
    """Configuration for the paper-scale solar experiment.

    The five phases reproduce the public Copernicus notebook's batch sizes,
    learning rates, beta values, and 20-to-50-step curriculum. By default,
    ``phase_steps`` are minibatch optimizer updates. Set
    ``full_dataset_epochs=True`` to interpret them as the full shuffled passes
    used by the original TensorFlow training loop (millions of updates).
    """

    output_dir: str = "results/solar_replication"
    seeds: tuple[int, ...] = (0,)
    data_seed: int = 2026
    models: tuple[str, ...] = ("reservoir", "scinet")
    train_samples: int = 95_000
    validation_samples: int = 5_000
    test_samples: int = 5_000
    series_length: int = 50
    delta_days: float = 7.0
    lifetime_days: int = COPERNICUS_LIFETIME_DAYS
    sampling_mode: str = "independent_catalog"
    nodes_1: int = 150
    nodes_2: int = 150
    reservoir_layers: int = 10
    latent_size: int = 2
    encoder_steps: int = 3
    second_reservoir_warmup_steps: int = 20
    second_reservoir_steps: int = 3
    scinet_hidden_size: int = 100
    spectral_radius: float = 0.9
    density: float = 0.1
    leak_rate: float = 1.0
    input_scale: float = 0.5
    interlayer_scale: float = 2.0
    phase_steps: tuple[int, ...] = (1_000, 1_000, 1_000, 1_000, 11_000)
    phase_batch_sizes: tuple[int, ...] = (256, 1_024, 1_024, 2_048, 2_048)
    phase_learning_rates: tuple[float, ...] = (1e-4, 1e-4, 1e-4, 1e-5, 1e-5)
    phase_betas: tuple[float, ...] = (0.1, 0.1, 0.1, 0.01, 0.001)
    phase_horizons: tuple[int, ...] = (20, 20, 50, 50, 50)
    mars_velocity_loss_weight: float = 1.0
    mars_curvature_loss_weight: float = 1.0
    euler_l2_coeff: float = 1.0
    full_dataset_epochs: bool = False
    training_log_interval: int = 100
    validation_interval: int = 250
    validation_subset: int = 1_024
    evaluation_batch_size: int = 1_024
    gradient_clip_value: float = 10.0
    analysis_grid_size: int = 35
    device: str = "auto"


def _json_dump(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _validate_config(config: SolarExperimentConfig) -> None:
    if not config.seeds:
        raise ValueError("at least one seed is required")
    if not config.models:
        raise ValueError("at least one model is required")
    unknown = set(config.models) - set(SOLAR_MODEL_NAMES)
    if unknown:
        raise ValueError(f"unknown solar models: {sorted(unknown)}")
    if config.sampling_mode not in SOLAR_SAMPLING_MODES:
        raise ValueError(f"sampling_mode must be one of {SOLAR_SAMPLING_MODES}")
    if min(config.train_samples, config.validation_samples) < 1:
        raise ValueError("all dataset sizes must be positive")
    if config.test_samples < 2:
        raise ValueError("test_samples must be at least two for held-out latent fits")
    if config.latent_size < 1:
        raise ValueError("latent_size must be positive")
    if config.reservoir_layers < 2:
        raise ValueError("reservoir_layers must be at least two")
    if config.second_reservoir_warmup_steps < 0:
        raise ValueError("second_reservoir_warmup_steps must be nonnegative")
    phase_lengths = {
        len(config.phase_steps),
        len(config.phase_batch_sizes),
        len(config.phase_learning_rates),
        len(config.phase_betas),
        len(config.phase_horizons),
    }
    if phase_lengths != {len(config.phase_steps)} or not config.phase_steps:
        raise ValueError("all phase settings must have the same nonzero length")
    if min(config.phase_steps) < 1 or min(config.phase_batch_sizes) < 1:
        raise ValueError("phase steps and batch sizes must be positive")
    if min(config.phase_learning_rates) <= 0.0:
        raise ValueError("phase learning rates must be positive")
    if min(config.phase_betas) < 0.0:
        raise ValueError("phase betas must be nonnegative")
    if (
        min(
            config.mars_velocity_loss_weight,
            config.mars_curvature_loss_weight,
        )
        < 0.0
    ):
        raise ValueError("Mars dynamics loss weights must be nonnegative")
    if config.euler_l2_coeff < 0.0:
        raise ValueError("euler_l2_coeff must be nonnegative")
    if (
        min(config.phase_horizons) < 1
        or max(config.phase_horizons) > config.series_length
    ):
        raise ValueError("phase horizons must be between one and series_length")
    if (
        min(
            config.validation_interval,
            config.validation_subset,
            config.evaluation_batch_size,
            config.training_log_interval,
        )
        < 1
    ):
        raise ValueError(
            "validation_interval, validation_subset, evaluation_batch_size, and "
            "training_log_interval must be positive"
        )
    if config.analysis_grid_size < 3:
        raise ValueError("analysis_grid_size must be at least three")


def _batched_prediction(
    model: SolarModelBase,
    dataset: SolarDataset,
    horizon: int,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    predictions = []
    latents = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(dataset), batch_size):
            observation = dataset.observation[start : start + batch_size].to(device)
            prediction, latent = model.predict_with_latents(observation, horizon)
            predictions.append(prediction.cpu().numpy())
            latents.append(latent.cpu().numpy())
    return np.concatenate(predictions), np.concatenate(latents)


def evaluate_solar_model(
    model: SolarModelBase,
    dataset: SolarDataset,
    horizon: int,
    batch_size: int,
    device: torch.device,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    prediction, latents = _batched_prediction(
        model, dataset, horizon, batch_size, device
    )
    target = dataset.target[:, :horizon].numpy()
    squared_error = (prediction - target) ** 2
    mse = float(np.mean(squared_error))
    rmse = math.sqrt(mse)
    return (
        {
            "mse": mse,
            "rmse_radians": rmse,
            "relative_rmse_2pi": rmse / (2.0 * np.pi),
            "sun_rmse_radians": float(np.sqrt(np.mean(squared_error[..., 0]))),
            "mars_rmse_radians": float(np.sqrt(np.mean(squared_error[..., 1]))),
        },
        prediction,
        latents,
    )


def _validation_mse(
    model: SolarModelBase,
    dataset: SolarDataset,
    horizon: int,
    subset: int,
    batch_size: int,
    device: torch.device,
) -> float:
    count = min(subset, len(dataset))
    limited = SolarDataset(
        observation=dataset.observation[:count],
        target=dataset.target[:count],
        heliocentric=dataset.heliocentric[:count],
    )
    metrics, _, _ = evaluate_solar_model(model, limited, horizon, batch_size, device)
    return metrics["mse"]


def _training_diagnostics(
    model: SolarModelBase,
    dataset: SolarDataset,
    subset: int,
    batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    """Measure latent training state on a fixed validation subset."""

    diagnostics: dict[str, Any] = {
        "latent_delta": model.latent_delta.detach().cpu().tolist(),
    }
    if not model.variational_latent:
        return diagnostics

    count = min(subset, len(dataset))
    log_sigma_parts = []
    model.eval()
    with torch.no_grad():
        for start in range(0, count, batch_size):
            observation = dataset.observation[
                start : min(start + batch_size, count)
            ].to(device)
            log_sigma = model.latent_log_sigma(observation)
            if log_sigma is None:
                raise RuntimeError("variational model did not return latent log sigma")
            log_sigma_parts.append(log_sigma.cpu())
    all_log_sigma = torch.cat(log_sigma_parts)
    all_sigma = torch.exp(all_log_sigma)
    diagnostics.update(
        {
            "latent_log_sigma_min": float(all_log_sigma.min()),
            "latent_log_sigma_mean": float(all_log_sigma.mean()),
            "latent_log_sigma_max": float(all_log_sigma.max()),
            "latent_sigma_min": float(all_sigma.min()),
            "latent_sigma_mean": float(all_sigma.mean()),
            "latent_sigma_max": float(all_sigma.max()),
            "latent_sigma_mean_per_dimension": all_sigma.mean(dim=0).tolist(),
        }
    )
    return diagnostics


def _mars_dynamics_losses(
    prediction: torch.Tensor, target: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return MSEs for the Mars angle's discrete velocity and curvature."""

    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have the same shape")
    if prediction.ndim != 3 or prediction.shape[-1] != 2:
        raise ValueError("prediction and target must have shape [batch, time, 2]")
    zero = prediction.new_zeros(())
    if prediction.shape[1] < 2:
        return zero, zero
    predicted_velocity = torch.diff(prediction[..., 1], dim=1)
    target_velocity = torch.diff(target[..., 1], dim=1)
    velocity_loss = nn.functional.mse_loss(predicted_velocity, target_velocity)
    if prediction.shape[1] < 3:
        return velocity_loss, zero
    predicted_curvature = torch.diff(predicted_velocity, dim=1)
    target_curvature = torch.diff(target_velocity, dim=1)
    curvature_loss = nn.functional.mse_loss(predicted_curvature, target_curvature)
    return velocity_loss, curvature_loss


def train_solar_model(
    model: SolarModelBase,
    train: SolarDataset,
    validation: SolarDataset,
    config: SolarExperimentConfig,
    device: torch.device,
    seed: int,
) -> tuple[dict[str, Any], dict[str, list[Any]]]:
    model.to(device)
    optimizer = torch.optim.Adam(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=config.phase_learning_rates[0],
    )
    sampler = torch.Generator(device="cpu").manual_seed(seed + 91_337)
    history: dict[str, list[Any]] = {
        "step": [],
        "phase": [],
        "horizon": [],
        "reconstruction_loss": [],
        "total_loss": [],
        "validation_step": [],
        "validation_mse": [],
        "diagnostic_step": [],
        "latent_delta": [],
    }
    regularization_key = (
        "kl_loss" if model.variational_latent else "representation_loss"
    )
    history[regularization_key] = []
    if model.uses_mars_dynamics_loss:
        history["mars_velocity_loss"] = []
        history["mars_curvature_loss"] = []
    if model.variational_latent:
        history["evolution_l2_loss"] = []
    global_step = 0
    start_time = time.perf_counter()
    best_validation_mse = float("inf")
    best_validation_step = -1
    final_diagnostics: dict[str, Any] = {}

    for phase_index, (steps, batch_size, learning_rate, beta, horizon) in enumerate(
        zip(
            config.phase_steps,
            config.phase_batch_sizes,
            config.phase_learning_rates,
            config.phase_betas,
            config.phase_horizons,
            strict=True,
        ),
        start=1,
    ):
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        updates_per_epoch = (
            max(1, len(train) // batch_size) if config.full_dataset_epochs else 1
        )
        phase_updates = steps * updates_per_epoch
        validation_updates = config.validation_interval * updates_per_epoch
        permutation: torch.Tensor | None = None
        for phase_step in range(phase_updates):
            global_step += 1
            if config.full_dataset_epochs and len(train) >= batch_size:
                batch_in_epoch = phase_step % updates_per_epoch
                if batch_in_epoch == 0:
                    permutation = torch.randperm(len(train), generator=sampler)
                if permutation is None:
                    raise RuntimeError("training permutation was not initialized")
                start = batch_in_epoch * batch_size
                indices = permutation[start : start + batch_size]
            else:
                indices = torch.randint(
                    len(train), (batch_size,), generator=sampler, device="cpu"
                )
            observation = train.observation[indices].to(device)
            target = train.target[indices, :horizon].to(device)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            prediction, _, representation_penalty = model.training_forward(
                observation, horizon
            )
            reconstruction = nn.functional.mse_loss(prediction, target)
            if model.uses_mars_dynamics_loss:
                mars_velocity_loss, mars_curvature_loss = _mars_dynamics_losses(
                    prediction, target
                )
            else:
                mars_velocity_loss = reconstruction.new_zeros(())
                mars_curvature_loss = reconstruction.new_zeros(())
            evolution_l2_loss = model.evolution_l2_loss()
            loss = (
                reconstruction
                + config.mars_velocity_loss_weight * mars_velocity_loss
                + config.mars_curvature_loss_weight * mars_curvature_loss
                + beta * representation_penalty
                + config.euler_l2_coeff * evolution_l2_loss
            )
            loss.backward()
            if config.gradient_clip_value > 0.0:
                nn.utils.clip_grad_value_(
                    model.parameters(), config.gradient_clip_value
                )
            optimizer.step()

            validate_now = (
                phase_step + 1
            ) % validation_updates == 0 or phase_step == phase_updates - 1
            log_now = (
                global_step == 1
                or phase_step == 0
                or global_step % config.training_log_interval == 0
                or validate_now
            )
            if log_now:
                history["step"].append(global_step)
                history["phase"].append(phase_index)
                history["horizon"].append(horizon)
                history["reconstruction_loss"].append(
                    float(reconstruction.detach().cpu())
                )
                history[regularization_key].append(
                    float(representation_penalty.detach().cpu())
                )
                if model.uses_mars_dynamics_loss:
                    history["mars_velocity_loss"].append(
                        float(mars_velocity_loss.detach().cpu())
                    )
                    history["mars_curvature_loss"].append(
                        float(mars_curvature_loss.detach().cpu())
                    )
                if model.variational_latent:
                    history["evolution_l2_loss"].append(
                        float(evolution_l2_loss.detach().cpu())
                    )
                history["total_loss"].append(float(loss.detach().cpu()))

            if validate_now:
                validation_mse = _validation_mse(
                    model,
                    validation,
                    horizon,
                    config.validation_subset,
                    config.evaluation_batch_size,
                    device,
                )
                history["validation_step"].append(global_step)
                history["validation_mse"].append(validation_mse)
                if validation_mse < best_validation_mse:
                    best_validation_mse = validation_mse
                    best_validation_step = global_step
                final_diagnostics = _training_diagnostics(
                    model,
                    validation,
                    config.validation_subset,
                    config.evaluation_batch_size,
                    device,
                )
                history["diagnostic_step"].append(global_step)
                for key, value in final_diagnostics.items():
                    history.setdefault(key, []).append(value)
                message = (
                    f"    phase={phase_index} step={global_step} horizon={horizon} "
                    f"reconstruction={float(reconstruction.detach().cpu()):.6g} "
                    f"{regularization_key}="
                    f"{float(representation_penalty.detach().cpu()):.6g} "
                    f"validation_mse={validation_mse:.6g}"
                )
                if model.uses_mars_dynamics_loss:
                    message += (
                        f" mars_velocity_loss="
                        f"{float(mars_velocity_loss.detach().cpu()):.6g}"
                        f" mars_curvature_loss="
                        f"{float(mars_curvature_loss.detach().cpu()):.6g}"
                    )
                if model.variational_latent:
                    delta = ", ".join(
                        f"{value:.6g}" for value in final_diagnostics["latent_delta"]
                    )
                    message += (
                        f" sigma[min/mean/max]="
                        f"{final_diagnostics['latent_sigma_min']:.3g}/"
                        f"{final_diagnostics['latent_sigma_mean']:.3g}/"
                        f"{final_diagnostics['latent_sigma_max']:.3g}"
                        f" latent_delta=[{delta}]"
                    )
                print(message, flush=True)

    train_info: dict[str, Any] = {
        "optimization_steps": global_step,
        "best_validation_mse": best_validation_mse,
        "best_validation_step": best_validation_step,
        "training_seconds": time.perf_counter() - start_time,
    }
    train_info.update(
        {f"final_{key}": value for key, value in final_diagnostics.items()}
    )
    return (
        train_info,
        history,
    )


def _fit_linear_map(
    source: np.ndarray,
    destination: np.ndarray,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
) -> tuple[np.ndarray, float, list[float], float]:
    source_augmented = np.column_stack((source, np.ones(len(source))))
    coefficients = np.linalg.lstsq(
        source_augmented[train_indices], destination[train_indices], rcond=None
    )[0]
    prediction = source_augmented[test_indices] @ coefficients
    truth = destination[test_indices]
    residual = np.sum((truth - prediction) ** 2, axis=0)
    total = np.sum((truth - np.mean(truth, axis=0, keepdims=True)) ** 2, axis=0)
    per_dimension = 1.0 - residual / np.maximum(total, 1e-12)
    pooled_r2 = 1.0 - float(np.sum(residual)) / max(float(np.sum(total)), 1e-12)
    rmse = float(np.sqrt(np.mean((truth - prediction) ** 2)))
    return coefficients, pooled_r2, per_dimension.tolist(), rmse


def latent_diagnostics(
    model: SolarModelBase,
    dataset: SolarDataset,
    initial_latents: np.ndarray,
    delta_days: float,
    seed: int,
    chart_phi_earth: np.ndarray,
    chart_phi_mars: np.ndarray,
    chart_observation: np.ndarray,
    chart_latent: np.ndarray,
) -> dict[str, Any]:
    # First retain a raw test-distribution diagnostic. Its arbitrary +/-pi
    # branch cut makes it deliberately secondary for cyclic coordinates.
    test_latent = initial_latents[:, 0]
    test_heliocentric = dataset.heliocentric[:, 0].numpy()
    test_permutation = np.random.default_rng(seed).permutation(len(dataset))
    test_split = max(1, int(0.8 * len(test_permutation)))
    test_split = min(test_split, len(test_permutation) - 1)
    test_train = test_permutation[:test_split]
    test_holdout = test_permutation[test_split:]
    _, test_branch_helio_r2, _, _ = _fit_linear_map(
        test_heliocentric, test_latent, test_train, test_holdout
    )
    _, test_branch_reverse_r2, _, _ = _fit_linear_map(
        test_latent, test_heliocentric, test_train, test_holdout
    )

    # The primary diagnostic follows the paper's Figure 3 analysis: evaluate a
    # full heliocentric grid after unwrapping the observed Mars-angle branch,
    # then assess linearity on cells held out from the least-squares fit.
    heliocentric = np.column_stack(
        (chart_phi_earth.reshape(-1), chart_phi_mars.reshape(-1))
    )
    geocentric = chart_observation.reshape(-1, 2)
    latent = chart_latent.reshape(-1, model.latent_size)
    permutation = np.random.default_rng(seed + 1).permutation(len(latent))
    split = max(1, int(0.8 * len(permutation)))
    split = min(split, len(permutation) - 1)
    train_indices, test_indices = permutation[:split], permutation[split:]
    heliocentric_coefficients, helio_r2, helio_per_latent, _ = _fit_linear_map(
        heliocentric, latent, train_indices, test_indices
    )
    _, geo_r2, geo_per_latent, _ = _fit_linear_map(
        geocentric, latent, train_indices, test_indices
    )
    _, reverse_r2, reverse_per_angle, reverse_rmse = _fit_linear_map(
        latent, heliocentric, train_indices, test_indices
    )
    angular_increment = np.asarray(
        [
            2.0 * np.pi * delta_days / EARTH_PERIOD_DAYS,
            2.0 * np.pi * delta_days / MARS_PERIOD_DAYS,
        ]
    )
    expected_delta = angular_increment @ heliocentric_coefficients[:2]
    actual_delta = model.latent_delta.detach().cpu().numpy()
    expected_norm = float(np.linalg.norm(expected_delta))
    actual_norm = float(np.linalg.norm(actual_delta))
    relative_delta_error = float(
        np.linalg.norm(actual_delta - expected_delta) / max(expected_norm, 1e-12)
    )
    cosine = float(
        np.dot(actual_delta, expected_delta) / max(actual_norm * expected_norm, 1e-12)
    )
    return {
        "heliocentric_to_latent_r2": helio_r2,
        "heliocentric_to_latent_r2_per_dimension": helio_per_latent,
        "geocentric_to_latent_r2": geo_r2,
        "geocentric_to_latent_r2_per_dimension": geo_per_latent,
        "latent_to_heliocentric_r2": reverse_r2,
        "latent_to_heliocentric_r2_per_angle": reverse_per_angle,
        "latent_to_heliocentric_rmse_radians": reverse_rmse,
        "test_branch_heliocentric_to_latent_r2": test_branch_helio_r2,
        "test_branch_latent_to_heliocentric_r2": test_branch_reverse_r2,
        "learned_latent_delta": actual_delta.tolist(),
        "heliocentric_fit_expected_delta": expected_delta.tolist(),
        "latent_delta_relative_error": relative_delta_error,
        "latent_delta_cosine_similarity": cosine,
        "heliocentric_fit_coefficients": heliocentric_coefficients.tolist(),
    }


def _save_training_plot(history: dict[str, list[Any]], path: Path) -> None:
    variational = "kl_loss" in history
    mars_dynamics = "mars_velocity_loss" in history
    row_count = 3 if variational else 2 if mars_dynamics else 1
    figure, axes = plt.subplots(
        row_count,
        1,
        figsize=(9, 4.5 if row_count == 1 else 10.5),
        sharex=row_count > 1,
        squeeze=False,
    )
    loss_axis = axes[0, 0]
    loss_axis.semilogy(
        history["step"],
        history["reconstruction_loss"],
        alpha=0.65,
        label="train reconstruction",
    )
    loss_axis.semilogy(
        history["validation_step"],
        history["validation_mse"],
        marker="o",
        markersize=3,
        label="validation",
    )
    loss_axis.set(ylabel="MSE", title="Solar forecast training")
    loss_axis.grid(alpha=0.25)
    loss_axis.legend()

    if mars_dynamics:
        dynamics_axis = axes[1, 0]
        dynamics_axis.semilogy(
            history["step"],
            history["mars_velocity_loss"],
            label="Mars velocity",
        )
        dynamics_axis.semilogy(
            history["step"],
            history["mars_curvature_loss"],
            label="Mars curvature",
        )
        dynamics_axis.set(xlabel="Optimizer step", ylabel="Unweighted MSE")
        dynamics_axis.grid(alpha=0.25)
        dynamics_axis.legend()
    elif variational:
        regularization_axis = axes[1, 0]
        regularization_axis.semilogy(
            history["step"], history["kl_loss"], label="KL loss"
        )
        regularization_axis.semilogy(
            history["step"],
            history["evolution_l2_loss"],
            label="Euler weight L2",
        )
        regularization_axis.set(ylabel="Unweighted loss")
        regularization_axis.grid(alpha=0.25)
        regularization_axis.legend()

        diagnostic_axis = axes[2, 0]
        for statistic in ("min", "mean", "max"):
            diagnostic_axis.semilogy(
                history["diagnostic_step"],
                history[f"latent_sigma_{statistic}"],
                marker=".",
                label=f"sigma {statistic}",
            )
        diagnostic_axis.set(xlabel="Optimizer step", ylabel="Latent sigma")
        diagnostic_axis.grid(alpha=0.25)
        delta_axis = diagnostic_axis.twinx()
        latent_delta = np.asarray(history["latent_delta"])
        for dimension in range(latent_delta.shape[1]):
            delta_axis.plot(
                history["diagnostic_step"],
                latent_delta[:, dimension],
                linestyle="--",
                label=f"delta[{dimension}]",
            )
        delta_axis.set_ylabel("Latent delta")
        sigma_handles, sigma_labels = diagnostic_axis.get_legend_handles_labels()
        delta_handles, delta_labels = delta_axis.get_legend_handles_labels()
        diagnostic_axis.legend(
            sigma_handles + delta_handles,
            sigma_labels + delta_labels,
            loc="best",
        )
    else:
        loss_axis.set_xlabel("Optimizer step")

    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _save_prediction_plot(
    prediction: np.ndarray, target: np.ndarray, path: Path
) -> None:
    labels = ("Sun from Earth", "Mars from Earth")
    figure, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    steps = np.arange(prediction.shape[1])
    for index, axis in enumerate(axes):
        axis.plot(steps, target[0, :, index], label="target", linewidth=1.8)
        axis.plot(steps, prediction[0, :, index], label="prediction", linewidth=1.3)
        axis.set(ylabel="Angle (rad)", title=labels[index])
        axis.grid(alpha=0.25)
    axes[0].legend()
    axes[-1].set_xlabel("Week")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _latent_surface_data(
    model: SolarModelBase, grid_size: int, device: torch.device
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    values = np.linspace(0.0, 2.0 * np.pi, grid_size, dtype=np.float32)
    phi_earth, phi_mars = np.meshgrid(values, values)
    observation_grid = earth_view_angles(phi_earth, phi_mars)
    # Match the public Figure 3 analysis helper: remove branch jumps along the
    # flattened plotting path before passing the Earth-view Mars angle through
    # the encoder. This exposes a coordinate chart instead of a +/-pi seam.
    mars_plot_angle = observation_grid[..., 1].reshape(-1)
    branch_offset = np.rint((mars_plot_angle[0] + np.pi / 2.0) / (2.0 * np.pi))
    mars_plot_angle = np.unwrap(
        mars_plot_angle - 2.0 * np.pi * branch_offset, discont=5.0
    )
    observation_grid[..., 1] = mars_plot_angle.reshape(grid_size, grid_size)
    observation = observation_grid.reshape(-1, 2)
    model.eval()
    with torch.no_grad():
        latent = model.encode(torch.from_numpy(observation).to(device)).cpu().numpy()
    return (
        phi_earth,
        phi_mars,
        observation,
        latent.reshape(grid_size, grid_size, model.latent_size),
    )


def _save_latent_surfaces(
    phi_earth: np.ndarray,
    phi_mars: np.ndarray,
    latent: np.ndarray,
    path: Path,
) -> None:
    shown = min(latent.shape[-1], 4)
    figure = plt.figure(figsize=(6.2 * shown, 5.0))
    for index in range(shown):
        axis = figure.add_subplot(1, shown, index + 1, projection="3d")
        axis.plot_surface(
            phi_earth,
            phi_mars,
            latent[..., index],
            cmap="inferno",
            linewidth=0,
            antialiased=True,
        )
        axis.set(
            xlabel=r"Heliocentric $\phi_E$",
            ylabel=r"Heliocentric $\phi_M$",
            zlabel=f"Latent {index + 1}",
        )
        axis.set_xticks([0.0, np.pi, 2.0 * np.pi])
        axis.set_yticks([0.0, np.pi, 2.0 * np.pi])
    figure.suptitle("Latent activations over heliocentric state space")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries = []
    for model_name in dict.fromkeys(row["model"] for row in rows):
        selected = [row for row in rows if row["model"] == model_name]
        summary: dict[str, Any] = {
            "model": model_name,
            "seeds": len(selected),
            "trainable_parameters": selected[0]["trainable_parameters"],
        }
        for metric in (
            "test_relative_rmse_2pi",
            "heliocentric_to_latent_r2",
            "latent_to_heliocentric_r2",
        ):
            values = np.asarray([row[metric] for row in selected], dtype=np.float64)
            summary[f"{metric}_mean"] = float(values.mean())
            summary[f"{metric}_std"] = (
                float(values.std(ddof=1)) if len(values) > 1 else 0.0
            )
        summaries.append(summary)
    return summaries


def run_solar_experiment(
    config: SolarExperimentConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    _validate_config(config)
    root = Path(config.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    device = resolve_device(config.device)
    config_dict = asdict(config)
    config_dict["resolved_device"] = str(device)
    _json_dump(root / "config.json", config_dict)
    splits = make_solar_splits(
        config.train_samples,
        config.validation_samples,
        config.test_samples,
        config.series_length,
        config.data_seed,
        delta_days=config.delta_days,
        lifetime_days=config.lifetime_days,
        sampling_mode=config.sampling_mode,
    )
    rows: list[dict[str, Any]] = []
    for model_name in config.models:
        for seed in config.seeds:
            print(f"[{model_name}] seed={seed} device={device}", flush=True)
            seed_everything(seed)
            model = build_solar_model(
                model_name,
                nodes_1=config.nodes_1,
                nodes_2=config.nodes_2,
                reservoir_layers=config.reservoir_layers,
                latent_size=config.latent_size,
                spectral_radius=config.spectral_radius,
                input_scale=config.input_scale,
                interlayer_scale=config.interlayer_scale,
                density=config.density,
                leak_rate=config.leak_rate,
                encoder_steps=config.encoder_steps,
                second_reservoir_warmup_steps=(config.second_reservoir_warmup_steps),
                second_reservoir_steps=config.second_reservoir_steps,
                scinet_hidden_size=config.scinet_hidden_size,
                seed=seed,
            )
            run_dir = root / model_name / f"seed_{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)
            train_info, history = train_solar_model(
                model,
                splits["train"],
                splits["validation"],
                config,
                device,
                seed,
            )
            validation_metrics, _, _ = evaluate_solar_model(
                model,
                splits["validation"],
                config.series_length,
                config.evaluation_batch_size,
                device,
            )
            test_metrics, prediction, latents = evaluate_solar_model(
                model,
                splits["test"],
                config.series_length,
                config.evaluation_batch_size,
                device,
            )
            phi_earth, phi_mars, grid_observation, grid_latent = _latent_surface_data(
                model, config.analysis_grid_size, device
            )
            diagnostics = latent_diagnostics(
                model,
                splits["test"],
                latents,
                config.delta_days,
                config.data_seed + seed,
                phi_earth,
                phi_mars,
                grid_observation,
                grid_latent,
            )
            metrics: dict[str, Any] = {
                "model": model_name,
                "seed": seed,
                "device": str(device),
                "trainable_parameters": count_trainable_parameters(model),
                **train_info,
                **{
                    f"validation_{key}": value
                    for key, value in validation_metrics.items()
                },
                **{f"test_{key}": value for key, value in test_metrics.items()},
                **diagnostics,
            }
            torch.save(
                {
                    "model_name": model_name,
                    "seed": seed,
                    "config": asdict(config),
                    "state_dict": {
                        key: value.detach().cpu()
                        for key, value in model.state_dict().items()
                    },
                    "metrics": metrics,
                },
                run_dir / "checkpoint.pt",
            )
            np.savez_compressed(
                run_dir / "predictions.npz",
                observation=splits["test"].observation.numpy(),
                target=splits["test"].target.numpy(),
                heliocentric=splits["test"].heliocentric.numpy(),
                prediction=prediction,
                latent=latents,
            )
            np.savez_compressed(
                run_dir / "latent_surface.npz",
                phi_earth=phi_earth,
                phi_mars=phi_mars,
                observation=grid_observation,
                latent=grid_latent,
            )
            _json_dump(run_dir / "history.json", history)
            _json_dump(run_dir / "metrics.json", metrics)
            _save_training_plot(history, run_dir / "training.png")
            _save_prediction_plot(
                prediction, splits["test"].target.numpy(), run_dir / "predictions.png"
            )
            _save_latent_surfaces(
                phi_earth, phi_mars, grid_latent, run_dir / "latent_surfaces.png"
            )
            rows.append(metrics)
            _write_csv(root / "metrics.csv", rows)
            _json_dump(root / "metrics.json", rows)
            print(
                f"  test RMSE/(2pi)={metrics['test_relative_rmse_2pi']:.4%} "
                f"helio->latent R2={metrics['heliocentric_to_latent_r2']:.4f} "
                f"latent->helio R2={metrics['latent_to_heliocentric_r2']:.4f}",
                flush=True,
            )
    summary = _summarize(rows)
    _write_csv(root / "summary.csv", summary)
    _json_dump(root / "summary.json", summary)
    return rows, summary
