#!/usr/bin/env python3
"""Compare a solar grid-search winner with the released SciNet checkpoint.

The script evaluates both models on the same deterministic held-out split and
produces plots that localize the forecast-performance gap. TensorFlow is only
needed for loading the released SciNet v1 checkpoint; the checkpoint graph is
imported directly, so the authors' Python-2 model code is not required.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Iterator

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "bp_reservoir_mpl")
)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from reservoir.solar_data import SolarDataset, earth_view_angles, make_solar_splits
from reservoir.solar_models import SolarReservoir, build_solar_model


TWO_PI = 2.0 * np.pi
EARTH_PERIOD_DAYS = 365.0
MARS_PERIOD_DAYS = 686.97959


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--grid-root",
        default="results/solar-grid-15k-108774",
        help="Grid-search directory containing best_validation_config.json",
    )
    parser.add_argument(
        "--run-dir",
        help="Override the selected grid run directory",
    )
    parser.add_argument(
        "--official-checkpoint-prefix",
        required=True,
        help="Path through copernicus.ckpt, without .meta/.index suffix",
    )
    parser.add_argument(
        "--output-dir",
        default="results/solar-performance-gap",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--grid-size", type=int, default=35)
    parser.add_argument(
        "--readout-ridge",
        type=float,
        default=1e-6,
        help="Ridge coefficient for the diagnostic validation-fit readout",
    )
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.grid_size < 3:
        parser.error("--grid-size must be at least three")
    if args.readout_ridge < 0.0:
        parser.error("--readout-ridge must be nonnegative")
    return args


def _json_dump(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _resolve_run_dir(grid_root: Path, explicit_run_dir: str | None) -> Path:
    if explicit_run_dir is not None:
        return Path(explicit_run_dir)
    winner_path = grid_root / "best_validation_config.json"
    winner = json.loads(winner_path.read_text(encoding="utf-8"))
    return Path(winner["run_dir"])


def _load_grid_model(
    run_dir: Path, device: torch.device
) -> tuple[SolarReservoir, dict[str, Any]]:
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    checkpoint = torch.load(
        run_dir / "checkpoint.pt", map_location="cpu", weights_only=True
    )
    model = build_solar_model(
        "reservoir",
        nodes_1=config["nodes_1"],
        nodes_2=config["nodes_2"],
        reservoir_layers=config["reservoir_layers"],
        latent_size=config["latent_size"],
        spectral_radius=config["spectral_radius"],
        input_scale=config["input_scale"],
        interlayer_scale=config["interlayer_scale"],
        density=config["density"],
        leak_rate=config["leak_rate"],
        encoder_steps=config["encoder_steps"],
        second_reservoir_warmup_steps=config["second_reservoir_warmup_steps"],
        second_reservoir_steps=config["second_reservoir_steps"],
        scinet_hidden_size=config["scinet_hidden_size"],
        seed=checkpoint["seed"],
        preserve_primary_latent=config.get("preserve_primary_latent", False),
        intermediate_latent_residual_scale=config.get(
            "intermediate_latent_residual_scale", 0.1
        ),
    )
    if not isinstance(model, SolarReservoir):
        raise TypeError("selected checkpoint is not a SolarReservoir")
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    return model, config


class OfficialSciNet:
    """Thin inference wrapper around the authors' saved TensorFlow v1 graph."""

    def __init__(self, checkpoint_prefix: Path) -> None:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
        import tensorflow as tf

        self.tf = tf
        tf.compat.v1.disable_eager_execution()
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.saver = tf.compat.v1.train.import_meta_graph(
                str(checkpoint_prefix) + ".meta", clear_devices=True
            )
        self.session = tf.compat.v1.Session(graph=self.graph)
        self.saver.restore(self.session, str(checkpoint_prefix))
        self.full_series = self.graph.get_tensor_by_name("full_time_series:0")
        self.epsilon = self.graph.get_tensor_by_name("epsilon:0")
        self.latent = self.graph.get_tensor_by_name("dynamic_state/Tanh:0")
        self.latent_delta = self.graph.get_tensor_by_name(
            "RNN/euler_vars/b_euler0:0"
        )
        self.outputs = [
            self.graph.get_tensor_by_name(
                "RNN/initial_euler_loss/b_add_2th_dec_layer:0"
            )
        ] + [
            self.graph.get_tensor_by_name(
                f"RNN/decode_{step}th_euler_step/b_add_2th_dec_layer:0"
            )
            for step in range(1, 50)
        ]

    def close(self) -> None:
        self.session.close()

    def _padded_input(self, observation: np.ndarray) -> np.ndarray:
        value = np.zeros((len(observation), 100), dtype=np.float32)
        value[:, :2] = observation
        return value

    def encode(self, observation: np.ndarray, batch_size: int) -> np.ndarray:
        encoded = []
        for start in range(0, len(observation), batch_size):
            selected = observation[start : start + batch_size]
            value = self.session.run(
                self.latent,
                feed_dict={
                    self.full_series: self._padded_input(selected),
                    self.epsilon: np.zeros((len(selected), 2), dtype=np.float32),
                },
            )
            encoded.append(value)
        return np.concatenate(encoded)

    def predict(
        self, observation: np.ndarray, batch_size: int
    ) -> tuple[np.ndarray, np.ndarray]:
        predictions = []
        latents = []
        for start in range(0, len(observation), batch_size):
            selected = observation[start : start + batch_size]
            values = self.session.run(
                self.outputs + [self.latent],
                feed_dict={
                    self.full_series: self._padded_input(selected),
                    self.epsilon: np.zeros((len(selected), 2), dtype=np.float32),
                },
            )
            predictions.append(np.stack(values[:-1], axis=1))
            latents.append(values[-1])
        return np.concatenate(predictions), np.concatenate(latents)

    def delta(self) -> np.ndarray:
        return np.asarray(self.session.run(self.latent_delta))


def _predict_grid(
    model: SolarReservoir,
    dataset: SolarDataset,
    horizon: int,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    predictions = []
    latents = []
    with torch.no_grad():
        for start in range(0, len(dataset), batch_size):
            observation = dataset.observation[start : start + batch_size].to(device)
            prediction, latent = model.predict_with_latents(observation, horizon)
            predictions.append(prediction.cpu().numpy())
            latents.append(latent.cpu().numpy())
    return np.concatenate(predictions), np.concatenate(latents)


def _metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    error = prediction.astype(np.float64) - target.astype(np.float64)
    squared = error**2
    mse = float(np.mean(squared))
    sun_mse = float(np.mean(squared[..., 0]))
    mars_mse = float(np.mean(squared[..., 1]))
    mars_velocity_error = np.diff(prediction[..., 1], axis=1) - np.diff(
        target[..., 1], axis=1
    )
    mars_curvature_error = np.diff(prediction[..., 1], n=2, axis=1) - np.diff(
        target[..., 1], n=2, axis=1
    )
    return {
        "mse": mse,
        "rmse_radians": math.sqrt(mse),
        "relative_rmse_2pi": math.sqrt(mse) / TWO_PI,
        "sun_mse": sun_mse,
        "sun_relative_rmse_2pi": math.sqrt(sun_mse) / TWO_PI,
        "mars_mse": mars_mse,
        "mars_relative_rmse_2pi": math.sqrt(mars_mse) / TWO_PI,
        "mars_velocity_mse": float(np.mean(mars_velocity_error**2)),
        "mars_curvature_mse": float(np.mean(mars_curvature_error**2)),
    }


def _error_curves(
    prediction: np.ndarray, target: np.ndarray
) -> dict[str, np.ndarray]:
    squared = (
        prediction.astype(np.float64) - target.astype(np.float64)
    ) ** 2
    per_week_overall_mse = np.mean(squared, axis=(0, 2))
    per_week_sun_mse = np.mean(squared[..., 0], axis=0)
    per_week_mars_mse = np.mean(squared[..., 1], axis=0)

    def relative(mse: np.ndarray) -> np.ndarray:
        return np.sqrt(mse) / TWO_PI

    def cumulative(mse: np.ndarray) -> np.ndarray:
        counts = np.arange(1, len(mse) + 1)
        return relative(np.cumsum(mse) / counts)

    return {
        "per_week_overall": relative(per_week_overall_mse),
        "per_week_sun": relative(per_week_sun_mse),
        "per_week_mars": relative(per_week_mars_mse),
        "cumulative_overall": cumulative(per_week_overall_mse),
        "cumulative_sun": cumulative(per_week_sun_mse),
        "cumulative_mars": cumulative(per_week_mars_mse),
    }


def _fit_linear_map(
    source: np.ndarray,
    destination: np.ndarray,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
) -> tuple[np.ndarray, float, list[float]]:
    augmented = np.column_stack((source, np.ones(len(source))))
    coefficients = np.linalg.lstsq(
        augmented[train_indices], destination[train_indices], rcond=None
    )[0]
    prediction = augmented[test_indices] @ coefficients
    truth = destination[test_indices]
    residual = np.sum((truth - prediction) ** 2, axis=0)
    total = np.sum((truth - np.mean(truth, axis=0, keepdims=True)) ** 2, axis=0)
    per_dimension = 1.0 - residual / np.maximum(total, 1e-12)
    pooled = 1.0 - float(np.sum(residual)) / max(float(np.sum(total)), 1e-12)
    return coefficients, pooled, per_dimension.tolist()


def _latent_grid(
    grid_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.linspace(0.0, TWO_PI, grid_size, dtype=np.float32)
    phi_earth, phi_mars = np.meshgrid(values, values)
    observation = earth_view_angles(phi_earth, phi_mars)
    mars = observation[..., 1].reshape(-1)
    branch_offset = np.rint((mars[0] + np.pi / 2.0) / TWO_PI)
    observation[..., 1] = np.unwrap(
        mars - TWO_PI * branch_offset, discont=5.0
    ).reshape(grid_size, grid_size)
    return phi_earth, phi_mars, observation


def _latent_diagnostics(
    phi_earth: np.ndarray,
    phi_mars: np.ndarray,
    observation: np.ndarray,
    latent: np.ndarray,
    delta: np.ndarray,
    seed: int,
) -> tuple[dict[str, Any], np.ndarray]:
    heliocentric = np.column_stack((phi_earth.ravel(), phi_mars.ravel()))
    geocentric = observation.reshape(-1, 2)
    flattened_latent = latent.reshape(-1, latent.shape[-1])
    permutation = np.random.default_rng(seed).permutation(len(flattened_latent))
    split = min(max(1, int(0.8 * len(permutation))), len(permutation) - 1)
    train, test = permutation[:split], permutation[split:]
    coefficients, helio_r2, helio_per_dimension = _fit_linear_map(
        heliocentric, flattened_latent, train, test
    )
    _, geo_r2, geo_per_dimension = _fit_linear_map(
        geocentric, flattened_latent, train, test
    )
    _, reverse_r2, reverse_per_dimension = _fit_linear_map(
        flattened_latent, heliocentric, train, test
    )
    angular_increment = np.asarray(
        [
            TWO_PI * 7.0 / EARTH_PERIOD_DAYS,
            TWO_PI * 7.0 / MARS_PERIOD_DAYS,
        ]
    )
    expected_delta = angular_increment @ coefficients[:2]
    expected_norm = float(np.linalg.norm(expected_delta))
    actual_norm = float(np.linalg.norm(delta))
    diagnostics = {
        "heliocentric_to_latent_r2": helio_r2,
        "heliocentric_to_latent_r2_per_dimension": helio_per_dimension,
        "geocentric_to_latent_r2": geo_r2,
        "geocentric_to_latent_r2_per_dimension": geo_per_dimension,
        "latent_to_heliocentric_r2": reverse_r2,
        "latent_to_heliocentric_r2_per_angle": reverse_per_dimension,
        "learned_latent_delta": delta.tolist(),
        "heliocentric_fit_expected_delta": expected_delta.tolist(),
        "latent_delta_relative_error": float(
            np.linalg.norm(delta - expected_delta) / max(expected_norm, 1e-12)
        ),
        "latent_delta_cosine_similarity": float(
            np.dot(delta, expected_delta)
            / max(actual_norm * expected_norm, 1e-12)
        ),
    }
    return diagnostics, coefficients


def _reservoir_features_from_latents(
    model: SolarReservoir, primary_latents: torch.Tensor
) -> torch.Tensor:
    states = [
        primary_latents.new_zeros((len(primary_latents), nodes))
        for nodes in model.reservoir_sizes[1:]
    ]
    features = []
    for time_index in range(primary_latents.shape[1]):
        current = primary_latents[:, time_index]
        update_count = (
            model.second_reservoir_warmup_steps
            if time_index == 0
            else model.second_reservoir_steps
        )
        for layer_index, state in enumerate(states):
            recurrent = getattr(model, model._recurrent_names[layer_index + 1])
            projection = getattr(model, model._projection_names[layer_index])
            drive = current @ projection.T
            for _ in range(update_count):
                state = model._update(state, recurrent, drive)
            states[layer_index] = state
            if layer_index < len(model.intermediate_weights):
                current = (
                    state @ model.intermediate_weights[layer_index].T
                    + model.intermediate_biases[layer_index]
                )
                if model.nonlinear:
                    current = torch.tanh(current)
        features.append(states[-1])
    return torch.stack(features, dim=1)


def _predict_from_primary_latents(
    model: SolarReservoir,
    primary_latents: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    predictions = []
    with torch.no_grad():
        for start in range(0, len(primary_latents), batch_size):
            latent = torch.from_numpy(
                primary_latents[start : start + batch_size].astype(np.float32)
            ).to(device)
            features = _reservoir_features_from_latents(model, latent)
            prediction = features @ model.W_out.T + model.c
            predictions.append(prediction.cpu().numpy())
    return np.concatenate(predictions)


def _primary_latents(
    model: SolarReservoir, observation: torch.Tensor, horizon: int
) -> torch.Tensor:
    initial = model.encode(observation)
    steps = torch.arange(
        horizon, device=observation.device, dtype=observation.dtype
    )
    return initial[:, None, :] + steps[None, :, None] * model.latent_delta


def _feature_batches(
    model: SolarReservoir,
    dataset: SolarDataset,
    batch_size: int,
    device: torch.device,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    with torch.no_grad():
        for start in range(0, len(dataset), batch_size):
            observation = dataset.observation[start : start + batch_size].to(device)
            primary = _primary_latents(model, observation, dataset.target.shape[1])
            features = _reservoir_features_from_latents(model, primary)
            yield (
                features.cpu().numpy().reshape(-1, features.shape[-1]),
                dataset.target[start : start + batch_size].numpy().reshape(-1, 2),
            )


def _fit_validation_readout(
    model: SolarReservoir,
    validation: SolarDataset,
    batch_size: int,
    device: torch.device,
    ridge: float,
) -> np.ndarray:
    width = model.reservoir_sizes[-1] + 1
    xtx = np.zeros((width, width), dtype=np.float64)
    xty = np.zeros((width, 2), dtype=np.float64)
    for features, target in _feature_batches(
        model, validation, batch_size, device
    ):
        augmented = np.column_stack(
            (features.astype(np.float64), np.ones(len(features)))
        )
        xtx += augmented.T @ augmented
        xty += augmented.T @ target.astype(np.float64)
    penalty = ridge * np.eye(width, dtype=np.float64)
    penalty[-1, -1] = 0.0
    return np.linalg.solve(xtx + penalty, xty)


def _predict_with_readout(
    model: SolarReservoir,
    dataset: SolarDataset,
    coefficients: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    predictions = []
    for features, _ in _feature_batches(model, dataset, batch_size, device):
        augmented = np.column_stack(
            (features.astype(np.float64), np.ones(len(features)))
        )
        predictions.append(augmented @ coefficients)
    return np.concatenate(predictions).reshape(
        len(dataset), dataset.target.shape[1], 2
    )


def _plot_error_by_week(
    grid_curves: dict[str, np.ndarray],
    official_curves: dict[str, np.ndarray],
    path: Path,
) -> None:
    weeks = np.arange(len(grid_curves["per_week_overall"]))
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    panels = (
        ("per_week_overall", "Both angles"),
        ("per_week_sun", "Sun"),
        ("per_week_mars", "Mars"),
    )
    for axis, (key, title) in zip(axes.flat[:3], panels, strict=True):
        axis.semilogy(
            weeks,
            100.0 * grid_curves[key],
            label="grid winner",
            color="#d55e00",
            linewidth=2,
        )
        axis.semilogy(
            weeks,
            100.0 * official_curves[key],
            label="official SciNet",
            color="#0072b2",
            linewidth=2,
        )
        axis.set(title=title, ylabel="RMSE / 2π (%)")
        axis.grid(alpha=0.25, which="both")
    ratio = grid_curves["per_week_overall"] / np.maximum(
        official_curves["per_week_overall"], 1e-12
    )
    axes[1, 1].plot(weeks, ratio, color="#6a3d9a", linewidth=2)
    axes[1, 1].axhline(1.0, color="black", linewidth=1, alpha=0.6)
    axes[1, 1].set(
        title="Overall RMSE ratio",
        xlabel="Forecast week",
        ylabel="grid / official",
    )
    axes[1, 1].grid(alpha=0.25)
    axes[1, 0].set_xlabel("Forecast week")
    axes[0, 0].legend()
    figure.suptitle("Forecast error by lead time")
    figure.tight_layout()
    figure.savefig(path, dpi=170)
    plt.close(figure)


def _plot_cumulative_error(
    grid_curves: dict[str, np.ndarray],
    official_curves: dict[str, np.ndarray],
    path: Path,
) -> None:
    weeks = np.arange(1, len(grid_curves["cumulative_overall"]) + 1)
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    for axis, component in zip(
        axes, ("overall", "sun", "mars"), strict=True
    ):
        key = f"cumulative_{component}"
        axis.semilogy(
            weeks,
            100.0 * grid_curves[key],
            label="grid winner",
            color="#d55e00",
            linewidth=2,
        )
        axis.semilogy(
            weeks,
            100.0 * official_curves[key],
            label="official SciNet",
            color="#0072b2",
            linewidth=2,
        )
        axis.set(
            title=component.capitalize(),
            xlabel="Forecast horizon (weeks)",
        )
        axis.grid(alpha=0.25, which="both")
    axes[0].set_ylabel("Cumulative RMSE / 2π (%)")
    axes[0].legend()
    figure.suptitle("Cumulative forecast error")
    figure.tight_layout()
    figure.savefig(path, dpi=170)
    plt.close(figure)


def _plot_examples(
    grid_prediction: np.ndarray,
    official_prediction: np.ndarray,
    target: np.ndarray,
    path: Path,
) -> list[int]:
    per_sample = np.mean((grid_prediction - target) ** 2, axis=(1, 2))
    ordered = np.argsort(per_sample)
    indices = [
        int(ordered[len(ordered) // 2]),
        int(ordered[min(len(ordered) - 1, int(0.9 * len(ordered)))]),
    ]
    weeks = np.arange(target.shape[1])
    figure, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
    for column, index in enumerate(indices):
        for row, (component, label) in enumerate(
            ((0, "Sun"), (1, "Mars"))
        ):
            axis = axes[row, column]
            axis.plot(
                weeks,
                target[index, :, component],
                color="black",
                linewidth=2.2,
                label="target",
            )
            axis.plot(
                weeks,
                grid_prediction[index, :, component],
                color="#d55e00",
                linewidth=1.7,
                label="grid winner",
            )
            axis.plot(
                weeks,
                official_prediction[index, :, component],
                color="#0072b2",
                linewidth=1.7,
                label="official SciNet",
            )
            quantile = "median-error" if column == 0 else "90th-percentile error"
            axis.set(
                title=f"{label}, {quantile} sample",
                ylabel="Angle (radians)",
            )
            axis.grid(alpha=0.25)
    axes[1, 0].set_xlabel("Forecast week")
    axes[1, 1].set_xlabel("Forecast week")
    axes[0, 0].legend()
    figure.suptitle("Representative held-out trajectories")
    figure.tight_layout()
    figure.savefig(path, dpi=170)
    plt.close(figure)
    return indices


def _binned_phase_error(
    phi_earth: np.ndarray,
    phi_mars: np.ndarray,
    sample_error: np.ndarray,
    bins: int,
) -> np.ndarray:
    earth_index = np.minimum((phi_earth / TWO_PI * bins).astype(int), bins - 1)
    mars_index = np.minimum((phi_mars / TWO_PI * bins).astype(int), bins - 1)
    total = np.zeros((bins, bins), dtype=np.float64)
    count = np.zeros((bins, bins), dtype=np.int64)
    np.add.at(total, (mars_index, earth_index), sample_error)
    np.add.at(count, (mars_index, earth_index), 1)
    return np.divide(
        total,
        count,
        out=np.full_like(total, np.nan),
        where=count > 0,
    )


def _plot_phase_error(
    dataset: SolarDataset,
    grid_prediction: np.ndarray,
    official_prediction: np.ndarray,
    path: Path,
    bins: int = 18,
) -> None:
    target = dataset.target.numpy()
    grid_error = np.sqrt(np.mean((grid_prediction - target) ** 2, axis=(1, 2)))
    official_error = np.sqrt(
        np.mean((official_prediction - target) ** 2, axis=(1, 2))
    )
    phi = np.mod(dataset.heliocentric[:, 0].numpy(), TWO_PI)
    grid_map = _binned_phase_error(phi[:, 0], phi[:, 1], grid_error, bins)
    official_map = _binned_phase_error(
        phi[:, 0], phi[:, 1], official_error, bins
    )
    ratio_map = grid_map / np.maximum(official_map, 1e-12)
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    extent = (0.0, TWO_PI, 0.0, TWO_PI)
    for axis, value, title, label in (
        (axes[0], 100.0 * grid_map / TWO_PI, "Grid winner", "RMSE / 2π (%)"),
        (
            axes[1],
            100.0 * official_map / TWO_PI,
            "Official SciNet",
            "RMSE / 2π (%)",
        ),
        (axes[2], ratio_map, "Gap ratio", "grid / official"),
    ):
        image = axis.imshow(
            value,
            origin="lower",
            extent=extent,
            aspect="auto",
            cmap="magma",
        )
        axis.set(
            title=title,
            xlabel=r"Initial heliocentric $\phi_E$",
            ylabel=r"Initial heliocentric $\phi_M$",
        )
        figure.colorbar(image, ax=axis, label=label)
    figure.suptitle("Forecast error over initial heliocentric state")
    figure.tight_layout()
    figure.savefig(path, dpi=170)
    plt.close(figure)


def _plot_latent_surfaces(
    phi_earth: np.ndarray,
    phi_mars: np.ndarray,
    grid_latent: np.ndarray,
    official_latent: np.ndarray,
    path: Path,
) -> None:
    figure = plt.figure(figsize=(14, 10))
    for row, (name, latent) in enumerate(
        (("Grid winner", grid_latent), ("Official SciNet", official_latent))
    ):
        for column in range(2):
            axis = figure.add_subplot(
                2, 2, row * 2 + column + 1, projection="3d"
            )
            axis.plot_surface(
                phi_earth,
                phi_mars,
                latent[..., column],
                cmap="inferno",
                linewidth=0,
                antialiased=True,
            )
            axis.set(
                title=f"{name}: latent {column + 1}",
                xlabel=r"$\phi_E$",
                ylabel=r"$\phi_M$",
                zlabel="activation",
            )
            axis.set_xticks((0.0, np.pi, TWO_PI))
            axis.set_yticks((0.0, np.pi, TWO_PI))
    figure.suptitle("Latent surfaces on the same heliocentric grid")
    figure.tight_layout()
    figure.savefig(path, dpi=170)
    plt.close(figure)


def _plot_decomposition(
    decompositions: dict[str, dict[str, float]], path: Path
) -> None:
    labels = list(decompositions)
    values = [
        100.0 * decompositions[label]["relative_rmse_2pi"] for label in labels
    ]
    colors = ["#d55e00", "#e69f00", "#cc79a7", "#009e73", "#0072b2"]
    figure, axis = plt.subplots(figsize=(11, 5.5))
    positions = np.arange(len(labels))
    bars = axis.bar(positions, values, color=colors[: len(labels)])
    axis.set_yscale("log")
    axis.set(
        xticks=positions,
        xticklabels=labels,
        ylabel="Test RMSE / 2π (%)",
        title="Controlled bottleneck decomposition",
    )
    axis.grid(alpha=0.25, axis="y", which="both")
    for bar, value in zip(bars, values, strict=True):
        axis.text(
            bar.get_x() + bar.get_width() / 2.0,
            value * 1.08,
            f"{value:.3f}%",
            ha="center",
            va="bottom",
        )
    figure.tight_layout()
    figure.savefig(path, dpi=170)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    run_dir = _resolve_run_dir(Path(args.grid_root), args.run_dir)
    print(f"Grid winner: {run_dir}", flush=True)
    model, config = _load_grid_model(run_dir, device)
    print(f"Device: {device}", flush=True)
    splits = make_solar_splits(
        config["train_samples"],
        config["validation_samples"],
        config["test_samples"],
        config["series_length"],
        config["data_seed"],
        delta_days=config["delta_days"],
        lifetime_days=config["lifetime_days"],
        sampling_mode=config["sampling_mode"],
    )
    target = splits["test"].target.numpy()
    grid_prediction, grid_latents = _predict_grid(
        model,
        splits["test"],
        config["series_length"],
        args.batch_size,
        device,
    )

    print("Restoring official TensorFlow checkpoint...", flush=True)
    official = OfficialSciNet(Path(args.official_checkpoint_prefix))
    try:
        official_prediction, official_initial_latent = official.predict(
            splits["test"].observation.numpy(), args.batch_size
        )
        official_delta = official.delta()

        phi_earth, phi_mars, grid_observation = _latent_grid(args.grid_size)
        flattened_observation = grid_observation.reshape(-1, 2)
        with torch.no_grad():
            grid_surface = (
                model.encode(
                    torch.from_numpy(flattened_observation).to(device)
                )
                .cpu()
                .numpy()
                .reshape(args.grid_size, args.grid_size, model.latent_size)
            )
        official_surface = official.encode(
            flattened_observation, args.batch_size
        ).reshape(args.grid_size, args.grid_size, 2)
    finally:
        official.close()

    grid_latent_metrics, _ = _latent_diagnostics(
        phi_earth,
        phi_mars,
        grid_observation,
        grid_surface,
        model.latent_delta.detach().cpu().numpy(),
        config["data_seed"] + 1,
    )
    official_latent_metrics, _ = _latent_diagnostics(
        phi_earth,
        phi_mars,
        grid_observation,
        official_surface,
        official_delta,
        config["data_seed"] + 1,
    )

    steps = np.arange(config["series_length"], dtype=np.float64)
    expected_delta = np.asarray(
        grid_latent_metrics["heliocentric_fit_expected_delta"]
    )
    corrected_increment_latents = (
        grid_latents[:, :1, :].astype(np.float64)
        + steps[None, :, None]
        * expected_delta[None, None, :]
    )
    corrected_increment_prediction = _predict_from_primary_latents(
        model, corrected_increment_latents, args.batch_size, device
    )

    print("Fitting diagnostic linear readout on validation features...", flush=True)
    readout = _fit_validation_readout(
        model,
        splits["validation"],
        args.batch_size,
        device,
        args.readout_ridge,
    )
    refit_prediction = _predict_with_readout(
        model, splits["test"], readout, args.batch_size, device
    )

    grid_metrics = _metrics(grid_prediction, target)
    official_metrics = _metrics(official_prediction, target)
    decompositions = {
        "Grid winner": grid_metrics,
        "Corrected latent increment": _metrics(
            corrected_increment_prediction, target
        ),
        "Refit final readout": _metrics(refit_prediction, target),
        "Official SciNet": official_metrics,
    }
    grid_curves = _error_curves(grid_prediction, target)
    official_curves = _error_curves(official_prediction, target)

    print("Writing figures...", flush=True)
    _plot_error_by_week(
        grid_curves, official_curves, output_dir / "error_by_week.png"
    )
    _plot_cumulative_error(
        grid_curves, official_curves, output_dir / "cumulative_error.png"
    )
    example_indices = _plot_examples(
        grid_prediction,
        official_prediction,
        target,
        output_dir / "representative_trajectories.png",
    )
    _plot_phase_error(
        splits["test"],
        grid_prediction,
        official_prediction,
        output_dir / "phase_space_error.png",
    )
    _plot_latent_surfaces(
        phi_earth,
        phi_mars,
        grid_surface,
        official_surface,
        output_dir / "latent_surfaces_comparison.png",
    )
    _plot_decomposition(
        decompositions, output_dir / "bottleneck_decomposition.png"
    )

    report = {
        "grid_run_dir": str(run_dir),
        "official_checkpoint_prefix": args.official_checkpoint_prefix,
        "test_samples": len(splits["test"]),
        "grid_winner": grid_metrics,
        "official_scinet": official_metrics,
        "controlled_decomposition": decompositions,
        "grid_latent_diagnostics": grid_latent_metrics,
        "official_latent_diagnostics": official_latent_metrics,
        "example_indices": example_indices,
        "curves": {
            "forecast_week": list(range(config["series_length"])),
            "grid": {key: value.tolist() for key, value in grid_curves.items()},
            "official": {
                key: value.tolist() for key, value in official_curves.items()
            },
        },
    }
    _json_dump(output_dir / "metrics.json", report)
    np.savez_compressed(
        output_dir / "diagnostics.npz",
        target=target,
        grid_prediction=grid_prediction,
        official_prediction=official_prediction,
        corrected_increment_prediction=corrected_increment_prediction,
        refit_prediction=refit_prediction,
        grid_latents=grid_latents,
        official_initial_latent=official_initial_latent,
        grid_surface=grid_surface,
        official_surface=official_surface,
        phi_earth=phi_earth,
        phi_mars=phi_mars,
    )
    print(
        "Grid winner RMSE/(2pi)="
        f"{grid_metrics['relative_rmse_2pi']:.4%}",
        flush=True,
    )
    print(
        "Official SciNet RMSE/(2pi)="
        f"{official_metrics['relative_rmse_2pi']:.4%}",
        flush=True,
    )
    print(f"Report written to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
