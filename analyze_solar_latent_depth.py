#!/usr/bin/env python3
"""Measure physical alignment of every 2D latent in a deep solar reservoir."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
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

from reservoir.solar_data import earth_view_angles, make_solar_splits
from reservoir.solar_models import SolarReservoir, build_solar_model


TWO_PI = 2.0 * np.pi


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        default="results/solar-10x150-shape-15k-106370/reservoir/seed_0",
        help="Directory containing checkpoint.pt",
    )
    parser.add_argument(
        "--output-dir",
        default=(
            "results/solar-10x150-shape-15k-106370/"
            "latent-depth-analysis"
        ),
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--grid-size", type=int, default=35)
    parser.add_argument(
        "--preserve-primary-latent",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override the checkpoint's intermediate latent preservation mode",
    )
    parser.add_argument(
        "--intermediate-latent-residual-scale",
        type=float,
        help="Override the checkpoint's bounded residual scale",
    )
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.grid_size < 3:
        parser.error("--grid-size must be at least three")
    if (
        args.intermediate_latent_residual_scale is not None
        and args.intermediate_latent_residual_scale < 0.0
    ):
        parser.error("--intermediate-latent-residual-scale must be nonnegative")
    return args


def _load_model(
    run_dir: Path,
    device: torch.device,
    preserve_primary_latent: bool | None,
    intermediate_latent_residual_scale: float | None,
) -> tuple[SolarReservoir, dict[str, Any], int]:
    checkpoint = torch.load(
        run_dir / "checkpoint.pt", map_location="cpu", weights_only=True
    )
    config = checkpoint["config"]
    seed = int(checkpoint["seed"])
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
        second_reservoir_warmup_steps=config[
            "second_reservoir_warmup_steps"
        ],
        second_reservoir_steps=config["second_reservoir_steps"],
        scinet_hidden_size=config["scinet_hidden_size"],
        seed=seed,
        preserve_primary_latent=(
            config.get("preserve_primary_latent", False)
            if preserve_primary_latent is None
            else preserve_primary_latent
        ),
        intermediate_latent_residual_scale=(
            config.get("intermediate_latent_residual_scale", 0.1)
            if intermediate_latent_residual_scale is None
            else intermediate_latent_residual_scale
        ),
    )
    if not isinstance(model, SolarReservoir):
        raise TypeError("checkpoint does not contain a SolarReservoir")
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    return model, config, seed


def _phase_grid(
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


def _all_initial_latents(
    model: SolarReservoir,
    observation: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    predictions = []
    latents = []
    flattened = observation.reshape(-1, 2)
    with torch.no_grad():
        for start in range(0, len(flattened), batch_size):
            batch = torch.from_numpy(
                flattened[start : start + batch_size].astype(np.float32)
            ).to(device)
            prediction, all_latents = model.predict_with_all_latents(batch, 1)
            predictions.append(prediction[:, 0].cpu().numpy())
            latents.append(all_latents[:, 0].cpu().numpy())
    return np.concatenate(predictions), np.concatenate(latents)


def _fit_score(
    source: np.ndarray,
    destination: np.ndarray,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
) -> tuple[float, list[float], float]:
    augmented = np.column_stack(
        (source.astype(np.float64), np.ones(len(source)))
    )
    destination = destination.astype(np.float64)
    coefficients = np.linalg.lstsq(
        augmented[train_indices], destination[train_indices], rcond=None
    )[0]
    prediction = augmented[test_indices] @ coefficients
    truth = destination[test_indices]
    squared_residual = np.sum((truth - prediction) ** 2, axis=0)
    centered = truth - np.mean(truth, axis=0, keepdims=True)
    total = np.sum(centered**2, axis=0)
    per_dimension = 1.0 - squared_residual / np.maximum(total, 1e-12)
    pooled = 1.0 - float(np.sum(squared_residual)) / max(
        float(np.sum(total)), 1e-12
    )
    rmse = float(np.sqrt(np.mean((truth - prediction) ** 2)))
    return pooled, per_dimension.tolist(), rmse


def _layer_scores(
    latents: np.ndarray,
    heliocentric: np.ndarray,
    geocentric: np.ndarray,
    seed: int,
) -> list[dict[str, Any]]:
    permutation = np.random.default_rng(seed).permutation(len(latents))
    split = min(max(1, int(0.8 * len(permutation))), len(permutation) - 1)
    train_indices = permutation[:split]
    test_indices = permutation[split:]
    results = []
    for layer_index in range(latents.shape[1]):
        layer_latent = latents[:, layer_index]
        helio_forward = _fit_score(
            heliocentric, layer_latent, train_indices, test_indices
        )
        helio_reverse = _fit_score(
            layer_latent, heliocentric, train_indices, test_indices
        )
        geo_forward = _fit_score(
            geocentric, layer_latent, train_indices, test_indices
        )
        geo_reverse = _fit_score(
            layer_latent, geocentric, train_indices, test_indices
        )
        results.append(
            {
                "latent_index": layer_index + 1,
                "source_reservoir": layer_index + 1,
                "heliocentric_to_latent_r2": helio_forward[0],
                "heliocentric_to_latent_r2_per_dimension": helio_forward[1],
                "latent_to_heliocentric_r2": helio_reverse[0],
                "latent_to_heliocentric_r2_per_angle": helio_reverse[1],
                "latent_to_heliocentric_rmse_radians": helio_reverse[2],
                "geocentric_to_latent_r2": geo_forward[0],
                "geocentric_to_latent_r2_per_dimension": geo_forward[1],
                "latent_to_geocentric_r2": geo_reverse[0],
                "latent_to_geocentric_r2_per_angle": geo_reverse[1],
                "latent_to_geocentric_rmse_radians": geo_reverse[2],
            }
        )
    return results


def _plot_scores(scores: list[dict[str, Any]], path: Path) -> None:
    layers = np.asarray([score["latent_index"] for score in scores])
    figure, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), sharey=True)
    relationships = (
        (
            "Heliocentric alignment",
            "heliocentric_to_latent_r2",
            "latent_to_heliocentric_r2",
        ),
        (
            "Geocentric alignment",
            "geocentric_to_latent_r2",
            "latent_to_geocentric_r2",
        ),
    )
    for axis, (title, forward_key, reverse_key) in zip(
        axes, relationships, strict=True
    ):
        axis.plot(
            layers,
            [score[forward_key] for score in scores],
            marker="o",
            linewidth=2,
            label="physical coordinates → latent",
        )
        axis.plot(
            layers,
            [score[reverse_key] for score in scores],
            marker="s",
            linewidth=2,
            label="latent → physical coordinates",
        )
        axis.set(
            title=title,
            xlabel="2D latent after reservoir",
            xticks=layers,
            ylim=(-0.05, 1.02),
        )
        axis.grid(alpha=0.25)
        axis.legend(frameon=False)
    axes[0].set_ylabel("Held-out linear-fit $R^2$")
    figure.suptitle("Physical alignment across the 10×150 reservoir stack")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_surfaces(
    latents: np.ndarray, grid_size: int, path: Path
) -> None:
    layer_count = latents.shape[1]
    figure, axes = plt.subplots(
        layer_count,
        2,
        figsize=(8.2, 2.35 * layer_count),
        constrained_layout=True,
    )
    surfaces = latents.reshape(grid_size, grid_size, layer_count, 2)
    for layer_index in range(layer_count):
        for dimension in range(2):
            axis = axes[layer_index, dimension]
            image = axis.imshow(
                surfaces[:, :, layer_index, dimension],
                origin="lower",
                extent=(0.0, TWO_PI, 0.0, TWO_PI),
                aspect="auto",
                cmap="viridis",
            )
            axis.set_title(
                f"After reservoir {layer_index + 1}: latent {dimension + 1}"
            )
            axis.set(xlabel="Earth phase", ylabel="Mars phase")
            figure.colorbar(image, ax=axis, shrink=0.82)
    figure.savefig(path, dpi=150)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    model, config, seed = _load_model(
        run_dir,
        device,
        args.preserve_primary_latent,
        args.intermediate_latent_residual_scale,
    )

    phi_earth, phi_mars, observation = _phase_grid(args.grid_size)
    grid_prediction, grid_latents = _all_initial_latents(
        model, observation, args.batch_size, device
    )
    heliocentric = np.column_stack((phi_earth.ravel(), phi_mars.ravel()))
    geocentric = observation.reshape(-1, 2)
    analysis_seed = int(config["data_seed"]) + seed
    grid_scores = _layer_scores(
        grid_latents, heliocentric, geocentric, analysis_seed + 1
    )

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
    test = splits["test"]
    test_prediction, test_latents = _all_initial_latents(
        model,
        test.observation.numpy(),
        args.batch_size,
        device,
    )
    test_heliocentric = test.heliocentric[:, 0].numpy()
    test_geocentric = test.target[:, 0].numpy()
    test_scores = _layer_scores(
        test_latents,
        test_heliocentric,
        test_geocentric,
        analysis_seed,
    )

    report = {
        "run_dir": str(run_dir),
        "reservoir_layers": model.reservoir_layers,
        "explicit_2d_latents": int(grid_latents.shape[1]),
        "preserve_primary_latent": model.preserve_primary_latent,
        "intermediate_latent_residual_scale": (
            model.intermediate_latent_residual_scale
        ),
        "note": (
            "The final reservoir maps directly to the model output, so a "
            "10-reservoir model has nine explicit 2D latent readouts."
        ),
        "grid_scores": grid_scores,
        "test_branch_scores": test_scores,
        "initial_grid_output_mse": float(
            np.mean(
                (
                    grid_prediction.astype(np.float64)
                    - geocentric.astype(np.float64)
                )
                ** 2
            )
        ),
        "initial_test_output_mse": float(
            np.mean(
                (
                    test_prediction.astype(np.float64)
                    - test_geocentric.astype(np.float64)
                )
                ** 2
            )
        ),
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _plot_scores(grid_scores, output_dir / "latent_r2_by_depth.png")
    _plot_surfaces(
        grid_latents,
        args.grid_size,
        output_dir / "latent_surfaces_by_depth.png",
    )
    np.savez_compressed(
        output_dir / "latent_depth_data.npz",
        phi_earth=phi_earth,
        phi_mars=phi_mars,
        observation=observation,
        latents=grid_latents,
        prediction=grid_prediction,
    )

    print(
        "layer  helio->latent  latent->helio  geo->latent  latent->geo",
        flush=True,
    )
    for score in grid_scores:
        print(
            f"{score['latent_index']:>5d}"
            f"  {score['heliocentric_to_latent_r2']:>13.6f}"
            f"  {score['latent_to_heliocentric_r2']:>13.6f}"
            f"  {score['geocentric_to_latent_r2']:>11.6f}"
            f"  {score['latent_to_geocentric_r2']:>11.6f}",
            flush=True,
        )
    print(f"Report written to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
