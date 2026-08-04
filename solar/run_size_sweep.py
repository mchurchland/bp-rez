#!/usr/bin/env python3
"""Compare matched two-reservoir solar models across reservoir widths."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "bp_reservoir_mpl")
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .experiment import SolarExperimentConfig, run_solar_experiment


DEFAULT_RESERVOIR_SIZES = (10, 20, 30, 50, 75, 100, 150, 200, 250, 300)
DEFAULT_MODEL_SEEDS = (0, 1, 2)

SUMMARY_METRICS = (
    "validation_relative_rmse_2pi",
    "test_relative_rmse_2pi",
    "heliocentric_to_latent_r2",
    "latent_to_heliocentric_r2",
    "geocentric_to_latent_r2",
    "latent_to_geocentric_r2",
    "heliocentric_advantage_forward_r2",
    "heliocentric_advantage_reverse_r2",
    "latent_delta_cosine_similarity",
    "latent_delta_relative_error",
    "training_seconds",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=list(DEFAULT_RESERVOIR_SIZES),
        help="Shared width of the first and second reservoirs",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_MODEL_SEEDS),
        help="Model/reservoir seeds evaluated at every size",
    )
    parser.add_argument("--data-seed", type=int, default=2026)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--max-sizes",
        type=int,
        help="Run at most this many previously incomplete sizes",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run one size, one seed, and one optimizer update",
    )
    args = parser.parse_args()
    try:
        _validate_sweep_inputs(args.sizes, args.seeds)
    except ValueError as error:
        parser.error(str(error))
    if args.max_sizes is not None and args.max_sizes < 1:
        parser.error("--max-sizes must be positive")
    return args


def _validate_sweep_inputs(sizes: Sequence[int], seeds: Sequence[int]) -> None:
    if not sizes:
        raise ValueError("at least one reservoir size is required")
    if any(size < 1 for size in sizes):
        raise ValueError("reservoir sizes must be positive")
    if list(sizes) != sorted(set(sizes)):
        raise ValueError("reservoir sizes must be unique and strictly increasing")
    if not seeds:
        raise ValueError("at least one model seed is required")
    if len(set(seeds)) != len(seeds):
        raise ValueError("model seeds must be unique")


def build_config(
    reservoir_size: int,
    output_dir: Path,
    *,
    seeds: Sequence[int],
    data_seed: int,
    device: str,
    smoke: bool,
) -> SolarExperimentConfig:
    """Build one controlled two-reservoir trial."""

    common = {
        "output_dir": str(output_dir),
        "seeds": tuple(seeds),
        "data_seed": data_seed,
        "models": ("reservoir",),
        "nodes_1": reservoir_size,
        "nodes_2": reservoir_size,
        "reservoir_layers": 2,
        "latent_size": 2,
        "preserve_primary_latent": True,
        "intermediate_latent_residual_scale": 0.1,
        "intermediate_latent_skip_mode": "sequential",
    }
    if smoke:
        return SolarExperimentConfig(
            **common,
            train_samples=20,
            validation_samples=10,
            test_samples=10,
            series_length=6,
            encoder_steps=2,
            second_reservoir_steps=1,
            decoder_bias_scale=1.0,
            spectral_radius=0.9,
            density=0.4,
            leak_rate=1.0,
            input_scale=0.5,
            interlayer_scale=2.0,
            phase_steps=(1,),
            phase_batch_sizes=(8,),
            phase_learning_rates=(1e-3,),
            phase_betas=(0.001,),
            phase_horizons=(6,),
            mars_velocity_loss_weight=1.0,
            mars_curvature_loss_weight=1.0,
            training_log_interval=1,
            validation_interval=1,
            validation_subset=8,
            evaluation_batch_size=8,
            analysis_grid_size=5,
            device="cpu",
        )
    return SolarExperimentConfig(
        **common,
        train_samples=95_000,
        validation_samples=5_000,
        test_samples=5_000,
        series_length=50,
        delta_days=7.0,
        lifetime_days=25_657,
        sampling_mode="independent_catalog",
        encoder_steps=3,
        second_reservoir_steps=3,
        decoder_bias_scale=1.0,
        spectral_radius=0.9,
        density=0.1,
        leak_rate=1.0,
        input_scale=0.5,
        interlayer_scale=2.0,
        phase_steps=(1_000, 1_000, 1_000, 1_000, 11_000),
        phase_batch_sizes=(256, 1_024, 1_024, 2_048, 2_048),
        phase_learning_rates=(1e-4, 1e-4, 1e-4, 1e-5, 1e-5),
        phase_betas=(0.1, 0.1, 0.1, 0.01, 0.001),
        phase_horizons=(20, 20, 50, 50, 50),
        mars_velocity_loss_weight=1.0,
        mars_curvature_loss_weight=1.0,
        training_log_interval=100,
        validation_interval=250,
        validation_subset=1_024,
        evaluation_batch_size=1_024,
        gradient_clip_value=10.0,
        analysis_grid_size=35,
        device=device,
    )


def fixed_matrix_scalars(reservoir_size: int, latent_size: int = 2) -> int:
    """Count the fixed recurrent and input-projection matrix entries."""

    return (
        2 * reservoir_size * reservoir_size
        + 2 * reservoir_size
        + latent_size * reservoir_size
    )


def _json_dump(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _sweep_definition(
    sizes: Sequence[int],
    seeds: Sequence[int],
    data_seed: int,
    smoke: bool,
) -> dict[str, Any]:
    prototype = build_config(
        sizes[0],
        Path("<size-run>"),
        seeds=seeds,
        data_seed=data_seed,
        device="<runtime-device>",
        smoke=smoke,
    )
    fixed_config = asdict(prototype)
    for key in ("output_dir", "nodes_1", "nodes_2", "seeds", "device"):
        fixed_config.pop(key)
    return {
        "experiment": "matched two-reservoir width sweep",
        "reservoir_sizes": list(sizes),
        "model_seeds": list(seeds),
        "data_seed": data_seed,
        "reservoir_layers": 2,
        "explicit_latent_bottlenecks": 1,
        "matched_first_and_second_reservoir_widths": True,
        "smoke": smoke,
        "fixed_config": fixed_config,
    }


def _prepare_definition(root: Path, definition: dict[str, Any]) -> None:
    path = root / "size_sweep_definition.json"
    normalized = json.loads(json.dumps(definition))
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != normalized:
            raise ValueError(
                f"{path} describes a different sweep; use a new output directory"
            )
    else:
        _json_dump(path, normalized)


def _record_from_metrics(
    reservoir_size: int,
    metrics: dict[str, Any],
    run_dir: Path,
) -> dict[str, Any]:
    trainable = int(metrics["trainable_parameters"])
    fixed = fixed_matrix_scalars(reservoir_size)
    depth_metrics = metrics["latent_depth_diagnostics"][0]
    latent_to_geocentric_r2 = float(depth_metrics["latent_to_geocentric_r2"])
    return {
        "reservoir_size": reservoir_size,
        "seed": int(metrics["seed"]),
        "run_dir": str(
            Path("runs") / run_dir.name / "reservoir" / f"seed_{int(metrics['seed'])}"
        ),
        "device": metrics["device"],
        "trainable_parameters": trainable,
        "fixed_matrix_scalars": fixed,
        "total_tensor_scalars": fixed + trainable,
        "optimization_steps": int(metrics["optimization_steps"]),
        "training_seconds": float(metrics["training_seconds"]),
        "best_validation_mse": float(metrics["best_validation_mse"]),
        "best_validation_step": int(metrics["best_validation_step"]),
        "validation_mse": float(metrics["validation_mse"]),
        "validation_relative_rmse_2pi": float(metrics["validation_relative_rmse_2pi"]),
        "test_mse": float(metrics["test_mse"]),
        "test_relative_rmse_2pi": float(metrics["test_relative_rmse_2pi"]),
        "test_sun_mse": float(metrics["test_sun_mse"]),
        "test_mars_mse": float(metrics["test_mars_mse"]),
        "test_mars_velocity_mse": float(metrics["test_mars_velocity_mse"]),
        "test_mars_curvature_mse": float(metrics["test_mars_curvature_mse"]),
        "heliocentric_to_latent_r2": float(metrics["heliocentric_to_latent_r2"]),
        "latent_to_heliocentric_r2": float(metrics["latent_to_heliocentric_r2"]),
        "geocentric_to_latent_r2": float(metrics["geocentric_to_latent_r2"]),
        "latent_to_geocentric_r2": latent_to_geocentric_r2,
        "heliocentric_advantage_forward_r2": float(
            metrics["heliocentric_to_latent_r2"] - metrics["geocentric_to_latent_r2"]
        ),
        "heliocentric_advantage_reverse_r2": float(
            metrics["latent_to_heliocentric_r2"] - latent_to_geocentric_r2
        ),
        "latent_delta_cosine_similarity": float(
            metrics["latent_delta_cosine_similarity"]
        ),
        "latent_delta_relative_error": float(metrics["latent_delta_relative_error"]),
    }


def _load_completed_records(
    reservoir_size: int,
    run_dir: Path,
    expected_seeds: Sequence[int],
) -> list[dict[str, Any]] | None:
    config_path = run_dir / "config.json"
    metrics_path = run_dir / "metrics.json"
    if not config_path.is_file() or not metrics_path.is_file():
        return None
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if (
        config.get("nodes_1") != reservoir_size
        or config.get("nodes_2") != reservoir_size
        or config.get("reservoir_layers") != 2
        or config.get("seeds") != list(expected_seeds)
        or config.get("models") != ["reservoir"]
    ):
        return None
    metrics_rows = json.loads(metrics_path.read_text(encoding="utf-8"))
    if not isinstance(metrics_rows, list):
        return None
    completed_seeds = [int(row.get("seed", -1)) for row in metrics_rows]
    if completed_seeds != list(expected_seeds):
        return None
    return [
        _record_from_metrics(reservoir_size, metrics, run_dir)
        for metrics in metrics_rows
    ]


def summarize_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries = []
    for reservoir_size in sorted({row["reservoir_size"] for row in records}):
        selected = [row for row in records if row["reservoir_size"] == reservoir_size]
        summary: dict[str, Any] = {
            "reservoir_size": reservoir_size,
            "seeds": len(selected),
            "trainable_parameters": selected[0]["trainable_parameters"],
            "fixed_matrix_scalars": selected[0]["fixed_matrix_scalars"],
            "total_tensor_scalars": selected[0]["total_tensor_scalars"],
        }
        for metric in SUMMARY_METRICS:
            values = np.asarray([row[metric] for row in selected], dtype=np.float64)
            summary[f"{metric}_mean"] = float(values.mean())
            summary[f"{metric}_std"] = (
                float(values.std(ddof=1)) if len(values) > 1 else 0.0
            )
        summaries.append(summary)
    return summaries


def _errorbar(
    axis: plt.Axes,
    summaries: list[dict[str, Any]],
    metric: str,
    *,
    label: str,
    scale: float = 1.0,
    linestyle: str = "-",
) -> None:
    sizes = [row["reservoir_size"] for row in summaries]
    means = [scale * row[f"{metric}_mean"] for row in summaries]
    errors = [scale * row[f"{metric}_std"] for row in summaries]
    axis.errorbar(
        sizes,
        means,
        yerr=errors,
        marker="o",
        linewidth=2,
        capsize=3,
        linestyle=linestyle,
        label=label,
    )


def _save_sweep_plot(
    summaries: list[dict[str, Any]],
    path: Path,
) -> None:
    sizes = [row["reservoir_size"] for row in summaries]
    figure, axes = plt.subplots(2, 2, figsize=(13.5, 9.0), sharex=True)

    _errorbar(
        axes[0, 0],
        summaries,
        "validation_relative_rmse_2pi",
        label="validation",
        scale=100.0,
    )
    _errorbar(
        axes[0, 0],
        summaries,
        "test_relative_rmse_2pi",
        label="test",
        scale=100.0,
    )
    axes[0, 0].set(
        title="Forecast error",
        ylabel="RMSE / (2π) [%]",
    )

    for metric, label, linestyle in (
        ("heliocentric_to_latent_r2", "heliocentric → latent", "-"),
        ("latent_to_heliocentric_r2", "latent → heliocentric", "-"),
        ("geocentric_to_latent_r2", "geocentric → latent", "--"),
        ("latent_to_geocentric_r2", "latent → geocentric", "--"),
    ):
        _errorbar(
            axes[0, 1],
            summaries,
            metric,
            label=label,
            linestyle=linestyle,
        )
    axes[0, 1].set(
        title="Latent linear alignment",
        ylabel="Held-out $R^2$",
    )

    _errorbar(
        axes[1, 0],
        summaries,
        "heliocentric_advantage_forward_r2",
        label="coordinates → latent",
    )
    _errorbar(
        axes[1, 0],
        summaries,
        "heliocentric_advantage_reverse_r2",
        label="latent → coordinates",
    )
    axes[1, 0].axhline(0.0, color="black", linewidth=1, alpha=0.55)
    axes[1, 0].set(
        title="Heliocentric advantage over geocentric",
        xlabel="Neurons per reservoir",
        ylabel="$\\Delta R^2$",
    )

    _errorbar(
        axes[1, 1],
        summaries,
        "training_seconds",
        label="training time",
    )
    axes[1, 1].set(
        title="Compute cost",
        xlabel="Neurons per reservoir",
        ylabel="Seconds",
    )

    for axis in axes.flat:
        axis.set_xticks(sizes)
        axis.grid(alpha=0.25)
        axis.legend(frameon=False)
    figure.suptitle(
        "Two-reservoir solar size sweep (one explicit 2D latent bottleneck)"
    )
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _write_sweep_outputs(
    root: Path,
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    ordered = sorted(records, key=lambda row: (row["reservoir_size"], row["seed"]))
    summaries = summarize_records(ordered)
    _json_dump(root / "size_sweep.json", ordered)
    _json_dump(root / "size_sweep_summary.json", summaries)
    _write_csv(root / "size_sweep.csv", ordered)
    _write_csv(root / "size_sweep_summary.csv", summaries)
    if summaries:
        best = min(
            summaries,
            key=lambda row: row["validation_relative_rmse_2pi_mean"],
        )
        _json_dump(root / "best_validation_size.json", best)
        _save_sweep_plot(summaries, root / "size_sweep.png")
    return summaries


def run_size_sweep(
    output_dir: Path,
    *,
    sizes: Sequence[int] = DEFAULT_RESERVOIR_SIZES,
    seeds: Sequence[int] = DEFAULT_MODEL_SEEDS,
    data_seed: int = 2026,
    device: str = "auto",
    max_sizes: int | None = None,
    smoke: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run or resume the controlled size sweep."""

    sizes = tuple(sizes)
    seeds = tuple(seeds)
    _validate_sweep_inputs(sizes, seeds)
    if max_sizes is not None and max_sizes < 1:
        raise ValueError("max_sizes must be positive")
    if smoke:
        sizes = sizes[:1]
        seeds = seeds[:1]

    root = Path(output_dir)
    runs_root = root / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)
    definition = _sweep_definition(sizes, seeds, data_seed, smoke)
    _prepare_definition(root, definition)

    records: list[dict[str, Any]] = []
    completed_this_call = 0
    for index, reservoir_size in enumerate(sizes, start=1):
        run_dir = runs_root / f"size_{reservoir_size:03d}"
        completed = _load_completed_records(reservoir_size, run_dir, seeds)
        if completed is not None:
            records.extend(completed)
            print(
                f"[{index}/{len(sizes)}] size={reservoir_size}: already complete",
                flush=True,
            )
            continue
        if max_sizes is not None and completed_this_call >= max_sizes:
            break

        config = build_config(
            reservoir_size,
            run_dir,
            seeds=seeds,
            data_seed=data_seed,
            device=device,
            smoke=smoke,
        )
        print(
            f"\n[{index}/{len(sizes)}] size={reservoir_size} seeds={list(seeds)}",
            flush=True,
        )
        try:
            metric_rows, _ = run_solar_experiment(config)
        except Exception as error:
            run_dir.mkdir(parents=True, exist_ok=True)
            _json_dump(
                run_dir / "failure.json",
                {"error_type": type(error).__name__, "message": str(error)},
            )
            print(
                f"size={reservoir_size} failed: {type(error).__name__}: {error}",
                flush=True,
            )
            continue

        records.extend(
            _record_from_metrics(reservoir_size, metrics, run_dir)
            for metrics in metric_rows
        )
        completed_this_call += 1
        _write_sweep_outputs(root, records)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summaries = _write_sweep_outputs(root, records)
    print(f"\nCompleted sizes available: {len(summaries)}/{len(sizes)}", flush=True)
    if summaries:
        best = min(
            summaries,
            key=lambda row: row["validation_relative_rmse_2pi_mean"],
        )
        print(
            "Best validation size: "
            f"{best['reservoir_size']} neurons/reservoir, "
            "RMSE/(2pi)="
            f"{best['validation_relative_rmse_2pi_mean']:.4%}",
            flush=True,
        )
    return records, summaries


def main() -> None:
    args = parse_args()
    run_size_sweep(
        Path(args.output_dir),
        sizes=args.sizes,
        seeds=args.seeds,
        data_seed=args.data_seed,
        device=args.device,
        max_sizes=args.max_sizes,
        smoke=args.smoke,
    )


if __name__ == "__main__":
    main()
