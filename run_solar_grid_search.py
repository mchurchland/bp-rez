#!/usr/bin/env python3
"""Run a resumable, time-budgeted grid search for the solar reservoir."""

from __future__ import annotations

import argparse
import csv
import gc
import itertools
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from reservoir.experiment import resolve_device, seed_everything
from reservoir.models import count_trainable_parameters
from reservoir.solar_data import make_solar_splits
from reservoir.solar_experiment import (
    SolarExperimentConfig,
    evaluate_solar_model,
    train_solar_model,
)
from reservoir.solar_models import build_solar_model


PAPER_BETAS = (0.1, 0.1, 0.1, 0.01, 0.001)
SELECTION_VELOCITY_WEIGHT = 10.0
SELECTION_CURVATURE_WEIGHT = 10.0


@dataclass(frozen=True)
class GridPoint:
    warmup_steps: int
    steps_per_week: int
    interlayer_scale: float
    velocity_weight: float
    curvature_weight: float
    beta_mode: str

    @property
    def slug(self) -> str:
        def token(value: float) -> str:
            return f"{value:g}".replace(".", "p")

        return (
            f"warmup-{self.warmup_steps:02d}"
            f"_steps-{self.steps_per_week:02d}"
            f"_scale-{token(self.interlayer_scale)}"
            f"_velocity-{token(self.velocity_weight)}"
            f"_curvature-{token(self.curvature_weight)}"
            f"_beta-{self.beta_mode}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--time-budget-hours", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-seed", type=int, default=2026)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-runs", type=int)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run one tiny trial to verify the complete grid-search pipeline",
    )
    args = parser.parse_args()
    if args.time_budget_hours <= 0.0:
        parser.error("--time-budget-hours must be positive")
    if args.max_runs is not None and args.max_runs < 1:
        parser.error("--max-runs must be positive")
    return args


def build_grid() -> list[GridPoint]:
    points = [
        GridPoint(warmup, steps, scale, velocity, curvature, beta_mode)
        for warmup, steps, scale, (velocity, curvature), beta_mode in itertools.product(
            (5, 20, 40),
            (1, 3, 5),
            (1.0, 2.0, 4.0),
            ((1.0, 1.0), (10.0, 10.0), (30.0, 10.0)),
            ("paper", "zero"),
        )
    ]
    random.Random(2026).shuffle(points)
    return points


def json_dump(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def build_config(
    point: GridPoint,
    output_dir: Path,
    *,
    seed: int,
    data_seed: int,
    device: str,
    smoke: bool,
) -> SolarExperimentConfig:
    phase_betas = PAPER_BETAS if point.beta_mode == "paper" else (0.0,) * 5
    if smoke:
        return SolarExperimentConfig(
            output_dir=str(output_dir),
            seeds=(seed,),
            data_seed=data_seed,
            models=("reservoir",),
            train_samples=20,
            validation_samples=10,
            test_samples=10,
            series_length=6,
            nodes_1=12,
            nodes_2=12,
            reservoir_layers=2,
            latent_size=2,
            encoder_steps=2,
            second_reservoir_warmup_steps=point.warmup_steps,
            second_reservoir_steps=point.steps_per_week,
            interlayer_scale=point.interlayer_scale,
            density=0.4,
            phase_steps=(1,),
            phase_batch_sizes=(8,),
            phase_learning_rates=(1e-3,),
            phase_betas=(0.001 if point.beta_mode == "paper" else 0.0,),
            phase_horizons=(6,),
            mars_velocity_loss_weight=point.velocity_weight,
            mars_curvature_loss_weight=point.curvature_weight,
            training_log_interval=1,
            validation_interval=1,
            validation_subset=8,
            evaluation_batch_size=8,
            analysis_grid_size=5,
            device="cpu",
        )
    return SolarExperimentConfig(
        output_dir=str(output_dir),
        seeds=(seed,),
        data_seed=data_seed,
        models=("reservoir",),
        train_samples=95_000,
        validation_samples=5_000,
        test_samples=5_000,
        series_length=50,
        nodes_1=150,
        nodes_2=150,
        reservoir_layers=2,
        latent_size=2,
        encoder_steps=3,
        second_reservoir_warmup_steps=point.warmup_steps,
        second_reservoir_steps=point.steps_per_week,
        spectral_radius=0.9,
        density=0.1,
        leak_rate=1.0,
        input_scale=0.5,
        interlayer_scale=point.interlayer_scale,
        phase_steps=(1_000, 1_000, 1_000, 1_000, 11_000),
        phase_batch_sizes=(256, 1_024, 1_024, 2_048, 2_048),
        phase_learning_rates=(1e-4, 1e-4, 1e-4, 1e-5, 1e-5),
        phase_betas=phase_betas,
        phase_horizons=(20, 20, 50, 50, 50),
        mars_velocity_loss_weight=point.velocity_weight,
        mars_curvature_loss_weight=point.curvature_weight,
        training_log_interval=100,
        validation_interval=250,
        validation_subset=1_024,
        evaluation_batch_size=1_024,
        analysis_grid_size=35,
        device=device,
    )


def validation_selection_score(metrics: dict[str, Any]) -> float:
    return float(
        metrics["validation_mars_mse"]
        + SELECTION_VELOCITY_WEIGHT * metrics["validation_mars_velocity_mse"]
        + SELECTION_CURVATURE_WEIGHT * metrics["validation_mars_curvature_mse"]
    )


def leaderboard_record(
    point: GridPoint,
    metrics: dict[str, Any],
    *,
    grid_index: int,
    run_dir: Path,
) -> dict[str, Any]:
    return {
        "grid_index": grid_index,
        "run_name": point.slug,
        "run_dir": str(run_dir),
        "selection_score": validation_selection_score(metrics),
        "validation_mse": metrics["validation_mse"],
        "validation_sun_mse": metrics["validation_sun_mse"],
        "validation_mars_mse": metrics["validation_mars_mse"],
        "validation_mars_velocity_mse": metrics["validation_mars_velocity_mse"],
        "validation_mars_curvature_mse": metrics["validation_mars_curvature_mse"],
        "warmup_steps": point.warmup_steps,
        "steps_per_week": point.steps_per_week,
        "interlayer_scale": point.interlayer_scale,
        "velocity_weight": point.velocity_weight,
        "curvature_weight": point.curvature_weight,
        "beta_mode": point.beta_mode,
        "training_seconds": metrics["training_seconds"],
        "trainable_parameters": metrics["trainable_parameters"],
    }


def write_leaderboard(root: Path, records: list[dict[str, Any]]) -> None:
    ranked = sorted(records, key=lambda row: row["selection_score"])
    json_dump(root / "leaderboard.json", ranked)
    if ranked:
        with (root / "leaderboard.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(ranked[0]))
            writer.writeheader()
            writer.writerows(ranked)
        json_dump(root / "best_validation_config.json", ranked[0])


def load_completed_record(
    point: GridPoint, run_dir: Path, grid_index: int
) -> dict[str, Any] | None:
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.is_file():
        return None
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    return leaderboard_record(point, metrics, grid_index=grid_index, run_dir=run_dir)


def run_trial(
    point: GridPoint,
    config: SolarExperimentConfig,
    splits: dict[str, Any],
    device: torch.device,
    run_dir: Path,
) -> dict[str, Any]:
    seed = config.seeds[0]
    seed_everything(seed)
    model = build_solar_model(
        "reservoir",
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
        second_reservoir_warmup_steps=config.second_reservoir_warmup_steps,
        second_reservoir_steps=config.second_reservoir_steps,
        scinet_hidden_size=config.scinet_hidden_size,
        seed=seed,
        preserve_primary_latent=config.preserve_primary_latent,
        intermediate_latent_residual_scale=(
            config.intermediate_latent_residual_scale
        ),
    )
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
    metrics: dict[str, Any] = {
        "model": "reservoir",
        "seed": seed,
        "device": str(device),
        "trainable_parameters": count_trainable_parameters(model),
        **train_info,
        **{f"validation_{key}": value for key, value in validation_metrics.items()},
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    json_dump(run_dir / "config.json", asdict(config))
    json_dump(run_dir / "grid_point.json", asdict(point))
    json_dump(run_dir / "history.json", history)
    json_dump(run_dir / "metrics.json", metrics)
    torch.save(
        {
            "model_name": "reservoir",
            "seed": seed,
            "config": asdict(config),
            "grid_point": asdict(point),
            "state_dict": {
                key: value.detach().cpu() for key, value in model.state_dict().items()
            },
            "metrics": metrics,
        },
        run_dir / "checkpoint.pt",
    )
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metrics


def main() -> None:
    args = parse_args()
    root = Path(args.output_dir)
    runs_root = root / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)
    points = build_grid()
    if args.smoke:
        points = points[:1]
        args.max_runs = 1

    json_dump(
        root / "grid_definition.json",
        {
            "candidate_count": len(points),
            "data_seed": args.data_seed,
            "model_seed": args.seed,
            "selection_score": (
                "validation_mars_mse"
                " + 10 * validation_mars_velocity_mse"
                " + 10 * validation_mars_curvature_mse"
            ),
            "selection_uses_test_data": False,
            "time_budget_hours": args.time_budget_hours,
            "grid": {
                "warmup_steps": [5, 20, 40],
                "steps_per_week": [1, 3, 5],
                "interlayer_scale": [1.0, 2.0, 4.0],
                "shape_loss_weights": [[1.0, 1.0], [10.0, 10.0], [30.0, 10.0]],
                "beta_mode": ["paper", "zero"],
            },
        },
    )

    first_config = build_config(
        points[0],
        runs_root / points[0].slug,
        seed=args.seed,
        data_seed=args.data_seed,
        device=args.device,
        smoke=args.smoke,
    )
    device = resolve_device(first_config.device)
    print(f"Grid candidates: {len(points)}", flush=True)
    print(f"Device: {device}", flush=True)
    print(f"Time budget: {args.time_budget_hours:.3g} hours", flush=True)
    print("Generating one shared train/validation split...", flush=True)
    splits = make_solar_splits(
        first_config.train_samples,
        first_config.validation_samples,
        first_config.test_samples,
        first_config.series_length,
        first_config.data_seed,
        delta_days=first_config.delta_days,
        lifetime_days=first_config.lifetime_days,
        sampling_mode=first_config.sampling_mode,
    )

    records = []
    completed_this_run = 0
    started = time.monotonic()
    budget_seconds = args.time_budget_hours * 3_600.0
    for grid_index, point in enumerate(points):
        run_dir = runs_root / point.slug
        completed = load_completed_record(point, run_dir, grid_index)
        if completed is not None:
            records.append(completed)
            continue
        if args.max_runs is not None and completed_this_run >= args.max_runs:
            break
        elapsed = time.monotonic() - started
        if elapsed >= budget_seconds:
            print("Time budget reached; stopping before the next trial.", flush=True)
            break

        config = build_config(
            point,
            run_dir,
            seed=args.seed,
            data_seed=args.data_seed,
            device=str(device),
            smoke=args.smoke,
        )
        print(
            f"\n[{grid_index + 1}/{len(points)}] {point.slug} "
            f"elapsed={elapsed / 3_600.0:.2f}h",
            flush=True,
        )
        try:
            metrics = run_trial(point, config, splits, device, run_dir)
        except Exception as error:
            run_dir.mkdir(parents=True, exist_ok=True)
            json_dump(
                run_dir / "failure.json",
                {"error_type": type(error).__name__, "message": str(error)},
            )
            print(f"Trial failed: {type(error).__name__}: {error}", flush=True)
            continue
        record = leaderboard_record(
            point, metrics, grid_index=grid_index, run_dir=run_dir
        )
        records.append(record)
        completed_this_run += 1
        write_leaderboard(root, records)
        print(
            f"selection_score={record['selection_score']:.6g} "
            f"Mars MSE={record['validation_mars_mse']:.6g} "
            f"velocity={record['validation_mars_velocity_mse']:.6g} "
            f"curvature={record['validation_mars_curvature_mse']:.6g}",
            flush=True,
        )

    write_leaderboard(root, records)
    ranked = sorted(records, key=lambda row: row["selection_score"])
    print(f"\nCompleted configurations available: {len(ranked)}", flush=True)
    print("Top validation configurations:", flush=True)
    for rank, row in enumerate(ranked[:10], start=1):
        print(
            f"{rank:2d}. score={row['selection_score']:.6g} {row['run_name']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
