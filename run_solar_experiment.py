#!/usr/bin/env python3
"""Run the Copernicus solar-system latent-representation experiment."""

from __future__ import annotations

import argparse

from reservoir.solar_data import SOLAR_SAMPLING_MODES
from reservoir.solar_experiment import SolarExperimentConfig, run_solar_experiment
from reservoir.solar_models import SOLAR_MODEL_NAMES


def parse_args() -> argparse.Namespace:
    defaults = SolarExperimentConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=defaults.output_dir)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(defaults.seeds))
    parser.add_argument("--data-seed", type=int, default=defaults.data_seed)
    parser.add_argument(
        "--models", nargs="+", choices=SOLAR_MODEL_NAMES, default=list(defaults.models)
    )
    parser.add_argument("--train-samples", type=int, default=defaults.train_samples)
    parser.add_argument(
        "--validation-samples", type=int, default=defaults.validation_samples
    )
    parser.add_argument("--test-samples", type=int, default=defaults.test_samples)
    parser.add_argument("--series-length", type=int, default=defaults.series_length)
    parser.add_argument("--delta-days", type=float, default=defaults.delta_days)
    parser.add_argument("--lifetime-days", type=int, default=defaults.lifetime_days)
    parser.add_argument(
        "--sampling-mode",
        choices=SOLAR_SAMPLING_MODES,
        default=defaults.sampling_mode,
    )
    parser.add_argument("--nodes-1", type=int, default=defaults.nodes_1)
    parser.add_argument("--nodes-2", type=int, default=defaults.nodes_2)
    parser.add_argument("--latent-size", type=int, default=defaults.latent_size)
    parser.add_argument("--encoder-steps", type=int, default=defaults.encoder_steps)
    parser.add_argument("--decoder-steps", type=int, default=defaults.decoder_steps)
    parser.add_argument(
        "--scinet-hidden-size", type=int, default=defaults.scinet_hidden_size
    )
    parser.add_argument("--spectral-radius", type=float, default=defaults.spectral_radius)
    parser.add_argument("--density", type=float, default=defaults.density)
    parser.add_argument("--leak-rate", type=float, default=defaults.leak_rate)
    parser.add_argument("--input-scale", type=float, default=defaults.input_scale)
    parser.add_argument(
        "--interlayer-scale", type=float, default=defaults.interlayer_scale
    )
    parser.add_argument(
        "--phase-steps", type=int, nargs="+", default=list(defaults.phase_steps)
    )
    parser.add_argument(
        "--phase-batch-sizes",
        type=int,
        nargs="+",
        default=list(defaults.phase_batch_sizes),
    )
    parser.add_argument(
        "--phase-learning-rates",
        type=float,
        nargs="+",
        default=list(defaults.phase_learning_rates),
    )
    parser.add_argument(
        "--phase-betas", type=float, nargs="+", default=list(defaults.phase_betas)
    )
    parser.add_argument(
        "--phase-horizons", type=int, nargs="+", default=list(defaults.phase_horizons)
    )
    parser.add_argument(
        "--full-dataset-epochs",
        action="store_true",
        help="Interpret phase steps as full shuffled dataset passes, as in the original code",
    )
    parser.add_argument(
        "--validation-interval", type=int, default=defaults.validation_interval
    )
    parser.add_argument(
        "--validation-subset", type=int, default=defaults.validation_subset
    )
    parser.add_argument(
        "--evaluation-batch-size", type=int, default=defaults.evaluation_batch_size
    )
    parser.add_argument(
        "--gradient-clip-value", type=float, default=defaults.gradient_clip_value
    )
    parser.add_argument(
        "--analysis-grid-size", type=int, default=defaults.analysis_grid_size
    )
    parser.add_argument("--device", default=defaults.device, help="auto, cpu, cuda, or cuda:N")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use a short three-phase development run instead of the paper-scale schedule",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.quick:
        args.train_samples = 2_000
        args.validation_samples = 500
        args.test_samples = 500
        args.phase_steps = [50, 50, 100]
        args.phase_batch_sizes = [64, 128, 128]
        args.phase_learning_rates = [1e-3, 5e-4, 1e-4]
        args.phase_betas = [0.01, 0.01, 0.001]
        args.phase_horizons = [20, 20, 50]
        args.validation_interval = 25
        args.validation_subset = 256
        args.evaluation_batch_size = 256
    config = SolarExperimentConfig(
        output_dir=args.output_dir,
        seeds=tuple(args.seeds),
        data_seed=args.data_seed,
        models=tuple(args.models),
        train_samples=args.train_samples,
        validation_samples=args.validation_samples,
        test_samples=args.test_samples,
        series_length=args.series_length,
        delta_days=args.delta_days,
        lifetime_days=args.lifetime_days,
        sampling_mode=args.sampling_mode,
        nodes_1=args.nodes_1,
        nodes_2=args.nodes_2,
        latent_size=args.latent_size,
        encoder_steps=args.encoder_steps,
        decoder_steps=args.decoder_steps,
        scinet_hidden_size=args.scinet_hidden_size,
        spectral_radius=args.spectral_radius,
        density=args.density,
        leak_rate=args.leak_rate,
        input_scale=args.input_scale,
        interlayer_scale=args.interlayer_scale,
        phase_steps=tuple(args.phase_steps),
        phase_batch_sizes=tuple(args.phase_batch_sizes),
        phase_learning_rates=tuple(args.phase_learning_rates),
        phase_betas=tuple(args.phase_betas),
        phase_horizons=tuple(args.phase_horizons),
        full_dataset_epochs=args.full_dataset_epochs,
        validation_interval=args.validation_interval,
        validation_subset=args.validation_subset,
        evaluation_batch_size=args.evaluation_batch_size,
        gradient_clip_value=args.gradient_clip_value,
        analysis_grid_size=args.analysis_grid_size,
        device=args.device,
    )
    _, summary = run_solar_experiment(config)
    print("\nAggregate solar results")
    for row in summary:
        print(
            f"{row['model']:12s} RMSE/(2pi)="
            f"{row['test_relative_rmse_2pi_mean']:.4%} "
            f"helio->latent R2={row['heliocentric_to_latent_r2_mean']:.4f} "
            f"latent->helio R2={row['latent_to_heliocentric_r2_mean']:.4f}"
        )


if __name__ == "__main__":
    main()
