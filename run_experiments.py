#!/usr/bin/env python3
"""CLI for the 300-node NARMA10 reservoir comparison."""

import argparse

from reservoir.experiment import ExperimentConfig, run_experiment
from reservoir.models import MODEL_NAMES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="results/narma10_300_nodes")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--data-seed", type=int, default=2026)
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=list(MODEL_NAMES))
    parser.add_argument("--nodes-1", type=int, default=150)
    parser.add_argument("--nodes-2", type=int, default=150)
    parser.add_argument("--latent-size", type=int, default=10)
    parser.add_argument("--train-length", type=int, default=2000)
    parser.add_argument("--val-length", type=int, default=500)
    parser.add_argument("--test-length", type=int, default=500)
    parser.add_argument("--washout", type=int, default=100)
    parser.add_argument("--spectral-radius", type=float, default=0.9)
    parser.add_argument("--density", type=float, default=0.1)
    parser.add_argument("--leak-rate", type=float, default=1.0)
    parser.add_argument("--input-scale", type=float, default=0.5)
    parser.add_argument("--interlayer-scale", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--min-delta", type=float, default=1e-7)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ExperimentConfig(
        output_dir=args.output_dir,
        seeds=tuple(args.seeds),
        data_seed=args.data_seed,
        models=tuple(args.models),
        nodes_1=args.nodes_1,
        nodes_2=args.nodes_2,
        latent_size=args.latent_size,
        train_length=args.train_length,
        val_length=args.val_length,
        test_length=args.test_length,
        washout=args.washout,
        spectral_radius=args.spectral_radius,
        density=args.density,
        leak_rate=args.leak_rate,
        input_scale=args.input_scale,
        interlayer_scale=args.interlayer_scale,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_epochs=args.max_epochs,
        patience=args.patience,
        min_delta=args.min_delta,
        grad_clip=args.grad_clip,
        device=args.device,
    )
    _, summary = run_experiment(config)
    print("\nAggregate test results")
    for row in summary:
        print(
            f"{row['model']:20s} MSE={row['test_mse_mean']:.6g}±{row['test_mse_std']:.3g} "
            f"NRMSE={row['test_nrmse_mean']:.6g}±{row['test_nrmse_std']:.3g} "
            f"params={row['trainable_parameters']}"
        )


if __name__ == "__main__":
    main()
