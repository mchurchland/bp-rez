#!/usr/bin/env python3
"""Run one 2D-latent car experiment with a direct linear readout."""

from __future__ import annotations

import argparse

from .experiment import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="car_acceleration/results/linear_readout_2latent_positive_seed8",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--covariance-weight", type=float, default=0.01)
    parser.add_argument("--device", default="auto", help="auto, cpu, or cuda")
    args = parser.parse_args()
    run_experiment(
        output_dir=args.output_dir,
        seed=args.seed,
        steps=args.steps,
        covariance_weight=args.covariance_weight,
        device_name=args.device,
    )


if __name__ == "__main__":
    main()
