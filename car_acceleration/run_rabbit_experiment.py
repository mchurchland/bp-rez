#!/usr/bin/env python3
"""Forecast rabbit and wolf populations from their observed histories."""

from __future__ import annotations

import argparse

from .rabbit_experiment import run_rabbit_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="car_acceleration/results/rabbit_wolf_narrow_rates_seed7",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--history-length", type=int, default=200)
    parser.add_argument("--forecast-horizon", type=int, default=600)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--covariance-weight", type=float, default=0.0)
    parser.add_argument("--device", default="auto", help="auto, cpu, or cuda")
    args = parser.parse_args()
    run_rabbit_experiment(
        output_dir=args.output_dir,
        seed=args.seed,
        steps=args.steps,
        history_length=args.history_length,
        forecast_horizon=args.forecast_horizon,
        dt=args.dt,
        covariance_weight=args.covariance_weight,
        device_name=args.device,
    )


if __name__ == "__main__":
    main()
