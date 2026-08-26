#!/usr/bin/env python3
"""Run the SciNet-style car experiment."""

from __future__ import annotations

import argparse

from .scinet_experiment import run_scinet_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", default="car_acceleration/results/scinet_2latent_seed8"
    )
    parser.add_argument("--seed", type=int, default=8)
    parser.add_argument("--device", default="auto", help="auto, cpu, or cuda")
    args = parser.parse_args()
    run_scinet_experiment(
        output_dir=args.output_dir,
        seed=args.seed,
        device_name=args.device,
    )


if __name__ == "__main__":
    main()

