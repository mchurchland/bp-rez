#!/usr/bin/env python3
"""Aggregate a pulled NARMA run from its immutable locked suite manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from narma.benchmark import SuiteConfig, aggregate_results
from narma.models import ReservoirConfig
from narma.training import OptimizerConfig


def load_locked_suite(output_root: Path) -> SuiteConfig:
    """Reconstruct the exact suite used to produce a locked result directory."""

    lock_path = output_root / "locked_configs.json"
    if not lock_path.is_file():
        raise FileNotFoundError(f"{lock_path} does not exist")
    locked = json.loads(lock_path.read_text(encoding="utf-8"))
    raw_suite: dict[str, Any] = dict(locked["suite"])

    recorded_root = Path(raw_suite["output_root"])
    if recorded_root.resolve() != output_root.resolve():
        raise RuntimeError(
            "the requested result directory does not match the output_root "
            f"recorded in {lock_path}: {recorded_root}"
        )

    for field in (
        "orders",
        "regimes",
        "models",
        "final_pair_ids",
        "tuning_pair_ids",
        "ridge_alphas",
    ):
        raw_suite[field] = tuple(raw_suite[field])
    raw_suite["reservoir"] = ReservoirConfig(**raw_suite["reservoir"])
    raw_suite["optimizer"] = OptimizerConfig(**raw_suite["optimizer"])
    return SuiteConfig(**raw_suite)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "output_root",
        type=Path,
        help="completed result directory containing locked_configs.json",
    )
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--allow-budget-limited", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    suite = load_locked_suite(args.output_root)
    rows, summaries = aggregate_results(
        suite,
        allow_incomplete=args.allow_incomplete,
        allow_budget_limited=args.allow_budget_limited,
    )
    print(f"Aggregated {len(rows)} model runs into {len(summaries)} summary rows")


if __name__ == "__main__":
    main()
