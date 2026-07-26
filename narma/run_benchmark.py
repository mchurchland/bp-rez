#!/usr/bin/env python3
"""CLI for the development-only tuning and locked NARMA final benchmark."""

from __future__ import annotations

import argparse
from dataclasses import replace

from .benchmark import (
    REGIMES,
    SuiteConfig,
    aggregate_results,
    lock_selected_configs,
    run_final_pair,
    run_tuning_condition,
)
from .data import TASKS
from .models import MODEL_NAMES, ReservoirConfig
from .training import OptimizerConfig


def _add_suite_arguments(parser: argparse.ArgumentParser) -> None:
    defaults = SuiteConfig()
    reservoir = defaults.reservoir
    optimizer = defaults.optimizer
    parser.add_argument("--output-root", default=defaults.output_root)
    parser.add_argument(
        "--orders",
        type=int,
        nargs="+",
        choices=sorted(TASKS),
        default=list(defaults.orders),
    )
    parser.add_argument(
        "--regimes", nargs="+", choices=sorted(REGIMES), default=list(defaults.regimes)
    )
    parser.add_argument(
        "--models", nargs="+", choices=MODEL_NAMES, default=list(defaults.models)
    )
    parser.add_argument(
        "--final-pair-ids", type=int, nargs="+", default=list(defaults.final_pair_ids)
    )
    parser.add_argument(
        "--tuning-pair-ids", type=int, nargs="+", default=list(defaults.tuning_pair_ids)
    )
    parser.add_argument("--base-data-seed", type=int, default=defaults.base_data_seed)
    parser.add_argument("--base-model-seed", type=int, default=defaults.base_model_seed)
    parser.add_argument(
        "--small-train-length", type=int, default=defaults.small_train_length
    )
    parser.add_argument(
        "--long-train-length", type=int, default=defaults.long_train_length
    )
    parser.add_argument(
        "--validation-length", type=int, default=defaults.validation_length
    )
    parser.add_argument("--test-length", type=int, default=defaults.test_length)
    parser.add_argument("--washout", type=int, default=defaults.washout)
    parser.add_argument("--search-trials", type=int, default=defaults.search_trials)
    parser.add_argument(
        "--bootstrap-samples", type=int, default=defaults.bootstrap_samples
    )
    parser.add_argument(
        "--max-epoch-cap-fraction",
        type=float,
        default=defaults.max_epoch_cap_fraction,
        help=(
            "maximum accepted fraction of learned runs that reach max_epochs "
            "within each model/order/regime group"
        ),
    )
    parser.add_argument("--device", default=defaults.device)
    parser.add_argument("--save-checkpoints", action="store_true")
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--nodes-1", type=int, default=reservoir.nodes_1)
    parser.add_argument("--nodes-2", type=int, default=reservoir.nodes_2)
    parser.add_argument("--latent-size", type=int, default=reservoir.latent_size)
    parser.add_argument(
        "--gru-hidden-size", type=int, default=reservoir.gru_hidden_size
    )
    parser.add_argument(
        "--spectral-radius-1", type=float, default=reservoir.spectral_radius_1
    )
    parser.add_argument(
        "--spectral-radius-2", type=float, default=reservoir.spectral_radius_2
    )
    parser.add_argument("--leak-rate-1", type=float, default=reservoir.leak_rate_1)
    parser.add_argument("--leak-rate-2", type=float, default=reservoir.leak_rate_2)
    parser.add_argument("--density-1", type=float, default=reservoir.density_1)
    parser.add_argument("--density-2", type=float, default=reservoir.density_2)
    parser.add_argument("--input-scale", type=float, default=reservoir.input_scale)
    parser.add_argument(
        "--interlayer-scale", type=float, default=reservoir.interlayer_scale
    )
    parser.add_argument("--bias-scale-1", type=float, default=reservoir.bias_scale_1)
    parser.add_argument("--bias-scale-2", type=float, default=reservoir.bias_scale_2)
    parser.add_argument("--max-epochs", type=int, default=optimizer.max_epochs)
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=optimizer.early_stopping_patience,
    )
    parser.add_argument(
        "--scheduler-patience", type=int, default=optimizer.scheduler_patience
    )
    parser.add_argument(
        "--scheduler-factor", type=float, default=optimizer.scheduler_factor
    )
    parser.add_argument(
        "--minimum-learning-rate",
        type=float,
        default=optimizer.minimum_learning_rate,
    )
    parser.add_argument("--learning-rate", type=float, default=optimizer.learning_rate)
    parser.add_argument("--weight-decay", type=float, default=optimizer.weight_decay)
    parser.add_argument("--gradient-clip", type=float, default=optimizer.gradient_clip)
    parser.add_argument("--min-delta", type=float, default=optimizer.min_delta)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("tune", "tune-array", "lock", "final", "final-array", "aggregate"):
        subparser = subparsers.add_parser(name)
        _add_suite_arguments(subparser)
        if name == "tune":
            subparser.add_argument(
                "--order", type=int, required=True, choices=sorted(TASKS)
            )
            subparser.add_argument("--regime", required=True, choices=sorted(REGIMES))
            subparser.add_argument("--model", required=True, choices=MODEL_NAMES)
        elif name == "tune-array":
            subparser.add_argument("--index", type=int, required=True)
            subparser.add_argument(
                "--array-models",
                nargs="+",
                choices=MODEL_NAMES,
                help=(
                    "optional model subset used only to map this tuning-array "
                    "index; the locked suite still contains --models"
                ),
            )
        elif name == "final":
            subparser.add_argument(
                "--order", type=int, required=True, choices=sorted(TASKS)
            )
            subparser.add_argument("--regime", required=True, choices=sorted(REGIMES))
            subparser.add_argument("--pair-id", type=int, required=True)
        elif name == "final-array":
            subparser.add_argument("--index", type=int, required=True)
        elif name == "aggregate":
            subparser.add_argument("--allow-incomplete", action="store_true")
            subparser.add_argument("--allow-budget-limited", action="store_true")
    return parser.parse_args()


def _suite(args: argparse.Namespace) -> SuiteConfig:
    if args.search_trials < 1:
        raise ValueError("--search-trials must be positive")
    if args.bootstrap_samples < 1:
        raise ValueError("--bootstrap-samples must be positive")
    if not args.final_pair_ids or not args.tuning_pair_ids:
        raise ValueError("final and tuning pair ID lists must be nonempty")
    if set(args.final_pair_ids) & set(args.tuning_pair_ids):
        raise ValueError("final and tuning pair IDs must be disjoint")
    for label, values in (
        ("orders", args.orders),
        ("regimes", args.regimes),
        ("models", args.models),
        ("final pair IDs", args.final_pair_ids),
        ("tuning pair IDs", args.tuning_pair_ids),
    ):
        if len(values) != len(set(values)):
            raise ValueError(f"{label} must not contain duplicates")
    if not 0.0 <= args.max_epoch_cap_fraction <= 1.0:
        raise ValueError("--max-epoch-cap-fraction must be in [0, 1]")
    if (
        min(
            args.small_train_length,
            args.long_train_length,
            args.validation_length,
            args.test_length,
        )
        < 2
    ):
        raise ValueError("every data split must contain at least two points")
    if args.small_train_length > args.long_train_length:
        raise ValueError("small training data must be a prefix of long training data")
    if args.washout < 0 or args.washout >= min(
        args.small_train_length,
        args.validation_length,
        args.test_length,
    ):
        raise ValueError("washout must be nonnegative and shorter than every split")
    reservoir = replace(
        ReservoirConfig(),
        nodes_1=args.nodes_1,
        nodes_2=args.nodes_2,
        latent_size=args.latent_size,
        gru_hidden_size=args.gru_hidden_size,
        spectral_radius_1=args.spectral_radius_1,
        spectral_radius_2=args.spectral_radius_2,
        leak_rate_1=args.leak_rate_1,
        leak_rate_2=args.leak_rate_2,
        density_1=args.density_1,
        density_2=args.density_2,
        input_scale=args.input_scale,
        interlayer_scale=args.interlayer_scale,
        bias_scale_1=args.bias_scale_1,
        bias_scale_2=args.bias_scale_2,
    )
    optimizer = replace(
        OptimizerConfig(),
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_epochs=args.max_epochs,
        early_stopping_patience=args.early_stopping_patience,
        min_delta=args.min_delta,
        gradient_clip=args.gradient_clip,
        scheduler_patience=args.scheduler_patience,
        scheduler_factor=args.scheduler_factor,
        minimum_learning_rate=args.minimum_learning_rate,
    )
    return SuiteConfig(
        output_root=args.output_root,
        orders=tuple(args.orders),
        regimes=tuple(args.regimes),
        models=tuple(args.models),
        final_pair_ids=tuple(args.final_pair_ids),
        tuning_pair_ids=tuple(args.tuning_pair_ids),
        base_data_seed=args.base_data_seed,
        base_model_seed=args.base_model_seed,
        small_train_length=args.small_train_length,
        long_train_length=args.long_train_length,
        validation_length=args.validation_length,
        test_length=args.test_length,
        washout=args.washout,
        search_trials=args.search_trials,
        bootstrap_samples=args.bootstrap_samples,
        max_epoch_cap_fraction=args.max_epoch_cap_fraction,
        device=args.device,
        save_checkpoints=args.save_checkpoints,
        save_predictions=args.save_predictions,
        reservoir=reservoir,
        optimizer=optimizer,
    )


def _tune_index(suite: SuiteConfig, index: int) -> tuple[int, str, str]:
    return _tune_subset_index(suite, index, suite.models)


def _tune_subset_index(
    suite: SuiteConfig, index: int, models: tuple[str, ...]
) -> tuple[int, str, str]:
    if not models or any(model not in suite.models for model in models):
        raise ValueError("tuning array models must be a nonempty suite subset")
    entries = [
        (order, regime, model)
        for order in suite.orders
        for regime in suite.regimes
        for model in models
    ]
    if index < 0 or index >= len(entries):
        raise ValueError(f"tuning array index must be in [0, {len(entries) - 1}]")
    return entries[index]


def _final_index(suite: SuiteConfig, index: int) -> tuple[int, str, int]:
    entries = [
        (order, regime, pair_id)
        for order in suite.orders
        for regime in suite.regimes
        for pair_id in suite.final_pair_ids
    ]
    if index < 0 or index >= len(entries):
        raise ValueError(f"final array index must be in [0, {len(entries) - 1}]")
    return entries[index]


def main() -> None:
    args = parse_args()
    suite = _suite(args)
    if args.command == "tune":
        run_tuning_condition(
            suite, order=args.order, regime=args.regime, model_name=args.model
        )
    elif args.command == "tune-array":
        array_models = (
            tuple(args.array_models) if args.array_models is not None else suite.models
        )
        order, regime, model = _tune_subset_index(suite, args.index, array_models)
        run_tuning_condition(suite, order=order, regime=regime, model_name=model)
    elif args.command == "lock":
        locked = lock_selected_configs(suite)
        print(f"Locked configuration: {locked['locked_config_hash']}")
    elif args.command == "final":
        run_final_pair(
            suite, order=args.order, regime=args.regime, pair_id=args.pair_id
        )
    elif args.command == "final-array":
        order, regime, pair_id = _final_index(suite, args.index)
        run_final_pair(suite, order=order, regime=regime, pair_id=pair_id)
    elif args.command == "aggregate":
        rows, summaries = aggregate_results(
            suite,
            allow_incomplete=args.allow_incomplete,
            allow_budget_limited=args.allow_budget_limited,
        )
        print(f"Aggregated {len(rows)} paired model runs into {len(summaries)} rows")
    else:
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
