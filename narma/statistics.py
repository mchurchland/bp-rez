"""Pair-level uncertainty and multiple-comparison-aware NARMA statistics."""

from __future__ import annotations

import itertools
import math
from collections.abc import Iterable
from typing import Any

import numpy as np


def bootstrap_mean_interval(
    values: np.ndarray,
    *,
    samples: int,
    seed: int,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Percentile interval from whole-pair bootstrap resamples."""

    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return float("nan"), float("nan")
    if len(values) == 1:
        return float(values[0]), float(values[0])
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(samples, len(values)))
    means = values[indices].mean(axis=1)
    tail = (1.0 - confidence) / 2.0
    low, high = np.quantile(means, (tail, 1.0 - tail))
    return float(low), float(high)


def exact_sign_flip_pvalue(differences: np.ndarray) -> float:
    """Two-sided sign-flip p-value under exchangeability/symmetry."""

    values = np.asarray(differences, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan")
    observed = abs(float(values.mean()))
    if len(values) <= 20:
        extreme = 0
        total = 1 << len(values)
        for mask in range(total):
            signs = np.fromiter(
                (1.0 if mask & (1 << i) else -1.0 for i in range(len(values))),
                dtype=np.float64,
                count=len(values),
            )
            if abs(float(np.mean(signs * values))) >= observed - 1e-15:
                extreme += 1
        return extreme / total
    rng = np.random.default_rng(90210)
    signs = rng.choice((-1.0, 1.0), size=(100_000, len(values)))
    null = np.abs(np.mean(signs * values, axis=1))
    return float((np.count_nonzero(null >= observed) + 1) / (len(null) + 1))


def exact_sign_test_pvalue(differences: np.ndarray) -> float:
    """Two-sided exact binomial sign-test sensitivity, discarding exact ties."""

    values = np.asarray(differences, dtype=np.float64)
    values = values[np.isfinite(values)]
    positive = int(np.count_nonzero(values > 0.0))
    negative = int(np.count_nonzero(values < 0.0))
    count = positive + negative
    if count == 0:
        return float("nan")
    tail = min(positive, negative)
    probability = sum(math.comb(count, index) for index in range(tail + 1))
    return min(1.0, 2.0 * probability / (2**count))


def holm_adjust(pvalues: Iterable[float]) -> list[float]:
    """Holm family-wise adjusted p-values in original input order."""

    raw = np.asarray(list(pvalues), dtype=np.float64)
    result = np.full(len(raw), np.nan)
    finite_indices = np.flatnonzero(np.isfinite(raw))
    if len(finite_indices) == 0:
        return result.tolist()
    ranked = finite_indices[np.argsort(raw[finite_indices])]
    running = 0.0
    count = len(ranked)
    for rank, index in enumerate(ranked):
        adjusted = min(1.0, (count - rank) * raw[index])
        running = max(running, adjusted)
        result[index] = running
    return result.tolist()


def _json_safe(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Represent undefined/non-finite statistics as JSON null."""

    for row in rows:
        for key, value in row.items():
            if isinstance(value, (float, np.floating)) and not np.isfinite(value):
                row[key] = None
    return rows


def summarize_rows(
    rows: list[dict[str, Any]],
    *,
    bootstrap_samples: int = 10_000,
    seed: int = 20260725,
) -> list[dict[str, Any]]:
    """Aggregate one row per independent pair into model-condition summaries."""

    summaries: list[dict[str, Any]] = []

    def key(row: dict[str, Any]) -> tuple[int, str, str]:
        return row["order"], row["regime"], row["model"]

    for (order, regime, model), selected_iter in itertools.groupby(
        sorted(rows, key=key), key=key
    ):
        selected = list(selected_iter)
        mse = np.asarray([row["test_mse"] for row in selected], dtype=np.float64)
        nrmse = np.asarray([row["test_nrmse"] for row in selected], dtype=np.float64)
        mse_ci = bootstrap_mean_interval(
            mse, samples=bootstrap_samples, seed=seed + order
        )
        nrmse_ci = bootstrap_mean_interval(
            nrmse, samples=bootstrap_samples, seed=seed + order + 100
        )
        summaries.append(
            {
                "order": order,
                "task_name": selected[0]["task_name"],
                "regime": regime,
                "model": model,
                "pairs": len(selected),
                "test_mse_mean": float(mse.mean()),
                "test_mse_sd": float(mse.std(ddof=1)) if len(mse) > 1 else 0.0,
                "test_mse_ci95_low": mse_ci[0],
                "test_mse_ci95_high": mse_ci[1],
                "test_nrmse_mean": float(nrmse.mean()),
                "test_nrmse_sd": (float(nrmse.std(ddof=1)) if len(nrmse) > 1 else 0.0),
                "test_nrmse_ci95_low": nrmse_ci[0],
                "test_nrmse_ci95_high": nrmse_ci[1],
                "gradient_trained_parameters": selected[0][
                    "gradient_trained_parameters"
                ],
                "encoder_gradient_parameters": selected[0][
                    "encoder_gradient_parameters"
                ],
                "ridge_fitted_parameters": selected[0]["ridge_fitted_parameters"],
                "pca_fitted_parameters": selected[0]["pca_fitted_parameters"],
                "normalization_fitted_parameters": selected[0][
                    "normalization_fitted_parameters"
                ],
                "fit_metadata_values": selected[0]["fit_metadata_values"],
                "trainable_parameters": selected[0]["trainable_parameters"],
                "data_fitted_parameters": selected[0]["data_fitted_parameters"],
                "fixed_parameters": selected[0]["fixed_parameters"],
                "nonzero_fixed_parameters": selected[0]["nonzero_fixed_parameters"],
                "total_fixed_plus_trainable_parameters": selected[0][
                    "total_fixed_plus_trainable_parameters"
                ],
                "total_parameters": selected[0]["total_parameters"],
                "total_stored_values": selected[0]["total_stored_values"],
                "recurrent_state_size": selected[0]["recurrent_state_size"],
                "bottleneck_size": selected[0]["bottleneck_size"],
                "training_seconds_mean": float(
                    np.mean([row["training_seconds"] for row in selected])
                ),
                "ridge_fit_seconds_mean": float(
                    np.mean([row["ridge_fit_seconds"] for row in selected])
                ),
                "total_fit_seconds_mean": float(
                    np.mean(
                        [
                            row["training_seconds"] + row["ridge_fit_seconds"]
                            for row in selected
                        ]
                    )
                ),
                "inference_seconds_mean": float(
                    np.mean([row["inference_seconds"] for row in selected])
                ),
                "best_epoch_mean": float(
                    np.mean([row["best_epoch"] for row in selected])
                ),
                "epoch_cap_hits": int(
                    sum(bool(row["hit_epoch_cap"]) for row in selected)
                ),
                "failures": int(
                    sum(not bool(row.get("completed", True)) for row in selected)
                ),
            }
        )
    return summaries


def paired_comparisons(
    rows: list[dict[str, Any]],
    *,
    primary_model: str = "learned_linear",
    bootstrap_samples: int = 10_000,
    seed: int = 20260725,
) -> list[dict[str, Any]]:
    """Compare the primary model with every control using whole paired runs."""

    comparisons: list[dict[str, Any]] = []
    conditions = sorted({(row["order"], row["regime"]) for row in rows})
    for order, regime in conditions:
        condition = [
            row for row in rows if row["order"] == order and row["regime"] == regime
        ]
        primary = {
            int(row["pair_id"]): row
            for row in condition
            if row["model"] == primary_model
        }
        if not primary:
            continue
        models = sorted({row["model"] for row in condition} - {primary_model})
        for comparator in models:
            control = {
                int(row["pair_id"]): row
                for row in condition
                if row["model"] == comparator
            }
            pair_ids = sorted(set(primary) & set(control))
            if not pair_ids:
                continue
            differences = np.asarray(
                [
                    primary[pair]["test_nrmse"] - control[pair]["test_nrmse"]
                    for pair in pair_ids
                ],
                dtype=np.float64,
            )
            ratios = np.asarray(
                [
                    primary[pair]["test_nrmse"] / control[pair]["test_nrmse"]
                    for pair in pair_ids
                ],
                dtype=np.float64,
            )
            interval = bootstrap_mean_interval(
                differences,
                samples=bootstrap_samples,
                seed=seed + order * 100 + len(comparisons),
            )
            sd = float(differences.std(ddof=1)) if len(differences) > 1 else 0.0
            comparisons.append(
                {
                    "order": order,
                    "regime": regime,
                    "primary_model": primary_model,
                    "comparator": comparator,
                    "is_confirmatory_contrast": (
                        order == 10
                        and regime == "long"
                        and primary_model == "learned_linear"
                        and comparator == "deep_esn_ridge"
                    ),
                    "pairs": len(pair_ids),
                    "mean_paired_nrmse_difference": float(differences.mean()),
                    "median_paired_nrmse_difference": float(np.median(differences)),
                    "difference_ci95_low": interval[0],
                    "difference_ci95_high": interval[1],
                    "ci_multiplicity_adjustment": "none",
                    "mean_nrmse_ratio": float(ratios.mean()),
                    "primary_wins": int(np.count_nonzero(differences < 0.0)),
                    "cohens_dz": (
                        float(differences.mean() / sd)
                        if len(differences) > 1 and sd > 0.0
                        else None
                    ),
                    "permutation_p_raw": exact_sign_flip_pvalue(differences),
                    "sign_test_p_raw": exact_sign_test_pvalue(differences),
                }
            )
    global_adjusted = holm_adjust(row["permutation_p_raw"] for row in comparisons)
    for row, value in zip(comparisons, global_adjusted, strict=True):
        row["permutation_p_holm_global"] = value
    global_sign_adjusted = holm_adjust(row["sign_test_p_raw"] for row in comparisons)
    for row, value in zip(comparisons, global_sign_adjusted, strict=True):
        row["sign_test_p_holm_global"] = value
    for order, regime in conditions:
        indices = [
            index
            for index, row in enumerate(comparisons)
            if row["order"] == order and row["regime"] == regime
        ]
        within_adjusted = holm_adjust(
            comparisons[index]["permutation_p_raw"] for index in indices
        )
        within_sign_adjusted = holm_adjust(
            comparisons[index]["sign_test_p_raw"] for index in indices
        )
        for index, value, sign_value in zip(
            indices, within_adjusted, within_sign_adjusted, strict=True
        ):
            comparisons[index]["permutation_p_holm"] = value
            comparisons[index]["sign_test_p_holm"] = sign_value
            comparisons[index]["multiplicity_family"] = (
                f"primary-versus-controls within NARMA-{order}/{regime}"
            )
    return _json_safe(comparisons)


def paired_regime_comparisons(
    rows: list[dict[str, Any]],
    *,
    small_regime: str = "small",
    long_regime: str = "long",
    bootstrap_samples: int = 10_000,
    seed: int = 20260725,
) -> list[dict[str, Any]]:
    """Compare long versus small training data using matched pair IDs."""

    comparisons: list[dict[str, Any]] = []
    orders = sorted({int(row["order"]) for row in rows})
    models = sorted({str(row["model"]) for row in rows})
    for order in orders:
        for model in models:
            selected = [
                row for row in rows if row["order"] == order and row["model"] == model
            ]
            small = {
                int(row["pair_id"]): row
                for row in selected
                if row["regime"] == small_regime
            }
            long = {
                int(row["pair_id"]): row
                for row in selected
                if row["regime"] == long_regime
            }
            pair_ids = sorted(set(small) & set(long))
            if not pair_ids:
                continue
            differences = np.asarray(
                [
                    long[pair]["test_nrmse"] - small[pair]["test_nrmse"]
                    for pair in pair_ids
                ],
                dtype=np.float64,
            )
            ratios = np.asarray(
                [
                    long[pair]["test_nrmse"] / small[pair]["test_nrmse"]
                    for pair in pair_ids
                ],
                dtype=np.float64,
            )
            interval = bootstrap_mean_interval(
                differences,
                samples=bootstrap_samples,
                seed=seed + order * 100 + len(comparisons),
            )
            sd = float(differences.std(ddof=1)) if len(differences) > 1 else 0.0
            comparisons.append(
                {
                    "order": order,
                    "model": model,
                    "contrast": f"{long_regime} minus {small_regime}",
                    "pairs": len(pair_ids),
                    "mean_paired_nrmse_difference": float(differences.mean()),
                    "median_paired_nrmse_difference": float(np.median(differences)),
                    "difference_ci95_low": interval[0],
                    "difference_ci95_high": interval[1],
                    "ci_multiplicity_adjustment": "none",
                    "mean_nrmse_ratio": float(ratios.mean()),
                    "long_regime_wins": int(np.count_nonzero(differences < 0.0)),
                    "cohens_dz": (
                        float(differences.mean() / sd)
                        if len(differences) > 1 and sd > 0.0
                        else None
                    ),
                    "permutation_p_raw": exact_sign_flip_pvalue(differences),
                    "sign_test_p_raw": exact_sign_test_pvalue(differences),
                }
            )
    global_adjusted = holm_adjust(row["permutation_p_raw"] for row in comparisons)
    for row, value in zip(comparisons, global_adjusted, strict=True):
        row["permutation_p_holm_global"] = value
    global_sign_adjusted = holm_adjust(row["sign_test_p_raw"] for row in comparisons)
    for row, value in zip(comparisons, global_sign_adjusted, strict=True):
        row["sign_test_p_holm_global"] = value
    for order in orders:
        indices = [
            index for index, row in enumerate(comparisons) if row["order"] == order
        ]
        within_adjusted = holm_adjust(
            comparisons[index]["permutation_p_raw"] for index in indices
        )
        within_sign_adjusted = holm_adjust(
            comparisons[index]["sign_test_p_raw"] for index in indices
        )
        for index, value, sign_value in zip(
            indices, within_adjusted, within_sign_adjusted, strict=True
        ):
            comparisons[index]["permutation_p_holm"] = value
            comparisons[index]["sign_test_p_holm"] = sign_value
            comparisons[index]["multiplicity_family"] = (
                f"long-versus-small across models within NARMA-{order}"
            )
    return _json_safe(comparisons)


def paired_difference_in_differences(
    rows: list[dict[str, Any]],
    *,
    primary_model: str = "learned_linear",
    small_regime: str = "small",
    long_regime: str = "long",
    bootstrap_samples: int = 10_000,
    seed: int = 20260725,
) -> list[dict[str, Any]]:
    """Test whether added data changes the primary model's relative advantage."""

    comparisons: list[dict[str, Any]] = []
    orders = sorted({int(row["order"]) for row in rows})
    models = sorted({str(row["model"]) for row in rows} - {primary_model})
    for order in orders:
        order_rows = [row for row in rows if row["order"] == order]

        def make_index(model: str, regime: str) -> dict[int, dict[str, Any]]:
            return {
                int(row["pair_id"]): row
                for row in order_rows
                if row["model"] == model and row["regime"] == regime
            }

        primary_small = make_index(primary_model, small_regime)
        primary_long = make_index(primary_model, long_regime)
        if not primary_small or not primary_long:
            continue
        for comparator in models:
            control_small = make_index(comparator, small_regime)
            control_long = make_index(comparator, long_regime)
            pair_ids = sorted(
                set(primary_small)
                & set(primary_long)
                & set(control_small)
                & set(control_long)
            )
            if not pair_ids:
                continue
            differences = np.asarray(
                [
                    (
                        primary_long[pair]["test_nrmse"]
                        - control_long[pair]["test_nrmse"]
                    )
                    - (
                        primary_small[pair]["test_nrmse"]
                        - control_small[pair]["test_nrmse"]
                    )
                    for pair in pair_ids
                ],
                dtype=np.float64,
            )
            interval = bootstrap_mean_interval(
                differences,
                samples=bootstrap_samples,
                seed=seed + order * 100 + len(comparisons),
            )
            sd = float(differences.std(ddof=1)) if len(differences) > 1 else 0.0
            comparisons.append(
                {
                    "order": order,
                    "primary_model": primary_model,
                    "comparator": comparator,
                    "contrast": (
                        f"({primary_model}-{comparator})_{long_regime} minus "
                        f"({primary_model}-{comparator})_{small_regime}"
                    ),
                    "interpretation": (
                        "negative means the primary model's NRMSE advantage "
                        "increases with the longer training set"
                    ),
                    "pairs": len(pair_ids),
                    "mean_paired_difference_in_differences": float(differences.mean()),
                    "median_paired_difference_in_differences": float(
                        np.median(differences)
                    ),
                    "difference_ci95_low": interval[0],
                    "difference_ci95_high": interval[1],
                    "ci_multiplicity_adjustment": "none",
                    "cohens_dz": (
                        float(differences.mean() / sd)
                        if len(differences) > 1 and sd > 0.0
                        else None
                    ),
                    "permutation_p_raw": exact_sign_flip_pvalue(differences),
                    "sign_test_p_raw": exact_sign_test_pvalue(differences),
                }
            )
    global_adjusted = holm_adjust(row["permutation_p_raw"] for row in comparisons)
    global_sign_adjusted = holm_adjust(row["sign_test_p_raw"] for row in comparisons)
    for row, value, sign_value in zip(
        comparisons, global_adjusted, global_sign_adjusted, strict=True
    ):
        row["permutation_p_holm_global"] = value
        row["sign_test_p_holm_global"] = sign_value
    for order in orders:
        indices = [
            index for index, row in enumerate(comparisons) if row["order"] == order
        ]
        within_adjusted = holm_adjust(
            comparisons[index]["permutation_p_raw"] for index in indices
        )
        within_sign_adjusted = holm_adjust(
            comparisons[index]["sign_test_p_raw"] for index in indices
        )
        for index, value, sign_value in zip(
            indices, within_adjusted, within_sign_adjusted, strict=True
        ):
            comparisons[index]["permutation_p_holm"] = value
            comparisons[index]["sign_test_p_holm"] = sign_value
            comparisons[index]["multiplicity_family"] = (
                "primary-versus-control training-data interaction contrasts "
                f"within NARMA-{order}"
            )
    return _json_safe(comparisons)
