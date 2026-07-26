"""Tuning, locked final evaluation, cost accounting and aggregation."""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
import os
import platform
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "bp_narma_mpl"))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from common.runtime import resolve_device, seed_everything

from .data import (
    TASKS,
    NARMASplit,
    make_development_splits,
    make_paired_splits,
)
from .models import (
    GRADIENT_MODEL_NAMES,
    MODEL_NAMES,
    BenchmarkModel,
    ReservoirConfig,
    build_model,
)
from .ridge import RidgeFit, tune_and_refit_ridge
from .statistics import (
    paired_comparisons,
    paired_difference_in_differences,
    paired_regime_comparisons,
    summarize_rows,
)
from .training import OptimizerConfig, train_gradient_model


RIDGE_ALPHAS = tuple(float(value) for value in np.logspace(-12, 4, 17))


@dataclass(frozen=True)
class DataRegime:
    name: str
    train_length: int


REGIMES = {
    "small": DataRegime("small", 2_000),
    "long": DataRegime("long", 10_000),
}


@dataclass(frozen=True)
class SuiteConfig:
    """Protocol-wide settings shared by tuning and final evaluation."""

    output_root: str = "narma/results/publication_benchmark"
    orders: tuple[int, ...] = (5, 10, 20, 30)
    regimes: tuple[str, ...] = ("small", "long")
    models: tuple[str, ...] = MODEL_NAMES
    final_pair_ids: tuple[int, ...] = tuple(range(10))
    tuning_pair_ids: tuple[int, ...] = (10_000, 10_001, 10_002)
    base_data_seed: int = 8_675_309
    base_model_seed: int = 4_294_967
    small_train_length: int = 2_000
    long_train_length: int = 10_000
    validation_length: int = 2_000
    test_length: int = 10_000
    washout: int = 200
    search_trials: int = 8
    ridge_alphas: tuple[float, ...] = RIDGE_ALPHAS
    bootstrap_samples: int = 10_000
    max_epoch_cap_fraction: float = 0.1
    device: str = "auto"
    save_checkpoints: bool = False
    save_predictions: bool = False
    reservoir: ReservoirConfig = ReservoirConfig()
    optimizer: OptimizerConfig = OptimizerConfig()

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["models"] = list(self.models)
        value["orders"] = list(self.orders)
        value["regimes"] = list(self.regimes)
        value["final_pair_ids"] = list(self.final_pair_ids)
        value["tuning_pair_ids"] = list(self.tuning_pair_ids)
        value["ridge_alphas"] = list(self.ridge_alphas)
        return value


def _regime_train_length(suite: SuiteConfig, regime: str) -> int:
    if regime == "small":
        return suite.small_train_length
    if regime == "long":
        return suite.long_train_length
    raise ValueError(f"unknown data regime {regime!r}")


def _json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _json_hash(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _source_hash() -> str:
    """Hash executable benchmark sources, including uncommitted changes."""

    repository = Path(__file__).resolve().parents[1]
    paths = sorted((repository / "common").glob("*.py")) + sorted(
        (repository / "narma").glob("*.py")
    )
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(repository)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _model_seed(base_seed: int, pair_id: int) -> int:
    sequence = np.random.SeedSequence([base_seed, pair_id, 0x4D4F444C])
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def _configure_deterministic_torch() -> None:
    torch.use_deterministic_algorithms(True, warn_only=True)


def _candidate_configs(
    suite: SuiteConfig,
    *,
    order: int,
    regime: str,
    model_name: str,
) -> list[tuple[ReservoirConfig, OptimizerConfig]]:
    """Return unique, model-effective candidates with an equal trial budget."""

    candidates = [(suite.reservoir, suite.optimizer)]
    if suite.search_trials == 1:
        return candidates
    model_index = MODEL_NAMES.index(model_name)
    regime_index = suite.regimes.index(regime)
    rng = np.random.default_rng(
        np.random.SeedSequence(
            [suite.base_model_seed, order, regime_index, model_index, 0x54554E45]
        )
    )
    radii = (0.5, 0.7, 0.9, 0.99, 1.1)
    leaks = (0.1, 0.3, 0.6, 1.0)
    densities = (0.05, 0.1, 0.2)
    input_scales = (0.1, 0.3, 0.5, 1.0)
    interlayer_scales = (0.1, 0.5, 1.0, 2.0)
    bias_scales = (0.0, 0.1, 0.3)
    learning_rates = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2)
    weight_decays = (0.0, 1e-6, 1e-5, 1e-4, 1e-3)

    def choose(values: tuple[float, ...]) -> float:
        return float(values[int(rng.integers(0, len(values)))])

    if model_name == "gru":
        default_optimizer = (
            suite.optimizer.learning_rate,
            suite.optimizer.weight_decay,
        )
        optimizer_grid = [
            pair
            for pair in itertools.product(learning_rates, weight_decays)
            if pair != default_optimizer
        ]
        rng.shuffle(optimizer_grid)
        if suite.search_trials - 1 > len(optimizer_grid):
            raise ValueError(
                "the GRU supports at most "
                f"{len(optimizer_grid) + 1} unique optimizer trials"
            )
        for learning_rate, weight_decay in optimizer_grid[: suite.search_trials - 1]:
            candidates.append(
                (
                    suite.reservoir,
                    replace(
                        suite.optimizer,
                        learning_rate=learning_rate,
                        weight_decay=weight_decay,
                    ),
                )
            )
        return candidates

    def signature(
        reservoir: ReservoirConfig, optimizer: OptimizerConfig
    ) -> tuple[float, ...]:
        first_layer = (
            reservoir.spectral_radius_1,
            reservoir.leak_rate_1,
            reservoir.density_1,
            reservoir.input_scale,
            reservoir.bias_scale_1,
        )
        if model_name in {"esn_ridge", "large_esn_ridge"}:
            return first_layer
        both_layers = first_layer + (
            reservoir.spectral_radius_2,
            reservoir.leak_rate_2,
            reservoir.density_2,
            reservoir.interlayer_scale,
            reservoir.bias_scale_2,
        )
        if model_name in {"learned_linear", "learned_nonlinear"}:
            return both_layers + (
                optimizer.learning_rate,
                optimizer.weight_decay,
            )
        return both_layers

    seen = {signature(*candidates[0])}
    attempts = 0
    while len(candidates) < suite.search_trials:
        attempts += 1
        if attempts > 10_000:
            raise RuntimeError("could not sample enough unique tuning candidates")
        first_layer = {
            "spectral_radius_1": choose(radii),
            "leak_rate_1": choose(leaks),
            "density_1": choose(densities),
            "input_scale": choose(input_scales),
            "bias_scale_1": choose(bias_scales),
        }
        if model_name in {"esn_ridge", "large_esn_ridge"}:
            reservoir = replace(suite.reservoir, **first_layer)
        else:
            reservoir = replace(
                suite.reservoir,
                **first_layer,
                spectral_radius_2=choose(radii),
                leak_rate_2=choose(leaks),
                density_2=choose(densities),
                interlayer_scale=choose(interlayer_scales),
                bias_scale_2=choose(bias_scales),
            )
        optimizer = suite.optimizer
        if model_name in {"learned_linear", "learned_nonlinear"}:
            optimizer = replace(
                optimizer,
                learning_rate=choose(learning_rates),
                weight_decay=choose(weight_decays),
            )
        key = signature(reservoir, optimizer)
        if key in seen:
            continue
        seen.add(key)
        candidates.append((reservoir, optimizer))
    return candidates


def _target_metrics(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    prediction = prediction.detach().cpu().to(torch.float64).reshape(-1)
    target = target.detach().cpu().to(torch.float64).reshape(-1)
    mse = float(torch.mean((prediction - target) ** 2))
    variance = float(torch.var(target, unbiased=False))
    nrmse = math.sqrt(mse / variance) if variance > 0.0 else float("nan")
    return {"mse": mse, "nrmse": nrmse, "target_variance": variance}


def _extract_features(
    model: BenchmarkModel,
    split: NARMASplit,
    *,
    washout: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    with torch.no_grad():
        features = model.features(split.u.to(device))[washout:].detach().cpu()
    return features, split.y[washout:].detach().cpu()


def _prepare_and_train(
    model: BenchmarkModel,
    train: NARMASplit,
    validation: NARMASplit,
    *,
    washout: int,
    optimizer_config: OptimizerConfig,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, list[float]]]:
    model.to(device)
    if model.gradient_model:
        return train_gradient_model(
            model,
            train,
            validation,
            washout=washout,
            optimizer_config=optimizer_config,
            device=device,
        )
    _synchronize(device)
    started = time.perf_counter()
    model.prepare(train.u.to(device), washout)
    _synchronize(device)
    preparation_seconds = time.perf_counter() - started
    information, history = train_gradient_model(
        model,
        train,
        validation,
        washout=washout,
        optimizer_config=optimizer_config,
        device=device,
    )
    information["preprocessor_fit_seconds"] = preparation_seconds
    information["training_seconds"] = preparation_seconds
    return information, history


def _fit_readout(
    model: BenchmarkModel,
    train: NARMASplit,
    validation: NARMASplit,
    *,
    washout: int,
    alphas: tuple[float, ...],
    device: torch.device,
) -> RidgeFit:
    started = time.perf_counter()
    train_features, train_target = _extract_features(
        model, train, washout=washout, device=device
    )
    validation_features, validation_target = _extract_features(
        model, validation, washout=washout, device=device
    )
    fit = tune_and_refit_ridge(
        train_features,
        train_target,
        validation_features,
        validation_target,
        alphas,
    )
    model.set_readout(fit.weight, fit.bias)
    return replace(fit, fit_seconds=time.perf_counter() - started)


def _validation_nrmse_from_tuning_fit(
    fit: RidgeFit, validation: NARMASplit, washout: int
) -> float:
    variance = float(
        torch.var(validation.y[washout:].to(torch.float64), unbiased=False)
    )
    return math.sqrt(fit.validation_mse / variance) if variance > 0.0 else float("nan")


def tuning_path(root: Path, order: int, regime: str, model_name: str) -> Path:
    return root / "tuning" / f"order_{order}" / regime / model_name


def run_tuning_condition(
    suite: SuiteConfig,
    *,
    order: int,
    regime: str,
    model_name: str,
) -> dict[str, Any]:
    """Tune one model-condition without constructing or evaluating test data."""

    if order not in suite.orders or order not in TASKS:
        raise ValueError(f"order {order} is not part of this suite")
    if regime not in suite.regimes or regime not in REGIMES:
        raise ValueError(f"unknown regime {regime!r}")
    if model_name not in suite.models:
        raise ValueError(f"model {model_name!r} is not part of this suite")
    root = Path(suite.output_root)
    run_root = tuning_path(root, order, regime, model_name)
    run_root.mkdir(parents=True, exist_ok=True)
    candidates = _candidate_configs(
        suite, order=order, regime=regime, model_name=model_name
    )
    run_manifest = {
        "order": order,
        "regime": regime,
        "model": model_name,
        "suite_protocol_hash": _json_hash(suite.to_dict()),
        "source_hash": _source_hash(),
        "candidates": [
            {
                "reservoir": reservoir.to_dict(),
                "optimizer": optimizer.to_dict(),
            }
            for reservoir, optimizer in candidates
        ],
    }
    manifest_path = run_root / "run_manifest.json"
    if manifest_path.is_file():
        existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing_manifest != run_manifest:
            raise RuntimeError(
                f"{run_root} contains tuning shards from a different source "
                "or suite; use a new output directory"
            )
    else:
        _json_dump(manifest_path, run_manifest)
    trials_path = run_root / "trials.json"
    trials: list[dict[str, Any]] = []
    if trials_path.is_file():
        trials = json.loads(trials_path.read_text(encoding="utf-8"))
        if not isinstance(trials, list) or len(trials) > len(candidates):
            raise RuntimeError(f"invalid resumable tuning state at {trials_path}")
        for trial_index, trial in enumerate(trials):
            reservoir, optimizer = candidates[trial_index]
            valid = (
                trial.get("trial") == trial_index
                and trial.get("reservoir") == reservoir.to_dict()
                and trial.get("optimizer") == optimizer.to_dict()
                and len(trial.get("pairs", ())) == len(suite.tuning_pair_ids)
            )
            if not valid:
                raise RuntimeError(
                    f"stale or incomplete resumable trial {trial_index} at "
                    f"{trials_path}"
                )
    device = resolve_device(suite.device)
    _configure_deterministic_torch()
    development_splits: dict[int, dict[str, NARMASplit]] = {}
    if len(trials) < len(candidates):
        development_splits = {
            tuning_pair_id: make_development_splits(
                order,
                train_length=_regime_train_length(suite, regime),
                long_train_length=suite.long_train_length,
                validation_length=suite.validation_length,
                base_seed=suite.base_data_seed,
                tuning_pair_id=tuning_pair_id,
            )
            for tuning_pair_id in suite.tuning_pair_ids
        }
    for trial_index in range(len(trials), len(candidates)):
        reservoir, optimizer = candidates[trial_index]
        pair_scores: list[float] = []
        pair_details: list[dict[str, Any]] = []
        for tuning_pair_id in suite.tuning_pair_ids:
            splits = development_splits[tuning_pair_id]
            model_seed = _model_seed(suite.base_model_seed, tuning_pair_id)
            seed_everything(model_seed)
            model = build_model(model_name, reservoir, model_seed)
            training, _ = _prepare_and_train(
                model,
                splits["train"],
                splits["validation"],
                washout=suite.washout,
                optimizer_config=optimizer,
                device=device,
            )
            ridge = _fit_readout(
                model,
                splits["train"],
                splits["validation"],
                washout=suite.washout,
                alphas=suite.ridge_alphas,
                device=device,
            )
            validation_nrmse = _validation_nrmse_from_tuning_fit(
                ridge, splits["validation"], suite.washout
            )
            pair_scores.append(validation_nrmse)
            pair_details.append(
                {
                    "tuning_pair_id": tuning_pair_id,
                    "model_seed": model_seed,
                    "validation_nrmse": validation_nrmse,
                    "validation_mse": ridge.validation_mse,
                    "selected_ridge_alpha": ridge.alpha,
                    "ridge_fit_seconds": ridge.fit_seconds,
                    "train_manifest": splits["train"].manifest(),
                    "validation_manifest": splits["validation"].manifest(),
                    **training,
                }
            )
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        trial = {
            "trial": trial_index,
            "mean_validation_nrmse": float(np.mean(pair_scores)),
            "sd_validation_nrmse": (
                float(np.std(pair_scores, ddof=1)) if len(pair_scores) > 1 else 0.0
            ),
            "reservoir": reservoir.to_dict(),
            "optimizer": optimizer.to_dict(),
            "pairs": pair_details,
        }
        trials.append(trial)
        _json_dump(trials_path, trials)
        print(
            f"[tune] order={order} regime={regime} model={model_name} "
            f"trial={trial_index + 1}/{suite.search_trials} "
            f"validation NRMSE={trial['mean_validation_nrmse']:.6g}",
            flush=True,
        )
    winner = min(trials, key=lambda row: row["mean_validation_nrmse"])
    selected = {
        "order": order,
        "task": TASKS[order].to_dict(),
        "benchmark_variant_name": TASKS[order].benchmark_variant_name,
        "regime": regime,
        "model": model_name,
        "selected_trial": winner["trial"],
        "mean_validation_nrmse": winner["mean_validation_nrmse"],
        "reservoir": winner["reservoir"],
        "optimizer": winner["optimizer"],
        "search_trials": suite.search_trials,
        "tuning_pair_ids": list(suite.tuning_pair_ids),
        "epoch_cap_hits": int(
            sum(bool(pair["hit_epoch_cap"]) for pair in winner["pairs"])
        ),
        "tuning_runs": len(winner["pairs"]),
        "epoch_cap_fraction": float(
            np.mean([bool(pair["hit_epoch_cap"]) for pair in winner["pairs"]])
        ),
        "test_data_constructed": False,
        "suite_protocol_hash": _json_hash(suite.to_dict()),
        "source_hash": _source_hash(),
    }
    selected["selection_hash"] = _json_hash(selected)
    _json_dump(run_root / "selected.json", selected)
    return selected


def lock_selected_configs(suite: SuiteConfig) -> dict[str, Any]:
    """Validate the tuning matrix and create one immutable selection manifest."""

    root = Path(suite.output_root)
    selected: dict[str, Any] = {}
    missing: list[str] = []
    stale: list[str] = []
    budget_limited: list[str] = []
    suite_hash = _json_hash(suite.to_dict())
    source_hash = _source_hash()
    for order in suite.orders:
        for regime in suite.regimes:
            for model_name in suite.models:
                path = tuning_path(root, order, regime, model_name) / "selected.json"
                key = f"{order}:{regime}:{model_name}"
                if not path.is_file():
                    missing.append(str(path))
                    continue
                value = json.loads(path.read_text(encoding="utf-8"))
                recorded_hash = value.pop("selection_hash", None)
                valid = (
                    value.get("order") == order
                    and value.get("regime") == regime
                    and value.get("model") == model_name
                    and value.get("suite_protocol_hash") == suite_hash
                    and value.get("source_hash") == source_hash
                    and recorded_hash == _json_hash(value)
                )
                if not valid:
                    stale.append(str(path))
                    continue
                value["selection_hash"] = recorded_hash
                selected[key] = value
                if (
                    model_name in GRADIENT_MODEL_NAMES
                    and float(value["epoch_cap_fraction"])
                    > suite.max_epoch_cap_fraction
                ):
                    budget_limited.append(
                        f"{key}: {value['epoch_cap_hits']}/"
                        f"{value['tuning_runs']} selected tuning runs hit "
                        "the epoch cap"
                    )
    if missing:
        raise RuntimeError(
            "cannot lock an incomplete tuning matrix; missing:\n" + "\n".join(missing)
        )
    if stale:
        raise RuntimeError(
            "cannot lock stale or tampered tuning selections:\n" + "\n".join(stale)
        )
    if budget_limited:
        raise RuntimeError(
            "cannot lock budget-limited learned configurations; increase "
            "--max-epochs (or deliberately relax "
            "--max-epoch-cap-fraction):\n" + "\n".join(budget_limited)
        )
    payload = {
        "suite": suite.to_dict(),
        "suite_protocol_hash": suite_hash,
        "source_hash": source_hash,
        "git_commit": _git_commit(),
        "selected": selected,
    }
    payload["locked_config_hash"] = _json_hash(payload)
    _json_dump(root / "locked_configs.json", payload)
    _write_protocol(root, suite, payload["locked_config_hash"])
    return payload


def _load_locked(suite: SuiteConfig) -> dict[str, Any]:
    path = Path(suite.output_root) / "locked_configs.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} does not exist; finish tuning and run the lock phase first"
        )
    locked = json.loads(path.read_text(encoding="utf-8"))
    recorded_lock_hash = locked.pop("locked_config_hash", None)
    if recorded_lock_hash != _json_hash(locked):
        raise RuntimeError(f"{path} failed its configuration-integrity check")
    locked["locked_config_hash"] = recorded_lock_hash
    if locked.get("source_hash") != _source_hash():
        raise RuntimeError(
            "benchmark source files changed after configuration locking; use "
            "a new output directory and repeat tuning"
        )
    if locked["suite_protocol_hash"] != _json_hash(suite.to_dict()):
        raise RuntimeError(
            "the requested suite does not match locked_configs.json; use the "
            "same arguments as tuning or create a new output directory"
        )
    return locked


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _evaluate_test_once(
    model: BenchmarkModel,
    split: NARMASplit,
    *,
    washout: int,
    device: torch.device,
) -> tuple[dict[str, float], np.ndarray]:
    model.eval()
    _synchronize(device)
    start = time.perf_counter()
    with torch.no_grad():
        prediction = model(split.u.to(device))[washout:]
    _synchronize(device)
    elapsed = time.perf_counter() - start
    target = split.y[washout:].to(device)
    metrics = _target_metrics(prediction, target)
    if not all(np.isfinite(value) for value in metrics.values()):
        raise FloatingPointError(f"non-finite test metrics: {metrics}")
    metrics["inference_seconds"] = elapsed
    metrics["steps_per_second"] = len(prediction) / elapsed
    return metrics, prediction.detach().cpu().numpy().squeeze(-1)


def _nonzero_fixed_coefficients(model: BenchmarkModel) -> int:
    return sum(
        int(torch.count_nonzero(value).item())
        for name, value in model.named_buffers()
        if name.startswith(("A", "B", "V", "Q", "R", "q"))
    )


def _costs(model: BenchmarkModel) -> dict[str, int]:
    encoder_gradient = model.encoder_gradient_parameters
    joint_gradient = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    normalization = model.normalization_statistics
    ridge = model.ridge_parameters
    pca = model.pca_fitted_parameters
    trainable = encoder_gradient + ridge
    data_fitted = trainable + pca
    fixed = model.fixed_coefficient_count
    return {
        "gradient_trained_parameters": joint_gradient,
        "encoder_gradient_parameters": encoder_gradient,
        "ridge_fitted_parameters": ridge,
        "pca_fitted_parameters": pca,
        "normalization_fitted_parameters": normalization,
        "fit_metadata_values": 2 * model.feature_size,
        "trainable_parameters": trainable,
        "data_fitted_parameters": data_fitted,
        "fixed_parameters": fixed,
        "total_fixed_plus_trainable_parameters": trainable + fixed,
        "total_parameters": data_fitted + fixed,
        "total_stored_values": data_fitted + fixed + normalization,
        "nonzero_fixed_parameters": _nonzero_fixed_coefficients(model),
        "recurrent_state_size": model.recurrent_state_size,
        "bottleneck_size": model.bottleneck_size,
    }


def final_pair_path(root: Path, order: int, regime: str, pair_id: int) -> Path:
    return root / "final" / f"order_{order}" / regime / f"pair_{pair_id:02d}"


def run_final_pair(
    suite: SuiteConfig,
    *,
    order: int,
    regime: str,
    pair_id: int,
) -> list[dict[str, Any]]:
    """Run all models on one paired dataset after configurations are locked."""

    if pair_id not in suite.final_pair_ids:
        raise ValueError(f"pair_id {pair_id} is not part of the final protocol")
    locked = _load_locked(suite)
    root = Path(suite.output_root)
    pair_root = final_pair_path(root, order, regime, pair_id)
    pair_root.mkdir(parents=True, exist_ok=True)
    completion_path = pair_root / "complete.json"
    if completion_path.is_file():
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        if completion.get("locked_config_hash") == locked["locked_config_hash"]:
            return [
                json.loads(
                    (pair_root / model_name / "metrics.json").read_text(
                        encoding="utf-8"
                    )
                )
                for model_name in suite.models
            ]
        raise RuntimeError(f"stale completion marker at {completion_path}")

    splits = make_paired_splits(
        order,
        train_length=_regime_train_length(suite, regime),
        long_train_length=suite.long_train_length,
        validation_length=suite.validation_length,
        test_length=suite.test_length,
        base_seed=suite.base_data_seed,
        pair_id=pair_id,
    )
    data_manifest = {
        "order": order,
        "task": TASKS[order].to_dict(),
        "benchmark_variant_name": TASKS[order].benchmark_variant_name,
        "regime": regime,
        "pair_id": pair_id,
        "washout": suite.washout,
        "state_initialization": "all recurrent and NARMA target states reset to zero",
        "alignment": "u[t] is consumed before predicting y[t+1]",
        "storage_dtype": "float32",
        "generation_dtype": "float64",
        "input_proposal_rng": "NumPy Generator(PCG64)",
        "acceptance_policy": (
            "first deterministic iid-uniform proposal whose generated target "
            "is finite and |y[t]| <= 1e6 over the recorded acceptance horizon"
        ),
        "splits": {name: split.manifest() for name, split in splits.items()},
    }
    _json_dump(pair_root / "data_manifest.json", data_manifest)
    device = resolve_device(suite.device)
    _configure_deterministic_torch()
    rows: list[dict[str, Any]] = []
    model_seed = _model_seed(suite.base_model_seed, pair_id)
    for model_name in suite.models:
        selected_key = f"{order}:{regime}:{model_name}"
        selected = locked["selected"][selected_key]
        reservoir = ReservoirConfig(**selected["reservoir"])
        optimizer = OptimizerConfig(**selected["optimizer"])
        model_root = pair_root / model_name
        model_root.mkdir(parents=True, exist_ok=True)
        metrics_path = model_root / "metrics.json"
        required_artifacts = [metrics_path, model_root / "history.json"]
        if suite.save_predictions:
            required_artifacts.append(model_root / "predictions.npz")
        if suite.save_checkpoints:
            required_artifacts.append(model_root / "checkpoint.pt")
        if metrics_path.is_file():
            previous = json.loads(metrics_path.read_text(encoding="utf-8"))
            current_identity = (
                previous.get("locked_config_hash") == locked["locked_config_hash"]
                and previous.get("selection_hash") == selected["selection_hash"]
                and previous.get("data_test_sha256") == splits["test"].digest()
                and previous.get("model") == model_name
                and int(previous.get("pair_id", -1)) == pair_id
            )
            if not current_identity:
                raise RuntimeError(f"stale result shard at {metrics_path}")
            if all(path.is_file() for path in required_artifacts):
                rows.append(previous)
                print(
                    f"[final/resume] order={order} regime={regime} "
                    f"pair={pair_id} model={model_name}",
                    flush=True,
                )
                continue
        seed_everything(model_seed)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        model = build_model(model_name, reservoir, model_seed)
        training, history = _prepare_and_train(
            model,
            splits["train"],
            splits["validation"],
            washout=suite.washout,
            optimizer_config=optimizer,
            device=device,
        )
        ridge = _fit_readout(
            model,
            splits["train"],
            splits["validation"],
            washout=suite.washout,
            alphas=suite.ridge_alphas,
            device=device,
        )
        test, prediction = _evaluate_test_once(
            model,
            splits["test"],
            washout=suite.washout,
            device=device,
        )
        peak_gpu = (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        )
        row: dict[str, Any] = {
            "completed": True,
            "order": order,
            "task_name": TASKS[order].benchmark_variant_name,
            "recurrence_name": TASKS[order].name,
            "regime": regime,
            "pair_id": pair_id,
            "model": model_name,
            "model_seed": model_seed,
            "locked_config_hash": locked["locked_config_hash"],
            "selection_hash": selected["selection_hash"],
            "data_test_sha256": splits["test"].digest(),
            "device": str(device),
            "torch_version": str(torch.__version__),
            "python_version": platform.python_version(),
            "hostname": platform.node(),
            "selected_ridge_alpha": ridge.alpha,
            "ridge_validation_mse_before_train_val_refit": ridge.validation_mse,
            "ridge_fit_seconds": ridge.fit_seconds,
            "test_mse": test["mse"],
            "test_nrmse": test["nrmse"],
            "test_target_variance": test["target_variance"],
            "inference_seconds": test["inference_seconds"],
            "test_steps_per_second": test["steps_per_second"],
            "inference_seconds_per_1000_steps": (1_000.0 / test["steps_per_second"]),
            "peak_gpu_memory_bytes": peak_gpu,
            **_costs(model),
            **training,
        }
        _json_dump(metrics_path, row)
        _json_dump(model_root / "history.json", history)
        if suite.save_predictions:
            np.savez_compressed(
                model_root / "predictions.npz",
                prediction=prediction,
                target=splits["test"].y[suite.washout :].numpy().squeeze(-1),
                input=splits["test"].u[suite.washout :].numpy().squeeze(-1),
                washout=np.asarray(suite.washout),
            )
        if suite.save_checkpoints:
            torch.save(
                {
                    "model": model_name,
                    "order": order,
                    "regime": regime,
                    "pair_id": pair_id,
                    "model_seed": model_seed,
                    "locked_config_hash": locked["locked_config_hash"],
                    "reservoir": reservoir.to_dict(),
                    "optimizer": optimizer.to_dict(),
                    "state_dict": {
                        name: value.detach().cpu()
                        for name, value in model.state_dict().items()
                    },
                    "ridge": {
                        "alpha": ridge.alpha,
                        "feature_mean": ridge.feature_mean,
                        "feature_scale": ridge.feature_scale,
                    },
                },
                model_root / "checkpoint.pt",
            )
        rows.append(row)
        print(
            f"[final] order={order} regime={regime} pair={pair_id} "
            f"model={model_name} NRMSE={row['test_nrmse']:.6g}",
            flush=True,
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    _json_dump(
        completion_path,
        {
            "locked_config_hash": locked["locked_config_hash"],
            "models": list(suite.models),
            "test_sha256": splits["test"].digest(),
        },
    )
    return rows


def _write_protocol(root: Path, suite: SuiteConfig, locked_hash: str) -> None:
    protocol = {
        "locked_config_hash": locked_hash,
        "primary_metric": "test NRMSE",
        "metric_definitions": {
            "MSE": "mean((prediction - target)^2) after washout",
            "NRMSE": "sqrt(MSE / population_variance(target)) after washout",
            "summary_NRMSE": "arithmetic mean of independently computed pair NRMSEs",
            "uncertainty_unit": "paired end-to-end replicate (pair_id)",
            "confidence_interval": (
                f"unadjusted descriptive 95% percentile bootstrap over whole "
                f"pairs, {suite.bootstrap_samples} resamples"
            ),
            "confirmatory_contrast": (
                "learned_linear versus deep_esn_ridge on NARMA-10/long"
            ),
            "paired_test": (
                "two-sided exact sign-flip test, interpreted under symmetric "
                "paired differences or label-exchangeability"
            ),
            "sensitivity_test": (
                "two-sided exact binomial sign test, which does not use "
                "difference magnitudes and tests a 0.5 sign probability after "
                "discarding exact ties"
            ),
            "multiplicity": (
                "primary reported Holm correction is across the seven "
                "learned_linear-versus-control contrasts within each "
                "order/regime family; a global 56-contrast Holm value is also "
                "reported as supplementary"
            ),
            "regime_comparison": (
                "paired long-minus-small NRMSE per model; Holm correction "
                "across eight models within each NARMA order, with a "
                "supplementary global correction"
            ),
            "training_data_interaction": (
                "paired difference-in-differences: "
                "(learned_linear-control)_long minus "
                "(learned_linear-control)_small; Holm correction across seven "
                "controls within each order, with a supplementary global "
                "correction"
            ),
            "family_scope": (
                "model, regime and interaction tables are separate secondary "
                "analysis families and are not claimed to provide one "
                "study-wide FWER adjustment"
            ),
        },
        "input_target_alignment": "consume u[t], then predict y[t+1]",
        "initialization": {
            "narma_target_history": "zero",
            "reservoir_state": "zero at the start of every independent split",
            "recurrent_matrix": (
                "Uniform(-1,1), Bernoulli sparsity mask, then exact spectral "
                "radius scaling; named RNG streams reproduce the same raw "
                "draws when models use the same reservoir configuration"
            ),
            "input_and_interlayer_projection": (
                "fan-in-scaled symmetric uniform distribution"
            ),
            "fixed_bias": (
                "fan-in-scaled symmetric uniform distribution when its "
                "validation-selected scale is nonzero"
            ),
        },
        "determinism": (
            "named RNG streams, seeded Python/NumPy/PyTorch, deterministic "
            "PyTorch algorithms requested with warnings, and Slurm sets "
            "PYTHONHASHSEED plus CUBLAS_WORKSPACE_CONFIG"
        ),
        "split_policy": (
            "development tuning creates only train/validation; final test is "
            "constructed only after hyperparameters are hashed and locked"
        ),
        "data_acceptance_policy": {
            "proposal": (
                "each input sample is independently proposed from the task's "
                "stated NumPy-PCG64 uniform distribution"
            ),
            "acceptance": (
                "model-blind deterministic rejection sampling: accept the "
                "first derived seed whose target remains finite and "
                "|y[t]| <= 1e6 over the complete split horizon"
            ),
            "accepted_distribution": (
                "the proposal input distribution conditioned on the target "
                "acceptance event; it is not claimed to remain exactly uniform"
            ),
            "small_regime": (
                "the small training set is a prefix of a stream accepted over "
                "the common acceptance horizon and is therefore conditioned "
                "on stability beyond its observed portion"
            ),
            "common_acceptance_horizon": max(
                suite.long_train_length,
                suite.validation_length,
                suite.test_length,
            ),
            "audit": (
                "requested/realized seeds, rejected seeds, attempts and "
                "acceptance horizons are saved in every data manifest"
            ),
        },
        "hyperparameter_pairing": (
            "each model is validation-tuned independently using the same "
            "number of unique, model-effective trials. Named RNG streams pair "
            "raw matrices only when the selected hyperparameter configurations "
            "are identical; tuned final models may select different scales, "
            "densities or spectral radii"
        ),
        "convergence_gate": (
            "configuration locking and final aggregation fail if the fraction "
            "of epoch-cap hits in any learned-model/order/regime group exceeds "
            f"{suite.max_epoch_cap_fraction:g}"
        ),
        "data_regimes": {
            name: {
                "name": name,
                "generated_train_length": _regime_train_length(suite, name),
                "post_washout_training_observations": (
                    _regime_train_length(suite, name) - suite.washout
                ),
            }
            for name in suite.regimes
        },
        "long_train_length": suite.long_train_length,
        "validation": {
            "generated_length": suite.validation_length,
            "post_washout_observations": (suite.validation_length - suite.washout),
        },
        "final_test": {
            "generated_length": suite.test_length,
            "post_washout_observations": suite.test_length - suite.washout,
        },
        "washout": suite.washout,
        "input_normalization": "none",
        "feature_normalization": (
            "ridge features standardized with training statistics during "
            "selection and train+validation statistics for the locked refit; "
            "first-reservoir bottleneck inputs use training-only statistics"
        ),
        "narma_tasks": {str(order): TASKS[order].to_dict() for order in suite.orders},
        "cross_order_warning": (
            "The cited NARMA orders use different coefficients, input ranges "
            "and, for NARMA-20, an outer tanh. Cross-order scores are not a "
            "pure measure of increasing memory length."
        ),
    }
    _json_dump(root / "protocol.json", protocol)
    lines = [
        "# Locked NARMA benchmark protocol",
        "",
        f"Configuration hash: `{locked_hash}`",
        "",
        "Each independently reset sequence consumes `u[t]` before predicting "
        "`y[t+1]`. MSE and NRMSE exclude the first "
        f"{suite.washout} samples. NRMSE is "
        "`sqrt(MSE / population_variance(target))`.",
        "",
        "The final test streams are constructed only by the final phase, after "
        "the selected configurations have been locked.",
        "",
        "## Task definitions",
        "",
    ]
    for order in suite.orders:
        task = TASKS[order]
        lines.extend(
            [
                f"### {task.name}",
                "",
                f"`{task.equation}`",
                "",
                "Each proposal has "
                f"`u[t] iid ~ Uniform({task.input_low:g}, {task.input_high:g})` "
                "from NumPy PCG64. "
                f"Source: [{task.citation}]({task.citation_url}).",
                "",
            ]
        )
    lines.extend(
        [
            "For unbounded recurrences, the benchmark accepts the first "
            "deterministically derived proposal whose target remains finite "
            "and `|y[t]| <= 1e6` over the complete split. Accepted inputs are "
            "therefore sampled from the stated proposal distribution "
            "**conditioned on this target-stability event**. Every rejected "
            "seed and acceptance horizon is recorded. Every stream is accepted "
            "over one common horizon equal to the maximum requested split "
            "length, and requested datasets are prefixes of those accepted "
            "streams.",
            "",
            "The definitions differ across orders; their errors must not be "
            "interpreted as a pure order/memory scaling curve.",
            "",
        ]
    )
    (root / "PROTOCOL.md").write_text("\n".join(lines), encoding="utf-8")


def _save_summary_plot(summaries: list[dict[str, Any]], path: Path) -> None:
    orders = sorted({int(row["order"]) for row in summaries})
    regimes = sorted({str(row["regime"]) for row in summaries})
    figure, axes = plt.subplots(
        len(regimes),
        len(orders),
        figsize=(4.3 * len(orders), 4.0 * len(regimes)),
        squeeze=False,
    )
    for regime_index, regime in enumerate(regimes):
        for order_index, order in enumerate(orders):
            axis = axes[regime_index][order_index]
            selected = [
                row
                for row in summaries
                if row["regime"] == regime and row["order"] == order
            ]
            selected.sort(key=lambda row: row["test_nrmse_mean"])
            labels = [row["model"].replace("_", "\n") for row in selected]
            means = [row["test_nrmse_mean"] for row in selected]
            lower = [
                max(0.0, row["test_nrmse_mean"] - row["test_nrmse_ci95_low"])
                for row in selected
            ]
            upper = [
                max(0.0, row["test_nrmse_ci95_high"] - row["test_nrmse_mean"])
                for row in selected
            ]
            axis.bar(
                np.arange(len(selected)),
                means,
                yerr=np.asarray((lower, upper)),
                capsize=3,
            )
            axis.set_xticks(np.arange(len(selected)), labels, fontsize=7)
            axis.set_title(f"NARMA-{order}, {regime}")
            axis.set_ylabel("Test NRMSE (paired-bootstrap 95% CI)")
            axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _convergence_report(
    rows: list[dict[str, Any]], suite: SuiteConfig
) -> dict[str, Any]:
    groups: list[dict[str, Any]] = []
    violations: list[str] = []
    for order in suite.orders:
        for regime in suite.regimes:
            for model_name in GRADIENT_MODEL_NAMES:
                if model_name not in suite.models:
                    continue
                selected = [
                    row
                    for row in rows
                    if row["order"] == order
                    and row["regime"] == regime
                    and row["model"] == model_name
                ]
                if not selected:
                    continue
                hits = sum(bool(row["hit_epoch_cap"]) for row in selected)
                fraction = hits / len(selected)
                accepted = fraction <= suite.max_epoch_cap_fraction
                groups.append(
                    {
                        "order": order,
                        "regime": regime,
                        "model": model_name,
                        "runs": len(selected),
                        "epoch_cap_hits": hits,
                        "epoch_cap_fraction": fraction,
                        "maximum_accepted_fraction": (suite.max_epoch_cap_fraction),
                        "accepted": accepted,
                    }
                )
                if not accepted:
                    violations.append(
                        f"NARMA-{order}/{regime}/{model_name}: "
                        f"{hits}/{len(selected)} runs hit the epoch cap"
                    )
    return {
        "criterion": (
            "within every learned-model/order/regime group, the fraction of "
            "runs reaching max_epochs must not exceed "
            f"{suite.max_epoch_cap_fraction:g}"
        ),
        "accepted": not violations,
        "violations": violations,
        "groups": groups,
    }


def aggregate_results(
    suite: SuiteConfig,
    *,
    allow_incomplete: bool = False,
    allow_budget_limited: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Validate all final shards, then write tables, tests and a summary plot."""

    locked = _load_locked(suite)
    root = Path(suite.output_root)
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for order in suite.orders:
        for regime in suite.regimes:
            for pair_id in suite.final_pair_ids:
                pair_root = final_pair_path(root, order, regime, pair_id)
                for model_name in suite.models:
                    path = pair_root / model_name / "metrics.json"
                    if not path.is_file():
                        missing.append(str(path))
                        continue
                    row = json.loads(path.read_text(encoding="utf-8"))
                    if row["locked_config_hash"] != locked["locked_config_hash"]:
                        raise RuntimeError(f"stale result shard: {path}")
                    rows.append(row)
    if missing and not allow_incomplete:
        raise RuntimeError(
            f"benchmark is incomplete ({len(missing)} missing shards); first:\n"
            + "\n".join(missing[:20])
        )
    paired_hashes: dict[tuple[int, int], set[str]] = {}
    for row in rows:
        key = (int(row["order"]), int(row["pair_id"]))
        paired_hashes.setdefault(key, set()).add(str(row["data_test_sha256"]))
    mismatched = {
        key: values for key, values in paired_hashes.items() if len(values) != 1
    }
    if mismatched:
        raise RuntimeError(
            "models or data regimes within a paired replicate used different "
            "final-test data: "
            f"{mismatched}"
        )
    expected = (
        len(suite.orders)
        * len(suite.regimes)
        * len(suite.final_pair_ids)
        * len(suite.models)
    )
    summaries = summarize_rows(
        rows,
        bootstrap_samples=suite.bootstrap_samples,
        seed=suite.base_data_seed,
    )
    comparisons = paired_comparisons(
        rows,
        primary_model="learned_linear",
        bootstrap_samples=suite.bootstrap_samples,
        seed=suite.base_data_seed,
    )
    regime_comparisons = paired_regime_comparisons(
        rows,
        bootstrap_samples=suite.bootstrap_samples,
        seed=suite.base_data_seed + 1,
    )
    interaction_comparisons = paired_difference_in_differences(
        rows,
        primary_model="learned_linear",
        bootstrap_samples=suite.bootstrap_samples,
        seed=suite.base_data_seed + 2,
    )
    convergence = _convergence_report(rows, suite)
    _json_dump(root / "metrics.json", rows)
    _write_csv(root / "metrics.csv", rows)
    _json_dump(root / "summary.json", summaries)
    _write_csv(root / "summary.csv", summaries)
    _json_dump(root / "paired_comparisons.json", comparisons)
    _write_csv(root / "paired_comparisons.csv", comparisons)
    _json_dump(root / "paired_regime_comparisons.json", regime_comparisons)
    _write_csv(root / "paired_regime_comparisons.csv", regime_comparisons)
    _json_dump(
        root / "paired_difference_in_differences.json",
        interaction_comparisons,
    )
    _write_csv(
        root / "paired_difference_in_differences.csv",
        interaction_comparisons,
    )
    _json_dump(root / "convergence.json", convergence)
    _json_dump(
        root / "completeness.json",
        {
            "expected_rows": expected,
            "observed_rows": len(rows),
            "missing_rows": len(missing),
            "missing_paths": missing,
            "locked_config_hash": locked["locked_config_hash"],
        },
    )
    if summaries:
        _save_summary_plot(summaries, root / "summary.png")
    if not convergence["accepted"] and not allow_budget_limited:
        raise RuntimeError(
            "publication convergence gate failed; increase --max-epochs and "
            "repeat the benchmark. Diagnostic output was written to "
            f"{root / 'convergence.json'}:\n" + "\n".join(convergence["violations"])
        )
    return rows, summaries
