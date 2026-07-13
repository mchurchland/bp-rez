"""Training, evaluation, artifact writing, and multi-seed aggregation."""

from __future__ import annotations

import csv
import json
import math
import os
import random
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "bp_reservoir_mpl"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from .data import NARMASplit, make_narma10_splits
from .models import MODEL_NAMES, ReservoirBase, build_model, count_trainable_parameters


@dataclass
class ExperimentConfig:
    output_dir: str = "results/narma10_300_nodes"
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    data_seed: int = 2026
    models: tuple[str, ...] = MODEL_NAMES
    nodes_1: int = 150
    nodes_2: int = 150
    latent_size: int = 10
    train_length: int = 2000
    val_length: int = 500
    test_length: int = 500
    washout: int = 100
    spectral_radius: float = 0.9
    density: float = 0.1
    leak_rate: float = 1.0
    input_scale: float = 0.5
    interlayer_scale: float = 1.0
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    max_epochs: int = 200
    patience: int = 25
    min_delta: float = 1e-7
    grad_clip: float = 1.0
    device: str = "auto"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def _loss_after_washout(prediction: torch.Tensor, target: torch.Tensor, washout: int) -> torch.Tensor:
    if washout >= len(target):
        raise ValueError("washout must be smaller than every sequence length")
    return nn.functional.mse_loss(prediction[washout:], target[washout:])


def evaluate(
    model: ReservoirBase,
    split: NARMASplit,
    washout: int,
    device: torch.device,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    model.eval()
    with torch.no_grad():
        target = split.y.to(device)
        prediction = model(split.u.to(device))
        prediction_eval = prediction[washout:]
        target_eval = target[washout:]
        mse = nn.functional.mse_loss(prediction_eval, target_eval).item()
        variance = torch.var(target_eval, unbiased=False).item()
        nrmse = math.sqrt(mse / variance) if variance > 0.0 else float("nan")
    return (
        {"mse": mse, "nrmse": nrmse},
        prediction.detach().cpu().numpy().squeeze(-1),
        target.detach().cpu().numpy().squeeze(-1),
    )


def train_one(
    model: ReservoirBase,
    train: NARMASplit,
    validation: NARMASplit,
    config: ExperimentConfig,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, list[float]]]:
    """Train all and only trainable tensors with full-sequence BPTT and Adam."""

    model.to(device)
    train_u = train.u.to(device)
    train_y = train.y.to(device)
    val_u = validation.u.to(device)
    val_y = validation.y.to(device)
    optimizer = torch.optim.Adam(
        (p for p in model.parameters() if p.requires_grad),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    best_loss = float("inf")
    best_epoch = -1
    best_state: dict[str, torch.Tensor] | None = None
    stale_epochs = 0
    history: dict[str, list[float]] = {"train_mse": [], "validation_mse": [], "grad_norm": []}
    start = time.perf_counter()

    for epoch in range(config.max_epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        train_prediction = model(train_u)
        train_loss = _loss_after_washout(train_prediction, train_y, config.washout)
        train_loss.backward()
        if config.grad_clip > 0:
            grad_norm_tensor = nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            grad_norm = float(grad_norm_tensor.detach().cpu())
        else:
            squared = sum(
                float(torch.sum(p.grad.detach() ** 2).cpu())
                for p in model.parameters()
                if p.grad is not None
            )
            grad_norm = math.sqrt(squared)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            validation_prediction = model(val_u)
            validation_loss = _loss_after_washout(
                validation_prediction, val_y, config.washout
            ).item()
        history["train_mse"].append(float(train_loss.detach().cpu()))
        history["validation_mse"].append(validation_loss)
        history["grad_norm"].append(grad_norm)

        if validation_loss < best_loss - config.min_delta:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= config.patience:
                break

    if best_state is None:
        raise RuntimeError("training did not produce a checkpoint")
    model.load_state_dict(best_state)
    model.to(device)
    return (
        {
            "best_epoch": best_epoch + 1,
            "epochs_ran": len(history["train_mse"]),
            "best_validation_mse": best_loss,
            "training_seconds": time.perf_counter() - start,
        },
        history,
    )


def _json_dump(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _save_history_plot(history: dict[str, list[float]], path: Path) -> None:
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.semilogy(history["train_mse"], label="train")
    axis.semilogy(history["validation_mse"], label="validation")
    axis.set(xlabel="Epoch", ylabel="MSE", title="Training history")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _save_prediction_plot(
    prediction: np.ndarray, target: np.ndarray, washout: int, path: Path
) -> None:
    start = washout
    stop = min(len(target), start + 250)
    figure, axis = plt.subplots(figsize=(10, 4))
    axis.plot(np.arange(start, stop), target[start:stop], label="target", linewidth=1.4)
    axis.plot(np.arange(start, stop), prediction[start:stop], label="prediction", linewidth=1.1)
    axis.set(xlabel="Time step", ylabel="NARMA10 output", title="Test prediction")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def run_one(
    model_name: str,
    seed: int,
    splits: dict[str, NARMASplit],
    config: ExperimentConfig,
    device: torch.device,
    root: Path,
) -> dict[str, Any]:
    seed_everything(seed)
    model = build_model(
        model_name,
        nodes_1=config.nodes_1,
        nodes_2=config.nodes_2,
        latent_size=config.latent_size,
        spectral_radius=config.spectral_radius,
        input_scale=config.input_scale,
        interlayer_scale=config.interlayer_scale,
        density=config.density,
        leak_rate=config.leak_rate,
        seed=seed,
    )
    run_dir = root / model_name / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    train_info, history = train_one(
        model, splits["train"], splits["validation"], config, device
    )
    validation_metrics, _, _ = evaluate(
        model, splits["validation"], config.washout, device
    )
    test_metrics, prediction, target = evaluate(
        model, splits["test"], config.washout, device
    )
    trainable_parameters = count_trainable_parameters(model)
    metrics: dict[str, Any] = {
        "model": model_name,
        "seed": seed,
        "device": str(device),
        "trainable_parameters": trainable_parameters,
        "validation_mse": validation_metrics["mse"],
        "validation_nrmse": validation_metrics["nrmse"],
        "test_mse": test_metrics["mse"],
        "test_nrmse": test_metrics["nrmse"],
        **train_info,
    }
    torch.save(
        {
            "model_name": model_name,
            "seed": seed,
            "config": asdict(config),
            "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "metrics": metrics,
        },
        run_dir / "checkpoint.pt",
    )
    np.savez_compressed(
        run_dir / "predictions.npz",
        prediction=prediction,
        target=target,
        input=splits["test"].u.numpy().squeeze(-1),
        washout=np.asarray(config.washout),
    )
    _json_dump(run_dir / "metrics.json", metrics)
    _json_dump(run_dir / "history.json", history)
    _save_history_plot(history, run_dir / "training.png")
    _save_prediction_plot(prediction, target, config.washout, run_dir / "predictions.png")
    return metrics


def _summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries = []
    for model_name in dict.fromkeys(row["model"] for row in rows):
        model_rows = [row for row in rows if row["model"] == model_name]
        mse = np.asarray([row["test_mse"] for row in model_rows])
        nrmse = np.asarray([row["test_nrmse"] for row in model_rows])
        summaries.append(
            {
                "model": model_name,
                "seeds": len(model_rows),
                "trainable_parameters": model_rows[0]["trainable_parameters"],
                "test_mse_mean": float(mse.mean()),
                "test_mse_std": float(mse.std(ddof=1)) if len(mse) > 1 else 0.0,
                "test_nrmse_mean": float(nrmse.mean()),
                "test_nrmse_std": float(nrmse.std(ddof=1)) if len(nrmse) > 1 else 0.0,
            }
        )
    return summaries


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _save_summary_plot(summary: list[dict[str, Any]], path: Path) -> None:
    labels = [row["model"].replace("_", "\n") for row in summary]
    means = [row["test_nrmse_mean"] for row in summary]
    errors = [row["test_nrmse_std"] for row in summary]
    figure, axis = plt.subplots(figsize=(9, 4.5))
    axis.bar(labels, means, yerr=errors, capsize=4)
    axis.set(ylabel="Test NRMSE (mean ± sample SD)", title="NARMA10 multi-seed comparison")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def run_experiment(config: ExperimentConfig) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if len(config.seeds) < 1:
        raise ValueError("at least one seed is required")
    if config.nodes_1 + config.nodes_2 != 300:
        raise ValueError("this study requires nodes_1 + nodes_2 == 300")
    if config.latent_size != 10:
        raise ValueError("this study requires a 10-dimensional latent space")
    if config.washout >= min(config.train_length, config.val_length, config.test_length):
        raise ValueError("washout must be shorter than every split")
    unknown = set(config.models) - set(MODEL_NAMES)
    if unknown:
        raise ValueError(f"unknown models: {sorted(unknown)}")

    root = Path(config.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    device = resolve_device(config.device)
    config_dict = asdict(config)
    config_dict["resolved_device"] = str(device)
    _json_dump(root / "config.json", config_dict)
    splits = make_narma10_splits(
        config.train_length, config.val_length, config.test_length, config.data_seed
    )
    rows = []
    for model_name in config.models:
        for seed in config.seeds:
            print(f"[{model_name}] seed={seed} device={device}", flush=True)
            row = run_one(model_name, seed, splits, config, device, root)
            rows.append(row)
            print(
                f"  test MSE={row['test_mse']:.6g} NRMSE={row['test_nrmse']:.6g} "
                f"best_epoch={row['best_epoch']}",
                flush=True,
            )
            _write_csv(root / "metrics.csv", rows)
            _json_dump(root / "metrics.json", rows)
    summary = _summarize(rows)
    _write_csv(root / "summary.csv", summary)
    _json_dump(root / "summary.json", summary)
    _save_summary_plot(summary, root / "summary.png")
    return rows, summary
