"""Train and plot the SciNet-style car experiment."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from common.reservoir import count_trainable_parameters
from common.runtime import resolve_device, seed_everything

from .data import generate_car_dataset
from .experiment import (
    POSITION_SCALE,
    _batches,
    _evaluate,
    _save_latent_plot,
    _save_physics_plot,
    _save_prediction_plot,
    _save_training_plot,
)
from .scinet_model import CarSciNet


def run_scinet_experiment(
    output_dir: str = "car_acceleration/results/scinet_2latent_seed8",
    *,
    seed: int = 8,
    train_samples: int = 3000,
    validation_samples: int = 500,
    test_samples: int = 500,
    steps: int = 5000,
    batch_size: int = 128,
    device_name: str = "auto",
) -> dict[str, float | list[float]]:
    """Train one SciNet-style model and save the same plots as the reservoir run."""

    seed_everything(seed)
    device = resolve_device(device_name)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    train = generate_car_dataset(train_samples, seed + 1)
    validation = generate_car_dataset(validation_samples, seed + 2)
    test = generate_car_dataset(test_samples, seed + 3)
    model = CarSciNet(seed=seed).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    rng = np.random.default_rng(seed + 4)
    history = {"step": [], "train_loss": [], "validation_loss": []}

    def validation_loss() -> float:
        model.eval()
        with torch.no_grad():
            history_tensor = (
                torch.from_numpy(validation.history).to(device) / POSITION_SCALE
            )
            target_tensor = (
                torch.from_numpy(validation.target).to(device) / POSITION_SCALE
            )
            prediction, _, _ = model.rollout(history_tensor, target_tensor.shape[1])
            return float(torch.mean((prediction - target_tensor) ** 2).cpu())

    for step in range(1, steps + 1):
        model.train()
        history_batch, target_batch = _batches(train, batch_size, rng)
        history_tensor = torch.from_numpy(history_batch).to(device) / POSITION_SCALE
        target_tensor = torch.from_numpy(target_batch).to(device) / POSITION_SCALE
        prediction, _, representation_loss = model.training_rollout(
            history_tensor, target_tensor.shape[1]
        )
        reconstruction_loss = torch.mean((prediction - target_tensor) ** 2)
        loss = reconstruction_loss + representation_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % 20 == 0 or step == steps:
            history["step"].append(step)
            history["train_loss"].append(float(loss.detach().cpu()))
            history["validation_loss"].append(validation_loss())
            if step % 100 == 0 or step == steps:
                print(
                    f"step={step:4d} train={history['train_loss'][-1]:.6g} "
                    f"validation={history['validation_loss'][-1]:.6g}",
                    flush=True,
                )

    metrics, prediction, future_latent, initial_latent = _evaluate(model, test, device)
    metrics.update(
        {
            "model": "scinet_style",
            "seed": seed,
            "device": str(device),
            "trainable_parameters": count_trainable_parameters(model),
            "reservoir_nodes_each": 0,
            "reservoir_count": 0,
            "latent_size": 2,
            "hidden_size": 100,
            "beta": model.beta,
            "target_latent_std": model.target_latent_std,
            "fixed_acceleration": 0.5,
            "history_length": 3,
            "forecast_horizon": 30,
        }
    )
    torch.save(model.state_dict(), output / "checkpoint.pt")
    np.savez_compressed(
        output / "predictions.npz",
        history=test.history,
        target=test.target,
        prediction=prediction,
        position=test.position,
        velocity=test.velocity,
        initial_latent=initial_latent,
        future_latent=future_latent,
    )
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    (output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    _save_training_plot(history, output / "training.png")
    _save_physics_plot(test, output / "ground_truth_dynamics.png")
    _save_prediction_plot(test, prediction, output / "predictions.png")
    _save_latent_plot(test, initial_latent, future_latent, output / "latent_dynamics.png")
    print(json.dumps(metrics, indent=2), flush=True)
    print(f"saved results to {output}", flush=True)
    return metrics

