"""Training, evaluation, and plots for the car acceleration experiment."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from common.reservoir import count_trainable_parameters
from common.runtime import resolve_device, seed_everything

from .data import CarDataset, generate_car_dataset
from .model import CarReservoir


POSITION_SCALE = 10.0
TIME_STEP = 1.0
COVARIANCE_WEIGHT = 0.1
LEARNING_RATE = 1e-3
LR_DECAY_STEP = 5000
LR_DECAY_FACTOR = 0.1


def covariance_penalty(latent: torch.Tensor) -> torch.Tensor:
    """Penalize covariance between latent coordinates across a batch."""

    #if latent.ndim != 2 or latent.shape[1] < 2:
    #    raise ValueError("latent must have shape [batch, latent_size >= 2]")
    centered = latent - latent.mean(dim=0, keepdim=True)
    covariance = centered.T @ centered / max(len(latent) - 1, 1)
    off_diagonal = covariance - torch.diag(torch.diag(covariance))
    return off_diagonal.square().sum()


def _r2(source: np.ndarray, destination: np.ndarray) -> tuple[float, list[float]]:
    source_augmented = np.column_stack((source, np.ones(len(source))))
    coefficients = np.linalg.lstsq(source_augmented, destination, rcond=None)[0]
    prediction = source_augmented @ coefficients
    residual = np.sum((destination - prediction) ** 2, axis=0)
    total = np.sum((destination - destination.mean(axis=0)) ** 2, axis=0)
    per_dimension = 1.0 - residual / np.maximum(total, 1e-12)
    pooled = 1.0 - float(residual.sum()) / max(float(total.sum()), 1e-12)
    return float(pooled), per_dimension.tolist()


def _batches(dataset: CarDataset, batch_size: int, rng: np.random.Generator):
    indices = rng.integers(0, len(dataset.history), size=batch_size)
    return dataset.history[indices], dataset.target[indices]


def _evaluate(
    model: CarReservoir,
    dataset: CarDataset,
    device: torch.device,
    batch_size: int = 256,
) -> tuple[dict[str, float | list[float]], np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    predictions = []
    latents = []
    initial_latents = []
    with torch.no_grad():
        for start in range(0, len(dataset.history), batch_size):
            history = torch.from_numpy(dataset.history[start : start + batch_size]).to(device)
            prediction, future_latent, initial_latent = model.rollout(
                history / POSITION_SCALE, dataset.target.shape[1]
            )
            predictions.append((prediction.cpu().numpy() * POSITION_SCALE))
            latents.append(future_latent.cpu().numpy())
            initial_latents.append(initial_latent.cpu().numpy())
    prediction_array = np.concatenate(predictions)
    latent_array = np.concatenate(latents)
    initial_latent_array = np.concatenate(initial_latents)
    rmse = float(np.sqrt(np.mean((prediction_array - dataset.target) ** 2)))
    last_index = dataset.history.shape[1] - 1
    physical_state = np.column_stack(
        (dataset.position[:, last_index], dataset.velocity[:, last_index])
    )
    latent_to_state_r2, latent_to_state_per_dimension = _r2(
        initial_latent_array, physical_state
    )
    state_to_latent_r2, state_to_latent_per_dimension = _r2(
        physical_state, initial_latent_array
    )
    latent_covariance = np.cov(initial_latent_array, rowvar=False)
    metrics = {
        "position_rmse": rmse,
        "latent_to_position_velocity_r2": latent_to_state_r2,
        "latent_to_position_velocity_r2_per_dimension": latent_to_state_per_dimension,
        "position_velocity_to_latent_r2": state_to_latent_r2,
        "position_velocity_to_latent_r2_per_dimension": state_to_latent_per_dimension,
        "latent_covariance": latent_covariance.tolist(),
        "latent_covariance_off_diagonal": float(latent_covariance[0, 1]),
    }
    return metrics, prediction_array, latent_array, initial_latent_array


def _save_training_plot(history: dict[str, list[float]], path: Path) -> None:
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.semilogy(history["step"], history["train_loss"], label="train")
    axis.semilogy(history["step"], history["validation_loss"], label="validation")
    axis.set(xlabel="Optimizer step", ylabel="Position MSE", title="Training")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _save_prediction_plot(
    dataset: CarDataset, prediction: np.ndarray, path: Path
) -> None:
    count = min(4, len(dataset.history))
    figure, axes = plt.subplots(count, 1, figsize=(9, 2.2 * count), squeeze=False)
    forecast_times = TIME_STEP * np.arange(1, dataset.target.shape[1] + 1)
    for row in range(count):
        axis = axes[row, 0]
        axis.plot(forecast_times, dataset.target[row], label="true position", linewidth=2)
        axis.plot(forecast_times, prediction[row], label="predicted position", linewidth=1.4)
        axis.set_ylabel("position")
        axis.grid(alpha=0.25)
    axes[0, 0].legend()
    axes[-1, 0].set_xlabel("time after observed history")
    figure.suptitle("Held-out car rollouts")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)

def _save_latent_plot(
    dataset: CarDataset,
    initial_latent: np.ndarray,
    future_latent: np.ndarray,
    path: Path,
) -> None:
    last_index = dataset.history.shape[1] - 1

    position = dataset.position[:, last_index]
    velocity = dataset.velocity[:, last_index]
    acceleration = dataset.acceleration[:, last_index]

    figure = plt.figure(figsize=(20, 5))

    ax_position = figure.add_subplot(1, 4, 1, projection="3d")
    ax_velocity = figure.add_subplot(1, 4, 2, projection="3d")
    ax_acceleration = figure.add_subplot(1, 4, 3, projection="3d")
    ax_trajectory = figure.add_subplot(1, 4, 4, projection="3d")

    # ------------------------------------------------------------
    # Latent colored by true position
    # ------------------------------------------------------------

    scatter = ax_position.scatter(
        initial_latent[:, 0],
        initial_latent[:, 1],
        initial_latent[:, 2],
        c=position,
        cmap="viridis",
        s=12,
        alpha=0.8,
    )

    ax_position.set(
        title="colored by position",
        xlabel="latent 1",
        ylabel="latent 2",
        zlabel="latent 3",
    )

    figure.colorbar(
        scatter,
        ax=ax_position,
        label="true position",
        shrink=0.7,
        pad=0.1,
    )

    # ------------------------------------------------------------
    # Latent colored by true velocity
    # ------------------------------------------------------------

    scatter = ax_velocity.scatter(
        initial_latent[:, 0],
        initial_latent[:, 1],
        initial_latent[:, 2],
        c=velocity,
        cmap="coolwarm",
        s=12,
        alpha=0.8,
    )

    ax_velocity.set(
        title="colored by velocity",
        xlabel="latent 1",
        ylabel="latent 2",
        zlabel="latent 3",
    )

    figure.colorbar(
        scatter,
        ax=ax_velocity,
        label="true velocity",
        shrink=0.7,
        pad=0.1,
    )

    # ------------------------------------------------------------
    # Latent colored by true acceleration
    # ------------------------------------------------------------

    scatter = ax_acceleration.scatter(
        initial_latent[:, 0],
        initial_latent[:, 1],
        initial_latent[:, 2],
        c=acceleration,
        cmap="plasma",
        s=12,
        alpha=0.8,
    )

    ax_acceleration.set(
        title="colored by acceleration",
        xlabel="latent 1",
        ylabel="latent 2",
        zlabel="latent 3",
    )

    figure.colorbar(
        scatter,
        ax=ax_acceleration,
        label="true acceleration",
        shrink=0.7,
        pad=0.1,
    )

    # ------------------------------------------------------------
    # Example trajectory through latent space
    # ------------------------------------------------------------

    example = future_latent[0]

    ax_trajectory.plot(
        example[:, 0],
        example[:, 1],
        example[:, 2],
        marker="o",
        markersize=2,
    )

    ax_trajectory.scatter(
        example[0, 0],
        example[0, 1],
        example[0, 2],
        color="green",
        s=40,
        label="first prediction",
    )

    ax_trajectory.scatter(
        example[-1, 0],
        example[-1, 1],
        example[-1, 2],
        color="red",
        s=40,
        label="last prediction",
    )

    ax_trajectory.set(
        title="one latent trajectory",
        xlabel="latent 1",
        ylabel="latent 2",
        zlabel="latent 3",
    )

    ax_trajectory.legend(fontsize=8)

    # ------------------------------------------------------------
    # Formatting
    # ------------------------------------------------------------

    for axis in (
        ax_position,
        ax_velocity,
        ax_acceleration,
        ax_trajectory,
    ):
        axis.grid(alpha=0.25)

    figure.suptitle(
        "What information is carried by the 3D latent?"
    )

    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
def _save_physics_plot(dataset: CarDataset, path: Path) -> None:
    dt = TIME_STEP
    times = dt * np.arange(dataset.position.shape[1])

    figure, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)

    num_to_plot = min(6, len(dataset.position))

    for index in range(num_to_plot):
        axes[0].plot(
            times,
            dataset.position[index],
            alpha=0.75,
        )

        axes[1].plot(
            times,
            dataset.velocity[index],
            alpha=0.75,
        )

        axes[2].plot(
            times,
            dataset.acceleration[index],
            alpha=0.75,
        )

    history_end = dataset.history.shape[1] * dt

    axes[0].axvline(
        history_end,
        color="black",
        linestyle="--",
        label="history ends",
    )

    axes[0].set_ylabel("position")
    axes[1].set_ylabel("velocity")
    axes[2].set_ylabel("acceleration")
    axes[2].set_xlabel("time")

    axes[0].set_title("Ground-truth accelerating cars")
    axes[0].legend()

    for axis in axes:
        axis.grid(alpha=0.25)

    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
def run_experiment(
    output_dir: str = "car_acceleration/results/linear_readout_2latent_positive_seed8",
    *,
    seed: int = 7,
    train_samples: int = 3000,
    validation_samples: int = 500,
    test_samples: int = 500,
    steps: int = 100,
    batch_size: int = 128,
    covariance_weight: float = COVARIANCE_WEIGHT,
    device_name: str = "auto",
) -> dict[str, float | list[float]]:
    """Train exactly one encoder-reservoir model and save results and figures."""

    seed_everything(seed)
    device = resolve_device(device_name)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    train = generate_car_dataset(train_samples, seed + 1, dt=TIME_STEP)
    validation = generate_car_dataset(validation_samples, seed + 2, dt=TIME_STEP)
    test = generate_car_dataset(test_samples, seed + 3, dt=TIME_STEP)
    model = CarReservoir(seed=seed).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[LR_DECAY_STEP],
        gamma=LR_DECAY_FACTOR,
    )
    rng = np.random.default_rng(seed + 4)
    history = {
        "step": [],
        "train_loss": [],
        "validation_loss": [],
        "learning_rate": [],
    }

    def validation_loss() -> float:
        model.eval()
        with torch.no_grad():
            history_tensor = torch.from_numpy(validation.history).to(device) / POSITION_SCALE
            target_tensor = torch.from_numpy(validation.target).to(device) / POSITION_SCALE
            prediction, _, _ = model.rollout(history_tensor, target_tensor.shape[1])
            return float(torch.mean((prediction - target_tensor) ** 2).cpu())

    for step in range(1, steps + 1):
        model.train()
        history_batch, target_batch = _batches(train, batch_size, rng)
        history_tensor = torch.from_numpy(history_batch).to(device) / POSITION_SCALE
        target_tensor = torch.from_numpy(target_batch).to(device) / POSITION_SCALE
        prediction, _, initial_latent = model.rollout(
            history_tensor, target_tensor.shape[1]
        )
        loss = torch.mean((prediction - target_tensor) ** 2)
        loss = loss + covariance_weight * covariance_penalty(initial_latent)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        if step == 1 or step % 20 == 0 or step == steps:
            history["step"].append(step)
            history["train_loss"].append(float(loss.detach().cpu()))
            history["validation_loss"].append(validation_loss())
            history["learning_rate"].append(optimizer.param_groups[0]["lr"])
            if step % 100 == 0 or step == steps:
                print(
                    f"step={step:4d} train={history['train_loss'][-1]:.6g} "
                    f"validation={history['validation_loss'][-1]:.6g} "
                    f"lr={history['learning_rate'][-1]:.3g}",
                    flush=True,
                )

    metrics, prediction, future_latent, initial_latent = _evaluate(model, test, device)

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
