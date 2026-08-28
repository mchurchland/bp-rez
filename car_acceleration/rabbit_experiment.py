"""Train on rabbit-wolf histories and forecast both future populations."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from common.reservoir import count_trainable_parameters
from common.runtime import resolve_device, seed_everything

from .data import RabbitDataset, generate_rabbit_dataset
from .rabbit_model import RabbitReservoir


LEARNING_RATE = 1e-3
LR_DECAY_STEPS = (5000, 8000)
LR_DECAY_FACTOR = 0.1
COVARIANCE_WEIGHT = 0.0


def forecast_curriculum_horizon(step: int, maximum_horizon: int) -> int:
    """Return the staged horizon for a 2,000-step training run."""

    if step < 1 or maximum_horizon < 1:
        raise ValueError("step and maximum_horizon must be positive")
    if step <= 400:
        requested_horizon = 25
    elif step <= 800:
        requested_horizon = 50
    elif step <= 1200:
        requested_horizon = 100
    elif step <= 1600:
        requested_horizon = 250
    else:
        requested_horizon = maximum_horizon
    return min(requested_horizon, maximum_horizon)


def covariance_penalty(latent: torch.Tensor) -> torch.Tensor:
    """Penalize off-diagonal covariance of the initial latent batch."""

    centered = latent - latent.mean(dim=0, keepdim=True)
    covariance = centered.T @ centered / max(len(latent) - 1, 1)
    off_diagonal = covariance - torch.diag(torch.diag(covariance))
    return off_diagonal.square().sum()


def _r2(source: np.ndarray, destination: np.ndarray) -> tuple[float, list[float]]:
    augmented = np.column_stack((source, np.ones(len(source))))
    coefficients = np.linalg.lstsq(augmented, destination, rcond=None)[0]
    prediction = augmented @ coefficients
    residual = np.sum((destination - prediction) ** 2, axis=0)
    total = np.sum((destination - destination.mean(axis=0)) ** 2, axis=0)
    per_dimension = 1.0 - residual / np.maximum(total, 1e-12)
    pooled = 1.0 - float(residual.sum()) / max(float(total.sum()), 1e-12)
    return pooled, per_dimension.tolist()


def _held_out_r2(
    train_source: np.ndarray,
    train_destination: np.ndarray,
    test_source: np.ndarray,
    test_destination: np.ndarray,
) -> tuple[float, list[float]]:
    """Fit an affine probe on training data and score it on held-out data."""

    if len(train_source) > 1000:
        probe_rng = np.random.default_rng(0)
        probe_indices = probe_rng.choice(
            len(train_source),
            size=1000,
            replace=False,
        )
        train_source = train_source[probe_indices]
        train_destination = train_destination[probe_indices]

    mean = train_source.mean(axis=0)
    scale = train_source.std(axis=0)
    retained = scale > 1e-8
    if not np.any(retained):
        raise ValueError("probe source has no varying features")
    normalized_train = (
        train_source[:, retained] - mean[retained]
    ) / scale[retained]
    normalized_test = (
        test_source[:, retained] - mean[retained]
    ) / scale[retained]
    destination_mean = train_destination.mean(axis=0)
    centered_destination = train_destination - destination_mean
    sample_count, feature_count = normalized_train.shape
    ridge_strength = 0.1 * max(sample_count, feature_count)
    if feature_count <= sample_count:
        gram = normalized_train.T @ normalized_train
        gram.flat[:: feature_count + 1] += ridge_strength
        coefficients = np.linalg.solve(
            gram,
            normalized_train.T @ centered_destination,
        )
    else:
        gram = normalized_train @ normalized_train.T
        gram.flat[:: sample_count + 1] += ridge_strength
        coefficients = normalized_train.T @ np.linalg.solve(
            gram,
            centered_destination,
        )
    prediction = normalized_test @ coefficients + destination_mean
    residual = np.sum((test_destination - prediction) ** 2, axis=0)
    total = np.sum(
        (test_destination - test_destination.mean(axis=0)) ** 2,
        axis=0,
    )
    per_dimension = 1.0 - residual / np.maximum(total, 1e-12)
    pooled = 1.0 - float(residual.sum()) / max(float(total.sum()), 1e-12)
    return pooled, per_dimension.tolist()


def _print_feature_probe_table(
    reservoir_scores: list[float],
    latent_scores: list[float],
) -> None:
    """Print reservoir and learned-latent physical-information probes."""

    labels = ("Rabbits", "Wolves", "alpha", "beta", "gamma", "delta")
    print("\nRequired information carried by the model state")
    print(f"{'Required information':<22}{'Reservoir R^2':>16}{'Latent R^2':>14}")
    for label, reservoir_score, latent_score in zip(
        labels,
        reservoir_scores,
        latent_scores,
        strict=True,
    ):
        print(f"{label:<22}{reservoir_score:>16.3f}{latent_score:>14.3f}")
    print("Probes fitted on training trajectories and scored on test trajectories.")


def _log_population(values: np.ndarray) -> np.ndarray:
    return np.log1p(values.astype(np.float64))


def _normalize(values: np.ndarray, mean: float, scale: float) -> np.ndarray:
    return ((_log_population(values) - mean) / scale).astype(np.float32)


def _normalized_targets(
    dataset: RabbitDataset,
    rabbit_log_mean: float,
    rabbit_log_scale: float,
    wolf_log_mean: float,
    wolf_log_scale: float,
) -> np.ndarray:
    """Return joint future targets with shape [trajectory, time, 2]."""

    history_length = dataset.history.shape[1]
    future_wolves = dataset.wolves[:, history_length:]
    return np.stack(
        (
            _normalize(dataset.target, rabbit_log_mean, rabbit_log_scale),
            _normalize(future_wolves, wolf_log_mean, wolf_log_scale),
        ),
        axis=2,
    )


def _physical_state(dataset: RabbitDataset) -> np.ndarray:
    """Return the six observable augmented-state coordinates."""

    last_observed = dataset.history.shape[1] - 1
    return np.column_stack(
        (
            np.log1p(dataset.rabbits[:, last_observed]),
            np.log1p(dataset.wolves[:, last_observed]),
            np.log(dataset.alpha),
            np.log(dataset.beta),
            np.log(dataset.gamma),
            np.log(dataset.delta),
        )
    )


def _precompute_reservoir_states(
    model: RabbitReservoir,
    dataset: RabbitDataset,
    device: torch.device,
    log_mean: float,
    log_scale: float,
    wolf_log_mean: float,
    wolf_log_scale: float,
    batch_size: int = 256,
) -> torch.Tensor:
    """Cache fixed reservoir states so optimizer steps only train the head."""

    states = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(dataset.history), batch_size):
            normalized_rabbits = _normalize(
                dataset.history[start : start + batch_size],
                log_mean,
                log_scale,
            )
            history_length = dataset.history.shape[1]
            normalized_wolves = _normalize(
                dataset.wolves[
                    start : start + batch_size,
                    :history_length,
                ],
                wolf_log_mean,
                wolf_log_scale,
            )
            normalized_history = np.stack(
                (normalized_rabbits, normalized_wolves),
                axis=2,
            )
            states.append(
                model.reservoir_state(
                    torch.from_numpy(normalized_history).to(device)
                )
            )
    return torch.cat(states)


def _evaluate(
    model: RabbitReservoir,
    dataset: RabbitDataset,
    device: torch.device,
    log_mean: float,
    log_scale: float,
    wolf_log_mean: float,
    wolf_log_scale: float,
    reservoir_states: torch.Tensor,
    batch_size: int = 128,
) -> tuple[dict[str, object], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    normalized_predictions = []
    future_latents = []
    initial_latents = []
    with torch.no_grad():
        for start in range(0, len(dataset.history), batch_size):
            prediction, future_latent, initial_latent = (
                model.rollout_from_reservoir_state(
                    reservoir_states[start : start + batch_size],
                    dataset.target.shape[1],
                )
            )
            normalized_predictions.append(prediction.cpu().numpy())
            future_latents.append(future_latent.cpu().numpy())
            initial_latents.append(initial_latent.cpu().numpy())

    normalized_prediction = np.concatenate(normalized_predictions)
    future_latent_array = np.concatenate(future_latents)
    initial_latent_array = np.concatenate(initial_latents)
    predicted_rabbit_log = normalized_prediction[:, :, 0] * log_scale + log_mean
    predicted_wolf_log = (
        normalized_prediction[:, :, 1] * wolf_log_scale + wolf_log_mean
    )
    true_rabbit_log = _log_population(dataset.target)
    history_length = dataset.history.shape[1]
    wolf_target = dataset.wolves[:, history_length:]
    true_wolf_log = _log_population(wolf_target)
    rabbit_prediction = np.expm1(np.clip(predicted_rabbit_log, 0.0, 20.0))
    wolf_prediction = np.expm1(np.clip(predicted_wolf_log, 0.0, 20.0))

    physical_state = _physical_state(dataset)
    probe_pooled, probe_per_dimension = _r2(initial_latent_array, physical_state)
    latent_covariance = np.cov(initial_latent_array, rowvar=False)
    off_diagonal = latent_covariance - np.diag(np.diag(latent_covariance))
    metrics: dict[str, object] = {
        "rabbit_log_rmse": float(
            np.sqrt(np.mean((predicted_rabbit_log - true_rabbit_log) ** 2))
        ),
        "wolf_log_rmse": float(
            np.sqrt(np.mean((predicted_wolf_log - true_wolf_log) ** 2))
        ),
        "rabbit_rmse": float(
            np.sqrt(np.mean((rabbit_prediction - dataset.target) ** 2))
        ),
        "rabbit_mae": float(np.mean(np.abs(rabbit_prediction - dataset.target))),
        "wolf_rmse": float(np.sqrt(np.mean((wolf_prediction - wolf_target) ** 2))),
        "wolf_mae": float(np.mean(np.abs(wolf_prediction - wolf_target))),
        "latent_to_observable_state_r2": probe_pooled,
        "latent_to_observable_state_r2_per_dimension": probe_per_dimension,
        "observable_state_labels": [
            "log_rabbits",
            "log_wolves",
            "log_alpha",
            "log_beta",
            "log_gamma",
            "log_delta",
        ],
        "latent_covariance": latent_covariance.tolist(),
        "latent_covariance_off_diagonal_mean_abs": float(
            np.mean(np.abs(off_diagonal[np.triu_indices_from(off_diagonal, k=1)]))
        ),
    }
    return (
        metrics,
        rabbit_prediction,
        wolf_prediction,
        future_latent_array,
        initial_latent_array,
    )


def _save_training_plot(history: dict[str, list[float]], path: Path) -> None:
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.semilogy(history["step"], history["train_loss"], label="train")
    axis.semilogy(history["step"], history["validation_loss"], label="validation")
    axis.set(
        xlabel="Optimizer step",
        ylabel="Standardized log-population loss",
        title="Rabbit training",
    )
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _save_prediction_plot(
    dataset: RabbitDataset,
    rabbit_prediction: np.ndarray,
    wolf_prediction: np.ndarray,
    path: Path,
) -> None:
    count = min(4, len(dataset.history))
    times = dataset.dt * np.arange(1, dataset.target.shape[1] + 1)
    history_length = dataset.history.shape[1]
    wolf_target = dataset.wolves[:, history_length:]
    figure, axes = plt.subplots(
        count,
        2,
        figsize=(14, 2.3 * count),
        squeeze=False,
        sharex=True,
    )
    for row in range(count):
        rabbit_axis, wolf_axis = axes[row]
        rabbit_axis.plot(
            times,
            dataset.target[row],
            label="true rabbits",
            linewidth=2,
        )
        rabbit_axis.plot(
            times,
            rabbit_prediction[row],
            label="predicted rabbits",
            linewidth=1.4,
        )
        wolf_axis.plot(
            times,
            wolf_target[row],
            label="true wolves",
            linewidth=2,
        )
        wolf_axis.plot(
            times,
            wolf_prediction[row],
            label="predicted wolves",
            linewidth=1.4,
        )
        for axis in (rabbit_axis, wolf_axis):
            axis.set_yscale("symlog", linthresh=1.0)
            axis.grid(alpha=0.25)
        rabbit_axis.set_ylabel("rabbits")
        wolf_axis.set_ylabel("wolves")
    axes[0, 0].legend()
    axes[0, 1].legend()
    axes[-1, 0].set_xlabel("time after observed history")
    axes[-1, 1].set_xlabel("time after observed history")
    figure.suptitle("Held-out predator-prey rollouts")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _save_physics_plot(dataset: RabbitDataset, path: Path) -> None:
    times = dataset.dt * np.arange(dataset.rabbits.shape[1])
    figure, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    for index in range(min(6, len(dataset.rabbits))):
        axes[0].plot(times, dataset.rabbits[index], alpha=0.75)
        axes[1].plot(times, dataset.wolves[index], alpha=0.75)
    history_end = dataset.history.shape[1] * dataset.dt
    for axis in axes:
        axis.axvline(history_end, color="black", linestyle="--")
        axis.set_yscale("symlog", linthresh=1.0)
        axis.grid(alpha=0.25)
    axes[0].set(ylabel="rabbits", title="Ground-truth predator-prey dynamics")
    axes[1].set(xlabel="time", ylabel="wolves")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _save_latent_plot(
    dataset: RabbitDataset,
    initial_latent: np.ndarray,
    future_latent: np.ndarray,
    path: Path,
) -> None:
    centered = initial_latent - initial_latent.mean(axis=0, keepdims=True)
    _, _, right_vectors = np.linalg.svd(centered, full_matrices=False)
    basis = right_vectors[:2].T
    projected = centered @ basis
    physical = _physical_state(dataset)
    labels = ("rabbits", "wolves", "alpha", "beta", "gamma", "delta")

    figure, axes = plt.subplots(2, 4, figsize=(18, 9))
    for index, (axis, label) in enumerate(zip(axes.flat[:6], labels, strict=True)):
        scatter = axis.scatter(
            projected[:, 0],
            projected[:, 1],
            c=physical[:, index],
            cmap="viridis",
            s=12,
            alpha=0.8,
        )
        axis.set(title=f"colored by {label}", xlabel="latent PC1", ylabel="latent PC2")
        figure.colorbar(scatter, ax=axis, shrink=0.8)

    trajectory_axis = axes.flat[6]
    trajectory = (future_latent[0] - initial_latent.mean(axis=0)) @ basis
    trajectory_axis.plot(trajectory[:, 0], trajectory[:, 1], linewidth=1.5)
    trajectory_axis.scatter(*trajectory[0], color="green", label="first prediction")
    trajectory_axis.scatter(*trajectory[-1], color="red", label="last prediction")
    trajectory_axis.set(
        title="one latent trajectory",
        xlabel="latent PC1",
        ylabel="latent PC2",
    )
    trajectory_axis.legend(fontsize=8)
    axes.flat[7].axis("off")
    for axis in axes.flat:
        axis.grid(alpha=0.25)
    figure.suptitle(
        f"What information is carried by the {initial_latent.shape[1]}D latent?"
    )
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def run_rabbit_experiment(
    output_dir: str = "car_acceleration/results/rabbit_wolf_narrow_rates_seed7",
    *,
    seed: int = 7,
    train_samples: int = 3000,
    validation_samples: int = 500,
    test_samples: int = 500,
    history_length: int = 100,
    forecast_horizon: int = 600,
    dt: float = 0.1,
    steps: int = 2000,
    batch_size: int = 64,
    covariance_weight: float = COVARIANCE_WEIGHT,
    device_name: str = "auto",
) -> dict[str, object]:
    """Train from rabbit-wolf histories and save joint-forecast diagnostics."""

    seed_everything(seed)
    device = resolve_device(device_name)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    data_arguments = {
        "history_length": history_length,
        "forecast_horizon": forecast_horizon,
        "dt": dt,
    }
    train = generate_rabbit_dataset(train_samples, seed + 1, **data_arguments)
    validation = generate_rabbit_dataset(
        validation_samples,
        seed + 2,
        **data_arguments,
    )
    test = generate_rabbit_dataset(test_samples, seed + 3, **data_arguments)

    train_log_population = _log_population(train.rabbits)
    log_mean = float(train_log_population.mean())
    log_scale = float(train_log_population.std())
    train_wolf_log_population = _log_population(train.wolves)
    wolf_log_mean = float(train_wolf_log_population.mean())
    wolf_log_scale = float(train_wolf_log_population.std())
    if log_scale < 1e-12 or wolf_log_scale < 1e-12:
        raise ValueError("training populations have zero log variance")

    model = RabbitReservoir(seed=seed).to(device)
    print("precomputing fixed reservoir states...", flush=True)
    train_reservoir_states = _precompute_reservoir_states(
        model,
        train,
        device,
        log_mean,
        log_scale,
        wolf_log_mean,
        wolf_log_scale,
    )
    validation_reservoir_states = _precompute_reservoir_states(
        model,
        validation,
        device,
        log_mean,
        log_scale,
        wolf_log_mean,
        wolf_log_scale,
    )
    test_reservoir_states = _precompute_reservoir_states(
        model,
        test,
        device,
        log_mean,
        log_scale,
        wolf_log_mean,
        wolf_log_scale,
    )
    train_targets = torch.from_numpy(
        _normalized_targets(
            train,
            log_mean,
            log_scale,
            wolf_log_mean,
            wolf_log_scale,
        )
    ).to(device)
    validation_targets = torch.from_numpy(
        _normalized_targets(
            validation,
            log_mean,
            log_scale,
            wolf_log_mean,
            wolf_log_scale,
        )
    ).to(device)
    print("fixed reservoir states cached", flush=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=LR_DECAY_STEPS,
        gamma=LR_DECAY_FACTOR,
    )
    rng = np.random.default_rng(seed + 4)
    history: dict[str, list[float]] = {
        "step": [],
        "train_loss": [],
        "validation_loss": [],
        "learning_rate": [],
        "forecast_horizon": [],
    }

    def validation_loss(horizon: int) -> float:
        model.eval()
        with torch.no_grad():
            validation_target = validation_targets[:, :horizon]
            prediction, _, _ = model.rollout_from_reservoir_state(
                validation_reservoir_states,
                horizon,
            )
            return float(torch.mean((prediction - validation_target) ** 2).cpu())

    for step in range(1, steps + 1):
        model.train()
        active_horizon = forecast_curriculum_horizon(step, forecast_horizon)
        indices = torch.from_numpy(
            rng.integers(0, len(train.history), size=batch_size)
        ).to(device)
        reservoir_state_batch = train_reservoir_states.index_select(0, indices)
        target_tensor = train_targets.index_select(0, indices)[:, :active_horizon]
        prediction, _, initial_latent = model.rollout_from_reservoir_state(
            reservoir_state_batch,
            active_horizon,
        )
        forecast_loss = torch.mean((prediction - target_tensor) ** 2)
        loss = forecast_loss + covariance_weight * covariance_penalty(initial_latent)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if step == 1 or step % 250 == 0 or step == steps:
            history["step"].append(step)
            history["train_loss"].append(float(loss.detach().cpu()))
            history["validation_loss"].append(validation_loss(active_horizon))
            history["learning_rate"].append(optimizer.param_groups[0]["lr"])
            history["forecast_horizon"].append(active_horizon)
            if step == 1 or step % 500 == 0 or step == steps:
                print(
                    f"step={step:6d} train={history['train_loss'][-1]:.6g} "
                    f"validation={history['validation_loss'][-1]:.6g} "
                    f"lr={history['learning_rate'][-1]:.3g} "
                    f"horizon={active_horizon}",
                    flush=True,
                )

    (
        metrics,
        rabbit_prediction,
        wolf_prediction,
        future_latent,
        initial_latent,
    ) = _evaluate(
        model,
        test,
        device,
        log_mean,
        log_scale,
        wolf_log_mean,
        wolf_log_scale,
        test_reservoir_states,
    )
    with torch.no_grad():
        train_latent = model.encode_reservoir_state(
            train_reservoir_states
        ).cpu().numpy()
    train_reservoir_array = train_reservoir_states.cpu().numpy()
    test_reservoir_array = test_reservoir_states.cpu().numpy()
    train_physical_state = _physical_state(train)
    test_physical_state = _physical_state(test)
    reservoir_probe_pooled, reservoir_probe_per_dimension = _held_out_r2(
        train_reservoir_array,
        train_physical_state,
        test_reservoir_array,
        test_physical_state,
    )
    latent_probe_pooled, latent_probe_per_dimension = _held_out_r2(
        train_latent,
        train_physical_state,
        initial_latent,
        test_physical_state,
    )
    metrics["reservoir_to_observable_state_r2"] = reservoir_probe_pooled
    metrics[
        "reservoir_to_observable_state_r2_per_dimension"
    ] = reservoir_probe_per_dimension
    metrics["latent_to_observable_state_r2"] = latent_probe_pooled
    metrics[
        "latent_to_observable_state_r2_per_dimension"
    ] = latent_probe_per_dimension
    metrics["trainable_parameters"] = count_trainable_parameters(model)
    metrics["latent_dimensions"] = model.latent_size
    metrics["reservoir_nodes"] = model.nodes
    metrics["reservoir_history_taps"] = model.history_taps
    metrics["reservoir_feature_size"] = model.reservoir_feature_size
    _print_feature_probe_table(
        reservoir_probe_per_dimension,
        latent_probe_per_dimension,
    )

    torch.save(model.state_dict(), output / "checkpoint.pt")
    np.savez_compressed(
        output / "predictions.npz",
        history=test.history,
        wolf_history=test.wolves[:, : test.history.shape[1]],
        target=test.target,
        prediction=rabbit_prediction,
        wolf_target=test.wolves[:, test.history.shape[1] :],
        wolf_prediction=wolf_prediction,
        rabbits=test.rabbits,
        wolves=test.wolves,
        alpha=test.alpha,
        beta=test.beta,
        gamma=test.gamma,
        delta=test.delta,
        initial_latent=initial_latent,
        future_latent=future_latent,
        dt=np.asarray(test.dt),
        log_mean=np.asarray(log_mean),
        log_scale=np.asarray(log_scale),
        wolf_log_mean=np.asarray(wolf_log_mean),
        wolf_log_scale=np.asarray(wolf_log_scale),
    )
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    (output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    (output / "normalization.json").write_text(
        json.dumps(
            {
                "rabbit_log_mean": log_mean,
                "rabbit_log_scale": log_scale,
                "wolf_log_mean": wolf_log_mean,
                "wolf_log_scale": wolf_log_scale,
            },
            indent=2,
        )
        + "\n"
    )
    _save_training_plot(history, output / "training.png")
    _save_prediction_plot(
        test,
        rabbit_prediction,
        wolf_prediction,
        output / "predictions.png",
    )
    _save_physics_plot(test, output / "ground_truth_dynamics.png")
    _save_latent_plot(
        test,
        initial_latent,
        future_latent,
        output / "latent_dynamics.png",
    )
    print(json.dumps(metrics, indent=2), flush=True)
    print(f"saved results to {output}", flush=True)
    return metrics
