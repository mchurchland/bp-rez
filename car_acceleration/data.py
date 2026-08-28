"""Synthetic one-dimensional car trajectories."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CarDataset:
    """Position histories, future targets, and hidden physical states."""

    history: np.ndarray       # [samples, history_length]
    target: np.ndarray        # [samples, forecast_horizon]
    position: np.ndarray      # [samples, total_length]
    velocity: np.ndarray      # [samples, total_length]
    acceleration: np.ndarray  # [samples, total_length]


@dataclass(frozen=True)
class RabbitDataset:
    """Predator-prey trajectories plus the legacy rabbit history/target views."""

    history: np.ndarray   # [samples, history_length]
    target: np.ndarray    # [samples, forecast_horizon]
    rabbits: np.ndarray   # [samples, total_length]
    wolves: np.ndarray    # [samples, total_length]
    alpha: np.ndarray     # [samples]
    beta: np.ndarray      # [samples]
    gamma: np.ndarray     # [samples]
    delta: np.ndarray     # [samples]
    dt: float


RABBIT_RATE_RANGES = {
    "alpha": (0.8, 1.2),
    "beta": (0.08, 0.12),
    "gamma": (0.8, 1.2),
    "delta": (0.08, 0.12),
}


def generate_car_dataset(
    samples: int,
    seed: int,
    *,
    history_length: int = 5,
    forecast_horizon: int = 50,
    dt: float = 1.0,
    acc: float = 0.5,
    acceleration_variable: bool = True,
) -> CarDataset:
    """Generate 1D car trajectories.

    Each car has a random initial position and velocity.

    If acceleration_variable is True, each car receives its own linearly
    varying acceleration:

        a_i(t) = offset_i + slope_i * t

    Otherwise every car has constant acceleration `acc`.
    """

    if (
        samples < 1
        or history_length < 2
        or forecast_horizon < 1
        or dt <= 0
    ):
        raise ValueError("invalid car-dataset dimensions or time step")

    rng = np.random.default_rng(seed)

    initial_position = rng.uniform(
        -5.0,
        5.0,
        size=samples,
    )

    initial_velocity = rng.uniform(
        -2.0,
        2.0,
        size=samples,
    )

    total_length = history_length + forecast_horizon

    # ------------------------------------------------------------
    # Acceleration
    # ------------------------------------------------------------

    if acceleration_variable:
        # Each trajectory gets an independently sampled acceleration law.
        acceleration_per_car = rng.uniform(
            -1.0,
            1.0,
            size=samples,
        )

        acceleration = np.repeat(
            acceleration_per_car[:, None],
            total_length,
            axis=1,
        )

    else:
        acceleration = np.full(
            (samples, total_length),
            acc,
            dtype=np.float64,
        )

    # ------------------------------------------------------------
    # State arrays
    # ------------------------------------------------------------

    position = np.zeros(
        (samples, total_length),
        dtype=np.float64,
    )

    velocity = np.zeros(
        (samples, total_length),
        dtype=np.float64,
    )

    position[:, 0] = initial_position
    velocity[:, 0] = initial_velocity

    # ------------------------------------------------------------
    # Integrate dynamics
    # ------------------------------------------------------------

    for t in range(1, total_length):
        a_previous = acceleration[:, t - 1]
        a_current = acceleration[:, t]

        # Acceleration changes linearly over each interval, so jerk is
        # constant within that interval.
        jerk = (a_current - a_previous) / dt

        position[:, t] = (
            position[:, t - 1]
            + velocity[:, t - 1] * dt
            + 0.5 * a_previous * dt**2
            + (1.0 / 6.0) * jerk * dt**3
        )

        velocity[:, t] = (
            velocity[:, t - 1]
            + a_previous * dt
            + 0.5 * jerk * dt**2
        )

    # ------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------

    return CarDataset(
        history=position[:, :history_length].astype(np.float32),
        target=position[:, history_length:].astype(np.float32),
        position=position.astype(np.float32),
        velocity=velocity.astype(np.float32),
        acceleration=acceleration.astype(np.float32),
    )


def generate_rabbit_dataset(
    samples: int,
    seed: int,
    *,
    history_length: int = 100,
    forecast_horizon: int = 100,
    dt: float = 0.1,
    integration_substeps: int = 10,
) -> RabbitDataset:
    """Generate paired rabbit-wolf Lotka-Volterra trajectories.

    Every trajectory receives independently sampled initial populations and
    rate constants. ``history`` and ``target`` remain rabbit-only compatibility
    views; the joint experiment slices the matching wolf history and target
    from ``wolves``.
    """

    if (
        samples < 1
        or history_length < 2
        or forecast_horizon < 1
        or dt <= 0
        or integration_substeps < 1
    ):
        raise ValueError("invalid rabbit-dataset dimensions or time step")

    rng = np.random.default_rng(seed)
    initial_rabbit = 10 ** rng.uniform(0, 2, size=samples)  # 1–100
    initial_wolf = 10 ** rng.uniform(-1, 1.5, size=samples)  # 0.1–31.6
    total_length = history_length + forecast_horizon

    alpha = rng.uniform(*RABBIT_RATE_RANGES["alpha"], size=samples)
    beta = rng.uniform(*RABBIT_RATE_RANGES["beta"], size=samples)
    gamma = rng.uniform(*RABBIT_RATE_RANGES["gamma"], size=samples)
    delta = rng.uniform(*RABBIT_RATE_RANGES["delta"], size=samples)

    rabbits = np.zeros(
        (samples, total_length),
        dtype=np.float64,
    )

    wolves = np.zeros(
        (samples, total_length),
        dtype=np.float64,
    )

    rabbits[:, 0] = initial_rabbit
    wolves[:, 0] = initial_wolf

    # ------------------------------------------------------------
    # Integrate dynamics
    # ------------------------------------------------------------

    integration_dt = dt / integration_substeps

    def derivative(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return alpha * x - beta * x * y, delta * x * y - gamma * y

    for t in range(1, total_length):
        x = rabbits[:, t - 1].copy()
        y = wolves[:, t - 1].copy()
        for _ in range(integration_substeps):
            k1_x, k1_y = derivative(x, y)
            k2_x, k2_y = derivative(
                x + 0.5 * integration_dt * k1_x,
                y + 0.5 * integration_dt * k1_y,
            )
            k3_x, k3_y = derivative(
                x + 0.5 * integration_dt * k2_x,
                y + 0.5 * integration_dt * k2_y,
            )
            k4_x, k4_y = derivative(
                x + integration_dt * k3_x,
                y + integration_dt * k3_y,
            )
            x = x + (integration_dt / 6.0) * (
                k1_x + 2.0 * k2_x + 2.0 * k3_x + k4_x
            )
            y = y + (integration_dt / 6.0) * (
                k1_y + 2.0 * k2_y + 2.0 * k3_y + k4_y
            )
            np.maximum(x, 0.0, out=x)
            np.maximum(y, 0.0, out=y)
        rabbits[:, t] = x
        wolves[:, t] = y

    if not np.all(np.isfinite(rabbits)) or not np.all(np.isfinite(wolves)):
        raise FloatingPointError("rabbit integration produced non-finite values")

    return RabbitDataset(
        history=rabbits[:, :history_length].astype(np.float32),
        target=rabbits[:, history_length:].astype(np.float32),
        rabbits=rabbits.astype(np.float32),
        wolves=wolves.astype(np.float32),
        alpha=alpha.astype(np.float32),
        beta=beta.astype(np.float32),
        gamma=gamma.astype(np.float32),
        delta=delta.astype(np.float32),
        dt=float(dt),
    )
