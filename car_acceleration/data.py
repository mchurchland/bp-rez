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


def generate_car_dataset(
    samples: int,
    seed: int,
    *,
    history_length: int = 5,
    forecast_horizon: int = 30,
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

    times = dt * np.arange(
        total_length,
        dtype=np.float64,
    )

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
