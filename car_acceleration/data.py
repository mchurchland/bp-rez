"""Synthetic one-dimensional car trajectories."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CarDataset:
    """Position histories, future targets, and the hidden physical state."""

    history: np.ndarray  # [samples, history_length]
    target: np.ndarray  # [samples, forecast_horizon]
    position: np.ndarray  # [samples, total_length]
    velocity: np.ndarray  # [samples, total_length]
    acceleration: np.ndarray  # [samples]


def generate_car_dataset(
    samples: int,
    seed: int,
    *,
    history_length: int = 2,
    forecast_horizon: int = 30,
    dt: float = 0.1,
    acceleration: float | NDArray[float64]  = 0.5,
    acceleration_variable: bool=  False
) -> CarDataset:
    """Generate cars with random initial position/velocity and fixed acceleration."""

    if samples < 1 or history_length < 2 or forecast_horizon < 1 or dt <= 0:
        raise ValueError("invalid car-dataset dimensions or time step")
    rng = np.random.default_rng(seed)
    initial_position = rng.uniform(-5.0, 5.0, size=samples)
    initial_velocity = rng.uniform(-2.0, 2.0, size=samples)
    total_length = history_length + forecast_horizon
    times = dt * np.arange(total_length, dtype=np.float64)[None, :]
    print(times.shape)
    print(times)
    

    if acceleration_variable ==True:
        acceleration = np.linspace(0,5,total_length)
    else:
        pass
        #acceleration = np.full()
    position = (
        initial_position[:, None]
        + initial_velocity[:, None] * times
        + 0.5 * acceleration * times**2
    )
    print(position.shape)
    print(initial_velocity[:, None].shape)
    #quit()
    velocity = initial_velocity[:, None] + acceleration * times
    return CarDataset(
        history=position[:, :history_length].astype(np.float32),
        target=position[:, history_length:].astype(np.float32),
        position=position.astype(np.float32),
        velocity=velocity.astype(np.float32),
        acceleration=np.full(samples, acceleration, dtype=np.float32),
    )
