"""Synthetic Earth-view observations for the Copernicus/SciNet experiment.

The simulation follows the circular-orbit setup from Iten et al. (2020):
Earth and Mars move at constant angular velocity, while the network only sees
the angles of the Sun and Mars measured from Earth.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


EARTH_ORBIT_RADIUS = 1.0
MARS_ORBIT_RADIUS = 1.52366231
# The public experiment uses 365.26 days when constructing the historical
# start-phase catalog and 365 days for subsequent simulated time evolution.
EARTH_CATALOG_PERIOD_DAYS = 365.26
EARTH_PERIOD_DAYS = 365.0
MARS_PERIOD_DAYS = 686.97959
COPERNICUS_LIFETIME_DAYS = 25_657
EARTH_ANGLE_AT_BIRTH = 0.965
MARS_ANGLE_AT_BIRTH = 5.938
SOLAR_SAMPLING_MODES = ("independent_catalog", "coupled_catalog", "continuous")


@dataclass(frozen=True)
class SolarDataset:
    """A collection of solar-system forecasting examples.

    ``observation`` contains only the two Earth-view angles at the initial
    time. ``target`` contains their complete unwrapped time series. The
    heliocentric angles are retained strictly for post-training analysis and
    are never passed to a model during training or prediction.
    """

    observation: torch.Tensor
    target: torch.Tensor
    heliocentric: torch.Tensor

    def __len__(self) -> int:
        return len(self.observation)


def earth_view_angles(phi_earth: np.ndarray, phi_mars: np.ndarray) -> np.ndarray:
    """Return ``(theta_sun, theta_mars)`` observed from Earth.

    Inputs are broadcast-compatible heliocentric angles in radians. The Sun's
    apparent angle uses the convention from the public SciNet implementation,
    ``theta_sun = phi_earth``.
    """

    distance = np.sqrt(
        MARS_ORBIT_RADIUS**2
        + EARTH_ORBIT_RADIUS**2
        - 2.0
        * MARS_ORBIT_RADIUS
        * EARTH_ORBIT_RADIUS
        * np.cos(phi_mars - phi_earth)
    )
    sin_theta_mars = (
        EARTH_ORBIT_RADIUS * np.sin(phi_earth)
        - MARS_ORBIT_RADIUS * np.sin(phi_mars)
    ) / distance
    cos_theta_mars = (
        EARTH_ORBIT_RADIUS * np.cos(phi_earth)
        - MARS_ORBIT_RADIUS * np.cos(phi_mars)
    ) / distance
    theta_mars = np.arctan2(sin_theta_mars, cos_theta_mars)
    return np.stack((phi_earth, theta_mars), axis=-1)


def _initial_heliocentric_angles(
    samples: int,
    rng: np.random.Generator,
    delta_days: float,
    lifetime_days: int,
    sampling_mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    if sampling_mode not in SOLAR_SAMPLING_MODES:
        raise ValueError(
            f"unknown sampling mode {sampling_mode!r}; choose from {SOLAR_SAMPLING_MODES}"
        )
    if sampling_mode == "continuous":
        return (
            rng.uniform(0.0, 2.0 * np.pi, size=samples),
            rng.uniform(0.0, 2.0 * np.pi, size=samples),
        )

    catalog_size = int(lifetime_days / delta_days)
    if catalog_size < 2:
        raise ValueError("lifetime_days must contain at least two observation intervals")
    offsets = delta_days * np.arange(catalog_size, dtype=np.float64)
    earth_catalog = (
        EARTH_ANGLE_AT_BIRTH
        + 2.0 * np.pi * offsets / EARTH_CATALOG_PERIOD_DAYS
    )
    mars_catalog = MARS_ANGLE_AT_BIRTH + 2.0 * np.pi * offsets / MARS_PERIOD_DAYS
    if sampling_mode == "coupled_catalog":
        indices = rng.integers(0, catalog_size, size=samples)
        earth_indices = mars_indices = indices
    else:
        # The public SciNet experiment independently samples the starting
        # phases. This covers the two-dimensional state space while keeping
        # each phase on the weekly Copernicus-lifetime observation catalog.
        earth_indices = rng.integers(0, catalog_size, size=samples)
        mars_indices = rng.integers(0, catalog_size, size=samples)
    return (
        np.mod(earth_catalog[earth_indices], 2.0 * np.pi),
        np.mod(mars_catalog[mars_indices], 2.0 * np.pi),
    )


def generate_solar_dataset(
    samples: int,
    series_length: int,
    seed: int,
    *,
    delta_days: float = 7.0,
    lifetime_days: int = COPERNICUS_LIFETIME_DAYS,
    sampling_mode: str = "independent_catalog",
) -> SolarDataset:
    """Generate deterministic circular-orbit solar-system sequences."""

    if samples < 1:
        raise ValueError("samples must be positive")
    if series_length < 2:
        raise ValueError("series_length must be at least two")
    if delta_days <= 0.0:
        raise ValueError("delta_days must be positive")

    rng = np.random.default_rng(seed)
    phi_earth_0, phi_mars_0 = _initial_heliocentric_angles(
        samples, rng, delta_days, lifetime_days, sampling_mode
    )
    elapsed = delta_days * np.arange(series_length, dtype=np.float64)
    phi_earth = phi_earth_0[:, None] + 2.0 * np.pi * elapsed / EARTH_PERIOD_DAYS
    phi_mars = phi_mars_0[:, None] + 2.0 * np.pi * elapsed / MARS_PERIOD_DAYS
    target = earth_view_angles(phi_earth, phi_mars)
    # The loss in the original experiment is ordinary MSE, not a circular
    # loss, so remove the +/-pi branch jumps independently in every sequence.
    target[..., 1] = np.unwrap(target[..., 1], axis=1)
    heliocentric = np.stack((phi_earth, phi_mars), axis=-1)
    target_tensor = torch.from_numpy(target.astype(np.float32))
    return SolarDataset(
        observation=target_tensor[:, 0].clone(),
        target=target_tensor,
        heliocentric=torch.from_numpy(heliocentric.astype(np.float32)),
    )


def make_solar_splits(
    train_samples: int,
    validation_samples: int,
    test_samples: int,
    series_length: int,
    data_seed: int,
    *,
    delta_days: float = 7.0,
    lifetime_days: int = COPERNICUS_LIFETIME_DAYS,
    sampling_mode: str = "independent_catalog",
) -> dict[str, SolarDataset]:
    """Create independent, deterministic train/validation/test draws."""

    seed_sequence = np.random.SeedSequence(data_seed)
    child_seeds = [int(child.generate_state(1)[0]) for child in seed_sequence.spawn(3)]
    counts = (train_samples, validation_samples, test_samples)
    return {
        name: generate_solar_dataset(
            count,
            series_length,
            seed,
            delta_days=delta_days,
            lifetime_days=lifetime_days,
            sampling_mode=sampling_mode,
        )
        for name, count, seed in zip(
            ("train", "validation", "test"), counts, child_seeds, strict=True
        )
    }
