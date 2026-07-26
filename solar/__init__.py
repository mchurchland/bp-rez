"""Copernicus/SciNet representation-discovery experiments."""

from .data import SOLAR_SAMPLING_MODES, SolarDataset, make_solar_splits
from .models import SOLAR_MODEL_NAMES, build_solar_model

__all__ = [
    "SOLAR_MODEL_NAMES",
    "SOLAR_SAMPLING_MODES",
    "SolarDataset",
    "build_solar_model",
    "make_solar_splits",
]
