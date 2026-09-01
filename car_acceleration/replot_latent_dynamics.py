"""Regenerate the car latent-dynamics GIF and static snapshot without retraining."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .data import CarDataset
from .experiment import _save_latent_animation, _save_latent_plot


DEFAULT_PREDICTIONS = (
    "car_acceleration/results/linear_readout_2latent_positive_seed8/predictions.npz"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", default=DEFAULT_PREDICTIONS)
    parser.add_argument("--output", default=None, help="output GIF path")
    args = parser.parse_args()

    predictions_path = Path(args.predictions)
    if not predictions_path.exists():
        parser.error(f"predictions file does not exist: {predictions_path}")
    gif_path = Path(args.output) if args.output else predictions_path.with_name(
        "latent_dynamics.gif"
    )

    with np.load(predictions_path) as data:
        required = {
            "history", "target", "position", "velocity",
            "initial_latent", "future_latent",
        }
        missing = required.difference(data.files)
        if missing:
            parser.error(f"predictions file is missing arrays: {sorted(missing)}")
        arrays = {name: data[name] for name in data.files}

    acceleration = arrays.get("acceleration")
    if acceleration is None:
        acceleration = np.empty_like(arrays["velocity"])
        acceleration[:, :-1] = np.diff(arrays["velocity"], axis=1)
        acceleration[:, -1] = acceleration[:, -2]

    dataset = CarDataset(
        history=arrays["history"], target=arrays["target"],
        position=arrays["position"], velocity=arrays["velocity"],
        acceleration=acceleration,
    )
    _save_latent_animation(
        dataset, arrays["initial_latent"], arrays["future_latent"], gif_path
    )
    _save_latent_plot(
        dataset, arrays["initial_latent"], arrays["future_latent"],
        gif_path.with_suffix(".png"),
    )
    print(f"saved {gif_path} and {gif_path.with_suffix('.png')}")


if __name__ == "__main__":
    main()
