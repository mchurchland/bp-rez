#!/usr/bin/env python3
"""Print learned latent dynamics and their physical-coordinate transform."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch


PHYSICAL_LABELS = ("position", "velocity", "acceleration")


def load_recurrence(checkpoint: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load the transition matrix and bias from a model checkpoint."""

    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    missing = sorted({"transition", "transition_bias"}.difference(state))
    if missing:
        raise ValueError(f"missing recurrence tensors in checkpoint: {missing}")

    transition = state["transition"].detach().cpu().numpy().astype(np.float64)
    bias = state["transition_bias"].detach().cpu().numpy().astype(np.float64)
    if transition.ndim != 2 or transition.shape[0] != transition.shape[1]:
        raise ValueError("transition must be a square matrix")
    if bias.shape != (transition.shape[0],):
        raise ValueError("transition bias has an incompatible shape")
    return transition, bias


def fit_latent_to_physical(
    predictions_path: Path,
    latent_size: int,
    dt: float,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray, float]:
    """Fit the affine probe ``physical = W @ latent + offset``."""

    if dt <= 0.0:
        raise ValueError("dt must be positive")

    with np.load(predictions_path) as data:
        required = {"history", "position", "velocity", "initial_latent"}
        missing = sorted(required.difference(data.files))
        if missing:
            raise ValueError(f"missing arrays in predictions file: {missing}")

        history = data["history"].astype(np.float64)
        position = data["position"].astype(np.float64)
        velocity = data["velocity"].astype(np.float64)
        latent = data["initial_latent"].astype(np.float64)
        acceleration = (
            data["acceleration"].astype(np.float64)
            if "acceleration" in data.files
            else None
        )

    if latent.ndim != 2 or latent.shape[1] != latent_size:
        raise ValueError(
            f"expected initial_latent shape [samples, {latent_size}], "
            f"got {latent.shape}"
        )
    if history.ndim != 2 or history.shape[1] < 2:
        raise ValueError("history must contain at least two observations")

    last_observed = history.shape[1] - 1
    if position.shape != velocity.shape or position.shape[0] != len(latent):
        raise ValueError("position, velocity, and latent arrays are incompatible")
    if position.shape[1] <= last_observed:
        raise ValueError("physical-state arrays are shorter than the history")

    if acceleration is None:
        acceleration_at_history_end = (
            velocity[:, last_observed] - velocity[:, last_observed - 1]
        ) / dt
    elif acceleration.ndim == 1:
        acceleration_at_history_end = acceleration
    elif acceleration.ndim == 2 and acceleration.shape[1] > last_observed:
        acceleration_at_history_end = acceleration[:, last_observed]
    else:
        raise ValueError("acceleration has an incompatible shape")

    physical = np.column_stack(
        (
            position[:, last_observed],
            velocity[:, last_observed],
            acceleration_at_history_end,
        )
    )
    if physical.shape[1] != latent_size:
        raise ValueError(
            "physicalization requires a 3D latent for position, velocity, "
            "and acceleration"
        )

    latent_augmented = np.column_stack((latent, np.ones(len(latent))))
    coefficients = np.linalg.lstsq(latent_augmented, physical, rcond=None)[0]
    weight = coefficients[:-1].T
    offset = coefficients[-1]
    if np.linalg.matrix_rank(weight) < latent_size:
        raise ValueError("fitted latent-to-physical matrix W is singular")

    fitted_physical = latent @ weight.T + offset
    residual = np.sum((physical - fitted_physical) ** 2, axis=0)
    total = np.sum((physical - physical.mean(axis=0)) ** 2, axis=0)
    r_squared_per_dimension = 1.0 - residual / np.maximum(total, 1e-12)
    pooled_r_squared = 1.0 - float(residual.sum()) / max(float(total.sum()), 1e-12)
    condition_number = float(np.linalg.cond(weight))
    return (
        weight,
        offset,
        pooled_r_squared,
        r_squared_per_dimension,
        condition_number,
    )


def physicalize_recurrence(
    transition: np.ndarray,
    bias: np.ndarray,
    weight: np.ndarray,
    offset: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Transform latent dynamics into the fitted physical coordinates."""

    physical_transition = weight @ transition @ np.linalg.inv(weight)
    physical_bias = (
        weight @ bias + offset - physical_transition @ offset
    )
    return physical_transition, physical_bias


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "car_acceleration/results/linear_readout_2latent_seed8/checkpoint.pt"
        ),
        help="checkpoint.pt produced at the end of training",
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=6,
        help="number of decimal places to display",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        help=(
            "predictions.npz used to fit physical coordinates; defaults to "
            "the file beside the checkpoint"
        ),
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=1.0,
        help="simulation time step used to infer acceleration (default: 1)",
    )
    args = parser.parse_args()
    if args.precision < 0:
        parser.error("--precision must be nonnegative")
    if args.dt <= 0.0:
        parser.error("--dt must be positive")

    transition, bias = load_recurrence(args.checkpoint)
    predictions_path = args.predictions or args.checkpoint.with_name("predictions.npz")
    if not predictions_path.exists():
        parser.error(f"predictions file does not exist: {predictions_path}")
    (
        weight,
        offset,
        probe_r_squared,
        probe_r_squared_per_dimension,
        condition_number,
    ) = fit_latent_to_physical(predictions_path, len(bias), args.dt)
    physical_transition, physical_bias = physicalize_recurrence(
        transition,
        bias,
        weight,
        offset,
    )
    eigenvalues = np.linalg.eigvals(transition)
    labels = ", ".join(f"z{index + 1}" for index in range(len(bias)))

    print(f"checkpoint: {args.checkpoint}")
    print(f"latent order: ({labels})")
    print("equation: z[t+1] = A @ z[t] + b")
    print("\nA =")
    print(
        np.array2string(
            transition,
            precision=args.precision,
            suppress_small=False,
        )
    )
    print("\nb =")
    print(
        np.array2string(
            bias,
            precision=args.precision,
            suppress_small=False,
        )
    )
    print("\neigenvalues =")
    print(
        np.array2string(
            eigenvalues,
            precision=args.precision,
            suppress_small=False,
        )
    )
    print(f"\nphysical-state data: {predictions_path}")
    print("physical order: (position, velocity, acceleration)")
    print("probe equation: physical = W @ z + c")
    print("\nW =")
    print(
        np.array2string(
            weight,
            precision=args.precision,
            suppress_small=False,
        )
    )
    print("\nc =")
    print(
        np.array2string(
            offset,
            precision=args.precision,
            suppress_small=False,
        )
    )
    print(f"\nprobe pooled R^2 = {probe_r_squared:.{args.precision}f}")
    print("probe R^2 by physical coordinate:")
    for label, value in zip(
        PHYSICAL_LABELS,
        probe_r_squared_per_dimension,
        strict=True,
    ):
        print(f"  {label:>12} = {value:.{args.precision}f}")
    print(f"condition number of W = {condition_number:.{args.precision}f}")
    print("\nphysicalized equation: s[t+1] = A_physicalized @ s[t] + b_physicalized")
    print("A_physicalized = W @ A @ inv(W)")
    print("\nA_physicalized =")
    print(
        np.array2string(
            physical_transition,
            precision=args.precision,
            suppress_small=False,
        )
    )
    print("\nb_physicalized =")
    print(
        np.array2string(
            physical_bias,
            precision=args.precision,
            suppress_small=False,
        )
    )


if __name__ == "__main__":
    main()
