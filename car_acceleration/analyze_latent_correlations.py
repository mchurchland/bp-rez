#!/usr/bin/env python3
"""Compare physical-state and latent correlations and align latent axes."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import torch


PHYSICAL_LABELS = ("position", "velocity", "acceleration")


def correlation_matrix(values: np.ndarray) -> np.ndarray:
    """Return a feature-by-feature correlation matrix."""

    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError("values must have shape [samples, features >= 2]")
    if len(values) < 2:
        raise ValueError("at least two samples are required")
    correlation = np.corrcoef(values, rowvar=False)
    if not np.all(np.isfinite(correlation)):
        raise ValueError("cannot correlate a feature with zero or invalid variance")
    return correlation


def upper_triangle(matrix: np.ndarray) -> np.ndarray:
    """Return the unique off-diagonal entries of a square matrix."""

    return matrix[np.triu_indices_from(matrix, k=1)]


def infer_acceleration(data: np.lib.npyio.NpzFile, dt: float) -> np.ndarray:
    """Load acceleration when available, otherwise infer it from velocity."""

    history_length = data["history"].shape[1]
    last_observed = history_length - 1
    if "acceleration" in data.files:
        acceleration = data["acceleration"]
        if acceleration.ndim == 2:
            return acceleration[:, last_observed].astype(np.float64)
        if acceleration.ndim == 1:
            return acceleration.astype(np.float64)
        raise ValueError("acceleration must have one or two dimensions")

    velocity = data["velocity"].astype(np.float64)
    if velocity.shape[1] < 2:
        raise ValueError("at least two velocity samples are needed to infer acceleration")
    return (velocity[:, 1] - velocity[:, 0]) / dt


def best_latent_order(
    physical_correlation: np.ndarray,
    latent_correlation: np.ndarray,
) -> tuple[tuple[int, ...], float, np.ndarray]:
    """Find the latent permutation closest to the physical correlation pattern."""

    if physical_correlation.shape != latent_correlation.shape:
        raise ValueError("physical and latent correlation matrices must have equal shape")

    target = upper_triangle(physical_correlation)
    best: tuple[tuple[int, ...], float, np.ndarray] | None = None
    for order in itertools.permutations(range(len(latent_correlation))):
        reordered = latent_correlation[np.ix_(order, order)]
        rmse = float(np.sqrt(np.mean((upper_triangle(reordered) - target) ** 2)))
        if best is None or rmse < best[1]:
            best = (order, rmse, reordered)

    if best is None:  # pragma: no cover - permutations are nonempty for valid input
        raise RuntimeError("could not find a latent-axis ordering")
    return best


def load_latent_recurrence(
    checkpoint_path: Path,
    order: tuple[int, ...],
) -> dict[str, object]:
    """Load and reorder the learned affine latent recurrence."""

    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    missing = sorted({"transition", "transition_bias"}.difference(state))
    if missing:
        raise ValueError(f"missing recurrence tensors in checkpoint: {missing}")

    transition = state["transition"].detach().cpu().numpy().astype(np.float64)
    bias = state["transition_bias"].detach().cpu().numpy().astype(np.float64)
    expected_shape = (len(order), len(order))
    if transition.shape != expected_shape or bias.shape != (len(order),):
        raise ValueError(
            f"expected transition shape {expected_shape} and bias shape "
            f"{(len(order),)}, got {transition.shape} and {bias.shape}"
        )

    reordered_transition = transition[np.ix_(order, order)]
    reordered_bias = bias[list(order)]
    eigenvalues = np.linalg.eigvals(transition)
    return {
        "checkpoint_path": str(checkpoint_path),
        "equation": "z[t+1] = A @ z[t] + b",
        "transition": transition.tolist(),
        "bias": bias.tolist(),
        "reordered_transition": reordered_transition.tolist(),
        "reordered_bias": reordered_bias.tolist(),
        "eigenvalues": [
            {"real": float(value.real), "imaginary": float(value.imag)}
            for value in eigenvalues
        ],
    }


def matrix_text(matrix: np.ndarray, labels: tuple[str, ...]) -> str:
    """Format a small labeled matrix for terminal output."""

    return rectangular_matrix_text(matrix, labels, labels)


def rectangular_matrix_text(
    matrix: np.ndarray,
    row_labels: tuple[str, ...],
    column_labels: tuple[str, ...],
) -> str:
    """Format a labeled rectangular matrix for terminal output."""

    if matrix.shape != (len(row_labels), len(column_labels)):
        raise ValueError("matrix shape does not match its row and column labels")
    width = max(
        10,
        max(len(label) for label in row_labels) + 1,
        max(len(label) for label in column_labels) + 1,
    )
    header = " " * width + "".join(
        f"{label:>{width}}" for label in column_labels
    )
    rows = [header]
    for label, row in zip(row_labels, matrix, strict=True):
        values = "".join(f"{value:>{width}.4f}" for value in row)
        rows.append(f"{label:<{width}}{values}")
    return "\n".join(rows)


def analyze(
    predictions_path: Path,
    dt: float,
    checkpoint_path: Path | None = None,
) -> dict[str, object]:
    """Compute correlations and, for a 3D latent, its physical-axis alignment."""

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
        acceleration = infer_acceleration(data, dt)

    last_observed = history.shape[1] - 1
    end_physical_labels = (f"x{last_observed}", f"v{last_observed}", "a")
    initial_physics = np.column_stack(
        (position[:, 0], velocity[:, 0], acceleration)
    )
    end_physics = np.column_stack(
        (position[:, last_observed], velocity[:, last_observed], acceleration)
    )

    initial_physics_correlation = correlation_matrix(initial_physics)
    end_physics_correlation = correlation_matrix(end_physics)
    history_correlation = correlation_matrix(history)
    latent_correlation = correlation_matrix(latent)
    end_physics_latent_correlation = np.corrcoef(
        end_physics, latent, rowvar=False
    )[: len(PHYSICAL_LABELS), len(PHYSICAL_LABELS) :]

    latent_labels = tuple(f"z{index + 1}" for index in range(latent.shape[1]))
    alignment_available = latent.shape[1] == len(PHYSICAL_LABELS)
    if alignment_available:
        order, alignment_rmse, reordered_latent_correlation = best_latent_order(
            end_physics_correlation,
            latent_correlation,
        )
        ordered_latent_labels = tuple(latent_labels[index] for index in order)
        mapping: dict[str, str] | None = {
            physical: latent_label
            for physical, latent_label in zip(
                PHYSICAL_LABELS,
                ordered_latent_labels,
                strict=True,
            )
        }

        physical_pairs = upper_triangle(end_physics_correlation)
        latent_pairs = upper_triangle(reordered_latent_correlation)
        pair_names = (
            "position_velocity",
            "position_acceleration",
            "velocity_acceleration",
        )
        pairwise_comparison: dict[str, dict[str, float]] | None = {
            name: {
                "physical": float(physical_value),
                "latent": float(latent_value),
                "absolute_difference": float(abs(physical_value - latent_value)),
            }
            for name, physical_value, latent_value in zip(
                pair_names,
                physical_pairs,
                latent_pairs,
                strict=True,
            )
        }
    else:
        order = tuple(range(latent.shape[1]))
        alignment_rmse = None
        reordered_latent_correlation = latent_correlation
        ordered_latent_labels = latent_labels
        mapping = None
        pairwise_comparison = None

    if checkpoint_path is None:
        candidate = predictions_path.with_name("checkpoint.pt")
        checkpoint_path = candidate if candidate.exists() else None
    recurrence = (
        load_latent_recurrence(checkpoint_path, order)
        if checkpoint_path is not None
        else None
    )

    return {
        "predictions_path": str(predictions_path),
        "dt": dt,
        "samples": len(history),
        "latent_size": latent.shape[1],
        "end_physical_labels": list(end_physical_labels),
        "alignment_available": alignment_available,
        "mapping": mapping,
        "latent_order_zero_based": list(order),
        "latent_order": list(ordered_latent_labels),
        "alignment_rmse": alignment_rmse,
        "pairwise_comparison": pairwise_comparison,
        "initial_physics_correlation": initial_physics_correlation.tolist(),
        "end_physics_correlation": end_physics_correlation.tolist(),
        "end_physics_latent_correlation": end_physics_latent_correlation.tolist(),
        "history_correlation": history_correlation.tolist(),
        "latent_correlation": latent_correlation.tolist(),
        "reordered_latent_correlation": reordered_latent_correlation.tolist(),
        "latent_recurrence": recurrence,
    }


def print_analysis(analysis: dict[str, object]) -> None:
    """Print the correlation analysis in a human-readable form."""

    physical = np.asarray(analysis["end_physics_correlation"])
    latent = np.asarray(analysis["latent_correlation"])
    reordered = np.asarray(analysis["reordered_latent_correlation"])
    latent_labels = tuple(f"z{index + 1}" for index in range(len(latent)))
    ordered_labels = tuple(analysis["latent_order"])
    end_physical_labels = tuple(analysis["end_physical_labels"])
    end_physics_latent = np.asarray(analysis["end_physics_latent_correlation"])

    print("Physical correlation at the end of the observed history")
    print(matrix_text(physical, PHYSICAL_LABELS))
    print(
        "\nCorrelation of actual "
        f"({', '.join(end_physical_labels)}) with the initial reservoir latent"
    )
    print(
        rectangular_matrix_text(
            end_physics_latent,
            end_physical_labels,
            latent_labels,
        )
    )
    print("\nOriginal latent correlation")
    print(matrix_text(latent, latent_labels))
    if analysis["alignment_available"]:
        print("\nBest latent order for (position, velocity, acceleration)")
        print("  " + " -> ".join(ordered_labels))
        for physical_name, latent_name in analysis["mapping"].items():
            print(f"  {physical_name:>12} <- {latent_name}")
        print(f"\nCorrelation-pattern RMSE: {analysis['alignment_rmse']:.6f}")
        print("\nReordered latent correlation")
        print(matrix_text(reordered, ordered_labels))
        print("\nPairwise comparison")
        for name, comparison in analysis["pairwise_comparison"].items():
            print(
                f"  {name:<25} physical={comparison['physical']:.4f} "
                f"latent={comparison['latent']:.4f} "
                f"difference={comparison['absolute_difference']:.4f}"
            )
    else:
        print(
            "\nPhysical-axis permutation skipped: it requires a 3D latent "
            f"but this run has {analysis['latent_size']} dimensions."
        )

    recurrence = analysis["latent_recurrence"]
    if recurrence is not None:
        transition = np.asarray(recurrence["transition"])
        bias = np.asarray(recurrence["bias"])
        reordered_transition = np.asarray(recurrence["reordered_transition"])
        reordered_bias = np.asarray(recurrence["reordered_bias"])
        print("\nLatent recurrence in original order")
        print("  z[t+1] = A @ z[t] + b")
        print(matrix_text(transition, latent_labels))
        print("  b = " + np.array2string(bias, precision=6))
        if analysis["alignment_available"]:
            print("\nLatent recurrence in physical-aligned order")
            print("  state order: " + ", ".join(PHYSICAL_LABELS))
            print("  latent order: " + ", ".join(ordered_labels))
            print(matrix_text(reordered_transition, ordered_labels))
            print("  b = " + np.array2string(reordered_bias, precision=6))
        eigenvalue_text = ", ".join(
            (
                f"{value['real']:.6f}{value['imaginary']:+.6f}j"
                if abs(value["imaginary"]) > 1e-12
                else f"{value['real']:.6f}"
            )
            for value in recurrence["eigenvalues"]
        )
        print("  eigenvalues = " + eigenvalue_text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path(
            "car_acceleration/results/linear_readout_2latent_seed8/predictions.npz"
        ),
        help="path to a predictions.npz file",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=1.0,
        help="simulation time step, used when acceleration is not saved",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        help="optional path for the complete machine-readable analysis",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help=(
            "checkpoint.pt to inspect; defaults to the checkpoint beside "
            "predictions.npz"
        ),
    )
    args = parser.parse_args()

    analysis = analyze(args.predictions, args.dt, args.checkpoint)
    print_analysis(analysis)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(analysis, indent=2) + "\n")
        print(f"\nsaved analysis to {args.output_json}")


if __name__ == "__main__":
    main()
