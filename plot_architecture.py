#!/usr/bin/env python3
"""Draw a configurable neuron-level deep reservoir architecture.

The default is a five-reservoir conceptual extension with 300 recurrent
neurons total (60 per reservoir) and four 10-neuron latent spaces.
Representative edges are rendered to keep the graph readable.
"""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "bp_reservoir_mpl"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch


FIXED = "#2878B5"
TRAINABLE = "#E07A1F"
NODE_FILL = "#CFE5F5"
LATENT_FILL = "#FFD49A"
INK = "#263238"


def disk_positions(
    center: tuple[float, float], count: int, radius: float, seed: int
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    angles = rng.uniform(0.0, 2.0 * np.pi, count)
    distances = radius * np.sqrt(rng.uniform(0.035, 0.88, count))
    return np.column_stack(
        (center[0] + distances * np.cos(angles), center[1] + distances * np.sin(angles))
    )


def latent_positions(center: tuple[float, float], count: int) -> np.ndarray:
    columns = 2
    rows = int(np.ceil(count / columns))
    y_values = np.linspace(center[1] + 0.64, center[1] - 0.64, rows)
    positions = [(center[0] + dx, y) for y in y_values for dx in (-0.13, 0.13)]
    return np.asarray(positions[:count])


def random_pairs(
    rng: np.random.Generator, sources: int, targets: int, count: int
) -> list[tuple[int, int]]:
    return list(
        zip(
            rng.integers(0, sources, count).tolist(),
            rng.integers(0, targets, count).tolist(),
            strict=True,
        )
    )


def draw_edges(
    axis: plt.Axes,
    starts: np.ndarray,
    ends: np.ndarray,
    pairs: list[tuple[int, int]],
    color: str,
    alpha: float,
    linewidth: float = 0.45,
) -> None:
    for source, target in pairs:
        axis.plot(
            [starts[source, 0], ends[target, 0]],
            [starts[source, 1], ends[target, 1]],
            color=color,
            alpha=alpha,
            linewidth=linewidth,
            zorder=1,
        )


def connection_arrow(
    axis: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    label: str,
    color: str,
) -> None:
    axis.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=1.8,
            color=color,
            zorder=7,
        )
    )
    axis.text(
        (start[0] + end[0]) / 2,
        start[1] + 0.12,
        label,
        ha="center",
        va="bottom",
        fontsize=7.8,
        color=color,
        fontweight="bold",
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.5, "alpha": 0.9},
        zorder=8,
    )


def recurrent_loop(axis: plt.Axes, center: tuple[float, float], index: int) -> None:
    axis.add_patch(
        FancyArrowPatch(
            (center[0] - 0.55, center[1] + 1.03),
            (center[0] + 0.55, center[1] + 1.03),
            connectionstyle="arc3,rad=-0.55",
            arrowstyle="-|>",
            mutation_scale=11,
            linewidth=1.45,
            color=FIXED,
            zorder=6,
        )
    )
    axis.text(
        center[0],
        center[1] + 1.48,
        f"$A_{index}$ fixed",
        ha="center",
        fontsize=8.5,
        color=FIXED,
        fontweight="bold",
    )


def draw_architecture(
    output: Path,
    reservoirs: int = 5,
    total_nodes: int = 300,
    latent_size: int = 10,
    dpi: int = 240,
) -> None:
    if reservoirs < 2:
        raise ValueError("at least two reservoirs are required")
    if total_nodes % reservoirs != 0:
        raise ValueError("total nodes must be divisible by the number of reservoirs")
    if latent_size < 1:
        raise ValueError("latent size must be positive")

    output.parent.mkdir(parents=True, exist_ok=True)
    nodes_per_reservoir = total_nodes // reservoirs
    rng = np.random.default_rng(2026)
    center_y = 3.25
    spacing = 3.05
    first_x = 2.0
    reservoir_centers = [(first_x + index * spacing, center_y) for index in range(reservoirs)]
    latent_centers = [
        ((reservoir_centers[index][0] + reservoir_centers[index + 1][0]) / 2, center_y)
        for index in range(reservoirs - 1)
    ]
    reservoir_nodes = [
        disk_positions(center, nodes_per_reservoir, 0.86, seed=index + 1)
        for index, center in enumerate(reservoir_centers)
    ]
    latent_nodes = [latent_positions(center, latent_size) for center in latent_centers]
    input_position = np.asarray([[0.42, center_y]])
    output_position = np.asarray([[reservoir_centers[-1][0] + 1.62, center_y]])
    x_limit = output_position[0, 0] + 0.55

    figure, axis = plt.subplots(figsize=(24, 7.8))
    axis.set_xlim(-0.1, x_limit)
    axis.set_ylim(0.0, 6.4)
    axis.set_aspect("equal")
    axis.axis("off")

    for index, (center, nodes) in enumerate(
        zip(reservoir_centers, reservoir_nodes, strict=True), start=1
    ):
        axis.add_patch(
            Circle(
                center,
                1.0,
                facecolor="#F4F9FC",
                edgecolor=FIXED,
                linewidth=1.8,
                zorder=0,
            )
        )
        draw_edges(
            axis,
            nodes,
            nodes,
            random_pairs(rng, nodes_per_reservoir, nodes_per_reservoir, nodes_per_reservoir * 3),
            FIXED,
            alpha=0.085,
        )
        recurrent_loop(axis, center, index)

    input_sources = np.repeat(input_position, min(30, nodes_per_reservoir), axis=0)
    input_targets = rng.choice(
        nodes_per_reservoir, len(input_sources), replace=len(input_sources) > nodes_per_reservoir
    )
    draw_edges(
        axis,
        input_sources,
        reservoir_nodes[0],
        [(index, int(target)) for index, target in enumerate(input_targets)],
        FIXED,
        alpha=0.18,
        linewidth=0.55,
    )

    for index, latent in enumerate(latent_nodes):
        edge_count = latent_size * 12
        draw_edges(
            axis,
            reservoir_nodes[index],
            latent,
            random_pairs(rng, nodes_per_reservoir, latent_size, edge_count),
            TRAINABLE,
            alpha=0.11,
            linewidth=0.5,
        )
        draw_edges(
            axis,
            latent,
            reservoir_nodes[index + 1],
            random_pairs(rng, latent_size, nodes_per_reservoir, edge_count),
            FIXED,
            alpha=0.11,
            linewidth=0.5,
        )

    output_sources = rng.choice(
        nodes_per_reservoir, min(35, nodes_per_reservoir), replace=False
    )
    output_targets = np.repeat(output_position, len(output_sources), axis=0)
    draw_edges(
        axis,
        reservoir_nodes[-1],
        output_targets,
        [(int(source), index) for index, source in enumerate(output_sources)],
        TRAINABLE,
        alpha=0.18,
        linewidth=0.55,
    )

    for index, nodes in enumerate(reservoir_nodes, start=1):
        axis.scatter(
            nodes[:, 0],
            nodes[:, 1],
            s=18,
            facecolor=NODE_FILL,
            edgecolor=FIXED,
            linewidth=0.5,
            zorder=3,
        )
        center = reservoir_centers[index - 1]
        axis.text(
            center[0],
            1.83,
            f"$x_{index}(t)$",
            ha="center",
            fontsize=12,
            fontweight="bold",
        )
        axis.text(
            center[0],
            1.57,
            f"Reservoir {index}\n{nodes_per_reservoir} neurons",
            ha="center",
            va="top",
            fontsize=8.7,
            linespacing=1.25,
        )

    for index, nodes in enumerate(latent_nodes, start=1):
        axis.scatter(
            nodes[:, 0],
            nodes[:, 1],
            s=72,
            facecolor=LATENT_FILL,
            edgecolor=TRAINABLE,
            linewidth=1.1,
            zorder=4,
        )
        for neuron_index, position in enumerate(nodes, start=1):
            axis.text(
                position[0],
                position[1],
                str(neuron_index),
                ha="center",
                va="center",
                fontsize=5.2,
                zorder=5,
            )
        axis.text(
            latent_centers[index - 1][0],
            1.48,
            f"$h_{index}(t)$\n{latent_size} latent",
            ha="center",
            va="top",
            fontsize=8.5,
            linespacing=1.2,
        )

    axis.add_patch(
        Circle(tuple(input_position[0]), 0.23, facecolor="#ECEFF1", edgecolor=INK, linewidth=1.6, zorder=5)
    )
    axis.add_patch(
        Circle(tuple(output_position[0]), 0.24, facecolor="#DDF3E4", edgecolor=INK, linewidth=1.6, zorder=5)
    )
    axis.text(*input_position[0], "$u(t)$", ha="center", va="center", fontsize=10, zorder=6)
    axis.text(*output_position[0], "$\\hat{y}(t)$", ha="center", va="center", fontsize=10, zorder=6)

    connection_arrow(axis, (0.70, center_y), (0.98, center_y), "$B_1$", FIXED)
    for index, latent_center in enumerate(latent_centers, start=1):
        left_center = reservoir_centers[index - 1]
        right_center = reservoir_centers[index]
        connection_arrow(
            axis,
            (left_center[0] + 1.04, center_y),
            (latent_center[0] - 0.27, center_y),
            f"$W_{index},b_{index}$",
            TRAINABLE,
        )
        connection_arrow(
            axis,
            (latent_center[0] + 0.27, center_y),
            (right_center[0] - 1.04, center_y),
            f"$R_{index}$",
            FIXED,
        )
    connection_arrow(
        axis,
        (reservoir_centers[-1][0] + 1.04, center_y),
        (output_position[0, 0] - 0.29, center_y),
        "$W_{out},c$",
        TRAINABLE,
    )

    axis.text(
        x_limit / 2,
        6.05,
        f"{reservoirs}-Reservoir Deep Network with Trainable Latent Readouts",
        ha="center",
        fontsize=19,
        fontweight="bold",
    )
    axis.text(
        x_limit / 2,
        5.68,
        f"{total_nodes} recurrent neurons total  •  {nodes_per_reservoir} per reservoir  •  "
        f"{reservoirs - 1} × {latent_size}-neuron latent spaces",
        ha="center",
        fontsize=11,
        color="#455A64",
    )

    legend = [
        Line2D([0], [0], color=FIXED, lw=2, label="Fixed random weights"),
        Line2D([0], [0], color=TRAINABLE, lw=2, label="Trainable weights"),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=NODE_FILL,
            markeredgecolor=FIXED,
            markersize=6,
            label="Reservoir neuron",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=LATENT_FILL,
            markeredgecolor=TRAINABLE,
            markersize=7,
            label="Latent neuron",
        ),
    ]
    axis.legend(
        handles=legend,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.01),
        ncol=4,
        frameon=False,
        fontsize=9,
    )
    axis.text(
        x_limit / 2,
        0.43,
        "All 300 reservoir neurons and all 40 latent neurons are drawn; edges are sampled for readability.",
        ha="center",
        fontsize=8.7,
        color="#607D8B",
    )

    figure.tight_layout(pad=0.25)
    figure.savefig(output, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("figures/five_reservoir_architecture.png"),
    )
    parser.add_argument("--reservoirs", type=int, default=5)
    parser.add_argument("--total-nodes", type=int, default=300)
    parser.add_argument("--latent-size", type=int, default=10)
    parser.add_argument("--dpi", type=int, default=240)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    draw_architecture(
        args.output,
        reservoirs=args.reservoirs,
        total_nodes=args.total_nodes,
        latent_size=args.latent_size,
        dpi=args.dpi,
    )
    print(f"Saved {args.reservoirs}-reservoir architecture diagram to {args.output}")


if __name__ == "__main__":
    main()
