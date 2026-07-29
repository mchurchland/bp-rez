#!/usr/bin/env python3
"""Draw compact presentation diagrams for the solar reservoir experiments."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Iterable

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "bp_reservoir_mpl")
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch


FIXED = "#2878B5"
FIXED_FILL = "#EDF6FC"
TRAINABLE = "#E07A1F"
TRAINABLE_FILL = "#FFE1B8"
PRESERVED = "#B8860B"
PRESERVED_FILL = "#FFF0B8"
SCINET = "#7651B5"
SCINET_FILL = "#EEE8FA"
OUTPUT = "#2E7D4F"
OUTPUT_FILL = "#DDF3E4"
INK = "#263238"
MUTED = "#52636B"
GRID = "#D7E0E4"

DEFAULT_RUN_DIR = Path("solar/results/solar-10x150-preserved-15k-109965")
DEFAULT_OUTPUT_DIR = Path(
    "solar/results/solar-10x150-preserved-vs-scinet-presentation"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def disk_positions(
    center: tuple[float, float], count: int, radius: float, seed: int
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    angles = rng.uniform(0.0, 2.0 * np.pi, count)
    distances = radius * np.sqrt(rng.uniform(0.03, 0.93, count))
    return np.column_stack(
        (
            center[0] + distances * np.cos(angles),
            center[1] + distances * np.sin(angles),
        )
    )


def vertical_positions(
    x_value: float, center_y: float, count: int, span: float
) -> np.ndarray:
    return np.column_stack(
        (
            np.full(count, x_value),
            np.linspace(center_y + span / 2.0, center_y - span / 2.0, count),
        )
    )


def random_pairs(
    rng: np.random.Generator, source_count: int, target_count: int, count: int
) -> list[tuple[int, int]]:
    return list(
        zip(
            rng.integers(0, source_count, count).tolist(),
            rng.integers(0, target_count, count).tolist(),
            strict=True,
        )
    )


def draw_edges(
    axis: plt.Axes,
    starts: np.ndarray,
    ends: np.ndarray,
    pairs: Iterable[tuple[int, int]],
    *,
    color: str,
    alpha: float,
    linewidth: float = 0.5,
    zorder: int = 1,
) -> None:
    for source, target in pairs:
        axis.plot(
            [starts[source, 0], ends[target, 0]],
            [starts[source, 1], ends[target, 1]],
            color=color,
            alpha=alpha,
            linewidth=linewidth,
            zorder=zorder,
        )


def arrow(
    axis: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str,
    label: str | None = None,
    linewidth: float = 2.2,
    connectionstyle: str = "arc3",
    zorder: int = 7,
) -> None:
    axis.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=15,
            linewidth=linewidth,
            color=color,
            connectionstyle=connectionstyle,
            zorder=zorder,
        )
    )
    if label:
        axis.text(
            (start[0] + end[0]) / 2.0,
            (start[1] + end[1]) / 2.0 + 0.18,
            label,
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            color=color,
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "pad": 0.8,
                "alpha": 0.92,
            },
            zorder=zorder + 1,
        )


def rounded_box(
    axis: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    *,
    facecolor: str,
    edgecolor: str,
    linewidth: float = 1.7,
    radius: float = 0.14,
    zorder: int = 2,
) -> FancyBboxPatch:
    box = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle=f"round,pad=0.04,rounding_size={radius}",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        zorder=zorder,
    )
    axis.add_patch(box)
    return box


def draw_reservoir(
    axis: plt.Axes,
    center: tuple[float, float],
    *,
    node_count: int,
    radius: float,
    seed: int,
    label: str,
) -> np.ndarray:
    rng = np.random.default_rng(seed + 100)
    nodes = disk_positions(center, node_count, radius * 0.84, seed)
    axis.add_patch(
        Circle(
            center,
            radius,
            facecolor=FIXED_FILL,
            edgecolor=FIXED,
            linewidth=2.4,
            zorder=0,
        )
    )
    draw_edges(
        axis,
        nodes,
        nodes,
        random_pairs(rng, node_count, node_count, node_count * 3),
        color=FIXED,
        alpha=0.075,
        linewidth=0.45,
    )
    axis.scatter(
        nodes[:, 0],
        nodes[:, 1],
        s=17,
        facecolor="#D8ECF8",
        edgecolor=FIXED,
        linewidth=0.55,
        zorder=3,
    )
    arrow(
        axis,
        (center[0] - radius * 0.58, center[1] + radius * 1.03),
        (center[0] + radius * 0.58, center[1] + radius * 1.03),
        color=FIXED,
        linewidth=1.8,
        connectionstyle="arc3,rad=-0.55",
    )
    axis.text(
        center[0],
        center[1] + radius * 1.42,
        r"$A$ fixed recurrent dynamics",
        ha="center",
        va="center",
        fontsize=10.5,
        color=FIXED,
        fontweight="bold",
    )
    axis.text(
        center[0],
        center[1] - radius - 0.42,
        label,
        ha="center",
        va="top",
        fontsize=13,
        fontweight="bold",
        color=INK,
    )
    axis.text(
        center[0],
        center[1] - radius - 0.80,
        f"{node_count} recurrent neurons",
        ha="center",
        va="top",
        fontsize=10,
        color=MUTED,
    )
    return nodes


def draw_input(
    axis: plt.Axes, center: tuple[float, float], reservoir_nodes: np.ndarray
) -> None:
    input_position = np.asarray([center])
    input_sources = np.repeat(input_position, 32, axis=0)
    rng = np.random.default_rng(900)
    targets = rng.choice(len(reservoir_nodes), len(input_sources), replace=False)
    draw_edges(
        axis,
        input_sources,
        reservoir_nodes,
        [(index, int(target)) for index, target in enumerate(targets)],
        color=FIXED,
        alpha=0.18,
        linewidth=0.55,
    )
    axis.add_patch(
        Circle(
            center,
            0.42,
            facecolor="#ECEFF1",
            edgecolor=INK,
            linewidth=2.0,
            zorder=5,
        )
    )
    axis.text(
        center[0],
        center[1] + 0.08,
        "Earth view",
        ha="center",
        va="center",
        fontsize=8.4,
        fontweight="bold",
        color=INK,
        zorder=6,
    )
    axis.text(
        center[0],
        center[1] - 0.15,
        r"$(\theta_S,\theta_M)_{t_0}$",
        ha="center",
        va="center",
        fontsize=7.6,
        color=MUTED,
        zorder=6,
    )


def draw_latent_pair(
    axis: plt.Axes,
    center: tuple[float, float],
    *,
    color: str,
    fill: str,
    label: str,
) -> np.ndarray:
    nodes = vertical_positions(center[0], center[1], 2, 0.58)
    axis.scatter(
        nodes[:, 0],
        nodes[:, 1],
        s=205,
        facecolor=fill,
        edgecolor=color,
        linewidth=1.8,
        zorder=5,
    )
    for index, position in enumerate(nodes, start=1):
        axis.text(
            position[0],
            position[1],
            str(index),
            ha="center",
            va="center",
            fontsize=8,
            fontweight="bold",
            color=INK,
            zorder=6,
        )
    axis.text(
        center[0],
        center[1] - 0.86,
        label,
        ha="center",
        va="top",
        fontsize=12,
        fontweight="bold",
        color=INK,
    )
    axis.text(
        center[0],
        center[1] - 1.24,
        "two latent neurons",
        ha="center",
        va="top",
        fontsize=9.5,
        color=MUTED,
    )
    return nodes


def save(figure: plt.Figure, stem: Path, dpi: int) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(stem.with_suffix(".png"), dpi=dpi, facecolor="white")
    figure.savefig(stem.with_suffix(".pdf"), facecolor="white")
    plt.close(figure)


def draw_compact_preserved(output_stem: Path, dpi: int) -> None:
    figure, axis = plt.subplots(figsize=(16, 9))
    axis.set_xlim(0.0, 16.0)
    axis.set_ylim(0.0, 9.0)
    axis.set_aspect("equal")
    axis.axis("off")

    center_y = 4.75
    reservoir_center = (4.0, center_y)
    latent_center = (8.05, center_y)
    input_center = (0.85, center_y)
    output_center = (14.75, center_y)
    rng = np.random.default_rng(2026)

    reservoir_nodes = draw_reservoir(
        axis,
        reservoir_center,
        node_count=150,
        radius=1.65,
        seed=17,
        label=r"Representative reservoir state  $x_k(t)$",
    )
    draw_input(axis, input_center, reservoir_nodes)
    arrow(
        axis,
        (1.30, center_y),
        (2.30, center_y),
        color=FIXED,
        label=r"$B_1$",
    )

    latent_nodes = draw_latent_pair(
        axis,
        latent_center,
        color=PRESERVED,
        fill=PRESERVED_FILL,
        label=r"Explicit bottleneck  $z_k(t)\in\mathbb{R}^2$",
    )
    draw_edges(
        axis,
        reservoir_nodes,
        latent_nodes,
        random_pairs(rng, len(reservoir_nodes), 2, 62),
        color=TRAINABLE,
        alpha=0.17,
        linewidth=0.65,
    )
    arrow(
        axis,
        (5.70, center_y),
        (7.70, center_y),
        color=TRAINABLE,
        label=r"$W_k,b_k$",
    )

    rounded_box(
        axis,
        (9.75, 4.04),
        2.15,
        1.42,
        facecolor="#FFF3E7",
        edgecolor=TRAINABLE,
        linewidth=2.0,
    )
    axis.text(
        10.825,
        4.91,
        "Final trainable readout",
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
        color=INK,
    )
    axis.text(
        10.825,
        4.51,
        r"$W_{\rm out}x_{10}(t)+c$",
        ha="center",
        va="center",
        fontsize=11.5,
        color=TRAINABLE,
    )
    arrow(
        axis,
        (8.38, center_y),
        (9.68, center_y),
        color=FIXED,
        label=r"repeat to $x_{10}(t)$",
    )
    arrow(
        axis,
        (11.97, center_y),
        (14.26, center_y),
        color=TRAINABLE,
    )
    axis.add_patch(
        Circle(
            output_center,
            0.47,
            facecolor=OUTPUT_FILL,
            edgecolor=OUTPUT,
            linewidth=2.2,
            zorder=5,
        )
    )
    axis.text(
        output_center[0],
        output_center[1] + 0.06,
        "Forecast",
        ha="center",
        va="center",
        fontsize=9,
        fontweight="bold",
        color=INK,
        zorder=6,
    )
    axis.text(
        output_center[0],
        output_center[1] - 0.18,
        r"$(\hat\theta_S,\hat\theta_M)_t$",
        ha="center",
        va="center",
        fontsize=8.4,
        color=MUTED,
        zorder=6,
    )

    rounded_box(
        axis,
        (4.86, 6.30),
        1.38,
        0.78,
        facecolor=FIXED,
        edgecolor=FIXED,
        linewidth=0,
        radius=0.34,
        zorder=8,
    )
    axis.text(
        5.55,
        6.69,
        r"$\mathbf{\times 10}$",
        ha="center",
        va="center",
        fontsize=25,
        fontweight="bold",
        color="white",
        zorder=9,
    )
    axis.text(
        6.42,
        6.69,
        "fixed reservoirs",
        ha="left",
        va="center",
        fontsize=12,
        fontweight="bold",
        color=FIXED,
    )
    rounded_box(
        axis,
        (8.64, 5.83),
        1.05,
        0.64,
        facecolor=PRESERVED_FILL,
        edgecolor=PRESERVED,
        linewidth=1.6,
        radius=0.27,
        zorder=8,
    )
    axis.text(
        9.165,
        6.15,
        r"$\mathbf{\times 9}$",
        ha="center",
        va="center",
        fontsize=18,
        fontweight="bold",
        color=PRESERVED,
        zorder=9,
    )
    axis.text(
        9.87,
        6.15,
        "inter-reservoir bottlenecks",
        ha="left",
        va="center",
        fontsize=10.5,
        fontweight="bold",
        color=PRESERVED,
    )

    rounded_box(
        axis,
        (2.15, 0.53),
        11.70,
        1.13,
        facecolor="#F7F9FA",
        edgecolor=GRID,
        linewidth=1.2,
    )
    axis.text(
        8.0,
        1.28,
        r"Preserved anchor:  $z_k(t)=z_1(t)+0.1\,\tanh(W_kx_k(t)+b_k)$, "
        r"$k=2,\ldots,9$",
        ha="center",
        va="center",
        fontsize=11,
        color=INK,
    )
    axis.text(
        8.0,
        0.88,
        r"1,500 fixed recurrent neurons  •  18 latent neurons  •  "
        r"3,022 trainable parameters",
        ha="center",
        va="center",
        fontsize=10.5,
        fontweight="bold",
        color=MUTED,
    )

    axis.text(
        8.0,
        8.42,
        "Preserved 10×150 Reservoir Model",
        ha="center",
        va="center",
        fontsize=23,
        fontweight="bold",
        color=INK,
    )
    axis.text(
        8.0,
        8.00,
        "Compact presentation view — the repeated reservoir motif is drawn once",
        ha="center",
        va="center",
        fontsize=11.5,
        color=MUTED,
    )
    legend = [
        Line2D([0], [0], color=FIXED, lw=2.5, label="Fixed random weights"),
        Line2D([0], [0], color=TRAINABLE, lw=2.5, label="Trainable weights"),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=PRESERVED_FILL,
            markeredgecolor=PRESERVED,
            markersize=8,
            label=r"Preserved 2D latent",
        ),
    ]
    axis.legend(
        handles=legend,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=3,
        frameon=False,
        fontsize=9.5,
    )
    figure.tight_layout(pad=0.2)
    save(figure, output_stem, dpi)


def draw_hybrid_scinet_decoder(output_stem: Path, dpi: int) -> None:
    figure, axis = plt.subplots(figsize=(16, 9))
    axis.set_xlim(0.0, 16.0)
    axis.set_ylim(0.0, 9.0)
    axis.set_aspect("equal")
    axis.axis("off")

    center_y = 4.65
    input_center = (0.72, center_y)
    reservoir_center = (3.25, center_y)
    latent_center = (6.35, center_y)
    reservoir_nodes = draw_reservoir(
        axis,
        reservoir_center,
        node_count=150,
        radius=1.48,
        seed=31,
        label=r"Reservoir state  $x(t)$",
    )
    draw_input(axis, input_center, reservoir_nodes)
    arrow(
        axis,
        (1.16, center_y),
        (1.72, center_y),
        color=FIXED,
        label=r"$B$",
    )

    rng = np.random.default_rng(808)
    latent_nodes = draw_latent_pair(
        axis,
        latent_center,
        color=TRAINABLE,
        fill=TRAINABLE_FILL,
        label=r"Learned latent  $z(t)\in\mathbb{R}^2$",
    )
    draw_edges(
        axis,
        reservoir_nodes,
        latent_nodes,
        random_pairs(rng, len(reservoir_nodes), 2, 68),
        color=TRAINABLE,
        alpha=0.19,
        linewidth=0.65,
    )
    arrow(
        axis,
        (4.78, center_y),
        (6.02, center_y),
        color=TRAINABLE,
        label=r"$W_z,b_z$",
    )

    decoder_left = 7.25
    decoder_bottom = 2.46
    decoder_width = 7.72
    decoder_height = 4.55
    rounded_box(
        axis,
        (decoder_left, decoder_bottom),
        decoder_width,
        decoder_height,
        facecolor="#FBF9FE",
        edgecolor=SCINET,
        linewidth=2.2,
        radius=0.22,
        zorder=0,
    )
    axis.text(
        decoder_left + decoder_width / 2.0,
        decoder_bottom + decoder_height - 0.42,
        "Reproduced SciNet decoder",
        ha="center",
        va="center",
        fontsize=16,
        fontweight="bold",
        color=SCINET,
    )
    axis.text(
        decoder_left + decoder_width / 2.0,
        decoder_bottom + decoder_height - 0.78,
        "shared at every forecast time",
        ha="center",
        va="center",
        fontsize=10,
        color=MUTED,
    )

    hidden_1 = vertical_positions(9.05, center_y, 9, 2.66)
    hidden_2 = vertical_positions(11.55, center_y, 9, 2.66)
    decoder_output = vertical_positions(13.72, center_y, 2, 0.64)

    draw_edges(
        axis,
        latent_nodes,
        hidden_1,
        [(source, target) for source in range(2) for target in range(9)],
        color=SCINET,
        alpha=0.23,
        linewidth=0.65,
    )
    draw_edges(
        axis,
        hidden_1,
        hidden_2,
        random_pairs(rng, 9, 9, 42),
        color=SCINET,
        alpha=0.14,
        linewidth=0.55,
    )
    draw_edges(
        axis,
        hidden_2,
        decoder_output,
        [(source, target) for source in range(9) for target in range(2)],
        color=SCINET,
        alpha=0.23,
        linewidth=0.65,
    )
    for nodes in (hidden_1, hidden_2):
        axis.scatter(
            nodes[:, 0],
            nodes[:, 1],
            s=92,
            facecolor=SCINET_FILL,
            edgecolor=SCINET,
            linewidth=1.2,
            zorder=4,
        )
    axis.scatter(
        decoder_output[:, 0],
        decoder_output[:, 1],
        s=205,
        facecolor=OUTPUT_FILL,
        edgecolor=OUTPUT,
        linewidth=1.8,
        zorder=5,
    )
    axis.text(
        9.05,
        2.86,
        "Dense 100\nELU",
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
        color=SCINET,
    )
    axis.text(
        11.55,
        2.86,
        "Dense 100\nELU",
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
        color=SCINET,
    )
    axis.text(
        13.72,
        3.49,
        "Dense 2",
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
        color=OUTPUT,
    )
    axis.text(
        13.72,
        3.08,
        r"$(\hat\theta_S,\hat\theta_M)_t$",
        ha="center",
        va="center",
        fontsize=10,
        color=INK,
    )
    arrow(
        axis,
        (6.66, center_y),
        (8.52, center_y),
        color=SCINET,
        label="decoder input",
        linewidth=2.4,
    )
    axis.text(
        8.0,
        8.43,
        "Hybrid Reservoir–SciNet Decoder",
        ha="center",
        va="center",
        fontsize=23,
        fontweight="bold",
        color=INK,
    )
    axis.text(
        8.0,
        8.02,
        "One fixed 150-neuron reservoir learns a 2D latent representation, "
        "then the SciNet decoder predicts future Earth-view angles",
        ha="center",
        va="center",
        fontsize=11.2,
        color=MUTED,
    )
    legend = [
        Line2D([0], [0], color=FIXED, lw=2.5, label="Fixed reservoir weights"),
        Line2D(
            [0], [0], color=TRAINABLE, lw=2.5, label="Trainable latent readout"
        ),
        Line2D([0], [0], color=SCINET, lw=2.5, label="Trainable SciNet decoder"),
    ]
    axis.legend(
        handles=legend,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.035),
        ncol=3,
        frameon=False,
        fontsize=9.4,
    )
    axis.text(
        8.0,
        0.33,
        r"Exact decoder topology from the reproduction:  "
        r"$2 \rightarrow 100\,{\rm ELU} \rightarrow 100\,{\rm ELU} "
        r"\rightarrow 2$",
        ha="center",
        va="center",
        fontsize=10.5,
        fontweight="bold",
        color=INK,
    )
    figure.tight_layout(pad=0.2)
    save(figure, output_stem, dpi)


def draw_two_reservoir(output_stem: Path, dpi: int) -> None:
    figure, axis = plt.subplots(figsize=(16, 9))
    axis.set_xlim(0.0, 16.0)
    axis.set_ylim(0.0, 9.0)
    axis.set_aspect("equal")
    axis.axis("off")

    center_y = 4.62
    input_center = (0.62, center_y)
    first_center = (3.15, center_y)
    latent_center = (7.30, center_y)
    second_center = (10.55, center_y)
    output_center = (14.72, center_y)
    rng = np.random.default_rng(2150)

    first_nodes = draw_reservoir(
        axis,
        first_center,
        node_count=150,
        radius=1.47,
        seed=41,
        label=r"Reservoir 1 state  $x_1(t)$",
    )
    second_nodes = draw_reservoir(
        axis,
        second_center,
        node_count=150,
        radius=1.47,
        seed=42,
        label=r"Reservoir 2 state  $x_2(t)$",
    )
    draw_input(axis, input_center, first_nodes)
    arrow(
        axis,
        (1.06, center_y),
        (1.62, center_y),
        color=FIXED,
        label=r"$B_1$",
    )

    latent_nodes = draw_latent_pair(
        axis,
        latent_center,
        color=TRAINABLE,
        fill=TRAINABLE_FILL,
        label=r"2D latent bottleneck  $z(t)$",
    )
    draw_edges(
        axis,
        first_nodes,
        latent_nodes,
        random_pairs(rng, 150, 2, 70),
        color=TRAINABLE,
        alpha=0.19,
        linewidth=0.65,
    )
    draw_edges(
        axis,
        latent_nodes,
        second_nodes,
        random_pairs(rng, 2, 150, 70),
        color=FIXED,
        alpha=0.17,
        linewidth=0.65,
    )
    arrow(
        axis,
        (4.67, center_y),
        (6.95, center_y),
        color=TRAINABLE,
        label=r"$W_1,b_1$",
    )
    arrow(
        axis,
        (7.64, center_y),
        (9.03, center_y),
        color=FIXED,
        label=r"$R_1$ fixed",
    )

    output_position = np.asarray([output_center])
    output_sources = rng.choice(150, 38, replace=False)
    output_targets = np.repeat(output_position, len(output_sources), axis=0)
    draw_edges(
        axis,
        second_nodes,
        output_targets,
        [(int(source), index) for index, source in enumerate(output_sources)],
        color=TRAINABLE,
        alpha=0.19,
        linewidth=0.65,
    )
    arrow(
        axis,
        (12.07, center_y),
        (14.22, center_y),
        color=TRAINABLE,
        label=r"$W_{\rm out},c$",
    )
    axis.add_patch(
        Circle(
            output_center,
            0.48,
            facecolor=OUTPUT_FILL,
            edgecolor=OUTPUT,
            linewidth=2.2,
            zorder=5,
        )
    )
    axis.text(
        output_center[0],
        output_center[1] + 0.08,
        "Forecast",
        ha="center",
        va="center",
        fontsize=9,
        fontweight="bold",
        color=INK,
        zorder=6,
    )
    axis.text(
        output_center[0],
        output_center[1] - 0.18,
        r"$(\hat\theta_S,\hat\theta_M)_t$",
        ha="center",
        va="center",
        fontsize=8.3,
        color=MUTED,
        zorder=6,
    )

    axis.text(
        8.0,
        8.43,
        "2×150 Reservoir Model",
        ha="center",
        va="center",
        fontsize=24,
        fontweight="bold",
        color=INK,
    )
    axis.text(
        8.0,
        8.02,
        "Two fixed 150-neuron reservoirs with one explicit vertical 2D bottleneck",
        ha="center",
        va="center",
        fontsize=11.5,
        color=MUTED,
    )

    rounded_box(
        axis,
        (3.05, 0.48),
        9.90,
        0.88,
        facecolor="#F7F9FA",
        edgecolor=GRID,
        linewidth=1.2,
    )
    axis.text(
        8.0,
        0.92,
        "300 fixed recurrent neurons  •  2 latent neurons  •  "
        "606 trainable parameters",
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
        color=MUTED,
    )
    legend = [
        Line2D([0], [0], color=FIXED, lw=2.5, label="Fixed random weights"),
        Line2D([0], [0], color=TRAINABLE, lw=2.5, label="Trainable weights"),
    ]
    axis.legend(
        handles=legend,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=2,
        frameon=False,
        fontsize=9.5,
    )
    figure.tight_layout(pad=0.2)
    save(figure, output_stem, dpi)


def validate_run(run_dir: Path) -> None:
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    expected = {
        "reservoir_layers": 10,
        "nodes_1": 150,
        "latent_size": 2,
        "preserve_primary_latent": True,
        "scinet_hidden_size": 100,
    }
    mismatches = {
        key: (config.get(key), expected_value)
        for key, expected_value in expected.items()
        if config.get(key) != expected_value
    }
    if mismatches:
        raise ValueError(f"unexpected experiment configuration: {mismatches}")


def main() -> None:
    args = parse_args()
    validate_run(args.run_dir)
    draw_compact_preserved(
        args.output_dir / "compact_preserved_model_architecture", args.dpi
    )
    draw_hybrid_scinet_decoder(
        args.output_dir / "reservoir_scinet_decoder_architecture", args.dpi
    )
    draw_two_reservoir(
        args.output_dir / "two_reservoir_2x150_architecture", args.dpi
    )
    print(f"Saved compact architecture diagrams to {args.output_dir}")


if __name__ == "__main__":
    main()
