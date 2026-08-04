#!/usr/bin/env python3
"""Draw the exact preserved 10-by-150 solar-reservoir architecture."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "bp_reservoir_mpl")
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch
import numpy as np


FIXED = "#0072B2"
FIXED_FILL = "#E6F2F8"
TRAINABLE = "#D55E00"
TRAINABLE_FILL = "#FDE9DD"
PRESERVED = "#C69214"
PRESERVED_FILL = "#FFF3C4"
OUTPUT_FILL = "#E4F3E8"
INK = "#263238"
MUTED = "#52636B"
GRID = "#D8E0E3"

DEFAULT_RUN_DIR = Path(
    "solar/results/solar-10x150-preserved-15k-109965"
)
DEFAULT_OUTPUT_DIR = Path(
    "solar/results/solar-10x150-preserved-vs-scinet-presentation"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def _rounded_box(
    axis: Any,
    center: tuple[float, float],
    width: float,
    height: float,
    *,
    facecolor: str,
    edgecolor: str,
    linewidth: float = 1.7,
    radius: float = 0.12,
    zorder: int = 3,
) -> FancyBboxPatch:
    patch = FancyBboxPatch(
        (center[0] - width / 2.0, center[1] - height / 2.0),
        width,
        height,
        boxstyle=f"round,pad=0.03,rounding_size={radius}",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        zorder=zorder,
    )
    axis.add_patch(patch)
    return patch


def _arrow(
    axis: Any,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str,
    linewidth: float = 1.8,
    mutation_scale: float = 12.0,
    connectionstyle: str = "arc3",
    zorder: int = 2,
) -> FancyArrowPatch:
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        color=color,
        linewidth=linewidth,
        mutation_scale=mutation_scale,
        connectionstyle=connectionstyle,
        shrinkA=0,
        shrinkB=0,
        zorder=zorder,
    )
    axis.add_patch(patch)
    return patch


def _validate_config(config: dict[str, Any], metrics: dict[str, Any]) -> None:
    expected = {
        "reservoir_layers": 10,
        "nodes_1": 150,
        "nodes_2": 150,
        "latent_size": 2,
        "preserve_primary_latent": True,
    }
    mismatches = {
        key: (config.get(key), value)
        for key, value in expected.items()
        if config.get(key) != value
    }
    if mismatches:
        raise ValueError(f"run does not match the preserved architecture: {mismatches}")
    if metrics.get("trainable_parameters") != 3022:
        raise ValueError(
            "the checkpoint does not have the expected 3,022 trainable parameters"
        )


def _save(figure: plt.Figure, output_stem: Path) -> None:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_stem.with_suffix(".png"), dpi=240)
    figure.savefig(output_stem.with_suffix(".pdf"))
    plt.close(figure)


def draw_architecture(
    config: dict[str, Any],
    metrics: dict[str, Any],
    output_stem: Path,
) -> None:
    _validate_config(config, metrics)

    figure, axis = plt.subplots(figsize=(13.33, 7.5))
    axis.set_xlim(0.0, 16.0)
    axis.set_ylim(0.0, 9.0)
    axis.axis("off")

    main_y = 5.0
    reservoir_x = np.linspace(1.45, 14.20, config["reservoir_layers"])
    latent_x = (reservoir_x[:-1] + reservoir_x[1:]) / 2.0
    input_center = (0.48, main_y)
    output_center = (15.40, main_y)
    reservoir_width = 0.78
    reservoir_height = 1.08
    latent_radius = 0.245

    # Main left-to-right computation path.
    _rounded_box(
        axis,
        input_center,
        0.98,
        1.18,
        facecolor="#F2F4F5",
        edgecolor=INK,
        linewidth=1.7,
    )
    axis.text(
        input_center[0],
        main_y + 0.13,
        "Input\nEarth view",
        ha="center",
        va="center",
        fontsize=8.5,
        fontweight="bold",
        linespacing=1.1,
        color=INK,
        zorder=5,
    )
    axis.text(
        input_center[0],
        main_y - 0.32,
        r"$(\theta_S,\theta_M)_{t_0}$",
        ha="center",
        va="center",
        fontsize=9,
        color=MUTED,
        zorder=5,
    )

    _rounded_box(
        axis,
        output_center,
        1.08,
        1.18,
        facecolor=OUTPUT_FILL,
        edgecolor="#2E7D4F",
        linewidth=1.7,
    )
    axis.text(
        output_center[0],
        main_y + 0.16,
        "Output\nEarth view",
        ha="center",
        va="center",
        fontsize=8.3,
        fontweight="bold",
        linespacing=1.1,
        color=INK,
        zorder=5,
    )
    axis.text(
        output_center[0],
        main_y - 0.31,
        r"$(\hat\theta_S,\hat\theta_M)_t$",
        ha="center",
        va="center",
        fontsize=8.8,
        color=MUTED,
        zorder=5,
    )

    for index, x_value in enumerate(reservoir_x, start=1):
        _rounded_box(
            axis,
            (float(x_value), main_y),
            reservoir_width,
            reservoir_height,
            facecolor=FIXED_FILL,
            edgecolor=FIXED,
        )
        axis.text(
            x_value,
            main_y + 0.15,
            rf"$\mathcal{{R}}_{{{index}}}$",
            ha="center",
            va="center",
            fontsize=11.5,
            fontweight="bold",
            color=FIXED,
            zorder=5,
        )
        axis.text(
            x_value,
            main_y - 0.24,
            "150 fixed\nneurons",
            ha="center",
            va="center",
            fontsize=6.7,
            linespacing=1.05,
            color=MUTED,
            zorder=5,
        )

    for index, x_value in enumerate(latent_x, start=1):
        facecolor = PRESERVED_FILL if index == 1 else TRAINABLE_FILL
        edgecolor = PRESERVED if index == 1 else TRAINABLE
        axis.add_patch(
            Circle(
                (float(x_value), main_y),
                latent_radius,
                facecolor=facecolor,
                edgecolor=edgecolor,
                linewidth=1.8,
                zorder=4,
            )
        )
        axis.text(
            x_value,
            main_y + 0.035,
            rf"$z_{index}$",
            ha="center",
            va="center",
            fontsize=8.7,
            fontweight="bold",
            color=edgecolor,
            zorder=5,
        )
        axis.text(
            x_value,
            main_y - 0.14,
            "2D",
            ha="center",
            va="center",
            fontsize=5.8,
            color=MUTED,
            zorder=5,
        )

    # Fixed projections into reservoirs and trainable low-dimensional readouts.
    _arrow(
        axis,
        (input_center[0] + 0.52, main_y),
        (reservoir_x[0] - reservoir_width / 2.0, main_y),
        color=FIXED,
    )
    for index, x_value in enumerate(latent_x):
        _arrow(
            axis,
            (reservoir_x[index] + reservoir_width / 2.0, main_y),
            (x_value - latent_radius, main_y),
            color=TRAINABLE,
        )
        _arrow(
            axis,
            (x_value + latent_radius, main_y),
            (reservoir_x[index + 1] - reservoir_width / 2.0, main_y),
            color=FIXED,
        )
    _arrow(
        axis,
        (reservoir_x[-1] + reservoir_width / 2.0, main_y),
        (output_center[0] - 0.58, main_y),
        color=TRAINABLE,
    )

    # The exact preservation mechanism in this checkpoint: every intermediate
    # bottleneck is a bounded residual around the primary latent.
    bus_y = 3.31
    axis.plot(
        [latent_x[0], latent_x[-1]],
        [bus_y, bus_y],
        color=PRESERVED,
        linewidth=2.2,
        zorder=1,
    )
    axis.plot(
        [latent_x[0], latent_x[0]],
        [main_y - latent_radius, bus_y],
        color=PRESERVED,
        linewidth=2.2,
        zorder=1,
    )
    for x_value in latent_x[1:]:
        _arrow(
            axis,
            (float(x_value), bus_y),
            (float(x_value), main_y - latent_radius - 0.03),
            color=PRESERVED,
            linewidth=1.5,
            mutation_scale=9,
            zorder=1,
        )
    axis.text(
        float(np.mean(latent_x)),
        bus_y - 0.29,
        "Preserved primary-latent anchor",
        ha="center",
        va="center",
        fontsize=9.3,
        fontweight="bold",
        color=PRESERVED,
    )
    axis.text(
        float(np.mean(latent_x)),
        bus_y - 0.64,
        r"$z_k(t)=z_1(t)+0.1\,\tanh(W_kx_k(t)+b_k),"
        r"\quad k=2,\ldots,9$",
        ha="center",
        va="center",
        fontsize=9.1,
        color=INK,
    )

    # Primary time evolution and stateful downstream update schedule.
    primary_box_center = (float(latent_x[0]), 6.56)
    _rounded_box(
        axis,
        primary_box_center,
        2.32,
        0.66,
        facecolor=PRESERVED_FILL,
        edgecolor=PRESERVED,
        linewidth=1.5,
    )
    axis.text(
        primary_box_center[0],
        primary_box_center[1] + 0.10,
        r"$z_1(t+1)=z_1(t)+\Delta z$",
        ha="center",
        va="center",
        fontsize=10.2,
        fontweight="bold",
        color=INK,
    )
    delta = metrics["learned_latent_delta"]
    axis.text(
        primary_box_center[0],
        primary_box_center[1] - 0.17,
        rf"learned $\Delta z=({delta[0]:.4f},{delta[1]:.4f})$",
        ha="center",
        va="center",
        fontsize=7.4,
        color=MUTED,
    )
    _arrow(
        axis,
        (primary_box_center[0], primary_box_center[1] - 0.36),
        (latent_x[0], main_y + latent_radius + 0.02),
        color=PRESERVED,
        linewidth=1.6,
        mutation_scale=10,
    )
    axis.text(
        reservoir_x[0],
        5.78,
        "3 encoder\nupdates",
        ha="center",
        va="bottom",
        fontsize=6.8,
        color=MUTED,
        linespacing=1.1,
    )
    schedule_start = float(reservoir_x[1] - reservoir_width / 2.0)
    schedule_end = float(reservoir_x[-1] + reservoir_width / 2.0)
    schedule_y = 6.47
    axis.plot(
        [schedule_start, schedule_end],
        [schedule_y, schedule_y],
        color=FIXED,
        linewidth=1.4,
    )
    axis.plot(
        [schedule_start, schedule_start],
        [schedule_y, schedule_y - 0.16],
        color=FIXED,
        linewidth=1.4,
    )
    axis.plot(
        [schedule_end, schedule_end],
        [schedule_y, schedule_y - 0.16],
        color=FIXED,
        linewidth=1.4,
    )
    axis.text(
        (schedule_start + schedule_end) / 2.0,
        schedule_y + 0.13,
        "Reservoirs 2–10 retain state across the trajectory"
        "  •  20 updates at $t_0$  •  3 updates per later week",
        ha="center",
        va="bottom",
        fontsize=8.2,
        color=FIXED,
    )

    # Header and compact implementation/training notes.
    axis.text(
        8.0,
        8.48,
        "Preserved 10×150 solar reservoir",
        ha="center",
        va="center",
        fontsize=21,
        fontweight="bold",
        color=INK,
    )
    axis.text(
        8.0,
        8.05,
        "Ten fixed recurrent reservoirs  •  nine explicit 2D bottlenecks"
        "  •  constant primary-latent dynamics",
        ha="center",
        va="center",
        fontsize=10.8,
        color=MUTED,
    )

    footer = FancyBboxPatch(
        (0.35, 0.34),
        15.30,
        1.28,
        boxstyle="round,pad=0.04,rounding_size=0.12",
        facecolor="#F7F9FA",
        edgecolor=GRID,
        linewidth=1.2,
        zorder=0,
    )
    axis.add_patch(footer)
    legend = [
        Line2D(
            [0],
            [0],
            color=FIXED,
            lw=2.4,
            label=r"Fixed random weights $(A_k,B_1,R_k)$",
        ),
        Line2D(
            [0],
            [0],
            color=TRAINABLE,
            lw=2.4,
            label=r"Trainable weights $(W_k,b_k,\Delta z,W_{\rm out},c)$",
        ),
        Line2D([0], [0], color=PRESERVED, lw=2.4, label="Preserved $z_1$ path"),
    ]
    axis.legend(
        handles=legend,
        loc="lower left",
        bbox_to_anchor=(0.035, 0.035),
        frameon=False,
        ncol=3,
        fontsize=8.6,
        handlelength=2.3,
        columnspacing=1.7,
    )
    axis.text(
        8.04,
        1.18,
        "Fixed dynamics: spectral radius 0.9, density 0.1, leak rate 1.0"
        "  •  input scale 0.5, interlayer scale 2.0",
        ha="center",
        va="center",
        fontsize=8.0,
        color=MUTED,
    )
    axis.text(
        8.04,
        0.84,
        r"Trainable: 9 latent readouts + $\Delta z$ + final readout"
        r"  =  3,022 parameters"
        r"  •  50-week shared computation",
        ha="center",
        va="center",
        fontsize=8.4,
        fontweight="bold",
        color=INK,
    )

    _save(figure, output_stem)


def main() -> None:
    args = parse_args()
    config = json.loads((args.run_dir / "config.json").read_text(encoding="utf-8"))
    metrics = json.loads(
        (args.run_dir / "reservoir/seed_0/metrics.json").read_text(encoding="utf-8")
    )
    _validate_config(config, metrics)

    repository_root = Path(__file__).resolve().parents[2]
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))
    from narma.plot_architecture import draw_architecture as draw_neuron_architecture

    output_stem = args.output_dir / "preserved_model_architecture"
    common = {
        "reservoirs": config["reservoir_layers"],
        "total_nodes": config["reservoir_layers"] * config["nodes_1"],
        "latent_size": config["latent_size"],
        "solar_preserved": True,
    }
    draw_neuron_architecture(output_stem.with_suffix(".png"), dpi=180, **common)
    draw_neuron_architecture(output_stem.with_suffix(".pdf"), **common)


if __name__ == "__main__":
    main()
