#!/usr/bin/env python3
"""Create a simple presentation graphic for SciNet's solar-system example."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "bp_reservoir_mpl")
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Arc, Circle, FancyArrowPatch, FancyBboxPatch, Polygon


INK = "#24323A"
MUTED = "#5B6B73"
BLUE = "#2878B5"
BLUE_LIGHT = "#EDF6FC"
ORANGE = "#E07A1F"
ORANGE_LIGHT = "#FFF3E5"
PURPLE = "#7651B5"
PURPLE_LIGHT = "#F4F0FC"
GOLD = "#E5A91A"
MARS = "#C9573A"
EARTH = "#2E83C2"
GREEN = "#2E7D4F"
GRID = "#D8E1E5"
BACKGROUND = "#FBFCFD"

DEFAULT_OUTPUT_DIR = Path(
    "solar/results/solar-10x150-preserved-vs-scinet-presentation"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def rounded_panel(
    axis: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    *,
    facecolor: str,
    edgecolor: str,
) -> None:
    axis.add_patch(
        FancyBboxPatch(
            xy,
            width,
            height,
            boxstyle="round,pad=0.04,rounding_size=0.20",
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=1.8,
            zorder=0,
        )
    )


def arrow(
    axis: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str,
    linewidth: float = 2.0,
    connectionstyle: str = "arc3",
    zorder: int = 5,
) -> None:
    axis.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=14,
            color=color,
            linewidth=linewidth,
            connectionstyle=connectionstyle,
            zorder=zorder,
        )
    )


def star_points(
    center: tuple[float, float],
    outer_radius: float,
    inner_radius: float,
    points: int = 5,
) -> np.ndarray:
    angles = np.linspace(np.pi / 2.0, np.pi / 2.0 + 2.0 * np.pi, points * 2 + 1)
    radii = np.resize(np.asarray([outer_radius, inner_radius]), points * 2)
    coordinates = np.column_stack(
        (
            center[0] + radii * np.cos(angles[:-1]),
            center[1] + radii * np.sin(angles[:-1]),
        )
    )
    return coordinates


def draw_sun(axis: plt.Axes, center: tuple[float, float], radius: float) -> None:
    for angle in np.linspace(0.0, 2.0 * np.pi, 12, endpoint=False):
        start = (
            center[0] + radius * 1.23 * np.cos(angle),
            center[1] + radius * 1.23 * np.sin(angle),
        )
        end = (
            center[0] + radius * 1.55 * np.cos(angle),
            center[1] + radius * 1.55 * np.sin(angle),
        )
        axis.plot(
            [start[0], end[0]],
            [start[1], end[1]],
            color=GOLD,
            linewidth=1.5,
            zorder=3,
        )
    axis.add_patch(
        Circle(
            center,
            radius,
            facecolor="#FFD75E",
            edgecolor="#C78A00",
            linewidth=1.8,
            zorder=4,
        )
    )


def draw_observation_panel(axis: plt.Axes) -> None:
    panel_x, panel_y, panel_w, panel_h = 0.35, 1.35, 4.72, 6.20
    rounded_panel(
        axis,
        (panel_x, panel_y),
        panel_w,
        panel_h,
        facecolor=BLUE_LIGHT,
        edgecolor=BLUE,
    )
    axis.text(
        panel_x + panel_w / 2.0,
        7.12,
        "1. What is observed?",
        ha="center",
        va="center",
        fontsize=15.5,
        fontweight="bold",
        color=BLUE,
    )
    axis.text(
        panel_x + panel_w / 2.0,
        6.72,
        "Only the sky as seen from Earth",
        ha="center",
        va="center",
        fontsize=10.5,
        color=MUTED,
    )

    axis.text(
        0.72,
        5.82,
        "Sun",
        ha="left",
        va="center",
        fontsize=11,
        fontweight="bold",
        color="#A56F00",
    )
    sun_positions = [(1.48, 5.82), (2.38, 5.82), (3.28, 5.82), (4.18, 5.82)]
    for index, position in enumerate(sun_positions):
        axis.add_patch(
            Circle(
                position,
                0.14,
                facecolor="#FFD75E",
                edgecolor="#C78A00",
                linewidth=1.4,
                zorder=4,
            )
        )
        axis.text(
            position[0],
            5.50,
            rf"$t_{index}$",
            ha="center",
            va="center",
            fontsize=8.2,
            color=MUTED,
        )
        if index:
            arrow(
                axis,
                (sun_positions[index - 1][0] + 0.18, 5.82),
                (position[0] - 0.18, 5.82),
                color="#C78A00",
                linewidth=1.9,
            )
    axis.text(
        2.83,
        6.18,
        "steady motion in one direction",
        ha="center",
        va="center",
        fontsize=8.8,
        fontweight="bold",
        color="#A56F00",
    )

    axis.text(
        0.72,
        4.62,
        "Mars",
        ha="left",
        va="center",
        fontsize=11,
        fontweight="bold",
        color=MARS,
    )
    mars_forward = [(1.48, 4.62), (2.38, 4.62), (3.28, 4.62)]
    for index, position in enumerate(mars_forward):
        axis.add_patch(
            Circle(
                position,
                0.14,
                facecolor="#DF7459",
                edgecolor="#9D3527",
                linewidth=1.4,
                zorder=4,
            )
        )
        axis.text(
            position[0],
            4.31,
            rf"$t_{index}$",
            ha="center",
            va="center",
            fontsize=8.2,
            color=MUTED,
        )
        if index:
            arrow(
                axis,
                (mars_forward[index - 1][0] + 0.18, 4.62),
                (position[0] - 0.18, 4.62),
                color=MARS,
                linewidth=1.9,
            )
    mars_reversed = (2.48, 3.82)
    arrow(
        axis,
        (3.34, 4.47),
        (mars_reversed[0] + 0.15, mars_reversed[1] + 0.10),
        color=MARS,
        linewidth=2.2,
        connectionstyle="arc3,rad=-0.20",
    )
    axis.add_patch(
        Circle(
            mars_reversed,
            0.14,
            facecolor="#DF7459",
            edgecolor="#9D3527",
            linewidth=1.4,
            zorder=4,
        )
    )
    axis.text(
        mars_reversed[0],
        3.51,
        r"$t_3$",
        ha="center",
        va="center",
        fontsize=8.2,
        color=MUTED,
    )
    axis.text(
        3.69,
        4.06,
        "apparent\nreversal",
        ha="center",
        va="center",
        fontsize=9.2,
        fontweight="bold",
        color=MARS,
        linespacing=1.1,
    )

    earth = (1.14, 2.48)
    axis.add_patch(
        Circle(
            earth,
            0.24,
            facecolor="#62A9DA",
            edgecolor="#176A9F",
            linewidth=1.6,
            zorder=4,
        )
    )
    axis.text(
        earth[0],
        earth[1] - 0.42,
        "Earth",
        ha="center",
        va="top",
        fontsize=9.5,
        fontweight="bold",
        color=EARTH,
    )
    rounded_panel(
        axis,
        (1.63, 2.00),
        2.95,
        0.96,
        facecolor="white",
        edgecolor=BLUE,
    )
    arrow(axis, (1.40, 2.48), (1.59, 2.48), color=BLUE, linewidth=1.8)
    axis.text(
        3.105,
        2.65,
        "Earth records two sky angles",
        ha="center",
        va="center",
        fontsize=9.5,
        fontweight="bold",
        color=BLUE,
    )
    axis.text(
        3.105,
        2.30,
        r"input at $t_0$:  $(\theta_S,\theta_M)$",
        ha="center",
        va="center",
        fontsize=10.2,
        color=INK,
    )
    axis.text(
        panel_x + panel_w / 2.0,
        1.64,
        "The observed motions look very different.",
        ha="center",
        va="center",
        fontsize=9.5,
        color=MUTED,
    )


def draw_learning_panel(axis: plt.Axes) -> None:
    panel_x, panel_y, panel_w, panel_h = 5.36, 1.35, 5.28, 6.20
    rounded_panel(
        axis,
        (panel_x, panel_y),
        panel_w,
        panel_h,
        facecolor=ORANGE_LIGHT,
        edgecolor=ORANGE,
    )
    axis.text(
        panel_x + panel_w / 2.0,
        7.12,
        "2. What must the model do?",
        ha="center",
        va="center",
        fontsize=15.5,
        fontweight="bold",
        color=ORANGE,
    )
    axis.text(
        panel_x + panel_w / 2.0,
        6.72,
        "Compress, evolve, and predict",
        ha="center",
        va="center",
        fontsize=10.5,
        color=MUTED,
    )

    rounded_panel(
        axis,
        (5.73, 4.68),
        1.30,
        1.10,
        facecolor="white",
        edgecolor=BLUE,
    )
    axis.text(
        6.38,
        5.40,
        "Earth-view\ninput",
        ha="center",
        va="center",
        fontsize=9.5,
        fontweight="bold",
        color=BLUE,
    )
    axis.text(
        6.38,
        4.91,
        r"$\theta_S,\theta_M$",
        ha="center",
        va="center",
        fontsize=10.5,
        color=INK,
    )

    latent_positions = [(8.05, 5.48), (8.05, 4.98)]
    for index, position in enumerate(latent_positions, start=1):
        axis.add_patch(
            Circle(
                position,
                0.19,
                facecolor="#FFD5A2",
                edgecolor=ORANGE,
                linewidth=1.8,
                zorder=4,
            )
        )
        axis.text(
            position[0],
            position[1],
            str(index),
            ha="center",
            va="center",
            fontsize=8,
            fontweight="bold",
            color=INK,
            zorder=5,
        )
    axis.text(
        8.05,
        5.94,
        "2-number latent",
        ha="center",
        va="bottom",
        fontsize=10.5,
        fontweight="bold",
        color=ORANGE,
    )

    rounded_panel(
        axis,
        (9.05, 4.68),
        1.24,
        1.10,
        facecolor="white",
        edgecolor=GREEN,
    )
    axis.text(
        9.67,
        5.40,
        "Future\nangles",
        ha="center",
        va="center",
        fontsize=9.5,
        fontweight="bold",
        color=GREEN,
    )
    axis.text(
        9.67,
        4.91,
        r"$\hat\theta_S,\hat\theta_M$",
        ha="center",
        va="center",
        fontsize=10.2,
        color=INK,
    )
    arrow(axis, (7.07, 5.23), (7.74, 5.23), color=ORANGE)
    arrow(axis, (8.28, 5.23), (9.00, 5.23), color=GREEN)

    rounded_panel(
        axis,
        (6.17, 3.25),
        3.65,
        0.75,
        facecolor="#FFF8ED",
        edgecolor="#C58B13",
    )
    axis.text(
        7.995,
        3.625,
        r"Same simple update each week:  $z \leftarrow z+\Delta z$",
        ha="center",
        va="center",
        fontsize=10.2,
        fontweight="bold",
        color="#A56F00",
    )
    arrow(
        axis,
        (8.35, 4.80),
        (8.35, 4.03),
        color="#C58B13",
        linewidth=1.6,
    )

    axis.text(
        8.0,
        2.65,
        r"$t_0\quad\longrightarrow\quad t_1\quad\longrightarrow\quad"
        r" t_2\quad\longrightarrow\quad\cdots\quad t_{49}$",
        ha="center",
        va="center",
        fontsize=11,
        color=INK,
    )
    axis.text(
        8.0,
        2.10,
        "Goal: predict both Earth-view angles\nat every weekly time step.",
        ha="center",
        va="center",
        fontsize=10,
        color=MUTED,
        linespacing=1.25,
    )


def orbit_point(
    center: tuple[float, float], radius: float, angle_degrees: float
) -> tuple[float, float]:
    angle = np.deg2rad(angle_degrees)
    return (
        center[0] + radius * np.cos(angle),
        center[1] + radius * np.sin(angle),
    )


def draw_solution_panel(axis: plt.Axes) -> None:
    panel_x, panel_y, panel_w, panel_h = 10.93, 1.35, 4.72, 6.20
    rounded_panel(
        axis,
        (panel_x, panel_y),
        panel_w,
        panel_h,
        facecolor=PURPLE_LIGHT,
        edgecolor=PURPLE,
    )
    axis.text(
        panel_x + panel_w / 2.0,
        7.12,
        "3. What simple state emerges?",
        ha="center",
        va="center",
        fontsize=15.0,
        fontweight="bold",
        color=PURPLE,
    )
    axis.text(
        panel_x + panel_w / 2.0,
        6.72,
        "Sun-centered coordinates",
        ha="center",
        va="center",
        fontsize=10.5,
        color=MUTED,
    )

    sun = (13.30, 4.55)
    inner_radius = 0.92
    outer_radius = 1.67
    earth = orbit_point(sun, inner_radius, 58.0)
    mars = orbit_point(sun, outer_radius, 145.0)
    axis.add_patch(
        Circle(
            sun,
            inner_radius,
            facecolor="none",
            edgecolor="#78A9CA",
            linewidth=1.6,
            zorder=1,
        )
    )
    axis.add_patch(
        Circle(
            sun,
            outer_radius,
            facecolor="none",
            edgecolor="#D69A8C",
            linewidth=1.6,
            zorder=1,
        )
    )
    axis.plot(
        [sun[0], sun[0] + 1.92],
        [sun[1], sun[1]],
        color=MUTED,
        linewidth=1.1,
        linestyle=(0, (4, 4)),
        zorder=1,
    )
    axis.plot(
        [sun[0], earth[0]],
        [sun[1], earth[1]],
        color=EARTH,
        linewidth=1.6,
        zorder=2,
    )
    axis.plot(
        [sun[0], mars[0]],
        [sun[1], mars[1]],
        color=MARS,
        linewidth=1.6,
        zorder=2,
    )
    axis.add_patch(
        Arc(
            sun,
            1.05,
            1.05,
            theta1=0,
            theta2=58,
            color=EARTH,
            linewidth=1.8,
            zorder=3,
        )
    )
    axis.add_patch(
        Arc(
            sun,
            1.52,
            1.52,
            theta1=58,
            theta2=145,
            color=MARS,
            linewidth=1.8,
            zorder=3,
        )
    )
    axis.text(
        13.75,
        4.78,
        r"$\phi_E$",
        fontsize=11.5,
        fontweight="bold",
        color=EARTH,
    )
    axis.text(
        12.96,
        5.25,
        r"$\phi_M$",
        fontsize=11.5,
        fontweight="bold",
        color=MARS,
    )
    draw_sun(axis, sun, 0.27)
    axis.text(
        sun[0],
        sun[1] - 0.50,
        "Sun",
        ha="center",
        va="top",
        fontsize=9.5,
        fontweight="bold",
        color="#9D6D00",
    )
    axis.add_patch(
        Circle(
            earth,
            0.16,
            facecolor="#62A9DA",
            edgecolor="#176A9F",
            linewidth=1.4,
            zorder=4,
        )
    )
    axis.text(
        earth[0] + 0.16,
        earth[1] + 0.14,
        "Earth",
        ha="left",
        va="bottom",
        fontsize=9,
        fontweight="bold",
        color=EARTH,
    )
    axis.add_patch(
        Circle(
            mars,
            0.17,
            facecolor="#DF7459",
            edgecolor="#9D3527",
            linewidth=1.4,
            zorder=4,
        )
    )
    axis.text(
        mars[0] - 0.04,
        mars[1] + 0.25,
        "Mars",
        ha="center",
        va="bottom",
        fontsize=9,
        fontweight="bold",
        color=MARS,
    )
    axis.add_patch(
        Arc(
            sun,
            inner_radius * 2.0,
            inner_radius * 2.0,
            theta1=205,
            theta2=330,
            color=EARTH,
            linewidth=3.2,
            zorder=3,
        )
    )
    axis.add_patch(
        FancyArrowPatch(
            orbit_point(sun, inner_radius, 319),
            orbit_point(sun, inner_radius, 331),
            arrowstyle="-|>",
            mutation_scale=21,
            color=EARTH,
            linewidth=3.0,
            connectionstyle="arc3,rad=0.05",
            zorder=5,
        )
    )
    axis.add_patch(
        Arc(
            sun,
            outer_radius * 2.0,
            outer_radius * 2.0,
            theta1=24,
            theta2=132,
            color=MARS,
            linewidth=3.2,
            zorder=3,
        )
    )
    axis.add_patch(
        FancyArrowPatch(
            orbit_point(sun, outer_radius, 122),
            orbit_point(sun, outer_radius, 133),
            arrowstyle="-|>",
            mutation_scale=21,
            color=MARS,
            linewidth=3.0,
            connectionstyle="arc3,rad=0.05",
            zorder=5,
        )
    )

    axis.text(
        panel_x + panel_w / 2.0,
        2.40,
        r"$\phi_E$ and $\phi_M$ advance at constant rates",
        ha="center",
        va="center",
        fontsize=11.1,
        fontweight="bold",
        color=INK,
    )
    axis.text(
        panel_x + panel_w / 2.0,
        1.90,
        "The learned 2D latent is a linear\ncombination of these heliocentric angles.",
        ha="center",
        va="center",
        fontsize=9.4,
        color=MUTED,
        linespacing=1.25,
    )


def draw_overview(output_stem: Path, dpi: int) -> None:
    figure, axis = plt.subplots(figsize=(16, 9))
    figure.patch.set_facecolor(BACKGROUND)
    axis.set_facecolor(BACKGROUND)
    axis.set_xlim(0.0, 16.0)
    axis.set_ylim(0.0, 9.0)
    axis.set_aspect("equal")
    axis.axis("off")

    axis.text(
        8.0,
        8.58,
        "The Solar-System Problem at a Glance",
        ha="center",
        va="center",
        fontsize=24,
        fontweight="bold",
        color=INK,
    )
    draw_observation_panel(axis)
    draw_learning_panel(axis)
    draw_solution_panel(axis)

    rounded_panel(
        axis,
        (1.28, 0.28),
        13.44,
        0.72,
        facecolor="#EFF7F1",
        edgecolor=GREEN,
    )
    axis.text(
        8.0,
        0.64,
        "Key idea: a complicated Earth-view signal becomes simple, constant "
        "motion in heliocentric coordinates.",
        ha="center",
        va="center",
        fontsize=12.2,
        fontweight="bold",
        color=GREEN,
    )
    axis.text(
        15.65,
        0.08,
        "Iten et al., 2020 • Fig. 3 concept",
        ha="right",
        va="bottom",
        fontsize=7.5,
        color=MUTED,
    )

    figure.tight_layout(pad=0.15)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_stem.with_suffix(".png"), dpi=dpi, facecolor=BACKGROUND)
    figure.savefig(output_stem.with_suffix(".pdf"), facecolor=BACKGROUND)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    draw_overview(args.output_dir / "solar_problem_at_a_glance", args.dpi)
    print(f"Saved solar problem overview to {args.output_dir}")


if __name__ == "__main__":
    main()
