#!/usr/bin/env python3
"""Create presentation figures comparing the preserved reservoir with SciNet.

The script consumes the saved latent grids and held-out predictions, so it does
not retrain either model.  The SciNet reference is the authors' released
checkpoint, already evaluated on the same latent grid and held-out targets.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "bp_reservoir_mpl")
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np


TWO_PI = 2.0 * np.pi
OURS_COLOR = "#D55E00"
SCINET_COLOR = "#0072B2"

DEFAULT_OURS_DIR = Path(
    "solar/results/solar-10x150-preserved-15k-109965/reservoir/seed_0"
)
DEFAULT_SCINET_ANALYSIS = Path(
    "solar/results/solar-performance-gap-grid-winner/diagnostics.npz"
)
DEFAULT_SCINET_METRICS = Path(
    "solar/results/solar-performance-gap-grid-winner/metrics.json"
)
DEFAULT_OUTPUT_DIR = Path(
    "solar/results/solar-10x150-preserved-vs-scinet-presentation"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ours-dir", type=Path, default=DEFAULT_OURS_DIR)
    parser.add_argument(
        "--scinet-analysis", type=Path, default=DEFAULT_SCINET_ANALYSIS
    )
    parser.add_argument(
        "--scinet-metrics", type=Path, default=DEFAULT_SCINET_METRICS
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def _load_npz(path: Path, required: tuple[str, ...]) -> dict[str, np.ndarray]:
    with np.load(path) as archive:
        missing = set(required) - set(archive.files)
        if missing:
            raise KeyError(f"{path} is missing arrays: {sorted(missing)}")
        return {key: archive[key] for key in required}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _assert_same(name: str, ours: np.ndarray, scinet: np.ndarray) -> None:
    if ours.shape != scinet.shape or not np.allclose(ours, scinet, atol=1e-7):
        raise ValueError(
            f"the saved runs do not use the same {name}: "
            f"{ours.shape} versus {scinet.shape}"
        )


def _standardize_components(latent: np.ndarray) -> np.ndarray:
    mean = np.mean(latent, axis=(0, 1), keepdims=True)
    standard_deviation = np.std(latent, axis=(0, 1), keepdims=True)
    return (latent - mean) / np.maximum(standard_deviation, 1e-12)


def _configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 11,
            "figure.titlesize": 20,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def _save_figure(figure: plt.Figure, output_stem: Path) -> None:
    figure.savefig(
        output_stem.with_suffix(".png"),
        dpi=240,
    )
    figure.savefig(
        output_stem.with_suffix(".pdf"),
    )
    plt.close(figure)


def _format_angle_axis(axis: Any) -> None:
    ticks = (0.0, np.pi, TWO_PI)
    labels = ("0", r"$\pi$", r"$2\pi$")
    axis.set_xticks(ticks, labels)
    axis.set_yticks(ticks, labels)
    axis.set_xlim(0.0, TWO_PI)
    axis.set_ylim(0.0, TWO_PI)
    axis.set_zlim(-3.25, 3.25)
    axis.set_zticks((-2.0, 0.0, 2.0))
    axis.view_init(elev=27, azim=-58)
    axis.set_box_aspect((1.2, 1.0, 0.62))
    axis.xaxis.pane.fill = False
    axis.yaxis.pane.fill = False
    axis.zaxis.pane.fill = False
    axis.grid(True, alpha=0.22)


def plot_latent_comparison(
    phi_earth: np.ndarray,
    phi_mars: np.ndarray,
    ours_latent: np.ndarray,
    scinet_latent: np.ndarray,
    ours_r2: float,
    ours_r2_per_component: list[float],
    scinet_r2: float,
    scinet_r2_per_component: list[float],
    output_stem: Path,
) -> None:
    """Plot matched 3D latent surfaces over the same physical grid."""

    ours_standardized = _standardize_components(ours_latent)
    scinet_standardized = _standardize_components(scinet_latent)
    scinet_display_signs = np.asarray(
        [
            1.0
            if np.sum(
                ours_standardized[..., component]
                * scinet_standardized[..., component]
            )
            >= 0.0
            else -1.0
            for component in range(scinet_standardized.shape[-1])
        ],
        dtype=np.float64,
    )
    scinet_standardized = scinet_standardized * scinet_display_signs
    all_values = np.concatenate(
        (ours_standardized.ravel(), scinet_standardized.ravel())
    )
    limit = max(3.25, float(np.ceil(np.max(np.abs(all_values)) * 4.0) / 4.0))
    normalization = colors.Normalize(vmin=-limit, vmax=limit)
    colormap = plt.get_cmap("plasma")

    figure = plt.figure(figsize=(13.33, 7.5))
    model_rows = (
        (
            "Preserved 10×150",
            ours_standardized,
            ours_r2_per_component,
        ),
        (
            "SciNet (released checkpoint)",
            scinet_standardized,
            scinet_r2_per_component,
        ),
    )
    for row, (
        model_title,
        latent,
        component_alignment_r2,
    ) in enumerate(model_rows):
        for component in range(2):
            axis = figure.add_subplot(
                2, 2, row * 2 + component + 1, projection="3d"
            )
            axis.plot_surface(
                phi_earth,
                phi_mars,
                latent[..., component],
                facecolors=colormap(normalization(latent[..., component])),
                rstride=1,
                cstride=1,
                linewidth=0,
                antialiased=True,
                shade=False,
            )
            axis.set_title(
                f"{model_title} — latent {component + 1}\n"
                rf"$R^2={component_alignment_r2[component]:.4f}$",
                pad=5,
                fontsize=12.5,
                fontweight="bold",
            )
            axis.set_xlabel(r"Earth phase $\phi_E$", labelpad=5)
            axis.set_ylabel(r"Mars phase $\phi_M$", labelpad=5)
            axis.set_zlabel("Standardized activation", labelpad=5)
            _format_angle_axis(axis)

    figure.subplots_adjust(
        left=0.01,
        right=0.99,
        bottom=0.10,
        top=0.86,
        wspace=0.02,
        hspace=0.26,
    )
    figure.suptitle(
        "Latent activations over the same heliocentric state space",
        y=0.975,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.015,
        "Standardized surfaces; SciNet latent 1 is sign-flipped for visual "
        "alignment ($R^2$ unchanged).\n"
        rf"Panel $R^2$: held-out heliocentric→latent. Pooled: preserved "
        rf"{ours_r2:.3f}, SciNet {scinet_r2:.3f}.",
        ha="center",
        va="bottom",
        fontsize=8.5,
        color="#4D4D4D",
    )
    _save_figure(figure, output_stem)


def _relative_rmse(
    prediction: np.ndarray, target: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    squared_error = (
        prediction.astype(np.float64) - target.astype(np.float64)
    ) ** 2
    component = 100.0 * np.sqrt(np.mean(squared_error, axis=(0, 1))) / TWO_PI
    overall = 100.0 * np.sqrt(np.mean(squared_error)) / TWO_PI
    by_lead = (
        100.0 * np.sqrt(np.mean(squared_error, axis=(0, 2))) / TWO_PI
    )
    return np.concatenate(([overall], component)), by_lead


def plot_error_comparison(
    ours_prediction: np.ndarray,
    scinet_prediction: np.ndarray,
    target: np.ndarray,
    output_stem: Path,
) -> dict[str, Any]:
    ours_summary, ours_by_lead = _relative_rmse(ours_prediction, target)
    scinet_summary, scinet_by_lead = _relative_rmse(scinet_prediction, target)

    figure, (bars_axis, curve_axis) = plt.subplots(
        1,
        2,
        figsize=(13.33, 7.5),
        gridspec_kw={"width_ratios": (1.0, 1.65)},
    )
    categories = ("Overall", "Sun angle", "Mars angle")
    x = np.arange(len(categories))
    width = 0.35
    ours_bars = bars_axis.bar(
        x - width / 2,
        ours_summary,
        width,
        label="Preserved 10×150 reservoir",
        color=OURS_COLOR,
    )
    scinet_bars = bars_axis.bar(
        x + width / 2,
        scinet_summary,
        width,
        label="SciNet (released checkpoint)",
        color=SCINET_COLOR,
    )
    bars_axis.bar_label(
        ours_bars,
        labels=[f"{value:.2f}%" for value in ours_summary],
        padding=4,
        fontsize=10,
        fontweight="bold",
    )
    bars_axis.bar_label(
        scinet_bars,
        labels=[f"{value:.2f}%" for value in scinet_summary],
        padding=4,
        fontsize=10,
        fontweight="bold",
    )
    bars_axis.set(
        title="Held-out test error",
        ylabel=r"Relative RMSE (% of $2\pi$)",
        xticks=x,
        xticklabels=categories,
    )
    bars_axis.set_ylim(0.0, max(ours_summary.max(), scinet_summary.max()) * 1.18)
    bars_axis.grid(axis="y", alpha=0.25)
    bars_axis.set_axisbelow(True)

    lead = np.arange(target.shape[1])
    curve_axis.plot(
        lead,
        ours_by_lead,
        color=OURS_COLOR,
        linewidth=2.8,
        label="Preserved 10×150 reservoir",
    )
    curve_axis.plot(
        lead,
        scinet_by_lead,
        color=SCINET_COLOR,
        linewidth=2.8,
        label="SciNet (released checkpoint)",
    )
    curve_axis.fill_between(
        lead,
        ours_by_lead,
        scinet_by_lead,
        color="#808080",
        alpha=0.10,
    )
    curve_axis.set(
        title="Error across the 50-week trajectory",
        xlabel="Lead time (weeks)",
        ylabel=r"Relative RMSE (% of $2\pi$)",
        xlim=(0, target.shape[1] - 1),
    )
    curve_axis.grid(alpha=0.25)
    curve_axis.legend(loc="upper left", frameon=False)

    figure.suptitle(
        "Forecast error: Preserved 10×150 vs SciNet",
        y=0.975,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.015,
        "Paired evaluation on the same 5,000 held-out trajectories.",
        ha="center",
        va="bottom",
        fontsize=9.5,
        color="#4D4D4D",
    )
    figure.subplots_adjust(
        left=0.075, right=0.985, bottom=0.12, top=0.88, wspace=0.28
    )
    _save_figure(figure, output_stem)

    return {
        "test_trajectories": int(target.shape[0]),
        "trajectory_steps": int(target.shape[1]),
        "metric": "RMSE / (2*pi), percent",
        "preserved_10x150": {
            "overall": float(ours_summary[0]),
            "sun": float(ours_summary[1]),
            "mars": float(ours_summary[2]),
        },
        "scinet_released_checkpoint": {
            "overall": float(scinet_summary[0]),
            "sun": float(scinet_summary[1]),
            "mars": float(scinet_summary[2]),
        },
        "overall_error_ratio_reservoir_over_scinet": float(
            ours_summary[0] / scinet_summary[0]
        ),
    }


def main() -> None:
    args = parse_args()
    _configure_style()

    ours_surface = _load_npz(
        args.ours_dir / "latent_surface.npz",
        ("phi_earth", "phi_mars", "latent"),
    )
    scinet_analysis = _load_npz(
        args.scinet_analysis,
        (
            "phi_earth",
            "phi_mars",
            "official_surface",
            "target",
            "official_prediction",
        ),
    )
    ours_predictions = _load_npz(
        args.ours_dir / "predictions.npz", ("target", "prediction")
    )
    ours_metrics = _load_json(args.ours_dir / "metrics.json")
    scinet_report = _load_json(args.scinet_metrics)
    scinet_metrics = scinet_report["official_latent_diagnostics"]

    _assert_same(
        "heliocentric Earth grid",
        ours_surface["phi_earth"],
        scinet_analysis["phi_earth"],
    )
    _assert_same(
        "heliocentric Mars grid",
        ours_surface["phi_mars"],
        scinet_analysis["phi_mars"],
    )
    _assert_same(
        "held-out test targets",
        ours_predictions["target"],
        scinet_analysis["target"],
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_latent_comparison(
        ours_surface["phi_earth"],
        ours_surface["phi_mars"],
        ours_surface["latent"],
        scinet_analysis["official_surface"],
        float(ours_metrics["heliocentric_to_latent_r2"]),
        [
            float(value)
            for value in ours_metrics[
                "heliocentric_to_latent_r2_per_dimension"
            ]
        ],
        float(scinet_metrics["heliocentric_to_latent_r2"]),
        [
            float(value)
            for value in scinet_metrics[
                "heliocentric_to_latent_r2_per_dimension"
            ]
        ],
        args.output_dir / "latent_space_comparison",
    )
    comparison = plot_error_comparison(
        ours_predictions["prediction"],
        scinet_analysis["official_prediction"],
        ours_predictions["target"],
        args.output_dir / "forecast_error_comparison",
    )
    comparison["latent_alignment_r2"] = {
        "preserved_10x150": float(
            ours_metrics["heliocentric_to_latent_r2"]
        ),
        "scinet_released_checkpoint": float(
            scinet_metrics["heliocentric_to_latent_r2"]
        ),
    }
    comparison["latent_alignment_r2_per_component"] = {
        "preserved_10x150": [
            float(value)
            for value in ours_metrics[
                "heliocentric_to_latent_r2_per_dimension"
            ]
        ],
        "scinet_released_checkpoint": [
            float(value)
            for value in scinet_metrics[
                "heliocentric_to_latent_r2_per_dimension"
            ]
        ],
    }
    (args.output_dir / "comparison_metrics.json").write_text(
        json.dumps(comparison, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(comparison, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
