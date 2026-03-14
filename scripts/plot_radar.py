#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_radar.py — Professional radar plotter with enhanced aesthetics.

Creates publication-ready radar charts with:
- Clean, modern visual design
- Improved typography and spacing
- Professional color schemes
- Clear metric labeling with units
"""

import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import math

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Circle

# Xiang: added for my system
plt.rcParams["font.family"] = "sans-serif"

# Enhanced visual constants
ERR = 0.03
FONT_FAMILY = "sans-serif"  # Professional font
PALETTE_NAME = "Set2"  # More professional color palette
BACKGROUND_COLOR = "#fafbfc"
GRID_COLOR = "#c6cbd1"  # Darker grid color for better visibility
TEXT_COLOR = "#24292e"
SPINE_COLOR = "#d0d7de"

# Metric display names and units for better clarity
METRIC_DISPLAY_NAMES = {
    "loss": "Loss",
    "mse": "MSE",
    "rmse": "RMSE",
    "psnr": "PSNR (dB)",
    "ssim": "SSIM",
    "lpips": "LPIPS",
    "accuracy": "Accuracy (%)",
    "f1": "F1-Score",
    "auc": "AUC",
    "recall": "Recall (%)",
    "precision": "Precision (%)",
}


def get_display_name(metric: str) -> str:
    """Convert raw metric name to professional display name."""
    metric_lower = metric.lower()

    # Check for exact matches first
    if metric_lower in METRIC_DISPLAY_NAMES:
        return METRIC_DISPLAY_NAMES[metric_lower]

    # Check for partial matches
    for key, display_name in METRIC_DISPLAY_NAMES.items():
        if key in metric_lower:
            return display_name

    # Default: capitalize and clean up
    return metric.replace("_", " ").title()


def get_display_name_new(metric: str) -> str:
    """Convert raw metric name to professional display name."""
    return metric.replace("-", "\n")


def metric_direction(raw_metric: str) -> str:
    m = raw_metric.lower()
    if any(
        k in m
        for k in [
            "loss",
            "mse",
            "rmse",
            "error",
            "err",
            "lpips",
            "cd",
            "point",
            "depth",
        ]
    ):
        return "down"
    if any(
        k in m
        for k in [
            "auc",
            "acc",
            "recall",
            "rra",
            "rta",
            "psnr",
            "ssim",
            "fs",
            "precision",
            "f1",
        ]
    ):
        return "up"
    return "up"


def normalize_with_margin(
    values: np.ndarray, directions: List[str], err: float = ERR
) -> np.ndarray:
    norm_vals = np.full_like(values, np.nan, dtype=float)
    for j in range(values.shape[1]):
        col = values[:, j]
        mask = np.isfinite(col)
        if not np.any(mask):
            continue
        vmin = np.nanmin(col[mask])
        vmax = np.nanmax(col[mask])
        if np.isclose(vmax, vmin, atol=1e-12):
            norm_vals[mask, j] = 1.0 - err
            continue
        den = max(vmax - vmin, 1e-12)
        if directions[j] == "down":
            norm = (vmax - col[mask]) / den
        else:
            norm = (col[mask] - vmin) / den
        norm_vals[mask, j] = err + (1.0 - 2.0 * err) * norm
    return norm_vals


def make_radar(
    df: pd.DataFrame, metrics: List[str], out_path: Path, annotate_axis: bool = False
):
    runs = df["run"].astype(str).tolist()
    vals = df[metrics].to_numpy(dtype=float)
    directions = [metric_direction(m) for m in metrics]

    # Store original value ranges for axis annotations
    value_ranges = {}
    for j, metric in enumerate(metrics):
        col = vals[:, j]
        mask = np.isfinite(col)
        if np.any(mask):
            vmin = np.nanmin(col[mask])
            vmax = np.nanmax(col[mask])
            value_ranges[metric] = (vmin, vmax, directions[j])

    # Normalize values
    norm_vals = normalize_with_margin(vals, directions, ERR)

    K = len(metrics)
    base_angles = np.linspace(0, 2 * np.pi, K, endpoint=False)

    # Professional styling setup
    plt.style.use("default")  # Reset to clean slate
    plt.rcParams.update(
        {
            "font.family": FONT_FAMILY,
            "font.size": 18,  # Increased from 16 for larger plot
            "axes.linewidth": 1.2,
            "axes.edgecolor": SPINE_COLOR,
            "axes.facecolor": BACKGROUND_COLOR,
            "figure.facecolor": "white",
            "text.color": TEXT_COLOR,
            "axes.labelcolor": TEXT_COLOR,
            "xtick.color": TEXT_COLOR,
            "ytick.color": TEXT_COLOR,
            "grid.color": GRID_COLOR,
            "grid.linewidth": 1.0,
            "grid.alpha": 0.9,
        }
    )

    # Larger figure size - increased from (12, 10) and scaled based on legend
    num_runs = len(runs)
    ncols = math.ceil(num_runs / 2)  # 2 rows maximum
    fig_width = max(16, ncols * 3.5)  # Increased from max(12, ncols * 2.5)
    fig_height = 16  # Increased from 10
    fig = plt.figure(figsize=(fig_width, fig_height), facecolor="white")
    ax = plt.subplot(111, polar=True)

    # Enhanced background styling
    ax.set_facecolor(BACKGROUND_COLOR)
    ax.spines["polar"].set_color(SPINE_COLOR)
    ax.spines["polar"].set_linewidth(1.5)

    # Add more visible concentric circles with better intervals
    for radius in [0.25, 0.5, 0.75, 1.0 - ERR]:  # Same intervals as yticks
        circle = Circle(
            (0, 0),
            radius,
            fill=False,
            color=GRID_COLOR,
            linewidth=1.5,
            alpha=0.9,
            linestyle=(0, (3, 3)),
            transform=ax.transData._b,
        )
        ax.add_patch(circle)

    # Professional color palette
    colors = sns.color_palette(PALETTE_NAME, n_colors=len(runs))
    legend_handles = []

    # Plot each run with enhanced styling - scaled markers for larger plot
    for i, rn in enumerate(runs):
        v = norm_vals[i, :]
        angles = base_angles
        finite = np.isfinite(v)
        color = colors[i]

        if np.all(finite):
            # Complete polygon
            vv = np.concatenate([v, v[:1]])
            aa = np.concatenate([angles, angles[:1]])

            # Main line with better styling - increased sizes for larger plot
            ax.plot(
                aa,
                vv,
                linewidth=4,
                alpha=0.9,
                color=color,  # Increased from 3
                marker="o",
                markersize=12,
                markeredgewidth=0,  # Increased from 10
                markerfacecolor=color,
                zorder=3,
            )

            # Fill with gradient-like effect
            ax.fill(aa, vv, alpha=0.15, color=color, zorder=1)

        else:
            # Handle missing data points
            ax.scatter(
                angles[finite],
                v[finite],
                s=120,
                color=color,  # Increased from 80
                edgecolor="none",
                linewidth=0,
                zorder=5,
            )

            # Connect available segments
            start = None
            for k in range(K * 2):
                idx = k % K
                if finite[idx] and start is None:
                    start = idx
                if (not finite[idx] or k == K * 2 - 1) and start is not None:
                    end = (idx - 1) % K
                    if end >= start:
                        a_seg = angles[start : end + 1]
                        v_seg = v[start : end + 1]
                    else:
                        a_seg = np.concatenate([angles[start:], angles[: end + 1]])
                        v_seg = np.concatenate([v[start:], v[: end + 1]])
                    ax.plot(
                        a_seg,
                        v_seg,
                        linewidth=4,
                        alpha=0.9,
                        color=color,  # Increased from 3
                        marker="o",
                        markersize=12,
                        markeredgewidth=0,  # Increased from 10
                        markerfacecolor=color,
                    )
                    start = None

        # Create legend handle - larger markers to match radar text scale
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="-",
                linewidth=4,
                markerfacecolor=color,
                markeredgecolor=color,
                markeredgewidth=0,
                markersize=20,
                color=color,
                label=rn,
            )
        )  # Increased from 16

    # Enhanced axis labels with larger font sizes for bigger plot
    ax.set_xticks(base_angles)
    display_labels = [get_display_name_new(m) for m in metrics]
    ax.set_xticklabels(
        display_labels,
        fontsize=22,
        fontweight="bold",  # Increased from 18
        color=TEXT_COLOR,
        ha="center",
    )

    # Improve label positioning with more padding for larger plot
    ax.tick_params(axis="x", pad=40)  # Increased from 30

    # Clean radial axis with matching tick positions
    ax.set_yticks([0.25, 0.5, 0.75, 1.0 - ERR])
    if annotate_axis:
        # Add individual text annotations for each axis showing actual values
        tick_levels = [0.25, 0.5, 0.75, 1.0 - ERR]

        for j, (metric, angle) in enumerate(zip(metrics, base_angles)):
            if metric in value_ranges:
                vmin, vmax, direction = value_ranges[metric]

                # Calculate actual values for this specific metric
                for tick_level in tick_levels:
                    # Reverse the normalization formula
                    true_norm = (tick_level - ERR) / (1.0 - 2.0 * ERR)
                    true_norm = max(0.0, min(1.0, true_norm))

                    if direction == "down":
                        actual_val = vmax - true_norm * (vmax - vmin)
                    else:
                        actual_val = vmin + true_norm * (vmax - vmin)

                    # Format the value
                    if abs(actual_val) >= 1000:
                        val_str = f"{actual_val:.0f}"
                    elif abs(actual_val) >= 10:
                        val_str = f"{actual_val:.1f}"
                    elif abs(actual_val) >= 1:
                        val_str = f"{actual_val:.2f}"
                    else:
                        val_str = f"{actual_val:.3f}"

                    # Add text annotation at this position
                    ax.text(
                        angle,
                        tick_level,
                        val_str,
                        fontsize=10,
                        ha="center",
                        va="center",
                        color="#8a9099",
                        weight="normal",
                        alpha=0.8,
                    )

        # Remove radial tick labels to avoid clutter
        ax.set_yticklabels([])
    else:
        ax.set_yticklabels([])
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.4)

    # Professional legend with 2 rows maximum - same font size as radar text
    if legend_handles:
        ncols = math.ceil(len(legend_handles) / 2)  # Calculate columns for 2 rows
        legend = fig.legend(
            handles=legend_handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.02),
            ncol=ncols,
            frameon=True,
            fancybox=True,
            shadow=True,
            framealpha=0.95,
            facecolor="white",
            edgecolor=SPINE_COLOR,
            fontsize=22,  # Matches radar text font size (was 16)
            handletextpad=0.8,  # Increased spacing for larger markers
            columnspacing=2.5,  # Increased for better spacing with larger elements
            borderaxespad=0,
        )
        legend.get_frame().set_linewidth(1.2)

    # Improved layout with more padding for larger plot and legend
    plt.tight_layout(pad=2.5)
    plt.subplots_adjust(
        top=0.94,
        bottom=0.12,  # Slightly increased for larger legend
        left=0.04,
        right=0.96,
    )

    # Save with high quality
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        out_path,
        dpi=600,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
        pad_inches=0.3,
    )
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(
        description="Generate professional radar plots from CSV data."
    )
    ap.add_argument(
        "--csv",
        required=True,
        help="Path to CSV with columns: run,<metric_1>,...,<metric_K>",
    )
    ap.add_argument(
        "--out", required=True, help="Output file path (e.g., radar.png or radar.pdf)"
    )
    ap.add_argument(
        "--title",
        default="Performance Comparison",
        help="Chart title (default: 'Performance Comparison')",
    )
    ap.add_argument(
        "--annotate-axis",
        action="store_true",
        help="Add scale annotations to radial axis (default: False)",
    )
    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if "run" not in df.columns:
        raise ValueError("CSV must include a 'run' column.")

    metrics = [c for c in df.columns if c != "run"]
    if not metrics:
        raise ValueError("No metric columns found after 'run'.")

    out_path = Path(args.out)
    make_radar(df[["run"] + metrics], metrics, out_path, args.annotate_axis)
    print(f"✅ Professional radar chart saved to: {out_path}")


if __name__ == "__main__":
    main()
