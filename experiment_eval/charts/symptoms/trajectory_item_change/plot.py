"""Panel A total-score trajectories; panels B/C run-level item changes."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D

from ....life_state import ITEM_LABELS_EN
from ....schema import SCALE_RANGES, group_color, group_sort_key, scale_sort_key
from ...shared.plotting import heatmap_text_color, plt, save_figure


def _display_timepoint(value: str) -> str:
    text = str(value)
    if text == "T0":
        return text
    if text.startswith("session_"):
        return f"S{text.rsplit('_', 1)[-1]}"
    return text.replace("_", " ")


def _plot_heatmap(
    ax: object,
    rows: pd.DataFrame,
    *,
    scale: str,
    norm: TwoSlopeNorm,
    cmap: object,
) -> object:
    items = sorted(int(value) for value in rows["item"].unique())
    runs = (
        rows[["run", "run_label", "run_order", "group"]]
        .drop_duplicates()
        .sort_values("run_order")
    )
    matrix = (
        rows.pivot(index="item", columns="run", values="score_change")
        .reindex(index=items, columns=runs["run"])
        .to_numpy(dtype=float)
    )
    image = ax.imshow(matrix, aspect="auto", cmap=cmap, norm=norm)
    ax.set_xticks(
        np.arange(len(runs)), runs["run_label"], rotation=48, ha="right", fontsize=7
    )
    labels = [
        f"I{item:02d} {ITEM_LABELS_EN.get(scale, {}).get(item, f'Item {item}')}"
        for item in items
    ]
    ax.set_yticks(np.arange(len(items)), labels, fontsize=7)
    ax.set_title(scale, fontsize=11, fontweight="bold")
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            value = matrix[row_index, column_index]
            if np.isfinite(value):
                ax.text(
                    column_index,
                    row_index,
                    f"{value:+.1f}",
                    ha="center",
                    va="center",
                    fontsize=5.8,
                    color=heatmap_text_color(cmap, norm, float(value)),
                )
    group_values = runs["group"].astype(str).tolist()
    for index in range(1, len(group_values)):
        if group_values[index] != group_values[index - 1]:
            ax.axvline(index - 0.5, color="#111827", linewidth=1.2)
    ax.tick_params(length=0)
    return image


def plot_symptom_trajectory_item_change(
    trajectory: pd.DataFrame,
    item_changes: pd.DataFrame,
    out_dir: Path,
    *,
    title: str | None = None,
) -> Path | None:
    if trajectory.empty or item_changes.empty:
        return None
    scales = sorted(trajectory["scale"].astype(str).unique(), key=scale_sort_key)
    groups = sorted(trajectory["group"].astype(str).unique(), key=group_sort_key)
    if len(scales) != 2:
        raise ValueError("Trajectory/item-change composite requires two scales")

    fig = plt.figure(figsize=(16.0, 8.8), constrained_layout=True)
    grid = fig.add_gridspec(2, 3, width_ratios=[1.35, 1.0, 1.28])
    trajectory_axes = [fig.add_subplot(grid[index, 0]) for index in range(2)]
    heatmap_axes = [fig.add_subplot(grid[:, 1]), fig.add_subplot(grid[:, 2])]

    for ax, scale in zip(trajectory_axes, scales):
        scale_rows = trajectory.loc[trajectory["scale"] == scale]
        for group in groups:
            selected = scale_rows.loc[scale_rows["group"] == group].sort_values(
                "time_order"
            )
            x = selected["time_order"].to_numpy(dtype=float)
            center = selected["mean"].to_numpy(dtype=float)
            lower = selected["ci95_lower"].to_numpy(dtype=float)
            upper = selected["ci95_upper"].to_numpy(dtype=float)
            errors = np.vstack([center - lower, upper - center])
            ax.errorbar(
                x,
                center,
                yerr=errors,
                color=group_color(group),
                marker="o",
                linewidth=1.8,
                capsize=3,
                markersize=4,
                label=group,
            )
        timepoints = (
            scale_rows[["timepoint", "time_order"]]
            .drop_duplicates()
            .sort_values("time_order")
        )
        ax.set_xticks(
            timepoints["time_order"],
            [_display_timepoint(value) for value in timepoints["timepoint"]],
            fontsize=8,
        )
        ax.set_ylabel(f"{scale} total score")
        if scale in SCALE_RANGES:
            ax.set_ylim(*SCALE_RANGES[scale])
        ax.grid(axis="y", alpha=0.2)
    trajectory_axes[-1].set_xlabel("Assessment timepoint")
    handles = [
        Line2D(
            [0], [0], color=group_color(group), marker="o", linewidth=1.8, label=group
        )
        for group in groups
    ]
    trajectory_axes[0].legend(handles=handles, title="Group", frameon=False, loc="best")
    trajectory_axes[0].text(
        -0.14, 1.07, "A", transform=trajectory_axes[0].transAxes, fontsize=14, fontweight="bold"
    )

    finite_changes = item_changes["score_change"].astype(float).to_numpy()
    limit = max(0.5, float(np.nanmax(np.abs(finite_changes))))
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
    cmap = plt.get_cmap("RdBu_r")
    images = []
    for panel, ax, scale in zip(("B", "C"), heatmap_axes, scales):
        image = _plot_heatmap(
            ax,
            item_changes.loc[item_changes["scale"] == scale],
            scale=scale,
            norm=norm,
            cmap=cmap,
        )
        images.append(image)
        ax.text(
            -0.18, 1.03, panel, transform=ax.transAxes, fontsize=14, fontweight="bold"
        )
    colorbar = fig.colorbar(images[-1], ax=heatmap_axes, fraction=0.025, pad=0.02)
    colorbar.set_label("Item score change (endpoint − T0)")

    fig.suptitle(
        title or "Two-scale trajectories and item-level change by independent run",
        fontsize=14,
        fontweight="bold",
    )
    fig.text(
        0.5,
        -0.012,
        "Panel A: mean total score and 95% t CI across independent outer runs. "
        "Panels B/C: one column per outer run; negative (blue) indicates symptom improvement, positive (red) worsening.",
        ha="center",
        fontsize=8,
        color="#475569",
    )
    return save_figure(fig, out_dir / "symptom_trajectory_item_change_composite.png")
