"""PHQ-9/BDI-II item small-multiple trajectory figures."""

from __future__ import annotations

import math

from pathlib import Path

import numpy as np

import pandas as pd

from matplotlib.lines import Line2D

from ....life_state import ITEM_LABELS_EN

from ....schema import group_color, group_sort_key, slugify

from ...shared.plotting import plt, save_figure


def plot_symptom_trajectory_small_multiples(
    summary: pd.DataFrame,
    out_dir: Path,
    *,
    scale: str,
    interval: str = "ci95",
    title: str | None = None,
) -> Path | None:
    if summary.empty:
        return None
    items = sorted(int(value) for value in summary["item"].unique())
    groups = sorted(summary["group"].astype(str).unique(), key=group_sort_key)
    timepoints = (
        summary[["timepoint", "time_order"]]
        .drop_duplicates()
        .sort_values("time_order")["timepoint"]
        .astype(str)
        .tolist()
    )
    columns = 3
    rows = math.ceil(len(items) / columns)
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(12.6, 3.25 * rows),
        sharex=True,
        sharey=True,
        squeeze=False,
        constrained_layout=True,
    )
    for ax, item in zip(axes.ravel(), items):
        item_rows = summary.loc[summary["item"] == item]
        for group in groups:
            selected = item_rows.loc[item_rows["group"] == group].sort_values(
                "time_order"
            )
            x = selected["time_order"].to_numpy(dtype=float)
            center = selected["mean"].to_numpy(dtype=float)
            color = group_color(group)
            ax.plot(x, center, marker="o", markersize=3.2, linewidth=1.45, color=color)
            if interval == "se":
                error = selected["se"].to_numpy(dtype=float)
                lower, upper = center - error, center + error
            else:
                lower = selected["ci95_lower"].to_numpy(dtype=float)
                upper = selected["ci95_upper"].to_numpy(dtype=float)
            finite = np.isfinite(lower) & np.isfinite(upper)
            if finite.any():
                ax.fill_between(
                    x[finite],
                    lower[finite],
                    upper[finite],
                    color=color,
                    alpha=0.14,
                    linewidth=0,
                )
        label = ITEM_LABELS_EN.get(scale, {}).get(item, f"Item {item}")
        ax.set_title(f"I{item:02d} · {label}", fontsize=9.2)
        ax.set_ylim(-0.05, 3.05)
        ax.set_yticks([0, 1, 2, 3])
        ax.grid(axis="y", alpha=0.2, linewidth=0.7)
        ax.tick_params(axis="both", labelsize=7.5)
    for ax in axes.ravel()[len(items) :]:
        ax.axis("off")
    for ax in axes[-1, :]:
        if ax.axison:
            ax.set_xticks(range(len(timepoints)), timepoints, rotation=42, ha="right")
            ax.set_xlabel("Assessment timepoint")
    for ax in axes[:, 0]:
        ax.set_ylabel("Mean item score (0–3)")
    handles = [
        Line2D(
            [0], [0], color=group_color(group), marker="o", linewidth=1.6, label=group
        )
        for group in groups
    ]
    fig.legend(
        handles=handles, loc="outside right upper", ncol=1, frameon=False, title="Group"
    )
    fig.suptitle(
        title or f"{scale} item trajectories by experimental group",
        fontsize=13,
        fontweight="bold",
    )
    interval_label = "SE" if interval == "se" else "95% t CI"
    fig.text(
        0.5,
        -0.012,
        f"Points are independent-run means; ribbons show {interval_label}. Frozen-snapshot measurement repeats are pre-averaged.",
        ha="center",
        fontsize=8,
        color="#475569",
    )
    return save_figure(fig, out_dir / f"symptom_trajectory_{slugify(scale)}.png")
