"""Run-preserving figures for cross-persona and cross-condition interpretation."""

from __future__ import annotations

import math

from pathlib import Path

from typing import Any

import numpy as np

from ....schema import group_sort_key, label_display, scale_sort_key

from ...shared.plotting import Line2D, plt, save_figure

AGREEMENT_CODES = {
    "两次均改善": -1.0,
    "两次均恶化": 1.0,
    "两次均稳定": 0.0,
    "方向不一致或含稳定": 0.45,
    "重复不足": np.nan,
}

CLASS_COLORS = {
    "推进且同期症状改善": "#16a34a",
    "主诉停滞但同期症状改善": "#2563eb",
    "主诉推进但同期症状未改善": "#f59e0b",
    "主诉停滞且同期症状未改善": "#dc2626",
}


def plot_complaint_interval_alignment(
    rows: list[dict[str, Any]],
    labels: list[str],
    entity_field: str,
    title: str,
    out_dir: Path,
) -> Path | None:
    if not rows:
        return None
    intervals = [
        (first, second)
        for first, second in zip(labels, labels[1:])
        if any(
            row["from_timepoint"] == first and row["to_timepoint"] == second
            for row in rows
        )
    ]
    column_count = min(3, len(intervals))
    row_count = math.ceil(len(intervals) / column_count)
    fig, axes = plt.subplots(
        row_count,
        column_count,
        figsize=(13, max(5.8, 4.2 * row_count)),
        squeeze=False,
        constrained_layout=True,
    )
    flat_axes = list(axes.flat)
    for ax, pair in zip(flat_axes, intervals):
        selected = [
            row for row in rows if (row["from_timepoint"], row["to_timepoint"]) == pair
        ]
        ax.axvspan(-0.03, 0.5, color="#2563eb", alpha=0.035)
        ax.axvspan(0.5, 1.03, color="#16a34a", alpha=0.035)
        for row in selected:
            x = float(row["complaint_advance_rate"])
            y = float(row["general_symptom_change"])
            ax.scatter(
                x,
                y,
                color=CLASS_COLORS[row["alignment_class"]],
                edgecolor="#111827" if row.get("risk_worsened") else "white",
                linewidth=1.3,
                s=48,
            )
            ax.annotate(
                f"{row[entity_field]}-{row['outer_run_id']}",
                (x, y),
                xytext=(3, 3),
                textcoords="offset points",
                fontsize=5.5,
                alpha=0.82,
            )
        ax.axvline(0.5, color="#64748b", linestyle="--", linewidth=0.8)
        ax.axhline(0, color="#111827", linewidth=0.8)
        ax.set_xlim(-0.03, 1.03)
        ax.set_title(
            f"{label_display(pair[0])}→{label_display(pair[1])}\nn={len(selected)}"
        )
        ax.set_xlabel("Complaint advance rate")
        ax.grid(alpha=0.16)
    unused_axes = flat_axes[len(intervals) :]
    for ax in unused_axes:
        ax.axis("off")
    for ax in axes[:, 0]:
        ax.set_ylabel("General life-state change\n(negative = improvement)")
    handles = [
        Line2D([0], [0], marker="o", linestyle="", color=color, label=label)
        for label, color in CLASS_COLORS.items()
    ]
    if unused_axes:
        legend_ax = unused_axes[0]
        legend_ax.legend(
            handles=handles, loc="center", ncol=1, fontsize=9, frameon=False
        )
        legend_ax.text(
            0.5,
            0.18,
            "Upper-left = clearest concern\nLower-right = aligned progress\nBlack outline = risk item worsened",
            transform=legend_ax.transAxes,
            ha="center",
            va="center",
            fontsize=8,
            color="#475569",
        )
    else:
        fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=8)
    fig.suptitle(f"{title}: is complaint progress accompanied by symptom improvement?")
    return save_figure(
        fig, out_dir / "complaint_01_interval_progress_outcome_alignment.png"
    )
