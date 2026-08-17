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


def plot_complaint_run_alignment(
    rows: list[dict[str, Any]], entity_field: str, title: str, out_dir: Path
) -> Path | None:
    if not rows:
        return None
    fig, ax = plt.subplots(figsize=(9, 7), constrained_layout=True)
    ax.axvspan(-0.03, 0.5, color="#2563eb", alpha=0.035)
    ax.axvspan(0.5, 1.03, color="#16a34a", alpha=0.035)
    for row in rows:
        x, y = float(row["complaint_advance_rate"]), float(
            row["general_symptom_change"]
        )
        ax.scatter(
            x,
            y,
            color=CLASS_COLORS[row["alignment_class"]],
            edgecolor="#111827" if row.get("risk_worsened") else "white",
            linewidth=1.4,
            s=72,
        )
        ax.annotate(
            f"{row[entity_field]}-{row['outer_run_id']}",
            (x, y),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
        )
    ax.axvline(0.5, color="#64748b", linestyle="--", linewidth=0.9)
    ax.axhline(0, color="#111827", linewidth=0.9)
    ax.set_xlim(-0.03, 1.03)
    ax.set_xlabel("Complaint reflection advance rate across available intervals")
    ax.set_ylabel("T0→session 20 general life-state change\n(negative = improvement)")
    ax.set_title(f"{title}: run-level complaint/outcome alignment")
    ax.grid(alpha=0.18)
    return save_figure(fig, out_dir / "complaint_02_run_progress_outcome_alignment.png")
