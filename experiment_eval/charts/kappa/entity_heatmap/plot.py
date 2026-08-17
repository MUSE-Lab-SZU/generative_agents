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


def plot_entity_kappa_heatmap(
    summary_rows: list[dict[str, Any]], entity_type: str, title: str, out_dir: Path
) -> Path | None:
    rows = [
        row
        for row in summary_rows
        if row["stratum_type"] == entity_type
        and row["weights"] == "quadratic"
        and row.get("kappa") is not None
    ]
    entities = sorted({str(row["stratum_value"]) for row in rows}, key=group_sort_key)
    scales = sorted({str(row["scale"]) for row in rows}, key=scale_sort_key)
    if not entities or not scales:
        return None
    matrix = np.full((len(entities), len(scales)), np.nan)
    for row in rows:
        matrix[entities.index(row["stratum_value"]), scales.index(row["scale"])] = (
            float(row["kappa"])
        )
    fig, ax = plt.subplots(
        figsize=(6.5, max(4.5, len(entities) * 0.65 + 2)), constrained_layout=True
    )
    image = ax.imshow(matrix, vmin=0, vmax=1, cmap="YlGnBu", aspect="auto")
    for row_index in range(len(entities)):
        for column_index in range(len(scales)):
            value = matrix[row_index, column_index]
            if not math.isnan(value):
                ax.text(
                    column_index, row_index, f"{value:.3f}", ha="center", va="center"
                )
    ax.set_xticks(range(len(scales)), scales)
    ax.set_yticks(range(len(entities)), entities)
    ax.set_title(f"{title}: entity-specific quadratic weighted Kappa")
    fig.colorbar(image, ax=ax, label="Weighted κ")
    fig.text(
        0.5,
        -0.02,
        "Each entity has two independent outer runs; values are descriptive and no entity-level CI is forced.",
        ha="center",
        fontsize=8,
        color="#475569",
    )
    return save_figure(fig, out_dir / "weighted_kappa_03_entity_scale_heatmap.png")
