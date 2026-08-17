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


def plot_replicate_agreement(
    agreement_rows: list[dict[str, Any]], entity_field: str, title: str, out_dir: Path
) -> Path | None:
    entities = sorted(
        {str(row[entity_field]) for row in agreement_rows}, key=group_sort_key
    )
    symptoms = list(dict.fromkeys(str(row["symptom_id"]) for row in agreement_rows))
    if not entities or not symptoms:
        return None
    matrix = np.full((len(entities), len(symptoms)), np.nan)
    annotations = [["" for _ in symptoms] for _ in entities]
    labels: dict[str, str] = {}
    for row in agreement_rows:
        row_index = entities.index(str(row[entity_field]))
        column_index = symptoms.index(str(row["symptom_id"]))
        matrix[row_index, column_index] = AGREEMENT_CODES[row["agreement"]]
        annotations[row_index][column_index] = (
            f"{row.get('R01_change', 'NA'):+.2f}/{row.get('R02_change', 'NA'):+.2f}"
            if row.get("R01_change") is not None and row.get("R02_change") is not None
            else "NA"
        )
        labels[row["symptom_id"]] = row["symptom_label_en"]
    from matplotlib.colors import BoundaryNorm, ListedColormap

    cmap = ListedColormap(["#16a34a", "#eab308", "#f59e0b", "#dc2626"])
    norm = BoundaryNorm([-1.1, -0.5, 0.2, 0.7, 1.1], cmap.N)
    fig, ax = plt.subplots(
        figsize=(15, max(5.0, len(entities) * 0.65 + 2.5)), constrained_layout=True
    )
    ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto")
    for row_index in range(len(entities)):
        for column_index in range(len(symptoms)):
            if annotations[row_index][column_index]:
                ax.text(
                    column_index,
                    row_index,
                    annotations[row_index][column_index],
                    ha="center",
                    va="center",
                    fontsize=7,
                )
    ax.set_xticks(
        range(len(symptoms)),
        [labels[value] for value in symptoms],
        rotation=30,
        ha="right",
    )
    ax.set_yticks(range(len(entities)), entities)
    ax.set_title(f"{title}: can the R01 finding be reproduced in R02?")
    handles = [
        Line2D(
            [0], [0], marker="s", linestyle="", color=color, markersize=9, label=label
        )
        for color, label in (
            ("#16a34a", "both improve"),
            ("#dc2626", "both worsen"),
            ("#eab308", "both stable"),
            ("#f59e0b", "mixed/one stable"),
        )
    ]
    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
        ncol=4,
        fontsize=8,
    )
    fig.text(
        0.5,
        -0.015,
        "Cell text = R01 change / R02 change. This is reproducibility, not an average effect.",
        ha="center",
        fontsize=8,
        color="#475569",
    )
    return save_figure(fig, out_dir / "life_state_02_r01_r02_reproducibility.png")
