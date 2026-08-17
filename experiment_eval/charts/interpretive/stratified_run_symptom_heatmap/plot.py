"""Single-run symptom heatmap for stratified interpretation."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from ....schema import group_sort_key
from ...shared.plotting import plt, save_figure
from ...shared.stratified import _run_order


def plot_run_symptom_heatmap(
    symptom_changes: list[dict[str, Any]], entity_field: str, title: str, out_dir: Path
) -> Path | None:
    rows = [
        row for row in symptom_changes if row["scale"] == "COMBINED_EQUAL_SCALE_WEIGHT"
    ]
    run_order = _run_order(rows, entity_field)
    symptoms = list(dict.fromkeys(str(row["symptom_id"]) for row in rows))
    if not run_order or not symptoms:
        return None
    matrix = np.full((len(run_order), len(symptoms)), np.nan)
    labels: dict[str, str] = {}
    stable_to_row = {
        stable_id: index
        for index, (_entity, _repeat, stable_id) in enumerate(run_order)
    }
    for row in rows:
        matrix[stable_to_row[row["stable_id"]], symptoms.index(row["symptom_id"])] = (
            float(row["change"])
        )
        labels[row["symptom_id"]] = row["symptom_label_en"]
    limit = max(0.5, float(np.nanmax(np.abs(matrix))))
    fig, ax = plt.subplots(
        figsize=(15, max(6.5, len(run_order) * 0.45 + 2.5)), constrained_layout=True
    )
    image = ax.imshow(matrix, vmin=-limit, vmax=limit, cmap="RdYlGn_r", aspect="auto")
    for row_index in range(len(run_order)):
        for column_index in range(len(symptoms)):
            value = matrix[row_index, column_index]
            if not math.isnan(value):
                ax.text(
                    column_index,
                    row_index,
                    f"{value:+.2f}",
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
    ax.set_yticks(
        range(len(run_order)),
        [f"{entity}-{repeat}" for entity, repeat, _stable_id in run_order],
        fontsize=8,
    )
    ax.set_title(f"{title}: each row is one independent simulation run")
    fig.colorbar(image, ax=ax, label="session 20 − T0 (negative/green = improvement)")
    fig.text(
        0.5,
        -0.018,
        "No averaging across personas or conditions. Every outer run remains separate; each snapshot averages only its measurement repeats.",
        ha="center",
        fontsize=8,
        color="#475569",
    )
    return save_figure(fig, out_dir / "life_state_01_single_run_symptom_heatmap.png")
