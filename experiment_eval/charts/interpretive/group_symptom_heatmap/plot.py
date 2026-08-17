"""Item-level and symptom-domain figures."""

from __future__ import annotations

import math

from pathlib import Path

from statistics import mean

from typing import Any

import numpy as np

from ....schema import group_sort_key, label_display, label_sort_key, scale_sort_key

from ....statistics import mean_ci95

from ...shared.plotting import plt, save_figure


def plot_group_symptom_heatmap(
    summary_rows: list[dict[str, Any]], out_dir: Path
) -> Path | None:
    rows = [
        row
        for row in summary_rows
        if row["stratum_type"] == "group"
        and row["metric_level"] == "symptom"
        and row["scale"] == "COMBINED_EQUAL_SCALE_WEIGHT"
    ]
    groups = sorted({row["stratum_value"] for row in rows}, key=group_sort_key)
    symptoms = list(dict.fromkeys(row["metric_id"] for row in rows))
    if not groups or not symptoms:
        return None
    matrix = np.full((len(groups), len(symptoms)), np.nan)
    labels = {}
    for row in rows:
        matrix[groups.index(row["stratum_value"]), symptoms.index(row["metric_id"])] = (
            float(row["mean_change"])
        )
        labels[row["metric_id"]] = row["metric_label_en"]
    limit = max(0.5, float(np.nanmax(np.abs(matrix))))
    fig, ax = plt.subplots(figsize=(15, 6.5), constrained_layout=True)
    image = ax.imshow(matrix, vmin=-limit, vmax=limit, cmap="RdYlGn_r", aspect="auto")
    for row_index in range(len(groups)):
        for column_index in range(len(symptoms)):
            value = matrix[row_index, column_index]
            if not math.isnan(value):
                ax.text(
                    column_index,
                    row_index,
                    f"{value:+.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                )
    ax.set_xticks(
        range(len(symptoms)),
        [labels[value] for value in symptoms],
        rotation=30,
        ha="right",
    )
    ax.set_yticks(range(len(groups)), groups)
    ax.set_title("Group-level simulated life-state change: session 20 − T0")
    fig.colorbar(image, ax=ax, label="Change (negative/green = improvement)")
    observed_ns = sorted(
        {
            int(row["n_outer_runs"])
            for row in rows
            if row.get("n_outer_runs") is not None
            and int(row["n_outer_runs"]) > 0
        }
    )
    fig.text(
        0.5,
        -0.02,
        "Descriptive only. Observed outer n/group="
        + (",".join(map(str, observed_ns)) or "unavailable")
        + "; the archive is a partial persona × group design, so group differences are not pure treatment effects.",
        ha="center",
        fontsize=8,
        color="#475569",
    )
    return save_figure(fig, out_dir / "life_state_03_group_symptom_change_heatmap.png")
