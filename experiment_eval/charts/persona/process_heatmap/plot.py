"""Persona-by-group process heatmap."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ....schema import group_sort_key
from ...shared.persona import _heatmap
from ...shared.plotting import plt, save_figure


def plot_process_heatmap(summary_rows: list[dict[str, Any]], out_dir: Path) -> Path:
    personas = sorted({row["persona"] for row in summary_rows}, key=group_sort_key)
    groups = sorted({row["group"] for row in summary_rows}, key=group_sort_key)
    by_cell = {(row["persona"], row["group"]): row for row in summary_rows}
    definitions = [
        ("mean_completed_consultation_meetings", "Doctor consultations", "meetings"),
        ("mean_total_turns", "Recorded doctor–patient turns", "turns"),
        ("mean_environment_tasks", "Environment tasks", "tasks"),
        ("process_raw_available_rate", "Raw process availability", "proportion"),
    ]
    fig, axes = plt.subplots(
        2, 2, figsize=(13.8, 8.0), squeeze=False, constrained_layout=True
    )
    for ax, (field, title, colorbar_label) in zip(axes.flat, definitions):
        matrix = np.asarray(
            [
                [
                    (
                        (by_cell.get((persona, group)) or {}).get(field, np.nan)
                        if (by_cell.get((persona, group)) or {}).get(field) is not None
                        else np.nan
                    )
                    for group in groups
                ]
                for persona in personas
            ],
            dtype=float,
        )
        finite = matrix[np.isfinite(matrix)]
        vmin = 0.0
        vmax = (
            1.0
            if field.endswith("_rate")
            else (float(finite.max()) if finite.size else 1.0)
        )

        def formatter(value: float, persona: str, group: str) -> str:
            if field.endswith("_rate"):
                return f"{value:.0%}"
            return f"{value:.1f}"

        _heatmap(
            ax,
            matrix,
            personas=personas,
            groups=groups,
            title=title,
            cmap_name="YlGnBu",
            vmin=vmin,
            vmax=max(vmax, 1e-9),
            formatter=formatter,
            colorbar_label=colorbar_label,
        )
    fig.suptitle("Persona × group process-data summary", fontsize=14)
    fig.text(
        0.5,
        -0.02,
        "NA means the archived extractor input is absent, not zero. Zero consultations for resident-chat groups "
        "describes the group design and does not mean zero social interaction; resident-chat dose is not yet normalized.",
        ha="center",
        fontsize=8,
        color="#475569",
    )
    return save_figure(
        fig, out_dir / "figure_03b_0718_persona_group_process_heatmap.png"
    )
