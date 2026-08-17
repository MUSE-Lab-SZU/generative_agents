"""Persona-by-group outcome heatmap."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ....schema import group_sort_key
from ...shared.persona import _heatmap
from ...shared.plotting import plt, save_figure


def plot_outcome_heatmap(summary_rows: list[dict[str, Any]], out_dir: Path) -> Path:
    personas = sorted({row["persona"] for row in summary_rows}, key=group_sort_key)
    groups = sorted({row["group"] for row in summary_rows}, key=group_sort_key)
    by_cell = {(row["persona"], row["group"]): row for row in summary_rows}
    definitions = [
        (
            "phq9_mean_endpoint_change",
            "PHQ-9 endpoint change",
            "RdBu_r",
            "Agent score change",
        ),
        (
            "bdi2_mean_endpoint_change",
            "BDI-II endpoint change",
            "RdBu_r",
            "Agent score change",
        ),
        (
            "phq_bdi_direction_agreement_rate",
            "PHQ-9 / BDI-II direction agreement",
            "Blues",
            "Proportion",
        ),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(16.2, 4.6), constrained_layout=True)
    for ax, (field, title, cmap, colorbar_label) in zip(axes, definitions):
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
        if field.endswith("agreement_rate"):
            vmin, vmax = 0.0, 1.0

            def formatter(value: float, persona: str, group: str) -> str:
                row = by_cell[(persona, group)]
                return f"{value:.0%}\nn={row['n_direction_pairs']}"

        else:
            finite = np.abs(matrix[np.isfinite(matrix)])
            bound = max(float(finite.max()), 1.0) if finite.size else 1.0
            vmin, vmax = -bound, bound

            def formatter(value: float, persona: str, group: str) -> str:
                row = by_cell[(persona, group)]
                return f"{value:+.1f}\nn={row['n_outer_runs']}"

        _heatmap(
            ax,
            matrix,
            personas=personas,
            groups=groups,
            title=title,
            cmap_name=cmap,
            vmin=vmin,
            vmax=vmax,
            formatter=formatter,
            colorbar_label=colorbar_label,
        )
    fig.suptitle("Persona × group outcome summary", fontsize=14)
    observed_ns = sorted(
        {
            int(row["n_outer_runs"])
            for row in summary_rows
            if row.get("n_outer_runs") is not None
            and int(row["n_outer_runs"]) > 0
        }
    )
    fig.text(
        0.5,
        -0.02,
        "Cells are outer-run means; negative score change indicates improvement. "
        "Direction agreement is descriptive; observed outer n/cell="
        + (",".join(map(str, observed_ns)) or "unavailable")
        + ".",
        ha="center",
        fontsize=8,
        color="#475569",
    )
    return save_figure(
        fig, out_dir / "figure_03a_0718_persona_group_outcome_heatmap.png"
    )
