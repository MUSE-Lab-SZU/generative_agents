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


def plot_item_life_state_change(
    summary_rows: list[dict[str, Any]], out_dir: Path
) -> Path | None:
    rows = [
        row
        for row in summary_rows
        if row["stratum_type"] == "overall" and row["metric_level"] == "item"
    ]
    scales = sorted({row["scale"] for row in rows}, key=scale_sort_key)
    if not rows:
        return None
    fig, axes = plt.subplots(
        1, len(scales), figsize=(14, 8.5), squeeze=False, constrained_layout=True
    )
    for scale_index, scale in enumerate(scales):
        ax = axes[0][scale_index]
        selected = sorted(
            (row for row in rows if row["scale"] == scale),
            key=lambda row: int(str(row["metric_id"]).rsplit("_", 1)[-1]),
        )
        y = np.arange(len(selected))
        values = [float(row["mean_change"]) for row in selected]
        lower = [row.get("ci95_lower") for row in selected]
        upper = [row.get("ci95_upper") for row in selected]
        errors = np.asarray(
            [
                [
                    value - float(lo) if lo is not None else 0
                    for value, lo in zip(values, lower)
                ],
                [
                    float(hi) - value if hi is not None else 0
                    for value, hi in zip(values, upper)
                ],
            ]
        )
        colors = [
            "#16a34a" if value < 0 else "#dc2626" if value > 0 else "#64748b"
            for value in values
        ]
        ax.errorbar(
            values, y, xerr=errors, fmt="none", ecolor="#94a3b8", capsize=2, linewidth=1
        )
        ax.scatter(values, y, c=colors, s=36, zorder=3)
        ax.axvline(0, color="#111827", linewidth=0.8)
        ax.set_yticks(
            y,
            [
                f"I{str(row['metric_id']).rsplit('_', 1)[-1]} {row['metric_label_en']}"
                for row in selected
            ],
        )
        ax.invert_yaxis()
        ax.set_title(scale)
        ax.set_xlabel("Endpoint − T0 item score")
        ax.grid(axis="x", alpha=0.2)
    fig.suptitle("Which simulated life-state items changed from T0 to session 20?")
    fig.text(
        0.5,
        -0.012,
        "Green/negative = fewer reported symptoms (improvement); red/positive = worsening. "
        "Points are means across 26 independent outer runs; bars are outer-run t intervals.",
        ha="center",
        fontsize=8,
        color="#475569",
    )
    return save_figure(fig, out_dir / "life_state_01_item_change_forest.png")
