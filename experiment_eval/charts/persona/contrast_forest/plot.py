"""Persona-specific contrast forest chart."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ....schema import group_sort_key, kbd_color, scale_sort_key
from ...shared.persona import _draw_clipped_interval, _interval_text
from ...shared.plotting import plt, save_figure


def plot_persona_contrast_forest(
    rows: list[dict[str, Any]],
    out_dir: Path,
    *,
    first_group: str,
    second_group: str,
) -> Path:
    """Per-persona G1−G9 adjusted differences and standardized effects."""
    scales = sorted({row["scale"] for row in rows}, key=scale_sort_key)
    personas = sorted({row["persona"] for row in rows}, key=group_sort_key)
    fig, axes = plt.subplots(
        len(scales),
        2,
        figsize=(13.8, max(7.2, 3.8 * len(scales))),
        squeeze=False,
        constrained_layout=True,
    )
    for scale_index, scale in enumerate(scales):
        scale_rows = {row["persona"]: row for row in rows if row["scale"] == scale}
        y_positions = np.arange(len(personas))
        adjusted_limits = (-25.0, 25.0) if scale == "PHQ-9" else (-65.0, 65.0)
        specifications = [
            (
                axes[scale_index][0],
                "adjusted_endpoint_difference",
                "adjusted_ci95_lower",
                "adjusted_ci95_upper",
                adjusted_limits,
                f"{scale}: baseline-adjusted endpoint difference",
                "Agent score difference",
            ),
            (
                axes[scale_index][1],
                "hedges_g",
                "hedges_g_ci95_lower",
                "hedges_g_ci95_upper",
                (-4.0, 4.0),
                f"{scale}: Hedges' g of endpoint change",
                "Standardized mean difference",
            ),
        ]
        for (
            ax,
            estimate_key,
            lower_key,
            upper_key,
            limits,
            title,
            xlabel,
        ) in specifications:
            for y, persona in zip(y_positions, personas):
                row = scale_rows.get(persona)
                if row is None or row.get(estimate_key) is None:
                    continue
                estimate = float(row[estimate_key])
                lower = (
                    float(row[lower_key]) if row.get(lower_key) is not None else None
                )
                upper = (
                    float(row[upper_key]) if row.get(upper_key) is not None else None
                )
                _draw_clipped_interval(
                    ax,
                    estimate=estimate,
                    lower=lower,
                    upper=upper,
                    y=float(y),
                    color=kbd_color(persona),
                    limits=limits,
                )
                ax.text(
                    1.02,
                    y,
                    _interval_text(estimate, lower, upper),
                    transform=ax.get_yaxis_transform(),
                    ha="left",
                    va="center",
                    fontsize=7.5,
                )
            ax.axvline(0, color="#111827", linewidth=0.9)
            ax.set_xlim(*limits)
            ax.set_yticks(y_positions, personas)
            ax.invert_yaxis()
            ax.set_title(title, fontsize=10.5)
            ax.set_xlabel(xlabel)
            ax.grid(axis="x", alpha=0.2)
    fig.suptitle(
        f"Persona-specific exploratory contrast: {first_group} − {second_group}",
        fontsize=14,
    )
    fig.text(
        0.5,
        -0.012,
        "Negative values favor the first group for symptom scores. Arrows denote 95% CIs extending beyond the "
        "plot window; full intervals are printed and exported. Adjusted CI: ANCOVA t interval; g CI: noncentral t. "
        "The 0718 endpoint is legacy immediate POST, not delayed follow-up.",
        ha="center",
        fontsize=8,
        color="#475569",
    )
    return save_figure(fig, out_dir / "figure_02_0718_g1_vs_g9_persona_forest.png")
