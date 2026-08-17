"""Leave-one-persona-out sensitivity chart."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from ....schema import group_sort_key, kbd_color, scale_sort_key
from ...shared.persona import _draw_clipped_interval
from ...shared.plotting import plt, save_figure


def plot_leave_one_persona_out(
    rows: list[dict[str, Any]],
    out_dir: Path,
    *,
    first_group: str,
    second_group: str,
) -> Path:
    scales = sorted({row["scale"] for row in rows}, key=scale_sort_key)
    order = [
        "NONE",
        *sorted(
            {
                row["omitted_persona"]
                for row in rows
                if row["omitted_persona"] != "NONE"
            },
            key=group_sort_key,
        ),
    ]
    labels = ["All personas", *[f"Without {persona}" for persona in order[1:]]]
    fig, axes = plt.subplots(
        1, len(scales), figsize=(12.8, 4.8), squeeze=False, constrained_layout=True
    )
    for column, scale in enumerate(scales):
        ax = axes[0][column]
        scale_rows = {
            row["omitted_persona"]: row for row in rows if row["scale"] == scale
        }
        finite_bounds = [
            abs(float(value))
            for row in scale_rows.values()
            for value in (row.get("ci95_lower"), row.get("ci95_upper"))
            if value is not None and math.isfinite(float(value))
        ]
        bound = max(finite_bounds, default=1.0) * 1.08
        limits = (-bound, bound)
        for y, omitted in enumerate(order):
            row = scale_rows.get(omitted)
            if row is None or row.get("adjusted_endpoint_difference") is None:
                continue
            estimate = float(row["adjusted_endpoint_difference"])
            lower = (
                float(row["ci95_lower"]) if row.get("ci95_lower") is not None else None
            )
            upper = (
                float(row["ci95_upper"]) if row.get("ci95_upper") is not None else None
            )
            color = "#111827" if omitted == "NONE" else kbd_color(omitted)
            _draw_clipped_interval(
                ax,
                estimate=estimate,
                lower=lower,
                upper=upper,
                y=float(y),
                color=color,
                limits=limits,
            )
        ax.axvline(0, color="#111827", linewidth=0.9)
        ax.set_xlim(*limits)
        ax.set_yticks(range(len(labels)), labels)
        ax.invert_yaxis()
        ax.set_title(scale)
        ax.set_xlabel("Baseline-adjusted endpoint difference")
        ax.grid(axis="x", alpha=0.2)
    fig.suptitle(
        f"Leave-one-persona-out sensitivity: {first_group} − {second_group}",
        fontsize=14,
    )
    fig.text(
        0.5,
        -0.015,
        "Endpoint ~ group + baseline + persona fixed effects. This checks influence of three selected cases; "
        "it is not a jackknife estimate for a sampled persona population.",
        ha="center",
        fontsize=8,
        color="#475569",
    )
    return save_figure(
        fig, out_dir / "figure_s01_0718_g1_vs_g9_leave_one_persona_out.png"
    )
