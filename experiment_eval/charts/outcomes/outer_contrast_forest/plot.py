"""Endpoint-change, rebound, and outer-run group-comparison figures."""

from __future__ import annotations

import math

from pathlib import Path

import numpy as np

from ....loader import available_severities

from ....schema import ExperimentRecord, group_color, label_display, slugify

from ....statistics import (
    baseline_adjusted_endpoint_contrasts,
    cross_scale_convergence,
    delta_ci,
    group_endpoint_contrasts,
    group_time_contrasts,
    trajectory_metrics,
)

from ...shared.plotting import (
    add_measurement_footer as _footer,
    figure_grid as _grid,
    Line2D,
    plt,
    save_figure,
    series_color as _series_color,
    wrapped as _wrapped,
)


def plot_outer_contrast_forest(
    records: list[ExperimentRecord],
    labels: list[str],
    scales: list[str],
    out_dir: Path,
    title_prefix: str,
) -> Path | None:
    adjusted_rows = baseline_adjusted_endpoint_contrasts(records, labels, scales)
    change_rows = group_endpoint_contrasts(records, labels, scales)
    rows = []
    for adjusted in adjusted_rows:
        if adjusted.get("estimate") is None:
            continue
        change = next(
            (
                row
                for row in change_rows
                if row["scale"] == adjusted["scale"]
                and row["contrast"] == adjusted["contrast"]
            ),
            None,
        )
        rows.append((adjusted, change))
    if not rows:
        return None
    fig, ax = plt.subplots(
        figsize=(8.6, max(3.8, 0.75 * len(rows) + 2.2)), constrained_layout=True
    )
    y_positions = np.arange(len(rows))
    for y, (row, change) in zip(y_positions, rows):
        estimate = row["estimate"]
        lower, upper = row.get("lower"), row.get("upper")
        error = None
        if lower is not None and upper is not None:
            error = np.asarray([[estimate - lower], [upper - estimate]])
        ax.errorbar(
            estimate,
            y,
            xerr=error,
            fmt="o",
            color=group_color(row["first_group"]),
            capsize=4,
            markersize=7,
        )
        effect = change["hedges_g"].get("estimate") if change else None
        effect_text = (
            "NA" if effect is None or not math.isfinite(effect) else f"{effect:.2f}"
        )
        ax.annotate(
            f"g={effect_text}",
            (estimate, y),
            xytext=(7, 6),
            textcoords="offset points",
            fontsize=8,
        )
    ax.axvline(0, color="#111827", linewidth=0.9)
    ax.set_yticks(
        y_positions, [f"{row['scale']}: {row['contrast']}" for row, _ in rows]
    )
    ax.invert_yaxis()
    ax.set_xlabel("Baseline-adjusted observed endpoint difference (first − second)")
    ax.set_title(f"{title_prefix}: exploratory baseline-adjusted endpoint contrasts")
    ax.grid(axis="x", alpha=0.22)
    fig.text(
        0.5,
        -0.02,
        "Pairwise ANCOVA 95% CI: endpoint ~ group + baseline. Hedges' g annotation uses unadjusted outer-run change SD. "
        "Exploratory small-n comparison; negative adjusted difference favors the first group for symptom scores.",
        ha="center",
        fontsize=8,
        color="#475569",
    )
    return save_figure(fig, out_dir / "core_01_outer_run_contrast_forest.png")
