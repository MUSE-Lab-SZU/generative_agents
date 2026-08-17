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


def plot_best_change_and_rebound(
    records: list[ExperimentRecord],
    labels: list[str],
    scales: list[str],
    out_dir: Path,
    title_prefix: str,
) -> Path:
    severities = available_severities(records)
    fig, axes = _grid(len(scales), len(severities), height=4.0)
    definitions = [
        ("best_change_from_baseline", "Baseline→nadir", "#2563eb"),
        ("rebound_from_nadir", "Nadir→endpoint", "#dc2626"),
        ("endpoint_change", "Baseline→endpoint", "#64748b"),
    ]
    for row, scale in enumerate(scales):
        for column, severity in enumerate(severities):
            ax = axes[row][column]
            panel_records = [
                record for record in records if record.severity == severity
            ]
            x = np.arange(len(panel_records))
            width = 0.24
            for metric_index, (field, label, color) in enumerate(definitions):
                values = [
                    trajectory_metrics(record, labels, scale).get(field)
                    for record in panel_records
                ]
                ax.bar(
                    x + (metric_index - 1) * width,
                    values,
                    width=width,
                    label=label,
                    color=color,
                    alpha=0.86,
                )
            for index, record in enumerate(panel_records):
                metrics = trajectory_metrics(record, labels, scale)
                best = metrics.get("best_change_from_baseline")
                rebound = metrics.get("rebound_from_nadir")
                if (
                    best is not None
                    and best < 0
                    and rebound is not None
                    and rebound > 0
                ):
                    ax.annotate(
                        f"rebound after {label_display(metrics['best_timepoint'])}",
                        (index, rebound),
                        xytext=(0, 6),
                        textcoords="offset points",
                        ha="center",
                        va="bottom",
                        fontsize=7,
                    )
            ax.axhline(0, color="#111827", linewidth=0.8)
            ax.set_xticks(
                x,
                [record.plot_label for record in panel_records],
                rotation=25,
                ha="right",
            )
            ax.set_title(f"{scale} — {severity}")
            ax.set_ylabel("Score change")
            ax.grid(axis="y", alpha=0.22)
            ax.margins(y=0.18)
    handles = [
        Line2D([0], [0], color=color, linewidth=8, label=label)
        for _, label, color in definitions
    ]
    fig.legend(
        handles=handles,
        loc="center left",
        bbox_to_anchor=(1.0, 0.5),
        ncol=1,
        frameon=False,
    )
    fig.suptitle(
        _wrapped(f"{title_prefix}: best change, rebound, and endpoint change"),
        fontsize=13,
    )
    _footer(fig, records, "none (point summaries)", labels)
    return save_figure(fig, out_dir / "presentation_03_best_change_and_rebound.png")
