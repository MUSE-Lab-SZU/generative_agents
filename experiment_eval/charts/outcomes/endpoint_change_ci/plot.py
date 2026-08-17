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


def plot_endpoint_delta_with_ci(
    records: list[ExperimentRecord],
    labels: list[str],
    scales: list[str],
    out_dir: Path,
    title_prefix: str,
) -> Path:
    severities = available_severities(records)
    fig, axes = _grid(len(scales), len(severities), height=3.9)
    for row, scale in enumerate(scales):
        for column, severity in enumerate(severities):
            ax = axes[row][column]
            panel_records = [
                record for record in records if record.severity == severity
            ]
            for index, record in enumerate(panel_records):
                interval = delta_ci(record, labels, scale)
                estimate = interval.get("estimate")
                if estimate is None:
                    continue
                lower, upper = interval.get("lower"), interval.get("upper")
                crosses = (
                    lower is not None and upper is not None and lower <= 0 <= upper
                )
                yerr = None
                if lower is not None and upper is not None:
                    yerr = np.asarray([[estimate - lower], [upper - estimate]])
                ax.errorbar(
                    [index],
                    [estimate],
                    yerr=yerr,
                    marker="o",
                    markersize=7,
                    markerfacecolor=(
                        "white" if crosses else _series_color(record, panel_records)
                    ),
                    markeredgecolor=_series_color(record, panel_records),
                    color=_series_color(record, panel_records),
                    capsize=4,
                    linewidth=1.6,
                )
                if crosses:
                    ax.annotate(
                        "CI crosses 0",
                        (index, estimate),
                        xytext=(0, 8),
                        textcoords="offset points",
                        ha="center",
                        fontsize=7,
                    )
            ax.axhline(0, color="#111827", linewidth=0.9)
            ax.set_xticks(
                range(len(panel_records)),
                [record.plot_label for record in panel_records],
                rotation=25,
                ha="right",
            )
            ax.set_title(f"{scale} — {severity}")
            ax.set_ylabel("Observed endpoint − baseline")
            ax.grid(axis="y", alpha=0.22)
    fig.suptitle(
        _wrapped(f"{title_prefix}: observed-endpoint change with 95% difference CI"),
        fontsize=13,
    )
    ns = sorted({record.expected_repeats for record in records})
    fig.text(
        0.5,
        -0.025,
        f"n={','.join(map(str, ns))} repeats/snapshot; observed endpoint: POST > NOW > last available (not inferred follow-up). "
        "Error=95% difference CI; Monte Carlo measurement uncertainty only, not patient or outer-experiment uncertainty. "
        "Default Welch unless metadata explicitly declares pairing.",
        ha="center",
        va="top",
        fontsize=8,
        color="#475569",
    )
    return save_figure(fig, out_dir / "presentation_02_endpoint_delta_with_ci.png")
