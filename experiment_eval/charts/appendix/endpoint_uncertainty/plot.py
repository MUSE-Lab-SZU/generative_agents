"""Backward-compatible appendix figures."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ....loader import available_severities

from ....schema import ExperimentRecord, label_display, repeat_style, slugify

from ....statistics import (
    endpoint_label_for_record,
    labels_with_score,
    score_at,
    stat_at,
    trajectory_metrics,
)

from ...shared.plotting import (
    add_measurement_footer as _footer,
    figure_grid as _grid,
    heatmap_text_color as _text_color,
    Line2D,
    plt,
    save_figure,
    series_color as _series_color,
    series_legend as _series_legend,
    wrapped as _wrapped,
)


def _appendix_endpoint_uncertainty(
    records: list[ExperimentRecord],
    labels: list[str],
    scales: list[str],
    out_dir: Path,
    title_prefix: str,
    index: int,
) -> Path:
    severities = available_severities(records)
    fig, axes = _grid(len(scales), len(severities), height=3.7)
    for row, scale in enumerate(scales):
        for column, severity in enumerate(severities):
            ax = axes[row][column]
            panel = [record for record in records if record.severity == severity]
            endpoints = [
                endpoint_label_for_record(record, labels, scale) for record in panel
            ]
            scores = [
                score_at(record, scale, endpoint)
                for record, endpoint in zip(panel, endpoints)
            ]
            sds = [
                stat_at(record, scale, endpoint, "sample_sd")
                for record, endpoint in zip(panel, endpoints)
            ]
            ax.bar(
                range(len(panel)),
                scores,
                yerr=sds,
                capsize=4,
                color=[_series_color(record, panel) for record in panel],
                alpha=0.82,
            )
            ax.set_xticks(
                range(len(panel)),
                [record.plot_label for record in panel],
                rotation=25,
                ha="right",
            )
            ax.set_title(f"{scale} endpoint — {severity}")
            ax.set_ylabel("Primary score ± within-snapshot SD")
            ax.grid(axis="y", alpha=0.22)
    fig.suptitle(_wrapped(f"{title_prefix}: endpoint repeat uncertainty"), fontsize=13)
    _footer(fig, records, "sd", labels)
    return save_figure(
        fig, out_dir / f"{index:02d}_endpoint_repeat_uncertainty_by_severity.png"
    )
