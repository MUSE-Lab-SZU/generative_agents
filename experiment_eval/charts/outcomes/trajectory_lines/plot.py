"""Trajectory figures for scale scores across observed timepoints."""

from __future__ import annotations

import math

from collections import defaultdict

from pathlib import Path

from statistics import mean, stdev

from typing import Any

import numpy as np

from scipy.stats import t

from ....loader import available_severities

from ....schema import (
    ExperimentRecord,
    group_color,
    label_display,
    repeat_style,
    slugify,
)

from ....statistics import labels_with_score, score_at

from ...shared.plotting import (
    add_measurement_footer as _footer,
    apply_score_y_axis as _apply_y_axis,
    figure_grid as _grid,
    lines_only_legend as _lines_only_legend,
    Line2D,
    measurement_error as _measurement_error,
    plt,
    save_figure,
    series_color as _series_color,
    series_legend as _series_legend,
    wrapped as _wrapped,
)


def plot_trajectory_lines_only(
    records: list[ExperimentRecord],
    labels: list[str],
    scale: str,
    out_dir: Path,
    title_prefix: str,
    *,
    y_axis: str,
) -> Path:
    """Plot only mean trajectories: no markers, error bars, or inner-repeat points."""
    severities = available_severities(records)
    fig, axes = _grid(1, len(severities), height=4.8)
    x_ticks = np.arange(len(labels))
    for column, severity in enumerate(severities):
        ax = axes[0][column]
        panel_records = [record for record in records if record.severity == severity]
        panel_values: list[float] = []
        for record in panel_records:
            available = labels_with_score(record, labels, scale)
            if not available:
                continue
            xs = [labels.index(label) for label in available]
            ys = [float(score_at(record, scale, label)) for label in available]
            _, linestyle = repeat_style(record.repeat_id)
            ax.plot(
                xs,
                ys,
                color=_series_color(record, panel_records),
                linestyle=linestyle,
                linewidth=2.2,
                alpha=0.94,
            )
            panel_values.extend(ys)
        ax.set_title(severity)
        ax.set_xticks(
            x_ticks,
            [label_display(label) for label in labels],
            rotation=25 if len(labels) > 7 else 0,
        )
        ax.set_ylabel(f"{scale} primary score")
        ax.grid(alpha=0.20)
        _apply_y_axis(ax, scale, y_axis, panel_values)
    handles = _lines_only_legend(records)
    if handles:
        fig.legend(
            handles=handles,
            loc="center left",
            bbox_to_anchor=(1.0, 0.5),
            ncol=1,
            frameon=False,
            fontsize=8,
        )
    fig.suptitle(
        _wrapped(f"{title_prefix}: {scale} trajectory — lines only"), fontsize=13
    )
    _footer(fig, records, "none (lines only)", labels)
    return save_figure(
        fig, out_dir / f"presentation_00_{slugify(scale)}_trajectory_lines_only.png"
    )
