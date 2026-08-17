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


def _draw_outer_summary(
    ax: Any, records: list[ExperimentRecord], labels: list[str], scale: str
) -> None:
    grouped: dict[str, list[ExperimentRecord]] = defaultdict(list)
    for record in records:
        grouped[record.group].append(record)
    for group, group_records in grouped.items():
        xs: list[int] = []
        centers: list[float] = []
        lowers: list[float] = []
        uppers: list[float] = []
        for index, label in enumerate(labels):
            values = [score_at(record, scale, label) for record in group_records]
            clean = [float(value) for value in values if value is not None]
            if not clean:
                continue
            center = mean(clean)
            if len(clean) > 1:
                half = (
                    float(t.ppf(0.975, len(clean) - 1))
                    * stdev(clean)
                    / math.sqrt(len(clean))
                )
            else:
                half = 0.0
            xs.append(index)
            centers.append(center)
            lowers.append(center - half)
            uppers.append(center + half)
        if centers:
            color = group_color(group)
            ax.plot(
                xs,
                centers,
                color=color,
                linewidth=3.2,
                alpha=0.95,
                label=f"{group} outer mean",
            )
            ax.fill_between(xs, lowers, uppers, color=color, alpha=0.12)


def plot_trajectory_with_ci(
    records: list[ExperimentRecord],
    labels: list[str],
    scale: str,
    out_dir: Path,
    title_prefix: str,
    *,
    error_bar: str,
    y_axis: str,
    show_repeat_points: bool,
    outer_summary: str,
) -> Path:
    severities = available_severities(records)
    fig, axes = _grid(1, len(severities), height=4.8)
    x_ticks = np.arange(len(labels))
    all_values: list[float] = []
    for column, severity in enumerate(severities):
        ax = axes[0][column]
        panel_records = [record for record in records if record.severity == severity]
        for record in panel_records:
            marker, linestyle = repeat_style(record.repeat_id)
            xs: list[int] = []
            ys: list[float] = []
            lower_error: list[float] = []
            upper_error: list[float] = []
            for index, label in enumerate(labels):
                stats = record.series.get(scale, {}).get(label)
                if not stats or stats.get("score") is None:
                    continue
                value = float(stats["score"])
                low, high = _measurement_error(stats, error_bar)
                xs.append(index)
                ys.append(value)
                lower_error.append(low)
                upper_error.append(high)
                all_values.extend([value - low, value + high])
                if show_repeat_points:
                    points = stats.get("values", [])
                    jitter = np.linspace(-0.10, 0.10, len(points)) if points else []
                    ax.scatter(
                        np.asarray([index] * len(points)) + jitter,
                        points,
                        s=12,
                        color=_series_color(record, panel_records),
                        alpha=0.18,
                        linewidths=0,
                        zorder=1,
                    )
            if not ys:
                continue
            alpha = 0.45 if outer_summary == "mean-ci" else 0.92
            ax.errorbar(
                xs,
                ys,
                yerr=(
                    np.asarray([lower_error, upper_error])
                    if error_bar != "none"
                    else None
                ),
                color=_series_color(record, panel_records),
                marker=marker,
                linestyle=linestyle,
                linewidth=1.8,
                markersize=5,
                capsize=3,
                alpha=alpha,
                zorder=2,
            )
        if outer_summary == "mean-ci":
            _draw_outer_summary(ax, panel_records, labels, scale)
        ax.set_title(severity)
        ax.set_xticks(
            x_ticks,
            [label_display(label) for label in labels],
            rotation=25 if len(labels) > 7 else 0,
        )
        ax.set_ylabel(f"{scale} primary score")
        ax.grid(alpha=0.22)
        _apply_y_axis(ax, scale, y_axis, all_values)
    handles = _series_legend(records)
    if outer_summary == "mean-ci":
        for group in sorted({record.group for record in records}):
            handles.append(
                Line2D(
                    [0],
                    [0],
                    color=group_color(group),
                    linewidth=3.2,
                    label=f"{group} outer mean ± 95% CI",
                )
            )
    if handles:
        fig.legend(
            handles=handles,
            loc="center left",
            bbox_to_anchor=(1.0, 0.5),
            ncol=1,
            frameon=False,
            fontsize=8,
        )
    error_label = {"ci95": "95% CI", "sd": "SD", "none": "no error bars"}[error_bar]
    fig.suptitle(
        _wrapped(f"{title_prefix}: {scale} trajectory with {error_label}"), fontsize=13
    )
    _footer(fig, records, error_bar, labels, outer_summary)
    return save_figure(
        fig, out_dir / f"presentation_01_{slugify(scale)}_trajectory_with_ci.png"
    )
