"""Reusable renderers that compose individual chart implementations into CLI families."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..charts.appendix.delta_heatmap.plot import _appendix_delta_heatmap
from ..charts.appendix.delta_trajectory.plot import _appendix_delta_trajectory
from ..charts.appendix.endpoint_uncertainty.plot import _appendix_endpoint_uncertainty
from ..charts.appendix.final_delta.plot import _appendix_final_delta
from ..charts.appendix.trajectory.plot import _appendix_trajectory
from ..charts.appendix.volatility.plot import _appendix_volatility
from ..charts.outcomes.change_rebound import plot_best_change_and_rebound
from ..charts.outcomes.cross_scale_convergence import (
    plot_cross_scale_concurrent_validity,
    plot_cross_scale_convergence,
)
from ..charts.outcomes.endpoint_change_ci import plot_endpoint_delta_with_ci
from ..charts.outcomes.endpoint_waterfall import plot_endpoint_waterfall
from ..charts.outcomes.group_time_contrasts import plot_group_time_contrasts
from ..charts.outcomes.outer_contrast_forest import plot_outer_contrast_forest
from ..charts.outcomes.trajectory_ci import plot_trajectory_with_ci
from ..charts.outcomes.trajectory_lines import plot_trajectory_lines_only
from ..charts.process.delivery import plot_process_delivery
from ..charts.reliability.measurement_heatmap import (
    plot_measurement_reliability_heatmaps,
)
from ..charts.reliability.measurement_icc import plot_measurement_error, plot_measurement_icc
from ..schema import ExperimentRecord


def render_lines_only(
    records: list[ExperimentRecord],
    labels: list[str],
    scales: list[str],
    out_dir: Path,
    title_prefix: str,
    *,
    y_axis: str,
) -> list[Path]:
    return [
        plot_trajectory_lines_only(
            records, labels, scale, out_dir, title_prefix, y_axis=y_axis
        )
        for scale in scales
    ]


def render_presentation(
    records: list[ExperimentRecord],
    labels: list[str],
    scales: list[str],
    out_dir: Path,
    title_prefix: str,
    *,
    error_bar: str,
    y_axis: str,
    show_repeat_points: bool,
    outer_summary: str,
) -> list[Path]:
    paths = render_lines_only(
        records, labels, scales, out_dir, title_prefix, y_axis=y_axis
    )
    paths.extend(
        plot_trajectory_with_ci(
            records,
            labels,
            scale,
            out_dir,
            title_prefix,
            error_bar=error_bar,
            y_axis=y_axis,
            show_repeat_points=show_repeat_points,
            outer_summary=outer_summary,
        )
        for scale in scales
    )
    paths.append(
        plot_endpoint_delta_with_ci(records, labels, scales, out_dir, title_prefix)
    )
    paths.append(
        plot_best_change_and_rebound(records, labels, scales, out_dir, title_prefix)
    )
    paths.extend(
        plot_measurement_reliability_heatmaps(
            records, labels, scales, out_dir, title_prefix
        )
    )
    return paths


def render_appendix(
    records: list[ExperimentRecord],
    labels: list[str],
    scales: list[str],
    out_dir: Path,
    title_prefix: str,
) -> list[Path]:
    paths = [_appendix_final_delta(records, labels, scales, out_dir, title_prefix)]
    index = 2
    for scale in scales:
        paths.append(
            _appendix_trajectory(records, labels, scale, out_dir, title_prefix, index)
        )
        index += 1
    paths.append(
        _appendix_delta_trajectory(
            records, labels, scales, out_dir, title_prefix, index
        )
    )
    index += 1
    paths.append(
        _appendix_volatility(records, labels, scales, out_dir, title_prefix, index)
    )
    index += 1
    paths.append(
        _appendix_endpoint_uncertainty(
            records, labels, scales, out_dir, title_prefix, index
        )
    )
    index += 1
    paths.append(
        _appendix_delta_heatmap(records, labels, scales, out_dir, title_prefix, index)
    )
    return paths


def render_core_figures(
    records: list[ExperimentRecord],
    labels: list[str],
    scales: list[str],
    out_dir: Path,
    title_prefix: str,
    *,
    process_rows: list[dict[str, Any]] | None = None,
    stage_rows: list[dict[str, Any]] | None = None,
) -> list[Path]:
    paths: list[Path] = []
    for path in (
        plot_group_time_contrasts(records, labels, scales, out_dir, title_prefix),
        plot_outer_contrast_forest(records, labels, scales, out_dir, title_prefix),
        plot_cross_scale_concurrent_validity(records, labels, out_dir, title_prefix),
        plot_cross_scale_convergence(records, labels, out_dir, title_prefix),
        plot_measurement_icc(records, labels, scales, out_dir, title_prefix),
        plot_measurement_error(records, labels, scales, out_dir, title_prefix),
    ):
        if path is not None:
            paths.append(path)
    paths.extend(
        plot_endpoint_waterfall(records, labels, scales, out_dir, title_prefix)
    )
    paths.extend(
        plot_process_delivery(
            process_rows or [], stage_rows or [], out_dir, title_prefix
        )
    )
    return paths
