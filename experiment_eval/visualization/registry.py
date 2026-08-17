"""Central figure-family registry for the main evaluation pipeline.

Adding a chart to an existing family only requires changing that family's
renderer.  A genuinely new family is registered here without coupling it to
input loading, statistics, report writing, or CLI batch selection.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..schema import ExperimentRecord
from ..charts.outcomes.change_ci import plot_change_ci_comparison
from ..charts.process.engagement import plot_engagement_process
from ..workflows.figure_families import (
    render_appendix,
    render_core_figures,
    render_lines_only,
    render_presentation,
)


def render_requested_figures(
    records: list[ExperimentRecord],
    labels: list[str],
    scales: list[str],
    out_dir: Path,
    title_prefix: str,
    *,
    report_mode: str,
    error_bar: str,
    y_axis: str,
    show_repeat_points: bool,
    outer_summary: str,
    core_figures: bool,
    process_rows: list[dict[str, Any]],
    stage_rows: list[dict[str, Any]],
    paper_figure_types: list[str],
    change_estimates: list[dict[str, Any]],
    change_contrasts: list[dict[str, Any]],
    change_significance_label: str,
    engagement_daily: list[dict[str, Any]],
    engagement_time_bins: list[dict[str, Any]],
    engagement_sessions: list[dict[str, Any]],
) -> list[Path]:
    paths: list[Path] = []
    if report_mode == "lines-only":
        paths.extend(
            render_lines_only(
                records, labels, scales, out_dir, title_prefix, y_axis=y_axis
            )
        )
    if report_mode in {"presentation", "all"}:
        paths.extend(
            render_presentation(
                records,
                labels,
                scales,
                out_dir,
                title_prefix,
                error_bar=error_bar,
                y_axis=y_axis,
                show_repeat_points=show_repeat_points,
                outer_summary=outer_summary,
            )
        )
    if report_mode in {"appendix", "all"}:
        paths.extend(render_appendix(records, labels, scales, out_dir, title_prefix))
    if core_figures and report_mode in {"core", "presentation", "all"}:
        paths.extend(
            render_core_figures(
                records,
                labels,
                scales,
                out_dir,
                title_prefix,
                process_rows=process_rows,
                stage_rows=stage_rows,
            )
        )
    if paper_figure_types and report_mode in {"core", "presentation", "all"}:
        if "change-ci" in paper_figure_types:
            paths.extend(
                plot_change_ci_comparison(
                    change_estimates,
                    change_contrasts,
                    out_dir,
                    title_prefix,
                    significance_label=change_significance_label,
                )
            )
        if "engagement" in paper_figure_types:
            paths.extend(
                plot_engagement_process(
                    engagement_daily,
                    engagement_time_bins,
                    out_dir,
                    title_prefix,
                    engagement_sessions,
                )
            )
    return paths
