"""Reusable chart-batch orchestration shared by CLI entry points."""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Any

from .cli_config import PROJECT_ROOT
from .analysis.change_ci import build_change_ci_analysis
from .analysis.engagement import (
    build_engagement_analysis,
    build_session_turn_score_analysis,
)
from .data.change_ci import extract_change_score_rows
from .loader import records_for_all_comparison
from .reports import write_batch_reports, write_paper_figure_data
from .schema import ExperimentRecord
from .visualization.registry import render_requested_figures


def _unique_plot_labels(records: list[ExperimentRecord]) -> list[ExperimentRecord]:
    labels = [(record.plot_label, record.severity) for record in records]
    return (
        records
        if len(labels) == len(set(labels))
        else records_for_all_comparison(records)
    )


def render_chart_batch(
    records: list[ExperimentRecord],
    labels: list[str],
    scales: list[str],
    out_dir: Path,
    title_prefix: str,
    *,
    batch_kind: str,
    report_mode: str,
    error_bar: str,
    y_axis: str,
    show_repeat_points: bool,
    outer_summary: str,
    core_figures: bool,
    process_rows: list[dict[str, Any]],
    stage_rows: list[dict[str, Any]],
    dataset_label: str,
    paper_figure_types: list[str],
    long_format_paper_figures: dict[str, Any] | None,
    paper_outcomes: list[str],
    change_panel_b: str,
    change_estimator: str,
    change_significance_label: str,
    engagement_events: list[dict[str, Any]],
    engagement_coverage: list[dict[str, Any]],
    engagement_availability: dict[str, Any],
    engagement_bin_hours: int,
    engagement_secondary: str,
    **additional_family_options: Any,
) -> dict[str, Any]:
    if not records:
        raise ValueError(f"No records selected for chart batch: {batch_kind}")
    records = _unique_plot_labels(records)
    out_dir.mkdir(parents=True, exist_ok=True)
    stable_ids = {record.stable_id for record in records}
    selected_process = [row for row in process_rows if row["stable_id"] in stable_ids]
    selected_stages = [row for row in stage_rows if row["stable_id"] in stable_ids]
    selected_engagement = [
        row for row in engagement_events if row["stable_id"] in stable_ids
    ]
    selected_coverage = [
        row for row in engagement_coverage if row["stable_id"] in stable_ids
    ]
    change_rows, change_availability = extract_change_score_rows(
        records, labels, paper_outcomes
    )
    change_estimates, change_contrasts, change_diagnostics = (
        build_change_ci_analysis(
            change_rows,
            panel_b=change_panel_b,
            estimator=change_estimator,
        )
        if "change-ci" in paper_figure_types
        else ([], [], {"status": "not_requested"})
    )
    engagement_daily, engagement_bins, engagement_diagnostics = (
        build_engagement_analysis(
            selected_engagement,
            selected_coverage,
            bin_hours=engagement_bin_hours,
            secondary_metric=engagement_secondary,
        )
        if "engagement" in paper_figure_types
        else ([], [], {"status": "not_requested"})
    )
    engagement_sessions, engagement_session_diagnostics = (
        build_session_turn_score_analysis(selected_engagement, change_rows)
        if "engagement" in paper_figure_types
        else ([], {"status": "not_requested"})
    )
    selected_engagement_availability = dict(engagement_availability)
    if isinstance(engagement_availability.get("runs"), list):
        selected_runs = [
            row
            for row in engagement_availability["runs"]
            if row.get("stable_id") in stable_ids
        ]
        selected_engagement_availability["runs"] = selected_runs
        selected_engagement_availability["run_count"] = len(stable_ids)
        selected_engagement_availability["runs_with_conversations"] = sum(
            bool(row.get("conversation_count")) for row in selected_runs
        )
    paper_diagnostics = {
        "change_data": change_availability,
        "change_statistics": change_diagnostics,
        "engagement_data": selected_engagement_availability,
        "engagement_statistics": engagement_diagnostics,
        "engagement_session_statistics": engagement_session_diagnostics,
    }
    paper_data_paths = (
        write_paper_figure_data(
            out_dir,
            change_rows=change_rows,
            change_estimates=change_estimates,
            change_contrasts=change_contrasts,
            engagement_events=selected_engagement,
            engagement_coverage=selected_coverage,
            engagement_daily=engagement_daily,
            engagement_time_bins=engagement_bins,
            engagement_sessions=engagement_sessions,
            diagnostics=paper_diagnostics,
        )
        if paper_figure_types
        else {}
    )
    chart_paths = render_requested_figures(
        records,
        labels,
        scales,
        out_dir,
        title_prefix,
        report_mode=report_mode,
        error_bar=error_bar,
        y_axis=y_axis,
        show_repeat_points=show_repeat_points,
        outer_summary=outer_summary,
        core_figures=core_figures,
        process_rows=selected_process,
        stage_rows=selected_stages,
        paper_figure_types=paper_figure_types,
        change_estimates=change_estimates,
        change_contrasts=change_contrasts,
        change_significance_label=change_significance_label,
        engagement_daily=engagement_daily,
        engagement_time_bins=engagement_bins,
        engagement_sessions=engagement_sessions,
    )
    # ``report-mode all`` can render dozens of 300-DPI figures before the
    # weighted-Kappa bootstrap starts.  Individual renderers close their
    # figures, but Matplotlib may retain disconnected artists/arrays until a
    # collection cycle.  Release them at the phase boundary so large archive
    # batches do not accumulate enough resident memory to be killed.
    from .charts.shared.plotting import plt

    plt.close("all")
    gc.collect()
    kappa_figure_batch = batch_kind in {
        "single",
        "combined KBD selection",
        "cross-KBD independent experiments",
    } or batch_kind.startswith("all independent experiments")
    result = write_batch_reports(
        out_dir,
        records,
        labels,
        scales,
        chart_paths,
        batch_kind=batch_kind,
        report_mode=report_mode,
        error_bar=error_bar,
        y_axis=y_axis,
        outer_summary=outer_summary,
        process_rows=selected_process,
        stage_rows=selected_stages,
        dataset_label=dataset_label,
        project_root=PROJECT_ROOT,
        title_prefix=title_prefix,
        render_weighted_kappa=(
            core_figures
            and kappa_figure_batch
            and report_mode in {"core", "presentation", "all"}
        ),
    )
    result["paper_figure_data"] = {
        key: (
            str(path.relative_to(PROJECT_ROOT))
            if path.is_relative_to(PROJECT_ROOT)
            else str(path)
        )
        for key, path in paper_data_paths.items()
    }
    if long_format_paper_figures is not None:
        result["long_format_paper_figures"] = long_format_paper_figures
    if additional_family_options:
        result["additional_family_options"] = additional_family_options
    return result


def combined_records(
    records: list[ExperimentRecord], *, include_group: bool = True
) -> list[ExperimentRecord]:
    """Make labels unique when several KBD variants share one chart batch."""
    return [
        record.with_plot_label(
            "-".join(
                part
                for part in (
                    record.kbd,
                    record.group if include_group else None,
                    record.repeat_id,
                )
                if part is not None
            )
        )
        for record in records
    ]
