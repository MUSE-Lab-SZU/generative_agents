"""Command-line orchestration for experimental metric reporting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .loader import (
    available_groups,
    available_kbds,
    available_severities,
    find_report_files,
    load_records,
    records_for_all_comparison,
    records_for_group,
    records_for_repeat,
)
from .process import extract_process_metrics
from .visualization.scale_plots import render_appendix, render_lines_only, render_presentation
from .visualization.core_plots import render_core_figures
from .reports import write_batch_reports
from .schema import ExperimentRecord, group_sort_key, normalize_group, normalize_kbd, normalize_repeat_id


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = PROJECT_ROOT / "results" / "experiment_data" / "reports"
OUT_DIR = PROJECT_ROOT / "docs" / "experiment_evaluation"
RECURSIVE = False
TITLE_PREFIX = "In-silico experiment evaluation"
BATCH_MODE = "compare-all"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot KBD repeat summary charts from archived report JSON files.")
    parser.add_argument("--reports-dir", type=Path, default=None)
    parser.add_argument("--report-file", type=Path, action="append", default=None)
    parser.add_argument("--group-alias", action="append", default=None, metavar="PATH_MATCH=LABEL")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument(
        "--experiment-data-root",
        type=Path,
        default=None,
        help="Optional raw experiment_data root used for conversation-dose/process metrics.",
    )
    parser.add_argument(
        "--checkpoints-root",
        type=Path,
        default=None,
        help="Optional checkpoint root used for CBT stage and prompt-completion metrics.",
    )
    parser.add_argument("--dataset-label", default=None, help="Dataset/protocol label written into every report.")
    parser.add_argument("--recursive", dest="recursive", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--title-prefix", default=None)
    parser.add_argument(
        "--batch-mode",
        choices=["single", "by-group", "by-repeat", "compare-all", "all"],
        default=None,
    )
    parser.add_argument("--kbd", action="append", default=None)
    parser.add_argument("--group", action="append", default=None)
    parser.add_argument("--repeat-id", action="append", default=None)
    parser.add_argument(
        "--combine-kbds",
        action="store_true",
        help="Combine selected KBD variants in the same chart batches instead of writing one subtree per KBD.",
    )
    parser.add_argument(
        "--report-mode",
        choices=["core", "lines-only", "presentation", "appendix", "all"],
        default="all",
        help="Generate core paper figures, lines-only samples, presentation charts, appendix charts, or all charts.",
    )
    parser.add_argument(
        "--error-bar",
        choices=["ci95", "sd", "none"],
        default="ci95",
        help="Uncertainty shown on presentation trajectories (default: ci95).",
    )
    parser.add_argument(
        "--y-axis",
        choices=["full", "adaptive"],
        default="full",
        help="Presentation trajectory y-axis; full uses PHQ-9 0–27 and BDI-II 0–63.",
    )
    parser.add_argument("--show-repeat-points", action="store_true", help="Overlay raw repeat total scores on trajectories.")
    parser.add_argument(
        "--outer-summary",
        choices=["none", "mean-ci"],
        default="none",
        help="Optionally overlay outer-experiment means and t CIs; independent runs remain visible.",
    )
    parser.add_argument(
        "--core-figures",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Generate paper-oriented contrast, convergence, ICC, waterfall, and process figures.",
    )
    return parser.parse_args(argv)


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _display(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _aliases(values: list[str] | None) -> list[tuple[str, str]]:
    result: list[tuple[str, str]] = []
    for item in values or []:
        if "=" not in item:
            raise SystemExit(f"Invalid --group-alias (expected PATH_MATCH=LABEL): {item}")
        path_match, label = (part.strip() for part in item.split("=", 1))
        if not path_match or not label:
            raise SystemExit(f"Invalid --group-alias (empty match or label): {item}")
        result.append((path_match, label))
    return result


def _unique_plot_labels(records: list[ExperimentRecord]) -> list[ExperimentRecord]:
    labels = [(record.plot_label, record.severity) for record in records]
    if len(labels) == len(set(labels)):
        return records
    return records_for_all_comparison(records)


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
) -> dict[str, Any]:
    if not records:
        raise ValueError(f"No records selected for chart batch: {batch_kind}")
    records = _unique_plot_labels(records)
    out_dir.mkdir(parents=True, exist_ok=True)
    chart_paths: list[Path] = []
    if report_mode == "lines-only":
        chart_paths.extend(render_lines_only(records, labels, scales, out_dir, title_prefix, y_axis=y_axis))
    if report_mode in {"presentation", "all"}:
        chart_paths.extend(
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
        chart_paths.extend(render_appendix(records, labels, scales, out_dir, title_prefix))
    if core_figures and report_mode in {"core", "presentation", "all"}:
        stable_ids = {record.stable_id for record in records}
        chart_paths.extend(
            render_core_figures(
                records,
                labels,
                scales,
                out_dir,
                title_prefix,
                process_rows=[row for row in process_rows if row["stable_id"] in stable_ids],
                stage_rows=[row for row in stage_rows if row["stable_id"] in stable_ids],
            )
        )
    kappa_figure_batch = (
        batch_kind in {"single", "combined KBD selection", "cross-KBD independent experiments"}
        or batch_kind.startswith("all independent experiments")
    )
    return write_batch_reports(
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
        process_rows=[row for row in process_rows if row["stable_id"] in {record.stable_id for record in records}],
        stage_rows=[row for row in stage_rows if row["stable_id"] in {record.stable_id for record in records}],
        dataset_label=dataset_label,
        project_root=PROJECT_ROOT,
        title_prefix=title_prefix,
        render_weighted_kappa=kappa_figure_batch and report_mode in {"core", "presentation", "all"},
    )


def _combined_records(records: list[ExperimentRecord], *, include_group: bool = True) -> list[ExperimentRecord]:
    """Label records uniquely when multiple KBD variants share one chart batch."""
    return [
        record.with_plot_label(
            "-".join(
                part
                for part in (record.kbd, record.group if include_group else None, record.repeat_id)
                if part is not None
            )
        )
        for record in records
    ]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    reports_dir = _resolve(args.reports_dir) if args.reports_dir is not None else REPORTS_DIR
    out_dir = _resolve(args.out_dir) if args.out_dir is not None else OUT_DIR
    recursive = args.recursive if args.recursive is not None else RECURSIVE
    title_prefix = args.title_prefix if args.title_prefix is not None else TITLE_PREFIX
    batch_mode = args.batch_mode if args.batch_mode is not None else BATCH_MODE

    if args.report_file:
        report_files = []
        for item in args.report_file:
            path = _resolve(item)
            if not path.is_file():
                raise SystemExit(f"Report file not found: {path}")
            report_files.append(path)
        report_files = sorted(set(report_files))
    else:
        try:
            report_files = find_report_files(reports_dir, recursive)
        except ValueError as exc:
            raise SystemExit(f"Failed to discover repeat summary data: {exc}") from exc
    if not report_files:
        raise SystemExit(f"No repeat summary JSON files found in {reports_dir}")

    group_aliases = _aliases(args.group_alias)
    try:
        records, labels, scales = load_records(report_files, group_aliases)
    except (ValueError, KeyError, TypeError) as exc:
        raise SystemExit(f"Failed to load repeat summary data: {exc}") from exc
    if not records or not labels or not scales:
        raise SystemExit("No plottable records found.")

    all_kbds = available_kbds(records)
    requested_kbds = [normalize_kbd(value) for value in args.kbd] if args.kbd else all_kbds
    missing_kbds = sorted(set(requested_kbds) - set(all_kbds), key=group_sort_key)
    if missing_kbds:
        raise SystemExit(f"Requested KBD variants not found: {', '.join(missing_kbds)}")
    records = [record for record in records if record.kbd in requested_kbds]

    all_groups = available_groups(records)
    all_repeats = sorted({record.repeat_id for record in records}, key=group_sort_key)
    requested_groups = [normalize_group(value) for value in args.group] if args.group else all_groups
    requested_repeats = [normalize_repeat_id(value) for value in args.repeat_id] if args.repeat_id else all_repeats
    missing_groups = sorted(set(requested_groups) - set(all_groups), key=group_sort_key)
    missing_repeats = sorted(set(requested_repeats) - set(all_repeats), key=group_sort_key)
    if missing_groups:
        raise SystemExit(f"Requested groups not found: {', '.join(missing_groups)}")
    if missing_repeats:
        raise SystemExit(f"Requested repeat ids not found: {', '.join(missing_repeats)}")
    records = [record for record in records if record.group in requested_groups]
    records = [record for record in records if record.repeat_id in requested_repeats]

    experiment_data_root = _resolve(args.experiment_data_root) if args.experiment_data_root else None
    checkpoints_root = _resolve(args.checkpoints_root) if args.checkpoints_root else None
    process_rows, stage_rows = extract_process_metrics(
        records,
        experiment_data_root=experiment_data_root,
        checkpoints_root=checkpoints_root,
        project_root=PROJECT_ROOT,
    )
    dataset_label = args.dataset_label or reports_dir.name

    common = {
        "report_mode": args.report_mode,
        "error_bar": args.error_bar,
        "y_axis": args.y_axis,
        "show_repeat_points": args.show_repeat_points,
        "outer_summary": args.outer_summary,
        "core_figures": args.core_figures,
        "process_rows": process_rows,
        "stage_rows": stage_rows,
        "dataset_label": dataset_label,
    }
    batches: list[dict[str, Any]] = []
    if args.combine_kbds:
        combined_title = f"{title_prefix}: {', '.join(requested_kbds)}"
        combined_groups = available_groups(records)
        if batch_mode == "single":
            batches.append(
                render_chart_batch(
                    _combined_records(records),
                    labels,
                    scales,
                    out_dir,
                    combined_title,
                    batch_kind="combined KBD selection",
                    **common,
                )
            )
        if batch_mode in {"by-group", "all"}:
            for group in combined_groups:
                group_records = [record for record in records if record.group == group]
                batches.append(
                    render_chart_batch(
                        _combined_records(group_records, include_group=False),
                        labels,
                        scales,
                        out_dir / "by_group" / group,
                        f"{combined_title}: {group} across KBD variants",
                        batch_kind="same-group cross-KBD comparison",
                        **common,
                    )
                )
        if batch_mode in {"by-repeat", "all"}:
            for repeat_id in requested_repeats:
                repeat_records = [record for record in records if record.repeat_id == repeat_id]
                if not repeat_records:
                    continue
                batches.append(
                    render_chart_batch(
                        _combined_records(repeat_records),
                        labels,
                        scales,
                        out_dir / "by_repeat" / repeat_id,
                        f"{combined_title}: {repeat_id} across KBD variants and groups",
                        batch_kind="same-id cross-KBD comparison",
                        **common,
                    )
                )
        if batch_mode in {"compare-all", "all"}:
            destination = out_dir / "all_independent_experiments" if batch_mode == "all" else out_dir
            batches.append(
                render_chart_batch(
                    _combined_records(records),
                    labels,
                    scales,
                    destination,
                    f"{combined_title}: all selected independent experiments",
                    batch_kind="cross-KBD independent experiments",
                    **common,
                )
            )

    for kbd in ([] if args.combine_kbds else requested_kbds):
        kbd_records = [record for record in records if record.kbd == kbd]
        if not kbd_records:
            continue
        multiple_kbds = len(requested_kbds) > 1
        kbd_out = out_dir / "by_kbd" / kbd if multiple_kbds else out_dir
        kbd_title = f"{title_prefix}: {kbd}" if multiple_kbds else title_prefix
        kbd_groups = available_groups(kbd_records)
        kbd_repeats = sorted({record.repeat_id for record in kbd_records}, key=group_sort_key)
        if batch_mode == "single":
            batches.append(render_chart_batch(kbd_records, labels, scales, kbd_out, kbd_title, batch_kind="single", **common))
        if batch_mode in {"by-group", "all"}:
            for group in kbd_groups:
                batches.append(
                    render_chart_batch(
                        records_for_group(kbd_records, group),
                        labels,
                        scales,
                        kbd_out / "by_group" / group,
                        f"{kbd_title}: {group} independent experiments",
                        batch_kind="within-group independent experiments",
                        **common,
                    )
                )
        if batch_mode in {"by-repeat", "all"}:
            for repeat_id in requested_repeats:
                if repeat_id not in kbd_repeats:
                    continue
                batches.append(
                    render_chart_batch(
                        records_for_repeat(kbd_records, repeat_id),
                        labels,
                        scales,
                        kbd_out / "by_repeat" / repeat_id,
                        f"{kbd_title}: {repeat_id} across groups",
                        batch_kind="same-id cross-group comparison",
                        **common,
                    )
                )
        if batch_mode in {"compare-all", "all"}:
            destination = kbd_out / "all_independent_experiments" if batch_mode == "all" else kbd_out
            batches.append(
                render_chart_batch(
                    records_for_all_comparison(kbd_records),
                    labels,
                    scales,
                    destination,
                    f"{kbd_title}: all independent experiments",
                    batch_kind=(
                        "all independent experiments; outer mean-ci overlay"
                        if args.outer_summary == "mean-ci"
                        else "all independent experiments; no outer-repeat pooling"
                    ),
                    **common,
                )
            )

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "batch_manifest.json"
    manifest = {
        "reports_dir": _display(reports_dir),
        "report_count": len(report_files),
        "report_mode": args.report_mode,
        "error_bar": args.error_bar,
        "y_axis": args.y_axis,
        "show_repeat_points": args.show_repeat_points,
        "outer_summary": args.outer_summary,
        "core_figures": args.core_figures,
        "dataset_label": dataset_label,
        "experiment_data_root": _display(experiment_data_root) if experiment_data_root else None,
        "checkpoints_root": _display(checkpoints_root) if checkpoints_root else None,
        "batch_mode": batch_mode,
        "combine_kbds": args.combine_kbds,
        "kbds": requested_kbds,
        "groups": available_groups(records),
        "repeat_ids": requested_repeats,
        "batches": batches,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    source = "explicit --report-file selection" if args.report_file else _display(reports_dir)
    print(f"Read {len(report_files)} report JSON files from {source}")
    print(f"Detected KBD variants: {', '.join(requested_kbds)}")
    print(f"Detected groups: {', '.join(available_groups(records))}")
    print(f"Selected repeats: {', '.join(requested_repeats)}")
    print(f"Detected severities: {', '.join(available_severities(records))}")
    print(f"Detected scales: {', '.join(scales)}")
    print(f"Wrote {len(batches)} chart batches to {_display(out_dir)}")
    for batch in batches:
        print(f"- {batch['batch_kind']}: {batch['out_dir']}")
        print(f"  report: {batch['statistics_report_path']}")
        print(f"  detail CSV: {batch['statistics_csv_path']}")
        print(f"  item CSV: {batch['item_statistics_csv_path']}")
    print(f"- manifest: {_display(manifest_path)}")
    return 0
