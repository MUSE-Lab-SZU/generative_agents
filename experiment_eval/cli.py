"""Main command-line entry point for experimental metric reporting."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .cli_config import (
    BATCH_MODE,
    OUT_DIR,
    PROJECT_ROOT,
    RECURSIVE,
    REPORTS_DIR,
    TITLE_PREFIX,
    parse_args,
)
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
from .data.longitudinal import long_data_from_records, read_long_data
from .paper_longitudinal import LONG_FIGURE_TYPES, render_long_format_paper_figures
from .persona_profile import PROFILE_FIGURE_TYPES, render_persona_profile
from .data.persona_profile import (
    persona_profile_data_from_records,
    read_persona_profile_data,
)
from .analysis.persona_predictor import read_feature_data
from .pipeline import combined_records, render_chart_batch
from .process import extract_process_metrics
from .data.engagement import extract_engagement_events
from .schema import group_sort_key, normalize_group, normalize_kbd, normalize_repeat_id


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
            raise SystemExit(
                f"Invalid --group-alias (expected PATH_MATCH=LABEL): {item}"
            )
        path_match, label = (part.strip() for part in item.split("=", 1))
        if not path_match or not label:
            raise SystemExit(f"Invalid --group-alias (empty match or label): {item}")
        result.append((path_match, label))
    return result


def _render_long_figures(args: Any, frame: Any, out_dir: Path) -> dict[str, Any]:
    feature_frame = None
    if args.feature_data:
        feature_path = _resolve(args.feature_data)
        if not feature_path.is_file():
            raise SystemExit(f"Feature data not found: {feature_path}")
        try:
            feature_frame = read_feature_data(feature_path)
        except (ValueError, TypeError) as exc:
            raise SystemExit(f"Failed to load feature data: {exc}") from exc
    try:
        return render_long_format_paper_figures(
            frame,
            out_dir / "long_format_paper_figures",
            figure_types=args.paper_figure or [],
            scales=args.symptom_scale or ["PHQ-9"],
            feature_frame=feature_frame,
            cbt_group=args.cbt_group,
            control_group=args.control_group,
            interval=args.symptom_interval,
            forest_timepoints=args.forest_timepoint,
            total_effect_reference=args.total_effect_reference,
            fdr_alpha=args.fdr_alpha,
            shap_target=args.shap_target,
            shap_scale=args.shap_scale,
            shap_endpoint=args.shap_endpoint,
            shap_min_samples=args.shap_min_samples,
            network_mode=args.symptom_network_mode,
            network_scale=args.symptom_network_scale,
            network_groups=args.symptom_network_group,
            network_timepoints=args.symptom_network_timepoint,
            network_min_n=args.symptom_network_min_n,
            network_ebic_gamma=args.symptom_network_ebic_gamma,
            network_alpha_min_ratio=args.symptom_network_alpha_min_ratio,
            network_alpha_count=args.symptom_network_alpha_count,
            network_edge_threshold=args.symptom_network_edge_threshold,
        )
    except (ValueError, KeyError, TypeError) as exc:
        raise SystemExit(f"Failed to render long-format paper figures: {exc}") from exc


def _render_persona_profile(
    args: Any, frame: Any, out_dir: Path, title_prefix: str
) -> dict[str, Any]:
    try:
        return render_persona_profile(
            frame,
            out_dir,
            method=args.profile_zscore,
            scales=args.profile_scale,
            persona_order=args.persona_order,
            group_order=args.profile_group_order,
            panel_by=args.profile_panel_by,
            setting_order=args.profile_setting_order,
            baseline_timepoint=args.profile_baseline,
            post_timepoint=args.profile_post,
            uncertainty=args.profile_uncertainty,
            title=f"{title_prefix}: persona treatment-response profile",
        )
    except (ValueError, KeyError, TypeError) as exc:
        raise SystemExit(f"Failed to render persona profile: {exc}") from exc


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    reports_dir = (
        _resolve(args.reports_dir) if args.reports_dir is not None else REPORTS_DIR
    )
    out_dir = _resolve(args.out_dir) if args.out_dir is not None else OUT_DIR
    recursive = args.recursive if args.recursive is not None else RECURSIVE
    title_prefix = args.title_prefix if args.title_prefix is not None else TITLE_PREFIX
    batch_mode = args.batch_mode if args.batch_mode is not None else BATCH_MODE
    paper_figure_types = args.paper_figure or ["change-ci", "engagement"]
    if not args.paper_figures:
        paper_figure_types = []
    long_figure_types = [
        value for value in paper_figure_types if value in LONG_FIGURE_TYPES
    ]
    profile_requested = any(
        value in PROFILE_FIGURE_TYPES for value in paper_figure_types
    )
    batch_paper_types = [
        value
        for value in paper_figure_types
        if value not in LONG_FIGURE_TYPES and value not in PROFILE_FIGURE_TYPES
    ]
    if not 0 < args.fdr_alpha <= 1:
        raise SystemExit("--fdr-alpha must be within (0, 1]")
    if args.shap_min_samples < 5:
        raise SystemExit("--shap-min-samples must be at least 5")
    if args.symptom_network_min_n is not None and args.symptom_network_min_n < 2:
        raise SystemExit("--symptom-network-min-n must be at least 2")
    if not 0.0 <= args.symptom_network_ebic_gamma <= 1.0:
        raise SystemExit("--symptom-network-ebic-gamma must be between 0 and 1")
    if not 0.0 < args.symptom_network_alpha_min_ratio <= 1.0:
        raise SystemExit("--symptom-network-alpha-min-ratio must be in (0, 1]")
    if args.symptom_network_alpha_count < 2:
        raise SystemExit("--symptom-network-alpha-count must be at least 2")
    if not 0.0 <= args.symptom_network_edge_threshold < 1.0:
        raise SystemExit("--symptom-network-edge-threshold must be in [0, 1)")

    # A canonical long table can drive the long-format families without any
    # repeat-summary directory.  Mixed old/new requests continue below.
    if (
        long_figure_types
        and args.long_data
        and not batch_paper_types
        and not profile_requested
    ):
        long_path = _resolve(args.long_data)
        if not long_path.is_file():
            raise SystemExit(f"Long data not found: {long_path}")
        try:
            long_frame = read_long_data(long_path)
        except (ValueError, TypeError) as exc:
            raise SystemExit(f"Failed to load long data: {exc}") from exc
        result = _render_long_figures(args, long_frame, out_dir)
        print(f"Read {len(long_frame)} canonical item rows from {_display(long_path)}")
        print(f"Long-format paper figures: {_display(Path(result['out_dir']))}")
        print(f"- manifest: {_display(Path(result['manifest']))}")
        return 0

    if (
        profile_requested
        and args.persona_profile_data
        and not batch_paper_types
        and not long_figure_types
    ):
        profile_path = _resolve(args.persona_profile_data)
        if not profile_path.is_file():
            raise SystemExit(f"Persona-profile data not found: {profile_path}")
        try:
            profile_frame = read_persona_profile_data(profile_path)
        except (ValueError, TypeError) as exc:
            raise SystemExit(f"Failed to load persona-profile data: {exc}") from exc
        result = _render_persona_profile(args, profile_frame, out_dir, title_prefix)
        print(
            f"Read {len(profile_frame)} persona-profile rows from {_display(profile_path)}"
        )
        print(f"Persona profile: {_display(Path(result['out_dir']))}")
        print(f"- manifest: {_display(Path(result['manifest']))}")
        return 0

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
    repeat_aliases = _aliases(args.repeat_alias)
    try:
        records, labels, scales = load_records(
            report_files, group_aliases, repeat_aliases=repeat_aliases
        )
    except (ValueError, KeyError, TypeError) as exc:
        raise SystemExit(f"Failed to load repeat summary data: {exc}") from exc
    if not records or not labels or not scales:
        raise SystemExit("No plottable records found.")

    all_kbds = available_kbds(records)
    requested_kbds = (
        [normalize_kbd(value) for value in args.kbd] if args.kbd else all_kbds
    )
    missing_kbds = sorted(set(requested_kbds) - set(all_kbds), key=group_sort_key)
    if missing_kbds:
        raise SystemExit(f"Requested KBD variants not found: {', '.join(missing_kbds)}")
    records = [record for record in records if record.kbd in requested_kbds]

    all_groups = available_groups(records)
    all_repeats = sorted({record.repeat_id for record in records}, key=group_sort_key)
    requested_groups = (
        [normalize_group(value) for value in args.group] if args.group else all_groups
    )
    requested_repeats = (
        [normalize_repeat_id(value) for value in args.repeat_id]
        if args.repeat_id
        else all_repeats
    )
    missing_groups = sorted(set(requested_groups) - set(all_groups), key=group_sort_key)
    missing_repeats = sorted(
        set(requested_repeats) - set(all_repeats), key=group_sort_key
    )
    if missing_groups:
        raise SystemExit(f"Requested groups not found: {', '.join(missing_groups)}")
    if missing_repeats:
        raise SystemExit(
            f"Requested repeat ids not found: {', '.join(missing_repeats)}"
        )
    records = [record for record in records if record.group in requested_groups]
    records = [record for record in records if record.repeat_id in requested_repeats]

    long_figure_result: dict[str, Any] | None = None
    if long_figure_types:
        if args.long_data:
            long_path = _resolve(args.long_data)
            if not long_path.is_file():
                raise SystemExit(f"Long data not found: {long_path}")
            try:
                long_frame = read_long_data(long_path)
            except (ValueError, TypeError) as exc:
                raise SystemExit(f"Failed to load long data: {exc}") from exc
        else:
            long_frame = long_data_from_records(records, labels, scales)
        long_figure_result = _render_long_figures(args, long_frame, out_dir)
    profile_result: dict[str, Any] | None = None
    if profile_requested:
        if args.persona_profile_data:
            profile_path = _resolve(args.persona_profile_data)
            if not profile_path.is_file():
                raise SystemExit(f"Persona-profile data not found: {profile_path}")
            try:
                profile_frame = read_persona_profile_data(profile_path)
            except (ValueError, TypeError) as exc:
                raise SystemExit(f"Failed to load persona-profile data: {exc}") from exc
        else:
            profile_frame = persona_profile_data_from_records(records, labels, scales)
        profile_result = _render_persona_profile(
            args, profile_frame, out_dir, title_prefix
        )

    if not batch_paper_types and (
        long_figure_result is not None or profile_result is not None
    ):
        if long_figure_result is not None:
            source = (
                _display(_resolve(args.long_data))
                if args.long_data
                else "validated repeat summaries"
            )
            print(f"Prepared {len(long_frame)} canonical item rows from {source}")
            print(
                f"Long-format paper figures: {_display(Path(long_figure_result['out_dir']))}"
            )
            print(f"- manifest: {_display(Path(long_figure_result['manifest']))}")
        if profile_result is not None:
            source = (
                _display(_resolve(args.persona_profile_data))
                if args.persona_profile_data
                else "validated repeat summaries"
            )
            print(f"Prepared persona-profile data from {source}")
            print(f"Persona profile: {_display(Path(profile_result['out_dir']))}")
            print(f"- manifest: {_display(Path(profile_result['manifest']))}")
        return 0

    experiment_data_root = (
        _resolve(args.experiment_data_root) if args.experiment_data_root else None
    )
    checkpoints_root = (
        _resolve(args.checkpoints_root) if args.checkpoints_root else None
    )
    process_rows, stage_rows = extract_process_metrics(
        records,
        experiment_data_root=experiment_data_root,
        checkpoints_root=checkpoints_root,
        project_root=PROJECT_ROOT,
    )
    if args.engagement_bin_hours <= 0 or 24 % args.engagement_bin_hours:
        raise SystemExit("--engagement-bin-hours must be a positive divisor of 24")
    engagement_events, engagement_coverage, engagement_availability = (
        extract_engagement_events(
            records,
            experiment_data_root=experiment_data_root,
            checkpoints_root=checkpoints_root,
            project_root=PROJECT_ROOT,
        )
        if "engagement" in batch_paper_types
        else ([], [], {"status": "not_requested"})
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
        "paper_figure_types": batch_paper_types,
        "long_format_paper_figures": long_figure_result,
        "paper_outcomes": args.paper_outcome or ["PHQ-9"],
        "change_panel_b": args.change_panel_b,
        "change_estimator": args.change_estimator,
        "change_significance_label": args.change_significance_label,
        "engagement_events": engagement_events,
        "engagement_coverage": engagement_coverage,
        "engagement_availability": engagement_availability,
        "engagement_bin_hours": args.engagement_bin_hours,
        "engagement_secondary": args.engagement_secondary,
    }
    batches: list[dict[str, Any]] = []
    if args.combine_kbds:
        combined_title = f"{title_prefix}: {', '.join(requested_kbds)}"
        combined_groups = available_groups(records)
        if batch_mode == "single":
            batches.append(
                render_chart_batch(
                    combined_records(records),
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
                        combined_records(group_records, include_group=False),
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
                repeat_records = [
                    record for record in records if record.repeat_id == repeat_id
                ]
                if not repeat_records:
                    continue
                batches.append(
                    render_chart_batch(
                        combined_records(repeat_records),
                        labels,
                        scales,
                        out_dir / "by_repeat" / repeat_id,
                        f"{combined_title}: {repeat_id} across KBD variants and groups",
                        batch_kind="same-id cross-KBD comparison",
                        **common,
                    )
                )
        if batch_mode in {"compare-all", "all"}:
            destination = (
                out_dir / "all_independent_experiments"
                if batch_mode == "all"
                else out_dir
            )
            batches.append(
                render_chart_batch(
                    combined_records(records),
                    labels,
                    scales,
                    destination,
                    f"{combined_title}: all selected independent experiments",
                    batch_kind="cross-KBD independent experiments",
                    **common,
                )
            )

    for kbd in [] if args.combine_kbds else requested_kbds:
        kbd_records = [record for record in records if record.kbd == kbd]
        if not kbd_records:
            continue
        multiple_kbds = len(requested_kbds) > 1
        kbd_out = out_dir / "by_kbd" / kbd if multiple_kbds else out_dir
        kbd_title = f"{title_prefix}: {kbd}" if multiple_kbds else title_prefix
        kbd_groups = available_groups(kbd_records)
        kbd_repeats = sorted(
            {record.repeat_id for record in kbd_records}, key=group_sort_key
        )
        if batch_mode == "single":
            batches.append(
                render_chart_batch(
                    kbd_records,
                    labels,
                    scales,
                    kbd_out,
                    kbd_title,
                    batch_kind="single",
                    **common,
                )
            )
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
            destination = (
                kbd_out / "all_independent_experiments"
                if batch_mode == "all"
                else kbd_out
            )
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
        "paper_figure_types": paper_figure_types,
        "persona_profile": profile_result,
        "paper_outcomes": args.paper_outcome or ["PHQ-9"],
        "change_panel_b": args.change_panel_b,
        "change_estimator": args.change_estimator,
        "change_significance_label": args.change_significance_label,
        "engagement_bin_hours": args.engagement_bin_hours,
        "engagement_secondary": args.engagement_secondary,
        "symptom_network_mode": args.symptom_network_mode,
        "symptom_network_scale": args.symptom_network_scale,
        "symptom_network_groups": args.symptom_network_group,
        "symptom_network_timepoints": args.symptom_network_timepoint,
        "symptom_network_min_n": args.symptom_network_min_n,
        "symptom_network_ebic_gamma": args.symptom_network_ebic_gamma,
        "symptom_network_alpha_min_ratio": args.symptom_network_alpha_min_ratio,
        "symptom_network_alpha_count": args.symptom_network_alpha_count,
        "symptom_network_edge_threshold": args.symptom_network_edge_threshold,
        "dataset_label": dataset_label,
        "experiment_data_root": (
            _display(experiment_data_root) if experiment_data_root else None
        ),
        "checkpoints_root": _display(checkpoints_root) if checkpoints_root else None,
        "batch_mode": batch_mode,
        "combine_kbds": args.combine_kbds,
        "kbds": requested_kbds,
        "groups": available_groups(records),
        "repeat_ids": requested_repeats,
        "batches": batches,
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    source = (
        "explicit --report-file selection"
        if args.report_file
        else _display(reports_dir)
    )
    print(f"Read {len(report_files)} report JSON files from {source}")
    print(f"Detected KBD variants: {', '.join(requested_kbds)}")
    print(f"Detected groups: {', '.join(available_groups(records))}")
    print(f"Selected repeats: {', '.join(requested_repeats)}")
    print(f"Detected severities: {', '.join(available_severities(records))}")
    print(f"Detected scales: {', '.join(scales)}")
    if long_figure_result:
        print(
            f"Long-format paper figures: {_display(Path(long_figure_result['out_dir']))}"
        )
        print(
            f"  audit: {_display(Path(long_figure_result['out_dir']) / 'paper_figure_feasibility.md')}"
        )
    if profile_result:
        print(f"Persona profile: {_display(Path(profile_result['out_dir']))}")
        print(
            f"  audit: {_display(Path(profile_result['out_dir']) / 'persona_profile_feasibility.md')}"
        )
    print(f"Wrote {len(batches)} chart batches to {_display(out_dir)}")
    for batch in batches:
        print(f"- {batch['batch_kind']}: {batch['out_dir']}")
        print(f"  report: {batch['statistics_report_path']}")
        print(f"  detail CSV: {batch['statistics_csv_path']}")
        print(f"  item CSV: {batch['item_statistics_csv_path']}")
    print(f"- manifest: {_display(manifest_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
