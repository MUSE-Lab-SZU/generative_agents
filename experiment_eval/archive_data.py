"""Build auditable analysis interfaces from a relocated ``results`` archive.

The exporter keeps one row identity per root simulation run.  Repeat scale
generations are collapsed by the strict repeat-summary loader, and follow-up
branches are attached only through archived parent metadata.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .data.engagement import extract_engagement_events
from .loader import (
    find_report_files,
    load_records,
    read_json,
    resolve_followup_source_summary,
)
from .schema import ExperimentRecord, group_sort_key


SCHEMA_VERSION = "experiment_eval_archive_interfaces_v2"


@dataclass
class ArchiveInterfaces:
    run_catalog: pd.DataFrame
    outcomes: pd.DataFrame
    items: pd.DataFrame
    baseline_features: pd.DataFrame
    followup_lineage: pd.DataFrame
    engagement_events: pd.DataFrame
    engagement_coverage: pd.DataFrame
    condition_coverage: pd.DataFrame
    availability: dict[str, Any]


def _parse_time(value: Any) -> datetime | None:
    text = str(value or "").strip()
    for fmt in ("%Y%m%d-%H:%M:%S", "%Y%m%d-%H:%M", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def _reports_dir(archive_root: Path) -> Path:
    candidates = (
        archive_root / "experiment_data" / "reports",
        archive_root / "reports",
        archive_root,
    )
    for candidate in candidates:
        if candidate.is_dir() and any(candidate.glob("**/repeat-*_summary.json")):
            return candidate
    raise ValueError(f"No repeat summaries found below archive root: {archive_root}")


def _resolve_report_reference(reports_dir: Path, value: Any) -> Path | None:
    basename = Path(str(value or "")).name
    if not basename:
        return None
    matches = [path for path in reports_dir.glob(f"**/{basename}") if path.is_file()]
    return matches[0] if len(matches) == 1 else None


def _condition(payload: dict[str, Any], record: ExperimentRecord) -> dict[str, Any]:
    conditions = [
        row for row in (payload.get("conditions") or []) if isinstance(row, dict)
    ]
    exact_run = [row for row in conditions if row.get("run_name") == record.run_name]
    if len(exact_run) == 1:
        return exact_run[0]
    exact_condition = [
        row for row in conditions if row.get("condition_name") == record.condition_name
    ]
    return exact_condition[0] if len(exact_condition) == 1 else {}


def _original_condition(
    reports_dir: Path,
    repeat_payload: dict[str, Any],
    record: ExperimentRecord,
) -> tuple[Path | None, dict[str, Any]]:
    path = _resolve_report_reference(
        reports_dir, repeat_payload.get("source_original_summary")
    )
    if path is None:
        return None, {}
    return path, _condition(read_json(path), record)


def _manifest_metadata(condition: dict[str, Any]) -> dict[str, Any]:
    manifest = condition.get("controller_manifest") or {}
    resolved = manifest.get("resolved_config") or {}
    config = resolved.get("config") if isinstance(resolved, dict) else {}
    config = config if isinstance(config, dict) else {}
    intervention = config.get("intervention") or {}
    staged_eval = config.get("staged_eval") or {}
    group_overlay = manifest.get("group_overlay") or {}
    agent_llm = ((config.get("agent_base") or {}).get("think") or {}).get("llm") or {}
    forced_llm = intervention.get("forced_llm") or {}
    start = config.get("time")
    if isinstance(start, dict):
        start = start.get("start")
    return {
        "controller_identity": manifest.get("controller_identity")
        or condition.get("controller_identity"),
        "controller_version": manifest.get("controller_version")
        or condition.get("controller_version"),
        "controller_mode": manifest.get("mode") or condition.get("mode"),
        "group_overlay_path": group_overlay.get("path"),
        "group_overlay_sha256": group_overlay.get("sha256"),
        "base_config_sha256": (manifest.get("base_config") or {}).get("sha256"),
        "runtime_config_sha256": resolved.get("sha256"),
        "target_agent": staged_eval.get("target_agent"),
        "doctor": intervention.get("doctor"),
        "simulation_model_provider": agent_llm.get("provider"),
        "simulation_model": agent_llm.get("model"),
        "intervention_model_provider": forced_llm.get("provider"),
        "intervention_model": forced_llm.get("model"),
        "simulation_start": start,
        "stride_minutes": config.get("stride"),
    }


def _root_run_id(record: ExperimentRecord) -> str:
    return record.run_name or record.stable_id


def _timepoint_role(label: str, *, followup: bool) -> str:
    if followup:
        return "followup"
    if label.strip().lower() in {"t0", "baseline", "pre"}:
        return "baseline"
    if label.strip().upper() in {"NOW", "POST", "ENDPOINT"}:
        return "endpoint"
    return "intervention_assessment"


def _evaluation_metadata(condition: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(row.get("trigger_label")): row
        for row in (condition.get("evaluations") or [])
        if isinstance(row, dict) and row.get("trigger_label")
    }


def _append_measurements(
    outcomes: list[dict[str, Any]],
    items: list[dict[str, Any]],
    *,
    record: ExperimentRecord,
    root: ExperimentRecord,
    repeat_payload: dict[str, Any],
    repeat_condition: dict[str, Any],
    root_baseline: datetime | None,
    followup: bool,
    linkage_source: str,
) -> None:
    evaluations = _evaluation_metadata(repeat_condition)
    run_id = _root_run_id(root)
    for scale, by_time in record.series.items():
        for timepoint, stats in by_time.items():
            evaluation = evaluations.get(timepoint, {})
            sim_time = stats.get("sim_time") or evaluation.get("sim_time")
            parsed_time = _parse_time(sim_time)
            elapsed_days = (
                (parsed_time - root_baseline).total_seconds() / 86400.0
                if parsed_time is not None and root_baseline is not None
                else None
            )
            common = {
                "run_id": run_id,
                "persona_id": root.kbd,
                "group": root.group,
                "repeat_id": root.repeat_id,
                "timepoint": timepoint,
                "timepoint_role": _timepoint_role(timepoint, followup=followup),
                "sim_time": sim_time,
                "elapsed_days": elapsed_days,
                "completed_session_count": stats.get("completed_session_count"),
                "scale": scale,
                "source_run_name": record.run_name,
                "source_summary": str(record.path),
                "source_kind": (
                    "linked_followup_repeat_summary"
                    if followup
                    else "root_repeat_summary"
                ),
                "lineage_source": linkage_source,
                "aggregation_method_version": repeat_payload.get(
                    "aggregation_method_version"
                ),
                "primary_score": (repeat_payload.get("evaluation_protocol") or {}).get(
                    "primary_score"
                ),
                "score_source": stats.get("score_source"),
                "expected_measurement_repeats": stats.get("expected_repeats"),
                "valid_measurement_repeats": stats.get("valid_repeats"),
            }
            outcomes.append(
                {
                    **common,
                    "score": stats.get("score"),
                    "measurement_sd": stats.get("sample_sd"),
                    "measurement_ci95_lower": stats.get("ci95_lower"),
                    "measurement_ci95_upper": stats.get("ci95_upper"),
                    "setting": "unspecified",
                }
            )
            for item in stats.get("item_statistics") or []:
                items.append(
                    {
                        "run": run_id,
                        "persona": root.kbd,
                        "group": root.group,
                        "timepoint": timepoint,
                        "scale": scale,
                        "item": item.get("item_id"),
                        "item_score": item.get("mean"),
                        **{
                            key: value
                            for key, value in common.items()
                            if key
                            not in {
                                "run_id",
                                "persona_id",
                                "group",
                                "timepoint",
                                "scale",
                            }
                        },
                    }
                )


def _baseline_features(
    roots: list[ExperimentRecord],
) -> pd.DataFrame:
    personas = sorted({record.kbd for record in roots}, key=group_sort_key)
    rows: list[dict[str, Any]] = []
    for record in roots:
        feature: dict[str, Any] = {
            "run": _root_run_id(record),
            "persona_id": record.kbd,
            "group": record.group,
        }
        for persona in personas:
            feature[f"persona__is_{persona.lower()}"] = float(record.kbd == persona)
        for scale, by_time in record.series.items():
            baseline_label = next(
                (
                    label
                    for label in by_time
                    if label.strip().lower() in {"t0", "baseline", "pre"}
                ),
                None,
            )
            if baseline_label is None:
                continue
            stats = by_time[baseline_label]
            scale_key = {"PHQ-9": "phq9", "BDI-II": "bdi2"}.get(
                scale, scale.lower().replace("-", "_")
            )
            feature[f"baseline__{scale_key}_total"] = stats.get("score")
            for item in stats.get("item_statistics") or []:
                feature[f"baseline__{scale_key}_item_{item.get('item_id')}"] = item.get(
                    "mean"
                )
        rows.append(feature)
    return pd.DataFrame.from_records(rows)


def _condition_coverage(
    run_catalog: pd.DataFrame,
    outcomes: pd.DataFrame,
    items: pd.DataFrame,
    *,
    expected_conditions: tuple[str, ...],
    expected_personas: tuple[str, ...],
) -> pd.DataFrame:
    """Materialize observed and missing persona/condition measurement cells.

    A condition that is absent from an archive cannot be inferred from filenames,
    so callers may declare the intended design with ``expected_conditions``.  The
    exporter always unions those values with observed values so unexpected data is
    visible rather than silently discarded.
    """

    observed_personas = set(run_catalog["persona"].dropna().astype(str))
    observed_conditions = set(run_catalog["condition"].dropna().astype(str))
    personas = sorted(
        observed_personas | {value for value in expected_personas if value},
        key=group_sort_key,
    )
    conditions = sorted(
        observed_conditions | {value for value in expected_conditions if value},
        key=group_sort_key,
    )
    measurement_cells = sorted(
        {
            (str(row.scale), str(row.timepoint))
            for row in outcomes[["scale", "timepoint"]].itertuples(index=False)
        },
        key=lambda value: (value[0], group_sort_key(value[1])),
    )
    run_counts = (
        run_catalog.groupby(["persona", "condition"], observed=True)["run_id"]
        .nunique()
        .to_dict()
    )
    outcome_counts = (
        outcomes.groupby(
            ["persona_id", "group", "scale", "timepoint"], observed=True
        )["run_id"]
        .nunique()
        .to_dict()
    )
    item_counts = (
        items.groupby(["persona", "group", "scale", "timepoint"], observed=True)[
            "run"
        ]
        .nunique()
        .to_dict()
        if not items.empty
        else {}
    )
    rows: list[dict[str, Any]] = []
    for persona in personas:
        for condition in conditions:
            root_n = int(run_counts.get((persona, condition), 0))
            for scale, timepoint in measurement_cells:
                outcome_n = int(
                    outcome_counts.get((persona, condition, scale, timepoint), 0)
                )
                item_n = int(item_counts.get((persona, condition, scale, timepoint), 0))
                if root_n == 0:
                    status = "missing_persona_condition"
                elif outcome_n == 0:
                    status = "missing_measurement"
                else:
                    status = "observed"
                rows.append(
                    {
                        "persona": persona,
                        "condition": condition,
                        "scale": scale,
                        "timepoint": timepoint,
                        "root_run_n": root_n,
                        "outcome_run_n": outcome_n,
                        "item_run_n": item_n,
                        "missing_run_n_within_cell": max(0, root_n - outcome_n),
                        "status": status,
                        "is_missing": status != "observed",
                        "condition_was_declared": condition in expected_conditions,
                        "persona_was_declared": persona in expected_personas,
                    }
                )
    return pd.DataFrame.from_records(rows)


def build_archive_interfaces(
    archive_root: Path,
    *,
    expected_conditions: tuple[str, ...] = (),
    expected_personas: tuple[str, ...] = (),
    repeat_aliases: tuple[tuple[str, str], ...] = (),
) -> ArchiveInterfaces:
    """Scan one archive and return canonical, lineage-safe tables."""
    archive_root = archive_root.resolve()
    reports_dir = _reports_dir(archive_root)
    repeat_paths = find_report_files(reports_dir, recursive=True, include_followup=True)
    payloads = {path: read_json(path) for path in repeat_paths}
    root_paths: list[Path] = []
    followup_repeat_by_original: dict[Path, Path] = {}
    unresolved_followup_repeats: list[str] = []
    for path, payload in payloads.items():
        declared_source = str(payload.get("source_original_summary") or "").strip()
        original_path = _resolve_report_reference(reports_dir, declared_source)
        if declared_source and original_path is None:
            original_path, _ = resolve_followup_source_summary(path, payload)
        if declared_source and original_path is None:
            raise ValueError(
                f"Cannot uniquely resolve source_original_summary for {path}: "
                f"{declared_source!r}"
            )
        original_payload = read_json(original_path) if original_path else {}
        if original_payload.get("artifact_kind") == "post_sim_followup_summary":
            if original_path in followup_repeat_by_original:
                raise ValueError(
                    "Multiple repeat follow-up summaries reference the same source: "
                    f"{original_path}"
                )
            followup_repeat_by_original[original_path] = path
        else:
            root_paths.append(path)

    roots, _, _ = load_records(root_paths, repeat_aliases=list(repeat_aliases))
    root_by_run = {record.run_name: record for record in roots if record.run_name}
    if len(root_by_run) != len(roots):
        raise ValueError(
            "Root repeat summaries require unique, non-empty run_name values"
        )

    root_context: dict[str, dict[str, Any]] = {}
    root_baselines: dict[str, datetime | None] = {}
    for record in roots:
        repeat_payload = payloads[record.path]
        repeat_condition = _condition(repeat_payload, record)
        original_path, original_condition = _original_condition(
            reports_dir, repeat_payload, record
        )
        manifest = _manifest_metadata(original_condition)
        evaluations = _evaluation_metadata(repeat_condition)
        baseline = next(
            (
                _parse_time(row.get("sim_time"))
                for label, row in evaluations.items()
                if label.strip().lower() in {"t0", "baseline", "pre"}
            ),
            None,
        )
        root_baselines[record.run_name] = baseline
        root_context[record.run_name] = {
            "repeat_payload": repeat_payload,
            "repeat_condition": repeat_condition,
            "original_path": original_path,
            "original_condition": original_condition,
            "manifest": manifest,
        }

    followup_paths = sorted(
        path
        for path in reports_dir.glob("**/*_summary.json")
        if path.is_file()
        and read_json(path).get("artifact_kind") == "post_sim_followup_summary"
    )
    lineage_rows: list[dict[str, Any]] = []
    linked_followup_records: list[
        tuple[ExperimentRecord, ExperimentRecord, dict[str, Any], dict[str, Any], str]
    ] = []
    for followup_path in followup_paths:
        followup_payload = read_json(followup_path)
        repeat_path = followup_repeat_by_original.get(followup_path)
        repeated_records = (
            load_records([repeat_path], repeat_aliases=list(repeat_aliases))[0]
            if repeat_path
            else []
        )
        repeated_by_branch = {record.run_name: record for record in repeated_records}
        for condition in followup_payload.get("conditions") or []:
            if not isinstance(condition, dict):
                continue
            manifest = condition.get("controller_manifest") or {}
            resume = manifest.get("resume") or {}
            parent_run_name = manifest.get("parent_run_name") or resume.get(
                "parent_run_name"
            )
            parent_label = resume.get("parent_label")
            branch_run_name = str(condition.get("run_name") or "")
            parent = root_by_run.get(str(parent_run_name or ""))
            repeated = repeated_by_branch.get(branch_run_name)
            linked = parent is not None and bool(parent_label)
            linkage_source = (
                "post_sim_followup_summary.conditions[].controller_manifest."
                "parent_run_name+resume.parent_label"
            )
            lineage_rows.append(
                {
                    "branch_run_name": branch_run_name,
                    "parent_run_name": parent_run_name,
                    "parent_run_id": _root_run_id(parent) if parent else None,
                    "parent_timepoint": parent_label,
                    "lineage_status": "linked" if linked else "unavailable",
                    "lineage_source": linkage_source if linked else None,
                    "followup_summary": str(followup_path),
                    "repeat_followup_summary": (
                        str(repeat_path) if repeat_path else None
                    ),
                    "repeat_followup_available": repeated is not None,
                    "followup_timepoints": "|".join(
                        str(value) for value in (condition.get("trigger_labels") or [])
                    ),
                    "controller_parent_manifest_sha256": manifest.get(
                        "parent_manifest_sha256"
                    ),
                }
            )
            if repeated is not None and linked and parent is not None:
                repeat_payload = payloads[repeat_path]
                repeat_condition = _condition(repeat_payload, repeated)
                linked_followup_records.append(
                    (
                        repeated,
                        parent,
                        repeat_payload,
                        repeat_condition,
                        linkage_source,
                    )
                )
            elif repeated is not None and not linked:
                unresolved_followup_repeats.append(str(repeat_path))

    outcomes: list[dict[str, Any]] = []
    items: list[dict[str, Any]] = []
    for record in roots:
        context = root_context[record.run_name]
        _append_measurements(
            outcomes,
            items,
            record=record,
            root=record,
            repeat_payload=context["repeat_payload"],
            repeat_condition=context["repeat_condition"],
            root_baseline=root_baselines[record.run_name],
            followup=False,
            linkage_source="root_repeat_summary.run_name",
        )
    for repeated, parent, payload, condition, linkage_source in linked_followup_records:
        _append_measurements(
            outcomes,
            items,
            record=repeated,
            root=parent,
            repeat_payload=payload,
            repeat_condition=condition,
            root_baseline=root_baselines[parent.run_name],
            followup=True,
            linkage_source=linkage_source,
        )

    engagement_events, engagement_coverage, engagement_audit = (
        extract_engagement_events(
            roots,
            experiment_data_root=archive_root,
            checkpoints_root=archive_root,
            project_root=archive_root,
        )
    )
    events_by_stable: dict[str, list[dict[str, Any]]] = {}
    for event in engagement_events:
        events_by_stable.setdefault(str(event["stable_id"]), []).append(event)
    coverage_by_stable = {str(row["stable_id"]): row for row in engagement_coverage}

    lineage_frame = pd.DataFrame.from_records(lineage_rows)
    branches_by_parent: dict[str, list[dict[str, Any]]] = {}
    for row in lineage_rows:
        if row.get("parent_run_name"):
            branches_by_parent.setdefault(str(row["parent_run_name"]), []).append(row)
    catalog_rows: list[dict[str, Any]] = []
    for record in roots:
        context = root_context[record.run_name]
        manifest = context["manifest"]
        branches = branches_by_parent.get(record.run_name, [])
        coverage = coverage_by_stable.get(record.stable_id, {})
        labels = sorted(
            {label for by_time in record.series.values() for label in by_time}
        )
        catalog_rows.append(
            {
                "run_id": _root_run_id(record),
                "stable_id": record.stable_id,
                "run_name": record.run_name,
                "persona": record.kbd,
                "condition": record.group,
                "repeat_id": record.repeat_id,
                "condition_name": record.condition_name,
                "archived_severity_label": record.severity,
                "grouping_dimensions": "persona|condition",
                "measurement_qc_passed": True,
                "measurement_qc_source": "strict_repeat_summary_validation",
                "simulation_qc_status": "unavailable_no_explicit_run_qc",
                "baseline_available": any(
                    label.lower() in {"t0", "baseline", "pre"} for label in labels
                ),
                "session_20_available": "session_20" in labels,
                "root_timepoints": "|".join(labels),
                "followup_branch_count": len(branches),
                "linked_followup_branch_count": sum(
                    row["lineage_status"] == "linked" for row in branches
                ),
                "scored_followup_branch_count": sum(
                    bool(row["repeat_followup_available"]) for row in branches
                ),
                "raw_process_available": bool(
                    coverage.get("conversation_log_available")
                    or coverage.get("simulation_event_log_available")
                ),
                "observed_interaction_count": len(
                    events_by_stable.get(record.stable_id, [])
                ),
                "original_summary": (
                    str(context["original_path"]) if context["original_path"] else None
                ),
                "repeat_summary": str(record.path),
                **manifest,
            }
        )

    run_catalog = pd.DataFrame.from_records(catalog_rows)
    outcomes_frame = pd.DataFrame.from_records(outcomes)
    items_frame = pd.DataFrame.from_records(items)
    baseline_features = _baseline_features(roots)
    condition_coverage = _condition_coverage(
        run_catalog,
        outcomes_frame,
        items_frame,
        expected_conditions=expected_conditions,
        expected_personas=expected_personas,
    )
    personas = sorted(run_catalog["persona"].unique().tolist(), key=group_sort_key)
    groups = sorted(run_catalog["condition"].unique().tolist(), key=group_sort_key)
    severity_values = sorted(run_catalog["archived_severity_label"].unique().tolist())
    persona_condition_n = (
        run_catalog.groupby(["persona", "condition"], observed=True)["run_id"]
        .nunique()
        .reset_index(name="n")
    )
    max_group_time_n = int(
        outcomes_frame.groupby(["group", "timepoint", "scale"], observed=True)["run_id"]
        .nunique()
        .max()
    )
    availability = {
        "schema_version": SCHEMA_VERSION,
        "archive_root": str(archive_root),
        "grouping": {
            "dimensions": ["persona", "condition"],
            "personas": personas,
            "conditions": groups,
            "archived_severity_values": severity_values,
            "severity_used_as_group": False,
            "reason": "The archive has no severity experiment dimension; its archived severity label is metadata only.",
        },
        "counts": {
            "root_outer_runs": len(roots),
            "repeat_summary_files": len(repeat_paths),
            "root_repeat_summary_files": len(root_paths),
            "followup_branches": len(lineage_rows),
            "linked_followup_branches": sum(
                row["lineage_status"] == "linked" for row in lineage_rows
            ),
            "followup_branches_with_repeat_scores": len(linked_followup_records),
            "outcome_rows": len(outcomes_frame),
            "item_rows": len(items_frame),
            "runs_with_raw_process": int(run_catalog["raw_process_available"].sum()),
        },
        "persona_condition_cells": persona_condition_n.to_dict("records"),
        "condition_coverage": {
            "status": "available",
            "scope": (
                "declared_design_plus_observed"
                if expected_conditions or expected_personas
                else "observed_design_only"
            ),
            "expected_conditions": list(expected_conditions),
            "expected_personas": list(expected_personas),
            "rows": len(condition_coverage),
            "missing_rows": int(condition_coverage["is_missing"].sum()),
            "limitation": (
                "A wholly absent condition is detectable only when supplied with "
                "--expected-condition."
            ),
        },
        "fields": {
            "root_run_identity": "available from repeat-summary condition.run_name",
            "followup_parent_lineage": (
                "available from archived controller parent metadata"
                if lineage_rows
                and all(row["lineage_status"] == "linked" for row in lineage_rows)
                else "partially_available"
            ),
            "simulation_time_and_elapsed_days": "available from evaluation.sim_time",
            "score_and_item_provenance": "available from strict repeat summaries",
            "controller_protocol_and_model": "available from source original summaries when embedded manifests exist",
            "main_vs_replication_role": "unavailable_no_explicit_metadata",
            "simulation_run_qc": "unavailable_no_explicit_behavior_qc_declaration",
            "true_conversation_elapsed_duration": "unavailable",
            "persona_static_configuration": "unavailable_beyond_persona_id_for_most_runs",
            "baseline_behavior_features": "unavailable_no_consistent_pre_treatment_behavior_window",
            "raw_process": f"available_for_{int(run_catalog['raw_process_available'].sum())}_of_{len(roots)}_root_runs",
        },
        "analysis_feasibility": {
            "change_ci_between_conditions": (
                "available" if len(groups) >= 2 else "unavailable_only_one_condition"
            ),
            "persona_profile": (
                "available_exploratory"
                if len(personas) >= 2 and persona_condition_n["n"].min() >= 2
                else "insufficient"
            ),
            "faceted_boxplot": "use persona and condition; severity facet unavailable",
            "symptom_trajectory": "available",
            "symptom_effect_forest": (
                "available_exploratory"
                if len(groups) >= 2
                else "unavailable_only_one_condition"
            ),
            "symptom_network": {
                "status": "insufficient_independent_runs",
                "maximum_group_time_cell_n": max_group_time_n,
                "recommended_minimum": "max(30, 5 x node_count): PHQ-9=45, BDI-II=105",
            },
            "persona_predictor_shap": {
                "status": "insufficient_independent_runs_or_conditions",
                "root_outer_runs": len(roots),
                "minimum_before_feature_rule": 40,
                "condition_count": len(groups),
            },
        },
        "warnings": [
            "Measurement repeats remain collapsed within each frozen snapshot and never increase outer-run n.",
            "Follow-up rows use the parent root run_id and never become independent samples.",
            *(
                [
                    "Some scored follow-up summaries could not be linked from explicit parent metadata: "
                    + ", ".join(unresolved_followup_repeats)
                ]
                if unresolved_followup_repeats
                else []
            ),
        ],
        "engagement": engagement_audit,
    }
    return ArchiveInterfaces(
        run_catalog=run_catalog,
        outcomes=outcomes_frame,
        items=items_frame,
        baseline_features=baseline_features,
        followup_lineage=lineage_frame,
        engagement_events=pd.DataFrame.from_records(engagement_events),
        engagement_coverage=pd.DataFrame.from_records(engagement_coverage),
        condition_coverage=condition_coverage,
        availability=availability,
    )


def write_archive_interfaces(
    interfaces: ArchiveInterfaces, out_dir: Path
) -> dict[str, Path]:
    """Write canonical tables and a manifest for downstream figure APIs."""
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = {
        "run_catalog": (interfaces.run_catalog, "archive_run_catalog.csv"),
        "outcomes_long": (interfaces.outcomes, "archive_outcomes_long.csv"),
        "items_long": (interfaces.items, "archive_items_long.csv"),
        "baseline_features": (
            interfaces.baseline_features,
            "archive_baseline_features.csv",
        ),
        "followup_lineage": (
            interfaces.followup_lineage,
            "archive_followup_lineage.csv",
        ),
        "engagement_events": (
            interfaces.engagement_events,
            "archive_engagement_events.csv",
        ),
        "engagement_coverage": (
            interfaces.engagement_coverage,
            "archive_engagement_coverage.csv",
        ),
        "condition_coverage": (
            interfaces.condition_coverage,
            "archive_condition_coverage.csv",
        ),
    }
    paths: dict[str, Path] = {}
    for key, (frame, filename) in frames.items():
        path = out_dir / filename
        frame.to_csv(path, index=False)
        paths[key] = path
    availability_path = out_dir / "archive_data_availability.json"
    availability_path.write_text(
        json.dumps(interfaces.availability, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    paths["availability"] = availability_path
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "statistical_unit": "independent root outer simulation run",
        "grouping_dimensions": ["persona", "condition"],
        "outputs": {key: path.name for key, path in paths.items()},
        "counts": interfaces.availability["counts"],
    }
    manifest_path = out_dir / "archive_interface_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    paths["manifest"] = manifest_path
    return paths


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build lineage-safe analysis tables from a results archive."
    )
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--repeat-alias",
        action="append",
        default=[],
        metavar="PATH_MATCH=REPEAT_ID",
        help="Remap an outer repeat by report-path substring; repeatable.",
    )
    parser.add_argument(
        "--expected-condition",
        action="append",
        default=[],
        help="Condition expected by the design (for example G1); repeatable.",
    )
    parser.add_argument(
        "--expected-persona",
        action="append",
        default=[],
        help="Persona expected by the design (for example KBD2); repeatable.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repeat_aliases: list[tuple[str, str]] = []
    for value in args.repeat_alias:
        if "=" not in value:
            raise SystemExit(f"Invalid --repeat-alias (expected PATH_MATCH=REPEAT_ID): {value}")
        path_match, label = value.split("=", 1)
        if not path_match or not label:
            raise SystemExit(f"Invalid --repeat-alias: {value}")
        repeat_aliases.append((path_match, label))
    interfaces = build_archive_interfaces(
        args.archive_root,
        expected_conditions=tuple(args.expected_condition),
        expected_personas=tuple(args.expected_persona),
        repeat_aliases=tuple(repeat_aliases),
    )
    paths = write_archive_interfaces(interfaces, args.out_dir.resolve())
    counts = interfaces.availability["counts"]
    print(
        f"Built archive interfaces for {counts['root_outer_runs']} root outer runs "
        f"({counts['linked_followup_branches']} linked follow-up branches)."
    )
    print(f"Manifest: {paths['manifest']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
