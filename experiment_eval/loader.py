"""Discovery, schema validation, and loading of repeat-summary JSON files."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from statistics import mean
from typing import Any

from .schema import (
    EXPECTED_AGGREGATION_METHOD_VERSION,
    EXPECTED_PRIMARY_SCORE,
    ExperimentRecord,
    as_float,
    group_sort_key,
    label_sort_key,
    normalize_group,
    normalize_repeat_id,
    scale_sort_key,
    severity_sort_key,
    stable_component,
)
from .statistics import category_metrics, descriptive_statistics, item_statistics


DEFAULT_TOLERANCE = 1e-8

# These are intentionally kept here instead of importing the runnable
# ``runshells`` module: archived reports must remain loadable when only the
# analysis package is installed.  The names mirror runshells.scale_protocol.
INTERMEDIATE_SHORT_SCALE_NAMES = {
    "总体抑郁水平及干扰程度量表",
    "总体焦虑水平及干扰程度量表",
}


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise ValueError(f"Top-level JSON must be an object: {path}")
    return payload


def display_path(path: Path, project_root: Path | None = None) -> str:
    if project_root is not None:
        try:
            return str(path.relative_to(project_root))
        except ValueError:
            pass
    return str(path)


def _local_source_summary(
    path: Path, payload: dict[str, Any]
) -> tuple[Path | None, dict[str, Any] | None]:
    """Resolve an archived ``source_original_summary`` by basename.

    Historical summaries commonly retain absolute paths from the remote
    machine.  Relocating an archive changes the prefix, but not the report
    basename.  A basename is accepted only when it resolves uniquely beside
    the repeat summary and the referenced artifact declares its kind.
    """
    source = str(payload.get("source_original_summary") or "").strip()
    if not source:
        return None, None
    basename = Path(source).name
    matches = [
        candidate
        for candidate in path.parent.glob(f"**/{basename}")
        if candidate.is_file()
    ]
    if len(matches) != 1:
        return None, None
    try:
        source_payload = read_json(matches[0])
    except (OSError, json.JSONDecodeError, ValueError):
        return matches[0], None
    return matches[0], source_payload


def resolve_followup_source_summary(
    path: Path, payload: dict[str, Any]
) -> tuple[Path | None, dict[str, Any] | None]:
    """Resolve a follow-up source even when its transient absolute path expired.

    Older ``followup_repeat_watch`` summaries reference an intermediate
    ``sources/*_source.json`` file that was not always archived.  The canonical
    ``post_sim_followup_summary`` is still present and can be matched without
    ambiguity by the branch ``run_name`` recorded in both artifacts.
    """

    source_path, source_payload = _local_source_summary(path, payload)
    if source_payload and source_payload.get("artifact_kind") == "post_sim_followup_summary":
        return source_path, source_payload
    branch_names = {
        str(condition.get("run_name"))
        for condition in (payload.get("conditions") or [])
        if isinstance(condition, dict) and condition.get("run_name")
    }
    if not branch_names:
        return None, None
    if not all(run_name.startswith("followup-") for run_name in branch_names):
        return None, None
    if path.name.startswith("repeat-") and path.name.endswith(
        "-followup_summary.json"
    ):
        canonical_name = (
            "followup-"
            + path.name[len("repeat-") : -len("-followup_summary.json")]
            + "_summary.json"
        )
        canonical = path.parent / canonical_name
        if canonical.is_file():
            try:
                canonical_payload = read_json(canonical)
            except (OSError, json.JSONDecodeError, ValueError):
                canonical_payload = {}
            canonical_names = {
                str(condition.get("run_name"))
                for condition in (canonical_payload.get("conditions") or [])
                if isinstance(condition, dict) and condition.get("run_name")
            }
            if (
                canonical_payload.get("artifact_kind")
                == "post_sim_followup_summary"
                and branch_names & canonical_names
            ):
                return canonical, canonical_payload
    matches: list[tuple[Path, dict[str, Any]]] = []
    for candidate in path.parent.glob("**/*_summary.json"):
        if candidate == path or not candidate.is_file():
            continue
        try:
            candidate_payload = read_json(candidate)
        except (OSError, json.JSONDecodeError, ValueError):
            continue
        if candidate_payload.get("artifact_kind") != "post_sim_followup_summary":
            continue
        candidate_names = {
            str(condition.get("run_name"))
            for condition in (candidate_payload.get("conditions") or [])
            if isinstance(condition, dict) and condition.get("run_name")
        }
        if branch_names & candidate_names:
            matches.append((candidate, candidate_payload))
    return matches[0] if len(matches) == 1 else (None, None)


def is_followup_repeat_summary(
    path: Path, payload: dict[str, Any] | None = None
) -> bool:
    """Return true only for a repeat summary with explicit follow-up provenance."""
    payload = payload or read_json(path)
    condition_runs = [
        str(condition.get("run_name") or "")
        for condition in (payload.get("conditions") or [])
        if isinstance(condition, dict)
    ]
    if (
        path.name.startswith("repeat-")
        and path.name.endswith("-followup_summary.json")
        and condition_runs
        and all(run_name.startswith("followup-") for run_name in condition_runs)
    ):
        return True
    _, source_payload = resolve_followup_source_summary(path, payload)
    return bool(
        source_payload
        and source_payload.get("artifact_kind") == "post_sim_followup_summary"
    )


def find_report_files(
    reports_dir: Path, recursive: bool, *, include_followup: bool = False
) -> list[Path]:
    pattern = "**/*_summary.json" if recursive else "*_summary.json"
    candidates = sorted(
        path
        for path in reports_dir.glob(pattern)
        if path.is_file() and "repeat-" in path.name
    )
    by_batch: dict[str, list[Path]] = {}
    unreadable: list[str] = []
    for path in candidates:
        try:
            payload = read_json(path)
            if not include_followup and is_followup_repeat_summary(path, payload):
                continue
            batch_name = str(
                payload.get("batch_name") or path.stem.removesuffix("_summary")
            )
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            unreadable.append(f"{path}: {exc}")
            continue
        by_batch.setdefault(batch_name, []).append(path)
    conflicts = {name: paths for name, paths in by_batch.items() if len(paths) > 1}
    if conflicts:
        details = "; ".join(
            f"{name}: {', '.join(str(path) for path in paths)}"
            for name, paths in conflicts.items()
        )
        raise ValueError(
            "Duplicate batch_name values found across report files; use explicit --report-file selection. "
            + details
        )
    if unreadable:
        raise ValueError("Unreadable repeat summary files: " + "; ".join(unreadable))
    return sorted(paths[0] for paths in by_batch.values())


def parse_repeat_id(path: Path, batch_name: str) -> str:
    for text in (batch_name, path.stem.removesuffix("_summary")):
        match = re.search(r"-(?:\d{4})-(\d+)$", text)
        if match:
            return f"R{int(match.group(1)):02d}"
    for text in (batch_name, path.stem.removesuffix("_summary")):
        match = re.search(r"-rep(?:eat)?[-_]?0*(\d+)$", text, re.IGNORECASE)
        if match:
            return f"R{int(match.group(1)):02d}"
    for text in (batch_name, path.stem.removesuffix("_summary")):
        if "repeat" not in text.lower():
            continue
        match = re.search(r"-(\d{2})$", text)
        if match:
            return f"R{int(match.group(1)):02d}"
    return stable_component(batch_name or path.stem).upper()


def parse_group_severity(path: Path, payload: dict[str, Any], condition: dict[str, Any]) -> tuple[str, str]:
    condition_texts = [
        str(condition.get("condition_name") or ""),
        str(condition.get("run_name") or ""),
        str(condition.get("source_condition_name") or ""),
    ]
    match = re.search(r"KBD\d+-(G\d+)-(MILD|MOD|SEV)", " ".join(condition_texts), re.IGNORECASE)
    if match:
        return normalize_group(match.group(1)), match.group(2).upper()
    raw_group = str(condition.get("group") or condition.get("source_group") or "").strip()
    raw = str(condition.get("severity") or "UNKNOWN").strip().lower()
    severity = {"mild": "MILD", "moderate": "MOD", "mod": "MOD", "severe": "SEV", "sev": "SEV"}.get(
        raw, raw.upper()
    )
    if raw_group:
        return normalize_group(raw_group), severity
    fallback = re.search(
        r"KBD\d+-(G\d+)-(MILD|MOD|SEV)",
        f"{payload.get('batch_name', '')} {path.name}",
        re.IGNORECASE,
    )
    if fallback:
        return normalize_group(fallback.group(1)), fallback.group(2).upper()
    return "UNKNOWN", severity


def parse_kbd(path: Path, payload: dict[str, Any], condition: dict[str, Any]) -> str:
    condition_texts = [
        str(condition.get("condition_name") or ""),
        str(condition.get("run_name") or ""),
        str(condition.get("variant") or ""),
    ]
    match = re.search(r"KBD\s*0*(\d+)", " ".join(condition_texts), re.IGNORECASE)
    if match:
        return f"KBD{int(match.group(1))}"
    fallback = re.search(r"KBD\s*0*(\d+)", f"{payload.get('batch_name', '')} {path.name}", re.IGNORECASE)
    return f"KBD{int(fallback.group(1))}" if fallback else "UNKNOWN"


def _explicit_pairing(payload: dict[str, Any]) -> tuple[bool, str]:
    protocol = payload.get("evaluation_protocol") or {}
    metadata = payload.get("metadata") or {}
    for location, container in (("evaluation_protocol", protocol), ("metadata", metadata)):
        for key in ("repeats_paired_across_timepoints", "paired_across_timepoints"):
            if container.get(key) is True:
                return True, f"{location}.{key}=true"
        pairing = str(container.get("repeat_pairing") or container.get("timepoint_repeat_pairing") or "").lower()
        if pairing in {"paired", "paired_across_timepoints", "same_repeat_paired"}:
            return True, f"{location}.repeat_pairing={pairing}"
    return False, "No explicit metadata declaration of cross-timepoint pairing"


def _require_protocol(payload: dict[str, Any], path: Path) -> tuple[int, int | None, bool, str]:
    version = str(payload.get("aggregation_method_version") or "")
    if version != EXPECTED_AGGREGATION_METHOD_VERSION:
        raise ValueError(
            f"{path}: expected aggregation_method_version={EXPECTED_AGGREGATION_METHOD_VERSION!r}, "
            f"got {version or '<missing>'!r}"
        )
    protocol = payload.get("evaluation_protocol")
    if not isinstance(protocol, dict):
        raise ValueError(f"{path}: missing evaluation_protocol object")
    expected = protocol.get("expected_repeats")
    if not isinstance(expected, int) or expected < 2:
        raise ValueError(f"{path}: evaluation_protocol.expected_repeats must be an integer >= 2")
    intermediate_expected = protocol.get("intermediate_scale_repeats")
    if intermediate_expected is not None and (
        not isinstance(intermediate_expected, int) or intermediate_expected < 2
    ):
        raise ValueError(
            f"{path}: evaluation_protocol.intermediate_scale_repeats must be an integer >= 2"
        )
    if protocol.get("strict_complete_k") is not True:
        raise ValueError(f"{path}: evaluation_protocol.strict_complete_k must be true")
    primary = str(protocol.get("primary_score") or "")
    if primary != EXPECTED_PRIMARY_SCORE:
        raise ValueError(
            f"{path}: evaluation_protocol.primary_score must be {EXPECTED_PRIMARY_SCORE!r}, got {primary!r}"
        )
    paired, evidence = _explicit_pairing(payload)
    return expected, intermediate_expected, paired, evidence


def _assert_close(path: Path, context: str, field: str, actual: Any, expected: Any, tolerance: float) -> None:
    actual_number = as_float(actual)
    expected_number = as_float(expected)
    if actual_number is None or expected_number is None:
        if actual_number != expected_number:
            raise ValueError(f"{path}: {context}.{field} mismatch: JSON={actual!r}, recomputed={expected!r}")
        return
    if not math.isclose(actual_number, expected_number, rel_tol=tolerance, abs_tol=tolerance):
        raise ValueError(
            f"{path}: {context}.{field} mismatch: JSON={actual_number:.12g}, recomputed={expected_number:.12g}"
        )


def _summarize_scale(
    path: Path,
    context: str,
    scale_name: str,
    scale_payload: dict[str, Any],
    protocol_expected: int,
    intermediate_protocol_expected: int | None,
    tolerance: float,
) -> dict[str, Any]:
    expected = scale_payload.get("expected_repeats")
    valid = scale_payload.get("valid_repeats")
    expected_for_scale = (
        intermediate_protocol_expected
        if scale_name in INTERMEDIATE_SHORT_SCALE_NAMES and intermediate_protocol_expected is not None
        else protocol_expected
    )
    if expected != expected_for_scale:
        raise ValueError(
            f"{path}: {context}.expected_repeats={expected!r}, "
            f"protocol requires {expected_for_scale}"
        )
    if valid != expected:
        raise ValueError(f"{path}: {context}.valid_repeats={valid!r} must equal expected_repeats={expected!r}")
    if scale_payload.get("aggregation_status") != "complete":
        raise ValueError(f"{path}: {context}.aggregation_status must be 'complete'")

    repeat_runs = scale_payload.get("repeat_runs")
    if not isinstance(repeat_runs, list):
        raise ValueError(f"{path}: {context}.repeat_runs must be present for recomputation")
    ok_runs = [run for run in repeat_runs if isinstance(run, dict) and run.get("status") == "ok"]
    if len(ok_runs) != expected:
        raise ValueError(f"{path}: {context} has {len(ok_runs)} ok repeat_runs; expected {expected}")
    repeat_ids = [str(run.get("repeat")) for run in ok_runs]
    if len(repeat_ids) != len(set(repeat_ids)):
        raise ValueError(f"{path}: {context} contains duplicate repeat identifiers")
    values = [as_float(run.get("total_score")) for run in ok_runs]
    if any(value is None for value in values):
        raise ValueError(f"{path}: {context} contains an ok repeat without total_score")
    clean_values = [float(value) for value in values if value is not None]
    recomputed = descriptive_statistics(clean_values)
    ci_json = scale_payload.get("ci95") or {}
    if not isinstance(ci_json, dict):
        raise ValueError(f"{path}: {context}.ci95 must be an object")
    comparisons = {
        "total_score": recomputed["mean"],
        "mean": recomputed["mean"],
        "sample_sd": recomputed["sample_sd"],
        "median": recomputed["median"],
        "q1": recomputed["q1"],
        "q3": recomputed["q3"],
        "iqr": recomputed["iqr"],
    }
    for field, expected_value in comparisons.items():
        _assert_close(path, context, field, scale_payload.get(field), expected_value, tolerance)
    _assert_close(path, context, "ci95.lower", ci_json.get("lower"), recomputed["ci95"]["lower"], tolerance)
    _assert_close(path, context, "ci95.upper", ci_json.get("upper"), recomputed["ci95"]["upper"], tolerance)
    if str(ci_json.get("method") or "") != "student_t":
        raise ValueError(f"{path}: {context}.ci95.method must be 'student_t'")

    categories = [str(run.get("severity") or "") for run in ok_runs]
    category = category_metrics(categories)
    for field in ("modal_confidence", "category_pairwise_flip_rate"):
        if scale_payload.get(field) is not None:
            _assert_close(path, context, field, scale_payload.get(field), category[field], tolerance)
    if scale_payload.get("any_category_flip") is not None and bool(scale_payload.get("any_category_flip")) != category[
        "any_category_flip"
    ]:
        raise ValueError(f"{path}: {context}.any_category_flip mismatch")

    item_rows: list[list[float]] = []
    item_values_by_repeat: dict[str, list[float]] = {}
    expected_item_count: int | None = None
    for run in ok_runs:
        raw_items = run.get("item_scores")
        if not isinstance(raw_items, list) or not raw_items:
            raise ValueError(f"{path}: {context} repeat {run.get('repeat')} is missing item_scores")
        items = [as_float(value) for value in raw_items]
        if any(value is None for value in items):
            raise ValueError(f"{path}: {context} repeat {run.get('repeat')} has invalid item_scores")
        item_values = [float(value) for value in items if value is not None]
        if expected_item_count is None:
            expected_item_count = len(item_values)
        elif len(item_values) != expected_item_count:
            raise ValueError(
                f"{path}: {context} repeat {run.get('repeat')} has {len(item_values)} items; "
                f"expected {expected_item_count}"
            )
        if not math.isclose(sum(item_values), float(run["total_score"]), abs_tol=tolerance, rel_tol=tolerance):
            raise ValueError(f"{path}: {context} repeat {run.get('repeat')} item_scores do not sum to total_score")
        item_rows.append(item_values)
        item_values_by_repeat[str(run.get("repeat"))] = item_values

    trimmed = mean(sorted(clean_values)[1:-1]) if len(clean_values) > 2 else None
    return {
        "score": recomputed["mean"],
        "total_score": recomputed["mean"],
        "repeat_mean": recomputed["mean"],
        "score_semantics": "mean_of_complete_reviewed_scale_totals",
        "sample_sd": recomputed["sample_sd"],
        "stddev": recomputed["sample_sd"],
        "ci95_lower": recomputed["ci95"]["lower"],
        "ci95_upper": recomputed["ci95"]["upper"],
        "ci95_width": recomputed["ci95"]["upper"] - recomputed["ci95"]["lower"],
        "ci95_method": "student_t",
        "median": recomputed["median"],
        "q1": recomputed["q1"],
        "q3": recomputed["q3"],
        "iqr": recomputed["iqr"],
        "trimmed_mean_drop_one_each": trimmed,
        "min": min(clean_values),
        "max": max(clean_values),
        "n": len(clean_values),
        "expected_repeats": expected,
        "valid_repeats": valid,
        "missing_repeats": scale_payload.get("missing_repeats") or [],
        "aggregation_status": "complete",
        "values": clean_values,
        "values_by_repeat": dict(zip(repeat_ids, clean_values)),
        "item_values_by_repeat": item_values_by_repeat,
        "score_source": scale_payload.get("score_source", ""),
        "severity": scale_payload.get("severity", ""),
        "category_counts": category["category_counts"],
        "modal_confidence": category["modal_confidence"],
        "category_pairwise_flip_rate": category["category_pairwise_flip_rate"],
        "any_category_flip": category["any_category_flip"],
        "item_statistics": item_statistics(item_rows),
    }


def load_records(
    paths: list[Path],
    group_aliases: list[tuple[str, str]] | None = None,
    *,
    repeat_aliases: list[tuple[str, str]] | None = None,
    tolerance: float = DEFAULT_TOLERANCE,
) -> tuple[list[ExperimentRecord], list[str], list[str]]:
    records: list[ExperimentRecord] = []
    discovered_labels: list[str] = []
    discovered_scales: set[str] = set()
    seen_batch_paths: dict[str, Path] = {}

    for path in paths:
        payload = read_json(path)
        protocol_expected, intermediate_protocol_expected, paired, pairing_evidence = _require_protocol(payload, path)
        batch_name = str(payload.get("batch_name") or path.stem.removesuffix("_summary"))
        if batch_name in seen_batch_paths and seen_batch_paths[batch_name] != path:
            raise ValueError(
                f"Duplicate batch_name {batch_name!r}: {seen_batch_paths[batch_name]} and {path}; "
                "select one explicitly"
            )
        seen_batch_paths[batch_name] = path
        repeat_matches = [
            label for path_match, label in (repeat_aliases or []) if path_match in str(path)
        ]
        if len(repeat_matches) > 1:
            raise ValueError(f"Multiple repeat aliases match report: {path}")
        repeat_id = (
            normalize_repeat_id(repeat_matches[0])
            if repeat_matches
            else parse_repeat_id(path, batch_name)
        )
        matches = [label for path_match, label in (group_aliases or []) if path_match in str(path)]
        if len(matches) > 1:
            raise ValueError(f"Multiple group aliases match report: {path}")
        conditions = [
            condition
            for condition in (payload.get("conditions") or [])
            if isinstance(condition, dict) and isinstance(condition.get("evaluations"), list)
        ]
        if not conditions:
            raise ValueError(f"{path}: no usable conditions")

        for condition in conditions:
            source_group, severity = parse_group_severity(path, payload, condition)
            group = normalize_group(matches[0]) if matches else source_group
            kbd = parse_kbd(path, payload, condition)
            condition_name = str(condition.get("condition_name") or "UNKNOWN")
            stable_id = "::".join(
                [kbd, stable_component(condition_name), group, severity, repeat_id]
            )
            series: dict[str, dict[str, dict[str, Any]]] = {}
            for evaluation in condition.get("evaluations") or []:
                label = str(evaluation.get("trigger_label") or "").strip()
                if not label:
                    continue
                if label not in discovered_labels:
                    discovered_labels.append(label)
                scales = evaluation.get("scales") or {}
                if not isinstance(scales, dict):
                    raise ValueError(f"{path}: {condition_name}/{label}.scales must be an object")
                for scale, scale_payload in scales.items():
                    if not isinstance(scale_payload, dict):
                        continue
                    scale_name = str(scale)
                    context = f"{condition_name}/{label}/{scale_name}"
                    stats = _summarize_scale(
                        path,
                        context,
                        scale_name,
                        scale_payload,
                        protocol_expected,
                        intermediate_protocol_expected,
                        tolerance,
                    )
                    stats["completed_session_count"] = evaluation.get("completed_session_count")
                    stats["sim_time"] = evaluation.get("sim_time")
                    series.setdefault(scale_name, {})[label] = stats
                    discovered_scales.add(scale_name)
            records.append(
                ExperimentRecord(
                    stable_id=stable_id,
                    key=stable_id,
                    kbd=kbd,
                    group=group,
                    source_group=source_group,
                    severity=severity,
                    repeat_id=repeat_id,
                    plot_label=group,
                    batch_name=batch_name,
                    path=path,
                    condition_name=condition_name,
                    run_name=str(condition.get("run_name") or ""),
                    series=series,
                    expected_repeats=protocol_expected,
                    repeats_paired_across_timepoints=paired,
                    pairing_evidence=pairing_evidence,
                )
            )

    identifiers = [record.stable_id for record in records]
    duplicates = sorted({identifier for identifier in identifiers if identifiers.count(identifier) > 1})
    if duplicates:
        raise ValueError("Duplicate stable experiment identifiers: " + ", ".join(duplicates))
    labels = sorted(discovered_labels, key=lambda label: label_sort_key(label, discovered_labels))
    scales = sorted(discovered_scales, key=scale_sort_key)
    records.sort(
        key=lambda record: (
            group_sort_key(record.kbd),
            severity_sort_key(record.severity),
            group_sort_key(record.group),
            group_sort_key(record.repeat_id),
            record.condition_name,
        )
    )
    return records, labels, scales


def available_groups(records: list[ExperimentRecord]) -> list[str]:
    return sorted({record.group for record in records}, key=group_sort_key)


def available_severities(records: list[ExperimentRecord]) -> list[str]:
    return sorted({record.severity for record in records}, key=severity_sort_key)


def available_kbds(records: list[ExperimentRecord]) -> list[str]:
    return sorted({record.kbd for record in records}, key=group_sort_key)


def records_for_group(records: list[ExperimentRecord], group: str) -> list[ExperimentRecord]:
    return [record.with_plot_label(record.repeat_id) for record in records if record.group == group]


def records_for_repeat(records: list[ExperimentRecord], repeat_id: str) -> list[ExperimentRecord]:
    return [record.with_plot_label(record.group) for record in records if record.repeat_id == repeat_id]


def records_for_all_comparison(records: list[ExperimentRecord]) -> list[ExperimentRecord]:
    labels = [f"{record.group}-{record.repeat_id}" for record in records]
    duplicated = {label for label in labels if labels.count(label) > 1}
    return [
        record.with_plot_label(
            f"{record.group}-{record.repeat_id}-{stable_component(record.condition_name)}"
            if f"{record.group}-{record.repeat_id}" in duplicated
            else f"{record.group}-{record.repeat_id}"
        )
        for record in records
    ]
