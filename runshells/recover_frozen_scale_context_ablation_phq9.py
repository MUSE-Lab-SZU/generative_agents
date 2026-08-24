#!/usr/bin/env python3
"""Recover missing PHQ-9 expert scores and write plot-ready batch summaries.

This utility is intentionally limited to an already-produced frozen context
ablation batch.  It never replays the simulation or regenerates answers:
missing expert scoring artifacts are recovered from the saved
``PHQ-9_answered.jsonl`` only.  Nodes without a complete answer sheet remain
explicitly unscored in the QC and summary outputs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev
from typing import Any

from run_frozen_scale_context_ablation_worker import direct_score, expert_score, extract_expert_total


EXPECTED_ANSWER_IDS = tuple(range(1, 11))


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line:
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"line {line_number} is not a JSON object")
            rows.append(value)
    return rows


def write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def numeric(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
        return float(value)
    return None


def validate_answers(path: Path) -> tuple[list[dict[str, Any]] | None, str]:
    if not path.is_file():
        return None, "missing"
    try:
        rows = load_jsonl(path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return None, f"invalid_jsonl:{type(exc).__name__}"
    ids = []
    for row in rows:
        value = row.get("id")
        if not isinstance(value, int) or isinstance(value, bool):
            return None, "invalid_item_id"
        if not str(row.get("answer", "") or "").strip():
            return None, f"empty_answer_item_{value}"
        ids.append(value)
    if tuple(ids) != EXPECTED_ANSWER_IDS:
        return None, "unexpected_answer_ids:" + ",".join(map(str, ids))
    return rows, "valid"


def load_direct_total(path: Path) -> tuple[float | None, str]:
    if not path.is_file():
        return None, "missing"
    try:
        payload = load_json(path)
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"invalid_json:{type(exc).__name__}"
    if not isinstance(payload, dict):
        return None, "invalid_payload"
    if payload.get("complete") is not True:
        return None, "incomplete"
    total = numeric(payload.get("total_score"))
    return (total, "valid") if total is not None else (None, "invalid_total")


def load_expert_total(path: Path) -> tuple[float | None, str]:
    if not path.is_file():
        return None, "missing"
    try:
        payload = load_json(path)
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"invalid_json:{type(exc).__name__}"
    if not isinstance(payload, dict):
        return None, "invalid_payload"
    total = numeric(extract_expert_total("PHQ-9", payload))
    return (total, "valid") if total is not None else (None, "invalid_score")


def attempt_expert_recovery(node_dir: Path, rows: list[dict[str, Any]]) -> tuple[bool, str]:
    """Score only a missing/invalid expert artifact; never touch answers."""

    score_path = node_dir / "PHQ-9_expert_scored.json"
    trace_path = node_dir / "PHQ-9_expert_scoring_trace.json"
    try:
        scored, trace = expert_score("PHQ-9", rows)
        total = numeric(extract_expert_total("PHQ-9", scored))
        if total is None:
            write_json(
                node_dir / "PHQ-9_expert_recovery_error.json",
                {
                    "status": "invalid_expert_response",
                    "generated_at": datetime.now().isoformat(timespec="seconds"),
                    "scored_payload": scored,
                },
            )
            return False, "invalid_expert_response"
        write_json(score_path, scored)
        write_json(trace_path, trace)
        return True, "recovered"
    except Exception as exc:  # keep the batch resumable after an API/model error
        write_json(
            node_dir / "PHQ-9_expert_recovery_error.json",
            {
                "status": "error",
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        )
        return False, f"recovery_error:{type(exc).__name__}"


def safe_job(node_dir: Path) -> tuple[dict[str, Any], str]:
    try:
        payload = load_json(node_dir / "job.json")
    except (OSError, json.JSONDecodeError) as exc:
        return {}, f"invalid_job:{type(exc).__name__}"
    return (payload, "valid") if isinstance(payload, dict) else ({}, "invalid_job_payload")


def grouped_score_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["group"], row["timepoint"], row["condition"])].append(row)
    result = []
    for (group, timepoint, condition), items in sorted(groups.items()):
        scores = [float(row["total_score"]) for row in items if row["total_score"] is not None]
        source_counts = Counter(row["score_source"] for row in items if row["total_score"] is not None)
        result.append(
            {
                "group": group,
                "timepoint": timepoint,
                "condition": condition,
                "n_evaluation_nodes": len(items),
                "n_valid_answers": sum(row["answer_status"] == "valid" for row in items),
                "n_missing_or_invalid_answers": sum(row["answer_status"] != "valid" for row in items),
                "n_expert_scored": sum(row["expert_score_status"] == "valid" for row in items),
                "n_direct_complete": sum(row["direct_score_status"] == "valid" for row in items),
                "n_plottable": len(scores),
                "score_source_counts": dict(sorted(source_counts.items())),
                "mean_total_score": mean(scores) if scores else None,
                "std_total_score": stdev(scores) if len(scores) >= 2 else None,
                "min_total_score": min(scores) if scores else None,
                "max_total_score": max(scores) if scores else None,
            }
        )
    return result


def paired_change_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cells: dict[tuple[str, int, int, str], dict[str, float]] = defaultdict(dict)
    for row in rows:
        total = row["total_score"]
        if total is None:
            continue
        key = (row["run_name"], row["outer_repeat"], row["ablation_repeat"], row["condition"])
        cells[key][row["timepoint"]] = float(total)
    grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for (run_name, _outer, _repeat, condition), values in cells.items():
        if "T0" not in values:
            continue
        group = next(row["group"] for row in rows if row["run_name"] == run_name)
        for timepoint, value in values.items():
            if timepoint != "T0":
                grouped[(group, timepoint, condition)].append(value - values["T0"])
    summary = []
    for (group, timepoint, condition), values in sorted(grouped.items()):
        summary.append(
            {
                "group": group,
                "timepoint": timepoint,
                "condition": condition,
                "n_paired_nodes": len(values),
                "mean_timepoint_minus_t0": mean(values),
                "std_timepoint_minus_t0": stdev(values) if len(values) >= 2 else None,
                "min_timepoint_minus_t0": min(values),
                "max_timepoint_minus_t0": max(values),
            }
        )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--recover-missing-expert",
        action="store_true",
        help="Call the existing ExpertLLM scorer for answer-valid nodes missing an expert score.",
    )
    parser.add_argument(
        "--recovery-limit",
        type=int,
        default=0,
        help="Maximum missing expert scores to recover in this invocation; 0 means no limit.",
    )
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    node_dirs = sorted(path.parent for path in output_root.rglob("job.json"))
    if not node_dirs:
        raise FileNotFoundError(f"no job.json found beneath {output_root}")

    recovered = []
    if args.recover_missing_expert:
        recovery_candidates = []
        for node_dir in node_dirs:
            rows, answer_status = validate_answers(node_dir / "PHQ-9_answered.jsonl")
            expert_total, expert_status = load_expert_total(node_dir / "PHQ-9_expert_scored.json")
            if answer_status == "valid" and expert_total is None:
                recovery_candidates.append((node_dir, rows or []))
        if args.recovery_limit > 0:
            recovery_candidates = recovery_candidates[: args.recovery_limit]
        for node_dir, rows in recovery_candidates:
            ok, detail = attempt_expert_recovery(node_dir, rows)
            recovered.append({"node_dir": str(node_dir.relative_to(output_root)), "status": detail, "ok": ok})

    node_rows: list[dict[str, Any]] = []
    for node_dir in node_dirs:
        job, job_status = safe_job(node_dir)
        answer_rows, answer_status = validate_answers(node_dir / "PHQ-9_answered.jsonl")
        direct_total, direct_status = load_direct_total(node_dir / "PHQ-9_direct_scored.json")
        expert_total, expert_status = load_expert_total(node_dir / "PHQ-9_expert_scored.json")
        if expert_total is not None:
            total_score, score_source = expert_total, "expert"
        elif direct_total is not None:
            total_score, score_source = direct_total, "direct"
        else:
            total_score, score_source = None, "missing"
        node_rows.append(
            {
                "node_dir": str(node_dir.relative_to(output_root)),
                "run_name": str(job.get("run_name", node_dir.parents[3].name)),
                "group": str(job.get("group", "unknown")),
                "outer_repeat": int(job.get("outer_repeat", 0) or 0),
                "ablation_repeat": int(job.get("ablation_repeat", 0) or 0),
                "timepoint": str(job.get("timepoint", "unknown")),
                "condition": str(job.get("condition", "unknown")),
                "job_status": job_status,
                "answer_status": answer_status,
                "answer_item_count": len(answer_rows) if answer_rows is not None else 0,
                "direct_score_status": direct_status,
                "direct_total_score": direct_total,
                "expert_score_status": expert_status,
                "expert_total_score": expert_total,
                "total_score": total_score,
                "score_source": score_source,
            }
        )

    node_rows.sort(key=lambda row: (row["group"], row["run_name"], row["timepoint"], row["condition"], row["ablation_repeat"]))
    plot_summary = grouped_score_summary(node_rows)
    change_summary = paired_change_summary(node_rows)
    qc = {
        "evaluation_node_count": len(node_rows),
        "valid_answer_count": sum(row["answer_status"] == "valid" for row in node_rows),
        "missing_or_invalid_answer_count": sum(row["answer_status"] != "valid" for row in node_rows),
        "expert_scored_count": sum(row["expert_score_status"] == "valid" for row in node_rows),
        "direct_complete_count": sum(row["direct_score_status"] == "valid" for row in node_rows),
        "plottable_score_count": sum(row["total_score"] is not None for row in node_rows),
        "unscored_answer_valid_count": sum(
            row["answer_status"] == "valid" and row["total_score"] is None for row in node_rows
        ),
    }
    summary = {
        "schema_version": 1,
        "artifact_kind": "frozen_scale_context_ablation_phq9_summary",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "output_root": str(output_root),
        "score_preference": ["expert", "direct"],
        "score_recovery": {
            "requested": bool(args.recover_missing_expert),
            "attempted_count": len(recovered),
            "recovered_count": sum(item["ok"] for item in recovered),
            "attempts": recovered,
        },
        "qc": qc,
        "plot_summary": plot_summary,
        "paired_change_summary": change_summary,
        "node_scores": node_rows,
    }
    write_json(output_root / "summary.json", summary)
    node_fields = list(node_rows[0])
    write_csv(output_root / "phq9_node_scores.csv", node_rows, node_fields)
    plot_fields = list(plot_summary[0]) if plot_summary else ["group", "timepoint", "condition"]
    write_csv(output_root / "phq9_plot_summary.csv", plot_summary, plot_fields)
    change_fields = list(change_summary[0]) if change_summary else ["group", "timepoint", "condition"]
    write_csv(output_root / "phq9_changes_summary.csv", change_summary, change_fields)
    print(json.dumps({"qc": qc, "score_recovery": summary["score_recovery"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
