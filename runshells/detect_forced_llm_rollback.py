#!/usr/bin/env python3
"""Detect sustained forced-LLM failures and recommend a safe resume anchor.

The detector is intentionally read-only.  Simulation traces are authoritative;
runtime logs only supplement meetings that have not reached a trace sidecar.
When the simulation is clean, archived repeat-evaluation traces are checked
independently so a bad repeat job can be rerun without rolling back simulation.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import prepare_sparse_checkpoint_resume as recovery


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_ROOT = BASE_DIR / "results"
FORCED_SIDECAR_SUBDIR = (
    Path("trace_state_sidecars") / "forced_prompt_trace_state"
)
MEETING_ID_RE = re.compile(r"\bmeeting_id=([^\s=]+)")
AGENT_ROUTE_RE = re.compile(
    r"\broute=(forced_llm|default)\s+fallback=(True|False)\s+reason=([^\s]+)"
)
LLM_CALL_ERROR_RE = re.compile(r"\[LLM_CALL_ERROR\]|forced_llm_(?:error|failed)")
FAILURE_TOKEN_RE = re.compile(
    r"(?:fallback|error|failed|failure|empty|exception|unavailable|json_parse_failed)",
    re.IGNORECASE,
)


@dataclass
class UnitStats:
    unit_id: str
    source: str
    total_calls: int
    failed_calls: int
    failure_ratio: float
    is_mass_failure: bool
    meeting_id: str = ""
    step_time: str = ""
    path: str = ""


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"JSON 文件无效: {path}: {exc}") from exc


def snapshot_step(path: Path) -> int:
    payload = load_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"snapshot JSON 顶层必须是对象: {path}")
    raw = payload.get("step")
    if isinstance(raw, bool):
        raise ValueError(f"snapshot step 无效: {path}")
    try:
        return int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"snapshot step 无效: {path}") from exc


def _text(value: Any) -> str:
    return str(value or "").strip()


def classify_forced_trace_record(record: Any) -> str | None:
    """Return ``success``, ``failure``, or ``None`` for one trace record."""

    if not isinstance(record, dict):
        return None
    meta = record.get("meta", {})
    if not isinstance(meta, dict):
        meta = {}
    role = _text(record.get("role")).lower()
    route = _text(meta.get("route")).lower()
    reason = _text(meta.get("reason")).lower()
    source = _text(meta.get("source")).lower()
    error = _text(meta.get("error"))

    if route == "think_llm":
        return None

    forced_context = (
        route == "forced_llm"
        or reason.startswith("forced_llm_")
        or "forced_llm" in source
        or role.endswith("_llm")
        or role
        in {
            "patient",
            "doctor",
            "state_tracker_llm",
            "dialog_judge_llm",
            "session_eval_llm",
            "consult_history_llm",
        }
    )
    failure = bool(error) or bool(
        FAILURE_TOKEN_RE.search(" ".join((route, reason, source)))
    )
    if failure and forced_context:
        return "failure"
    if not forced_context:
        return None
    if reason == "forced_llm_success" or route == "forced_llm":
        return "success"
    if role.endswith("_llm") and source and not FAILURE_TOKEN_RE.search(source):
        return "success"
    return None


def make_unit_stats(
    *,
    unit_id: str,
    source: str,
    outcomes: Iterable[str],
    min_calls: int,
    failure_ratio: float,
    meeting_id: str = "",
    step_time: str = "",
    path: str = "",
) -> UnitStats:
    values = list(outcomes)
    total = len(values)
    failed = sum(item == "failure" for item in values)
    ratio = failed / total if total else 0.0
    return UnitStats(
        unit_id=unit_id,
        source=source,
        total_calls=total,
        failed_calls=failed,
        failure_ratio=round(ratio, 6),
        is_mass_failure=bool(total >= min_calls and ratio >= failure_ratio),
        meeting_id=meeting_id,
        step_time=step_time,
        path=path,
    )


def sidecar_files(checkpoint_dir: Path) -> list[tuple[int, Path]]:
    sidecar_dir = checkpoint_dir / FORCED_SIDECAR_SUBDIR
    candidates: list[tuple[int, Path]] = []
    for path in sidecar_dir.glob("simulate-*.json"):
        snapshot_path = checkpoint_dir / path.name
        if not snapshot_path.is_file():
            continue
        try:
            candidates.append((snapshot_step(snapshot_path), path))
        except ValueError:
            continue
    return sorted(candidates, key=lambda item: (item[0], item[1].name))


def extract_sidecar_units(
    sidecar_path: Path,
    *,
    min_calls: int,
    failure_ratio: float,
) -> list[UnitStats]:
    payload = load_json(sidecar_path)
    if not isinstance(payload, dict):
        raise ValueError(f"forced trace sidecar 顶层必须是对象: {sidecar_path}")
    sessions = payload.get("sessions", [])
    if not isinstance(sessions, list):
        raise ValueError(f"forced trace sidecar sessions 必须是列表: {sidecar_path}")

    units: list[UnitStats] = []
    for index, session in enumerate(sessions, start=1):
        if not isinstance(session, dict):
            continue
        meeting = session.get("meeting", {})
        if not isinstance(meeting, dict):
            meeting = {}
        meeting_id = _text(meeting.get("meeting_id"))
        unit_id = meeting_id or f"meeting_unknown_{index:03d}"
        records = session.get("records", [])
        outcomes = []
        if isinstance(records, list):
            outcomes = [
                outcome
                for outcome in (classify_forced_trace_record(item) for item in records)
                if outcome is not None
            ]
        units.append(
            make_unit_stats(
                unit_id=unit_id,
                source="simulation_sidecar",
                outcomes=outcomes,
                min_calls=min_calls,
                failure_ratio=failure_ratio,
                meeting_id=meeting_id,
                step_time=_text(meeting.get("step_time")),
                path=str(sidecar_path),
            )
        )
    return units


def extract_unsaved_log_units(
    log_paths: Iterable[Path],
    *,
    saved_meeting_ids: set[str],
    min_calls: int,
    failure_ratio: float,
) -> list[UnitStats]:
    outcomes_by_meeting: dict[str, list[str]] = {}
    order: list[str] = []
    current_meeting = ""

    for log_path in log_paths:
        if not log_path.is_file():
            continue
        with log_path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                meeting_match = MEETING_ID_RE.search(line)
                if meeting_match:
                    candidate = meeting_match.group(1).strip()
                    if candidate:
                        current_meeting = candidate
                        if (
                            current_meeting not in saved_meeting_ids
                            and current_meeting not in outcomes_by_meeting
                        ):
                            outcomes_by_meeting[current_meeting] = []
                            order.append(current_meeting)
                if not current_meeting or current_meeting in saved_meeting_ids:
                    continue
                route_match = AGENT_ROUTE_RE.search(line)
                if route_match:
                    route, fallback, reason = route_match.groups()
                    if reason == "forced_llm_success" and route == "forced_llm":
                        outcomes_by_meeting[current_meeting].append("success")
                    elif fallback == "True" and reason.startswith("forced_llm_"):
                        outcomes_by_meeting[current_meeting].append("failure")
                    continue
                if LLM_CALL_ERROR_RE.search(line):
                    outcomes_by_meeting[current_meeting].append("failure")

    return [
        make_unit_stats(
            unit_id=meeting_id,
            source="simulation_log",
            outcomes=outcomes_by_meeting.get(meeting_id, []),
            min_calls=min_calls,
            failure_ratio=failure_ratio,
            meeting_id=meeting_id,
        )
        for meeting_id in order
    ]


def first_sustained_failure(
    units: list[UnitStats],
    consecutive: int,
) -> tuple[int, list[UnitStats]] | None:
    needed = max(1, int(consecutive))
    meeting_positions = [
        index for index, item in enumerate(units) if item.meeting_id
    ]
    meeting_units = [units[index] for index in meeting_positions]
    for start in range(0, len(meeting_units) - needed + 1):
        window = meeting_units[start : start + needed]
        if all(item.is_mass_failure for item in window):
            return meeting_positions[start], window
    return None


def first_snapshot_containing_meeting(
    checkpoint_dir: Path,
    meeting_id: str,
) -> tuple[int, str] | None:
    if not meeting_id:
        return None
    for step_no, path in sidecar_files(checkpoint_dir):
        payload = load_json(path)
        sessions = payload.get("sessions", []) if isinstance(payload, dict) else []
        for session in sessions if isinstance(sessions, list) else []:
            meeting = session.get("meeting", {}) if isinstance(session, dict) else {}
            if isinstance(meeting, dict) and _text(meeting.get("meeting_id")) == meeting_id:
                return step_no, path.name
    return None


def anchor_payload(anchor: recovery.RecoveryAnchor) -> dict[str, Any]:
    return {
        "label": anchor.label,
        "step_no": anchor.step_no,
        "sim_time": anchor.sim_time,
        "snapshot_name": anchor.snapshot_name,
        "job_path": str(anchor.job_path),
        "manifest_path": str(anchor.manifest_path),
    }


def classify_repeat_trace_item(item: Any) -> str | None:
    if not isinstance(item, dict):
        return None
    trace = item.get("trace", {})
    if not isinstance(trace, dict):
        return None
    route = _text(trace.get("llm_route")).lower()
    reason = _text(trace.get("llm_route_reason")).lower()
    if not route:
        return None
    if route == "forced_llm" and reason in {"", "success"}:
        return "success"
    if "forced" in route and (
        "fallback" in route or FAILURE_TOKEN_RE.search(reason)
    ):
        return "failure"
    return None


def inspect_repeat_eval(
    condition_dir: Path | None,
    *,
    min_calls: int,
    failure_ratio: float,
) -> dict[str, Any]:
    if condition_dir is None:
        return {
            "checked": False,
            "condition_dir": "",
            "reason": "not_configured",
            "units": [],
            "bad_unit_dirs": [],
        }
    if not condition_dir.is_dir():
        return {
            "checked": False,
            "condition_dir": str(condition_dir),
            "reason": "directory_not_found",
            "units": [],
            "bad_unit_dirs": [],
        }

    units: list[UnitStats] = []
    for label_dir in sorted(
        {
            path.parent
            for path in condition_dir.glob("r*/*/*_trace.json")
            if path.is_file()
        },
        key=lambda path: path.as_posix(),
    ):
        outcomes: list[str] = []
        for trace_path in sorted(label_dir.glob("*_trace.json")):
            payload = load_json(trace_path)
            if not isinstance(payload, list):
                continue
            outcomes.extend(
                outcome
                for outcome in (classify_repeat_trace_item(item) for item in payload)
                if outcome is not None
            )
        units.append(
            make_unit_stats(
                unit_id=str(label_dir.relative_to(condition_dir)),
                source="repeat_eval_trace",
                outcomes=outcomes,
                min_calls=min_calls,
                failure_ratio=failure_ratio,
                path=str(label_dir),
            )
        )
    return {
        "checked": True,
        "condition_dir": str(condition_dir),
        "reason": "ok",
        "units": [asdict(item) for item in units],
        "bad_unit_dirs": [item.path for item in units if item.is_mass_failure],
    }


def repeat_dirs_after_anchor(
    condition_dir: Path | None,
    anchor_step: int,
) -> list[str]:
    if condition_dir is None or not condition_dir.is_dir():
        return []
    invalid: list[str] = []
    for label_dir in sorted(
        {
            path.parent
            for path in condition_dir.glob("r*/*/job.json")
            if path.is_file()
        },
        key=lambda path: path.as_posix(),
    ):
        label = label_dir.name
        job_path = label_dir / "job.json"
        job = load_json(job_path)
        if not isinstance(job, dict):
            continue
        try:
            step_no = int(job.get("step_no", 0) or 0)
        except (TypeError, ValueError):
            step_no = 0
        if label == "POST" or step_no > int(anchor_step):
            invalid.append(str(label_dir))
    return invalid


def detect(
    checkpoint_dir: Path,
    *,
    repeat_eval_condition_dir: Path | None = None,
    simulation_logs: Iterable[Path] = (),
    min_calls: int = 6,
    failure_ratio: float = 0.5,
    consecutive_sim_meetings: int = 2,
) -> dict[str, Any]:
    if min_calls < 1:
        raise ValueError("min_calls 必须至少为 1")
    if not 0.0 <= failure_ratio <= 1.0:
        raise ValueError("failure_ratio 必须在 0 到 1 之间")
    if consecutive_sim_meetings < 1:
        raise ValueError("consecutive_sim_meetings 必须至少为 1")
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"checkpoint 不存在: {checkpoint_dir}")

    anchors = recovery.discover_recovery_anchors(checkpoint_dir)
    latest_anchor = anchors[-1]
    sidecars = sidecar_files(checkpoint_dir)
    if not sidecars:
        raise RuntimeError(f"没有可分析的 forced trace sidecar: {checkpoint_dir}")
    latest_sidecar_step, latest_sidecar = sidecars[-1]
    units = extract_sidecar_units(
        latest_sidecar,
        min_calls=min_calls,
        failure_ratio=failure_ratio,
    )
    saved_meeting_ids = {item.meeting_id for item in units if item.meeting_id}
    log_paths = list(simulation_logs)
    if not log_paths:
        log_paths = sorted(checkpoint_dir.glob("*.log"))
    log_units = extract_unsaved_log_units(
        log_paths,
        saved_meeting_ids=saved_meeting_ids,
        min_calls=min_calls,
        failure_ratio=failure_ratio,
    )
    all_units = [*units, *log_units]
    sustained = first_sustained_failure(
        all_units,
        consecutive_sim_meetings,
    )

    simulation: dict[str, Any] = {
        "checked": True,
        "latest_sidecar": str(latest_sidecar),
        "latest_sidecar_step": latest_sidecar_step,
        "log_paths": [str(path) for path in log_paths if path.is_file()],
        "units": [asdict(item) for item in all_units],
        "mass_failure_detected": bool(sustained),
        "failure_start": None,
        "failure_window": [],
    }
    recommended_anchor = latest_anchor
    action = "normal_resume"
    status = "ok"
    error = ""

    if sustained:
        start_index, window = sustained
        first_bad = all_units[start_index]
        simulation["failure_start"] = asdict(first_bad)
        simulation["failure_window"] = [asdict(item) for item in window]
        containing = first_snapshot_containing_meeting(
            checkpoint_dir,
            first_bad.meeting_id,
        )
        if containing is None:
            contaminated_step = max(
                [snapshot_step(path) for path in checkpoint_dir.glob("simulate-*.json")],
                default=latest_sidecar_step,
            ) + 1
            simulation["failure_start_snapshot"] = {
                "source": "unsaved_log",
                "step_no": contaminated_step,
                "snapshot_name": "",
            }
        else:
            contaminated_step, sidecar_name = containing
            simulation["failure_start_snapshot"] = {
                "source": "sidecar",
                "step_no": contaminated_step,
                "snapshot_name": sidecar_name,
            }
        safe_anchors = [
            anchor for anchor in anchors if anchor.step_no < contaminated_step
        ]
        if not safe_anchors:
            status = "error"
            action = "no_safe_anchor"
            error = (
                "forced_llm 持续故障之前没有通过完整性校验的 staged 恢复锚点"
            )
            recommended_anchor = None
        else:
            recommended_anchor = safe_anchors[-1]
            action = "simulation_rollback"

    repeat_eval = inspect_repeat_eval(
        repeat_eval_condition_dir,
        min_calls=min_calls,
        failure_ratio=failure_ratio,
    )
    repeat_eval["invalid_after_anchor_dirs"] = (
        repeat_dirs_after_anchor(
            repeat_eval_condition_dir,
            recommended_anchor.step_no,
        )
        if action == "simulation_rollback" and recommended_anchor is not None
        else []
    )
    if not sustained and repeat_eval["bad_unit_dirs"]:
        action = "repeat_eval_repair"

    return {
        "schema_version": 1,
        "status": status,
        "error": error,
        "action": action,
        "checkpoint_dir": str(checkpoint_dir),
        "run_name": checkpoint_dir.name,
        "thresholds": {
            "min_calls": min_calls,
            "failure_ratio": failure_ratio,
            "consecutive_sim_meetings": consecutive_sim_meetings,
        },
        "simulation": simulation,
        "repeat_eval": repeat_eval,
        "available_anchors": [anchor_payload(item) for item in anchors],
        "recommended_anchor": (
            anchor_payload(recommended_anchor)
            if recommended_anchor is not None
            else None
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="只读检测 forced_llm 持续故障并推荐安全恢复锚点"
    )
    selector = parser.add_mutually_exclusive_group(required=True)
    selector.add_argument("--checkpoint-dir", help="直接指定 checkpoint 目录")
    selector.add_argument("--run-name", help="指定 results/checkpoints 下的 run_name")
    parser.add_argument(
        "--results-root",
        default=str(DEFAULT_RESULTS_ROOT),
        help="与 --run-name 配套的 results 根目录",
    )
    parser.add_argument(
        "--repeat-eval-condition-dir",
        help="可选的 repeat_scale_eval/<name>/<condition> 目录",
    )
    parser.add_argument(
        "--simulation-log",
        action="append",
        default=[],
        help="补充检查的仿真日志，可重复；默认检查 checkpoint 根目录下 *.log",
    )
    parser.add_argument("--min-calls", type=int, default=6)
    parser.add_argument("--failure-ratio", type=float, default=0.5)
    parser.add_argument("--consecutive-sim-meetings", type=int, default=2)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.checkpoint_dir:
        checkpoint_dir = Path(args.checkpoint_dir).expanduser().resolve()
    else:
        results_root = Path(args.results_root).expanduser().resolve()
        run_name = _text(args.run_name)
        if not run_name or Path(run_name).name != run_name:
            raise ValueError("--run-name 必须是单个目录名")
        checkpoint_dir = results_root / "checkpoints" / run_name
    repeat_dir = (
        Path(args.repeat_eval_condition_dir).expanduser().resolve()
        if args.repeat_eval_condition_dir
        else None
    )
    log_paths = [Path(item).expanduser().resolve() for item in args.simulation_log]
    payload = detect(
        checkpoint_dir,
        repeat_eval_condition_dir=repeat_dir,
        simulation_logs=log_paths,
        min_calls=int(args.min_calls),
        failure_ratio=float(args.failure_ratio),
        consecutive_sim_meetings=int(args.consecutive_sim_meetings),
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload["status"] == "ok" else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, FileNotFoundError) as exc:
        print(f"错误: {exc}", file=sys.stderr)
        raise SystemExit(2)
