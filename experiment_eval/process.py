"""Read-only extraction of conversation dose and CBT delivery metrics."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .schema import ExperimentRecord


def _json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _nested_root(root: Path | None, child: str) -> Path | None:
    if root is None:
        return None
    nested = root / child
    return nested if nested.is_dir() else root


def _find_run(root: Path | None, run_name: str) -> Path | None:
    if root is None:
        return None
    direct = root / run_name
    if direct.is_dir():
        return direct
    matches = [path for path in root.glob(f"**/{run_name}") if path.is_dir()]
    return matches[0] if len(matches) == 1 else None


def _checkpoint_sort_key(path: Path) -> tuple[str, str]:
    match = re.search(r"simulate-(\d{8})-(\d{4})", path.name)
    return match.groups() if match else ("", path.name)


def _last_checkpoint(run_dir: Path | None) -> tuple[Path | None, dict[str, Any] | None]:
    if run_dir is None:
        return None, None
    candidates = sorted(run_dir.glob("simulate-*.json"), key=_checkpoint_sort_key)
    if not candidates:
        return None, None
    path = candidates[-1]
    return path, _json(path)


def _judge_sessions(run_dir: Path | None) -> tuple[Path | None, list[dict[str, Any]]]:
    if run_dir is None:
        return None, []
    for relative in (Path("traces/judge_conversation.json"), Path("judge_traces/judge_conversation.json")):
        path = run_dir / relative
        if path.is_file():
            sessions = _json(path).get("sessions") or []
            return path, [value for value in sessions if isinstance(value, dict)]
    return None, []


def _meeting_number(meeting_id: str) -> int | None:
    match = re.search(r"_(\d+)$", meeting_id)
    return int(match.group(1)) if match else None


def extract_process_metrics(
    records: list[ExperimentRecord],
    *,
    experiment_data_root: Path | None,
    checkpoints_root: Path | None,
    project_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return run-level process metrics and stage-history rows.

    The extractor never writes into results. Missing raw artifacts produce an
    explicit unavailable row instead of silently treating missing values as zero.
    """
    data_root = _nested_root(experiment_data_root, "experiment_data")
    checkpoint_root = _nested_root(checkpoints_root, "checkpoints")
    process_rows: list[dict[str, Any]] = []
    stage_rows: list[dict[str, Any]] = []
    for record in records:
        data_run = _find_run(data_root, record.run_name)
        checkpoint_run = _find_run(checkpoint_root, record.run_name)
        judge_path, sessions = _judge_sessions(data_run or checkpoint_run)
        checkpoint_path, checkpoint = _last_checkpoint(checkpoint_run)
        judge_trace_available = judge_path is not None

        patient_turns = 0
        doctor_turns = 0
        patient_chars = 0
        doctor_chars = 0
        meeting_rows: list[dict[str, Any]] = []
        for session_index, session in enumerate(sessions, start=1):
            turns = session.get("turns") or []
            session_patient_turns = 0
            session_doctor_turns = 0
            session_patient_chars = 0
            session_doctor_chars = 0
            for turn in turns:
                if not isinstance(turn, dict):
                    continue
                patient = str(turn.get("patient") or "")
                doctor = str(turn.get("doctor") or "")
                if patient:
                    patient_turns += 1
                    session_patient_turns += 1
                    patient_chars += len(patient)
                    session_patient_chars += len(patient)
                if doctor:
                    doctor_turns += 1
                    session_doctor_turns += 1
                    doctor_chars += len(doctor)
                    session_doctor_chars += len(doctor)
            meeting = session.get("meeting") or {}
            meeting_rows.append(
                {
                    "meeting_index": session_index,
                    "meeting_id": meeting.get("meeting_id"),
                    "patient_turns": session_patient_turns,
                    "doctor_turns": session_doctor_turns,
                    "patient_chars": session_patient_chars,
                    "doctor_chars": session_doctor_chars,
                    "has_session_eval": isinstance(session.get("session_eval"), dict),
                }
            )

        completed_meetings = len(sessions)
        prompt_completed = None
        prompt_completion_meeting = None
        environment_task_batches = None
        environment_tasks = None
        environment_outcomes = None
        if checkpoint:
            state = checkpoint.get("intervention_state") or {}
            prompt_pairs = (state.get("session_prompt_state") or {}).get("pairs") or {}
            prompt_state = next(iter(prompt_pairs.values()), {})
            prompt_completed = bool(prompt_state.get("completed"))
            completed_state = state.get("completed_meeting_state") or {}
            meeting_ids_by_pair = completed_state.get("meeting_ids_by_pair") or {}
            meeting_ids = list(next(iter(meeting_ids_by_pair.values()), []))
            completed_meetings = len(meeting_ids) or completed_meetings
            advance_meeting_id = str(prompt_state.get("last_advance_meeting_id") or "")
            if advance_meeting_id in meeting_ids:
                prompt_completion_meeting = meeting_ids.index(advance_meeting_id) + 1
            elif advance_meeting_id:
                prompt_completion_meeting = _meeting_number(advance_meeting_id)
            task_audit = (state.get("environment_task_state") or {}).get("audit") or []
            environment_task_batches = len(task_audit)
            environment_tasks = sum(len(row.get("tasks") or []) for row in task_audit if isinstance(row, dict))
            environment_outcomes = sum(len(row.get("outcomes") or []) for row in task_audit if isinstance(row, dict))

            control = state.get("progressive_d_control_state") or {}
            histories = control.get("history_by_pair") or {}
            history = list(next(iter(histories.values()), []))
            for index, entry in enumerate(history, start=1):
                if not isinstance(entry, dict):
                    continue
                adapter = entry.get("transition_adapter") or {}
                subgoals = entry.get("subgoals") or []
                stage_rows.append(
                    {
                        "stable_id": record.stable_id,
                        "kbd": record.kbd,
                        "group": record.group,
                        "severity": record.severity,
                        "outer_run_id": record.repeat_id,
                        "meeting_index": index,
                        "meeting_id": entry.get("meeting_id"),
                        "legacy_session": entry.get("legacy_session"),
                        "macro_stage": entry.get("macro_stage"),
                        "subgoal_complete": sum(
                            item.get("progress") == "complete" for item in subgoals if isinstance(item, dict)
                        ),
                        "subgoal_total": len(subgoals),
                        "completion_score": adapter.get("completion_score"),
                        "completion_threshold": adapter.get("completion_threshold"),
                        "effective_action": adapter.get("effective_action"),
                        "valid": entry.get("valid"),
                    }
                )

        fixed_prompt_meetings = (
            completed_meetings - prompt_completion_meeting
            if prompt_completion_meeting is not None
            else None
        )
        process_rows.append(
            {
                "stable_id": record.stable_id,
                "kbd": record.kbd,
                "group": record.group,
                "severity": record.severity,
                "outer_run_id": record.repeat_id,
                "run_name": record.run_name,
                "raw_artifacts_available": bool(judge_trace_available or checkpoint),
                "completed_consultation_meetings": (
                    completed_meetings if judge_trace_available or checkpoint else None
                ),
                "cbt_prompt_completed": prompt_completed,
                "cbt_prompt_completion_meeting": prompt_completion_meeting,
                "fixed_prompt_post_completion_meetings": fixed_prompt_meetings,
                "true_no_intervention_followup": False,
                "followup_interpretation": (
                    "Fixed follow-up-prompt consultations after CBT prompt completion; "
                    "not a no-intervention delayed follow-up."
                ),
                "patient_turns": patient_turns if judge_trace_available else None,
                "doctor_turns": doctor_turns if judge_trace_available else None,
                "total_turns": patient_turns + doctor_turns if judge_trace_available else None,
                "patient_chars": patient_chars if judge_trace_available else None,
                "doctor_chars": doctor_chars if judge_trace_available else None,
                "total_chars": patient_chars + doctor_chars if judge_trace_available else None,
                "mean_turns_per_meeting": (
                    (patient_turns + doctor_turns) / len(sessions) if sessions else None
                ),
                "mean_chars_per_meeting": (
                    (patient_chars + doctor_chars) / len(sessions) if sessions else None
                ),
                "environment_task_batches": environment_task_batches,
                "environment_tasks": environment_tasks,
                "environment_outcomes": environment_outcomes,
                "judge_trace_path": _display(judge_path, project_root),
                "final_checkpoint_path": _display(checkpoint_path, project_root),
                "meeting_rows": meeting_rows,
            }
        )
    return process_rows, stage_rows


def _display(path: Path | None, project_root: Path) -> str | None:
    if path is None:
        return None
    try:
        return str(path.relative_to(project_root))
    except ValueError:
        return str(path)
