#!/usr/bin/env python3
"""Merge judge traces and consultation records for one checkpoint archive."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any


JSON_INDENT = 2
STRICT_MODE = True
JUDGE_SPEAKER = "判断LLM"

PROJECT_ROOT = Path(__file__).resolve().parent
CHECKPOINTS_DIR = PROJECT_ROOT / "results" / "checkpoints"
# Default runtime parameters (you can edit these two values directly).
DEFAULT_ARCHIVE_NAME = "sim-test-kbd-0421-2"
EXPERIMENT_DATA_DIR = PROJECT_ROOT / "results" / "experiment_data"
DEFAULT_OUTPUT_FILE = EXPERIMENT_DATA_DIR / DEFAULT_ARCHIVE_NAME / "traces" / "merge_consultation_dialogues.json"


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=JSON_INDENT)


def iso_to_slot_key(iso_datetime: str) -> str:
    dt = datetime.fromisoformat(iso_datetime)
    return dt.strftime("%Y%m%d-%H:%M")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge consultation records and judge conversation under "
            "results/checkpoints/<archive_name>."
        )
    )
    parser.add_argument(
        "archive_name",
        nargs="?",
        default=DEFAULT_ARCHIVE_NAME,
        help=(
            "Checkpoint archive name (optional). "
            f"Default: {DEFAULT_ARCHIVE_NAME}"
        ),
    )
    parser.add_argument(
        "output_file",
        nargs="?",
        default=None,
        help=(
            "Output JSON file path (optional). "
            "Default: results/experiment_data/<archive_name>/traces/merge_consultation_dialogues.json"
        ),
    )
    args = parser.parse_args()
    if args.output_file is None:
        args.output_file = (
            EXPERIMENT_DATA_DIR
            / args.archive_name
            / "traces"
            / "merge_consultation_dialogues.json"
        )
    return args


def find_latest_simulate_file(checkpoint_dir: Path) -> Path:
    simulate_files = sorted(checkpoint_dir.glob("simulate-*.json"), key=lambda p: p.name)
    if not simulate_files:
        raise FileNotFoundError(f"No simulate-*.json found under: {checkpoint_dir}")
    return simulate_files[-1]


def find_first_value_by_key(data: Any, target_key: str) -> Any | None:
    if isinstance(data, dict):
        if target_key in data:
            return data[target_key]
        for value in data.values():
            found = find_first_value_by_key(value, target_key)
            if found is not None:
                return found
    elif isinstance(data, list):
        for item in data:
            found = find_first_value_by_key(item, target_key)
            if found is not None:
                return found
    return None


def is_consult_record_enabled(simulate_data: Any) -> bool:
    if not isinstance(simulate_data, dict):
        raise TypeError("simulate json root must be an object")

    intervention = simulate_data.get("intervention") or {}
    if isinstance(intervention, dict):
        consult_cfg = intervention.get("consult_record") or {}
        if isinstance(consult_cfg, dict) and "enabled" in consult_cfg:
            return bool(consult_cfg.get("enabled"))

    fallback = find_first_value_by_key(simulate_data, "consult_record")
    if isinstance(fallback, dict) and "enabled" in fallback:
        return bool(fallback.get("enabled"))
    return False


def extract_records_by_id(simulate_data: Any, *, required: bool) -> dict[str, Any]:
    if not isinstance(simulate_data, dict):
        raise TypeError("simulate json root must be an object")

    direct = simulate_data.get("records_by_id")
    if isinstance(direct, dict):
        return direct

    nested = (
        (simulate_data.get("intervention_state") or {})
        .get("consult_record_state", {})
        .get("records_by_id")
    )
    if isinstance(nested, dict):
        return nested

    fallback = find_first_value_by_key(simulate_data, "records_by_id")
    if isinstance(fallback, dict):
        return fallback

    if required:
        raise KeyError("records_by_id was not found in simulate json")
    return {}


def build_session_index_by_meeting(simulate_data: Any) -> dict[str, str]:
    if not isinstance(simulate_data, dict):
        raise TypeError("simulate json root must be an object")

    intervention_state = simulate_data.get("intervention_state") or {}
    by_meeting: dict[str, str] = {}
    if not isinstance(intervention_state, dict):
        return by_meeting

    # Prefer the legacy/general session_eval index, then fill Progressive D
    # meetings from its compatible local/LLM control history.
    for state_name, session_key in (
        ("session_eval_state", "session_id"),
        ("progressive_d_control_state", "legacy_session"),
    ):
        state = intervention_state.get(state_name) or {}
        histories = state.get("history_by_pair") if isinstance(state, dict) else {}
        if not isinstance(histories, dict):
            continue
        for history in histories.values():
            if not isinstance(history, list):
                continue
            for item in history:
                if not isinstance(item, dict):
                    continue
                meeting_id = str(item.get("meeting_id") or "").strip()
                session_id = str(item.get(session_key) or "").strip()
                if meeting_id and session_id and meeting_id not in by_meeting:
                    by_meeting[meeting_id] = session_id
    return by_meeting


def build_record_index(records_by_id: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], int]:
    by_meeting: dict[str, dict[str, Any]] = {}
    duplicate_meeting_count = 0

    for record in records_by_id.values():
        if not isinstance(record, dict):
            continue
        meeting_id = record.get("meeting_id")
        if not meeting_id:
            continue
        if meeting_id in by_meeting:
            duplicate_meeting_count += 1
            continue
        by_meeting[meeting_id] = record
    return by_meeting, duplicate_meeting_count


def normalize_time(record: dict[str, Any] | None, meeting: dict[str, Any]) -> str:
    if isinstance(record, dict):
        started_at = record.get("session_started_at")
        if isinstance(started_at, str):
            try:
                return iso_to_slot_key(started_at)
            except ValueError:
                pass

    step_time = meeting.get("step_time")
    if isinstance(step_time, str):
        return step_time
    return ""


def resolve_participants(record: dict[str, Any] | None, meeting: dict[str, Any]) -> tuple[str, str]:
    doctor = None
    patient = None

    if isinstance(record, dict):
        participants = record.get("participants") or {}
        if isinstance(participants, dict):
            doctor = participants.get("doctor")
            patient = participants.get("patient")

    if not isinstance(doctor, str) or not isinstance(patient, str):
        pair_key = meeting.get("pair_key")
        if isinstance(pair_key, str) and "::" in pair_key:
            pair_doctor, pair_patient = pair_key.split("::", 1)
            if not isinstance(doctor, str):
                doctor = pair_doctor
            if not isinstance(patient, str):
                patient = pair_patient

    if not isinstance(doctor, str):
        doctor = "doctor"
    if not isinstance(patient, str):
        patient = "patient"
    return doctor, patient


def extract_dialogue(turns: Any, doctor: str, patient: str) -> list[list[str]]:
    dialogue: list[list[str]] = []
    if not isinstance(turns, list):
        return dialogue

    for turn in turns:
        if not isinstance(turn, dict):
            continue

        patient_text = turn.get("patient")
        if isinstance(patient_text, str):
            dialogue.append([patient, patient_text])

        advice = (
            (turn.get("judge") or {})
            .get("doctor_turn_judge_cache", {})
            .get("advice")
        )
        if isinstance(advice, str):
            dialogue.append([JUDGE_SPEAKER, advice])

        doctor_text = turn.get("doctor")
        if isinstance(doctor_text, str):
            dialogue.append([doctor, doctor_text])

    return dialogue


def merge_sessions(
    judge_sessions: list[Any],
    records_by_meeting: dict[str, dict[str, Any]],
    session_by_meeting: dict[str, str],
    *,
    consult_record_enabled: bool,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    merged: list[dict[str, Any]] = []
    stats = {"total_sessions": 0, "matched": 0, "missing_record": 0}

    for session in judge_sessions:
        if not isinstance(session, dict):
            continue
        stats["total_sessions"] += 1

        meeting = session.get("meeting") or {}
        meeting_id = meeting.get("meeting_id")
        if not isinstance(meeting_id, str) or not meeting_id:
            stats["missing_record"] += 1
            continue

        record = records_by_meeting.get(meeting_id)
        if consult_record_enabled and not isinstance(record, dict):
            stats["missing_record"] += 1
            continue

        doctor, patient = resolve_participants(record, meeting)
        dialogue = extract_dialogue(session.get("turns"), doctor=doctor, patient=patient)
        reason = (session.get("session_eval") or {}).get("reason")
        current_session = session_by_meeting.get(meeting_id, "")
        if isinstance(record, dict):
            current_session = str(record.get("current_session") or current_session)

        item = {
            "time": normalize_time(record, meeting),
            "current_session": current_session,
            "dialogue": dialogue,
            "reason": reason,
        }
        if consult_record_enabled and isinstance(record, dict):
            item["consultation_record"] = {"soap": record.get("soap")}

        merged.append(item)
        stats["matched"] += 1

    merged.sort(key=lambda item: item.get("time") or "")
    return merged, stats


def main() -> None:
    args = parse_args()

    checkpoint_dir = CHECKPOINTS_DIR / args.archive_name
    if not checkpoint_dir.exists():
        available = sorted(
            p.name for p in CHECKPOINTS_DIR.iterdir() if p.is_dir()
        )
        raise FileNotFoundError(
            f"Checkpoint directory not found: {checkpoint_dir}\n"
            f"Available checkpoint archives: {', '.join(available)}"
        )

    judge_path = checkpoint_dir / "judge_traces" / "judge_conversation.json"
    if not judge_path.exists():
        raise FileNotFoundError(f"judge_conversation.json not found: {judge_path}")

    latest_simulate_path = find_latest_simulate_file(checkpoint_dir)
    simulate_data = read_json(latest_simulate_path)
    consult_record_enabled = is_consult_record_enabled(simulate_data)
    records_by_id = extract_records_by_id(simulate_data, required=consult_record_enabled)
    records_by_meeting, duplicate_meeting_count = build_record_index(records_by_id)
    session_by_meeting = build_session_index_by_meeting(simulate_data)

    judge_data = read_json(judge_path)
    judge_sessions = judge_data.get("sessions")
    if not isinstance(judge_sessions, list):
        raise TypeError(f"Invalid sessions format in: {judge_path}")

    merged, stats = merge_sessions(
        judge_sessions,
        records_by_meeting,
        session_by_meeting,
        consult_record_enabled=consult_record_enabled,
    )

    if consult_record_enabled and STRICT_MODE and (duplicate_meeting_count > 0 or stats["missing_record"] > 0):
        raise RuntimeError(
            "STRICT_MODE=True and duplicate/missing records detected: "
            f"duplicates={duplicate_meeting_count}, missing_record={stats['missing_record']}"
        )

    output_path = Path(args.output_file)
    if not output_path.is_absolute():
        output_path = (PROJECT_ROOT / output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(output_path, merged)

    print(f"[Input] latest_simulate={latest_simulate_path}")
    print(f"[Input] judge_conversation={judge_path}")
    print(
        "[Stats] "
        f"consult_record_enabled={consult_record_enabled}, "
        f"total_sessions={stats['total_sessions']}, "
        f"matched={stats['matched']}, "
        f"missing_record={stats['missing_record']}, "
        f"duplicate_meeting={duplicate_meeting_count}"
    )
    print(f"[Done] {output_path} (records={len(merged)})")


if __name__ == "__main__":
    main()
