"""Offline extraction of conversation/session events for engagement figures."""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from ..schema import ExperimentRecord


TIME_FORMATS = ("%Y%m%d-%H:%M:%S", "%Y%m%d-%H:%M", "%Y-%m-%dT%H:%M:%S")
INTERACTION_EVENT_TYPES = {"doctor_consult", "resident_chat", "normal_chat"}


def _parse_time(value: Any) -> datetime | None:
    text = str(value or "").strip()
    for fmt in TIME_FORMATS:
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def _nested_root(root: Path | None, child: str) -> Path | None:
    if root is None:
        return None
    nested = root / child
    return nested if nested.is_dir() else root


def _find_run(root: Path | None, run_name: str) -> Path | None:
    if root is None or not run_name:
        return None
    direct = root / run_name
    if direct.is_dir():
        return direct
    matches = [path for path in root.glob(f"**/{run_name}") if path.is_dir()]
    return matches[0] if len(matches) == 1 else None


def _read_event_log(path: Path | None) -> tuple[list[dict[str, Any]], int]:
    if path is None or not path.is_file():
        return [], 0
    rows: list[dict[str, Any]] = []
    malformed = 0
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                malformed += 1
                continue
            if isinstance(payload, dict):
                rows.append(payload)
    return rows, malformed


def _read_conversations(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    return payload if isinstance(payload, dict) else None


def _read_object(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _artifact(
    data_run: Path | None, checkpoint_run: Path | None, name: str
) -> Path | None:
    for run_dir in (data_run, checkpoint_run):
        path = run_dir / name if run_dir else None
        if path is not None and path.is_file():
            return path
    return None


def _participants(turns: Any, label: str) -> list[str]:
    names: list[str] = []
    if isinstance(turns, list):
        for turn in turns:
            if isinstance(turn, list) and turn:
                name = str(turn[0]).strip()
                if name and name not in names:
                    names.append(name)
    if len(names) >= 2:
        return names
    prefix = label.split("@", 1)[0]
    if "->" in prefix:
        for name in prefix.split("->", 1):
            cleaned = name.strip()
            if cleaned and cleaned not in names:
                names.append(cleaned)
    return names


def _archived_roles(manifest: dict[str, Any] | None) -> dict[str, Any]:
    """Read role labels from the run's archived resolved configuration."""
    if not manifest:
        return {
            "target_agent": None,
            "doctor": None,
            "patients": [],
            "source": None,
        }
    resolved = manifest.get("resolved_config") or {}
    config = resolved.get("config") if isinstance(resolved, dict) else None
    if not isinstance(config, dict):
        return {
            "target_agent": None,
            "doctor": None,
            "patients": [],
            "source": None,
        }
    staged_eval = config.get("staged_eval") or {}
    intervention = config.get("intervention") or {}
    target = str(staged_eval.get("target_agent") or "").strip() or None
    doctor = str(intervention.get("doctor") or "").strip() or None
    patients = [
        str(value).strip()
        for value in (intervention.get("patients") or [])
        if str(value).strip()
    ]
    return {
        "target_agent": target,
        "doctor": doctor,
        "patients": patients,
        "source": (
            "cbt_condition_manifest.resolved_config.config"
            if target or doctor or patients
            else None
        ),
    }


def _turn_metrics(
    turns: list[Any], *, target_agent: str | None, doctor: str | None
) -> dict[str, Any]:
    speaker_turns: dict[str, int] = defaultdict(int)
    speaker_characters: dict[str, int] = defaultdict(int)
    total_characters = 0
    for turn in turns:
        if not isinstance(turn, list) or len(turn) < 2:
            continue
        speaker = str(turn[0]).strip()
        text = str(turn[1])
        if speaker:
            speaker_turns[speaker] += 1
            speaker_characters[speaker] += len(text)
        total_characters += len(text)
    return {
        "character_count": total_characters,
        "target_turn_count": (
            speaker_turns.get(target_agent, 0) if target_agent else None
        ),
        "target_character_count": (
            speaker_characters.get(target_agent, 0) if target_agent else None
        ),
        "doctor_turn_count": speaker_turns.get(doctor, 0) if doctor else None,
        "doctor_character_count": (
            speaker_characters.get(doctor, 0) if doctor else None
        ),
    }


def extract_engagement_events(
    records: list[ExperimentRecord],
    *,
    experiment_data_root: Path | None,
    checkpoints_root: Path | None,
    project_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Return conversation-level rows, run coverage rows and field diagnostics."""
    data_root = _nested_root(experiment_data_root, "experiment_data")
    checkpoint_root = _nested_root(checkpoints_root, "checkpoints")
    events: list[dict[str, Any]] = []
    coverage: list[dict[str, Any]] = []
    run_diagnostics: list[dict[str, Any]] = []
    for record in records:
        data_run = _find_run(data_root, record.run_name)
        checkpoint_run = _find_run(checkpoint_root, record.run_name)
        conversation_path = _artifact(data_run, checkpoint_run, "conversation.json")
        event_path = _artifact(data_run, checkpoint_run, "simulation_events.jsonl")
        manifest_path = _artifact(
            data_run, checkpoint_run, "cbt_condition_manifest.json"
        )
        conversations = _read_conversations(conversation_path)
        event_log, malformed_lines = _read_event_log(event_path)
        roles = _archived_roles(_read_object(manifest_path))
        target_agent = roles["target_agent"]
        doctor = roles["doctor"]

        parsed_event_times = [
            value
            for value in (_parse_time(row.get("simulation_time")) for row in event_log)
            if value is not None
        ]
        conversation_times = [
            value
            for value in (_parse_time(key) for key in (conversations or {}))
            if value is not None
        ]
        all_times = parsed_event_times + conversation_times
        observation_start = min(all_times) if all_times else None
        observation_end = max(all_times) if all_times else None

        interaction_by_index: dict[tuple[datetime, int], dict[str, Any]] = {}
        duration_by_pair: dict[tuple[datetime, frozenset[str]], list[float]] = (
            defaultdict(list)
        )
        for row in event_log:
            timestamp = _parse_time(row.get("simulation_time"))
            if timestamp is None:
                continue
            if row.get("event_type") in INTERACTION_EVENT_TYPES:
                index = row.get("conversation_index")
                if isinstance(index, int):
                    interaction_by_index[(timestamp, index)] = row
            if (
                row.get("event_type") == "agent_state"
                and str(row.get("activity") or "") == "对话"
            ):
                agent, peer = str(row.get("agent") or ""), str(row.get("peer") or "")
                duration = row.get("action_duration_minutes")
                if agent and peer and isinstance(duration, (int, float)):
                    duration_by_pair[(timestamp, frozenset((agent, peer)))].append(
                        float(duration)
                    )

        extracted_for_run = 0
        duration_proxy_count = 0
        if conversations:
            for raw_time, blocks in conversations.items():
                timestamp = _parse_time(raw_time)
                if timestamp is None or not isinstance(blocks, list):
                    continue
                conversation_index = 0
                for block in blocks:
                    if not isinstance(block, dict):
                        continue
                    for label, turns in block.items():
                        if not isinstance(turns, list):
                            continue
                        names = _participants(turns, str(label))
                        matched = interaction_by_index.get(
                            (timestamp, conversation_index), {}
                        )
                        duration_values = (
                            duration_by_pair.get((timestamp, frozenset(names)), [])
                            if len(names) >= 2
                            else []
                        )
                        duration_proxy = (
                            max(duration_values) if duration_values else None
                        )
                        if duration_proxy is not None:
                            duration_proxy_count += 1
                        origin = observation_start or timestamp
                        target_involved = (
                            target_agent in names if target_agent is not None else None
                        )
                        target_partners = (
                            [name for name in names if name != target_agent]
                            if target_involved
                            else []
                        )
                        turn_metrics = _turn_metrics(
                            turns, target_agent=target_agent, doctor=doctor
                        )
                        events.append(
                            {
                                "stable_id": record.stable_id,
                                "run_name": record.run_name,
                                "persona": record.kbd,
                                "group": record.group,
                                "replicate_id": record.repeat_id,
                                "timestamp": timestamp.isoformat(),
                                "simulation_day": (
                                    timestamp.date() - origin.date()
                                ).days
                                + 1,
                                "hour": timestamp.hour + timestamp.minute / 60.0,
                                "participants": "|".join(names),
                                "participant_count": len(names),
                                "turn_count": len(turns),
                                **turn_metrics,
                                "target_agent": target_agent,
                                "target_agent_source": roles["source"],
                                "target_involved": target_involved,
                                "target_partners": "|".join(target_partners),
                                "doctor": doctor,
                                "doctor_target_interaction": (
                                    doctor in names and bool(target_involved)
                                    if doctor and target_involved is not None
                                    else None
                                ),
                                "duration_minutes_proxy": duration_proxy,
                                "duration_source": (
                                    "simulation_events.action_duration_minutes"
                                    if duration_proxy is not None
                                    else None
                                ),
                                "interaction_type": matched.get("event_type")
                                or "conversation_log",
                                "session_id": matched.get("session_id") or None,
                                "rule_id": matched.get("rule_id") or None,
                                "meeting_source": matched.get("meeting_source") or None,
                                "initiator": matched.get("agent") or None,
                                "initiator_source": (
                                    "simulation_events.agent"
                                    if matched.get("agent")
                                    else None
                                ),
                                "conversation_index": conversation_index,
                            }
                        )
                        extracted_for_run += 1
                        conversation_index += 1

        day_count = (
            (observation_end.date() - observation_start.date()).days + 1
            if observation_start is not None and observation_end is not None
            else 0
        )
        coverage.append(
            {
                "stable_id": record.stable_id,
                "run_name": record.run_name,
                "persona": record.kbd,
                "group": record.group,
                "replicate_id": record.repeat_id,
                "observation_start": (
                    observation_start.isoformat() if observation_start else None
                ),
                "observation_end": (
                    observation_end.isoformat() if observation_end else None
                ),
                "observation_days": day_count,
                "conversation_count": extracted_for_run,
                "target_agent": target_agent,
                "target_agent_source": roles["source"],
                "target_conversation_count": sum(
                    1
                    for row in events
                    if row["stable_id"] == record.stable_id
                    and row.get("target_involved") is True
                ),
                "conversation_log_available": conversations is not None,
                "simulation_event_log_available": event_path is not None
                and event_path.is_file(),
            }
        )
        run_diagnostics.append(
            {
                "stable_id": record.stable_id,
                "conversation_path": _display(conversation_path, project_root),
                "simulation_event_path": _display(event_path, project_root),
                "conversation_count": extracted_for_run,
                "duration_proxy_count": duration_proxy_count,
                "target_agent": target_agent,
                "target_agent_source": roles["source"],
                "condition_manifest_path": _display(manifest_path, project_root),
                "malformed_simulation_event_lines_skipped": malformed_lines,
                "missing_fields": [
                    field
                    for field, absent in (
                        ("conversation_timestamp", conversations is None),
                        ("participants", conversations is None),
                        ("turn_count", conversations is None),
                        ("simulation_day_time", observation_start is None),
                        ("target_agent", target_agent is None),
                        # No current artifact records start/end timestamps for
                        # actual conversational elapsed duration.
                        ("true_elapsed_conversation_duration", True),
                    )
                    if absent
                ],
            }
        )
    diagnostics = {
        "event_count": len(events),
        "run_count": len(records),
        "runs_with_conversations": sum(
            row["conversation_log_available"] for row in coverage
        ),
        "runs_with_target_agent": sum(
            row.get("target_agent") is not None for row in coverage
        ),
        "field_availability": {
            "conversation_timestamp": "available from conversation.json keys",
            "participants": "available from turn speakers with label fallback",
            "turn_count": "available from conversation turn arrays",
            "simulation_day_time": "derived from simulation timestamps and observation origin",
            "target_agent": "optional archived cbt_condition_manifest resolved config",
            "target_involvement": "derived only when target_agent is available",
            "turn_balance_and_character_count": "available from conversation turn arrays",
            "initiator": "available only when a matching simulation interaction event records agent",
            "interaction_rule_and_source": "available only from matching simulation interaction events",
            "duration_minutes_proxy": "optional simulation_events.action_duration_minutes",
            "true_elapsed_conversation_duration": "missing",
        },
        "duration_warning": (
            "action_duration_minutes is a planned action-duration proxy, not observed conversation elapsed time"
        ),
        "runs": run_diagnostics,
    }
    return events, coverage, diagnostics


def _display(path: Path | None, project_root: Path) -> str | None:
    if path is None:
        return None
    try:
        return str(path.relative_to(project_root))
    except ValueError:
        return str(path)
