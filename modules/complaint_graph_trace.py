"""Compact complaint-graph traces for judge conversations.

The depression engine keeps a rich graph snapshot.  Judge traces only need the
state transition that belongs to a patient utterance, plus a small session
summary.  This module deliberately contains no simulation side effects so the
same schema can be used by the live runtime and by checkpoint backfills.
"""

from __future__ import annotations

import copy
import re
from datetime import datetime
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


MEETING_TIMESTAMP_RE = re.compile(
    r"(?P<date>[0-9]{8})-(?P<time>[0-9]{6})(?:_|$)"
)


class ComplaintGraphTraceAlignmentError(ValueError):
    """Raised when archived records cannot be aligned without guessing."""


def unavailable_trace(reason: str, provenance: Any) -> Dict[str, Any]:
    return {
        "status": "unavailable",
        "provenance": _normalize_provenance(provenance),
        "reason": str(reason or "unknown"),
    }


def compact_stage(
    stage_id: str,
    stage_label: str,
    stage_index: Any,
    stage_catalog: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> Dict[str, Any]:
    catalog = stage_catalog if isinstance(stage_catalog, Mapping) else {}
    stage = catalog.get(str(stage_id or ""), {})
    stage = stage if isinstance(stage, Mapping) else {}
    try:
        normalized_index: Optional[int] = int(stage_index)
    except (TypeError, ValueError):
        normalized_index = None
    return {
        "stage_id": str(stage_id or stage.get("id", "") or ""),
        "stage_label": str(stage_label or stage.get("label", "") or ""),
        "summary": str(stage.get("summary", "") or ""),
        "core_belief": str(stage.get("core_belief", "") or ""),
        "stage_index": normalized_index,
    }


def build_transition_trace(
    history_row: Mapping[str, Any],
    stage_catalog: Optional[Mapping[str, Mapping[str, Any]]] = None,
    provenance: Any = "runtime_capture",
) -> Dict[str, Any]:
    row = history_row if isinstance(history_row, Mapping) else {}
    pointer_before = _safe_int_or_none(row.get("pointer_before"))
    pointer_after = _safe_int_or_none(row.get("pointer_after"))
    from_id = str(row.get("from_stage_id", "") or "")
    to_id = str(row.get("to_stage_id", "") or "")
    changed = bool(
        from_id != to_id
        or (
            pointer_before is not None
            and pointer_after is not None
            and pointer_before != pointer_after
        )
    )
    return {
        "status": "available",
        "provenance": _normalize_provenance(provenance),
        "transition": {
            "timestamp": _serialize_timestamp(row.get("timestamp")),
            "source": str(row.get("source", "") or ""),
            "action": str(row.get("action", "hold") or "hold"),
            "matched": bool(row.get("matched", False)),
            "match_reason": str(row.get("match_reason", "") or ""),
            "pointer_before": pointer_before,
            "pointer_after": pointer_after,
        },
        "before": compact_stage(
            from_id,
            str(row.get("from_stage_label", "") or ""),
            pointer_before,
            stage_catalog,
        ),
        "after": compact_stage(
            to_id,
            str(row.get("to_stage_label", "") or ""),
            pointer_after,
            stage_catalog,
        ),
        "changed": changed,
    }


def capture_runtime_chat_trace(patient: Any, patient_text: str) -> Dict[str, Any]:
    provenance = {"mode": "runtime_capture"}
    text = str(patient_text or "").strip()
    if not text:
        return unavailable_trace("no_patient_utterance", provenance)
    manager = _runtime_graph_manager(patient)
    if manager is None:
        return unavailable_trace("depression_dynamic_unavailable", provenance)
    history = getattr(manager, "stage_history", None)
    dialogue = getattr(manager, "dialogue_history", None)
    if not isinstance(history, list) or not history:
        return unavailable_trace("stage_history_missing", provenance)
    if not isinstance(dialogue, list) or not dialogue:
        return unavailable_trace("dialogue_history_missing", provenance)
    history_row = history[-1]
    dialogue_row = dialogue[-1]
    if not isinstance(history_row, Mapping) or not isinstance(dialogue_row, Mapping):
        return unavailable_trace("latest_history_invalid", provenance)
    if str(history_row.get("source", "") or "") != "chat":
        return unavailable_trace("latest_transition_not_chat", provenance)
    if str(dialogue_row.get("source", "") or "") != "chat":
        return unavailable_trace("latest_dialogue_not_chat", provenance)
    if _serialize_timestamp(history_row.get("timestamp")) != _serialize_timestamp(
        dialogue_row.get("timestamp")
    ):
        return unavailable_trace("latest_history_timestamp_mismatch", provenance)
    excerpt = str(dialogue_row.get("conversation_excerpt", "") or "").strip()
    if not _excerpt_matches(excerpt, text):
        return unavailable_trace("latest_transition_not_bound_to_patient_turn", provenance)
    return build_transition_trace(
        history_row,
        _runtime_stage_catalog(manager),
        provenance,
    )


def capture_runtime_reflection_trace(patient: Any, meeting_id: str) -> Dict[str, Any]:
    provenance = {"mode": "runtime_capture"}
    target_meeting = str(meeting_id or "").strip()
    if not target_meeting:
        return unavailable_trace("meeting_id_missing", provenance)
    manager = _runtime_graph_manager(patient)
    if manager is None:
        return unavailable_trace("depression_dynamic_unavailable", provenance)
    history = getattr(manager, "stage_history", None)
    dialogue = getattr(manager, "dialogue_history", None)
    if not isinstance(history, list) or not history:
        return unavailable_trace("stage_history_missing", provenance)
    if not isinstance(dialogue, list) or not dialogue:
        return unavailable_trace("dialogue_history_missing", provenance)
    history_row = history[-1]
    dialogue_row = dialogue[-1]
    if not isinstance(history_row, Mapping) or not isinstance(dialogue_row, Mapping):
        return unavailable_trace("latest_history_invalid", provenance)
    if str(history_row.get("source", "") or "") != "reflection":
        return unavailable_trace("reflection_not_committed", provenance)
    if str(dialogue_row.get("source", "") or "") != "reflection":
        return unavailable_trace("latest_dialogue_not_reflection", provenance)
    if _serialize_timestamp(history_row.get("timestamp")) != _serialize_timestamp(
        dialogue_row.get("timestamp")
    ):
        return unavailable_trace("latest_history_timestamp_mismatch", provenance)
    metadata = dialogue_row.get("metadata", {})
    metadata = metadata if isinstance(metadata, Mapping) else {}
    trigger_context = metadata.get("trigger_context", {})
    trigger_context = trigger_context if isinstance(trigger_context, Mapping) else {}
    if str(trigger_context.get("meeting_id", "") or "") != target_meeting:
        return unavailable_trace("reflection_not_bound_to_meeting", provenance)
    return build_transition_trace(
        history_row,
        _runtime_stage_catalog(manager),
        provenance,
    )


def update_session_chat_summary(
    session_item: Dict[str, Any],
    turn_trace: Mapping[str, Any],
) -> None:
    summary = _ensure_session_summary(session_item)
    if str(turn_trace.get("status", "") or "") != "available":
        return
    if str((turn_trace.get("transition", {}) or {}).get("source", "") or "") != "chat":
        return
    if summary.get("before_chat") is None:
        summary["before_chat"] = copy.deepcopy(turn_trace.get("before"))
    summary["after_chat"] = copy.deepcopy(turn_trace.get("after"))
    summary["chat_transition_count"] = int(
        summary.get("chat_transition_count", 0) or 0
    ) + 1


def update_session_reflection_summary(
    session_item: Dict[str, Any],
    reflection_trace: Mapping[str, Any],
) -> None:
    summary = _ensure_session_summary(session_item)
    summary["reflection"] = copy.deepcopy(dict(reflection_trace))
    if str(reflection_trace.get("status", "") or "") == "available":
        summary["after_reflection"] = copy.deepcopy(reflection_trace.get("after"))


def parse_meeting_timestamp(meeting_id: str) -> str:
    match = MEETING_TIMESTAMP_RE.search(str(meeting_id or ""))
    if not match:
        raise ComplaintGraphTraceAlignmentError(
            "meeting_id does not contain YYYYMMDD-HHMMSS: {!r}".format(
                str(meeting_id or "")
            )
        )
    date = match.group("date")
    time = match.group("time")
    return "{}-{}-{}T{}:{}:{}".format(
        date[0:4],
        date[4:6],
        date[6:8],
        time[0:2],
        time[2:4],
        time[4:6],
    )


def enrich_archived_judge_trace(
    trace_payload: Mapping[str, Any],
    graph_state: Mapping[str, Any],
    forced_prompt_payload: Mapping[str, Any],
    source_snapshot: str,
) -> Tuple[Dict[str, Any], Dict[str, int]]:
    """Return an enriched copy, or raise before mutating the input."""

    result = copy.deepcopy(dict(trace_payload))
    sessions = result.get("sessions", [])
    if not isinstance(sessions, list):
        raise ComplaintGraphTraceAlignmentError("judge trace sessions must be a list")
    stage_history = graph_state.get("stage_history", [])
    if not isinstance(stage_history, list):
        raise ComplaintGraphTraceAlignmentError("stage_history must be a list")
    catalog = _catalog_from_graph_state(graph_state)
    forced_index = _forced_session_index(forced_prompt_payload)
    provenance = {
        "mode": "checkpoint_backfill",
        "source_snapshot": str(source_snapshot or ""),
    }
    stats = {
        "sessions": 0,
        "patient_transitions": 0,
        "reflection_transitions": 0,
        "changed_transitions": 0,
    }

    for session in sessions:
        if not isinstance(session, dict):
            raise ComplaintGraphTraceAlignmentError("judge session must be an object")
        meeting = session.get("meeting", {})
        meeting = meeting if isinstance(meeting, Mapping) else {}
        meeting_id = str(meeting.get("meeting_id", "") or "")
        pair_key = str(meeting.get("pair_key", "") or "")
        timestamp = parse_meeting_timestamp(meeting_id)
        turns = session.get("turns", [])
        if not isinstance(turns, list):
            raise ComplaintGraphTraceAlignmentError(
                "{} turns must be a list".format(meeting_id)
            )
        patient_texts = [
            str(turn.get("patient", "") or "").strip()
            for turn in turns
            if isinstance(turn, Mapping)
            and str(turn.get("patient", "") or "").strip()
        ]
        forced_texts = _forced_patient_outputs(
            forced_index,
            meeting_id,
            pair_key,
        )
        if forced_texts != patient_texts:
            raise ComplaintGraphTraceAlignmentError(
                "{} forced patient outputs do not match judge turns".format(
                    meeting_id
                )
            )
        chat_rows, reflection_row = _align_history_block(
            stage_history,
            timestamp,
            len(patient_texts),
            meeting_id,
        )

        chat_iter = iter(chat_rows)
        for turn in turns:
            if not isinstance(turn, dict):
                raise ComplaintGraphTraceAlignmentError(
                    "{} contains a non-object turn".format(meeting_id)
                )
            patient_text = str(turn.get("patient", "") or "").strip()
            if patient_text:
                trace = build_transition_trace(next(chat_iter), catalog, provenance)
                stats["patient_transitions"] += 1
                stats["changed_transitions"] += int(bool(trace.get("changed", False)))
            else:
                trace = unavailable_trace("no_patient_utterance", provenance)
            turn["complaint_graph"] = trace

        reflection_trace = build_transition_trace(
            reflection_row,
            catalog,
            provenance,
        )
        session["complaint_graph"] = {
            "before_chat": (
                copy.deepcopy(turns[next(
                    idx
                    for idx, turn in enumerate(turns)
                    if str(turn.get("patient", "") or "").strip()
                )]["complaint_graph"]["before"])
                if patient_texts
                else None
            ),
            "after_chat": (
                copy.deepcopy(
                    next(
                        turn["complaint_graph"]
                        for turn in reversed(turns)
                        if str(turn.get("patient", "") or "").strip()
                    )["after"]
                )
                if patient_texts
                else None
            ),
            "after_reflection": copy.deepcopy(reflection_trace.get("after")),
            "chat_transition_count": len(chat_rows),
            "reflection": reflection_trace,
        }
        stats["sessions"] += 1
        stats["reflection_transitions"] += 1

    return result, stats


def _align_history_block(
    stage_history: Sequence[Any],
    timestamp: str,
    patient_count: int,
    meeting_id: str,
) -> Tuple[List[Mapping[str, Any]], Mapping[str, Any]]:
    if patient_count <= 0:
        raise ComplaintGraphTraceAlignmentError(
            "{} has no patient outputs to align".format(meeting_id)
        )
    candidates: List[Tuple[List[Mapping[str, Any]], Mapping[str, Any]]] = []
    for index, raw_row in enumerate(stage_history):
        if not isinstance(raw_row, Mapping):
            continue
        if _serialize_timestamp(raw_row.get("timestamp")) != timestamp:
            continue
        if str(raw_row.get("source", "") or "") != "reflection":
            continue
        block_start = index - patient_count
        if block_start < 0:
            continue
        block = stage_history[block_start:index]
        if len(block) != patient_count:
            continue
        if all(
            isinstance(row, Mapping)
            and _serialize_timestamp(row.get("timestamp")) == timestamp
            and str(row.get("source", "") or "") == "chat"
            for row in block
        ):
            candidates.append((list(block), raw_row))
    if len(candidates) != 1:
        raise ComplaintGraphTraceAlignmentError(
            "{} expected one chat/reflection history block, found {}".format(
                meeting_id,
                len(candidates),
            )
        )
    return candidates[0]


def _forced_session_index(
    payload: Mapping[str, Any],
) -> Dict[Tuple[str, str], Mapping[str, Any]]:
    sessions = payload.get("sessions", []) if isinstance(payload, Mapping) else []
    if not isinstance(sessions, list):
        raise ComplaintGraphTraceAlignmentError(
            "forced prompt trace sessions must be a list"
        )
    result: Dict[Tuple[str, str], Mapping[str, Any]] = {}
    for session in sessions:
        if not isinstance(session, Mapping):
            continue
        meeting = session.get("meeting", {})
        meeting = meeting if isinstance(meeting, Mapping) else {}
        meeting_id = str(meeting.get("meeting_id", "") or "")
        pair_key = str(meeting.get("pair_key", "") or "")
        if not meeting_id:
            continue
        key = (meeting_id, pair_key)
        if key in result:
            raise ComplaintGraphTraceAlignmentError(
                "duplicate forced prompt session: {} {}".format(
                    meeting_id,
                    pair_key,
                )
            )
        result[key] = session
    return result


def _forced_patient_outputs(
    index: Mapping[Tuple[str, str], Mapping[str, Any]],
    meeting_id: str,
    pair_key: str,
) -> List[str]:
    session = index.get((meeting_id, pair_key))
    if session is None:
        raise ComplaintGraphTraceAlignmentError(
            "forced prompt session missing: {} {}".format(meeting_id, pair_key)
        )
    records = session.get("records", [])
    if not isinstance(records, list):
        raise ComplaintGraphTraceAlignmentError(
            "{} forced records must be a list".format(meeting_id)
        )
    return [
        str(record.get("output", "") or "").strip()
        for record in records
        if isinstance(record, Mapping)
        and str(record.get("role", "") or "") == "patient"
    ]


def _runtime_graph_manager(patient: Any) -> Any:
    engine = getattr(patient, "depression_dynamic", None)
    manager = getattr(engine, "graph_manager", None)
    return manager


def _runtime_stage_catalog(manager: Any) -> Dict[str, Mapping[str, Any]]:
    raw = getattr(manager, "stage_catalog", {})
    if not isinstance(raw, Mapping):
        return {}
    return {
        str(stage_id): stage
        for stage_id, stage in raw.items()
        if isinstance(stage, Mapping)
    }


def _catalog_from_graph_state(
    graph_state: Mapping[str, Any],
) -> Dict[str, Mapping[str, Any]]:
    result: Dict[str, Mapping[str, Any]] = {}
    stages = graph_state.get("runtime_stages", [])
    if isinstance(stages, list):
        for stage in stages:
            if not isinstance(stage, Mapping):
                continue
            stage_id = str(stage.get("id", "") or "")
            if stage_id:
                result[stage_id] = stage
    return result


def _ensure_session_summary(session_item: Dict[str, Any]) -> Dict[str, Any]:
    summary = session_item.get("complaint_graph")
    if not isinstance(summary, dict):
        summary = {}
        session_item["complaint_graph"] = summary
    summary.setdefault("before_chat", None)
    summary.setdefault("after_chat", None)
    summary.setdefault("after_reflection", None)
    try:
        count = int(summary.get("chat_transition_count", 0) or 0)
    except (TypeError, ValueError):
        count = 0
    summary["chat_transition_count"] = max(0, count)
    summary.setdefault(
        "reflection",
        unavailable_trace("reflection_not_recorded", {"mode": "runtime_capture"}),
    )
    return summary


def _normalize_provenance(value: Any) -> Dict[str, Any]:
    if isinstance(value, Mapping):
        result = copy.deepcopy(dict(value))
        result["mode"] = str(result.get("mode", "") or "")
        return result
    return {"mode": str(value or "")}


def _excerpt_matches(excerpt: str, full_text: str) -> bool:
    left = str(excerpt or "").strip()
    right = str(full_text or "").strip()
    if left == right:
        return True
    if left.endswith("…"):
        return right.startswith(left[:-1])
    return False


def _serialize_timestamp(value: Any) -> str:
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value or "")


def _safe_int_or_none(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
