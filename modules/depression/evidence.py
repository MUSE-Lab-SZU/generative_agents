"""Immutable accepted evidence records; IDs are identities, never text matches."""

import copy
import hashlib
import json
import uuid

CLAIM_KINDS = {
    "symptom_report",
    "function_behavior",
    "belief_appraisal",
    "life_event",
    "interaction_state",
    "plan",
}


def new_id(prefix):
    return prefix + "_" + uuid.uuid4().hex


def digest(value):
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()


def check_id(value):
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 160
        or value.strip() != value
    ):
        raise ValueError("INVALID_ID")
    return value


def register_accepted_message(ledger, record):
    record = copy.deepcopy(record)
    key = check_id(record["message_id"])
    for field in ("case_id", "origin_namespace", "session_instance_id", "speaker_id"):
        check_id(record.get(field))
    if type(record.get("ordinal")) is not int or record["ordinal"] < 1:
        raise ValueError("INVALID_MESSAGE_ORDINAL")
    if not isinstance(record.get("text"), str):
        raise ValueError("INVALID_MESSAGE_TEXT")
    if (
        record.get("acceptance_status") != "accepted"
        or not str(record.get("text", "")).strip()
    ):
        raise ValueError("MESSAGE_NOT_ACCEPTED")
    if record.get("speaker_role") not in {"patient", "counterpart"}:
        raise ValueError("INVALID_SPEAKER")
    expected_source = (
        "patient_utterance" if record["speaker_role"] == "patient" else "context_only"
    )
    if record.get("source_kind") != expected_source:
        raise ValueError("INVALID_MESSAGE_SOURCE")
    record.update(
        evidence_id=key, record_kind="message", text_hash=digest(record["text"])
    )
    if key in ledger and ledger[key] != record:
        raise ValueError("MESSAGE_ID_CONFLICT")
    ledger[key] = record
    return key


def resolve_messages(ledger, refs, case_id, patient_only=True):
    records = []
    for ref in refs:
        check_id(ref)
        row = ledger.get(ref)
        if (
            not row
            or row.get("record_kind") != "message"
            or row.get("message_id") != ref
        ):
            raise ValueError("LEGACY_EVIDENCE_UNRESOLVED")
        if row.get("speaker_role") not in {"patient", "counterpart"} or row.get(
            "source_kind"
        ) != (
            "patient_utterance"
            if row.get("speaker_role") == "patient"
            else "context_only"
        ):
            raise ValueError("INVALID_MESSAGE_SOURCE")
        if row.get("case_id") != case_id or row.get("acceptance_status") != "accepted":
            raise ValueError("INVALID_MESSAGE_SOURCE")
        if (
            digest(row.get("text")) != row.get("text_hash")
            or not row.get("text", "").strip()
        ):
            raise ValueError("MESSAGE_TEXT_CONFLICT")
        if patient_only and (
            row.get("speaker_role") != "patient"
            or row.get("source_kind") != "patient_utterance"
        ):
            raise ValueError("NON_PATIENT_EVIDENCE")
        records.append(copy.deepcopy(row))
    return records


def validate_event_sources(manager, event):
    if event.get("source", "chat") not in {"chat", "reflection"}:
        raise ValueError("INVALID_EVENT_SOURCE")
    meta = event.get("metadata", event)
    refs = meta.get("message_refs", [])
    if not isinstance(refs, list) or not refs:
        raise ValueError("LEGACY_EVIDENCE_UNRESOLVED")
    if len(set(refs)) != len(refs):
        raise ValueError("DUPLICATE_MESSAGE_REF")
    session = meta.get("session_instance_id")
    if session not in manager.session_records:
        raise ValueError("INVALID_SESSION")
    allowed = manager.session_records[session]["message_refs"]
    if any(ref not in allowed for ref in refs):
        raise ValueError("UNAUTHORIZED_MESSAGE_REF")
    rows = resolve_messages(manager.evidence_ledger, refs, manager.case_id)
    if any(row["session_instance_id"] != session for row in rows):
        raise ValueError("INVALID_SESSION")
    snapshot = meta.get("generation_snapshot_ref")
    if snapshot and snapshot not in manager.generation_snapshots:
        raise ValueError("INVALID_GENERATION_SNAPSHOT")
    if event.get("source", "chat") == "chat" and len(rows) == 1:
        if snapshot != rows[0].get("generation_snapshot_ref") or meta.get(
            "generation_attempt_id"
        ) != rows[0].get("generation_attempt_id"):
            raise ValueError("GENERATION_MESSAGE_BINDING_CONFLICT")
    return rows


def validate_claim(claim):
    c = copy.deepcopy(claim)
    required = {
        "claim_id",
        "revision",
        "topic_id",
        "kind",
        "subject",
        "text",
        "assertion_status",
        "actuality",
        "time",
        "acceptance_basis",
        "evidence_refs",
        "accepted_at",
        "accepted_version",
        "supersedes",
    }
    if not isinstance(c, dict) or not required <= c.keys():
        raise ValueError("INCOMPLETE_CLAIM")
    if "claim_status" in c:
        raise ValueError("CLAIM_STATUS_NOT_AUTHORITY")
    check_id(c["claim_id"])
    check_id(c["topic_id"])
    if type(c.get("revision")) is not int or c["revision"] < 1:
        raise ValueError("INVALID_CLAIM_REVISION")
    for key, choices in [
        ("kind", CLAIM_KINDS | {"unknown"}),
        ("assertion_status", {"affirmed", "denied", "uncertain", "unknown"}),
        ("actuality", {"occurred", "ongoing", "intended", "hypothetical", "unknown"}),
        ("acceptance_basis", {"configured", "patient_report"}),
    ]:
        if c.get(key) not in choices:
            raise ValueError("INVALID_CLAIM_" + key.upper())
    if c["kind"] == "unknown" and c["acceptance_basis"] != "configured":
        raise ValueError("UNKNOWN_RUNTIME_CLAIM")
    if (
        not isinstance(c.get("evidence_refs"), list)
        or not c["evidence_refs"]
        or any(not isinstance(r, str) for r in c["evidence_refs"])
    ):
        raise ValueError("INVALID_CLAIM_EVIDENCE")
    for ref in c["evidence_refs"]:
        check_id(ref)
    if (
        not isinstance(c.get("text"), str)
        or not c["text"].strip()
        or not isinstance(c.get("subject"), str)
        or not c["subject"].strip()
    ):
        raise ValueError("INCOMPLETE_CLAIM")
    if (
        not isinstance(c.get("time"), dict)
        or not {"reported_time_text", "effective_start", "effective_end", "precision"}
        <= c["time"].keys()
    ):
        raise ValueError("INVALID_CLAIM_TIME")
    if "accepted_at" not in c or type(c.get("accepted_version")) is not int:
        raise ValueError("INVALID_CLAIM_ACCEPTANCE")
    return c


def json_safe(value):
    """Convert simulated timestamps anywhere in an audit envelope to ISO text."""
    from datetime import datetime

    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return copy.deepcopy(value)


def gate_message_refs(manager, refs, allowed):
    if not isinstance(refs, list) or not refs or len(set(refs)) != len(refs):
        raise ValueError("INVALID_EVIDENCE_REFS")
    if not set(refs) <= set(allowed):
        raise ValueError("UNAUTHORIZED_MESSAGE_REF")
    return resolve_messages(manager.evidence_ledger, refs, manager.case_id)


def update_key(diff, update_kind):
    return digest(
        dict(
            source_message_ids=sorted(diff["supporting_message_ids"]),
            previous_claim_ref=diff.get("previous_claim_ref"),
            claim=diff["proposed_claim"],
            update_kind=update_kind,
        )
    )
