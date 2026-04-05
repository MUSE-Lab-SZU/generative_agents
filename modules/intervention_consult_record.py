"""Consult record domain helpers for forced intervention."""

from __future__ import annotations

import hashlib
import re
from typing import Any, Dict, List, Tuple


SOAP_FIELD_MAP = {
    "s": ["expr", "focus", "impact"],
    "o": ["obs", "source", "resp"],
    "a": ["formulation", "outcome", "understanding"],
}

ISO8601_WITH_TZ_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:Z|[+-]\d{2}:\d{2})$")


CONSULT_RECORD_SCHEMA: Dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "ForcedInterventionConsultRecord",
    "type": "object",
    "additionalProperties": False,
    "required": [
        "record_id",
        "meeting_id",
        "current_session",
        "session_started_at",
        "participants",
        "soap",
    ],
    "properties": {
        "record_id": {"type": "string", "minLength": 1},
        "meeting_id": {"type": "string", "minLength": 1},
        "current_session": {"type": "string", "minLength": 1},
        "session_started_at": {
            "type": "string",
            "pattern": "^\\d{4}-\\d{2}-\\d{2}T\\d{2}:\\d{2}:\\d{2}(?:Z|[+-]\\d{2}:\\d{2})$",
        },
        "participants": {
            "type": "object",
            "additionalProperties": False,
            "required": ["doctor", "patient"],
            "properties": {
                "doctor": {"type": "string", "minLength": 1},
                "patient": {"type": "string", "minLength": 1},
            },
        },
        "soap": {
            "type": "object",
            "additionalProperties": False,
            "required": ["s", "o", "a"],
            "properties": {
                "s": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["expr", "focus", "impact"],
                    "properties": {
                        "expr": {"type": "string", "minLength": 1, "maxLength": 120},
                        "focus": {"type": "string", "minLength": 1, "maxLength": 120},
                        "impact": {"type": "string", "minLength": 1, "maxLength": 120},
                    },
                },
                "o": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["obs", "source", "resp"],
                    "properties": {
                        "obs": {"type": "string", "minLength": 1, "maxLength": 120},
                        "source": {"type": "string", "minLength": 1, "maxLength": 120},
                        "resp": {"type": "string", "minLength": 1, "maxLength": 120},
                    },
                },
                "a": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["formulation", "outcome", "understanding"],
                    "properties": {
                        "formulation": {"type": "string", "minLength": 1, "maxLength": 120},
                        "outcome": {"type": "string", "minLength": 1, "maxLength": 120},
                        "understanding": {"type": "string", "minLength": 1, "maxLength": 120},
                    },
                },
            },
        },
    },
}


class ConsultRecordError(Exception):
    def __init__(self, reason: str, message: str = "", retryable: bool = False):
        super().__init__(message or reason)
        self.reason = str(reason or "unknown_error")
        self.message = str(message or "")
        self.retryable = bool(retryable)


class ConsultRecordPromptError(ConsultRecordError):
    pass


class ConsultRecordValidationError(ConsultRecordError):
    pass


class ConsultRecordWriteError(ConsultRecordError):
    pass


def to_conversation_text(chats: Any) -> str:
    lines: List[str] = []
    for item in chats or []:
        if isinstance(item, (tuple, list)) and len(item) >= 2:
            speaker = str(item[0] or "").strip()
            text = str(item[1] or "").strip()
            if speaker or text:
                lines.append(f"{speaker}: {text}".strip())
    return "\n".join(lines).strip()


def normalize_text(value: Any) -> str:
    text = str(value or "")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_pair_key(doctor: str, patient: str) -> str:
    return f"{str(doctor or '').strip()}::{str(patient or '').strip()}"


def build_dedup_key(meeting_id: str, summary: Any) -> str:
    meeting = str(meeting_id or "").strip()
    normalized_summary = normalize_text(summary)
    digest = hashlib.sha1(normalized_summary.encode("utf-8")).hexdigest()
    return f"{meeting}|{digest}"


def project_whitelist(payload: Any) -> Tuple[Dict[str, Any], List[str]]:
    dropped: List[str] = []
    projected: Dict[str, Any] = {"soap": {"s": {}, "o": {}, "a": {}}}

    if not isinstance(payload, dict):
        return projected, dropped

    for key in payload.keys():
        if key != "soap":
            dropped.append(str(key))

    soap = payload.get("soap", {})
    if not isinstance(soap, dict):
        return projected, dropped

    for section, fields in SOAP_FIELD_MAP.items():
        section_obj = soap.get(section)
        if not isinstance(section_obj, dict):
            continue
        for key in section_obj.keys():
            if key not in fields:
                dropped.append(f"soap.{section}.{key}")
        for field in fields:
            if field in section_obj:
                projected["soap"][section][field] = section_obj.get(field)

    for section in soap.keys():
        if section not in SOAP_FIELD_MAP:
            dropped.append(f"soap.{section}")

    return projected, dropped


def validate_soap_only(payload: Dict[str, Any], soap_text_max_len: int = 120) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        raise ConsultRecordValidationError(reason="soap_structure_invalid")

    soap = payload.get("soap")
    if not isinstance(soap, dict):
        raise ConsultRecordValidationError(reason="soap_structure_invalid")

    normalized = {"soap": {"s": {}, "o": {}, "a": {}}}
    max_len = max(1, int(soap_text_max_len or 120))

    for section, fields in SOAP_FIELD_MAP.items():
        section_obj = soap.get(section)
        if not isinstance(section_obj, dict):
            raise ConsultRecordValidationError(reason="soap_structure_invalid")
        for field in fields:
            if field not in section_obj:
                raise ConsultRecordValidationError(reason="soap_structure_invalid")
            value = section_obj.get(field)
            if not isinstance(value, str):
                raise ConsultRecordValidationError(reason="soap_structure_invalid")
            text = value.strip()
            if not text:
                raise ConsultRecordValidationError(reason="required_field_empty")
            if len(text) > max_len:
                raise ConsultRecordValidationError(reason="soap_length_out_of_range")
            normalized["soap"][section][field] = text

    return normalized


def build_full_record(
    record_id: str,
    meeting_id: str,
    current_session: str,
    session_started_at: str,
    doctor: str,
    patient: str,
    soap: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "record_id": str(record_id or "").strip(),
        "meeting_id": str(meeting_id or "").strip(),
        "current_session": str(current_session or "").strip(),
        "session_started_at": str(session_started_at or "").strip(),
        "participants": {
            "doctor": str(doctor or "").strip(),
            "patient": str(patient or "").strip(),
        },
        "soap": soap or {"s": {}, "o": {}, "a": {}},
    }


def validate_full_record(
    record: Dict[str, Any],
    expected_meeting_id: str,
    expected_doctor: str,
    expected_patient: str,
    soap_text_max_len: int = 120,
) -> Dict[str, Any]:
    if not isinstance(record, dict):
        raise ConsultRecordValidationError(reason="soap_structure_invalid")

    for key in ["record_id", "meeting_id", "current_session", "session_started_at"]:
        if not str(record.get(key, "") or "").strip():
            raise ConsultRecordValidationError(reason="required_field_empty")

    session_started_at = str(record.get("session_started_at", "") or "").strip()
    if not ISO8601_WITH_TZ_RE.match(session_started_at):
        raise ConsultRecordValidationError(reason="session_started_at_invalid")

    meeting_id = str(record.get("meeting_id", "") or "").strip()
    if meeting_id != str(expected_meeting_id or "").strip():
        raise ConsultRecordValidationError(reason="meeting_id_mismatch")

    participants = record.get("participants", {})
    if not isinstance(participants, dict):
        raise ConsultRecordValidationError(reason="pair_mismatch")

    doctor = str(participants.get("doctor", "") or "").strip()
    patient = str(participants.get("patient", "") or "").strip()
    if doctor != str(expected_doctor or "").strip() or patient != str(expected_patient or "").strip():
        raise ConsultRecordValidationError(reason="pair_mismatch")

    normalized_soap = validate_soap_only(record, soap_text_max_len=soap_text_max_len)
    normalized = dict(record)
    normalized["soap"] = normalized_soap["soap"]
    normalized["participants"] = {"doctor": doctor, "patient": patient}
    return normalized


def flatten_record_for_injection(record: Dict[str, Any]) -> Dict[str, str]:
    soap = (record or {}).get("soap", {}) if isinstance(record, dict) else {}
    s = soap.get("s", {}) if isinstance(soap, dict) else {}
    o = soap.get("o", {}) if isinstance(soap, dict) else {}
    a = soap.get("a", {}) if isinstance(soap, dict) else {}
    return {
        "s_expr": str(s.get("expr", "") or "").strip(),
        "s_focus": str(s.get("focus", "") or "").strip(),
        "s_impact": str(s.get("impact", "") or "").strip(),
        "o_obs": str(o.get("obs", "") or "").strip(),
        "o_source": str(o.get("source", "") or "").strip(),
        "o_resp": str(o.get("resp", "") or "").strip(),
        "a_formulation": str(a.get("formulation", "") or "").strip(),
        "a_outcome": str(a.get("outcome", "") or "").strip(),
        "a_understanding": str(a.get("understanding", "") or "").strip(),
    }
