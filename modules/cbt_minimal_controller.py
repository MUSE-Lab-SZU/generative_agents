"""Deterministic helpers for the minimal CBT controller."""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence


PATIENT_STATE_DEFAULT: Dict[str, Any] = {
    "emotion_level": "medium",
    "engagement": "medium",
    "resistance": False,
    "problem_clarity": "vague",
    "risk_level": "possible",
    "evidence": "",
}

PROGRESSIVE_D_PATIENT_STATE_DEFAULT: Dict[str, Any] = {
    "emotion_level": "medium",
    "engagement": "medium",
    "resistance": False,
    "problem_clarity": "vague",
    "risk_level": "none",
}

MINIMAL_PROGRESS_DEFAULT: Dict[str, Any] = {
    "latest_patient_state": {},
    "tracker_meeting_id": "",
    "high_risk_seen": False,
    "last_progress": "none",
    "last_evidence_turn_id": "",
    "last_blocking_factor": "",
    "last_reason": "",
    "meeting_id": "",
}

SIDE_STEP_DEFAULT: Dict[str, Any] = {
    "active": False,
    "reason": "",
    "return_to_subgoal_id": "",
    "started_meeting_id": "",
}

SOFT_STEP_BACK_DEFAULT: Dict[str, Any] = {
    "active": False,
    "target_subgoal_id": "",
    "return_to_subgoal_id": "",
    "reason": "",
}

MINIMAL_STAGE_ADAPTER_VERSION = "minimal_v2_stage_adapter"
PROGRESSIVE_D_CONTROLLER_VERSION = (
    "progressive_d_v3_1_calibrated_batch_control"
)

PROGRESSIVE_D_SUBGOAL_PROGRESS = {"none", "partial", "complete"}
STAGE_ADAPTER_ACTIONS = {
    "stay",
    "advance_subgoal",
    "side_step",
    "advance_stage",
    "soft_step_back",
    "hold",
}
SUBGOAL_PROGRESS_STATUS_RANK = {
    "none": 0,
    "partial": 1,
    "complete": 2,
}
PROGRESSIVE_D_CONTROL_FIELDS = {
    "subgoals",
    "recommended_action",
    "target_subgoal_id",
    "reason",
}
PROGRESSIVE_D_CONTROL_ACTIONS = {
    "stay",
    "side_step",
    "soft_step_back",
    "hold",
}
PROGRESSIVE_D_SUBGOAL_FIELDS = {
    "subgoal_id",
    "progress",
    "blocking_factor",
}
SESSION_TASK_PLANNER_UPDATE_FIELDS = {
    "subgoal_id",
    "status",
}
SESSION_TASK_PLANNER_UPDATE_STATUS = {"partial", "completed"}
SESSION_TASK_PLANNER_FIELDS = {
    "subgoal_updates",
    "next_subgoal_id",
}
SESSION_TASK_PLANNER_CLOSE = "准备收尾"

CANONICAL_MICRO_SKILLS = (
    "choice_scaffolding",
    "focused_question",
    "memory_link",
    "normalization",
    "one_step_pacing",
    "open_question",
    "permission_check",
    "psychoeducation",
    "reflection",
    "reinforcement",
    "scaling",
    "socratic_question",
    "summary",
    "teach_back",
    "validation",
)
_CANONICAL_MICRO_SKILL_SET = frozenset(CANONICAL_MICRO_SKILLS)

SUPPORTIVE_PRIMARY = (
    "safety_check",
    "support_stabilize",
    "alliance_repair",
    "acute_affect_regulation",
    "agenda_focus",
)
SUPPORTIVE_MICRO = (
    "validation",
    "reflection",
    "focused_question",
    "one_step_pacing",
    "open_question",
)
HIGH_LOAD_EXCLUDED_PRIMARY = {
    "downward_arrow",
    "shared_formulation",
    "distortion_labeling",
    "socratic_question",
    "alternative_hypothesis",
    "balanced_belief_consolidation",
    "graded_experiment_design",
    "experiment_review",
    "prediction_capture",
    "implementation_plan",
    "task_grading",
    "relapse_prevention_plan",
}
VAGUE_EXCLUDED_PRIMARY = HIGH_LOAD_EXCLUDED_PRIMARY | {
    "cognitive_distancing",
    "neutral_data_sampling",
    "conditional_rule_elicitation",
}
ACKNOWLEDGEMENT_ONLY = {
    "好",
    "好的",
    "好的谢谢",
    "嗯",
    "嗯嗯",
    "行",
    "好吧",
    "明白了",
    "我明白了",
    "知道了",
    "我知道了",
    "可以",
    "可以的",
    "可以试试看",
    "我试试",
    "那我试试",
    "我可以试试",
    "也许可以试试",
    "也许我可以试试",
    "没问题",
    "谢谢",
}
MIN_EVIDENCE_CONTENT_CHARS = 4

_PLACEHOLDER_RE = re.compile(r"\{\{([A-Z][A-Z0-9_]*)\}\}")
_ACKNOWLEDGEMENT_RE = re.compile(
    r"^(?:"
    r"[嗯啊哦]+"
    r"|好+|好的|好吧|行"
    r"|可以(?:的|吧)?|没问题"
    r"|(?:我)?(?:明白|知道)了?"
    r"|(?:那)?(?:我)?(?:也许|可能)?(?:会|可以|先)?试试(?:看)?(?:吧|的)?"
    r")(?:谢谢)?$"
)


class MinimalControllerConfigError(ValueError):
    """Raised when a minimal-controller prompt or strategy config is invalid."""


class ProgressiveDControlValidationError(ValueError):
    """Raised when a subgoal shadow evaluator output violates its contract."""


class SubgoalProgressValidationError(ValueError):
    """Raised when a persisted subgoal progress update is invalid."""


def load_prompt_template(
    prompt_file: str,
    values: Mapping[str, Any],
    required_placeholders: Iterable[str],
) -> str:
    """Load a UTF-8 ``{{NAME}}`` template and replace all declared values."""

    path_text = str(prompt_file or "").strip()
    if not path_text:
        raise MinimalControllerConfigError("minimal CBT prompt path is empty")
    path = Path(path_text)
    if not path.is_file():
        raise MinimalControllerConfigError(f"minimal CBT prompt file not found: {path_text}")
    try:
        template = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise MinimalControllerConfigError(
            f"minimal CBT prompt file is not readable UTF-8: {path_text}"
        ) from exc
    if not template.strip():
        raise MinimalControllerConfigError(f"minimal CBT prompt file is empty: {path_text}")

    required = {str(name or "").strip() for name in required_placeholders}
    missing_in_template = sorted(name for name in required if f"{{{{{name}}}}}" not in template)
    if missing_in_template:
        raise MinimalControllerConfigError(
            "minimal CBT prompt missing required placeholders in {}: {}".format(
                path_text,
                ", ".join(missing_in_template),
            )
        )
    missing_values = sorted(name for name in required if name not in values)
    if missing_values:
        raise MinimalControllerConfigError(
            "minimal CBT prompt values missing for {}: {}".format(
                path_text,
                ", ".join(missing_values),
            )
        )

    rendered = template
    for name, value in values.items():
        rendered = rendered.replace(f"{{{{{name}}}}}", str(value if value is not None else ""))
    unresolved = sorted(set(_PLACEHOLDER_RE.findall(rendered)))
    if unresolved:
        raise MinimalControllerConfigError(
            "minimal CBT prompt has unresolved placeholders in {}: {}".format(
                path_text,
                ", ".join(unresolved),
            )
        )
    return rendered


def normalize_patient_state(payload: Any, patient_utterances: Sequence[str] = ()) -> Dict[str, Any]:
    raw = payload if isinstance(payload, dict) else {}
    normalized = dict(PATIENT_STATE_DEFAULT)
    enum_fields = {
        "emotion_level": {"low", "medium", "high"},
        "engagement": {"low", "medium", "high"},
        "problem_clarity": {"vague", "clear"},
        "risk_level": {"none", "possible", "high"},
    }
    for field, allowed in enum_fields.items():
        value = raw.get(field)
        if isinstance(value, str) and value.strip().lower() in allowed:
            normalized[field] = value.strip().lower()
    if isinstance(raw.get("resistance"), bool):
        normalized["resistance"] = raw["resistance"]
    evidence = raw.get("evidence")
    if isinstance(evidence, str) and quote_in_utterances(evidence, patient_utterances):
        normalized["evidence"] = evidence.strip()
    return normalized


def normalize_progressive_d_patient_state(payload: Any) -> Dict[str, Any]:
    """Normalize the evidence-free five-field Progressive D tracker output."""

    raw = payload if isinstance(payload, dict) else {}
    normalized = dict(PROGRESSIVE_D_PATIENT_STATE_DEFAULT)
    enum_fields = {
        "emotion_level": {"low", "medium", "high"},
        "engagement": {"low", "medium", "high"},
        "problem_clarity": {"vague", "clear"},
        "risk_level": {"none", "possible", "high"},
    }
    for field, allowed in enum_fields.items():
        value = raw.get(field)
        if isinstance(value, str) and value.strip().lower() in allowed:
            normalized[field] = value.strip().lower()
    if isinstance(raw.get("resistance"), bool):
        normalized["resistance"] = raw["resistance"]
    return normalized


def load_strategy_map(strategy_file: str) -> Dict[str, Dict[str, List[str]]]:
    path_text = str(strategy_file or "").strip()
    if not path_text:
        raise MinimalControllerConfigError("CBT strategy map path is empty")
    path = Path(path_text)
    if not path.is_file():
        raise MinimalControllerConfigError(f"CBT strategy map file not found: {path_text}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise MinimalControllerConfigError(f"CBT strategy map is invalid: {path_text}") from exc
    if not isinstance(payload, dict):
        raise MinimalControllerConfigError(f"CBT strategy map root must be an object: {path_text}")

    normalized: Dict[str, Dict[str, List[str]]] = {}
    for session_id, item in payload.items():
        if not isinstance(item, dict):
            raise MinimalControllerConfigError(
                f"CBT strategy map entry must be an object: {session_id}"
            )
        primary = _string_list(item.get("primary_strategies"))
        micro = _string_list(item.get("micro_skills"))
        if not primary or not micro:
            raise MinimalControllerConfigError(
                f"CBT strategy map entry requires non-empty strategy lists: {session_id}"
            )
        unknown_micro = sorted(set(micro) - _CANONICAL_MICRO_SKILL_SET)
        if unknown_micro:
            raise MinimalControllerConfigError(
                "CBT strategy map entry contains non-canonical micro skills for {}: {}".format(
                    session_id,
                    ", ".join(unknown_micro),
                )
            )
        normalized[str(session_id)] = {
            "primary_strategies": primary,
            "micro_skills": micro,
        }
    return normalized


def load_term_glossary(
    glossary_file: str,
    strategy_map: Mapping[str, Mapping[str, Sequence[str]]],
) -> Dict[str, Any]:
    path_text = str(glossary_file or "").strip()
    path = Path(path_text)
    if not path_text or not path.is_file():
        raise MinimalControllerConfigError(
            f"CBT term glossary file not found: {path_text}"
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise MinimalControllerConfigError(
            f"CBT term glossary is invalid: {path_text}"
        ) from exc
    if not isinstance(payload, dict) or set(payload) != {
        "primary_strategies",
        "micro_skills",
        "macro_stages",
        "actions",
    }:
        raise MinimalControllerConfigError(
            "CBT term glossary must contain primary_strategies, "
            "micro_skills, macro_stages and actions"
        )

    primary_ids = {
        item
        for session in strategy_map.values()
        if isinstance(session, Mapping)
        for item in _string_list(session.get("primary_strategies"))
    }
    micro_ids = {
        item
        for session in strategy_map.values()
        if isinstance(session, Mapping)
        for item in _string_list(session.get("micro_skills"))
    }
    normalized: Dict[str, Any] = {
        "primary_strategies": {},
        "micro_skills": {},
        "macro_stages": {},
        "actions": {},
    }
    for section, required_ids in (
        ("primary_strategies", primary_ids),
        ("micro_skills", micro_ids),
    ):
        raw_section = payload.get(section)
        if not isinstance(raw_section, dict):
            raise MinimalControllerConfigError(
                f"CBT term glossary section must be an object: {section}"
            )
        missing = sorted(required_ids - set(raw_section))
        if missing:
            raise MinimalControllerConfigError(
                "CBT term glossary is missing {}: {}".format(
                    section,
                    ", ".join(missing),
                )
            )
        for term_id, item in raw_section.items():
            if not isinstance(item, dict) or set(item) != {
                "label",
                "instruction",
            }:
                raise MinimalControllerConfigError(
                    f"CBT term glossary entry is invalid: {section}.{term_id}"
                )
            label = _safe_text(item.get("label"))
            instruction = _safe_text(item.get("instruction"))
            if not label or not instruction:
                raise MinimalControllerConfigError(
                    f"CBT term glossary entry is empty: {section}.{term_id}"
                )
            normalized[section][str(term_id)] = {
                "label": label,
                "instruction": instruction,
            }
    for section in ("macro_stages", "actions"):
        raw_section = payload.get(section)
        if not isinstance(raw_section, dict):
            raise MinimalControllerConfigError(
                f"CBT term glossary section must be an object: {section}"
            )
        normalized[section] = {
            str(term_id): _safe_text(label)
            for term_id, label in raw_section.items()
            if _safe_text(label)
        }
    missing_actions = sorted(STAGE_ADAPTER_ACTIONS - set(normalized["actions"]))
    if missing_actions:
        raise MinimalControllerConfigError(
            "CBT term glossary is missing actions: {}".format(
                ", ".join(missing_actions)
            )
        )
    return normalized


def load_stage_subgoal_map(
    stage_subgoals_file: str,
) -> Dict[str, Dict[str, Any]]:
    """Load the static legacy-session to macro-stage/subgoal mapping."""

    path_text = str(stage_subgoals_file or "").strip()
    if not path_text:
        raise MinimalControllerConfigError("CBT stage/subgoal map path is empty")
    path = Path(path_text)
    if not path.is_file():
        raise MinimalControllerConfigError(
            f"CBT stage/subgoal map file not found: {path_text}"
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise MinimalControllerConfigError(
            f"CBT stage/subgoal map is invalid: {path_text}"
        ) from exc
    if not isinstance(payload, dict):
        raise MinimalControllerConfigError(
            f"CBT stage/subgoal map root must be an object: {path_text}"
        )

    normalized: Dict[str, Dict[str, Any]] = {}
    all_subgoal_ids = set()
    for session_id, item in payload.items():
        session_text = str(session_id or "").strip()
        if not session_text or not isinstance(item, dict):
            raise MinimalControllerConfigError(
                f"CBT stage/subgoal map entry must be an object: {session_id}"
            )
        if set(item) != {"macro_stage", "subgoals"}:
            raise MinimalControllerConfigError(
                f"CBT stage/subgoal map entry has invalid fields: {session_text}"
            )
        macro_stage = _safe_text(item.get("macro_stage"))
        raw_subgoals = item.get("subgoals")
        if not macro_stage or not isinstance(raw_subgoals, list) or not raw_subgoals:
            raise MinimalControllerConfigError(
                f"CBT stage/subgoal map entry requires macro_stage and subgoals: {session_text}"
            )
        subgoals: List[Dict[str, str]] = []
        session_subgoal_ids = set()
        for raw_subgoal in raw_subgoals:
            if not isinstance(raw_subgoal, dict) or set(raw_subgoal) != {
                "id",
                "goal",
                "completion_hint",
            }:
                raise MinimalControllerConfigError(
                    f"CBT stage/subgoal entry has invalid fields: {session_text}"
                )
            subgoal = {
                "id": _safe_text(raw_subgoal.get("id")),
                "goal": _safe_text(raw_subgoal.get("goal")),
                "completion_hint": _safe_text(raw_subgoal.get("completion_hint")),
            }
            if not all(subgoal.values()):
                raise MinimalControllerConfigError(
                    f"CBT stage/subgoal entry requires non-empty values: {session_text}"
                )
            subgoal_id = subgoal["id"]
            if subgoal_id in session_subgoal_ids or subgoal_id in all_subgoal_ids:
                raise MinimalControllerConfigError(
                    f"CBT stage/subgoal ID must be globally unique: {subgoal_id}"
                )
            session_subgoal_ids.add(subgoal_id)
            all_subgoal_ids.add(subgoal_id)
            subgoals.append(subgoal)
        normalized[session_text] = {
            "macro_stage": macro_stage,
            "subgoals": subgoals,
        }
    return normalized


def initialize_subgoal_progress_state(
    current_legacy_session: str,
    stage_item: Mapping[str, Any],
    existing: Any = None,
    preserve_active: bool = False,
    include_side_step: bool = False,
    include_stage_adapter: bool = False,
    include_evidence_turn_id: bool = True,
    include_batch_control: bool = False,
    controller_version: str = MINIMAL_STAGE_ADAPTER_VERSION,
) -> Dict[str, Any]:
    """Return compact progress for exactly one configured legacy Prompt."""

    legacy_session = _safe_text(current_legacy_session)
    macro_stage = _safe_text(stage_item.get("macro_stage"))
    configured_subgoals = stage_item.get("subgoals")
    if (
        not legacy_session
        or not macro_stage
        or not isinstance(configured_subgoals, list)
        or not configured_subgoals
    ):
        raise SubgoalProgressValidationError(
            "subgoal progress requires a configured legacy session and macro stage"
        )

    existing_state = existing if isinstance(existing, dict) else {}
    same_prompt = bool(
        str(existing_state.get("current_legacy_session", "") or "") == legacy_session
        and str(existing_state.get("current_macro_stage", "") or "") == macro_stage
    )
    existing_subgoals = (
        existing_state.get("subgoals", {})
        if same_prompt and isinstance(existing_state.get("subgoals"), dict)
        else {}
    )
    subgoals: Dict[str, Dict[str, str]] = {}
    for configured in configured_subgoals:
        if not isinstance(configured, Mapping):
            raise SubgoalProgressValidationError(
                "subgoal progress received an invalid static subgoal definition"
            )
        subgoal_id = _safe_text(configured.get("id"))
        if not subgoal_id:
            raise SubgoalProgressValidationError(
                "subgoal progress static subgoal ID is empty"
            )
        prior = existing_subgoals.get(subgoal_id, {})
        if not isinstance(prior, dict):
            prior = {}
        status = str(prior.get("status", "none") or "none").strip().lower()
        if status not in SUBGOAL_PROGRESS_STATUS_RANK:
            status = "none"
        normalized_subgoal = {
            "status": status,
            "blocking_factor": _safe_text(prior.get("blocking_factor")),
            "last_meeting_id": _safe_text(prior.get("last_meeting_id")),
        }
        if include_evidence_turn_id:
            normalized_subgoal["evidence_turn_id"] = _safe_text(
                prior.get("evidence_turn_id")
            )
        subgoals[subgoal_id] = normalized_subgoal

    state = {
        "current_legacy_session": legacy_session,
        "current_macro_stage": macro_stage,
        "subgoals": subgoals,
        "active_subgoal_id": "",
    }
    allowed_ids = list(subgoals)
    prior_active = _safe_text(existing_state.get("active_subgoal_id"))
    if preserve_active and same_prompt and prior_active in allowed_ids:
        state["active_subgoal_id"] = prior_active
    else:
        active = select_active_subgoal(stage_item, state)
        state["active_subgoal_id"] = active["id"]
    if include_side_step or include_stage_adapter:
        prior_side_step = (
            existing_state.get("side_step", {})
            if same_prompt and isinstance(existing_state.get("side_step"), dict)
            else {}
        )
        return_to = _safe_text(prior_side_step.get("return_to_subgoal_id"))
        active_side_step = bool(prior_side_step.get("active", False))
        if return_to not in allowed_ids:
            active_side_step = False
            return_to = ""
        state["side_step"] = {
            "active": active_side_step,
            "reason": (
                _safe_text(prior_side_step.get("reason"))
                if active_side_step
                else ""
            ),
            "return_to_subgoal_id": return_to if active_side_step else "",
            "started_meeting_id": (
                _safe_text(prior_side_step.get("started_meeting_id"))
                if active_side_step
                else ""
            ),
        }
    if include_stage_adapter:
        state["cbt_controller_version"] = (
            _safe_text(controller_version)
            or MINIMAL_STAGE_ADAPTER_VERSION
        )
        prior_soft_step_back = (
            existing_state.get("soft_step_back", {})
            if same_prompt
            and isinstance(existing_state.get("soft_step_back"), dict)
            else {}
        )
        target = _safe_text(prior_soft_step_back.get("target_subgoal_id"))
        return_to = _safe_text(
            prior_soft_step_back.get("return_to_subgoal_id")
        )
        active_soft_step_back = bool(
            prior_soft_step_back.get("active", False)
        )
        if return_to not in allowed_ids or not target:
            active_soft_step_back = False
            target = ""
            return_to = ""
        state["soft_step_back"] = {
            "active": active_soft_step_back,
            "target_subgoal_id": target if active_soft_step_back else "",
            "return_to_subgoal_id": return_to if active_soft_step_back else "",
            "reason": (
                _safe_text(prior_soft_step_back.get("reason"))
                if active_soft_step_back
                else ""
            ),
        }
        if (
            state.get("side_step", {}).get("active", False)
            and state["soft_step_back"]["active"]
        ):
            state["side_step"] = copy_side_step_default()
            state["soft_step_back"] = copy_soft_step_back_default()
    if include_batch_control:
        prior_meeting_ids = (
            existing_state.get("meeting_ids_in_current_prompt", [])
            if same_prompt
            else []
        )
        meeting_ids: List[str] = []
        if isinstance(prior_meeting_ids, list):
            meeting_ids = _dedupe(_string_list(prior_meeting_ids))
        state["meeting_ids_in_current_prompt"] = meeting_ids
        state["meetings_in_current_prompt"] = len(meeting_ids)
    return state


def select_active_subgoal(
    stage_item: Mapping[str, Any],
    progress_state: Mapping[str, Any],
) -> Dict[str, Any]:
    """Select the first incomplete configured subgoal, or the final one."""

    configured_subgoals = stage_item.get("subgoals")
    if not isinstance(configured_subgoals, list) or not configured_subgoals:
        raise SubgoalProgressValidationError(
            "active subgoal selection requires configured subgoals"
        )
    progress_subgoals = progress_state.get("subgoals", {})
    if not isinstance(progress_subgoals, Mapping):
        progress_subgoals = {}

    selected: Mapping[str, Any] = configured_subgoals[-1]
    all_complete = True
    for configured in configured_subgoals:
        if not isinstance(configured, Mapping):
            raise SubgoalProgressValidationError(
                "active subgoal selection received an invalid definition"
            )
        subgoal_id = _safe_text(configured.get("id"))
        item = progress_subgoals.get(subgoal_id, {})
        status = (
            str(item.get("status", "none") or "none").strip().lower()
            if isinstance(item, Mapping)
            else "none"
        )
        if status != "complete":
            selected = configured
            all_complete = False
            break
    return {
        "id": _safe_text(selected.get("id")),
        "goal": _safe_text(selected.get("goal")),
        "all_complete": all_complete,
    }



def compact_progressive_d_subgoal_progress_text(
    stage_item: Mapping[str, Any],
    progress_state: Mapping[str, Any],
    macro_stage_labels: Mapping[str, str] | None = None,
) -> str:
    """Render every D subgoal status instead of one active target."""

    configured = stage_item.get("subgoals")
    if not isinstance(configured, list) or not configured:
        raise SubgoalProgressValidationError(
            "Progressive D progress rendering requires configured subgoals"
        )
    stored = progress_state.get("subgoals", {})
    if not isinstance(stored, Mapping):
        stored = {}
    status_labels = {
        "none": "无进展",
        "partial": "部分完成",
        "complete": "完成",
    }
    macro_id = _safe_text(
        progress_state.get("current_macro_stage")
    ) or "未知"
    labels = (
        macro_stage_labels
        if isinstance(macro_stage_labels, Mapping)
        else {}
    )
    macro_label = _safe_text(labels.get(macro_id))
    lines = [
        "当前治疗阶段={}{}".format(
            macro_label or macro_id,
            f"（{macro_id}）" if macro_label else "",
        ),
        "当前固定会谈提纲的全部小目标：",
    ]
    for index, item in enumerate(configured, start=1):
        if not isinstance(item, Mapping):
            continue
        subgoal_id = _safe_text(item.get("id"))
        current = stored.get(subgoal_id, {})
        status = (
            _safe_text(current.get("status")).lower()
            if isinstance(current, Mapping)
            else "none"
        ) or "none"
        blocker = (
            _safe_text(current.get("blocking_factor"))
            if isinstance(current, Mapping)
            else ""
        )
        line = "- {}. [{}] {}：{}".format(
            index,
            subgoal_id,
            _safe_text(item.get("goal")) or subgoal_id,
            status_labels.get(status, "无进展"),
        )
        if blocker and status != "complete":
            line += "；当前阻碍={}".format(blocker)
        lines.append(line)
    lines.extend(
        [
            "当前提纲已处理会面数={}".format(
                int(
                    progress_state.get(
                        "meetings_in_current_prompt",
                        0,
                    )
                    or 0
                )
            ),
            "默认按以上顺序优先推进尚无进展的小目标；这不是硬约束，患者自然涉及后续目标时可同步处理。",
            "部分完成表示已有足够进展，可以继续推进其他目标；仅在后续确有必要时再自然补充，不要求先完全达标。",
        ]
    )
    return "\n".join(lines)



def copy_side_step_default() -> Dict[str, Any]:
    return dict(SIDE_STEP_DEFAULT)


def copy_soft_step_back_default() -> Dict[str, Any]:
    return dict(SOFT_STEP_BACK_DEFAULT)



def normalize_progressive_d_control_eval(
    payload: Any,
    configured_subgoal_ids: Sequence[str],
    allowed_soft_targets: Sequence[str] = (),
) -> Dict[str, Any]:
    """Strictly validate the evidence-free Progressive D batch result."""

    if not isinstance(payload, dict) or set(payload) != PROGRESSIVE_D_CONTROL_FIELDS:
        raise ProgressiveDControlValidationError(
            "Progressive D control output must contain exactly the four allowed fields"
        )
    configured_ids = _string_list(configured_subgoal_ids)
    raw_subgoals = payload.get("subgoals")
    if not isinstance(raw_subgoals, list) or len(raw_subgoals) != len(configured_ids):
        raise ProgressiveDControlValidationError(
            "Progressive D control output must cover every configured subgoal exactly once"
        )
    normalized_subgoals: List[Dict[str, str]] = []
    seen: List[str] = []
    for index, item in enumerate(raw_subgoals):
        if not isinstance(item, dict) or set(item) != PROGRESSIVE_D_SUBGOAL_FIELDS:
            raise ProgressiveDControlValidationError(
                "Progressive D subgoal item must contain exactly three allowed fields"
            )
        subgoal_id = _safe_text(item.get("subgoal_id"))
        if subgoal_id != configured_ids[index] or subgoal_id in seen:
            raise ProgressiveDControlValidationError(
                "Progressive D subgoals must use configured IDs in configured order"
            )
        progress = _safe_text(item.get("progress")).lower()
        if progress not in PROGRESSIVE_D_SUBGOAL_PROGRESS:
            raise ProgressiveDControlValidationError(
                f"Progressive D control output contains invalid progress: {progress}"
            )
        blocking_factor = item.get("blocking_factor")
        if not isinstance(blocking_factor, str):
            raise ProgressiveDControlValidationError(
                "Progressive D blocking_factor must be a string"
            )
        seen.append(subgoal_id)
        normalized_subgoals.append(
            {
                "subgoal_id": subgoal_id,
                "progress": progress,
                "blocking_factor": blocking_factor.strip(),
            }
        )

    action = _safe_text(payload.get("recommended_action")).lower()
    if action not in PROGRESSIVE_D_CONTROL_ACTIONS:
        raise ProgressiveDControlValidationError(
            f"Progressive D control output contains invalid action: {action}"
        )
    target = _safe_text(payload.get("target_subgoal_id"))
    allowed_targets = set(_string_list(allowed_soft_targets))
    if action == "soft_step_back":
        if not target or target not in allowed_targets:
            raise ProgressiveDControlValidationError(
                "Progressive D soft_step_back target is not allowed"
            )
    elif target:
        raise ProgressiveDControlValidationError(
            "Progressive D target_subgoal_id must be empty unless soft_step_back is selected"
        )
    reason = payload.get("reason")
    if not isinstance(reason, str):
        raise ProgressiveDControlValidationError(
            "Progressive D reason must be a string"
        )
    return {
        "subgoals": normalized_subgoals,
        "recommended_action": action,
        "target_subgoal_id": target,
        "reason": reason.strip(),
    }


def merge_progressive_d_batch_progress(
    progress_state: Dict[str, Any],
    stage_item: Mapping[str, Any],
    batch_subgoals: Sequence[Mapping[str, Any]],
    meeting_id: str,
) -> bool:
    """Merge one validated batch without regression or duplicate counting."""

    configured = stage_item.get("subgoals")
    if not isinstance(configured, list) or not configured:
        raise SubgoalProgressValidationError(
            "Progressive D batch merge requires configured subgoals"
        )
    configured_ids = [
        _safe_text(item.get("id"))
        for item in configured
        if isinstance(item, Mapping)
    ]
    incoming_ids = [
        _safe_text(item.get("subgoal_id"))
        for item in batch_subgoals
        if isinstance(item, Mapping)
    ]
    if incoming_ids != configured_ids:
        raise SubgoalProgressValidationError(
            "Progressive D batch subgoals do not match configuration order"
        )
    meeting_id_text = _safe_text(meeting_id)
    if not meeting_id_text:
        raise SubgoalProgressValidationError(
            "Progressive D batch merge requires meeting_id"
        )
    processed = progress_state.setdefault("meeting_ids_in_current_prompt", [])
    if not isinstance(processed, list):
        processed = []
        progress_state["meeting_ids_in_current_prompt"] = processed
    if meeting_id_text in processed:
        return False
    subgoal_state = progress_state.get("subgoals")
    if not isinstance(subgoal_state, dict) or set(subgoal_state) != set(configured_ids):
        raise SubgoalProgressValidationError(
            "Progressive D persisted subgoals do not match configuration"
        )
    for incoming in batch_subgoals:
        subgoal_id = _safe_text(incoming.get("subgoal_id"))
        progress = _safe_text(incoming.get("progress")).lower()
        if progress not in SUBGOAL_PROGRESS_STATUS_RANK:
            raise SubgoalProgressValidationError(
                f"Progressive D batch contains invalid progress: {progress}"
            )
        current = subgoal_state[subgoal_id]
        current_status = _safe_text(current.get("status")).lower() or "none"
        if (
            SUBGOAL_PROGRESS_STATUS_RANK[progress]
            >= SUBGOAL_PROGRESS_STATUS_RANK.get(current_status, 0)
        ):
            current.update(
                {
                    "status": progress,
                    "blocking_factor": _safe_text(
                        incoming.get("blocking_factor")
                    ),
                    "last_meeting_id": meeting_id_text,
                }
            )
            current.pop("evidence_turn_id", None)
    processed.append(meeting_id_text)
    progress_state["meetings_in_current_prompt"] = len(processed)
    selected = select_active_subgoal(stage_item, progress_state)
    progress_state["active_subgoal_id"] = selected["id"]
    return True


def progressive_d_completion_audit(
    progress_state: Mapping[str, Any],
    *,
    high_risk_seen: bool,
    threshold_ratio: float = 0.60,
    meeting_cap: int = 3,
    meeting_cap_threshold_ratio: float = 0.40,
) -> Dict[str, Any]:
    """Compute the deterministic calibrated completion decision."""

    subgoals = progress_state.get("subgoals", {})
    values = list(subgoals.values()) if isinstance(subgoals, Mapping) else []
    score = 0.0
    complete_count = 0
    for item in values:
        status = (
            _safe_text(item.get("status")).lower()
            if isinstance(item, Mapping)
            else "none"
        )
        if status == "complete":
            score += 1.0
            complete_count += 1
        elif status == "partial":
            score += 0.5
    threshold = len(values) * float(threshold_ratio)
    try:
        meetings = int(progress_state.get("meetings_in_current_prompt", 0) or 0)
    except (TypeError, ValueError):
        meetings = 0
    score_ready = bool(
        values
        and score >= threshold
        and complete_count >= 1
    )
    cap_ready = bool(
        values
        and meetings >= max(1, int(meeting_cap))
        and complete_count >= 1
        and score >= len(values) * float(meeting_cap_threshold_ratio)
    )
    completed = bool((score_ready or cap_ready) and not high_risk_seen)
    reason = ""
    if completed:
        reason = "score_threshold" if score_ready else "meeting_cap"
    elif high_risk_seen:
        reason = "high_risk_seen"
    else:
        reason = "not_ready"
    return {
        "session_prompt_complete": completed,
        "completion_reason": reason,
        "completion_score": score,
        "completion_threshold": threshold,
        "meetings_in_current_prompt": meetings,
    }


def route_strategies(
    current_session: str,
    patient_state: Mapping[str, Any],
    strategy_map: Mapping[str, Mapping[str, Sequence[str]]],
    side_step_active: bool = False,
) -> Dict[str, Any]:
    item = strategy_map.get(str(current_session or ""))
    if not isinstance(item, Mapping):
        raise MinimalControllerConfigError(
            f"CBT strategy map has no entry for current session: {current_session}"
        )
    base_primary = _dedupe(_string_list(item.get("primary_strategies")))
    base_micro = _dedupe(_string_list(item.get("micro_skills")))
    if not base_primary or not base_micro:
        raise MinimalControllerConfigError(
            f"CBT strategy map entry requires non-empty strategy lists: {current_session}"
        )
    unknown_micro = sorted(set(base_micro) - _CANONICAL_MICRO_SKILL_SET)
    if unknown_micro:
        raise MinimalControllerConfigError(
            "CBT strategy map entry contains non-canonical micro skills for {}: {}".format(
                current_session,
                ", ".join(unknown_micro),
            )
        )

    risk = str(patient_state.get("risk_level", "possible") or "possible")
    emotion = str(patient_state.get("emotion_level", "medium") or "medium")
    resistance = bool(patient_state.get("resistance", False))
    clarity = str(patient_state.get("problem_clarity", "vague") or "vague")

    if risk == "high":
        return {
            "allowed_primary_strategies": ["safety_check"],
            "allowed_micro_skills": [
                "validation",
                "reflection",
                "focused_question",
                "one_step_pacing",
            ],
            "applied_rule": "risk_high",
            "explanation": "高风险覆盖常规 CBT 推进，仅保留安全检查与支持性微技能。",
        }

    if risk == "possible":
        return {
            "allowed_primary_strategies": ["safety_check"],
            "allowed_micro_skills": [
                "validation",
                "reflection",
                "focused_question",
                "one_step_pacing",
            ],
            "applied_rule": "risk_possible",
            "explanation": "风险信息不足或抽取失败，先完成安全核对，不进入常规 CBT 推进。",
        }

    if bool(side_step_active):
        preferred_primary = [
            value
            for value in (
                "support_stabilize",
                "alliance_repair",
                "acute_affect_regulation",
                "agenda_focus",
            )
            if value in base_primary or value in SUPPORTIVE_PRIMARY
        ]
        remaining_primary = [
            value for value in base_primary if value not in HIGH_LOAD_EXCLUDED_PRIMARY
        ]
        return {
            "allowed_primary_strategies": _dedupe(
                preferred_primary + remaining_primary
            ),
            "allowed_micro_skills": _dedupe(
                [
                    "validation",
                    "reflection",
                    "focused_question",
                    "one_step_pacing",
                    "open_question",
                ]
                + base_micro
            ),
            "applied_rule": "side_step_active",
            "explanation": "临时任务激活，优先支持、澄清、联盟修复与稳定。",
        }

    if emotion == "high" or resistance:
        preferred_primary = [
            value
            for value in ("support_stabilize", "alliance_repair", "acute_affect_regulation")
            if value in base_primary or value in SUPPORTIVE_PRIMARY
        ]
        remaining_primary = [
            value for value in base_primary if value not in HIGH_LOAD_EXCLUDED_PRIMARY
        ]
        return {
            "allowed_primary_strategies": _dedupe(preferred_primary + remaining_primary),
            "allowed_micro_skills": _dedupe(
                [
                    "validation",
                    "reflection",
                    "focused_question",
                    "one_step_pacing",
                ]
                + base_micro
            ),
            "applied_rule": "high_emotion_or_resistance",
            "explanation": "情绪负荷高或存在抵触，优先承接、确认感受与澄清，并排除高负担技术。",
        }

    if clarity == "vague":
        preferred_primary = [
            value for value in ("agenda_focus", "cognitive_triangle_mapping", "functional_analysis")
            if value in base_primary
        ]
        remaining_primary = [
            value for value in base_primary if value not in VAGUE_EXCLUDED_PRIMARY
        ]
        if not preferred_primary:
            preferred_primary = [
                "support_stabilize"
                if "support_stabilize" in remaining_primary
                else (remaining_primary[0] if remaining_primary else base_primary[0])
            ]
        return {
            "allowed_primary_strategies": _dedupe(preferred_primary + remaining_primary),
            "allowed_micro_skills": _dedupe(
                ["focused_question", "open_question"] + base_micro
            ),
            "applied_rule": "problem_vague",
            "explanation": "问题材料仍模糊，优先澄清、具体化与开放式提问。",
        }

    return {
        "allowed_primary_strategies": base_primary,
        "allowed_micro_skills": base_micro,
        "applied_rule": "session_default",
        "explanation": "患者状态未触发覆盖规则，使用当前 Prompt 的配置候选。",
    }


def normalize_minimal_judge_output(
    payload: Any,
    route: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Validate the Judge's goal/termination decision.

    ``route`` remains an optional ignored argument so callers outside the runtime
    do not break while the strategy choice moves to the local selector.
    """

    raw = payload if isinstance(payload, dict) else {}
    goal = raw.get("turn_goal")
    terminate = raw.get("terminate")
    expected_fields = {"turn_goal", "terminate"}
    valid = (
        set(raw) == expected_fields
        and isinstance(goal, str)
        and bool(goal.strip())
        and isinstance(terminate, bool)
    )
    if valid:
        return {
            "turn_goal": goal.strip(),
            "terminate": terminate,
        }
    return safe_judge_fallback()


def normalize_progressive_d_judge_output(
    payload: Any,
    route: Mapping[str, Any] | None = None,
    termination_check_enabled: bool = True,
) -> Dict[str, Any]:
    """Normalize the D Judge output under the dynamic termination contract."""

    if termination_check_enabled:
        return normalize_minimal_judge_output(payload, route)

    raw = payload if isinstance(payload, dict) else {}
    goal = raw.get("turn_goal")
    terminate_present = "terminate" in raw
    valid = bool(
        set(raw) in ({"turn_goal"}, {"turn_goal", "terminate"})
        and isinstance(goal, str)
        and goal.strip()
        and (
            not terminate_present
            or isinstance(raw.get("terminate"), bool)
        )
    )
    if valid:
        return {
            "turn_goal": goal.strip(),
            "terminate": False,
        }
    return safe_judge_fallback()


def normalize_response_strategy_selector_output(
    payload: Any,
    route: Mapping[str, Any],
) -> Dict[str, str]:
    """Validate a selector choice strictly against the router candidates."""

    raw = payload if isinstance(payload, dict) else {}
    allowed_primary = _string_list(route.get("allowed_primary_strategies"))
    allowed_micro = _string_list(route.get("allowed_micro_skills"))
    primary = raw.get("primary_strategy")
    micro = raw.get("micro_skill")
    if (
        set(raw) == {"primary_strategy", "micro_skill"}
        and isinstance(primary, str)
        and primary in allowed_primary
        and isinstance(micro, str)
        and micro in allowed_micro
    ):
        return {
            "primary_strategy": primary,
            "micro_skill": micro,
        }
    return safe_strategy_selector_fallback(allowed_primary, allowed_micro)


def normalize_session_task_planner_output(
    payload: Any,
    configured_subgoal_ids: Sequence[str],
) -> Dict[str, Any]:
    """Validate the deliberately small per-turn Session Task Planner contract."""

    raw = payload if isinstance(payload, dict) else {}
    if set(raw) != SESSION_TASK_PLANNER_FIELDS:
        raise SubgoalProgressValidationError(
            "Session Task Planner output must contain exactly two fields"
        )

    raw_updates = raw.get("subgoal_updates")
    if not isinstance(raw_updates, list):
        raise SubgoalProgressValidationError(
            "Session Task Planner subgoal_updates must be a list"
        )
    allowed_ids = set(_string_list(configured_subgoal_ids))
    updates: List[Dict[str, str]] = []
    seen = set()
    for item in raw_updates:
        if (
            not isinstance(item, dict)
            or set(item) != SESSION_TASK_PLANNER_UPDATE_FIELDS
        ):
            raise SubgoalProgressValidationError(
                "Session Task Planner update must contain exactly two fields"
            )
        subgoal_id = _safe_text(item.get("subgoal_id"))
        status = _safe_text(item.get("status")).lower()
        if (
            subgoal_id not in allowed_ids
            or subgoal_id in seen
            or status not in SESSION_TASK_PLANNER_UPDATE_STATUS
        ):
            raise SubgoalProgressValidationError(
                "Session Task Planner subgoal update is invalid"
            )
        seen.add(subgoal_id)
        updates.append({"subgoal_id": subgoal_id, "status": status})

    next_subgoal_id = _safe_text(raw.get("next_subgoal_id"))
    if (
        next_subgoal_id not in allowed_ids
        and next_subgoal_id != SESSION_TASK_PLANNER_CLOSE
    ):
        raise SubgoalProgressValidationError(
            "Session Task Planner next_subgoal_id is invalid"
        )
    return {
        "subgoal_updates": updates,
        "next_subgoal_id": next_subgoal_id,
    }


def compact_session_task_plan_text(
    planner_output: Mapping[str, Any],
    stage_item: Mapping[str, Any],
) -> str:
    """Render Planner JSON as concise, non-binding task advice for the Judge."""

    configured = stage_item.get("subgoals", [])
    goal_by_id = {
        _safe_text(item.get("id")): _safe_text(item.get("goal"))
        for item in configured
        if isinstance(item, Mapping) and _safe_text(item.get("id"))
    }
    updates = planner_output.get("subgoal_updates", [])
    lines = ["本轮新增小目标进展："]
    if isinstance(updates, list) and updates:
        status_labels = {"partial": "部分完成", "completed": "完成"}
        for item in updates:
            if not isinstance(item, Mapping):
                continue
            subgoal_id = _safe_text(item.get("subgoal_id"))
            lines.append(
                "- [{}] {}：{}".format(
                    subgoal_id,
                    goal_by_id.get(subgoal_id, subgoal_id),
                    status_labels.get(_safe_text(item.get("status")), "有新增进展"),
                )
            )
    else:
        lines.append("- 无")

    next_subgoal_id = _safe_text(planner_output.get("next_subgoal_id"))
    if next_subgoal_id == SESSION_TASK_PLANNER_CLOSE:
        lines.append(
            "当前任务建议：准备收尾。这只是任务节奏信号，不构成 terminate=true 的必要条件；"
            "是否结束仍由 Judge 在终止检查开启后依据临床闭环判断。"
        )
    else:
        lines.append(
            "当前任务建议：[{}] {}。这是非强制建议，可依据患者最新状态和临床需要调整。".format(
                next_subgoal_id,
                goal_by_id.get(next_subgoal_id, next_subgoal_id),
            )
        )
    return "\n".join(lines)


def merge_progressive_d_planner_updates(
    progress_state: Dict[str, Any],
    stage_item: Mapping[str, Any],
    updates: Sequence[Mapping[str, Any]],
    meeting_id: str,
) -> List[Dict[str, str]]:
    """Apply only real Planner progress, without counting or advancing a meeting."""

    configured = stage_item.get("subgoals")
    if not isinstance(configured, list) or not configured:
        raise SubgoalProgressValidationError(
            "Progressive D Planner merge requires configured subgoals"
        )
    configured_ids = [
        _safe_text(item.get("id"))
        for item in configured
        if isinstance(item, Mapping)
    ]
    stored = progress_state.get("subgoals")
    if not isinstance(stored, dict) or set(stored) != set(configured_ids):
        raise SubgoalProgressValidationError(
            "Progressive D persisted subgoals do not match configuration"
        )
    meeting_id_text = _safe_text(meeting_id)
    if not meeting_id_text:
        raise SubgoalProgressValidationError(
            "Progressive D Planner merge requires meeting_id"
        )

    normalized_updates: List[Dict[str, str]] = []
    seen = set()
    for item in updates:
        if not isinstance(item, Mapping):
            raise SubgoalProgressValidationError(
                "Progressive D Planner update must be an object"
            )
        subgoal_id = _safe_text(item.get("subgoal_id"))
        status = _safe_text(item.get("status")).lower()
        if (
            subgoal_id not in stored
            or subgoal_id in seen
            or status not in SESSION_TASK_PLANNER_UPDATE_STATUS
        ):
            raise SubgoalProgressValidationError(
                "Progressive D Planner update is invalid"
            )
        seen.add(subgoal_id)
        normalized_updates.append(
            {"subgoal_id": subgoal_id, "status": status}
        )

    applied: List[Dict[str, str]] = []
    for item in normalized_updates:
        subgoal_id = item["subgoal_id"]
        external_status = item["status"]
        internal_status = (
            "complete" if external_status == "completed" else "partial"
        )
        current = stored[subgoal_id]
        current_status = _safe_text(current.get("status")).lower() or "none"
        if (
            SUBGOAL_PROGRESS_STATUS_RANK[internal_status]
            <= SUBGOAL_PROGRESS_STATUS_RANK.get(current_status, 0)
        ):
            continue
        current.update(
            {
                "status": internal_status,
                "blocking_factor": "",
                "last_meeting_id": meeting_id_text,
            }
        )
        current.pop("evidence_turn_id", None)
        applied.append(dict(item))

    selected = select_active_subgoal(stage_item, progress_state)
    progress_state["active_subgoal_id"] = selected["id"]
    return applied


def merge_progressive_d_judge_updates(
    progress_state: Dict[str, Any],
    stage_item: Mapping[str, Any],
    updates: Sequence[Mapping[str, Any]],
    meeting_id: str,
) -> List[Dict[str, str]]:
    """Compatibility alias for checkpoints/tests created before Planner split."""

    return merge_progressive_d_planner_updates(
        progress_state,
        stage_item,
        updates,
        meeting_id,
    )


def safe_strategy_selector_fallback(
    allowed_primary: Sequence[str],
    allowed_micro: Sequence[str],
) -> Dict[str, str]:
    primary = _first_preferred(allowed_primary, SUPPORTIVE_PRIMARY)
    micro = _first_preferred(allowed_micro, SUPPORTIVE_MICRO)
    if not primary or not micro:
        raise MinimalControllerConfigError(
            "Response Strategy Selector requires non-empty router candidates"
        )
    return {
        "primary_strategy": primary,
        "micro_skill": micro,
    }


def safe_judge_fallback(
    allowed_primary: Sequence[str] = (),
    allowed_micro: Sequence[str] = (),
) -> Dict[str, Any]:
    # Candidate arguments are accepted for source compatibility only. Strategy
    # fallback now belongs exclusively to ``safe_strategy_selector_fallback``.
    _ = allowed_primary, allowed_micro
    return {
        "turn_goal": "获得患者对当前感受或最小问题线索的进一步表达",
        "terminate": False,
    }


def normalize_minimal_session_eval(payload: Any) -> Dict[str, str]:
    raw = payload if isinstance(payload, dict) else {}
    progress = raw.get("progress")
    if not isinstance(progress, str) or progress.strip().lower() not in {
        "none",
        "partial",
        "complete",
    }:
        progress = "none"
    else:
        progress = progress.strip().lower()
    return {
        "progress": progress,
        "evidence_turn_id": _safe_text(raw.get("evidence_turn_id")),
        "blocking_factor": _safe_text(raw.get("blocking_factor")),
        "reason": _safe_text(raw.get("reason")),
    }


def validate_minimal_session_eval_completion(
    session_eval: Mapping[str, Any],
    chats: Sequence[Any],
    patient_name: str,
    risk_level: str,
    high_risk_seen: bool = False,
) -> Dict[str, Any]:
    """Validate completion evidence without deciding a legacy transition."""

    evidence_turn_id = _safe_text(session_eval.get("evidence_turn_id"))
    _, patient_turns = tagged_dialogue(chats, patient_name)
    evidence = resolve_evidence_turn(evidence_turn_id, patient_turns)
    evidence_turn_valid = bool(evidence)
    acknowledgement_only = is_acknowledgement_only(evidence)
    evidence_too_short = bool(evidence and is_evidence_too_short(evidence))
    normalized_risk = str(risk_level or "possible").strip().lower()
    validation_errors: List[str] = []
    if session_eval.get("progress") != "complete":
        validation_errors.append("session_eval_not_complete")
    else:
        if not evidence_turn_id:
            validation_errors.append("evidence_warning:evidence_turn_id_empty")
        elif not evidence_turn_valid:
            validation_errors.append("evidence_warning:evidence_turn_id_invalid")
        if acknowledgement_only:
            validation_errors.append("evidence_warning:acknowledgement_only")
        if evidence_too_short:
            validation_errors.append("evidence_warning:evidence_turn_too_short")
        if normalized_risk == "high":
            validation_errors.append("risk_high")
        elif normalized_risk != "none":
            validation_errors.append("risk_possible")
        if bool(high_risk_seen):
            validation_errors.append("risk_high_seen")
    return {
        "evidence_turn_valid": evidence_turn_valid,
        "acknowledgement_only": acknowledgement_only,
        "evidence_too_short": evidence_too_short,
        "validation_errors": validation_errors,
        "valid_for_advance": not any(
            not item.startswith("evidence_warning:") for item in validation_errors
        ),
    }


def generate_session_end(
    session_eval: Mapping[str, Any],
    chats: Sequence[Any],
    patient_name: str,
    risk_level: str,
    high_risk_seen: bool = False,
) -> Dict[str, Any]:
    validation = validate_minimal_session_eval_completion(
        session_eval=session_eval,
        chats=chats,
        patient_name=patient_name,
        risk_level=risk_level,
        high_risk_seen=high_risk_seen,
    )
    session_end = bool(validation.pop("valid_for_advance", False))
    result = dict(validation)
    result["session_end"] = session_end
    return result


def eligible_soft_step_back_targets(
    stage_map: Mapping[str, Any],
    legacy_order: Sequence[str],
    current_session: str,
    current_macro_stage: str,
    active_subgoal_id: str,
    configured_predecessors: Mapping[str, Any] | None = None,
) -> List[str]:
    """Return configured or same-stage predecessor subgoals in stable order."""

    order = _string_list(legacy_order)
    current_session_text = _safe_text(current_session)
    active_id = _safe_text(active_subgoal_id)
    if current_session_text not in order or not active_id:
        return []

    flattened: List[tuple[str, str, str]] = []
    active_position = -1
    for session_id in order:
        item = stage_map.get(session_id, {})
        if not isinstance(item, Mapping):
            continue
        macro_stage = _safe_text(item.get("macro_stage"))
        subgoals = item.get("subgoals", [])
        if not isinstance(subgoals, list):
            continue
        for subgoal in subgoals:
            if not isinstance(subgoal, Mapping):
                continue
            subgoal_id = _safe_text(subgoal.get("id"))
            if not subgoal_id:
                continue
            flattened.append((session_id, macro_stage, subgoal_id))
            if session_id == current_session_text and subgoal_id == active_id:
                active_position = len(flattened) - 1
    if active_position < 0:
        return []

    explicit_values = (
        configured_predecessors.get(active_id, [])
        if isinstance(configured_predecessors, Mapping)
        else []
    )
    explicit = set(_string_list(explicit_values))
    targets: List[str] = []
    for position, (_, macro_stage, subgoal_id) in enumerate(flattened):
        if position >= active_position:
            break
        if (
            macro_stage == _safe_text(current_macro_stage)
            or subgoal_id in explicit
        ):
            targets.append(subgoal_id)
    return targets


def apply_legacy_stage_transition_adapter(
    *,
    legacy_state: Mapping[str, Any],
    stage_map: Mapping[str, Any],
    current_macro_stage: str,
    progress_state: Dict[str, Any],
    tracker_state: Mapping[str, Any],
    evaluator_result: Mapping[str, Any],
    patient_turns: Mapping[str, str],
    meeting_id: str,
    allowed_actions: Sequence[str],
    configured_predecessors: Mapping[str, Any] | None = None,
    recommended_target_session: str = "",
    evaluator_valid: bool = True,
    controller_version: str = "",
    expected_controller_version: str = "",
) -> Dict[str, Any]:
    """Execute the minimal legacy-compatible transition projection."""

    recommended = _safe_text(
        evaluator_result.get("recommended_action")
    ).lower()
    output = {
        "recommended_action": recommended,
        "effective_action": "hold",
        "session_end": False,
        "reason": "",
    }
    reasons: List[str] = []
    order = _string_list(legacy_state.get("order", []))
    current_session = _safe_text(legacy_state.get("current_session"))
    try:
        current_index = int(legacy_state.get("current_index", 0) or 0)
    except (TypeError, ValueError):
        current_index = -1
    active_id = _safe_text(progress_state.get("active_subgoal_id"))
    configured_actions = set(_string_list(allowed_actions))
    tracker_risk = _safe_text(tracker_state.get("risk_level")).lower()
    tracker_meeting_id = _safe_text(tracker_state.get("meeting_id"))
    high_risk_seen = bool(tracker_state.get("high_risk_seen", False))
    evidence_turn_id = _safe_text(evaluator_result.get("evidence_turn_id"))
    evidence = resolve_evidence_turn(evidence_turn_id, patient_turns)
    evidence_warnings: List[str] = []
    if not evidence_turn_id:
        evidence_warnings.append("evidence_warning:evidence_turn_id_empty")
    elif not evidence:
        evidence_warnings.append("evidence_warning:evidence_turn_id_invalid")
    elif is_acknowledgement_only(evidence):
        evidence_warnings.append("evidence_warning:acknowledgement_only")
    elif is_evidence_too_short(evidence):
        evidence_warnings.append("evidence_warning:evidence_turn_too_short")
    actual_version = _safe_text(controller_version)
    expected_version = _safe_text(expected_controller_version)

    if not evaluator_valid:
        reasons.append("subgoal_evaluator_invalid")
    if (
        not actual_version
        or not expected_version
        or actual_version != expected_version
    ):
        reasons.append("controller_version_mismatch")
    if recommended not in STAGE_ADAPTER_ACTIONS:
        reasons.append("action_not_supported")
    elif recommended not in configured_actions:
        reasons.append("action_not_allowed_by_config")
    if (
        not current_session
        or current_session not in order
        or current_index < 0
        or current_index >= len(order)
        or order[current_index] != current_session
    ):
        reasons.append("legacy_state_invalid")
    stage_item = stage_map.get(current_session, {})
    if (
        not isinstance(stage_item, Mapping)
        or _safe_text(stage_item.get("macro_stage"))
        != _safe_text(current_macro_stage)
    ):
        reasons.append("macro_stage_mismatch")
    if tracker_risk == "high":
        reasons.append("risk_high")
    elif tracker_risk != "none":
        reasons.append("risk_possible")
    if high_risk_seen:
        reasons.append("risk_high_seen")
    meeting_id_text = _safe_text(meeting_id)
    if (
        not meeting_id_text
        or not tracker_meeting_id
        or meeting_id_text != tracker_meeting_id
    ):
        reasons.append("meeting_state_stale")

    if reasons or recommended == "hold":
        if recommended == "hold" and not reasons:
            reasons.append("recommended_hold")
        output["reason"] = "|".join(_dedupe(reasons + evidence_warnings))
        return output

    side_step = progress_state.get("side_step")
    if not isinstance(side_step, dict):
        side_step = copy_side_step_default()
        progress_state["side_step"] = side_step
    soft_step_back = progress_state.get("soft_step_back")
    if not isinstance(soft_step_back, dict):
        soft_step_back = copy_soft_step_back_default()
        progress_state["soft_step_back"] = soft_step_back
    side_active = bool(side_step.get("active", False))
    soft_active = bool(soft_step_back.get("active", False))
    subgoal_id = _safe_text(evaluator_result.get("subgoal_id"))
    progress = _safe_text(evaluator_result.get("progress")).lower()

    if recommended == "stay":
        output["effective_action"] = "stay"
        if side_active:
            return_to = _safe_text(side_step.get("return_to_subgoal_id"))
            if return_to not in progress_state.get("subgoals", {}):
                reasons.append("side_step_return_target_invalid")
                output["effective_action"] = "hold"
            else:
                progress_state["active_subgoal_id"] = return_to
                side_step.clear()
                side_step.update(copy_side_step_default())
                reasons.append("side_step_resolved")
        elif soft_active:
            target = _safe_text(soft_step_back.get("target_subgoal_id"))
            return_to = _safe_text(
                soft_step_back.get("return_to_subgoal_id")
            )
            if (
                subgoal_id == target
                and progress == "complete"
                and return_to in progress_state.get("subgoals", {})
            ):
                progress_state["active_subgoal_id"] = return_to
                soft_step_back.clear()
                soft_step_back.update(copy_soft_step_back_default())
                reasons.append("soft_step_back_resolved")
            else:
                reasons.append("soft_step_back_kept")
        else:
            reasons.append("stay")

    elif recommended == "advance_subgoal":
        if side_active or soft_active:
            reasons.append("temporary_work_active")
        elif subgoal_id != active_id:
            reasons.append("subgoal_not_active")
        else:
            subgoals = progress_state.get("subgoals", {})
            configured = stage_item.get("subgoals", [])
            allowed_ids = [
                _safe_text(item.get("id"))
                for item in configured
                if isinstance(item, Mapping)
            ] if isinstance(configured, list) else []
            current = (
                subgoals.get(active_id, {})
                if isinstance(subgoals, Mapping)
                else {}
            )
            if (
                progress != "complete"
                or not isinstance(current, Mapping)
                or _safe_text(current.get("status")).lower() != "complete"
            ):
                reasons.append("active_subgoal_not_complete")
            elif active_id not in allowed_ids:
                reasons.append("active_subgoal_not_configured")
            else:
                position = allowed_ids.index(active_id)
                if position + 1 >= len(allowed_ids):
                    reasons.append("no_legal_next_subgoal")
                else:
                    progress_state["active_subgoal_id"] = allowed_ids[
                        position + 1
                    ]
                    output["effective_action"] = "advance_subgoal"
                    reasons.append("advanced_to_adjacent_subgoal")

    elif recommended == "side_step":
        reason = _safe_text(
            evaluator_result.get("blocking_factor")
        ) or _safe_text(evaluator_result.get("reason"))
        if side_active or soft_active:
            reasons.append("temporary_work_conflict")
        elif not reason:
            reasons.append("side_step_reason_empty")
        else:
            side_step.update(
                {
                    "active": True,
                    "reason": reason,
                    "return_to_subgoal_id": active_id,
                    "started_meeting_id": meeting_id_text,
                }
            )
            output["effective_action"] = "side_step"
            reasons.append("side_step_activated")

    elif recommended == "soft_step_back":
        reason = _safe_text(
            evaluator_result.get("blocking_factor")
        ) or _safe_text(evaluator_result.get("reason"))
        eligible_targets = eligible_soft_step_back_targets(
            stage_map=stage_map,
            legacy_order=order,
            current_session=current_session,
            current_macro_stage=current_macro_stage,
            active_subgoal_id=active_id,
            configured_predecessors=configured_predecessors,
        )
        if side_active:
            reasons.append("side_step_active")
        elif soft_active:
            reasons.append("nested_soft_step_back_not_allowed")
        elif subgoal_id not in eligible_targets:
            reasons.append("soft_step_back_target_not_allowed")
        elif not reason:
            reasons.append("soft_step_back_reason_empty")
        else:
            soft_step_back.update(
                {
                    "active": True,
                    "target_subgoal_id": subgoal_id,
                    "return_to_subgoal_id": active_id,
                    "reason": reason,
                }
            )
            output["effective_action"] = "soft_step_back"
            reasons.append("soft_step_back_activated")

    elif recommended == "advance_stage":
        subgoal_states = progress_state.get("subgoals", {})
        all_complete = bool(subgoal_states) and all(
            isinstance(item, Mapping)
            and _safe_text(item.get("status")).lower() == "complete"
            for item in subgoal_states.values()
        )
        if not all_complete:
            reasons.append("required_subgoals_incomplete")
        if side_active:
            reasons.append("side_step_active")
        if soft_active:
            reasons.append("soft_step_back_active")

        terminal = current_index == len(order) - 1
        next_session = "" if terminal else order[current_index + 1]
        requested_target = _safe_text(recommended_target_session)
        if requested_target and requested_target != next_session:
            reasons.append("non_adjacent_legacy_target")
        if not terminal:
            next_item = stage_map.get(next_session, {})
            next_macro_stage = (
                _safe_text(next_item.get("macro_stage"))
                if isinstance(next_item, Mapping)
                else ""
            )
            macro_order: List[str] = []
            for session_id in order:
                item = stage_map.get(session_id, {})
                macro = (
                    _safe_text(item.get("macro_stage"))
                    if isinstance(item, Mapping)
                    else ""
                )
                if macro and macro not in macro_order:
                    macro_order.append(macro)
            legal_macro = next_macro_stage == _safe_text(current_macro_stage)
            if (
                not legal_macro
                and current_macro_stage in macro_order
                and macro_order.index(current_macro_stage) + 1 < len(macro_order)
            ):
                legal_macro = (
                    next_macro_stage
                    == macro_order[macro_order.index(current_macro_stage) + 1]
                )
            if not next_session or not legal_macro:
                reasons.append("illegal_next_macro_stage")

        if not reasons:
            output["effective_action"] = "advance_stage"
            output["session_end"] = True
            reasons.append(
                "terminal_legacy_prompt_completed"
                if terminal
                else "adjacent_legacy_prompt_approved"
            )

    if output["effective_action"] == "hold" and not reasons:
        reasons.append("transition_guard_failed")
    output["reason"] = "|".join(_dedupe(reasons + evidence_warnings))
    return output


def split_utterances(chats: Sequence[Any], patient_name: str) -> tuple[List[str], List[str]]:
    patient: List[str] = []
    doctor: List[str] = []
    for item in chats or []:
        if not isinstance(item, (tuple, list)) or len(item) < 2:
            continue
        target = patient if str(item[0] or "") == str(patient_name or "") else doctor
        target.append(str(item[1] or ""))
    return patient, doctor


def tagged_dialogue(
    chats: Sequence[Any],
    patient_name: str,
) -> tuple[str, Dict[str, str]]:
    """Render meeting-local turn IDs and return the allowed patient-turn map."""

    patient_no = 0
    doctor_no = 0
    lines: List[str] = []
    patient_turns: Dict[str, str] = {}
    for item in chats or []:
        if not isinstance(item, (tuple, list)) or len(item) < 2:
            continue
        speaker = str(item[0] or "")
        utterance = str(item[1] or "")
        if speaker == str(patient_name or ""):
            patient_no += 1
            turn_id = f"P{patient_no}"
            patient_turns[turn_id] = utterance
        else:
            doctor_no += 1
            turn_id = f"D{doctor_no}"
        lines.append(f"[{turn_id}] {speaker}: {utterance}")
    return "\n".join(lines), patient_turns


def resolve_evidence_turn(
    evidence_turn_id: str,
    patient_turns: Mapping[str, str],
) -> str:
    """Resolve only an exact current-meeting patient turn identifier."""

    turn_id = _safe_text(evidence_turn_id)
    if not re.fullmatch(r"P[1-9][0-9]*", turn_id):
        return ""
    utterance = patient_turns.get(turn_id)
    return str(utterance or "") if isinstance(utterance, str) else ""


def quote_in_utterances(quote: str, utterances: Sequence[str]) -> bool:
    needle = normalize_for_match(quote)
    return bool(needle and any(needle in normalize_for_match(text) for text in utterances))


def normalize_for_match(text: Any) -> str:
    normalized = unicodedata.normalize("NFKC", str(text or ""))
    return "".join(normalized.split())


def is_acknowledgement_only(text: str) -> bool:
    normalized = normalize_evidence_content(text)
    return bool(
        normalized
        and (
            normalized in ACKNOWLEDGEMENT_ONLY
            or _ACKNOWLEDGEMENT_RE.fullmatch(normalized)
        )
    )


def is_evidence_too_short(text: str) -> bool:
    return len(normalize_evidence_content(text)) < MIN_EVIDENCE_CONTENT_CHARS


def normalize_evidence_content(text: Any) -> str:
    normalized = unicodedata.normalize("NFKC", str(text or ""))
    return "".join(
        char
        for char in normalized
        if not char.isspace()
        and not unicodedata.category(char).startswith(("P", "S", "Z"))
    )


def recent_dialogue(chats: Sequence[Any], max_utterances: int) -> str:
    limit = max(1, int(max_utterances or 1))
    lines = []
    for item in list(chats or [])[-limit:]:
        if isinstance(item, (tuple, list)) and len(item) >= 2:
            lines.append(f"{item[0]}: {item[1]}")
    return "\n".join(lines)


def compact_patient_state_text(patient_state: Mapping[str, Any]) -> str:
    return (
        "情绪负荷={}; 参与度={}; 抵触={}; 问题清晰度={}; 风险={}; 患者证据={}"
    ).format(
        patient_state.get("emotion_level", "medium"),
        patient_state.get("engagement", "medium"),
        "是" if patient_state.get("resistance", False) else "否",
        patient_state.get("problem_clarity", "vague"),
        patient_state.get("risk_level", "possible"),
        patient_state.get("evidence", "") or "无",
    )


def compact_progressive_d_patient_state_text(
    patient_state: Mapping[str, Any],
) -> str:
    """Render the evidence-free five-field control state."""

    return (
        "情绪负荷={}; 参与度={}; 抵触={}; 问题清晰度={}; 风险={}"
    ).format(
        patient_state.get("emotion_level", "medium"),
        patient_state.get("engagement", "medium"),
        "是" if patient_state.get("resistance", False) else "否",
        patient_state.get("problem_clarity", "vague"),
        patient_state.get("risk_level", "none"),
    )


def compact_prior_progress_text(progress_state: Mapping[str, Any]) -> str:
    return (
        "上次会谈材料评估={}；上次阻塞={}；材料评估说明={}。"
        "该评估仅用于累计证据判断，不代表固定会谈提纲已经推进。"
    ).format(
        _progress_label(progress_state.get("last_progress", "none")),
        progress_state.get("last_blocking_factor", "") or "无",
        progress_state.get("last_reason", "") or "无",
    )


def compact_route_text(
    route: Mapping[str, Any],
    glossary: Mapping[str, Any] | None = None,
) -> str:
    glossary_map = glossary if isinstance(glossary, Mapping) else {}
    return "主要策略候选={}; 微技能候选={}".format(
        "；".join(
            _format_candidate_term(
                item,
                glossary_map.get("primary_strategies", {}),
            )
            for item in _string_list(
                route.get("allowed_primary_strategies")
            )
        ),
        "；".join(
            _format_candidate_term(
                item,
                glossary_map.get("micro_skills", {}),
            )
            for item in _string_list(route.get("allowed_micro_skills"))
        ),
    )


def _safe_text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _progress_label(value: Any) -> str:
    return {
        "none": "无进展",
        "partial": "部分完成",
        "complete": "完成",
    }.get(str(value or "").strip().lower(), "未知")


def _subgoal_goal(
    configured_by_id: Mapping[str, Any],
    subgoal_id: str,
) -> str:
    item = configured_by_id.get(_safe_text(subgoal_id), {})
    if isinstance(item, Mapping):
        goal = _safe_text(item.get("goal"))
        if goal:
            return goal
    return "无"


def _format_candidate_term(term_id: str, terms: Any) -> str:
    item = terms.get(term_id, {}) if isinstance(terms, Mapping) else {}
    if not isinstance(item, Mapping):
        return term_id
    label = _safe_text(item.get("label"))
    instruction = _safe_text(item.get("instruction"))
    if not label or not instruction:
        return term_id
    return f"{term_id}（{label}：{instruction}）"


def _string_list(value: Any) -> List[str]:
    if not isinstance(value, (list, tuple)):
        return []
    return [str(item).strip() for item in value if isinstance(item, str) and item.strip()]


def _dedupe(values: Sequence[str]) -> List[str]:
    result: List[str] = []
    seen = set()
    for value in values:
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _first_preferred(values: Sequence[str], preferred: Sequence[str]) -> str:
    for candidate in preferred:
        if candidate in values:
            return candidate
    return str(values[0]) if values else ""
