"""主诉链管理器。"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple


logger = logging.getLogger(__name__)


@dataclass
class ComplaintStage:
    """单个主诉链节点。"""

    id: str
    label: str
    summary: str = ""
    core_belief: str = ""
    narrative_focus: List[str] = field(default_factory=list)
    speaking_style: Dict[str, Any] = field(default_factory=dict)
    emotion_vector: Dict[str, float] = field(default_factory=dict)
    bias_profile: Dict[str, Any] = field(default_factory=dict)
    advance_signals: List[str] = field(default_factory=list)
    hold_signals: List[str] = field(default_factory=list)
    relation_modifiers: Dict[str, Any] = field(default_factory=dict)
    next_candidates: List[str] = field(default_factory=list)
    is_terminal_stage: bool = False
    source: str = "config"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ComplaintChainManager:
    """仅围绕主诉链推进的人设动态管理器。"""

    STOPWORDS = {
        "自己", "觉得", "感觉", "因为", "然后", "已经", "还是", "不是", "就是", "一个", "一种",
        "有点", "这样", "那种", "事情", "问题", "别人", "什么", "没有", "不会", "如果", "真的",
        "可能", "一直", "最近", "现在", "让我", "我们", "他们", "只是", "还有", "其实", "一下",
    }

    DEFAULT_STAGE: Dict[str, Any] = {
        "id": "unconfigured_stage",
        "label": "尚未配置主诉链",
        "summary": "当前角色还没有被配置可推进的主诉链节点。",
        "core_belief": "我还没有准备好描述自己的痛苦。",
        "narrative_focus": ["沉默", "空白", "等待配置"],
        "speaking_style": {
            "tempo": "slow",
            "disclosure": "guarded",
            "tone": "flat",
            "repair_pattern": "回答偏短，容易停住",
        },
        "emotion_vector": {
            "valence": 0.25,
            "arousal": 0.35,
            "defensiveness": 0.55,
            "shame": 0.40,
            "hopelessness": 0.35,
            "trust": 0.20,
        },
        "bias_profile": {
            "dominant": ["mental_filter"],
            "secondary": ["emotional_reasoning"],
            "max_active": 1,
        },
        "advance_signals": [],
        "hold_signals": ["沉默", "不知道说什么"],
        "relation_modifiers": {},
        "next_candidates": [],
        "is_terminal_stage": False,
        "source": "fallback",
    }

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        now_provider: Optional[Callable[[], datetime]] = None,
    ):
        config = config if isinstance(config, dict) else {}
        chain_config = config.get("complaint_chain", config)
        if not isinstance(chain_config, dict):
            chain_config = {}

        self._raw_chain_config = copy.deepcopy(chain_config)
        self.planner: Dict[str, Any] = copy.deepcopy(chain_config.get("planner", {}))
        self.window_size = self._bounded_int(self.planner.get("window_size"), 3, 1, 8)
        self.min_match_confidence = self._bounded_float(
            self.planner.get("min_match_confidence"), 0.60, 0.0, 1.0
        )
        self.allow_replan = self._coerce_bool(self.planner.get("allow_replan", True))
        self.allow_jump = self._coerce_bool(self.planner.get("allow_jump", True))
        self.llm_enabled = self._coerce_bool(self.planner.get("llm_enabled", True))
        self.allow_runtime_stage_creation = self._coerce_bool(
            self.planner.get("allow_runtime_stage_creation", True)
        )
        self.max_dialog_history = self._bounded_int(
            self.planner.get("max_dialog_history"), 12, 4, 50
        )

        self._now_provider: Callable[[], datetime] = (
            now_provider if callable(now_provider) else datetime.now
        )

        self.stage_catalog: Dict[str, Dict[str, Any]] = {}
        raw_stages = chain_config.get("stages", [])
        if isinstance(raw_stages, list):
            for item in raw_stages:
                stage = self._sanitize_stage(item, source=str(item.get("source", "config")) if isinstance(item, dict) else "config")
                self.stage_catalog[stage["id"]] = stage

        if not self.stage_catalog:
            default_stage = self._sanitize_stage(self.DEFAULT_STAGE, source="fallback")
            self.stage_catalog[default_stage["id"]] = default_stage

        initial_stage_id = str(
            chain_config.get("initial_stage_id")
            or next(iter(self.stage_catalog.keys()))
        ).strip()
        if initial_stage_id not in self.stage_catalog:
            initial_stage_id = next(iter(self.stage_catalog.keys()))
        self.initial_stage_id = initial_stage_id

        self.planned_chain: List[str] = []
        self.stage_index = 0
        self.stage_history: List[Dict[str, Any]] = []
        self.dialogue_history: List[Dict[str, Any]] = []
        self.pending_stage_specs: Dict[str, Dict[str, Any]] = {}
        self.last_session_context: Dict[str, Any] = {}
        self.last_evaluation: Dict[str, Any] = {}
        self.stage_start_time = self._now()

        self.reset()

    def _now(self) -> datetime:
        try:
            value = self._now_provider()
        except Exception:
            value = datetime.now()
        return _coerce_datetime(value)

    def set_now_provider(self, now_provider: Optional[Callable[[], datetime]]) -> None:
        if callable(now_provider):
            self._now_provider = now_provider

    def reset(self) -> None:
        self.planned_chain = self._build_default_chain(self.initial_stage_id)
        self.stage_index = 0
        self.stage_history = []
        self.dialogue_history = []
        self.pending_stage_specs = {}
        self.last_session_context = {}
        self.last_evaluation = {}
        self.stage_start_time = self._now()

    def get_current_stage(self) -> Dict[str, Any]:
        if not self.planned_chain:
            self.reset()
        idx = min(max(0, int(self.stage_index)), len(self.planned_chain) - 1)
        stage_id = self.planned_chain[idx]
        return copy.deepcopy(self.stage_catalog.get(stage_id, self._sanitize_stage(self.DEFAULT_STAGE, source="fallback")))

    def get_current_stage_id(self) -> str:
        return str(self.get_current_stage().get("id", ""))

    def get_current_chain_window(self, count: Optional[int] = None) -> List[Dict[str, Any]]:
        if not self.planned_chain:
            return []
        width = self._bounded_int(count, self.window_size + 1, 1, 20)
        start = min(max(0, int(self.stage_index)), len(self.planned_chain) - 1)
        ids = self.planned_chain[start : start + width]
        return [copy.deepcopy(self.stage_catalog.get(stage_id, {})) for stage_id in ids]

    def get_chain_snapshot(self) -> Dict[str, Any]:
        current_stage = self.get_current_stage()
        return {
            "mode": "complaint_chain",
            "current_stage_id": str(current_stage.get("id", "")),
            "current_stage_label": str(current_stage.get("label", "")),
            "current_stage": current_stage,
            "planned_chain": [str(item) for item in self.planned_chain],
            "stage_index": int(self.stage_index),
            "current_chain_window": self.get_current_chain_window(self.window_size + 1),
            "current_stage_branch_options": self._get_branch_options(current_stage),
            "pending_stage_specs": copy.deepcopy(self.pending_stage_specs),
            "stage_start_time": self.stage_start_time.isoformat(),
            "stage_history": copy.deepcopy(self.stage_history),
            "dialogue_history": copy.deepcopy(self.dialogue_history[-self.max_dialog_history :]),
            "last_session_context": copy.deepcopy(self.last_session_context),
            "last_evaluation": copy.deepcopy(self.last_evaluation),
        }

    def get_roadmap_snapshot(self) -> Dict[str, Any]:
        return self.get_chain_snapshot()

    def get_state_history(self) -> List[Dict[str, Any]]:
        return copy.deepcopy(self.stage_history)

    def get_state_duration(self) -> float:
        return (self._now() - self.stage_start_time).total_seconds() / 60.0

    def _normalize_planned_chain(self, value: Any) -> List[str]:
        items = self._to_list(value)
        normalized: List[str] = []
        last_id = ""
        for item in items:
            stage_id = ""
            if isinstance(item, dict):
                candidate = str(item.get("id", "") or "").strip()
                if candidate in self.stage_catalog:
                    stage_id = candidate
            else:
                candidate = str(item or "").strip()
                if candidate in self.stage_catalog:
                    stage_id = candidate
            if not stage_id or stage_id == last_id:
                continue
            normalized.append(stage_id)
            last_id = stage_id
        return normalized

    def _normalize_path_tokens(self, value: Any, allow_unknown: bool = True) -> List[str]:
        items = self._to_list(value)
        normalized: List[str] = []
        last_token = ""
        for item in items:
            token = ""
            if isinstance(item, dict):
                token = str(item.get("id", "") or "").strip()
            else:
                token = str(item or "").strip()
            if not token:
                continue
            if not allow_unknown and token not in self.stage_catalog and token not in self.pending_stage_specs:
                continue
            if token == last_token:
                continue
            normalized.append(token)
            last_token = token
        return normalized

    def _normalize_candidate_ids(self, value: Any, allow_pending: bool = True) -> List[str]:
        normalized: List[str] = []
        seen = set()
        for token in self._normalize_path_tokens(value, allow_unknown=True):
            if token in self.stage_catalog or (allow_pending and token in self.pending_stage_specs):
                if token in seen:
                    continue
                seen.add(token)
                normalized.append(token)
        return normalized

    def _merge_candidate_ids(self, primary: Any, secondary: Any) -> List[str]:
        merged: List[str] = []
        seen = set()
        for token in self._normalize_path_tokens(primary, allow_unknown=True) + self._normalize_path_tokens(secondary, allow_unknown=True):
            if token in seen:
                continue
            seen.add(token)
            merged.append(token)
        return merged

    def _attach_candidate(self, predecessor_id: str, candidate_id: str, prefer_front: bool = False) -> None:
        predecessor_key = str(predecessor_id or "").strip()
        candidate_key = str(candidate_id or "").strip()
        if not predecessor_key or not candidate_key:
            return
        stage = self.stage_catalog.get(predecessor_key)
        if not isinstance(stage, dict):
            return
        current_candidates = self._normalize_candidate_ids(stage.get("next_candidates", []), allow_pending=True)
        if prefer_front:
            merged = [candidate_key] + [item for item in current_candidates if item != candidate_key]
        else:
            merged = current_candidates + [candidate_key]
        stage["next_candidates"] = self._normalize_candidate_ids(merged, allow_pending=True)

    def _build_branch_option(self, stage_id: str) -> Dict[str, Any]:
        candidate_id = str(stage_id or "").strip()
        if candidate_id in self.stage_catalog:
            stage = copy.deepcopy(self.stage_catalog.get(candidate_id, {}))
            stage["is_pending"] = False
            return stage
        spec = self.pending_stage_specs.get(candidate_id, {}) if isinstance(self.pending_stage_specs.get(candidate_id, {}), dict) else {}
        return {
            "id": candidate_id,
            "label": str(spec.get("label", candidate_id or "待生成节点") or candidate_id or "待生成节点"),
            "summary": str(spec.get("new_stage_brief", "待生成的新主诉节点") or "待生成的新主诉节点"),
            "next_candidates": self._normalize_path_tokens(spec.get("remaining_tail", []), allow_unknown=True),
            "source": "pending",
            "is_pending": True,
        }

    def _get_branch_options(self, current_stage: Dict[str, Any]) -> List[Dict[str, Any]]:
        current_stage = current_stage if isinstance(current_stage, dict) else {}
        branch_ids = self._normalize_candidate_ids(current_stage.get("next_candidates", []), allow_pending=True)
        return [self._build_branch_option(stage_id) for stage_id in branch_ids]

    def _get_recent_history_summary(self, limit: int = 4) -> List[Dict[str, Any]]:
        rows = self.stage_history[-max(0, int(limit)) :]
        summary: List[Dict[str, Any]] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            summary.append(
                {
                    "from_stage_id": str(row.get("from_stage_id", "") or ""),
                    "to_stage_id": str(row.get("to_stage_id", "") or ""),
                    "action": str(row.get("action", "") or ""),
                    "match_reason": str(row.get("match_reason", "") or "")[:120],
                }
            )
        return summary

    def _register_pending_path(self, requested_chain: List[str], new_stage_brief: str) -> List[str]:
        normalized = self._normalize_path_tokens(requested_chain, allow_unknown=True)
        if len(normalized) <= 1:
            return normalized
        for idx in range(1, len(normalized)):
            predecessor_id = str(normalized[idx - 1] or "").strip()
            stage_id = str(normalized[idx] or "").strip()
            if not predecessor_id or not stage_id:
                continue
            tail = self._normalize_path_tokens(normalized[idx + 1 :], allow_unknown=True)
            if stage_id in self.stage_catalog:
                self._attach_candidate(predecessor_id, stage_id, prefer_front=True)
                continue
            existing = self.pending_stage_specs.get(stage_id, {}) if isinstance(self.pending_stage_specs.get(stage_id, {}), dict) else {}
            updated = copy.deepcopy(existing)
            updated["id"] = stage_id
            updated["label"] = str(updated.get("label", stage_id) or stage_id)
            updated["predecessor_id"] = predecessor_id
            updated["new_stage_brief"] = str(new_stage_brief or updated.get("new_stage_brief", "") or "")[:180]
            updated["remaining_tail"] = tail
            self.pending_stage_specs[stage_id] = updated
            self._attach_candidate(predecessor_id, stage_id, prefer_front=True)
        return normalized

    def _would_continue_short_cycle(self, current_stage_id: str, target_stage_id: str) -> bool:
        current_id = str(current_stage_id or "").strip()
        target_id = str(target_stage_id or "").strip()
        if not current_id or not target_id or current_id == target_id or len(self.stage_history) < 2:
            return False
        last_to = str(self.stage_history[-1].get("to_stage_id", "") or "") if isinstance(self.stage_history[-1], dict) else ""
        prev_to = str(self.stage_history[-2].get("to_stage_id", "") or "") if isinstance(self.stage_history[-2], dict) else ""
        return bool(last_to == current_id and prev_to == target_id)

    def _resolve_requested_chain(
        self,
        current_stage: Dict[str, Any],
        requested_chain: List[str],
        needs_new_stage: bool,
        new_stage_brief: str,
        completion_func: Optional[Callable[[str], str]],
        session_context: Dict[str, Any],
        conversation_content: str,
        llm_cfg: Optional[Dict[str, Any]],
    ) -> Tuple[List[str], Optional[Dict[str, Any]]]:
        current_stage = current_stage if isinstance(current_stage, dict) else {}
        current_id = str(current_stage.get("id", "") or "").strip()
        if not current_id:
            return [], None
        existing_pending_ids = set(self.pending_stage_specs.keys())
        requested = self._normalize_next_chain(requested_chain, current_id)
        requested = self._register_pending_path(requested, new_stage_brief)
        resolved: List[str] = [current_id]
        generated_stage: Optional[Dict[str, Any]] = None
        can_generate = bool(callable(completion_func) and self.llm_enabled and self.allow_runtime_stage_creation)
        tail_source = requested[1:] if len(requested) > 1 else self._normalize_path_tokens(current_stage.get("next_candidates", []), allow_unknown=True)
        tail_items = list(tail_source)

        for idx, token in enumerate(tail_items):
            stage_id = str(token or "").strip()
            if not stage_id:
                continue
            if stage_id in self.stage_catalog:
                resolved.append(stage_id)
                continue
            spec = self.pending_stage_specs.get(stage_id, {}) if isinstance(self.pending_stage_specs.get(stage_id, {}), dict) else {}
            can_materialize_pending = bool(
                can_generate
                and stage_id in self.pending_stage_specs
                and (needs_new_stage or stage_id in existing_pending_ids)
            )
            if not can_materialize_pending or len(resolved) > 1:
                break
            remaining_tail = self._normalize_path_tokens(spec.get("remaining_tail", tail_items[idx + 1 :]), allow_unknown=True)
            brief = str(spec.get("new_stage_brief", new_stage_brief) or new_stage_brief or "").strip()
            runtime_stage = self._generate_runtime_stage(
                completion_func=completion_func,
                session_context=session_context,
                conversation_content=conversation_content,
                llm_cfg=llm_cfg,
                current_stage=current_stage,
                new_stage_brief=brief,
                requested_stage_id=stage_id,
                requested_tail=remaining_tail,
            )
            if not isinstance(runtime_stage, dict):
                break
            generated_stage = runtime_stage
            generated_id = str(runtime_stage.get("id", "") or "").strip()
            if not generated_id:
                break
            resolved.append(generated_id)
            for tail_token in remaining_tail:
                if tail_token in self.stage_catalog:
                    resolved.append(str(tail_token))
                    continue
                break
            break

        if len(resolved) <= 1:
            fallback = self._preview_future_chain(current_stage)
            fallback = self._normalize_next_chain(fallback, current_id)
            if len(fallback) > 1:
                resolved = fallback
        return self._normalize_next_chain(resolved, current_id), generated_stage

    def get_state_characteristics(self) -> Dict[str, float]:
        stage = self.get_current_stage()
        emotion = stage.get("emotion_vector", {}) if isinstance(stage.get("emotion_vector", {}), dict) else {}
        disclosure = self._disclosure_to_score(stage.get("speaking_style", {}).get("disclosure", "guarded"))
        return {
            "valence": self._bounded_float(emotion.get("valence"), 0.25, 0.0, 1.0),
            "arousal": self._bounded_float(emotion.get("arousal"), 0.35, 0.0, 1.0),
            "defensiveness": self._bounded_float(emotion.get("defensiveness"), 0.55, 0.0, 1.0),
            "shame": self._bounded_float(emotion.get("shame"), 0.40, 0.0, 1.0),
            "hopelessness": self._bounded_float(emotion.get("hopelessness"), 0.35, 0.0, 1.0),
            "trust": self._bounded_float(emotion.get("trust"), 0.20, 0.0, 1.0),
            "disclosure": disclosure,
        }

    def get_symptom_intensity(self, symptom: str, time_of_day: Optional[str] = None) -> float:
        del time_of_day
        return self._bounded_float(self.get_state_characteristics().get(symptom), 0.0, 0.0, 1.0)

    def evaluate_turn(
        self,
        session_context: Dict[str, Any],
        conversation_content: str,
        completion_func: Optional[Callable[[str], str]] = None,
        llm_cfg: Optional[Dict[str, Any]] = None,
        llm_signal: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        session_context = session_context if isinstance(session_context, dict) else {}
        conversation = str(conversation_content or "").strip()
        current_stage = self.get_current_stage()
        current_id = str(current_stage.get("id", "") or "").strip()

        normalized_signal = self._normalize_llm_signal(llm_signal)
        if callable(completion_func) and self.llm_enabled:
            inferred_signal = self._infer_chain_signal(
                completion_func=completion_func,
                session_context=session_context,
                conversation_content=conversation,
                llm_cfg=llm_cfg,
            )
            if inferred_signal:
                normalized_signal = inferred_signal

        heur_matched, heur_confidence, heur_reason = self._heuristic_match(
            current_stage=current_stage,
            session_context=session_context,
            conversation_content=conversation,
        )

        matched = heur_matched
        match_confidence = heur_confidence
        match_reason = heur_reason
        action = "hold"
        requested_chain: List[str] = self._preview_future_chain(current_stage)
        needs_new_stage = False
        new_stage_brief = ""

        if normalized_signal:
            matched = bool(normalized_signal.get("matched", matched))
            match_confidence = self._bounded_float(
                normalized_signal.get("match_confidence", match_confidence),
                match_confidence,
                0.0,
                1.0,
            )
            match_reason = str(normalized_signal.get("match_reason", match_reason) or match_reason)
            action = str(normalized_signal.get("action", action) or action).strip().lower() or "hold"
            candidate_chain = normalized_signal.get("next_chain", [])
            if isinstance(candidate_chain, list) and candidate_chain:
                requested_chain = self._normalize_next_chain(candidate_chain, current_id)
            needs_new_stage = bool(normalized_signal.get("needs_new_stage", False))
            new_stage_brief = str(normalized_signal.get("new_stage_brief", "") or "").strip()

        if action not in {"hold", "advance", "replan", "jump"}:
            action = "hold"

        if not normalized_signal:
            action = self._decide_action(
                current_stage=current_stage,
                matched=matched,
                match_confidence=match_confidence,
                session_context=session_context,
                conversation_content=conversation,
            )

        if action == "replan" and not self.allow_replan:
            action = "hold"
        if action == "jump" and not self.allow_jump:
            action = "hold"

        next_chain_ids, generated_stage = self._resolve_requested_chain(
            current_stage=current_stage,
            requested_chain=requested_chain,
            needs_new_stage=needs_new_stage,
            new_stage_brief=new_stage_brief,
            completion_func=completion_func,
            session_context=session_context,
            conversation_content=conversation,
            llm_cfg=llm_cfg,
        )

        next_stage: Optional[Dict[str, Any]] = None
        if action in {"advance", "jump"}:
            if len(next_chain_ids) > 1 and str(next_chain_ids[1] or "").strip() in self.stage_catalog:
                next_stage = copy.deepcopy(self.stage_catalog.get(str(next_chain_ids[1] or "").strip(), {}))
            else:
                action = "hold"

        evaluation = {
            "action": action,
            "matched": bool(matched),
            "match_confidence": round(float(match_confidence), 4),
            "match_reason": self._clip_text(match_reason, limit=180),
            "current_stage": copy.deepcopy(current_stage),
            "next_stage": copy.deepcopy(next_stage) if isinstance(next_stage, dict) else None,
            "generated_stage": copy.deepcopy(generated_stage) if isinstance(generated_stage, dict) else None,
            "next_chain": [str(item) for item in next_chain_ids],
            "session_context": copy.deepcopy(session_context),
            "conversation_excerpt": self._clip_text(conversation, limit=220),
            "needs_new_stage": bool(needs_new_stage),
            "new_stage_brief": self._clip_text(new_stage_brief, limit=180),
        }
        return evaluation

    def commit_turn(self, evaluation: Dict[str, Any]) -> Dict[str, Any]:
        evaluation = evaluation if isinstance(evaluation, dict) else {}
        current_before = self.get_current_stage()
        current_before_id = str(current_before.get("id", "") or "").strip()
        action = str(evaluation.get("action", "hold") or "hold").strip().lower()
        matched = bool(evaluation.get("matched", False))
        match_confidence = self._bounded_float(evaluation.get("match_confidence"), 0.0, 0.0, 1.0)
        match_reason = str(evaluation.get("match_reason", "") or "")[:180]
        next_chain = self._normalize_next_chain(evaluation.get("next_chain", []), current_before_id)
        session_context = (
            evaluation.get("session_context", {}) if isinstance(evaluation.get("session_context", {}), dict) else {}
        )
        conversation_excerpt = str(evaluation.get("conversation_excerpt", "") or "")

        if not next_chain:
            next_chain = self._preview_future_chain(current_before)

        pointer_before = int(self.stage_index)
        duration_minutes = self.get_state_duration()

        if action in {"hold", "replan"}:
            self._replace_future_chain(next_chain)
        elif action == "advance":
            target_stage_id = str(next_chain[1] or "").strip() if len(next_chain) > 1 else ""
            if not target_stage_id or target_stage_id not in self.stage_catalog or self._would_continue_short_cycle(current_before_id, target_stage_id):
                action = "hold"
                self._replace_future_chain([current_before_id] + [item for item in next_chain[1:] if str(item or "").strip() in self.stage_catalog])
            else:
                self._replace_future_chain(next_chain)
                if self.stage_index < len(self.planned_chain) - 1:
                    self.stage_index += 1
                    self.stage_start_time = self._now()
                else:
                    action = "hold"
        elif action == "jump":
            target_stage = evaluation.get("next_stage") if isinstance(evaluation.get("next_stage"), dict) else None
            target_stage_id = ""
            if isinstance(target_stage, dict):
                target_stage_id = str(target_stage.get("id", "") or "").strip()
            if not target_stage_id and len(next_chain) > 1:
                target_stage_id = str(next_chain[1] or "").strip()
            if target_stage_id and target_stage_id in self.stage_catalog and not self._would_continue_short_cycle(current_before_id, target_stage_id):
                self.planned_chain = [target_stage_id] + self._build_future_chain_from_stage(target_stage_id, self.window_size)
                self.stage_index = 0
                self.stage_start_time = self._now()
                self.stage_index = min(self.stage_index, max(0, len(self.planned_chain) - 1))
            else:
                logger.info(
                    "[DEPRESSION_DYNAMIC_STAGE_REJECT] current_stage=%s needs_new_stage=%s new_stage=%s reason=%s",
                    str(current_before.get("id", "") or ""),
                    bool(evaluation.get("needs_new_stage", False)),
                    str(target_stage_id or ""),
                    "jump_target_missing_or_cycle",
                )
                action = "hold"
                next_chain = self._preview_future_chain(current_before)
                self._replace_future_chain(next_chain)

        self._ensure_future_window()
        self.stage_index = min(self.stage_index, max(0, len(self.planned_chain) - 1))
        current_after = self.get_current_stage()
        if action == "advance" and str(current_after.get("id", "") or "") == current_before_id:
            action = "hold"

        self.last_session_context = copy.deepcopy(session_context)
        self.last_evaluation = {
            "action": action,
            "matched": matched,
            "match_confidence": round(float(match_confidence), 4),
            "match_reason": match_reason,
        }

        self._track_dialogue(
            current_stage=current_before,
            evaluation=self.last_evaluation,
            conversation_excerpt=conversation_excerpt,
            session_context=session_context,
        )
        self._record_history(
            action=action,
            from_stage=current_before,
            to_stage=current_after,
            matched=matched,
            match_confidence=match_confidence,
            match_reason=match_reason,
            pointer_before=pointer_before,
            pointer_after=int(self.stage_index),
            duration_minutes=duration_minutes,
        )
        return self.get_chain_snapshot()

    def update_state(
        self,
        context_analysis: Dict,
        llm_transition_signal: Optional[Dict[str, Any]] = None,
        conversation_content: str = "",
        roadmap_context: Optional[Dict[str, Any]] = None,
        roadmap_completion_func: Optional[Callable[[str], str]] = None,
        roadmap_llm_cfg: Optional[Dict[str, Any]] = None,
    ) -> bool:
        session_context = context_analysis if isinstance(context_analysis, dict) else {}
        if isinstance(roadmap_context, dict) and roadmap_context:
            merged = copy.deepcopy(session_context)
            merged.setdefault("runtime", {})
            merged["runtime"]["roadmap_context"] = copy.deepcopy(roadmap_context)
            session_context = merged
        evaluation = self.evaluate_turn(
            session_context=session_context,
            conversation_content=conversation_content,
            completion_func=roadmap_completion_func,
            llm_cfg=roadmap_llm_cfg,
            llm_signal=llm_transition_signal,
        )
        self.commit_turn(evaluation)
        return str(evaluation.get("action", "hold")) in {"advance", "jump"}

    def force_stage(self, stage_id: str, reason: str = "manual") -> None:
        stage_key = str(stage_id or "").strip()
        if not stage_key:
            return
        if stage_key not in self.stage_catalog:
            self.stage_catalog[stage_key] = self._sanitize_stage(
                {"id": stage_key, "label": stage_key, "summary": stage_key},
                source="manual",
            )
        previous_stage = self.get_current_stage()
        duration_minutes = self.get_state_duration()
        self.planned_chain = [stage_key] + self._build_future_chain_from_stage(stage_key, self.window_size)
        self.stage_index = 0
        self.stage_start_time = self._now()
        current_stage = self.get_current_stage()
        self._record_history(
            action="jump",
            from_stage=previous_stage,
            to_stage=current_stage,
            matched=True,
            match_confidence=1.0,
            match_reason=str(reason or "manual")[:180],
            pointer_before=0,
            pointer_after=0,
            duration_minutes=duration_minutes,
        )

    def force_transition(self, new_state: Any, reason: str = "manual") -> None:
        if isinstance(new_state, dict):
            stage = self._sanitize_stage(new_state, source="manual")
            self.stage_catalog[stage["id"]] = stage
            self.force_stage(stage["id"], reason=reason)
            return
        self.force_stage(str(new_state or "").strip(), reason=reason)

    def to_dict(self) -> Dict[str, Any]:
        history_rows: List[Dict[str, Any]] = []
        for row in self.stage_history:
            item = dict(row)
            ts = item.get("timestamp")
            if isinstance(ts, datetime):
                item["timestamp"] = ts.isoformat()
            history_rows.append(item)
        return {
            "mode": "complaint_chain",
            "config": {
                "planner": copy.deepcopy(self.planner),
                "initial_stage_id": self.initial_stage_id,
                "stages": [copy.deepcopy(item) for item in self.stage_catalog.values()],
            },
            "planned_chain": [str(item) for item in self.planned_chain],
            "stage_index": int(self.stage_index),
            "pending_stage_specs": copy.deepcopy(self.pending_stage_specs),
            "stage_history": history_rows,
            "dialogue_history": copy.deepcopy(self.dialogue_history),
            "last_session_context": copy.deepcopy(self.last_session_context),
            "last_evaluation": copy.deepcopy(self.last_evaluation),
            "stage_start_time": self.stage_start_time.isoformat(),
        }

    @classmethod
    def from_dict(
        cls,
        payload: Dict[str, Any],
        now_provider: Optional[Callable[[], datetime]] = None,
    ) -> "ComplaintChainManager":
        payload = payload if isinstance(payload, dict) else {}
        cfg = payload.get("config", {}) if isinstance(payload.get("config", {}), dict) else {}
        manager = cls(config={"complaint_chain": cfg}, now_provider=now_provider)
        pending_specs = payload.get("pending_stage_specs", {}) if isinstance(payload.get("pending_stage_specs", {}), dict) else {}
        manager.pending_stage_specs = {
            str(key or "").strip(): copy.deepcopy(value)
            for key, value in pending_specs.items()
            if str(key or "").strip() and isinstance(value, dict)
        }
        planned_chain = payload.get("planned_chain", [])
        if isinstance(planned_chain, list) and planned_chain:
            manager.planned_chain = manager._normalize_planned_chain(planned_chain)
        else:
            manager.planned_chain = manager._build_default_chain(manager.initial_stage_id)
        manager.stage_index = manager._bounded_int(
            payload.get("stage_index"), 0, 0, max(0, len(manager.planned_chain) - 1)
        )
        manager.stage_start_time = _coerce_datetime(payload.get("stage_start_time"))
        history = payload.get("stage_history", [])
        manager.stage_history = []
        if isinstance(history, list):
            for row in history:
                if not isinstance(row, dict):
                    continue
                item = dict(row)
                item["timestamp"] = _coerce_datetime(item.get("timestamp"))
                manager.stage_history.append(item)
        dialogue_history = payload.get("dialogue_history", [])
        manager.dialogue_history = [dict(item) for item in dialogue_history if isinstance(item, dict)]
        if len(manager.dialogue_history) > manager.max_dialog_history:
            manager.dialogue_history = manager.dialogue_history[-manager.max_dialog_history :]
        manager.last_session_context = copy.deepcopy(payload.get("last_session_context", {})) if isinstance(payload.get("last_session_context", {}), dict) else {}
        manager.last_evaluation = copy.deepcopy(payload.get("last_evaluation", {})) if isinstance(payload.get("last_evaluation", {}), dict) else {}
        manager._ensure_future_window()
        manager.stage_index = min(manager.stage_index, max(0, len(manager.planned_chain) - 1))
        return manager

    def _preview_future_chain(self, current_stage: Dict[str, Any]) -> List[str]:
        current_id = str(current_stage.get("id", "") or "").strip()
        if not current_id:
            return []
        future_ids = [current_id]
        if self.stage_index < len(self.planned_chain) - 1:
            future_ids.extend([str(item) for item in self.planned_chain[self.stage_index + 1 : self.stage_index + self.window_size + 1]])
        if len(future_ids) <= 1:
            future_ids.extend(self._build_future_chain_from_stage(current_id, self.window_size))
        return self._normalize_next_chain(future_ids, current_id)

    def _replace_future_chain(self, next_chain: List[str]) -> None:
        current_prefix = self.planned_chain[: self.stage_index]
        normalized = self._normalize_next_chain(next_chain, self.get_current_stage_id())
        concrete_suffix = self._normalize_planned_chain(normalized)
        self.planned_chain = current_prefix + concrete_suffix
        self.stage_index = min(self.stage_index, max(0, len(self.planned_chain) - 1))

    def _ensure_future_window(self) -> None:
        if not self.planned_chain:
            self.planned_chain = self._build_default_chain(self.initial_stage_id)
            self.stage_index = 0
            return
        remaining = len(self.planned_chain) - self.stage_index - 1
        if remaining >= self.window_size:
            return
        current_id = self.get_current_stage_id()
        needed = self.window_size - remaining
        self.planned_chain.extend(self._build_future_chain_from_stage(current_id, needed))
        self.planned_chain = self._normalize_planned_chain(self.planned_chain)
        self.stage_index = min(self.stage_index, max(0, len(self.planned_chain) - 1))

    def _build_default_chain(self, initial_stage_id: str) -> List[str]:
        start_id = str(initial_stage_id or "").strip()
        if not start_id or start_id not in self.stage_catalog:
            start_id = next(iter(self.stage_catalog.keys()))
        chain = [start_id]
        chain.extend(self._build_future_chain_from_stage(start_id, self.window_size))
        return self._normalize_planned_chain(chain)

    def _build_future_chain_from_stage(self, stage_id: str, count: int) -> List[str]:
        results: List[str] = []
        current_id = str(stage_id or "").strip()
        seen = {current_id}
        for _ in range(max(0, int(count))):
            next_id = ""
            for pos in range(max(0, int(self.stage_index)), max(0, len(self.planned_chain) - 1)):
                if str(self.planned_chain[pos] or "").strip() == current_id:
                    planned_next = str(self.planned_chain[pos + 1] or "").strip()
                    if planned_next and planned_next in self.stage_catalog and planned_next not in seen:
                        next_id = planned_next
                    break
            if not next_id:
                stage = self.stage_catalog.get(current_id, {})
                next_candidates = self._normalize_path_tokens(stage.get("next_candidates", []), allow_unknown=True) if isinstance(stage, dict) else []
                for candidate_id in next_candidates:
                    if candidate_id in seen:
                        continue
                    if candidate_id in self.stage_catalog:
                        next_id = candidate_id
                        break
                    if candidate_id in self.pending_stage_specs:
                        break
            if not next_id:
                break
            seen.add(next_id)
            results.append(next_id)
            current_id = next_id
        return results

    def _heuristic_match(
        self,
        current_stage: Dict[str, Any],
        session_context: Dict[str, Any],
        conversation_content: str,
    ) -> Tuple[bool, float, str]:
        conversation = str(conversation_content or "").strip()
        if not conversation:
            return False, 0.0, "empty_utterance"

        stage_focus = [str(item) for item in self._to_list(current_stage.get("narrative_focus", []))]
        stage_text = " ".join(
            [
                str(current_stage.get("label", "") or ""),
                str(current_stage.get("summary", "") or ""),
                str(current_stage.get("core_belief", "") or ""),
                " ".join(stage_focus),
            ]
        )
        stage_keywords = self._extract_keywords(stage_text)
        conversation_keywords = set(self._extract_keywords(conversation))
        overlap = [kw for kw in stage_keywords if kw in conversation or kw in conversation_keywords]

        semantic = session_context.get("semantic_cues", {}) if isinstance(session_context.get("semantic_cues", {}), dict) else {}
        topics = [str(item) for item in self._to_list(semantic.get("topics", []))]
        speech_acts = [str(item) for item in self._to_list(semantic.get("speech_acts", []))]
        stance = [str(item) for item in self._to_list(semantic.get("stance", []))]

        focus_hits = [item for item in stage_focus if item and any(item in topic or topic in item for topic in topics)]
        signal_hits = self._count_signal_hits(current_stage.get("advance_signals", []), conversation, topics)
        hold_alignment = self._count_signal_hits(current_stage.get("hold_signals", []), conversation, topics)

        score = 0.0
        if stage_keywords:
            score += min(0.42, 0.10 * float(len(overlap)))
        if focus_hits:
            score += min(0.24, 0.12 * float(len(focus_hits)))
        if "自我暴露" in speech_acts:
            score += 0.10
        if "具体叙述" in speech_acts:
            score += 0.10
        if "含蓄求助" in speech_acts or "求助尝试" in speech_acts:
            score += 0.06
        if "谨慎" in stance or "试探" in stance:
            score += 0.05
        if hold_alignment > 0:
            score += min(0.10, 0.04 * float(hold_alignment))
        if signal_hits > 0:
            score += min(0.10, 0.05 * float(signal_hits))

        threshold = max(0.32, min(0.58, self.min_match_confidence * 0.72))
        matched = score >= threshold
        reason = "heuristic_score={:.3f}; overlap={}; topics={}; speech_acts={}".format(
            score,
            ",".join(overlap[:4]) if overlap else "none",
            ",".join(topics[:3]) if topics else "none",
            ",".join(speech_acts[:3]) if speech_acts else "none",
        )
        return matched, round(float(score), 4), reason

    def _decide_action(
        self,
        current_stage: Dict[str, Any],
        matched: bool,
        match_confidence: float,
        session_context: Dict[str, Any],
        conversation_content: str,
    ) -> str:
        if not matched:
            return "hold"
        if self._coerce_bool(current_stage.get("is_terminal_stage", False)):
            return "hold"
        next_candidates = current_stage.get("next_candidates", []) if isinstance(current_stage.get("next_candidates", []), list) else []
        if not next_candidates:
            return "hold"

        semantic = session_context.get("semantic_cues", {}) if isinstance(session_context.get("semantic_cues", {}), dict) else {}
        topics = [str(item) for item in self._to_list(semantic.get("topics", []))]
        speech_acts = [str(item) for item in self._to_list(semantic.get("speech_acts", []))]
        advance_hits = self._count_signal_hits(current_stage.get("advance_signals", []), conversation_content, topics)
        hold_hits = self._count_signal_hits(current_stage.get("hold_signals", []), conversation_content, topics)

        detailed = "具体叙述" in speech_acts or len(str(conversation_content or "")) >= 36
        if advance_hits > hold_hits and (advance_hits > 0 or detailed):
            return "advance"
        if detailed and match_confidence >= max(0.46, self.min_match_confidence * 0.80):
            return "advance"
        return "hold"

    def _count_signal_hits(self, signals: Any, conversation: str, topics: List[str]) -> int:
        count = 0
        for signal in self._to_list(signals):
            text = str(signal or "").strip()
            if not text:
                continue
            if text in conversation or any(text in topic or topic in text for topic in topics):
                count += 1
        return count

    def _infer_chain_signal(
        self,
        completion_func: Callable[[str], str],
        session_context: Dict[str, Any],
        conversation_content: str,
        llm_cfg: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        prompt = self._build_chain_prompt(session_context, conversation_content, llm_cfg)
        raw = ""
        try:
            raw = str(completion_func(prompt) or "")
        except Exception:
            raw = ""
        parsed = self._parse_json_object(raw)
        return self._normalize_llm_signal(parsed)

    def _build_chain_prompt(
        self,
        session_context: Dict[str, Any],
        conversation_content: str,
        llm_cfg: Optional[Dict[str, Any]],
    ) -> str:
        cfg = llm_cfg if isinstance(llm_cfg, dict) else {}
        text_limit = self._bounded_int(cfg.get("max_text_length"), 1200, 200, 6000)
        current_stage = self.get_current_stage()
        payload = {
            "current_stage": current_stage,
            "current_chain_window": self.get_current_chain_window(self.window_size + 1),
            "current_stage_branch_options": self._get_branch_options(current_stage),
            "recent_stage_history": self._get_recent_history_summary(limit=4),
            "pending_stage_ids": list(self.pending_stage_specs.keys())[:8],
            "session_context": session_context,
            "conversation_content": self._clip_text(conversation_content, limit=text_limit),
            "allowed_actions": ["hold", "advance", "replan", "jump"],
        }
        payload_json = json.dumps(payload, ensure_ascii=False)
        return (
            "你是主诉链规划器。\n"
            "任务：依据当前主诉节点与本轮会话，判断本轮是否实质触及当前节点，并给出一条以当前节点为锚点的活跃路径候选。\n"
            "你只负责链路判断，不负责生成完整的新节点对象。系统会按顺序逐步物化 next_chain 里的新占位节点。\n"
            "只输出 JSON 对象，不要输出解释、markdown 或额外文本。\n"
            "输出字段固定为：\n"
            "{\n"
            '  "matched_current_stage": true,\n'
            '  "match_confidence": 0.0,\n'
            '  "match_reason": "10到120字",\n'
            '  "action": "hold|advance|replan|jump",\n'
            '  "next_chain": ["当前节点id", "后续id1", "后续id2"],\n'
            '  "needs_new_stage": false,\n'
            '  "new_stage_brief": "若需要新增节点，用一句话说明本轮首先要生成的新节点为何必要；否则为空"\n'
            "}\n"
            "约束：\n"
            "1. 不要输出病情等级或严重程度。\n"
            "2. 只能围绕主诉链节点是否被触及来判断。\n"
            "3. next_chain[0] 必须是当前节点 id。\n"
            "4. next_chain 后续可以写多个 id，既可以是已有节点 id，也可以是新节点占位 id。\n"
            "5. 若首次引入新的占位 id，needs_new_stage 必须为 true，并给出 new_stage_brief。\n"
            "6. 可以回退到旧节点、切换分支或继续前进，但不要制造 A↔B 式机械往返。\n"
            "7. 不要在第一阶段输出完整 stage 对象。\n"
            f"输入：{payload_json}\n"
        )

    def _normalize_llm_signal(self, payload: Any) -> Optional[Dict[str, Any]]:
        if not isinstance(payload, dict):
            return None

        matched = self._coerce_bool(
            payload.get("matched_current_stage", payload.get("matched", False))
        )
        match_confidence = self._bounded_float(
            payload.get("match_confidence", payload.get("confidence", 0.0)),
            0.0,
            0.0,
            1.0,
        )
        match_reason = str(
            payload.get("match_reason", payload.get("match_evidence", "")) or ""
        ).strip()[:180]
        action = str(payload.get("action", payload.get("chain_action", "hold")) or "hold").strip().lower()
        if action not in {"hold", "advance", "replan", "jump"}:
            action = "hold"

        raw_next = payload.get("next_chain", payload.get("next_stages", []))
        next_chain: List[str] = []
        if isinstance(raw_next, list):
            for item in raw_next:
                text = str(item or "").strip()
                if text:
                    next_chain.append(text)

        needs_new_stage = self._coerce_bool(payload.get("needs_new_stage", False))
        new_stage_brief = str(payload.get("new_stage_brief", "") or "").strip()[:180]

        if matched and match_confidence < self.min_match_confidence:
            matched = False
            if action == "advance":
                action = "hold"

        return {
            "matched": bool(matched),
            "match_confidence": round(float(match_confidence), 4),
            "match_reason": match_reason,
            "action": action,
            "next_chain": next_chain,
            "needs_new_stage": bool(needs_new_stage),
            "new_stage_brief": new_stage_brief,
        }

    def _normalize_next_chain(self, value: Any, current_stage_id: str) -> List[str]:
        current_id = str(current_stage_id or "").strip()
        normalized = self._normalize_path_tokens(value, allow_unknown=True)
        if not normalized:
            return [current_id] if current_id else []
        if current_id:
            if current_id not in normalized:
                normalized = [current_id] + normalized
            elif normalized[0] != current_id:
                first_idx = normalized.index(current_id)
                normalized = [current_id] + normalized[:first_idx] + normalized[first_idx + 1 :]
        return self._normalize_path_tokens(normalized, allow_unknown=True)

    def _normalize_chain_ids(self, value: Any) -> List[str]:
        items = self._to_list(value)
        normalized: List[str] = []
        seen = set()
        for item in items:
            stage_id = ""
            if isinstance(item, dict):
                stage = self._sanitize_stage(item, source=str(item.get("source", "config")) if isinstance(item, dict) else "config")
                candidate_id = str(stage.get("id", "") or "").strip()
                if candidate_id in self.stage_catalog:
                    stage_id = candidate_id
            else:
                text = str(item or "").strip()
                if text in self.stage_catalog:
                    stage_id = text
            if not stage_id or stage_id in seen:
                continue
            seen.add(stage_id)
            normalized.append(stage_id)
        return normalized

    def _build_stage_generation_prompt(
        self,
        current_stage: Dict[str, Any],
        session_context: Dict[str, Any],
        conversation_content: str,
        llm_cfg: Optional[Dict[str, Any]],
        new_stage_brief: str,
        requested_stage_id: str = "",
        requested_tail: Optional[List[str]] = None,
    ) -> str:
        cfg = llm_cfg if isinstance(llm_cfg, dict) else {}
        text_limit = self._bounded_int(cfg.get("max_text_length"), 1200, 200, 6000)
        payload = {
            "current_stage": copy.deepcopy(current_stage),
            "current_chain_window": self.get_current_chain_window(self.window_size + 1),
            "current_stage_branch_options": self._get_branch_options(current_stage),
            "recent_stage_history": self._get_recent_history_summary(limit=4),
            "session_context": copy.deepcopy(session_context),
            "conversation_content": self._clip_text(conversation_content, limit=text_limit),
            "requested_stage_id": str(requested_stage_id or "").strip(),
            "requested_tail": self._normalize_path_tokens(requested_tail or [], allow_unknown=True),
            "new_stage_brief": self._clip_text(new_stage_brief, limit=180),
        }
        payload_json = json.dumps(payload, ensure_ascii=False)
        return (
            "你是主诉链新节点生成器。\n"
            "任务：根据当前主诉节点、会话上下文和新增节点简述，生成当前应接入活跃路径的下一个新主诉节点。\n"
            "只输出单个 JSON 对象，不要输出解释、markdown 或额外文本。\n"
            "输出字段至少包含：\n"
            "{\n"
            '  "id": "new_stage_id",\n'
            '  "label": "节点标题",\n'
            '  "summary": "节点概述",\n'
            '  "core_belief": "可选",\n'
            '  "narrative_focus": ["可选"],\n'
            '  "next_candidates": ["可选，可连接已有节点或留空"],\n'
            '  "is_terminal_stage": false\n'
            "}\n"
            "约束：\n"
            "1. 只生成当前应物化的一个新节点，不要一次展开完整子树。\n"
            "2. 若输入中给了 requested_stage_id，就使用这个 id。\n"
            "3. 该节点应紧接当前节点，但不必是叶子；它可以承接后续分支。\n"
            "4. next_candidates 可以写多个，优先与输入中的 requested_tail 保持连续。\n"
            "5. 可以把 next_candidates 连接到已有节点，也可以留空；若拿不准，宁可保守也不要生成孤立死节点。\n"
            "6. summary 必须具体，不能只重复 label。\n"
            f"输入：{payload_json}\n"
        )

    def _validate_runtime_stage(self, payload: Any) -> Tuple[bool, str]:
        if not isinstance(payload, dict):
            return False, "payload_not_dict"
        stage_id = str(payload.get("id", "") or "").strip()
        label = str(payload.get("label", "") or "").strip()
        summary = str(payload.get("summary", payload.get("description", "")) or "").strip()
        if not stage_id:
            return False, "missing_id"
        if not label:
            return False, "missing_label"
        if not summary:
            return False, "missing_summary"
        existing = self.stage_catalog.get(stage_id)
        if isinstance(existing, dict) and str(existing.get("source", "") or "") == "config":
            return False, "conflicts_with_config_stage"
        return True, "ok"

    def _log_runtime_stage_event(
        self,
        event: str,
        current_stage_id: str,
        needs_new_stage: bool,
        stage_id: str = "",
        reason: str = "",
    ) -> None:
        logger.info(
            "[DEPRESSION_DYNAMIC_STAGE_%s] current_stage=%s needs_new_stage=%s new_stage=%s reason=%s",
            str(event or "UNKNOWN").upper(),
            str(current_stage_id or ""),
            bool(needs_new_stage),
            str(stage_id or ""),
            str(reason or ""),
        )

    def _has_existing_followup(self, current_stage: Dict[str, Any], next_chain_ids: List[str]) -> bool:
        if len(self._normalize_next_chain(next_chain_ids, str(current_stage.get("id", "") or ""))) > 1:
            return True
        next_candidates = self._normalize_candidate_ids(current_stage.get("next_candidates", []), allow_pending=True) if isinstance(current_stage, dict) else []
        return bool(next_candidates)

    def _generate_runtime_stage(
        self,
        completion_func: Callable[[str], str],
        session_context: Dict[str, Any],
        conversation_content: str,
        llm_cfg: Optional[Dict[str, Any]],
        current_stage: Dict[str, Any],
        new_stage_brief: str,
        requested_stage_id: str = "",
        requested_tail: Optional[List[str]] = None,
    ) -> Optional[Dict[str, Any]]:
        prompt = self._build_stage_generation_prompt(
            current_stage=current_stage,
            session_context=session_context,
            conversation_content=conversation_content,
            llm_cfg=llm_cfg,
            new_stage_brief=new_stage_brief,
            requested_stage_id=requested_stage_id,
            requested_tail=requested_tail,
        )
        raw = ""
        try:
            raw = str(completion_func(prompt) or "")
        except Exception as exc:
            self._log_runtime_stage_event(
                event="reject",
                current_stage_id=str(current_stage.get("id", "") or ""),
                needs_new_stage=True,
                reason="completion_error:{}".format(exc),
            )
            return None
        parsed = self._parse_json_object(raw)
        if requested_stage_id and isinstance(parsed, dict):
            parsed["id"] = str(requested_stage_id or "").strip()
        is_valid, reason = self._validate_runtime_stage(parsed)
        if not is_valid:
            stage_id = str(parsed.get("id", "") or "").strip() if isinstance(parsed, dict) else ""
            self._log_runtime_stage_event(
                event="reject",
                current_stage_id=str(current_stage.get("id", "") or ""),
                needs_new_stage=True,
                stage_id=stage_id,
                reason=reason,
            )
            return None
        stage = self._sanitize_stage(parsed, source="llm")
        stage_tail = self._normalize_path_tokens(requested_tail or [], allow_unknown=True)
        stage["next_candidates"] = self._merge_candidate_ids(stage_tail, stage.get("next_candidates", []))
        self.stage_catalog[stage["id"]] = stage
        self.pending_stage_specs.pop(str(stage.get("id", "") or ""), None)
        self._attach_candidate(str(current_stage.get("id", "") or ""), str(stage.get("id", "") or ""), prefer_front=True)
        self._log_runtime_stage_event(
            event="create",
            current_stage_id=str(current_stage.get("id", "") or ""),
            needs_new_stage=True,
            stage_id=str(stage.get("id", "") or ""),
            reason="ok",
        )
        return copy.deepcopy(stage)

    def _sanitize_stage(self, raw: Any, source: str = "config") -> Dict[str, Any]:
        payload = raw if isinstance(raw, dict) else {}
        label = str(payload.get("label", "") or "").strip() or "未命名主诉节点"
        stage_id = str(payload.get("id", "") or "").strip() or self._make_stage_id(label)
        summary = str(payload.get("summary", payload.get("description", label)) or label).strip()
        core_belief = str(payload.get("core_belief", "") or "").strip()
        speaking_style = payload.get("speaking_style", {}) if isinstance(payload.get("speaking_style", {}), dict) else {}
        emotion_vector = payload.get("emotion_vector", {}) if isinstance(payload.get("emotion_vector", {}), dict) else {}
        bias_profile = payload.get("bias_profile", {}) if isinstance(payload.get("bias_profile", {}), dict) else {}
        relation_modifiers = payload.get("relation_modifiers", {}) if isinstance(payload.get("relation_modifiers", {}), dict) else {}

        normalized = ComplaintStage(
            id=stage_id[:80],
            label=label[:80],
            summary=self._clip_text(summary, limit=220),
            core_belief=self._clip_text(core_belief, limit=220),
            narrative_focus=[str(item)[:60] for item in self._to_list(payload.get("narrative_focus", [])) if str(item).strip()][:12],
            speaking_style={
                "tempo": str(speaking_style.get("tempo", "slow") or "slow")[:32],
                "disclosure": str(speaking_style.get("disclosure", "guarded") or "guarded")[:32],
                "tone": str(speaking_style.get("tone", "flat") or "flat")[:48],
                "repair_pattern": self._clip_text(speaking_style.get("repair_pattern", "说一点、收一点"), limit=120),
            },
            emotion_vector={
                "valence": round(self._bounded_float(emotion_vector.get("valence"), 0.25, 0.0, 1.0), 4),
                "arousal": round(self._bounded_float(emotion_vector.get("arousal"), 0.35, 0.0, 1.0), 4),
                "defensiveness": round(self._bounded_float(emotion_vector.get("defensiveness"), 0.55, 0.0, 1.0), 4),
                "shame": round(self._bounded_float(emotion_vector.get("shame"), 0.40, 0.0, 1.0), 4),
                "hopelessness": round(self._bounded_float(emotion_vector.get("hopelessness"), 0.35, 0.0, 1.0), 4),
                "trust": round(self._bounded_float(emotion_vector.get("trust"), 0.20, 0.0, 1.0), 4),
            },
            bias_profile=copy.deepcopy(bias_profile),
            advance_signals=[str(item)[:80] for item in self._to_list(payload.get("advance_signals", [])) if str(item).strip()][:12],
            hold_signals=[str(item)[:80] for item in self._to_list(payload.get("hold_signals", [])) if str(item).strip()][:12],
            relation_modifiers=copy.deepcopy(relation_modifiers),
            next_candidates=[str(item)[:80] for item in self._to_list(payload.get("next_candidates", [])) if str(item).strip()][:8],
            is_terminal_stage=self._coerce_bool(payload.get("is_terminal_stage", payload.get("terminal_recovery", False))),
            source=str(payload.get("source", source) or source)[:32],
        )
        return normalized.to_dict()

    @staticmethod
    def _make_stage_id(label: str) -> str:
        text = str(label or "").strip()
        if not text:
            text = "stage"
        digest = hashlib.md5(text.encode("utf-8")).hexdigest()[:10]
        safe = re.sub(r"[^A-Za-z0-9_\-]+", "_", text).strip("_")
        if not safe:
            safe = "stage"
        return f"{safe[:24]}_{digest}"

    def _track_dialogue(
        self,
        current_stage: Dict[str, Any],
        evaluation: Dict[str, Any],
        conversation_excerpt: str,
        session_context: Dict[str, Any],
    ) -> None:
        participants = session_context.get("participants", {}) if isinstance(session_context.get("participants", {}), dict) else {}
        scene = session_context.get("scene", {}) if isinstance(session_context.get("scene", {}), dict) else {}
        row = {
            "timestamp": self._now().isoformat(),
            "stage_id": str(current_stage.get("id", "") or ""),
            "stage_label": str(current_stage.get("label", "") or ""),
            "conversation_excerpt": self._clip_text(conversation_excerpt, limit=220),
            "matched": bool(evaluation.get("matched", False)),
            "match_confidence": self._bounded_float(evaluation.get("match_confidence"), 0.0, 0.0, 1.0),
            "match_reason": str(evaluation.get("match_reason", "") or "")[:180],
            "action": str(evaluation.get("action", "hold") or "hold"),
            "other_agent": str(participants.get("other_agent", "") or ""),
            "relationship": str(participants.get("relationship", "") or ""),
            "interaction_type": str(scene.get("interaction_type", "") or ""),
        }
        self.dialogue_history.append(row)
        if len(self.dialogue_history) > self.max_dialog_history:
            self.dialogue_history = self.dialogue_history[-self.max_dialog_history :]

    def _record_history(
        self,
        action: str,
        from_stage: Dict[str, Any],
        to_stage: Dict[str, Any],
        matched: bool,
        match_confidence: float,
        match_reason: str,
        pointer_before: int,
        pointer_after: int,
        duration_minutes: float,
    ) -> None:
        self.stage_history.append(
            {
                "timestamp": self._now(),
                "action": str(action or "hold"),
                "from_stage_id": str(from_stage.get("id", "") or ""),
                "from_stage_label": str(from_stage.get("label", "") or ""),
                "to_stage_id": str(to_stage.get("id", "") or ""),
                "to_stage_label": str(to_stage.get("label", "") or ""),
                "matched": bool(matched),
                "match_confidence": round(float(match_confidence), 4),
                "match_reason": str(match_reason or "")[:180],
                "pointer_before": int(pointer_before),
                "pointer_after": int(pointer_after),
                "duration_minutes": float(duration_minutes),
            }
        )

    @staticmethod
    def _clip_text(value: Any, limit: int = 1200) -> str:
        text = str(value or "").strip()
        if len(text) <= int(limit):
            return text
        return text[: max(0, int(limit) - 1)] + "…"

    @classmethod
    def _extract_keywords(cls, text: Any) -> List[str]:
        source = str(text or "")
        chunks = re.findall(r"[\u4e00-\u9fff]{2,8}|[A-Za-z0-9_\-]{3,}", source)
        results: List[str] = []
        seen = set()
        for chunk in chunks:
            token = str(chunk).strip()
            if not token or token in cls.STOPWORDS or token in seen:
                continue
            seen.add(token)
            results.append(token)
        return results[:16]

    @staticmethod
    def _parse_json_object(raw: Any) -> Optional[Dict[str, Any]]:
        text = str(raw or "").strip()
        if not text:
            return None
        fenced = re.findall(r"```(?:json)?\s*([\s\S]*?)\s*```", text, flags=re.IGNORECASE)
        for chunk in [text] + fenced:
            data = str(chunk or "").strip()
            if not data:
                continue
            try:
                parsed = json.loads(data)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                continue
        left = text.find("{")
        right = text.rfind("}")
        if left >= 0 and right > left:
            snippet = text[left : right + 1]
            try:
                parsed = json.loads(snippet)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                return None
        return None

    @staticmethod
    def _coerce_bool(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        return str(value or "").strip().lower() in {"1", "true", "yes", "y", "on"}

    @staticmethod
    def _to_list(value: Any) -> List[Any]:
        if value is None:
            return []
        if isinstance(value, list):
            return list(value)
        return [value]

    @staticmethod
    def _bounded_float(value: Any, default: float, lower: float, upper: float) -> float:
        try:
            num = float(value)
        except Exception:
            num = float(default)
        num = max(float(lower), num)
        num = min(float(upper), num)
        return num

    @staticmethod
    def _bounded_int(value: Any, default: int, lower: int, upper: int) -> int:
        try:
            num = int(value)
        except Exception:
            num = int(default)
        num = max(int(lower), num)
        num = min(int(upper), num)
        return num

    @staticmethod
    def _disclosure_to_score(value: Any) -> float:
        text = str(value or "").strip().lower()
        mapping = {
            "sealed": 0.10,
            "guarded": 0.22,
            "shielded": 0.18,
            "cautious": 0.34,
            "partial": 0.48,
            "tentative": 0.44,
            "open": 0.66,
            "full": 0.82,
        }
        return float(mapping.get(text, 0.28))


SymptomStateMachine = ComplaintChainManager


def _coerce_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        text = value.strip()
        if text:
            try:
                return datetime.fromisoformat(text)
            except Exception:
                pass
    return datetime.now()
