"""主诉图管理器。"""

from __future__ import annotations

import copy
import difflib
import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from .prompt_builder import DynamicPromptBuilder
from .prompt_templates import render_prompt


def _simulation_now() -> datetime:
    """Return the active simulated-world time, falling back to wall time only outside simulation."""
    try:
        from modules import utils

        return utils.get_timer().get_date()
    except Exception:
        return datetime.now()


@dataclass
class ComplaintStage:
    """单个主诉图节点。"""

    id: str
    label: str
    summary: str = ""
    core_belief: str = ""
    narrative_focus: List[str] = field(default_factory=list)
    speaking_style: Dict[str, Any] = field(default_factory=dict)
    emotion_vector: Dict[str, float] = field(default_factory=dict)
    advance_signals: List[str] = field(default_factory=list)
    hold_signals: List[str] = field(default_factory=list)
    relation_modifiers: Dict[str, Any] = field(default_factory=dict)
    next_candidates: List[str] = field(default_factory=list)
    is_terminal_stage: bool = False
    source: str = "config"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ComplaintGraphManager:
    """仅围绕主诉图推进的人设动态管理器。

    它维护的不是“病情数值总表”，而是一个更离散的对象：
    当前角色正卡在哪个主诉节点、下一步可能走向哪里、为什么推进/停留。

    建议把内部状态理解为三部分：
    - `stage_catalog`：所有可能节点的运行态图索引；
    - `planned_graph + stage_index`：当前生效的“前瞻路径”；
    - `stage_history + dialogue_history`：运行过程留下的“证据轨迹”。
    """

    DEFAULT_STAGE: Dict[str, Any] = {
        "id": "unconfigured_stage",
        "label": "尚未配置主诉图",
        "summary": "当前角色还没有被配置可推进的主诉图节点。",
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
        "advance_signals": [],
        "hold_signals": ["沉默", "不知道说什么"],
        "relation_modifiers": {},
        "next_candidates": [],
        "is_terminal_stage": False,
        "source": "fallback",
    }

    DOMAIN_KEYS = (
        "mood_anhedonia",
        "sleep",
        "appetite",
        "energy",
        "attention",
        "function",
        "avoidance",
    )
    DOMAIN_DIRECTIONS = {"improved", "worsened", "unchanged"}
    DOMAIN_EVIDENCE_DIRECTIONS = {"improved", "worsened", "stable", "unclear"}
    DOMAIN_UPDATE_VALUES = {"no_update", "-10", "-5", "0", "+5", "+10"}
    DOMAIN_ANCHOR_UPDATES = {
        "improved": {"+5", "+10"},
        "worsened": {"-5", "-10"},
        "stable": {"0"},
        "insufficient": {"no_update"},
    }
    DOMAIN_CHANGE_STEP = 5
    DOMAIN_EVIDENCE_LIMIT = 5
    DOMAIN_WINDOW_CANDIDATE_LIMIT = 60
    TRANSITION_CANDIDATE_COUNT = 3

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        now_provider: Optional[Callable[[], datetime]] = None,
        domain_state_enabled: bool = True,
    ):
        config = config if isinstance(config, dict) else {}
        # 支持两种传法：
        # 1. 直接传 complaint_graph 配置本身；
        # 2. 传整个 depression_config.json，并优先取 `complaint_graph`。
        graph_config = self._select_graph_config(config)

        self._raw_graph_config = copy.deepcopy(graph_config)
        self.domain_state_enabled = bool(domain_state_enabled)
        self.planner: Dict[str, Any] = copy.deepcopy(graph_config.get("planner", {}))
        self.window_size = self._bounded_int(self.planner.get("window_size"), 3, 1, 8)
        self.llm_enabled = self._coerce_bool(self.planner.get("llm_enabled", True))
        self.max_dialog_history = self._bounded_int(
            self.planner.get("max_dialog_history"), 12, 4, 50
        )
        self._initial_domain_values = self._sanitize_initial_domain_state(
            graph_config.get("initial_domain_state", {})
        )

        self._now_provider: Callable[[], datetime] = (
            now_provider if callable(now_provider) else _simulation_now
        )

        self.stage_catalog: Dict[str, Dict[str, Any]] = {}
        raw_stages = graph_config.get("stages", graph_config.get("stage_catalog", []))
        if isinstance(raw_stages, list):
            for item in raw_stages:
                # 每个 stage 都先经过 sanitize，再进入 catalog。
                # 也就是说，JSON 配置不是直接拿来跑，而是会被统一清洗。
                stage = self._sanitize_stage(item, source=str(item.get("source", "config")) if isinstance(item, dict) else "config")
                self.stage_catalog[stage["id"]] = stage

        if not self.stage_catalog:
            default_stage = self._sanitize_stage(self.DEFAULT_STAGE, source="fallback")
            self.stage_catalog[default_stage["id"]] = default_stage

        initial_stage_id = str(
            graph_config.get("initial_stage_id")
            or graph_config.get("current_stage_id")
            or next(iter(self.stage_catalog.keys()))
        ).strip()
        if initial_stage_id not in self.stage_catalog:
            initial_stage_id = next(iter(self.stage_catalog.keys()))
        self.initial_stage_id = initial_stage_id
        configured_core_belief = self._normalize_core_belief(
            graph_config.get("core_belief", "")
        )
        initial_core_belief = self._normalize_core_belief(
            self.stage_catalog.get(self.initial_stage_id, {}).get("core_belief", "")
        )
        self._core_belief = configured_core_belief or initial_core_belief
        for stage in self.stage_catalog.values():
            stage["core_belief"] = self._core_belief
        configured_root_anchor = self._normalize_root_complaint_anchor(
            graph_config.get("root_complaint_anchor", "")
        )
        self._root_complaint_anchor = configured_root_anchor or self._derive_root_complaint_anchor(
            self.stage_catalog.get(self.initial_stage_id, {})
        )

        self.planned_graph: List[str] = []
        self.stage_index = 0
        self.stage_history: List[Dict[str, Any]] = []
        self.dialogue_history: List[Dict[str, Any]] = []
        self.last_session_context: Dict[str, Any] = {}
        self.last_evaluation: Dict[str, Any] = {}
        self.domain_state: Dict[str, Any] = {}
        self.pending_domain_window: Dict[str, Any] = {}
        self.last_domain_window_update: Dict[str, Any] = {}
        self.last_domain_window_id = ""
        self.stage_start_time = self._now()

        self.reset()

    @staticmethod
    def _select_graph_config(config: Dict[str, Any]) -> Dict[str, Any]:
        if "complaint_graph" in config and isinstance(config.get("complaint_graph", {}), dict):
            return copy.deepcopy(config.get("complaint_graph", {}))
        return copy.deepcopy(config if isinstance(config, dict) else {})

    @classmethod
    def _sanitize_initial_domain_state(cls, raw: Any) -> Dict[str, int]:
        payload = raw if isinstance(raw, dict) else {}
        values: Dict[str, int] = {}
        for domain in cls.DOMAIN_KEYS:
            value = payload.get(domain)
            if isinstance(value, bool) or not isinstance(value, int):
                continue
            if 0 <= value <= 100:
                values[domain] = int(value)
        return values

    def _now(self) -> datetime:
        try:
            value = self._now_provider()
        except Exception:
            value = _simulation_now()
        return _coerce_datetime(value, fallback=_simulation_now)

    def _build_initial_domain_state(self) -> Dict[str, Any]:
        observed_at = self._now().isoformat()
        return {
            "schema_version": 1,
            "domains": {
                domain: {
                    "initial_value": int(value),
                    "value": int(value),
                    "last_change": "unchanged",
                    "last_observed_at": observed_at,
                    "evidence": [],
                }
                for domain, value in self._initial_domain_values.items()
            },
        }

    def _restore_domain_state(self, raw: Any) -> None:
        if not self.domain_state_enabled:
            self.domain_state = {}
            return
        payload = raw if isinstance(raw, dict) else {}
        saved_domains = (
            payload.get("domains", {})
            if isinstance(payload.get("domains", {}), dict)
            else {}
        )
        current_domains = self.domain_state.get("domains", {})
        if not isinstance(current_domains, dict):
            current_domains = {}
        for domain in self.DOMAIN_KEYS:
            saved = saved_domains.get(domain, {})
            baseline = self._initial_domain_values.get(domain)
            saved_initial = saved.get("initial_value") if isinstance(saved, dict) else None
            if baseline is None and (
                isinstance(saved_initial, bool)
                or not isinstance(saved_initial, int)
                or not 0 <= saved_initial <= 100
            ):
                continue
            if baseline is None:
                baseline = int(saved_initial)
            if not isinstance(saved, dict):
                continue
            value = saved.get("value")
            if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 100:
                value = baseline
            initial_value = saved.get("initial_value")
            if (
                isinstance(initial_value, bool)
                or not isinstance(initial_value, int)
                or not 0 <= initial_value <= 100
            ):
                initial_value = baseline
            direction = str(saved.get("last_change", "unchanged") or "unchanged").strip()
            if direction not in self.DOMAIN_DIRECTIONS:
                direction = "unchanged"
            evidence_rows: List[Dict[str, str]] = []
            for item in self._to_list(saved.get("evidence", []))[-self.DOMAIN_EVIDENCE_LIMIT :]:
                if not isinstance(item, dict):
                    continue
                evidence_direction = str(item.get("direction", "") or "").strip()
                quote = self._clip_text(item.get("quote", ""), limit=240)
                if evidence_direction not in self.DOMAIN_DIRECTIONS or not quote:
                    continue
                evidence_row = {
                    "direction": evidence_direction,
                    "quote": quote,
                    "source": str(item.get("source", "chat") or "chat")[:32],
                    "observed_at": str(item.get("observed_at", "") or "")[:40],
                }
                if str(item.get("window_id", "") or "").strip():
                    evidence_row["window_id"] = str(item.get("window_id", "") or "")[:160]
                if str(item.get("update", "") or "") in self.DOMAIN_UPDATE_VALUES:
                    evidence_row["update"] = str(item.get("update", "") or "")
                evidence_rows.append(evidence_row)
            current_domains[domain] = {
                "initial_value": int(initial_value),
                "value": int(value),
                "last_change": direction,
                "last_observed_at": str(
                    saved.get("last_observed_at", self._now().isoformat()) or self._now().isoformat()
                )[:40],
                "evidence": evidence_rows,
            }
        self.domain_state = {"schema_version": 1, "domains": current_domains}

    def _domain_state_for_detector(self) -> Dict[str, Any]:
        domains = self.domain_state.get("domains", {})
        if not isinstance(domains, dict):
            domains = {}
        return {
            "scale": {
                "min": 0,
                "max": 100,
                "higher_is_better": True,
                "fixed_step": self.DOMAIN_CHANGE_STEP,
            },
            "domains": {
                domain: int(item.get("value", 0))
                for domain, item in domains.items()
                if domain in self.DOMAIN_KEYS
                and isinstance(item, dict)
                and isinstance(item.get("value"), int)
                and not isinstance(item.get("value"), bool)
            },
        }

    def _domain_state_for_window(self) -> Dict[str, Any]:
        view = self._domain_state_for_detector()
        view["scale"] = {
            "min": 0,
            "max": 100,
            "higher_is_better": True,
            "allowed_updates": ["no_update", "-10", "-5", "0", "+5", "+10"],
        }
        view["domains"] = {
            domain: {
                "value": value,
                "current_description": DynamicPromptBuilder.render_domain_description(
                    domain, value
                ),
            }
            for domain, value in view.get("domains", {}).items()
        }
        return view

    def reset(self) -> None:
        self.planned_graph = self._build_default_graph(self.initial_stage_id)
        self.stage_index = 0
        self.stage_history = []
        self.dialogue_history = []
        self.last_session_context = {}
        self.last_evaluation = {}
        self.domain_state = (
            self._build_initial_domain_state() if self.domain_state_enabled else {}
        )
        self.pending_domain_window = {}
        self.last_domain_window_update = {}
        self.last_domain_window_id = ""
        self.stage_start_time = self._now()

    def get_current_stage(self) -> Dict[str, Any]:
        if not self.planned_graph:
            self.reset()
        idx = min(max(0, int(self.stage_index)), len(self.planned_graph) - 1)
        stage_id = self.planned_graph[idx]
        return copy.deepcopy(self.stage_catalog.get(stage_id, self._sanitize_stage(self.DEFAULT_STAGE, source="fallback")))

    def get_current_stage_id(self) -> str:
        return str(self.get_current_stage().get("id", ""))

    @property
    def root_complaint_anchor(self) -> str:
        """Return the immutable long-term case background fixed at initialization."""
        return self._root_complaint_anchor

    @property
    def core_belief(self) -> str:
        """Return the immutable case-level belief fixed at initialization."""
        return self._core_belief

    @classmethod
    def _normalize_core_belief(cls, value: Any) -> str:
        text = re.sub(r"\s+", " ", str(value or "")).strip()
        return cls._clip_text(text, limit=220)

    @classmethod
    def _normalize_root_complaint_anchor(cls, value: Any) -> str:
        """Keep the anchor compact without importing stage symptom/detail fields."""
        text = re.sub(r"\s+", " ", str(value or "")).strip()
        return text[:240].rstrip("，,；;：:")

    @classmethod
    def _derive_root_complaint_anchor(cls, initial_stage: Any) -> str:
        """Conservatively derive a fallback from the initial complaint stage."""
        stage = initial_stage if isinstance(initial_stage, dict) else {}
        summary = cls._normalize_root_complaint_anchor(stage.get("summary", ""))
        core_belief = cls._normalize_root_complaint_anchor(stage.get("core_belief", ""))
        if summary and core_belief and core_belief not in summary:
            return cls._normalize_root_complaint_anchor(
                "{} 核心自我评价是：{}".format(summary, core_belief)
            )
        return summary or core_belief

    def get_current_graph_window(self, count: Optional[int] = None) -> List[Dict[str, Any]]:
        """返回当前节点及其直接候选分支。"""
        if not self.planned_graph:
            return []
        width = self._bounded_int(count, self.window_size + 1, 1, 20)
        current_stage = self.get_current_stage()
        current_id = str(current_stage.get("id", "") or "").strip()
        ids = [current_id] + self._candidate_ids_for_stage(current_stage, width - 1)
        return [copy.deepcopy(self.stage_catalog.get(stage_id, {})) for stage_id in ids]

    def get_graph_snapshot(self) -> Dict[str, Any]:
        current_stage = self.get_current_stage()
        graph_window = self.get_current_graph_window(self.window_size + 1)
        stages = [copy.deepcopy(item) for item in self.stage_catalog.values()]
        return {
            "mode": "complaint_graph",
            "core_belief": self.core_belief,
            "current_stage_id": str(current_stage.get("id", "")),
            "current_stage_label": str(current_stage.get("label", "")),
            "current_stage": current_stage,
            "stages": stages,
            "current_graph_window": graph_window,
            "window_size": int(self.window_size),
            "planned_graph": [str(item) for item in self.planned_graph],
            "stage_index": int(self.stage_index),
            "stage_start_time": self.stage_start_time.isoformat(),
            "stage_history": copy.deepcopy(self.stage_history),
            "dialogue_history": copy.deepcopy(self.dialogue_history[-self.max_dialog_history :]),
            "last_session_context": copy.deepcopy(self.last_session_context),
            "last_evaluation": copy.deepcopy(self.last_evaluation),
            "domain_state": copy.deepcopy(self.domain_state),
            "pending_domain_window": copy.deepcopy(self.pending_domain_window),
            "last_domain_window_update": copy.deepcopy(self.last_domain_window_update),
            "last_domain_window_id": str(self.last_domain_window_id),
        }

    def get_state_duration(self) -> float:
        return (self._now() - self.stage_start_time).total_seconds() / 60.0

    def evaluate_turn(
        self,
        session_context: Dict[str, Any],
        conversation_content: str,
        counterpart_utterance: str = "",
        completion_func: Optional[Callable[[str], str]] = None,
        llm_cfg: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """只评估，不提交。

        输出的是一个 evaluation dict，类似“事务草稿”：
        - 当前节点是否被触及；
        - 置信度如何；
        - 动作是 hold 还是 advance；
        - 如果要前进，本轮 transition 选中了哪个临时候选。
        真正改状态要等 `commit_turn()`。

        动作判定只交给 LLM（graph_transition.txt）：没有可用的 completion_func
        或判定结果时，一律默认 hold，不再走启发式规则或外部传入的信号。
        """
        session_context = session_context if isinstance(session_context, dict) else {}
        conversation = str(conversation_content or "").strip()
        current_stage = self.get_current_stage()

        # 动作判定唯一来源：graph_transition.txt 的 LLM 输出。
        transition_signal: Optional[Dict[str, Any]] = None
        if callable(completion_func) and self.llm_enabled:
            transition_signal = self._infer_transition_signal(
                completion_func=completion_func,
                current_stage=current_stage,
                session_context=session_context,
                conversation_content=conversation,
                counterpart_utterance=str(counterpart_utterance or "").strip(),
                llm_cfg=llm_cfg,
            )
        if not transition_signal:
            # 拿不到 LLM 判定时保持当前节点，不触发候选规划或图写入。
            transition_signal = {
                "matched": False,
                "match_reason": "no_llm_decision",
                "action": "hold",
                "next_graph": self._preview_future_graph(current_stage),
            }

        matched = bool(transition_signal.get("matched", False))
        match_reason = str(transition_signal.get("match_reason", "") or "")
        action = str(transition_signal.get("action", "hold") or "hold").strip().lower()
        next_stage = (
            copy.deepcopy(transition_signal.get("selected_stage"))
            if isinstance(transition_signal.get("selected_stage"), dict)
            else None
        )
        next_graph_value = transition_signal.get(
            "next_graph", self._preview_future_graph(current_stage)
        )
        raw_graph_ids = [
            str(item or "").strip()
            for item in self._to_list(next_graph_value)
            if not isinstance(item, dict) and str(item or "").strip()
        ]
        next_graph_ids = self._normalize_graph_path(raw_graph_ids, current_stage["id"])
        selected_id = str((next_stage or {}).get("id", "") or "").strip()
        raw_target_id = str(raw_graph_ids[1] if len(raw_graph_ids) > 1 else "").strip()
        if selected_id and raw_target_id == selected_id:
            # 临时候选尚未进入 catalog，不能让旧的路径规范化提前丢掉它。
            # 只有 transition 同时返回完整 selected_stage 和相同 target ID 时放行。
            next_graph_ids = [str(current_stage["id"]), selected_id]
        if action not in {"hold", "advance"}:
            action = "hold"
        if action == "advance" and self._coerce_bool(current_stage.get("is_terminal_stage", False)):
            # 终态节点不再向外推进，即便 LLM 给出了 advance。
            action = "hold"

        if action == "advance":
            target_id = str(next_graph_ids[1] if len(next_graph_ids) > 1 else "").strip()
            if (
                not target_id
                or not isinstance(next_stage, dict)
                or str(next_stage.get("id", "") or "").strip() != target_id
            ):
                # transition 必须显式选中一个本轮临时候选；没有 target 时
                # 不能再回退到“自动取第一个候选”。
                action = "hold"
                next_graph_ids = self._preview_future_graph(current_stage)
                next_stage = None
        if action != "advance":
            matched = False

        # evaluation 是“提交前快照”：
        # 后续 commit_turn() 只消费这个 dict，不再重新做一次理解。
        evaluation = {
            "action": action,
            "matched": bool(matched),
            "match_reason": self._clip_text(match_reason, limit=180),
            "current_stage": copy.deepcopy(current_stage),
            "next_stage": copy.deepcopy(next_stage) if isinstance(next_stage, dict) else None,
            "next_graph": [str(item) for item in next_graph_ids],
            "session_context": copy.deepcopy(session_context),
            "conversation_excerpt": self._clip_text(conversation, limit=220),
        }
        return evaluation

    def commit_turn(self, evaluation: Dict[str, Any]) -> Dict[str, Any]:
        """把 evaluation 真正写入状态机。

        这里最值得审查的点是：
        - `hold` 不改变图和指针；
        - `advance` 只把 transition 最终选中的临时候选写入图。
        """
        evaluation = evaluation if isinstance(evaluation, dict) else {}
        current_before = self.get_current_stage()
        action = str(evaluation.get("action", "hold") or "hold").strip().lower()
        if action not in {"hold", "advance"}:
            action = "hold"
        matched = bool(evaluation.get("matched", False))
        match_reason = str(evaluation.get("match_reason", "") or "")[:180]
        next_graph_value = evaluation.get("next_graph", [])
        selected_stage = (
            copy.deepcopy(evaluation.get("next_stage"))
            if isinstance(evaluation.get("next_stage"), dict)
            else None
        )
        selected_id = str((selected_stage or {}).get("id", "") or "").strip()
        raw_graph_ids = [
            str(item or "").strip()
            for item in self._to_list(next_graph_value)
            if not isinstance(item, dict) and str(item or "").strip()
        ]
        next_graph = self._normalize_graph_path(
            raw_graph_ids if action == "advance" else [],
            current_before.get("id", ""),
        )
        raw_target_id = str(raw_graph_ids[1] if len(raw_graph_ids) > 1 else "").strip()
        if selected_id and raw_target_id == selected_id:
            next_graph = [str(current_before.get("id", "") or ""), selected_id]
        session_context = (
            evaluation.get("session_context", {}) if isinstance(evaluation.get("session_context", {}), dict) else {}
        )
        conversation_excerpt = str(evaluation.get("conversation_excerpt", "") or "")
        runtime_event = self._runtime_event_from_context(session_context)
        source = str(runtime_event.get("source", "chat") or "chat")

        if not next_graph:
            next_graph = self._preview_future_graph(current_before)

        pointer_before = int(self.stage_index)
        # 记录在当前节点已经停留了多久，便于后续 history 分析。
        duration_minutes = self.get_state_duration()

        if action == "advance":
            target_stage_id = str(next_graph[1] if len(next_graph) > 1 else "").strip()
            is_terminal = self._coerce_bool(current_before.get("is_terminal_stage", False))
            visited = self._visited_stage_ids()
            existing_candidate_ids = self._candidate_ids_for_stage(
                current_before, self.window_size
            )
            can_use_existing = bool(
                target_stage_id
                and target_stage_id in existing_candidate_ids
                and target_stage_id not in visited
                and not is_terminal
            )
            can_materialize = bool(
                target_stage_id
                and selected_stage
                and selected_id == target_stage_id
                and target_stage_id not in visited
                and target_stage_id != str(current_before.get("id", "") or "")
                and target_stage_id not in self.stage_catalog
                and not is_terminal
            )
            if can_materialize:
                normalized = self._sanitize_stage(selected_stage, source="llm")
                normalized["id"] = target_stage_id[:80]
                normalized["next_candidates"] = []
                self.stage_catalog[target_stage_id] = normalized
                self._apply_branch_candidates(current_before.get("id", ""), [target_stage_id])
            if can_use_existing or can_materialize:
                self.planned_graph = self.planned_graph[: self.stage_index + 1]
                if not self.planned_graph or self.planned_graph[-1] != target_stage_id:
                    self.planned_graph.append(target_stage_id)
                self.stage_index = len(self.planned_graph) - 1
                self.stage_start_time = self._now()
            else:
                # 目标非法：不移动指针，也不写入任何临时候选。
                action = "hold"
                matched = False
        self._ensure_future_window()
        current_after = self.get_current_stage()

        if (
            self.domain_state_enabled
            and source == "chat"
            and not self.pending_domain_window
        ):
            self.pending_domain_window = {
                "started_at": self._now().isoformat(),
                "start_domain_state": self._domain_state_for_window(),
            }

        self.last_session_context = copy.deepcopy(session_context)
        self.last_evaluation = {
            "action": action,
            "matched": matched,
            "match_reason": match_reason,
            "source": source,
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
            match_reason=match_reason,
            pointer_before=pointer_before,
            pointer_after=int(self.stage_index),
            duration_minutes=duration_minutes,
            source=source,
        )
        return self.get_graph_snapshot()

    def initialize_graph_window(
        self,
        session_context: Dict[str, Any],
        conversation_content: str,
        completion_func: Optional[Callable[[str], str]] = None,
        llm_cfg: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """兼容旧调用；evidence-first 模式下启动时不再预生成候选。"""
        del session_context, conversation_content, completion_func, llm_cfg
        return self.get_graph_snapshot()

    def ensure_graph_window(
        self,
        session_context: Dict[str, Any],
        conversation_content: str,
        completion_func: Optional[Callable[[str], str]] = None,
        llm_cfg: Optional[Dict[str, Any]] = None,
        source: str = "graph_window_expansion",
        record_evaluation: bool = False,
    ) -> Dict[str, Any]:
        """兼容旧调用；没有 verified change 时不再补候选窗口。"""
        del (
            session_context,
            conversation_content,
            completion_func,
            llm_cfg,
            source,
            record_evaluation,
        )
        return self.get_graph_snapshot()

    def to_dict(self) -> Dict[str, Any]:
        history_rows: List[Dict[str, Any]] = []
        for row in self.stage_history:
            item = dict(row)
            ts = item.get("timestamp")
            if isinstance(ts, datetime):
                item["timestamp"] = ts.isoformat()
            history_rows.append(item)
        static_stage_ids = self._configured_stage_ids()
        reachable_stage_ids = self._reachable_stage_ids()
        runtime_stages = [
            copy.deepcopy(stage)
            for stage_id, stage in self.stage_catalog.items()
            if stage_id in reachable_stage_ids
            and (
                stage_id not in static_stage_ids
                or str(stage.get("source", "") or "") != "config"
                or self._runtime_candidates_changed(stage_id, stage)
            )
        ]
        return {
            "mode": "complaint_graph",
            "domain_state_enabled": bool(self.domain_state_enabled),
            "initial_stage_id": self.initial_stage_id,
            "root_complaint_anchor": self.root_complaint_anchor,
            "core_belief": self.core_belief,
            "current_stage_id": self.get_current_stage_id(),
            "runtime_stages": runtime_stages,
            "planned_graph": [str(item) for item in self.planned_graph],
            "stage_index": int(self.stage_index),
            "stage_history": history_rows,
            "dialogue_history": copy.deepcopy(self.dialogue_history),
            "last_session_context": copy.deepcopy(self.last_session_context),
            "last_evaluation": copy.deepcopy(self.last_evaluation),
            "domain_state": copy.deepcopy(self.domain_state),
            "pending_domain_window": copy.deepcopy(self.pending_domain_window),
            "last_domain_window_update": copy.deepcopy(self.last_domain_window_update),
            "last_domain_window_id": str(self.last_domain_window_id),
            "stage_start_time": self.stage_start_time.isoformat(),
        }

    @classmethod
    def from_dict(
        cls,
        payload: Dict[str, Any],
        now_provider: Optional[Callable[[], datetime]] = None,
        base_config: Optional[Dict[str, Any]] = None,
        domain_state_enabled: Optional[bool] = None,
    ) -> "ComplaintGraphManager":
        payload = payload if isinstance(payload, dict) else {}
        payload_cfg = payload.get("config", {}) if isinstance(payload.get("config", {}), dict) else {}
        base_cfg = base_config if isinstance(base_config, dict) else {}
        stage_catalog = base_cfg.get("stages", payload_cfg.get("stages", []))
        if isinstance(stage_catalog, dict):
            stage_catalog = list(stage_catalog.values())
        runtime_stages = payload.get("runtime_stages", [])
        if isinstance(runtime_stages, list):
            stage_catalog = list(stage_catalog if isinstance(stage_catalog, list) else [])
            stage_catalog.extend([copy.deepcopy(item) for item in runtime_stages if isinstance(item, dict)])
        initial_stage_id = str(
            payload.get(
                "initial_stage_id",
                payload.get(
                    "current_stage_id",
                    payload_cfg.get("initial_stage_id", base_cfg.get("initial_stage_id", "")),
                ),
            )
            or ""
        )

        def _initial_belief(stages: Any) -> str:
            items = list(stages.values()) if isinstance(stages, dict) else stages
            for item in items if isinstance(items, list) else []:
                if not isinstance(item, dict):
                    continue
                if str(item.get("id", "") or "").strip() != initial_stage_id:
                    continue
                return cls._normalize_core_belief(item.get("core_belief", ""))
            return ""

        checkpoint_core_belief = cls._normalize_core_belief(payload.get("core_belief", ""))
        payload_config_core_belief = cls._normalize_core_belief(
            payload_cfg.get("core_belief", "")
        ) or _initial_belief(payload_cfg.get("stages", []))
        runtime_core_belief = _initial_belief(runtime_stages)
        base_core_belief = cls._normalize_core_belief(
            base_cfg.get("core_belief", "")
        ) or _initial_belief(base_cfg.get("stages", []))
        cfg = {
            "planner": copy.deepcopy(payload_cfg.get("planner", base_cfg.get("planner", {}))),
            # The checkpoint value wins over the current persona file so an
            # existing case keeps the anchor fixed across resume.
            "root_complaint_anchor": str(
                payload.get(
                    "root_complaint_anchor",
                    payload_cfg.get(
                        "root_complaint_anchor",
                        base_cfg.get("root_complaint_anchor", ""),
                    ),
                )
                or ""
            ),
            # New checkpoints persist the case-level value explicitly. Legacy
            # checkpoints fall back to their saved initial stage before the
            # currently installed persona file, so resume does not adopt a
            # dynamically rewritten descendant belief.
            "core_belief": (
                checkpoint_core_belief
                or payload_config_core_belief
                or runtime_core_belief
                or base_core_belief
            ),
            "initial_domain_state": copy.deepcopy(
                payload_cfg.get(
                    "initial_domain_state", base_cfg.get("initial_domain_state", {})
                )
            ),
            "initial_stage_id": initial_stage_id,
            "stages": copy.deepcopy(stage_catalog) if isinstance(stage_catalog, list) else [],
        }
        resolved_domain_state_enabled = (
            bool(payload.get("domain_state_enabled", True))
            if domain_state_enabled is None
            else bool(domain_state_enabled)
        )
        manager = cls(
            config={"complaint_graph": cfg},
            now_provider=now_provider,
            domain_state_enabled=resolved_domain_state_enabled,
        )
        planned_graph = payload.get("planned_graph", [])
        if isinstance(planned_graph, list) and planned_graph:
            manager.planned_graph = manager._normalize_graph_ids(planned_graph)
        else:
            manager.planned_graph = manager._build_default_graph(manager.initial_stage_id)
        manager.stage_index = manager._bounded_int(
            payload.get("stage_index"), 0, 0, max(0, len(manager.planned_graph) - 1)
        )
        manager.stage_start_time = _coerce_datetime(payload.get("stage_start_time"), fallback=manager._now)
        history = payload.get("stage_history", [])
        manager.stage_history = []
        if isinstance(history, list):
            for row in history:
                if not isinstance(row, dict):
                    continue
                item = dict(row)
                item["timestamp"] = _coerce_datetime(item.get("timestamp"), fallback=manager._now)
                manager.stage_history.append(item)
        dialogue_history = payload.get("dialogue_history", [])
        manager.dialogue_history = [dict(item) for item in dialogue_history if isinstance(item, dict)]
        if len(manager.dialogue_history) > manager.max_dialog_history:
            manager.dialogue_history = manager.dialogue_history[-manager.max_dialog_history :]
        manager.last_session_context = copy.deepcopy(payload.get("last_session_context", {})) if isinstance(payload.get("last_session_context", {}), dict) else {}
        manager.last_evaluation = copy.deepcopy(payload.get("last_evaluation", {})) if isinstance(payload.get("last_evaluation", {}), dict) else {}
        manager._restore_domain_state(payload.get("domain_state", {}))
        if manager.domain_state_enabled:
            manager.pending_domain_window = copy.deepcopy(payload.get("pending_domain_window", {})) if isinstance(payload.get("pending_domain_window", {}), dict) else {}
            manager.last_domain_window_update = copy.deepcopy(payload.get("last_domain_window_update", {})) if isinstance(payload.get("last_domain_window_update", {}), dict) else {}
            manager.last_domain_window_id = str(payload.get("last_domain_window_id", "") or "")[:160]
        manager._link_graph_edges(manager.planned_graph)
        manager._ensure_future_window()
        return manager

    def _configured_stage_ids(self) -> set:
        """返回静态配置里声明过的节点 ID。"""
        raw_stages = self._raw_graph_config.get("stages", self._raw_graph_config.get("stage_catalog", []))
        if isinstance(raw_stages, dict):
            raw_stages = list(raw_stages.values())
        stage_ids = set()
        if isinstance(raw_stages, list):
            for item in raw_stages:
                if not isinstance(item, dict):
                    continue
                stage_id = str(item.get("id", "") or "").strip()
                if stage_id:
                    stage_ids.add(stage_id)
        return stage_ids

    def _runtime_candidates_changed(self, stage_id: str, stage: Dict[str, Any]) -> bool:
        """判断静态节点的候选分支是否被运行态规划改写。"""
        configured = self._configured_next_candidates(stage_id)
        current = [
            str(item or "").strip()
            for item in self._to_list(stage.get("next_candidates", []) if isinstance(stage, dict) else [])
            if str(item or "").strip()
        ]
        return current != configured

    def _configured_next_candidates(self, stage_id: str) -> List[str]:
        """读取静态配置中某个节点原始的 next_candidates。"""
        target = str(stage_id or "").strip()
        raw_stages = self._raw_graph_config.get("stages", self._raw_graph_config.get("stage_catalog", []))
        if isinstance(raw_stages, dict):
            raw_stages = list(raw_stages.values())
        for item in raw_stages if isinstance(raw_stages, list) else []:
            if not isinstance(item, dict) or str(item.get("id", "") or "").strip() != target:
                continue
            return [
                str(candidate or "").strip()
                for candidate in self._to_list(item.get("next_candidates", []))
                if str(candidate or "").strip()
            ]
        return []

    def _preview_future_graph(self, current_stage: Dict[str, Any]) -> List[str]:
        """返回“当前节点 + 直接候选分支”的窗口。"""
        current_id = str(current_stage.get("id", "") or "").strip()
        if not current_id:
            return []
        return self._normalize_graph_path(
            [current_id] + self._candidate_ids_for_stage(current_stage, self.window_size),
            current_id,
        )

    def _replace_future_graph(self, next_graph: List[str]) -> None:
        """用传入窗口替换当前节点的直接候选分支。"""
        normalized = self._normalize_graph_path(next_graph, self.get_current_stage_id())
        if normalized:
            self._apply_branch_candidates(normalized[0], normalized[1:])

    def _ensure_future_window(self) -> None:
        """规整当前路径指针，不再自动臆造未来节点。"""
        if not self.planned_graph:
            self.planned_graph = self._build_default_graph(self.initial_stage_id)
            self.stage_index = 0
            return
        self.planned_graph = self._normalize_graph_ids(self.planned_graph)
        self.stage_index = min(max(0, int(self.stage_index)), max(0, len(self.planned_graph) - 1))

    def _build_default_graph(self, initial_stage_id: str) -> List[str]:
        """构造只包含起始节点的已访问路径。"""
        start_id = str(initial_stage_id or "").strip()
        if not start_id or start_id not in self.stage_catalog:
            start_id = next(iter(self.stage_catalog.keys()))
        return self._normalize_graph_ids([start_id])

    def _visited_stage_ids(self) -> set:
        """返回当前已走过的路径节点，用于禁止候选分支回指旧节点。"""
        if not isinstance(self.planned_graph, list) or not self.planned_graph:
            return set()
        max_idx = min(max(0, int(self.stage_index)), len(self.planned_graph) - 1)
        visited = set()
        for item in self.planned_graph[: max_idx + 1]:
            stage_id = str(item or "").strip()
            if stage_id and stage_id in self.stage_catalog:
                visited.add(stage_id)
        return visited

    def _candidate_ids_for_stage(self, stage: Dict[str, Any], count: int) -> List[str]:
        """清洗并截断单个节点的 next_candidates。"""
        if not isinstance(stage, dict):
            return []
        stage_id = str(stage.get("id", "") or "").strip()
        limit = max(0, int(count))
        results: List[str] = []
        seen = {stage_id}
        visited = self._visited_stage_ids()
        for candidate in self._to_list(stage.get("next_candidates", [])):
            candidate_id = str(candidate or "").strip()
            if (
                not candidate_id
                or candidate_id in seen
                or candidate_id in visited
                or candidate_id not in self.stage_catalog
            ):
                continue
            seen.add(candidate_id)
            results.append(candidate_id)
            if len(results) >= limit:
                break
        return results

    def _apply_branch_candidates(self, parent_id: str, child_ids: List[str]) -> None:
        """把若干子分支写回父节点的 next_candidates。"""
        parent_key = str(parent_id or "").strip()
        if parent_key not in self.stage_catalog:
            return
        normalized: List[str] = []
        seen = {parent_key}
        visited = self._visited_stage_ids()
        for child_id in child_ids:
            key = str(child_id or "").strip()
            if not key or key in seen or key in visited or key not in self.stage_catalog:
                continue
            seen.add(key)
            normalized.append(key[:80])
            if len(normalized) >= self.window_size:
                break
        self.stage_catalog[parent_key]["next_candidates"] = normalized

    def _ensure_branch_candidates(self, parent_id: str, child_ids: List[str]) -> None:
        """把目标分支补进父节点候选中，保留原有兄弟分支。"""
        parent_key = str(parent_id or "").strip()
        if parent_key not in self.stage_catalog:
            return
        existing = self._candidate_ids_for_stage(self.stage_catalog[parent_key], self.window_size)
        merged = existing + [str(item or "").strip() for item in child_ids]
        self._apply_branch_candidates(parent_key, merged)

    def _reachable_stage_ids(self) -> set:
        """返回已访问路径和候选分支能到达的节点 ID。"""
        roots = self._normalize_graph_ids(self.planned_graph or [self.initial_stage_id])
        reachable = set()
        stack = list(roots)
        while stack:
            stage_id = str(stack.pop() or "").strip()
            if not stage_id or stage_id in reachable or stage_id not in self.stage_catalog:
                continue
            reachable.add(stage_id)
            stage = self.stage_catalog.get(stage_id, {})
            for candidate_id in self._candidate_ids_for_stage(stage, 8):
                if candidate_id not in reachable:
                    stack.append(candidate_id)
        return reachable

    def _link_graph_edges(self, graph_ids: List[str]) -> None:
        ids = self._normalize_graph_ids(graph_ids)
        for idx in range(len(ids) - 1):
            from_id = str(ids[idx] or "").strip()
            to_id = str(ids[idx + 1] or "").strip()
            if not from_id or not to_id or from_id == to_id:
                continue
            if from_id not in self.stage_catalog or to_id not in self.stage_catalog:
                continue
            stage = self.stage_catalog[from_id]
            next_candidates = stage.get("next_candidates", [])
            if not isinstance(next_candidates, list):
                next_candidates = []
            normalized_candidates: List[str] = []
            seen = set()
            for item in next_candidates + [to_id]:
                candidate_id = str(item or "").strip()
                if (
                    not candidate_id
                    or candidate_id == from_id
                    or candidate_id in seen
                    or candidate_id not in self.stage_catalog
                ):
                    continue
                seen.add(candidate_id)
                normalized_candidates.append(candidate_id[:80])
                if len(normalized_candidates) >= 8:
                    break
            stage["next_candidates"] = normalized_candidates
        self._prune_unknown_next_candidates(ids)

    def _prune_unknown_next_candidates(self, stage_ids: Optional[List[str]] = None) -> None:
        ids = self._normalize_graph_ids(stage_ids) if stage_ids else list(self.stage_catalog.keys())
        for stage_id in ids:
            stage = self.stage_catalog.get(stage_id)
            if not isinstance(stage, dict):
                continue
            next_candidates = stage.get("next_candidates", [])
            if not isinstance(next_candidates, list):
                stage["next_candidates"] = []
                continue
            normalized_candidates: List[str] = []
            seen = set()
            for item in next_candidates:
                candidate_id = str(item or "").strip()
                if (
                    not candidate_id
                    or candidate_id == stage_id
                    or candidate_id in seen
                    or candidate_id not in self.stage_catalog
                ):
                    continue
                seen.add(candidate_id)
                normalized_candidates.append(candidate_id[:80])
                if len(normalized_candidates) >= 8:
                    break
            stage["next_candidates"] = normalized_candidates

    def _infer_transition_signal(
        self,
        completion_func: Callable[[str], str],
        current_stage: Dict[str, Any],
        session_context: Dict[str, Any],
        conversation_content: str,
        llm_cfg: Optional[Dict[str, Any]],
        counterpart_utterance: str = "",
    ) -> Optional[Dict[str, Any]]:
        """先盲判本轮新增变化，再将已确认变化匹配到候选节点。"""
        change_prompt = self._build_transition_change_prompt(
            current_stage=current_stage,
            session_context=session_context,
            conversation_content=conversation_content,
            counterpart_utterance=counterpart_utterance,
            llm_cfg=llm_cfg,
        )
        try:
            raw_change = str(completion_func(change_prompt) or "")
        except Exception:
            raw_change = ""
        runtime_event = self._runtime_event_from_context(session_context)
        source = str(runtime_event.get("source", "chat") or "chat").strip()
        reflection_evidence = self._reflection_evidence_for_prompt(runtime_event)
        detected_change = self._normalize_transition_change(
            payload=self._parse_json_object(raw_change),
            source=source,
            conversation_content=conversation_content,
            reflection_evidence=reflection_evidence,
        )
        if not detected_change.get("has_new_change", False):
            return {
                "matched": False,
                "match_reason": str(
                    detected_change.get("reason", "no_new_patient_change")
                    or "no_new_patient_change"
                )[:180],
                "action": "hold",
                "next_graph": [str(current_stage.get("id", "") or "")],
            }

        temporary_seeds = self._infer_branch_plan(
            completion_func=completion_func,
            parent_stage=current_stage,
            session_context=session_context,
            conversation_content=conversation_content,
            llm_cfg=llm_cfg,
            verified_change=detected_change,
        )
        if not temporary_seeds:
            return {
                "matched": False,
                "match_reason": "verified_change_without_candidate",
                "action": "hold",
                "next_graph": self._preview_future_graph(current_stage),
            }

        temporary_candidates: List[Dict[str, Any]] = []
        for seed in temporary_seeds:
            detail_payload = self._call_graph_planner(
                completion_func=completion_func,
                mode="branch_detail",
                parent_stage=current_stage,
                session_context=session_context,
                conversation_content=conversation_content,
                llm_cfg=llm_cfg,
                candidate_seed=seed,
                verified_change=detected_change,
            )
            temporary_candidates.append(
                self._normalize_branch_detail(
                    detail_payload,
                    seed,
                    parent_stage=current_stage,
                )
            )

        match_prompt = self._build_transition_prompt(
            current_stage=current_stage,
            session_context=session_context,
            conversation_content=conversation_content,
            counterpart_utterance=counterpart_utterance,
            llm_cfg=llm_cfg,
            detected_change=detected_change,
            candidate_stages=temporary_candidates,
        )
        try:
            raw_match = str(completion_func(match_prompt) or "")
        except Exception:
            raw_match = ""
        parsed_match = self._parse_json_object(raw_match)
        signal = self._normalize_transition_signal(
            parsed_match,
            current_stage,
            candidate_ids=[
                str(stage.get("id", "") or "").strip()
                for stage in temporary_candidates
                if isinstance(stage, dict)
            ],
        )
        if not signal:
            return {
                "matched": False,
                "match_reason": "no_transition_decision",
                "action": "hold",
                "next_graph": self._preview_future_graph(current_stage),
            }
        if str(signal.get("action", "hold") or "hold") != "advance":
            return signal
        target_id = str(
            (signal.get("next_graph", []) or ["", ""])[1]
            if len(signal.get("next_graph", []) or []) > 1
            else ""
        ).strip()
        selected_stage = next(
            (
                copy.deepcopy(stage)
                for stage in temporary_candidates
                if str(stage.get("id", "") or "").strip() == target_id
            ),
            None,
        )
        if not selected_stage:
            return {
                "matched": False,
                "match_reason": "selected_candidate_missing",
                "action": "hold",
                "next_graph": self._preview_future_graph(current_stage),
            }
        signal["selected_stage"] = selected_stage
        return signal

    def _build_transition_change_prompt(
        self,
        current_stage: Dict[str, Any],
        session_context: Dict[str, Any],
        conversation_content: str,
        llm_cfg: Optional[Dict[str, Any]],
        counterpart_utterance: str = "",
    ) -> str:
        """构造候选不可见的本轮新增状态变化提取 prompt。"""
        cfg = llm_cfg if isinstance(llm_cfg, dict) else {}
        text_limit = self._bounded_int(cfg.get("max_text_length"), 1200, 200, 6000)
        runtime_event = self._runtime_event_from_context(session_context)
        source = str(runtime_event.get("source", "chat") or "chat").strip()
        payload: Dict[str, Any] = {
            "task": "transition_change_detection",
            "source": source,
            "current_stage": self._transition_stage_view(current_stage, current=True),
        }
        if source == "reflection":
            payload["reflection_evidence"] = self._reflection_evidence_for_prompt(runtime_event)
        else:
            scene = (
                session_context.get("scene", {})
                if isinstance(session_context.get("scene", {}), dict)
                else {}
            )
            participants = (
                session_context.get("participants", {})
                if isinstance(session_context.get("participants", {}), dict)
                else {}
            )
            payload.update(
                {
                    "patient_evidence": self._clip_text(conversation_content, limit=text_limit),
                    "counterpart_context": self._clip_text(counterpart_utterance, limit=text_limit),
                    "interaction_type": str(scene.get("interaction_type", "") or "").strip(),
                    "relationship": str(participants.get("relationship", "") or "").strip(),
                }
            )
        return render_prompt(
            "depression/graph_transition_change",
            {"payload_json": json.dumps(payload, ensure_ascii=False)},
        )

    def _build_transition_prompt(
        self,
        current_stage: Dict[str, Any],
        session_context: Dict[str, Any],
        conversation_content: str,
        llm_cfg: Optional[Dict[str, Any]],
        counterpart_utterance: str = "",
        detected_change: Optional[Dict[str, Any]] = None,
        candidate_stages: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """构造已确认变化到候选节点的匹配 prompt。"""
        del conversation_content, counterpart_utterance, llm_cfg
        candidates = [
            copy.deepcopy(item)
            for item in self._to_list(candidate_stages)
            if isinstance(item, dict)
        ][: self.TRANSITION_CANDIDATE_COUNT]
        if candidate_stages is None:
            candidate_ids = self._candidate_ids_for_stage(current_stage, self.window_size)
            candidates = [copy.deepcopy(self.stage_catalog[item]) for item in candidate_ids]
        runtime_event = self._runtime_event_from_context(session_context)
        payload = {
            "task": "transition_decision",
            "source": str(runtime_event.get("source", "chat") or "chat").strip(),
            "current_stage": self._transition_stage_view(current_stage, current=True),
            "candidate_stages": [
                self._transition_stage_view(item, current=False) for item in candidates
            ],
            "detected_change": {
                "change": self._clip_text(
                    (detected_change or {}).get("change", ""), limit=240
                ),
                "evidence_quotes": [
                    self._clip_text(item, limit=240)
                    for item in self._to_list(
                        (detected_change or {}).get("evidence_quotes", [])
                    )[:6]
                    if self._clip_text(item, limit=240)
                ],
            },
        }
        return render_prompt(
            "depression/graph_transition",
            {"payload_json": json.dumps(payload, ensure_ascii=False)},
        )

    def _normalize_transition_change(
        self,
        payload: Any,
        source: str,
        conversation_content: str,
        reflection_evidence: Dict[str, Any],
    ) -> Dict[str, Any]:
        """规范化核心变化；evidence 仅作为可选审计信息。"""
        if not isinstance(payload, dict):
            return {
                "has_new_change": False,
                "change": "",
                "evidence_quotes": [],
                "reason": "no_change_detection_result",
            }
        has_new_change = self._coerce_bool(payload.get("has_new_change", False))
        change = self._clip_text(payload.get("change", ""), limit=240)
        reason = self._clip_text(payload.get("reason", ""), limit=180)
        normalized_source = str(source or "chat").strip()
        if normalized_source == "reflection":
            evidence_texts = [
                self._clip_text(item, limit=360)
                for key in ("patient_key_utterances", "behavior_or_life_events")
                for item in self._to_list(reflection_evidence.get(key, []))
                if self._clip_text(item, limit=360)
            ]
        else:
            evidence_texts = [self._clip_text(conversation_content, limit=6000)]

        valid_quotes: List[str] = []
        seen = set()
        had_consumed_quote = False
        had_unresolved_quote = False
        for item in self._to_list(payload.get("evidence_quotes", [])):
            quote = self._resolve_evidence_quote(item, evidence_texts)
            if not quote:
                had_unresolved_quote = True
                continue
            if quote in seen:
                continue
            if self._advance_evidence_was_consumed(quote):
                had_consumed_quote = True
            seen.add(quote)
            valid_quotes.append(quote)
            if len(valid_quotes) >= 6:
                break

        if not has_new_change or not change:
            return {
                "has_new_change": False,
                "change": "",
                "evidence_quotes": [],
                "reason": reason or "no_new_patient_change",
            }
        evidence_warnings: List[str] = []
        if not valid_quotes:
            evidence_warnings.append("missing_or_unverified_evidence")
        elif had_unresolved_quote:
            evidence_warnings.append("partially_unverified_evidence")
        if had_consumed_quote:
            evidence_warnings.append("reused_evidence")
        if normalized_source == "reflection" and len(valid_quotes) < 2:
            evidence_warnings.append("reflection_evidence_count_below_hint")
        if evidence_warnings:
            warning_text = "evidence_warning: " + ",".join(evidence_warnings)
            reason = "{}; {}".format(reason, warning_text) if reason else warning_text
        return {
            "has_new_change": True,
            "change": change,
            "evidence_quotes": valid_quotes,
            "reason": reason,
        }

    def _normalize_domain_evidence(
        self, raw: Any, evidence_texts: List[str]
    ) -> List[Dict[str, str]]:
        configured_domains = self.domain_state.get("domains", {})
        if not isinstance(configured_domains, dict):
            return []
        grouped: Dict[str, List[Dict[str, str]]] = {}
        for item in self._to_list(raw):
            if not isinstance(item, dict):
                continue
            domain = str(item.get("domain", "") or "").strip()
            direction = str(item.get("direction_hint", "") or "").strip().lower()
            quote = self._resolve_evidence_quote(
                item.get("evidence_quote", ""), evidence_texts
            )
            evidence_type = str(item.get("evidence_type", "symptom_report") or "symptom_report").strip().lower()
            if domain not in self.DOMAIN_KEYS or domain not in configured_domains:
                continue
            if direction not in self.DOMAIN_EVIDENCE_DIRECTIONS:
                continue
            if quote and not any(quote in evidence for evidence in evidence_texts if evidence):
                continue
            candidate = {
                "domain": domain,
                "direction_hint": direction,
                "evidence_quote": quote,
                "evidence_type": evidence_type[:32],
            }
            if candidate not in grouped.setdefault(domain, []):
                grouped[domain].append(candidate)

        normalized: List[Dict[str, str]] = []
        for domain in self.DOMAIN_KEYS:
            candidates = grouped.get(domain, [])
            if not candidates:
                continue
            directions = {item["direction_hint"] for item in candidates}
            if len(directions) == 1:
                normalized.append(candidates[0])
            else:
                candidate = copy.deepcopy(candidates[0])
                candidate["direction_hint"] = "unclear"
                normalized.append(candidate)
            if len(normalized) >= 3:
                break
        return normalized

    def _normalize_domain_changes(
        self, raw: Any, evidence_texts: List[str]
    ) -> List[Dict[str, str]]:
        """Compatibility adapter for old callers; detector runtime no longer uses it."""
        candidates = []
        for item in self._to_list(raw):
            if not isinstance(item, dict):
                continue
            direction = str(item.get("direction", "") or "").strip().lower()
            candidates.append(
                {
                    "domain": item.get("domain", ""),
                    "direction_hint": "stable" if direction == "unchanged" else direction,
                    "evidence_quote": item.get("evidence_quote", ""),
                    "evidence_type": "symptom_report",
                }
            )
        return self._normalize_domain_evidence(candidates, evidence_texts)

    def _apply_domain_changes(self, raw: Any, source: str) -> List[Dict[str, Any]]:
        """Legacy reducer retained for API compatibility; runtime uses window updates."""
        domains = self.domain_state.get("domains", {})
        if not isinstance(domains, dict):
            return []
        applied: List[Dict[str, Any]] = []
        observed_at = self._now().isoformat()
        for change in self._to_list(raw)[:3]:
            if not isinstance(change, dict):
                continue
            domain = str(change.get("domain", "") or "").strip()
            direction = str(change.get("direction", "") or "").strip().lower()
            quote = self._clip_text(change.get("evidence_quote", ""), limit=240)
            state = domains.get(domain)
            if (
                domain not in self.DOMAIN_KEYS
                or direction not in self.DOMAIN_DIRECTIONS
                or not isinstance(state, dict)
            ):
                continue
            evidence = [
                copy.deepcopy(item)
                for item in self._to_list(state.get("evidence", []))
                if isinstance(item, dict)
            ][-self.DOMAIN_EVIDENCE_LIMIT :]
            if any(
                str(item.get("direction", "") or "") == direction
                and str(item.get("quote", "") or "") == quote
                for item in evidence
            ):
                continue
            old_value = state.get("value")
            if isinstance(old_value, bool) or not isinstance(old_value, int):
                continue
            delta = {
                "improved": self.DOMAIN_CHANGE_STEP,
                "worsened": -self.DOMAIN_CHANGE_STEP,
                "unchanged": 0,
            }[direction]
            new_value = min(100, max(0, int(old_value) + delta))
            evidence.append(
                {
                    "direction": direction,
                    "quote": quote,
                    "source": str(source or "chat")[:32],
                    "observed_at": observed_at,
                }
            )
            state["value"] = new_value
            state["last_change"] = direction
            state["last_observed_at"] = observed_at
            state["evidence"] = evidence[-self.DOMAIN_EVIDENCE_LIMIT :]
            applied.append(
                {
                    "domain": domain,
                    "direction": direction,
                    "evidence_quote": quote,
                    "old_value": int(old_value),
                    "new_value": int(new_value),
                }
            )
        return applied

    def _collect_domain_evidence(self, raw: Any, source: str) -> List[Dict[str, str]]:
        """Collect detector hints without mutating any domain value."""
        if not self.domain_state_enabled:
            return []
        if not self.pending_domain_window:
            self.pending_domain_window = {
                "started_at": self._now().isoformat(),
                "start_domain_state": self._domain_state_for_window(),
                "candidates": [],
            }
        candidates = self.pending_domain_window.setdefault("candidates", [])
        if not isinstance(candidates, list):
            candidates = []
            self.pending_domain_window["candidates"] = candidates
        collected: List[Dict[str, str]] = []
        for item in self._to_list(raw):
            if not isinstance(item, dict):
                continue
            domain = str(item.get("domain", "") or "").strip()
            direction = str(item.get("direction_hint", "") or "").strip().lower()
            quote = self._clip_text(item.get("evidence_quote", ""), limit=240)
            evidence_type = str(item.get("evidence_type", "symptom_report") or "symptom_report").strip().lower()[:32]
            if (
                domain not in self.DOMAIN_KEYS
                or direction not in self.DOMAIN_EVIDENCE_DIRECTIONS
            ):
                continue
            candidate = {
                "domain": domain,
                "direction_hint": direction,
                "evidence_quote": quote,
                "evidence_type": evidence_type,
                "source": str(source or "chat")[:32],
            }
            if candidate not in candidates:
                candidates.append(candidate)
                collected.append(copy.deepcopy(candidate))
        self.pending_domain_window["candidates"] = candidates[-self.DOMAIN_WINDOW_CANDIDATE_LIMIT :]
        return collected

    def finalize_domain_window(
        self,
        patient_dialogue: str,
        completion_func: Optional[Callable[[str], str]],
        window_id: str = "",
        window_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Adjudicate all seven domains once at a completed conversation boundary."""
        if not self.domain_state_enabled:
            return {"enabled": False, "domain_updates": [], "applied_changes": []}
        dialogue = str(patient_dialogue or "").strip()
        resolved_id = str(window_id or "").strip()[:160]
        if not resolved_id:
            digest = hashlib.sha1(dialogue.encode("utf-8")).hexdigest()[:16]
            resolved_id = "{}|{}".format(self._now().isoformat(), digest)
        if resolved_id == self.last_domain_window_id:
            return copy.deepcopy(self.last_domain_window_update)

        pending = copy.deepcopy(self.pending_domain_window)
        if not pending:
            pending = {
                "started_at": str(
                    (window_context or {}).get("started_at", "")
                    if isinstance(window_context, dict)
                    else ""
                )
                or self._now().isoformat(),
                "start_domain_state": self._domain_state_for_window(),
            }
        start_state = self._normalize_window_start_domain_state(
            pending.get("start_domain_state", {})
        )
        context = window_context if isinstance(window_context, dict) else {}
        completed_at = self._now().isoformat()
        metadata = {
            "window_id": resolved_id,
            "started_at": str(
                context.get("started_at", pending.get("started_at", "")) or ""
            ),
            "completed_at": completed_at,
            "participants": copy.deepcopy(context.get("participants", [])),
            "patient": str(context.get("patient", "") or ""),
            "interaction_type": str(context.get("interaction_type", "") or ""),
            "location": str(context.get("location", "") or ""),
            "time_of_day": str(context.get("time_of_day", "") or ""),
        }
        payload = {
            "task": "domain_window_update",
            "window_metadata": metadata,
            "window_start_domain_state": start_state,
            "session_transcript": dialogue,
        }
        prompt = render_prompt(
            "depression/domain_window_update",
            {"payload_json": json.dumps(payload, ensure_ascii=False)},
        )
        raw_response = ""
        if callable(completion_func):
            try:
                raw_response = str(completion_func(prompt) or "")
            except Exception:
                raw_response = ""
        normalized = self._normalize_domain_window_updates(
            self._parse_json_object(raw_response),
            dialogue,
            patient_name=metadata["patient"],
        )
        applied = self._apply_domain_window_updates(normalized, resolved_id)
        validation_trace = [
            {"domain": item["domain"], "reason": item["reason"]}
            for item in normalized
            if str(item.get("reason", "") or "").startswith("validation_failed:")
        ]
        result = {
            "window_id": resolved_id,
            "started_at": str(pending.get("started_at", "") or ""),
            "completed_at": completed_at,
            "domain_updates": normalized,
            "applied_changes": applied,
            "validation_trace": validation_trace,
            "llm_success": bool(raw_response.strip()),
        }
        self.last_domain_window_id = resolved_id
        self.last_domain_window_update = copy.deepcopy(result)
        self.pending_domain_window = {}
        return result

    def _normalize_domain_window_updates(
        self, payload: Any, session_transcript: str, patient_name: str = ""
    ) -> List[Dict[str, Any]]:
        raw_updates = payload.get("domain_updates", []) if isinstance(payload, dict) else []
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        patient_text = self._patient_text_from_transcript(
            session_transcript, patient_name=patient_name
        )
        for item in self._to_list(raw_updates):
            if not isinstance(item, dict):
                continue
            domain = str(item.get("domain", "") or "").strip()
            update_value = item.get("update", "no_update")
            update = (
                str(update_value or "no_update").strip()
                if isinstance(update_value, str)
                else ""
            )
            if domain not in self.DOMAIN_KEYS or update not in self.DOMAIN_UPDATE_VALUES:
                continue
            anchor = str(item.get("comparison_anchor", "") or "").strip().lower()
            quotes: List[str] = []
            seen = set()
            for raw_quote in self._to_list(item.get("evidence_quotes", [])):
                quote = self._resolve_evidence_quote(raw_quote, [patient_text])
                if not quote or quote in seen:
                    continue
                seen.add(quote)
                quotes.append(quote)
                if len(quotes) >= 4:
                    break
            reason = self._clip_text(item.get("reason", ""), limit=240)
            validation_error = ""
            if update not in self.DOMAIN_ANCHOR_UPDATES.get(anchor, set()):
                validation_error = "comparison_anchor_update_mismatch"
            elif update != "no_update" and not quotes:
                warning_text = "evidence_warning: missing_or_unverified_patient_evidence"
                reason = "{}; {}".format(reason, warning_text) if reason else warning_text
            if validation_error:
                anchor = "insufficient"
                update = "no_update"
                reason = self._clip_text(
                    "validation_failed: {}; {}".format(validation_error, reason),
                    limit=240,
                )
            grouped.setdefault(domain, []).append(
                {
                    "domain": domain,
                    "comparison_anchor": anchor,
                    "update": update,
                    "evidence_quotes": quotes,
                    "reason": reason,
                }
            )

        normalized: List[Dict[str, Any]] = []
        for domain in self.DOMAIN_KEYS:
            items = grouped.get(domain, [])
            if len(items) == 1:
                normalized.append(items[0])
            else:
                normalized.append(
                    {
                        "domain": domain,
                        "comparison_anchor": "insufficient",
                        "update": "no_update",
                        "evidence_quotes": [],
                        "reason": "validation_failed: missing_or_conflicting_window_decision",
                    }
                )
        return normalized

    def _normalize_window_start_domain_state(self, raw: Any) -> Dict[str, Any]:
        """Upgrade legacy numeric window snapshots to the new prompt-only view."""
        payload = raw if isinstance(raw, dict) else {}
        raw_domains = payload.get("domains", {})
        if not isinstance(raw_domains, dict):
            return self._domain_state_for_window()
        current_domains = self.domain_state.get("domains", {})
        if not isinstance(current_domains, dict):
            current_domains = {}
        domains: Dict[str, Dict[str, Any]] = {}
        for domain in self.DOMAIN_KEYS:
            item = raw_domains.get(domain)
            value = item.get("value") if isinstance(item, dict) else item
            if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 100:
                current = current_domains.get(domain, {})
                value = current.get("value") if isinstance(current, dict) else None
            if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 100:
                continue
            domains[domain] = {
                "value": int(value),
                "current_description": DynamicPromptBuilder.render_domain_description(
                    domain, int(value)
                ),
            }
        return {
            "scale": {
                "min": 0,
                "max": 100,
                "higher_is_better": True,
                "allowed_updates": ["no_update", "-10", "-5", "0", "+5", "+10"],
            },
            "domains": domains,
        }

    @staticmethod
    def _patient_text_from_transcript(transcript: str, patient_name: str = "") -> str:
        """Extract patient turns while retaining unlabeled continuation lines."""
        text = str(transcript or "")
        expected = str(patient_name or "").strip()
        patient_aliases = {"患者", "patient"}
        if expected:
            patient_aliases.add(expected)
        patient_lines: List[str] = []
        saw_speaker_label = False
        current_is_patient = False
        for raw_line in text.splitlines():
            match = re.match(r"^\s*([^:：\n]{1,80})\s*[:：]\s*(.*)$", raw_line)
            if match:
                saw_speaker_label = True
                speaker = match.group(1).strip()
                if expected:
                    current_is_patient = speaker == expected
                else:
                    current_is_patient = (
                        speaker.lower() in patient_aliases
                        or speaker.startswith("卡布达")
                    )
                if current_is_patient and match.group(2).strip():
                    patient_lines.append(match.group(2).strip())
            elif current_is_patient and raw_line.strip():
                patient_lines.append(raw_line.strip())
        if saw_speaker_label:
            return "\n".join(patient_lines)
        # Backward compatibility for direct callers that passed patient-only text.
        return text.strip()

    def _apply_domain_window_updates(
        self, updates: List[Dict[str, Any]], window_id: str
    ) -> List[Dict[str, Any]]:
        domains = self.domain_state.get("domains", {})
        if not isinstance(domains, dict):
            return []
        observed_at = self._now().isoformat()
        applied: List[Dict[str, Any]] = []
        for decision in updates:
            domain = str(decision.get("domain", "") or "")
            update = str(decision.get("update", "no_update") or "no_update")
            state = domains.get(domain)
            if update == "no_update" or not isinstance(state, dict):
                continue
            old_value = state.get("value")
            if isinstance(old_value, bool) or not isinstance(old_value, int):
                continue
            delta = int(update)
            new_value = min(100, max(0, int(old_value) + delta))
            direction = "improved" if delta > 0 else "worsened" if delta < 0 else "unchanged"
            quote = "；".join(
                self._clip_text(item, limit=120)
                for item in self._to_list(decision.get("evidence_quotes", []))[:2]
                if self._clip_text(item, limit=120)
            )
            evidence = [
                copy.deepcopy(item)
                for item in self._to_list(state.get("evidence", []))
                if isinstance(item, dict)
            ][-self.DOMAIN_EVIDENCE_LIMIT :]
            evidence.append(
                {
                    "direction": direction,
                    "quote": quote,
                    "source": "domain_window",
                    "observed_at": observed_at,
                    "window_id": str(window_id)[:160],
                    "update": update,
                }
            )
            state["value"] = new_value
            state["last_change"] = direction
            state["last_observed_at"] = observed_at
            state["evidence"] = evidence[-self.DOMAIN_EVIDENCE_LIMIT :]
            applied.append(
                {
                    "domain": domain,
                    "update": update,
                    "direction": direction,
                    "old_value": int(old_value),
                    "new_value": int(new_value),
                    "effective": new_value != old_value or delta == 0,
                    "saturated_noop": new_value == old_value and delta != 0,
                    "evidence_quotes": copy.deepcopy(decision.get("evidence_quotes", [])),
                }
            )
        return applied

    def _advance_evidence_was_consumed(self, evidence_quote: str) -> bool:
        """检测重复证据，仅用于审计告警，不再阻断推进。"""
        quote = str(evidence_quote or "").strip()
        if not quote:
            return True
        for row in reversed(self.dialogue_history):
            if not isinstance(row, dict) or str(row.get("action", "") or "") != "advance":
                continue
            consumed_text = str(row.get("conversation_excerpt", "") or "").strip()
            if consumed_text and self._resolve_evidence_quote(quote, [consumed_text]):
                return True
        return False

    def _reflection_evidence_for_prompt(self, runtime_event: Dict[str, Any]) -> Dict[str, Any]:
        """提取反思判定可用的原始证据，不透传反思结论或运行态元数据。"""
        metadata = (
            runtime_event.get("metadata", {})
            if isinstance(runtime_event.get("metadata", {}), dict)
            else {}
        )
        evidence = (
            metadata.get("reflection_evidence", {})
            if isinstance(metadata.get("reflection_evidence", {}), dict)
            else {}
        )

        def _texts(key: str, limit: int) -> List[str]:
            results: List[str] = []
            seen = set()
            for item in self._to_list(evidence.get(key, [])):
                text = self._clip_text(item, limit=360)
                if not text or text in seen:
                    continue
                seen.add(text)
                results.append(text)
                if len(results) >= limit:
                    break
            return results

        return {
            "interaction_background": self._clip_text(
                evidence.get("interaction_background", ""), limit=120
            ),
            "patient_key_utterances": _texts("patient_key_utterances", 6),
            "behavior_or_life_events": _texts("behavior_or_life_events", 5),
        }

    @staticmethod
    def _transition_stage_view(stage: Dict[str, Any], current: bool) -> Dict[str, Any]:
        """仅保留推进判定直接需要的节点语义。"""
        stage = stage if isinstance(stage, dict) else {}
        payload = {
            "id": str(stage.get("id", "") or ""),
            "label": str(stage.get("label", "") or ""),
            "summary": str(stage.get("summary", "") or ""),
            "core_belief": str(stage.get("core_belief", "") or ""),
            "narrative_focus": copy.deepcopy(
                stage.get("narrative_focus", [])
                if isinstance(stage.get("narrative_focus", []), list)
                else []
            ),
        }
        if current:
            payload["hold_signals"] = copy.deepcopy(
                stage.get("hold_signals", [])
                if isinstance(stage.get("hold_signals", []), list)
                else []
            )
            payload["is_terminal_stage"] = bool(stage.get("is_terminal_stage", False))
        return payload

    @staticmethod
    def _transition_seed_view(stage: Dict[str, Any]) -> Dict[str, Any]:
        """兼容旧调用：返回 transition 所需的完整候选语义。"""
        payload = stage if isinstance(stage, dict) else {}
        return {
            "id": str(payload.get("id", "") or "")[:80],
            "label": str(payload.get("label", "") or "")[:80],
            "summary": str(payload.get("summary", "") or "")[:220],
            "core_belief": str(payload.get("core_belief", "") or "")[:160],
            "narrative_focus": [
                str(item or "")[:80]
                for item in payload.get("narrative_focus", [])[:6]
                if str(item or "").strip()
            ]
            if isinstance(payload.get("narrative_focus", []), list)
            else [],
        }

    def _normalize_transition_signal(
        self,
        payload: Any,
        current_stage: Dict[str, Any],
        candidate_ids: Optional[List[str]] = None,
    ) -> Optional[Dict[str, Any]]:
        """清洗 LLM 的推进判定结果。"""
        if not isinstance(payload, dict):
            return None
        current_id = str(current_stage.get("id", "") or "").strip()
        allowed_candidate_ids = (
            [
                str(item or "").strip()
                for item in candidate_ids
                if str(item or "").strip()
            ][: self.TRANSITION_CANDIDATE_COUNT]
            if candidate_ids is not None
            else self._candidate_ids_for_stage(current_stage, self.window_size)
        )
        action = str(payload.get("action", "hold") or "hold").strip().lower()
        if action not in {"hold", "advance"}:
            action = "hold"
        matched_current_stage = self._coerce_bool(
            payload.get("matched_current_stage", action == "advance")
        )
        target_id = str(
            payload.get("target_stage_id", payload.get("next_stage_id", "")) or ""
        ).strip()
        if action == "advance" and not matched_current_stage:
            action = "hold"
            target_id = ""
        if action == "advance" and target_id not in allowed_candidate_ids:
            action = "hold"
            target_id = ""
        if action != "advance":
            target_id = ""
        next_graph = [current_id]
        if target_id:
            next_graph.append(target_id)
        return {
            "matched": bool(action == "advance" and target_id),
            "match_reason": str(payload.get("reason", payload.get("match_reason", "")) or "")[:180],
            "action": action,
            "next_graph": next_graph,
        }

    def _stage_needs_branch_plan(self, stage: Dict[str, Any]) -> bool:
        """判断当前节点是否需要调用规划器补直接子分支。"""
        if not isinstance(stage, dict):
            return False
        if self._coerce_bool(stage.get("is_terminal_stage", False)):
            return False
        return len(self._candidate_ids_for_stage(stage, self.window_size)) < self.window_size

    def _infer_branch_plan(
        self,
        completion_func: Callable[[str], str],
        parent_stage: Dict[str, Any],
        session_context: Dict[str, Any],
        conversation_content: str,
        llm_cfg: Optional[Dict[str, Any]],
        verified_change: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """围绕已验证变化固定生成三个临时 seed，不写运行态图。"""
        change = verified_change if isinstance(verified_change, dict) else {}
        if not change.get("has_new_change", False):
            return []
        if not str(change.get("change", "") or "").strip():
            return []
        candidate_limit = self.TRANSITION_CANDIDATE_COUNT

        seed_payload = self._call_graph_planner(
            completion_func=completion_func,
            mode="branch_seed",
            parent_stage=parent_stage,
            session_context=session_context,
            conversation_content=conversation_content,
            llm_cfg=llm_cfg,
            verified_change=change,
            max_candidates=candidate_limit,
        )
        seeds, _rejected = self._normalize_branch_seed_result(
            seed_payload,
            accepted_seed_ids=set(),
            limit=candidate_limit,
        )

        return seeds[:candidate_limit]

    def _call_graph_branch_repair(
        self,
        completion_func: Callable[[str], str],
        parent_stage: Dict[str, Any],
        session_context: Dict[str, Any],
        conversation_content: str,
        llm_cfg: Optional[Dict[str, Any]],
        accepted_branches: List[Dict[str, Any]],
        rejected_branches: List[Dict[str, Any]],
        needed_count: int,
    ) -> Dict[str, Any]:
        """调用补分支 prompt，为被拒绝分支留下明确上下文。"""
        prompt = self._build_graph_repair_prompt(
            parent_stage=parent_stage,
            session_context=session_context,
            conversation_content=conversation_content,
            llm_cfg=llm_cfg,
            accepted_branches=accepted_branches,
            rejected_branches=rejected_branches,
            needed_count=needed_count,
        )
        try:
            raw = str(completion_func(prompt) or "")
        except Exception:
            raw = ""
        parsed = self._parse_json_object(raw)
        return parsed if isinstance(parsed, dict) else {}

    def _call_graph_planner(
        self,
        completion_func: Callable[[str], str],
        mode: str,
        parent_stage: Dict[str, Any],
        session_context: Dict[str, Any],
        conversation_content: str,
        llm_cfg: Optional[Dict[str, Any]],
        candidate_seed: Optional[Dict[str, Any]] = None,
        verified_change: Optional[Dict[str, Any]] = None,
        max_candidates: Optional[int] = None,
    ) -> Dict[str, Any]:
        """调用主诉图规划 prompt 并解析 JSON 对象。"""
        prompt = self._build_graph_prompt(
            mode=mode,
            parent_stage=parent_stage,
            session_context=session_context,
            conversation_content=conversation_content,
            llm_cfg=llm_cfg,
            candidate_seed=candidate_seed,
            verified_change=verified_change,
            max_candidates=max_candidates,
        )
        try:
            raw = str(completion_func(prompt) or "")
        except Exception:
            raw = ""
        parsed = self._parse_json_object(raw)
        return parsed if isinstance(parsed, dict) else {}

    def _normalize_branch_seeds(self, payload: Any) -> List[Dict[str, Any]]:
        """清洗第一阶段返回的简略子节点。"""
        seeds, _rejected = self._normalize_branch_seed_result(payload)
        return seeds

    def _normalize_branch_seed_result(
        self,
        payload: Any,
        accepted_seed_ids: Optional[set] = None,
        limit: Optional[int] = None,
    ) -> tuple:
        """清洗简略子节点；LLM ID 只作基础 ID，冲突由程序唯一化。"""
        if not isinstance(payload, dict):
            return [], []
        raw_items = (
            payload.get("children")
            or payload.get("branches")
            or payload.get("next_candidates")
            or payload.get("stage_updates")
            or payload.get("next_graph")
            or []
        )
        seeds: List[Dict[str, Any]] = []
        rejected: List[Dict[str, Any]] = []
        seen = {str(item or "").strip() for item in self.stage_catalog.keys() if str(item or "").strip()}
        seen.add(self.get_current_stage_id())
        for item in accepted_seed_ids or set():
            text = str(item or "").strip()
            if text:
                seen.add(text)
        max_count = self._bounded_int(
            limit,
            self.window_size,
            0,
            max(self.window_size, self.TRANSITION_CANDIDATE_COUNT),
        )
        for item in self._to_list(raw_items):
            if not isinstance(item, dict):
                rejected.append({"id": "", "label": "", "summary": "", "reason": "not_an_object"})
                continue
            label = str(item.get("label", item.get("name", "")) or "").strip()
            summary = str(item.get("summary", item.get("description", label)) or label).strip()
            base_stage_id = (
                str(item.get("id", "") or "").strip()
                or self._make_stage_id(label or summary)
            )[:80]
            stage_id = self._unique_stage_id(base_stage_id, seen)
            seen.add(stage_id)
            seeds.append(
                {
                    "id": stage_id,
                    "label": (label or stage_id)[:80],
                    "summary": self._clip_text(summary or label or stage_id, limit=220),
                }
            )
            if len(seeds) >= max_count:
                break
        return seeds, rejected

    def _normalize_branch_detail(
        self,
        payload: Any,
        seed: Dict[str, Any],
        parent_stage: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """清洗选中 seed 的 detail；稳定字段默认精确继承父节点。"""
        detail: Dict[str, Any] = {}
        if isinstance(payload, dict):
            if isinstance(payload.get("stage"), dict):
                detail = copy.deepcopy(payload.get("stage", {}))
            elif isinstance(payload.get("stage_update"), dict):
                detail = copy.deepcopy(payload.get("stage_update", {}))
            elif any(key in payload for key in ["id", "label", "summary", "narrative_focus"]):
                detail = copy.deepcopy(payload)
            else:
                updates = self._to_list(payload.get("stage_updates", []))
                for item in updates:
                    if isinstance(item, dict):
                        detail = copy.deepcopy(item)
                        break
        parent = parent_stage if isinstance(parent_stage, dict) else {}
        merged = copy.deepcopy(parent)
        merged.update(copy.deepcopy(seed if isinstance(seed, dict) else {}))
        merged.update(detail)
        seed_id = str((seed if isinstance(seed, dict) else {}).get("id", "") or "").strip()
        if seed_id:
            merged["id"] = seed_id

        # 这些字段有运行时消费者或属于既有 I/O 形状，但单轮 verified_change
        # 不负责改写。保留字段、精确继承，禁止 detail 自由改写。
        stable_defaults = {
            "speaking_style": {},
            "emotion_vector": {},
            "relation_modifiers": {},
            "advance_signals": [],
            "hold_signals": [],
        }
        for field_name, default_value in stable_defaults.items():
            merged[field_name] = copy.deepcopy(parent.get(field_name, default_value))

        # core_belief 是病例级稳定值。即使旧 planner 或脏数据仍输出该键，
        # candidate 也只镜像初始化时锁定的值；动态认知变化由 label/summary 表达。
        merged["core_belief"] = self.core_belief
        return self._sanitize_stage(merged, source="llm")

    def _materialize_branch_plan(self, parent_id: str, child_stages: List[Dict[str, Any]]) -> List[str]:
        """写入子节点并更新父节点 next_candidates。"""
        parent_key = str(parent_id or "").strip()
        parent_stage = self.stage_catalog.get(parent_key, {})
        child_ids: List[str] = self._candidate_ids_for_stage(parent_stage, self.window_size)
        existing_ids = {str(item or "").strip() for item in self.stage_catalog.keys() if str(item or "").strip()}
        for stage in child_stages:
            normalized = self._sanitize_stage(stage, source=str(stage.get("source", "llm") or "llm"))
            stage_id = str(normalized.get("id", "") or "").strip()
            if not stage_id or stage_id == parent_id or stage_id in existing_ids or stage_id in child_ids:
                continue
            normalized["next_candidates"] = []
            self.stage_catalog[stage_id] = normalized
            existing_ids.add(stage_id)
            child_ids.append(stage_id)
            if len(child_ids) >= self.window_size:
                break
        self._apply_branch_candidates(parent_id, child_ids)
        self._prune_unknown_next_candidates(child_ids + [str(parent_id or "").strip()])
        return child_ids

    def _accepted_branch_summaries(self, branch_ids: List[str]) -> List[Dict[str, Any]]:
        """把已录用候选压缩成补分支 prompt 需要的摘要。"""
        results: List[Dict[str, Any]] = []
        for branch_id in branch_ids:
            stage = self.stage_catalog.get(str(branch_id or "").strip(), {})
            if not isinstance(stage, dict):
                continue
            results.append(
                {
                    "id": str(stage.get("id", "") or "")[:80],
                    "label": str(stage.get("label", "") or "")[:80],
                    "summary": self._clip_text(stage.get("summary", ""), limit=220),
                }
            )
        return results

    def _seed_branch_summaries(self, seeds: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """把本轮已通过校验的 seed 压缩成补分支 prompt 需要的摘要。"""
        results: List[Dict[str, Any]] = []
        for seed in seeds:
            if not isinstance(seed, dict):
                continue
            results.append(
                {
                    "id": str(seed.get("id", "") or "")[:80],
                    "label": str(seed.get("label", "") or "")[:80],
                    "summary": self._clip_text(seed.get("summary", ""), limit=220),
                }
            )
        return results

    def _stage_summary_for_prompt(self, stage: Dict[str, Any]) -> Dict[str, Any]:
        """返回候选分支列表使用的紧凑节点摘要。"""
        payload = stage if isinstance(stage, dict) else {}
        return {
            "id": str(payload.get("id", "") or "")[:80],
            "label": str(payload.get("label", "") or "")[:80],
            "semantic": self._clip_text(
                payload.get("summary", "")
                or payload.get("core_belief", "")
                or "；".join(str(item) for item in self._to_list(payload.get("narrative_focus", []))),
                limit=120,
            ),
        }

    def _planner_stage_view(self, stage: Dict[str, Any]) -> Dict[str, Any]:
        """仅保留规划直接需要的当前节点语义。"""
        payload = stage if isinstance(stage, dict) else {}
        return {
            "id": str(payload.get("id", "") or "")[:80],
            "label": str(payload.get("label", "") or "")[:80],
            "summary": self._clip_text(payload.get("summary", ""), limit=220),
            "core_belief": self._clip_text(payload.get("core_belief", ""), limit=160),
            "narrative_focus": [
                self._clip_text(item, limit=80)
                for item in self._to_list(payload.get("narrative_focus", []))[:6]
                if self._clip_text(item, limit=80)
            ],
        }

    @staticmethod
    def _planner_semantic_units(value: Any) -> set:
        """以中英文双字符片段做轻量相关性排序，不引入额外 NLP 依赖。"""
        text = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]+", "", str(value or "").lower())
        if not text:
            return set()
        if len(text) == 1:
            return {text}
        return {text[index : index + 2] for index in range(len(text) - 1)}

    def _semantic_guard_for_prompt(
        self,
        parent_stage: Dict[str, Any],
        mode: str,
        candidate_seed: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """返回少量最相关旧节点；它们只用于防重复，不充当生成示例。"""
        cfg = self.planner if isinstance(self.planner, dict) else {}
        configured_limit = self._bounded_int(
            cfg.get("prompt_semantic_guard_limit"), 20, 0, 80
        )
        mode_cap = 2 if str(mode or "") == "branch_detail" else 4
        guard_limit = min(configured_limit, mode_cap)
        if guard_limit <= 0:
            return []

        parent_view = self._planner_stage_view(parent_stage)
        seed_view = candidate_seed if isinstance(candidate_seed, dict) else {}
        reference = json.dumps(
            [
                parent_view.get("label", ""),
                parent_view.get("summary", ""),
                parent_view.get("core_belief", ""),
                parent_view.get("narrative_focus", []),
                seed_view.get("label", ""),
                seed_view.get("summary", ""),
            ],
            ensure_ascii=False,
        )
        generic_units = {
            "患者",
            "主诉",
            "阶段",
            "节点",
            "当前",
            "状态",
            "变化",
            "开始",
            "意识",
            "自己",
        }
        reference_units = self._planner_semantic_units(reference) - generic_units
        current_id = str(parent_stage.get("id", "") or "").strip()
        seed_id = str(
            (candidate_seed if isinstance(candidate_seed, dict) else {}).get("id", "") or ""
        ).strip()
        excluded_ids = {current_id, seed_id}
        if str(mode or "") != "branch_detail":
            excluded_ids.update(
                self._candidate_ids_for_stage(parent_stage, self.window_size)
            )
        ranked: List[tuple] = []
        for index, stage in enumerate(self.stage_catalog.values()):
            stage_id = str(stage.get("id", "") or "").strip()
            if not stage_id or stage_id in excluded_ids:
                continue
            compact = self._stage_summary_for_prompt(stage)
            stage_units = self._planner_semantic_units(
                "{} {}".format(compact.get("label", ""), compact.get("semantic", ""))
            ) - generic_units
            overlap = len(reference_units.intersection(stage_units))
            if overlap <= 0:
                continue
            ranked.append((overlap, index, compact))
        ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return [item[2] for item in ranked[:guard_limit]]

    def _build_prompt_graph_snapshot(
        self,
        parent_stage: Dict[str, Any],
        mode: str,
    ) -> Dict[str, Any]:
        """构造 branch_seed/repair 共用的紧凑分支规划视图。"""
        child_ids = self._candidate_ids_for_stage(parent_stage, self.window_size)
        existing_children = [
            self._stage_summary_for_prompt(self.stage_catalog[stage_id])
            for stage_id in child_ids
            if stage_id in self.stage_catalog
        ]
        return {
            "mode": str(mode or "branch_seed"),
            "current_stage": self._planner_stage_view(parent_stage),
            "existing_children": existing_children,
            "semantic_guard": self._semantic_guard_for_prompt(
                parent_stage=parent_stage,
                mode=mode,
            ),
        }

    def _build_graph_prompt(
        self,
        mode: str,
        parent_stage: Dict[str, Any],
        session_context: Dict[str, Any],
        conversation_content: str,
        llm_cfg: Optional[Dict[str, Any]],
        candidate_seed: Optional[Dict[str, Any]] = None,
        verified_change: Optional[Dict[str, Any]] = None,
        max_candidates: Optional[int] = None,
    ) -> str:
        """构造分支规划器 prompt。"""
        del session_context, conversation_content, llm_cfg, max_candidates
        normalized_mode = str(mode or "branch_seed")
        change_view = self._verified_change_for_planner(verified_change)
        if normalized_mode == "branch_detail":
            seed = copy.deepcopy(candidate_seed if isinstance(candidate_seed, dict) else {})
            payload = {
                "mode": normalized_mode,
                "current_stage": self._planner_stage_view(parent_stage),
                "candidate_seed": seed,
                "verified_change": change_view,
            }
        else:
            payload = {
                "mode": normalized_mode,
                "current_stage": self._planner_stage_view(parent_stage),
                "verified_change": change_view,
                "max_candidates": self.TRANSITION_CANDIDATE_COUNT,
            }
        payload_json = json.dumps(payload, ensure_ascii=False)
        return render_prompt(
            "depression/graph_planner",
            {"payload_json": payload_json},
        )

    def _verified_change_for_planner(self, value: Any) -> Dict[str, Any]:
        """只向 planner 透传已规范化的最小变化及可用审计证据。"""
        payload = value if isinstance(value, dict) else {}
        return {
            "change": self._clip_text(payload.get("change", ""), limit=240),
            "evidence_quotes": [
                self._clip_text(item, limit=240)
                for item in self._to_list(payload.get("evidence_quotes", []))[:6]
                if self._clip_text(item, limit=240)
            ],
        }

    def _build_graph_repair_prompt(
        self,
        parent_stage: Dict[str, Any],
        session_context: Dict[str, Any],
        conversation_content: str,
        llm_cfg: Optional[Dict[str, Any]],
        accepted_branches: List[Dict[str, Any]],
        rejected_branches: List[Dict[str, Any]],
        needed_count: int,
    ) -> str:
        """构造候选分支补齐 prompt。"""
        del session_context, conversation_content, llm_cfg
        payload = self._build_prompt_graph_snapshot(
            parent_stage,
            "branch_repair",
        )
        payload.update({
            "accepted_branch_candidates": copy.deepcopy(accepted_branches),
            "rejected_branch_candidates": copy.deepcopy(rejected_branches),
            "needed_count": max(0, int(needed_count)),
        })
        repair_context = (
            "之前被拒绝使用的子分支数量：{}\n"
            "已经录用的子分支数量：{}\n"
            "还需要生成的合法子分支数量：{}"
        ).format(
            len(rejected_branches),
            len(accepted_branches),
            max(0, int(needed_count)),
        )
        return render_prompt(
            "depression/graph_branch_repair",
            {
                "payload_json": json.dumps(payload, ensure_ascii=False),
                "repair_context": repair_context,
            },
        )

    def _normalize_graph_path(self, value: Any, current_stage_id: str) -> List[str]:
        current_id = str(current_stage_id or "").strip()
        normalized = self._normalize_graph_ids(value)
        if not normalized and current_id:
            return [current_id]
        if current_id:
            # 无论外部给什么 next_graph，第一位都必须强制对齐当前节点。
            # 这样 commit_turn() 才能把它理解为“当前窗口”，而不是“纯未来列表”。
            if current_id not in normalized:
                normalized = [current_id] + normalized
            elif normalized[0] != current_id:
                normalized = [current_id] + [item for item in normalized if item != current_id]
        return normalized

    def _normalize_graph_ids(self, value: Any) -> List[str]:
        items = self._to_list(value)
        normalized: List[str] = []
        seen = set()
        for item in items:
            stage_id = ""
            if isinstance(item, dict):
                # 允许图路径里混入“完整 stage 对象”，并在这里即时注册到 catalog。
                stage = self._sanitize_stage(item, source=str(item.get("source", "config")) if isinstance(item, dict) else "config")
                self.stage_catalog[stage["id"]] = stage
                stage_id = stage["id"]
            else:
                text = str(item or "").strip()
                if text in self.stage_catalog:
                    stage_id = text
            if not stage_id or stage_id in seen:
                continue
            seen.add(stage_id)
            normalized.append(stage_id)
        return normalized

    def _sanitize_stage(self, raw: Any, source: str = "config") -> Dict[str, Any]:
        """把外部配置 / LLM 生成的 stage 清洗成统一结构。"""
        payload = raw if isinstance(raw, dict) else {}
        label = str(payload.get("label", "") or "").strip() or "未命名主诉节点"
        stage_id = str(payload.get("id", "") or "").strip() or self._make_stage_id(label)
        summary = str(payload.get("summary", payload.get("description", label)) or label).strip()
        core_belief = (
            self.core_belief
            if hasattr(self, "_core_belief")
            else str(payload.get("core_belief", "") or "").strip()
        )
        speaking_style = payload.get("speaking_style", {}) if isinstance(payload.get("speaking_style", {}), dict) else {}
        emotion_vector = payload.get("emotion_vector", {}) if isinstance(payload.get("emotion_vector", {}), dict) else {}
        relation_modifiers = payload.get("relation_modifiers", {}) if isinstance(payload.get("relation_modifiers", {}), dict) else {}

        # 这里就是 JSON 配置落地为运行时 stage 的关键位置。
        # 例如你在 `depression_config.json` 里看到的：
        # - narrative_focus
        # - speaking_style
        # - emotion_vector
        # 最终都会被规整成下面这个 ComplaintStage 结构。
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

    @staticmethod
    def _unique_stage_id(base_stage_id: str, occupied_ids: Any) -> str:
        """Return an at-most-80-char ID without changing any stage semantics."""
        base = str(base_stage_id or "").strip()[:80] or "stage"
        occupied = {
            str(item or "").strip()
            for item in (occupied_ids or set())
            if str(item or "").strip()
        }
        if base not in occupied:
            return base
        suffix_number = 2
        while True:
            suffix = "_{}".format(suffix_number)
            candidate = "{}{}".format(base[: 80 - len(suffix)], suffix)
            if candidate not in occupied:
                return candidate
            suffix_number += 1

    def _runtime_event_from_context(self, session_context: Dict[str, Any]) -> Dict[str, Any]:
        runtime_event = (
            session_context.get("runtime_event", {})
            if isinstance(session_context.get("runtime_event", {}), dict)
            else {}
        )
        metadata = (
            copy.deepcopy(runtime_event.get("metadata", {}))
            if isinstance(runtime_event.get("metadata", {}), dict)
            else {}
        )
        raw_evidence = runtime_event.get("evidence_ids", metadata.get("evidence_ids", []))
        evidence_ids: List[str] = []
        seen = set()
        for item in self._to_list(raw_evidence):
            text = str(item or "").strip()
            if not text or text in seen:
                continue
            seen.add(text)
            evidence_ids.append(text[:80])
            if len(evidence_ids) >= 20:
                break
        metadata["evidence_ids"] = evidence_ids
        source = str(runtime_event.get("source", "") or "").strip() or "chat"
        return {
            "source": source[:32],
            "evidence_ids": evidence_ids,
            "metadata": metadata,
        }

    def _graph_expansion_context(self, session_context: Dict[str, Any]) -> Dict[str, Any]:
        payload = copy.deepcopy(session_context if isinstance(session_context, dict) else {})
        payload.setdefault("scene", {})
        if isinstance(payload.get("scene", {}), dict):
            payload["scene"]["interaction_type"] = "主诉图补足"
        payload["runtime_event"] = {
            "source": "graph_window_expansion",
            "metadata": {"purpose": "maintain_complaint_graph_window"},
            "evidence_ids": [],
        }
        return payload

    def _track_dialogue(
        self,
        current_stage: Dict[str, Any],
        evaluation: Dict[str, Any],
        conversation_excerpt: str,
        session_context: Dict[str, Any],
    ) -> None:
        # 对话历史是“为什么会推进到这里”的轻量证据，
        # 供 debug / 回放 / 人工审查使用，不参与复杂推理。
        participants = session_context.get("participants", {}) if isinstance(session_context.get("participants", {}), dict) else {}
        scene = session_context.get("scene", {}) if isinstance(session_context.get("scene", {}), dict) else {}
        runtime_event = self._runtime_event_from_context(session_context)
        row = {
            "timestamp": self._now().isoformat(),
            "stage_id": str(current_stage.get("id", "") or ""),
            "stage_label": str(current_stage.get("label", "") or ""),
            "conversation_excerpt": self._clip_text(conversation_excerpt, limit=220),
            "matched": bool(evaluation.get("matched", False)),
            "match_reason": str(evaluation.get("match_reason", "") or "")[:180],
            "action": str(evaluation.get("action", "hold") or "hold"),
            "other_agent": str(participants.get("other_agent", "") or ""),
            "relationship": str(participants.get("relationship", "") or ""),
            "interaction_type": str(scene.get("interaction_type", "") or ""),
            "source": str(runtime_event.get("source", "chat") or "chat"),
            "evidence_ids": copy.deepcopy(runtime_event.get("evidence_ids", [])),
            "metadata": copy.deepcopy(runtime_event.get("metadata", {})),
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
        match_reason: str,
        pointer_before: int,
        pointer_after: int,
        duration_minutes: float,
        source: str = "chat",
    ) -> None:
        # stage_history 更像状态迁移日志，记录 from/to/action 和匹配证据。
        self.stage_history.append(
            {
                "timestamp": self._now(),
                "action": str(action or "hold"),
                "source": str(source or "chat")[:32],
                "from_stage_id": str(from_stage.get("id", "") or ""),
                "from_stage_label": str(from_stage.get("label", "") or ""),
                "to_stage_id": str(to_stage.get("id", "") or ""),
                "to_stage_label": str(to_stage.get("label", "") or ""),
                "matched": bool(matched),
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

    @staticmethod
    def _evidence_compact_text(value: Any) -> tuple[str, List[int]]:
        """Return a punctuation-insensitive view and its offsets in the source text."""
        text = str(value or "")
        characters: List[str] = []
        offsets: List[int] = []
        for index, char in enumerate(text):
            if char.isalnum():
                characters.append(char.lower())
                offsets.append(index)
        return "".join(characters), offsets

    @staticmethod
    def _evidence_safety_signature(text: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """Keep small quote repairs from crossing negation or number boundaries."""
        compact, _ = ComplaintGraphManager._evidence_compact_text(text)
        negations = tuple(char for char in compact if char in "不没无未否别勿")
        numbers = tuple(re.findall(r"\d+", compact))
        return negations, numbers

    @classmethod
    def _resolve_evidence_quote(cls, quote: Any, evidence_texts: List[str]) -> str:
        """Map a near-verbatim model quote back to the actual patient excerpt.

        This only repairs cosmetic punctuation/spacing differences and at most
        two character-level slips in a sufficiently long quote, then persists
        source text rather than the model's wording. Failed repair leaves the
        optional audit field empty; it does not reject a semantic state change.
        """
        requested = cls._clip_text(quote, limit=240)
        if not requested:
            return ""
        requested_compact, _ = cls._evidence_compact_text(requested)
        if not requested_compact:
            return ""
        requested_signature = cls._evidence_safety_signature(requested)
        best_match: Optional[tuple[tuple[int, int, float], str]] = None

        for evidence in evidence_texts:
            source = cls._clip_text(evidence, limit=6000)
            if not source:
                continue
            direct_index = source.find(requested)
            if direct_index >= 0:
                return source[direct_index : direct_index + len(requested)].strip()

            compact_source, offsets = cls._evidence_compact_text(source)
            compact_index = compact_source.find(requested_compact)
            if compact_index >= 0:
                start = offsets[compact_index]
                end = offsets[compact_index + len(requested_compact) - 1] + 1
                return source[start:end].strip()

            # Do not fuzzy-match short quotes: a one-character difference can
            # reverse their meaning (for example, adding a negation).
            if len(requested_compact) < 12:
                continue
            allowed_edits = 2
            lower = max(1, len(requested_compact) - allowed_edits)
            upper = min(len(compact_source), len(requested_compact) + allowed_edits)
            for candidate_length in range(lower, upper + 1):
                for start_index in range(0, len(compact_source) - candidate_length + 1):
                    candidate = compact_source[
                        start_index : start_index + candidate_length
                    ]
                    if cls._evidence_safety_signature(candidate) != requested_signature:
                        continue
                    matcher = difflib.SequenceMatcher(
                        None, requested_compact, candidate, autojunk=False
                    )
                    matched_characters = sum(
                        block.size for block in matcher.get_matching_blocks()
                    )
                    edits = max(len(requested_compact), len(candidate)) - matched_characters
                    if edits > allowed_edits or matcher.ratio() < 0.85:
                        continue
                    source_start = offsets[start_index]
                    source_end = offsets[start_index + candidate_length - 1] + 1
                    resolved = source[source_start:source_end].strip()
                    # Prefer a source span with the same length before a
                    # slightly higher-ratio substring that silently drops a
                    # meaningful trailing word.
                    score = (
                        edits,
                        abs(len(requested_compact) - len(candidate)),
                        -matcher.ratio(),
                    )
                    if best_match is None or score < best_match[0]:
                        best_match = (score, resolved)
        return best_match[1] if best_match else ""

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


def _coerce_datetime(value: Any, fallback: Optional[Callable[[], datetime]] = None) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        text = value.strip()
        if text:
            try:
                return datetime.fromisoformat(text)
            except Exception:
                pass
    if callable(fallback):
        try:
            fallback_value = fallback()
            if isinstance(fallback_value, datetime):
                return fallback_value
        except Exception:
            pass
    return _simulation_now()
