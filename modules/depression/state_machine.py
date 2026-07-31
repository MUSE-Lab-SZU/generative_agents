"""主诉图管理器。"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

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

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        now_provider: Optional[Callable[[], datetime]] = None,
    ):
        config = config if isinstance(config, dict) else {}
        # 支持两种传法：
        # 1. 直接传 complaint_graph 配置本身；
        # 2. 传整个 depression_config.json，并优先取 `complaint_graph`。
        graph_config = self._select_graph_config(config)

        self._raw_graph_config = copy.deepcopy(graph_config)
        self.planner: Dict[str, Any] = copy.deepcopy(graph_config.get("planner", {}))
        self.window_size = self._bounded_int(self.planner.get("window_size"), 3, 1, 8)
        self.llm_enabled = self._coerce_bool(self.planner.get("llm_enabled", True))
        self.max_dialog_history = self._bounded_int(
            self.planner.get("max_dialog_history"), 12, 4, 50
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

        self.planned_graph: List[str] = []
        self.stage_index = 0
        self.stage_history: List[Dict[str, Any]] = []
        self.dialogue_history: List[Dict[str, Any]] = []
        self.last_session_context: Dict[str, Any] = {}
        self.last_evaluation: Dict[str, Any] = {}
        self.stage_start_time = self._now()

        self.reset()

    @staticmethod
    def _select_graph_config(config: Dict[str, Any]) -> Dict[str, Any]:
        if "complaint_graph" in config and isinstance(config.get("complaint_graph", {}), dict):
            return copy.deepcopy(config.get("complaint_graph", {}))
        return copy.deepcopy(config if isinstance(config, dict) else {})

    def _now(self) -> datetime:
        try:
            value = self._now_provider()
        except Exception:
            value = _simulation_now()
        return _coerce_datetime(value, fallback=_simulation_now)

    def reset(self) -> None:
        self.planned_graph = self._build_default_graph(self.initial_stage_id)
        self.stage_index = 0
        self.stage_history = []
        self.dialogue_history = []
        self.last_session_context = {}
        self.last_evaluation = {}
        self.stage_start_time = self._now()

    def get_current_stage(self) -> Dict[str, Any]:
        if not self.planned_graph:
            self.reset()
        idx = min(max(0, int(self.stage_index)), len(self.planned_graph) - 1)
        stage_id = self.planned_graph[idx]
        return copy.deepcopy(self.stage_catalog.get(stage_id, self._sanitize_stage(self.DEFAULT_STAGE, source="fallback")))

    def get_current_stage_id(self) -> str:
        return str(self.get_current_stage().get("id", ""))

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
        - 如果要前进，当前节点的候选分支是什么。
        真正改状态要等 `commit_turn()`。

        动作判定只交给 LLM（graph_transition.txt）：没有可用的 completion_func
        或判定结果时，一律默认 hold，不再走启发式规则或外部传入的信号。
        """
        session_context = session_context if isinstance(session_context, dict) else {}
        conversation = str(conversation_content or "").strip()
        current_stage = self.get_current_stage()

        if callable(completion_func) and self.llm_enabled:
            # 分支规划器只补候选，不参与本轮动作裁决。
            self.ensure_graph_window(
                session_context=session_context,
                conversation_content=conversation,
                completion_func=completion_func,
                llm_cfg=llm_cfg,
                record_evaluation=False,
            )
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
            # 拿不到 LLM 判定时保持当前节点，只回填候选窗口。
            transition_signal = {
                "matched": False,
                "match_reason": "no_llm_decision",
                "action": "hold",
                "next_graph": self._preview_future_graph(current_stage),
            }

        matched = bool(transition_signal.get("matched", False))
        match_reason = str(transition_signal.get("match_reason", "") or "")
        action = str(transition_signal.get("action", "hold") or "hold").strip().lower()
        next_graph_ids = self._normalize_graph_path(
            transition_signal.get("next_graph", self._preview_future_graph(current_stage)),
            current_stage["id"],
        )
        if action not in {"hold", "advance"}:
            action = "hold"
        if action == "advance" and self._coerce_bool(current_stage.get("is_terminal_stage", False)):
            # 终态节点不再向外推进，即便 LLM 给出了 advance。
            action = "hold"

        next_stage: Optional[Dict[str, Any]] = None
        if action == "advance":
            # advance 的前提：必须能推出一个合法的下一节点。
            # 合法目标只有一类：当前节点已有的候选分支；
            # 禁止只给一个裸 id 就凭空推进到与当前节点无边的未知节点。
            candidate_ids = self._candidate_ids_for_stage(current_stage, self.window_size)
            target_id = str(next_graph_ids[1] if len(next_graph_ids) > 1 else "").strip()
            if target_id and target_id in candidate_ids:
                next_stage = copy.deepcopy(self.stage_catalog[target_id])
            elif not target_id and candidate_ids:
                target_id = candidate_ids[0]
                next_graph_ids = [current_stage["id"], target_id]
                next_stage = copy.deepcopy(self.stage_catalog[target_id])
            if not next_stage:
                # 没有可推进的候选时，不凭规则臆造主诉节点。
                action = "hold"
                next_graph_ids = self._preview_future_graph(current_stage)

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
        - `hold` 只更新当前节点候选分支；
        - `advance` 选择一个候选分支成为新的当前节点。
        """
        evaluation = evaluation if isinstance(evaluation, dict) else {}
        current_before = self.get_current_stage()
        action = str(evaluation.get("action", "hold") or "hold").strip().lower()
        if action not in {"hold", "advance"}:
            action = "hold"
        matched = bool(evaluation.get("matched", False))
        match_reason = str(evaluation.get("match_reason", "") or "")[:180]
        next_graph_value = evaluation.get("next_graph", [])
        next_graph = self._normalize_graph_path(next_graph_value, current_before.get("id", ""))
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

        if action == "hold":
            # hold 只改当前节点的候选分支，不移动指针。
            self._replace_future_graph(next_graph)
        elif action == "advance":
            target_stage_id = str(next_graph[1] if len(next_graph) > 1 else "").strip()
            # 防御性复检：只有当目标确实是当前节点的合法候选、且当前节点非终态时
            # 才允许移动指针，否则降级为 hold。这样即便上游传入了被篡改的
            # evaluation，状态机底层也不会推进到与当前节点无边的任意节点。
            candidate_ids = self._candidate_ids_for_stage(current_before, self.window_size)
            is_terminal = self._coerce_bool(current_before.get("is_terminal_stage", False))
            if target_stage_id and target_stage_id in candidate_ids and not is_terminal:
                # advance 表示从候选分支中选中一个节点，兄弟分支仍留在父节点上。
                self._ensure_branch_candidates(current_before.get("id", ""), [target_stage_id])
                self.planned_graph = self.planned_graph[: self.stage_index + 1]
                if not self.planned_graph or self.planned_graph[-1] != target_stage_id:
                    self.planned_graph.append(target_stage_id)
                self.stage_index = len(self.planned_graph) - 1
                self.stage_start_time = self._now()
            else:
                # 目标非法：不移动指针，按 hold 处理当前候选窗口。
                action = "hold"
                self._replace_future_graph(self._preview_future_graph(current_before))
        self._ensure_future_window()
        current_after = self.get_current_stage()

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
        """用 LLM 为当前节点补候选分支，但不推进当前节点。"""
        return self.ensure_graph_window(
            session_context=session_context,
            conversation_content=conversation_content,
            completion_func=completion_func,
            llm_cfg=llm_cfg,
            source="graph_window_init",
            record_evaluation=True,
        )

    def ensure_graph_window(
        self,
        session_context: Dict[str, Any],
        conversation_content: str,
        completion_func: Optional[Callable[[str], str]] = None,
        llm_cfg: Optional[Dict[str, Any]] = None,
        source: str = "graph_window_expansion",
        record_evaluation: bool = False,
    ) -> Dict[str, Any]:
        """当当前节点没有 next_candidates 时，按 window_size 规划候选分支。"""
        if not callable(completion_func) or not self.llm_enabled:
            return self.get_graph_snapshot()
        current_stage = self.get_current_stage()
        if not self._stage_needs_branch_plan(current_stage):
            return self.get_graph_snapshot()
        if self._coerce_bool(current_stage.get("is_terminal_stage", False)):
            return self.get_graph_snapshot()

        planning_context = self._graph_expansion_context(session_context)
        child_stages = self._infer_branch_plan(
            completion_func=completion_func,
            parent_stage=current_stage,
            session_context=planning_context,
            conversation_content=str(conversation_content or ""),
            llm_cfg=llm_cfg,
        )
        if child_stages:
            child_ids = self._materialize_branch_plan(current_stage["id"], child_stages)
            if record_evaluation:
                self.last_session_context = copy.deepcopy(planning_context)
                self.last_evaluation = {
                    "action": "plan_branches",
                    "matched": False,
                    "match_reason": "planned {} candidate branches".format(len(child_ids)),
                    "source": str(source or "graph_window_expansion")[:32],
                }
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
            "initial_stage_id": self.initial_stage_id,
            "current_stage_id": self.get_current_stage_id(),
            "runtime_stages": runtime_stages,
            "planned_graph": [str(item) for item in self.planned_graph],
            "stage_index": int(self.stage_index),
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
        base_config: Optional[Dict[str, Any]] = None,
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
        cfg = {
            "planner": copy.deepcopy(payload_cfg.get("planner", base_cfg.get("planner", {}))),
            "initial_stage_id": str(
                payload.get(
                    "initial_stage_id",
                    payload.get(
                        "current_stage_id",
                        payload_cfg.get("initial_stage_id", base_cfg.get("initial_stage_id", "")),
                    ),
                )
                or ""
            ),
            "stages": copy.deepcopy(stage_catalog) if isinstance(stage_catalog, list) else [],
        }
        manager = cls(config={"complaint_graph": cfg}, now_provider=now_provider)
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
        """调用 LLM 判定本轮是否推进主诉节点。"""
        prompt = self._build_transition_prompt(
            current_stage=current_stage,
            session_context=session_context,
            conversation_content=conversation_content,
            counterpart_utterance=counterpart_utterance,
            llm_cfg=llm_cfg,
        )
        try:
            raw = str(completion_func(prompt) or "")
        except Exception:
            raw = ""
        parsed = self._parse_json_object(raw)
        return self._normalize_transition_signal(parsed, current_stage)

    def _build_transition_prompt(
        self,
        current_stage: Dict[str, Any],
        session_context: Dict[str, Any],
        conversation_content: str,
        llm_cfg: Optional[Dict[str, Any]],
        counterpart_utterance: str = "",
    ) -> str:
        """构造主诉图推进判定 prompt。"""
        cfg = llm_cfg if isinstance(llm_cfg, dict) else {}
        text_limit = self._bounded_int(cfg.get("max_text_length"), 1200, 200, 6000)
        candidate_ids = self._candidate_ids_for_stage(current_stage, self.window_size)
        payload = {
            "task": "transition_decision",
            "current_stage": copy.deepcopy(current_stage),
            "candidate_stages": [copy.deepcopy(self.stage_catalog[item]) for item in candidate_ids],
            "candidate_ids": candidate_ids,
            "session_context": session_context,
            "conversation_content": self._clip_text(conversation_content, limit=text_limit),
            "counterpart_utterance": self._clip_text(counterpart_utterance, limit=text_limit),
            "state_duration_minutes": round(float(self.get_state_duration()), 4),
        }
        return render_prompt(
            "depression/graph_transition",
            {"payload_json": json.dumps(payload, ensure_ascii=False)},
        )

    def _normalize_transition_signal(
        self,
        payload: Any,
        current_stage: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """清洗 LLM 的推进判定结果。"""
        if not isinstance(payload, dict):
            return None
        current_id = str(current_stage.get("id", "") or "").strip()
        candidate_ids = self._candidate_ids_for_stage(current_stage, self.window_size)
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
        if action == "advance" and target_id not in candidate_ids:
            action = "hold"
            target_id = ""
        if action != "advance":
            target_id = ""
        next_graph = [current_id]
        if target_id:
            next_graph.append(target_id)
        return {
            "matched": bool(matched_current_stage),
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
    ) -> List[Dict[str, Any]]:
        """两阶段生成当前节点的候选子分支。"""
        existing_candidates = self._accepted_branch_summaries(
            self._candidate_ids_for_stage(parent_stage, self.window_size)
        )
        needed_count = max(0, self.window_size - len(existing_candidates))
        if needed_count <= 0:
            return []

        seed_payload = self._call_graph_planner(
            completion_func=completion_func,
            mode="branch_seed",
            parent_stage=parent_stage,
            session_context=session_context,
            conversation_content=conversation_content,
            llm_cfg=llm_cfg,
        )
        seeds, rejected = self._normalize_branch_seed_result(
            seed_payload,
            accepted_seed_ids=set(),
            limit=needed_count,
        )
        repair_attempt = 0
        max_repair_attempts = self._bounded_int(
            self.planner.get("max_branch_repair_attempts"),
            self.window_size + 2,
            1,
            16,
        )
        while len(seeds) < needed_count and repair_attempt < max_repair_attempts:
            repair_attempt += 1
            repair_payload = self._call_graph_branch_repair(
                completion_func=completion_func,
                parent_stage=parent_stage,
                session_context=session_context,
                conversation_content=conversation_content,
                llm_cfg=llm_cfg,
                accepted_branches=existing_candidates + self._seed_branch_summaries(seeds),
                rejected_branches=rejected,
                needed_count=needed_count - len(seeds),
            )
            supplement, supplement_rejected = self._normalize_branch_seed_result(
                repair_payload,
                accepted_seed_ids={str(seed.get("id", "") or "").strip() for seed in seeds},
                limit=needed_count - len(seeds),
            )
            rejected.extend(supplement_rejected)
            if not supplement:
                continue
            seeds.extend(supplement)

        stages: List[Dict[str, Any]] = []
        for seed in seeds[:needed_count]:
            detail_payload = self._call_graph_planner(
                completion_func=completion_func,
                mode="branch_detail",
                parent_stage=parent_stage,
                session_context=session_context,
                conversation_content=conversation_content,
                llm_cfg=llm_cfg,
                candidate_seed=seed,
            )
            stages.append(self._normalize_branch_detail(detail_payload, seed))
        return stages

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
    ) -> Dict[str, Any]:
        """调用主诉图规划 prompt 并解析 JSON 对象。"""
        prompt = self._build_graph_prompt(
            mode=mode,
            parent_stage=parent_stage,
            session_context=session_context,
            conversation_content=conversation_content,
            llm_cfg=llm_cfg,
            candidate_seed=candidate_seed,
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
        """清洗简略子节点，并返回被拒绝的候选，供补分支 prompt 使用。"""
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
        max_count = self._bounded_int(limit, self.window_size, 0, self.window_size)
        for item in self._to_list(raw_items):
            if not isinstance(item, dict):
                rejected.append({"id": "", "label": "", "summary": "", "reason": "not_an_object"})
                continue
            label = str(item.get("label", item.get("name", "")) or "").strip()
            summary = str(item.get("summary", item.get("description", label)) or label).strip()
            stage_id = (str(item.get("id", "") or "").strip() or self._make_stage_id(label or summary))[:80]
            if not stage_id or stage_id in seen:
                rejected.append(
                    {
                        "id": stage_id,
                        "label": (label or stage_id)[:80],
                        "summary": self._clip_text(summary or label or stage_id, limit=220),
                        "reason": "duplicate_or_existing_stage_id" if stage_id else "empty_stage_id",
                    }
                )
                continue
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

    def _normalize_branch_detail(self, payload: Any, seed: Dict[str, Any]) -> Dict[str, Any]:
        """清洗第二阶段返回的完整子节点，失败时用 seed 补默认字段。"""
        detail: Dict[str, Any] = {}
        if isinstance(payload, dict):
            if isinstance(payload.get("stage"), dict):
                detail = copy.deepcopy(payload.get("stage", {}))
            elif isinstance(payload.get("stage_update"), dict):
                detail = copy.deepcopy(payload.get("stage_update", {}))
            elif any(key in payload for key in ["id", "label", "summary", "core_belief"]):
                detail = copy.deepcopy(payload)
            else:
                updates = self._to_list(payload.get("stage_updates", []))
                for item in updates:
                    if isinstance(item, dict):
                        detail = copy.deepcopy(item)
                        break
        merged = copy.deepcopy(seed if isinstance(seed, dict) else {})
        merged.update(detail)
        seed_id = str((seed if isinstance(seed, dict) else {}).get("id", "") or "").strip()
        if seed_id:
            merged["id"] = seed_id
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
        """返回用于 prompt 去重和语义参照的紧凑节点摘要。"""
        payload = stage if isinstance(stage, dict) else {}
        return {
            "id": str(payload.get("id", "") or "")[:80],
            "label": str(payload.get("label", "") or "")[:80],
            "summary": self._clip_text(payload.get("summary", ""), limit=220),
        }

    def _prompt_known_stage_ids(self) -> List[str]:
        """返回 prompt 所需的已占用 ID；程序端仍负责最终唯一性校验。"""
        limit = self._bounded_int(
            self.planner.get("prompt_known_stage_id_limit"), 512, 50, 5000
        )
        ids = [str(stage_id or "").strip()[:80] for stage_id in self.stage_catalog]
        return [stage_id for stage_id in ids if stage_id][:limit]

    def _build_prompt_graph_snapshot(
        self,
        parent_stage: Dict[str, Any],
        mode: str,
    ) -> Dict[str, Any]:
        """构造有界的主诉图 prompt 视图，不改变运行态 catalog。

        完整 ``stage_catalog`` 是程序的权威图索引，不能为了缩短 prompt 而删减。
        LLM 只需看到当前节点、直接子节点、近期路径和少量语义参照；所有已
        占用 ID 则以短字符串索引传递，用于降低重复生成的概率。
        """
        cfg = self.planner if isinstance(self.planner, dict) else {}
        path_limit = self._bounded_int(cfg.get("prompt_recent_path_limit"), 4, 1, 12)
        guard_limit = self._bounded_int(cfg.get("prompt_semantic_guard_limit"), 20, 0, 80)
        current_id = str(parent_stage.get("id", "") or "").strip()
        child_ids = self._candidate_ids_for_stage(parent_stage, self.window_size)

        # planned_graph 可能包含前瞻节点；仅取已经走到 current 的实际路径。
        path_ids = self._normalize_graph_ids(
            self.planned_graph[: min(max(0, int(self.stage_index)) + 1, len(self.planned_graph))]
        )
        recent_path = [
            self._stage_summary_for_prompt(self.stage_catalog[stage_id])
            for stage_id in path_ids[-path_limit:]
            if stage_id in self.stage_catalog
        ]
        existing_children = [
            copy.deepcopy(self.stage_catalog[stage_id])
            for stage_id in child_ids
            if stage_id in self.stage_catalog
        ]

        # 优先使用近期新增节点作为语义防重参照；当前节点、路径和直接子节点
        # 已通过其他字段完整表达，无须重复。
        excluded_ids = {current_id, *child_ids, *path_ids}
        semantic_guard: List[Dict[str, Any]] = []
        for stage in reversed(list(self.stage_catalog.values())):
            if len(semantic_guard) >= guard_limit:
                break
            stage_id = str(stage.get("id", "") or "").strip()
            if not stage_id or stage_id in excluded_ids:
                continue
            semantic_guard.append(self._stage_summary_for_prompt(stage))

        return {
            "snapshot_version": 1,
            "mode": str(mode or "branch_seed"),
            "current_stage": copy.deepcopy(parent_stage),
            "local_graph": {
                "recent_path": recent_path,
                "existing_children": existing_children,
                "semantic_guard": semantic_guard,
            },
            "known_stage_ids": self._prompt_known_stage_ids(),
        }

    def _build_graph_prompt(
        self,
        mode: str,
        parent_stage: Dict[str, Any],
        session_context: Dict[str, Any],
        conversation_content: str,
        llm_cfg: Optional[Dict[str, Any]],
        candidate_seed: Optional[Dict[str, Any]] = None,
    ) -> str:
        """构造分支规划器 prompt。"""
        cfg = llm_cfg if isinstance(llm_cfg, dict) else {}
        text_limit = self._bounded_int(cfg.get("max_text_length"), 1200, 200, 6000)
        payload = self._build_prompt_graph_snapshot(parent_stage, mode)
        payload.update({
            "existing_next_candidates": self._candidate_ids_for_stage(parent_stage, self.window_size),
            "needed_count": max(
                0,
                self.window_size - len(self._candidate_ids_for_stage(parent_stage, self.window_size)),
            ),
            "window_size": int(self.window_size),
            "candidate_seed": copy.deepcopy(candidate_seed if isinstance(candidate_seed, dict) else {}),
            "session_context": session_context,
            "conversation_content": self._clip_text(conversation_content, limit=text_limit),
        })
        payload_json = json.dumps(payload, ensure_ascii=False)
        return render_prompt(
            "depression/graph_planner",
            {"payload_json": payload_json},
        )

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
        cfg = llm_cfg if isinstance(llm_cfg, dict) else {}
        text_limit = self._bounded_int(cfg.get("max_text_length"), 1200, 200, 6000)
        payload = self._build_prompt_graph_snapshot(parent_stage, "branch_repair")
        payload.update({
            "existing_next_candidates": self._candidate_ids_for_stage(parent_stage, self.window_size),
            "accepted_branch_candidates": copy.deepcopy(accepted_branches),
            "rejected_branch_candidates": copy.deepcopy(rejected_branches),
            "needed_count": max(0, int(needed_count)),
            "window_size": int(self.window_size),
            "session_context": session_context,
            "conversation_content": self._clip_text(conversation_content, limit=text_limit),
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
        core_belief = str(payload.get("core_belief", "") or "").strip()
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
