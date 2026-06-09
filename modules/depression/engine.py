"""主诉图化后的动态抑郁模块主引擎。"""

from __future__ import annotations

import copy
import json
import os
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Union

from .context_analyzer import SessionContextBuilder
from .emotion_inferencer import EmotionInferencer
from .memory_system import TraumaMemorySystem
from .prompt_builder import DynamicPromptBuilder
from .state_machine import ComplaintGraphManager


def _simulation_now() -> datetime:
    """Return the active simulated-world time, falling back to wall time only outside simulation."""
    try:
        from modules import utils

        return utils.get_timer().get_date()
    except Exception:
        return datetime.now()


class DepressionSimulationEngine:
    """由主诉图驱动的动态抑郁表现引擎。

    可以把本类理解成一个“编排器”：
    - `SessionContextBuilder`：先把对话场景整理成结构化上下文；
    - `ComplaintGraphManager`：判断当前是否仍处在同一主诉节点，是否推进；
    - `EmotionInferencer`：给出本轮瞬时情绪；
    - `DynamicPromptBuilder`：把以上内容拼回 prompt。

    最重要的接口有两个：
    - `preview_interaction_prompt()`：只预览，不落盘状态；
    - `commit_interaction()`：真正提交本轮并推进内部状态。
    """

    def __init__(
        self,
        config: Optional[Union[Dict[str, Any], str]] = None,
        clock_provider: Optional[Callable[[], datetime]] = None,
    ):
        """解析配置并初始化主诉图、上下文、记忆、情绪和 prompt 子系统。"""
        resolved = self._resolve_config(config)
        self.raw_config = copy.deepcopy(resolved)
        self.config_path = str(resolved.get("_config_path", "") or "")
        self.agent_dir = str(resolved.get("_agent_dir", "") or "")

        self.enabled = bool(resolved.get("enabled", True))
        self.base_prompt = ""
        self.interaction_count = 0
        self.last_emotion: Dict[str, Any] = {}
        self.last_session_context: Dict[str, Any] = {}

        self._clock_provider: Callable[[], datetime] = (
            clock_provider if callable(clock_provider) else _simulation_now
        )
        self.last_update_time = self._now()

        agent_name = self._infer_agent_name(resolved)
        self.graph_manager = ComplaintGraphManager(resolved, now_provider=self._clock_provider)
        self.context_builder = SessionContextBuilder(self_name=agent_name)
        self.context_analyzer = self.context_builder  # 兼容旧字段名
        self.memory_system = TraumaMemorySystem(resolved.get("memory", {}))
        self.emotion_inferencer = EmotionInferencer(resolved.get("emotion", {}))
        self.prompt_builder = DynamicPromptBuilder(resolved.get("prompt", {}))

    def _now(self) -> datetime:
        """返回当前引擎时间，时间源异常或返回值非法时回退到模拟时钟。"""
        try:
            now_obj = self._clock_provider()
        except Exception:
            now_obj = _simulation_now()
        return now_obj if isinstance(now_obj, datetime) else _simulation_now()

    def set_clock_provider(self, clock_provider: Optional[Callable[[], datetime]]) -> None:
        """替换引擎时间源，并同步给主诉图管理器。"""
        if callable(clock_provider):
            self._clock_provider = clock_provider
            self.graph_manager.set_now_provider(clock_provider)

    def set_base_prompt(self, base_prompt: str) -> None:
        """设置所有动态抑郁信息附着前的基础角色 prompt。"""
        self.base_prompt = str(base_prompt or "")

    def process_interaction(
        self,
        location: str,
        time_of_day: str,
        other_agent: Optional[str] = None,
        relationship: Optional[str] = None,
        interaction_type: Optional[str] = None,
        conversation_content: str = "",
    ) -> str:
        """提交一轮已发生互动并返回注入抑郁运行态后的完整 prompt。"""
        # 这个接口会直接 commit，因此更适合“确认发生过的互动”；
        # 如果只是为了生成 prompt 预览，优先看 `preview_interaction_prompt()`。
        runtime = self.commit_interaction(
            location=location,
            time_of_day=time_of_day,
            other_agent=other_agent,
            relationship=relationship,
            interaction_type=interaction_type,
            conversation_content=conversation_content,
        )
        if not runtime.get("enabled", False):
            return self.base_prompt
        return self.prompt_builder.build_prompt(
            base_prompt=self.base_prompt,
            current_stage=runtime.get("current_stage", {}),
            graph_snapshot=runtime.get("graph", {}),
            session_context=runtime.get("session_context", {}),
            activated_memories=runtime.get("memory_context", []),
            emotion=runtime.get("emotion", {}),
        )

    def preview_interaction_prompt(
        self,
        location: str,
        time_of_day: str,
        other_agent: Optional[str] = None,
        relationship: Optional[str] = None,
        interaction_type: Optional[str] = None,
        conversation_content: str = "",
        llm_transition_signal: Optional[Dict[str, Any]] = None,
        roadmap_completion_func: Optional[Callable[[str], str]] = None,
        roadmap_llm_cfg: Optional[Dict[str, Any]] = None,
        emotion_completion_func: Optional[Callable[[str], str]] = None,
    ) -> str:
        """预览一轮互动会生成的动态 prompt，但不写入真实主诉图状态。"""
        if not self.enabled:
            return self.base_prompt

        # 第 1 步：把调用方传来的碎片信息整理成统一的 session_context。
        session_context = self.context_builder.build_context(
            location=location,
            time_of_day=time_of_day,
            other_agent=other_agent,
            relationship=relationship,
            interaction_type=interaction_type,
            conversation_content=conversation_content,
        )
        # 第 2 步：先在当前状态上做“如果这轮发生，会怎样”的评估。
        evaluation = self.graph_manager.evaluate_turn(
            session_context=session_context,
            conversation_content=conversation_content,
            completion_func=roadmap_completion_func,
            llm_cfg=roadmap_llm_cfg,
            llm_signal=llm_transition_signal,
        )
        # 第 3 步：克隆一个 manager 做 preview commit，避免污染真实状态。
        preview_manager = ComplaintGraphManager.from_dict(
            self.graph_manager.to_dict(), now_provider=self._clock_provider
        )
        preview_graph = preview_manager.commit_turn(copy.deepcopy(evaluation))
        current_stage = preview_manager.get_current_stage()
        # 第 4 步：在 preview 后的节点上推断记忆和瞬时情绪。
        memory_context = self.memory_system.prepare_memory_context(current_stage, session_context, conversation_content)
        emotion = self._infer_emotion(
            current_stage=current_stage,
            graph_snapshot=preview_graph,
            session_context=session_context,
            conversation_content=conversation_content,
            completion_func=emotion_completion_func or roadmap_completion_func,
        )
        return self.prompt_builder.build_prompt(
            base_prompt=self.base_prompt,
            current_stage=current_stage,
            graph_snapshot=preview_graph,
            session_context=session_context,
            activated_memories=memory_context,
            emotion=emotion,
        )

    def commit_interaction(
        self,
        location: str,
        time_of_day: str,
        other_agent: Optional[str] = None,
        relationship: Optional[str] = None,
        interaction_type: Optional[str] = None,
        conversation_content: str = "",
        llm_transition_signal: Optional[Dict[str, Any]] = None,
        roadmap_completion_func: Optional[Callable[[str], str]] = None,
        roadmap_llm_cfg: Optional[Dict[str, Any]] = None,
        emotion_completion_func: Optional[Callable[[str], str]] = None,
    ) -> Dict[str, Any]:
        """提交一轮对话互动，推进主诉图并返回本轮完整运行态快照。"""
        if not self.enabled:
            return self._disabled_runtime()

        # commit 版本与 preview 共享同一条流水线，
        # 差别在于这里会真正修改 `graph_manager` / `interaction_count`。
        session_context = self.context_builder.build_context(
            location=location,
            time_of_day=time_of_day,
            other_agent=other_agent,
            relationship=relationship,
            interaction_type=interaction_type,
            conversation_content=conversation_content,
        )
        return self._commit_context(
            session_context=session_context,
            conversation_content=conversation_content,
            llm_transition_signal=llm_transition_signal,
            roadmap_completion_func=roadmap_completion_func,
            roadmap_llm_cfg=roadmap_llm_cfg,
            emotion_completion_func=emotion_completion_func,
        )

    def commit_event(
        self,
        source: str,
        location: str,
        time_of_day: str,
        interaction_type: str,
        content: str,
        other_agent: Optional[str] = None,
        relationship: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        llm_transition_signal: Optional[Dict[str, Any]] = None,
        roadmap_completion_func: Optional[Callable[[str], str]] = None,
        roadmap_llm_cfg: Optional[Dict[str, Any]] = None,
        emotion_completion_func: Optional[Callable[[str], str]] = None,
    ) -> Dict[str, Any]:
        """提交一个会影响主诉图的非对话运行时事件。

        事件仍复用 interaction 管线，只是在 session_context 中额外标记
        `runtime_event`，让状态记录能区分对话与反思等来源。
        """
        if not self.enabled:
            return self._disabled_runtime()

        event_source = str(source or "").strip()
        event_content = str(content or "").strip()
        if not event_source or not event_content:
            return self._current_runtime(enabled=True)

        session_context = self.context_builder.build_context(
            location=location,
            time_of_day=time_of_day,
            other_agent=other_agent,
            relationship=relationship,
            interaction_type=interaction_type,
            conversation_content=event_content,
        )
        self._attach_runtime_event(
            session_context=session_context,
            source=event_source,
            metadata=metadata,
        )
        return self._commit_context(
            session_context=session_context,
            conversation_content=event_content,
            llm_transition_signal=llm_transition_signal,
            roadmap_completion_func=roadmap_completion_func,
            roadmap_llm_cfg=roadmap_llm_cfg,
            emotion_completion_func=emotion_completion_func,
            disallow_jump=(event_source == "reflection"),
        )

    def _commit_context(
        self,
        session_context: Dict[str, Any],
        conversation_content: str,
        llm_transition_signal: Optional[Dict[str, Any]] = None,
        roadmap_completion_func: Optional[Callable[[str], str]] = None,
        roadmap_llm_cfg: Optional[Dict[str, Any]] = None,
        emotion_completion_func: Optional[Callable[[str], str]] = None,
        disallow_jump: bool = False,
    ) -> Dict[str, Any]:
        """执行提交流水线：评估主诉图、落盘状态、推断记忆和情绪。"""
        if not self.enabled:
            return self._disabled_runtime()

        session_context = session_context if isinstance(session_context, dict) else {}
        conversation_content = str(conversation_content or "")
        stage_catalog_before = (
            copy.deepcopy(self.graph_manager.stage_catalog)
            if disallow_jump
            else None
        )
        evaluation = self.graph_manager.evaluate_turn(
            session_context=session_context,
            conversation_content=conversation_content,
            completion_func=roadmap_completion_func,
            llm_cfg=roadmap_llm_cfg,
            llm_signal=llm_transition_signal,
        )
        if disallow_jump and str(evaluation.get("action", "") or "").strip().lower() == "jump":
            if isinstance(stage_catalog_before, dict):
                self.graph_manager.stage_catalog = stage_catalog_before
            evaluation = self._downgrade_jump_evaluation(evaluation)
        graph_snapshot = self.graph_manager.commit_turn(evaluation)
        if callable(roadmap_completion_func):
            graph_snapshot = self.graph_manager.ensure_graph_window(
                session_context=session_context,
                conversation_content=conversation_content,
                completion_func=roadmap_completion_func,
                llm_cfg=roadmap_llm_cfg,
                record_evaluation=False,
            )
        current_stage = self.graph_manager.get_current_stage()

        memory_context = self.memory_system.prepare_memory_context(current_stage, session_context, conversation_content)
        emotion = self._infer_emotion(
            current_stage=current_stage,
            graph_snapshot=graph_snapshot,
            session_context=session_context,
            conversation_content=conversation_content,
            completion_func=emotion_completion_func or roadmap_completion_func,
        )
        self.memory_system.commit_turn(current_stage=current_stage, session_context=session_context)

        self.last_emotion = copy.deepcopy(emotion)
        self.last_session_context = copy.deepcopy(session_context)
        self.interaction_count += 1
        self.last_update_time = self._now()

        return {
            "enabled": True,
            "current_stage": current_stage,
            "graph": graph_snapshot,
            "evaluation": copy.deepcopy(evaluation),
            "session_context": session_context,
            "emotion": emotion,
            "memory_context": memory_context,
            "interaction_count": self.interaction_count,
            "last_update": self.last_update_time.isoformat(),
        }

    def _disabled_runtime(self) -> Dict[str, Any]:
        """构造模块禁用时的最小运行态快照。"""
        return {
            "enabled": False,
            "current_stage": self.graph_manager.get_current_stage(),
            "graph": self.graph_manager.get_graph_snapshot(),
            "session_context": {},
        }

    def _current_runtime(self, enabled: bool = True) -> Dict[str, Any]:
        """返回当前已缓存的运行态快照，不触发任何状态推进。"""
        return {
            "enabled": bool(enabled),
            "current_stage": self.graph_manager.get_current_stage(),
            "graph": self.graph_manager.get_graph_snapshot(),
            "session_context": copy.deepcopy(self.last_session_context),
            "emotion": copy.deepcopy(self.last_emotion),
            "interaction_count": self.interaction_count,
            "last_update": self.last_update_time.isoformat(),
        }

    def initialize_graph_window(
        self,
        location: str = "",
        time_of_day: str = "",
        interaction_type: str = "主诉图初始化",
        roadmap_completion_func: Optional[Callable[[str], str]] = None,
        roadmap_llm_cfg: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """启动或恢复时用 LLM 补足运行态主诉图窗口，不改变当前 stage。"""
        if not self.enabled:
            return self._disabled_runtime()
        if not callable(roadmap_completion_func):
            return self._current_runtime(enabled=True)
        session_context = self.context_builder.build_context(
            location=location,
            time_of_day=time_of_day,
            other_agent="",
            relationship="",
            interaction_type=interaction_type,
            conversation_content="初始化主诉图窗口：请基于当前主诉节点，规划自然、保守、可推进的后续主诉图节点。",
        )
        graph_snapshot = self.graph_manager.initialize_graph_window(
            session_context=session_context,
            conversation_content=session_context.get("conversation", {}).get("content", ""),
            completion_func=roadmap_completion_func,
            llm_cfg=roadmap_llm_cfg,
        )
        self.last_session_context = copy.deepcopy(session_context)
        self.last_update_time = self._now()
        return {
            "enabled": True,
            "current_stage": self.graph_manager.get_current_stage(),
            "graph": graph_snapshot,
            "session_context": session_context,
            "interaction_count": self.interaction_count,
            "last_update": self.last_update_time.isoformat(),
        }

    def _attach_runtime_event(
        self,
        session_context: Dict[str, Any],
        source: str,
        metadata: Optional[Dict[str, Any]],
    ) -> None:
        """把标准化后的运行时事件挂到 session_context，并同步上下文缓存。"""
        payload = self._normalize_runtime_event(source=source, metadata=metadata)
        session_context["runtime_event"] = payload
        self.context_builder.current_context = copy.deepcopy(session_context)
        if self.context_builder.context_history:
            self.context_builder.context_history[-1] = copy.deepcopy(session_context)

    def _normalize_runtime_event(
        self,
        source: str,
        metadata: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """清洗事件来源和证据 ID，生成可安全写入 session_context 的事件载荷。"""
        meta = copy.deepcopy(metadata) if isinstance(metadata, dict) else {}
        evidence_ids = meta.get("evidence_ids", [])
        if not isinstance(evidence_ids, list):
            evidence_ids = [evidence_ids] if evidence_ids else []
        normalized_evidence = []
        seen = set()
        for item in evidence_ids:
            text = str(item or "").strip()
            if not text or text in seen:
                continue
            seen.add(text)
            normalized_evidence.append(text[:80])
            if len(normalized_evidence) >= 20:
                break
        meta["evidence_ids"] = normalized_evidence
        return {
            "source": str(source or "").strip()[:32],
            "metadata": meta,
            "evidence_ids": normalized_evidence,
        }

    def _downgrade_jump_evaluation(self, evaluation: Dict[str, Any]) -> Dict[str, Any]:
        """把不允许跳转场景中的 jump 评估降级为 hold，并保留当前窗口。"""
        payload = copy.deepcopy(evaluation if isinstance(evaluation, dict) else {})
        if str(payload.get("action", "") or "").strip().lower() != "jump":
            return payload
        current_graph = self.graph_manager.get_current_graph_window(
            getattr(self.graph_manager, "window_size", 3) + 1
        )
        payload["action"] = "hold"
        payload["next_stage"] = None
        payload["stage_updates"] = []
        payload["next_graph"] = [
            str(stage.get("id", "") or "")
            for stage in current_graph
            if isinstance(stage, dict) and str(stage.get("id", "") or "").strip()
        ]
        payload["next_graph"] = copy.deepcopy(payload["next_graph"])
        reason = str(payload.get("match_reason", "") or "").strip()
        suffix = "reflection_disallows_jump"
        payload["match_reason"] = "{}; {}".format(reason, suffix) if reason else suffix
        return payload

    def _infer_emotion(
        self,
        current_stage: Dict[str, Any],
        graph_snapshot: Dict[str, Any],
        session_context: Dict[str, Any],
        conversation_content: str,
        completion_func: Optional[Callable[[str], str]] = None,
    ) -> Dict[str, Any]:
        """基于当前主诉节点、对话上下文和上一轮情绪推断本轮瞬时情绪。"""
        # 情绪推断刻意使用“上一轮情绪 + 本轮上下文”的组合，
        # 目的是让说话状态连续变化，而不是每轮都从头随机生成。
        payload = {
            "current_stage": copy.deepcopy(current_stage),
            "graph_snapshot": copy.deepcopy(graph_snapshot),
            "session_context": copy.deepcopy(session_context),
            "conversation_content": str(conversation_content or ""),
            "previous_emotion": copy.deepcopy(self.last_emotion),
            "turn_key": f"{self.interaction_count}|{session_context.get('scene', {}).get('time_of_day', '')}",
        }
        return self.emotion_inferencer.infer(payload, completion_func=completion_func)

    def get_simple_prompt(self) -> str:
        """生成只包含基础 prompt、当前主诉节点和主诉图摘要的简版 prompt。"""
        if not self.enabled:
            return self.base_prompt
        return self.prompt_builder.build_simple_prompt(
            base_prompt=self.base_prompt,
            current_stage=self.graph_manager.get_current_stage(),
            graph_snapshot=self.graph_manager.get_graph_snapshot(),
        )

    def get_current_state_info(self) -> Dict[str, Any]:
        """返回当前引擎状态概览，供调试、存档或前端展示使用。"""
        return {
            "enabled": self.enabled,
            "current_stage": self.graph_manager.get_current_stage(),
            "graph": self.graph_manager.get_graph_snapshot(),
            "state_duration_minutes": self.graph_manager.get_state_duration(),
            "memory_context": copy.deepcopy(self.memory_system.memory_context),
            "emotion": copy.deepcopy(self.last_emotion),
            "interaction_count": self.interaction_count,
            "last_update": self.last_update_time.isoformat(),
        }

    def force_stage(
        self,
        stage_id: str,
        reason: str = "manual",
        roadmap_completion_func: Optional[Callable[[str], str]] = None,
        roadmap_llm_cfg: Optional[Dict[str, Any]] = None,
    ) -> None:
        """强制切换到指定主诉节点，并可选择让 LLM 重新补足后续图窗口。"""
        self.graph_manager.force_stage(stage_id, reason=reason)
        if callable(roadmap_completion_func):
            session_context = self.context_builder.build_context(
                location="",
                time_of_day="",
                other_agent="",
                relationship="",
                interaction_type="手动主诉图切换",
                conversation_content=str(reason or "manual"),
            )
            self.graph_manager.ensure_graph_window(
                session_context=session_context,
                conversation_content=str(reason or "manual"),
                completion_func=roadmap_completion_func,
                llm_cfg=roadmap_llm_cfg,
                record_evaluation=False,
            )
        self.last_update_time = self._now()

    def force_state_transition(self, new_state: Any, reason: str = "manual") -> None:
        """强制切换当前主诉节点，支持传入 stage 字典或字符串。"""
        if isinstance(new_state, dict):
            stage_id = str(new_state.get("id", new_state.get("label", "")) or "").strip()
            if stage_id:
                self.graph_manager.force_stage(stage_id, reason=reason)
        else:
            self.graph_manager.force_stage(str(new_state or "").strip(), reason=reason)
        self.last_update_time = self._now()

    def get_state_history(self) -> List[Dict[str, Any]]:
        """读取主诉图管理器记录的历史状态变化。"""
        return self.graph_manager.get_state_history()

    def get_symptom_intensity(self, symptom: str) -> float:
        """查询当前主诉节点上某个症状的强度值。"""
        return self.graph_manager.get_symptom_intensity(symptom)

    def simulate_therapy_progress(self, effectiveness: float = 0.1) -> None:
        """保留的治疗进展模拟接口；当前版本不改变任何状态。"""
        del effectiveness
        return None

    def reset(self) -> None:
        """重置交互计数、缓存情绪、上下文、主诉图和记忆运行态。"""
        self.interaction_count = 0
        self.last_emotion = {}
        self.last_session_context = {}
        self.last_update_time = self._now()
        self.graph_manager.reset()
        self.memory_system.memory_context = []

    @classmethod
    def from_config_file(cls, config_path: str) -> "DepressionSimulationEngine":
        """从 depression_config.json 路径创建引擎实例。"""
        return cls(config=config_path)

    @classmethod
    def from_agent_dir(cls, agent_dir: str) -> "DepressionSimulationEngine":
        """从 agent 目录创建引擎实例，默认读取其中的 depression_config.json。"""
        agent_dir = os.path.abspath(str(agent_dir or ""))
        config_path = os.path.join(agent_dir, "depression_config.json")
        return cls(config={"config_path": config_path, "agent_dir": agent_dir})

    def to_dict(self) -> Dict[str, Any]:
        """序列化引擎配置引用和各子系统运行态，用于存档或迁移。"""
        config_reference = self._state_config_reference()
        payload = {
            "config_reference": copy.deepcopy(config_reference),
            "enabled": bool(self.enabled),
            "interaction_count": int(self.interaction_count),
            "last_update_time": self.last_update_time.isoformat(),
            "last_emotion": copy.deepcopy(self.last_emotion),
            "last_session_context": copy.deepcopy(self.last_session_context),
            "complaint_graph_manager": self.graph_manager.to_dict(),
        }
        if not (config_reference.get("config_path") or config_reference.get("agent_dir")) and isinstance(self.raw_config, dict) and self.raw_config:
            payload["inline_config"] = copy.deepcopy(self.raw_config)
        memory_payload = self.memory_system.to_dict()
        if memory_payload.get("memory_context"):
            payload["memory_system"] = memory_payload
        return payload

    def load_state(self, payload: Dict[str, Any]) -> None:
        """从序列化结果恢复引擎状态。

        阅读时可重点留意：
        1. 配置会被重新解析；
        2. graph/context/memory 都会分别恢复；
        3. emotion/prompt_builder 会按最新配置重新实例化。
        """
        payload = payload if isinstance(payload, dict) else {}
        config = self._select_state_config(payload)
        refreshed = self._resolve_config(config)
        self.raw_config = copy.deepcopy(refreshed)
        self.config_path = str(refreshed.get("_config_path", "") or "")
        self.agent_dir = str(refreshed.get("_agent_dir", "") or "")
        self.enabled = bool(payload.get("enabled", refreshed.get("enabled", True)))
        self.base_prompt = str(payload.get("base_prompt", self.base_prompt or "") or "")
        try:
            self.interaction_count = int(payload.get("interaction_count", 0) or 0)
        except Exception:
            self.interaction_count = 0
        # `last_update_time` is a runtime freshness marker, not a historical event.
        # Refresh it with the current simulated clock so old wall-time checkpoints self-correct on resume.
        self.last_update_time = self._now()

        self.last_emotion = copy.deepcopy(payload.get("last_emotion", {})) if isinstance(payload.get("last_emotion", {}), dict) else {}
        self.last_session_context = copy.deepcopy(payload.get("last_session_context", {})) if isinstance(payload.get("last_session_context", {}), dict) else {}

        graph_payload = (
            payload.get("complaint_graph_manager", {})
            if isinstance(payload.get("complaint_graph_manager", {}), dict)
            else {}
        )
        base_graph_config = (
            refreshed.get("complaint_graph", {})
            if isinstance(refreshed.get("complaint_graph", {}), dict)
            else {}
        )
        self.graph_manager = ComplaintGraphManager.from_dict(
            graph_payload,
            now_provider=self._clock_provider,
            base_config=base_graph_config,
        )

        self.context_builder = SessionContextBuilder(self_name=self._infer_agent_name(refreshed))
        if self.last_session_context:
            self.context_builder.current_context = copy.deepcopy(self.last_session_context)
            self.context_builder.context_history = [copy.deepcopy(self.last_session_context)]
        self.context_builder.set_self_name(self._infer_agent_name(refreshed))
        self.context_analyzer = self.context_builder

        memory_payload = payload.get("memory_system", {}) if isinstance(payload.get("memory_system", {}), dict) else {}
        self.memory_system = TraumaMemorySystem(refreshed.get("memory", {}))
        memory_context = memory_payload.get("memory_context", [])
        if isinstance(memory_context, list):
            self.memory_system.memory_context = [
                copy.deepcopy(item) for item in memory_context if isinstance(item, dict)
            ]

        self.emotion_inferencer = EmotionInferencer(refreshed.get("emotion", {}))
        self.prompt_builder = DynamicPromptBuilder(refreshed.get("prompt", {}))

    @classmethod
    def from_dict(
        cls,
        payload: Dict[str, Any],
        clock_provider: Optional[Callable[[], datetime]] = None,
    ) -> "DepressionSimulationEngine":
        """从序列化 payload 创建引擎，并恢复其中保存的运行态。"""
        payload = payload if isinstance(payload, dict) else {}
        config = payload.get("config_reference", {})
        if not isinstance(config, dict) or not (config.get("config_path") or config.get("agent_dir")):
            config = payload.get("inline_config", {})
        engine = cls(config=config, clock_provider=clock_provider)
        engine.load_state(copy.deepcopy(payload))
        return engine

    @staticmethod
    def _load_json_file(path: str) -> Dict[str, Any]:
        """读取 JSON 配置文件；文件内容不是对象时返回空字典。"""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}

    @classmethod
    def _resolve_config(cls, config: Optional[Union[Dict[str, Any], str]]) -> Dict[str, Any]:
        """解析配置来源，支持路径、配置字典、agent 目录和嵌套配置结构。"""
        if isinstance(config, str):
            config_path = os.path.abspath(config)
            data = cls._load_json_file(config_path)
            data["_config_path"] = config_path
            data["_agent_dir"] = os.path.dirname(config_path)
            return data

        config = config if isinstance(config, dict) else {}
        config_path_value = config.get("config_path", config.get("_config_path", ""))
        if isinstance(config_path_value, str) and config_path_value.strip():
            config_path = os.path.abspath(str(config_path_value or "").strip())
            if os.path.isfile(config_path):
                data = cls._load_json_file(config_path)
                data["_config_path"] = config_path
                data["_agent_dir"] = os.path.dirname(config_path)
                return data
        agent_dir = str(config.get("agent_dir", config.get("_agent_dir", "")) or "").strip()
        if agent_dir:
            config_path = os.path.join(os.path.abspath(agent_dir), "depression_config.json")
            if os.path.isfile(config_path):
                data = cls._load_json_file(config_path)
                data["_config_path"] = config_path
                data["_agent_dir"] = os.path.dirname(config_path)
                return data

        data = copy.deepcopy(config)
        if "depression_simulation" in data and isinstance(data.get("depression_simulation"), dict):
            nested = copy.deepcopy(data.get("depression_simulation", {}))
            nested.setdefault("_config_path", str(data.get("_config_path", "") or ""))
            nested.setdefault("_agent_dir", str(data.get("_agent_dir", "") or ""))
            return nested
        return data

    def _state_config_reference(self) -> Dict[str, Any]:
        """生成轻量配置引用，避免序列化时把完整配置重复写入状态。"""
        ref: Dict[str, Any] = {}
        if self.config_path:
            ref["config_path"] = str(self.config_path)
        if self.agent_dir:
            ref["agent_dir"] = str(self.agent_dir)
        agent_name = self._infer_agent_name(self.raw_config)
        if agent_name:
            ref["agent_name"] = agent_name
        return ref

    def _select_state_config(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """选择恢复状态时应使用的配置引用，优先保留当前实例的路径信息。"""
        current_ref = self._state_config_reference()
        if current_ref.get("config_path") or current_ref.get("agent_dir"):
            return current_ref

        for key in ("config_reference", "config"):
            value = payload.get(key, {})
            if isinstance(value, dict) and (
                value.get("config_path")
                or value.get("agent_dir")
                or value.get("complaint_graph")
                or value.get("depression_simulation")
            ):
                return copy.deepcopy(value)

        inline_config = payload.get("inline_config", {})
        if isinstance(inline_config, dict) and inline_config:
            return copy.deepcopy(inline_config)

        return copy.deepcopy(self.raw_config if isinstance(self.raw_config, dict) and self.raw_config else current_ref)

    def _infer_agent_name(self, config: Dict[str, Any]) -> str:
        """从显式字段或 agent 目录名推断当前角色名。"""
        explicit = str(config.get("agent_name", "") or config.get("name", "") or "").strip()
        if explicit:
            return explicit
        agent_dir = str(config.get("_agent_dir", "") or self.agent_dir or "").strip()
        if agent_dir:
            return os.path.basename(agent_dir)
        return ""
