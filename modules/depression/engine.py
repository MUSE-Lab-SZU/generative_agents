"""主诉链化后的动态抑郁模块主引擎。"""

from __future__ import annotations

import copy
import json
import os
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Union

from .bias_injector import ComplaintBiasInjector
from .context_analyzer import SessionContextBuilder
from .emotion_inferencer import EmotionInferencer
from .memory_system import TraumaMemorySystem
from .prompt_builder import DynamicPromptBuilder
from .state_machine import ComplaintChainManager


class DepressionSimulationEngine:
    """由主诉链驱动的动态抑郁表现引擎。

    可以把本类理解成一个“编排器”：
    - `SessionContextBuilder`：先把对话场景整理成结构化上下文；
    - `ComplaintChainManager`：判断当前是否仍处在同一主诉节点，是否推进；
    - `ComplaintBiasInjector`：决定本轮要显性呈现哪些认知偏差；
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
        resolved = self._resolve_config(config)
        self.raw_config = copy.deepcopy(resolved)
        self.config_path = str(resolved.get("_config_path", "") or "")
        self.agent_dir = str(resolved.get("_agent_dir", "") or "")

        self.enabled = bool(resolved.get("enabled", True))
        self.profile = resolved.get("profile", {}) if isinstance(resolved.get("profile", {}), dict) else {}
        self.base_prompt = ""
        self.interaction_count = 0
        self.last_emotion: Dict[str, Any] = {}
        self.last_session_context: Dict[str, Any] = {}

        self._clock_provider: Callable[[], datetime] = (
            clock_provider if callable(clock_provider) else datetime.now
        )
        self.last_update_time = self._now()

        agent_name = self._infer_agent_name(resolved)
        self.chain_manager = ComplaintChainManager(resolved, now_provider=self._clock_provider)
        self.state_machine = self.chain_manager  # 兼容旧字段名
        self.context_builder = SessionContextBuilder(self_name=agent_name)
        self.context_analyzer = self.context_builder  # 兼容旧字段名
        self.bias_injector = ComplaintBiasInjector(
            library_override=((resolved.get("bias", {}) or {}).get("library_override", {}) if isinstance(resolved.get("bias", {}), dict) else {}),
            selection_policy=((resolved.get("bias", {}) or {}).get("selection_policy", {}) if isinstance(resolved.get("bias", {}), dict) else {}),
        )
        self.memory_system = TraumaMemorySystem(resolved.get("memory", {}))
        self.emotion_inferencer = EmotionInferencer(resolved.get("emotion", {}))
        self.prompt_builder = DynamicPromptBuilder(resolved.get("prompt", {}))

    def _now(self) -> datetime:
        try:
            now_obj = self._clock_provider()
        except Exception:
            now_obj = datetime.now()
        return now_obj if isinstance(now_obj, datetime) else datetime.now()

    def set_clock_provider(self, clock_provider: Optional[Callable[[], datetime]]) -> None:
        if callable(clock_provider):
            self._clock_provider = clock_provider
            self.chain_manager.set_now_provider(clock_provider)

    def set_base_prompt(self, base_prompt: str) -> None:
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
            chain_snapshot=runtime.get("chain", {}),
            session_context=runtime.get("session_context", {}),
            cognitive_biases=runtime.get("biases", []),
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
        evaluation = self.chain_manager.evaluate_turn(
            session_context=session_context,
            conversation_content=conversation_content,
            completion_func=roadmap_completion_func,
            llm_cfg=roadmap_llm_cfg,
            llm_signal=llm_transition_signal,
        )
        # 第 3 步：克隆一个 manager 做 preview commit，避免污染真实状态。
        preview_manager = ComplaintChainManager.from_dict(
            self.chain_manager.to_dict(), now_provider=self._clock_provider
        )
        preview_chain = preview_manager.commit_turn(copy.deepcopy(evaluation))
        current_stage = preview_manager.get_current_stage()
        # 第 4 步：在 preview 后的节点上推断偏差、记忆和瞬时情绪。
        biases = self.bias_injector.inject_bias(current_stage, session_context, conversation_content)
        memory_context = self.memory_system.prepare_memory_context(current_stage, session_context, conversation_content)
        emotion = self._infer_emotion(
            current_stage=current_stage,
            chain_snapshot=preview_chain,
            session_context=session_context,
            conversation_content=conversation_content,
            completion_func=emotion_completion_func or roadmap_completion_func,
        )
        return self.prompt_builder.build_prompt(
            base_prompt=self.base_prompt,
            current_stage=current_stage,
            chain_snapshot=preview_chain,
            session_context=session_context,
            cognitive_biases=biases,
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
        if not self.enabled:
            return {
                "enabled": False,
                "current_stage": self.chain_manager.get_current_stage(),
                "chain": self.chain_manager.get_chain_snapshot(),
                "session_context": {},
            }

        # commit 版本与 preview 共享同一条流水线，
        # 差别在于这里会真正修改 `chain_manager` / `interaction_count`。
        session_context = self.context_builder.build_context(
            location=location,
            time_of_day=time_of_day,
            other_agent=other_agent,
            relationship=relationship,
            interaction_type=interaction_type,
            conversation_content=conversation_content,
        )
        evaluation = self.chain_manager.evaluate_turn(
            session_context=session_context,
            conversation_content=conversation_content,
            completion_func=roadmap_completion_func,
            llm_cfg=roadmap_llm_cfg,
            llm_signal=llm_transition_signal,
        )
        chain_snapshot = self.chain_manager.commit_turn(evaluation)
        current_stage = self.chain_manager.get_current_stage()

        biases = self.bias_injector.inject_bias(current_stage, session_context, conversation_content)
        memory_context = self.memory_system.prepare_memory_context(current_stage, session_context, conversation_content)
        emotion = self._infer_emotion(
            current_stage=current_stage,
            chain_snapshot=chain_snapshot,
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
            "chain": chain_snapshot,
            "evaluation": copy.deepcopy(evaluation),
            "session_context": session_context,
            "emotion": emotion,
            "biases": biases,
            "active_biases": [item.get("type") for item in biases if isinstance(item, dict)],
            "memory_context": memory_context,
            "interaction_count": self.interaction_count,
            "last_update": self.last_update_time.isoformat(),
        }

    def _infer_emotion(
        self,
        current_stage: Dict[str, Any],
        chain_snapshot: Dict[str, Any],
        session_context: Dict[str, Any],
        conversation_content: str,
        completion_func: Optional[Callable[[str], str]] = None,
    ) -> Dict[str, Any]:
        # 情绪推断刻意使用“上一轮情绪 + 本轮上下文”的组合，
        # 目的是让说话状态连续变化，而不是每轮都从头随机生成。
        payload = {
            "current_stage": copy.deepcopy(current_stage),
            "chain_snapshot": copy.deepcopy(chain_snapshot),
            "session_context": copy.deepcopy(session_context),
            "conversation_content": str(conversation_content or ""),
            "previous_emotion": copy.deepcopy(self.last_emotion),
            "turn_key": f"{self.interaction_count}|{session_context.get('scene', {}).get('time_of_day', '')}",
        }
        return self.emotion_inferencer.infer(payload, completion_func=completion_func)

    def get_simple_prompt(self) -> str:
        if not self.enabled:
            return self.base_prompt
        return self.prompt_builder.build_simple_prompt(
            base_prompt=self.base_prompt,
            current_stage=self.chain_manager.get_current_stage(),
            chain_snapshot=self.chain_manager.get_chain_snapshot(),
        )

    def get_current_state_info(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "current_stage": self.chain_manager.get_current_stage(),
            "chain": self.chain_manager.get_chain_snapshot(),
            "state_duration_minutes": self.chain_manager.get_state_duration(),
            "active_biases": list(self.bias_injector.active_biases),
            "memory_context": copy.deepcopy(self.memory_system.memory_context),
            "emotion": copy.deepcopy(self.last_emotion),
            "interaction_count": self.interaction_count,
            "last_update": self.last_update_time.isoformat(),
        }

    def force_stage(self, stage_id: str, reason: str = "manual") -> None:
        self.chain_manager.force_stage(stage_id, reason=reason)
        self.last_update_time = self._now()

    def force_state_transition(self, new_state: Any, reason: str = "manual") -> None:
        if isinstance(new_state, dict):
            stage_id = str(new_state.get("id", new_state.get("label", "")) or "").strip()
            if stage_id:
                self.chain_manager.force_stage(stage_id, reason=reason)
        else:
            self.chain_manager.force_stage(str(new_state or "").strip(), reason=reason)
        self.last_update_time = self._now()

    def get_state_history(self) -> List[Dict[str, Any]]:
        return self.chain_manager.get_state_history()

    def get_symptom_intensity(self, symptom: str) -> float:
        return self.chain_manager.get_symptom_intensity(symptom)

    def simulate_therapy_progress(self, effectiveness: float = 0.1) -> None:
        del effectiveness
        return None

    def reset(self) -> None:
        self.interaction_count = 0
        self.last_emotion = {}
        self.last_session_context = {}
        self.last_update_time = self._now()
        self.chain_manager.reset()
        self.bias_injector.active_biases.clear()
        self.memory_system.memory_context = []

    @classmethod
    def from_config_file(cls, config_path: str) -> "DepressionSimulationEngine":
        return cls(config=config_path)

    @classmethod
    def from_agent_dir(cls, agent_dir: str) -> "DepressionSimulationEngine":
        agent_dir = os.path.abspath(str(agent_dir or ""))
        config_path = os.path.join(agent_dir, "depression_config.json")
        return cls(config={"config_path": config_path, "agent_dir": agent_dir})

    def to_dict(self) -> Dict[str, Any]:
        config_reference = self._state_config_reference()
        return {
            "config": copy.deepcopy(config_reference),
            "config_reference": copy.deepcopy(config_reference),
            "enabled": bool(self.enabled),
            "base_prompt": str(self.base_prompt or ""),
            "interaction_count": int(self.interaction_count),
            "last_update_time": self.last_update_time.isoformat(),
            "last_emotion": copy.deepcopy(self.last_emotion),
            "last_session_context": copy.deepcopy(self.last_session_context),
            "chain_manager": self.chain_manager.to_dict(),
            "context_builder": self.context_builder.to_dict(),
            "bias_injector": self.bias_injector.to_dict(),
            "memory_system": self.memory_system.to_dict(),
        }

    def load_state(self, payload: Dict[str, Any]) -> None:
        """从序列化结果恢复引擎状态。

        阅读时可重点留意：
        1. 配置会被重新解析；
        2. chain/context/bias/memory 都会分别恢复；
        3. emotion/prompt_builder 会按最新配置重新实例化。
        """
        payload = payload if isinstance(payload, dict) else {}
        config = self._select_state_config(payload)
        refreshed = self._resolve_config(config)
        self.raw_config = copy.deepcopy(refreshed)
        self.config_path = str(refreshed.get("_config_path", "") or "")
        self.agent_dir = str(refreshed.get("_agent_dir", "") or "")
        self.profile = refreshed.get("profile", {}) if isinstance(refreshed.get("profile", {}), dict) else {}
        self.enabled = bool(payload.get("enabled", refreshed.get("enabled", True)))
        self.base_prompt = str(payload.get("base_prompt", self.base_prompt or "") or "")
        try:
            self.interaction_count = int(payload.get("interaction_count", 0) or 0)
        except Exception:
            self.interaction_count = 0
        try:
            self.last_update_time = datetime.fromisoformat(str(payload.get("last_update_time", "") or ""))
        except Exception:
            self.last_update_time = self._now()

        self.last_emotion = copy.deepcopy(payload.get("last_emotion", {})) if isinstance(payload.get("last_emotion", {}), dict) else {}
        self.last_session_context = copy.deepcopy(payload.get("last_session_context", {})) if isinstance(payload.get("last_session_context", {}), dict) else {}

        chain_payload = payload.get("chain_manager", {}) if isinstance(payload.get("chain_manager", {}), dict) else {}
        self.chain_manager = ComplaintChainManager.from_dict(
            chain_payload,
            now_provider=self._clock_provider,
            base_config=refreshed.get("complaint_chain", {}) if isinstance(refreshed.get("complaint_chain", {}), dict) else {},
        )
        self.state_machine = self.chain_manager

        context_payload = payload.get("context_builder", {}) if isinstance(payload.get("context_builder", {}), dict) else {}
        self.context_builder = SessionContextBuilder.from_dict(context_payload)
        self.context_builder.set_self_name(self._infer_agent_name(refreshed))
        self.context_analyzer = self.context_builder

        bias_payload = payload.get("bias_injector", {}) if isinstance(payload.get("bias_injector", {}), dict) else {}
        self.bias_injector = ComplaintBiasInjector.from_dict(bias_payload)

        memory_payload = payload.get("memory_system", {}) if isinstance(payload.get("memory_system", {}), dict) else {}
        self.memory_system = TraumaMemorySystem.from_dict(memory_payload)

        self.emotion_inferencer = EmotionInferencer(refreshed.get("emotion", {}))
        self.prompt_builder = DynamicPromptBuilder(refreshed.get("prompt", {}))

    @classmethod
    def from_dict(
        cls,
        payload: Dict[str, Any],
        clock_provider: Optional[Callable[[], datetime]] = None,
    ) -> "DepressionSimulationEngine":
        payload = payload if isinstance(payload, dict) else {}
        engine = cls(config=payload.get("config", {}), clock_provider=clock_provider)
        engine.load_state(copy.deepcopy(payload))
        return engine

    @staticmethod
    def _load_json_file(path: str) -> Dict[str, Any]:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}

    @classmethod
    def _resolve_config(cls, config: Optional[Union[Dict[str, Any], str]]) -> Dict[str, Any]:
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
        current_ref = self._state_config_reference()
        if current_ref.get("config_path") or current_ref.get("agent_dir"):
            return current_ref

        for key in ("config_reference", "config"):
            value = payload.get(key, {})
            if isinstance(value, dict) and value:
                return copy.deepcopy(value)

        return copy.deepcopy(self.raw_config if isinstance(self.raw_config, dict) and self.raw_config else current_ref)

    def _infer_agent_name(self, config: Dict[str, Any]) -> str:
        explicit = str(config.get("agent_name", "") or config.get("name", "") or self.profile.get("name", "") or "").strip()
        if explicit:
            return explicit
        agent_dir = str(config.get("_agent_dir", "") or self.agent_dir or "").strip()
        if agent_dir:
            return os.path.basename(agent_dir)
        return ""
