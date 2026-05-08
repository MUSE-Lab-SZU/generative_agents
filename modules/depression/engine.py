"""
动态抑郁模块主引擎。

当前版本已由“症状状态转换”改为“主诉认知路线图推进”，但对外仍尽量保留
DepressionSimulationEngine 的既有接口，以减少外围代码改动。
"""

from __future__ import annotations

import copy
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from .bias_injector import CognitiveBiasInjector
from .context_analyzer import ContextAnalyzer
from .memory_system import TraumaMemorySystem
from .prompt_builder import DynamicPromptBuilder
from .state_machine import DepressionState, SymptomStateMachine


class DepressionSimulationEngine:
    """动态抑郁引擎：由主诉路线图驱动表现层。"""

    def __init__(
        self,
        config: Optional[Dict] = None,
        clock_provider: Optional[Callable[[], datetime]] = None,
    ):
        config = config or {}

        roadmap_config = config.get("complaint_roadmap", {})
        if not isinstance(roadmap_config, dict) or not roadmap_config:
            roadmap_config = config.get("state_machine", {})
        if not isinstance(roadmap_config, dict):
            roadmap_config = {}

        self.state_machine = SymptomStateMachine(
            initial_state=roadmap_config.get("initial_state", DepressionState.SEVERE_EPISODE),
            transition_sensitivity=roadmap_config.get("transition_sensitivity", 0.7),
            minimum_state_duration=roadmap_config.get("minimum_state_duration", 0),
            window_size=roadmap_config.get("window_size", 2),
            initial_stage=roadmap_config.get("initial_stage"),
            seed_chain=roadmap_config.get("seed_chain"),
            max_dialog_history=roadmap_config.get("max_dialog_history", 12),
        )

        self.context_analyzer = ContextAnalyzer()
        self.bias_injector = CognitiveBiasInjector()

        trauma_memories = config.get("trauma_memories")
        self.memory_system = TraumaMemorySystem(trauma_memories)

        self.prompt_builder = DynamicPromptBuilder()

        self.enabled = config.get("enabled", True)
        self.base_prompt = ""

        self.interaction_count = 0
        self._clock_provider: Callable[[], datetime] = (
            clock_provider if callable(clock_provider) else datetime.now
        )
        self.last_update_time = self._now()
        self.state_machine.set_now_provider(self._clock_provider)

    def _now(self) -> datetime:
        try:
            now_obj = self._clock_provider()
        except Exception:
            now_obj = datetime.now()
        if isinstance(now_obj, datetime):
            return now_obj
        return datetime.now()

    def set_clock_provider(self, clock_provider: Optional[Callable[[], datetime]]) -> None:
        if callable(clock_provider):
            self._clock_provider = clock_provider
            self.state_machine.set_now_provider(clock_provider)

    def set_base_prompt(self, base_prompt: str):
        self.base_prompt = base_prompt

    def process_interaction(
        self,
        location: str,
        time_of_day: str,
        other_agent: Optional[str] = None,
        relationship: Optional[str] = None,
        interaction_type: Optional[str] = None,
        conversation_content: str = "",
    ) -> str:
        return self._generate_prompt_for_interaction(
            location=location,
            time_of_day=time_of_day,
            other_agent=other_agent,
            relationship=relationship,
            interaction_type=interaction_type,
            conversation_content=conversation_content,
            commit=True,
        )

    def preview_interaction_prompt(
        self,
        location: str,
        time_of_day: str,
        other_agent: Optional[str] = None,
        relationship: Optional[str] = None,
        interaction_type: Optional[str] = None,
        conversation_content: str = "",
    ) -> str:
        return self._generate_prompt_for_interaction(
            location=location,
            time_of_day=time_of_day,
            other_agent=other_agent,
            relationship=relationship,
            interaction_type=interaction_type,
            conversation_content=conversation_content,
            commit=False,
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
    ) -> Dict:
        """
        提交一次真实互动，更新路线图状态与运行时统计。

        兼容说明：
        - llm_transition_signal 参数保留，但现在可承载“路线图识别/预测”结果。
        - roadmap_completion_func + roadmap_llm_cfg 是新的主诉链 LLM 推演入口。
        """
        if not self.enabled:
            return {
                "enabled": False,
                "current_state": self.state_machine.get_current_state(),
                "current_stage": self.state_machine.get_current_stage(),
                "context": {},
            }

        context = self.context_analyzer.analyze_full_context(
            location=location,
            time_of_day=time_of_day,
            other_agent=other_agent,
            relationship=relationship,
            interaction_type=interaction_type,
            conversation_content=conversation_content,
        )

        self.interaction_count += 1
        advanced = self.state_machine.update_state(
            context,
            llm_transition_signal=llm_transition_signal,
            conversation_content=conversation_content,
            roadmap_context={
                "base_prompt": self.base_prompt,
                "location": location,
                "time_of_day": time_of_day,
                "other_agent": other_agent,
                "relationship": relationship,
                "interaction_type": interaction_type,
            },
            roadmap_completion_func=roadmap_completion_func,
            roadmap_llm_cfg=roadmap_llm_cfg,
        )
        current_state = self.state_machine.get_current_state()
        current_stage = self.state_machine.get_current_stage()
        roadmap_snapshot = self.state_machine.get_roadmap_snapshot()

        activated_memories = self.memory_system.check_memory_activation(
            context, conversation_content
        )
        cognitive_biases = self.bias_injector.inject_bias(
            current_state, context, conversation_content
        )
        self.last_update_time = self._now()

        return {
            "enabled": True,
            "advanced": advanced,
            "transitioned": advanced,
            "current_state": current_state,
            "current_stage": current_stage,
            "roadmap": roadmap_snapshot,
            "is_recovered": bool(roadmap_snapshot.get("reached_terminal_recovery", False)),
            "context": context,
            "activated_memories": [m.get("id") for m in activated_memories],
            "active_biases": [b.get("type") for b in cognitive_biases],
            "interaction_count": self.interaction_count,
            "last_update": self.last_update_time.isoformat(),
        }

    def _generate_prompt_for_interaction(
        self,
        location: str,
        time_of_day: str,
        other_agent: Optional[str] = None,
        relationship: Optional[str] = None,
        interaction_type: Optional[str] = None,
        conversation_content: str = "",
        commit: bool = False,
    ) -> str:
        if not self.enabled:
            return self.base_prompt

        context = self.context_analyzer.analyze_full_context(
            location=location,
            time_of_day=time_of_day,
            other_agent=other_agent,
            relationship=relationship,
            interaction_type=interaction_type,
            conversation_content=conversation_content,
        )

        if commit:
            self.interaction_count += 1
            self.state_machine.update_state(
                context,
                conversation_content=conversation_content,
                roadmap_context={
                    "base_prompt": self.base_prompt,
                    "location": location,
                    "time_of_day": time_of_day,
                    "other_agent": other_agent,
                    "relationship": relationship,
                    "interaction_type": interaction_type,
                },
            )
            current_state = self.state_machine.get_current_state()
            current_stage = self.state_machine.get_current_stage()
            roadmap_snapshot = self.state_machine.get_roadmap_snapshot()
            activated_memories = self.memory_system.check_memory_activation(
                context, conversation_content
            )
            cognitive_biases = self.bias_injector.inject_bias(
                current_state, context, conversation_content
            )
            self.last_update_time = self._now()
        else:
            current_state = self.state_machine.get_current_state()
            current_stage = self.state_machine.get_current_stage()
            roadmap_snapshot = self.state_machine.get_roadmap_snapshot()
            activated_memories = self._preview_activated_memories(
                context, conversation_content
            )
            cognitive_biases = self._preview_cognitive_biases(
                current_state, context, conversation_content
            )

        state_characteristics = self.state_machine.get_state_characteristics()

        return self.prompt_builder.build_prompt(
            base_prompt=self.base_prompt,
            current_state=current_state,
            current_stage=current_stage,
            roadmap_snapshot=roadmap_snapshot,
            state_characteristics=state_characteristics,
            context=context,
            cognitive_biases=cognitive_biases,
            activated_memories=activated_memories,
        )

    def _preview_activated_memories(
        self,
        context: Dict,
        conversation_content: str,
    ) -> List[Dict]:
        previous = list(self.memory_system.activated_memories)
        try:
            current = self.memory_system.check_memory_activation(
                context, conversation_content
            )
            return list(current)
        finally:
            self.memory_system.activated_memories = previous

    def _preview_cognitive_biases(
        self,
        current_state: str,
        context: Dict,
        conversation_content: str,
    ) -> List[Dict]:
        previous = list(self.bias_injector.active_biases)
        try:
            current = self.bias_injector.inject_bias(
                current_state, context, conversation_content
            )
            return list(current)
        finally:
            self.bias_injector.active_biases = previous

    def get_simple_prompt(self) -> str:
        if not self.enabled:
            return self.base_prompt

        current_state = self.state_machine.get_current_state()
        current_stage = self.state_machine.get_current_stage()
        roadmap_snapshot = self.state_machine.get_roadmap_snapshot()
        state_characteristics = self.state_machine.get_state_characteristics()

        return self.prompt_builder.build_simple_prompt(
            self.base_prompt,
            current_state,
            state_characteristics,
            current_stage=current_stage,
            roadmap_snapshot=roadmap_snapshot,
        )

    def get_current_state_info(self) -> Dict:
        current_state = self.state_machine.get_current_state()
        current_stage = self.state_machine.get_current_stage()
        state_characteristics = self.state_machine.get_state_characteristics()
        roadmap_snapshot = self.state_machine.get_roadmap_snapshot()

        return {
            "enabled": self.enabled,
            "current_state": current_state,
            "current_stage": current_stage,
            "roadmap": roadmap_snapshot,
            "state_duration_minutes": self.state_machine.get_state_duration(),
            "state_characteristics": state_characteristics,
            "active_biases": self.bias_injector.active_biases,
            "activated_memories": [m["id"] for m in self.memory_system.activated_memories],
            "interaction_count": self.interaction_count,
            "last_update": self.last_update_time.isoformat(),
        }

    def force_state_transition(self, new_state: Any, reason: str = "manual"):
        self.state_machine.force_transition(new_state, reason)

    def get_symptom_intensity(self, symptom: str) -> float:
        hour = self._now().hour
        if 5 <= hour < 12:
            time_of_day = "morning"
        elif 12 <= hour < 18:
            time_of_day = "afternoon"
        elif 18 <= hour < 22:
            time_of_day = "evening"
        else:
            time_of_day = "night"

        return self.state_machine.get_symptom_intensity(symptom, time_of_day)

    def simulate_therapy_progress(self, effectiveness: float = 0.1):
        for memory in self.memory_system.trauma_memories:
            self.memory_system.reduce_memory_intensity(
                memory["id"],
                reduction=effectiveness,
            )

    def get_state_history(self) -> List[Dict]:
        return self.state_machine.get_state_history()

    def reset(self):
        self.interaction_count = 0
        self.last_update_time = self._now()
        self.state_machine.accumulated_triggers.clear()
        self.state_machine.dialogue_history.clear()
        self.bias_injector.active_biases.clear()
        self.memory_system.activated_memories.clear()

    @classmethod
    def from_config_file(cls, config_path: str) -> "DepressionSimulationEngine":
        import json

        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        return cls(config.get("depression_simulation", {}))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": bool(self.enabled),
            "base_prompt": str(self.base_prompt or ""),
            "interaction_count": int(self.interaction_count or 0),
            "last_update_time": self.last_update_time.isoformat(),
            "state_machine": self.state_machine.to_dict(),
            "context_analyzer": self.context_analyzer.to_dict(),
            "bias_injector": self.bias_injector.to_dict(),
            "memory_system": self.memory_system.to_dict(),
        }

    def load_state(self, payload: Dict[str, Any]) -> None:
        payload = payload or {}
        self.enabled = bool(payload.get("enabled", self.enabled))
        self.base_prompt = str(payload.get("base_prompt", self.base_prompt or "") or "")
        try:
            self.interaction_count = int(payload.get("interaction_count", 0) or 0)
        except Exception:
            self.interaction_count = 0

        last_update = payload.get("last_update_time")
        if isinstance(last_update, str) and last_update.strip():
            try:
                self.last_update_time = datetime.fromisoformat(last_update)
            except Exception:
                self.last_update_time = self._now()
        else:
            self.last_update_time = self._now()

        state_payload = payload.get("state_machine", {})
        if isinstance(state_payload, dict):
            self.state_machine = SymptomStateMachine.from_dict(
                state_payload,
                now_provider=self._clock_provider,
            )
        else:
            self.state_machine.set_now_provider(self._clock_provider)

        context_payload = payload.get("context_analyzer", {})
        if isinstance(context_payload, dict):
            self.context_analyzer = ContextAnalyzer.from_dict(context_payload)

        bias_payload = payload.get("bias_injector", {})
        if isinstance(bias_payload, dict):
            self.bias_injector = CognitiveBiasInjector.from_dict(bias_payload)

        memory_payload = payload.get("memory_system", {})
        if isinstance(memory_payload, dict):
            self.memory_system = TraumaMemorySystem.from_dict(memory_payload)

    @classmethod
    def from_dict(
        cls,
        payload: Dict[str, Any],
        clock_provider: Optional[Callable[[], datetime]] = None,
    ) -> "DepressionSimulationEngine":
        payload = payload or {}
        init_cfg = {"enabled": bool(payload.get("enabled", True))}
        engine = cls(config=init_cfg, clock_provider=clock_provider)
        engine.load_state(copy.deepcopy(payload))
        return engine
