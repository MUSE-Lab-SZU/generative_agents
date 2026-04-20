"""
动态抑郁症状模拟引擎 - 主引擎类
"""
import copy
from typing import Any, Callable, Dict, List, Optional
from datetime import datetime

from .state_machine import SymptomStateMachine, DepressionState
from .context_analyzer import ContextAnalyzer
from .bias_injector import CognitiveBiasInjector
from .memory_system import TraumaMemorySystem
from .prompt_builder import DynamicPromptBuilder


class DepressionSimulationEngine:
    """
    动态抑郁症状模拟引擎
    
    整合所有子系统，提供统一的接口来模拟抑郁症状的动态表现
    """
    
    def __init__(
        self,
        config: Optional[Dict] = None,
        clock_provider: Optional[Callable[[], datetime]] = None,
    ):
        """
        初始化抑郁症状模拟引擎
        
        Args:
            config: 配置字典，包含各子系统的配置参数
        """
        config = config or {}
        
        # 初始化各子系统
        state_config = config.get("state_machine", {})
        self.state_machine = SymptomStateMachine(
            initial_state=state_config.get("initial_state", DepressionState.SEVERE_EPISODE),
            transition_sensitivity=state_config.get("transition_sensitivity", 0.7),
            minimum_state_duration=state_config.get("minimum_state_duration", 30),
        )
        
        self.context_analyzer = ContextAnalyzer()
        self.bias_injector = CognitiveBiasInjector()
        
        trauma_memories = config.get("trauma_memories") # 加载创伤记忆
        self.memory_system = TraumaMemorySystem(trauma_memories)
        
        self.prompt_builder = DynamicPromptBuilder()
        
        # 配置参数
        self.enabled = config.get("enabled", True)
        self.base_prompt = ""
        
        # 运行时数据
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
        """设置时钟提供器，并同步到状态机。"""
        if callable(clock_provider):
            self._clock_provider = clock_provider
            self.state_machine.set_now_provider(clock_provider)
        
    def set_base_prompt(self, base_prompt: str):
        """设置基础prompt（从agent配置中获取）"""
        self.base_prompt = base_prompt
    
    def process_interaction(self, location: str, time_of_day: str,
                          other_agent: Optional[str] = None,
                          relationship: Optional[str] = None,
                          interaction_type: Optional[str] = None,
                          conversation_content: str = "") -> str:
        """
        处理一次交互，返回动态构建的prompt
        
        Args:
            location: 当前位置
            time_of_day: 时间段 (morning/afternoon/evening/night)
            other_agent: 对方agent名称（如果有）
            relationship: 关系类型（如果有）
            interaction_type: 互动类型（如果有）
            conversation_content: 对话内容
            
        Returns:
            动态构建的完整prompt
        """
        return self._generate_prompt_for_interaction(
            location=location,
            time_of_day=time_of_day,
            other_agent=other_agent,
            relationship=relationship,
            interaction_type=interaction_type,
            conversation_content=conversation_content,
            commit=True,
        )

    def preview_interaction_prompt(self, location: str, time_of_day: str,
                                   other_agent: Optional[str] = None,
                                   relationship: Optional[str] = None,
                                   interaction_type: Optional[str] = None,
                                   conversation_content: str = "") -> str:
        """
        预览某次互动对应的动态prompt，不更新内部症状状态。
        """
        return self._generate_prompt_for_interaction(
            location=location,
            time_of_day=time_of_day,
            other_agent=other_agent,
            relationship=relationship,
            interaction_type=interaction_type,
            conversation_content=conversation_content,
            commit=False,
        )

    def commit_interaction(self, location: str, time_of_day: str,
                           other_agent: Optional[str] = None,
                           relationship: Optional[str] = None,
                           interaction_type: Optional[str] = None,
                           conversation_content: str = "",
                           llm_transition_signal: Optional[Dict[str, Any]] = None) -> Dict:
        """
        提交一次真实互动，更新症状状态与运行时统计，不构建prompt文本。
        llm_transition_signal 为可选辅助评分信号，缺省时走纯规则状态机。
        """
        if not self.enabled:
            return {
                "enabled": False,
                "current_state": self.state_machine.get_current_state(),
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
        transitioned = self.state_machine.update_state(
            context,
            llm_transition_signal=llm_transition_signal,
        )
        current_state = self.state_machine.get_current_state()

        activated_memories = self.memory_system.check_memory_activation(
            context, conversation_content
        )
        cognitive_biases = self.bias_injector.inject_bias(
            current_state, context, conversation_content
        )
        self.last_update_time = self._now()

        return {
            "enabled": True,
            "transitioned": transitioned,
            "current_state": current_state,
            "context": context,
            "activated_memories": [m.get("id") for m in activated_memories],
            "active_biases": [b.get("type") for b in cognitive_biases],
            "interaction_count": self.interaction_count,
            "last_update": self.last_update_time.isoformat(),
        }

    def _generate_prompt_for_interaction(self, location: str, time_of_day: str,
                                         other_agent: Optional[str] = None,
                                         relationship: Optional[str] = None,
                                         interaction_type: Optional[str] = None,
                                         conversation_content: str = "",
                                         commit: bool = False) -> str:
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
            self.state_machine.update_state(context)
            current_state = self.state_machine.get_current_state()
            activated_memories = self.memory_system.check_memory_activation(
                context, conversation_content
            )
            cognitive_biases = self.bias_injector.inject_bias(
                current_state, context, conversation_content
            )
            self.last_update_time = self._now()
        else:
            current_state = self.state_machine.get_current_state()
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
            state_characteristics=state_characteristics,
            context=context,
            cognitive_biases=cognitive_biases,
            activated_memories=activated_memories,
        )

    def _preview_activated_memories(self, context: Dict,
                                    conversation_content: str) -> List[Dict]:
        """计算记忆激活但不持久化到运行时状态。"""
        previous = list(self.memory_system.activated_memories)
        try:
            current = self.memory_system.check_memory_activation(
                context, conversation_content
            )
            return list(current)
        finally:
            self.memory_system.activated_memories = previous

    def _preview_cognitive_biases(self, current_state: str, context: Dict,
                                  conversation_content: str) -> List[Dict]:
        """计算认知偏差但不持久化到运行时状态。"""
        previous = list(self.bias_injector.active_biases)
        try:
            current = self.bias_injector.inject_bias(
                current_state, context, conversation_content
            )
            return list(current)
        finally:
            self.bias_injector.active_biases = previous
    
    def get_simple_prompt(self) -> str:
        """
        获取简化版prompt（用于不需要完整情境分析的场景）
        
        Returns:
            简化的prompt
        """
        if not self.enabled:
            return self.base_prompt
        
        current_state = self.state_machine.get_current_state()
        state_characteristics = self.state_machine.get_state_characteristics()
        
        return self.prompt_builder.build_simple_prompt(
            self.base_prompt, current_state, state_characteristics
        )
    
    def get_current_state_info(self) -> Dict:
        """
        获取当前状态信息（用于调试和监控）
        
        Returns:
            包含当前状态各项信息的字典
        """
        current_state = self.state_machine.get_current_state()
        state_characteristics = self.state_machine.get_state_characteristics()
        
        return {
            "enabled": self.enabled,
            "current_state": current_state,
            "state_duration_minutes": self.state_machine.get_state_duration(),
            "state_characteristics": state_characteristics,
            "active_biases": self.bias_injector.active_biases,
            "activated_memories": [m["id"] for m in self.memory_system.activated_memories],
            "interaction_count": self.interaction_count,
            "last_update": self.last_update_time.isoformat(),
        }
    
    def force_state_transition(self, new_state: str, reason: str = "manual"):
        """
        强制转换到指定状态（用于测试或特殊情况）
        
        Args:
            new_state: 目标状态
            reason: 转换原因
        """
        self.state_machine.force_transition(new_state, reason)
    
    def get_symptom_intensity(self, symptom: str) -> float:
        """
        获取特定症状的当前强度
        
        Args:
            symptom: 症状名称
            
        Returns:
            症状强度 (0-1)
        """
        # 根据当前时间确定时间段
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
        """
        模拟治疗进展（降低创伤记忆强度）
        
        Args:
            effectiveness: 治疗效果 (0-1)
        """
        for memory in self.memory_system.trauma_memories:
            self.memory_system.reduce_memory_intensity(
                memory["id"], 
                reduction=effectiveness
            )
    
    def get_state_history(self) -> List[Dict]:
        """获取状态转换历史"""
        return self.state_machine.get_state_history()
    
    def reset(self):
        """重置引擎状态（用于测试）"""
        self.interaction_count = 0
        self.last_update_time = self._now()
        self.state_machine.accumulated_triggers.clear()
        self.bias_injector.active_biases.clear()
        self.memory_system.activated_memories.clear()
    
    @classmethod
    def from_config_file(cls, config_path: str) -> 'DepressionSimulationEngine':
        """
        从配置文件创建引擎实例
        
        Args:
            config_path: 配置文件路径
            
        Returns:
            引擎实例
        """
        import json
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        return cls(config.get("depression_simulation", {}))

    def to_dict(self) -> Dict[str, Any]:
        """导出可序列化运行状态。"""
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
        """从序列化状态恢复运行时。"""
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
        """从序列化数据构建引擎实例。"""
        payload = payload or {}
        init_cfg = {"enabled": bool(payload.get("enabled", True))}
        engine = cls(config=init_cfg, clock_provider=clock_provider)
        engine.load_state(copy.deepcopy(payload))
        return engine
