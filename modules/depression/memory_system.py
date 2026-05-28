"""记忆接口占位实现。"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional


class TraumaMemorySystem:
    """当前版本只保留接口，不做记忆激活逻辑。"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config if isinstance(config, dict) else {}
        self.enabled = bool(self.config.get("enabled", False))
        self.memory_context: List[Dict[str, Any]] = []

    def prepare_memory_context(
        self,
        current_stage: Dict[str, Any],
        session_context: Dict[str, Any],
        conversation_content: str = "",
    ) -> List[Dict[str, Any]]:
        # 目前是占位实现：始终返回空。
        # 因此如果你在阅读时看到 prompt builder 支持 memory layer，
        # 要知道当前工程里这条链路实际上还没有真正启用。
        del current_stage, session_context, conversation_content
        self.memory_context = []
        return []

    def check_memory_activation(
        self,
        current_context: Dict[str, Any],
        conversation_content: str = "",
    ) -> List[Dict[str, Any]]:
        # 为兼容旧接口名而保留，底层仍然走同一个占位实现。
        return self.prepare_memory_context({}, current_context, conversation_content)

    def commit_turn(self, *args, **kwargs) -> None:
        # 未来若真的实现“记忆被触发后需要写回衰减/访问痕迹”，
        # 很可能就是从这个接口开始扩展。
        del args, kwargs
        return None

    def get_activated_thoughts(self) -> List[str]:
        # 当前没有真实 memory_context，因此这里只能返回空列表。
        return []

    def get_physical_reactions(self) -> List[str]:
        # 例如心悸、僵住、胃部紧缩等身体反应，本版本尚未建模。
        return []

    def get_memory_description(self) -> str:
        return "当前版本未启用记忆激活逻辑"

    def add_trauma_memory(self, memory: Dict[str, Any]) -> None:
        del memory
        return None

    def remove_trauma_memory(self, memory_id: str) -> None:
        del memory_id
        return None

    def reduce_memory_intensity(self, memory_id: str, reduction: float = 0.1) -> None:
        del memory_id, reduction
        return None

    def to_dict(self) -> Dict[str, Any]:
        # 即使现在是空实现，也先把序列化接口留好，
        # 这样以后补功能时不用改上层 checkpoint 结构。
        return {
            "config": copy.deepcopy(self.config),
            "enabled": bool(self.enabled),
            "memory_context": copy.deepcopy(self.memory_context),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "TraumaMemorySystem":
        payload = payload if isinstance(payload, dict) else {}
        inst = cls(config=copy.deepcopy(payload.get("config", {})) if isinstance(payload.get("config", {}), dict) else {})
        inst.enabled = bool(payload.get("enabled", inst.enabled))
        context = payload.get("memory_context", [])
        if isinstance(context, list):
            inst.memory_context = [copy.deepcopy(item) for item in context if isinstance(item, dict)]
        return inst
