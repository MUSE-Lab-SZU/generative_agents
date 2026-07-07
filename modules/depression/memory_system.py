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

    def commit_turn(self, *args, **kwargs) -> None:
        # 未来若真的实现“记忆被触发后需要写回衰减/访问痕迹”，
        # 很可能就是从这个接口开始扩展。
        del args, kwargs
        return None

    def to_dict(self) -> Dict[str, Any]:
        # 即使现在是空实现，也先把序列化接口留好，
        # 这样以后补功能时不用改上层 checkpoint 结构。
        return {
            "memory_context": copy.deepcopy(self.memory_context),
        }
