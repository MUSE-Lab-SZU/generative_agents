"""generative_agents.utils.namespace"""

from typing import Any, Optional
import copy


class GenerativeAgentsMap:
    """Global Namespace map for Land"""

    # 类变量，作为全局状态容器使用
    MAP = {}

    @classmethod # 直接通过类调用，不用实例化对象
    def set(cls, key: str, value: Any):
        cls.MAP[key] = value

    @classmethod
    def get(cls, key: str, default: Optional[Any] = None):
        return cls.MAP.get(key, default)

    @classmethod
    def clone(cls, key: str, default: Optional[Any] = None):
        return copy.deepcopy(cls.get(key, default))

    @classmethod
    def delete(cls, key: str):
        if key in cls.MAP:
            return cls.MAP.pop(key)
        return None

    @classmethod
    def contains(cls, key: str):
        return key in cls.MAP

    @classmethod
    def reset(cls):
        cls.MAP = {}


class GenerativeAgentsKey:
    """Keys for the LandMap"""

    GAME = "game"
    TIMER = "timer"
    MODELS = "models"
