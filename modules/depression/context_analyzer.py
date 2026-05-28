"""会话上下文整理器。"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional


class SessionContextBuilder:
    """将当前一轮互动整理成主诉链可消费的会话上下文。"""

    TOPIC_RULES = {
        "工作挫败": ["工作", "辞职", "辞退", "公司", "绩效", "老板", "上班", "失业", "开除"],
        "学业受挫": ["学习", "考试", "成绩", "论文", "毕业", "不及格", "导师"],
        "自我否定": ["没用", "失败", "废物", "不行", "负担", "价值", "意义", "拖累"],
        "关系疏离": ["朋友", "家人", "理解", "离开", "孤独", "排斥", "疏远", "没人"],
        "未来无望": ["未来", "前途", "以后", "希望", "不会好", "没希望", "看不到"],
        "疲惫停滞": ["累", "撑不住", "不想动", "疲惫", "睡不着", "发呆", "空掉"],
        "求助摇摆": ["帮助", "求助", "咨询", "治疗", "要不要", "能不能", "算了", "没必要"],
    }

    def __init__(self, self_name: str = ""):
        self.self_name = str(self_name or "").strip()
        self.current_context: Dict[str, Any] = {}
        self.context_history: List[Dict[str, Any]] = []

    def set_self_name(self, self_name: str) -> None:
        self.self_name = str(self_name or "").strip()

    def build_context(
        self,
        location: str,
        time_of_day: str,
        other_agent: Optional[str] = None,
        relationship: Optional[str] = None,
        interaction_type: Optional[str] = None,
        conversation_content: str = "",
    ) -> Dict[str, Any]:
        # 这里是一个明显的“规则系统”，不是语义理解模型：
        # 所有 topic / speech_act / stance 都来自关键词匹配。
        conversation = str(conversation_content or "").strip()
        scene = {
            "location": str(location or "").strip(),
            "time_of_day": str(time_of_day or "").strip(),
            "interaction_type": str(interaction_type or "").strip(),
        }
        participants = {
            "self_name": self.self_name,
            "other_agent": str(other_agent or "").strip(),
            "relationship": str(relationship or "").strip(),
        }
        semantic_cues = self._build_semantic_cues(
            conversation_content=conversation,
            relationship=participants["relationship"],
            interaction_type=scene["interaction_type"],
        )
        session_flags = self._build_session_flags(
            relationship=participants["relationship"],
            interaction_type=scene["interaction_type"],
            conversation_content=conversation,
            semantic_cues=semantic_cues,
        )
        context = {
            "scene": scene,
            "participants": participants,
            "conversation": {
                "content": conversation,
                "excerpt": self._clip_text(conversation, limit=220),
                "turn_length": len(conversation),
            },
            "session_flags": session_flags,
            "semantic_cues": semantic_cues,
        }
        self.current_context = copy.deepcopy(context)
        self.context_history.append(copy.deepcopy(context))
        if len(self.context_history) > 100:
            self.context_history = self.context_history[-100:]
        return context

    def analyze_full_context(
        self,
        location: str,
        time_of_day: str,
        other_agent: Optional[str] = None,
        relationship: Optional[str] = None,
        interaction_type: Optional[str] = None,
        conversation_content: str = "",
    ) -> Dict[str, Any]:
        return self.build_context(
            location=location,
            time_of_day=time_of_day,
            other_agent=other_agent,
            relationship=relationship,
            interaction_type=interaction_type,
            conversation_content=conversation_content,
        )

    def get_context_description(self) -> str:
        if not self.current_context:
            return "当前没有可用的会话上下文"
        scene = self.current_context.get("scene", {}) if isinstance(self.current_context.get("scene", {}), dict) else {}
        participants = self.current_context.get("participants", {}) if isinstance(self.current_context.get("participants", {}), dict) else {}
        semantic = self.current_context.get("semantic_cues", {}) if isinstance(self.current_context.get("semantic_cues", {}), dict) else {}
        topics = "、".join([str(item) for item in semantic.get("topics", [])[:3]]) or "未识别主题"
        return "地点：{}；对象：{}；关系：{}；主题：{}".format(
            scene.get("location", "未知"),
            participants.get("other_agent", "未知"),
            participants.get("relationship", "未知"),
            topics,
        )

    def _build_semantic_cues(
        self,
        conversation_content: str,
        relationship: str,
        interaction_type: str,
    ) -> Dict[str, Any]:
        # 语义线索的目标不是精确 NLP，而是给后续状态机提供
        # “足够稳定、可解释”的粗粒度标签。
        conversation = str(conversation_content or "")
        topics: List[str] = []
        for topic, keywords in self.TOPIC_RULES.items():
            if any(keyword in conversation for keyword in keywords):
                topics.append(topic)
        if not topics:
            topics.append("一般低落叙述")

        speech_acts: List[str] = []
        if any(token in conversation for token in ["我觉得", "我最近", "我一直", "我很", "我现在"]):
            speech_acts.append("自我暴露")
        if any(token in conversation for token in ["其实", "比如", "因为", "后来", "那天", "具体"]):
            speech_acts.append("具体叙述")
        if any(token in conversation for token in ["帮", "能不能", "要不要", "想聊", "想说", "该怎么办"]):
            speech_acts.append("求助尝试")
        if any(token in conversation for token in ["算了", "不用", "没事", "还好", "只是有点"]):
            speech_acts.append("淡化")
        if any(token in conversation for token in ["不想说", "不太想", "没必要", "懒得"]):
            speech_acts.append("回避")
        if any(token in conversation for token in ["也许", "可能", "不知道", "好像"]):
            speech_acts.append("含蓄求助")
        if not speech_acts:
            speech_acts.append("简短应答")

        stance: List[str] = []
        if any(token in conversation for token in ["有点", "也许", "可能", "不知道"]):
            stance.append("试探")
        if any(token in conversation for token in ["没事", "算了", "不用", "还好"]):
            stance.append("保留")
        if any(token in conversation for token in ["失败", "没用", "废物", "丢脸", "羞耻"]):
            stance.append("羞耻")
        if any(token in conversation for token in ["不想", "别问", "烦", "没必要"]):
            stance.append("防御")
        if any(token in conversation for token in ["累", "空", "麻木", "没劲"]):
            stance.append("低落")
        if relationship == "治疗师" or interaction_type == "治疗对话":
            stance.append("被评估感")
        if not stance:
            stance.append("谨慎")

        return {
            "topics": topics[:6],
            "speech_acts": self._dedupe_keep_order(speech_acts)[:6],
            "stance": self._dedupe_keep_order(stance)[:6],
        }

    def _build_session_flags(
        self,
        relationship: str,
        interaction_type: str,
        conversation_content: str,
        semantic_cues: Dict[str, Any],
    ) -> Dict[str, Any]:
        speech_acts = [str(item) for item in semantic_cues.get("speech_acts", [])]
        return {
            "is_help_seeking_frame": (
                interaction_type in {"寻求帮助", "治疗对话"}
                or relationship == "治疗师"
                or "求助尝试" in speech_acts
                or "含蓄求助" in speech_acts
            ),
            "is_evaluative_frame": (
                interaction_type in {"被询问状况", "治疗对话"}
                or relationship in {"治疗师", "家人", "父母"}
            ),
            "is_close_relationship": relationship in {"亲密朋友", "朋友", "家人", "父母", "伴侣"},
            "is_professional_frame": relationship == "治疗师" or interaction_type == "治疗对话",
            "is_minimizing": any(token in conversation_content for token in ["没事", "还好", "只是有点", "算了"]),
            "is_withdrawing": any(token in conversation_content for token in ["不想说", "不太想", "没必要", "懒得"]),
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "self_name": self.self_name,
            "current_context": copy.deepcopy(self.current_context),
            "context_history": copy.deepcopy(self.context_history[-100:]),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "SessionContextBuilder":
        payload = payload if isinstance(payload, dict) else {}
        inst = cls(self_name=str(payload.get("self_name", "") or ""))
        inst.current_context = copy.deepcopy(payload.get("current_context", {})) if isinstance(payload.get("current_context", {}), dict) else {}
        history = payload.get("context_history", [])
        if isinstance(history, list):
            inst.context_history = [copy.deepcopy(item) for item in history if isinstance(item, dict)][-100:]
        return inst

    @staticmethod
    def _clip_text(value: Any, limit: int = 220) -> str:
        text = str(value or "").strip()
        if len(text) <= int(limit):
            return text
        return text[: max(0, int(limit) - 1)] + "…"

    @staticmethod
    def _dedupe_keep_order(values: List[str]) -> List[str]:
        results: List[str] = []
        seen = set()
        for item in values:
            text = str(item or "").strip()
            if not text or text in seen:
                continue
            seen.add(text)
            results.append(text)
        return results


ContextAnalyzer = SessionContextBuilder
