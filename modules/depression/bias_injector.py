"""主诉节点驱动的认知偏差注入器。"""

from __future__ import annotations

import copy
import random
from typing import Any, Dict, List, Optional

from .prompt_templates import load_prompt_json


class ComplaintBiasInjector:
    """根据当前主诉节点与会话上下文选择认知偏差。"""

    PROMPT_CONFIG: Dict[str, Any] = load_prompt_json("depression/depression_prompt_config", {})
    DEFAULT_BIASES: Dict[str, Dict[str, Any]] = (
        PROMPT_CONFIG.get("bias_library", {}) if isinstance(PROMPT_CONFIG.get("bias_library", {}), dict) else {}
    )

    def __init__(
        self,
        library_override: Optional[Dict[str, Any]] = None,
        selection_policy: Optional[Dict[str, Any]] = None,
    ):
        base_library = self.DEFAULT_BIASES if isinstance(self.DEFAULT_BIASES, dict) else {}
        self.bias_library = copy.deepcopy(base_library)
        if isinstance(library_override, dict):
            for key, value in library_override.items():
                if not isinstance(value, dict):
                    continue
                merged = copy.deepcopy(self.bias_library.get(key, {}))
                merged.update(copy.deepcopy(value))
                self.bias_library[str(key)] = merged
        self.selection_policy = selection_policy if isinstance(selection_policy, dict) else {}
        self.active_biases: List[str] = []

    def inject_bias(
        self,
        current_stage: Dict[str, Any],
        session_context: Dict[str, Any],
        conversation_content: str = "",
    ) -> List[Dict[str, Any]]:
        # dominant 偏差直接优先激活；
        # secondary 偏差则要求当前会话上下文“支持”它出现。
        current_stage = current_stage if isinstance(current_stage, dict) else {}
        session_context = session_context if isinstance(session_context, dict) else {}
        conversation = str(conversation_content or "")

        bias_profile = current_stage.get("bias_profile", {}) if isinstance(current_stage.get("bias_profile", {}), dict) else {}
        dominant = [str(item) for item in self._to_list(bias_profile.get("dominant", [])) if str(item).strip()]
        secondary = [str(item) for item in self._to_list(bias_profile.get("secondary", [])) if str(item).strip()]
        max_active = self._bounded_int(
            bias_profile.get("max_active"),
            self.selection_policy.get("default_max_active", 2),
            1,
            5,
        )

        selected: List[str] = []
        selected.extend([item for item in dominant if item in self.bias_library])

        for bias_type in secondary:
            if bias_type not in self.bias_library:
                continue
            if self._conversation_supports_bias(bias_type, conversation, session_context, current_stage):
                selected.append(bias_type)

        relationship = self._relationship(session_context)
        if relationship in {"家人", "父母"} and "personalization" in self.bias_library:
            selected.append("personalization")
        if relationship == "治疗师" and self._is_help_context(session_context):
            if "should_statements" in self.bias_library:
                selected.append("should_statements")
        if self._has_self_negation(conversation):
            selected.append("all_or_nothing")

        selected = self._dedupe_keep_order(selected)
        selected = selected[:max_active]
        self.active_biases = list(selected)

        source_stage_id = str(current_stage.get("id", "") or "")
        stage_templates = bias_profile.get("thought_templates", {}) if isinstance(bias_profile.get("thought_templates", {}), dict) else {}
        outputs: List[Dict[str, Any]] = []
        for index, bias_type in enumerate(selected):
            # 输出结果除了 thought 文本，还保留 stage 来源和 confidence，
            # 便于后续 prompt 展示与人工审查。
            thought = self._generate_biased_thought(
                bias_type=bias_type,
                current_stage=current_stage,
                stage_templates=stage_templates,
                conversation_content=conversation,
            )
            confidence = self._estimate_confidence(
                bias_type=bias_type,
                index=index,
                dominant=dominant,
                conversation_content=conversation,
                session_context=session_context,
            )
            outputs.append(
                {
                    "type": bias_type,
                    "name": self.bias_library.get(bias_type, {}).get("name", bias_type),
                    "thought": thought,
                    "source_stage_id": source_stage_id,
                    "confidence": round(float(confidence), 4),
                }
            )
        return outputs

    def get_bias_description(self, biases: List[Dict[str, Any]]) -> str:
        if not biases:
            return "当前未突出显现明显的自动化认知偏差"
        rows = ["当前活跃的认知偏差："]
        for item in biases:
            rows.append(f"- {item.get('name', item.get('type', '未知偏差'))}：{item.get('thought', '')}")
        return "\n".join(rows)

    def should_activate_bias(self, bias_type: str, conversation_topic: str) -> bool:
        return self._conversation_supports_bias(
            bias_type=str(bias_type or "").strip(),
            conversation_content=str(conversation_topic or ""),
            session_context={},
            current_stage={},
        )

    def _conversation_supports_bias(
        self,
        bias_type: str,
        conversation_content: str,
        session_context: Dict[str, Any],
        current_stage: Dict[str, Any],
    ) -> bool:
        # 这里同样是规则化设计：关键词 / topic / help-context。
        bias_meta = self.bias_library.get(str(bias_type or "").strip(), {})
        keywords = [str(item) for item in self._to_list(bias_meta.get("cue_keywords", []))]
        conversation = str(conversation_content or "")
        if any(keyword and keyword in conversation for keyword in keywords):
            return True

        semantic = session_context.get("semantic_cues", {}) if isinstance(session_context.get("semantic_cues", {}), dict) else {}
        topics = [str(item) for item in self._to_list(semantic.get("topics", []))]
        if bias_type in {"all_or_nothing", "personalization"} and any(topic in {"自我否定", "工作挫败", "学业受挫"} for topic in topics):
            return True
        if bias_type in {"catastrophizing", "fortune_telling"} and "未来无望" in topics:
            return True
        if bias_type == "should_statements" and self._is_help_context(session_context):
            return True
        if bias_type == "emotional_reasoning" and any(topic in {"疲惫停滞", "一般低落叙述"} for topic in topics):
            return True

        focus = [str(item) for item in self._to_list(current_stage.get("narrative_focus", []))]
        return any(item and item in conversation for item in focus[:2]) and bias_type in {"mental_filter", "emotional_reasoning"}

    def _generate_biased_thought(
        self,
        bias_type: str,
        current_stage: Dict[str, Any],
        stage_templates: Dict[str, Any],
        conversation_content: str,
    ) -> str:
        templates: List[str] = []
        if isinstance(stage_templates.get(bias_type), list):
            templates = [str(item) for item in stage_templates.get(bias_type, []) if str(item).strip()]
        if not templates:
            templates = [str(item) for item in self._to_list(self.bias_library.get(bias_type, {}).get("templates", [])) if str(item).strip()]
        if templates:
            return random.choice(templates)
        core_belief = str(current_stage.get("core_belief", "") or "").strip()
        if core_belief:
            return core_belief
        conversation = str(conversation_content or "").strip()
        if conversation:
            return f"我会不由自主地把“{conversation[:20]}”往更糟糕的方向理解。"
        return "我会不由自主地把事情解释成对自己更不利的样子。"

    def _estimate_confidence(
        self,
        bias_type: str,
        index: int,
        dominant: List[str],
        conversation_content: str,
        session_context: Dict[str, Any],
    ) -> float:
        confidence = 0.58
        if bias_type in dominant:
            confidence += 0.20
        if self._conversation_supports_bias(bias_type, conversation_content, session_context, {}):
            confidence += 0.08
        confidence -= min(0.10, 0.03 * float(index))
        return max(0.30, min(0.95, confidence))

    def _relationship(self, session_context: Dict[str, Any]) -> str:
        participants = session_context.get("participants", {}) if isinstance(session_context.get("participants", {}), dict) else {}
        return str(participants.get("relationship", "") or "").strip()

    def _is_help_context(self, session_context: Dict[str, Any]) -> bool:
        flags = session_context.get("session_flags", {}) if isinstance(session_context.get("session_flags", {}), dict) else {}
        return bool(flags.get("is_help_seeking_frame", False) or flags.get("is_professional_frame", False))

    @staticmethod
    def _has_self_negation(conversation_content: str) -> bool:
        conversation = str(conversation_content or "")
        return any(token in conversation for token in ["我不行", "我就是失败", "我没用", "整个人都", "彻底失败"])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "active_biases": [str(item) for item in self.active_biases],
            "selection_policy": copy.deepcopy(self.selection_policy),
            "bias_library": copy.deepcopy(self.bias_library),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ComplaintBiasInjector":
        payload = payload if isinstance(payload, dict) else {}
        inst = cls(
            library_override=payload.get("bias_library", {}),
            selection_policy=payload.get("selection_policy", {}),
        )
        active = payload.get("active_biases", [])
        if isinstance(active, list):
            inst.active_biases = [str(item) for item in active if str(item).strip()]
        return inst

    @staticmethod
    def _to_list(value: Any) -> List[Any]:
        if value is None:
            return []
        if isinstance(value, list):
            return list(value)
        return [value]

    @staticmethod
    def _bounded_int(value: Any, default: int, lower: int, upper: int) -> int:
        try:
            num = int(value)
        except Exception:
            num = int(default)
        num = max(int(lower), num)
        num = min(int(upper), num)
        return num

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


CognitiveBiasInjector = ComplaintBiasInjector
