"""基于主诉链的瞬时说话情绪推断器。"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Callable, Dict, Optional

from .prompt_templates import render_prompt


class EmotionInferencer:
    """推断角色在当前轮对话中的瞬时情绪风格。"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config if isinstance(config, dict) else {}
        self.volatility_limit = self._bounded_float(
            self.config.get("volatility_limit"), 0.12, 0.02, 0.35
        )
        self.llm_enabled = bool(self.config.get("llm_enabled", True))

    def infer(
        self,
        payload: Dict[str, Any],
        completion_func: Optional[Callable[[str], str]] = None,
    ) -> Dict[str, Any]:
        # 默认先构造 fallback，再尝试 LLM；
        # 这样即使 LLM 不可用，系统也仍然能给出连续的瞬时情绪。
        payload = payload if isinstance(payload, dict) else {}
        fallback = self.build_fallback(payload)
        if not callable(completion_func) or not self.llm_enabled:
            return fallback

        raw = ""
        try:
            raw = str(completion_func(self.build_prompt(payload)) or "")
        except Exception:
            raw = ""

        parsed = self._parse_json_object(raw)
        if not isinstance(parsed, dict):
            return fallback
        return self._sanitize(parsed, fallback)

    def build_prompt(self, payload: Dict[str, Any]) -> str:
        current_stage = json.dumps(payload.get("current_stage", {}) or {}, ensure_ascii=False)
        chain_snapshot = json.dumps(payload.get("chain_snapshot", {}) or {}, ensure_ascii=False)
        session_context = json.dumps(payload.get("session_context", {}) or {}, ensure_ascii=False)
        previous_emotion = json.dumps(payload.get("previous_emotion", {}) or {}, ensure_ascii=False)
        conversation_content = self._clip_text(payload.get("conversation_content", ""), limit=1200)
        return render_prompt(
            "depression/emotion_inferencer",
            {
                "current_stage": current_stage,
                "chain_snapshot": chain_snapshot,
                "session_context": session_context,
                "previous_emotion": previous_emotion,
                "conversation_content": conversation_content or "（暂无明确话语内容）",
            },
        )

    def build_fallback(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """基于 stage + relationship + flags 的规则化情绪兜底。"""
        stage = payload.get("current_stage", {}) if isinstance(payload.get("current_stage", {}), dict) else {}
        session_context = payload.get("session_context", {}) if isinstance(payload.get("session_context", {}), dict) else {}
        previous = payload.get("previous_emotion", {}) if isinstance(payload.get("previous_emotion", {}), dict) else {}
        conversation = str(payload.get("conversation_content", "") or "")

        emotion_vector = stage.get("emotion_vector", {}) if isinstance(stage.get("emotion_vector", {}), dict) else {}
        speaking_style = stage.get("speaking_style", {}) if isinstance(stage.get("speaking_style", {}), dict) else {}
        relationship = self._relationship(session_context)
        relation_modifiers = stage.get("relation_modifiers", {}) if isinstance(stage.get("relation_modifiers", {}), dict) else {}
        relation_delta = relation_modifiers.get(relationship, {}) if isinstance(relation_modifiers.get(relationship, {}), dict) else {}

        valence = self._bounded_float(emotion_vector.get("valence"), 0.25, 0.0, 1.0)
        arousal = self._bounded_float(emotion_vector.get("arousal"), 0.35, 0.0, 1.0)
        hopelessness = self._bounded_float(emotion_vector.get("hopelessness"), 0.35, 0.0, 1.0)
        shame = self._bounded_float(emotion_vector.get("shame"), 0.40, 0.0, 1.0)
        trust = self._bounded_float(emotion_vector.get("trust"), 0.20, 0.0, 1.0)
        defensiveness = self._bounded_float(emotion_vector.get("defensiveness"), 0.55, 0.0, 1.0)

        trust += self._bounded_float(relation_delta.get("trust_delta"), 0.0, -0.5, 0.5)
        defensiveness += self._bounded_float(relation_delta.get("defensiveness_delta"), 0.0, -0.5, 0.5)
        disclosure_level = self._disclosure_score(speaking_style.get("disclosure", "guarded"))
        disclosure_level += self._bounded_float(relation_delta.get("disclosure_delta"), 0.0, -0.5, 0.5)

        flags = session_context.get("session_flags", {}) if isinstance(session_context.get("session_flags", {}), dict) else {}
        speech_acts = [str(item) for item in self._to_list(session_context.get("semantic_cues", {}).get("speech_acts", []) if isinstance(session_context.get("semantic_cues", {}), dict) else [])]
        stance = [str(item) for item in self._to_list(session_context.get("semantic_cues", {}).get("stance", []) if isinstance(session_context.get("semantic_cues", {}), dict) else [])]

        if flags.get("is_professional_frame", False):
            trust += 0.05
            disclosure_level += 0.05
        if flags.get("is_evaluative_frame", False):
            defensiveness += 0.05
        if flags.get("is_minimizing", False):
            disclosure_level -= 0.08
            defensiveness += 0.06
        if flags.get("is_withdrawing", False):
            disclosure_level -= 0.10
            defensiveness += 0.08
        if "具体叙述" in speech_acts:
            disclosure_level += 0.08
        if "求助尝试" in speech_acts or "含蓄求助" in speech_acts:
            disclosure_level += 0.06
            trust += 0.04
        if "羞耻" in stance:
            defensiveness += 0.05
        if self._contains_negative_cues(conversation):
            hopelessness += 0.05
            arousal += 0.03
        if self._contains_softening_cues(conversation):
            trust += 0.03

        trust = self._bounded_float(trust, 0.20, 0.0, 1.0)
        defensiveness = self._bounded_float(defensiveness, 0.55, 0.0, 1.0)
        disclosure_level = self._bounded_float(disclosure_level, 0.28, 0.0, 1.0)
        intensity = self._bounded_float(
            (1.0 - valence) * 0.44 + hopelessness * 0.28 + shame * 0.18 + arousal * 0.10,
            0.62,
            0.0,
            1.0,
        )

        previous_initialized = bool(str(previous.get("label", "") or "").strip() or "intensity" in previous)
        previous_intensity = self._bounded_float(previous.get("intensity"), intensity, 0.0, 1.0)
        previous_disclosure = self._bounded_float(previous.get("disclosure_level"), disclosure_level, 0.0, 1.0)
        previous_defensiveness = self._bounded_float(previous.get("defensiveness"), defensiveness, 0.0, 1.0)
        jitter = self._stable_jitter(
            "{}|{}|{}|{}".format(
                stage.get("id", ""),
                relationship,
                payload.get("turn_key", ""),
                conversation[-120:],
            )
        )

        if previous_initialized:
            # 这一段很关键：通过裁剪 delta，让情绪只做“小幅、可逆”波动，
            # 避免模型一轮比一轮跳得太离谱。
            intensity = previous_intensity + self._clip_delta(intensity - previous_intensity + jitter * 0.20, self.volatility_limit)
            disclosure_level = previous_disclosure + self._clip_delta(disclosure_level - previous_disclosure + jitter * 0.12, self.volatility_limit)
            defensiveness = previous_defensiveness + self._clip_delta(defensiveness - previous_defensiveness - jitter * 0.12, self.volatility_limit)

        intensity = self._bounded_float(intensity, 0.62, 0.0, 1.0)
        disclosure_level = self._bounded_float(disclosure_level, 0.28, 0.0, 1.0)
        defensiveness = self._bounded_float(defensiveness, 0.55, 0.0, 1.0)

        label = self._select_label(intensity, disclosure_level, defensiveness, relationship)
        style = self._build_style(
            tempo=str(speaking_style.get("tempo", "slow") or "slow"),
            tone=str(speaking_style.get("tone", "flat") or "flat"),
            repair_pattern=str(speaking_style.get("repair_pattern", "说一点、收一点") or "说一点、收一点"),
            disclosure_level=disclosure_level,
            defensiveness=defensiveness,
        )
        delta = intensity - previous_intensity
        volatility_note = self._build_volatility_note(delta, relationship, flags)

        return {
            "label": label,
            "style": style,
            "intensity": round(float(intensity), 4),
            "disclosure_level": round(float(disclosure_level), 4),
            "defensiveness": round(float(defensiveness), 4),
            "volatility_note": volatility_note,
            "source": "fallback",
        }

    def _sanitize(self, parsed: Dict[str, Any], fallback: Dict[str, Any]) -> Dict[str, Any]:
        label = str(parsed.get("label", "") or "").strip() or str(fallback.get("label", "") or "")
        style = str(parsed.get("style", "") or "").strip() or str(fallback.get("style", "") or "")
        volatility_note = str(parsed.get("volatility_note", "") or "").strip() or str(fallback.get("volatility_note", "") or "")
        intensity = self._bounded_float(parsed.get("intensity"), fallback.get("intensity", 0.62), 0.0, 1.0)
        disclosure_level = self._bounded_float(parsed.get("disclosure_level"), fallback.get("disclosure_level", 0.28), 0.0, 1.0)
        defensiveness = self._bounded_float(parsed.get("defensiveness"), fallback.get("defensiveness", 0.55), 0.0, 1.0)
        return {
            "label": label[:24],
            "style": style[:220],
            "intensity": round(float(intensity), 4),
            "disclosure_level": round(float(disclosure_level), 4),
            "defensiveness": round(float(defensiveness), 4),
            "volatility_note": volatility_note[:160],
            "source": "llm",
        }

    @staticmethod
    def _relationship(session_context: Dict[str, Any]) -> str:
        participants = session_context.get("participants", {}) if isinstance(session_context.get("participants", {}), dict) else {}
        return str(participants.get("relationship", "") or "").strip()

    @staticmethod
    def _disclosure_score(value: Any) -> float:
        text = str(value or "").strip().lower()
        mapping = {
            "sealed": 0.10,
            "guarded": 0.22,
            "shielded": 0.18,
            "cautious": 0.34,
            "partial": 0.48,
            "tentative": 0.44,
            "open": 0.66,
            "full": 0.82,
        }
        return float(mapping.get(text, 0.28))

    @staticmethod
    def _select_label(intensity: float, disclosure_level: float, defensiveness: float, relationship: str) -> str:
        if defensiveness >= 0.72 and intensity >= 0.64:
            return "低落防御"
        if disclosure_level >= 0.55 and intensity >= 0.55:
            return "低落松动"
        if relationship == "治疗师" and disclosure_level >= 0.45:
            return "谨慎试探"
        if intensity < 0.46:
            return "克制低落"
        return "低落收着"

    @staticmethod
    def _build_style(
        tempo: str,
        tone: str,
        repair_pattern: str,
        disclosure_level: float,
        defensiveness: float,
    ) -> str:
        parts = []
        if tempo == "slow":
            parts.append("语速偏慢")
        elif tempo == "hesitant":
            parts.append("起句略犹豫")
        else:
            parts.append("说话节奏相对平")

        if tone in {"flat_shame", "shame", "flat"}:
            parts.append("语气偏平、带一点压着的羞耻感")
        elif tone in {"guarded", "dry"}:
            parts.append("语气收着、不过多外露")
        else:
            parts.append("语气谨慎")

        if disclosure_level <= 0.28:
            parts.append("通常只肯说很少一部分")
        elif disclosure_level <= 0.52:
            parts.append("能说出部分真实感受，但会马上收回来")
        else:
            parts.append("比平时更愿意展开一点细节")

        if defensiveness >= 0.68:
            parts.append("一旦感觉被评估就会立刻加厚防备")
        elif defensiveness >= 0.45:
            parts.append("防备仍在，但不是完全封死")
        else:
            parts.append("防备感有所松动")

        parts.append(f"常见修正方式是：{repair_pattern}")
        return "，".join(parts) + "。"

    def _build_volatility_note(self, delta: float, relationship: str, flags: Dict[str, Any]) -> str:
        if delta >= max(0.08, self.volatility_limit * 0.75):
            if flags.get("is_evaluative_frame", False):
                return "比上一轮更收紧一些，主要是当前场景带来了更明显的被评估感。"
            return "比上一轮更绷一点，但这种收紧仍在当前主诉节点允许的波动范围内。"
        if delta <= -max(0.08, self.volatility_limit * 0.75):
            if relationship == "治疗师":
                return "比上一轮稍微松一点，像是在被接住后短暂愿意多说一点。"
            return "比上一轮稍微松一点，但这种放松仍然不稳。"
        return "整体接近上一轮，只出现细小而可逆的情绪摆动。"

    @staticmethod
    def _contains_negative_cues(conversation: str) -> bool:
        return any(token in conversation for token in ["失败", "没用", "糟", "完了", "不会好", "撑不住", "不行"])

    @staticmethod
    def _contains_softening_cues(conversation: str) -> bool:
        return any(token in conversation for token in ["谢谢", "愿意", "想说", "其实", "试试", "可以"])

    @staticmethod
    def _clip_delta(value: float, limit: float) -> float:
        return max(-float(limit), min(float(limit), float(value)))

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
    def _clip_text(value: Any, limit: int = 1200) -> str:
        text = str(value or "").strip()
        if len(text) <= int(limit):
            return text
        return text[: max(0, int(limit) - 1)] + "…"

    @staticmethod
    def _bounded_float(value: Any, default: float, low: float, high: float) -> float:
        try:
            val = float(value)
        except Exception:
            val = float(default)
        return max(float(low), min(float(high), val))

    @staticmethod
    def _to_list(value: Any):
        if value is None:
            return []
        if isinstance(value, list):
            return list(value)
        return [value]

    @staticmethod
    def _stable_jitter(seed_text: str) -> float:
        digest = hashlib.md5(str(seed_text or "").encode("utf-8")).hexdigest()
        bucket = int(digest[:8], 16) % 11
        return (float(bucket) - 5.0) * 0.015
