"""LLM-based inferencer for transient speaking emotion during chats."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Callable, Dict, Optional


class EmotionInferencer:
    """Infer the current speaking emotion for one chat turn."""

    def infer(
        self,
        payload: Dict[str, Any],
        completion_func: Optional[Callable[[str], str]] = None,
    ) -> Dict[str, Any]:
        """Infer a transient emotion snapshot.

        The returned payload is designed to be persisted into runtime state and
        injected only into chat prompts.
        """
        fallback = self.build_fallback(payload)
        if not callable(completion_func):
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
        static_profile_json = json.dumps(
            payload.get("static_profile", {}) or {}, ensure_ascii=False
        )
        current_event_json = json.dumps(
            payload.get("current_event", {}) or {}, ensure_ascii=False
        )
        merged_case_config_json = json.dumps(
            payload.get("merged_case_config", {}) or {}, ensure_ascii=False
        )
        previous_emotion_json = json.dumps(
            payload.get("previous_emotion", {}) or {}, ensure_ascii=False
        )
        conversation_content = self._clip_text(
            payload.get("conversation_content", ""), limit=1200
        )
        return (
            "你是 Emotion Inferencer。\n"
            "任务：根据稳定的人设、当前抑郁画像、最近对话与上一轮情绪，"
            "推断角色『当前这一轮说话时』的情绪风格。\n"
            "注意：\n"
            "1. Emotion 是当前交流中的瞬时说话状态，不是永久人格标签。\n"
            "2. 必须模拟波动性：相对上一轮可出现小幅、非线性、可逆的起伏，"
            "例如略微更紧绷、稍有放松、短暂愿意多说一点后又收回；"
            "但不能脱离病例设定，也不能突然明显康复。\n"
            "3. 输出要直接指导说话方式，重点体现语气、节奏、是否掩饰、是否防御、"
            "是否愿意展开表达。\n"
            "4. 只输出 JSON 对象，不要输出解释、markdown 或额外文本。\n"
            "输出字段固定为：\n"
            "{\n"
            '  "label": "2到12字的当前情绪概括",\n'
            '  "style": "15到80字的说话风格描述",\n'
            '  "intensity": 0.0,\n'
            '  "volatility_note": "10到60字，描述相对上一轮的小幅波动"\n'
            "}\n"
            "字段约束：\n"
            "- intensity 必须在 0 到 1 之间，数值越大表示当前说话时越紧绷、低落、压抑或防御。\n"
            "- 如果信息不足，优先保持与上一轮相近，但允许有轻微波动。\n"
            "- 不要把 emotion 写成长期诊断结论。\n\n"
            f"静态Profile：{static_profile_json}\n"
            f"当前主导事件：{current_event_json}\n" # 想通过主诉链实现
            f"上一轮Emotion：{previous_emotion_json}\n"
            f"对话对象：{str(payload.get('other_agent', '') or '')}\n"
            f"最近对话：{conversation_content if conversation_content else '（暂无对话内容）'}\n"
        )

    def build_fallback(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        previous = (
            payload.get("previous_emotion", {})
            if isinstance(payload.get("previous_emotion", {}), dict)
            else {}
        )
        relationship = str(payload.get("relationship", "") or "").strip()
        conversation = str(payload.get("conversation_content", "") or "")

        previous_initialized = bool(
            str(previous.get("label", "") or "").strip()
            or str(previous.get("style", "") or "").strip()
            or int(previous.get("updated_step", -1) or -1) >= 0
        )
        prev_intensity = self._bounded_float(
            previous.get("intensity"),
            0.62 if not previous_initialized else 0.62,
            0.0,
            1.0,
        )
        if not previous_initialized and prev_intensity <= 0.0:
            prev_intensity = 0.62
        relation_bias = {
            "治疗师": -0.04,
            "亲密朋友": -0.02,
            "普通朋友": 0.00,
            "冲突关系": 0.10,
        }.get(relationship, 0.03)

        positive_hits = sum(
            1 for k in ["谢谢", "理解", "支持", "陪", "可以", "愿意", "放松"] if k in conversation
        )
        negative_hits = sum(
            1 for k in ["烦", "累", "压力", "没用", "算了", "不想", "失望", "难受"] if k in conversation
        )
        sentiment_bias = 0.04 * float(negative_hits) - 0.03 * float(positive_hits)
        jitter = self._stable_jitter(
            "{}|{}|{}|{}".format(
                payload.get("other_agent", ""),
                relationship,
                payload.get("now_step", ""),
                conversation[-200:],
            )
        )

        intensity = self._bounded_float(
            prev_intensity + relation_bias + sentiment_bias + jitter,
            0.62,
            0.18,
            0.95,
        )

        if relationship == "治疗师":
            label = "低落试探" if intensity < 0.60 else "克制压抑"
            style = "回答偏轻，通常先保守应答；感到被接住时，会慢慢补充更多细节。"
        elif relationship == "冲突关系":
            label = "紧绷防御"
            style = "说话更短，带戒备和回避，容易先挡开问题，不愿暴露自己的脆弱。"
        elif relationship in {"亲密朋友", "普通朋友"}:
            label = "低落谨慎"
            style = "常先说还好，再一点点补充；整体偏克制，情绪不会一下子完全放开。"
        else:
            label = "低落克制"
            style = "语速略慢，停顿稍多，表达收着说，不会主动把感受一下子讲透。"

        delta = intensity - prev_intensity
        if delta >= 0.08:
            volatility_note = "比上一轮略紧一些，但这种收紧仍是小幅、可逆的。"
        elif delta <= -0.08:
            volatility_note = "比上一轮稍微松一点，但这种放松并不稳定。"
        else:
            volatility_note = "整体接近上一轮，只出现细小而非线性的情绪起伏。"

        return {
            "label": label,
            "style": style,
            "intensity": round(float(intensity), 4),
            "volatility_note": volatility_note,
            "source": "fallback",
        }

    def _sanitize(self, parsed: Dict[str, Any], fallback: Dict[str, Any]) -> Dict[str, Any]:
        label = str(parsed.get("label", "") or "").strip()
        style = str(parsed.get("style", "") or "").strip()
        volatility_note = str(parsed.get("volatility_note", "") or "").strip()
        intensity = self._bounded_float(
            parsed.get("intensity"),
            fallback.get("intensity", 0.62),
            0.0,
            1.0,
        )

        if len(label) < 2:
            label = str(fallback.get("label", "") or "")
        if len(style) < 8:
            style = str(fallback.get("style", "") or "")
        if len(volatility_note) < 8:
            volatility_note = str(fallback.get("volatility_note", "") or "")

        return {
            "label": label[:24],
            "style": style[:180],
            "intensity": round(float(intensity), 4),
            "volatility_note": volatility_note[:120],
            "source": "llm",
        }

    def _parse_json_object(self, raw: Any) -> Optional[Dict[str, Any]]:
        text = str(raw or "").strip()
        if not text:
            return None

        fenced = re.findall(
            r"```(?:json)?\s*([\s\S]*?)\s*```", text, flags=re.IGNORECASE
        )
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
    def _stable_jitter(seed_text: str) -> float:
        digest = hashlib.md5(str(seed_text or "").encode("utf-8")).hexdigest()
        bucket = int(digest[:8], 16) % 11  # 0..10
        return (float(bucket) - 5.0) * 0.015  # [-0.075, +0.075]
