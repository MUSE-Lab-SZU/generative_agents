"""
主诉认知路线图管理器。

说明：
- 保留 SymptomStateMachine / DepressionState 这些既有导出名，尽量减少外围调用改动。
- 但内部逻辑已不再采用“固定症状状态 + 规则状态转移”模型，
  而是改为“主诉认知路线图（Complaint Roadmap）”推进模型。
- current_state 仍保留为一个由当前主诉阶段投影得到的粗粒度严重程度标签，
  用于兼容旧的调试、分析与认知偏差子模块；真正驱动演化的是 current_stage/current_chain。
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple


class DepressionState:
    """兼容旧代码的粗粒度状态枚举。"""

    SEVERE_EPISODE = "severe_episode"
    MODERATE_EPISODE = "moderate_episode"
    MILD_EPISODE = "mild_episode"
    REMISSION = "remission"
    CRISIS = "crisis"


@dataclass
class ComplaintStage:
    """单个主诉认知阶段。"""

    label: str
    description: str
    distress_level: float = 0.75
    openness_level: float = 0.25
    hopefulness_level: float = 0.12
    terminal_recovery: bool = False
    source: str = "fallback"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SymptomStateMachine:
    """兼容旧接口的主诉路线图推进器。"""

    LEGACY_STAGE_SEEDS: Dict[str, Dict[str, Any]] = {
        DepressionState.SEVERE_EPISODE: {
            "label": "把失败感当成人生定论",
            "description": "容易把近期受挫扩展成对整个人生的否定，觉得再努力也只会再次让人失望。",
            "distress_level": 0.86,
            "openness_level": 0.18,
            "hopefulness_level": 0.10,
            "terminal_recovery": False,
        },
        DepressionState.MODERATE_EPISODE: {
            "label": "在自责和求助之间摇摆",
            "description": "已经意识到自己状态不对，但仍倾向于把痛苦解释成自己不够努力、不够坚强。",
            "distress_level": 0.70,
            "openness_level": 0.30,
            "hopefulness_level": 0.18,
            "terminal_recovery": False,
        },
        DepressionState.MILD_EPISODE: {
            "label": "能描述困扰但仍难松动",
            "description": "能够说出一些具体困扰，但很难真正把它们和自我价值切开，仍会反复回到低自我评价。",
            "distress_level": 0.52,
            "openness_level": 0.42,
            "hopefulness_level": 0.32,
            "terminal_recovery": False,
        },
        DepressionState.REMISSION: {
            "label": "开始把困难当作可处理的问题",
            "description": "可以承认自己仍会低落，但不再自动把低落等同于整个人没有价值，已经有了较稳定的区分能力。",
            "distress_level": 0.24,
            "openness_level": 0.72,
            "hopefulness_level": 0.78,
            "terminal_recovery": True,
        },
        DepressionState.CRISIS: {
            "label": "被痛苦压到只想停止承受",
            "description": "几乎被痛苦完全吞没，表达里会出现强烈的绝望、崩塌感，甚至把结束一切视为唯一出口。",
            "distress_level": 0.98,
            "openness_level": 0.06,
            "hopefulness_level": 0.03,
            "terminal_recovery": False,
        },
    }

    STATE_SEVERITY = {
        DepressionState.REMISSION: 0,
        DepressionState.MILD_EPISODE: 1,
        DepressionState.MODERATE_EPISODE: 2,
        DepressionState.SEVERE_EPISODE: 3,
        DepressionState.CRISIS: 4,
    }

    STOPWORDS = {
        "自己", "觉得", "感觉", "因为", "然后", "已经", "还是", "不是", "就是", "一个", "一种",
        "有点", "这样", "那种", "事情", "问题", "别人", "什么", "没有", "不会", "如果", "真的",
        "可能", "一直", "最近", "现在", "让我", "我们", "他们", "只是", "还有", "其实", "一下",
    }
    NEGATIVE_HINTS = [
        "累", "痛苦", "没用", "失败", "算了", "不想", "撑不住", "绝望", "没意义", "没有希望",
        "完了", "糟糕", "受不了", "结束", "活着", "拖累", "负担", "废物", "又搞砸", "黑暗", "做不好", "失望",
    ]
    POSITIVE_HINTS = [
        "也许", "试试", "可以", "想", "愿意", "谢谢", "理解", "帮助", "慢慢", "好一点",
        "有用", "能说", "想聊", "希望", "试着", "可能会", "先做", "接受", "分开看",
    ]
    SUPPORT_HINTS = ["谢谢", "理解", "支持", "陪", "帮助", "愿意", "可以试试", "被接住"]
    THERAPY_HINTS = ["咨询", "治疗", "医生", "心理", "复盘", "谈谈", "方法"]
    CRISIS_HINTS = ["死", "自杀", "结束", "解脱", "不想活", "世界没有我", "消失", "撑不下去"]

    DECAY_FACTOR = 0.88
    TRIGGER_PRUNE_THRESHOLD = 0.20

    def __init__(
        self,
        initial_state: str = DepressionState.SEVERE_EPISODE,
        transition_sensitivity: float = 0.7,
        minimum_state_duration: int = 0,
        window_size: int = 2,
        initial_stage: Optional[Dict[str, Any]] = None,
        seed_chain: Optional[List[Dict[str, Any]]] = None,
        max_dialog_history: int = 12,
        now_provider: Optional[Callable[[], datetime]] = None,
    ):
        self.transition_sensitivity = self._clamp_score(transition_sensitivity)
        self.minimum_state_duration = max(0, int(minimum_state_duration or 0))
        self.window_size = max(1, int(window_size or 2))
        self.max_dialog_history = max(4, int(max_dialog_history or 12))

        self._now_provider: Callable[[], datetime] = (
            now_provider if callable(now_provider) else datetime.now
        )
        self.state_start_time = self._now()

        self.current_chain: List[Dict[str, Any]] = []
        self.current_pointer = 0
        self.state_history: List[Dict[str, Any]] = []
        self.dialogue_history: List[Dict[str, Any]] = []
        self.accumulated_triggers: Dict[str, float] = {}
        self.reached_terminal_recovery = False

        self._bootstrap(initial_state=initial_state, initial_stage=initial_stage, seed_chain=seed_chain)

    def _now(self) -> datetime:
        try:
            now_obj = self._now_provider()
        except Exception:
            now_obj = datetime.now()
        return _coerce_datetime(now_obj)

    def set_now_provider(self, now_provider: Optional[Callable[[], datetime]]) -> None:
        if callable(now_provider):
            self._now_provider = now_provider

    def _bootstrap(
        self,
        initial_state: str,
        initial_stage: Optional[Dict[str, Any]],
        seed_chain: Optional[List[Dict[str, Any]]],
    ) -> None:
        start_stage = self._sanitize_stage(
            initial_stage or self.LEGACY_STAGE_SEEDS.get(initial_state, self.LEGACY_STAGE_SEEDS[DepressionState.SEVERE_EPISODE]),
            source="seed",
        )
        self.current_chain = [start_stage]
        if isinstance(seed_chain, list):
            for item in seed_chain:
                self.current_chain.append(self._sanitize_stage(item, source="seed_chain"))
        self._ensure_future_window({}, "", llm_signal=None)
        self.state_start_time = self._now()

    def get_current_state(self) -> str:
        return self._derive_legacy_state(self.get_current_stage())

    def get_current_stage(self) -> Dict[str, Any]:
        if not self.current_chain:
            self._bootstrap(
                initial_state=DepressionState.SEVERE_EPISODE,
                initial_stage=None,
                seed_chain=None,
            )
        idx = min(max(0, int(self.current_pointer or 0)), len(self.current_chain) - 1)
        return copy.deepcopy(self.current_chain[idx])

    def get_current_chain_window(self, count: Optional[int] = None) -> List[Dict[str, Any]]:
        if not self.current_chain:
            return []
        width = max(1, int(count or (self.window_size + 1)))
        start = min(max(0, int(self.current_pointer or 0)), len(self.current_chain) - 1)
        end = min(len(self.current_chain), start + width)
        return [copy.deepcopy(item) for item in self.current_chain[start:end]]

    def get_roadmap_snapshot(self) -> Dict[str, Any]:
        current_stage = self.get_current_stage()
        return {
            "current_state": self.get_current_state(),
            "current_stage": current_stage,
            "current_pointer": int(self.current_pointer or 0),
            "window_size": int(self.window_size or 2),
            "reached_terminal_recovery": bool(self.reached_terminal_recovery),
            "current_chain": self.get_current_chain_window(self.window_size + 1),
            "total_chain_length": len(self.current_chain),
            "state_start_time": self.state_start_time.isoformat(),
            "dialogue_history": copy.deepcopy(self.dialogue_history[-self.max_dialog_history :]),
        }

    def get_state_characteristics(self) -> Dict[str, float]:
        stage = self.get_current_stage()
        return self._derive_state_characteristics(stage)

    def update_state(
        self,
        context_analysis: Dict,
        llm_transition_signal: Optional[Dict[str, Any]] = None,
        conversation_content: str = "",
        roadmap_context: Optional[Dict[str, Any]] = None,
        roadmap_completion_func: Optional[Callable[[str], str]] = None,
        roadmap_llm_cfg: Optional[Dict[str, Any]] = None,
    ) -> bool:
        context = context_analysis if isinstance(context_analysis, dict) else {}
        conversation = str(conversation_content or "").strip()
        roadmap_context = roadmap_context if isinstance(roadmap_context, dict) else {}

        time_in_state = self.get_state_duration()
        if time_in_state < float(self.minimum_state_duration):
            self._track_dialogue(
                current_stage=self.get_current_stage(),
                conversation_content=conversation,
                context=context,
                matched=False,
                match_confidence=0.0,
                match_evidence="minimum_duration_not_reached",
                roadmap_context=roadmap_context,
            )
            return False

        self._refresh_trigger_stats(context.get("triggers", []))

        llm_signal = self._normalize_llm_roadmap_signal(
            llm_transition_signal,
            min_confidence=self._roadmap_cfg_min_confidence(roadmap_llm_cfg),
        )
        if callable(roadmap_completion_func):
            inferred = self._infer_roadmap_signal(
                completion_func=roadmap_completion_func,
                roadmap_context=roadmap_context,
                context=context,
                conversation_content=conversation,
                roadmap_llm_cfg=roadmap_llm_cfg,
            )
            if inferred:
                llm_signal = inferred

        self._ensure_future_window(context, conversation, llm_signal=llm_signal)
        current_stage = self.get_current_stage()
        matched, match_confidence, match_evidence = self._is_stage_matched(
            current_stage=current_stage,
            conversation_content=conversation,
            context=context,
            llm_signal=llm_signal,
        )

        self._track_dialogue(
            current_stage=current_stage,
            conversation_content=conversation,
            context=context,
            matched=matched,
            match_confidence=match_confidence,
            match_evidence=match_evidence,
            roadmap_context=roadmap_context,
        )

        if not matched:
            return False

        if bool(current_stage.get("terminal_recovery", False)):
            self.reached_terminal_recovery = True
            self._record_history(
                action="matched_terminal_recovery",
                from_stage=current_stage,
                to_stage=current_stage,
                triggers=context.get("triggers", []),
                match_confidence=match_confidence,
                match_evidence=match_evidence,
                duration_minutes=self.get_state_duration(),
            )
            return True

        previous_stage = current_stage
        previous_pointer = int(self.current_pointer or 0)
        previous_duration = self.get_state_duration()

        if self.current_pointer >= len(self.current_chain) - 1:
            self._ensure_future_window(context, conversation, llm_signal=llm_signal, minimum_extra=1)

        if self.current_pointer < len(self.current_chain) - 1:
            self.current_pointer += 1
            self.state_start_time = self._now()
            next_stage = self.get_current_stage()
            self.reached_terminal_recovery = bool(next_stage.get("terminal_recovery", False)) and bool(matched)
            self._record_history(
                action="advance",
                from_stage=previous_stage,
                to_stage=next_stage,
                triggers=context.get("triggers", []),
                match_confidence=match_confidence,
                match_evidence=match_evidence,
                pointer_before=previous_pointer,
                pointer_after=self.current_pointer,
                duration_minutes=previous_duration,
            )
            self._ensure_future_window(context, conversation, llm_signal=llm_signal)
            return True

        return False

    def _infer_roadmap_signal(
        self,
        completion_func: Callable[[str], str],
        roadmap_context: Dict[str, Any],
        context: Dict[str, Any],
        conversation_content: str,
        roadmap_llm_cfg: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        prompt = self._build_roadmap_prompt(
            roadmap_context=roadmap_context,
            context=context,
            conversation_content=conversation_content,
            roadmap_llm_cfg=roadmap_llm_cfg,
        )
        raw = ""
        try:
            raw = str(completion_func(prompt) or "")
        except Exception:
            raw = ""
        parsed = self._parse_json_object(raw)
        if not isinstance(parsed, dict):
            return None
        return self._normalize_llm_roadmap_signal(
            parsed,
            min_confidence=self._roadmap_cfg_min_confidence(roadmap_llm_cfg),
        )

    def _build_roadmap_prompt(
        self,
        roadmap_context: Dict[str, Any],
        context: Dict[str, Any],
        conversation_content: str,
        roadmap_llm_cfg: Optional[Dict[str, Any]],
    ) -> str:
        cfg = roadmap_llm_cfg if isinstance(roadmap_llm_cfg, dict) else {}
        max_text_length = self._bounded_int(cfg.get("max_text_length"), 1600, 200, 8000)
        clipped_conversation = self._clip_text(conversation_content, limit=max_text_length)
        base_prompt = self._clip_text(roadmap_context.get("base_prompt", ""), limit=1800)
        current_stage = self.get_current_stage()
        roadmap_window = self.get_current_chain_window(self.window_size + 1)
        history_rows = self.state_history[-6:]
        dialogue_rows = self.dialogue_history[-6:]

        prompt_payload = {
            "cfg_c": base_prompt,
            "current_stage": current_stage,
            "current_chain_window": roadmap_window,
            "current_pointer": self.current_pointer,
            "state_history_tail": history_rows,
            "dialogue_history_tail": dialogue_rows,
            "context": {
                "location": str(roadmap_context.get("location", "") or ""),
                "time_of_day": str(roadmap_context.get("time_of_day", "") or ""),
                "other_agent": str(roadmap_context.get("other_agent", "") or ""),
                "relationship": str(roadmap_context.get("relationship", "") or ""),
                "interaction_type": str(roadmap_context.get("interaction_type", "") or ""),
                "context_analysis": context,
            },
            "current_utterance": clipped_conversation or "（暂无明确话语内容，仅依据情境与历史推演）",
            "N": self.window_size,
        }
        payload_json = json.dumps(prompt_payload, ensure_ascii=False)
        return (
            "# Role\n"
            "你是一位专注于‘计算心理学’和‘行为轨迹预测’的研究专家。\n"
            "你负责维护一个抑郁症 Agent 的‘主诉认知路线图（Complaint Roadmap）’。\n\n"
            "# Task\n"
            "基于 Agent 的初始画像（cfg_c）、当前路线图、已发生历史和本轮话语，完成两件事：\n"
            "1. 判断当前话语是否实质性达到了 current_stage 的描述（is_matched）。\n"
            "2. 预测接下来最可能出现的 N 个认知阶段。\n\n"
            "# Core Rules\n"
            "- 必须保持人格一致性，不要脱离病例设定。\n"
            "- 允许重复、反复、原地踏步；不需要强行改善。\n"
            "- 只有当逻辑上真的成立时，才允许把某个节点标记为 terminal_recovery=true。\n"
            "- 若积极变化出现，也必须保持缓慢、试探、可逆，不得突然痊愈。\n"
            "- 只输出 JSON 对象，不要输出解释、markdown 或额外文本。\n\n"
            "输出字段固定为：\n"
            "{\n"
            '  "matched_current_stage": true,\n'
            '  "match_confidence": 0.0,\n'
            '  "match_evidence": "10到80字，说明是否触及当前阶段",\n'
            '  "next_stages": [\n'
            "    {\n"
            '      "label": "2到24字",\n'
            '      "description": "20到120字",\n'
            '      "distress_level": 0.0,\n'
            '      "openness_level": 0.0,\n'
            '      "hopefulness_level": 0.0,\n'
            '      "terminal_recovery": false\n'
            "    }\n"
            "  ]\n"
            "}\n"
            "约束：\n"
            "- match_confidence / distress_level / openness_level / hopefulness_level 都必须在 0 到 1 之间。\n"
            "- next_stages 必须返回恰好 N 个阶段；如果判断会反复或原地踏步，可以重复或近似重复阶段。\n"
            "- label 要简洁，description 要能描述该阶段的内在认知组织方式。\n\n"
            f"输入：{payload_json}\n"
        )

    def _ensure_future_window(
        self,
        context: Dict[str, Any],
        conversation_content: str,
        llm_signal: Optional[Dict[str, Any]],
        minimum_extra: int = 0,
    ) -> None:
        lookahead = max(0, len(self.current_chain) - int(self.current_pointer or 0) - 1)
        target_lookahead = max(self.window_size, int(minimum_extra or 0))
        missing = max(0, target_lookahead - lookahead)
        if missing <= 0:
            return

        predicted: List[Dict[str, Any]] = []
        if isinstance(llm_signal, dict):
            raw_stages = llm_signal.get("next_stages", [])
            if isinstance(raw_stages, list):
                predicted = [self._sanitize_stage(item, source="llm") for item in raw_stages if isinstance(item, dict)]

        if len(predicted) < missing:
            predicted.extend(
                self._fallback_predict_next_stages(
                    context=context,
                    conversation_content=conversation_content,
                    count=missing - len(predicted),
                    anchor_stage=(predicted[-1] if predicted else self.current_chain[-1]),
                )
            )

        for item in predicted[:missing]:
            self.current_chain.append(self._sanitize_stage(item, source=item.get("source", "fallback")))

    def _fallback_predict_next_stages(
        self,
        context: Dict[str, Any],
        conversation_content: str,
        count: int,
        anchor_stage: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        stages: List[Dict[str, Any]] = []
        previous = self._sanitize_stage(anchor_stage or self.get_current_stage(), source="fallback")
        for idx in range(max(0, int(count or 0))):
            theme = self._detect_theme(context, conversation_content, previous)
            candidate = self._build_fallback_stage(theme, previous, context, idx)
            stages.append(candidate)
            previous = candidate
        return stages

    def _detect_theme(
        self,
        context: Dict[str, Any],
        conversation_content: str,
        current_stage: Dict[str, Any],
    ) -> str:
        content = str(conversation_content or "")
        triggers = [str(item) for item in (context.get("triggers", []) or [])]
        lowered = content.lower()

        if any(word in content for word in self.CRISIS_HINTS) or any(word in lowered for word in ["suicide", "die"]):
            return "collapse"
        if "自杀意念" in triggers:
            return "collapse"
        if "学业失败" in triggers or "academic_failure" in lowered:
            return "failure_self_blame"
        if "自我价值" in triggers:
            return "failure_self_blame"
        if "社交压力" in triggers:
            return "rejection_isolation"
        if "未来焦虑" in triggers:
            return "future_hopeless"

        support_level = 0.0
        relationship = ""
        interaction_type = ""
        social = context.get("social", {}) if isinstance(context.get("social", {}), dict) else {}
        if social:
            try:
                support_level = float(social.get("emotional_support", 0.0) or 0.0)
            except Exception:
                support_level = 0.0
            relationship = str(social.get("relationship", "") or "")
            interaction_type = str(social.get("interaction_type", "") or "")

        if support_level >= 0.72 or relationship == "治疗师" or interaction_type == "治疗对话":
            if float(current_stage.get("hopefulness_level", 0.0) or 0.0) >= 0.45:
                return "recovery_window"
            return "therapy_opening"
        if support_level >= 0.55 or any(token in content for token in self.SUPPORT_HINTS + self.POSITIVE_HINTS):
            return "support_ambivalent"

        if float(current_stage.get("distress_level", 0.0) or 0.0) >= 0.82:
            return "stuck_loop"
        return "stuck_loop"

    def _build_fallback_stage(
        self,
        theme: str,
        previous: Dict[str, Any],
        context: Dict[str, Any],
        index: int,
    ) -> Dict[str, Any]:
        distress = self._bounded_float(previous.get("distress_level"), 0.75, 0.0, 1.0)
        openness = self._bounded_float(previous.get("openness_level"), 0.25, 0.0, 1.0)
        hope = self._bounded_float(previous.get("hopefulness_level"), 0.12, 0.0, 1.0)
        terminal = False

        if theme == "collapse":
            distress = self._bounded_float(distress + 0.03, 0.96, 0.0, 1.0)
            openness = self._bounded_float(openness - 0.04, 0.08, 0.0, 1.0)
            hope = self._bounded_float(hope - 0.03, 0.03, 0.0, 1.0)
            label = "被痛苦压到只想停止承受" if index % 2 == 0 else "把结束一切想成唯一解脱"
            description = "你几乎只剩下‘别再让我继续扛下去’的念头，痛苦被理解为无法被稀释，只想尽快停止承受。"
        elif theme == "failure_self_blame":
            distress = self._bounded_float(max(distress, 0.80) + 0.01, 0.82, 0.0, 1.0)
            openness = self._bounded_float(min(openness, 0.24) - 0.01, 0.20, 0.0, 1.0)
            hope = self._bounded_float(min(hope, 0.14) - 0.01, 0.10, 0.0, 1.0)
            label = "把受挫当成自我失败的证据" if index % 2 == 0 else "从一次做不好扩展到整个人都不行"
            description = "你会把具体失误迅速升级成对整个人的否定，觉得问题不在事件本身，而在‘我就是不行’。"
        elif theme == "rejection_isolation":
            distress = self._bounded_float(max(distress, 0.76) + 0.01, 0.78, 0.0, 1.0)
            openness = self._bounded_float(min(openness, 0.22), 0.22, 0.0, 1.0)
            hope = self._bounded_float(min(hope, 0.15), 0.15, 0.0, 1.0)
            label = "把疏离感理解成自己不值得被靠近"
            description = "你更容易把别人的迟疑、距离或沉默理解成‘我本来就不值得被理解’，于是进一步退回去。"
        elif theme == "future_hopeless":
            distress = self._bounded_float(max(distress, 0.74), 0.74, 0.0, 1.0)
            openness = self._bounded_float(openness, 0.24, 0.0, 1.0)
            hope = self._bounded_float(min(hope, 0.12), 0.12, 0.0, 1.0)
            label = "觉得未来只会继续变糟"
            description = "你会在事情真正发生前就预判结局必然更坏，把不确定感直接理解成无望。"
        elif theme == "support_ambivalent":
            if distress >= 0.68 or hope < 0.28:
                distress = self._bounded_float(distress - 0.05, 0.66, 0.0, 1.0)
                openness = self._bounded_float(openness + 0.10, 0.38, 0.0, 1.0)
                hope = self._bounded_float(hope + 0.08, 0.24, 0.0, 1.0)
                label = "短暂承认自己需要帮助，但马上又退回去"
                description = "你会出现一点‘也许可以说出来’的松动，但很快又补上一句‘算了，反正也不会真的改变什么’。"
            else:
                distress = self._bounded_float(distress - 0.06, 0.52, 0.0, 1.0)
                openness = self._bounded_float(openness + 0.08, 0.52, 0.0, 1.0)
                hope = self._bounded_float(hope + 0.10, 0.38, 0.0, 1.0)
                label = "愿意把痛苦说得更具体，但仍不稳固"
                description = "你开始能说出自己具体在累什么、怕什么，但这份松动还很脆弱，随时可能又被自责盖回去。"
        elif theme == "therapy_opening":
            if distress >= 0.64 or hope < 0.30:
                distress = self._bounded_float(distress - 0.07, 0.62, 0.0, 1.0)
                openness = self._bounded_float(openness + 0.12, 0.44, 0.0, 1.0)
                hope = self._bounded_float(hope + 0.10, 0.28, 0.0, 1.0)
                label = "开始把问题说成可讨论的困境，而不只是自己差"
                description = "你会短暂把注意力从‘我这个人不行’移到‘我现在被什么困住了’，但仍常回头怀疑自己是否值得被帮助。"
            elif distress >= 0.42 or hope < 0.56:
                distress = self._bounded_float(distress - 0.08, 0.44, 0.0, 1.0)
                openness = self._bounded_float(openness + 0.10, 0.60, 0.0, 1.0)
                hope = self._bounded_float(hope + 0.12, 0.48, 0.0, 1.0)
                label = "能区分症状和自我价值，但还会反复"
                description = "你开始承认自己是被困住了，而不是彻底坏掉了；不过一旦遇到挫折，旧有自责仍会迅速回潮。"
            else:
                distress = self._bounded_float(distress - 0.10, 0.22, 0.0, 1.0)
                openness = self._bounded_float(openness + 0.08, 0.78, 0.0, 1.0)
                hope = self._bounded_float(hope + 0.16, 0.74, 0.0, 1.0)
                terminal = hope >= 0.56
                label = "能把低落和自我价值分开看"
                description = "你依然会难受，但不再自动把一次低落或挫败解释成整个人没有价值，已经具备较稳定的区分和修正能力。"
        elif theme == "recovery_window":
            if distress > 0.34 or hope < 0.62:
                distress = self._bounded_float(distress - 0.08, 0.34, 0.0, 1.0)
                openness = self._bounded_float(openness + 0.08, 0.68, 0.0, 1.0)
                hope = self._bounded_float(hope + 0.12, 0.62, 0.0, 1.0)
                label = "开始把困难看成阶段性问题"
                description = "你能承认困难仍在，但不再把它们理解成一条不会改变的命运线，而是开始看到阶段性和可处理性。"
            else:
                distress = self._bounded_float(distress - 0.12, 0.18, 0.0, 1.0)
                openness = self._bounded_float(openness + 0.06, 0.82, 0.0, 1.0)
                hope = self._bounded_float(hope + 0.14, 0.82, 0.0, 1.0)
                terminal = True
                label = "能区分低落与整体自我，不再把挫败当成人生定论"
                description = "你已经能够持续地把暂时低落、现实受挫与整体自我价值分开，不再被单一失败牵引回全盘否定。"
        else:
            drift = self._stable_micro_drift(previous.get("label", ""), index)
            distress = self._bounded_float(distress + drift, distress, 0.0, 1.0)
            openness = self._bounded_float(openness - drift / 2.0, openness, 0.0, 1.0)
            hope = self._bounded_float(hope - max(0.0, drift) / 2.0 + min(0.0, drift) * -0.2, hope, 0.0, 1.0)
            label = "在同一套自责叙事里来回打转"
            description = "你并没有真正离开当前的痛苦解释框架，只是在相似的自责、无力和回避之间反复打转。"

        return self._sanitize_stage(
            {
                "label": label,
                "description": description,
                "distress_level": distress,
                "openness_level": openness,
                "hopefulness_level": hope,
                "terminal_recovery": terminal,
                "source": "fallback",
            },
            source="fallback",
        )

    def _is_stage_matched(
        self,
        current_stage: Dict[str, Any],
        conversation_content: str,
        context: Dict[str, Any],
        llm_signal: Optional[Dict[str, Any]],
    ) -> Tuple[bool, float, str]:
        if isinstance(llm_signal, dict):
            match_confidence = self._bounded_float(llm_signal.get("match_confidence"), 0.0, 0.0, 1.0)
            if bool(llm_signal.get("matched_current_stage", False)):
                evidence = str(llm_signal.get("match_evidence", "") or "").strip() or "llm_match"
                return True, match_confidence, evidence

        content = str(conversation_content or "").strip()
        if not content:
            return False, 0.0, "empty_utterance"

        keywords = self._extract_keywords(
            "{} {}".format(
                current_stage.get("label", ""),
                current_stage.get("description", ""),
            )
        )
        overlap_hits = [kw for kw in keywords if kw in content]
        overlap_score = 0.0
        if keywords:
            overlap_score = min(1.0, float(len(overlap_hits)) / max(1.0, len(keywords) * 0.45))

        negative_hits = sum(1 for item in self.NEGATIVE_HINTS if item in content)
        positive_hits = sum(1 for item in self.POSITIVE_HINTS if item in content)
        crisis_hits = sum(1 for item in self.CRISIS_HINTS if item in content)

        distress = self._bounded_float(current_stage.get("distress_level"), 0.75, 0.0, 1.0)
        openness = self._bounded_float(current_stage.get("openness_level"), 0.25, 0.0, 1.0)
        hope = self._bounded_float(current_stage.get("hopefulness_level"), 0.12, 0.0, 1.0)

        affect_alignment = 0.0
        if distress >= 0.80:
            affect_alignment += min(0.30, 0.08 * float(negative_hits + crisis_hits))
        elif hope >= 0.50:
            affect_alignment += min(0.24, 0.08 * float(positive_hits))
        else:
            affect_alignment += min(0.18, 0.06 * float(max(negative_hits, positive_hits)))

        length_alignment = 0.0
        content_len = len(content)
        if openness <= 0.22 and content_len <= 40:
            length_alignment = 0.12
        elif openness >= 0.45 and content_len >= 28:
            length_alignment = 0.12
        elif 0.22 < openness < 0.45 and 12 <= content_len <= 80:
            length_alignment = 0.08

        trigger_alignment = 0.0
        triggers = [str(item) for item in (context.get("triggers", []) or [])]
        stage_text = "{} {}".format(current_stage.get("label", ""), current_stage.get("description", ""))
        for trigger in triggers:
            if trigger and trigger in stage_text:
                trigger_alignment = 0.12
                break
        if not trigger_alignment:
            theme = self._detect_theme(context, content, current_stage)
            if theme in stage_text or any(token in stage_text for token in ["失败", "靠近", "未来", "帮助", "价值", "低落"]):
                trigger_alignment = 0.06

        semantic_alignment = 0.0
        cue_pairs = [
            (["失败", "不行", "失望", "价值", "负担", "拖累"], ["做不好", "失败", "失望", "不行", "没用", "负担", "拖累", "废物"]),
            (["帮助", "求助", "说出来", "被理解", "被接住"], ["帮助", "理解", "愿意", "说出来", "试试", "聊", "谢谢"]),
            (["未来", "无望", "希望", "变好"], ["未来", "没希望", "不会好", "变好", "希望"]),
            (["靠近", "关系", "孤独", "离开"], ["孤独", "没人", "离开", "关系", "被排斥", "疏远"]),
        ]
        for stage_cues, content_cues in cue_pairs:
            if any(cue in stage_text for cue in stage_cues) and any(cue in content for cue in content_cues):
                semantic_alignment = 0.18
                break

        score = min(1.0, overlap_score * 0.44 + affect_alignment + length_alignment + trigger_alignment + semantic_alignment)
        threshold = 0.30 + self.transition_sensitivity * 0.15
        matched = score >= threshold
        evidence = "heuristic_score={:.3f}; overlap={}; negative_hits={}; positive_hits={}".format(
            score,
            ",".join(overlap_hits[:4]) if overlap_hits else "none",
            negative_hits + crisis_hits,
            positive_hits,
        )
        return matched, round(float(score), 4), evidence

    def _track_dialogue(
        self,
        current_stage: Dict[str, Any],
        conversation_content: str,
        context: Dict[str, Any],
        matched: bool,
        match_confidence: float,
        match_evidence: str,
        roadmap_context: Dict[str, Any],
    ) -> None:
        row = {
            "timestamp": self._now().isoformat(),
            "stage_label": str(current_stage.get("label", "") or ""),
            "derived_state": self._derive_legacy_state(current_stage),
            "conversation_excerpt": self._clip_text(conversation_content, limit=220),
            "matched_current_stage": bool(matched),
            "match_confidence": self._bounded_float(match_confidence, 0.0, 0.0, 1.0),
            "match_evidence": str(match_evidence or "")[:200],
            "triggers": [str(item) for item in (context.get("triggers", []) or [])][:12],
            "other_agent": str(roadmap_context.get("other_agent", "") or ""),
            "relationship": str(roadmap_context.get("relationship", "") or ""),
            "interaction_type": str(roadmap_context.get("interaction_type", "") or ""),
        }
        self.dialogue_history.append(row)
        if len(self.dialogue_history) > self.max_dialog_history:
            self.dialogue_history = self.dialogue_history[-self.max_dialog_history :]

    def _record_history(
        self,
        action: str,
        from_stage: Dict[str, Any],
        to_stage: Dict[str, Any],
        triggers: Any,
        match_confidence: float,
        match_evidence: str,
        pointer_before: Optional[int] = None,
        pointer_after: Optional[int] = None,
        duration_minutes: Optional[float] = None,
    ) -> None:
        now_obj = self._now()
        row = {
            "action": str(action or "advance"),
            "from_state": self._derive_legacy_state(from_stage),
            "to_state": self._derive_legacy_state(to_stage),
            "from_stage_label": str(from_stage.get("label", "") or ""),
            "to_stage_label": str(to_stage.get("label", "") or ""),
            "timestamp": now_obj,
            "duration_minutes": float(duration_minutes) if duration_minutes is not None else (now_obj - self.state_start_time).total_seconds() / 60,
            "triggers": [str(item) for item in self._to_list(triggers)],
            "match_confidence": self._bounded_float(match_confidence, 0.0, 0.0, 1.0),
            "match_evidence": str(match_evidence or "")[:200],
            "pointer_before": int(pointer_before if pointer_before is not None else self.current_pointer),
            "pointer_after": int(pointer_after if pointer_after is not None else self.current_pointer),
        }
        self.state_history.append(row)

    def _refresh_trigger_stats(self, raw_triggers: Any) -> None:
        decayed: Dict[str, float] = {}
        for key, value in (self.accumulated_triggers or {}).items():
            try:
                next_value = float(value) * self.DECAY_FACTOR
            except Exception:
                continue
            if next_value >= self.TRIGGER_PRUNE_THRESHOLD:
                decayed[str(key)] = next_value
        self.accumulated_triggers = decayed

        for trigger in self._to_list(raw_triggers):
            text = str(trigger or "").strip()
            if not text:
                continue
            self.accumulated_triggers[text] = self.accumulated_triggers.get(text, 0.0) + 1.0

    def _sanitize_stage(self, raw: Any, source: str = "fallback") -> Dict[str, Any]:
        if isinstance(raw, ComplaintStage):
            payload = raw.to_dict()
        elif isinstance(raw, dict):
            payload = copy.deepcopy(raw)
        elif isinstance(raw, str):
            payload = {
                "label": str(raw).strip() or "尚未命名阶段",
                "description": str(raw).strip() or "尚未命名阶段",
            }
        else:
            payload = {}

        label = str(payload.get("label", "") or "").strip() or "尚未命名阶段"
        description = str(payload.get("description", "") or "").strip() or label
        distress_level = self._bounded_float(payload.get("distress_level"), 0.75, 0.0, 1.0)
        openness_level = self._bounded_float(payload.get("openness_level"), 0.25, 0.0, 1.0)
        hopefulness_level = self._bounded_float(payload.get("hopefulness_level"), 0.12, 0.0, 1.0)
        terminal_recovery = self._coerce_bool(payload.get("terminal_recovery", False))
        normalized_source = str(payload.get("source", source) or source).strip() or source

        if terminal_recovery:
            distress_level = min(distress_level, 0.40)
            openness_level = max(openness_level, 0.55)
            hopefulness_level = max(hopefulness_level, 0.65)

        return ComplaintStage(
            label=label[:32],
            description=description[:220],
            distress_level=round(float(distress_level), 4),
            openness_level=round(float(openness_level), 4),
            hopefulness_level=round(float(hopefulness_level), 4),
            terminal_recovery=bool(terminal_recovery),
            source=normalized_source[:32],
        ).to_dict()

    def _normalize_llm_roadmap_signal(
        self,
        payload: Any,
        min_confidence: float,
    ) -> Optional[Dict[str, Any]]:
        if not isinstance(payload, dict):
            return None

        match_confidence = self._bounded_float(
            payload.get("match_confidence", payload.get("confidence", 0.0)),
            0.0,
            0.0,
            1.0,
        )
        matched = self._coerce_bool(payload.get("matched_current_stage", False))
        if matched and match_confidence < min_confidence:
            matched = False

        match_evidence = str(payload.get("match_evidence", "") or "").strip()[:180]
        next_stages: List[Dict[str, Any]] = []
        raw_next = payload.get("next_stages", payload.get("predicted_stages", []))
        if isinstance(raw_next, list):
            for item in raw_next:
                if not isinstance(item, dict):
                    continue
                next_stages.append(self._sanitize_stage(item, source="llm"))

        return {
            "matched_current_stage": bool(matched),
            "match_confidence": round(float(match_confidence), 4),
            "match_evidence": match_evidence,
            "next_stages": next_stages,
        }

    def _roadmap_cfg_min_confidence(self, roadmap_llm_cfg: Optional[Dict[str, Any]]) -> float:
        cfg = roadmap_llm_cfg if isinstance(roadmap_llm_cfg, dict) else {}
        return self._bounded_float(cfg.get("min_confidence"), 0.60, 0.0, 1.0)

    def _derive_legacy_state(self, stage: Dict[str, Any]) -> str:
        characteristics = self._derive_state_characteristics(stage)
        distress = self._bounded_float(stage.get("distress_level"), 0.75, 0.0, 1.0)
        suicidal = characteristics.get("suicidal_ideation", 0.0)
        if suicidal >= 0.84 or distress >= 0.94 or self._contains_crisis_words(stage):
            return DepressionState.CRISIS
        if distress >= 0.78:
            return DepressionState.SEVERE_EPISODE
        if distress >= 0.58:
            return DepressionState.MODERATE_EPISODE
        if distress >= 0.38:
            return DepressionState.MILD_EPISODE
        return DepressionState.REMISSION

    def _contains_crisis_words(self, stage: Dict[str, Any]) -> bool:
        text = "{} {}".format(stage.get("label", ""), stage.get("description", ""))
        return any(word in text for word in self.CRISIS_HINTS)

    def _derive_state_characteristics(self, stage: Dict[str, Any]) -> Dict[str, float]:
        distress = self._bounded_float(stage.get("distress_level"), 0.75, 0.0, 1.0)
        openness = self._bounded_float(stage.get("openness_level"), 0.25, 0.0, 1.0)
        hope = self._bounded_float(stage.get("hopefulness_level"), 0.12, 0.0, 1.0)
        crisis_boost = 0.18 if self._contains_crisis_words(stage) else 0.0
        terminal_bonus = 0.12 if bool(stage.get("terminal_recovery", False)) else 0.0

        mood_score = self._bounded_float((1.0 - distress) * 0.70 + hope * 0.30 + terminal_bonus * 0.30, 0.25, 0.0, 1.0)
        energy_level = self._bounded_float((1.0 - distress) * 0.58 + hope * 0.18 + openness * 0.08, 0.25, 0.0, 1.0)
        social_avoidance = self._bounded_float(distress * 0.52 + (1.0 - openness) * 0.34, 0.55, 0.0, 1.0)
        cognitive_distortion = self._bounded_float(distress * 0.60 + (1.0 - hope) * 0.26, 0.62, 0.0, 1.0)
        suicidal_ideation = self._bounded_float(max(0.0, distress - 0.62) * 1.55 + crisis_boost, 0.15, 0.0, 1.0)
        sleep_disturbance = self._bounded_float(0.26 + distress * 0.56, 0.50, 0.0, 1.0)
        appetite_change = self._bounded_float(0.20 + distress * 0.44, 0.45, 0.0, 1.0)
        concentration = self._bounded_float((1.0 - distress) * 0.46 + hope * 0.22 + openness * 0.16, 0.35, 0.0, 1.0)
        self_worth = self._bounded_float((1.0 - distress) * 0.34 + hope * 0.42 + terminal_bonus * 0.40, 0.20, 0.0, 1.0)
        hopelessness = self._bounded_float(distress * 0.60 + (1.0 - hope) * 0.32 + crisis_boost * 0.60, 0.72, 0.0, 1.0)

        return {
            "mood_score": round(float(mood_score), 4),
            "energy_level": round(float(energy_level), 4),
            "social_avoidance": round(float(social_avoidance), 4),
            "cognitive_distortion": round(float(cognitive_distortion), 4),
            "suicidal_ideation": round(float(suicidal_ideation), 4),
            "sleep_disturbance": round(float(sleep_disturbance), 4),
            "appetite_change": round(float(appetite_change), 4),
            "concentration": round(float(concentration), 4),
            "self_worth": round(float(self_worth), 4),
            "hopelessness": round(float(hopelessness), 4),
        }

    def force_transition(self, new_state: Any, reason: str = "manual"):
        previous = self.get_current_stage()
        if isinstance(new_state, dict):
            next_stage = self._sanitize_stage(new_state, source="manual")
        else:
            text = str(new_state or "").strip()
            if text in self.LEGACY_STAGE_SEEDS:
                next_stage = self._sanitize_stage(self.LEGACY_STAGE_SEEDS[text], source="manual")
            else:
                fallback = {
                    "label": text or "手动指定阶段",
                    "description": text or "手动指定阶段",
                    "distress_level": previous.get("distress_level", 0.75),
                    "openness_level": previous.get("openness_level", 0.25),
                    "hopefulness_level": previous.get("hopefulness_level", 0.12),
                    "terminal_recovery": False,
                }
                next_stage = self._sanitize_stage(fallback, source="manual")

        previous_duration = self.get_state_duration()
        if not self.current_chain:
            self.current_chain = [next_stage]
            self.current_pointer = 0
        else:
            self.current_chain = self.current_chain[: self.current_pointer] + [next_stage]
        self.state_start_time = self._now()
        self.reached_terminal_recovery = bool(next_stage.get("terminal_recovery", False))
        self._record_history(
            action="manual",
            from_stage=previous,
            to_stage=next_stage,
            triggers=[reason],
            match_confidence=1.0,
            match_evidence=str(reason or "manual")[:120],
            duration_minutes=previous_duration,
        )
        self._ensure_future_window({}, "", llm_signal=None)

    def get_state_duration(self) -> float:
        return (self._now() - self.state_start_time).total_seconds() / 60

    def get_state_history(self) -> List[Dict]:
        return copy.deepcopy(self.state_history)

    def get_symptom_intensity(self, symptom: str, time_of_day: Optional[str] = None) -> float:
        base_intensity = self.get_state_characteristics().get(symptom, 0.5)
        if time_of_day and symptom in ["mood_score", "energy_level"]:
            circadian_factors = {
                "morning": 0.72,
                "afternoon": 1.0,
                "evening": 1.08,
                "night": 0.90,
            }
            base_intensity *= circadian_factors.get(time_of_day, 1.0)
        return min(1.0, max(0.0, base_intensity))

    def to_dict(self) -> Dict[str, Any]:
        history_payload: List[Dict[str, Any]] = []
        for item in self.state_history:
            if not isinstance(item, dict):
                continue
            row = dict(item)
            ts = row.get("timestamp")
            if isinstance(ts, datetime):
                row["timestamp"] = ts.isoformat()
            elif ts is None:
                row["timestamp"] = ""
            else:
                row["timestamp"] = str(ts)
            row["triggers"] = [str(t) for t in self._to_list(row.get("triggers", []))]
            history_payload.append(row)

        return {
            "mode": "complaint_roadmap",
            "current_state": self.get_current_state(),
            "current_stage": self.get_current_stage(),
            "current_pointer": int(self.current_pointer or 0),
            "window_size": int(self.window_size or 2),
            "transition_sensitivity": float(self.transition_sensitivity),
            "minimum_state_duration": int(self.minimum_state_duration),
            "state_start_time": self.state_start_time.isoformat(),
            "state_history": history_payload,
            "current_chain": [copy.deepcopy(item) for item in self.current_chain],
            "dialogue_history": copy.deepcopy(self.dialogue_history),
            "reached_terminal_recovery": bool(self.reached_terminal_recovery),
            "accumulated_triggers": {
                str(k): float(v)
                for k, v in (self.accumulated_triggers or {}).items()
            },
        }

    @classmethod
    def from_dict(
        cls,
        payload: Dict[str, Any],
        now_provider: Optional[Callable[[], datetime]] = None,
    ) -> "SymptomStateMachine":
        payload = payload or {}
        roadmap_payload = payload.get("current_chain", [])
        current_stage = payload.get("current_stage")
        current_state = str(payload.get("current_state", DepressionState.SEVERE_EPISODE) or DepressionState.SEVERE_EPISODE)

        machine = cls(
            initial_state=current_state if current_state in cls.LEGACY_STAGE_SEEDS else DepressionState.SEVERE_EPISODE,
            transition_sensitivity=float(payload.get("transition_sensitivity", 0.7) or 0.7),
            minimum_state_duration=int(payload.get("minimum_state_duration", 0) or 0),
            window_size=int(payload.get("window_size", 2) or 2),
            initial_stage=current_stage if isinstance(current_stage, dict) else None,
            seed_chain=None,
            max_dialog_history=12,
            now_provider=now_provider,
        )

        chain: List[Dict[str, Any]] = []
        if isinstance(roadmap_payload, list) and roadmap_payload:
            for item in roadmap_payload:
                if isinstance(item, dict):
                    chain.append(machine._sanitize_stage(item, source=item.get("source", "load")))
        elif isinstance(current_stage, dict):
            chain = [machine._sanitize_stage(current_stage, source="load")]
        else:
            seed = cls.LEGACY_STAGE_SEEDS.get(current_state, cls.LEGACY_STAGE_SEEDS[DepressionState.SEVERE_EPISODE])
            chain = [machine._sanitize_stage(seed, source="legacy_load")]

        machine.current_chain = chain or machine.current_chain

        try:
            pointer = int(payload.get("current_pointer", 0) or 0)
        except Exception:
            pointer = 0
        machine.current_pointer = min(max(0, pointer), max(0, len(machine.current_chain) - 1))

        machine.state_start_time = _coerce_datetime(payload.get("state_start_time"))
        machine.reached_terminal_recovery = bool(payload.get("reached_terminal_recovery", False))

        history_raw = payload.get("state_history", [])
        machine.state_history = []
        if isinstance(history_raw, list):
            for item in history_raw:
                if not isinstance(item, dict):
                    continue
                row = dict(item)
                row["timestamp"] = _coerce_datetime(row.get("timestamp"))
                row["triggers"] = [str(t) for t in machine._to_list(row.get("triggers", []))]
                try:
                    row["duration_minutes"] = float(row.get("duration_minutes", 0.0) or 0.0)
                except Exception:
                    row["duration_minutes"] = 0.0
                row["from_state"] = str(row.get("from_state", "") or "")
                row["to_state"] = str(row.get("to_state", "") or "")
                row["from_stage_label"] = str(row.get("from_stage_label", "") or "")
                row["to_stage_label"] = str(row.get("to_stage_label", "") or "")
                machine.state_history.append(row)

        dialogue_raw = payload.get("dialogue_history", [])
        machine.dialogue_history = []
        if isinstance(dialogue_raw, list):
            for item in dialogue_raw:
                if not isinstance(item, dict):
                    continue
                machine.dialogue_history.append(
                    {
                        "timestamp": str(item.get("timestamp", "") or ""),
                        "stage_label": str(item.get("stage_label", "") or ""),
                        "derived_state": str(item.get("derived_state", "") or ""),
                        "conversation_excerpt": str(item.get("conversation_excerpt", "") or ""),
                        "matched_current_stage": bool(item.get("matched_current_stage", False)),
                        "match_confidence": machine._bounded_float(item.get("match_confidence"), 0.0, 0.0, 1.0),
                        "match_evidence": str(item.get("match_evidence", "") or ""),
                        "triggers": [str(t) for t in machine._to_list(item.get("triggers", []))],
                        "other_agent": str(item.get("other_agent", "") or ""),
                        "relationship": str(item.get("relationship", "") or ""),
                        "interaction_type": str(item.get("interaction_type", "") or ""),
                    }
                )
        if len(machine.dialogue_history) > machine.max_dialog_history:
            machine.dialogue_history = machine.dialogue_history[-machine.max_dialog_history :]

        acc = payload.get("accumulated_triggers", {})
        machine.accumulated_triggers = {}
        if isinstance(acc, dict):
            for key, value in acc.items():
                try:
                    machine.accumulated_triggers[str(key)] = float(value)
                except Exception:
                    continue

        machine._ensure_future_window({}, "", llm_signal=None)
        return machine

    def _extract_keywords(self, text: Any) -> List[str]:
        source = str(text or "")
        chunks = re.findall(r"[\u4e00-\u9fff]{2,8}|[A-Za-z0-9_\-]{3,}", source)
        keywords: List[str] = []
        seen = set()
        for chunk in chunks:
            token = str(chunk).strip()
            if not token or token in self.STOPWORDS or token in seen:
                continue
            seen.add(token)
            keywords.append(token)
        return keywords[:12]

    def _parse_json_object(self, raw: Any) -> Optional[Dict[str, Any]]:
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
    def _clamp_score(value: Any) -> float:
        try:
            score = float(value)
        except Exception:
            score = 0.0
        return max(0.0, min(1.0, score))

    @staticmethod
    def _bounded_float(value: Any, default: float, lower: float, upper: float) -> float:
        try:
            num = float(value)
        except Exception:
            num = float(default)
        num = max(float(lower), num)
        num = min(float(upper), num)
        return num

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
    def _coerce_bool(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        text = str(value or "").strip().lower()
        return text in {"1", "true", "yes", "y", "on"}

    @staticmethod
    def _to_list(value: Any) -> List[Any]:
        if value is None:
            return []
        if isinstance(value, list):
            return list(value)
        return [value]

    @staticmethod
    def _stable_micro_drift(seed_text: Any, index: int) -> float:
        digest = hashlib.md5(f"{seed_text}|{index}".encode("utf-8")).hexdigest()
        bucket = int(digest[:8], 16) % 9  # 0..8
        return (float(bucket) - 4.0) * 0.01  # [-0.04, +0.04]


def _coerce_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        text = value.strip()
        if text:
            try:
                return datetime.fromisoformat(text)
            except Exception:
                pass
    return datetime.now()
