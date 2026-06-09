"""主诉图驱动的动态 Prompt 构建器。"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .prompt_templates import load_prompt_json, render_prompt_section


class DynamicPromptBuilder:
    """构建主诉节点、会话与瞬时情绪组成的多层 Prompt。"""

    PROMPT_CONFIG: Dict[str, Any] = load_prompt_json("depression/depression_prompt_config", {})
    CONTEXT_FLAG_TEXTS: Dict[str, str] = (
        PROMPT_CONFIG.get("context_flags", {}) if isinstance(PROMPT_CONFIG.get("context_flags", {}), dict) else {}
    )
    EMOTION_DEFAULTS: Dict[str, str] = (
        PROMPT_CONFIG.get("emotion_defaults", {}) if isinstance(PROMPT_CONFIG.get("emotion_defaults", {}), dict) else {}
    )

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config if isinstance(config, dict) else {}
        self.include_graph_window = bool(self.config.get("include_graph_window", True))
        self.include_emotion_layer = bool(self.config.get("include_emotion_layer", True))

    def build_prompt(
        self,
        base_prompt: str,
        current_stage: Dict[str, Any],
        graph_snapshot: Dict[str, Any],
        session_context: Dict[str, Any],
        activated_memories: List[Dict[str, Any]],
        emotion: Optional[Dict[str, Any]] = None,
    ) -> str:
        # 层次顺序很重要：
        # 基础人格 -> 当前主诉节点 -> 当前会话 -> 瞬时情绪。
        # 它体现的是“稳定人设在前，当前轮波动在后”的约束方向。
        layers = [
            self._build_base_layer(base_prompt),
            self._build_stage_layer(current_stage, graph_snapshot),
            self._build_context_layer(session_context),
        ]
        if self.include_emotion_layer:
            layers.append(self._build_emotion_layer(current_stage, emotion or {}, activated_memories))
        return self._combine_layers(layers)

    def build_simple_prompt(
        self,
        base_prompt: str,
        current_stage: Dict[str, Any],
        graph_snapshot: Optional[Dict[str, Any]] = None,
    ) -> str:
        # simple prompt 只保留最核心的“人格 + 当前主诉节点”，
        # 适合反思/摘要等不需要完整会话层的场景。
        return self._combine_layers(
            [
                self._build_base_layer(base_prompt),
                self._build_stage_layer(
                    current_stage=current_stage if isinstance(current_stage, dict) else {},
                    graph_snapshot=graph_snapshot if isinstance(graph_snapshot, dict) else {},
                ),
            ]
        )

    def _build_base_layer(self, base_prompt: str) -> str:
        return self._with_trailing_newline(
            render_prompt_section(
                "depression/dynamic_prompt_layers",
                "base_layer",
                {"base_prompt": str(base_prompt or "").strip()},
            ).strip()
        )

    def _build_stage_layer(self, current_stage: Dict[str, Any], graph_snapshot: Dict[str, Any]) -> str:
        current_stage = current_stage if isinstance(current_stage, dict) else {}
        current_window = graph_snapshot.get("current_graph_window", [])
        current_window = current_window if isinstance(current_window, list) else []

        # 这些字段都直接来自 complaint_graph 配置中的单个 stage。
        label = str(current_stage.get("label", "未命名主诉节点") or "未命名主诉节点").strip()
        summary = str(current_stage.get("summary", "") or "").strip()
        core_belief = str(current_stage.get("core_belief", "") or "").strip()
        narrative_focus = [str(item) for item in current_stage.get("narrative_focus", [])[:6]] if isinstance(current_stage.get("narrative_focus", []), list) else []
        speaking_style = current_stage.get("speaking_style", {}) if isinstance(current_stage.get("speaking_style", {}), dict) else {}

        optional_sections = []
        if summary:
            optional_sections.append(
                self._render_block(
                    "dynamic_stage_summary_section",
                    {"summary": summary},
                )
            )
        if core_belief:
            optional_sections.append(
                self._render_block(
                    "dynamic_stage_core_belief_section",
                    {"core_belief": core_belief},
                )
            )
        if narrative_focus:
            # narrative_focus 不是“必须逐字复述的关键词”，
            # 更像给 LLM 的“优先围绕哪些痛点组织表达”的提醒。
            optional_sections.append(
                self._render_block(
                    "dynamic_stage_narrative_focus_section",
                    {"narrative_focus": "、".join(narrative_focus)},
                )
            )

        if speaking_style:
            # 这里把结构化风格字段翻译回自然语言提示，
            # 让 LLM 更容易在输出中体现节奏/语气/修正模式。
            optional_sections.append(
                self._render_block(
                    "dynamic_stage_speaking_style_section",
                    {
                        "tempo": speaking_style.get("tempo", "slow"),
                        "disclosure": speaking_style.get("disclosure", "guarded"),
                        "tone": speaking_style.get("tone", "flat"),
                        "repair_pattern": speaking_style.get("repair_pattern", "说一点、收一点"),
                    },
                )
            )

        graph_window_section = ""
        if self.include_graph_window and current_window:
            graph_window_items = []
            for idx, stage in enumerate(current_window):
                if not isinstance(stage, dict):
                    continue
                # idx=0 永远是当前节点；后面的节点是并列候选分支。
                marker = "当前" if idx == 0 else f"分支{idx}"
                graph_window_items.append(
                    self._render_block(
                        "dynamic_stage_graph_window_item",
                        {
                            "marker": marker,
                            "label": str(stage.get("label", "未知节点") or "未知节点"),
                            "summary": str(stage.get("summary", "") or "").strip(),
                        },
                    )
                )
            next_limit_section = ""
            if len(current_window) > 1:
                candidate_labels = [
                    str(stage.get("label", "") or "").strip()
                    for stage in current_window[1:]
                    if isinstance(stage, dict) and str(stage.get("label", "") or "").strip()
                ]
                if candidate_labels:
                    next_limit_section = self._render_block(
                        "dynamic_stage_next_limit_section",
                        {"candidate_labels": "、".join(candidate_labels[:6])},
                    )
            graph_window_section = render_prompt_section(
                "depression/dynamic_prompt_layers",
                "stage_graph_window_section",
                {
                    "graph_window_items": "".join(graph_window_items),
                    "next_limit_section": next_limit_section,
                },
            )

        return self._with_trailing_newline(
            render_prompt_section(
                "depression/dynamic_prompt_layers",
                "stage_layer",
                {
                    "label": label,
                    "optional_sections": "".join(optional_sections),
                    "graph_window_section": graph_window_section,
                },
            ).strip()
        )

    def _build_context_layer(self, session_context: Dict[str, Any]) -> str:
        session_context = session_context if isinstance(session_context, dict) else {}
        scene = session_context.get("scene", {}) if isinstance(session_context.get("scene", {}), dict) else {}
        participants = session_context.get("participants", {}) if isinstance(session_context.get("participants", {}), dict) else {}
        semantic = session_context.get("semantic_cues", {}) if isinstance(session_context.get("semantic_cues", {}), dict) else {}
        flags = session_context.get("session_flags", {}) if isinstance(session_context.get("session_flags", {}), dict) else {}

        optional_sections = []
        if participants.get("other_agent"):
            optional_sections.append(
                self._render_block(
                    "dynamic_context_other_agent_section",
                    {
                        "other_agent": participants.get("other_agent", "未知"),
                        "relationship": participants.get("relationship", "未知"),
                    },
                )
            )
        if scene.get("interaction_type"):
            optional_sections.append(
                self._render_block(
                    "dynamic_context_interaction_type_section",
                    {"interaction_type": scene.get("interaction_type", "未知")},
                )
            )

        topics = [str(item) for item in semantic.get("topics", [])[:6]] if isinstance(semantic.get("topics", []), list) else []
        speech_acts = [str(item) for item in semantic.get("speech_acts", [])[:6]] if isinstance(semantic.get("speech_acts", []), list) else []
        stance = [str(item) for item in semantic.get("stance", [])[:6]] if isinstance(semantic.get("stance", []), list) else []
        if topics:
            # topics 是“你在围绕什么痛点说话”。
            optional_sections.append(
                self._render_block(
                    "dynamic_context_topics_section",
                    {"topics": "、".join(topics)},
                )
            )
        if speech_acts:
            # speech_acts 是“你是怎么说的”，比如求助、淡化、回避。
            optional_sections.append(
                self._render_block(
                    "dynamic_context_speech_acts_section",
                    {"speech_acts": "、".join(speech_acts)},
                )
            )
        if stance:
            # stance 更接近说话姿态，例如试探、防御、羞耻、低落。
            optional_sections.append(
                self._render_block(
                    "dynamic_context_stance_section",
                    {"stance": "、".join(stance)},
                )
            )

        flag_texts = []
        for flag_key in [
            "is_help_seeking_frame",
            "is_evaluative_frame",
            "is_close_relationship",
            "is_professional_frame",
            "is_minimizing",
            "is_withdrawing",
        ]:
            flag_text = str(self.CONTEXT_FLAG_TEXTS.get(flag_key, "") or "").strip()
            if flags.get(flag_key, False) and flag_text:
                flag_texts.append(flag_text)
        if flag_texts:
            optional_sections.append(
                self._render_block(
                    "dynamic_context_flags_section",
                    {"flags": "；".join(flag_texts)},
                )
            )

        return self._with_trailing_newline(
            render_prompt_section(
                "depression/dynamic_prompt_layers",
                "context_layer",
                {
                    "location": scene.get("location", "未知"),
                    "time_of_day": scene.get("time_of_day", "未知"),
                    "optional_sections": "".join(optional_sections).rstrip(),
                },
            ).strip()
        )

    def _build_emotion_layer(
        self,
        current_stage: Dict[str, Any],
        emotion: Dict[str, Any],
        activated_memories: List[Dict[str, Any]],
    ) -> str:
        current_stage = current_stage if isinstance(current_stage, dict) else {}
        emotion = emotion if isinstance(emotion, dict) else {}
        speaking_style = current_stage.get("speaking_style", {}) if isinstance(current_stage.get("speaking_style", {}), dict) else {}

        disclosure_level = float(emotion.get("disclosure_level", 0.3) or 0.3)
        defensiveness = float(emotion.get("defensiveness", 0.5) or 0.5)
        intensity = float(emotion.get("intensity", 0.6) or 0.6)

        # 下面不是再算一次 emotion，而是把数值区间翻译成 LLM 更容易执行的语言提示。
        instruction_sections = []
        if intensity >= 0.72:
            instruction_sections.append(self._render_block("dynamic_emotion_intensity_high"))
        elif intensity >= 0.50:
            instruction_sections.append(self._render_block("dynamic_emotion_intensity_medium"))
        else:
            instruction_sections.append(self._render_block("dynamic_emotion_intensity_low"))

        if disclosure_level <= 0.28:
            instruction_sections.append(self._render_block("dynamic_emotion_disclosure_low"))
        elif disclosure_level <= 0.55:
            instruction_sections.append(self._render_block("dynamic_emotion_disclosure_medium"))
        else:
            instruction_sections.append(self._render_block("dynamic_emotion_disclosure_high"))

        if defensiveness >= 0.68:
            instruction_sections.append(self._render_block("dynamic_emotion_defensiveness_high"))
        elif defensiveness >= 0.45:
            instruction_sections.append(self._render_block("dynamic_emotion_defensiveness_medium"))
        else:
            instruction_sections.append(self._render_block("dynamic_emotion_defensiveness_low"))

        if speaking_style.get("repair_pattern"):
            instruction_sections.append(
                self._render_block(
                    "dynamic_emotion_repair_section",
                    {"repair_pattern": speaking_style.get("repair_pattern")},
                )
            )

        if activated_memories:
            # 当前 memory_system 还是占位实现，所以这条一般不会出现；
            # 但接口预留好了，未来启用记忆激活时这里能直接承接。
            instruction_sections.append(self._render_block("dynamic_emotion_memory_section"))

        return self._with_trailing_newline(
            render_prompt_section(
                "depression/dynamic_prompt_layers",
                "emotion_layer",
                {
                    "label": emotion.get("label", self.EMOTION_DEFAULTS.get("label", "")),
                    "style": emotion.get("style", self.EMOTION_DEFAULTS.get("style", "")),
                    "intensity": int(round(float(emotion.get("intensity", 0.6) or 0.6) * 10)),
                    "disclosure_level": int(round(float(emotion.get("disclosure_level", 0.3) or 0.3) * 10)),
                    "defensiveness": int(round(float(emotion.get("defensiveness", 0.5) or 0.5) * 10)),
                    "volatility_note": emotion.get(
                        "volatility_note",
                        self.EMOTION_DEFAULTS.get("volatility_note", ""),
                    ),
                    "instruction_sections": "".join(instruction_sections).rstrip(),
                },
            ).strip()
        )

    @staticmethod
    def _combine_layers(layers: List[str]) -> str:
        separator_text = (
            render_prompt_section(
                "depression/dynamic_prompt_layers",
                "layer_separator",
                {},
            ).strip()
            or ("=" * 50)
        )
        separator = "\n" + separator_text + "\n\n"
        # 统一用分隔线拼层，方便人工查看 prompt，也方便 debug 时定位是哪一层出了问题。
        return separator.join([str(item or "").strip() for item in layers if str(item or "").strip()])

    @staticmethod
    def _with_trailing_newline(text: str) -> str:
        value = str(text or "").rstrip()
        return value + "\n" if value else ""

    @staticmethod
    def _render_block(template: str, data: Optional[Dict[str, Any]] = None) -> str:
        rendered = render_prompt_section(
            "depression/dynamic_prompt_layers",
            template,
            data or {},
        ).strip()
        return rendered + "\n" if rendered else ""
