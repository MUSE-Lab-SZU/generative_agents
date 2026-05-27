"""主诉链驱动的动态 Prompt 构建器。"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


class DynamicPromptBuilder:
    """构建主诉节点、会话、偏差与瞬时情绪组成的多层 Prompt。"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config if isinstance(config, dict) else {}
        self.include_chain_window = bool(self.config.get("include_chain_window", True))
        self.include_bias_layer = bool(self.config.get("include_bias_layer", True))
        self.include_emotion_layer = bool(self.config.get("include_emotion_layer", True))

    def build_prompt(
        self,
        base_prompt: str,
        current_stage: Dict[str, Any],
        chain_snapshot: Dict[str, Any],
        session_context: Dict[str, Any],
        cognitive_biases: List[Dict[str, Any]],
        activated_memories: List[Dict[str, Any]],
        emotion: Optional[Dict[str, Any]] = None,
    ) -> str:
        # 层次顺序很重要：
        # 基础人格 -> 当前主诉节点 -> 当前会话 -> 偏差 -> 瞬时情绪。
        # 它体现的是“稳定人设在前，当前轮波动在后”的约束方向。
        layers = [
            self._build_base_layer(base_prompt),
            self._build_stage_layer(current_stage, chain_snapshot),
            self._build_context_layer(session_context),
        ]
        if self.include_bias_layer:
            layers.append(self._build_bias_layer(cognitive_biases))
        if self.include_emotion_layer:
            layers.append(self._build_emotion_layer(current_stage, emotion or {}, activated_memories))
        return self._combine_layers(layers)

    def build_simple_prompt(
        self,
        base_prompt: str,
        current_stage: Dict[str, Any],
        chain_snapshot: Optional[Dict[str, Any]] = None,
    ) -> str:
        # simple prompt 只保留最核心的“人格 + 当前主诉节点”，
        # 适合反思/摘要等不需要完整会话层的场景。
        return self._combine_layers(
            [
                self._build_base_layer(base_prompt),
                self._build_stage_layer(
                    current_stage=current_stage if isinstance(current_stage, dict) else {},
                    chain_snapshot=chain_snapshot if isinstance(chain_snapshot, dict) else {},
                ),
            ]
        )

    def _build_base_layer(self, base_prompt: str) -> str:
        return f"=== 基础人格层 ===\n{str(base_prompt or '').strip()}\n"

    def _build_stage_layer(self, current_stage: Dict[str, Any], chain_snapshot: Dict[str, Any]) -> str:
        current_stage = current_stage if isinstance(current_stage, dict) else {}
        current_window = chain_snapshot.get("current_chain_window", []) if isinstance(chain_snapshot.get("current_chain_window", []), list) else []

        # 这些字段都直接来自 complaint_chain 配置中的单个 stage。
        label = str(current_stage.get("label", "未命名主诉节点") or "未命名主诉节点").strip()
        summary = str(current_stage.get("summary", "") or "").strip()
        core_belief = str(current_stage.get("core_belief", "") or "").strip()
        narrative_focus = [str(item) for item in current_stage.get("narrative_focus", [])[:6]] if isinstance(current_stage.get("narrative_focus", []), list) else []
        speaking_style = current_stage.get("speaking_style", {}) if isinstance(current_stage.get("speaking_style", {}), dict) else {}

        lines = [
            "=== 当前主诉节点层 ===",
            f"当前节点：{label}",
        ]
        if summary:
            lines.append(f"节点概述：{summary}")
        if core_belief:
            lines.append(f"核心信念：{core_belief}")
        if narrative_focus:
            # narrative_focus 不是“必须逐字复述的关键词”，
            # 更像给 LLM 的“优先围绕哪些痛点组织表达”的提醒。
            lines.append("当前最容易围绕这些问题组织表达：" + "、".join(narrative_focus))

        if speaking_style:
            lines.append("当前节点下的表达方式：")
            # 这里把结构化风格字段翻译回自然语言提示，
            # 让 LLM 更容易在输出中体现节奏/语气/修正模式。
            lines.append(f"- 语速/节奏：{speaking_style.get('tempo', 'slow')}")
            lines.append(f"- 暴露程度：{speaking_style.get('disclosure', 'guarded')}")
            lines.append(f"- 语气底色：{speaking_style.get('tone', 'flat')}")
            lines.append(f"- 常见修正方式：{speaking_style.get('repair_pattern', '说一点、收一点')}")

        lines.extend(
            [
                "推进原则：",
                "- 你应围绕当前节点说话，不要突然跳到完全无关的远端好转。",
                "- 即使出现一点松动，也应表现为试探性的、会反复的变化。",
                "- 若当前对话没有真正触及这个节点，就不要跨越到下一个节点。",
            ]
        )

        if self.include_chain_window and current_window:
            lines.append("")
            lines.append("当前窗口中的主诉走向：")
            for idx, stage in enumerate(current_window):
                if not isinstance(stage, dict):
                    continue
                # idx=0 永远是当前节点；后面的节点只是“可能的后续方向”，
                # 不是命令式要求角色立即跨过去。
                marker = "当前" if idx == 0 else f"后续{idx}"
                lines.append(
                    f"- [{marker}] {str(stage.get('label', '未知节点') or '未知节点')}：{str(stage.get('summary', '') or '').strip()}"
                )
            if len(current_window) > 1:
                next_label = str(current_window[1].get("label", "") or "").strip() if isinstance(current_window[1], dict) else ""
                if next_label:
                    lines.append(f"- 若本轮确实出现推进，最多只自然靠近「{next_label}」，不要跨越式跳转。")

        return "\n".join(lines) + "\n"

    def _build_context_layer(self, session_context: Dict[str, Any]) -> str:
        session_context = session_context if isinstance(session_context, dict) else {}
        scene = session_context.get("scene", {}) if isinstance(session_context.get("scene", {}), dict) else {}
        participants = session_context.get("participants", {}) if isinstance(session_context.get("participants", {}), dict) else {}
        semantic = session_context.get("semantic_cues", {}) if isinstance(session_context.get("semantic_cues", {}), dict) else {}
        flags = session_context.get("session_flags", {}) if isinstance(session_context.get("session_flags", {}), dict) else {}

        lines = ["=== 会话上下文层 ==="]
        lines.append(f"地点：{scene.get('location', '未知')}")
        lines.append(f"时间：{scene.get('time_of_day', '未知')}")
        if participants.get("other_agent"):
            lines.append(
                f"对话对象：{participants.get('other_agent', '未知')}（关系：{participants.get('relationship', '未知')}）"
            )
        if scene.get("interaction_type"):
            lines.append(f"互动类型：{scene.get('interaction_type', '未知')}")

        topics = [str(item) for item in semantic.get("topics", [])[:6]] if isinstance(semantic.get("topics", []), list) else []
        speech_acts = [str(item) for item in semantic.get("speech_acts", [])[:6]] if isinstance(semantic.get("speech_acts", []), list) else []
        stance = [str(item) for item in semantic.get("stance", [])[:6]] if isinstance(semantic.get("stance", []), list) else []
        if topics:
            # topics 是“你在围绕什么痛点说话”。
            lines.append("识别到的主诉主题：" + "、".join(topics))
        if speech_acts:
            # speech_acts 是“你是怎么说的”，比如求助、淡化、回避。
            lines.append("这轮话语动作：" + "、".join(speech_acts))
        if stance:
            # stance 更接近说话姿态，例如试探、防御、羞耻、低落。
            lines.append("这轮说话姿态：" + "、".join(stance))

        flag_texts = []
        if flags.get("is_help_seeking_frame", False):
            flag_texts.append("当前会话带有求助框架")
        if flags.get("is_evaluative_frame", False):
            flag_texts.append("当前会话带有被评估感")
        if flags.get("is_close_relationship", False):
            flag_texts.append("对方属于较近关系")
        if flags.get("is_professional_frame", False):
            flag_texts.append("当前属于专业支持场景")
        if flags.get("is_minimizing", False):
            flag_texts.append("角色正在淡化严重性")
        if flags.get("is_withdrawing", False):
            flag_texts.append("角色正倾向回避/收缩")
        if flag_texts:
            lines.append("场景附加判断：" + "；".join(flag_texts))

        return "\n".join(lines) + "\n"

    def _build_bias_layer(self, biases: List[Dict[str, Any]]) -> str:
        if not biases:
            return "=== 认知偏差层 ===\n当前没有特别突出的自动化偏差被显性调用。\n"
        lines = [
            "=== 认知偏差层 ===",
            "以下自动化偏差会影响你如何解释现实：",
        ]
        for item in biases:
            # bias 层给的是“内心自动化想法”的示例，而不是要求逐字照搬。
            lines.append(f"- {item.get('name', item.get('type', '未知偏差'))}：「{item.get('thought', '')}」")
        lines.append("这些偏差会让你更容易用绝对化、自责化或悲观化的方式理解正在发生的事。")
        return "\n".join(lines) + "\n"

    def _build_emotion_layer(
        self,
        current_stage: Dict[str, Any],
        emotion: Dict[str, Any],
        activated_memories: List[Dict[str, Any]],
    ) -> str:
        current_stage = current_stage if isinstance(current_stage, dict) else {}
        emotion = emotion if isinstance(emotion, dict) else {}
        speaking_style = current_stage.get("speaking_style", {}) if isinstance(current_stage.get("speaking_style", {}), dict) else {}

        lines = [
            "=== 瞬时情绪与表达指导层 ===",
            f"当前瞬时情绪标签：{emotion.get('label', '低落克制')}",
            f"情绪风格：{emotion.get('style', '说话收着，带保留与迟疑。')}",
            f"情绪强度：{int(round(float(emotion.get('intensity', 0.6) or 0.6) * 10))}/10",
            f"暴露程度：{int(round(float(emotion.get('disclosure_level', 0.3) or 0.3) * 10))}/10",
            f"防御水平：{int(round(float(emotion.get('defensiveness', 0.5) or 0.5) * 10))}/10",
            f"相对上一轮波动：{emotion.get('volatility_note', '当前只允许出现小幅、可逆的情绪摆动。')}",
        ]

        disclosure_level = float(emotion.get("disclosure_level", 0.3) or 0.3)
        defensiveness = float(emotion.get("defensiveness", 0.5) or 0.5)
        intensity = float(emotion.get("intensity", 0.6) or 0.6)

        # 下面不是再算一次 emotion，而是把数值区间翻译成 LLM 更容易执行的语言提示。
        if intensity >= 0.72:
            lines.append("- 表达时应更容易卡住、停顿、语气压着，避免显得轻松流畅。")
        elif intensity >= 0.50:
            lines.append("- 表达时保持低落与谨慎，不要突然变得鲜活或外放。")
        else:
            lines.append("- 表面可以稍微稳定一点，但底层仍要保留脆弱与迟疑。")

        if disclosure_level <= 0.28:
            lines.append("- 你很难直接讲透真实感受，更可能轻描淡写、绕开重点或说一半就收回。")
        elif disclosure_level <= 0.55:
            lines.append("- 你能透露部分真实内容，但通常会马上补一句否认、淡化或自我修正。")
        else:
            lines.append("- 你会比平时更愿意给出一点细节，但这种打开仍然有限且可逆。")

        if defensiveness >= 0.68:
            lines.append("- 一旦感觉被追问或被判断，你要本能地变短、变硬、变保留。")
        elif defensiveness >= 0.45:
            lines.append("- 你仍有防备，但不是完全封死，偶尔会松一点口。")
        else:
            lines.append("- 当前防备稍有松动，但仍不能表现成完全卸下保护。")

        if speaking_style.get("repair_pattern"):
            lines.append(f"- 当前节点典型的自我修正模式：{speaking_style.get('repair_pattern')}。")

        if activated_memories:
            # 当前 memory_system 还是占位实现，所以这条一般不会出现；
            # 但接口预留好了，未来启用记忆激活时这里能直接承接。
            lines.append("- 记忆层当前虽非主驱动，但如有相关旧叙事浮现，也只能作为轻微背景，不要压过主诉节点。")

        lines.append("\n请将以上各层综合起来，真实地表现这个角色在当前主诉节点下的说话方式。")
        return "\n".join(lines) + "\n"

    @staticmethod
    def _combine_layers(layers: List[str]) -> str:
        separator = "\n" + "=" * 50 + "\n\n"
        # 统一用分隔线拼层，方便人工查看 prompt，也方便 debug 时定位是哪一层出了问题。
        return separator.join([str(item or "").strip() for item in layers if str(item or "").strip()])
