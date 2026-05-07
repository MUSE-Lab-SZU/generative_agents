"""
动态 Prompt 构建器。

本版本不再围绕“症状状态转换层”组织，而是围绕“主诉认知路线图层”组织，
同时保留由当前主诉阶段投影得到的衍生症状侧写，供下游表现层使用。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


class DynamicPromptBuilder:
    """构建主诉链驱动的多层动态 prompt。"""

    def build_prompt(
        self,
        base_prompt: str,
        current_state: str,
        current_stage: Dict[str, Any],
        roadmap_snapshot: Dict[str, Any],
        state_characteristics: Dict[str, float],
        context: Dict[str, Any],
        cognitive_biases: List[Dict],
        activated_memories: List[Dict],
    ) -> str:
        base_layer = self._build_base_layer(base_prompt)
        roadmap_layer = self._build_roadmap_layer(
            current_state=current_state,
            current_stage=current_stage,
            roadmap_snapshot=roadmap_snapshot,
            state_characteristics=state_characteristics,
        )
        context_layer = self._build_context_layer(context)
        bias_layer = self._build_bias_layer(cognitive_biases)
        memory_layer = self._build_memory_layer(activated_memories)
        behavior_layer = self._build_behavior_layer(
            current_state=current_state,
            current_stage=current_stage,
            roadmap_snapshot=roadmap_snapshot,
            state_characteristics=state_characteristics,
            context=context,
        )
        return self._combine_layers(
            [
                base_layer,
                roadmap_layer,
                context_layer,
                bias_layer,
                memory_layer,
                behavior_layer,
            ]
        )

    def _build_base_layer(self, base_prompt: str) -> str:
        return f"=== 基础人格层 ===\n{base_prompt}\n"

    def _build_roadmap_layer(
        self,
        current_state: str,
        current_stage: Dict[str, Any],
        roadmap_snapshot: Dict[str, Any],
        state_characteristics: Dict[str, float],
    ) -> str:
        state_name = self._legacy_state_name(current_state)
        pointer = int(roadmap_snapshot.get("current_pointer", 0) or 0)
        total = int(roadmap_snapshot.get("total_chain_length", 0) or 0)
        current_chain = roadmap_snapshot.get("current_chain", [])
        if not isinstance(current_chain, list):
            current_chain = []

        lines = [
            "=== 主诉认知路线图层 ===",
            f"当前粗粒度状态：{state_name}",
            f"当前认知阶段：{self._format_stage_title(current_stage)}",
            f"阶段描述：{str(current_stage.get('description', '') or '').strip()}",
            f"路线图位置：第{pointer + 1}阶段 / 当前已规划{max(total, len(current_chain))}阶段",
            "",
            "推进原则：",
            "- 你当前的表达应优先围绕“当前认知阶段”展开，不要突然跳到远端好转。",
            "- 即使出现松动，也应表现为试探性、可逆、会反复的变化。",
            "- 只有真正触及当前阶段的表达后，路线图才会向后推进。",
            "",
        ]

        if current_chain:
            lines.append("当前窗口内最可能的主诉走向：")
            for idx, stage in enumerate(current_chain, start=1):
                if not isinstance(stage, dict):
                    continue
                marker = "当前" if idx == 1 else f"后续{idx - 1}"
                lines.append(f"- [{marker}] {self._format_stage_title(stage)}：{str(stage.get('description', '') or '').strip()}")
            lines.append("")

        lines.extend(
            [
                "由当前阶段投影出的衍生表现：",
                *self._render_characteristics(state_characteristics),
            ]
        )

        if any(bool(item.get("terminal_recovery", False)) for item in current_chain if isinstance(item, dict)):
            lines.append(
                "注意：若后续窗口中出现 [TERMINAL_RECOVERY] 节点，那只代表潜在认知突破窗口，不代表此刻已经康复。"
            )
        if bool(roadmap_snapshot.get("reached_terminal_recovery", False)):
            lines.append("当前路线图已触达临床痊愈节点，但表达仍应保持人格连续性，而非突然变得完全轻松。")

        return "\n".join(lines) + "\n"

    def _render_characteristics(self, characteristics: Dict[str, float]) -> List[str]:
        symptom_descriptions = {
            "mood_score": {
                "name": "情绪低落",
                "levels": ["情绪相对稳定", "轻度低落", "较为低落", "非常低落", "极度低落"],
                "inverse": True,
            },
            "energy_level": {
                "name": "精力水平",
                "levels": ["精力相对可用", "轻度疲惫", "明显疲惫", "严重疲惫", "极度疲惫"],
                "inverse": True,
            },
            "social_avoidance": {
                "name": "社交回避",
                "levels": ["能够维持接触", "轻度回避", "明显回避", "强烈回避", "完全回避"],
                "inverse": False,
            },
            "cognitive_distortion": {
                "name": "认知扭曲",
                "levels": ["思路较清晰", "轻度扭曲", "中度扭曲", "明显扭曲", "严重扭曲"],
                "inverse": False,
            },
            "sleep_disturbance": {
                "name": "睡眠障碍",
                "levels": ["睡眠相对稳定", "轻度失眠/紊乱", "睡眠不佳", "明显失眠/紊乱", "严重失眠/紊乱"],
                "inverse": False,
            },
            "hopelessness": {
                "name": "绝望感",
                "levels": ["保留一定希望", "轻度绝望", "明显绝望", "深度绝望", "极度绝望"],
                "inverse": False,
            },
        }
        lines: List[str] = []
        for symptom, meta in symptom_descriptions.items():
            value = float(characteristics.get(symptom, 0.5) or 0.5)
            severity = (1.0 - value) if meta["inverse"] else value
            severity = max(0.0, min(1.0, severity))
            level_idx = min(4, int(round(severity * 4)))
            level_desc = meta["levels"][level_idx]
            intensity = int(round(severity * 10))
            lines.append(f"- {meta['name']}：{level_desc}（强度：{intensity}/10）")
        return lines

    def _build_context_layer(self, context: Dict) -> str:
        lines = ["=== 情境感知层 ==="]

        env = context.get("environment", {}) if isinstance(context.get("environment", {}), dict) else {}
        social = context.get("social", {}) if isinstance(context.get("social", {}), dict) else {}

        if env:
            lines.append(f"当前环境：{env.get('location', '未知')}")
            lines.append(f"时间：{env.get('time_of_day', '未知')}")
            stress = env.get("stress_level", 0)
            if stress > 0.7:
                lines.append("环境压力：高（容易放大负性体验和防御）")
            elif stress > 0.4:
                lines.append("环境压力：中等（会带来一定紧绷感）")
            else:
                lines.append("环境压力：低（相对更容易松动一点）")

        if social:
            other = social.get("other_agent", "")
            interaction = social.get("interaction_type", "")
            relationship = social.get("relationship", "")
            if other:
                lines.append(f"社交情境：正在与{other}进行{interaction}")
                lines.append(f"关系类型：{relationship}")

                support = social.get("emotional_support", 0)
                if support > 0.7:
                    lines.append("对方态度：感受到较强支持，但你未必能稳定接住")
                elif support > 0.4:
                    lines.append("对方态度：感受到一定支持，但仍可能半信半疑")
                else:
                    lines.append("对方态度：几乎感受不到稳定支持")

        triggers = context.get("triggers", [])
        if isinstance(triggers, list) and triggers:
            lines.append("当前显著触发词：" + "、".join([str(item) for item in triggers[:8]]))

        return "\n".join(lines) + "\n"

    def _build_bias_layer(self, biases: List[Dict]) -> str:
        if not biases:
            return "=== 认知偏差层 ===\n当前认知暂未出现特别突出的自动化偏差\n"

        lines = [
            "=== 认知偏差层 ===",
            "当前活跃的自动化偏差（会影响你如何解释现实）：",
        ]
        for bias in biases:
            lines.append(f"- {bias['name']}：「{bias['thought']}」")
        lines.append("\n这些偏差会让你更容易用负向、绝对化或自责化的方式理解发生的事。")
        return "\n".join(lines) + "\n"

    def _build_memory_layer(self, memories: List[Dict]) -> str:
        if not memories:
            return "=== 记忆影响层 ===\n当前没有特别强烈的创伤记忆被激活\n"

        lines = [
            "=== 记忆影响层 ===",
            "被激活的创伤记忆（这些记忆会拉扯你回到旧的解释框架）：",
        ]
        for memory in memories:
            lines.append(f"\n【{memory['event']}】")
            lines.append(f"  {memory.get('description', '')}")
            thoughts = memory.get("associated_thoughts", [])
            if thoughts:
                lines.append(f"  相关想法：{thoughts[0]}")
            reactions = memory.get("physical_reactions", [])
            if reactions:
                lines.append(f"  生理反应：{', '.join(reactions[:2])}")
        return "\n".join(lines) + "\n"

    def _build_behavior_layer(
        self,
        current_state: str,
        current_stage: Dict[str, Any],
        roadmap_snapshot: Dict[str, Any],
        state_characteristics: Dict[str, float],
        context: Dict[str, Any],
    ) -> str:
        lines = [
            "=== 行为表现指导层 ===",
            "在当前路线图节点下，你的表现应遵循以下主轴：",
            f"- 主诉主轴：围绕「{str(current_stage.get('label', '') or '').strip()}」组织表达。",
            f"- 阶段内核：{str(current_stage.get('description', '') or '').strip()}",
        ]

        mood = state_characteristics.get("mood_score", 0.5)
        social_avoidance = state_characteristics.get("social_avoidance", 0.5)
        openness = float(current_stage.get("openness_level", 0.25) or 0.25)
        hope = float(current_stage.get("hopefulness_level", 0.12) or 0.12)

        if mood < 0.3:
            lines.append("- 语言表达：语速偏慢，停顿较多，用词更灰暗，容易出现“没用”“算了”“没什么意义”之类表述。")
        elif mood < 0.6:
            lines.append("- 语言表达：语气平，情感起伏不大，会不自觉流露疲惫和消极。")
        else:
            lines.append("- 语言表达：表面上较为平稳，但底层仍保留迟疑、谨慎与残余低落。")

        if social_avoidance > 0.7:
            lines.append("- 社交反应：本能地想回避深谈，回答偏短，倾向把话题往外推或尽快结束。")
        elif social_avoidance > 0.4:
            lines.append("- 社交反应：会勉强维持交流，但不太主动展开，常常说一半又收回去。")
        else:
            lines.append("- 社交反应：可以维持基本互动，偶尔愿意多说一点，但不会彻底放下防备。")

        if openness < 0.25:
            lines.append("- 暴露程度：你很难直接讲透真实感受，更多是模糊、绕开、轻描淡写或否认严重度。")
        elif openness < 0.5:
            lines.append("- 暴露程度：你能透露部分真实感受，但往往会马上补一句淡化、否认或自我修正。")
        else:
            lines.append("- 暴露程度：你愿意把困扰说得更具体，但仍会保留迟疑，不会突然全盘通透。")

        if hope < 0.25:
            lines.append("- 变化感：即使别人关心你，你也更容易先想到“这没什么用”或“迟早还是会回去”。")
        elif hope < 0.55:
            lines.append("- 变化感：你可能会短暂出现一点松动，但这种松动并不稳定，很容易反复。")
        else:
            lines.append("- 变化感：你开始能想象“也许可以不完全按旧方式理解自己”，但仍需保留残余困难。")

        social = context.get("social", {}) if isinstance(context.get("social", {}), dict) else {}
        if social.get("emotional_support", 0) > 0.6:
            if hope < 0.30:
                lines.append("- 对关心的反应：虽然能感觉到对方在接住你，但你会下意识怀疑自己是否配得上，甚至想推开。")
            else:
                lines.append("- 对关心的反应：你能部分感到被理解，可能愿意多说一点，但仍保留试探与保留。")

        current_chain = roadmap_snapshot.get("current_chain", [])
        if isinstance(current_chain, list) and len(current_chain) > 1:
            next_stage = current_chain[1] if isinstance(current_chain[1], dict) else None
            if isinstance(next_stage, dict):
                lines.append(f"- 潜在下一步：若本轮出现松动，最多只自然靠近「{self._format_stage_title(next_stage)}」，不要跨越式跳转。")

        lines.append("\n请根据以上所有层次的信息，真实地表现出该抑郁角色此刻的认知路线与说话方式。")
        return "\n".join(lines) + "\n"

    def _combine_layers(self, layers: List[str]) -> str:
        separator = "\n" + "=" * 50 + "\n\n"
        return separator.join(layers)

    def build_simple_prompt(
        self,
        base_prompt: str,
        state: str,
        state_characteristics: Dict,
        current_stage: Optional[Dict[str, Any]] = None,
        roadmap_snapshot: Optional[Dict[str, Any]] = None,
    ) -> str:
        base_layer = self._build_base_layer(base_prompt)
        roadmap_layer = self._build_roadmap_layer(
            current_state=state,
            current_stage=current_stage if isinstance(current_stage, dict) else {},
            roadmap_snapshot=roadmap_snapshot if isinstance(roadmap_snapshot, dict) else {},
            state_characteristics=state_characteristics,
        )
        return f"{base_layer}\n{roadmap_layer}"

    def _legacy_state_name(self, state: str) -> str:
        state_names = {
            "severe_episode": "重度抑郁发作期（衍生映射）",
            "moderate_episode": "中度抑郁发作期（衍生映射）",
            "mild_episode": "轻度抑郁发作期（衍生映射）",
            "remission": "症状缓解期（衍生映射）",
            "crisis": "危机状态（衍生映射）",
        }
        return state_names.get(str(state or ""), str(state or "未知"))

    def _format_stage_title(self, stage: Dict[str, Any]) -> str:
        if not isinstance(stage, dict):
            return "未知阶段"
        label = str(stage.get("label", "") or "未知阶段").strip()
        if bool(stage.get("terminal_recovery", False)):
            return f"{label} [TERMINAL_RECOVERY]"
        return label
