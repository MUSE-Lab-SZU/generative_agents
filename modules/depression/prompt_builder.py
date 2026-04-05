"""
动态Prompt构建器 - 构建多层次的动态prompt
"""
from typing import Dict, List, Optional


class DynamicPromptBuilder:
    """动态Prompt构建器 - 构建动态的多层次prompt"""

    def build_prompt(self, base_prompt: str, current_state: str, 
                    state_characteristics: Dict, context: Dict,
                    cognitive_biases: List[Dict], activated_memories: List[Dict]) -> str:
        """
        构建动态的多层次prompt
        
        Args:
            base_prompt: 基础人格和背景
            current_state: 当前症状状态
            state_characteristics: 状态特征
            context: 情境分析结果
            cognitive_biases: 认知偏差列表
            activated_memories: 被激活的创伤记忆
            
        Returns:
            完整的动态prompt
        """
        # 第一层：基础人格和背景
        base_layer = self._build_base_layer(base_prompt)
        
        # 第二层：当前症状状态
        symptom_layer = self._build_symptom_layer(current_state, state_characteristics)
        
        # 第三层：情境感知
        context_layer = self._build_context_layer(context)
        
        # 第四层：认知偏差
        bias_layer = self._build_bias_layer(cognitive_biases)
        
        # 第五层：记忆影响
        memory_layer = self._build_memory_layer(activated_memories)
        
        # 第六层：行为指导
        behavior_layer = self._build_behavior_layer(current_state, state_characteristics, context)
        
        # 组合所有层
        return self._combine_layers([
            base_layer,
            symptom_layer,
            context_layer,
            bias_layer,
            memory_layer,
            behavior_layer,
        ])
    
    def _build_base_layer(self, base_prompt: str) -> str:
        """构建基础层"""
        return f"=== 基础人格层 ===\n{base_prompt}\n"
    
    def _build_symptom_layer(self, state: str, characteristics: Dict) -> str:
        """构建症状状态层"""
        state_names = {
            "severe_episode": "重度抑郁发作期",
            "moderate_episode": "中度抑郁发作期",
            "mild_episode": "轻度抑郁发作期",
            "remission": "症状缓解期",
            "crisis": "危机状态",
        }
        
        state_name = state_names.get(state, state)
        
        lines = [
            "=== 当前症状状态层 ===",
            f"当前处于：{state_name}",
            "",
            "症状强度评估：",
        ]
        
        # 添加主要症状描述
        symptom_descriptions = {
            "mood_score": ("情绪低落", "极度低落", "非常低落", "较为低落", "轻度低落", "情绪稳定"),
            "energy_level": ("精力水平", "极度疲惫", "严重疲惫", "明显疲惫", "轻度疲惫", "精力充沛"),
            "social_avoidance": ("社交回避", "完全回避", "强烈回避", "明显回避", "轻度回避", "社交正常"),
            "cognitive_distortion": ("认知扭曲", "严重扭曲", "明显扭曲", "中度扭曲", "轻度扭曲", "思维清晰"),
            "sleep_disturbance": ("睡眠障碍", "严重失眠", "明显失眠", "睡眠不佳", "轻度失眠", "睡眠正常"),
            "hopelessness": ("绝望感", "极度绝望", "深度绝望", "明显绝望", "轻度绝望", "有希望感"),
        }
        
        for symptom, (name, *levels) in symptom_descriptions.items():
            value = characteristics.get(symptom, 0.5)
            level_idx = min(4, int(value * 5))
            level_desc = levels[level_idx] if level_idx < len(levels) else levels[-1]
            intensity = int(value * 10)
            lines.append(f"- {name}：{level_desc}（强度：{intensity}/10）")
        
        return "\n".join(lines) + "\n"
    
    def _build_context_layer(self, context: Dict) -> str:
        """构建情境感知层"""
        lines = ["=== 情境感知层 ==="]
        
        env = context.get("environment", {})
        social = context.get("social", {})
        
        # 环境信息
        if env:
            lines.append(f"当前环境：{env.get('location', '未知')}")
            lines.append(f"时间：{env.get('time_of_day', '未知')}")
            
            stress = env.get("stress_level", 0)
            if stress > 0.7:
                lines.append("环境压力：高（感到非常不适和焦虑）")
            elif stress > 0.4:
                lines.append("环境压力：中等（感到一定程度的不适）")
            else:
                lines.append("环境压力：低（相对舒适）")
        
        # 社交信息
        if social:
            other = social.get("other_agent", '')
            interaction = social.get("interaction_type", '')
            relationship = social.get("relationship", '')
            
            if other:
                lines.append(f"社交情境：正在与{other}进行{interaction}")
                lines.append(f"关系类型：{relationship}")
                
                support = social.get("emotional_support", 0)
                if support > 0.7:
                    lines.append("对方态度：感受到强烈的支持和关心")
                elif support > 0.4:
                    lines.append("对方态度：感受到一定的支持")
                else:
                    lines.append("对方态度：感受不到明显支持")
        
        return "\n".join(lines) + "\n"
    
    def _build_bias_layer(self, biases: List[Dict]) -> str:
        """构建认知偏差层"""
        if not biases:
            return "=== 认知偏差层 ===\n当前认知相对清晰\n"
        
        lines = [
            "=== 认知偏差层 ===",
            "当前活跃的认知偏差（这些想法会影响你对事物的看法）：",
        ]
        
        for bias in biases:
            lines.append(f"- {bias['name']}：「{bias['thought']}」")
        
        lines.append("\n这些认知偏差会让你倾向于用消极的方式解读事物。")
        
        return "\n".join(lines) + "\n"
    
    def _build_memory_layer(self, memories: List[Dict]) -> str:
        """构建记忆影响层"""
        if not memories:
            return "=== 记忆影响层 ===\n当前没有特别强烈的创伤记忆被激活\n"
        
        lines = [
            "=== 记忆影响层 ===",
            "被激活的创伤记忆（这些记忆正在影响你的情绪和反应）：",
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
    
    def _build_behavior_layer(self, state: str, characteristics: Dict, context: Dict) -> str:
        """构建行为指导层"""
        lines = [
            "=== 行为表现指导层 ===",
            "在当前状态下，你的表现特征：",
        ]
        
        # 根据症状状态给出行为指导
        mood = characteristics.get("mood_score", 0.5)
        energy = characteristics.get("energy_level", 0.5)
        social_avoidance = characteristics.get("social_avoidance", 0.5)
        
        # 语言表达
        if mood < 0.3:
            lines.append("- 语言表达：语速缓慢，声音低沉，用词消极，经常使用「没用」「不行」「算了」等词")
        elif mood < 0.6:
            lines.append("- 语言表达：语气平淡，缺乏情感起伏，偶尔表现出消极")
        else:
            lines.append("- 语言表达：相对正常，但仍可能带有一些消极色彩")
        
        # 社交反应
        if social_avoidance > 0.7:
            lines.append("- 社交反应：强烈想要回避交流，回答简短，试图尽快结束对话")
        elif social_avoidance > 0.4:
            lines.append("- 社交反应：对交流感到疲惫，但会勉强维持，回答相对简短")
        else:
            lines.append("- 社交反应：能够进行基本的社交互动，但不太主动")
        
        # 情绪表达
        if mood < 0.3:
            lines.append("- 情绪表达：可能会表现出绝望、无助、自责，甚至提及自杀想法")
        elif mood < 0.6:
            lines.append("- 情绪表达：表现出明显的悲伤、疲惫和无力感")
        else:
            lines.append("- 情绪表达：情绪相对稳定，但仍带有一些低落")
        
        # 对帮助的反应
        social = context.get("social", {})
        if social.get("emotional_support", 0) > 0.6:
            if mood < 0.3:
                lines.append("- 对关心的反应：虽然感受到关心，但认为自己不配，可能会推开对方")
            else:
                lines.append("- 对关心的反应：能够感受到关心，但难以完全接受和相信")
        
        lines.append("\n请根据以上所有层次的信息，真实地表现出抑郁症患者的状态。")
        
        return "\n".join(lines) + "\n"
    
    def _combine_layers(self, layers: List[str]) -> str:
        """组合所有层"""
        separator = "\n" + "="*50 + "\n\n"
        return separator.join(layers)
    
    def build_simple_prompt(self, base_prompt: str, state: str, 
                          state_characteristics: Dict) -> str:
        """构建简化版prompt（用于不需要完整情境的场景）"""
        base_layer = self._build_base_layer(base_prompt)
        symptom_layer = self._build_symptom_layer(state, state_characteristics)
        
        return f"{base_layer}\n{symptom_layer}"
