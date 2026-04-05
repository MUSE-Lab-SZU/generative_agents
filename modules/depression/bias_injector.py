"""
认知偏差注入器 - 实时注入符合当前症状状态的认知偏差
"""
import random
from typing import Any, Dict, List


class CognitiveBiasInjector:
    """认知偏差注入器 - 根据当前状态和情境注入相应的认知偏差"""
    
    # 认知偏差类型库
    COGNITIVE_BIASES = {
        "catastrophizing": {  # 灾难化思维
            "name": "灾难化思维",
            "templates": [
                "这件事会彻底毁掉我的生活",
                "一切都会变得更糟",
                "我永远无法从这个困境中走出来",
                "这次失败意味着我的人生完了",
                "没有人能帮我，我注定要失败",
            ],
            "trigger_contexts": ["failure", "criticism", "uncertainty", "学业失败", "未来焦虑"],
            "severity_modifiers": {
                "mild": "可能会",
                "moderate": "一定会",
                "severe": "必然会",
            }
        },
        "all_or_nothing": {  # 全或无思维
            "name": "全或无思维",
            "templates": [
                "我要么完美，要么就是彻底的失败者",
                "如果我不能做到最好，那就没有意义",
                "这次失败证明我什么都做不好",
                "我必须在所有方面都成功，否则就是失败",
                "一个错误就说明我是个废物",
            ],
            "trigger_contexts": ["performance", "achievement", "comparison", "学业失败"],
        },
        "personalization": {  # 个人化
            "name": "个人化",
            "templates": [
                "这都是我的错",
                "我让所有人失望了",
                "如果我更努力一点，事情就不会这样",
                "别人的不快乐都是因为我",
                "我是个负担，给大家带来麻烦",
            ],
            "trigger_contexts": ["conflict", "disappointment", "social", "自我价值"],
        },
        "mental_filter": {  # 心理过滤
            "name": "心理过滤",
            "templates": [
                "虽然有些好的方面，但那些都不重要",
                "一个负面评价抹杀了所有正面反馈",
                "我只能看到自己的缺点",
                "别人的鼓励只是客套话，不是真心的",
                "好的事情只是暂时的，坏的才是真实的",
            ],
            "trigger_contexts": ["praise", "success", "support"],
        },
        "should_statements": {  # 应该陈述
            "name": "应该陈述",
            "templates": [
                "我应该更坚强",
                "我不应该有这些感受",
                "我必须让所有人满意",
                "我应该能够独自处理这一切",
                "我不应该需要帮助",
            ],
            "trigger_contexts": ["weakness", "emotion", "help_seeking"],
        },
        "fortune_telling": {  # 算命式思维
            "name": "算命式思维",
            "templates": [
                "我知道这不会有好结果",
                "事情一定会变得更糟",
                "我永远不会好起来",
                "没有人会真正理解我",
                "我的未来一片黑暗",
            ],
            "trigger_contexts": ["future", "treatment", "relationship", "未来焦虑"],
        },
        "emotional_reasoning": {  # 情绪推理
            "name": "情绪推理",
            "templates": [
                "我感觉自己很糟糕，所以我一定很糟糕",
                "我感到绝望，说明情况真的没有希望",
                "我觉得自己是负担，所以我就是负担",
                "我感到害怕，说明真的有危险",
                "我的感受就是事实",
            ],
            "trigger_contexts": ["emotion", "self_evaluation"],
        },
    }
    
    def __init__(self):
        self.active_biases: List[str] = []
        
    def inject_bias(self, current_state: str, context: Dict, 
                   conversation_topic: str = "") -> List[Dict]:
        """
        根据当前状态和情境注入相应的认知偏差
        
        Args:
            current_state: 当前抑郁状态
            context: 情境分析结果
            conversation_topic: 对话主题
            
        Returns:
            注入的认知偏差列表
        """
        injected_biases = []
        
        # 根据症状严重程度确定激活的偏差数量
        severity_map = {
            "severe_episode": 4,
            "moderate_episode": 3,
            "mild_episode": 2,
            "remission": 1,
            "crisis": 5,
        }
        num_biases = severity_map.get(current_state, 2)
        
        # 选择相关的认知偏差
        relevant_biases = self._select_relevant_biases(context, conversation_topic)
        
        # 随机选择要激活的偏差
        selected_biases = random.sample(
            relevant_biases, 
            min(num_biases, len(relevant_biases))
        )
        
        # 生成偏差思维
        for bias_type in selected_biases:
            bias_data = self.COGNITIVE_BIASES[bias_type]
            thought = self._generate_biased_thought(bias_type, current_state, context)
            
            injected_biases.append({
                "type": bias_type,
                "name": bias_data["name"],
                "thought": thought,
                "intensity": self._calculate_bias_intensity(current_state),
            })
        
        self.active_biases = selected_biases
        return injected_biases
    
    def _select_relevant_biases(self, context: Dict, topic: str) -> List[str]:
        """选择与当前情境相关的认知偏差"""
        relevant = []
        
        # 获取触发因素
        triggers = context.get("triggers", [])
        
        for bias_type, bias_data in self.COGNITIVE_BIASES.items():
            # 检查是否与触发因素匹配
            trigger_contexts = bias_data.get("trigger_contexts", [])
            
            # 检查触发匹配
            if any(t in triggers for t in trigger_contexts):
                relevant.append(bias_type)
                continue
            
            # 检查话题匹配
            if any(ctx in topic for ctx in trigger_contexts):
                relevant.append(bias_type)
                continue
        
        # 如果没有特别相关的，返回所有偏差类型
        if not relevant:
            relevant = list(self.COGNITIVE_BIASES.keys())
        
        return relevant
    
    def _generate_biased_thought(self, bias_type: str, state: str, context: Dict) -> str:
        """生成具体的偏差思维"""
        bias_data = self.COGNITIVE_BIASES[bias_type]
        template = random.choice(bias_data["templates"])
        
        # 根据严重程度修改
        severity_modifiers = bias_data.get("severity_modifiers", {})
        if severity_modifiers:
            if state in ["severe_episode", "crisis"]:
                modifier = severity_modifiers.get("severe", "")
            elif state == "moderate_episode":
                modifier = severity_modifiers.get("moderate", "")
            else:
                modifier = severity_modifiers.get("mild", "")
            
            if modifier and "会" in template:
                template = template.replace("会", modifier)
        
        return template
    
    def _calculate_bias_intensity(self, state: str) -> float:
        """计算认知偏差的强度"""
        intensity_map = {
            "severe_episode": 0.9,
            "moderate_episode": 0.7,
            "mild_episode": 0.5,
            "remission": 0.3,
            "crisis": 1.0,
        }
        return intensity_map.get(state, 0.5)
    
    def get_bias_description(self, biases: List[Dict]) -> str:
        """获取认知偏差的文字描述"""
        if not biases:
            return "当前没有明显的认知偏差"
        
        desc_parts = ["当前活跃的认知偏差："]
        for bias in biases:
            desc_parts.append(f"- {bias['name']}：{bias['thought']}")
        
        return "\n".join(desc_parts)
    
    def should_activate_bias(self, bias_type: str, conversation_topic: str) -> bool:
        """判断是否应该激活特定的认知偏差"""
        bias_data = self.COGNITIVE_BIASES.get(bias_type)
        if not bias_data:
            return False
        
        trigger_contexts = bias_data.get("trigger_contexts", [])
        return any(ctx in conversation_topic for ctx in trigger_contexts)

    def to_dict(self) -> Dict[str, Any]:
        """导出可序列化状态。"""
        return {
            "active_biases": [str(item) for item in self.active_biases],
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "CognitiveBiasInjector":
        """从序列化状态恢复实例。"""
        payload = payload or {}
        inst = cls()
        raw = payload.get("active_biases", [])
        if isinstance(raw, list):
            inst.active_biases = [str(item) for item in raw]
        return inst
