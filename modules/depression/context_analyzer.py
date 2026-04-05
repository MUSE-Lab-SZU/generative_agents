"""
情境感知引擎 - 分析环境和社交情境对症状的影响
"""
import copy
from typing import Any, Dict, List, Optional


class ContextAnalyzer:
    """情境感知引擎 - 根据环境和社交情境动态调整症状表现"""
    
    # 环境压力评分
    LOCATION_STRESS = {
        "家": 0.3,
        "卧室": 0.2,
        "公园": 0.4,
        "心理咨询室": 0.6,
        "公共场所": 0.7,
        "陌生环境": 0.8,
    }
    
    # 社交密度压力
    SOCIAL_DENSITY_STRESS = {
        "独处": 0.2,
        "一对一": 0.5,
        "小群体": 0.7,
        "大群体": 0.9,
    }
    
    # 关系类型影响
    RELATIONSHIP_IMPACT = {
        "亲密朋友": {"support": 0.8, "pressure": 0.3},
        "普通朋友": {"support": 0.5, "pressure": 0.5},
        "治疗师": {"support": 0.9, "pressure": 0.4},
        "陌生人": {"support": 0.1, "pressure": 0.8},
        "冲突关系": {"support": 0.0, "pressure": 0.9},
    }
    
    # 互动类型影响
    INTERACTION_TYPE_IMPACT = {
        "日常活动": {"stress": 0.3, "trigger_potential": 0.2},
        "闲聊": {"stress": 0.4, "trigger_potential": 0.2},
        "资源等待": {"stress": 0.45, "trigger_potential": 0.3},
        "关系回顾": {"stress": 0.35, "trigger_potential": 0.3},
        "深度交流": {"stress": 0.6, "trigger_potential": 0.5},
        "寻求帮助": {"stress": 0.7, "trigger_potential": 0.6},
        "被询问状况": {"stress": 0.8, "trigger_potential": 0.7},
        "冲突": {"stress": 0.95, "trigger_potential": 0.9},
        "治疗对话": {"stress": 0.5, "trigger_potential": 0.8},
    }
    
    # 触发词库
    TRIGGER_KEYWORDS = {
        "学业失败": ["学习", "成绩", "考试", "作业", "毕业", "学业", "失败", "不及格"],
        "自我价值": ["没用", "失败者", "废物", "负担", "价值", "意义", "能力"],
        "社交压力": ["聚会", "见面", "社交", "朋友", "孤独", "被排斥"],
        "未来焦虑": ["未来", "计划", "目标", "希望", "前途", "工作"],
        "自杀意念": ["死", "结束", "解脱", "活着", "痛苦", "自杀"],
    }
    
    def __init__(self):
        self.current_context = {}
        self.trigger_history: List[str] = []
        
    def analyze_environment(self, location: str, time_of_day: str, 
                          social_density: str = "独处") -> Dict:
        """
        分析环境对症状的影响
        
        Args:
            location: 当前位置
            time_of_day: 时间段
            social_density: 社交密度
            
        Returns:
            环境分析结果
        """
        # 计算环境压力
        stress_level = self._calculate_location_stress(location)
        stress_level += self.SOCIAL_DENSITY_STRESS.get(social_density, 0.5)
        stress_level = min(1.0, stress_level)
        
        # 计算舒适度
        comfort_level = 1.0 - stress_level
        if location in ["家", "卧室"]:
            comfort_level += 0.2
        comfort_level = min(1.0, comfort_level)
        
        # 识别潜在触发因素
        trigger_potential = self._assess_trigger_potential(location, time_of_day)
        
        return {
            "stress_level": stress_level,
            "comfort_level": comfort_level,
            "trigger_potential": trigger_potential,
            "location": location,
            "time_of_day": time_of_day,
            "social_density": social_density,
        }
    
    def analyze_social_interaction(self, other_agent: str, 
                                   relationship: str,
                                   interaction_type: str,
                                   conversation_content: str = "") -> Dict:
        """
        分析社交互动对症状的影响
        
        Args:
            other_agent: 对方agent名称
            relationship: 关系类型
            interaction_type: 互动类型
            conversation_content: 对话内容
            
        Returns:
            社交互动分析结果
        """
        # 评估情感支持
        relationship_data = self.RELATIONSHIP_IMPACT.get(relationship, 
                                                        {"support": 0.3, "pressure": 0.6})
        emotional_support = relationship_data["support"]
        
        # 评估社交压力
        social_pressure = relationship_data["pressure"]
        interaction_data = self.INTERACTION_TYPE_IMPACT.get(interaction_type,
                                                            {"stress": 0.5, "trigger_potential": 0.5})
        social_pressure += interaction_data["stress"]
        social_pressure = min(1.0, social_pressure / 2)
        
        # 检查触发词
        triggers = self._check_triggers(conversation_content)
        trigger_activation = len(triggers) > 0
        
        return {
            "emotional_support": emotional_support,
            "social_pressure": social_pressure,
            "trigger_activation": trigger_activation,
            "triggers": triggers,
            "other_agent": other_agent,
            "relationship": relationship,
            "interaction_type": interaction_type,
        }
    
    def analyze_full_context(self, location: str, time_of_day: str,
                           other_agent: Optional[str] = None,
                           relationship: Optional[str] = None,
                           interaction_type: Optional[str] = None,
                           conversation_content: str = "") -> Dict:
        """
        综合分析当前完整情境
        
        Returns:
            完整的情境分析结果
        """
        # 环境分析
        social_density = "独处" if not other_agent else "一对一"
        env_analysis = self.analyze_environment(location, time_of_day, social_density)
        
        # 社交分析（如果有）
        social_analysis = {}
        if other_agent and relationship and interaction_type:
            social_analysis = self.analyze_social_interaction(
                other_agent, relationship, interaction_type, conversation_content
            )
        
        # 综合评估
        overall_stress = env_analysis["stress_level"]
        if social_analysis:
            overall_stress = (overall_stress + social_analysis["social_pressure"]) / 2
        
        # 收集所有触发因素
        all_triggers = []
        if social_analysis and social_analysis.get("triggers"):
            all_triggers.extend(social_analysis["triggers"])
        
        # 根据压力水平添加状态触发
        if overall_stress > 0.8:
            all_triggers.append("extreme_stress")
        elif overall_stress > 0.6:
            all_triggers.append("stress")
        
        if social_analysis and social_analysis.get("emotional_support", 0) > 0.7:
            all_triggers.append("support")

        if all_triggers:
            self.trigger_history.extend([str(item) for item in all_triggers])
            # 仅保留最近触发，避免无界增长
            if len(self.trigger_history) > 200:
                self.trigger_history = self.trigger_history[-200:]
        
        self.current_context = {
            "environment": env_analysis,
            "social": social_analysis,
            "overall_stress": overall_stress,
            "triggers": all_triggers,
            "timestamp": time_of_day,
        }
        
        return self.current_context
    
    def _calculate_location_stress(self, location: str) -> float:
        """计算位置压力"""
        for key, value in self.LOCATION_STRESS.items():
            if key in location:
                return value
        return 0.5  # 默认中等压力
    
    def _assess_trigger_potential(self, location: str, time_of_day: str) -> float:
        """评估触发潜力"""
        trigger_potential = 0.3
        
        # 夜晚和清晨触发潜力更高
        if time_of_day in ["night", "morning"]:
            trigger_potential += 0.2
        
        # 某些地点触发潜力更高
        if "心理咨询室" in location:
            trigger_potential += 0.3
        
        return min(1.0, trigger_potential)
    
    def _check_triggers(self, content: str) -> List[str]:
        """检查对话内容中的触发词"""
        triggered = []
        
        for trigger_type, keywords in self.TRIGGER_KEYWORDS.items():
            for keyword in keywords:
                if keyword in content:
                    triggered.append(trigger_type)
                    break
        
        return triggered

    def get_context_description(self) -> str:
        """获取当前情境的文字描述"""
        if not self.current_context:
            return "当前情境未分析"
        
        env = self.current_context.get("environment", {})
        social = self.current_context.get("social", {})
        
        desc_parts = []
        
        # 环境描述
        desc_parts.append(f"当前环境：{env.get('location', '未知')}")
        desc_parts.append(f"时间：{env.get('time_of_day', '未知')}")
        
        # 社交描述
        if social:
            desc_parts.append(f"社交情境：与{social.get('other_agent', '某人')}进行{social.get('interaction_type', '交流')}")
            desc_parts.append(f"关系类型：{social.get('relationship', '未知')}")
        
        # 压力水平
        stress = self.current_context.get("overall_stress", 0)
        if stress > 0.7:
            desc_parts.append("环境压力：高")
        elif stress > 0.4:
            desc_parts.append("环境压力：中等")
        else:
            desc_parts.append("环境压力：低")
        
        return "，".join(desc_parts)

    def to_dict(self) -> Dict[str, Any]:
        """导出可序列化状态。"""
        current_context = (
            copy.deepcopy(self.current_context)
            if isinstance(self.current_context, dict)
            else {}
        )
        trigger_history = list(self.trigger_history) if isinstance(self.trigger_history, list) else []
        return {
            "current_context": current_context,
            "trigger_history": [str(item) for item in trigger_history],
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ContextAnalyzer":
        """从序列化状态恢复实例。"""
        payload = payload or {}
        inst = cls()
        context = payload.get("current_context", {})
        if isinstance(context, dict):
            inst.current_context = copy.deepcopy(context)
        history = payload.get("trigger_history", [])
        if isinstance(history, list):
            inst.trigger_history = [str(item) for item in history][-200:]
        return inst
