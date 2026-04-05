"""
记忆影响系统 - 基于历史经历动态影响当前症状表现
"""
import copy
from typing import Any, Dict, List, Optional


class TraumaMemorySystem:
    """创伤记忆系统 - 管理和激活创伤记忆"""
    
    def __init__(self, trauma_memories: Optional[List[Dict]] = None):
        """
        初始化创伤记忆系统
        
        Args:
            trauma_memories: 创伤记忆列表
        """
        self.trauma_memories = trauma_memories or self._get_default_memories()
        self.activated_memories: List[Dict] = []
        
    def _get_default_memories(self) -> List[Dict]:
        """获取默认的创伤记忆配置"""
        return [
            {
                "id": "academic_failure",
                "event": "学业失败",
                "description": "大二时因无法完成毕业设计而被迫休学，感到极度羞耻和失败",
                "emotional_charge": 0.9,
                "triggers": ["学习", "评价", "比较", "期望", "成绩", "作业", "设计", "学业"],
                "associated_thoughts": [
                    "我又要让所有人失望了",
                    "我永远都不够好",
                    "我注定是个失败者",
                    "我连最基本的事情都做不好",
                ],
                "physical_reactions": ["心跳加速", "呼吸困难", "胸闷"],
                "activation_threshold": 0.6,
            },
            {
                "id": "social_rejection",
                "event": "社交被拒",
                "description": "因抑郁症状被朋友疏远，感到被抛弃和孤立",
                "emotional_charge": 0.8,
                "triggers": ["朋友", "聚会", "社交", "孤独", "被排斥", "不理解"],
                "associated_thoughts": [
                    "没有人真正关心我",
                    "我是个负担",
                    "我不配拥有朋友",
                    "最终所有人都会离开我",
                ],
                "physical_reactions": ["胸口发紧", "想哭"],
                "activation_threshold": 0.5,
            },
            {
                "id": "perfectionism_trauma",
                "event": "完美主义创伤",
                "description": "从小被要求完美，任何失误都会受到严厉批评",
                "emotional_charge": 0.85,
                "triggers": ["完美", "错误", "批评", "标准", "要求", "失误"],
                "associated_thoughts": [
                    "我必须做到完美",
                    "一个错误就证明我是失败的",
                    "我不能让任何人看到我的弱点",
                    "不完美就等于失败",
                ],
                "physical_reactions": ["肌肉紧张", "头痛"],
                "activation_threshold": 0.7,
            },
            {
                "id": "hopelessness_experience",
                "event": "绝望体验",
                "description": "多次治疗失败后产生的深刻绝望感",
                "emotional_charge": 0.95,
                "triggers": ["治疗", "希望", "未来", "改变", "好转", "康复"],
                "associated_thoughts": [
                    "我永远不会好起来",
                    "治疗没有用",
                    "我的情况是无药可救的",
                    "活着只是在受苦",
                ],
                "physical_reactions": ["极度疲惫", "身体沉重"],
                "activation_threshold": 0.65,
            },
            {
                "id": "self_harm_ideation",
                "event": "自伤意念",
                "description": "在最痛苦时出现的自伤和自杀想法",
                "emotional_charge": 1.0,
                "triggers": ["痛苦", "解脱", "结束", "死", "自杀", "活着"],
                "associated_thoughts": [
                    "也许结束一切是唯一的解脱",
                    "我的存在只会给别人带来痛苦",
                    "死亡可能比活着更好",
                    "我太累了，不想再坚持了",
                ],
                "physical_reactions": ["麻木感", "解离感"],
                "activation_threshold": 0.8,
            },
        ]
    
    def check_memory_activation(self, current_context: Dict, 
                               conversation_content: str = "") -> List[Dict]:
        """
        检查是否有创伤记忆被激活
        
        Args:
            current_context: 当前情境
            conversation_content: 对话内容
            
        Returns:
            被激活的创伤记忆列表
        """
        self.activated_memories.clear()
        
        # 获取情境触发因素
        context_triggers = current_context.get("triggers", [])
        stress_level = current_context.get("overall_stress", 0)
        
        for memory in self.trauma_memories:
            # 检查是否被触发
            is_triggered = self._is_memory_triggered(
                memory, context_triggers, conversation_content, stress_level
            )
            
            if is_triggered:
                self.activated_memories.append(memory)

        return self.activated_memories
    
    def _is_memory_triggered(self, memory: Dict, context_triggers: List[str],
                            content: str, stress_level: float) -> bool:
        """判断记忆是否被触发"""
        # 检查触发词
        memory_triggers = memory.get("triggers", [])
        trigger_match = False
        
        for trigger in memory_triggers:
            if trigger in content or trigger in str(context_triggers):
                trigger_match = True
                break
        
        if not trigger_match:
            return False
        
        # 检查激活阈值
        activation_threshold = memory.get("activation_threshold", 0.5)
        
        # 压力越大，越容易激活记忆
        adjusted_threshold = activation_threshold * (1 - stress_level * 0.3)
        
        # 情感强度越高的记忆越容易被激活
        emotional_charge = memory.get("emotional_charge", 0.5)
        
        return emotional_charge >= adjusted_threshold
    
    def get_activated_thoughts(self) -> List[str]:
        """获取所有被激活记忆的相关想法"""
        thoughts = []
        for memory in self.activated_memories:
            thoughts.extend(memory.get("associated_thoughts", []))
        return thoughts
    
    def get_physical_reactions(self) -> List[str]:
        """获取所有被激活记忆的生理反应"""
        reactions = []
        for memory in self.activated_memories:
            reactions.extend(memory.get("physical_reactions", []))
        return list(set(reactions))  # 去重
    
    def get_memory_description(self) -> str:
        """获取被激活记忆的文字描述"""
        if not self.activated_memories:
            return "当前没有创伤记忆被激活"
        
        desc_parts = ["被激活的创伤记忆："]
        for memory in self.activated_memories:
            desc_parts.append(f"- {memory['event']}：{memory['description']}")
            
            # 添加相关想法
            thoughts = memory.get("associated_thoughts", [])
            if thoughts:
                desc_parts.append(f"  相关想法：{thoughts[0]}")
        
        return "\n".join(desc_parts)
    
    def add_trauma_memory(self, memory: Dict):
        """添加新的创伤记忆"""
        required_fields = ["id", "event", "triggers", "associated_thoughts"]
        if not all(field in memory for field in required_fields):
            raise ValueError(f"创伤记忆必须包含字段: {required_fields}")
        
        # 设置默认值
        memory.setdefault("emotional_charge", 0.7)
        memory.setdefault("activation_threshold", 0.6)
        memory.setdefault("physical_reactions", [])
        memory.setdefault("description", "")
        
        self.trauma_memories.append(memory)
    
    def remove_trauma_memory(self, memory_id: str):
        """移除创伤记忆（用于模拟治疗效果）"""
        self.trauma_memories = [
            m for m in self.trauma_memories if m["id"] != memory_id
        ]
    
    def reduce_memory_intensity(self, memory_id: str, reduction: float = 0.1):
        """降低创伤记忆的情感强度（用于模拟治疗进展）"""
        for memory in self.trauma_memories:
            if memory["id"] == memory_id:
                memory["emotional_charge"] = max(0.1, memory["emotional_charge"] - reduction)
                memory["activation_threshold"] = min(0.9, memory["activation_threshold"] + reduction)
                break

    def to_dict(self) -> Dict[str, Any]:
        """导出可序列化状态。"""
        return {
            "trauma_memories": copy.deepcopy(self.trauma_memories),
            "activated_memory_ids": [
                str(item.get("id", ""))
                for item in self.activated_memories
                if isinstance(item, dict)
            ],
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "TraumaMemorySystem":
        """从序列化状态恢复实例。"""
        payload = payload or {}
        trauma_memories = payload.get("trauma_memories")
        if not isinstance(trauma_memories, list):
            trauma_memories = None
        inst = cls(trauma_memories=copy.deepcopy(trauma_memories) if trauma_memories else None)
        active_ids = payload.get("activated_memory_ids", [])
        if isinstance(active_ids, list):
            active_id_set = {str(item) for item in active_ids if str(item).strip()}
            inst.activated_memories = [
                copy.deepcopy(item)
                for item in inst.trauma_memories
                if isinstance(item, dict) and str(item.get("id", "")) in active_id_set
            ]
        return inst
