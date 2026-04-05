"""
症状状态机 - 管理抑郁症状的动态状态转换
"""
import random
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional


class DepressionState:
    """抑郁症状态枚举"""
    SEVERE_EPISODE = "severe_episode"      # 重度发作期
    MODERATE_EPISODE = "moderate_episode"  # 中度发作期
    MILD_EPISODE = "mild_episode"          # 轻度发作期
    REMISSION = "remission"                # 缓解期
    CRISIS = "crisis"                      # 危机状态


class SymptomStateMachine:
    """症状状态机 - 基于有限状态机的症状动态模型"""
    
    # 状态特征矩阵
    STATE_CHARACTERISTICS = {
        DepressionState.SEVERE_EPISODE: {
            "mood_score": 0.1,           # 情绪评分 (0-1)
            "energy_level": 0.2,         # 精力水平
            "social_avoidance": 0.9,     # 社交回避程度
            "cognitive_distortion": 0.8,  # 认知扭曲程度
            "suicidal_ideation": 0.7,    # 自杀意念强度
            "sleep_disturbance": 0.9,    # 睡眠障碍程度
            "appetite_change": 0.8,      # 食欲改变程度
            "concentration": 0.2,        # 注意力集中程度
            "self_worth": 0.1,          # 自我价值感
            "hopelessness": 0.9,        # 绝望感
        },
        DepressionState.MODERATE_EPISODE: {
            "mood_score": 0.3,
            "energy_level": 0.4,
            "social_avoidance": 0.7,
            "cognitive_distortion": 0.6,
            "suicidal_ideation": 0.4,
            "sleep_disturbance": 0.7,
            "appetite_change": 0.6,
            "concentration": 0.4,
            "self_worth": 0.3,
            "hopelessness": 0.7,
        },
        DepressionState.MILD_EPISODE: {
            "mood_score": 0.5,
            "energy_level": 0.6,
            "social_avoidance": 0.5,
            "cognitive_distortion": 0.4,
            "suicidal_ideation": 0.2,
            "sleep_disturbance": 0.5,
            "appetite_change": 0.4,
            "concentration": 0.6,
            "self_worth": 0.5,
            "hopelessness": 0.5,
        },
        DepressionState.REMISSION: {
            "mood_score": 0.7,
            "energy_level": 0.8,
            "social_avoidance": 0.3,
            "cognitive_distortion": 0.2,
            "suicidal_ideation": 0.0,
            "sleep_disturbance": 0.3,
            "appetite_change": 0.2,
            "concentration": 0.8,
            "self_worth": 0.7,
            "hopelessness": 0.2,
        },
        DepressionState.CRISIS: {
            "mood_score": 0.0,
            "energy_level": 0.1,
            "social_avoidance": 1.0,
            "cognitive_distortion": 1.0,
            "suicidal_ideation": 0.95,
            "sleep_disturbance": 1.0,
            "appetite_change": 0.9,
            "concentration": 0.1,
            "self_worth": 0.0,
            "hopelessness": 1.0,
        },
    }
    
    # 状态转换规则
    TRANSITION_RULES = {
        DepressionState.SEVERE_EPISODE: {
            DepressionState.CRISIS: {"triggers": ["extreme_stress", "trauma", "severe_conflict"], "probability": 0.15},
            DepressionState.MODERATE_EPISODE: {"triggers": ["positive_interaction", "therapy", "time"], "probability": 0.05},
        },
        DepressionState.MODERATE_EPISODE: {
            DepressionState.SEVERE_EPISODE: {"triggers": ["stress", "negative_event", "isolation"], "probability": 0.2},
            DepressionState.MILD_EPISODE: {"triggers": ["support", "therapy", "positive_event"], "probability": 0.1},
        },
        DepressionState.MILD_EPISODE: {
            DepressionState.MODERATE_EPISODE: {"triggers": ["stress", "setback"], "probability": 0.15},
            DepressionState.REMISSION: {"triggers": ["sustained_support", "therapy_progress"], "probability": 0.08},
        },
        DepressionState.REMISSION: {
            DepressionState.MILD_EPISODE: {"triggers": ["stress", "trigger_event"], "probability": 0.1},
        },
        DepressionState.CRISIS: {
            DepressionState.SEVERE_EPISODE: {"triggers": ["intervention", "support", "time"], "probability": 0.3},
        },
    }
    
    def __init__(
        self,
        initial_state: str = DepressionState.SEVERE_EPISODE,
        transition_sensitivity: float = 0.7,
        minimum_state_duration: int = 30,
        now_provider: Optional[Callable[[], datetime]] = None,
    ):
        """
        初始化症状状态机
        
        Args:
            initial_state: 初始状态
            transition_sensitivity: 状态转换敏感度 (0-1)
            minimum_state_duration: 最小状态持续时间（分钟）
        """
        if initial_state not in self.STATE_CHARACTERISTICS:
            initial_state = DepressionState.SEVERE_EPISODE
        self.current_state = initial_state
        self.transition_sensitivity = float(transition_sensitivity)
        self.minimum_state_duration = int(minimum_state_duration)

        self._now_provider: Callable[[], datetime] = (
            now_provider if callable(now_provider) else datetime.now
        )
        self.state_start_time = self._now()
        self.state_history: List[Dict] = []
        self.accumulated_triggers: Dict[str, float] = {}

    def _now(self) -> datetime:
        """返回当前时间，支持外部注入仿真时钟。"""
        try:
            now_obj = self._now_provider()
        except Exception:
            now_obj = datetime.now()
        return _coerce_datetime(now_obj)

    def set_now_provider(self, now_provider: Optional[Callable[[], datetime]]) -> None:
        """设置当前时间提供器。"""
        if callable(now_provider):
            self._now_provider = now_provider
        
    def get_current_state(self) -> str:
        """获取当前状态"""
        return self.current_state
    
    def get_state_characteristics(self) -> Dict[str, float]:
        """获取当前状态的特征"""
        state = self.current_state
        if state not in self.STATE_CHARACTERISTICS:
            state = DepressionState.SEVERE_EPISODE
        return self.STATE_CHARACTERISTICS[state].copy()
    
    def update_state(self, context_analysis: Dict) -> bool:
        """
        根据情境分析更新状态
        
        Args:
            context_analysis: 情境分析结果
            
        Returns:
            是否发生了状态转换
        """
        # 检查是否满足最小持续时间
        time_in_state = (self._now() - self.state_start_time).total_seconds() / 60
        if time_in_state < self.minimum_state_duration:
            return False
        
        # 累积触发因子
        triggers = context_analysis.get("triggers", [])
        for trigger in triggers:
            self.accumulated_triggers[trigger] = self.accumulated_triggers.get(trigger, 0) + 1
        
        # 检查可能的状态转换
        possible_transitions = self.TRANSITION_RULES.get(self.current_state, {})
        
        for next_state, rule in possible_transitions.items():
            # 检查触发条件
            trigger_match = any(t in self.accumulated_triggers for t in rule["triggers"])
            
            if trigger_match:
                # 计算转换概率
                base_probability = rule["probability"]
                adjusted_probability = base_probability * self.transition_sensitivity
                
                # 根据触发强度调整概率
                trigger_strength = sum(
                    self.accumulated_triggers.get(t, 0) 
                    for t in rule["triggers"]
                ) / len(rule["triggers"])
                adjusted_probability *= (1 + trigger_strength * 0.1)
                
                # 随机决定是否转换
                if random.random() < adjusted_probability:
                    self._transition_to(next_state, triggers)
                    return True
        
        return False
    
    def _transition_to(self, new_state: str, triggers: List[str]):
        """执行状态转换"""
        now_obj = self._now()
        # 记录历史
        self.state_history.append({
            "from_state": self.current_state,
            "to_state": new_state,
            "timestamp": now_obj,
            "duration_minutes": (now_obj - self.state_start_time).total_seconds() / 60,
            "triggers": triggers.copy(),
        })
        
        # 更新状态
        self.current_state = new_state
        self.state_start_time = now_obj
        self.accumulated_triggers.clear()
    
    def force_transition(self, new_state: str, reason: str = "manual"):
        """强制转换到指定状态（用于测试或特殊情况）"""
        self._transition_to(new_state, [reason])
    
    def get_state_duration(self) -> float:
        """获取当前状态持续时间（分钟）"""
        return (self._now() - self.state_start_time).total_seconds() / 60
    
    def get_state_history(self) -> List[Dict]:
        """获取状态历史"""
        return self.state_history.copy()
    
    def get_symptom_intensity(self, symptom: str, time_of_day: Optional[str] = None) -> float:
        """
        获取特定症状的强度
        
        Args:
            symptom: 症状名称
            time_of_day: 时间段 (morning/afternoon/evening/night)
            
        Returns:
            症状强度 (0-1)
        """
        base_intensity = self.get_state_characteristics().get(symptom, 0.5)
        
        # 根据时间调整（抑郁症状的昼夜节律）
        if time_of_day and symptom in ["mood_score", "energy_level"]:
            circadian_factors = {
                "morning": 0.7,    # 晨重
                "afternoon": 1.0,
                "evening": 1.1,
                "night": 0.9,
            }
            base_intensity *= circadian_factors.get(time_of_day, 1.0)
        
        return min(1.0, max(0.0, base_intensity))

    def to_dict(self) -> Dict[str, Any]:
        """导出可序列化状态。"""
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
            triggers = row.get("triggers", [])
            if not isinstance(triggers, list):
                triggers = [str(triggers)]
            row["triggers"] = [str(t) for t in triggers]
            history_payload.append(row)

        return {
            "current_state": self.current_state,
            "transition_sensitivity": float(self.transition_sensitivity),
            "minimum_state_duration": int(self.minimum_state_duration),
            "state_start_time": self.state_start_time.isoformat(),
            "state_history": history_payload,
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
        """从序列化状态恢复实例。"""
        payload = payload or {}
        machine = cls(
            initial_state=str(payload.get("current_state", DepressionState.SEVERE_EPISODE)),
            transition_sensitivity=float(payload.get("transition_sensitivity", 0.7) or 0.7),
            minimum_state_duration=int(payload.get("minimum_state_duration", 30) or 30),
            now_provider=now_provider,
        )
        current_state = str(payload.get("current_state", machine.current_state))
        if current_state in machine.STATE_CHARACTERISTICS:
            machine.current_state = current_state

        machine.state_start_time = _coerce_datetime(payload.get("state_start_time"))

        history_raw = payload.get("state_history", [])
        machine.state_history = []
        if isinstance(history_raw, list):
            for item in history_raw:
                if not isinstance(item, dict):
                    continue
                row = dict(item)
                row["timestamp"] = _coerce_datetime(row.get("timestamp"))
                triggers = row.get("triggers", [])
                if not isinstance(triggers, list):
                    triggers = [str(triggers)]
                row["triggers"] = [str(t) for t in triggers]
                try:
                    row["duration_minutes"] = float(row.get("duration_minutes", 0.0) or 0.0)
                except Exception:
                    row["duration_minutes"] = 0.0
                row["from_state"] = str(row.get("from_state", ""))
                row["to_state"] = str(row.get("to_state", ""))
                machine.state_history.append(row)

        acc = payload.get("accumulated_triggers", {})
        machine.accumulated_triggers = {}
        if isinstance(acc, dict):
            for key, value in acc.items():
                try:
                    machine.accumulated_triggers[str(key)] = float(value)
                except Exception:
                    continue
        return machine


def _coerce_datetime(value: Any) -> datetime:
    """把任意值尽量转换为 datetime，失败时返回当前时间。"""
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
