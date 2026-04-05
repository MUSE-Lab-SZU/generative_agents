"""
动态抑郁症状模拟系统 (Multi-layered Dynamic Symptom Simulation, MDSS)
"""

from .engine import DepressionSimulationEngine
from .state_machine import SymptomStateMachine, DepressionState
from .context_analyzer import ContextAnalyzer
from .bias_injector import CognitiveBiasInjector
from .memory_system import TraumaMemorySystem
from .prompt_builder import DynamicPromptBuilder

__all__ = [
    'DepressionSimulationEngine',
    'SymptomStateMachine',
    'DepressionState',
    'ContextAnalyzer',
    'CognitiveBiasInjector',
    'TraumaMemorySystem',
    'DynamicPromptBuilder',
]
