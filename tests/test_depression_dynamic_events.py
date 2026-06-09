import json
from pathlib import Path
from string import Template

from modules import utils
from modules.depression import DepressionSimulationEngine

try:
    import pytest
except ModuleNotFoundError:  # pragma: no cover - local lightweight runner fallback
    pytest = None

try:
    from modules.agent import Agent
except ModuleNotFoundError as exc:  # pragma: no cover - depends on optional runtime deps
    Agent = None
    AGENT_IMPORT_ERROR = exc


def _engine_config():
    return {
        "enabled": True,
        "agent_name": "测试角色",
        "complaint_graph": {
            "initial_stage_id": "stage_a",
            "planner": {
                "llm_enabled": False,
                "min_match_confidence": 0.6,
                "window_size": 2,
            },
            "stages": [
                {
                    "id": "stage_a",
                    "label": "阶段 A",
                    "summary": "围绕失业的自我否定。",
                    "narrative_focus": ["失业", "失败"],
                    "advance_signals": ["具体描述"],
                    "next_candidates": ["stage_b"],
                },
                {
                    "id": "stage_b",
                    "label": "阶段 B",
                    "summary": "开始承认疲惫。",
                    "narrative_focus": ["疲惫"],
                    "is_terminal_stage": True,
                },
            ],
        },
        "emotion": {"llm_enabled": False},
        "memory": {"enabled": False},
    }


def _jump_signal():
    return {
        "matched": True,
        "match_confidence": 1.0,
        "match_reason": "测试跳转",
        "action": "jump",
        "next_graph": ["stage_a", "stage_b"],
    }


def _single_stage_empty_candidates_config():
    return {
        "enabled": True,
        "agent_name": "测试角色",
        "complaint_graph": {
            "initial_stage_id": "stage_a",
            "planner": {
                "llm_enabled": False,
                "min_match_confidence": 0.6,
                "window_size": 2,
            },
            "stages": [
                {
                    "id": "stage_a",
                    "label": "阶段 A",
                    "summary": "围绕失业、失败和自我否定。",
                    "narrative_focus": ["失业", "失败", "自我否定"],
                    "advance_signals": ["具体描述"],
                    "next_candidates": [],
                },
            ],
        },
        "emotion": {"llm_enabled": False},
        "memory": {"enabled": False},
    }


def test_generate_chat_template_keeps_persona_description_single_source():
    template = Template(Path("data/prompts/generate_chat.txt").read_text(encoding="utf-8"))
    common = {
        "agent": "卡布达",
        "memory": "记忆",
        "address": "家，餐桌",
        "current_time": "09:00",
        "previous_context": "",
        "current_context": "当前场景",
        "another": "朋友",
        "conversation": "[对话尚未开始]",
    }

    normal_prompt = template.safe_substitute(
        {
            **common,
            "base_desc_block": "以下是对 卡布达 的简要描述：\n普通基础描述\n",
            "depression_chat_block": "",
        }
    )
    assert "以下是对 卡布达 的简要描述：" in normal_prompt
    assert "普通基础描述" in normal_prompt

    dynamic_prompt = template.safe_substitute(
        {
            **common,
            "base_desc_block": "",
            "depression_chat_block": "=== 基础人格层 ===\n动态基础描述\n",
        }
    )
    assert "以下是对 卡布达 的简要描述：" not in dynamic_prompt
    assert dynamic_prompt.count("动态基础描述") == 1
    assert dynamic_prompt.count("=== 基础人格层 ===") == 1


def test_chat_event_keeps_existing_jump_behavior():
    engine = DepressionSimulationEngine(_engine_config())

    engine.commit_event(
        source="chat",
        location="家",
        time_of_day="morning",
        interaction_type="闲聊",
        content="我因为失业觉得自己彻底失败。",
        other_agent="朋友",
        relationship="朋友",
        metadata={"evidence_ids": ["chat-1"]},
        llm_transition_signal=_jump_signal(),
    )

    state = engine.get_current_state_info()
    assert state["current_stage"]["id"] == "stage_b"
    assert state["graph"]["last_evaluation"]["action"] == "jump"
    assert state["graph"]["last_evaluation"]["source"] == "chat"
    assert state["graph"]["dialogue_history"][-1]["source"] == "chat"
    assert state["graph"]["dialogue_history"][-1]["evidence_ids"] == ["chat-1"]
    assert state["graph"]["stage_history"][-1]["source"] == "chat"


def test_depression_dynamic_uses_simulated_clock_for_runtime_timestamps():
    utils.set_timer(start="20260609-16:10")
    engine = DepressionSimulationEngine(_engine_config())

    assert engine.to_dict()["last_update_time"].startswith("2026-06-09T16:10")

    engine.commit_event(
        source="chat",
        location="家",
        time_of_day="afternoon",
        interaction_type="闲聊",
        content="我因为失业觉得自己彻底失败。",
        other_agent="朋友",
        relationship="朋友",
        metadata={"evidence_ids": ["chat-1"]},
        llm_transition_signal=_jump_signal(),
    )

    saved = engine.to_dict()
    graph = saved["complaint_graph_manager"]
    assert saved["last_update_time"].startswith("2026-06-09T16:10")
    assert graph["stage_start_time"].startswith("2026-06-09T16:10")
    assert graph["dialogue_history"][-1]["timestamp"].startswith("2026-06-09T16:10")
    assert graph["stage_history"][-1]["timestamp"].startswith("2026-06-09T16:10")

    saved["last_update_time"] = "2026-06-09T15:10:53.809975"
    restored = DepressionSimulationEngine.from_dict(saved)
    assert restored.to_dict()["last_update_time"].startswith("2026-06-09T16:10")


def test_reflection_event_downgrades_jump_to_hold():
    engine = DepressionSimulationEngine(_engine_config())

    engine.commit_event(
        source="reflection",
        location="家",
        time_of_day="night",
        interaction_type="内在反思",
        content="反思结论：我把失业和整个人的失败绑得太紧。",
        metadata={"evidence_ids": ["thought-1", "thought-1", "event-2"]},
        llm_transition_signal=_jump_signal(),
    )

    state = engine.get_current_state_info()
    assert state["current_stage"]["id"] == "stage_a"
    assert state["interaction_count"] == 1
    assert state["graph"]["last_evaluation"]["action"] == "hold"
    assert state["graph"]["last_evaluation"]["source"] == "reflection"
    assert state["graph"]["dialogue_history"][-1]["source"] == "reflection"
    assert state["graph"]["dialogue_history"][-1]["evidence_ids"] == ["thought-1", "event-2"]
    assert state["graph"]["stage_history"][-1]["source"] == "reflection"


def test_empty_candidate_stage_without_llm_stage_holds_current_stage():
    engine = DepressionSimulationEngine(_single_stage_empty_candidates_config())
    initial_state = engine.get_current_state_info()
    assert initial_state["graph"]["planned_graph"] == ["stage_a"]

    engine.commit_event(
        source="chat",
        location="家",
        time_of_day="morning",
        interaction_type="闲聊",
        content="我最近因为失业觉得自己很失败。其实那天以后我想具体描述一下，我每天都卡住。",
        other_agent="朋友",
        relationship="朋友",
    )

    state = engine.get_current_state_info()
    assert state["current_stage"]["id"] == "stage_a"
    assert state["graph"]["last_evaluation"]["action"] == "hold"
    assert state["graph"]["stage_index"] == 0

    saved = engine.to_dict()
    assert len(engine.raw_config["complaint_graph"]["stages"]) == 1
    assert saved["complaint_graph_manager"]["runtime_stages"] == []
    assert saved["complaint_graph_manager"]["planned_graph"] == ["stage_a"]


def test_llm_id_only_next_graph_does_not_materialize_unknown_stage():
    engine = DepressionSimulationEngine(_single_stage_empty_candidates_config())

    engine.commit_event(
        source="chat",
        location="家",
        time_of_day="morning",
        interaction_type="闲聊",
        content="我开始能说出失业后具体卡住的地方。",
        other_agent="朋友",
        relationship="朋友",
        llm_transition_signal={
            "matched": True,
            "match_confidence": 1.0,
            "match_reason": "只返回后续 id 的测试信号",
            "action": "advance",
            "next_graph": ["stage_a", "stage_b"],
        },
    )

    state = engine.get_current_state_info()
    assert state["current_stage"]["id"] == "stage_a"
    assert state["graph"]["last_evaluation"]["action"] == "hold"
    assert state["graph"]["planned_graph"] == ["stage_a"]


def test_llm_full_stage_extends_empty_candidate_graph():
    engine = DepressionSimulationEngine(_single_stage_empty_candidates_config())

    engine.commit_event(
        source="chat",
        location="家",
        time_of_day="morning",
        interaction_type="闲聊",
        content="我开始能说出失业后具体卡住的地方。",
        other_agent="朋友",
        relationship="朋友",
        llm_transition_signal={
            "matched": True,
            "match_confidence": 1.0,
            "match_reason": "完整节点测试信号",
            "action": "advance",
            "next_graph": [
                "stage_a",
                {
                    "id": "stage_b",
                    "label": "开始描述失业后的卡住",
                    "summary": "从抽象自我否定转向描述失业后日常被卡住的具体影响。",
                    "narrative_focus": ["失业后日常", "卡住", "具体影响"],
                },
            ],
        },
    )

    state = engine.get_current_state_info()
    assert state["current_stage"]["id"] == "stage_b"
    assert state["current_stage"]["source"] == "llm"
    assert state["graph"]["planned_graph"] == ["stage_a", "stage_b"]
    assert engine.graph_manager.stage_catalog["stage_a"]["next_candidates"] == ["stage_b"]


def test_initialize_graph_window_uses_llm_full_stage_without_committing_turn():
    config = _single_stage_empty_candidates_config()
    config["complaint_graph"]["planner"]["llm_enabled"] = True
    engine = DepressionSimulationEngine(config)

    def completion(_prompt):
        return json.dumps(
            {
                "matched_current_stage": True,
                "match_confidence": 0.9,
                "match_reason": "补足起始主诉图窗口",
                "action": "replan",
                "next_graph": [
                    "stage_a",
                    {
                        "id": "stage_b",
                        "label": "开始承认失业后的疲惫",
                        "summary": "从单纯自责延伸到承认失业后持续疲惫和日常停滞。",
                        "narrative_focus": ["疲惫", "日常停滞", "自责后的耗竭"],
                    },
                ],
            },
            ensure_ascii=False,
        )

    state = engine.initialize_graph_window(
        location="家",
        time_of_day="morning",
        roadmap_completion_func=completion,
    )

    assert state["graph"]["planned_graph"] == ["stage_a", "stage_b"]
    assert state["interaction_count"] == 0
    assert state["graph"]["stage_history"] == []
    assert state["graph"]["stage_index"] == 0
    assert state["current_stage"]["id"] == "stage_a"
    assert engine.graph_manager.stage_catalog["stage_b"]["source"] == "llm"
    assert engine.graph_manager.stage_catalog["stage_a"]["next_candidates"] == ["stage_b"]


def test_llm_graph_window_links_generated_stages_as_graph_edges():
    config = _single_stage_empty_candidates_config()
    config["complaint_graph"]["planner"]["llm_enabled"] = True
    engine = DepressionSimulationEngine(config)

    def completion(_prompt):
        return json.dumps(
            {
                "matched_current_stage": True,
                "match_confidence": 0.9,
                "match_reason": "补足多节点窗口",
                "action": "replan",
                "next_graph": [
                    "stage_a",
                    {
                        "id": "stage_b",
                        "label": "开始描述失业后的卡住",
                        "summary": "开始把失业后的日常停滞说具体。",
                    },
                    {
                        "id": "stage_c",
                        "label": "意识到自责会加重退缩",
                        "summary": "开始看见自责和退缩之间的关系。",
                    },
                ],
            },
            ensure_ascii=False,
        )

    engine.initialize_graph_window(
        location="家",
        time_of_day="morning",
        roadmap_completion_func=completion,
    )

    assert engine.graph_manager.stage_catalog["stage_a"]["next_candidates"] == ["stage_b"]
    assert engine.graph_manager.stage_catalog["stage_b"]["next_candidates"] == ["stage_c"]


def test_unknown_llm_next_candidates_are_pruned_from_runtime_graph():
    config = _single_stage_empty_candidates_config()
    config["complaint_graph"]["planner"]["llm_enabled"] = True
    engine = DepressionSimulationEngine(config)

    def completion(_prompt):
        return json.dumps(
            {
                "matched_current_stage": True,
                "match_confidence": 0.9,
                "match_reason": "补足窗口但只给了一个完整节点",
                "action": "replan",
                "next_graph": [
                    "stage_a",
                    {
                        "id": "stage_b",
                        "label": "阶段 B",
                        "summary": "运行态生成的阶段。",
                        "next_candidates": ["stage_c"],
                    },
                ],
            },
            ensure_ascii=False,
        )

    engine.initialize_graph_window(
        location="家",
        time_of_day="morning",
        roadmap_completion_func=completion,
    )

    assert engine.graph_manager.stage_catalog["stage_a"]["next_candidates"] == ["stage_b"]
    assert engine.graph_manager.stage_catalog["stage_b"]["next_candidates"] == []
    assert "stage_c" not in engine.graph_manager.stage_catalog


def test_initialize_graph_window_retries_until_window_target_is_reached():
    config = _single_stage_empty_candidates_config()
    config["complaint_graph"]["planner"]["llm_enabled"] = True
    engine = DepressionSimulationEngine(config)
    responses = [
        {
            "matched_current_stage": True,
            "match_confidence": 0.9,
            "match_reason": "第一次只补一个节点",
            "action": "replan",
            "next_graph": [
                "stage_a",
                {
                    "id": "stage_b",
                    "label": "阶段 B",
                    "summary": "运行态生成的第一段延续。",
                },
            ],
        },
        {
            "matched_current_stage": True,
            "match_confidence": 0.9,
            "match_reason": "第二次补足窗口",
            "action": "replan",
            "next_graph": [
                "stage_a",
                "stage_b",
                {
                    "id": "stage_c",
                    "label": "阶段 C",
                    "summary": "运行态生成的第二段延续。",
                },
            ],
        },
    ]
    last_response = responses[-1]

    def completion(_prompt):
        payload = responses.pop(0) if responses else last_response
        return json.dumps(payload, ensure_ascii=False)

    state = engine.initialize_graph_window(
        location="家",
        time_of_day="morning",
        roadmap_completion_func=completion,
    )

    assert state["graph"]["planned_graph"] == ["stage_a", "stage_b", "stage_c"]
    assert engine.graph_manager.stage_catalog["stage_a"]["next_candidates"] == ["stage_b"]
    assert engine.graph_manager.stage_catalog["stage_b"]["next_candidates"] == ["stage_c"]


def test_initialize_graph_window_maintains_window_size_future_nodes():
    config = _single_stage_empty_candidates_config()
    config["complaint_graph"]["planner"]["llm_enabled"] = True
    config["complaint_graph"]["planner"]["window_size"] = 3
    engine = DepressionSimulationEngine(config)

    def completion(_prompt):
        return json.dumps(
            {
                "matched_current_stage": True,
                "match_confidence": 0.9,
                "match_reason": "补足三个未来节点",
                "action": "replan",
                "next_graph": [
                    "stage_a",
                    {"id": "stage_b", "label": "阶段 B", "summary": "第一段自然延续。"},
                    {"id": "stage_c", "label": "阶段 C", "summary": "第二段自然延续。"},
                    {"id": "stage_d", "label": "阶段 D", "summary": "第三段自然延续。"},
                ],
            },
            ensure_ascii=False,
        )

    state = engine.initialize_graph_window(
        location="家",
        time_of_day="morning",
        roadmap_completion_func=completion,
    )

    assert [stage["id"] for stage in state["graph"]["current_graph_window"]] == [
        "stage_a",
        "stage_b",
        "stage_c",
        "stage_d",
    ]
    assert engine.graph_manager.stage_catalog["stage_c"]["next_candidates"] == ["stage_d"]


def test_preview_interaction_prompt_does_not_materialize_llm_graph_updates():
    config = _single_stage_empty_candidates_config()
    config["complaint_graph"]["planner"]["llm_enabled"] = True
    engine = DepressionSimulationEngine(config)

    def completion(_prompt):
        return json.dumps(
            {
                "matched_current_stage": True,
                "match_confidence": 1.0,
                "match_reason": "预览生成完整节点",
                "action": "advance",
                "next_graph": [
                    "stage_a",
                    {
                        "id": "stage_b",
                        "label": "预览阶段 B",
                        "summary": "只应出现在 preview clone 中。",
                    },
                ],
            },
            ensure_ascii=False,
        )

    engine.preview_interaction_prompt(
        location="家",
        time_of_day="morning",
        other_agent="朋友",
        relationship="朋友",
        interaction_type="闲聊",
        conversation_content="我开始能说出失业后具体卡住的地方。",
        roadmap_completion_func=completion,
    )

    assert list(engine.graph_manager.stage_catalog.keys()) == ["stage_a"]
    assert engine.graph_manager.stage_catalog["stage_a"]["next_candidates"] == []
    assert engine.get_current_state_info()["current_stage"]["id"] == "stage_a"


def test_chat_replan_with_advance_evidence_moves_to_next_stage():
    engine = DepressionSimulationEngine(_single_stage_empty_candidates_config())

    engine.commit_event(
        source="chat",
        location="家",
        time_of_day="morning",
        interaction_type="闲聊",
        content="我最近因为失业觉得自己失败，但我开始描述失业后具体卡住的地方，也承认自己真的很累。",
        other_agent="朋友",
        relationship="朋友",
        llm_transition_signal={
            "matched": True,
            "match_confidence": 0.95,
            "match_reason": "补充后续路线图",
            "action": "replan",
            "next_graph": [
                "stage_a",
                {
                    "id": "stage_b",
                    "label": "开始描述失业后的卡住",
                    "summary": "从抽象自责转向描述失业后日常卡住和疲惫。",
                },
            ],
        },
    )

    state = engine.get_current_state_info()
    assert state["current_stage"]["id"] == "stage_b"
    assert state["graph"]["stage_index"] == 1
    assert state["graph"]["last_evaluation"]["action"] == "advance"


def test_commit_advance_then_expands_new_current_stage_future_window():
    config = _single_stage_empty_candidates_config()
    config["complaint_graph"]["planner"]["llm_enabled"] = True
    config["complaint_graph"]["planner"]["window_size"] = 3
    engine = DepressionSimulationEngine(config)
    responses = [
        {
            "matched_current_stage": True,
            "match_confidence": 1.0,
            "match_reason": "先推进到阶段 B",
            "action": "advance",
            "next_graph": [
                "stage_a",
                {"id": "stage_b", "label": "阶段 B", "summary": "推进后的当前节点。"},
            ],
        },
        {
            "matched_current_stage": True,
            "match_confidence": 0.9,
            "match_reason": "给新当前节点补足未来窗口",
            "action": "replan",
            "next_graph": [
                "stage_b",
                {"id": "stage_c", "label": "阶段 C", "summary": "B 后的第一段。"},
                {"id": "stage_d", "label": "阶段 D", "summary": "B 后的第二段。"},
                {"id": "stage_e", "label": "阶段 E", "summary": "B 后的第三段。"},
            ],
        },
    ]

    def completion(_prompt):
        return json.dumps(responses.pop(0), ensure_ascii=False)

    state = engine.commit_event(
        source="chat",
        location="家",
        time_of_day="morning",
        interaction_type="闲聊",
        content="我开始能说出失业后具体卡住的地方。",
        other_agent="朋友",
        relationship="朋友",
        roadmap_completion_func=completion,
    )

    assert state["current_stage"]["id"] == "stage_b"
    assert [stage["id"] for stage in state["graph"]["current_graph_window"]] == [
        "stage_b",
        "stage_c",
        "stage_d",
        "stage_e",
    ]
    assert engine.graph_manager.stage_catalog["stage_b"]["next_candidates"] == ["stage_c"]
    assert state["graph"]["last_evaluation"]["action"] == "advance"


def test_restore_links_existing_planned_graph_edges():
    engine = DepressionSimulationEngine(_single_stage_empty_candidates_config())
    payload = engine.to_dict()
    payload["complaint_graph_manager"]["runtime_stages"].append(
        {
            "id": "stage_b",
            "label": "阶段 B",
            "summary": "运行态生成的阶段。",
            "source": "llm",
            "next_candidates": [],
        }
    )
    payload["complaint_graph_manager"]["planned_graph"] = ["stage_a", "stage_b"]

    restored = DepressionSimulationEngine.from_dict(payload)

    assert restored.graph_manager.stage_catalog["stage_a"]["next_candidates"] == ["stage_b"]


class _FakeDepressionDynamic:
    def __init__(self):
        self.base_prompt = ""
        self.calls = []
        self.init_calls = []

    def set_base_prompt(self, prompt):
        self.base_prompt = prompt

    def commit_event(self, **kwargs):
        self.calls.append(kwargs)
        return {"enabled": True}

    def initialize_graph_window(self, **kwargs):
        self.init_calls.append(kwargs)
        return {"enabled": True}


class _FakeLlm:
    def is_available(self):
        return True


def _agent_with_fake_dynamic():
    if Agent is None:
        if pytest is not None:
            pytest.skip("Agent import requires optional runtime dependencies")
        raise RuntimeError("Agent import requires optional runtime dependencies: {}".format(AGENT_IMPORT_ERROR))
    agent = Agent.__new__(Agent)
    agent.name = "卡布达"
    agent.logger = None
    agent.depression_dynamic = _FakeDepressionDynamic()
    agent._llm = _FakeLlm()
    agent._build_depression_base_prompt = lambda: "base prompt"
    agent._dynamic_location = lambda: "家"
    agent._dynamic_time_of_day = lambda: "night"
    return agent


def test_agent_reset_initializes_depression_graph_window():
    agent = _agent_with_fake_dynamic()
    agent.think_config = {"llm": {}}

    agent.reset()

    assert len(agent.depression_dynamic.init_calls) == 1
    call = agent.depression_dynamic.init_calls[0]
    assert call["location"] == "家"
    assert call["time_of_day"] == "night"
    assert callable(call["roadmap_completion_func"])
    assert agent.depression_dynamic.calls == []


def test_agent_depression_event_only_allows_chat_and_reflection():
    agent = _agent_with_fake_dynamic()

    agent._commit_depression_event(
        source="observation",
        location="家",
        time_of_day="morning",
        interaction_type="环境观察",
        content="看到一个事件",
    )
    agent._commit_depression_event(
        source="action",
        location="家",
        time_of_day="morning",
        interaction_type="行动",
        content="做了一件事",
    )

    assert agent.depression_dynamic.calls == []

    agent._commit_depression_event(
        source="chat",
        location="家",
        time_of_day="morning",
        interaction_type="闲聊",
        content="我最近觉得自己很失败。",
        metadata={"evidence_ids": ["chat-1"]},
    )

    assert len(agent.depression_dynamic.calls) == 1
    assert agent.depression_dynamic.calls[0]["source"] == "chat"
    assert agent.depression_dynamic.calls[0]["metadata"]["evidence_ids"] == ["chat-1"]


def test_agent_reflection_payload_is_committed_once():
    agent = _agent_with_fake_dynamic()

    agent._commit_depression_reflection(
        focus=["失业", "自我否定"],
        entries=[
            {"thought": "我把失业理解成自己没用。", "evidence": ["event-1"], "node_id": "thought-1"},
            {"thought": "我把失业理解成自己没用。", "evidence": ["event-1", "chat-2"], "node_id": "thought-2"},
        ],
    )

    assert len(agent.depression_dynamic.calls) == 1
    call = agent.depression_dynamic.calls[0]
    assert call["source"] == "reflection"
    assert call["interaction_type"] == "内在反思"
    assert "反思焦点：失业；自我否定" in call["content"]
    assert "反思结论：我把失业理解成自己没用。" in call["content"]
    assert call["metadata"]["evidence_ids"] == ["event-1", "chat-2"]
    assert call["metadata"]["thought_node_ids"] == ["thought-1", "thought-2"]
