import json
from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest

from modules.depression.memory_system import DynamicMemorySystem
from modules.depression.engine import DepressionSimulationEngine

NOW = datetime(2026, 9, 15, 9)


class Embeddings:
    """Controlled semantic vectors: paraphrases share meaning, not characters."""
    def get_text_embedding(self, text):
        return [1, 0] if text != "unrelated" else [0, 1]

    def get_query_embedding(self, text):
        return [0, 1] if text == "天气如何" else [1, 0]


def config():
    return {"enabled": True, "initial_trust": 0.3, "min_similarity": 0.6,
            "max_items": 4, "public_persona": "我是小镇居民。",
            "items": [
                {"memory_id": "surface", "content": "最近失业了。", "disclosure_threshold": 0.2},
                {"memory_id": "secret", "content": "家人曾在童年否定我。", "disclosure_threshold": 0.7,
                 "generates_discomfort": True},
            ]}


def memory(tmp_path, cfg=None):
    mem = DynamicMemorySystem(cfg or config(), "patient", lambda: NOW)
    mem.bind(tmp_path / "patient", Embeddings(), "test-model-v1")
    return mem


def engine(tmp_path):
    cfg = {"enabled": True, "agent_name": "patient", "memory": config(),
           "emotion": {"llm_enabled": False}, "complaint_graph": {
               "initial_stage_id": "a", "planner": {"llm_enabled": False},
               "stages": [{"id": "a", "label": "家人曾在童年否定我。",
                           "summary": "家人曾在童年否定我。", "core_belief": "家人曾在童年否定我。",
                           "speaking_style": {"repair_pattern": "家人曾在童年否定我。"}}]}}
    obj = DepressionSimulationEngine(cfg, clock_provider=lambda: NOW)
    obj.memory_system.bind(tmp_path, Embeddings(), "test-v1")
    obj.set_base_prompt("家人曾在童年否定我。")
    return obj


def test_vector_paraphrase_and_low_trust_block(tmp_path):
    mem = memory(tmp_path)
    result = mem.prepare("饭碗丢了以后怎么样", "doctor", "t1")
    assert [r["memory_id"] for r in result["allowed_memories"]] == ["surface"]
    assert result["blocked_ids"] == ["secret"]
    assert "童年" not in json.dumps(result["blocked_signal"], ensure_ascii=False)
    assert mem.prepare("天气如何", "doctor", "t2")["allowed_memories"] == []


def test_allowed_budget_not_crowded_out_by_blocked(tmp_path):
    cfg = config()
    cfg["max_items"] = 1
    cfg["items"][1]["importance"] = 1
    mem = memory(tmp_path, cfg)
    result = mem.prepare("工作", "doctor", "t1")
    assert result["allowed_memories"][0]["memory_id"] == "surface"
    assert result["blocked_ids"] == ["secret"]


def test_partner_and_owner_isolation(tmp_path):
    mem = memory(tmp_path)
    mem.trust["doctor"] = 0.8
    assert len(mem.prepare("工作", "doctor", "t1")["allowed_memories"]) == 2
    assert len(mem.prepare("工作", "neighbor", "t2")["allowed_memories"]) == 1
    with pytest.raises(ValueError):
        mem.add_memory({"owner_id": "someone_else", "content": "secret"})
    other = DynamicMemorySystem(config(), "other", lambda: NOW)
    with pytest.raises(ValueError):
        other.load_state(mem.to_dict())


def test_repeated_preview_does_not_change_persistent_state(tmp_path):
    obj = engine(tmp_path)
    before = obj.to_dict()
    for _ in range(2):
        prompt = obj.preview_interaction_prompt("家", "morning", "doctor", conversation_content="工作", turn_id="t1")
        assert "家人曾在童年否定我" not in prompt
        assert "最近失业了" in prompt
    assert obj.to_dict() == before
    assert obj.context_builder.context_history == []
    assert "家人曾在童年否定我" not in obj.get_simple_prompt()


def test_commit_trust_next_turn_and_idempotency(tmp_path):
    mem = memory(tmp_path)
    decision = mem.prepare("工作", "doctor", "t1")
    judge = lambda _: json.dumps({"direction": "increased_slightly", "evidence": "不用急",
                                  "disclosed_ids": ["surface", "secret", "invented"]})
    assert mem.commit_turn(decision, "最近失业了。", "不用急", completion_func=judge)
    assert mem.trust["doctor"] == pytest.approx(0.34)
    assert decision["trust"] == 0.3
    assert mem.records["surface"]["disclosed_to"] == {"doctor": NOW.isoformat()}
    assert not mem.records["secret"]["disclosed_to"]
    before = mem.to_dict()
    assert not mem.commit_turn(decision, "最近失业了。", "不用急", completion_func=judge)
    assert mem.to_dict() == before


def test_judge_failure_and_invalid_evidence_do_not_inflate_trust(tmp_path):
    mem = memory(tmp_path)
    for i, judge in enumerate([lambda _: "invalid", lambda _: json.dumps({
            "direction": "increased_significantly", "evidence": "not actually said"})]):
        mem.commit_turn(mem.prepare("工作", "doctor", str(i)), "嗯", "你好", completion_func=judge)
    assert mem.trust["doctor"] == 0.3


def test_reflection_does_not_change_trust_and_inherits_threshold(tmp_path):
    mem = memory(tmp_path)
    mem.trust["doctor"] = 0.9
    decision = mem.prepare("工作", "doctor", "reflection1")
    mem.commit_turn(decision, "我对此的理解", source="reflection",
                    completion_func=lambda _: pytest.fail("reflection must not evaluate trust"))
    assert mem.trust["doctor"] == 0.9
    new = next(r for r in mem.records.values() if r["source_type"] == "reflection")
    assert new["disclosure_threshold"] >= 0.7
    assert new["kind"] == "subjective_reflection"
    assert not new["disclosed_to"]


def test_same_listener_keeps_actual_words_after_trust_drop(tmp_path):
    mem = memory(tmp_path)
    mem.commit_turn(mem.prepare("工作", "doctor", "t1"), "我曾被否定。", "你想聊什么")
    mem.trust["doctor"] = 0
    own = mem.prepare("工作", "doctor", "t2")
    assert any(r["content"] == "我曾被否定。" and r["already_shared_only"] for r in own["allowed_memories"])
    assert not any(r["content"] == "我曾被否定。" for r in mem.prepare("工作", "neighbor", "t3")["allowed_memories"])


def test_checkpoint_older_than_disk_wins_and_vectors_rebuild(tmp_path):
    mem = memory(tmp_path)
    old = mem.to_dict()
    mem.trust["doctor"] = 1
    mem.prepare("工作", "doctor", "t1")
    mem.save()
    restored = DynamicMemorySystem(config(), "patient", lambda: NOW)
    restored.load_state(old)
    restored.bind(tmp_path / "patient", Embeddings(), "different-model")
    assert restored.trust_for("doctor") == 0.3
    assert len(restored.prepare("工作", "doctor", "t2")["allowed_memories"]) == 1
    assert len(restored.vectors) > len(mem.vectors)
    disk = DynamicMemorySystem(config(), "patient", lambda: NOW)
    disk.bind(tmp_path / "patient", Embeddings(), "test-model-v1")
    assert disk.trust_for("doctor") == 1


def test_expiration_and_embedding_failure(tmp_path):
    mem = memory(tmp_path)
    mem.records["surface"]["expires_at"] = (NOW - timedelta(days=1)).isoformat()
    assert not mem.prepare("工作", "doctor", "t1")["allowed_memories"]
    mem.embedder = None
    result = mem.prepare("工作", "doctor", "t2")
    assert result["retrieval_error"] and not result["allowed_memories"]
    assert not result["blocked_signal"]["present"]


def test_disabled_has_no_storage_or_embedding_calls(tmp_path):
    mem = DynamicMemorySystem({"enabled": False, "items": [{"content": "private"}]}, "normal")
    mem.bind(tmp_path / "normal", None, "")
    assert not mem.prepare("hello", "doctor", "t1")["allowed_memories"]
    mem.save()
    assert not (tmp_path / "normal").exists()


def test_engine_commit_idempotent_and_restore(tmp_path):
    obj = engine(tmp_path)
    obj.preview_interaction_prompt("家", "morning", "doctor", conversation_content="工作", turn_id="t1")
    args = dict(source="chat", location="家", time_of_day="morning", interaction_type="闲聊",
                content="最近失业了。", counterpart_utterance="不用急", other_agent="doctor", metadata={"turn_id": "t1"})
    obj.commit_event(**args)
    state = obj.to_dict()
    obj.commit_event(**args)
    assert obj.to_dict() == state
    restored = DepressionSimulationEngine.from_dict(state, clock_provider=lambda: NOW)
    assert restored.memory_system.to_dict() == obj.memory_system.to_dict()
    restored.commit_event(**args)
    assert restored.interaction_count == 1


def test_chat_prompt_filters_old_private_memory(tmp_path):
    from modules.prompt.scratch import Scratch
    from modules import utils
    utils.set_timer(start="20260915-09:00")
    node = SimpleNamespace(describe="历史秘密不应该显示", create=NOW, node_id="old")
    associate = SimpleNamespace(retrieve_focus=lambda *a, **kw: [node], retrieve_chats=lambda *a, **kw: [node])
    event = SimpleNamespace(get_describe=lambda *a: "活动中的私密内容")
    agent = SimpleNamespace(name="patient", associate=associate, logger=None,
                            public_memory_nodes=lambda nodes: [], dynamic_memory_enabled=lambda: True,
                            get_tile=lambda: SimpleNamespace(get_address=lambda: ["town", "home"]),
                            get_event=lambda: event)
    other = SimpleNamespace(name="doctor", get_event=lambda: event)
    scratch = Scratch("patient", "旧私人背景", {})
    result = scratch.prompt_generate_chat(agent, other, "关系", [], depression_chat_block="公开信息")
    assert "历史秘密" not in result["prompt"]
    assert "活动中的私密内容" not in result["prompt"]
    assert "旧私人背景" not in result["prompt"]


def test_new_sensitive_concept_is_stored_separately(tmp_path):
    from modules.agent import Agent
    from modules.memory import Event
    agent = Agent.__new__(Agent)
    agent.name = "patient"
    agent.depression_dynamic = SimpleNamespace(enabled=True, memory_system=memory(tmp_path))
    agent.completion = lambda *a: 5
    agent.logger = SimpleNamespace(debug=lambda *a: None)
    saved = []
    def add_node(kind, event, *args, **kwargs):
        saved.append(event)
        return SimpleNamespace(node_id="new")
    agent.associate = SimpleNamespace(add_node=add_node)
    event = Event("patient", "此时", "私密想法", describe="深层私人解释")
    agent._add_concept("thought", event)
    assert saved[0].get_describe() == "patient 进行了一次反思"
    assert event.get_describe() == "patient 深层私人解释"
    assert "new" in agent.depression_dynamic.memory_system.public_node_ids
    assert any(r["content"] == "patient 深层私人解释" for r in agent.depression_dynamic.memory_system.records.values())


@pytest.mark.parametrize("value", [-1, 2, float("nan"), float("inf")])
def test_invalid_threshold_rejected(value):
    cfg = config()
    cfg["items"][0]["disclosure_threshold"] = value
    with pytest.raises(ValueError):
        DynamicMemorySystem(cfg, "patient")


def test_associate_filter_is_opt_in_and_prevents_empty_filter_retrieval():
    from modules.memory.associate import Associate
    associate = Associate.__new__(Associate)
    associate.retention = 8
    associate.memory = {"event": ["private"], "thought": [], "chat": []}
    associate._valid_memory_ids = lambda kind: associate.memory[kind]
    associate._index = SimpleNamespace(retrieve=lambda *a, **kw: pytest.fail("empty scope must not query all nodes"))
    assert associate._visible_ids(["private"]) == ["private"]
    associate.visibility_filter = lambda node_id: False
    assert associate.retrieve_focus(["question"]) == []
    assert associate.retrieve_events("question") == []
    assert associate._retrieve_chats_direct() == []


def test_context_budget_keeps_records_atomic(tmp_path):
    cfg = config()
    cfg["max_context_chars"] = 1
    mem = memory(tmp_path, cfg)
    assert not mem.prepare("工作", "doctor", "t1")["allowed_memories"]


def test_enabled_config_binds_existing_embedder_and_quarantines_legacy(tmp_path):
    from modules.agent import Agent
    from modules.prompt.scratch import Scratch
    from modules import utils
    utils.set_timer(start="20260915-09:00")
    obj = engine(tmp_path / "seed")
    config_path = tmp_path / "depression_config.json"
    config_path.write_text(json.dumps(obj.raw_config), encoding="utf-8")
    agent = Agent.__new__(Agent)
    agent.name, agent.logger = "patient", None
    agent.scratch = Scratch("patient", "private background", {})
    agent._build_depression_base_prompt = lambda: "private background"
    index = SimpleNamespace(embedding_model=Embeddings(), get_nodes=lambda: [SimpleNamespace(id_="old", text="legacy private")])
    agent.associate = SimpleNamespace(index=index)
    dynamic = agent._init_depression_dynamic({
        "depression_config_path": str(config_path), "storage_root": str(tmp_path / "run"),
        "associate": {"embedding": {"provider": "test", "model": "v1"}}})
    assert dynamic.memory_system.path == tmp_path / "run" / "depression_memory"
    assert dynamic.memory_system.records["legacy_old"]["disclosure_threshold"] == 1
    assert not agent.associate.visibility_filter("old")
    assert agent.scratch._base_desc() == "我是小镇居民。"


def test_normal_agent_does_not_bind_memory_or_change_persona(tmp_path):
    from modules.agent import Agent
    agent = Agent.__new__(Agent)
    agent.name, agent.logger = "normal", None
    assert agent._init_depression_dynamic({"agent_dir": str(tmp_path)}) is None
    assert not agent.dynamic_memory_enabled()
    original = [SimpleNamespace(node_id="old")]
    assert agent.public_memory_nodes(original) is original


def test_real_llama_index_embedding_adapter(tmp_path):
    from llama_index.core import VectorStoreIndex
    from llama_index.core.embeddings import MockEmbedding
    from modules.storage.index import LlamaIndex
    wrapper = LlamaIndex.__new__(LlamaIndex)
    embedder = MockEmbedding(embed_dim=8)
    wrapper._index = VectorStoreIndex([], embed_model=embedder)
    mem = DynamicMemorySystem(config(), "patient", lambda: NOW)
    mem.bind(tmp_path, wrapper.embedding_model, "llama-mock-8")
    decision = mem.prepare("semantic query", "doctor", "t1")
    assert not decision["retrieval_error"]
    assert len(decision["allowed_memories"]) == 1
    assert all(len(v) == 8 for v in mem.vectors.values())


def test_reused_turn_cannot_cross_partners(tmp_path):
    obj = engine(tmp_path)
    obj.preview_interaction_prompt("家", "morning", "doctor", conversation_content="工作", turn_id="t1")
    with pytest.raises(ValueError):
        obj.commit_event(source="chat", location="家", time_of_day="morning", interaction_type="闲聊",
                         content="你好", other_agent="neighbor", metadata={"turn_id": "t1"})
    assert obj.interaction_count == 0


def test_generated_but_rejected_reply_does_not_commit():
    from modules.agent import Agent
    agent = Agent.__new__(Agent)
    agent.name = "patient"
    agent.scratch = SimpleNamespace(prompt_generate_chat=lambda **kwargs: {"failsafe": "候选话语"})
    agent.logger = SimpleNamespace(debug=lambda *a: None)
    agent.llm_available = lambda: False
    agent._prepare_depression_generate_chat = lambda args, kwargs: (kwargs, {"turn_id": "t1"})
    committed = []
    agent._commit_depression_generate_chat = lambda context, output: committed.append(output)
    agent.completion("generate_chat", _defer_depression_commit=True)
    assert committed == []
    agent._finish_depression_chat(False)
    assert committed == []
    agent.completion("generate_chat", _defer_depression_commit=True)
    agent._finish_depression_chat(True)
    agent._finish_depression_chat(True)
    assert committed == ["候选话语"]


def write_seed_config(root, items=None, **overrides):
    payload = {"schema_version": 1, "owner_id": "patient", "items": config()["items"] if items is None else items}
    payload.update(overrides)
    source = root / "initial_dynamic_memories.json"
    source.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    cfg = {"enabled": True, "agent_name": "patient", "memory": {
        "enabled": True, "initial_memories_file": source.name},
        "emotion": {"llm_enabled": False}}
    path = root / "depression_config.json"
    path.write_text(json.dumps(cfg), encoding="utf-8")
    return path, source


def test_external_seeds_resolve_relative_to_agent_not_cwd(tmp_path, monkeypatch):
    agent_dir = tmp_path / "resident"
    agent_dir.mkdir()
    path, source = write_seed_config(agent_dir)
    monkeypatch.chdir(tmp_path)
    obj = DepressionSimulationEngine(str(path), clock_provider=lambda: NOW)
    assert set(obj.memory_system.records) == {"surface", "secret"}
    assert "items" not in obj.raw_config["memory"]
    assert obj.memory_system.records["secret"]["owner_id"] == "patient"
    before = source.read_bytes()
    obj.memory_system.bind(tmp_path / "runtime", Embeddings(), "test-v1")
    obj.memory_system.commit_turn(obj.memory_system.prepare("工作", "doctor", "t1"), "嗯")
    obj.memory_system.save()
    assert source.read_bytes() == before
    assert (tmp_path / "runtime" / "state.json").exists()


def test_seed_edits_apply_to_new_runs_but_checkpoint_keeps_snapshot(tmp_path):
    path, source = write_seed_config(tmp_path)
    obj = DepressionSimulationEngine(str(path), clock_provider=lambda: NOW)
    saved = obj.to_dict()
    payload = json.loads(source.read_text())
    payload["items"].append({"memory_id": "new", "content": "新增设定", "disclosure_threshold": 0.2})
    source.write_text(json.dumps(payload), encoding="utf-8")
    fresh = DepressionSimulationEngine(str(path), clock_provider=lambda: NOW)
    restored = DepressionSimulationEngine.from_dict(saved, clock_provider=lambda: NOW)
    assert "new" in fresh.memory_system.records
    assert "new" not in restored.memory_system.records
    assert restored.memory_system.to_dict() == saved["memory_system"]


@pytest.mark.parametrize("overrides", [
    {"owner_id": "other"}, {"schema_version": 99}, {"items": {}},
    {"items": [{"memory_id": "dup", "content": "a"}, {"memory_id": "dup", "content": "a"}]},
    {"items": [{"content": "missing id"}]},
])
def test_invalid_external_seed_file_rejected(tmp_path, overrides):
    path, _ = write_seed_config(tmp_path, **overrides)
    with pytest.raises(ValueError):
        DepressionSimulationEngine(str(path), clock_provider=lambda: NOW)


def test_missing_seed_file_and_ambiguous_sources_rejected(tmp_path):
    path, source = write_seed_config(tmp_path)
    source.unlink()
    with pytest.raises(FileNotFoundError):
        DepressionSimulationEngine(str(path))
    cfg = config()
    cfg["initial_memories_file"] = "initial_dynamic_memories.json"
    with pytest.raises(ValueError, match="not both"):
        DynamicMemorySystem(cfg, "patient", agent_dir=tmp_path)
    with pytest.raises(ValueError, match="requires agent_dir"):
        DynamicMemorySystem({"enabled": True, "initial_memories_file": "seeds.json"}, "patient")


def test_disabled_memory_does_not_read_seed_file(tmp_path):
    obj = DynamicMemorySystem({"enabled": False, "initial_memories_file": "missing.json"}, "patient", agent_dir=tmp_path)
    assert obj.records == {}


def test_kabuda_external_memories_are_complete_and_shared_by_variants():
    from pathlib import Path
    root = Path(__file__).resolve().parents[1] / "frontend/static/assets/village/agents/卡布达"
    obj = DepressionSimulationEngine(str(root / "depression_config.json"), clock_provider=lambda: NOW)
    records = list(obj.memory_system.records.values())
    assert len(records) == 20
    assert {r["provenance"] for r in records} == {"existing_persona", "authored_extension"}
    assert any(r["kind"] == "protective_factor" for r in records)
    assert any(r["disclosure_threshold"] >= 0.7 for r in records)
    assert any(r["disclosure_threshold"] <= 0.2 for r in records)
    for path in root.glob("depression_config*.json"):
        cfg = json.loads(path.read_text())
        assert "items" not in cfg["memory"]
        assert cfg["memory"]["initial_memories_file"] == "initial_dynamic_memories.json"
