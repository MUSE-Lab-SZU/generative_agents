from __future__ import annotations

import json
import os
import sys
import types
from pathlib import Path
from types import MethodType, SimpleNamespace

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(CURRENT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def _install_llama_index_stub() -> None:
    if "llama_index" in sys.modules:
        return

    llama_index_pkg = types.ModuleType("llama_index")
    core_mod = types.ModuleType("llama_index.core")
    retrievers_mod = types.ModuleType("llama_index.core.retrievers")
    vector_stores_mod = types.ModuleType("llama_index.core.vector_stores")
    retriever_impl_mod = types.ModuleType("llama_index.core.indices.vector_store.retrievers")
    schema_mod = types.ModuleType("llama_index.core.schema")
    node_parser_mod = types.ModuleType("llama_index.core.node_parser")

    class BaseRetriever:
        def __init__(self, *args, **kwargs) -> None:
            pass

    class MetadataFilters:
        def __init__(self, filters=None) -> None:
            self.filters = filters or []

    class ExactMatchFilter:
        def __init__(self, key=None, value=None) -> None:
            self.key = key
            self.value = value

    class VectorIndexRetriever:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def retrieve(self, query_bundle):
            return []

    class TextNode:
        def __init__(
            self,
            text="",
            id_=None,
            metadata=None,
            excluded_llm_metadata_keys=None,
            excluded_embed_metadata_keys=None,
        ) -> None:
            self.text = text
            self.id_ = id_
            self.metadata = metadata or {}

    class SentenceSplitter:
        def __init__(self, *args, **kwargs) -> None:
            pass

    retrievers_mod.BaseRetriever = BaseRetriever
    vector_stores_mod.MetadataFilters = MetadataFilters
    vector_stores_mod.ExactMatchFilter = ExactMatchFilter
    retriever_impl_mod.VectorIndexRetriever = VectorIndexRetriever
    schema_mod.TextNode = TextNode
    node_parser_mod.SentenceSplitter = SentenceSplitter
    core_mod.Settings = SimpleNamespace(
        embed_model=None,
        node_parser=None,
        num_output=None,
        context_window=None,
    )

    llama_index_pkg.core = core_mod
    sys.modules["llama_index"] = llama_index_pkg
    sys.modules["llama_index.core"] = core_mod
    sys.modules["llama_index.core.retrievers"] = retrievers_mod
    sys.modules["llama_index.core.vector_stores"] = vector_stores_mod
    sys.modules["llama_index.core.indices.vector_store.retrievers"] = retriever_impl_mod
    sys.modules["llama_index.core.schema"] = schema_mod
    sys.modules["llama_index.core.node_parser"] = node_parser_mod


_install_llama_index_stub()

from modules.agent import Agent
from modules.depression import DepressionSimulationEngine


class DummyLogger:
    def __init__(self) -> None:
        self.logs = []

    def info(self, message: str) -> None:
        self.logs.append(message)

    def warning(self, message: str) -> None:
        self.logs.append(message)

    def debug(self, message: str) -> None:
        self.logs.append(message)


class DummyTile:
    def get_address(self):
        return ["the Ville", "诊所", "心理咨询室"]


def _write_agent_config(tmp_path: Path, payload: dict) -> Path:
    agent_dir = tmp_path / "agent"
    agent_dir.mkdir(parents=True, exist_ok=True)
    config_path = agent_dir / "depression_config.json"
    config_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return agent_dir


def _base_simulation_payload(*, enabled: bool = True) -> dict:
    return {
        "enabled": enabled,
        "profile": {
            "name": "卡布达",
        },
        "complaint_chain": {
            "initial_stage_id": "job_loss",
            "stages": [
                {
                    "id": "job_loss",
                    "label": "失业即失败",
                    "summary": "把失业解释成整个人失败。",
                    "core_belief": "我这个人没有价值。",
                }
            ],
        },
        "emotion": {
            "llm_enabled": False,
        },
        "memory": {
            "enabled": False,
        },
        "prompt": {
            "include_chain_window": True,
            "include_bias_layer": True,
            "include_emotion_layer": True,
        },
    }


def _build_agent_stub(*, agent_dir: Path, state_payload: dict | None = None) -> Agent:
    agent = Agent.__new__(Agent)
    agent.name = "卡布达"
    agent.logger = DummyLogger()
    agent._llm = None
    agent.think_config = {
        "llm": {
            "retry": 3,
            "temperature": 0.5,
        }
    }
    agent.scratch = SimpleNamespace(
        currently="最近过得不太好。",
        _base_desc=lambda: "你是卡布达。最近情绪低落。",
    )
    agent.depression_dynamic_global = {
        "enabled": True,
    }
    agent.get_tile = MethodType(lambda self: DummyTile(), agent)
    agent.status = {}
    agent.schedule = SimpleNamespace(to_dict=lambda: {})
    agent.associate = SimpleNamespace(to_dict=lambda: {})
    agent.chats = []
    agent.action = SimpleNamespace(to_dict=lambda: {})
    agent.depression_dynamic = None
    config = {
        "agent_dir": str(agent_dir),
    }
    if state_payload is not None:
        config["depression_dynamic_state"] = state_payload
    agent.depression_dynamic = Agent._init_depression_dynamic(agent, config)
    return agent


def test_init_depression_dynamic_enables_flat_top_level_config(tmp_path: Path) -> None:
    agent_dir = _write_agent_config(tmp_path, _base_simulation_payload(enabled=True))
    agent = _build_agent_stub(agent_dir=agent_dir)

    assert isinstance(agent.depression_dynamic, DepressionSimulationEngine)
    assert agent.depression_dynamic.raw_config["enabled"] is True
    assert (
        agent.depression_dynamic.get_current_state_info()["current_stage"]["id"]
        == "job_loss"
    )


def test_init_depression_dynamic_preserves_nested_config_shape(tmp_path: Path) -> None:
    agent_dir = _write_agent_config(
        tmp_path,
        {"depression_simulation": _base_simulation_payload(enabled=True)},
    )
    agent = _build_agent_stub(agent_dir=agent_dir)

    assert isinstance(agent.depression_dynamic, DepressionSimulationEngine)
    assert "depression_simulation" in agent.depression_dynamic.raw_config
    assert agent.depression_dynamic.config_path.endswith("depression_config.json")


def test_init_depression_dynamic_keeps_disabled_engine_state(tmp_path: Path) -> None:
    agent_dir = _write_agent_config(tmp_path, _base_simulation_payload(enabled=False))
    agent = _build_agent_stub(agent_dir=agent_dir)

    assert isinstance(agent.depression_dynamic, DepressionSimulationEngine)
    assert agent.depression_dynamic.get_current_state_info()["enabled"] is False


def test_prepare_and_commit_generate_chat_use_direct_engine(tmp_path: Path) -> None:
    agent_dir = _write_agent_config(tmp_path, _base_simulation_payload(enabled=True))
    agent = _build_agent_stub(agent_dir=agent_dir)
    other = SimpleNamespace(name="蜻蜓队长")
    chats = [("用户", "我最近状态很差")]

    prompt_kwargs, context = Agent._prepare_depression_generate_chat(
        agent,
        args=(agent, other, "治疗师", chats),
        kwargs={},
    )

    assert context is not None
    assert prompt_kwargs["depression_chat_block"]
    before = agent.depression_dynamic.get_current_state_info()["interaction_count"]

    Agent._commit_depression_generate_chat(agent, context, "我最近真的很累。")

    after = agent.depression_dynamic.get_current_state_info()["interaction_count"]
    assert after == before + 1


def test_to_dict_persists_raw_engine_state(tmp_path: Path) -> None:
    agent_dir = _write_agent_config(tmp_path, _base_simulation_payload(enabled=True))
    agent = _build_agent_stub(agent_dir=agent_dir)
    payload = Agent.to_dict(agent)

    assert "depression_dynamic_state" in payload
    assert "depression_config_path" in payload
    state = payload["depression_dynamic_state"]
    assert isinstance(state, dict)
    assert "config" in state
    assert "chain_manager" in state
    assert "runtime" not in state
    assert "schema_version" not in state
    assert payload["depression_config_path"].endswith("depression_config.json")
