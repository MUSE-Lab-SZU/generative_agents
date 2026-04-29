from __future__ import annotations

import datetime
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(CURRENT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from modules.external_memory_bridge import ExternalMemoryBridge


class DummyLogger:
    def __init__(self) -> None:
        self.logs: List[str] = []

    def info(self, message: str) -> None:
        self.logs.append(message)

    def warning(self, message: str) -> None:
        self.logs.append(message)


class DummyEvent:
    def __init__(self, text: str, address: Optional[List[str]] = None) -> None:
        self._text = text
        self.address = address or ["arena", "room"]

    def get_describe(self) -> str:
        return self._text


class DummyClient:
    def __init__(
        self,
        *,
        ingest_response: Optional[Dict[str, Any]] = None,
        retrieve_response: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.ingest_response = ingest_response or {}
        self.retrieve_response = retrieve_response or {}
        self.ingest_calls: List[Dict[str, Any]] = []
        self.retrieve_calls: List[Dict[str, Any]] = []

    def ingest_memory(self, content: str, user_id: str, remote_id: Optional[str] = None) -> Dict[str, Any]:
        self.ingest_calls.append({"content": content, "user_id": user_id, "remote_id": remote_id})
        return self.ingest_response

    def retrieve_memory_context(
        self,
        query: str,
        user_id: str,
        active_project: Optional[str] = None,
    ) -> Dict[str, Any]:
        self.retrieve_calls.append({"query": query, "user_id": user_id, "active_project": active_project})
        return self.retrieve_response


def _build_bridge(
    tmp_path: Path,
    user_id: str = "agent_demo",
    short_term_recent_n: int = 40,
) -> ExternalMemoryBridge:
    storage_dir = tmp_path / "results" / "checkpoints" / "save_a" / "storage" / "agent_1" / "associate"
    logger = DummyLogger()
    bridge = ExternalMemoryBridge(
        agent_name="agent_1",
        cfg={
            "enabled": False,
            "user_id": user_id,
            "short_term_recent_n": short_term_recent_n,
        },
        storage_dir=str(storage_dir),
        logger=logger,
    )
    bridge.enabled = True
    return bridge


def _sample_formatted_prompt_with_recent_raw() -> str:
    return (
        "【相关记忆 (Long Term)】:\n"
        "1. 示例长期记忆\n\n"
        "【近期原文（本服务入库）】:\n"
        "- [2026-04-25T06:51:05] [msg_a] 记忆时间：20260427-16:30:00\n"
        "记忆类型：thought\n"
        "内容：A\n"
        "- [2026-04-25T06:51:08] [msg_b] 记忆时间：20260427-16:31:00\n"
        "记忆类型：thought\n"
        "内容：B\n"
        "- [2026-04-25T06:51:09] [msg_c] 记忆时间：20260427-16:32:00\n"
        "记忆类型：thought\n"
        "内容：C\n\n"
        "【近期情绪（本服务入库）】:\n"
        "- [2026-04-25 06:51:10] Happy (Intensity: 0.8)"
    )


def test_scoped_user_id_uses_checkpoint_name(tmp_path: Path) -> None:
    bridge = _build_bridge(tmp_path, user_id="agent_kabuda")
    assert bridge.save_name == "save_a"
    assert bridge.get_scoped_user_id() == "agent_kabuda+save_a"
    assert bridge.build_scoped_user_id("agent_kabuda", "") == "agent_kabuda"


def test_ingest_requires_success_status_and_remote_id(tmp_path: Path) -> None:
    bridge = _build_bridge(tmp_path)
    bridge.client = DummyClient(
        ingest_response={
            "status": "completed",
            "message": "ok",
            "remote_id": "msg_001",
            "long_term_pending": True,
            "analysis_ok": None,
            "mws_score": None,
            "stored_raw_fallback": False,
        }
    )
    result = bridge.ingest_from_local_node(
        node_id="node_1",
        node_type="chat",
        event=DummyEvent("hello world"),
        create_time=datetime.datetime(2026, 4, 24, 10, 0, 0),
    )
    assert result["ok"] is True
    assert result["reason"] == "ok"
    assert result["remote_id"] == "msg_001"
    assert bridge.client.ingest_calls[0]["user_id"] == "agent_demo+save_a"
    assert Path(bridge.map_path).is_file()
    ingest_logs = [line for line in bridge.logger.logs if "[EXT_MEMORY_INGEST]" in line]
    assert ingest_logs
    assert "message=ok" in ingest_logs[-1]
    assert "long_term_pending=True" in ingest_logs[-1]
    assert "stored_raw_fallback=False" in ingest_logs[-1]
    assert "elapsed_ms=" in ingest_logs[-1]


def test_ingest_rejects_error_status_even_with_remote_id(tmp_path: Path) -> None:
    bridge = _build_bridge(tmp_path)
    bridge.client = DummyClient(
        ingest_response={
            "status": "error",
            "message": "failed",
            "remote_id": "msg_999",
        }
    )
    result = bridge.ingest_from_local_node(
        node_id="node_1",
        node_type="chat",
        event=DummyEvent("hello world"),
        create_time=datetime.datetime(2026, 4, 24, 10, 0, 0),
    )
    assert result["ok"] is False
    assert result["reason"] == "ingest_status_error"
    assert result["remote_id"] == "msg_999"
    assert not Path(bridge.map_path).exists()


def test_ingest_rejects_missing_remote_id(tmp_path: Path) -> None:
    bridge = _build_bridge(tmp_path)
    bridge.client = DummyClient(
        ingest_response={
            "status": "completed",
            "message": "ok",
            "remote_id": "",
        }
    )
    result = bridge.ingest_from_local_node(
        node_id="node_1",
        node_type="chat",
        event=DummyEvent("hello world"),
        create_time=datetime.datetime(2026, 4, 24, 10, 0, 0),
    )
    assert result["ok"] is False
    assert result["reason"] == "missing_remote_id"
    assert not Path(bridge.map_path).exists()


def test_retrieve_passes_scoped_user_id_and_exposes_details(tmp_path: Path) -> None:
    bridge = _build_bridge(tmp_path)
    bridge.client = DummyClient(
        retrieve_response={
            "formatted_prompt": "memory context",
            "details": {
                "ranked_memories": [{"remote_id": "msg_1"}],
                "recent_emotion": [{"tag": "Happy", "intensity": 0.9}],
            },
        }
    )
    result = bridge.retrieve_chat_context(
        chats=[("other", "How are you?")],
        other_name="other",
        is_initiator=False,
        turn_no=2,
    )
    assert result["ok"] is True
    assert result["context"] == "memory context"
    assert "ranked_memories" in result["details"]
    assert bridge.client.retrieve_calls[0]["user_id"] == "agent_demo+save_a"
    retrieve_logs = [line for line in bridge.logger.logs if "[EXT_MEMORY_RETRIEVE]" in line]
    assert retrieve_logs
    assert "hit_remote_count=1" in retrieve_logs[-1]
    assert "hit_remote_ids=msg_1" in retrieve_logs[-1]
    assert "elapsed_ms=" in retrieve_logs[-1]


def test_retrieve_keeps_details_when_formatted_prompt_is_empty(tmp_path: Path) -> None:
    bridge = _build_bridge(tmp_path)
    bridge.client = DummyClient(
        retrieve_response={
            "formatted_prompt": "",
            "details": {"short_term_recent": [{"remote_id": "msg_7"}]},
        }
    )
    result = bridge.retrieve_chat_context(
        chats=[("other", "How are you?")],
        other_name="other",
        is_initiator=False,
        turn_no=2,
    )
    assert result["ok"] is False
    assert result["reason"] == "empty_formatted_prompt"
    assert "short_term_recent" in result["details"]


def test_retrieve_recent_raw_keeps_last_n_entries_and_order(tmp_path: Path) -> None:
    bridge = _build_bridge(tmp_path, short_term_recent_n=2)
    bridge.client = DummyClient(
        retrieve_response={
            "formatted_prompt": _sample_formatted_prompt_with_recent_raw(),
            "details": {},
        }
    )
    result = bridge.retrieve_chat_context(
        chats=[("other", "How are you?")],
        other_name="other",
        is_initiator=False,
        turn_no=2,
    )
    context = result["context"]
    assert result["ok"] is True
    assert "[msg_a]" not in context
    assert "[msg_b]" in context
    assert "[msg_c]" in context
    assert context.index("[msg_b]") < context.index("[msg_c]")


def test_retrieve_recent_raw_strips_timestamp_prefix(tmp_path: Path) -> None:
    bridge = _build_bridge(tmp_path, short_term_recent_n=40)
    bridge.client = DummyClient(
        retrieve_response={
            "formatted_prompt": _sample_formatted_prompt_with_recent_raw(),
            "details": {},
        }
    )
    result = bridge.retrieve_chat_context(
        chats=[("other", "How are you?")],
        other_name="other",
        is_initiator=False,
        turn_no=2,
    )
    context = result["context"]
    assert result["ok"] is True
    assert "- [2026-04-25T06:51:05] [msg_a]" not in context
    assert "- [2026-04-25T06:51:08] [msg_b]" not in context
    assert "- [2026-04-25T06:51:09] [msg_c]" not in context
    assert "- [msg_a] 记忆时间：20260427-16:30:00" in context
    assert "- [msg_b] 记忆时间：20260427-16:31:00" in context
    assert "- [msg_c] 记忆时间：20260427-16:32:00" in context


def test_retrieve_recent_raw_zero_keeps_title_only(tmp_path: Path) -> None:
    bridge = _build_bridge(tmp_path, short_term_recent_n=0)
    bridge.client = DummyClient(
        retrieve_response={
            "formatted_prompt": _sample_formatted_prompt_with_recent_raw(),
            "details": {},
        }
    )
    result = bridge.retrieve_chat_context(
        chats=[("other", "How are you?")],
        other_name="other",
        is_initiator=False,
        turn_no=2,
    )
    context = result["context"]
    assert result["ok"] is True
    assert "【近期原文（本服务入库）】:" in context
    assert "[msg_a]" not in context
    assert "[msg_b]" not in context
    assert "[msg_c]" not in context
    assert "【近期情绪（本服务入库）】:" in context


def test_retrieve_recent_raw_no_section_keeps_original_context(tmp_path: Path) -> None:
    original_context = "【相关记忆 (Long Term)】:\n1. 示例长期记忆"
    bridge = _build_bridge(tmp_path, short_term_recent_n=2)
    bridge.client = DummyClient(
        retrieve_response={
            "formatted_prompt": original_context,
            "details": {},
        }
    )
    result = bridge.retrieve_chat_context(
        chats=[("other", "How are you?")],
        other_name="other",
        is_initiator=False,
        turn_no=2,
    )
    assert result["ok"] is True
    assert result["context"] == original_context
