"""基于 MEMORY_SERVICE.md 的客户端示例测试（mock HTTP，无需真实服务）。"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import pytest
import requests

from modules.ec_doll_memory_service_client import ECDollMemoryServiceClient


class MockResponse:
    """最小可用的 requests.Response 替身。"""

    def __init__(self, status_code: int, json_data: Any) -> None:
        self.status_code = status_code
        self._json_data = json_data
        self.content = b"" if json_data is None else b"{}"
        self.text = str(json_data)

    def json(self) -> Any:
        return self._json_data

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}: {self.text}")


def _build_memory_rows() -> List[Dict[str, Any]]:
    return [
        {
            "remote_id": "msg_999",
            "level": "L1",
            "content": "我今天中了彩票!",
            "mws_score": 0.85,
            "mws_components": {"emotion": 1.0, "info": 0.5, "user": 0.0},
            "is_locked": False,
            "is_milestone": False,
            "timestamp": "2026-04-22T08:45:10.000000",
            "user_id": "user_123",
        },
        {
            "remote_id": "msg_101",
            "level": "L2",
            "content": "我之前提到拿到了奖学金。",
            "mws_score": 0.92,
            "mws_components": {"emotion": 0.9, "info": 0.8, "user": 0.5},
            "is_locked": True,
            "is_milestone": True,
            "timestamp": "2026-04-22T08:45:11.000000",
            "user_id": "user_123",
        },
        {
            "remote_id": "msg_888",
            "level": "L2",
            "content": "我在新的项目里负责算法部分。",
            "mws_score": 0.74,
            "mws_components": {"emotion": 0.5, "info": 0.9, "user": 0.4},
            "is_locked": False,
            "is_milestone": False,
            "timestamp": "2026-04-22T08:45:12.000000",
            "user_id": "user_456",
        },
    ]


def _apply_memory_filters(
    data: List[Dict[str, Any]],
    level: Optional[str],
    locked: Optional[Any],
    user_id: Optional[str],
) -> List[Dict[str, Any]]:
    result = data
    if level is not None:
        result = [item for item in result if item["level"] == level]
    if user_id is not None:
        result = [item for item in result if item["user_id"] == user_id]
    if locked is not None:
        if isinstance(locked, str):
            locked = locked.lower() == "true"
        result = [item for item in result if item["is_locked"] is bool(locked)]
    return result


@pytest.fixture
def client_with_mock_service(monkeypatch: pytest.MonkeyPatch) -> ECDollMemoryServiceClient:
    milestones = [
        {
            "id": 1,
            "user_id": "user_123",
            "title": "拿到奖学金",
            "level": 1,
            "happened_at": "2026-04-22T08:45:11",
            "description": "获得全额奖学金",
            "remote_id": "msg_101",
            "created_at": "2026-04-22T08:45:11",
        }
    ]
    memories = _build_memory_rows()
    lock_states: Dict[str, bool] = {m["remote_id"]: bool(m["is_locked"]) for m in memories}

    def dispatch(
        method: str,
        url: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        json_body: Optional[Dict[str, Any]] = None,
    ) -> MockResponse:
        method = method.upper()
        path = urlparse(url).path
        params = params or {}
        json_body = json_body or {}

        if method == "POST" and path == "/ingest":
            return MockResponse(
                200,
                {
                    "status": "completed",
                    "message": "Short-term stored; long-term analysis pending.",
                    "remote_id": json_body.get("remote_id") or "msg_12",
                    "analysis_ok": None,
                    "mws_score": None,
                    "stored_raw_fallback": False,
                    "long_term_pending": True,
                },
            )

        if method == "POST" and path == "/retrieve":
            return MockResponse(
                200,
                {
                    "formatted_prompt": "memory context prompt",
                    "details": {
                        "l0_history": [{"role": "user", "content": "你好"}],
                        "ranked_memories": memories[:2],
                        "profile_data": {"major": "psychology"},
                        "recent_emotion": [{"tag": "Happy", "intensity": 0.9}],
                        "milestones": milestones,
                        "short_term_recent": [
                            {"timestamp": "2026-04-22T08:45:05", "remote_id": "msg_12", "content": "最近在准备答辩"}
                        ],
                    },
                },
            )

        if method == "GET" and path == "/api/milestones":
            return MockResponse(200, milestones)

        if method == "GET" and path == "/api/memories":
            filtered = _apply_memory_filters(
                data=memories,
                level=params.get("level"),
                locked=params.get("locked"),
                user_id=params.get("user_id"),
            )
            page = params.get("page")
            page_size = params.get("page_size")
            if page is not None and page_size is not None:
                page_i = int(page)
                page_size_i = int(page_size)
                total = len(filtered)
                start = (page_i - 1) * page_size_i
                items = filtered[start : start + page_size_i]
                return MockResponse(
                    200,
                    {
                        "items": items,
                        "total": total,
                        "page": page_i,
                        "page_size": page_size_i,
                    },
                )
            return MockResponse(200, filtered)

        if path.startswith("/api/memories/"):
            parts = path.strip("/").split("/")
            if len(parts) == 3 and method == "PUT":
                memory_id = parts[2]
                return MockResponse(
                    200,
                    {
                        "status": "success",
                        "id": memory_id,
                        "updated_fields": sorted(json_body.keys()),
                    },
                )
            if len(parts) == 3 and method == "DELETE":
                memory_id = parts[2]
                return MockResponse(
                    200,
                    {
                        "status": "success",
                        "message": "Memory deleted",
                        "id": memory_id,
                    },
                )
            if len(parts) == 4 and method == "POST":
                memory_id = parts[2]
                action = parts[3]
                if action == "promote":
                    return MockResponse(
                        200,
                        {
                            "status": "success",
                            "message": f"Memory {memory_id} promoted",
                        },
                    )
                if action == "demote":
                    return MockResponse(
                        200,
                        {
                            "status": "success",
                            "message": f"Memory {memory_id} demoted",
                        },
                    )
                if action == "toggle_lock":
                    now_locked = not lock_states.get(memory_id, False)
                    lock_states[memory_id] = now_locked
                    return MockResponse(
                        200,
                        {
                            "status": "success",
                            "locked": now_locked,
                        },
                    )
                if action == "milestone":
                    return MockResponse(
                        200,
                        {
                            "status": "success",
                            "is_milestone": bool(json_body.get("is_milestone")),
                        },
                    )

        if method == "POST" and path == "/api/settings/memory":
            threshold = json_body["l2_threshold"]
            return MockResponse(
                200,
                {
                    "status": "success",
                    "message": (
                        f"L2 threshold updated to {threshold}. "
                        "Note: V2 architecture uses unified storage, no physical migration triggered."
                    ),
                },
            )

        if method == "GET" and path == "/health":
            return MockResponse(200, {"status": "ok", "version": "v2"})

        if method == "GET" and path == "/ready":
            return MockResponse(200, {"status": "ready", "redis": True, "postgres": True, "chroma": True})

        return MockResponse(404, {"message": f"Unhandled route: {method} {path}"})

    def patched_request(
        self: requests.Session,
        method: str,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        json: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> MockResponse:
        _ = timeout, kwargs
        return dispatch(method, url, params=params, json_body=json)

    monkeypatch.setattr(requests.Session, "request", patched_request)
    return ECDollMemoryServiceClient(base_url="http://localhost:8031")


def get_all_memories_paginated(
    client: ECDollMemoryServiceClient,
    user_id: Optional[str] = None,
    page_size: int = 50,
) -> List[Dict[str, Any]]:
    """对应文档中的分页获取全部记忆示例。"""
    all_memories: List[Dict[str, Any]] = []
    page = 1

    while True:
        data = client.list_memories(user_id=user_id, page=page, page_size=page_size)
        if not isinstance(data, dict) or "items" not in data:
            return data if isinstance(data, list) else all_memories

        items = data.get("items", [])
        total = int(data.get("total", 0))
        current_page = int(data.get("page", page))
        current_page_size = int(data.get("page_size", page_size))
        all_memories.extend(items if isinstance(items, list) else [])

        if len(all_memories) >= total:
            break
        if not isinstance(items, list) or len(items) < current_page_size:
            break
        page = current_page + 1

    return all_memories


def test_example_01_ingest_memory(client_with_mock_service: ECDollMemoryServiceClient) -> None:
    result = client_with_mock_service.ingest_memory(
        content="我今天拿到了全额奖学金！",
        user_id="user_123",
    )
    for key in (
        "status",
        "message",
        "remote_id",
        "long_term_pending",
        "analysis_ok",
        "mws_score",
        "stored_raw_fallback",
    ):
        assert key in result


def test_example_02_retrieve_memory(client_with_mock_service: ECDollMemoryServiceClient) -> None:
    result = client_with_mock_service.retrieve_memory_context(
        query="我之前提到过什么奖项？",
        user_id="user_123",
    )
    assert "formatted_prompt" in result
    details = result.get("details", {})
    for key in (
        "l0_history",
        "ranked_memories",
        "profile_data",
        "recent_emotion",
        "milestones",
        "short_term_recent",
    ):
        assert key in details


def test_example_03_query_milestones(client_with_mock_service: ECDollMemoryServiceClient) -> None:
    milestones = client_with_mock_service.list_milestones(user_id="user_123")
    assert len(milestones) == 1
    assert milestones[0]["title"] == "拿到奖学金"


def test_example_04_list_all_memories_no_pagination(
    client_with_mock_service: ECDollMemoryServiceClient,
) -> None:
    memories = client_with_mock_service.list_memories()
    assert isinstance(memories, list)
    assert len(memories) == 3


def test_example_05_list_user_l2_memories_no_pagination(
    client_with_mock_service: ECDollMemoryServiceClient,
) -> None:
    memories = client_with_mock_service.list_memories(user_id="user_123", level="L2")
    assert isinstance(memories, list)
    assert len(memories) == 1
    assert memories[0]["remote_id"] == "msg_101"


def test_example_06_list_memories_with_pagination(
    client_with_mock_service: ECDollMemoryServiceClient,
) -> None:
    paged = client_with_mock_service.list_memories(user_id="user_123", page=1, page_size=20)
    assert isinstance(paged, dict)
    for key in ("items", "total", "page", "page_size"):
        assert key in paged
    assert paged["page"] == 1
    assert paged["total"] == 2


def test_example_07_get_all_memories_paginated_function(
    client_with_mock_service: ECDollMemoryServiceClient,
) -> None:
    memories = get_all_memories_paginated(
        client_with_mock_service,
        user_id="user_123",
        page_size=1,
    )
    assert len(memories) == 2
    assert {m["remote_id"] for m in memories} == {"msg_999", "msg_101"}


def test_example_08_update_memory_content(client_with_mock_service: ECDollMemoryServiceClient) -> None:
    result = client_with_mock_service.update_memory(
        memory_id="msg_123",
        content="修正后的记忆内容：用户实际上更喜欢吃甜豆腐脑。",
    )
    assert result["status"] == "success"
    assert result["id"] == "msg_123"
    assert "content" in result["updated_fields"]


def test_example_09_update_memory_mws_score(client_with_mock_service: ECDollMemoryServiceClient) -> None:
    result = client_with_mock_service.update_memory(
        memory_id="msg_123",
        mws_score=0.88,
    )
    assert result["status"] == "success"
    assert "mws_score" in result["updated_fields"]


def test_example_10_update_memory_requires_fields(client_with_mock_service: ECDollMemoryServiceClient) -> None:
    with pytest.raises(ValueError):
        client_with_mock_service.update_memory(memory_id="msg_123")


def test_example_11_delete_memory(client_with_mock_service: ECDollMemoryServiceClient) -> None:
    result = client_with_mock_service.delete_memory(memory_id="msg_123")
    assert result["status"] == "success"
    assert result["message"] == "Memory deleted"
    assert result["id"] == "msg_123"


def test_example_12_promote_memory(client_with_mock_service: ECDollMemoryServiceClient) -> None:
    result = client_with_mock_service.promote_memory(memory_id="msg_101")
    assert result["status"] == "success"
    assert result["message"] == "Memory msg_101 promoted"


def test_example_13_demote_memory(client_with_mock_service: ECDollMemoryServiceClient) -> None:
    result = client_with_mock_service.demote_memory(memory_id="msg_101")
    assert result["status"] == "success"
    assert result["message"] == "Memory msg_101 demoted"


def test_example_14_toggle_lock(client_with_mock_service: ECDollMemoryServiceClient) -> None:
    result1 = client_with_mock_service.toggle_memory_lock(memory_id="msg_101")
    result2 = client_with_mock_service.toggle_memory_lock(memory_id="msg_101")
    assert result1["status"] == "success"
    assert result2["status"] == "success"
    assert result1["locked"] is False
    assert result2["locked"] is True


def test_example_15_set_milestone(client_with_mock_service: ECDollMemoryServiceClient) -> None:
    result = client_with_mock_service.set_memory_milestone(
        memory_id="msg_101",
        is_milestone=True,
    )
    assert result["status"] == "success"
    assert result["is_milestone"] is True


def test_example_16_update_memory_settings(
    client_with_mock_service: ECDollMemoryServiceClient,
) -> None:
    result = client_with_mock_service.update_memory_settings(l2_threshold=0.8)
    assert result["status"] == "success"
    assert "L2 threshold updated to 0.8" in result["message"]


def test_example_17_health_check(client_with_mock_service: ECDollMemoryServiceClient) -> None:
    result = client_with_mock_service.health_check()
    assert result == {"status": "ok", "version": "v2"}


def test_example_18_ready_check(client_with_mock_service: ECDollMemoryServiceClient) -> None:
    result = client_with_mock_service.ready_check()
    assert result["status"] == "ready"
    assert result["redis"] is True
    assert result["postgres"] is True
    assert result["chroma"] is True
