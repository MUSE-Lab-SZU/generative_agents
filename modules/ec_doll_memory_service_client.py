"""EC-Doll 记忆服务 HTTP 客户端（对齐 MEMORY_SERVICE.md）。"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, TypedDict, Union

import requests

JsonDict = Dict[str, Any]
JsonList = List[JsonDict]
JsonData = Union[JsonDict, JsonList]


class CognitiveSchema(TypedDict):
    """`/api/cognitive/schemas` 返回的图式对象（已弃用接口）。"""

    schema_id: str
    name: str
    activation_score: float
    definition: str
    associated_thoughts: List[str]


class CognitiveHistoryItem(TypedDict):
    """认知接口 history 列表中的单条对话记录（已弃用接口）。"""

    user_input: str
    ai_response: str
    active_schemas: List[Any]


class ECDollMemoryServiceClient:
    """EC-Doll 记忆微服务客户端。"""

    ENDPOINT_INDEX: Dict[str, str] = {
        "ingest": "POST /ingest",
        "retrieve": "POST /retrieve",
        "milestones": "GET /api/milestones",
        "memories": "GET /api/memories",
        "user_delete": "DELETE /api/users/{user_id}",
        "memory_update": "PUT /api/memories/{id}",
        "memory_delete": "DELETE /api/memories/{id}",
        "memory_promote": "POST /api/memories/{id}/promote",
        "memory_demote": "POST /api/memories/{id}/demote",
        "memory_toggle_lock": "POST /api/memories/{id}/toggle_lock",
        "memory_milestone": "POST /api/memories/{id}/milestone",
        "memory_settings": "POST /api/settings/memory",
        "health": "GET /health",
        "ready": "GET /ready",
    }

    def __init__(
        self,
        base_url: str = "http://localhost:8031",
        timeout: float = 30.0,
        session: Optional[requests.Session] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = session or requests.Session()

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        json_body: Optional[Dict[str, Any]] = None,
    ) -> JsonData:
        response = self.session.request(
            method=method,
            url=f"{self.base_url}{path}",
            params=self._drop_none(params),
            json=self._drop_none(json_body),
            timeout=self.timeout,
        )
        response.raise_for_status()
        if not response.content:
            return {}
        return response.json()

    @staticmethod
    def _drop_none(data: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if data is None:
            return None
        return {key: value for key, value in data.items() if value is not None}

    @staticmethod
    def _warn_deprecated(name: str, replacement: str = "") -> None:
        tip = f"`{name}` 已弃用，不在最新 MEMORY_SERVICE.md 接口清单中。"
        if replacement:
            tip += f" 建议改用 `{replacement}`。"
        warnings.warn(tip, DeprecationWarning, stacklevel=2)

    def get_emotion_timeline(
        self,
        limit: int = 50,
        user_id: Optional[str] = None,
    ) -> JsonList:
        """
        已弃用接口：
        - 对应旧版路径：`GET /api/v1/visual/emotion_timeline`
        - 最新文档 `MEMORY_SERVICE.md` 已不再定义该接口

        输入：
        - `limit`(int): 返回最近情绪点数量。
        - `user_id`(str|None): 旧接口中的用户 ID 过滤参数。

        输出：
        - List[dict]：旧接口情绪点数组。常见字段：
          `timestamp`, `intensity`, `tag`, `related_msg_id`。
        """
        self._warn_deprecated("get_emotion_timeline")
        data = self._request(
            "GET",
            "/api/v1/visual/emotion_timeline",
            params={"limit": limit, "user_id": user_id},
        )
        return data if isinstance(data, list) else []

    def ingest_memory(
        self,
        content: str,
        user_id: str,
        remote_id: Optional[str] = None,
    ) -> JsonDict:
        """
        写入一条用户消息：`POST /ingest`。

        输入（JSON）：
        - `user_id`(str, 必填): 用户唯一 ID。
        - `content`(str, 必填): 本条消息原文。
        - `remote_id`(str|None, 可选): 外部消息 ID；不传由服务生成。

        输出（JSON）：
        - `status`(str): 例如 `completed` / `stored_raw` / `error`。
        - `message`(str): 人类可读说明。
        - `remote_id`(str): 本条消息最终 ID。
        - `long_term_pending`(bool): 是否已进入长期分析后台队列。
        - `analysis_ok`(bool|None): 分析结果是否可用。
        - `mws_score`(float|None): 消息 MWS 分数。
        - `stored_raw_fallback`(bool): LLM 不可用时是否退化为仅存原文。
        """
        data = self._request(
            "POST",
            "/ingest",
            json_body={
                "content": content,
                "user_id": user_id,
                "remote_id": remote_id,
            },
        )
        return data if isinstance(data, dict) else {}

    def retrieve_memory_context(
        self,
        query: str,
        user_id: str,
        active_project: Optional[str] = None,
    ) -> JsonDict:
        """
        组装记忆上下文：`POST /retrieve`。

        输入（JSON）：
        - `user_id`(str, 必填): 用户唯一 ID。
        - `query`(str, 必填): 当前轮查询文本，用于向量检索。
        - `active_project`(str|None, 可选): 预留字段，暂未使用。

        输出（JSON）：
        - `formatted_prompt`(str): 可直接拼接到系统提示词的完整上下文。
        - `details`(dict): 结构化上下文，常见字段：
          - `l0_history`(list): 外部原文历史。
          - `ranked_memories`(list): 长期召回与重排后的记忆。
          - `profile_data`(dict): 用户画像键值。
          - `recent_emotion`(list): 最近情绪记录。
          - `milestones`(list): 里程碑列表。
          - `short_term_recent`(list): 近期原文短期记忆。
        """
        data = self._request(
            "POST",
            "/retrieve",
            json_body={
                "query": query,
                "user_id": user_id,
                "active_project": active_project,
            },
        )
        return data if isinstance(data, dict) else {}

    def list_milestones(self, user_id: str) -> JsonList:
        """
        查询里程碑列表：`GET /api/milestones`。

        输入（Query）：
        - `user_id`(str, 必填): 用户唯一 ID。

        输出（JSON List）：
        - 每项常见字段：`id`, `user_id`, `title`, `level`, `happened_at`,
          `description`, `remote_id`, `created_at`。
        """
        data = self._request("GET", "/api/milestones", params={"user_id": user_id})
        return data if isinstance(data, list) else []

    def list_memories(
        self,
        level: Optional[str] = None,
        locked: Optional[bool] = None,
        user_id: Optional[str] = None,
        page: Optional[int] = None,
        page_size: Optional[int] = None,
    ) -> JsonData:
        """
        查询记忆列表：`GET /api/memories`。

        输入（Query）：
        - `user_id`(str|None): 用户 ID 过滤。
        - `level`(str|None): 记忆层级过滤（`L1`/`L2`）。
        - `locked`(bool|None): 锁定状态过滤。
        - `page`(int|None): 分页页码（从 1 开始）。
        - `page_size`(int|None): 分页大小。

        输出（JSON）：
        - 非分页：`List[dict]`。
        - 分页：`dict`，固定字段为 `items`, `total`, `page`, `page_size`。
        """
        return self._request(
            "GET",
            "/api/memories",
            params={
                "level": level,
                "locked": locked,
                "user_id": user_id,
                "page": page,
                "page_size": page_size,
            },
        )

    def update_memory(
        self,
        memory_id: str,
        content: Optional[str] = None,
        mws_score: Optional[float] = None,
    ) -> JsonDict:
        """
        更新记忆：`PUT /api/memories/{id}`。

        输入（Path + JSON）：
        - `memory_id`(str, 必填): 目标记忆 ID。
        - `content`(str|None): 新内容摘要。
        - `mws_score`(float|None): 新 MWS 分数。
        - 约束：`content` 与 `mws_score` 至少传一个。

        输出（JSON）：
        - 透传服务端返回，常见字段：`status`, `id`, `message`。
        """
        if content is None and mws_score is None:
            raise ValueError("update_memory requires at least one of content or mws_score")
        data = self._request(
            "PUT",
            f"/api/memories/{memory_id}",
            json_body={"content": content, "mws_score": mws_score},
        )
        return data if isinstance(data, dict) else {}

    def delete_memory(self, memory_id: str) -> JsonDict:
        """
        删除记忆：`DELETE /api/memories/{id}`。

        输入（Path）：
        - `memory_id`(str, 必填): 目标记忆 ID。

        输出（JSON）：
        - 透传服务端返回，常见字段：`status`, `message`, `id`。
        """
        data = self._request("DELETE", f"/api/memories/{memory_id}")
        return data if isinstance(data, dict) else {}

    def delete_user_data(self, user_id: str, dry_run: bool = False) -> JsonDict:
        """
        按 user_id 级联清理：`DELETE /api/users/{user_id}`。

        输入（Path + Query）：
        - `user_id`(str, 必填): 目标用户 ID。
        - `dry_run`(bool, 可选): 默认 `False`；为 `True` 时只预览匹配条数，不执行删除。

        输出（JSON）：
        - `status`(str): `ok` / `partial` / `dry_run` / `error`。
        - `deleted_chroma`(int): Chroma 匹配条数（或删除条数）。
        - `deleted_pg`(dict): PG 各表计数。
        - `deleted_redis`(dict): Redis key 命中情况。
        - `errors`(list): 分层删除失败信息。
        """
        user_id = str(user_id or "").strip()
        if not user_id:
            raise ValueError("delete_user_data requires non-empty user_id")
        data = self._request(
            "DELETE",
            f"/api/users/{user_id}",
            params={"dry_run": dry_run},
        )
        return data if isinstance(data, dict) else {}

    def promote_memory(self, memory_id: str) -> JsonDict:
        """
        强制升级记忆：`POST /api/memories/{id}/promote`。

        输入（Path）：
        - `memory_id`(str, 必填): 目标记忆 ID。

        输出（JSON）：
        - 透传服务端返回，常见字段：`status`, `message`。
        """
        data = self._request("POST", f"/api/memories/{memory_id}/promote")
        return data if isinstance(data, dict) else {}

    def demote_memory(self, memory_id: str) -> JsonDict:
        """
        强制降级记忆：`POST /api/memories/{id}/demote`。

        输入（Path）：
        - `memory_id`(str, 必填): 目标记忆 ID。

        输出（JSON）：
        - 透传服务端返回，常见字段：`status`, `message`。
        """
        data = self._request("POST", f"/api/memories/{memory_id}/demote")
        return data if isinstance(data, dict) else {}

    def toggle_memory_lock(self, memory_id: str) -> JsonDict:
        """
        切换锁定状态：`POST /api/memories/{id}/toggle_lock`。

        输入（Path）：
        - `memory_id`(str, 必填): 目标记忆 ID。

        输出（JSON）：
        - 透传服务端返回，常见字段：`status`, `locked`。
        """
        data = self._request("POST", f"/api/memories/{memory_id}/toggle_lock")
        return data if isinstance(data, dict) else {}

    def set_memory_milestone(self, memory_id: str, is_milestone: bool) -> JsonDict:
        """
        设置里程碑标记：`POST /api/memories/{id}/milestone`。

        输入（Path + JSON）：
        - `memory_id`(str, 必填): 目标记忆 ID。
        - `is_milestone`(bool, 必填): 是否设为里程碑。

        输出（JSON）：
        - 透传服务端返回，常见字段：`status`, `is_milestone`。
        """
        data = self._request(
            "POST",
            f"/api/memories/{memory_id}/milestone",
            json_body={"is_milestone": is_milestone},
        )
        return data if isinstance(data, dict) else {}

    def update_memory_settings(self, l2_threshold: float) -> JsonDict:
        """
        更新记忆阈值配置：`POST /api/settings/memory`。

        输入（JSON）：
        - `l2_threshold`(float, 必填): 新的 L2 阈值。

        输出（JSON）：
        - 透传服务端返回，常见字段：`status`, `message`。
        """
        data = self._request(
            "POST",
            "/api/settings/memory",
            json_body={"l2_threshold": l2_threshold},
        )
        return data if isinstance(data, dict) else {}

    def get_cognitive_schemas(self, user_id: str) -> List[CognitiveSchema]:
        """
        已弃用接口：`GET /api/cognitive/schemas`（最新文档不再定义）。

        输入：
        - `user_id`(str): 用户 ID。

        输出：
        - List[dict]：认知图式数组。
        """
        self._warn_deprecated("get_cognitive_schemas")
        data = self._request(
            "GET",
            "/api/cognitive/schemas",
            params={"user_id": user_id},
        )
        return data if isinstance(data, list) else []

    def run_cognitive_workflow(
        self,
        user_id: str,
        content: str,
        history: Optional[List[CognitiveHistoryItem]] = None,
    ) -> JsonDict:
        """
        已弃用接口：`POST /api/cognitive/workflow`（最新文档不再定义）。

        输入：
        - `user_id`(str): 用户 ID。
        - `content`(str): 当前输入文本。
        - `history`(list|None): 历史对话记录。

        输出：
        - dict：常见字段 `inner_monologue`, `active_schemas`, `reflection_result`。
        """
        self._warn_deprecated("run_cognitive_workflow")
        data = self._request(
            "POST",
            "/api/cognitive/workflow",
            json_body={"user_id": user_id, "content": content, "history": history},
        )
        return data if isinstance(data, dict) else {}

    def interpret_cognitive(self, user_id: str, content: str) -> JsonDict:
        """
        已弃用接口：`POST /api/cognitive/interpret`（最新文档不再定义）。

        输入：
        - `user_id`(str): 用户 ID。
        - `content`(str): 当前输入文本。

        输出：
        - dict：常见字段 `inner_monologue`, `active_schemas`。
        """
        self._warn_deprecated("interpret_cognitive")
        data = self._request(
            "POST",
            "/api/cognitive/interpret",
            json_body={"user_id": user_id, "content": content},
        )
        return data if isinstance(data, dict) else {}

    def reflect_cognitive(
        self,
        user_id: str,
        history: List[CognitiveHistoryItem],
    ) -> JsonDict:
        """
        已弃用接口：`POST /api/cognitive/reflect`（最新文档不再定义）。

        输入：
        - `user_id`(str): 用户 ID。
        - `history`(list): 历史对话记录。

        输出：
        - dict：常见字段 `status`, `insights`, `updates`。
        """
        self._warn_deprecated("reflect_cognitive")
        data = self._request(
            "POST",
            "/api/cognitive/reflect",
            json_body={"user_id": user_id, "history": history},
        )
        return data if isinstance(data, dict) else {}

    def health_check(self) -> JsonDict:
        """
        健康检查：`GET /health`。

        输出（JSON）：
        - `status`(str): 典型值 `ok`。
        - `version`(str): 服务版本，文档示例为 `v2`。
        """
        data = self._request("GET", "/health")
        return data if isinstance(data, dict) else {}

    def ready_check(self) -> JsonDict:
        """
        依赖就绪检查：`GET /ready`。

        输出（JSON）：
        - `status`(str): 典型值 `ready`。
        - `redis`(bool): Redis 依赖是否可用。
        - `postgres`(bool): PostgreSQL 依赖是否可用。
        - `chroma`(bool): Chroma 依赖是否可用。
        - 语义：任一依赖不可用时，服务应返回 HTTP 503。
        """
        try:
            data = self._request("GET", "/ready")
            return data if isinstance(data, dict) else {}
        except requests.HTTPError as exc:
            response = exc.response
            if response is not None and response.status_code == 503 and response.content:
                try:
                    data = response.json()
                except ValueError:
                    return {}
                return data if isinstance(data, dict) else {}
            raise
