"""EC-Doll 记忆框架微服务接口聚合封装（V2 - 0413 文档版）。"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict, Union

import requests

JsonDict = Dict[str, Any]
JsonList = List[JsonDict]
JsonData = Union[JsonDict, JsonList]


class CognitiveSchema(TypedDict):
    """`/api/cognitive/schemas` 返回的图式对象。"""

    schema_id: str
    name: str
    activation_score: float
    definition: str
    associated_thoughts: List[str]


class CognitiveHistoryItem(TypedDict):
    """认知接口 history 列表中的单条对话记录。"""

    user_input: str
    ai_response: str
    active_schemas: List[Any]


class ECDollMemoryServiceClient:
    """EC-Doll 记忆微服务 HTTP 客户端。"""

    ENDPOINT_INDEX: Dict[str, str] = {
        "emotion_timeline": "GET /api/v1/visual/emotion_timeline",
        "ingest": "POST /ingest",
        "retrieve": "POST /retrieve",
        "milestones": "GET /api/milestones",
        "memories": "GET /api/memories",
        "memory_update": "PUT /api/memories/{id}",
        "memory_delete": "DELETE /api/memories/{id}",
        "memory_promote": "POST /api/memories/{id}/promote",
        "memory_demote": "POST /api/memories/{id}/demote",
        "memory_toggle_lock": "POST /api/memories/{id}/toggle_lock",
        "memory_milestone": "POST /api/memories/{id}/milestone",
        "memory_settings": "POST /api/settings/memory",
        "cognitive_schemas": "GET /api/cognitive/schemas",
        "cognitive_workflow": "POST /api/cognitive/workflow",
        "cognitive_interpret": "POST /api/cognitive/interpret",
        "cognitive_reflect": "POST /api/cognitive/reflect",
        "health": "GET /health",
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

    def get_emotion_timeline(
        self,
        limit: int = 50,
        user_id: Optional[str] = None,
    ) -> JsonList:
        """
        作用:
        获取情绪时间线数据，用于前端绘制实时情绪折线图。

        输入:
        - limit(int): 最近数据点数量，默认 50。
        - user_id(str|None): 用户 ID。文档总则说明所有接口支持 user_id。

        输出:
        - List[dict]: 按时间正序返回的情绪点数组，每项包含
          timestamp、intensity、tag、related_msg_id 等字段。
        """
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
        作用:
        向记忆系统写入一条用户消息，触发后端分析与存储流程。

        输入:
        - content(str): 用户输入文本。
        - user_id(str): 用户 ID（V2 必填）。
        - remote_id(str|None): 可选消息 ID，不传则由系统生成。

        输出:
        - dict: 通常包含 status、message、remote_id。
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
        作用:
        在回复生成前检索记忆上下文，返回可直接拼接到 Prompt 的背景信息。

        输入:
        - query(str): 当前用户输入。
        - user_id(str): 用户 ID（V2 必填）。
        - active_project(str|None): 可选项目/话题名。

        输出:
        - dict: 典型包含 formatted_prompt 与 details。
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
        作用:
        查询用户的里程碑事件列表（按发生时间倒序）。

        输入:
        - user_id(str): 用户 ID。

        输出:
        - List[dict]: 里程碑数组，常见字段包括 id、title、level、
          happened_at、description、remote_id、created_at。
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
        作用:
        获取记忆列表，支持按层级/锁定状态/用户过滤，并支持分页。

        输入:
        - level(str|None): 记忆层级过滤，文档示例值为 L1/L2。
        - locked(bool|None): 锁定状态过滤。
        - user_id(str|None): 用户 ID 过滤。
        - page(int|None): 页码（从 1 开始）。
        - page_size(int|None): 每页数量（文档约束 1-100）。

        输出:
        - 非分页模式: List[dict]，直接返回记忆对象数组。
        - 分页模式: dict，包含 items/total/page/page_size/total_pages。
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
        content: str,
        mws_score: Optional[float] = None,
        user_id: Optional[str] = None,
    ) -> JsonDict:
        """
        作用:
        更新指定记忆内容，并触发存储与向量索引更新。

        输入:
        - memory_id(str): 记忆 ID（路径参数，对应 {id}）。
        - content(str): 新的记忆内容或摘要。
        - mws_score(float|None): 文档标注为已弃用，客户端保留该参数仅为兼容旧调用，传入会被忽略。
        - user_id(str|None): 用户 ID（可选；用于多用户隔离）。

        输出:
        - dict: 通常包含 status 与 id。
        """
        data = self._request(
            "PUT",
            f"/api/memories/{memory_id}",
            params={"user_id": user_id},
            json_body={"content": content},
        )
        return data if isinstance(data, dict) else {}

    def delete_memory(self, memory_id: str, user_id: Optional[str] = None) -> JsonDict:
        """
        作用:
        删除指定记忆，并清理相关存储记录。

        输入:
        - memory_id(str): 记忆 ID（路径参数，对应 {id}）。
        - user_id(str|None): 用户 ID（可选；用于多用户隔离）。

        输出:
        - dict: 通常包含 status、message、id。
        """
        data = self._request("DELETE", f"/api/memories/{memory_id}", params={"user_id": user_id})
        return data if isinstance(data, dict) else {}

    def promote_memory(self, memory_id: str, user_id: Optional[str] = None) -> JsonDict:
        """
        作用:
        强制升级记忆（提高 MWS 概念分并锁定）。

        输入:
        - memory_id(str): 记忆 ID（路径参数，对应 {id}）。
        - user_id(str|None): 用户 ID（可选；用于多用户隔离）。

        输出:
        - dict: 通常包含 status、message。
        """
        data = self._request("POST", f"/api/memories/{memory_id}/promote", params={"user_id": user_id})
        return data if isinstance(data, dict) else {}

    def demote_memory(self, memory_id: str, user_id: Optional[str] = None) -> JsonDict:
        """
        作用:
        强制降级记忆（降低 MWS 概念分并解锁，同时取消里程碑标记）。

        输入:
        - memory_id(str): 记忆 ID（路径参数，对应 {id}）。
        - user_id(str|None): 用户 ID（可选；用于多用户隔离）。

        输出:
        - dict: 通常包含 status、message。
        """
        data = self._request("POST", f"/api/memories/{memory_id}/demote", params={"user_id": user_id})
        return data if isinstance(data, dict) else {}

    def toggle_memory_lock(
        self,
        memory_id: str,
        locked: bool,
        user_id: Optional[str] = None,
    ) -> JsonDict:
        """
        作用:
        设置记忆锁定状态，防止或允许被自动清理/迁移流程处理。

        输入:
        - memory_id(str): 记忆 ID（路径参数，对应 {id}）。
        - locked(bool): True=锁定，False=解锁。
        - user_id(str|None): 用户 ID（可选；用于多用户隔离）。

        输出:
        - dict: 通常包含 status、locked。
        """
        data = self._request(
            "POST",
            f"/api/memories/{memory_id}/toggle_lock",
            params={"user_id": user_id},
            json_body={"locked": locked},
        )
        return data if isinstance(data, dict) else {}

    def set_memory_milestone(
        self,
        memory_id: str,
        is_milestone: bool,
        user_id: Optional[str] = None,
    ) -> JsonDict:
        """
        作用:
        设置或取消记忆的里程碑标记。

        输入:
        - memory_id(str): 记忆 ID（路径参数，对应 {id}）。
        - is_milestone(bool): True=设为里程碑，False=取消里程碑。
        - user_id(str|None): 用户 ID（可选；用于多用户隔离）。

        输出:
        - dict: 通常包含 status、is_milestone。
        """
        data = self._request(
            "POST",
            f"/api/memories/{memory_id}/milestone",
            params={"user_id": user_id},
            json_body={"is_milestone": is_milestone},
        )
        return data if isinstance(data, dict) else {}

    def update_memory_settings(self, l2_threshold: float, user_id: Optional[str] = None) -> JsonDict:
        """
        作用:
        更新全局记忆阈值配置（V2 中为概念阈值更新，不触发物理迁移）。

        输入:
        - l2_threshold(float): 新的 L2 阈值，文档范围为 0.0-1.0。
        - user_id(str|None): 用户 ID（可选；用于多用户隔离）。

        输出:
        - dict: 通常包含 status、message。
        """
        data = self._request(
            "POST",
            "/api/settings/memory",
            params={"user_id": user_id},
            json_body={"l2_threshold": l2_threshold},
        )
        return data if isinstance(data, dict) else {}

    def get_cognitive_schemas(self, user_id: str) -> List[CognitiveSchema]:
        """
        作用:
        获取用户当前全部认知图式及其激活状态（Layer 2）。

        输入:
        - user_id(str): 目标用户 ID（必填，query 参数）。

        输出:
        - List[dict]: 图式对象数组。单个对象字段与文档一致：
          schema_id, name, activation_score, definition, associated_thoughts。
        """
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
        作用:
        执行完整认知工作流：图式检索 -> 内心独白 -> 条件反思更新。

        输入:
        - user_id(str): 用户 ID。
        - content(str): 当前输入文本。
        - history(list[dict]|None): 最近对话历史（可选），格式为
          [{"user_input": "...", "ai_response": "...", "active_schemas": [...]}]。

        输出:
        - dict: 返回字段与文档一致：
          inner_monologue(str),
          active_schemas(List[Dict]),
          reflection_result(Dict|null)。
        """
        data = self._request(
            "POST",
            "/api/cognitive/workflow",
            json_body={"user_id": user_id, "content": content, "history": history},
        )
        return data if isinstance(data, dict) else {}

    def interpret_cognitive(self, user_id: str, content: str) -> JsonDict:
        """
        作用:
        执行认知解读原子能力（仅图式检索 + 内心独白，不触发反思）。

        输入:
        - user_id(str): 用户 ID。
        - content(str): 当前输入文本。

        输出:
        - dict: 返回字段与文档一致：
          inner_monologue(str),
          active_schemas(List[Dict])。
        """
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
        作用:
        手动触发认知反思流程，根据历史记录更新图式分数。

        输入:
        - user_id(str): 用户 ID。
        - history(list[dict]): 待分析的对话历史记录（必填）。

        输出:
        - dict: 返回字段与文档一致：
          status("success"|"error"),
          insights(List[str]),
          updates(List[Dict])。
        """
        data = self._request(
            "POST",
            "/api/cognitive/reflect",
            json_body={"user_id": user_id, "history": history},
        )
        return data if isinstance(data, dict) else {}

    def health_check(self) -> JsonDict:
        """
        作用:
        执行服务健康检查。

        输入:
        - 无。

        输出:
        - dict: 典型包含 status("ok")、version("v2")。
        """
        data = self._request("GET", "/health")
        return data if isinstance(data, dict) else {}
