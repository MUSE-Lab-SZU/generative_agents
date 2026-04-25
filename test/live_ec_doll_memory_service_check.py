"""EC-Doll 记忆微服务真实联调检查脚本（直连真实服务，不使用 mock）。"""

from __future__ import annotations

import argparse
import os
import sys
import uuid
from typing import List, Optional, Tuple

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(CURRENT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from modules.ec_doll_memory_service_client import ECDollMemoryServiceClient


class LiveCheckRunner:
    def __init__(
        self,
        client: ECDollMemoryServiceClient,
        user_id: str,
        memory_id: Optional[str] = None,
        delete_memory_id: Optional[str] = None,
        run_write_checks: bool = False,
        run_delete_check: bool = False,
        run_settings_check: bool = False,
    ) -> None:
        self.client = client
        self.user_id = user_id
        self.memory_id = memory_id
        self.delete_memory_id = delete_memory_id
        self.run_write_checks = run_write_checks
        self.run_delete_check = run_delete_check
        self.run_settings_check = run_settings_check
        self.results: List[Tuple[str, str, str]] = []

    def _record(self, name: str, status: str, detail: str) -> None:
        self.results.append((name, status, detail))
        print(f"[{status}] {name}: {detail}")

    def _pass(self, name: str, detail: str) -> None:
        self._record(name, "PASS", detail)

    def _fail(self, name: str, detail: str) -> None:
        self._record(name, "FAIL", detail)

    def _skip(self, name: str, detail: str) -> None:
        self._record(name, "SKIP", detail)

    def _run_check(self, name: str, fn) -> None:
        try:
            detail = fn()
            self._pass(name, detail)
        except Exception as exc:  # noqa: BLE001
            self._fail(name, f"{type(exc).__name__}: {exc}")

    def run(self) -> int:
        print(f"Base URL: {self.client.base_url}")
        print(f"User ID : {self.user_id}")
        print("-" * 80)

        self._run_check("GET /health", self.check_health)
        self._run_check("GET /ready", self.check_ready)
        self._run_check("POST /retrieve", self.check_retrieve)
        self._run_check("GET /api/milestones", self.check_milestones)
        self._run_check("GET /api/memories (no pagination)", self.check_memories_no_paging)
        self._run_check("GET /api/memories (pagination)", self.check_memories_paging)

        if self.run_write_checks:
            self._run_check("POST /ingest", self.check_ingest)
        else:
            self._skip("POST /ingest", "未启用 --run-write-checks")

        if self.run_write_checks and self.memory_id:
            self._run_check("PUT /api/memories/{id}", self.check_update_memory)
            self._run_check("POST /api/memories/{id}/promote", self.check_promote_memory)
            self._run_check("POST /api/memories/{id}/demote", self.check_demote_memory)
            self._run_check("POST /api/memories/{id}/toggle_lock", self.check_toggle_lock)
            self._run_check("POST /api/memories/{id}/milestone", self.check_set_milestone)
        elif self.run_write_checks and not self.memory_id:
            self._skip(
                "memory mutation endpoints",
                "已启用 --run-write-checks，但未提供 --memory-id，跳过 update/promote/demote/toggle/milestone",
            )
        else:
            self._skip("memory mutation endpoints", "未启用 --run-write-checks")

        if self.run_settings_check:
            self._run_check("POST /api/settings/memory", self.check_settings)
        else:
            self._skip("POST /api/settings/memory", "未启用 --run-settings-check")

        if self.run_delete_check and self.delete_memory_id:
            self._run_check("DELETE /api/memories/{id}", self.check_delete_memory)
        elif self.run_delete_check and not self.delete_memory_id:
            self._skip(
                "DELETE /api/memories/{id}",
                "已启用 --run-delete-check，但未提供 --delete-memory-id",
            )
        else:
            self._skip("DELETE /api/memories/{id}", "未启用 --run-delete-check")

        print("-" * 80)
        pass_count = sum(1 for _, status, _ in self.results if status == "PASS")
        fail_count = sum(1 for _, status, _ in self.results if status == "FAIL")
        skip_count = sum(1 for _, status, _ in self.results if status == "SKIP")
        print(f"Summary: PASS={pass_count}, FAIL={fail_count}, SKIP={skip_count}")

        return 1 if fail_count > 0 else 0

    def check_health(self) -> str:
        result = self.client.health_check()
        if not isinstance(result, dict):
            raise ValueError(f"health 返回类型异常: {type(result)}")
        if "status" not in result:
            raise ValueError(f"health 缺少 status 字段: {result}")
        return f"status={result.get('status')}, version={result.get('version')}"

    def check_ready(self) -> str:
        result = self.client.ready_check()
        if not isinstance(result, dict):
            raise ValueError(f"ready 返回类型异常: {type(result)}")
        for key in ("status", "redis", "postgres", "chroma"):
            if key not in result:
                raise ValueError(f"ready 缺少字段 {key}: {result}")
        return (
            f"status={result.get('status')}, redis={result.get('redis')}, "
            f"postgres={result.get('postgres')}, chroma={result.get('chroma')}"
        )

    def check_ingest(self) -> str:
        token = uuid.uuid4().hex[:8]
        result = self.client.ingest_memory(
            content=f"[live-check-{token}] EC-Doll ingest 可用性联调",
            user_id=self.user_id,
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
            if key not in result:
                raise ValueError(f"ingest 缺少字段 {key}: {result}")
        return f"status={result.get('status')}, remote_id={result.get('remote_id')}"

    def check_retrieve(self) -> str:
        result = self.client.retrieve_memory_context(
            query="这是服务连通性测试，请返回上下文。",
            user_id=self.user_id,
            active_project="live_service_probe",
        )
        if not isinstance(result, dict):
            raise ValueError(f"retrieve 返回类型异常: {type(result)}")
        if "formatted_prompt" not in result:
            raise ValueError(f"retrieve 缺少 formatted_prompt: {result}")
        details = result.get("details")
        if not isinstance(details, dict):
            raise ValueError(f"retrieve.details 类型异常: {type(details)}")
        for key in ("l0_history", "ranked_memories", "profile_data", "recent_emotion", "milestones", "short_term_recent"):
            if key not in details:
                raise ValueError(f"retrieve.details 缺少字段 {key}: {details}")
        return f"formatted_prompt_len={len(result.get('formatted_prompt', ''))}"

    def check_milestones(self) -> str:
        data = self.client.list_milestones(user_id=self.user_id)
        if not isinstance(data, list):
            raise ValueError(f"milestones 返回类型异常: {type(data)}")
        return f"items={len(data)}"

    def check_memories_no_paging(self) -> str:
        data = self.client.list_memories(user_id=self.user_id)
        if not isinstance(data, list):
            raise ValueError(f"memories(无分页) 返回类型异常: {type(data)}")
        return f"items={len(data)}"

    def check_memories_paging(self) -> str:
        data = self.client.list_memories(user_id=self.user_id, page=1, page_size=5)
        if not isinstance(data, dict):
            raise ValueError(f"memories(分页) 返回类型异常: {type(data)}")
        for key in ("items", "total", "page", "page_size"):
            if key not in data:
                raise ValueError(f"分页结果缺少字段 {key}: {data}")
        return f"page={data['page']}, page_size={data['page_size']}, total={data['total']}"

    def check_update_memory(self) -> str:
        assert self.memory_id is not None
        token = uuid.uuid4().hex[:6]
        result = self.client.update_memory(
            memory_id=self.memory_id,
            content=f"[live-check-update-{token}] 接口联调更新内容",
        )
        if result.get("status") != "success":
            raise ValueError(f"update_memory 失败: {result}")
        return f"status={result.get('status')}, id={result.get('id')}"

    def check_delete_memory(self) -> str:
        assert self.delete_memory_id is not None
        result = self.client.delete_memory(memory_id=self.delete_memory_id)
        if result.get("status") != "success":
            raise ValueError(f"delete_memory 失败: {result}")
        return f"status={result.get('status')}, id={result.get('id')}"

    def check_promote_memory(self) -> str:
        assert self.memory_id is not None
        result = self.client.promote_memory(memory_id=self.memory_id)
        if result.get("status") != "success":
            raise ValueError(f"promote_memory 失败: {result}")
        return f"message={result.get('message')}"

    def check_demote_memory(self) -> str:
        assert self.memory_id is not None
        result = self.client.demote_memory(memory_id=self.memory_id)
        if result.get("status") != "success":
            raise ValueError(f"demote_memory 失败: {result}")
        return f"message={result.get('message')}"

    def check_toggle_lock(self) -> str:
        assert self.memory_id is not None
        result = self.client.toggle_memory_lock(memory_id=self.memory_id)
        if result.get("status") != "success":
            raise ValueError(f"toggle_lock 失败: {result}")
        if "locked" not in result:
            raise ValueError(f"toggle_lock 缺少 locked 字段: {result}")
        return f"locked={result.get('locked')}"

    def check_set_milestone(self) -> str:
        assert self.memory_id is not None
        result = self.client.set_memory_milestone(memory_id=self.memory_id, is_milestone=True)
        if result.get("status") != "success":
            raise ValueError(f"set_milestone(is_milestone=True) 失败: {result}")
        if result.get("is_milestone") is not True:
            raise ValueError(f"set_milestone 返回 is_milestone 非 True: {result}")
        return "is_milestone=True"

    def check_settings(self) -> str:
        result = self.client.update_memory_settings(l2_threshold=0.8)
        if result.get("status") != "success":
            raise ValueError(f"update_memory_settings 失败: {result}")
        return f"message={result.get('message')}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EC-Doll 记忆微服务真实联调检查")
    parser.add_argument(
        "--base-url",
        default="http://localhost:8031",
        help="服务地址，默认 http://localhost:8031",
    )
    parser.add_argument(
        "--user-id",
        required=True,
        help="用于检索与查询类接口的 user_id",
    )
    parser.add_argument(
        "--memory-id",
        default=None,
        help="用于 update/promote/demote/toggle/milestone 的目标记忆 ID",
    )
    parser.add_argument(
        "--delete-memory-id",
        default=None,
        help="用于 delete 接口的目标记忆 ID（高风险）",
    )
    parser.add_argument(
        "--run-write-checks",
        action="store_true",
        help="启用写接口检查（ingest + memory mutation）",
    )
    parser.add_argument(
        "--run-delete-check",
        action="store_true",
        help="启用 delete 接口检查（必须同时传 --delete-memory-id）",
    )
    parser.add_argument(
        "--run-settings-check",
        action="store_true",
        help="启用 settings 接口检查（会修改 l2_threshold=0.8）",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=15.0,
        help="HTTP 超时时间（秒），默认 15",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    client = ECDollMemoryServiceClient(base_url=args.base_url, timeout=args.timeout)
    runner = LiveCheckRunner(
        client=client,
        user_id=args.user_id,
        memory_id=args.memory_id,
        delete_memory_id=args.delete_memory_id,
        run_write_checks=args.run_write_checks,
        run_delete_check=args.run_delete_check,
        run_settings_check=args.run_settings_check,
    )
    return runner.run()


if __name__ == "__main__":
    raise SystemExit(main())
