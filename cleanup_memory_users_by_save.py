"""按存档批量清理外置记忆 user_id 的执行脚本。

使用方式：
1) 在脚本顶部修改 SAVE_NAME / BASE_URL / EXECUTE_DELETE。
2) 直接运行：python cleanup_memory_users_by_save.py
3) 可选参数：--report-file <path> 保存执行报告 JSON。
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from typing import Any, Dict, List

from memory_user_admin_utils import (
    as_int as _as_int,
    build_targets as _build_targets,
    discover_roles_by_save as _discover_roles_by_save,
    load_json_file as _load_json_file,
    print_targets as _print_targets,
    read_base_user_id as _read_base_user_id,
    resolve_config_path as _resolve_config_path,
    write_report as _write_report,
)
from modules.ec_doll_memory_service_client import ECDollMemoryServiceClient

# ====== 顶部配置区（按需编辑） ======
SAVE_NAME = "sim-trace-test-0526"
BASE_URL = "http://localhost:8031"
EXECUTE_DELETE = True
# ===================================

CLIENT_TIMEOUT_SECONDS = 30.0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="按存档批量清理外置记忆 user_id。")
    parser.add_argument(
        "--report-file",
        default="",
        help="可选。将执行结果写入 JSON 报告文件。",
    )
    return parser.parse_args()


def _delete_one(
    client: ECDollMemoryServiceClient,
    scoped_user_id: str,
    dry_run: bool,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "user_id": scoped_user_id,
        "dry_run": bool(dry_run),
        "status": "request_exception",
        "errors": [],
        "ok": False,
    }
    try:
        response = client.delete_user_data(user_id=scoped_user_id, dry_run=dry_run)
    except Exception as exc:
        result["errors"] = [str(exc)]
        return result

    if not isinstance(response, dict):
        result["status"] = "invalid_response"
        result["errors"] = ["response is not a JSON object"]
        return result

    result.update(response)
    status = str(response.get("status", "") or "")
    if dry_run:
        result["ok"] = status in {"dry_run", "ok"}
    else:
        result["ok"] = status == "ok"
    return result


def _summarize(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "total": len(results),
        "ok_count": 0,
        "partial_count": 0,
        "error_count": 0,
        "status_counter": {},
        "deleted_chroma_total": 0,
        "deleted_pg_total": {
            "l3_emotion_log": 0,
            "l4_milestones": 0,
            "l4_profile_attributes": 0,
            "l4_profile_core": 0,
        },
        "deleted_redis_hit_count": {
            "short_term": 0,
            "l0_cache": 0,
        },
        "failed_user_ids": [],
    }
    for item in results:
        status = str(item.get("status", "") or "")
        status_counter = summary["status_counter"]
        status_counter[status] = _as_int(status_counter.get(status, 0)) + 1

        if bool(item.get("ok", False)):
            summary["ok_count"] = _as_int(summary["ok_count"]) + 1
        elif status == "partial":
            summary["partial_count"] = _as_int(summary["partial_count"]) + 1
            summary["failed_user_ids"].append(str(item.get("user_id", "") or ""))
        else:
            summary["error_count"] = _as_int(summary["error_count"]) + 1
            summary["failed_user_ids"].append(str(item.get("user_id", "") or ""))

        summary["deleted_chroma_total"] = _as_int(summary["deleted_chroma_total"]) + _as_int(
            item.get("deleted_chroma", 0)
        )

        deleted_pg = item.get("deleted_pg", {})
        if isinstance(deleted_pg, dict):
            for key in summary["deleted_pg_total"]:
                summary["deleted_pg_total"][key] = _as_int(summary["deleted_pg_total"][key]) + _as_int(
                    deleted_pg.get(key, 0)
                )

        deleted_redis = item.get("deleted_redis", {})
        if isinstance(deleted_redis, dict):
            for key in summary["deleted_redis_hit_count"]:
                if bool(deleted_redis.get(key, False)):
                    summary["deleted_redis_hit_count"][key] = _as_int(summary["deleted_redis_hit_count"][key]) + 1

    return summary


def _confirm_stage_1() -> bool:
    try:
        text = input("\n第一次确认：输入 YES 继续真删流程，否则取消：").strip()
    except EOFError:
        return False
    return text == "YES"


def _confirm_stage_2(save_name: str) -> bool:
    expected = "DELETE {}".format(save_name)
    prompt = "\n第二次确认：输入 `{}` 才会执行删除，否则取消：".format(expected)
    try:
        text = input(prompt).strip()
    except EOFError:
        return False
    return text == expected


def main() -> int:
    args = _parse_args()
    save_name = str(SAVE_NAME or "").strip()
    if not save_name:
        print("[ERROR] SAVE_NAME 为空，请先在脚本顶部配置。")
        return 1

    targets, role_source = _build_targets(save_name)
    if not targets:
        print("[ERROR] 未找到可用目标，请检查 SAVE_NAME 或角色配置。")
        return 1

    _print_targets(targets)
    client = ECDollMemoryServiceClient(
        base_url=BASE_URL,
        timeout=CLIENT_TIMEOUT_SECONDS,
    )

    preview_results = [
        _delete_one(client, str(item["scoped_user_id"]), dry_run=True)
        for item in targets
    ]
    preview_summary = _summarize(preview_results)

    report: Dict[str, Any] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "config": {
            "save_name": save_name,
            "base_url": BASE_URL,
            "execute_delete": bool(EXECUTE_DELETE),
            "role_source": role_source,
        },
        "targets": targets,
        "preview": {
            "summary": preview_summary,
            "results": preview_results,
        },
        "execute": None,
    }

    print("\n[INFO] dry-run 汇总：")
    print(json.dumps(preview_summary, ensure_ascii=False, indent=2))

    if not EXECUTE_DELETE:
        _write_report(str(args.report_file or ""), report)
        print("\n[DONE] EXECUTE_DELETE=False，仅完成 dry-run 预览。")
        return 0

    print("\n[WARN] 即将进入不可恢复删除流程。")
    if not _confirm_stage_1():
        print("[CANCELLED] 第一次确认未通过，已取消删除。")
        _write_report(str(args.report_file or ""), report)
        return 0

    _print_targets(targets)
    if not _confirm_stage_2(save_name):
        print("[CANCELLED] 第二次确认未通过，已取消删除。")
        _write_report(str(args.report_file or ""), report)
        return 0

    execute_results = [
        _delete_one(client, str(item["scoped_user_id"]), dry_run=False)
        for item in targets
    ]
    execute_summary = _summarize(execute_results)
    report["execute"] = {
        "summary": execute_summary,
        "results": execute_results,
    }

    print("\n[INFO] 真删汇总：")
    print(json.dumps(execute_summary, ensure_ascii=False, indent=2))
    _write_report(str(args.report_file or ""), report)

    if _as_int(execute_summary.get("partial_count", 0)) > 0 or _as_int(execute_summary.get("error_count", 0)) > 0:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
