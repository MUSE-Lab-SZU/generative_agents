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
from pathlib import Path
from typing import Any, Dict, List, Tuple

from modules.ec_doll_memory_service_client import ECDollMemoryServiceClient

# ====== 顶部配置区（按需编辑） ======
SAVE_NAME = "sim-test-memory-0426"
BASE_URL = "http://localhost:8031"
EXECUTE_DELETE = False
# ===================================

CLIENT_TIMEOUT_SECONDS = 30.0

PROJECT_ROOT = Path(__file__).resolve().parent
CHECKPOINTS_ROOT = PROJECT_ROOT / "results" / "checkpoints"
AGENTS_ROOT = PROJECT_ROOT / "frontend" / "static" / "assets" / "village" / "agents"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="按存档批量清理外置记忆 user_id。")
    parser.add_argument(
        "--report-file",
        default="",
        help="可选。将执行结果写入 JSON 报告文件。",
    )
    return parser.parse_args()


def _load_json_file(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as file_obj:
        data = json.load(file_obj)
    return data if isinstance(data, dict) else {}


def _resolve_config_path(config_path_from_snapshot: str, agent_name: str) -> Path | None:
    path_text = str(config_path_from_snapshot or "").strip()
    candidates: List[Path] = []
    if path_text:
        raw_path = Path(path_text)
        if raw_path.is_absolute():
            candidates.append(raw_path)
        else:
            candidates.append(PROJECT_ROOT / raw_path)
            candidates.append(PROJECT_ROOT / "frontend" / "static" / raw_path)
    candidates.append(AGENTS_ROOT / str(agent_name).strip() / "agent.json")
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _discover_roles_by_save(save_name: str) -> Tuple[List[Dict[str, Any]], str]:
    roles: List[Dict[str, Any]] = []
    snapshot_dir = CHECKPOINTS_ROOT / save_name
    if snapshot_dir.is_dir():
        snapshot_files = sorted(snapshot_dir.glob("simulate-*.json"))
        if snapshot_files:
            latest_snapshot = snapshot_files[-1]
            try:
                snapshot_data = _load_json_file(latest_snapshot)
                agents = snapshot_data.get("agents", {})
                if isinstance(agents, dict):
                    for agent_name, meta in agents.items():
                        if not isinstance(meta, dict):
                            continue
                        config_path = _resolve_config_path(
                            str(meta.get("config_path", "") or ""),
                            str(agent_name or ""),
                        )
                        if config_path is None:
                            continue
                        roles.append(
                            {
                                "agent_name": str(agent_name),
                                "config_path": str(config_path),
                                "source": "snapshot",
                            }
                        )
            except Exception as exc:
                print("[WARN] 读取存档快照失败：{} error={}".format(latest_snapshot, exc))
            if roles:
                return roles, "snapshot"

    for agent_dir in sorted(AGENTS_ROOT.glob("*")):
        config_path = agent_dir / "agent.json"
        if not config_path.is_file():
            continue
        roles.append(
            {
                "agent_name": agent_dir.name,
                "config_path": str(config_path),
                "source": "fallback_all_agents",
            }
        )
    return roles, "fallback_all_agents"


def _read_base_user_id(config_path: Path) -> str:
    try:
        data = _load_json_file(config_path)
    except Exception:
        return ""
    external_memory = data.get("external_memory", {})
    if not isinstance(external_memory, dict):
        return ""
    return str(external_memory.get("user_id", "") or "").strip()


def _build_targets(save_name: str) -> Tuple[List[Dict[str, Any]], str]:
    roles, role_source = _discover_roles_by_save(save_name)
    targets: List[Dict[str, Any]] = []
    for role in roles:
        config_path = Path(str(role.get("config_path", "") or ""))
        base_user_id = _read_base_user_id(config_path)
        if not base_user_id:
            continue
        scoped_user_id = "{}+{}".format(base_user_id, save_name)
        targets.append(
            {
                "agent_name": str(role.get("agent_name", "") or ""),
                "base_user_id": base_user_id,
                "scoped_user_id": scoped_user_id,
                "config_path": str(config_path),
                "source": str(role.get("source", role_source) or role_source),
            }
        )
    return targets, role_source


def _as_int(value: Any) -> int:
    try:
        return int(value)
    except Exception:
        return 0


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


def _print_targets(targets: List[Dict[str, Any]]) -> None:
    print("\n[INFO] 本次目标 user_id 列表（scoped）:")
    for item in targets:
        print(
            "  - agent={} scoped_user_id={} config_path={}".format(
                item.get("agent_name", ""),
                item.get("scoped_user_id", ""),
                item.get("config_path", ""),
            )
        )


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


def _write_report(report_file: str, payload: Dict[str, Any]) -> None:
    if not report_file:
        return
    out_path = Path(report_file)
    if not out_path.is_absolute():
        out_path = PROJECT_ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, ensure_ascii=False, indent=2)
    print("[INFO] 报告已写入：{}".format(out_path))


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
