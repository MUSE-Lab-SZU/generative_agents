"""按存档批量查看外置记忆 user stats 的脚本。

使用方式：
1) 在脚本顶部修改 SAVE_NAME / BASE_URL。
2) 直接运行：python inspect_memory_user_stats_by_save.py
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
SAVE_NAME = "sim-test-0516"
BASE_URL = "http://localhost:8031"
# ===================================

CLIENT_TIMEOUT_SECONDS = 30.0

PROJECT_ROOT = Path(__file__).resolve().parent
CHECKPOINTS_ROOT = PROJECT_ROOT / "results" / "checkpoints"
AGENTS_ROOT = PROJECT_ROOT / "frontend" / "static" / "assets" / "village" / "agents"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="按存档批量查看外置记忆 user stats。")
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


def _fetch_one_stats(
    client: ECDollMemoryServiceClient,
    target: Dict[str, Any],
) -> Dict[str, Any]:
    scoped_user_id = str(target.get("scoped_user_id", "") or "")
    result: Dict[str, Any] = {
        "agent_name": str(target.get("agent_name", "") or ""),
        "base_user_id": str(target.get("base_user_id", "") or ""),
        "user_id": scoped_user_id,
        "status": "request_exception",
        "ok": False,
        "stats": {},
        "errors": [],
    }
    try:
        response = client.get_user_stats(user_id=scoped_user_id)
    except Exception as exc:
        result["errors"] = [str(exc)]
        return result

    if not isinstance(response, dict):
        result["status"] = "invalid_response"
        result["errors"] = ["response is not a JSON object"]
        return result

    result["status"] = "ok"
    result["ok"] = True
    result["stats"] = response
    stats_errors = response.get("errors", [])
    if isinstance(stats_errors, list):
        result["errors"] = [str(item) for item in stats_errors if str(item or "").strip()]
    return result


def _summarize(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "total": len(results),
        "ok_count": 0,
        "error_count": 0,
        "users_with_stats_errors": [],
        "users_with_embed_mock": [],
        "users_with_raw_fallback": [],
        "chroma_total": {
            "total": 0,
            "l1": 0,
            "l2": 0,
            "milestones": 0,
            "locked": 0,
            "raw_fallback": 0,
        },
        "postgres_total": {
            "l3_emotion_log": 0,
            "l4_milestones": 0,
            "l4_profile_attributes": 0,
            "l4_profile_core": 0,
        },
        "redis_total": {
            "short_term": 0,
            "l0_cache": 0,
        },
    }
    for item in results:
        user_id = str(item.get("user_id", "") or "")
        if bool(item.get("ok", False)):
            summary["ok_count"] = _as_int(summary["ok_count"]) + 1
        else:
            summary["error_count"] = _as_int(summary["error_count"]) + 1
            continue

        stats = item.get("stats", {})
        if not isinstance(stats, dict):
            continue

        chroma = stats.get("chroma", {})
        if isinstance(chroma, dict):
            for key in summary["chroma_total"]:
                summary["chroma_total"][key] = _as_int(summary["chroma_total"][key]) + _as_int(chroma.get(key, 0))
            if _as_int(chroma.get("raw_fallback", 0)) > 0:
                summary["users_with_raw_fallback"].append(
                    {
                        "user_id": user_id,
                        "raw_fallback": _as_int(chroma.get("raw_fallback", 0)),
                    }
                )

        postgres = stats.get("postgres", {})
        if isinstance(postgres, dict):
            for key in summary["postgres_total"]:
                summary["postgres_total"][key] = _as_int(summary["postgres_total"][key]) + _as_int(postgres.get(key, 0))

        redis_stats = stats.get("redis", {})
        if isinstance(redis_stats, dict):
            for key in summary["redis_total"]:
                summary["redis_total"][key] = _as_int(summary["redis_total"][key]) + _as_int(redis_stats.get(key, 0))

        flags = stats.get("flags", {})
        if isinstance(flags, dict) and bool(flags.get("embed_mock", False)):
            summary["users_with_embed_mock"].append(user_id)

        errors = item.get("errors", [])
        if isinstance(errors, list) and errors:
            summary["users_with_stats_errors"].append(
                {
                    "user_id": user_id,
                    "errors": [str(err) for err in errors],
                }
            )
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


def _print_results(results: List[Dict[str, Any]]) -> None:
    print("\n[INFO] 分角色 stats 摘要：")
    for item in results:
        stats = item.get("stats", {})
        chroma = stats.get("chroma", {}) if isinstance(stats, dict) else {}
        postgres = stats.get("postgres", {}) if isinstance(stats, dict) else {}
        redis_stats = stats.get("redis", {}) if isinstance(stats, dict) else {}
        flags = stats.get("flags", {}) if isinstance(stats, dict) else {}
        print(
            "- agent={agent} user_id={user_id} status={status} chroma(total={total}, l1={l1}, l2={l2}, milestones={milestones}, locked={locked}, raw_fallback={raw_fallback}) postgres(l3={l3}, l4_ms={l4_ms}, l4_attr={l4_attr}, l4_core={l4_core}) redis(short_term={short_term}, l0_cache={l0_cache}) embed_mock={embed_mock}".format(
                agent=item.get("agent_name", ""),
                user_id=item.get("user_id", ""),
                status=item.get("status", ""),
                total=_as_int(chroma.get("total", 0)) if isinstance(chroma, dict) else 0,
                l1=_as_int(chroma.get("l1", 0)) if isinstance(chroma, dict) else 0,
                l2=_as_int(chroma.get("l2", 0)) if isinstance(chroma, dict) else 0,
                milestones=_as_int(chroma.get("milestones", 0)) if isinstance(chroma, dict) else 0,
                locked=_as_int(chroma.get("locked", 0)) if isinstance(chroma, dict) else 0,
                raw_fallback=_as_int(chroma.get("raw_fallback", 0)) if isinstance(chroma, dict) else 0,
                l3=_as_int(postgres.get("l3_emotion_log", 0)) if isinstance(postgres, dict) else 0,
                l4_ms=_as_int(postgres.get("l4_milestones", 0)) if isinstance(postgres, dict) else 0,
                l4_attr=_as_int(postgres.get("l4_profile_attributes", 0)) if isinstance(postgres, dict) else 0,
                l4_core=_as_int(postgres.get("l4_profile_core", 0)) if isinstance(postgres, dict) else 0,
                short_term=_as_int(redis_stats.get("short_term", 0)) if isinstance(redis_stats, dict) else 0,
                l0_cache=_as_int(redis_stats.get("l0_cache", 0)) if isinstance(redis_stats, dict) else 0,
                embed_mock=bool(flags.get("embed_mock", False)) if isinstance(flags, dict) else False,
            )
        )
        errors = item.get("errors", [])
        if isinstance(errors, list) and errors:
            print("  errors={}".format(json.dumps(errors, ensure_ascii=False)))


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
    results = [_fetch_one_stats(client, item) for item in targets]
    summary = _summarize(results)

    report: Dict[str, Any] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "config": {
            "save_name": save_name,
            "base_url": BASE_URL,
            "role_source": role_source,
        },
        "targets": targets,
        "summary": summary,
        "results": results,
    }

    _print_results(results)
    print("\n[INFO] 汇总：")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    _write_report(str(args.report_file or ""), report)

    if _as_int(summary.get("error_count", 0)) > 0:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
