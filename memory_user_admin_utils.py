"""Shared helpers for save-scoped external-memory administration CLIs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


PROJECT_ROOT = Path(__file__).resolve().parent
CHECKPOINTS_ROOT = PROJECT_ROOT / "results" / "checkpoints"
AGENTS_ROOT = PROJECT_ROOT / "frontend" / "static" / "assets" / "village" / "agents"


def load_json_file(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as file_obj:
        data = json.load(file_obj)
    return data if isinstance(data, dict) else {}


def resolve_config_path(config_path_from_snapshot: str, agent_name: str) -> Path | None:
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


def discover_roles_by_save(save_name: str) -> Tuple[List[Dict[str, Any]], str]:
    roles: List[Dict[str, Any]] = []
    snapshot_dir = CHECKPOINTS_ROOT / save_name
    if snapshot_dir.is_dir():
        snapshot_files = sorted(snapshot_dir.glob("simulate-*.json"))
        if snapshot_files:
            latest_snapshot = snapshot_files[-1]
            try:
                snapshot_data = load_json_file(latest_snapshot)
                agents = snapshot_data.get("agents", {})
                if isinstance(agents, dict):
                    for agent_name, meta in agents.items():
                        if not isinstance(meta, dict):
                            continue
                        config_path = resolve_config_path(
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


def read_base_user_id(config_path: Path) -> str:
    try:
        data = load_json_file(config_path)
    except Exception:
        return ""
    external_memory = data.get("external_memory", {})
    if not isinstance(external_memory, dict):
        return ""
    return str(external_memory.get("user_id", "") or "").strip()


def build_targets(save_name: str) -> Tuple[List[Dict[str, Any]], str]:
    roles, role_source = discover_roles_by_save(save_name)
    targets: List[Dict[str, Any]] = []
    for role in roles:
        config_path = Path(str(role.get("config_path", "") or ""))
        base_user_id = read_base_user_id(config_path)
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


def as_int(value: Any) -> int:
    try:
        return int(value)
    except Exception:
        return 0


def print_targets(targets: List[Dict[str, Any]]) -> None:
    print("\n[INFO] 本次目标 user_id 列表（scoped）:")
    for item in targets:
        print(
            "  - agent={} scoped_user_id={} config_path={}".format(
                item.get("agent_name", ""),
                item.get("scoped_user_id", ""),
                item.get("config_path", ""),
            )
        )


def write_report(report_file: str, payload: Dict[str, Any]) -> None:
    if not report_file:
        return
    out_path = Path(report_file)
    if not out_path.is_absolute():
        out_path = PROJECT_ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, ensure_ascii=False, indent=2)
    print("[INFO] 报告已写入：{}".format(out_path))
