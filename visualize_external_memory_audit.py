#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
External memory audit and visualization script.

How to use:
1) Edit the configuration block at the top of this file, then run: python visualize_external_memory_audit.py
2) Or override via CLI, for example:
   python visualize_external_memory_audit.py --cp-name sim-test-0513 --agent 卡布达
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from modules.ec_doll_memory_service_client import ECDollMemoryServiceClient


# =========================
# Configuration (edit here)
# =========================
CHECKPOINTS_ROOT = CURRENT_DIR / "results" / "checkpoints"
OUTPUT_ROOT = CURRENT_DIR / "results" / "external_memory_audit"

# Optional filters:
# [] means "all saves"/"all agents"
TARGET_SAVE_NAMES: List[str] = ["sim-test-0515-2"]
TARGET_AGENT_NAMES: List[str] = ["卡布达"]

MEMORY_SERVICE_BASE_URL = "http://localhost:8031"
MEMORY_SERVICE_TIMEOUT_SECONDS = 15.0
MEMORIES_PAGE_SIZE = 200
RETRIEVE_QUERY = "请给出最近相关记忆上下文"

INCLUDE_LOCAL_SUPPLEMENT = True
INCLUDE_TIMELINE_REPLAY = True
INCLUDE_HEALTH_CHECKS = True

WRITE_JSON = True
WRITE_CSV = True
WRITE_HTML = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="审计并可视化外置记忆")
    parser.add_argument("--cp-name", dest="save_names", action="append", default=None, help="实验名，可重复传入多次")
    parser.add_argument("--agent", dest="agent_names", action="append", default=None, help="角色名，可重复传入多次")
    parser.add_argument("--checkpoints-root", default=None, help="checkpoint 根目录")
    parser.add_argument("--output-root", default=None, help="输出根目录")
    parser.add_argument("--service-base-url", default=None, help="外置记忆服务地址")
    parser.add_argument("--service-timeout", type=float, default=None, help="外置记忆服务超时秒数")
    parser.add_argument("--retrieve-query", default=None, help="retrieve 接口使用的查询文本")
    parser.add_argument("--no-local-supplement", action="store_true", help="跳过本地补充信息")
    parser.add_argument("--no-timeline", action="store_true", help="跳过日志时间线回放")
    parser.add_argument("--no-health-checks", action="store_true", help="跳过 health/ready 检查")
    return parser.parse_args()


def _apply_cli_overrides(args: argparse.Namespace) -> None:
    global CHECKPOINTS_ROOT
    global OUTPUT_ROOT
    global TARGET_SAVE_NAMES
    global TARGET_AGENT_NAMES
    global MEMORY_SERVICE_BASE_URL
    global MEMORY_SERVICE_TIMEOUT_SECONDS
    global RETRIEVE_QUERY
    global INCLUDE_LOCAL_SUPPLEMENT
    global INCLUDE_TIMELINE_REPLAY
    global INCLUDE_HEALTH_CHECKS

    if args.checkpoints_root:
        checkpoints_root = Path(args.checkpoints_root)
        if not checkpoints_root.is_absolute():
            checkpoints_root = checkpoints_root.resolve()
        CHECKPOINTS_ROOT = checkpoints_root
    if args.output_root:
        output_root = Path(args.output_root)
        if not output_root.is_absolute():
            output_root = output_root.resolve()
        OUTPUT_ROOT = output_root
    if args.save_names is not None:
        TARGET_SAVE_NAMES = [name for name in args.save_names if name]
    if args.agent_names is not None:
        TARGET_AGENT_NAMES = [name for name in args.agent_names if name]
    if args.service_base_url is not None:
        MEMORY_SERVICE_BASE_URL = args.service_base_url
    if args.service_timeout is not None and args.service_timeout > 0:
        MEMORY_SERVICE_TIMEOUT_SECONDS = args.service_timeout
    if args.retrieve_query is not None:
        RETRIEVE_QUERY = args.retrieve_query
    if args.no_local_supplement:
        INCLUDE_LOCAL_SUPPLEMENT = False
    if args.no_timeline:
        INCLUDE_TIMELINE_REPLAY = False
    if args.no_health_checks:
        INCLUDE_HEALTH_CHECKS = False


@dataclass
class AgentTarget:
    save_name: str
    save_dir: Path
    snapshot_path: Path
    agent_name: str
    scoped_user_id: str
    user_id: str
    map_path: Path
    log_path: Path


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    errors = []
    for enc in ("utf-8", "utf-8-sig"):
        try:
            obj = json.loads(path.read_text(encoding=enc))
            return obj if isinstance(obj, dict) else {"_payload": obj}
        except Exception as exc:
            errors.append(f"{enc}: {exc}")
    raise ValueError(f"Failed to parse JSON: {path}\n" + "\n".join(errors))


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        for key in row.keys():
            if key in seen:
                continue
            seen.add(key)
            fieldnames.append(str(key))
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = {}
            for k, v in row.items():
                if isinstance(v, (dict, list)):
                    out[k] = json.dumps(v, ensure_ascii=False)
                else:
                    out[k] = v
            writer.writerow(out)


def _safe_filename(name: str) -> str:
    return re.sub(r'[\\/:*?"<>|]+', "_", str(name or "")).strip() or "unknown"


def _parse_time(raw: Any) -> Optional[datetime]:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    fmts = [
        "%Y%m%d-%H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%S.%f",
    ]
    for fmt in fmts:
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def _fmt_time(raw: Any) -> str:
    dt = _parse_time(raw)
    if dt is None:
        return str(raw or "")
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except Exception:
        return 0


def _resolve_saves() -> List[Path]:
    if TARGET_SAVE_NAMES:
        paths = [CHECKPOINTS_ROOT / name for name in TARGET_SAVE_NAMES]
        return [p for p in paths if p.is_dir()]
    if not CHECKPOINTS_ROOT.exists():
        return []
    return sorted([p for p in CHECKPOINTS_ROOT.iterdir() if p.is_dir()])


def _latest_snapshot(save_dir: Path) -> Optional[Path]:
    snapshots = sorted(save_dir.glob("simulate-*.json"))
    if not snapshots:
        return None
    return snapshots[-1]


def _resolve_agent_config_path(raw_path: str) -> Optional[Path]:
    if not raw_path:
        return None
    p = Path(raw_path)
    candidates = [
        CURRENT_DIR / p,
        CURRENT_DIR / "frontend" / "static" / p,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _resolve_user_id(snapshot: Dict[str, Any], agent_name: str) -> str:
    agent_payload = (snapshot.get("agents", {}) or {}).get(agent_name, {}) or {}
    raw_config_path = str(agent_payload.get("config_path", "") or "")
    cfg_path = _resolve_agent_config_path(raw_config_path)
    if cfg_path and cfg_path.exists():
        try:
            agent_cfg = _load_json(cfg_path)
            ext = agent_cfg.get("external_memory", {})
            if isinstance(ext, dict):
                uid = str(ext.get("user_id", "") or "").strip()
                if uid:
                    return uid
        except Exception:
            pass
    try:
        global_cfg = _load_json(CURRENT_DIR / "data" / "config.json")
        uid = str((((global_cfg.get("agent", {}) or {}).get("external_memory", {}) or {}).get("user_id", "") or "").strip())
        if uid:
            return uid
    except Exception:
        pass
    return ""


def _scoped_user_id(user_id: str, save_name: str) -> str:
    base = str(user_id or "").strip()
    save = str(save_name or "").strip()
    if not base:
        return ""
    if not save:
        return base
    return f"{base}+{save}"


def _collect_targets() -> List[AgentTarget]:
    out: List[AgentTarget] = []
    for save_dir in _resolve_saves():
        save_name = save_dir.name
        snapshot_path = _latest_snapshot(save_dir)
        if not snapshot_path:
            continue
        snapshot = _load_json(snapshot_path)
        all_agents = sorted(list((snapshot.get("agents", {}) or {}).keys()))
        if TARGET_AGENT_NAMES:
            agent_names = [name for name in TARGET_AGENT_NAMES if name in all_agents]
        else:
            agent_names = all_agents
        for agent_name in agent_names:
            user_id = _resolve_user_id(snapshot=snapshot, agent_name=agent_name)
            scoped = _scoped_user_id(user_id=user_id, save_name=save_name)
            if not scoped:
                continue
            map_path = save_dir / "storage" / agent_name / "associate" / "external_memory_node_map.json"
            log_path = save_dir / f"{save_name}.log"
            out.append(
                AgentTarget(
                    save_name=save_name,
                    save_dir=save_dir,
                    snapshot_path=snapshot_path,
                    agent_name=agent_name,
                    scoped_user_id=scoped,
                    user_id=user_id,
                    map_path=map_path,
                    log_path=log_path,
                )
            )
    return out


def _health_bundle(client: ECDollMemoryServiceClient) -> Dict[str, Any]:
    data: Dict[str, Any] = {}
    if not INCLUDE_HEALTH_CHECKS:
        return data
    try:
        data["health"] = client.health_check()
    except Exception as exc:
        data["health_error"] = str(exc)
    try:
        data["ready"] = client.ready_check()
    except Exception as exc:
        data["ready_error"] = str(exc)
    return data


def _fetch_all_memories(client: ECDollMemoryServiceClient, scoped_user_id: str) -> List[Dict[str, Any]]:
    memories: List[Dict[str, Any]] = []
    page = 1
    while True:
        data = client.list_memories(
            user_id=scoped_user_id,
            page=page,
            page_size=MEMORIES_PAGE_SIZE,
        )
        if isinstance(data, list):
            memories.extend([x for x in data if isinstance(x, dict)])
            break
        if not isinstance(data, dict):
            break
        items = data.get("items", [])
        if not isinstance(items, list):
            break
        page_items = [x for x in items if isinstance(x, dict)]
        memories.extend(page_items)
        total = data.get("total")
        if isinstance(total, int) and len(memories) >= total:
            break
        if len(page_items) < MEMORYS_PAGE_SIZE_SAFE():
            break
        page += 1
    return memories


def MEMORYS_PAGE_SIZE_SAFE() -> int:
    # keep one place to guard non-positive values
    if isinstance(MEMORIES_PAGE_SIZE, int) and MEMORIES_PAGE_SIZE > 0:
        return MEMORIES_PAGE_SIZE
    return 200


def _fetch_milestones(client: ECDollMemoryServiceClient, scoped_user_id: str) -> List[Dict[str, Any]]:
    data = client.list_milestones(user_id=scoped_user_id)
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    return []


def _fetch_retrieve_details(client: ECDollMemoryServiceClient, scoped_user_id: str) -> Dict[str, Any]:
    try:
        data = client.retrieve_memory_context(query=RETRIEVE_QUERY, user_id=scoped_user_id)
    except Exception as exc:
        return {"error": str(exc)}
    if not isinstance(data, dict):
        return {}
    details = data.get("details", {})
    if not isinstance(details, dict):
        details = {}
    return {"formatted_prompt": str(data.get("formatted_prompt", "") or ""), "details": details}


def _fetch_user_stats(client: ECDollMemoryServiceClient, scoped_user_id: str) -> Dict[str, Any]:
    try:
        data = client.get_user_stats(user_id=scoped_user_id)
    except Exception as exc:
        return {"error": str(exc)}
    return data if isinstance(data, dict) else {}


def _load_map(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"node_to_remote": {}, "remote_to_node": {}, "updated_at": "", "missing": True}
    data = _load_json(path)
    node_to_remote = data.get("node_to_remote", {})
    remote_to_node = data.get("remote_to_node", {})
    if not isinstance(node_to_remote, dict):
        node_to_remote = {}
    if not isinstance(remote_to_node, dict):
        remote_to_node = {}
    return {
        "node_to_remote": node_to_remote,
        "remote_to_node": remote_to_node,
        "updated_at": str(data.get("updated_at", "") or ""),
        "missing": False,
    }


def _load_local_supplement(target: AgentTarget) -> Dict[str, Any]:
    if not INCLUDE_LOCAL_SUPPLEMENT:
        return {}
    assoc_dir = target.save_dir / "storage" / target.agent_name / "associate"
    out: Dict[str, Any] = {"associate_dir": str(assoc_dir)}
    for filename in ("docstore.json", "index_store.json", "graph_store.json"):
        path = assoc_dir / filename
        if path.exists():
            try:
                raw = _load_json(path)
                out[filename] = {
                    "path": str(path),
                    "size_bytes": path.stat().st_size,
                    "top_keys": list(raw.keys())[:20],
                }
            except Exception as exc:
                out[filename] = {"path": str(path), "error": str(exc)}
        else:
            out[filename] = {"path": str(path), "missing": True}
    return out


def _memory_remote_id(item: Dict[str, Any]) -> str:
    remote_id = str(item.get("remote_id", "") or "").strip()
    if remote_id:
        return remote_id
    return str(item.get("id", "") or "").strip()


def _memory_level(item: Dict[str, Any]) -> str:
    return str(item.get("level", "") or "").upper()


def _build_relations(
    memories: List[Dict[str, Any]],
    map_data: Dict[str, Any],
    retrieve_data: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    node_to_remote = map_data.get("node_to_remote", {}) or {}
    remote_to_nodes = defaultdict(list)
    for node_id, rel in node_to_remote.items():
        if not isinstance(rel, dict):
            continue
        rid = str(rel.get("remote_id", "") or "").strip()
        if rid:
            remote_to_nodes[rid].append(str(node_id))

    memory_by_remote: Dict[str, Dict[str, Any]] = {}
    for item in memories:
        rid = _memory_remote_id(item)
        if rid:
            memory_by_remote[rid] = item

    details = retrieve_data.get("details", {}) if isinstance(retrieve_data, dict) else {}
    if not isinstance(details, dict):
        details = {}
    short_term_recent = details.get("short_term_recent", [])
    if not isinstance(short_term_recent, list):
        short_term_recent = []
    short_term_by_remote: Dict[str, Dict[str, Any]] = {}
    for item in short_term_recent:
        if not isinstance(item, dict):
            continue
        rid = str(item.get("remote_id", "") or "").strip()
        if rid:
            short_term_by_remote[rid] = item

    rows: List[Dict[str, Any]] = []
    for node_id, rel in node_to_remote.items():
        if not isinstance(rel, dict):
            continue
        remote_id = str(rel.get("remote_id", "") or "").strip()
        mem = memory_by_remote.get(remote_id, {})
        short_term = short_term_by_remote.get(remote_id, {})
        conflict = len(remote_to_nodes.get(remote_id, [])) > 1
        if conflict:
            status = "conflict"
        elif remote_id in memory_by_remote:
            status = "matched"
        else:
            status = "node_only"
        rows.append(
            {
                "node_id": str(node_id),
                "remote_id": remote_id,
                "node_type": str(rel.get("node_type", "") or ""),
                "create_time": str(rel.get("create_time", "") or ""),
                "memory_level": _memory_level(mem) if mem else "",
                "memory_timestamp": str(mem.get("timestamp", "") or ""),
                "memory_content": str(mem.get("content", "") or ""),
                "short_term_timestamp": str(short_term.get("ts", "") or ""),
                "short_term_content": str(short_term.get("content", "") or ""),
                "is_locked": bool(mem.get("is_locked")) if mem else False,
                "is_milestone": bool(mem.get("is_milestone")) if mem else False,
                "embed_mock_at_write": _memory_embed_mock_at_write(mem) if mem else False,
                "ingest_mode": _memory_ingest_mode(mem) if mem else "",
                "fallback_reason": _memory_fallback_reason(mem) if mem else "",
                "write_path_status": _memory_write_path_status(mem) if mem else "",
                "relation_status": status,
            }
        )

    map_remote_ids = set([str(r.get("remote_id", "") or "").strip() for r in node_to_remote.values() if isinstance(r, dict)])
    for remote_id, mem in memory_by_remote.items():
        if remote_id in map_remote_ids:
            continue
        short_term = short_term_by_remote.get(remote_id, {})
        rows.append(
            {
                "node_id": "",
                "remote_id": remote_id,
                "node_type": "",
                "create_time": "",
                "memory_level": _memory_level(mem),
                "memory_timestamp": str(mem.get("timestamp", "") or ""),
                "memory_content": str(mem.get("content", "") or ""),
                "short_term_timestamp": str(short_term.get("ts", "") or ""),
                "short_term_content": str(short_term.get("content", "") or ""),
                "is_locked": bool(mem.get("is_locked")),
                "is_milestone": bool(mem.get("is_milestone")),
                "embed_mock_at_write": _memory_embed_mock_at_write(mem),
                "ingest_mode": _memory_ingest_mode(mem),
                "fallback_reason": _memory_fallback_reason(mem),
                "write_path_status": _memory_write_path_status(mem),
                "relation_status": "remote_only",
            }
        )

    rows.sort(key=lambda r: (str(r.get("relation_status", "")), str(r.get("remote_id", "")), str(r.get("node_id", ""))))

    audit = {
        "matched": sum(1 for r in rows if r.get("relation_status") == "matched"),
        "remote_only": sum(1 for r in rows if r.get("relation_status") == "remote_only"),
        "node_only": sum(1 for r in rows if r.get("relation_status") == "node_only"),
        "conflict": sum(1 for r in rows if r.get("relation_status") == "conflict"),
        "conflict_remote_ids": sorted([rid for rid, nodes in remote_to_nodes.items() if len(nodes) > 1]),
    }
    return rows, audit


def _layer_summary(memories: List[Dict[str, Any]], milestones: List[Dict[str, Any]], retrieve_data: Dict[str, Any]) -> Dict[str, Any]:
    details = retrieve_data.get("details", {}) if isinstance(retrieve_data, dict) else {}
    if not isinstance(details, dict):
        details = {}

    level_counter = Counter([_memory_level(m) for m in memories if _memory_level(m)])
    profile_data = details.get("profile_data", {})
    if not isinstance(profile_data, dict):
        profile_data = {}
    return {
        "L0_history_count": len(details.get("l0_history", [])) if isinstance(details.get("l0_history", []), list) else 0,
        "short_term_recent_count": len(details.get("short_term_recent", [])) if isinstance(details.get("short_term_recent", []), list) else 0,
        "L1_count": level_counter.get("L1", 0),
        "L2_count": level_counter.get("L2", 0),
        "L3_recent_emotion_count": len(details.get("recent_emotion", [])) if isinstance(details.get("recent_emotion", []), list) else 0,
        "L4_milestone_count": len(milestones),
        "L4_profile_key_count": len(profile_data),
    }


def _l3_view(retrieve_data: Dict[str, Any]) -> Dict[str, Any]:
    details = retrieve_data.get("details", {}) if isinstance(retrieve_data, dict) else {}
    emotions = details.get("recent_emotion", []) if isinstance(details, dict) else []
    if not isinstance(emotions, list):
        emotions = []
    tag_counter: Counter = Counter()
    timeline: List[Dict[str, Any]] = []
    for item in emotions:
        if not isinstance(item, dict):
            continue
        tag = str(item.get("tag", item.get("emotion_tag", "")) or "").strip()
        intensity = item.get("intensity")
        ts = str(item.get("timestamp", item.get("created_at", "")) or "")
        if tag:
            tag_counter[tag] += 1
        timeline.append({"time": ts, "tag": tag, "intensity": intensity})
    return {"timeline": timeline, "tag_distribution": dict(tag_counter)}


def _l4_view(milestones: List[Dict[str, Any]], retrieve_data: Dict[str, Any]) -> Dict[str, Any]:
    details = retrieve_data.get("details", {}) if isinstance(retrieve_data, dict) else {}
    profile = details.get("profile_data", {}) if isinstance(details, dict) else {}
    if not isinstance(profile, dict):
        profile = {}
    ordered_milestones = sorted(
        [
            {
                "id": m.get("id"),
                "title": m.get("title"),
                "level": m.get("level"),
                "happened_at": m.get("happened_at"),
                "description": m.get("description"),
                "remote_id": m.get("remote_id"),
            }
            for m in milestones
        ],
        key=lambda x: str(x.get("happened_at", "") or ""),
    )
    return {"milestone_timeline": ordered_milestones, "profile_data": profile}


def _mws_stats(memories: List[Dict[str, Any]]) -> Dict[str, Any]:
    values: List[float] = []
    for m in memories:
        v = m.get("mws_score")
        if isinstance(v, bool):
            continue
        try:
            values.append(float(v))
        except Exception:
            continue
    if not values:
        return {"count": 0}
    bins = {"0.0-0.2": 0, "0.2-0.4": 0, "0.4-0.6": 0, "0.6-0.8": 0, "0.8-1.0": 0}
    for v in values:
        if v < 0.2:
            bins["0.0-0.2"] += 1
        elif v < 0.4:
            bins["0.2-0.4"] += 1
        elif v < 0.6:
            bins["0.4-0.6"] += 1
        elif v < 0.8:
            bins["0.6-0.8"] += 1
        else:
            bins["0.8-1.0"] += 1
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": statistics.mean(values),
        "distribution": bins,
    }


def _memory_ingest_mode(item: Dict[str, Any]) -> str:
    return str(item.get("ingest_mode", "") or "").strip()


def _memory_fallback_reason(item: Dict[str, Any]) -> str:
    return str(item.get("fallback_reason", "") or "").strip()


def _memory_embed_mock_at_write(item: Dict[str, Any]) -> bool:
    return bool(item.get("embed_mock_at_write", False))


def _memory_write_path_status(item: Dict[str, Any]) -> str:
    ingest_mode = _memory_ingest_mode(item)
    embed_mock_at_write = _memory_embed_mock_at_write(item)
    if ingest_mode == "raw_fallback" and embed_mock_at_write:
        return "raw_fallback+mock_write"
    if ingest_mode == "raw_fallback":
        return "raw_fallback"
    if embed_mock_at_write:
        return "mock_write"
    return "normal"


def _write_path_summary(memories: List[Dict[str, Any]]) -> Dict[str, Any]:
    status_counter: Counter[str] = Counter()
    fallback_reasons: List[str] = []
    for item in memories:
        if not isinstance(item, dict):
            continue
        status_counter[_memory_write_path_status(item)] += 1
        reason = _memory_fallback_reason(item)
        if reason:
            fallback_reasons.append(reason)
    unique_reasons = sorted(set(fallback_reasons))
    return {
        "embed_mock_at_write_count": sum(1 for item in memories if _memory_embed_mock_at_write(item)),
        "raw_fallback_memory_count": sum(1 for item in memories if _memory_ingest_mode(item) == "raw_fallback"),
        "write_path_status_counts": dict(status_counter),
        "fallback_reason_samples": unique_reasons[:5],
    }


def _consistency_with_stats(
    memories: List[Dict[str, Any]],
    milestones: List[Dict[str, Any]],
    retrieve_data: Dict[str, Any],
    user_stats: Dict[str, Any],
) -> Dict[str, Any]:
    details = retrieve_data.get("details", {}) if isinstance(retrieve_data, dict) else {}
    if not isinstance(details, dict):
        details = {}
    ranked_memories = details.get("ranked_memories", [])
    if not isinstance(ranked_memories, list):
        ranked_memories = []

    level_counter = Counter([_memory_level(m) for m in memories if _memory_level(m)])
    locked = sum(1 for m in memories if bool(m.get("is_locked")))
    milestone_flag = sum(1 for m in memories if bool(m.get("is_milestone")))
    observed_counts = {
        "listed_memories": len(memories),
        "listed_l1": level_counter.get("L1", 0),
        "listed_l2": level_counter.get("L2", 0),
        "listed_milestone_flags": milestone_flag,
        "listed_locked_flags": locked,
        "listed_milestones": len(milestones),
        "retrieve_ranked_memories": len(ranked_memories),
    }
    if not user_stats:
        return {
            "service_available": False,
            "matches": {},
            "service_counts": {},
            "observed_counts": observed_counts,
            "stats_errors": [],
            "warnings": ["本次未拿到 /api/users/{user_id}/stats 返回，因此无法做服务端权威计数对照。"],
        }

    chroma = user_stats.get("chroma", {}) if isinstance(user_stats, dict) else {}
    postgres = user_stats.get("postgres", {}) if isinstance(user_stats, dict) else {}
    flags = user_stats.get("flags", {}) if isinstance(user_stats, dict) else {}
    stats_errors = user_stats.get("errors", []) if isinstance(user_stats, dict) else []
    if not isinstance(chroma, dict):
        chroma = {}
    if not isinstance(postgres, dict):
        postgres = {}
    if not isinstance(flags, dict):
        flags = {}
    if not isinstance(stats_errors, list):
        stats_errors = []

    service_total = _safe_int(chroma.get("total", 0))
    service_l1 = _safe_int(chroma.get("l1", 0))
    service_l2 = _safe_int(chroma.get("l2", 0))
    service_milestones = _safe_int(chroma.get("milestones", 0))
    service_locked = _safe_int(chroma.get("locked", 0))
    service_raw_fallback = _safe_int(chroma.get("raw_fallback", 0))
    service_embed_mock_at_write = _safe_int(chroma.get("embed_mock_at_write", 0))
    service_pg_milestones = _safe_int(postgres.get("l4_milestones", 0))
    ranked_count = len(ranked_memories)
    embed_mock = bool(flags.get("embed_mock", False))

    warnings: List[str] = []
    if embed_mock:
        warnings.append("flags.embed_mock=true，说明 embedding 服务已退化到 MOCK，当前检索质量会下降。")
    if service_raw_fallback > 0:
        warnings.append(
            "chroma.raw_fallback={}，说明有一部分长期记忆是在分析链路异常时以 raw_fallback 方式写入。".format(
                service_raw_fallback
            )
        )
    if service_embed_mock_at_write > 0:
        warnings.append(
            "chroma.embed_mock_at_write={}，说明库里存在写入当下 embedding 已退到 MOCK 的零向量记录；这类旧数据容易表现为‘明明在库里但就是召不回来’。".format(
                service_embed_mock_at_write
            )
        )
    if service_l2 > ranked_count:
        warnings.append(
            "chroma.l2={} 但本次 retrieve.ranked_memories={}，这通常是 /retrieve 的 Top-K 召回截断，不代表 L2 数据丢失。".format(
                service_l2,
                ranked_count,
            )
        )
    if stats_errors:
        warnings.append("stats 接口返回了底层存储读取错误，请结合 errors 字段判断统计是否完整。")

    return {
        "service_available": bool(user_stats),
        "matches": {
            "memory_total_match": service_total == len(memories),
            "l1_match": service_l1 == level_counter.get("L1", 0),
            "l2_match": service_l2 == level_counter.get("L2", 0),
            "milestone_flag_match": service_milestones == milestone_flag,
            "locked_flag_match": service_locked == locked,
            "milestones_pg_vs_listing_match": service_pg_milestones == len(milestones),
        },
        "service_counts": {
            "chroma_total": service_total,
            "chroma_l1": service_l1,
            "chroma_l2": service_l2,
            "chroma_milestones": service_milestones,
            "chroma_locked": service_locked,
            "chroma_raw_fallback": service_raw_fallback,
            "chroma_embed_mock_at_write": service_embed_mock_at_write,
            "postgres_l4_milestones": service_pg_milestones,
            "embed_mock": embed_mock,
        },
        "observed_counts": observed_counts,
        "stats_errors": [str(item) for item in stats_errors],
        "warnings": warnings,
    }


def _stats_dashboard(
    memories: List[Dict[str, Any]],
    milestones: List[Dict[str, Any]],
    relation_audit: Dict[str, Any],
    user_stats: Dict[str, Any],
    consistency_with_stats: Dict[str, Any],
) -> Dict[str, Any]:
    total = len(memories)
    level_counter = Counter([_memory_level(m) for m in memories if _memory_level(m)])
    locked = sum(1 for m in memories if bool(m.get("is_locked")))
    milestone_flag = sum(1 for m in memories if bool(m.get("is_milestone")))
    flags = user_stats.get("flags", {}) if isinstance(user_stats, dict) else {}
    chroma = user_stats.get("chroma", {}) if isinstance(user_stats, dict) else {}
    if not isinstance(flags, dict):
        flags = {}
    if not isinstance(chroma, dict):
        chroma = {}
    write_path_summary = _write_path_summary(memories)
    service_raw_fallback = _safe_int(chroma.get("raw_fallback", 0))
    observed_raw_fallback = _safe_int(write_path_summary.get("raw_fallback_memory_count", 0))
    metadata_gap_warnings: List[str] = []
    if service_raw_fallback > 0 and observed_raw_fallback == 0 and total > 0:
        metadata_gap_warnings.append(
            "stats.chroma.raw_fallback={}，但本次 /api/memories 明细里没有任何 ingest_mode=raw_fallback；Node/Remote 关系视图当前无法逐条定位这些 raw_fallback，说明服务端 list_memories 返回未透出逐条 ingest metadata。".format(
                service_raw_fallback
            )
        )
    return {
        "memory_total": total,
        "L1_count": level_counter.get("L1", 0),
        "L2_count": level_counter.get("L2", 0),
        "locked_rate": (locked / total) if total else 0.0,
        "milestone_flag_rate": (milestone_flag / total) if total else 0.0,
        "milestone_total": len(milestones),
        "relation_audit": relation_audit,
        "mws": _mws_stats(memories),
        "embed_mock": bool(flags.get("embed_mock", False)),
        "raw_fallback_count": service_raw_fallback,
        "service_embed_mock_at_write_count": _safe_int(chroma.get("embed_mock_at_write", 0)),
        "embed_mock_at_write_count": _safe_int(write_path_summary.get("embed_mock_at_write_count", 0)),
        "raw_fallback_memory_count": observed_raw_fallback,
        "write_path_status_counts": write_path_summary.get("write_path_status_counts", {}),
        "fallback_reason_samples": write_path_summary.get("fallback_reason_samples", []),
        "metadata_gap_warnings": metadata_gap_warnings,
        "consistency_with_stats": consistency_with_stats,
    }


def _extract_log_timeline(log_path: Path, agent_name: str) -> List[Dict[str, Any]]:
    if not INCLUDE_TIMELINE_REPLAY or not log_path.exists():
        return []
    pattern = re.compile(
        r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+ .*\[(?P<tag>EXT_MEMORY_[A-Z_]+)\].*agent=(?P<agent>[^ ]+)"
    )
    out: List[Dict[str, Any]] = []
    for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = pattern.match(line.strip())
        if not m:
            continue
        if m.group("agent") != agent_name:
            continue
        out.append(
            {
                "time": m.group("ts"),
                "source": "log",
                "event_type": m.group("tag"),
                "raw": line.strip(),
            }
        )
    return out


def _build_timeline(target: AgentTarget, map_data: Dict[str, Any], memories: List[Dict[str, Any]], milestones: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []

    node_to_remote = map_data.get("node_to_remote", {}) or {}
    for node_id, rel in node_to_remote.items():
        if not isinstance(rel, dict):
            continue
        events.append(
            {
                "time": str(rel.get("create_time", "") or ""),
                "source": "map",
                "event_type": "node_bind",
                "node_id": str(node_id),
                "remote_id": str(rel.get("remote_id", "") or ""),
                "detail": str(rel.get("node_type", "") or ""),
            }
        )

    for mem in memories:
        events.append(
            {
                "time": str(mem.get("timestamp", "") or ""),
                "source": "remote_memory",
                "event_type": "memory_record",
                "node_id": "",
                "remote_id": _memory_remote_id(mem),
                "detail": _memory_level(mem),
            }
        )

    for ms in milestones:
        events.append(
            {
                "time": str(ms.get("happened_at", "") or ""),
                "source": "milestone",
                "event_type": "milestone_record",
                "node_id": "",
                "remote_id": str(ms.get("remote_id", "") or ""),
                "detail": str(ms.get("title", "") or ""),
            }
        )

    events.extend(_extract_log_timeline(log_path=target.log_path, agent_name=target.agent_name))

    def _sort_key(item: Dict[str, Any]):
        dt = _parse_time(item.get("time"))
        return (dt or datetime.min, str(item.get("source", "")), str(item.get("event_type", "")))

    events.sort(key=_sort_key)
    for e in events:
        e["time_fmt"] = _fmt_time(e.get("time"))
    return events


def _html_escape(text: Any) -> str:
    s = "" if text is None else str(text)
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


def _build_html(report: Dict[str, Any]) -> str:
    meta = report.get("meta", {})
    layer = report.get("layer_summary", {})
    stats = report.get("stats_dashboard", {})
    relations = report.get("relations", [])
    l3 = report.get("l3_visualization", {})
    l4 = report.get("l4_visualization", {})
    health = report.get("health", {})
    timeline = report.get("timeline_replay", [])

    metadata_gap_warnings = stats.get("metadata_gap_warnings", []) if isinstance(stats, dict) else []
    if not isinstance(metadata_gap_warnings, list):
        metadata_gap_warnings = []
    relation_warning_html = ""
    if metadata_gap_warnings:
        relation_warning_html = "".join(
            f'<div class="warn">{_html_escape(item)}</div>'
            for item in metadata_gap_warnings
            if str(item or "").strip()
        )

    relation_rows = "".join(
        "<tr>"
        f"<td>{_html_escape(r.get('relation_status', ''))}</td>"
        f"<td>{_html_escape(r.get('node_id', ''))}</td>"
        f"<td>{_html_escape(r.get('remote_id', ''))}</td>"
        f"<td>{_html_escape(r.get('node_type', ''))}</td>"
        f"<td>{_html_escape(r.get('create_time', ''))}</td>"
        f"<td>{_html_escape(r.get('memory_level', ''))}</td>"
        f"<td>{_html_escape(r.get('memory_timestamp', ''))}</td>"
        f"<td>{_html_escape(r.get('memory_content', ''))}</td>"
        f"<td>{_html_escape(r.get('short_term_timestamp', ''))}</td>"
        f"<td>{_html_escape(r.get('short_term_content', ''))}</td>"
        f"<td>{_html_escape(r.get('write_path_status', ''))}</td>"
        f"<td>{_html_escape(r.get('embed_mock_at_write', ''))}</td>"
        f"<td>{_html_escape(r.get('ingest_mode', ''))}</td>"
        f"<td>{_html_escape(r.get('fallback_reason', ''))}</td>"
        "</tr>"
        for r in relations
    )
    l3_rows = "".join(
        "<tr>"
        f"<td>{_html_escape(x.get('time', ''))}</td>"
        f"<td>{_html_escape(x.get('tag', ''))}</td>"
        f"<td>{_html_escape(x.get('intensity', ''))}</td>"
        "</tr>"
        for x in l3.get("timeline", [])
    )
    milestone_rows = "".join(
        "<tr>"
        f"<td>{_html_escape(x.get('happened_at', ''))}</td>"
        f"<td>{_html_escape(x.get('title', ''))}</td>"
        f"<td>{_html_escape(x.get('level', ''))}</td>"
        f"<td>{_html_escape(x.get('remote_id', ''))}</td>"
        "</tr>"
        for x in l4.get("milestone_timeline", [])
    )
    profile_rows = "".join(
        "<tr>"
        f"<td>{_html_escape(k)}</td><td>{_html_escape(v)}</td>"
        "</tr>"
        for k, v in sorted((l4.get("profile_data", {}) or {}).items(), key=lambda kv: str(kv[0]))
    )
    timeline_rows = "".join(
        "<tr>"
        f"<td>{_html_escape(x.get('time_fmt', ''))}</td>"
        f"<td>{_html_escape(x.get('source', ''))}</td>"
        f"<td>{_html_escape(x.get('event_type', ''))}</td>"
        f"<td>{_html_escape(x.get('node_id', ''))}</td>"
        f"<td>{_html_escape(x.get('remote_id', ''))}</td>"
        f"<td>{_html_escape(x.get('detail', x.get('raw', '')))}</td>"
        "</tr>"
        for x in timeline
    )

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>External Memory Audit - {_html_escape(meta.get("save_name", ""))} / {_html_escape(meta.get("agent_name", ""))}</title>
  <style>
    body {{ font-family: "Segoe UI", "Microsoft YaHei", sans-serif; margin: 0; background: #f5f6fa; color: #222; }}
    .wrap {{ max-width: 1500px; margin: 0 auto; padding: 16px; }}
    .card {{ background: #fff; border: 1px solid #e5e7eb; border-radius: 10px; padding: 12px; margin-bottom: 12px; }}
    h1, h2 {{ margin: 0 0 8px 0; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    th, td {{ border: 1px solid #e5e7eb; padding: 6px 8px; text-align: left; vertical-align: top; }}
    th {{ background: #f0f4f8; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 8px; }}
    .kv {{ line-height: 1.8; white-space: pre-wrap; word-break: break-all; }}
    .mono {{ font-family: Consolas, "Courier New", monospace; }}
    .warn {{ margin: 0 0 10px 0; padding: 10px 12px; border-radius: 8px; border: 1px solid #f59e0b; background: #fffbeb; color: #92400e; line-height: 1.6; }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="card">
      <h1>外置记忆审计报告</h1>
      <div class="kv">save={_html_escape(meta.get("save_name", ""))}\nagent={_html_escape(meta.get("agent_name", ""))}\nuser_id={_html_escape(meta.get("user_id", ""))}\nscoped_user_id={_html_escape(meta.get("scoped_user_id", ""))}\nmap_path={_html_escape(meta.get("map_path", ""))}</div>
    </div>

    <div class="card">
      <h2>健康检查整合</h2>
      <pre class="mono">{_html_escape(json.dumps(health, ensure_ascii=False, indent=2))}</pre>
    </div>

    <div class="card">
      <h2>统计看板</h2>
      <pre class="mono">{_html_escape(json.dumps(stats, ensure_ascii=False, indent=2))}</pre>
    </div>

    <div class="card">
      <h2>各层级记忆汇总</h2>
      <pre class="mono">{_html_escape(json.dumps(layer, ensure_ascii=False, indent=2))}</pre>
    </div>

    <div class="card">
      <h2>Node/Remote 关系视图</h2>
      {relation_warning_html}
      <table>
        <thead><tr><th>status</th><th>node_id</th><th>remote_id</th><th>node_type</th><th>create_time</th><th>level</th><th>memory_ts</th><th>memory_content</th><th>short_term_ts</th><th>short_term_content</th><th>write_path_status</th><th>embed_mock_at_write</th><th>ingest_mode</th><th>fallback_reason</th></tr></thead>
        <tbody>{relation_rows}</tbody>
      </table>
    </div>

    <div class="card">
      <h2>L3 可视化（recent_emotion）</h2>
      <div class="grid">
        <div><h3>tag distribution</h3><pre class="mono">{_html_escape(json.dumps(l3.get("tag_distribution", {}), ensure_ascii=False, indent=2))}</pre></div>
      </div>
      <table>
        <thead><tr><th>time</th><th>tag</th><th>intensity</th></tr></thead>
        <tbody>{l3_rows}</tbody>
      </table>
    </div>

    <div class="card">
      <h2>L4 可视化（milestone/profile）</h2>
      <h3>Milestone timeline</h3>
      <table>
        <thead><tr><th>happened_at</th><th>title</th><th>level</th><th>remote_id</th></tr></thead>
        <tbody>{milestone_rows}</tbody>
      </table>
      <h3>Profile key-value</h3>
      <table>
        <thead><tr><th>key</th><th>value</th></tr></thead>
        <tbody>{profile_rows}</tbody>
      </table>
    </div>

    <div class="card">
      <h2>时间线回放</h2>
      <table>
        <thead><tr><th>time</th><th>source</th><th>type</th><th>node_id</th><th>remote_id</th><th>detail</th></tr></thead>
        <tbody>{timeline_rows}</tbody>
      </table>
    </div>
  </div>
</body>
</html>"""


def _build_user_stats_markdown(report: Dict[str, Any]) -> str:
    meta = report.get("meta", {}) if isinstance(report, dict) else {}
    user_stats = report.get("user_stats", {}) if isinstance(report, dict) else {}
    consistency = report.get("consistency_with_stats", {}) if isinstance(report, dict) else {}
    stats_dashboard = report.get("stats_dashboard", {}) if isinstance(report, dict) else {}
    errors = report.get("errors", {}) if isinstance(report, dict) else {}

    chroma = user_stats.get("chroma", {}) if isinstance(user_stats, dict) else {}
    postgres = user_stats.get("postgres", {}) if isinstance(user_stats, dict) else {}
    redis_stats = user_stats.get("redis", {}) if isinstance(user_stats, dict) else {}
    flags = user_stats.get("flags", {}) if isinstance(user_stats, dict) else {}
    stats_errors = user_stats.get("errors", []) if isinstance(user_stats, dict) else []
    if not isinstance(chroma, dict):
        chroma = {}
    if not isinstance(postgres, dict):
        postgres = {}
    if not isinstance(redis_stats, dict):
        redis_stats = {}
    if not isinstance(flags, dict):
        flags = {}
    if not isinstance(stats_errors, list):
        stats_errors = []

    warnings = consistency.get("warnings", []) if isinstance(consistency, dict) else []
    matches = consistency.get("matches", {}) if isinstance(consistency, dict) else {}
    service_counts = consistency.get("service_counts", {}) if isinstance(consistency, dict) else {}
    observed_counts = consistency.get("observed_counts", {}) if isinstance(consistency, dict) else {}
    if not isinstance(warnings, list):
        warnings = []
    if not isinstance(matches, dict):
        matches = {}
    if not isinstance(service_counts, dict):
        service_counts = {}
    if not isinstance(observed_counts, dict):
        observed_counts = {}
    if not isinstance(stats_dashboard, dict):
        stats_dashboard = {}

    def _mark(flag: Any) -> str:
        return "一致" if bool(flag) else "不一致"

    lines = [
        "# 外置记忆用户 Stats 诊断",
        "",
        "## 基本信息",
        "- save_name: `{}`".format(meta.get("save_name", "")),
        "- agent_name: `{}`".format(meta.get("agent_name", "")),
        "- user_id: `{}`".format(meta.get("user_id", "")),
        "- scoped_user_id: `{}`".format(meta.get("scoped_user_id", "")),
        "- service_base_url: `{}`".format(meta.get("service_base_url", "")),
        "- generated_at: `{}`".format(meta.get("generated_at", "")),
        "",
    ]

    if not user_stats:
        lines.extend(
            [
                "## 接口结果",
                "- 未获取到 `/api/users/{user_id}/stats` 返回。",
                "- user_stats_error: `{}`".format(errors.get("user_stats_error", "")),
            ]
        )
        return "\n".join(lines) + "\n"

    lines.extend(
        [
            "## 服务端权威统计（`GET /api/users/{user_id}/stats`）",
            "- `chroma.total`: **{}**，表示 Chroma 中该 user_id 的长期记忆总数（含 L1/L2，也包含 raw_fallback 条目）。".format(_safe_int(chroma.get("total", 0))),
            "- `chroma.l1`: **{}**，表示按 MWS 阈值划分后落在 L1 的条数。".format(_safe_int(chroma.get("l1", 0))),
            "- `chroma.l2`: **{}**，表示按 MWS 阈值划分后落在 L2 的条数。".format(_safe_int(chroma.get("l2", 0))),
            "- `chroma.milestones`: **{}**，表示 metadata 中 `is_milestone=true` 的长期记忆条数。".format(_safe_int(chroma.get("milestones", 0))),
            "- `chroma.locked`: **{}**，表示 metadata 中 `is_locked=true` 的长期记忆条数。".format(_safe_int(chroma.get("locked", 0))),
            "- `chroma.raw_fallback`: **{}**，如果大于 0，说明有一段时间分析链路不健康，长期记忆以 raw_fallback 方式落盘。".format(_safe_int(chroma.get("raw_fallback", 0))),
            "- `chroma.embed_mock_at_write`: **{}**，v2.1.3 起表示写入当下 embedding 退到 MOCK 的长期记忆条数；它是定位\"老数据明明在库里，但就是召不回来\"的重要信号。".format(
                _safe_int(chroma.get("embed_mock_at_write", 0))
            ),
            "",
            "## PostgreSQL 侧统计",
            "- `postgres.l3_emotion_log`: **{}**，对应 L3 情绪日志表条数。".format(_safe_int(postgres.get("l3_emotion_log", 0))),
            "- `postgres.l4_milestones`: **{}**，对应 L4 milestone 表条数。".format(_safe_int(postgres.get("l4_milestones", 0))),
            "- `postgres.l4_profile_attributes`: **{}**，对应画像属性表条数。".format(_safe_int(postgres.get("l4_profile_attributes", 0))),
            "- `postgres.l4_profile_core`: **{}**，对应画像核心表条数。".format(_safe_int(postgres.get("l4_profile_core", 0))),
            "",
            "## Redis 侧统计",
            "- `redis.short_term`: **{}**，对应 `memory_st:{{user_id}}` 的列表长度。".format(_safe_int(redis_stats.get("short_term", 0))),
            "- `redis.l0_cache`: **{}**，对应 `l0_history:{{user_id}}` 的字段数。".format(_safe_int(redis_stats.get("l0_cache", 0))),
            "",
            "## 运行标记",
            "- `flags.embed_mock`: **{}**。为 `true` 时表示 embedding 服务退化到了 MOCK，检索质量会下降。".format(bool(flags.get("embed_mock", False))),
        ]
    )

    if stats_errors:
        lines.append("")
        lines.append("## stats 接口 errors")
        for item in stats_errors:
            lines.append("- {}".format(item))

    lines.extend(
        [
            "",
            "## 单条写入退化观察（基于 `/api/memories` metadata）",
            "- `embed_mock_at_write=true` 条数：**{}**。这表示这些记忆写入当下 BGE 不可达，向量被写成零向量；它和 `flags.embed_mock` 的区别是：前者是历史逐条事实，后者是当前服务状态。".format(
                _safe_int(stats_dashboard.get("embed_mock_at_write_count", 0))
            ),
            "- `ingest_mode=raw_fallback` 条数：**{}**。这些记忆写入时 LLM 分析链路异常，因此只有原文降级写入，不会自然进入 L2/L3/L4。".format(
                _safe_int(stats_dashboard.get("raw_fallback_memory_count", 0))
            ),
            "- `write_path_status` 分布：`{}`。".format(
                json.dumps(stats_dashboard.get("write_path_status_counts", {}), ensure_ascii=False)
            ),
            "- `fallback_reason` 样例：`{}`。".format(
                json.dumps(stats_dashboard.get("fallback_reason_samples", []), ensure_ascii=False)
            ),
            *[
                "- 元数据缺口提示：{}".format(item)
                for item in (stats_dashboard.get("metadata_gap_warnings", []) if isinstance(stats_dashboard, dict) else [])
                if str(item or "").strip()
            ],
            "",
            "## 与本次审计结果对照",
            "- 长期记忆总数：服务端 **{}** / 当前 list_memories **{}** / {}".format(
                _safe_int(service_counts.get("chroma_total", 0)),
                _safe_int(observed_counts.get("listed_memories", 0)),
                _mark(matches.get("memory_total_match", False)),
            ),
            "- L1 条数：服务端 **{}** / 当前 list_memories **{}** / {}".format(
                _safe_int(service_counts.get("chroma_l1", 0)),
                _safe_int(observed_counts.get("listed_l1", 0)),
                _mark(matches.get("l1_match", False)),
            ),
            "- L2 条数：服务端 **{}** / 当前 list_memories **{}** / {}".format(
                _safe_int(service_counts.get("chroma_l2", 0)),
                _safe_int(observed_counts.get("listed_l2", 0)),
                _mark(matches.get("l2_match", False)),
            ),
            "- milestone 标记数：服务端 **{}** / 当前 list_memories **{}** / {}".format(
                _safe_int(service_counts.get("chroma_milestones", 0)),
                _safe_int(observed_counts.get("listed_milestone_flags", 0)),
                _mark(matches.get("milestone_flag_match", False)),
            ),
            "- locked 标记数：服务端 **{}** / 当前 list_memories **{}** / {}".format(
                _safe_int(service_counts.get("chroma_locked", 0)),
                _safe_int(observed_counts.get("listed_locked_flags", 0)),
                _mark(matches.get("locked_flag_match", False)),
            ),
            "- PG milestone 表 vs milestone listing：服务端 **{}** / 当前 list_milestones **{}** / {}".format(
                _safe_int(service_counts.get("postgres_l4_milestones", 0)),
                _safe_int(observed_counts.get("listed_milestones", 0)),
                _mark(matches.get("milestones_pg_vs_listing_match", False)),
            ),
            "- retrieve.ranked_memories 当前返回 **{}** 条；它是 Top-K 召回结果，不应直接当作 L2 全量计数。".format(
                _safe_int(observed_counts.get("retrieve_ranked_memories", 0))
            ),
        ]
    )

    if warnings:
        lines.append("")
        lines.append("## 诊断提示")
        for item in warnings:
            lines.append("- {}".format(item))

    lines.extend(
        [
            "",
            "## 原始返回 JSON",
            "```json",
            json.dumps(user_stats, ensure_ascii=False, indent=2),
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def _build_report(target: AgentTarget) -> Dict[str, Any]:
    client = ECDollMemoryServiceClient(
        base_url=MEMORY_SERVICE_BASE_URL,
        timeout=MEMORY_SERVICE_TIMEOUT_SECONDS,
    )

    health = _health_bundle(client)
    memories: List[Dict[str, Any]] = []
    milestones: List[Dict[str, Any]] = []
    retrieve_data: Dict[str, Any] = {}
    user_stats: Dict[str, Any] = {}
    errors: Dict[str, str] = {}

    try:
        memories = _fetch_all_memories(client=client, scoped_user_id=target.scoped_user_id)
    except Exception as exc:
        errors["memories_error"] = str(exc)
    try:
        milestones = _fetch_milestones(client=client, scoped_user_id=target.scoped_user_id)
    except Exception as exc:
        errors["milestones_error"] = str(exc)
    retrieve_data = _fetch_retrieve_details(client=client, scoped_user_id=target.scoped_user_id)
    if "error" in retrieve_data:
        errors["retrieve_error"] = str(retrieve_data.get("error"))
    user_stats = _fetch_user_stats(client=client, scoped_user_id=target.scoped_user_id)
    if "error" in user_stats:
        errors["user_stats_error"] = str(user_stats.get("error"))
        user_stats = {}

    map_data = _load_map(target.map_path)
    relations, relation_audit = _build_relations(memories=memories, map_data=map_data, retrieve_data=retrieve_data)
    layer_summary = _layer_summary(memories=memories, milestones=milestones, retrieve_data=retrieve_data)
    consistency_with_stats = _consistency_with_stats(
        memories=memories,
        milestones=milestones,
        retrieve_data=retrieve_data,
        user_stats=user_stats,
    )
    stats_dashboard = _stats_dashboard(
        memories=memories,
        milestones=milestones,
        relation_audit=relation_audit,
        user_stats=user_stats,
        consistency_with_stats=consistency_with_stats,
    )
    l3_visualization = _l3_view(retrieve_data)
    l4_visualization = _l4_view(milestones=milestones, retrieve_data=retrieve_data)
    timeline = _build_timeline(target=target, map_data=map_data, memories=memories, milestones=milestones)
    local_supplement = _load_local_supplement(target)

    report = {
        "meta": {
            "save_name": target.save_name,
            "agent_name": target.agent_name,
            "user_id": target.user_id,
            "scoped_user_id": target.scoped_user_id,
            "snapshot_path": str(target.snapshot_path),
            "map_path": str(target.map_path),
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "service_base_url": MEMORY_SERVICE_BASE_URL,
        },
        "health": health,
        "errors": errors,
        "layer_summary": layer_summary,
        "stats_dashboard": stats_dashboard,
        "consistency_audit": relation_audit,
        "consistency_with_stats": consistency_with_stats,
        "user_stats": user_stats,
        "retrieve_snapshot": retrieve_data,
        "relations": relations,
        "memories": memories,
        "milestones": milestones,
        "l3_visualization": l3_visualization,
        "l4_visualization": l4_visualization,
        "timeline_replay": timeline,
        "local_supplement": local_supplement,
    }
    return report


def _export_report(report: Dict[str, Any]) -> None:
    meta = report.get("meta", {})
    save_name = _safe_filename(meta.get("save_name", "unknown_save"))
    agent_name = _safe_filename(meta.get("agent_name", "unknown_agent"))
    out_dir = OUTPUT_ROOT / save_name / agent_name
    out_dir.mkdir(parents=True, exist_ok=True)

    if WRITE_JSON:
        _write_json(out_dir / "report.json", report)
    if WRITE_CSV:
        _write_csv(out_dir / "relations.csv", report.get("relations", []))
        _write_csv(out_dir / "memories.csv", report.get("memories", []))
        _write_csv(out_dir / "timeline.csv", report.get("timeline_replay", []))
    (out_dir / "user_stats.md").write_text(_build_user_stats_markdown(report), encoding="utf-8")
    if WRITE_HTML:
        (out_dir / "report.html").write_text(_build_html(report), encoding="utf-8")

    print(f"[OK] save={meta.get('save_name')} agent={meta.get('agent_name')} -> {out_dir}")


def main() -> int:
    args = parse_args()
    _apply_cli_overrides(args)

    targets = _collect_targets()
    if not targets:
        print("[WARN] No valid targets found. Check TARGET_SAVE_NAMES/TARGET_AGENT_NAMES and user_id resolution.")
        return 1
    for target in targets:
        try:
            report = _build_report(target)
            _export_report(report)
        except Exception as exc:
            print(f"[FAIL] save={target.save_name} agent={target.agent_name} error={exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
