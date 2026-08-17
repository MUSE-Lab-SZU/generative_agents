#!/usr/bin/env python3
"""Refresh only model endpoint routing in the snapshot used by --resume.

``start.py --resume`` deliberately restores the runtime configuration from the
latest ``simulate-*.json`` checkpoint, instead of from ``data/config.json``.
This helper makes an explicit, auditable exception for the two model routing
sections, without changing simulation/controller state.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_ROOT = BASE_DIR / "results"
ROUTING_PATHS = (
    ("agent_base", "think", "llm"),
    ("agent_base", "associate", "embedding"),
)
SOURCE_PATHS = (
    ("agent", "think", "llm"),
    ("agent", "associate", "embedding"),
)
ROUTING_KEYS = ("base_url", "load_balancing")


def load_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"JSON 文件无效: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"JSON 顶层必须是 object: {path}")
    return payload


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise


def get_mapping(payload: dict[str, Any], path: tuple[str, ...], *, context: str) -> dict[str, Any]:
    current: Any = payload
    for key in path:
        if not isinstance(current, dict) or key not in current:
            raise ValueError(f"{context} 缺少配置路径: {'.'.join(path)}")
        current = current[key]
    if not isinstance(current, dict):
        raise ValueError(f"{context} 配置路径不是 object: {'.'.join(path)}")
    return current


def normalized_route(source: dict[str, Any], *, context: str) -> dict[str, Any]:
    base_url = source.get("base_url")
    if not isinstance(base_url, str) or not base_url.strip():
        raise ValueError(f"{context}.base_url 必须是非空字符串")
    route: dict[str, Any] = {"base_url": base_url.strip()}
    if "load_balancing" in source:
        load_balancing = source["load_balancing"]
        if not isinstance(load_balancing, dict):
            raise ValueError(f"{context}.load_balancing 必须是 object")
        if bool(load_balancing.get("enabled", False)):
            ports = load_balancing.get("ports")
            if not isinstance(ports, list) or not ports:
                raise ValueError(
                    f"{context}.load_balancing.enabled=true 时 ports 必须为非空数组"
                )
        route["load_balancing"] = copy.deepcopy(load_balancing)
    return route


def latest_snapshot(checkpoint_dir: Path) -> Path:
    snapshots = sorted(checkpoint_dir.glob("simulate-*.json"))
    if not snapshots:
        raise FileNotFoundError(f"没有可刷新路由的 checkpoint 快照: {checkpoint_dir}")
    return snapshots[-1]


def refresh(*, run_name: str, routing_config_path: Path, results_root: Path, dry_run: bool) -> dict[str, Any]:
    if not run_name or Path(run_name).name != run_name:
        raise ValueError("--run-name 必须是单个 checkpoint 目录名")
    checkpoint_dir = results_root / "checkpoints" / run_name
    snapshot_path = latest_snapshot(checkpoint_dir)
    snapshot = load_object(snapshot_path)
    routing_config = load_object(routing_config_path)

    changes: dict[str, dict[str, Any]] = {}
    for target_path, source_path, label in zip(
        ROUTING_PATHS, SOURCE_PATHS, ("llm", "embedding"), strict=True
    ):
        target = get_mapping(snapshot, target_path, context="checkpoint")
        source = get_mapping(routing_config, source_path, context="routing config")
        route = normalized_route(source, context="routing config." + ".".join(source_path))
        before = {key: copy.deepcopy(target.get(key)) for key in ROUTING_KEYS if key in target}
        for key in ROUTING_KEYS:
            target.pop(key, None)
        target.update(route)
        after = {key: copy.deepcopy(target.get(key)) for key in ROUTING_KEYS if key in target}
        changes[label] = {"before": before, "after": after}

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    backup_dir = results_root / "recovery_backups" / run_name / f"model-routing-{timestamp}"
    result: dict[str, Any] = {
        "status": "dry_run" if dry_run else "refreshed",
        "run_name": run_name,
        "snapshot": str(snapshot_path),
        "routing_config": str(routing_config_path),
        "backup_dir": str(backup_dir),
        "changes": changes,
    }
    if dry_run:
        return result

    backup_dir.mkdir(parents=True, exist_ok=False)
    original = load_object(snapshot_path)
    try:
        write_json_atomic(backup_dir / snapshot_path.name, original)
        write_json_atomic(snapshot_path, snapshot)
        write_json_atomic(backup_dir / "model_routing_refresh.json", result)
    except BaseException:
        if (backup_dir / snapshot_path.name).is_file():
            write_json_atomic(snapshot_path, original)
        raise
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="刷新 --resume 使用的 LLM/BGE endpoint 路由")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--routing-config", default="data/config.json")
    parser.add_argument("--results-root", default=str(DEFAULT_RESULTS_ROOT))
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    routing_config_path = Path(args.routing_config).expanduser()
    if not routing_config_path.is_absolute():
        routing_config_path = BASE_DIR / routing_config_path
    result = refresh(
        run_name=str(args.run_name),
        routing_config_path=routing_config_path.resolve(),
        results_root=Path(args.results_root).expanduser().resolve(),
        dry_run=bool(args.dry_run),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, FileNotFoundError) as exc:
        print(f"错误: {exc}")
        raise SystemExit(2)
