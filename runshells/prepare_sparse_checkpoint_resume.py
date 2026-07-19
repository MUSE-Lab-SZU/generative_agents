#!/usr/bin/env python3
"""Prepare a sparse simulation checkpoint for a state-consistent resume.

The live storage may be newer than the latest sparse ``simulate-*.json``.  This
tool finds the newest staged-eval capture whose runtime snapshot and full
storage bundle describe the same completed step, validates it, backs up the
newer live artifacts, and restores that consistent point in place.
"""

from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_ROOT = BASE_DIR / "results"
SNAPSHOT_RE = re.compile(r"^simulate-.+\.json$")
TRACE_STATE_KEYS = (
    "dialog_judge_trace_state",
    "forced_prompt_trace_state",
)


@dataclass(frozen=True)
class RecoveryAnchor:
    label: str
    step_no: int
    sim_time: str
    snapshot_name: str
    snapshot_path: Path
    snapshot_config: dict[str, Any]
    job_path: Path
    job: dict[str, Any]
    storage_root: Path
    manifest_path: Path
    manifest: dict[str, Any]


def load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"JSON 文件无效: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"JSON 顶层必须是对象: {path}")
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


def canonical_json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_child(root: Path, raw_relpath: Any, field_name: str) -> Path:
    text = str(raw_relpath or "").strip()
    relpath = Path(text)
    if not text or relpath.is_absolute():
        raise ValueError(f"bundle {field_name} 必须是非空相对路径: {raw_relpath}")
    root_resolved = root.resolve()
    resolved = (root_resolved / relpath).resolve()
    try:
        resolved.relative_to(root_resolved)
    except ValueError as exc:
        raise ValueError(f"bundle {field_name} 越出允许目录: {raw_relpath}") from exc
    return resolved


def validate_snapshot_bundle(job_path: Path, job: dict[str, Any]) -> tuple[Path, Path, dict[str, Any]]:
    bundle = job.get("snapshot_bundle")
    if not isinstance(bundle, dict) or not bundle:
        raise ValueError(f"staged job 没有 snapshot_bundle: {job_path}")
    if int(bundle.get("schema_version", 0) or 0) != 1:
        raise ValueError(f"不支持的 snapshot bundle schema: {job_path}")
    if str(bundle.get("storage_scope", "") or "") != "full":
        raise ValueError(f"snapshot bundle 不是 full storage: {job_path}")

    trigger_dir = job_path.parent
    storage_root = resolve_child(trigger_dir, bundle.get("storage_relpath"), "storage_relpath")
    manifest_path = resolve_child(trigger_dir, bundle.get("manifest_relpath"), "manifest_relpath")
    if not storage_root.is_dir():
        raise FileNotFoundError(f"snapshot storage 不存在: {storage_root}")
    manifest = load_json(manifest_path)
    label = str(job.get("trigger_label", "") or "")
    if int(manifest.get("schema_version", 0) or 0) != 1:
        raise ValueError(f"snapshot manifest schema 不匹配: {manifest_path}")
    if str(manifest.get("artifact_kind", "") or "") != "repeat_eval_snapshot":
        raise ValueError(f"snapshot manifest 类型不匹配: {manifest_path}")
    if str(manifest.get("storage_scope", "") or "") != "full":
        raise ValueError(f"snapshot manifest 不是 full storage: {manifest_path}")
    if str(manifest.get("trigger_label", "") or "") != label:
        raise ValueError(f"snapshot manifest label 不匹配: {manifest_path}")

    runtime_digest = canonical_json_sha256(job.get("runtime_config", {}))
    conversation_digest = canonical_json_sha256(job.get("conversation", {}) or {})
    if runtime_digest != str(manifest.get("runtime_config_sha256", "") or ""):
        raise ValueError(f"runtime_config SHA-256 不匹配: {job_path}")
    if runtime_digest != str(bundle.get("runtime_config_sha256", "") or ""):
        raise ValueError(f"bundle runtime_config SHA-256 不匹配: {job_path}")
    if conversation_digest != str(manifest.get("conversation_sha256", "") or ""):
        raise ValueError(f"conversation SHA-256 不匹配: {job_path}")
    if conversation_digest != str(bundle.get("conversation_sha256", "") or ""):
        raise ValueError(f"bundle conversation SHA-256 不匹配: {job_path}")

    records = manifest.get("files")
    if not isinstance(records, list):
        raise ValueError(f"snapshot manifest files 必须是列表: {manifest_path}")
    expected_paths: set[str] = set()
    total_size = 0
    for record in records:
        if not isinstance(record, dict):
            raise ValueError(f"snapshot manifest 文件记录无效: {manifest_path}")
        relpath = str(record.get("path", "") or "")
        if not relpath or relpath in expected_paths:
            raise ValueError(f"snapshot manifest 路径为空或重复: {relpath}")
        expected_paths.add(relpath)
        file_path = resolve_child(storage_root, relpath, "files.path")
        if not file_path.is_file():
            raise FileNotFoundError(f"snapshot bundle 文件缺失: {file_path}")
        expected_size = int(record.get("size", -1))
        actual_size = file_path.stat().st_size
        if actual_size != expected_size:
            raise ValueError(f"snapshot bundle 文件大小不匹配: {file_path}")
        if file_sha256(file_path) != str(record.get("sha256", "") or ""):
            raise ValueError(f"snapshot bundle 文件 SHA-256 不匹配: {file_path}")
        total_size += actual_size

    actual_paths = {
        path.relative_to(storage_root).as_posix()
        for path in storage_root.rglob("*")
        if path.is_file()
    }
    if actual_paths != expected_paths:
        missing = sorted(expected_paths - actual_paths)
        unexpected = sorted(actual_paths - expected_paths)
        raise ValueError(
            "snapshot bundle 文件集合不匹配: "
            f"missing={missing[:5]} unexpected={unexpected[:5]}"
        )
    if len(records) != int(manifest.get("file_count", -1)):
        raise ValueError(f"snapshot bundle 文件数不匹配: {manifest_path}")
    if total_size != int(manifest.get("total_size", -1)):
        raise ValueError(f"snapshot bundle 总大小不匹配: {manifest_path}")
    return storage_root, manifest_path, manifest


def _snapshot_step(snapshot_path: Path) -> int:
    payload = load_json(snapshot_path)
    raw_step = payload.get("step")
    if isinstance(raw_step, bool):
        raise ValueError(f"snapshot step 无效: {snapshot_path}")
    try:
        return int(raw_step)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"snapshot step 无效: {snapshot_path}") from exc


def discover_recovery_anchor(checkpoint_dir: Path) -> RecoveryAnchor:
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"checkpoint 不存在: {checkpoint_dir}")
    candidates: list[RecoveryAnchor] = []
    errors: list[str] = []
    for job_path in sorted((checkpoint_dir / "staged_eval").glob("*/job.json")):
        try:
            job = load_json(job_path)
            label = str(job.get("trigger_label", "") or "").strip()
            if not label or label == "T0":
                continue
            step_no = int(job.get("step_no", 0) or 0)
            sim_time = str(job.get("sim_time", "") or "")
            snapshot_name = str(job.get("snapshot_name", "") or "")
            if step_no <= 0 or not SNAPSHOT_RE.fullmatch(snapshot_name):
                continue
            snapshot_path = checkpoint_dir / snapshot_name
            if not snapshot_path.is_file():
                continue
            snapshot_config = load_json(snapshot_path)
            if int(snapshot_config.get("step", -1)) != step_no:
                continue
            if str(snapshot_config.get("time", "") or "") != sim_time:
                continue
            staged_state = snapshot_config.get("staged_eval_state", {})
            labels = staged_state.get("triggered_labels", []) if isinstance(staged_state, dict) else []
            if label not in labels:
                raise ValueError(f"匹配快照未记录 staged label={label}: {snapshot_path}")
            snapshot_stem = Path(snapshot_name).stem
            for state_key in TRACE_STATE_KEYS:
                sidecar_path = (
                    checkpoint_dir
                    / "trace_state_sidecars"
                    / state_key
                    / f"{snapshot_stem}.json"
                )
                if not sidecar_path.is_file():
                    raise FileNotFoundError(f"恢复锚点缺少 trace sidecar: {sidecar_path}")
                load_json(sidecar_path)
            storage_root, manifest_path, manifest = validate_snapshot_bundle(job_path, job)
            if int(manifest.get("step_no", -1)) != step_no:
                raise ValueError(f"manifest step 与 job 不匹配: {manifest_path}")
            if str(manifest.get("sim_time", "") or "") != sim_time:
                raise ValueError(f"manifest time 与 job 不匹配: {manifest_path}")
            if str(manifest.get("snapshot_name", "") or "") != snapshot_name:
                raise ValueError(f"manifest snapshot 与 job 不匹配: {manifest_path}")
            candidates.append(
                RecoveryAnchor(
                    label=label,
                    step_no=step_no,
                    sim_time=sim_time,
                    snapshot_name=snapshot_name,
                    snapshot_path=snapshot_path,
                    snapshot_config=snapshot_config,
                    job_path=job_path,
                    job=job,
                    storage_root=storage_root,
                    manifest_path=manifest_path,
                    manifest=manifest,
                )
            )
        except Exception as exc:
            errors.append(f"{job_path}: {exc}")
    if not candidates:
        detail = "\n".join(errors[-5:]) if errors else "没有非 T0 的匹配 bundle"
        raise RuntimeError(f"未找到可验证的一致恢复锚点: {checkpoint_dir}\n{detail}")
    return max(candidates, key=lambda item: (item.step_no, item.snapshot_name, item.label))


def resolve_run_from_batch_state(results_root: Path, batch_name: str, condition: str) -> tuple[str, str, Path]:
    if not re.fullmatch(r"[A-Za-z0-9._-]+", batch_name):
        raise ValueError(f"batch name 含有非法字符: {batch_name}")
    if not re.fullmatch(r"[A-Za-z0-9._-]+", condition):
        raise ValueError(f"condition 含有非法字符: {condition}")
    state_path = results_root / "experiment_data" / "batch_state" / batch_name / f"{condition}.json"
    state = load_json(state_path)
    if str(state.get("batch_name", "") or "") != batch_name:
        raise ValueError(f"batch state 中的 batch_name 不匹配: {state_path}")
    if str(state.get("condition_name", "") or "") != condition:
        raise ValueError(f"batch state 中的 condition_name 不匹配: {state_path}")
    run_name = str(state.get("run_name", "") or "").strip()
    if not run_name or Path(run_name).name != run_name:
        raise ValueError(f"batch state 中的 run_name 无效: {state_path}")
    expected_prefix = f"{batch_name}-{condition}-"
    if not (run_name == f"{batch_name}-{condition}" or run_name.startswith(expected_prefix)):
        raise ValueError(f"run_name 与 batch/condition 不匹配: {run_name}")
    status = str(state.get("status", "") or "").strip()
    return run_name, status, state_path


def find_active_processes(run_name: str) -> list[tuple[int, str]]:
    matches: list[tuple[int, str]] = []
    own_pid = os.getpid()
    excluded_pids = {own_pid}
    ancestor_pid = os.getppid()
    while ancestor_pid > 1 and ancestor_pid not in excluded_pids:
        excluded_pids.add(ancestor_pid)
        try:
            stat_fields = (Path("/proc") / str(ancestor_pid) / "stat").read_text().split()
            ancestor_pid = int(stat_fields[3])
        except (FileNotFoundError, PermissionError, ValueError, IndexError):
            break
    proc_root = Path("/proc")
    if not proc_root.is_dir():
        return matches
    for entry in proc_root.iterdir():
        if not entry.name.isdigit() or int(entry.name) in excluded_pids:
            continue
        try:
            raw = (entry / "cmdline").read_bytes()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        command = raw.replace(b"\0", b" ").decode("utf-8", errors="replace").strip()
        if run_name in command:
            matches.append((int(entry.name), command))
    return matches


def _effective_base_configs(snapshot_config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    agent_base = snapshot_config.get("agent_base", {})
    if not isinstance(agent_base, dict):
        agent_base = {}
    think = agent_base.get("think", {}) if isinstance(agent_base.get("think", {}), dict) else {}
    associate = agent_base.get("associate", {}) if isinstance(agent_base.get("associate", {}), dict) else {}
    llm = think.get("llm", {}) if isinstance(think.get("llm", {}), dict) else {}
    embedding = associate.get("embedding", {}) if isinstance(associate.get("embedding", {}), dict) else {}
    return llm, embedding


def check_openai_compatible_endpoint(config: dict[str, Any], label: str) -> None:
    if str(config.get("provider", "") or "").lower() != "openai":
        return
    base_url = str(config.get("base_url", "") or "").strip().rstrip("/")
    if not base_url:
        raise RuntimeError(f"{label} 缺少 base_url")
    request = urllib.request.Request(
        f"{base_url}/models",
        headers={"Authorization": f"Bearer {str(config.get('api_key', 'EMPTY') or 'EMPTY')}"},
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=8) as response:
            if not 200 <= int(response.status) < 300:
                raise RuntimeError(f"HTTP {response.status}")
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as exc:
        raise RuntimeError(f"{label} 服务不可用: {base_url}: {exc}") from exc


def _dotenv_has_value(path: Path, key: str) -> bool:
    if not path.is_file():
        return False
    prefix = f"{key}="
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or not line.startswith(prefix):
            continue
        return bool(line[len(prefix):].strip().strip("'\""))
    return False


def check_runtime_requirements(anchor: RecoveryAnchor) -> None:
    active = find_active_processes(anchor.snapshot_path.parent.name)
    if active:
        sample = "\n".join(f"pid={pid} {command}" for pid, command in active[:5])
        raise RuntimeError(f"目标 checkpoint 仍有活跃进程，拒绝恢复:\n{sample}")
    llm, embedding = _effective_base_configs(anchor.snapshot_config)
    check_openai_compatible_endpoint(llm, "Chat LLM")
    check_openai_compatible_endpoint(embedding, "Embedding")
    intervention = anchor.snapshot_config.get("intervention", {})
    forced = intervention.get("forced_llm", {}) if isinstance(intervention, dict) else {}
    if isinstance(forced, dict) and bool(forced.get("enabled", False)):
        env_name = str(forced.get("api_key_env", "DEEPSEEK_API_KEY") or "DEEPSEEK_API_KEY").strip()
        if env_name and not os.environ.get(env_name) and not _dotenv_has_value(BASE_DIR / ".env", env_name):
            raise RuntimeError(f"forced LLM 所需环境变量未设置: {env_name}")


def build_staged_index(state: dict[str, Any], run_name: str) -> dict[str, Any]:
    return {
        "run_name": run_name,
        "triggered_labels": list(state.get("triggered_labels", []) or []),
        "queued_labels": list(state.get("queued_labels", []) or []),
        "failed_labels": list(state.get("failed_labels", []) or []),
        "pending_jobs": list(state.get("pending_jobs", []) or []),
        "running_job": copy.deepcopy(state.get("running_job")),
        "records": list(state.get("records", []) or []),
    }


def artifacts_after_anchor(checkpoint_dir: Path, anchor: RecoveryAnchor) -> tuple[list[Path], list[str]]:
    paths: list[Path] = []
    moved_snapshot_stems: list[str] = []
    for snapshot_path in sorted(checkpoint_dir.glob("simulate-*.json")):
        if snapshot_path == anchor.snapshot_path:
            continue
        if _snapshot_step(snapshot_path) > anchor.step_no:
            paths.append(snapshot_path)
            moved_snapshot_stems.append(snapshot_path.stem)

    sidecar_root = checkpoint_dir / "trace_state_sidecars"
    for state_key in TRACE_STATE_KEYS:
        state_dir = sidecar_root / state_key
        for stem in moved_snapshot_stems:
            sidecar = state_dir / f"{stem}.json"
            if sidecar.is_file():
                paths.append(sidecar)

    staged_root = checkpoint_dir / "staged_eval"
    for trigger_dir in sorted(path for path in staged_root.iterdir() if path.is_dir()) if staged_root.is_dir() else []:
        if trigger_dir.name.startswith("_tmp"):
            paths.append(trigger_dir)
            continue
        job_path = trigger_dir / "job.json"
        if not job_path.is_file():
            continue
        job = load_json(job_path)
        if int(job.get("step_no", 0) or 0) > anchor.step_no:
            paths.append(trigger_dir)
    return paths, moved_snapshot_stems


def _backup_destination(backup_dir: Path, checkpoint_dir: Path, source: Path) -> Path:
    relative = source.relative_to(checkpoint_dir)
    return backup_dir / "checkpoint_artifacts" / relative


def _copy_and_verify_anchor_storage(anchor: RecoveryAnchor, temp_parent: Path) -> Path:
    temp_parent.mkdir(parents=True, exist_ok=True)
    temp_storage = Path(tempfile.mkdtemp(prefix="storage-", dir=str(temp_parent)))
    try:
        shutil.copytree(anchor.storage_root, temp_storage, dirs_exist_ok=True)
        records = anchor.manifest.get("files", [])
        actual_paths = {
            path.relative_to(temp_storage).as_posix()
            for path in temp_storage.rglob("*")
            if path.is_file()
        }
        expected_paths = {str(record.get("path", "") or "") for record in records}
        if actual_paths != expected_paths:
            raise ValueError("复制后的 recovery storage 文件集合不匹配")
        for record in records:
            path = temp_storage / str(record["path"])
            if path.stat().st_size != int(record["size"]) or file_sha256(path) != str(record["sha256"]):
                raise ValueError(f"复制后的 recovery storage 校验失败: {path}")
        return temp_storage
    except BaseException:
        shutil.rmtree(temp_storage, ignore_errors=True)
        raise


def prepare_checkpoint_resume(
    checkpoint_dir: Path,
    results_root: Path,
    run_name: str,
    target_step: int,
    *,
    dry_run: bool,
    check_health: bool = True,
) -> dict[str, Any]:
    anchor = discover_recovery_anchor(checkpoint_dir)
    if target_step <= anchor.step_no:
        raise ValueError(f"目标总步数必须大于恢复锚点: target={target_step}, anchor={anchor.step_no}")
    active = find_active_processes(run_name)
    if active:
        sample = "\n".join(f"pid={pid} {command}" for pid, command in active[:5])
        raise RuntimeError(f"目标 checkpoint 仍有活跃进程，拒绝恢复:\n{sample}")
    if check_health and not dry_run:
        check_runtime_requirements(anchor)

    extra_paths, moved_snapshot_stems = artifacts_after_anchor(checkpoint_dir, anchor)
    live_paths = [
        checkpoint_dir / "storage",
        checkpoint_dir / "conversation.json",
        checkpoint_dir / "judge_traces",
        checkpoint_dir / "forced_prompt_traces",
        checkpoint_dir / "staged_eval" / "state.json",
        checkpoint_dir / "staged_eval" / "index.json",
    ]
    sources: list[Path] = []
    seen: set[Path] = set()
    for path in [*live_paths, *extra_paths]:
        if path.exists() and path not in seen:
            sources.append(path)
            seen.add(path)

    now = datetime.now()
    recovery_id = now.strftime("%Y%m%d-%H%M%S-%f")
    backup_dir = results_root / "recovery_backups" / run_name / recovery_id
    plan = {
        "schema_version": 1,
        "status": "dry_run" if dry_run else "prepared",
        "run_name": run_name,
        "checkpoint_dir": str(checkpoint_dir),
        "target_step": int(target_step),
        "remaining_steps": int(target_step - anchor.step_no),
        "anchor": {
            "label": anchor.label,
            "step_no": anchor.step_no,
            "sim_time": anchor.sim_time,
            "snapshot_name": anchor.snapshot_name,
            "job_path": str(anchor.job_path),
            "manifest_path": str(anchor.manifest_path),
            "storage_file_count": int(anchor.manifest.get("file_count", 0) or 0),
            "storage_total_size": int(anchor.manifest.get("total_size", 0) or 0),
            "runtime_config_sha256": str(anchor.manifest.get("runtime_config_sha256", "") or ""),
            "conversation_sha256": str(anchor.manifest.get("conversation_sha256", "") or ""),
        },
        "backup_dir": str(backup_dir),
        "backed_up_paths": [str(path.relative_to(checkpoint_dir)) for path in sources],
        "moved_post_anchor_snapshot_stems": moved_snapshot_stems,
        "prepared_at": now.isoformat(timespec="seconds"),
    }
    if dry_run:
        return plan

    temp_parent = results_root / ".recovery_tmp" / run_name
    temp_storage = _copy_and_verify_anchor_storage(anchor, temp_parent)
    backup_dir.mkdir(parents=True, exist_ok=False)
    moved: list[tuple[Path, Path]] = []
    installed: list[Path] = []
    try:
        applying_plan = dict(plan)
        applying_plan["status"] = "applying"
        write_json_atomic(backup_dir / "recovery_manifest.json", applying_plan)
        for source in sources:
            destination = _backup_destination(backup_dir, checkpoint_dir, source)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source), str(destination))
            moved.append((destination, source))

        live_storage = checkpoint_dir / "storage"
        os.replace(temp_storage, live_storage)
        installed.append(live_storage)

        conversation = anchor.job.get("conversation", {}) or {}
        write_json_atomic(checkpoint_dir / "conversation.json", conversation)
        installed.append(checkpoint_dir / "conversation.json")

        staged_state = copy.deepcopy(anchor.snapshot_config.get("staged_eval_state", {}))
        if not isinstance(staged_state, dict):
            raise ValueError(f"恢复锚点缺少 staged_eval_state: {anchor.snapshot_path}")
        staged_state["pending_jobs"] = []
        staged_state["queued_labels"] = []
        staged_state["failed_labels"] = []
        staged_state["running_job"] = None
        write_json_atomic(checkpoint_dir / "staged_eval" / "state.json", staged_state)
        installed.append(checkpoint_dir / "staged_eval" / "state.json")
        write_json_atomic(
            checkpoint_dir / "staged_eval" / "index.json",
            build_staged_index(staged_state, run_name),
        )
        installed.append(checkpoint_dir / "staged_eval" / "index.json")

        snapshot_stem = Path(anchor.snapshot_name).stem
        judge_sidecar = (
            checkpoint_dir
            / "trace_state_sidecars"
            / "dialog_judge_trace_state"
            / f"{snapshot_stem}.json"
        )
        if judge_sidecar.is_file():
            write_json_atomic(
                checkpoint_dir / "judge_traces" / "judge_conversation.json",
                load_json(judge_sidecar),
            )
            installed.append(checkpoint_dir / "judge_traces")
        (checkpoint_dir / "forced_prompt_traces").mkdir(parents=True, exist_ok=True)
        installed.append(checkpoint_dir / "forced_prompt_traces")

        write_json_atomic(backup_dir / "recovery_manifest.json", plan)
        try:
            temp_parent.rmdir()
            temp_parent.parent.rmdir()
        except OSError:
            pass
        return plan
    except BaseException:
        for path in sorted(set(installed), key=lambda item: len(item.parts), reverse=True):
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
            else:
                path.unlink(missing_ok=True)
        for backup_path, original_path in reversed(moved):
            if not backup_path.exists():
                continue
            original_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(backup_path), str(original_path))
        shutil.rmtree(temp_storage, ignore_errors=True)
        try:
            temp_parent.rmdir()
            temp_parent.parent.rmdir()
        except OSError:
            pass
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="验证并准备稀疏 checkpoint 的一致性续跑")
    selector = parser.add_mutually_exclusive_group(required=True)
    selector.add_argument("--run-name", help="直接指定 results/checkpoints 下的 run_name")
    selector.add_argument("--batch-name", help="通过 batch state 精确选择，例如 batch-0718-01")
    parser.add_argument("--condition", help="与 --batch-name 配套，例如 Counsel-KBD6-G9-SEV")
    parser.add_argument("--results-root", default=str(DEFAULT_RESULTS_ROOT), help="results 根目录")
    parser.add_argument("--target-step", type=int, default=120, help="实验目标总步数，默认 120")
    parser.add_argument("--dry-run", action="store_true", help="只验证并显示恢复计划，不移动文件")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    results_root = Path(args.results_root).expanduser().resolve()
    if args.batch_name:
        if not args.condition:
            raise ValueError("--batch-name 必须同时提供 --condition")
        run_name, status, state_path = resolve_run_from_batch_state(
            results_root,
            str(args.batch_name),
            str(args.condition),
        )
    else:
        if args.condition:
            raise ValueError("--condition 只能与 --batch-name 一起使用")
        run_name = str(args.run_name or "").strip()
        if not run_name or Path(run_name).name != run_name:
            raise ValueError("--run-name 必须是单个目录名")
        status, state_path = "", None

    checkpoint_dir = results_root / "checkpoints" / run_name
    if args.dry_run:
        plan = prepare_checkpoint_resume(
            checkpoint_dir,
            results_root,
            run_name,
            int(args.target_step),
            dry_run=bool(args.dry_run),
            check_health=True,
        )
    else:
        lock_dir = results_root / "recovery_locks"
        lock_dir.mkdir(parents=True, exist_ok=True)
        lock_path = lock_dir / f"{run_name}.lock"
        with lock_path.open("a+", encoding="utf-8") as lock_handle:
            try:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError(f"已有恢复进程持有锁: {lock_path}") from exc
            plan = prepare_checkpoint_resume(
                checkpoint_dir,
                results_root,
                run_name,
                int(args.target_step),
                dry_run=False,
                check_health=True,
            )
    plan["batch_state_status"] = status
    plan["batch_state_path"] = str(state_path) if state_path else ""
    print(json.dumps(plan, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, FileNotFoundError) as exc:
        print(f"错误: {exc}", file=sys.stderr)
        raise SystemExit(2)
