#!/usr/bin/env python3
"""Create no-intervention post-simulation follow-up branches.

Only strict staged-eval snapshot bundles are accepted. Source checkpoints are
read-only; every follow-up is resumed from an independent copied branch.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import copy
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from artifact_digest import canonical_json_sha256
from run_archived_repeat_scale_eval import (
    target_external_memory_enabled,
    validate_snapshot_bundle,
)


BASE_DIR = Path(__file__).resolve().parents[1]
CHECKPOINTS_ROOT = BASE_DIR / "results" / "checkpoints"
REPORTS_DIR = BASE_DIR / "results" / "experiment_data" / "reports"
OVERLAY_PATH = (
    BASE_DIR
    / "experiments"
    / "config"
    / "phases"
    / "post_sim_followup_no_intervention.json"
)
CONTROLLER_MANIFEST_FILENAME = "cbt_condition_manifest.json"
FOLLOWUP_MANIFEST_FILENAME = "followup_manifest.json"
FOLLOWUP_SCHEMA_VERSION = 1
IDENTITY_KEYS = (
    "controller_identity",
    "kind",
    "mode",
    "controller_version",
    "progressive_stage",
    "capabilities",
)


@dataclass(frozen=True)
class FollowupConfig:
    name: str
    steps: int
    interval: int
    max_parallel: int
    dry_run: bool
    force: bool
    resume_partial: bool
    verbose: str
    checkpoints_root: Path = CHECKPOINTS_ROOT
    reports_dir: Path = REPORTS_DIR
    overlay_path: Path = OVERLAY_PATH


@dataclass(frozen=True)
class SourceSpec:
    node_dir: Path
    condition: dict[str, Any]


@dataclass(frozen=True)
class PreparedSource:
    spec: SourceSpec
    source_run: str
    source_label: str
    source_job: dict[str, Any]
    source_metadata: dict[str, Any]
    source_bundle_manifest: dict[str, Any]
    source_storage: Path
    source_controller_manifest: dict[str, Any]
    source_step: int
    source_time: str
    source_stride: int
    target_agent: str
    destination_run: str
    expected_labels: tuple[str, ...]


@dataclass(frozen=True)
class PartialResume:
    """A strictly validated, restorable point in an interrupted follow-up."""

    step: int
    label: str
    completed_labels: tuple[str, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="从严格 staged snapshot 创建仿真后无干预回访分支"
    )
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--source-summary", help="原批量实验 summary JSON")
    inputs.add_argument(
        "--source-node",
        action="append",
        help="staged_eval/<label> 节点目录；可重复传入",
    )
    parser.add_argument(
        "--source-label",
        default="session_20",
        help="--source-summary 模式选择的 staged label，默认 session_20",
    )
    parser.add_argument("--name", required=True, help="follow-up batch 名称")
    parser.add_argument("--steps", type=int, default=120, help="继续运行步数，默认 120")
    parser.add_argument("--interval", type=int, default=30, help="冻结节点间隔，默认 30")
    parser.add_argument("--max-parallel", type=int, default=1, help="并行 follow-up 数，默认 1")
    parser.add_argument("--verbose", default="info", help="透传给 start.py 的日志级别")
    parser.add_argument(
        "--force",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="重建时把旧派生 run 移入 _superseded_followups",
    )
    parser.add_argument(
        "--resume-partial",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="默认开启：严格校验中断回访的连续节点，并从最后一个完整节点续跑",
    )
    parser.add_argument(
        "--dry-run",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="只校验并展示计划，不创建任何文件或目录",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except FileNotFoundError:
        raise FileNotFoundError(f"required JSON not found: {path}") from None
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw_temp = tempfile.mkstemp(prefix=f".{path.name}-", dir=path.parent)
    temp_path = Path(raw_temp)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        os.replace(temp_path, path)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


def deep_merge(base: Mapping[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(dict(base))
    for key, value in overlay.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def safe_token(value: str, field: str) -> str:
    token = str(value or "").strip()
    if not token or token in {".", ".."} or "/" in token or "\\" in token:
        raise ValueError(f"invalid {field}: {value!r}")
    cleaned = re.sub(r"[^\w.-]+", "-", token, flags=re.UNICODE).strip("-.")
    if not cleaned:
        raise ValueError(f"invalid {field}: {value!r}")
    return cleaned


def resolve_path(raw: str) -> Path:
    path = Path(str(raw or "")).expanduser()
    if not path.is_absolute():
        path = BASE_DIR / path
    return path.resolve()


def parse_condition_dimensions(condition_name: str) -> dict[str, str]:
    tokens = str(condition_name or "").split("-")
    variant = ""
    group = ""
    severity = ""
    severity_map = {"MILD": "mild", "MOD": "moderate", "MODERATE": "moderate", "SEV": "severe", "SEVERE": "severe"}
    for token in tokens:
        upper = token.upper()
        if re.fullmatch(r"KBD\d+", upper):
            variant = upper.lower()
        elif re.fullmatch(r"G\d+", upper):
            group = upper.lower()
        elif upper in severity_map:
            severity = severity_map[upper]
    if not variant and group == "g4":
        variant = "kbd1"
    return {"variant": variant, "group": group, "severity": severity}


def source_specs_from_summary(summary_path: Path, label: str, checkpoints_root: Path) -> list[SourceSpec]:
    summary = load_json(summary_path)
    conditions = summary.get("conditions")
    if not isinstance(conditions, list) or not conditions:
        raise ValueError(f"source summary has no conditions: {summary_path}")
    specs: list[SourceSpec] = []
    for raw_condition in conditions:
        if not isinstance(raw_condition, dict):
            raise ValueError(f"source summary contains invalid condition: {summary_path}")
        condition = copy.deepcopy(raw_condition)
        condition_name = str(condition.get("condition_name", "") or "").strip()
        run_name = str(condition.get("run_name", "") or "").strip()
        if not condition_name or not run_name:
            raise ValueError(f"source summary condition identity is incomplete: {summary_path}")
        evaluations = condition.get("evaluations", []) or []
        if not any(
            isinstance(item, dict) and str(item.get("trigger_label", "") or "") == label
            for item in evaluations
        ):
            raise ValueError(f"source condition has no requested label: {condition_name} {label}")
        specs.append(
            SourceSpec(
                node_dir=(checkpoints_root / safe_token(run_name, "source run") / "staged_eval" / safe_token(label, "source label")).resolve(),
                condition=condition,
            )
        )
    return specs


def source_specs_from_nodes(raw_nodes: list[str], checkpoints_root: Path) -> list[SourceSpec]:
    specs: list[SourceSpec] = []
    checkpoints_resolved = checkpoints_root.resolve()
    for raw_node in raw_nodes:
        node = resolve_path(raw_node)
        if node.parent.name != "staged_eval":
            raise ValueError(f"source node must be staged_eval/<label>: {node}")
        source_run_dir = node.parent.parent.resolve()
        try:
            source_run_dir.relative_to(checkpoints_resolved)
        except ValueError as exc:
            raise ValueError(f"source node is outside checkpoints root: {node}") from exc
        controller_manifest = load_json(source_run_dir / CONTROLLER_MANIFEST_FILENAME)
        condition_name = str(controller_manifest.get("condition_name", "") or "").strip()
        if not condition_name:
            raise ValueError(f"source controller manifest has no condition_name: {source_run_dir}")
        condition = {
            "condition_name": condition_name,
            "condition_key": str(controller_manifest.get("condition_key", "") or ""),
            "run_name": source_run_dir.name,
            **parse_condition_dimensions(condition_name),
            **{key: copy.deepcopy(controller_manifest.get(key)) for key in IDENTITY_KEYS},
            "controller_manifest": copy.deepcopy(controller_manifest),
        }
        specs.append(SourceSpec(node_dir=node, condition=condition))
    return specs


def ensure_unique_conditions(specs: list[SourceSpec]) -> None:
    seen: set[str] = set()
    for spec in specs:
        name = str(spec.condition.get("condition_name", "") or "").strip()
        if name in seen:
            raise ValueError(f"each follow-up batch accepts one source per condition_name: {name}")
        seen.add(name)


def effective_time(runtime_config: Mapping[str, Any]) -> str:
    raw = runtime_config.get("time", "")
    if isinstance(raw, Mapping):
        raw = raw.get("start", "")
    value = str(raw or "").strip()
    try:
        datetime.strptime(value, "%Y%m%d-%H:%M")
    except ValueError as exc:
        raise ValueError(f"source runtime_config.time is invalid: {value!r}") from exc
    return value


def source_controller_manifest(spec: SourceSpec, source_run_dir: Path) -> dict[str, Any]:
    checkpoint_manifest = load_json(source_run_dir / CONTROLLER_MANIFEST_FILENAME)
    embedded = spec.condition.get("controller_manifest")
    if isinstance(embedded, dict) and embedded:
        for key in ("condition_name", "condition_key", "run_name", *IDENTITY_KEYS):
            if embedded.get(key) != checkpoint_manifest.get(key):
                raise ValueError(f"source summary/checkpoint controller manifest mismatch: {key}")
    if str(checkpoint_manifest.get("artifact_kind", "") or "") != "cbt_experiment_condition":
        raise ValueError(f"source controller manifest artifact kind mismatch: {source_run_dir}")
    if str(checkpoint_manifest.get("run_name", "") or "") != source_run_dir.name:
        raise ValueError(f"source controller manifest run_name mismatch: {source_run_dir}")
    return checkpoint_manifest


def prepare_source(spec: SourceSpec, cfg: FollowupConfig) -> PreparedSource:
    node_dir = spec.node_dir.resolve()
    if not node_dir.is_dir():
        raise FileNotFoundError(f"source staged node not found: {node_dir}")
    source_label = safe_token(node_dir.name, "source label")
    source_run_dir = node_dir.parent.parent.resolve()
    source_run = safe_token(source_run_dir.name, "source run")
    if node_dir.parent.name != "staged_eval":
        raise ValueError(f"source node must be staged_eval/<label>: {node_dir}")

    job = load_json(node_dir / "job.json")
    metadata = load_json(node_dir / "metadata.json")
    if str(job.get("trigger_label", "") or "") != source_label:
        raise ValueError(f"source job label mismatch: {node_dir}")
    if str(metadata.get("trigger_label", "") or "") != source_label:
        raise ValueError(f"source metadata label mismatch: {node_dir}")
    if str(metadata.get("artifact_kind", "") or "") != "repeat_eval_snapshot":
        raise ValueError(f"source metadata is not a repeat-eval snapshot: {node_dir}")
    source_storage = validate_snapshot_bundle(node_dir, job, source_label)
    bundle = job.get("snapshot_bundle", {})
    bundle_manifest_path = node_dir / str(bundle.get("manifest_relpath", "") or "")
    bundle_manifest = load_json(bundle_manifest_path)

    runtime_config = job.get("runtime_config")
    if not isinstance(runtime_config, dict):
        raise ValueError(f"source runtime_config must be an object: {node_dir / 'job.json'}")
    source_step = int(runtime_config.get("step", -1))
    if source_step < 1 or int(job.get("step_no", -1)) != source_step:
        raise ValueError(f"source runtime_config.step/job.step_no mismatch: {node_dir}")
    source_time = effective_time(runtime_config)
    if str(job.get("sim_time", "") or "") != source_time:
        raise ValueError(f"source runtime_config.time/job.sim_time mismatch: {node_dir}")
    source_stride = int(runtime_config.get("stride", 0) or 0)
    if source_stride <= 0:
        raise ValueError(f"source runtime_config.stride must be positive: {node_dir}")
    target_agent = str(job.get("target_agent", "") or "").strip()
    agents = runtime_config.get("agents", {})
    if not target_agent or not isinstance(agents, dict) or target_agent not in agents:
        raise ValueError(f"source target agent is missing from runtime config: {node_dir}")
    if target_external_memory_enabled(runtime_config, target_agent):
        raise ValueError(
            f"post-simulation follow-up does not support enabled external memory: {node_dir}"
        )

    controller_manifest = source_controller_manifest(spec, source_run_dir)
    condition_key = str(
        spec.condition.get("condition_key") or controller_manifest.get("condition_key") or ""
    ).strip()
    if not condition_key:
        raise ValueError(f"source condition_key is missing: {source_run_dir}")
    destination_run = "{}-{}-from-{}".format(
        safe_token(cfg.name, "follow-up name"),
        safe_token(condition_key, "condition key"),
        source_label,
    )
    if (cfg.checkpoints_root / destination_run).resolve() == source_run_dir:
        raise ValueError(f"follow-up destination must differ from source run: {source_run_dir}")
    trigger_count = cfg.steps // cfg.interval
    expected_labels = tuple(
        f"followup_step_{index * cfg.interval}" for index in range(1, trigger_count + 1)
    )
    return PreparedSource(
        spec=spec,
        source_run=source_run,
        source_label=source_label,
        source_job=job,
        source_metadata=metadata,
        source_bundle_manifest=bundle_manifest,
        source_storage=source_storage,
        source_controller_manifest=controller_manifest,
        source_step=source_step,
        source_time=source_time,
        source_stride=source_stride,
        target_agent=target_agent,
        destination_run=destination_run,
        expected_labels=expected_labels,
    )


def reset_control_state(runtime_config: dict[str, Any]) -> dict[str, int]:
    state = runtime_config.get("intervention_state")
    if not isinstance(state, dict):
        state = {}
        runtime_config["intervention_state"] = state
    active = state.get("active_meetings", {})
    queues = state.get("doctor_meeting_queues", {})
    current = state.get("doctor_current_meeting", {})
    summary = {
        "active_meetings_cleared": len(active) if isinstance(active, dict) else 0,
        "queued_meetings_cleared": sum(len(items) for items in queues.values() if isinstance(items, list)) if isinstance(queues, dict) else 0,
        "doctor_current_meetings_cleared": len(current) if isinstance(current, dict) else 0,
        "agent_locks_cleared": 0,
        "forced_chat_residuals_cleared": 0,
    }
    state["active_meetings"] = {}
    state["doctor_meeting_queues"] = {}
    state["doctor_current_meeting"] = {}

    agents = runtime_config.get("agents", {})
    if isinstance(agents, dict):
        for agent_cfg in agents.values():
            if not isinstance(agent_cfg, dict):
                continue
            status = agent_cfg.get("status")
            if not isinstance(status, dict):
                continue
            intervention = status.get("intervention")
            lock = intervention.get("lock") if isinstance(intervention, dict) else None
            has_meeting_lock = bool(
                isinstance(lock, dict)
                and (lock.get("enabled") or lock.get("meeting_id") or lock.get("target_agent"))
            )
            chat_state = status.get("chat_action_state")
            has_forced_chat_state = bool(
                isinstance(chat_state, dict)
                and (chat_state.get("forced") or chat_state.get("meeting_id"))
            )
            action = agent_cfg.get("action")
            action_event = action.get("event") if isinstance(action, dict) else None
            is_chat_action = bool(
                isinstance(action_event, dict)
                and str(action_event.get("predicate", "") or "") == "对话"
            )
            if has_forced_chat_state or (has_meeting_lock and is_chat_action):
                status["chat_action_state"] = {}
                if is_chat_action:
                    agent_cfg.pop("action", None)
                summary["forced_chat_residuals_cleared"] += 1
            if not isinstance(lock, dict):
                continue
            if has_meeting_lock:
                summary["agent_locks_cleared"] += 1
            intervention["lock"] = {
                "enabled": False,
                "mode": "meeting",
                "priority": "hard",
                "target_agent": "",
                "meeting_id": "",
                "expire_at": "",
                "trigger_step": -1,
                "deferred_by_sleep": False,
                "defer_count": 0,
            }
    runtime_config["staged_eval_state"] = {}
    return summary


def derive_followup_runtime(
    prepared: PreparedSource,
    cfg: FollowupConfig,
    overlay: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, int]]:
    source_runtime = prepared.source_job["runtime_config"]
    derived = deep_merge(source_runtime, overlay)
    cleanup = reset_control_state(derived)
    derived["checkpointing"] = {
        "enabled": True,
        "mode": "consult_sparse",
        "periodic_every_steps": 0,
        "write_final_snapshot": True,
        "write_on_staged_eval_trigger": True,
    }
    staged = derived.setdefault("staged_eval", {})
    if not isinstance(staged, dict):
        staged = {}
        derived["staged_eval"] = staged
    staged.update(
        {
            "enabled": True,
            "execution_mode": "capture_only",
            "target_agent": prepared.target_agent,
            "include_t0": False,
            "session_interval": 0,
            "t4_enabled": False,
            "auto_stop_after_t4_done": False,
            "step_interval": {
                "enabled": True,
                "every_steps": cfg.interval,
                "anchor_step": prepared.source_step,
                "label_prefix": "followup_step",
                "label_value_mode": "relative_step",
                "max_triggers": cfg.steps // cfg.interval,
            },
        }
    )
    return derived, cleanup


def derived_controller_manifest(
    prepared: PreparedSource,
    derived_runtime: Mapping[str, Any],
) -> dict[str, Any]:
    parent = prepared.source_controller_manifest
    manifest = copy.deepcopy(parent)
    manifest.update(
        {
            "run_name": prepared.destination_run,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "phase": "post_sim_followup",
            "parent_run_name": prepared.source_run,
            "parent_manifest_sha256": canonical_json_sha256(parent),
        }
    )
    resolved = manifest.get("resolved_config")
    if not isinstance(resolved, dict):
        resolved = {}
        manifest["resolved_config"] = resolved
    resolved["path"] = baseline_snapshot_name(prepared.source_time)
    resolved["sha256"] = canonical_json_sha256(derived_runtime)
    resolved["config"] = copy.deepcopy(dict(derived_runtime))
    manifest["resume"] = {
        "requested": True,
        "checkpoint_exists": True,
        "snapshot": baseline_snapshot_name(prepared.source_time),
        "validated": True,
        "parent_run_name": prepared.source_run,
        "parent_label": prepared.source_label,
    }
    return manifest


def request_payload(
    prepared: PreparedSource,
    cfg: FollowupConfig,
    overlay: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "source_run": prepared.source_run,
        "source_label": prepared.source_label,
        "source_runtime_config_sha256": canonical_json_sha256(prepared.source_job["runtime_config"]),
        "source_conversation_sha256": canonical_json_sha256(
            prepared.source_job.get("conversation", {}) or {}
        ),
        "source_bundle_manifest_sha256": canonical_json_sha256(
            prepared.source_bundle_manifest
        ),
        "destination_run": prepared.destination_run,
        "followup_steps": cfg.steps,
        "interval": cfg.interval,
        "target_step": prepared.source_step + cfg.steps,
        "expected_labels": list(prepared.expected_labels),
        "overlay_sha256": canonical_json_sha256(overlay),
    }


def initial_followup_manifest(
    prepared: PreparedSource,
    cfg: FollowupConfig,
    overlay: Mapping[str, Any],
    derived_runtime: Mapping[str, Any],
    cleanup: Mapping[str, int],
) -> dict[str, Any]:
    request = request_payload(prepared, cfg, overlay)
    source_bundle = prepared.source_job.get("snapshot_bundle", {})
    manifest_relpath = str(source_bundle.get("manifest_relpath", "") or "")
    return {
        "schema_version": FOLLOWUP_SCHEMA_VERSION,
        "artifact_kind": "post_sim_followup",
        "status": "prepared",
        "prepared_at": datetime.now().isoformat(timespec="seconds"),
        "started_at": "",
        "completed_at": "",
        "failed_at": "",
        "error": "",
        "request": request,
        "request_sha256": canonical_json_sha256(request),
        "source": {
            "run_name": prepared.source_run,
            "condition_name": str(prepared.spec.condition.get("condition_name", "") or ""),
            "condition_key": str(prepared.source_controller_manifest.get("condition_key", "") or ""),
            "label": prepared.source_label,
            "node_path": str(prepared.spec.node_dir),
            "job_path": str(prepared.spec.node_dir / "job.json"),
            "metadata_path": str(prepared.spec.node_dir / "metadata.json"),
            "bundle_manifest_path": str(prepared.spec.node_dir / manifest_relpath),
            "runtime_config_sha256": canonical_json_sha256(prepared.source_job["runtime_config"]),
            "conversation_sha256": canonical_json_sha256(prepared.source_job.get("conversation", {}) or {}),
            "bundle_manifest_sha256": canonical_json_sha256(prepared.source_bundle_manifest),
            "step": prepared.source_step,
            "time": prepared.source_time,
            "stride": prepared.source_stride,
        },
        "destination_run": prepared.destination_run,
        "target_step": prepared.source_step + cfg.steps,
        "expected_labels": list(prepared.expected_labels),
        "overlay": {
            "path": str(cfg.overlay_path),
            "sha256": canonical_json_sha256(overlay),
            "config": copy.deepcopy(dict(overlay)),
        },
        "control_state_cleanup": dict(cleanup),
        "external_memory": {
            "target_agent": prepared.target_agent,
            "enabled": False,
            "validated": True,
        },
        "source_runtime_config_sha256": canonical_json_sha256(prepared.source_job["runtime_config"]),
        "derived_runtime_config_sha256": canonical_json_sha256(derived_runtime),
        "actual_labels": [],
        "actual_nodes": [],
    }


def baseline_snapshot_name(source_time: str) -> str:
    return f"simulate-{source_time.replace(':', '')}.json"


def archive_existing(destination: Path, checkpoints_root: Path) -> Path:
    archive_root = checkpoints_root / "_superseded_followups"
    archive_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    candidate = archive_root / f"{destination.name}-{stamp}"
    suffix = 1
    while candidate.exists():
        candidate = archive_root / f"{destination.name}-{stamp}-{suffix}"
        suffix += 1
    destination.rename(candidate)
    return candidate


def completed_destination_matches(destination: Path, request: Mapping[str, Any]) -> bool:
    manifest_path = destination / FOLLOWUP_MANIFEST_FILENAME
    if not manifest_path.is_file():
        return False
    existing = load_json(manifest_path)
    return (
        str(existing.get("status", "") or "") == "completed"
        and str(existing.get("request_sha256", "") or "") == canonical_json_sha256(request)
    )


def snapshot_path_for_name(destination: Path, snapshot_name: Any) -> Path:
    name = str(snapshot_name or "").strip()
    path = Path(name)
    if (
        not name
        or path.name != name
        or not name.startswith("simulate-")
        or not name.endswith(".json")
    ):
        raise ValueError(f"invalid follow-up snapshot name: {snapshot_name!r}")
    return destination / name


def validate_followup_node(
    prepared: PreparedSource,
    destination: Path,
    label: str,
) -> tuple[dict[str, Any], Path]:
    """Validate one completed relative-step bundle and its matching root snapshot."""
    relative_step = int(label.removeprefix("followup_step_"))
    expected_step = prepared.source_step + relative_step
    node_dir = destination / "staged_eval" / label
    job = load_json(node_dir / "job.json")
    metadata = load_json(node_dir / "metadata.json")
    storage = validate_snapshot_bundle(node_dir, job, label)
    if str(metadata.get("artifact_kind", "") or "") != "repeat_eval_snapshot":
        raise ValueError(f"follow-up node artifact kind mismatch: {node_dir}")
    if str(job.get("trigger_label", "") or "") != label:
        raise ValueError(f"follow-up job label mismatch: expected={label} path={node_dir}")
    if str(metadata.get("trigger_label", "") or "") != label:
        raise ValueError(f"follow-up metadata label mismatch: expected={label} path={node_dir}")
    runtime_config = job.get("runtime_config")
    if not isinstance(runtime_config, dict):
        raise ValueError(f"follow-up job runtime_config must be an object: {node_dir}")
    observed_steps = {
        "job.step_no": int(job.get("step_no", -1)),
        "metadata.step_no": int(metadata.get("step_no", -1)),
        "runtime_config.step": int(runtime_config.get("step", -1)),
    }
    if any(step != expected_step for step in observed_steps.values()):
        raise ValueError(
            f"follow-up node step mismatch: label={label} expected={expected_step} actual={observed_steps}"
        )
    expected_metadata = {
        "step_interval_anchor_step": prepared.source_step,
        "step_interval_elapsed_steps": relative_step,
        "step_interval_label_value_mode": "relative_step",
        "followup_phase": True,
    }
    mismatched_metadata = {
        key: {"expected": value, "actual": metadata.get(key)}
        for key, value in expected_metadata.items()
        if metadata.get(key) != value
    }
    if mismatched_metadata:
        raise ValueError(
            f"follow-up node relative metadata mismatch: label={label} fields={mismatched_metadata}"
        )
    snapshot_path = snapshot_path_for_name(destination, job.get("snapshot_name"))
    if not snapshot_path.is_file():
        raise FileNotFoundError(f"follow-up root snapshot missing: {snapshot_path}")
    snapshot = load_json(snapshot_path)
    if int(snapshot.get("step", -1)) != expected_step:
        raise ValueError(
            f"follow-up root snapshot step mismatch: expected={expected_step} path={snapshot_path}"
        )
    if str(snapshot.get("time", "") or "") != str(job.get("sim_time", "") or ""):
        raise ValueError(f"follow-up root snapshot time mismatch: {snapshot_path}")
    return job, storage


def restore_partial_runtime(
    destination: Path,
    storage_source: Path,
    conversation: Any,
    resume_snapshot: Path,
    tail_node_dirs: tuple[Path, ...] = (),
) -> tuple[str, list[str], list[str]]:
    """Restore mutable runtime files from a validated staged bundle.

    Preserve interrupted live state outside the run before replacing it, so a
    recovery never silently discards writes made after the frozen node.
    """
    backup_root = destination.parent / "_partial_followup_resume_backups"
    backup_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup = backup_root / f"{destination.name}-{stamp}"
    suffix = 1
    while backup.exists():
        backup = backup_root / f"{destination.name}-{stamp}-{suffix}"
        suffix += 1
    temporary = Path(tempfile.mkdtemp(prefix=".resume-storage-", dir=destination))
    live_storage = destination / "storage"
    live_conversation = destination / "conversation.json"
    live_staged_state = destination / "staged_eval" / "state.json"
    live_staged_index = destination / "staged_eval" / "index.json"
    moved_live_storage = False
    moved_live_conversation = False
    moved_staged_files: list[tuple[Path, Path]] = []
    moved_tail_nodes: list[tuple[Path, Path]] = []
    installed_restored_storage = False
    wrote_conversation = False
    wrote_staged_state = False
    moved_snapshots: list[tuple[Path, Path]] = []
    try:
        restored_storage = temporary / "storage"
        shutil.copytree(storage_source, restored_storage)
        resume_runtime = load_json(resume_snapshot)
        staged_state = resume_runtime.get("staged_eval_state", {}) or {}
        if not isinstance(staged_state, dict):
            raise ValueError(
                f"partial resume staged_eval_state must be an object: {resume_snapshot}"
            )
        # start.py resumes from the lexicographically latest simulate-*.json.
        # A process can be interrupted after writing a newer root snapshot but
        # before its staged bundle is complete; retain those snapshots in the
        # backup and make the validated node the selected resume point.
        for snapshot in sorted(destination.glob("simulate-*.json")):
            if snapshot.name <= resume_snapshot.name:
                continue
            backup_snapshots = backup / "root_snapshots_after_resume_point"
            backup_snapshots.mkdir(parents=True, exist_ok=True)
            archived_snapshot = backup_snapshots / snapshot.name
            snapshot.rename(archived_snapshot)
            moved_snapshots.append((snapshot, archived_snapshot))
        if live_storage.exists():
            backup.mkdir(parents=True, exist_ok=True)
            live_storage.rename(backup / "storage")
            moved_live_storage = True
        if live_conversation.exists():
            backup.mkdir(parents=True, exist_ok=True)
            live_conversation.rename(backup / "conversation.json")
            moved_live_conversation = True
        for live_file in (live_staged_state, live_staged_index):
            if not live_file.exists():
                continue
            staged_backup = backup / "staged_eval_runtime"
            staged_backup.mkdir(parents=True, exist_ok=True)
            archived_file = staged_backup / live_file.name
            live_file.rename(archived_file)
            moved_staged_files.append((live_file, archived_file))
        for node_dir in tail_node_dirs:
            if not node_dir.exists():
                continue
            node_backup = backup / "staged_eval_nodes_after_resume_point"
            node_backup.mkdir(parents=True, exist_ok=True)
            archived_node = node_backup / node_dir.name
            node_dir.rename(archived_node)
            moved_tail_nodes.append((node_dir, archived_node))
        os.replace(restored_storage, live_storage)
        installed_restored_storage = True
        write_json_atomic(live_conversation, conversation or {})
        wrote_conversation = True
        write_json_atomic(live_staged_state, staged_state)
        wrote_staged_state = True
    except Exception:
        if wrote_staged_state:
            live_staged_state.unlink(missing_ok=True)
        if wrote_conversation:
            live_conversation.unlink(missing_ok=True)
        if installed_restored_storage and live_storage.exists():
            shutil.rmtree(live_storage, ignore_errors=True)
        for original, archived in reversed(moved_tail_nodes):
            if archived.exists() and not original.exists():
                archived.rename(original)
        for original, archived in reversed(moved_staged_files):
            if archived.exists() and not original.exists():
                archived.rename(original)
        for original, archived in reversed(moved_snapshots):
            if archived.exists() and not original.exists():
                archived.rename(original)
        backup_conversation = backup / "conversation.json"
        if (
            moved_live_conversation
            and not live_conversation.exists()
            and backup_conversation.exists()
        ):
            backup_conversation.rename(live_conversation)
        backup_storage = backup / "storage"
        if moved_live_storage and not live_storage.exists() and backup_storage.exists():
            backup_storage.rename(live_storage)
        raise
    finally:
        shutil.rmtree(temporary, ignore_errors=True)
    return (
        str(backup) if backup.exists() else "",
        [str(archived.relative_to(backup)) for _, archived in moved_snapshots],
        [str(archived.relative_to(backup)) for _, archived in moved_tail_nodes],
    )


def prepare_partial_resume(
    prepared: PreparedSource,
    cfg: FollowupConfig,
    destination: Path,
    expected_manifest: Mapping[str, Any],
) -> tuple[dict[str, Any], PartialResume | None]:
    """Validate and restore an interrupted branch, returning its exact resume point."""
    manifest_path = destination / FOLLOWUP_MANIFEST_FILENAME
    existing = load_json(manifest_path)
    expected_request_sha = str(expected_manifest.get("request_sha256", "") or "")
    if (
        str(existing.get("artifact_kind", "") or "") != "post_sim_followup"
        or str(existing.get("destination_run", "") or "") != prepared.destination_run
        or str(existing.get("request_sha256", "") or "") != expected_request_sha
        or str(existing.get("status", "") or "") not in {"prepared", "running", "failed"}
    ):
        raise ValueError(
            "existing follow-up does not exactly match the requested source/config; "
            "refusing partial resume"
        )

    completed: list[str] = []
    latest_job: dict[str, Any] | None = None
    latest_storage: Path | None = None
    missing_seen = False
    tail_node_dirs: list[Path] = []
    for label in prepared.expected_labels:
        node_dir = destination / "staged_eval" / label
        if not node_dir.exists():
            missing_seen = True
            continue
        if missing_seen:
            tail_node_dirs.append(node_dir)
            continue
        try:
            latest_job, latest_storage = validate_followup_node(
                prepared, destination, label
            )
        except FileNotFoundError:
            # A power loss can leave a node directory or bundle without the
            # matching full snapshot. It is not a valid anchor, so preserve it
            # as interrupted tail state and roll back to the prior strict node.
            missing_seen = True
            tail_node_dirs.append(node_dir)
            continue
        completed.append(label)

    if len(completed) == len(prepared.expected_labels):
        nodes = validate_completed_branch(prepared, cfg, destination)
        existing["status"] = "completed"
        existing["completed_at"] = existing.get("completed_at") or datetime.now().isoformat(
            timespec="seconds"
        )
        existing["failed_at"] = ""
        existing["error"] = ""
        existing["actual_labels"] = list(prepared.expected_labels)
        existing["actual_nodes"] = nodes
        write_json_atomic(manifest_path, existing)
        return existing, None

    if latest_job is None:
        resume_step = prepared.source_step
        resume_label = prepared.source_label
        resume_snapshot = destination / baseline_snapshot_name(prepared.source_time)
        storage_source = prepared.source_storage
        conversation = prepared.source_job.get("conversation", {}) or {}
    else:
        resume_step = int(latest_job["step_no"])
        resume_label = completed[-1]
        resume_snapshot = snapshot_path_for_name(destination, latest_job.get("snapshot_name"))
        storage_source = latest_storage
        conversation = latest_job.get("conversation", {}) or {}

    snapshot = load_json(resume_snapshot)
    if int(snapshot.get("step", -1)) != resume_step:
        raise ValueError(f"partial resume snapshot step mismatch: {resume_snapshot}")
    backup_path, archived_snapshots, archived_nodes = restore_partial_runtime(
        destination,
        storage_source,
        conversation,
        resume_snapshot,
        tuple(tail_node_dirs),
    )
    history = existing.get("resume_history")
    if not isinstance(history, list):
        history = []
    history.append(
        {
            "at": datetime.now().isoformat(timespec="seconds"),
            "from_label": resume_label,
            "from_step": resume_step,
            "completed_labels": list(completed),
            "runtime_backup": backup_path,
            "root_snapshots_after_resume_point": archived_snapshots,
            "staged_eval_nodes_after_resume_point": archived_nodes,
        }
    )
    existing["resume_history"] = history
    existing["status"] = "running"
    existing["failed_at"] = ""
    existing["error"] = ""
    write_json_atomic(manifest_path, existing)
    return existing, PartialResume(resume_step, resume_label, tuple(completed))


def install_branch(
    prepared: PreparedSource,
    cfg: FollowupConfig,
    overlay: Mapping[str, Any],
) -> tuple[Path, dict[str, Any]]:
    derived_runtime, cleanup = derive_followup_runtime(prepared, cfg, overlay)
    manifest = initial_followup_manifest(prepared, cfg, overlay, derived_runtime, cleanup)
    request = manifest["request"]
    destination = cfg.checkpoints_root / prepared.destination_run
    if destination.exists():
        if completed_destination_matches(destination, request):
            return destination, load_json(destination / FOLLOWUP_MANIFEST_FILENAME)
        if not cfg.force:
            raise FileExistsError(
                f"destination exists but is not the same completed follow-up: {destination}"
            )
        archive_existing(destination, cfg.checkpoints_root)

    cfg.checkpoints_root.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix=f".{prepared.destination_run}-", dir=cfg.checkpoints_root))
    try:
        shutil.copytree(prepared.source_storage, temp_dir / "storage")
        write_json_atomic(temp_dir / "conversation.json", prepared.source_job.get("conversation", {}) or {})
        write_json_atomic(temp_dir / baseline_snapshot_name(prepared.source_time), derived_runtime)
        write_json_atomic(
            temp_dir / CONTROLLER_MANIFEST_FILENAME,
            derived_controller_manifest(prepared, derived_runtime),
        )
        write_json_atomic(temp_dir / FOLLOWUP_MANIFEST_FILENAME, manifest)
        os.replace(temp_dir, destination)
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise
    return destination, manifest


def validate_completed_branch(
    prepared: PreparedSource,
    cfg: FollowupConfig,
    destination: Path,
) -> list[dict[str, Any]]:
    snapshots = sorted(destination.glob("simulate-*.json"))
    if not snapshots:
        raise ValueError(f"follow-up produced no full snapshot: {destination}")
    final_runtime = load_json(snapshots[-1])
    target_step = prepared.source_step + cfg.steps
    if int(final_runtime.get("step", -1)) != target_step:
        raise ValueError(
            f"follow-up final step mismatch: expected={target_step} actual={final_runtime.get('step')}"
        )
    nodes: list[dict[str, Any]] = []
    for label in prepared.expected_labels:
        job, _storage = validate_followup_node(prepared, destination, label)
        node_dir = destination / "staged_eval" / label
        metadata = load_json(node_dir / "metadata.json")
        manifest_path = node_dir / "snapshot_manifest.json"
        bundle_manifest = load_json(manifest_path)
        nodes.append(
            {
                "trigger_label": label,
                "step_no": int(job.get("step_no", 0) or 0),
                "sim_time": str(job.get("sim_time", "") or ""),
                "snapshot_name": str(job.get("snapshot_name", "") or ""),
                "completed_session_count": int(
                    metadata.get("completed_session_count", 0) or 0
                ),
                "snapshot_manifest_sha256": canonical_json_sha256(bundle_manifest),
            }
        )
    return nodes


def run_start(prepared: PreparedSource, cfg: FollowupConfig, destination: Path) -> None:
    command = [
        sys.executable,
        str(BASE_DIR / "start.py"),
        "--name",
        prepared.destination_run,
        "--resume",
        "--step",
        str(cfg.steps),
        "--stride",
        str(prepared.source_stride),
        "--verbose",
        cfg.verbose,
        "--log",
        "followup.log",
    ]
    subprocess.run(command, cwd=BASE_DIR, check=True)


def run_partial_resume(
    prepared: PreparedSource,
    cfg: FollowupConfig,
    destination: Path,
    resume: PartialResume,
) -> None:
    remaining_steps = prepared.source_step + cfg.steps - resume.step
    if remaining_steps <= 0:
        raise ValueError(
            "partial follow-up resume has no remaining steps: "
            f"step={resume.step} target={prepared.source_step + cfg.steps}"
        )
    command = [
        sys.executable,
        str(BASE_DIR / "start.py"),
        "--name",
        prepared.destination_run,
        "--resume",
        "--step",
        str(remaining_steps),
        "--stride",
        str(prepared.source_stride),
        "--verbose",
        cfg.verbose,
        "--log",
        "followup.log",
    ]
    subprocess.run(command, cwd=BASE_DIR, check=True)


def execute_one(
    prepared: PreparedSource,
    cfg: FollowupConfig,
    overlay: Mapping[str, Any],
) -> dict[str, Any]:
    destination = cfg.checkpoints_root / prepared.destination_run
    partial_resume: PartialResume | None = None
    if (
        destination.exists()
        and not cfg.force
        and not completed_destination_matches(
            destination, request_payload(prepared, cfg, overlay)
        )
        and cfg.resume_partial
    ):
        derived_runtime, cleanup = derive_followup_runtime(prepared, cfg, overlay)
        expected_manifest = initial_followup_manifest(
            prepared, cfg, overlay, derived_runtime, cleanup
        )
        manifest, partial_resume = prepare_partial_resume(
            prepared, cfg, destination, expected_manifest
        )
        if partial_resume is None:
            return {"status": "skipped", "prepared": prepared, "manifest": manifest}
    else:
        destination, manifest = install_branch(prepared, cfg, overlay)
    if str(manifest.get("status", "") or "") == "completed":
        validate_completed_branch(prepared, cfg, destination)
        return {"status": "skipped", "prepared": prepared, "manifest": manifest}
    manifest["status"] = "running"
    if partial_resume is None:
        manifest["started_at"] = datetime.now().isoformat(timespec="seconds")
    else:
        manifest["resumed_at"] = datetime.now().isoformat(timespec="seconds")
    write_json_atomic(destination / FOLLOWUP_MANIFEST_FILENAME, manifest)
    try:
        if partial_resume is None:
            run_start(prepared, cfg, destination)
        else:
            run_partial_resume(prepared, cfg, destination, partial_resume)
        nodes = validate_completed_branch(prepared, cfg, destination)
        manifest["status"] = "completed"
        manifest["completed_at"] = datetime.now().isoformat(timespec="seconds")
        manifest["actual_labels"] = [item["trigger_label"] for item in nodes]
        manifest["actual_nodes"] = nodes
        write_json_atomic(destination / FOLLOWUP_MANIFEST_FILENAME, manifest)
        return {"status": "completed", "prepared": prepared, "manifest": manifest}
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["failed_at"] = datetime.now().isoformat(timespec="seconds")
        manifest["error"] = str(exc)
        write_json_atomic(destination / FOLLOWUP_MANIFEST_FILENAME, manifest)
        raise


def print_plan(prepared: PreparedSource, cfg: FollowupConfig, overlay: Mapping[str, Any]) -> None:
    command = [
        sys.executable,
        "start.py",
        "--name",
        prepared.destination_run,
        "--resume",
        "--step",
        str(cfg.steps),
        "--stride",
        str(prepared.source_stride),
        "--verbose",
        cfg.verbose,
        "--log",
        "followup.log",
    ]
    print(f"[DRY-RUN] source condition: {prepared.spec.condition.get('condition_name', '')}")
    print(f"[DRY-RUN] source node: {prepared.spec.node_dir}")
    print(f"[DRY-RUN] destination run: {prepared.destination_run}")
    print(f"[DRY-RUN] source/target step: {prepared.source_step}/{prepared.source_step + cfg.steps}")
    print(f"[DRY-RUN] expected labels: {','.join(prepared.expected_labels)}")
    print("[DRY-RUN] no-intervention overlay:\n" + json.dumps(overlay, ensure_ascii=False, indent=2))
    print("[DRY-RUN] start.py command: " + " ".join(command))


def build_followup_condition(result: Mapping[str, Any], cfg: FollowupConfig) -> dict[str, Any]:
    prepared: PreparedSource = result["prepared"]
    manifest = result["manifest"]
    source_condition = prepared.spec.condition
    destination_manifest_path = (
        cfg.checkpoints_root / prepared.destination_run / CONTROLLER_MANIFEST_FILENAME
    )
    controller_manifest = load_json(destination_manifest_path)
    evaluations = []
    node_by_label = {
        str(item.get("trigger_label", "") or ""): item
        for item in manifest.get("actual_nodes", [])
        if isinstance(item, dict)
    }
    for label in prepared.expected_labels:
        node = node_by_label.get(label, {})
        evaluations.append(
            {
                "trigger_label": label,
                "completed_session_count": int(
                    node.get(
                        "completed_session_count",
                        prepared.source_metadata.get("completed_session_count", 0),
                    )
                    or 0
                ),
                "sim_time": str(node.get("sim_time", "") or ""),
                "snapshot_name": str(node.get("snapshot_name", "") or ""),
                "source": "staged_eval_snapshot",
                "evaluation_executed": False,
                "scales": {},
            }
        )
    payload = {
        "condition_name": str(source_condition.get("condition_name", "") or ""),
        "run_name": prepared.destination_run,
        "variant": str(source_condition.get("variant", "") or ""),
        "group": str(source_condition.get("group", "") or ""),
        "severity": str(source_condition.get("severity", "") or ""),
        "condition_key": str(controller_manifest.get("condition_key", "") or ""),
        "controller_manifest_path": str(destination_manifest_path),
        "controller_manifest": controller_manifest,
        "evaluations": evaluations,
        "trigger_labels": list(prepared.expected_labels),
    }
    for key in IDENTITY_KEYS:
        payload[key] = copy.deepcopy(controller_manifest.get(key))
    return payload


def write_summary(cfg: FollowupConfig, results: list[dict[str, Any]]) -> Path:
    conditions = sorted(
        (build_followup_condition(item, cfg) for item in results),
        key=lambda item: str(item.get("condition_name", "") or ""),
    )
    labels: list[str] = []
    for condition in conditions:
        for label in condition.get("trigger_labels", []):
            if label not in labels:
                labels.append(label)
    payload = {
        "schema_version": FOLLOWUP_SCHEMA_VERSION,
        "artifact_kind": "post_sim_followup_summary",
        "batch_name": cfg.name,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "conditions": conditions,
        "trigger_labels": labels,
        "warnings": [],
    }
    output = cfg.reports_dir / f"{safe_token(cfg.name, 'follow-up name')}_summary.json"
    write_json_atomic(output, payload)
    return output


def main() -> None:
    args = parse_args()
    if args.steps <= 0 or args.interval <= 0 or args.max_parallel <= 0:
        raise SystemExit("--steps, --interval and --max-parallel must be positive integers")
    cfg = FollowupConfig(
        name=safe_token(args.name, "follow-up name"),
        steps=args.steps,
        interval=args.interval,
        max_parallel=args.max_parallel,
        dry_run=bool(args.dry_run),
        force=bool(args.force),
        resume_partial=bool(args.resume_partial),
        verbose=str(args.verbose or "info"),
    )
    overlay = load_json(cfg.overlay_path)
    if args.source_summary:
        specs = source_specs_from_summary(
            resolve_path(args.source_summary),
            safe_token(args.source_label, "source label"),
            cfg.checkpoints_root,
        )
    else:
        specs = source_specs_from_nodes(args.source_node or [], cfg.checkpoints_root)
    ensure_unique_conditions(specs)
    prepared = [prepare_source(spec, cfg) for spec in specs]

    if cfg.dry_run:
        for item in prepared:
            print_plan(item, cfg, overlay)
        print("[DRY-RUN] no checkpoint, summary, or directory was created")
        return

    results: list[dict[str, Any]] = []
    failures: list[str] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(cfg.max_parallel, len(prepared))) as executor:
        futures = {executor.submit(execute_one, item, cfg, overlay): item for item in prepared}
        for future in concurrent.futures.as_completed(futures):
            item = futures[future]
            try:
                result = future.result()
                results.append(result)
                print(f"[{result['status'].upper()}] {item.destination_run}")
            except Exception as exc:
                failures.append(f"{item.destination_run}: {exc}")
                print(f"[FAILED] {item.destination_run}: {exc}", file=sys.stderr)
    if failures:
        raise SystemExit("follow-up failed:\n" + "\n".join(failures))
    summary = write_summary(cfg, results)
    print(f"[Done] follow-up summary: {summary}")


if __name__ == "__main__":
    main()
