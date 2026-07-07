#!/usr/bin/env python3
"""Repeat archived staged/POST scale evaluation and compare with the original report.

Typical usage:
    python3 runshells/run_archived_repeat_scale_eval.py \
      --archive-results-root results/0630/results-batch-0628-KBD2-KBD3-G1-SEV/results \
      --condition Counsel-KBD2-G1-SEV --labels T0 --repeat 1

The script never overwrites the original archived experiment data. Repeated
answers/scores are written under experiment_data/repeat_scale_eval/<name>/,
and a batch-summary-like report is written under experiment_data/reports/.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import copy
import json
import math
import shutil
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from run_one_experiment import (
    BASE_DIR,
    POST_EVAL_TIMEOUT_SECONDS,
    SCALE_SCORING_DIR,
    SCALES,
    SCORE_WORKER_SCRIPT,
    WORKER_PYTHON,
)
from run_batch_experiment import (
    GROUPS,
    SEVERITIES,
    SEVERITY_SHORT_NAMES,
    VARIANTS,
    VARIANT_SHORT_NAMES,
    aggregate_dimension_trajectory,
    aggregate_final_deltas,
    apply_trajectory_deltas,
    build_group_severity_matrix,
    build_scale_score_validation,
    build_scale_snapshot,
    extract_scale_severity,
    extract_scale_total,
    expected_scale_severity,
    format_delta,
    format_number,
    load_json_file,
    mean,
    ordered_trigger_labels,
    render_condition_trajectory_markdown,
    render_dimension_trajectory_markdown,
    render_final_delta_markdown,
    render_group_severity_matrix_markdown,
    render_scale_score_validation_markdown,
    resolve_conditions,
    trigger_sort_key,
    write_json_file,
)


DEFAULT_LABELS = ["T0", "session_4", "session_8", "session_12", "session_16", "T4", "POST"]
REPEAT_OUTPUT_SUBDIR = "repeat_scale_eval"
STAGED_WORKER_SCRIPT = BASE_DIR / "runshells" / "run_staged_eval_worker.py"
CHECKPOINT_RESULTS_PREFIX = "/workspace/project/results"

# ============================================================
# ↓↓↓ 可调参数：直接修改这里即可；命令行 --xxx 会覆盖这里 ↓↓↓
# ============================================================

# 存档中的 results 目录；留空时必须通过 --archive-results-root 传入
ARCHIVE_RESULTS_ROOT = ""

# 原始 *_summary.json；留空时自动从 archive results 的 reports 目录选择
ORIGINAL_SUMMARY = None

# 输出批次名；留空则自动生成，例如 repeat-scale-0630-1530
NAME = None

# 每个评估点完整重复评估次数
REPEAT = 3

# 评估点；可填字符串 "T0,POST"，也可直接使用 ",".join(DEFAULT_LABELS)
LABELS = ",".join(DEFAULT_LABELS)

# 只处理指定 condition；留空表示处理原始 summary 中全部 condition
CONDITIONS: list[str] = []

# 并行任务数
MAX_PARALLEL = 1

# 是否覆盖已有重复输出
FORCE = False

# 是否只打印计划，不实际执行
DRY_RUN = False

# 是否只基于已有重复结果重新生成报告
REPORT_ONLY = False

# 是否检测条目评分稳定性，并只对不稳定条目追加补跑
STABILITY_RERUN = True

# 每个不稳定条目最多追加补跑轮数
MAX_EXTRA_REPEAT = 4

# 条目分数极差达到该值时判为不稳定；1 表示重复分数不完全一致即不稳定
STABILITY_RANGE_THRESHOLD = 1

# 是否把目标患者动态抑郁状态重置为初始 depression_config 状态后复评
RESET_TARGET_DEPRESSION_STATE = False

# 复评输出使用的新 group 标签，例如 "g8"；留空则沿用原始 condition
OUTPUT_GROUP = ""

# ============================================================
# ↑↑↑ 可调参数：直接修改这里即可；命令行 --xxx 会覆盖这里 ↑↑↑
# ============================================================


@dataclass(frozen=True)
class RuntimeConfig:
    archive_results_root: Path
    original_summary: Path
    name: str
    repeat: int
    labels: list[str]
    conditions: list[str]
    max_parallel: int
    force: bool
    dry_run: bool
    report_only: bool
    stability_rerun: bool
    max_extra_repeat: int
    stability_range_threshold: int
    reset_target_depression_state: bool
    output_group: str


@dataclass(frozen=True)
class RepeatTask:
    condition: dict[str, Any]
    label: str
    repeat_idx: int


@dataclass(frozen=True)
class StabilityRerunTask:
    condition: dict[str, Any]
    label: str
    repeat_idx: int
    scale_item_ids: dict[str, list[int]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Repeat archived T0/session/T4/POST scale evaluations")
    archive_results_default = str(ARCHIVE_RESULTS_ROOT or "").strip()
    parser.add_argument(
        "--archive-results-root",
        default=archive_results_default or None,
        required=not bool(archive_results_default),
        help="存档中的 results 目录",
    )
    parser.add_argument("--original-summary", default=ORIGINAL_SUMMARY, help="原始 *_summary.json；默认自动从 reports 目录选择")
    parser.add_argument("--name", default=NAME, help="输出批次名，默认 repeat-scale-<MMdd-HHmm>")
    parser.add_argument("--repeat", type=int, default=REPEAT, help="每个评估点重复次数")
    parser.add_argument("--labels", default=LABELS, help="逗号分隔评估点")
    parser.add_argument("--condition", action="append", default=None, help="只处理指定 condition，可重复传入")
    parser.add_argument("--max-parallel", type=int, default=MAX_PARALLEL, help="并行任务数")
    parser.add_argument("--force", action=argparse.BooleanOptionalAction, default=FORCE, help="覆盖已有重复输出")
    parser.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=DRY_RUN, help="只打印计划，不实际执行")
    parser.add_argument("--report-only", action=argparse.BooleanOptionalAction, default=REPORT_ONLY, help="只基于已有重复结果重新生成报告")
    parser.add_argument("--stability-rerun", action=argparse.BooleanOptionalAction, default=STABILITY_RERUN, help="检测条目评分稳定性，并只对不稳定条目追加补跑")
    parser.add_argument("--max-extra-repeat", type=int, default=MAX_EXTRA_REPEAT, help="每个不稳定条目最多追加补跑轮数")
    parser.add_argument("--stability-range-threshold", type=int, default=STABILITY_RANGE_THRESHOLD, help="条目分数极差达到该值时判为不稳定")
    parser.add_argument("--reset-target-depression-state", action=argparse.BooleanOptionalAction, default=RESET_TARGET_DEPRESSION_STATE, help="把目标患者动态抑郁状态重置为初始 depression_config 状态后复评")
    parser.add_argument("--output-group", default=OUTPUT_GROUP, help="复评输出使用的新 group 标签，例如 G8；留空则沿用原始 condition")
    return parser.parse_args()


def generate_name() -> str:
    return f"repeat-scale-{datetime.now().strftime('%m%d-%H%M')}"


def normalize_labels(raw: str) -> list[str]:
    labels = [item.strip() for item in str(raw or "").split(",") if item.strip()]
    return labels or list(DEFAULT_LABELS)


def resolve_original_summary(archive_results_root: Path, raw: str | None) -> Path:
    if raw:
        path = Path(raw).expanduser()
        if not path.is_absolute():
            path = BASE_DIR / path
        return path.resolve()

    reports_dir = archive_results_root / "experiment_data" / "reports"
    candidates = sorted(
        path
        for path in reports_dir.glob("*_summary.json")
        if not path.name.startswith("repeat-scale-")
    )
    if not candidates:
        return synthesize_original_summary(archive_results_root)
    if len(candidates) > 1:
        candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
        print(f"[WARN] 检测到多个原始 summary，默认使用最新: {candidates[0]}")
    return candidates[0].resolve()


def empty_original_scale_payload(scale_name: str) -> dict[str, Any]:
    return {
        "total_score": None,
        "severity": "—",
        "scored_file": "",
        "source": "synthesized_original_summary",
        "scale": scale_name,
    }


def condition_metadata_from_name(condition_name: str) -> dict[str, str]:
    try:
        condition = resolve_conditions(condition_name)[0]
    except Exception:
        return {"variant": "", "group": "", "severity": ""}
    return {
        "variant": condition.variant,
        "group": condition.group,
        "severity": condition.severity,
    }


def synthesize_evaluations_from_checkpoint(checkpoint_dir: Path) -> list[dict[str, Any]]:
    records = []
    index = load_optional_json(checkpoint_dir / "staged_eval" / "index.json") or {}
    for record in index.get("records", []) or []:
        if not isinstance(record, dict) or record.get("status") != "ok":
            continue
        label = str(record.get("trigger_label", "") or "")
        if not label:
            continue
        records.append(
            {
                "trigger_label": label,
                "completed_session_count": int(record.get("completed_session_count", 0) or 0),
                "sim_time": str(record.get("sim_time", "") or ""),
                "snapshot_name": str(record.get("snapshot_name", "") or ""),
                "source": "synthesized_staged_eval",
                "metadata_path": str(checkpoint_dir / "staged_eval" / label / "metadata.json"),
                "scales": {
                    scale_name: empty_original_scale_payload(scale_name)
                    for scale_name in SCALES
                },
            }
        )
    return records


def synthesize_original_summary(archive_results_root: Path) -> Path:
    batch_state_root = archive_results_root / "experiment_data" / "batch_state"
    state_paths = sorted(batch_state_root.glob("*/*.json"))
    if not state_paths:
        raise FileNotFoundError(
            f"未找到原始 summary JSON，也无法从 batch_state 合成: {archive_results_root}"
        )

    conditions: list[dict[str, Any]] = []
    warnings: list[str] = []
    for state_path in state_paths:
        state = load_optional_json(state_path)
        if not state:
            warnings.append(f"无法读取 batch_state: {state_path}")
            continue
        condition_name = str(state.get("condition_name", "") or "")
        run_name = str(state.get("run_name", "") or "")
        if not condition_name or not run_name:
            warnings.append(f"batch_state 缺少 condition_name/run_name: {state_path}")
            continue
        checkpoint_dir = archive_results_root / "checkpoints" / run_name
        if not checkpoint_dir.is_dir():
            warnings.append(f"checkpoint 不存在，跳过 {condition_name}: {checkpoint_dir}")
            continue
        metadata = condition_metadata_from_name(condition_name)
        evaluations = synthesize_evaluations_from_checkpoint(checkpoint_dir)
        conditions.append(
            {
                "condition_name": condition_name,
                "run_name": run_name,
                "run_dir": str(archive_results_root / "experiment_data" / run_name),
                "variant": metadata["variant"],
                "group": metadata["group"],
                "severity": metadata["severity"],
                "evaluations": evaluations,
                "final_deltas": {},
            }
        )

    if not conditions:
        raise FileNotFoundError(
            f"未找到原始 summary JSON，且 batch_state 中没有可用 condition: {batch_state_root}"
        )

    summary_dir = Path("/tmp") / "generative_agents_archived_repeat_summaries"
    summary_dir.mkdir(parents=True, exist_ok=True)
    safe_archive_name = archive_results_root.parent.name.replace("/", "_")
    summary_path = summary_dir / f"{safe_archive_name}_synthesized_original_summary.json"
    payload = {
        "batch_name": "synthesized-archive-summary",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "summary_only": True,
        "conditions": conditions,
        "warnings": [
            "原始 batch summary 缺失；此文件由 run_archived_repeat_scale_eval.py 从 batch_state/checkpoints 自动合成。",
            "原始量表分数留空，因此重复评估报告中的“与原报告差异”只适合检查重复结果是否完整。",
            *warnings,
        ],
        "trigger_labels": ordered_trigger_labels(conditions),
    }
    write_json_file(summary_path, payload)
    print(f"[WARN] 未找到原始 summary，已自动合成: {summary_path}")
    return summary_path.resolve()


def resolve_runtime_config(args: argparse.Namespace) -> RuntimeConfig:
    archive_results_root = Path(args.archive_results_root).expanduser()
    if not archive_results_root.is_absolute():
        archive_results_root = BASE_DIR / archive_results_root
    archive_results_root = archive_results_root.resolve()
    if not archive_results_root.is_dir():
        raise FileNotFoundError(f"archive results root not found: {archive_results_root}")

    original_summary = resolve_original_summary(archive_results_root, args.original_summary)
    if not original_summary.is_file():
        raise FileNotFoundError(f"original summary not found: {original_summary}")

    repeat = max(1, int(args.repeat or 1))
    max_parallel = max(1, int(args.max_parallel or 1))
    return RuntimeConfig(
        archive_results_root=archive_results_root,
        original_summary=original_summary,
        name=str(args.name or "").strip() or generate_name(),
        repeat=repeat,
        labels=normalize_labels(args.labels),
        conditions=list(args.condition if args.condition is not None else CONDITIONS),
        max_parallel=max_parallel,
        force=bool(args.force),
        dry_run=bool(args.dry_run),
        report_only=bool(args.report_only),
        stability_rerun=bool(args.stability_rerun),
        max_extra_repeat=max(0, int(args.max_extra_repeat or 0)),
        stability_range_threshold=max(1, int(args.stability_range_threshold or 1)),
        reset_target_depression_state=bool(args.reset_target_depression_state),
        output_group=str(args.output_group or "").strip().lower(),
    )


def experiment_data_root(cfg: RuntimeConfig) -> Path:
    return cfg.archive_results_root / "experiment_data"


def checkpoints_root(cfg: RuntimeConfig) -> Path:
    return cfg.archive_results_root / "checkpoints"


def repeat_root(cfg: RuntimeConfig) -> Path:
    return experiment_data_root(cfg) / REPEAT_OUTPUT_SUBDIR / cfg.name


def reports_dir(cfg: RuntimeConfig) -> Path:
    return experiment_data_root(cfg) / "reports"


def condition_output_dir(cfg: RuntimeConfig, condition_name: str) -> Path:
    return repeat_root(cfg) / condition_name


def repeat_label_dir(cfg: RuntimeConfig, condition_name: str, repeat_idx: int, label: str) -> Path:
    return condition_output_dir(cfg, condition_name) / f"r{repeat_idx:02d}" / label


def to_display_path(path: Path) -> str:
    try:
        return str(path.relative_to(BASE_DIR))
    except ValueError:
        return str(path)


def report_group_values(cfg: RuntimeConfig) -> list[str]:
    values = list(GROUPS)
    output_group = str(cfg.output_group or "").strip().lower()
    if output_group and output_group not in values:
        values.append(output_group)
    return values


def build_repeat_group_severity_matrix(cfg: RuntimeConfig, results: list[dict]) -> dict:
    group_values = report_group_values(cfg)
    matrix = {}
    for scale_name in SCALES:
        scale_payload = {}
        for severity in SEVERITIES:
            row = {}
            for group in group_values:
                values = []
                for result in results:
                    if result.get("severity") != severity or result.get("group") != group:
                        continue
                    value = result.get("final_deltas", {}).get(scale_name)
                    if value is not None:
                        values.append(value)
                row[group] = mean(values)
            scale_payload[severity] = row
        matrix[scale_name] = scale_payload
    return matrix


def print_effective_config(cfg: RuntimeConfig, original_summary: dict[str, Any]) -> None:
    selected = selected_conditions(cfg, original_summary)
    task_count = len(selected) * len(cfg.labels) * cfg.repeat
    score_count = task_count * len(SCALES)
    print("==========================================")
    print(" 存档重复量表评估配置")
    print("==========================================")
    print(f"  archive results: {cfg.archive_results_root}")
    print(f"  original summary: {cfg.original_summary}")
    print(f"  output name:     {cfg.name}")
    print(f"  conditions:      {len(selected)}")
    print(f"  labels:          {', '.join(cfg.labels)}")
    print(f"  repeat:          {cfg.repeat}")
    print(f"  tasks:           {task_count} answer jobs, {score_count} score jobs")
    print(f"  max_parallel:    {cfg.max_parallel}")
    print(f"  force:           {cfg.force}")
    print(f"  dry-run:         {cfg.dry_run}")
    print(f"  report-only:     {cfg.report_only}")
    print(f"  stability-rerun: {cfg.stability_rerun}")
    print(f"  reset depression:{cfg.reset_target_depression_state}")
    print(f"  output group:    {cfg.output_group or '(source)'}")
    if cfg.stability_rerun:
        print(f"  max-extra-repeat:{cfg.max_extra_repeat}")
        print(f"  item range thres:{cfg.stability_range_threshold}")
    print("==========================================")


def selected_conditions(cfg: RuntimeConfig, original_summary: dict[str, Any]) -> list[dict[str, Any]]:
    conditions = original_summary.get("conditions", [])
    if not isinstance(conditions, list):
        return []
    wanted = set(cfg.conditions)
    source_selected = [
        item
        for item in conditions
        if isinstance(item, dict) and (not wanted or str(item.get("condition_name", "")) in wanted)
    ]
    found = {str(item.get("condition_name", "")) for item in source_selected}
    missing = sorted(wanted - found)
    if missing:
        raise ValueError(f"原始 summary 中找不到 condition: {', '.join(missing)}")
    return [condition_for_output(cfg, item) for item in source_selected]


def condition_for_output(cfg: RuntimeConfig, condition: dict[str, Any]) -> dict[str, Any]:
    payload = copy.deepcopy(condition)
    output_group = str(cfg.output_group or "").strip().lower()
    if not output_group:
        return payload

    source_name = str(payload.get("condition_name", "") or "")
    source_group = str(payload.get("group", "") or "")
    payload["source_condition_name"] = source_name
    payload["source_group"] = source_group
    payload["group"] = output_group
    payload["condition_name"] = build_output_condition_name(payload, output_group)
    return payload


def build_output_condition_name(condition: dict[str, Any], output_group: str) -> str:
    variant = str(condition.get("variant", "") or "").strip()
    severity = str(condition.get("severity", "") or "").strip()
    group_token = str(output_group or "").strip().upper()
    if variant in VARIANT_SHORT_NAMES and severity in SEVERITY_SHORT_NAMES:
        return "Counsel-{}-{}-{}".format(
            VARIANT_SHORT_NAMES[variant],
            group_token,
            SEVERITY_SHORT_NAMES[severity],
        )

    source_name = str(condition.get("source_condition_name", "") or condition.get("condition_name", "") or "")
    tokens = source_name.split("-")
    if len(tokens) == 4 and tokens[0].upper() == "COUNSEL":
        return "-".join([tokens[0], tokens[1], group_token, tokens[3]])
    if len(tokens) == 3 and tokens[0].upper() == "COUNSEL":
        return "-".join([tokens[0], group_token, tokens[2]])
    return "{}-{}".format(source_name or "Counsel", group_token)


def load_optional_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = load_json_file(path)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if isinstance(item, dict):
                rows.append(item)
    return rows


def latest_snapshot_name(checkpoint_dir: Path) -> str:
    snapshots = sorted(
        path.name
        for path in checkpoint_dir.iterdir()
        if path.is_file() and path.name.startswith("simulate-") and path.name.endswith(".json")
    )
    if not snapshots:
        raise FileNotFoundError(f"no simulate-*.json found under: {checkpoint_dir}")
    return snapshots[-1]


def rewrite_path_string(value: str, cfg: RuntimeConfig) -> str:
    if not value:
        return value
    text = str(value)
    archive_root = str(cfg.archive_results_root)
    if text.startswith(CHECKPOINT_RESULTS_PREFIX):
        return archive_root + text[len(CHECKPOINT_RESULTS_PREFIX):]
    if text.startswith("results/"):
        return archive_root + text[len("results"):]
    return text


def rewrite_paths(value: Any, cfg: RuntimeConfig) -> Any:
    if isinstance(value, dict):
        return {key: rewrite_paths(item, cfg) for key, item in value.items()}
    if isinstance(value, list):
        return [rewrite_paths(item, cfg) for item in value]
    if isinstance(value, str):
        return rewrite_path_string(value, cfg)
    return value


def load_conversation(checkpoint_dir: Path) -> dict[str, Any]:
    path = checkpoint_dir / "conversation.json"
    payload = load_optional_json(path)
    return payload or {}


def target_agent_for_condition(condition: dict[str, Any], checkpoint_dir: Path) -> str:
    for evaluation in condition.get("evaluations", []) or []:
        metadata_path = Path(str(evaluation.get("metadata_path", "") or ""))
        if metadata_path.name == "metadata.json":
            archive_metadata = checkpoint_dir / "staged_eval" / str(evaluation.get("trigger_label", "")) / "metadata.json"
            metadata = load_optional_json(archive_metadata)
            if metadata and metadata.get("target_agent"):
                return str(metadata["target_agent"])
    t0_job = load_optional_json(checkpoint_dir / "staged_eval" / "T0" / "job.json")
    if t0_job and t0_job.get("target_agent"):
        return str(t0_job["target_agent"])
    return "卡布达"


def reset_target_depression_state_if_needed(
    cfg: RuntimeConfig,
    runtime_config: dict[str, Any],
    target_agent: str,
) -> dict[str, Any]:
    if not cfg.reset_target_depression_state:
        return runtime_config
    if not isinstance(runtime_config, dict):
        return runtime_config

    agents = runtime_config.get("agents", {})
    if not isinstance(agents, dict):
        return runtime_config
    agent_cfg = agents.get(target_agent)
    if not isinstance(agent_cfg, dict):
        return runtime_config

    removed = bool("depression_dynamic_state" in agent_cfg)
    agent_cfg.pop("depression_dynamic_state", None)
    agent_cfg["_depression_state_reset"] = {
        "enabled": True,
        "mode": "initial_from_config",
        "removed_snapshot_state": removed,
        "target_agent": target_agent,
    }
    return runtime_config


def repeat_metadata_overrides(cfg: RuntimeConfig, condition: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    if condition.get("source_condition_name"):
        payload["source_condition_name"] = str(condition.get("source_condition_name", "") or "")
    if condition.get("source_group"):
        payload["source_group"] = str(condition.get("source_group", "") or "")
    if cfg.reset_target_depression_state:
        payload["depression_state_reset"] = "initial_from_config"
    if cfg.output_group:
        payload["output_group"] = str(cfg.output_group or "")
    return payload


def condition_evaluation(condition: dict[str, Any], label: str) -> dict[str, Any] | None:
    for evaluation in condition.get("evaluations", []) or []:
        if str(evaluation.get("trigger_label", "")) == label:
            return evaluation
    return None


def build_staged_job(
    cfg: RuntimeConfig,
    condition: dict[str, Any],
    label: str,
    repeat_idx: int,
    output_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    run_name = str(condition.get("run_name", "") or "").strip()
    checkpoint_dir = checkpoints_root(cfg) / run_name
    source_dir = checkpoint_dir / "staged_eval" / label
    source_job_path = source_dir / "job.json"
    job = load_optional_json(source_job_path)
    if job is None:
        raise FileNotFoundError(f"missing staged job: {source_job_path}")
    metadata = load_optional_json(source_dir / "metadata.json") or {}
    job = rewrite_paths(copy.deepcopy(job), cfg)
    target_agent = str(job.get("target_agent", "") or target_agent_for_condition(condition, checkpoint_dir))
    job["target_agent"] = target_agent
    runtime_config = job.get("runtime_config", {})
    if isinstance(runtime_config, dict):
        job["runtime_config"] = reset_target_depression_state_if_needed(
            cfg,
            runtime_config,
            target_agent,
        )
    job["trigger_dir"] = str(output_dir)
    job["worker_result_path"] = str(output_dir / "worker_result.json")
    job["tmp_root_parent"] = str(output_dir / "_tmp")
    job["storage_source_root"] = str(checkpoint_dir / "storage")
    job["cleanup_tmp_storage"] = True
    job["run_name"] = run_name
    return job, metadata


def build_post_job(
    cfg: RuntimeConfig,
    condition: dict[str, Any],
    repeat_idx: int,
    output_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    run_name = str(condition.get("run_name", "") or "").strip()
    checkpoint_dir = checkpoints_root(cfg) / run_name
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_dir}")
    snapshot = latest_snapshot_name(checkpoint_dir)
    runtime_config = rewrite_paths(load_json_file(checkpoint_dir / snapshot), cfg)
    target_agent = target_agent_for_condition(condition, checkpoint_dir)
    runtime_config = reset_target_depression_state_if_needed(
        cfg,
        runtime_config,
        target_agent,
    )
    evaluation = condition_evaluation(condition, "POST") or {}
    completed = int(evaluation.get("completed_session_count", 0) or 0)
    metadata = {
        "status": "ok",
        "trigger_label": "POST",
        "completed_session_count": completed,
        "sim_time": "",
        "snapshot_name": snapshot,
        "target_agent": target_agent,
        "source": "post_scale",
    }
    job = {
        "run_name": run_name,
        "trigger_label": "POST",
        "completed_session_count": completed,
        "step_no": 0,
        "sim_time": "",
        "snapshot_name": snapshot,
        "target_agent": target_agent,
        "scales": list(SCALES.keys()),
        "scale_question_files": {
            scale_name: scale_cfg["question_file"]
            for scale_name, scale_cfg in SCALES.items()
        },
        "runtime_config": runtime_config,
        "conversation": load_conversation(checkpoint_dir),
        "trigger_dir": str(output_dir),
        "worker_result_path": str(output_dir / "worker_result.json"),
        "storage_source_root": str(checkpoint_dir / "storage"),
        "tmp_root_parent": str(output_dir / "_tmp"),
        "cleanup_tmp_storage": True,
    }
    return job, metadata


def score_scale_answers(output_dir: Path, scale_name: str, *, dry_run: bool) -> None:
    answers_path = output_dir / f"{scale_name}_answered.jsonl"
    scored_path = output_dir / f"{scale_name}_scored.json"
    scoring_prompt = SCALE_SCORING_DIR / SCALES[scale_name]["scoring_prompt"]
    cmd = [
        WORKER_PYTHON,
        str(SCORE_WORKER_SCRIPT),
        "--answers",
        str(answers_path),
        "--scoring-prompt",
        str(scoring_prompt),
        "--output",
        str(scored_path),
    ]
    print(f"[RUN] {' '.join(cmd)}")
    if dry_run:
        return
    subprocess.run(cmd, cwd=BASE_DIR, check=True, timeout=POST_EVAL_TIMEOUT_SECONDS)


def item_scoring_prompt(scale_name: str, item_ids: list[int]) -> str:
    item_list = ", ".join(str(item_id) for item_id in sorted(item_ids))
    timeframe = "过去两周" if scale_name == "PHQ-9" else "近两周（包括今天）"
    return f"""你是一位专业、严谨、温和的“{scale_name}条目评分Agent”。
你的任务是只根据输入中的个别量表条目问答，对这些条目分别给出0-3分评分。

评分范围：
- 0分：无该症状或完全没有。
- 1分：轻度、有几天、略有变化。
- 2分：中度、超过一半天数、明显或经常。
- 3分：重度、几乎每天、总是、完全不能或不堪忍受。

要求：
1. 只评分输入中出现的条目，不要补全整套量表。
2. 只输出目标条目：{item_list}。
3. 每个条目必须给出唯一整数分数，范围为0-3。
4. 每个条目必须引用用户原文片段作为basis。
5. 时间范围按{timeframe}理解。

必须严格输出JSON，不要输出多余解释：
{{
  "scale": "{scale_name}",
  "item_scores": [
    {{"id": 条目id, "score": 分数, "basis": "依据用户原文片段"}}
  ]
}}
"""


def score_scale_items(output_dir: Path, scale_name: str, item_ids: list[int], *, dry_run: bool) -> None:
    answers_path = output_dir / f"{scale_name}_answered.jsonl"
    scored_path = output_dir / f"{scale_name}_item_scored.json"
    prompt_path = output_dir / f"{scale_name}_item_scoring_prompt.md"
    cmd = [
        WORKER_PYTHON,
        str(SCORE_WORKER_SCRIPT),
        "--answers",
        str(answers_path),
        "--scoring-prompt",
        str(prompt_path),
        "--output",
        str(scored_path),
    ]
    print(f"[RUN] {' '.join(cmd)}")
    if dry_run:
        return
    prompt_path.write_text(item_scoring_prompt(scale_name, item_ids), encoding="utf-8")
    subprocess.run(cmd, cwd=BASE_DIR, check=True, timeout=POST_EVAL_TIMEOUT_SECONDS)


def stability_task_complete(cfg: RuntimeConfig, task: StabilityRerunTask) -> bool:
    condition_name = str(task.condition.get("condition_name", ""))
    output_dir = repeat_label_dir(cfg, condition_name, task.repeat_idx, task.label)
    if not (output_dir / "metadata.json").is_file():
        return False
    return all(
        (output_dir / f"{scale_name}_item_scored.json").is_file()
        for scale_name in task.scale_item_ids
    )


def run_stability_rerun_task(cfg: RuntimeConfig, task: StabilityRerunTask) -> dict[str, Any]:
    condition = task.condition
    condition_name = str(condition.get("condition_name", ""))
    run_name = str(condition.get("run_name", "") or "").strip()
    output_dir = repeat_label_dir(cfg, condition_name, task.repeat_idx, task.label)
    checkpoint_dir = checkpoints_root(cfg) / run_name
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_dir}")

    if stability_task_complete(cfg, task) and not cfg.force:
        print(f"[SKIP] stability complete: {condition_name} r{task.repeat_idx:02d} {task.label}")
        return {
            "condition_name": condition_name,
            "label": task.label,
            "repeat": task.repeat_idx,
            "status": "skipped",
        }

    if cfg.force and output_dir.exists() and not cfg.dry_run:
        shutil.rmtree(output_dir)
    if not cfg.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    if task.label == "POST":
        job, source_metadata = build_post_job(cfg, condition, task.repeat_idx, output_dir)
    else:
        job, source_metadata = build_staged_job(cfg, condition, task.label, task.repeat_idx, output_dir)
    target_scales = sorted(task.scale_item_ids)
    job["scales"] = target_scales
    job["scale_item_ids"] = {
        scale_name: sorted({int(item_id) for item_id in item_ids})
        for scale_name, item_ids in task.scale_item_ids.items()
    }

    metadata = copy.deepcopy(source_metadata)
    metadata.update(
        {
            "repeat_batch_name": cfg.name,
            "repeat": task.repeat_idx,
            "condition_name": condition_name,
            "run_name": run_name,
            "trigger_label": task.label,
            "source_repeat_mode": "stability_adaptive_item_rerun",
            "scale_item_ids": job["scale_item_ids"],
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            **repeat_metadata_overrides(cfg, condition),
        }
    )

    job_path = output_dir / "job.json"
    if cfg.dry_run:
        print(f"[DRY-RUN] write item-rerun job: {job_path}")
        print(f"[DRY-RUN] run staged worker: {STAGED_WORKER_SCRIPT} --job {job_path}")
    else:
        write_json_file(job_path, job)
        cmd = [WORKER_PYTHON, str(STAGED_WORKER_SCRIPT), "--job", str(job_path)]
        print(f"[RUN] {' '.join(cmd)}")
        subprocess.run(cmd, cwd=BASE_DIR, check=True, timeout=POST_EVAL_TIMEOUT_SECONDS)

    for scale_name, item_ids in job["scale_item_ids"].items():
        score_scale_items(output_dir, scale_name, item_ids, dry_run=cfg.dry_run)

    if not cfg.dry_run:
        worker_result = load_optional_json(output_dir / "worker_result.json") or {}
        metadata["worker_result"] = worker_result
        write_json_file(output_dir / "metadata.json", metadata)
    return {
        "condition_name": condition_name,
        "label": task.label,
        "repeat": task.repeat_idx,
        "status": "ok",
    }


def task_complete(cfg: RuntimeConfig, task: RepeatTask) -> bool:
    condition_name = str(task.condition.get("condition_name", ""))
    output_dir = repeat_label_dir(cfg, condition_name, task.repeat_idx, task.label)
    if not (output_dir / "metadata.json").is_file():
        return False
    return all((output_dir / f"{scale_name}_scored.json").is_file() for scale_name in SCALES)


def run_repeat_task(cfg: RuntimeConfig, task: RepeatTask) -> dict[str, Any]:
    condition = task.condition
    condition_name = str(condition.get("condition_name", ""))
    run_name = str(condition.get("run_name", "") or "").strip()
    output_dir = repeat_label_dir(cfg, condition_name, task.repeat_idx, task.label)
    checkpoint_dir = checkpoints_root(cfg) / run_name
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_dir}")

    if task_complete(cfg, task) and not cfg.force:
        print(f"[SKIP] complete: {condition_name} r{task.repeat_idx:02d} {task.label}")
        return {
            "condition_name": condition_name,
            "label": task.label,
            "repeat": task.repeat_idx,
            "status": "skipped",
        }

    if cfg.force and output_dir.exists() and not cfg.dry_run:
        shutil.rmtree(output_dir)
    if not cfg.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    if task.label == "POST":
        job, source_metadata = build_post_job(cfg, condition, task.repeat_idx, output_dir)
    else:
        job, source_metadata = build_staged_job(cfg, condition, task.label, task.repeat_idx, output_dir)

    metadata = copy.deepcopy(source_metadata)
    metadata.update(
        {
            "repeat_batch_name": cfg.name,
            "repeat": task.repeat_idx,
            "condition_name": condition_name,
            "run_name": run_name,
            "trigger_label": task.label,
            "source_repeat_mode": "rerun_answers_and_scores",
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            **repeat_metadata_overrides(cfg, condition),
        }
    )

    job_path = output_dir / "job.json"
    if cfg.dry_run:
        print(f"[DRY-RUN] write job: {job_path}")
        print(f"[DRY-RUN] run staged worker: {STAGED_WORKER_SCRIPT} --job {job_path}")
    else:
        write_json_file(job_path, job)
        cmd = [WORKER_PYTHON, str(STAGED_WORKER_SCRIPT), "--job", str(job_path)]
        print(f"[RUN] {' '.join(cmd)}")
        subprocess.run(cmd, cwd=BASE_DIR, check=True, timeout=POST_EVAL_TIMEOUT_SECONDS)

    for scale_name in SCALES:
        score_scale_answers(output_dir, scale_name, dry_run=cfg.dry_run)

    if not cfg.dry_run:
        worker_result = load_optional_json(output_dir / "worker_result.json") or {}
        metadata["worker_result"] = worker_result
        write_json_file(output_dir / "metadata.json", metadata)
    return {
        "condition_name": condition_name,
        "label": task.label,
        "repeat": task.repeat_idx,
        "status": "ok",
    }


def build_tasks(cfg: RuntimeConfig, original_summary: dict[str, Any]) -> list[RepeatTask]:
    return [
        RepeatTask(condition=condition, label=label, repeat_idx=repeat_idx)
        for condition in selected_conditions(cfg, original_summary)
        for label in cfg.labels
        for repeat_idx in range(1, cfg.repeat + 1)
    ]


def run_tasks(cfg: RuntimeConfig, original_summary: dict[str, Any]) -> list[dict[str, Any]]:
    tasks = build_tasks(cfg, original_summary)
    print(f"[PLAN] {len(tasks)} answer jobs, {len(tasks) * len(SCALES)} score jobs")
    if cfg.dry_run:
        for task in tasks:
            condition_name = str(task.condition.get("condition_name", ""))
            print(f"[DRY-RUN] {condition_name} r{task.repeat_idx:02d} {task.label}")
        return []

    if cfg.max_parallel <= 1:
        results = []
        for task in tasks:
            condition_name = str(task.condition.get("condition_name", ""))
            try:
                result = run_repeat_task(cfg, task)
                print(f"[RESULT] {condition_name} r{task.repeat_idx:02d} {task.label}: {result['status']}")
                results.append(result)
            except Exception as exc:
                print(f"[ERROR] {condition_name} r{task.repeat_idx:02d} {task.label}: {exc}")
                results.append(
                    {
                        "condition_name": condition_name,
                        "label": task.label,
                        "repeat": task.repeat_idx,
                        "status": "error",
                        "error": str(exc),
                    }
                )
        return results

    results: list[dict[str, Any]] = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=cfg.max_parallel) as executor:
        future_map = {executor.submit(run_repeat_task, cfg, task): task for task in tasks}
        for future in concurrent.futures.as_completed(future_map):
            task = future_map[future]
            condition_name = str(task.condition.get("condition_name", ""))
            try:
                result = future.result()
                print(f"[RESULT] {condition_name} r{task.repeat_idx:02d} {task.label}: {result['status']}")
                results.append(result)
            except Exception as exc:
                print(f"[ERROR] {condition_name} r{task.repeat_idx:02d} {task.label}: {exc}")
                results.append(
                    {
                        "condition_name": condition_name,
                        "label": task.label,
                        "repeat": task.repeat_idx,
                        "status": "error",
                        "error": str(exc),
                    }
                )
    return results


def condition_map_by_name(cfg: RuntimeConfig, original_summary: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(condition.get("condition_name", "") or ""): condition
        for condition in selected_conditions(cfg, original_summary)
    }


def run_stability_reruns(cfg: RuntimeConfig, original_summary: dict[str, Any]) -> list[dict[str, Any]]:
    if not cfg.stability_rerun or cfg.max_extra_repeat <= 0:
        return []
    all_results: list[dict[str, Any]] = []
    conditions = condition_map_by_name(cfg, original_summary)
    for extra_round in range(1, cfg.max_extra_repeat + 1):
        analysis = build_stability_analysis(cfg, original_summary)
        pending = pending_stability_rerun_groups(analysis)
        if not pending:
            print("[STABILITY] 无需补跑：所有不稳定条目已定稿或没有不稳定条目")
            break

        repeat_idx = cfg.repeat + extra_round
        tasks: list[StabilityRerunTask] = []
        for (condition_name, label), scale_item_ids in sorted(pending.items()):
            condition = conditions.get(condition_name)
            if not condition:
                continue
            tasks.append(
                StabilityRerunTask(
                    condition=condition,
                    label=label,
                    repeat_idx=repeat_idx,
                    scale_item_ids=scale_item_ids,
                )
            )
        print(f"[STABILITY] round {extra_round}/{cfg.max_extra_repeat}: {len(tasks)} item-rerun jobs")
        if cfg.dry_run:
            for task in tasks:
                condition_name = str(task.condition.get("condition_name", ""))
                print(
                    f"[DRY-RUN] stability {condition_name} r{task.repeat_idx:02d} "
                    f"{task.label}: {task.scale_item_ids}"
                )
            break

        if cfg.max_parallel <= 1:
            for task in tasks:
                condition_name = str(task.condition.get("condition_name", ""))
                try:
                    result = run_stability_rerun_task(cfg, task)
                    print(f"[RESULT] stability {condition_name} r{task.repeat_idx:02d} {task.label}: {result['status']}")
                    all_results.append(result)
                except Exception as exc:
                    print(f"[ERROR] stability {condition_name} r{task.repeat_idx:02d} {task.label}: {exc}")
                    all_results.append(
                        {
                            "condition_name": condition_name,
                            "label": task.label,
                            "repeat": task.repeat_idx,
                            "status": "error",
                            "error": str(exc),
                        }
                    )
            continue

        with concurrent.futures.ProcessPoolExecutor(max_workers=cfg.max_parallel) as executor:
            future_map = {executor.submit(run_stability_rerun_task, cfg, task): task for task in tasks}
            for future in concurrent.futures.as_completed(future_map):
                task = future_map[future]
                condition_name = str(task.condition.get("condition_name", ""))
                try:
                    result = future.result()
                    print(f"[RESULT] stability {condition_name} r{task.repeat_idx:02d} {task.label}: {result['status']}")
                    all_results.append(result)
                except Exception as exc:
                    print(f"[ERROR] stability {condition_name} r{task.repeat_idx:02d} {task.label}: {exc}")
                    all_results.append(
                        {
                            "condition_name": condition_name,
                            "label": task.label,
                            "repeat": task.repeat_idx,
                            "status": "error",
                            "error": str(exc),
                        }
                    )
    return all_results


def stddev(values: list[float]) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return 0.0
    avg = sum(values) / len(values)
    return math.sqrt(sum((value - avg) ** 2 for value in values) / len(values))


def choose_repeated_severity(rows: list[dict[str, Any]], mean_score: float | None) -> str:
    severities = [str(row.get("severity", "") or "") for row in rows if row.get("severity")]
    if not severities:
        return "—"
    counts = Counter(severities)
    max_count = max(counts.values())
    candidates = sorted(severity for severity, count in counts.items() if count == max_count)
    if len(candidates) == 1 or mean_score is None:
        return candidates[0]
    best = None
    best_distance = None
    for row in rows:
        severity = str(row.get("severity", "") or "")
        if severity not in candidates:
            continue
        score = row.get("total_score")
        if score is None:
            continue
        distance = abs(float(score) - float(mean_score))
        if best is None or best_distance is None or distance < best_distance:
            best = severity
            best_distance = distance
    return best or candidates[0]


def summarize_scale_repeats(rows: list[dict[str, Any]], scale_name: str) -> dict[str, Any]:
    scores = [
        float(row["total_score"])
        for row in rows
        if row.get("total_score") is not None
    ]
    score_mean = mean(scores)
    severity_votes = dict(Counter(str(row.get("severity", "") or "—") for row in rows))
    return {
        "total_score": score_mean,
        "severity": choose_repeated_severity(rows, score_mean),
        "scored_file": "",
        "repeat_runs": rows,
        "mean": score_mean,
        "stddev": stddev(scores),
        "min": min(scores) if scores else None,
        "max": max(scores) if scores else None,
        "severity_votes": severity_votes,
    }


def load_repeat_rows(cfg: RuntimeConfig, condition_name: str, label: str, scale_name: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for repeat_idx in range(1, cfg.repeat + 1):
        output_dir = repeat_label_dir(cfg, condition_name, repeat_idx, label)
        scored_path = output_dir / f"{scale_name}_scored.json"
        answers_path = output_dir / f"{scale_name}_answered.jsonl"
        scored = load_optional_json(scored_path)
        if scored is None:
            rows.append(
                {
                    "repeat": repeat_idx,
                    "status": "missing",
                    "answer_file": str(answers_path),
                    "scored_file": str(scored_path),
                    "total_score": None,
                    "severity": "—",
                }
            )
            continue
        rows.append(
            {
                "repeat": repeat_idx,
                "status": "ok",
                "answer_file": str(answers_path),
                "scored_file": str(scored_path),
                "total_score": extract_scale_total(scored, scale_name),
                "severity": extract_scale_severity(scored, scale_name),
            }
        )
    return rows


SCALE_ITEM_SCORE_KEYS = {
    "PHQ-9": "phq9_scores",
    "BDI-II": "bdi_ii_scores",
}


def expected_item_count(scale_name: str) -> int:
    return 9 if scale_name == "PHQ-9" else 21


def normalize_score(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        score = int(value)
    except (TypeError, ValueError):
        return None
    if score < 0 or score > 3:
        return None
    return score


def extract_scored_item_records(scored: dict[str, Any] | None, scale_name: str) -> dict[int, dict[str, Any]]:
    if not isinstance(scored, dict):
        return {}
    item_key = SCALE_ITEM_SCORE_KEYS.get(scale_name)
    items = scored.get(item_key) if item_key else None
    if not isinstance(items, list):
        return {}
    records: dict[int, dict[str, Any]] = {}
    for index, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            continue
        score = normalize_score(item.get("score"))
        if score is None:
            continue
        records[index] = {
            "item_id": index,
            "score": score,
            "item": str(item.get("item", "") or ""),
            "basis": str(item.get("basis", "") or ""),
        }
    return records


def extract_partial_item_records(
    scored: dict[str, Any] | None,
    scale_name: str,
    expected_ids: list[int] | None = None,
) -> dict[int, dict[str, Any]]:
    if not isinstance(scored, dict):
        return {}
    items = scored.get("item_scores")
    if not isinstance(items, list):
        item_key = SCALE_ITEM_SCORE_KEYS.get(scale_name)
        items = scored.get(item_key) if item_key else None
    if not isinstance(items, list):
        return {}

    expected_ids = sorted({int(item_id) for item_id in expected_ids or []})
    records: dict[int, dict[str, Any]] = {}
    for index, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            continue
        raw_item_id = (
            item.get("id")
            if item.get("id") is not None
            else item.get("item_id", item.get("question_id"))
        )
        if raw_item_id is None and len(items) == len(expected_ids):
            raw_item_id = expected_ids[index - 1]
        try:
            item_id = int(raw_item_id)
        except (TypeError, ValueError):
            continue
        score = normalize_score(item.get("score"))
        if score is None:
            continue
        records[item_id] = {
            "item_id": item_id,
            "score": score,
            "item": str(item.get("item", "") or ""),
            "basis": str(item.get("basis", "") or ""),
        }
    return records


def load_stability_samples(
    cfg: RuntimeConfig,
    condition_name: str,
    label: str,
    scale_name: str,
) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for repeat_idx in range(1, cfg.repeat + 1):
        output_dir = repeat_label_dir(cfg, condition_name, repeat_idx, label)
        scored_path = output_dir / f"{scale_name}_scored.json"
        answers_path = output_dir / f"{scale_name}_answered.jsonl"
        for item_id, item in extract_scored_item_records(load_optional_json(scored_path), scale_name).items():
            samples.append(
                {
                    "condition_name": condition_name,
                    "trigger_label": label,
                    "scale": scale_name,
                    "item_id": item_id,
                    "repeat": repeat_idx,
                    "source": "full_repeat",
                    "score": item["score"],
                    "item": item.get("item", ""),
                    "basis": item.get("basis", ""),
                    "answer_file": str(answers_path),
                    "scored_file": str(scored_path),
                }
            )

    for repeat_idx in range(cfg.repeat + 1, cfg.repeat + cfg.max_extra_repeat + 1):
        output_dir = repeat_label_dir(cfg, condition_name, repeat_idx, label)
        metadata = load_optional_json(output_dir / "metadata.json") or {}
        scale_item_ids = metadata.get("scale_item_ids", {}) if isinstance(metadata, dict) else {}
        expected_ids = scale_item_ids.get(scale_name, []) if isinstance(scale_item_ids, dict) else []
        scored_path = output_dir / f"{scale_name}_item_scored.json"
        answers_path = output_dir / f"{scale_name}_answered.jsonl"
        for item_id, item in extract_partial_item_records(load_optional_json(scored_path), scale_name, expected_ids).items():
            samples.append(
                {
                    "condition_name": condition_name,
                    "trigger_label": label,
                    "scale": scale_name,
                    "item_id": item_id,
                    "repeat": repeat_idx,
                    "source": "item_rerun",
                    "score": item["score"],
                    "item": item.get("item", ""),
                    "basis": item.get("basis", ""),
                    "answer_file": str(answers_path),
                    "scored_file": str(scored_path),
                }
            )
    return samples


def strict_majority_score(scores: list[int]) -> tuple[int | None, dict[str, int]]:
    votes = Counter(scores)
    vote_payload = {str(score): int(count) for score, count in sorted(votes.items())}
    if not scores:
        return None, vote_payload
    best_score, best_count = votes.most_common(1)[0]
    if best_count > len(scores) / 2:
        return int(best_score), vote_payload
    return None, vote_payload


def final_review_score(
    scores: list[int],
    *,
    allow_lowest_tied_vote: bool,
) -> tuple[int | None, dict[str, int], str, str]:
    majority, votes = strict_majority_score(scores)
    if majority is not None:
        return majority, votes, "strict_majority", ""
    if not votes:
        return None, votes, "unresolved", ""
    if not allow_lowest_tied_vote:
        return None, votes, "unresolved", "未达严格多数；等待补跑后定稿。"

    max_count = max(votes.values())
    candidates = sorted(int(score) for score, count in votes.items() if int(count) == max_count)
    selected = candidates[0]
    return (
        selected,
        votes,
        "lowest_tied_vote",
        "无严格多数；按最高票并列分中的最低分定稿。",
    )


def build_stability_analysis(cfg: RuntimeConfig, original_summary: dict[str, Any]) -> dict[str, Any]:
    item_rows: list[dict[str, Any]] = []
    rerun_records: list[dict[str, Any]] = []
    for condition in selected_conditions(cfg, original_summary):
        condition_name = str(condition.get("condition_name", "") or "")
        for repeat_idx in range(cfg.repeat + 1, cfg.repeat + cfg.max_extra_repeat + 1):
            for label in cfg.labels:
                output_dir = repeat_label_dir(cfg, condition_name, repeat_idx, label)
                metadata = load_optional_json(output_dir / "metadata.json")
                if not metadata or metadata.get("source_repeat_mode") != "stability_adaptive_item_rerun":
                    continue
                rerun_records.append(
                    {
                        "condition_name": condition_name,
                        "trigger_label": label,
                        "repeat": repeat_idx,
                        "scale_item_ids": metadata.get("scale_item_ids", {}),
                        "metadata_file": str(output_dir / "metadata.json"),
                    }
                )

        for label in cfg.labels:
            for scale_name in SCALES:
                samples = load_stability_samples(cfg, condition_name, label, scale_name)
                by_item: dict[int, list[dict[str, Any]]] = {}
                for sample in samples:
                    by_item.setdefault(int(sample["item_id"]), []).append(sample)
                for item_id in sorted(by_item):
                    group = sorted(by_item[item_id], key=lambda item: int(item.get("repeat", 0) or 0))
                    initial_scores = [
                        int(item["score"])
                        for item in group
                        if item.get("source") == "full_repeat"
                    ]
                    extra_scores = [
                        int(item["score"])
                        for item in group
                        if item.get("source") == "item_rerun"
                    ]
                    all_scores = [int(item["score"]) for item in group]
                    if not initial_scores:
                        continue
                    initial_range = max(initial_scores) - min(initial_scores)
                    initially_unstable = initial_range >= cfg.stability_range_threshold
                    initial_majority, _initial_votes = strict_majority_score(initial_scores)
                    must_finish_extra_reruns = initially_unstable and initial_majority is None
                    strict_majority, votes = strict_majority_score(all_scores)
                    if must_finish_extra_reruns and len(extra_scores) < cfg.max_extra_repeat:
                        final_score = None
                        resolution_method = "unresolved"
                        resolution_note = "已补跑 {}/{} 轮；等待补跑完成后定稿。".format(
                            len(extra_scores),
                            cfg.max_extra_repeat,
                        )
                    else:
                        final_score, votes, resolution_method, resolution_note = final_review_score(
                            all_scores,
                            allow_lowest_tied_vote=True,
                        )
                    if initially_unstable:
                        status = "resolved" if final_score is not None else "unresolved"
                    else:
                        status = "stable"
                    item_rows.append(
                        {
                            "condition_name": condition_name,
                            "trigger_label": label,
                            "scale": scale_name,
                            "item_id": item_id,
                            "item": next((str(item.get("item", "") or "") for item in group if item.get("item")), ""),
                            "initial_scores": initial_scores,
                            "extra_scores": extra_scores,
                            "scores": all_scores,
                            "initial_range": initial_range,
                            "range": max(all_scores) - min(all_scores) if all_scores else None,
                            "votes": votes,
                            "majority_score": final_score,
                            "strict_majority_score": strict_majority,
                            "resolution_method": resolution_method,
                            "resolution_note": resolution_note,
                            "initially_unstable": initially_unstable,
                            "status": status,
                            "samples": group,
                        }
                    )

    scale_counts: dict[str, dict[str, int]] = {}
    for scale_name in SCALES:
        scale_rows = [row for row in item_rows if row.get("scale") == scale_name and row.get("initially_unstable")]
        scale_counts[scale_name] = {
            "unstable_item_count": len(scale_rows),
            "resolved_count": sum(1 for row in scale_rows if row.get("status") == "resolved"),
            "unresolved_count": sum(1 for row in scale_rows if row.get("status") == "unresolved"),
        }

    adjusted_totals = build_stability_adjusted_totals(item_rows)
    return {
        "enabled": cfg.stability_rerun,
        "range_threshold": cfg.stability_range_threshold,
        "max_extra_repeat": cfg.max_extra_repeat,
        "item_rows": item_rows,
        "unstable_items": [row for row in item_rows if row.get("initially_unstable")],
        "scale_counts": scale_counts,
        "rerun_records": rerun_records,
        "adjusted_totals": adjusted_totals,
    }


def build_stability_adjusted_totals(item_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in item_rows:
        grouped.setdefault(
            (
                str(row.get("condition_name", "") or ""),
                str(row.get("trigger_label", "") or ""),
                str(row.get("scale", "") or ""),
            ),
            [],
        ).append(row)

    totals: list[dict[str, Any]] = []
    for (condition_name, label, scale_name), rows in sorted(grouped.items()):
        expected_count = expected_item_count(scale_name)
        score_by_item = {
            int(row.get("item_id", 0) or 0): row.get("majority_score")
            for row in rows
            if row.get("majority_score") is not None
        }
        unresolved = [
            int(row.get("item_id", 0) or 0)
            for row in rows
            if row.get("initially_unstable") and row.get("status") == "unresolved"
        ]
        tie_break_items = [
            int(row.get("item_id", 0) or 0)
            for row in rows
            if row.get("resolution_method") == "lowest_tied_vote"
        ]
        adjusted_total = None
        severity = "—"
        if len(score_by_item) == expected_count:
            adjusted_total = float(sum(int(score_by_item[item_id]) for item_id in sorted(score_by_item)))
            severity = expected_scale_severity(scale_name, adjusted_total)
        totals.append(
            {
                "condition_name": condition_name,
                "trigger_label": label,
                "scale": scale_name,
                "expected_item_count": expected_count,
                "scored_item_count": len(score_by_item),
                "adjusted_total": adjusted_total,
                "adjusted_severity": severity,
                "unresolved_item_ids": sorted(unresolved),
                "tie_break_item_ids": sorted(tie_break_items),
            }
        )
    return totals


def apply_reviewed_scale_totals(results: list[dict[str, Any]], analysis: dict[str, Any]) -> None:
    adjusted_totals = analysis.get("adjusted_totals", []) if isinstance(analysis, dict) else []
    if not isinstance(adjusted_totals, list):
        adjusted_totals = []
    by_key = {
        (
            str(row.get("condition_name", "") or ""),
            str(row.get("trigger_label", "") or ""),
            str(row.get("scale", "") or ""),
        ): row
        for row in adjusted_totals
        if isinstance(row, dict)
    }

    for result in results:
        condition_name = str(result.get("condition_name", "") or "")
        for evaluation in result.get("evaluations", []) or []:
            label = str(evaluation.get("trigger_label", "") or "")
            for scale_name, scale_payload in (evaluation.get("scales", {}) or {}).items():
                reviewed = by_key.get((condition_name, label, scale_name))
                if not reviewed or reviewed.get("adjusted_total") is None:
                    continue
                scale_payload["repeat_mean_total"] = scale_payload.get("total_score")
                scale_payload["repeat_mean_severity"] = scale_payload.get("severity")
                scale_payload["total_score"] = reviewed.get("adjusted_total")
                scale_payload["severity"] = reviewed.get("adjusted_severity", "—")
                scale_payload["score_source"] = "reviewed_item_final_scores"
                scale_payload["reviewed_item_count"] = reviewed.get("scored_item_count", 0)
                scale_payload["expected_item_count"] = reviewed.get("expected_item_count", 0)
                scale_payload["review_unresolved_item_ids"] = reviewed.get("unresolved_item_ids", [])
                scale_payload["review_tie_break_item_ids"] = reviewed.get("tie_break_item_ids", [])
                if reviewed.get("tie_break_item_ids"):
                    scale_payload["review_note"] = "含无严格多数条目，按最高票并列分中的最低分定稿。"
                else:
                    scale_payload["review_note"] = "评分复核后总分为条目最终分之和。"

        apply_trajectory_deltas(result.get("evaluations", []))
        final_deltas = {}
        for scale_name in SCALES:
            final_delta = None
            for evaluation in result.get("evaluations", []):
                delta = evaluation["scales"].get(scale_name, {}).get("delta_from_baseline")
                if delta is not None:
                    final_delta = delta
            final_deltas[scale_name] = final_delta
        result["final_deltas"] = final_deltas


def pending_stability_rerun_groups(analysis: dict[str, Any]) -> dict[tuple[str, str], dict[str, list[int]]]:
    pending: dict[tuple[str, str], dict[str, list[int]]] = {}
    rows = analysis.get("unstable_items", []) if isinstance(analysis, dict) else []
    for row in rows:
        if row.get("status") != "unresolved":
            continue
        key = (str(row.get("condition_name", "") or ""), str(row.get("trigger_label", "") or ""))
        scale_name = str(row.get("scale", "") or "")
        item_id = int(row.get("item_id", 0) or 0)
        if not scale_name or item_id <= 0:
            continue
        pending.setdefault(key, {}).setdefault(scale_name, []).append(item_id)
    return {
        key: {scale: sorted(set(item_ids)) for scale, item_ids in scale_map.items()}
        for key, scale_map in pending.items()
    }


def metadata_for_label(cfg: RuntimeConfig, condition_name: str, label: str) -> dict[str, Any]:
    for repeat_idx in range(1, cfg.repeat + 1):
        metadata = load_optional_json(repeat_label_dir(cfg, condition_name, repeat_idx, label) / "metadata.json")
        if metadata:
            return metadata
    return {"trigger_label": label, "completed_session_count": 0, "sim_time": "", "snapshot_name": ""}


def build_condition_result(cfg: RuntimeConfig, condition: dict[str, Any]) -> dict[str, Any] | None:
    condition_name = str(condition.get("condition_name", ""))
    evaluations: list[dict[str, Any]] = []
    for label in cfg.labels:
        metadata = metadata_for_label(cfg, condition_name, label)
        scales_payload = {
            scale_name: summarize_scale_repeats(
                load_repeat_rows(cfg, condition_name, label, scale_name),
                scale_name,
            )
            for scale_name in SCALES
        }
        evaluations.append(
            {
                "trigger_label": label,
                "completed_session_count": int(metadata.get("completed_session_count", 0) or 0),
                "sim_time": str(metadata.get("sim_time", "") or ""),
                "snapshot_name": str(metadata.get("snapshot_name", "") or ""),
                "source": "repeat_scale_eval",
                "metadata_path": str(repeat_label_dir(cfg, condition_name, 1, label) / "metadata.json"),
                "scales": scales_payload,
            }
        )

    evaluations.sort(
        key=lambda item: trigger_sort_key(
            str(item.get("trigger_label", "")),
            int(item.get("completed_session_count", 0) or 0),
        )
    )
    apply_trajectory_deltas(evaluations)
    final_deltas = {}
    for scale_name in SCALES:
        final_delta = None
        for evaluation in evaluations:
            delta = evaluation["scales"].get(scale_name, {}).get("delta_from_baseline")
            if delta is not None:
                final_delta = delta
        final_deltas[scale_name] = final_delta

    return {
        "condition_name": condition_name,
        "run_name": str(condition.get("run_name", "") or ""),
        "run_dir": str(condition_output_dir(cfg, condition_name)),
        "variant": str(condition.get("variant", "") or ""),
        "group": str(condition.get("group", "") or ""),
        "severity": str(condition.get("severity", "") or ""),
        "source_condition_name": str(condition.get("source_condition_name", "") or ""),
        "source_group": str(condition.get("source_group", "") or ""),
        "depression_state_reset": "initial_from_config" if cfg.reset_target_depression_state else "",
        "evaluations": evaluations,
        "final_deltas": final_deltas,
    }


def build_repeat_validation(results: list[dict[str, Any]]) -> dict[str, Any]:
    synthetic_results: list[dict[str, Any]] = []
    for result in results:
        for repeat_idx in range(1, 1000):
            has_repeat = False
            evaluations = []
            for evaluation in result.get("evaluations", []):
                scales = {}
                for scale_name, scale_payload in evaluation.get("scales", {}).items():
                    for row in scale_payload.get("repeat_runs", []) or []:
                        if int(row.get("repeat", 0) or 0) != repeat_idx:
                            continue
                        if row.get("status") != "ok":
                            continue
                        has_repeat = True
                        scales[scale_name] = {
                            "total_score": row.get("total_score"),
                            "severity": row.get("severity"),
                            "scored_file": row.get("scored_file"),
                        }
                if scales:
                    evaluations.append(
                        {
                            "trigger_label": f"{evaluation.get('trigger_label', '')}/r{repeat_idx:02d}",
                            "completed_session_count": evaluation.get("completed_session_count", 0),
                            "sim_time": evaluation.get("sim_time", ""),
                            "snapshot_name": evaluation.get("snapshot_name", ""),
                            "source": "staged_eval",
                            "metadata_path": "",
                            "scales": scales,
                        }
                    )
            if not has_repeat:
                if repeat_idx == 1:
                    continue
                break
            synthetic_results.append(
                {
                    **result,
                    "run_dir": "",
                    "evaluations": evaluations,
                }
            )
    try:
        return build_scale_score_validation(synthetic_results)
    except Exception as exc:
        return {
            "total_mismatches": [],
            "item_mismatches": [],
            "error": str(exc),
        }


def build_summary_payload(
    cfg: RuntimeConfig,
    original_summary: dict[str, Any],
    warnings: list[str],
) -> dict[str, Any]:
    stability_analysis = build_stability_analysis(cfg, original_summary)
    results = []
    for condition in selected_conditions(cfg, original_summary):
        result = build_condition_result(cfg, condition)
        if result is not None:
            results.append(result)
    apply_reviewed_scale_totals(results, stability_analysis)
    return {
        "batch_name": cfg.name,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source_original_summary": str(cfg.original_summary),
        "archive_results_root": str(cfg.archive_results_root),
        "repeat": cfg.repeat,
        "labels": cfg.labels,
        "summary_only": cfg.report_only,
        "conditions": results,
        "warnings": warnings,
        "trigger_labels": ordered_trigger_labels(results),
        "variant_trajectory": aggregate_dimension_trajectory(results, dimension_key="variant", dimension_values=VARIANTS),
        "group_values": report_group_values(cfg),
        "group_trajectory": aggregate_dimension_trajectory(results, dimension_key="group", dimension_values=report_group_values(cfg)),
        "severity_trajectory": aggregate_dimension_trajectory(results, dimension_key="severity", dimension_values=SEVERITIES),
        "variant_final_delta": aggregate_final_deltas(results, dimension_key="variant", dimension_values=VARIANTS),
        "group_final_delta": aggregate_final_deltas(results, dimension_key="group", dimension_values=report_group_values(cfg)),
        "severity_final_delta": aggregate_final_deltas(results, dimension_key="severity", dimension_values=SEVERITIES),
        "group_severity_final_delta_matrix": build_repeat_group_severity_matrix(cfg, results),
        "scale_score_validation": build_repeat_validation(results),
        "stability_analysis": stability_analysis,
        "original_diff": build_original_diff(original_summary, results),
    }


def original_eval_map(original_summary: dict[str, Any]) -> dict[tuple[str, str, str], dict[str, Any]]:
    payload: dict[tuple[str, str, str], dict[str, Any]] = {}
    for condition in original_summary.get("conditions", []) or []:
        condition_name = str(condition.get("condition_name", "") or "")
        for evaluation in condition.get("evaluations", []) or []:
            label = str(evaluation.get("trigger_label", "") or "")
            for scale_name, scale_payload in (evaluation.get("scales", {}) or {}).items():
                payload[(condition_name, label, scale_name)] = scale_payload
    return payload


def build_original_diff(original_summary: dict[str, Any], results: list[dict[str, Any]]) -> dict[str, Any]:
    originals = original_eval_map(original_summary)
    rows = []
    by_scale: dict[str, list[dict[str, Any]]] = {scale_name: [] for scale_name in SCALES}
    for result in results:
        condition_name = str(result.get("condition_name", "") or "")
        source_condition_name = str(result.get("source_condition_name", "") or condition_name)
        for evaluation in result.get("evaluations", []) or []:
            label = str(evaluation.get("trigger_label", "") or "")
            for scale_name, scale_payload in (evaluation.get("scales", {}) or {}).items():
                original = originals.get((source_condition_name, label, scale_name), {})
                original_total = original.get("total_score")
                repeat_total = scale_payload.get("total_score")
                delta = None
                if original_total is not None and repeat_total is not None:
                    delta = float(repeat_total) - float(original_total)
                missing_repeats = [
                    row.get("repeat")
                    for row in scale_payload.get("repeat_runs", []) or []
                    if row.get("status") != "ok" or row.get("total_score") is None
                ]
                row = {
                    "condition_name": condition_name,
                    "trigger_label": label,
                    "scale": scale_name,
                    "original_total": original_total,
                    "reviewed_total": repeat_total,
                    "repeat_mean_total": scale_payload.get("repeat_mean_total"),
                    "delta": delta,
                    "original_severity": original.get("severity", "—"),
                    "repeat_severity": scale_payload.get("severity", "—"),
                    "severity_changed": bool(original.get("severity") != scale_payload.get("severity")),
                    "missing_repeats": missing_repeats,
                    "stddev": scale_payload.get("stddev"),
                    "min": scale_payload.get("min"),
                    "max": scale_payload.get("max"),
                }
                rows.append(row)
                by_scale.setdefault(scale_name, []).append(row)

    summary = {}
    for scale_name, scale_rows in by_scale.items():
        abs_deltas = [abs(float(row["delta"])) for row in scale_rows if row.get("delta") is not None]
        summary[scale_name] = {
            "count": len(scale_rows),
            "mean_abs_delta": mean(abs_deltas),
            "max_abs_delta": max(abs_deltas) if abs_deltas else None,
            "severity_changed_count": sum(1 for row in scale_rows if row.get("severity_changed")),
            "missing_repeat_entry_count": sum(1 for row in scale_rows if row.get("missing_repeats")),
        }
    return {"rows": rows, "summary": summary}


def render_original_diff_markdown(diff: dict[str, Any]) -> list[str]:
    rows = diff.get("rows", []) if isinstance(diff, dict) else []
    summary = diff.get("summary", {}) if isinstance(diff, dict) else {}
    lines = []
    lines.append("## 与原报告差异\n")
    lines.append("### 差异概览\n")
    lines.append("| 量表 | 条目数 | 平均绝对差 | 最大绝对差 | 程度变化数 | 有失败/缺失重复的条目数 |")
    lines.append("|------|--------|------------|------------|------------|--------------------------|")
    for scale_name in SCALES:
        item = summary.get(scale_name, {}) if isinstance(summary, dict) else {}
        lines.append(
            "| {scale} | {count} | {mad} | {maxd} | {sev} | {missing} |".format(
                scale=scale_name,
                count=item.get("count", 0),
                mad=format_number(item.get("mean_abs_delta")),
                maxd=format_number(item.get("max_abs_delta")),
                sev=item.get("severity_changed_count", 0),
                missing=item.get("missing_repeat_entry_count", 0),
            )
        )
    lines.append("")
    lines.append("### 明细\n")
    lines.append("| 条件 | 评估点 | 量表 | 原总分 | 复核总分 | 重复均值 | Δ | 原程度 | 复核程度 | 重复标准差 | 缺失重复 |")
    lines.append("|------|--------|------|--------|----------|----------|---|--------|----------|------------|----------|")
    for row in rows:
        missing = ",".join(f"r{int(item):02d}" for item in row.get("missing_repeats", []) if item) or "—"
        original_severity = row.get("original_severity", "—")
        repeat_severity = row.get("repeat_severity", "—")
        if row.get("severity_changed"):
            repeat_severity = f"**{repeat_severity}**"
        lines.append(
            "| {condition} | {label} | {scale} | {orig} | {reviewed} | {mean} | {delta} | {orig_sev} | {repeat_sev} | {std} | {missing} |".format(
                condition=row.get("condition_name", "—"),
                label=row.get("trigger_label", "—"),
                scale=row.get("scale", "—"),
                orig=format_number(row.get("original_total")),
                reviewed=format_number(row.get("reviewed_total")),
                mean=format_number(row.get("repeat_mean_total")),
                delta=format_delta(row.get("delta")),
                orig_sev=original_severity,
                repeat_sev=repeat_severity,
                std=format_number(row.get("stddev")),
                missing=missing,
            )
        )
    lines.append("")
    return lines


def render_repeat_detail_markdown(results: list[dict[str, Any]]) -> list[str]:
    lines = []
    lines.append("## 重复评分明细附录\n")
    for result in results:
        lines.append(f"### {result.get('condition_name', '—')}\n")
        lines.append("| 评估点 | 量表 | repeat | 总分 | 程度 | scored_file |")
        lines.append("|--------|------|--------|------|------|-------------|")
        for evaluation in result.get("evaluations", []):
            label = evaluation.get("trigger_label", "—")
            for scale_name, scale_payload in (evaluation.get("scales", {}) or {}).items():
                for row in scale_payload.get("repeat_runs", []) or []:
                    scored_file = row.get("scored_file", "")
                    if scored_file:
                        scored_file = to_display_path(Path(scored_file))
                    lines.append(
                        "| {label} | {scale} | r{repeat:02d} | {total} | {severity} | `{file}` |".format(
                            label=label,
                            scale=scale_name,
                            repeat=int(row.get("repeat", 0) or 0),
                            total=format_number(row.get("total_score")),
                            severity=row.get("severity", "—"),
                            file=scored_file or "—",
                        )
                    )
        lines.append("")
    return lines


def render_score_list(values: Any) -> str:
    if not isinstance(values, list):
        return "—"
    return ",".join(str(value) for value in values) if values else "—"


def render_votes(votes: Any) -> str:
    if not isinstance(votes, dict) or not votes:
        return "—"
    return ", ".join(f"{score}:{count}" for score, count in sorted(votes.items()))


def render_repeat_group_severity_matrix_markdown(matrix_payload: dict, group_values: list[str]) -> list[str]:
    lines = []
    lines.append("## Group × Severity 最终变化矩阵\n")
    groups = list(group_values or GROUPS)
    for scale_name in SCALES:
        lines.append(f"### {scale_name}\n")
        header = "| severity \\ group | " + " | ".join(groups) + " |"
        sep = "|--------------------|" + "|".join(["----"] * len(groups)) + "|"
        lines.append(header)
        lines.append(sep)
        for severity in SEVERITIES:
            row = matrix_payload.get(scale_name, {}).get(severity, {})
            rendered = [format_delta(row.get(group)) for group in groups]
            lines.append(f"| {severity} | " + " | ".join(rendered) + " |")
        lines.append("")
    return lines


def render_status(value: Any) -> str:
    status = str(value or "")
    return {
        "stable": "稳定",
        "resolved": "已定稿",
        "unresolved": "未定稿",
    }.get(status, status or "—")


def render_stability_analysis_markdown(analysis: dict[str, Any]) -> list[str]:
    if not isinstance(analysis, dict):
        return []
    if not analysis.get("enabled") and not analysis.get("unstable_items"):
        return []

    lines = []
    unstable_items = analysis.get("unstable_items", [])
    if not isinstance(unstable_items, list):
        unstable_items = []
    adjusted_totals = analysis.get("adjusted_totals", [])
    if not isinstance(adjusted_totals, list):
        adjusted_totals = []
    rerun_records = analysis.get("rerun_records", [])
    if not isinstance(rerun_records, list):
        rerun_records = []

    lines.append("## 条目稳定性与补跑\n")
    lines.append(
        "- 判定规则：同一条目初始重复分数极差 `>= {}` 视为不稳定。".format(
            analysis.get("range_threshold", "—")
        )
    )
    lines.append(f"- 最大补跑轮数：{analysis.get('max_extra_repeat', '—')}")
    lines.append(f"- 不稳定条目数：{len(unstable_items)}")
    lines.append(f"- 已记录补跑任务数：{len(rerun_records)}")
    lines.append("")

    scale_counts = analysis.get("scale_counts", {})
    if isinstance(scale_counts, dict):
        lines.append("### 概览\n")
        lines.append("| 量表 | 不稳定条目 | 已定稿 | 未定稿 |")
        lines.append("|------|------------|--------|--------|")
        for scale_name in SCALES:
            item = scale_counts.get(scale_name, {}) if isinstance(scale_counts.get(scale_name, {}), dict) else {}
            lines.append(
                "| {scale} | {unstable} | {resolved} | {unresolved} |".format(
                    scale=scale_name,
                    unstable=item.get("unstable_item_count", 0),
                    resolved=item.get("resolved_count", 0),
                    unresolved=item.get("unresolved_count", 0),
                )
            )
        lines.append("")

    if unstable_items:
        lines.append("### 不稳定条目明细\n")
        lines.append("| 条件 | 评估点 | 量表 | 条目 | 初始分数 | 补跑分数 | 投票 | 最终分 | 状态 | 备注 |")
        lines.append("|------|--------|------|------|----------|----------|------|--------|------|------|")
        for row in unstable_items:
            lines.append(
                "| {condition} | {label} | {scale} | {item_id} | {initial} | {extra} | {votes} | {majority} | {status} | {note} |".format(
                    condition=row.get("condition_name", "—"),
                    label=row.get("trigger_label", "—"),
                    scale=row.get("scale", "—"),
                    item_id=row.get("item_id", "—"),
                    initial=render_score_list(row.get("initial_scores")),
                    extra=render_score_list(row.get("extra_scores")),
                    votes=render_votes(row.get("votes")),
                    majority=format_number(row.get("majority_score")),
                    status=render_status(row.get("status")),
                    note=row.get("resolution_note", "") or "—",
                )
            )
        lines.append("")

    if adjusted_totals:
        lines.append("### 评分复核后总分\n")
        lines.append("| 条件 | 评估点 | 量表 | 条目数 | 复核总分 | 复核程度 | 未定稿条目 | 低分定稿条目 |")
        lines.append("|------|--------|------|--------|----------|----------|------------|--------------|")
        for row in adjusted_totals:
            unresolved = row.get("unresolved_item_ids", [])
            unresolved_text = ",".join(str(item) for item in unresolved) if isinstance(unresolved, list) and unresolved else "—"
            tie_break_items = row.get("tie_break_item_ids", [])
            tie_break_text = ",".join(str(item) for item in tie_break_items) if isinstance(tie_break_items, list) and tie_break_items else "—"
            lines.append(
                "| {condition} | {label} | {scale} | {count}/{expected} | {total} | {severity} | {unresolved} | {tie_break} |".format(
                    condition=row.get("condition_name", "—"),
                    label=row.get("trigger_label", "—"),
                    scale=row.get("scale", "—"),
                    count=row.get("scored_item_count", 0),
                    expected=row.get("expected_item_count", 0),
                    total=format_number(row.get("adjusted_total")),
                    severity=row.get("adjusted_severity", "—"),
                    unresolved=unresolved_text,
                    tie_break=tie_break_text,
                )
            )
        lines.append("")

    if rerun_records:
        lines.append("### 补跑记录\n")
        lines.append("| 条件 | 评估点 | repeat | 条目 | metadata |")
        lines.append("|------|--------|--------|------|----------|")
        for record in rerun_records:
            output_file = record.get("metadata_file", "")
            display_file = to_display_path(Path(output_file)) if output_file else "—"
            scale_items = record.get("scale_item_ids", {})
            lines.append(
                "| {condition} | {label} | r{repeat:02d} | `{items}` | `{file}` |".format(
                    condition=record.get("condition_name", "—"),
                    label=record.get("trigger_label", "—"),
                    repeat=int(record.get("repeat", 0) or 0),
                    items=json.dumps(scale_items, ensure_ascii=False, sort_keys=True),
                    file=display_file,
                )
            )
        lines.append("")

    return lines


def render_markdown_report(payload: dict[str, Any]) -> str:
    results = payload.get("conditions", [])
    warnings = payload.get("warnings", [])
    lines = []
    lines.append(f"# 存档重复量表评估汇总：{payload['batch_name']}\n")
    lines.append(f"生成时间：{payload['generated_at']}\n")
    lines.append("> 本汇总只统计 PHQ-9 / BDI-II；主分数为评分复核后的条目最终分之和，重复均值保留在附录。\n")
    lines.append("## 汇总范围\n")
    lines.append(f"- 条件数：{len(results)}")
    lines.append(f"- 重复次数：{payload.get('repeat', '—')}")
    lines.append(f"- 原始报告：`{payload.get('source_original_summary', '')}`")
    lines.append(f"- 评估点：{', '.join(payload['trigger_labels']) if payload.get('trigger_labels') else '—'}")
    if warnings:
        lines.append("- 警告：")
        for warning in warnings:
            lines.append(f"  - {warning}")
    lines.append("")

    lines.append("## 各条件量表变化过程\n")
    for result in results:
        lines.extend(render_condition_trajectory_markdown(result))

    lines.extend(
        render_dimension_trajectory_markdown(
            "按 Variant 的评估轨迹对比",
            payload["variant_trajectory"],
            VARIANTS,
            payload["trigger_labels"],
        )
    )
    lines.extend(
        render_dimension_trajectory_markdown(
            "按 Group 的评估轨迹对比",
            payload["group_trajectory"],
            payload.get("group_values", GROUPS),
            payload["trigger_labels"],
        )
    )
    lines.extend(
        render_dimension_trajectory_markdown(
            "按 Severity 的评估轨迹对比",
            payload["severity_trajectory"],
            SEVERITIES,
            payload["trigger_labels"],
        )
    )
    lines.extend(render_final_delta_markdown("按 Variant 的最终变化对比", payload["variant_final_delta"], VARIANTS))
    lines.extend(render_final_delta_markdown("按 Group 的最终变化对比", payload["group_final_delta"], payload.get("group_values", GROUPS)))
    lines.extend(render_final_delta_markdown("按 Severity 的最终变化对比", payload["severity_final_delta"], SEVERITIES))
    lines.extend(render_repeat_group_severity_matrix_markdown(
        payload["group_severity_final_delta_matrix"],
        payload.get("group_values", GROUPS),
    ))
    lines.extend(render_scale_score_validation_markdown(payload["scale_score_validation"]))
    lines.extend(render_stability_analysis_markdown(payload.get("stability_analysis", {})))
    lines.extend(render_repeat_detail_markdown(results))
    lines.extend(render_original_diff_markdown(payload["original_diff"]))
    return "\n".join(lines)


def write_report_outputs(cfg: RuntimeConfig, payload: dict[str, Any]) -> tuple[Path, Path]:
    json_path = reports_dir(cfg) / f"{cfg.name}_summary.json"
    md_path = reports_dir(cfg) / f"{cfg.name}_summary.md"
    if cfg.dry_run:
        print(f"[DRY-RUN] write summary json: {json_path}")
        print(f"[DRY-RUN] write summary md:   {md_path}")
        return json_path, md_path
    write_json_file(json_path, payload)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(render_markdown_report(payload), encoding="utf-8")
    print(f"[WRITE] {json_path}")
    print(f"[WRITE] {md_path}")
    return json_path, md_path


def main() -> None:
    args = parse_args()
    cfg = resolve_runtime_config(args)
    original_summary = load_json_file(cfg.original_summary)
    if not isinstance(original_summary, dict):
        raise ValueError(f"original summary must be JSON object: {cfg.original_summary}")

    print_effective_config(cfg, original_summary)
    warnings: list[str] = []
    if not cfg.report_only:
        task_results = run_tasks(cfg, original_summary)
        for item in task_results:
            if item.get("status") == "error":
                warnings.append(
                    "{condition} {label} r{repeat:02d}: {error}".format(
                        condition=item.get("condition_name", "—"),
                        label=item.get("label", "—"),
                        repeat=int(item.get("repeat", 0) or 0),
                        error=item.get("error", ""),
                    )
                )
        stability_results = run_stability_reruns(cfg, original_summary)
        for item in stability_results:
            if item.get("status") == "error":
                warnings.append(
                    "stability {condition} {label} r{repeat:02d}: {error}".format(
                        condition=item.get("condition_name", "—"),
                        label=item.get("label", "—"),
                        repeat=int(item.get("repeat", 0) or 0),
                        error=item.get("error", ""),
                    )
                )
    else:
        print("[INFO] report-only: 跳过 worker 调用，只汇总已有重复结果")

    payload = build_summary_payload(cfg, original_summary, warnings)
    _json_path, md_path = write_report_outputs(cfg, payload)
    print(f"\n[Done] 存档重复评估报告: {md_path}")


if __name__ == "__main__":
    main()
