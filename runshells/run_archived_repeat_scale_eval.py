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
import os
import re
import shutil
import subprocess
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from artifact_digest import (
    canonical_json_sha256 as _canonical_json_sha256,
    file_sha256 as _file_sha256,
)

from scipy.stats import t as student_t

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
    extract_direct_answer_score,
    expected_scale_severity,
    format_delta,
    format_number,
    load_json_file,
    mean,
    ordered_trigger_labels,
    render_condition_trajectory_markdown,
    render_dimension_trajectory_markdown,
    render_final_delta_markdown,
    render_scale_score_validation_markdown,
    resolve_conditions,
    trigger_sort_key,
    write_json_file,
)
from cbt_experiment_config import (
    MANIFEST_FILENAME,
    assert_identity_matches,
    manifest_identity,
)


DEFAULT_LABELS = ["T0", "session_4", "session_8", "session_12", "session_16", "T4", "POST"]
REPEAT_OUTPUT_SUBDIR = "repeat_scale_eval"
STAGED_WORKER_SCRIPT = BASE_DIR / "runshells" / "run_staged_eval_worker.py"
CHECKPOINT_RESULTS_PREFIX = "/workspace/project/results"
REPEAT_VLLM_ENV = "GA_REPEAT_USE_VLLM_MODELS"
SNAPSHOT_BUNDLE_SCHEMA_VERSION = 1
AGGREGATION_METHOD_VERSION = "fixed_complete_scale_reviewed_v2"
SCALE_ITEM_SCORE_KEYS = {
    "PHQ-9": "phq9_scores",
    "BDI-II": "bdi_ii_scores",
}
SCALE_ITEM_COUNTS = {
    "PHQ-9": 9,
    "BDI-II": 21,
}

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
REPEAT = 10

# 评估点；可填字符串 "T0,POST"，也可直接使用 ",".join(DEFAULT_LABELS)
LABELS = ",".join(DEFAULT_LABELS)

# 只处理指定 condition；留空表示处理原始 summary 中全部 condition
CONDITIONS: list[str] = []

# 并行任务数
MAX_PARALLEL = 1

# 是否覆盖已有重复输出
FORCE = False

# 是否断点续跑：复用完整的 answered/scored，只补齐缺失阶段
RESUME_PARTIAL = True

# 是否只打印计划，不实际执行
DRY_RUN = False

# 是否只基于已有重复结果重新生成报告
REPORT_ONLY = False

# 是否在完整报告写入后清理复评工作底稿。底层入口默认关闭，由批处理包装脚本显式开启。
CLEANUP_COMPLETED_ARTIFACTS = False

# 是否把目标患者动态抑郁状态重置为初始 depression_config 状态后复评
RESET_TARGET_DEPRESSION_STATE = False

# 旧 staged job 没有按时间点冻结 storage。默认拒绝；只有显式开启才复用最终 storage。
ALLOW_LEGACY_FINAL_STORAGE = False

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
    resume_partial: bool
    dry_run: bool
    report_only: bool
    reset_target_depression_state: bool
    allow_legacy_final_storage: bool
    output_group: str
    cleanup_completed_artifacts: bool = False
    require_controller_manifest: bool = False


@dataclass(frozen=True)
class RepeatTask:
    condition: dict[str, Any]
    label: str
    repeat_idx: int


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
    parser.add_argument("--repeat", type=int, default=REPEAT, help="每个评估点固定完整重复次数，默认 10")
    parser.add_argument("--labels", default=LABELS, help="逗号分隔评估点")
    parser.add_argument("--condition", action="append", default=None, help="只处理指定 condition，可重复传入")
    parser.add_argument("--max-parallel", type=int, default=MAX_PARALLEL, help="并行任务数")
    parser.add_argument("--force", action=argparse.BooleanOptionalAction, default=FORCE, help="覆盖已有重复输出")
    parser.add_argument(
        "--resume-partial",
        action=argparse.BooleanOptionalAction,
        default=RESUME_PARTIAL,
        help="复用已完整生成的 answered/scored，只补齐缺失的回答、评分和后续步骤",
    )
    parser.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=DRY_RUN, help="只打印计划，不实际执行")
    parser.add_argument("--report-only", action=argparse.BooleanOptionalAction, default=REPORT_ONLY, help="只基于已有重复结果重新生成报告")
    parser.add_argument(
        "--cleanup-completed-artifacts",
        action=argparse.BooleanOptionalAction,
        default=CLEANUP_COMPLETED_ARTIFACTS,
        help="完整报告写入并校验后，清理复评 job/trace 和 experiment_data 中重复的阶段快照",
    )
    parser.add_argument("--reset-target-depression-state", action=argparse.BooleanOptionalAction, default=RESET_TARGET_DEPRESSION_STATE, help="把目标患者动态抑郁状态重置为初始 depression_config 状态后复评")
    parser.add_argument(
        "--allow-legacy-final-storage",
        action=argparse.BooleanOptionalAction,
        default=ALLOW_LEGACY_FINAL_STORAGE,
        help="允许旧 staged job 复用最终 storage；结果会被标记为非严格历史快照",
    )
    parser.add_argument("--output-group", default=OUTPUT_GROUP, help="复评输出使用的新 group 标签，例如 G8；留空则沿用原始 condition")
    parser.add_argument(
        "--require-controller-manifest",
        action="store_true",
        help="要求并校验生成阶段 CBT condition manifest",
    )
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
        resume_partial=bool(args.resume_partial),
        dry_run=bool(args.dry_run),
        report_only=bool(args.report_only),
        reset_target_depression_state=bool(args.reset_target_depression_state),
        allow_legacy_final_storage=bool(args.allow_legacy_final_storage),
        output_group=str(args.output_group or "").strip().lower(),
        cleanup_completed_artifacts=bool(args.cleanup_completed_artifacts),
        require_controller_manifest=bool(args.require_controller_manifest),
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
    print(f"  resume-partial:  {cfg.resume_partial}")
    print(f"  dry-run:         {cfg.dry_run}")
    print(f"  report-only:     {cfg.report_only}")
    print(f"  cleanup complete:{cfg.cleanup_completed_artifacts}")
    print(f"  aggregation:     {AGGREGATION_METHOD_VERSION}")
    print(f"  reset depression:{cfg.reset_target_depression_state}")
    print(f"  legacy storage:  {cfg.allow_legacy_final_storage}")
    print(f"  output group:    {cfg.output_group or '(source)'}")
    print(f"  controller manifest required: {cfg.require_controller_manifest}")
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


def generation_manifest_output_path(
    cfg: RuntimeConfig,
    condition_name: str,
) -> Path:
    return condition_output_dir(cfg, condition_name) / MANIFEST_FILENAME


def generation_manifest_for_condition(
    cfg: RuntimeConfig,
    condition: Mapping[str, Any],
) -> dict[str, Any] | None:
    embedded = condition.get("controller_manifest")
    manifest: dict[str, Any] | None = (
        copy.deepcopy(dict(embedded))
        if isinstance(embedded, Mapping)
        else None
    )
    source_path = str(condition.get("controller_manifest_path", "") or "").strip()
    if manifest is None and source_path:
        path = Path(source_path).expanduser()
        if not path.is_absolute():
            path = BASE_DIR / path
        manifest = load_optional_json(path.resolve())
    required = bool(getattr(cfg, "require_controller_manifest", False))
    if manifest is None:
        if required:
            raise ValueError(
                "生成阶段 summary 缺少 controller manifest: "
                + str(condition.get("condition_name", "") or "")
            )
        return None
    if str(manifest.get("artifact_kind", "") or "") != "cbt_experiment_condition":
        raise ValueError("生成阶段 controller manifest artifact_kind 无效")
    identity = manifest_identity(manifest)
    if (
        not identity["controller_identity"]
        or identity["mode"] not in {"legacy", "minimal"}
        or not identity["controller_version"]
    ):
        raise ValueError("生成阶段 controller manifest identity 不完整")
    if condition.get("controller_identity"):
        assert_identity_matches(
            condition,
            manifest,
            context="repeat eval generation summary",
        )
    expected_condition_key = str(condition.get("condition_key", "") or "")
    if expected_condition_key and (
        str(manifest.get("condition_key", "") or "") != expected_condition_key
    ):
        raise ValueError("repeat eval generation manifest condition key 不匹配")
    run_name = str(condition.get("run_name", "") or "")
    if run_name and str(manifest.get("run_name", "") or "") != run_name:
        raise ValueError("repeat eval generation manifest run_name 不匹配")
    checkpoint_manifest_path = checkpoints_root(cfg) / run_name / MANIFEST_FILENAME
    checkpoint_manifest = load_optional_json(checkpoint_manifest_path)
    if checkpoint_manifest is not None:
        assert_identity_matches(
            manifest,
            checkpoint_manifest,
            context=f"repeat eval checkpoint manifest {checkpoint_manifest_path}",
        )
        if str(checkpoint_manifest.get("condition_key", "") or "") != str(
            manifest.get("condition_key", "") or ""
        ):
            raise ValueError(
                "repeat eval checkpoint manifest condition key 不匹配"
            )
        if str(checkpoint_manifest.get("run_name", "") or "") != run_name:
            raise ValueError(
                "repeat eval checkpoint manifest run_name 不匹配"
            )
    return manifest


def prepare_generation_manifests(
    cfg: RuntimeConfig,
    original_summary: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    manifests: dict[str, dict[str, Any]] = {}
    for condition in selected_conditions(cfg, original_summary):
        condition_name = str(condition.get("condition_name", "") or "")
        manifest = generation_manifest_for_condition(cfg, condition)
        if manifest is None:
            continue
        output_path = generation_manifest_output_path(cfg, condition_name)
        existing = load_optional_json(output_path)
        if existing is not None:
            assert_identity_matches(
                manifest,
                existing,
                context=f"repeat eval resume manifest {output_path}",
            )
            if str(existing.get("condition_key", "") or "") != str(
                manifest.get("condition_key", "") or ""
            ):
                raise ValueError(
                    f"repeat eval resume manifest condition key 不匹配: {output_path}"
                )
            if str(existing.get("run_name", "") or "") != str(
                manifest.get("run_name", "") or ""
            ):
                raise ValueError(
                    f"repeat eval resume manifest run_name 不匹配: {output_path}"
                )
        if cfg.dry_run:
            print(
                f"[DRY-RUN] copy generation manifest: "
                f"{manifest.get('controller_identity')} -> {output_path}"
            )
        else:
            write_json_file(output_path, manifest)
        manifests[condition_name] = manifest
    return manifests


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
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        # Disk exhaustion can truncate a JSON artifact in the middle of a
        # multibyte character.  In resume mode it must be regenerated, not
        # allowed to abort task inspection or final report generation.
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


def normalized_item_ids(rows: list[dict[str, Any]]) -> list[int] | None:
    item_ids: list[int] = []
    for row in rows:
        try:
            item_id = int(row.get("id"))
        except (TypeError, ValueError):
            return None
        if item_id <= 0:
            return None
        item_ids.append(item_id)
    return item_ids


def expected_answer_item_ids(job: dict[str, Any], scale_name: str) -> list[int]:
    requested = (job.get("scale_item_ids", {}) or {}).get(scale_name)
    if requested is not None:
        return sorted({int(item_id) for item_id in requested})

    question_file = str((job.get("scale_question_files", {}) or {}).get(scale_name, "") or "").strip()
    if not question_file:
        raise ValueError(f"missing question file for scale: {scale_name}")
    question_path = (
        BASE_DIR
        / "customization"
        / "depression_scale_agent"
        / "questions"
        / "templates"
        / question_file
    )
    rows = load_jsonl(question_path)
    item_ids = normalized_item_ids(rows)
    if not rows or item_ids is None:
        raise ValueError(f"invalid question template: {question_path}")
    return sorted(item_ids)


def answer_file_complete(output_dir: Path, job: dict[str, Any], scale_name: str) -> bool:
    answers_path = output_dir / f"{scale_name}_answered.jsonl"
    if not answers_path.is_file():
        return False
    try:
        rows = load_jsonl(answers_path)
    except (OSError, json.JSONDecodeError):
        return False
    if not rows or any("answer" not in row for row in rows):
        return False
    observed_ids = normalized_item_ids(rows)
    if observed_ids is None or len(observed_ids) != len(set(observed_ids)):
        return False
    try:
        expected_ids = expected_answer_item_ids(job, scale_name)
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return False
    return sorted(observed_ids) == expected_ids


def completed_answer_spec(
    output_dir: Path,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Load the large worker job, or its compact post-completion protocol."""

    job = load_optional_json(output_dir / "job.json")
    if job:
        return job
    if metadata is None:
        metadata = load_optional_json(output_dir / "metadata.json")
    if not isinstance(metadata, dict):
        return None
    protocol = metadata.get("evaluation_protocol")
    if not isinstance(protocol, dict):
        return None
    question_files = protocol.get("scale_question_files")
    if not isinstance(question_files, dict) or not question_files:
        return None
    return protocol


def score_result_complete(
    path: Path,
    scale_name: str,
) -> bool:
    payload = load_optional_json(path)
    if not payload or payload.get("parse_error") or payload.get("raw_reply"):
        return False
    if str(payload.get("scale", "") or "") != scale_name:
        return False

    raw_items = payload.get(SCALE_ITEM_SCORE_KEYS.get(scale_name, ""))
    return (
        isinstance(raw_items, list)
        and len(raw_items) == expected_item_count(scale_name)
        and len(extract_scored_item_records(payload, scale_name)) == expected_item_count(scale_name)
    )


def resume_scales_for_answers(
    cfg: RuntimeConfig,
    output_dir: Path,
    job: dict[str, Any],
    target_scales: list[str],
) -> list[str]:
    if not cfg.resume_partial or cfg.force:
        return list(target_scales)
    return [
        scale_name
        for scale_name in target_scales
        if not answer_file_complete(output_dir, job, scale_name)
    ]


def run_answer_worker(
    cfg: RuntimeConfig,
    output_dir: Path,
    job: dict[str, Any],
    target_scales: list[str],
    *,
    description: str,
) -> tuple[Path | None, dict[str, Any]]:
    missing_scales = resume_scales_for_answers(cfg, output_dir, job, target_scales)
    reused_scales = [scale_name for scale_name in target_scales if scale_name not in missing_scales]
    if reused_scales:
        print(f"[RESUME] reuse answered: {output_dir} ({', '.join(reused_scales)})")
    if not missing_scales:
        print(f"[RESUME] all answers complete; skip staged worker: {output_dir}")
        return None, {
            "resumed_answer_scales": [],
            "reused_answer_scales": reused_scales,
        }

    worker_job = copy.deepcopy(job)
    worker_job["scales"] = missing_scales
    partial_resume = bool(reused_scales)
    job_path = output_dir / ("resume_job.json" if partial_resume else "job.json")
    result_path = output_dir / ("resume_worker_result.json" if partial_resume else "worker_result.json")
    worker_job["worker_result_path"] = str(result_path)

    if cfg.dry_run:
        print(f"[DRY-RUN] write {description}: {job_path} (scales={', '.join(missing_scales)})")
        print(f"[DRY-RUN] run staged worker: {STAGED_WORKER_SCRIPT} --job {job_path}")
    else:
        write_json_file(job_path, worker_job)
        cmd = [WORKER_PYTHON, str(STAGED_WORKER_SCRIPT), "--job", str(job_path)]
        print(f"[RUN] {' '.join(cmd)}")
        try:
            subprocess.run(cmd, cwd=BASE_DIR, check=True, timeout=POST_EVAL_TIMEOUT_SECONDS)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"staged worker failed with returncode={exc.returncode}\n"
                f"{worker_failure_detail(output_dir, result_path=result_path)}"
            ) from exc

    return result_path, {
        "resumed_answer_scales": missing_scales,
        "reused_answer_scales": reused_scales,
    }


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
    if text.startswith(("http://", "https://")):
        return text
    archive_root = str(cfg.archive_results_root)
    if text == "results":
        return archive_root
    if text.startswith(CHECKPOINT_RESULTS_PREFIX):
        return archive_root + text[len(CHECKPOINT_RESULTS_PREFIX):]
    if text == CHECKPOINT_RESULTS_PREFIX:
        return archive_root
    if text.startswith("results/"):
        return archive_root + text[len("results"):]
    if text.endswith("/results"):
        return archive_root
    marker = "/results/"
    if marker in text:
        return archive_root + "/" + text.split(marker, 1)[1]
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


def env_flag_enabled(name: str) -> bool:
    value = str(os.environ.get(name, "") or "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def apply_runtime_model_env_override(runtime_config: dict[str, Any]) -> dict[str, Any]:
    if not env_flag_enabled(REPEAT_VLLM_ENV):
        return runtime_config
    if not isinstance(runtime_config, dict):
        return runtime_config

    think_payload = {
        "provider": "openai",
        "model": os.environ.get("GA_REPEAT_VLLM_THINK_MODEL", "qwen3-8b-vllm"),
        "base_url": os.environ.get("GA_REPEAT_VLLM_THINK_BASE_URL", "http://127.0.0.1:18000/v1"),
        "api_key": os.environ.get("GA_REPEAT_VLLM_API_KEY", "EMPTY"),
    }
    embedding_payload = {
        "provider": "openai",
        "model": os.environ.get("GA_REPEAT_VLLM_EMBED_MODEL", "bge-m3-vllm"),
        "base_url": os.environ.get("GA_REPEAT_VLLM_EMBED_BASE_URL", "http://127.0.0.1:18001/v1"),
        "api_key": os.environ.get("GA_REPEAT_VLLM_API_KEY", "EMPTY"),
    }
    forced_llm_payload = {
        "enabled": True,
        "provider": "openai",
        "model": think_payload["model"],
        "base_url": think_payload["base_url"],
        "api_key_env": "GA_REPEAT_VLLM_API_KEY",
    }

    agent_base = runtime_config.setdefault("agent_base", {})
    if not isinstance(agent_base, dict):
        agent_base = {}
        runtime_config["agent_base"] = agent_base

    think_cfg = agent_base.setdefault("think", {})
    if not isinstance(think_cfg, dict):
        think_cfg = {}
        agent_base["think"] = think_cfg
    llm_cfg = think_cfg.setdefault("llm", {})
    if not isinstance(llm_cfg, dict):
        llm_cfg = {}
        think_cfg["llm"] = llm_cfg
    llm_cfg.update(think_payload)

    associate_cfg = agent_base.setdefault("associate", {})
    if not isinstance(associate_cfg, dict):
        associate_cfg = {}
        agent_base["associate"] = associate_cfg
    embedding_cfg = associate_cfg.setdefault("embedding", {})
    if not isinstance(embedding_cfg, dict):
        embedding_cfg = {}
        associate_cfg["embedding"] = embedding_cfg
    embedding_cfg.update(embedding_payload)

    agents = runtime_config.get("agents", {})
    if isinstance(agents, dict):
        for agent_cfg in agents.values():
            if not isinstance(agent_cfg, dict):
                continue
            local_think = agent_cfg.get("think")
            if isinstance(local_think, dict) and isinstance(local_think.get("llm"), dict):
                local_think["llm"].update(think_payload)
            local_associate = agent_cfg.get("associate")
            if isinstance(local_associate, dict) and isinstance(local_associate.get("embedding"), dict):
                local_associate["embedding"].update(embedding_payload)

    intervention_cfg = runtime_config.setdefault("intervention", {})
    if not isinstance(intervention_cfg, dict):
        intervention_cfg = {}
        runtime_config["intervention"] = intervention_cfg
    forced_llm_cfg = intervention_cfg.setdefault("forced_llm", {})
    if not isinstance(forced_llm_cfg, dict):
        forced_llm_cfg = {}
        intervention_cfg["forced_llm"] = forced_llm_cfg
    forced_llm_cfg.update(forced_llm_payload)

    runtime_config["_repeat_runtime_model_override"] = {
        "enabled": True,
        "source": REPEAT_VLLM_ENV,
        "think_llm": copy.deepcopy(think_payload),
        "embedding": copy.deepcopy(embedding_payload),
        "forced_llm": copy.deepcopy(forced_llm_payload),
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


def public_model_config(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        return {}
    allowed = ("provider", "model", "base_url", "temperature", "top_p", "seed")
    return {key: copy.deepcopy(raw.get(key)) for key in allowed if key in raw}


def evaluation_protocol_metadata(cfg: RuntimeConfig, job: dict[str, Any]) -> dict[str, Any]:
    runtime_config = job.get("runtime_config", {}) if isinstance(job.get("runtime_config"), dict) else {}
    agent_config = runtime_config.get("agent", {}) if isinstance(runtime_config.get("agent"), dict) else {}
    think_config = agent_config.get("think", {}) if isinstance(agent_config.get("think"), dict) else {}
    target_agent = str(job.get("target_agent", "") or "")
    agents_config = runtime_config.get("agents", {}) if isinstance(runtime_config.get("agents"), dict) else {}
    target_config = agents_config.get(target_agent, {}) if isinstance(agents_config.get(target_agent), dict) else {}
    target_think_config = target_config.get("think", {}) if isinstance(target_config.get("think"), dict) else {}
    intervention = runtime_config.get("intervention", {}) if isinstance(runtime_config.get("intervention"), dict) else {}
    scale_question_files = job.get("scale_question_files", {}) if isinstance(job.get("scale_question_files"), dict) else {}
    scale_item_ids = job.get("scale_item_ids", {}) if isinstance(job.get("scale_item_ids"), dict) else {}
    return {
        "method_version": AGGREGATION_METHOD_VERSION,
        "expected_repeats": cfg.repeat,
        "scales": list(SCALES),
        "scale_order": list(SCALES),
        "scale_question_files": {
            scale_name: str(scale_question_files.get(scale_name, SCALES[scale_name]["question_file"]))
            for scale_name in SCALES
        },
        "scale_item_ids": {
            scale_name: list(scale_item_ids[scale_name])
            for scale_name in SCALES
            if isinstance(scale_item_ids.get(scale_name), list)
        },
        "scale_scoring_prompts": {
            scale_name: SCALES[scale_name]["scoring_prompt"]
            for scale_name in SCALES
        },
        "snapshot_name": str(job.get("snapshot_name", "") or ""),
        "target_agent": target_agent,
        "context_isolation": "fresh_worker_and_chat_session_per_repeat; answer_without_memory_per_item",
        "think_llm": public_model_config(think_config.get("llm")),
        "target_agent_think_llm": public_model_config(target_think_config.get("llm")),
        "forced_llm": public_model_config(intervention.get("forced_llm")),
        "seed_controlled": bool(
            (isinstance(think_config.get("llm"), dict) and think_config["llm"].get("seed") is not None)
            or (
                isinstance(target_think_config.get("llm"), dict)
                and target_think_config["llm"].get("seed") is not None
            )
        ),
        "api_call_date": datetime.now().isoformat(timespec="seconds"),
    }


def condition_evaluation(condition: dict[str, Any], label: str) -> dict[str, Any] | None:
    for evaluation in condition.get("evaluations", []) or []:
        if str(evaluation.get("trigger_label", "")) == label:
            return evaluation
    return None


def canonical_json_sha256(payload: Any) -> str:
    return _canonical_json_sha256(payload)


def file_sha256(path: Path) -> str:
    return _file_sha256(path)


def resolve_bundle_path(source_dir: Path, raw_relpath: Any, field_name: str) -> Path:
    relpath_text = str(raw_relpath or "").strip()
    relpath = Path(relpath_text)
    if not relpath_text or relpath.is_absolute():
        raise ValueError(f"invalid snapshot bundle {field_name}: {raw_relpath}")
    source_root = source_dir.resolve()
    resolved = (source_root / relpath).resolve()
    try:
        resolved.relative_to(source_root)
    except ValueError as exc:
        raise ValueError(f"snapshot bundle {field_name} escapes trigger directory: {raw_relpath}") from exc
    return resolved


def validate_snapshot_bundle(source_dir: Path, job: dict[str, Any], label: str) -> Path:
    bundle = job.get("snapshot_bundle")
    if not isinstance(bundle, dict) or not bundle:
        raise ValueError(f"staged job has no historical snapshot bundle: {source_dir / 'job.json'}")
    if int(bundle.get("schema_version", 0) or 0) != SNAPSHOT_BUNDLE_SCHEMA_VERSION:
        raise ValueError(f"unsupported snapshot bundle schema: {bundle.get('schema_version')}")
    if str(bundle.get("storage_scope", "") or "") != "full":
        raise ValueError(f"snapshot bundle is not full storage: {source_dir}")

    storage_root = resolve_bundle_path(source_dir, bundle.get("storage_relpath"), "storage_relpath")
    manifest_path = resolve_bundle_path(source_dir, bundle.get("manifest_relpath"), "manifest_relpath")
    if not storage_root.is_dir():
        raise FileNotFoundError(f"snapshot storage not found: {storage_root}")
    manifest = load_optional_json(manifest_path)
    if manifest is None:
        raise FileNotFoundError(f"snapshot manifest not found or invalid: {manifest_path}")

    if int(manifest.get("schema_version", 0) or 0) != SNAPSHOT_BUNDLE_SCHEMA_VERSION:
        raise ValueError(f"snapshot manifest schema mismatch: {manifest_path}")
    if str(manifest.get("artifact_kind", "") or "") != "repeat_eval_snapshot":
        raise ValueError(f"snapshot manifest artifact kind mismatch: {manifest_path}")
    if str(manifest.get("trigger_label", "") or "") != str(label or ""):
        raise ValueError(f"snapshot manifest label mismatch: expected={label} path={manifest_path}")

    runtime_digest = canonical_json_sha256(job.get("runtime_config", {}))
    conversation_digest = canonical_json_sha256(job.get("conversation", {}) or {})
    expected_runtime_digest = str(manifest.get("runtime_config_sha256", "") or "")
    expected_conversation_digest = str(manifest.get("conversation_sha256", "") or "")
    if runtime_digest != expected_runtime_digest or runtime_digest != str(bundle.get("runtime_config_sha256", "") or ""):
        raise ValueError(f"snapshot runtime_config digest mismatch: {source_dir / 'job.json'}")
    if conversation_digest != expected_conversation_digest or conversation_digest != str(bundle.get("conversation_sha256", "") or ""):
        raise ValueError(f"snapshot conversation digest mismatch: {source_dir / 'job.json'}")

    files = manifest.get("files")
    if not isinstance(files, list):
        raise ValueError(f"snapshot manifest files must be a list: {manifest_path}")
    seen_paths: set[str] = set()
    total_size = 0
    for record in files:
        if not isinstance(record, dict):
            raise ValueError(f"invalid snapshot manifest file record: {manifest_path}")
        relpath = str(record.get("path", "") or "")
        if not relpath or relpath in seen_paths:
            raise ValueError(f"invalid or duplicate snapshot file path: {relpath}")
        seen_paths.add(relpath)
        file_path = resolve_bundle_path(storage_root, relpath, "files.path")
        if not file_path.is_file():
            raise FileNotFoundError(f"snapshot bundle file missing: {file_path}")
        size = file_path.stat().st_size
        expected_size = int(record.get("size", -1))
        if size != expected_size:
            raise ValueError(f"snapshot bundle file size mismatch: {file_path}")
        if file_sha256(file_path) != str(record.get("sha256", "") or ""):
            raise ValueError(f"snapshot bundle file hash mismatch: {file_path}")
        total_size += size
    actual_paths = {
        path.relative_to(storage_root).as_posix()
        for path in storage_root.rglob("*")
        if path.is_file()
    }
    if actual_paths != seen_paths:
        missing = sorted(seen_paths - actual_paths)
        unexpected = sorted(actual_paths - seen_paths)
        raise ValueError(
            f"snapshot bundle file set mismatch: missing={missing[:5]} unexpected={unexpected[:5]}"
        )
    if len(files) != int(manifest.get("file_count", -1)):
        raise ValueError(f"snapshot bundle file count mismatch: {manifest_path}")
    if total_size != int(manifest.get("total_size", -1)):
        raise ValueError(f"snapshot bundle total size mismatch: {manifest_path}")
    return storage_root


def target_external_memory_enabled(runtime_config: dict[str, Any], target_agent: str) -> bool:
    agent_base = runtime_config.get("agent_base", {}) if isinstance(runtime_config, dict) else {}
    base_external = agent_base.get("external_memory", {}) if isinstance(agent_base, dict) else {}
    agents = runtime_config.get("agents", {}) if isinstance(runtime_config, dict) else {}
    target_cfg = agents.get(target_agent, {}) if isinstance(agents, dict) else {}
    local_external = target_cfg.get("external_memory", {}) if isinstance(target_cfg, dict) else {}
    effective = copy.deepcopy(base_external) if isinstance(base_external, dict) else {}
    if isinstance(local_external, dict):
        effective.update(local_external)
    return bool(effective.get("enabled", False))


def build_staged_job(
    cfg: RuntimeConfig,
    condition: dict[str, Any],
    label: str,
    output_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    run_name = str(condition.get("run_name", "") or "").strip()
    checkpoint_dir = checkpoints_root(cfg) / run_name
    source_dir = checkpoint_dir / "staged_eval" / label
    source_job_path = source_dir / "job.json"
    source_job = load_optional_json(source_job_path)
    if source_job is None:
        raise FileNotFoundError(f"missing staged job: {source_job_path}")
    metadata = load_optional_json(source_dir / "metadata.json") or {}
    source_target_agent = str(
        source_job.get("target_agent", "") or target_agent_for_condition(condition, checkpoint_dir)
    )
    source_runtime_config = source_job.get("runtime_config", {})
    if not isinstance(source_runtime_config, dict):
        raise ValueError(f"staged job runtime_config must be an object: {source_job_path}")
    if target_external_memory_enabled(source_runtime_config, source_target_agent):
        raise ValueError(
            f"strict historical repeat does not support external memory reads: {source_job_path}"
        )

    if isinstance(source_job.get("snapshot_bundle"), dict) and source_job.get("snapshot_bundle"):
        storage_source_root = validate_snapshot_bundle(source_dir, source_job, label)
        integrity_metadata = {
            "historical_snapshot_integrity": "verified_bundle",
            "snapshot_bundle_verified": True,
            "snapshot_bundle_manifest": str(source_dir / str(source_job["snapshot_bundle"].get("manifest_relpath", ""))),
        }
    else:
        if not bool(getattr(cfg, "allow_legacy_final_storage", False)):
            raise ValueError(
                "legacy staged job has no timepoint storage snapshot; "
                "rerun with --allow-legacy-final-storage only if an unverified historical input is acceptable: "
                f"{source_job_path}"
            )
        storage_source_root = checkpoint_dir / "storage"
        if not storage_source_root.is_dir():
            raise FileNotFoundError(f"legacy final storage not found: {storage_source_root}")
        integrity_metadata = {
            "historical_snapshot_integrity": "unverified_legacy",
            "snapshot_bundle_verified": False,
            "legacy_final_storage_used": True,
        }

    job = rewrite_paths(copy.deepcopy(source_job), cfg)
    target_agent = str(job.get("target_agent", "") or target_agent_for_condition(condition, checkpoint_dir))
    job["target_agent"] = target_agent
    runtime_config = job.get("runtime_config", {})
    if isinstance(runtime_config, dict):
        job["runtime_config"] = reset_target_depression_state_if_needed(
            cfg,
            runtime_config,
            target_agent,
        )
        job["runtime_config"] = apply_runtime_model_env_override(job["runtime_config"])
    job["trigger_dir"] = str(output_dir)
    job["worker_result_path"] = str(output_dir / "worker_result.json")
    job["tmp_root_parent"] = str(output_dir / "_tmp")
    job["storage_source_root"] = str(storage_source_root)
    job["cleanup_tmp_storage"] = True
    job["run_name"] = run_name
    metadata.update(integrity_metadata)
    return job, metadata


def build_post_job(
    cfg: RuntimeConfig,
    condition: dict[str, Any],
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
    runtime_config = apply_runtime_model_env_override(runtime_config)
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


def worker_failure_detail(output_dir: Path, *, result_path: Path | None = None) -> str:
    result_path = result_path or (output_dir / "worker_result.json")
    worker_result = load_optional_json(result_path) or {}
    if not worker_result:
        return f"worker result not found: {result_path}"
    parts = []
    status = str(worker_result.get("status", "") or "")
    if status:
        parts.append(f"worker_status={status}")
    error = str(worker_result.get("error", "") or "")
    if error:
        parts.append(f"error={error}")
    traceback_text = str(worker_result.get("traceback", "") or "")
    if traceback_text:
        lines = traceback_text.strip().splitlines()
        tail = "\n".join(lines[-12:])
        parts.append("traceback_tail:\n" + tail)
    return "\n".join(parts) if parts else f"worker_result has no error detail: {result_path}"


def task_complete(cfg: RuntimeConfig, task: RepeatTask) -> bool:
    condition_name = str(task.condition.get("condition_name", ""))
    output_dir = repeat_label_dir(cfg, condition_name, task.repeat_idx, task.label)
    metadata = load_optional_json(output_dir / "metadata.json")
    if not isinstance(metadata, dict) or not metadata:
        return False
    metadata_repeat = metadata.get("repeat")
    if (
        str(metadata.get("condition_name", "") or "") != condition_name
        or str(metadata.get("trigger_label", "") or "") != task.label
        or not isinstance(metadata_repeat, int)
        or isinstance(metadata_repeat, bool)
        or metadata_repeat != task.repeat_idx
        or not isinstance(metadata.get("evaluation_protocol"), dict)
    ):
        return False
    answer_spec = completed_answer_spec(output_dir, metadata)
    if not answer_spec:
        return False
    return all(
        answer_file_complete(output_dir, answer_spec, scale_name)
        and score_result_complete(output_dir / f"{scale_name}_scored.json", scale_name)
        for scale_name in SCALES
    )


def select_resumable_job(
    cfg: RuntimeConfig,
    job_path: Path,
    rebuilt_job: dict[str, Any],
) -> tuple[dict[str, Any], bool]:
    if not cfg.resume_partial or cfg.force:
        return rebuilt_job, False
    existing_job = load_optional_json(job_path)
    if not isinstance(existing_job, dict) or not existing_job:
        return rebuilt_job, False
    return existing_job, True


def run_repeat_task(cfg: RuntimeConfig, task: RepeatTask) -> dict[str, Any]:
    condition = task.condition
    condition_name = str(condition.get("condition_name", ""))
    run_name = str(condition.get("run_name", "") or "").strip()
    output_dir = repeat_label_dir(cfg, condition_name, task.repeat_idx, task.label)
    checkpoint_dir = checkpoints_root(cfg) / run_name
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_dir}")
    generation_manifest = generation_manifest_for_condition(cfg, condition)

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
        job, source_metadata = build_post_job(cfg, condition, output_dir)
    else:
        job, source_metadata = build_staged_job(cfg, condition, task.label, output_dir)

    job_path = output_dir / "job.json"
    active_job, preserve_existing_job = select_resumable_job(cfg, job_path, job)
    if not cfg.dry_run and not preserve_existing_job:
        write_json_file(job_path, active_job)

    metadata = copy.deepcopy(source_metadata)
    metadata.update(
        {
            "repeat_batch_name": cfg.name,
            "repeat": task.repeat_idx,
            "condition_name": condition_name,
            "run_name": run_name,
            "trigger_label": task.label,
            "source_repeat_mode": "fixed_complete_scale_repeat",
            "evaluation_protocol": evaluation_protocol_metadata(cfg, active_job),
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "generation_controller_manifest": (
                str(generation_manifest_output_path(cfg, condition_name))
                if generation_manifest is not None
                else ""
            ),
            "controller_identity": (
                str(generation_manifest.get("controller_identity", "") or "")
                if generation_manifest is not None
                else ""
            ),
            "controller_version": (
                str(generation_manifest.get("controller_version", "") or "")
                if generation_manifest is not None
                else ""
            ),
            "progressive_stage": (
                generation_manifest.get("progressive_stage")
                if generation_manifest is not None
                else None
            ),
            **repeat_metadata_overrides(cfg, condition),
        }
    )

    result_path, resume_metadata = run_answer_worker(
        cfg,
        output_dir,
        active_job,
        list(SCALES),
        description="job",
    )

    if not cfg.dry_run:
        for scale_name in SCALES:
            if not answer_file_complete(output_dir, active_job, scale_name):
                raise RuntimeError(f"answer worker produced incomplete answers: {output_dir} ({scale_name})")

    for scale_name in SCALES:
        scored_path = output_dir / f"{scale_name}_scored.json"
        if cfg.resume_partial and not cfg.force and score_result_complete(scored_path, scale_name):
            print(f"[RESUME] reuse score: {scored_path}")
            continue
        score_scale_answers(output_dir, scale_name, dry_run=cfg.dry_run)
        if not cfg.dry_run and not score_result_complete(scored_path, scale_name):
            raise RuntimeError(f"score worker produced invalid item scores: {scored_path}")

    if not cfg.dry_run:
        worker_result = load_optional_json(output_dir / "worker_result.json") or {}
        metadata["worker_result"] = worker_result
        metadata.update(resume_metadata)
        if result_path and result_path.name != "worker_result.json":
            metadata["resume_worker_result"] = load_optional_json(result_path) or {}
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
        results = []
        for task in tasks:
            condition_name = str(task.condition.get("condition_name", ""))
            print(f"[DRY-RUN] {condition_name} r{task.repeat_idx:02d} {task.label}")
            results.append(run_repeat_task(cfg, task))
        return results

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


def stddev(values: list[float]) -> float | None:
    """Sample standard deviation (ddof=1)."""
    if len(values) < 2:
        return None
    avg = sum(values) / len(values)
    return math.sqrt(sum((value - avg) ** 2 for value in values) / (len(values) - 1))


def linear_quantile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * probability
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return ordered[lower_index]
    fraction = position - lower_index
    return ordered[lower_index] + fraction * (ordered[upper_index] - ordered[lower_index])


def describe_scores(values: list[float]) -> dict[str, Any]:
    scores = [float(value) for value in values]
    if not scores:
        return {
            "count": 0,
            "mean": None,
            "sample_sd": None,
            "mean_absolute_deviation": None,
            "ci95": {"lower": None, "upper": None, "method": "student_t", "df": None},
            "median": None,
            "q1": None,
            "q3": None,
            "iqr": None,
            "trimmed_mean_drop_one_each": None,
            "min": None,
            "max": None,
        }
    score_mean = sum(scores) / len(scores)
    sample_sd = stddev(scores)
    ci_lower = None
    ci_upper = None
    degrees_of_freedom = len(scores) - 1 if len(scores) >= 2 else None
    if sample_sd is not None and degrees_of_freedom is not None:
        critical = float(student_t.ppf(0.975, degrees_of_freedom))
        margin = critical * sample_sd / math.sqrt(len(scores))
        ci_lower = score_mean - margin
        ci_upper = score_mean + margin
    q1 = linear_quantile(scores, 0.25)
    q3 = linear_quantile(scores, 0.75)
    ordered = sorted(scores)
    trimmed_mean = None
    if len(ordered) >= 3:
        trimmed_values = ordered[1:-1]
        trimmed_mean = sum(trimmed_values) / len(trimmed_values)
    return {
        "count": len(scores),
        "mean": score_mean,
        "sample_sd": sample_sd,
        "mean_absolute_deviation": sum(abs(value - score_mean) for value in scores) / len(scores),
        "ci95": {
            "lower": ci_lower,
            "upper": ci_upper,
            "method": "student_t",
            "df": degrees_of_freedom,
        },
        "median": linear_quantile(scores, 0.5),
        "q1": q1,
        "q3": q3,
        "iqr": (q3 - q1) if q1 is not None and q3 is not None else None,
        "trimmed_mean_drop_one_each": trimmed_mean,
        "min": min(scores),
        "max": max(scores),
    }


def summarize_categories(rows: list[dict[str, Any]]) -> dict[str, Any]:
    categories = [str(row.get("severity", "") or "") for row in rows if row.get("severity")]
    if not categories:
        return {
            "counts": {},
            "proportions": {},
            "modal_categories": [],
            "modal_confidence": None,
            "uncertainty": None,
            "pairwise_flip_rate": None,
            "any_flip": False,
        }
    counts = Counter(categories)
    total = len(categories)
    highest = max(counts.values())
    modal_categories = sorted(category for category, count in counts.items() if count == highest)
    agreeing_pairs = sum(count * (count - 1) // 2 for count in counts.values())
    total_pairs = total * (total - 1) // 2
    return {
        "counts": {category: int(count) for category, count in sorted(counts.items())},
        "proportions": {category: count / total for category, count in sorted(counts.items())},
        "modal_categories": modal_categories,
        "modal_confidence": highest / total,
        "uncertainty": 1.0 - highest / total,
        "pairwise_flip_rate": (1.0 - agreeing_pairs / total_pairs) if total_pairs else 0.0,
        "any_flip": len(counts) > 1,
    }


def summarize_scale_repeats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    expected_repeats = len(rows)
    valid_rows = [row for row in rows if row.get("status") == "ok" and row.get("total_score") is not None]
    scores = [float(row["total_score"]) for row in valid_rows]
    missing_repeats = [
        int(row.get("repeat", 0) or 0)
        for row in rows
        if row.get("status") != "ok" or row.get("total_score") is None
    ]
    complete = len(valid_rows) == expected_repeats and not missing_repeats
    diagnostics = describe_scores(scores)
    categories = summarize_categories(valid_rows)
    sensitivity = {
        "median_total": diagnostics["median"] if complete else None,
        "trimmed_mean_drop_one_each": diagnostics["trimmed_mean_drop_one_each"] if complete else None,
        "r01_total": next(
            (float(row["total_score"]) for row in valid_rows if int(row.get("repeat", 0) or 0) == 1),
            None,
        ) if complete else None,
    }
    modes = categories["modal_categories"] if complete else []
    severity = modes[0] if len(modes) == 1 else ("并列（{}）".format("/".join(modes)) if modes else "—")
    return {
        "total_score": diagnostics["mean"] if complete else None,
        "severity": severity,
        "score_source": "repeat_reviewed_total_mean",
        "scored_file": "",
        "repeat_runs": rows,
        "expected_repeats": expected_repeats,
        "valid_repeats": len(valid_rows),
        "missing_repeats": missing_repeats,
        "aggregation_status": "complete" if complete else "incomplete",
        "mean": diagnostics["mean"] if complete else None,
        "sample_sd": diagnostics["sample_sd"] if complete else None,
        "stddev": diagnostics["sample_sd"] if complete else None,
        "mean_absolute_deviation": diagnostics["mean_absolute_deviation"] if complete else None,
        "ci95": diagnostics["ci95"] if complete else {"lower": None, "upper": None, "method": "student_t", "df": None},
        "median": diagnostics["median"] if complete else None,
        "q1": diagnostics["q1"] if complete else None,
        "q3": diagnostics["q3"] if complete else None,
        "iqr": diagnostics["iqr"] if complete else None,
        "trimmed_mean_drop_one_each": diagnostics["trimmed_mean_drop_one_each"] if complete else None,
        "min": diagnostics["min"] if complete else None,
        "max": diagnostics["max"] if complete else None,
        "category_counts": categories["counts"] if complete else {},
        "category_proportions": categories["proportions"] if complete else {},
        "modal_categories": modes,
        "modal_confidence": categories["modal_confidence"] if complete else None,
        "category_uncertainty": categories["uncertainty"] if complete else None,
        "category_pairwise_flip_rate": categories["pairwise_flip_rate"] if complete else None,
        "any_category_flip": categories["any_flip"] if complete else False,
        "severity_votes": categories["counts"] if complete else {},
        "sensitivity_analysis": sensitivity,
        "partial_statistics": diagnostics if not complete else None,
    }


def load_repeat_rows(cfg: RuntimeConfig, condition_name: str, label: str, scale_name: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for repeat_idx in range(1, cfg.repeat + 1):
        output_dir = repeat_label_dir(cfg, condition_name, repeat_idx, label)
        scored_path = output_dir / f"{scale_name}_scored.json"
        answers_path = output_dir / f"{scale_name}_answered.jsonl"
        scored = load_optional_json(scored_path)
        metadata = load_optional_json(output_dir / "metadata.json")
        answer_spec = completed_answer_spec(output_dir, metadata)
        item_records = extract_scored_item_records(scored, scale_name)
        answers_complete = bool(answer_spec) and answer_file_complete(
            output_dir,
            answer_spec,
            scale_name,
        )
        try:
            answer_rows = load_jsonl(answers_path) if answers_complete else []
        except (OSError, json.JSONDecodeError):
            answer_rows = []
            answers_complete = False
        score_complete = score_result_complete(scored_path, scale_name)
        if scored is None or not answers_complete or not score_complete:
            rows.append(
                {
                    "repeat": repeat_idx,
                    "status": "missing" if scored is None else "invalid",
                    "answer_file": str(answers_path),
                    "scored_file": str(scored_path),
                    "total_score": None,
                    "reported_total_score": scored.get("total_score") if isinstance(scored, dict) else None,
                    "llm_item_scores": [],
                    "llm_item_sum_total": None,
                    "reviewed_item_scores": [],
                    "reviewed_total_score": None,
                    "severity": "—",
                    "item_scores": [],
                    "direct_answer_count": 0,
                    "direct_answer_reviews": [],
                    "answer_override_count": 0,
                    "answer_overrides": [],
                }
            )
            continue
        review = build_reviewed_item_scores(item_records, answer_rows, scale_name)
        reviewed_total = float(review["reviewed_total_score"])
        rows.append(
            {
                "repeat": repeat_idx,
                "status": "ok",
                "answer_file": str(answers_path),
                "scored_file": str(scored_path),
                "score_source": "reviewed_item_sum",
                "total_score": reviewed_total,
                "reported_total_score": scored.get("total_score"),
                "llm_item_scores": review["llm_item_scores"],
                "llm_item_sum_total": review["llm_item_sum_total"],
                "reviewed_item_scores": review["reviewed_item_scores"],
                "reviewed_total_score": reviewed_total,
                "severity": expected_scale_severity(scale_name, reviewed_total),
                "item_scores": review["reviewed_item_scores"],
                "direct_answer_count": review["direct_answer_count"],
                "direct_answer_reviews": review["direct_answer_reviews"],
                "answer_override_count": review["answer_override_count"],
                "answer_overrides": review["answer_overrides"],
            }
        )
    return rows


def expected_item_count(scale_name: str) -> int:
    return SCALE_ITEM_COUNTS[scale_name]


def normalize_score(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    score = value
    if score < 0 or score > 3:
        return None
    return score


def parse_scored_item_id(item_label: Any, scale_name: str) -> int | None:
    text = str(item_label or "").strip().upper()
    if scale_name == "PHQ-9":
        identifier_pattern = r"PHQ\s*(?:第\s*)?([0-9]+)"
        valid_ids = set(range(1, 10))
    elif scale_name == "BDI-II":
        identifier_pattern = r"BDI(?:-II)?\s*(?:第\s*)?([0-9]+)"
        valid_ids = set(range(1, 22))
    else:
        return None
    identifiers = [int(value) for value in re.findall(identifier_pattern, text)]
    prefix_match = re.match(rf"^{identifier_pattern}", text)
    if not prefix_match or len(identifiers) != 1 or identifiers[0] not in valid_ids:
        return None
    return identifiers[0]


def extract_unambiguous_answer_score(answer: str) -> tuple[int | None, str]:
    answer_score, status = extract_direct_answer_score(answer)
    if status != "direct" or answer_score is None:
        return answer_score, status

    score_values = {"0": 0, "1": 1, "2": 2, "3": 3, "零": 0, "一": 1, "二": 2, "两": 2, "三": 3}
    explicit_tokens = re.findall(
        r"(?:选|选择)\s*(?:了)?\s*([0-3零一二两三])|([0-3零一二两三])\s*分",
        str(answer or ""),
    )
    explicit_scores = {
        score_values[token]
        for groups in explicit_tokens
        for token in groups
        if token
    }
    if len(explicit_scores) > 1:
        return None, "ambiguous"
    return answer_score, status


def extract_scored_item_records(scored: dict[str, Any] | None, scale_name: str) -> dict[int, dict[str, Any]]:
    if not isinstance(scored, dict):
        return {}
    item_key = SCALE_ITEM_SCORE_KEYS.get(scale_name)
    items = scored.get(item_key) if item_key else None
    expected_count = expected_item_count(scale_name)
    if not isinstance(items, list) or len(items) != expected_count:
        return {}
    records: dict[int, dict[str, Any]] = {}
    for index, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            return {}
        item_id = parse_scored_item_id(item.get("item"), scale_name)
        if item_id != index or item_id in records:
            return {}
        score = normalize_score(item.get("score"))
        if score is None:
            return {}
        records[item_id] = {
            "item_id": item_id,
            "score": score,
            "item": str(item.get("item", "") or ""),
            "basis": str(item.get("basis", "") or ""),
        }
    return records


def build_reviewed_item_scores(
    item_records: dict[int, dict[str, Any]],
    answer_rows: list[dict[str, Any]],
    scale_name: str,
) -> dict[str, Any]:
    expected_count = expected_item_count(scale_name)
    llm_item_scores = [int(item_records[item_id]["score"]) for item_id in range(1, expected_count + 1)]
    reviewed_item_scores = list(llm_item_scores)
    direct_answer_count = 0
    direct_answer_reviews: list[dict[str, Any]] = []
    overrides: list[dict[str, Any]] = []

    for row in answer_rows:
        item_id = row.get("id") if isinstance(row, dict) else None
        if not isinstance(item_id, int) or isinstance(item_id, bool):
            continue
        if scale_name == "PHQ-9" and item_id == 10:
            continue
        if item_id < 1 or item_id > expected_count:
            continue
        answer_text = str(row.get("answer", "") or "")
        answer_score, status = extract_unambiguous_answer_score(answer_text)
        if status != "direct" or answer_score is None:
            continue
        direct_answer_count += 1
        llm_score = reviewed_item_scores[item_id - 1]
        review_detail = {
            "item_id": item_id,
            "llm_score": llm_score,
            "answer_score": int(answer_score),
            "overridden": llm_score != answer_score,
            "answer_excerpt": answer_text[:200],
        }
        direct_answer_reviews.append(review_detail)
        if llm_score == answer_score:
            continue
        reviewed_item_scores[item_id - 1] = int(answer_score)
        overrides.append(review_detail)

    return {
        "llm_item_scores": llm_item_scores,
        "llm_item_sum_total": float(sum(llm_item_scores)),
        "reviewed_item_scores": reviewed_item_scores,
        "reviewed_total_score": float(sum(reviewed_item_scores)),
        "direct_answer_count": direct_answer_count,
        "direct_answer_reviews": direct_answer_reviews,
        "answer_override_count": len(overrides),
        "answer_overrides": overrides,
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
                "historical_snapshot_integrity": str(
                    metadata.get("historical_snapshot_integrity", "") or ""
                ),
                "snapshot_bundle_verified": metadata.get("snapshot_bundle_verified"),
                "legacy_final_storage_used": bool(metadata.get("legacy_final_storage_used", False)),
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
    total_mismatches: list[dict[str, Any]] = []
    item_mismatches: list[dict[str, Any]] = []
    for result in results:
        for evaluation in result.get("evaluations", []):
            trigger_label = str(evaluation.get("trigger_label", "") or "—")
            for scale_name, scale_payload in (evaluation.get("scales", {}) or {}).items():
                for row in scale_payload.get("repeat_runs", []) or []:
                    if row.get("status") != "ok":
                        continue
                    repeat_idx = int(row.get("repeat", 0) or 0)
                    reported_total = row.get("reported_total_score")
                    llm_item_sum = row.get("llm_item_sum_total")
                    try:
                        reported_number = float(reported_total)
                        item_sum_number = float(llm_item_sum)
                    except (TypeError, ValueError):
                        reported_number = None
                        item_sum_number = None
                    if (
                        reported_number is not None
                        and item_sum_number is not None
                        and reported_number != item_sum_number
                    ):
                        total_mismatches.append(
                            {
                                "condition_name": result.get("condition_name", ""),
                                "trigger_label": f"{trigger_label}/r{repeat_idx:02d}",
                                "scale": scale_name,
                                "item_sum": item_sum_number,
                                "reported_total": reported_number,
                                "delta": reported_number - item_sum_number,
                            }
                        )
                    for override in row.get("answer_overrides", []) or []:
                        if not isinstance(override, dict):
                            continue
                        item_mismatches.append(
                            {
                                "condition_name": result.get("condition_name", ""),
                                "trigger_label": f"{trigger_label}/r{repeat_idx:02d}",
                                "scale": scale_name,
                                "item_id": override.get("item_id"),
                                "llm_score": override.get("llm_score"),
                                "answer_score": override.get("answer_score"),
                            }
                        )
    return {
        "total_mismatches": total_mismatches,
        "item_mismatches": item_mismatches,
    }


SENSITIVITY_METHODS = {
    "median_total": ("median_total", "median_total"),
    "trimmed_mean": ("trimmed_mean_drop_one_each", "trimmed_mean_drop_one_each"),
    "r01": ("r01_total", "r01_total"),
}


def sensitivity_condition_results(results: list[dict[str, Any]], metric_key: str) -> list[dict[str, Any]]:
    transformed: list[dict[str, Any]] = []
    for source_result in results:
        result = {
            key: copy.deepcopy(source_result.get(key))
            for key in (
                "condition_name",
                "run_name",
                "variant",
                "group",
                "severity",
                "source_condition_name",
                "source_group",
            )
        }
        evaluations = []
        for source_evaluation in source_result.get("evaluations", []) or []:
            scales = {}
            for scale_name, source_scale_payload in (source_evaluation.get("scales", {}) or {}).items():
                sensitivity = source_scale_payload.get("sensitivity_analysis", {}) or {}
                value = sensitivity.get(metric_key)
                scales[scale_name] = {
                    "total_score": value,
                    "severity": expected_scale_severity(scale_name, float(value)) if value is not None else "—",
                }
            evaluations.append(
                {
                    "trigger_label": source_evaluation.get("trigger_label", ""),
                    "completed_session_count": source_evaluation.get("completed_session_count", 0),
                    "sim_time": source_evaluation.get("sim_time", ""),
                    "snapshot_name": source_evaluation.get("snapshot_name", ""),
                    "source": "repeat_scale_eval_sensitivity",
                    "scales": scales,
                }
            )
        result["evaluations"] = evaluations
        apply_trajectory_deltas(evaluations)
        result["final_deltas"] = {
            scale_name: next(
                (
                    evaluation.get("scales", {}).get(scale_name, {}).get("delta_from_baseline")
                    for evaluation in reversed(evaluations)
                    if evaluation.get("scales", {}).get(scale_name, {}).get("delta_from_baseline") is not None
                ),
                None,
            )
            for scale_name in SCALES
        }
        transformed.append(result)
    return transformed


def build_sensitivity_analysis(cfg: RuntimeConfig, results: list[dict[str, Any]]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for method_name, (_label, metric_key) in SENSITIVITY_METHODS.items():
        method_results = sensitivity_condition_results(results, metric_key)
        payload[method_name] = {
            "conditions": method_results,
            "group_values": report_group_values(cfg),
            "variant_trajectory": aggregate_dimension_trajectory(
                method_results, dimension_key="variant", dimension_values=VARIANTS
            ),
            "group_trajectory": aggregate_dimension_trajectory(
                method_results, dimension_key="group", dimension_values=report_group_values(cfg)
            ),
            "severity_trajectory": aggregate_dimension_trajectory(
                method_results, dimension_key="severity", dimension_values=SEVERITIES
            ),
            "variant_final_delta": aggregate_final_deltas(
                method_results, dimension_key="variant", dimension_values=VARIANTS
            ),
            "group_final_delta": aggregate_final_deltas(
                method_results, dimension_key="group", dimension_values=report_group_values(cfg)
            ),
            "severity_final_delta": aggregate_final_deltas(
                method_results, dimension_key="severity", dimension_values=SEVERITIES
            ),
        }
    return payload


def calculate_icc_one_way(matrix: list[list[float]]) -> dict[str, Any]:
    target_count = len(matrix)
    repeat_count = len(matrix[0]) if matrix else 0
    base = {
        "method": "one_way_random_absolute_agreement",
        "target_count": target_count,
        "repeat_count": repeat_count,
        "icc_1_1": None,
        "icc_1_k": None,
        "reason": "",
    }
    if target_count < 2:
        base["reason"] = "insufficient_targets"
        return base
    if repeat_count < 2 or any(len(row) != repeat_count for row in matrix):
        base["reason"] = "insufficient_or_unequal_repeats"
        return base
    target_means = [sum(row) / repeat_count for row in matrix]
    grand_mean = sum(target_means) / target_count
    ms_between = repeat_count * sum((value - grand_mean) ** 2 for value in target_means) / (target_count - 1)
    within_sum_squares = sum(
        sum((value - target_mean) ** 2 for value in row)
        for row, target_mean in zip(matrix, target_means)
    )
    ms_within = within_sum_squares / (target_count * (repeat_count - 1))
    denominator = ms_between + (repeat_count - 1) * ms_within
    if math.isclose(denominator, 0.0, abs_tol=1e-12):
        base["reason"] = "zero_total_variance"
        return base
    base["icc_1_1"] = (ms_between - ms_within) / denominator
    if math.isclose(ms_between, 0.0, abs_tol=1e-12):
        base["reason"] = "zero_between_target_variance_for_icc_1_k"
    else:
        base["icc_1_k"] = (ms_between - ms_within) / ms_between
    base["ms_between"] = ms_between
    base["ms_within"] = ms_within
    return base


def build_reliability_analysis(results: list[dict[str, Any]], labels: list[str], repeat: int) -> dict[str, Any]:
    by_scale: dict[str, Any] = {}
    for scale_name in SCALES:
        label_payload: dict[str, Any] = {}
        for label in labels:
            matrix: list[list[float]] = []
            target_names: list[str] = []
            excluded_targets: list[str] = []
            for result in results:
                condition_name = str(result.get("condition_name", "") or "")
                evaluation = next(
                    (
                        item
                        for item in result.get("evaluations", []) or []
                        if str(item.get("trigger_label", "") or "") == label
                    ),
                    None,
                )
                scale_payload = (evaluation.get("scales", {}) or {}).get(scale_name, {}) if evaluation else {}
                rows = scale_payload.get("repeat_runs", []) if isinstance(scale_payload, dict) else []
                scores = [
                    float(row["total_score"])
                    for row in rows or []
                    if row.get("status") == "ok" and row.get("total_score") is not None
                ]
                if scale_payload.get("aggregation_status") == "complete" and len(scores) == repeat:
                    matrix.append(scores)
                    target_names.append(condition_name)
                else:
                    excluded_targets.append(condition_name)
            item = calculate_icc_one_way(matrix)
            item["targets"] = target_names
            item["excluded_targets"] = excluded_targets
            label_payload[label] = item
        by_scale[scale_name] = label_payload
    return {
        "unit": "condition/persona target within each scale and trigger label",
        "complete_repeats_required": repeat,
        "by_scale": by_scale,
    }


def build_report_completion(results: list[dict[str, Any]]) -> dict[str, Any]:
    incomplete_targets: list[dict[str, Any]] = []
    total_targets = 0
    for result in results:
        for evaluation in result.get("evaluations", []) or []:
            for scale_name, scale_payload in (evaluation.get("scales", {}) or {}).items():
                total_targets += 1
                if scale_payload.get("aggregation_status") == "complete":
                    continue
                incomplete_targets.append(
                    {
                        "condition_name": result.get("condition_name", ""),
                        "trigger_label": evaluation.get("trigger_label", ""),
                        "scale": scale_name,
                        "missing_repeats": list(scale_payload.get("missing_repeats", []) or []),
                        "valid_repeats": int(scale_payload.get("valid_repeats", 0) or 0),
                        "expected_repeats": int(scale_payload.get("expected_repeats", 0) or 0),
                    }
                )
    ready = total_targets > 0 and not incomplete_targets
    return {
        "status": "complete" if ready else "incomplete",
        "ready_for_final_report": ready,
        "total_scale_targets": total_targets,
        "complete_scale_targets": total_targets - len(incomplete_targets),
        "incomplete_scale_targets": incomplete_targets,
    }


def build_summary_payload(
    cfg: RuntimeConfig,
    original_summary: dict[str, Any],
    warnings: list[str],
) -> dict[str, Any]:
    results = []
    for condition in selected_conditions(cfg, original_summary):
        result = build_condition_result(cfg, condition)
        if result is not None:
            results.append(result)
    sensitivity_analysis = build_sensitivity_analysis(cfg, results)
    reliability_analysis = build_reliability_analysis(results, ordered_trigger_labels(results), cfg.repeat)
    completion = build_report_completion(results)
    return {
        "batch_name": cfg.name,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "aggregation_method_version": AGGREGATION_METHOD_VERSION,
        "evaluation_protocol": {
            "expected_repeats": cfg.repeat,
            "scales": list(SCALES),
            "scale_order": list(SCALES),
            "strict_complete_k": True,
            "primary_score": "mean_of_complete_reviewed_scale_totals",
            "within_repeat_review": "direct_unambiguous_answer_score_overrides_llm_item_score",
            "repeat_unit": "same agent and frozen snapshot; Monte Carlo measurement repeat, not independent patient",
        },
        "source_original_summary": str(cfg.original_summary),
        "generation_controller_manifests": {
            str(condition.get("condition_name", "") or ""): {
                **manifest_identity(manifest),
                "condition_key": manifest.get("condition_key", ""),
                "manifest_path": str(
                    generation_manifest_output_path(
                        cfg,
                        str(condition.get("condition_name", "") or ""),
                    )
                ),
            }
            for condition in selected_conditions(cfg, original_summary)
            for manifest in [generation_manifest_for_condition(cfg, condition)]
            if manifest is not None
        },
        "archive_results_root": str(cfg.archive_results_root),
        "repeat": cfg.repeat,
        "labels": cfg.labels,
        "summary_only": cfg.report_only,
        "completion": completion,
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
        "sensitivity_analysis": sensitivity_analysis,
        "reliability_analysis": reliability_analysis,
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
                    "repeat_total": repeat_total,
                    "repeat_mean_total": scale_payload.get("mean"),
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
    lines.append("| 条件 | 评估点 | 量表 | 原总分 | 固定K均值 | Δ | 原程度 | 众数程度 | 样本SD | 缺失重复 |")
    lines.append("|------|--------|------|--------|-----------|---|--------|----------|--------|----------|")
    for row in rows:
        missing = ",".join(f"r{int(item):02d}" for item in row.get("missing_repeats", []) if item) or "—"
        original_severity = row.get("original_severity", "—")
        repeat_severity = row.get("repeat_severity", "—")
        if row.get("severity_changed"):
            repeat_severity = f"**{repeat_severity}**"
        lines.append(
            "| {condition} | {label} | {scale} | {orig} | {mean} | {delta} | {orig_sev} | {repeat_sev} | {std} | {missing} |".format(
                condition=row.get("condition_name", "—"),
                label=row.get("trigger_label", "—"),
                scale=row.get("scale", "—"),
                orig=format_number(row.get("original_total")),
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
        lines.append("| 评估点 | 量表 | repeat | LLM自报总分 | LLM条目和 | 复核总分 | 覆盖条目数 | 程度 | scored_file |")
        lines.append("|--------|------|--------|-------------|-----------|----------|------------|------|-------------|")
        for evaluation in result.get("evaluations", []):
            label = evaluation.get("trigger_label", "—")
            for scale_name, scale_payload in (evaluation.get("scales", {}) or {}).items():
                for row in scale_payload.get("repeat_runs", []) or []:
                    scored_file = row.get("scored_file", "")
                    if scored_file:
                        scored_file = to_display_path(Path(scored_file))
                    lines.append(
                        "| {label} | {scale} | r{repeat:02d} | {reported} | {llm_sum} | {reviewed} | {overrides} | {severity} | `{file}` |".format(
                            label=label,
                            scale=scale_name,
                            repeat=int(row.get("repeat", 0) or 0),
                            reported=format_number(row.get("reported_total_score")),
                            llm_sum=format_number(row.get("llm_item_sum_total")),
                            reviewed=format_number(row.get("reviewed_total_score")),
                            overrides=int(row.get("answer_override_count", 0) or 0),
                            severity=row.get("severity", "—"),
                            file=scored_file or "—",
                        )
                    )
        lines.append("")
    return lines


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


def format_interval(interval: Any) -> str:
    if not isinstance(interval, dict):
        return "—"
    lower = interval.get("lower")
    upper = interval.get("upper")
    if lower is None or upper is None:
        return "—"
    return "[{}, {}]".format(format_number(lower, 2), format_number(upper, 2))


def format_category_distribution(scale_payload: dict[str, Any]) -> str:
    counts = scale_payload.get("category_counts", {})
    proportions = scale_payload.get("category_proportions", {})
    if not isinstance(counts, dict) or not counts:
        return "—"
    return "; ".join(
        "{}: {}/{} ({:.1%})".format(
            category,
            count,
            scale_payload.get("expected_repeats", 0),
            float(proportions.get(category, 0.0) or 0.0),
        )
        for category, count in counts.items()
    )


def render_fixed_repeat_statistics_markdown(results: list[dict[str, Any]]) -> list[str]:
    lines = ["## 固定完整重复统计\n"]
    lines.append("| 条件 | 评估点 | 量表 | 状态 | 有效/K | 均值 | 样本SD | MAD | 95% CI | 中位数 | IQR | 截尾均值 | 众数程度 | 置信度 | 不确定性 | 翻转率 |")
    lines.append("|------|--------|------|------|--------|------|--------|-----|--------|--------|-----|----------|----------|--------|----------|--------|")
    for result in results:
        for evaluation in result.get("evaluations", []) or []:
            for scale_name, scale_payload in (evaluation.get("scales", {}) or {}).items():
                lines.append(
                    "| {condition} | {label} | {scale} | {status} | {valid}/{expected} | {mean} | {sd} | {mad} | {ci} | {median} | {iqr} | {trimmed} | {severity} | {confidence} | {uncertainty} | {flip} |".format(
                        condition=result.get("condition_name", "—"),
                        label=evaluation.get("trigger_label", "—"),
                        scale=scale_name,
                        status=scale_payload.get("aggregation_status", "—"),
                        valid=scale_payload.get("valid_repeats", 0),
                        expected=scale_payload.get("expected_repeats", 0),
                        mean=format_number(scale_payload.get("mean"), 2),
                        sd=format_number(scale_payload.get("sample_sd"), 2),
                        mad=format_number(scale_payload.get("mean_absolute_deviation"), 2),
                        ci=format_interval(scale_payload.get("ci95")),
                        median=format_number(scale_payload.get("median"), 2),
                        iqr=format_number(scale_payload.get("iqr"), 2),
                        trimmed=format_number(scale_payload.get("trimmed_mean_drop_one_each"), 2),
                        severity=scale_payload.get("severity", "—"),
                        confidence=format_number(scale_payload.get("modal_confidence"), 3),
                        uncertainty=format_number(scale_payload.get("category_uncertainty"), 3),
                        flip=format_number(scale_payload.get("category_pairwise_flip_rate"), 3),
                    )
                )
    lines.append("")
    lines.append("### 严重程度类别分布\n")
    lines.append("| 条件 | 评估点 | 量表 | 类别分布 | 是否翻转 |")
    lines.append("|------|--------|------|----------|----------|")
    for result in results:
        for evaluation in result.get("evaluations", []) or []:
            for scale_name, scale_payload in (evaluation.get("scales", {}) or {}).items():
                lines.append(
                    "| {} | {} | {} | {} | {} |".format(
                        result.get("condition_name", "—"),
                        evaluation.get("trigger_label", "—"),
                        scale_name,
                        format_category_distribution(scale_payload),
                        "是" if scale_payload.get("any_category_flip") else "否",
                    )
                )
    lines.append("")
    return lines


def render_sensitivity_markdown(payload: dict[str, Any]) -> list[str]:
    lines = ["## 敏感性分析\n"]
    method_labels = {
        "median_total": "总分中位数",
        "trimmed_mean": "去除一个最高/最低总分后的均值",
        "r01": "首轮 r01",
    }
    for method_name, label in method_labels.items():
        method = payload.get(method_name, {}) if isinstance(payload, dict) else {}
        lines.append(f"### {label}\n")
        lines.append("| 条件 | 评估点 | 量表 | 敏感性总分 | 相对首次Δ |")
        lines.append("|------|--------|------|------------|-----------|")
        for result in method.get("conditions", []) or []:
            for evaluation in result.get("evaluations", []) or []:
                for scale_name, scale_payload in (evaluation.get("scales", {}) or {}).items():
                    lines.append(
                        "| {} | {} | {} | {} | {} |".format(
                            result.get("condition_name", "—"),
                            evaluation.get("trigger_label", "—"),
                            scale_name,
                            format_number(scale_payload.get("total_score"), 2),
                            format_delta(scale_payload.get("delta_from_baseline")),
                        )
                    )
        lines.append("")
        lines.extend(
            render_final_delta_markdown(
                f"{label}：按 Variant 的最终变化",
                method.get("variant_final_delta", {}),
                VARIANTS,
            )
        )
        lines.extend(
            render_final_delta_markdown(
                f"{label}：按 Group 的最终变化",
                method.get("group_final_delta", {}),
                method.get("group_values", GROUPS),
            )
        )
    return lines


def render_reliability_markdown(payload: dict[str, Any]) -> list[str]:
    lines = ["## 批次重复测量可靠性\n"]
    lines.append("ICC 按量表和评估点分别计算；condition/persona 是测量目标，repeat 是重复测量。\n")
    lines.append("| 量表 | 评估点 | 目标数 | K | ICC(1,1) | ICC(1,K) | 不可计算原因 | 排除目标 |")
    lines.append("|------|--------|--------|---|----------|----------|--------------|----------|")
    by_scale = payload.get("by_scale", {}) if isinstance(payload, dict) else {}
    for scale_name in SCALES:
        for label, item in (by_scale.get(scale_name, {}) or {}).items():
            lines.append(
                "| {} | {} | {} | {} | {} | {} | {} | {} |".format(
                    scale_name,
                    label,
                    item.get("target_count", 0),
                    item.get("repeat_count", 0),
                    format_number(item.get("icc_1_1"), 3),
                    format_number(item.get("icc_1_k"), 3),
                    item.get("reason", "") or "—",
                    ", ".join(item.get("excluded_targets", []) or []) or "—",
                )
            )
    lines.append("")
    return lines


def render_snapshot_integrity_markdown(results: list[dict[str, Any]]) -> list[str]:
    integrity_rows = [
        (result, evaluation)
        for result in results
        for evaluation in result.get("evaluations", []) or []
        if evaluation.get("historical_snapshot_integrity")
    ]
    if not integrity_rows:
        return []
    lines = [
        "## 历史快照完整性\n",
        "| 条件 | 评估点 | 完整性 | Bundle 已校验 | 使用最终 storage 兼容 |",
        "|------|--------|--------|----------------|-------------------------|",
    ]
    for result, evaluation in integrity_rows:
        lines.append(
            "| {} | {} | {} | {} | {} |".format(
                result.get("condition_name", "—"),
                evaluation.get("trigger_label", "—"),
                evaluation.get("historical_snapshot_integrity", "—"),
                "是" if evaluation.get("snapshot_bundle_verified") is True else "否",
                "是" if evaluation.get("legacy_final_storage_used") else "否",
            )
        )
    lines.append("")
    return lines


def render_markdown_report(payload: dict[str, Any]) -> str:
    results = payload.get("conditions", [])
    warnings = payload.get("warnings", [])
    completion = payload.get("completion", {}) if isinstance(payload.get("completion"), dict) else {}
    ready_for_final = completion.get("ready_for_final_report", True)
    lines = [
        f"# 存档重复量表评估汇总：{payload['batch_name']}\n",
        f"生成时间：{payload['generated_at']}\n",
        (
            "> 状态：完整。所有 condition × 评估点 × 量表均已取得固定 K 次有效复核总分。\n"
            if ready_for_final
            else "> **状态：未完成。此文件仅用于续跑诊断，不是最终报告；请使用相同 `--name` 重新运行。**\n"
        ),
        "> 本汇总只统计 PHQ-9 / BDI-II。每次以 LLM 条目分为底稿，用患者明确且无歧义的 0–3 分回答覆盖冲突条目；主分数是固定 K 次复核后完整量表总分的均值。重复测量不是独立患者样本。\n",
        "## 汇总范围\n",
        f"- 完成状态：`{completion.get('status', 'unknown')}`",
        f"- 完整量表目标：{completion.get('complete_scale_targets', '—')} / {completion.get('total_scale_targets', '—')}",
        f"- 方法版本：`{payload.get('aggregation_method_version', '—')}`",
        f"- 条件数：{len(results)}",
        f"- 固定完整重复次数 K：{payload.get('repeat', '—')}",
        f"- 原始报告：`{payload.get('source_original_summary', '')}`",
        f"- 评估点：{', '.join(payload['trigger_labels']) if payload.get('trigger_labels') else '—'}",
    ]
    if warnings:
        lines.append("- 警告：")
        lines.extend(f"  - {warning}" for warning in warnings)
    lines.append("")
    lines.extend(render_snapshot_integrity_markdown(results))
    lines.extend(render_fixed_repeat_statistics_markdown(results))
    lines.append("## 主结果：各条件量表变化过程\n")
    for result in results:
        lines.extend(render_condition_trajectory_markdown(result))
    lines.extend(render_dimension_trajectory_markdown("按 Variant 的主结果轨迹", payload["variant_trajectory"], VARIANTS, payload["trigger_labels"]))
    lines.extend(render_dimension_trajectory_markdown("按 Group 的主结果轨迹", payload["group_trajectory"], payload.get("group_values", GROUPS), payload["trigger_labels"]))
    lines.extend(render_dimension_trajectory_markdown("按 Severity 的主结果轨迹", payload["severity_trajectory"], SEVERITIES, payload["trigger_labels"]))
    lines.extend(render_final_delta_markdown("按 Variant 的主结果最终变化", payload["variant_final_delta"], VARIANTS))
    lines.extend(render_final_delta_markdown("按 Group 的主结果最终变化", payload["group_final_delta"], payload.get("group_values", GROUPS)))
    lines.extend(render_final_delta_markdown("按 Severity 的主结果最终变化", payload["severity_final_delta"], SEVERITIES))
    lines.extend(render_repeat_group_severity_matrix_markdown(payload["group_severity_final_delta_matrix"], payload.get("group_values", GROUPS)))
    lines.extend(render_sensitivity_markdown(payload.get("sensitivity_analysis", {})))
    lines.extend(render_reliability_markdown(payload.get("reliability_analysis", {})))
    lines.extend(render_scale_score_validation_markdown(payload["scale_score_validation"]))
    lines.extend(render_repeat_detail_markdown(results))
    lines.extend(render_original_diff_markdown(payload["original_diff"]))
    return "\n".join(lines)


def write_report_outputs(cfg: RuntimeConfig, payload: dict[str, Any]) -> tuple[Path, Path]:
    completion = payload.get("completion", {}) if isinstance(payload.get("completion"), dict) else {}
    ready_for_final = completion.get("ready_for_final_report", True)
    report_suffix = "summary" if ready_for_final else "incomplete"
    json_path = reports_dir(cfg) / f"{cfg.name}_{report_suffix}.json"
    md_path = reports_dir(cfg) / f"{cfg.name}_{report_suffix}.md"
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


def require_complete_report(cfg: RuntimeConfig, payload: dict[str, Any], md_path: Path) -> None:
    completion = payload.get("completion", {}) or {}
    if cfg.dry_run or completion.get("ready_for_final_report", False):
        return
    incomplete_count = len(completion.get("incomplete_scale_targets", []) or [])
    raise RuntimeError(
        f"重复评估尚未完整（{incomplete_count} 个量表目标缺少有效 K 次结果）。\n"
        f"已写入续跑诊断报告: {md_path}\n"
        "请使用相同 --name 重新运行；默认会复用完整 answered/scored 并只补缺失阶段。"
    )


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return path != root


def _validate_cleanup_scope(path: Path, allowed_root: Path) -> None:
    allowed_resolved = allowed_root.resolve()
    target_resolved = path.resolve(strict=False)
    if not _is_within(target_resolved, allowed_resolved):
        raise ValueError(f"cleanup target escapes allowed root: {path}")


def _validate_cleanup_target(path: Path, allowed_root: Path) -> None:
    """Reject path escapes, including symlinks that resolve outside the scope."""

    _validate_cleanup_scope(path, allowed_root)
    allowed_resolved = allowed_root.resolve()
    if not path.is_dir() or path.is_symlink():
        return
    for current_root, dirnames, filenames in os.walk(path, followlinks=False):
        current = Path(current_root)
        for name in [*dirnames, *filenames]:
            nested = current / name
            if not nested.is_symlink():
                continue
            nested_resolved = nested.resolve(strict=False)
            if not _is_within(nested_resolved, allowed_resolved):
                raise ValueError(f"cleanup tree contains escaping symlink: {nested}")


def _cleanup_target_stats(path: Path) -> tuple[int, int]:
    if path.is_symlink() or path.is_file():
        try:
            return 1, int(path.lstat().st_size)
        except FileNotFoundError:
            return 0, 0
    if not path.is_dir():
        return 0, 0
    file_count = 0
    total_bytes = 0
    for current_root, dirnames, filenames in os.walk(path, followlinks=False):
        current = Path(current_root)
        for name in [*dirnames, *filenames]:
            child = current / name
            if child.is_dir() and not child.is_symlink():
                continue
            try:
                total_bytes += int(child.lstat().st_size)
                file_count += 1
            except FileNotFoundError:
                continue
    return file_count, total_bytes


def _remove_cleanup_target(path: Path) -> None:
    try:
        if path.is_symlink() or path.is_file():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)
    except FileNotFoundError:
        # A concurrent cleanup for the same source run may have removed it.
        return


def _add_cleanup_candidate(
    candidates: dict[str, tuple[Path, Path, str]],
    path: Path,
    allowed_root: Path,
    category: str,
) -> None:
    if not path.exists() and not path.is_symlink():
        return
    candidates.setdefault(str(path.absolute()), (path, allowed_root, category))


def _build_cleanup_candidates(
    cfg: RuntimeConfig,
    original_summary: dict[str, Any],
) -> tuple[list[tuple[Path, Path, str]], list[str]]:
    candidates: dict[str, tuple[Path, Path, str]] = {}
    warnings: list[str] = []
    output_root = repeat_root(cfg)

    for condition in selected_conditions(cfg, original_summary):
        condition_name = str(condition.get("condition_name", "") or "")
        for repeat_idx in range(1, cfg.repeat + 1):
            for label in cfg.labels:
                label_dir = repeat_label_dir(
                    cfg,
                    condition_name,
                    repeat_idx,
                    label,
                )
                for name in ("job.json", "resume_job.json"):
                    _add_cleanup_candidate(
                        candidates,
                        label_dir / name,
                        output_root,
                        "repeat_job",
                    )
                for pattern in ("*_trace.json", "*_trace.jsonl"):
                    for trace_path in label_dir.glob(pattern):
                        _add_cleanup_candidate(
                            candidates,
                            trace_path,
                            output_root,
                            "repeat_scale_trace",
                        )
                _add_cleanup_candidate(
                    candidates,
                    label_dir / "_tmp",
                    output_root,
                    "repeat_tmp",
                )

        run_name = str(condition.get("run_name", "") or "").strip()
        if not run_name:
            continue
        staged_root = experiment_data_root(cfg) / run_name / "scales" / "staged"
        if not staged_root.exists() and not staged_root.is_symlink():
            continue
        try:
            _validate_cleanup_target(staged_root, experiment_data_root(cfg))
            for name in ("job.json", "resume_job.json"):
                for job_path in staged_root.rglob(name):
                    _add_cleanup_candidate(
                        candidates,
                        job_path,
                        experiment_data_root(cfg),
                        "experiment_staged_job",
                    )
            for directory_name in ("snapshot_storage", "_tmp"):
                for directory in staged_root.rglob(directory_name):
                    _add_cleanup_candidate(
                        candidates,
                        directory,
                        experiment_data_root(cfg),
                        "experiment_staged_snapshot"
                        if directory_name == "snapshot_storage"
                        else "experiment_staged_tmp",
                    )
        except (OSError, ValueError) as exc:
            warnings.append(f"{staged_root}: {exc}")

    for suffix in ("json", "md"):
        _add_cleanup_candidate(
            candidates,
            reports_dir(cfg) / f"{cfg.name}_incomplete.{suffix}",
            reports_dir(cfg),
            "superseded_incomplete_report",
        )

    return (
        sorted(candidates.values(), key=lambda item: len(item[0].parts)),
        warnings,
    )


def cleanup_completed_artifacts(
    cfg: RuntimeConfig,
    original_summary: dict[str, Any],
    payload: dict[str, Any],
    json_path: Path,
    md_path: Path,
) -> dict[str, Any] | None:
    if not cfg.cleanup_completed_artifacts or cfg.dry_run:
        return None
    completion = payload.get("completion", {})
    persisted_report = load_optional_json(json_path)
    persisted_completion = (
        persisted_report.get("completion", {})
        if isinstance(persisted_report, dict)
        else {}
    )
    if (
        not isinstance(completion, dict)
        or completion.get("ready_for_final_report") is not True
        or not isinstance(persisted_report, dict)
        or not isinstance(persisted_completion, dict)
        or persisted_completion.get("ready_for_final_report") is not True
        or not md_path.is_file()
    ):
        print("[WARN] 正式完整报告未通过落盘校验，跳过复评底稿清理。")
        return None

    repeat_base = experiment_data_root(cfg) / REPEAT_OUTPUT_SUBDIR
    try:
        _validate_cleanup_scope(repeat_root(cfg), repeat_base)
    except (OSError, ValueError) as exc:
        print(f"[WARN] 复评输出目录未通过边界校验，跳过清理：{exc}")
        return None

    candidates, warnings = _build_cleanup_candidates(cfg, original_summary)
    removed_target_count = 0
    removed_file_count = 0
    removed_bytes = 0
    by_category: dict[str, dict[str, int]] = {}
    for path, allowed_root, category in candidates:
        if not path.exists() and not path.is_symlink():
            continue
        try:
            _validate_cleanup_target(path, allowed_root)
            file_count, byte_count = _cleanup_target_stats(path)
            _remove_cleanup_target(path)
        except (OSError, ValueError) as exc:
            warnings.append(f"{path}: {exc}")
            continue
        if file_count == 0 and byte_count == 0:
            continue
        removed_target_count += 1
        removed_file_count += file_count
        removed_bytes += byte_count
        category_stats = by_category.setdefault(
            category,
            {"removed_targets": 0, "removed_files": 0, "removed_bytes": 0},
        )
        category_stats["removed_targets"] += 1
        category_stats["removed_files"] += file_count
        category_stats["removed_bytes"] += byte_count

    manifest_path = repeat_root(cfg) / "artifact_cleanup_manifest.json"
    previous = load_optional_json(manifest_path) or {}
    manifest = {
        "schema_version": 1,
        "artifact_kind": "completed_repeat_eval_cleanup",
        "policy": "repeat_workfiles_only",
        "status": "partial" if warnings else "complete",
        "completed_at": datetime.now().isoformat(timespec="seconds"),
        "summary_path": str(json_path),
        "summary_sha256": file_sha256(json_path),
        "checkpoint_cleanup": False,
        "removed_target_count": int(previous.get("removed_target_count", 0) or 0)
        + removed_target_count,
        "removed_file_count": int(previous.get("removed_file_count", 0) or 0)
        + removed_file_count,
        "removed_bytes": int(previous.get("removed_bytes", 0) or 0) + removed_bytes,
        "last_run": {
            "removed_target_count": removed_target_count,
            "removed_file_count": removed_file_count,
            "removed_bytes": removed_bytes,
            "by_category": by_category,
        },
        "retained_classes": [
            "answered",
            "scored",
            "metadata",
            "worker_result",
            "controller_manifest",
            "formal_reports",
            "all_checkpoints",
        ],
        "warnings": warnings,
    }
    write_json_file(manifest_path, manifest)
    if warnings:
        print(
            f"[WARN] 复评底稿清理部分完成：释放约 {removed_bytes / (1024 ** 2):.2f} MiB，"
            f"{len(warnings)} 项未清理；详见 {manifest_path}"
        )
    else:
        print(
            f"[CLEANUP] 复评底稿清理完成：删除 {removed_file_count} 个文件，"
            f"释放约 {removed_bytes / (1024 ** 2):.2f} MiB；manifest={manifest_path}"
        )
    return manifest


def main() -> None:
    args = parse_args()
    cfg = resolve_runtime_config(args)
    original_summary = load_json_file(cfg.original_summary)
    if not isinstance(original_summary, dict):
        raise ValueError(f"original summary must be JSON object: {cfg.original_summary}")

    print_effective_config(cfg, original_summary)
    prepare_generation_manifests(cfg, original_summary)
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
    else:
        print("[INFO] report-only: 跳过 worker 调用，只汇总已有重复结果")

    payload = build_summary_payload(cfg, original_summary, warnings)
    json_path, md_path = write_report_outputs(cfg, payload)
    require_complete_report(cfg, payload, md_path)
    try:
        cleanup_completed_artifacts(
            cfg,
            original_summary,
            payload,
            json_path,
            md_path,
        )
    except Exception as exc:
        # Cleanup is operational housekeeping and must not invalidate a
        # scientifically complete report.
        print(f"[WARN] 正式报告已完成，但复评底稿清理失败：{exc}")
    print(f"\n[Done] 存档重复评估报告: {md_path}")


if __name__ == "__main__":
    main()
