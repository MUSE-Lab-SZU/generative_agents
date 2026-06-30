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
    VARIANTS,
    aggregate_dimension_trajectory,
    aggregate_final_deltas,
    apply_trajectory_deltas,
    build_group_severity_matrix,
    build_scale_score_validation,
    build_scale_snapshot,
    extract_scale_severity,
    extract_scale_total,
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
    trigger_sort_key,
    write_json_file,
)


DEFAULT_LABELS = ["T0", "session_4", "session_8", "session_12", "T4", "POST"]
REPEAT_OUTPUT_SUBDIR = "repeat_scale_eval"
STAGED_WORKER_SCRIPT = BASE_DIR / "runshells" / "run_staged_eval_worker.py"
CHECKPOINT_RESULTS_PREFIX = "/workspace/project/results"


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


@dataclass(frozen=True)
class RepeatTask:
    condition: dict[str, Any]
    label: str
    repeat_idx: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Repeat archived T0/session/T4/POST scale evaluations")
    parser.add_argument("--archive-results-root", required=True, help="存档中的 results 目录")
    parser.add_argument("--original-summary", default=None, help="原始 *_summary.json；默认自动从 reports 目录选择")
    parser.add_argument("--name", default=None, help="输出批次名，默认 repeat-scale-<MMdd-HHmm>")
    parser.add_argument("--repeat", type=int, default=3, help="每个评估点重复次数")
    parser.add_argument("--labels", default=",".join(DEFAULT_LABELS), help="逗号分隔评估点")
    parser.add_argument("--condition", action="append", default=None, help="只处理指定 condition，可重复传入")
    parser.add_argument("--max-parallel", type=int, default=1, help="并行任务数")
    parser.add_argument("--force", action="store_true", help="覆盖已有重复输出")
    parser.add_argument("--dry-run", action="store_true", help="只打印计划，不实际执行")
    parser.add_argument("--report-only", action="store_true", help="只基于已有重复结果重新生成报告")
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
        raise FileNotFoundError(f"未找到原始 summary JSON: {reports_dir}/*_summary.json")
    if len(candidates) > 1:
        candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
        print(f"[WARN] 检测到多个原始 summary，默认使用最新: {candidates[0]}")
    return candidates[0].resolve()


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
        conditions=list(args.condition or []),
        max_parallel=max_parallel,
        force=bool(args.force),
        dry_run=bool(args.dry_run),
        report_only=bool(args.report_only),
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
    print("==========================================")


def selected_conditions(cfg: RuntimeConfig, original_summary: dict[str, Any]) -> list[dict[str, Any]]:
    conditions = original_summary.get("conditions", [])
    if not isinstance(conditions, list):
        return []
    wanted = set(cfg.conditions)
    selected = [
        item
        for item in conditions
        if isinstance(item, dict) and (not wanted or str(item.get("condition_name", "")) in wanted)
    ]
    found = {str(item.get("condition_name", "")) for item in selected}
    missing = sorted(wanted - found)
    if missing:
        raise ValueError(f"原始 summary 中找不到 condition: {', '.join(missing)}")
    return selected


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
                "total_score": extract_scale_total(scored),
                "severity": extract_scale_severity(scored),
            }
        )
    return rows


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
    results = []
    for condition in selected_conditions(cfg, original_summary):
        result = build_condition_result(cfg, condition)
        if result is not None:
            results.append(result)
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
        "group_trajectory": aggregate_dimension_trajectory(results, dimension_key="group", dimension_values=GROUPS),
        "severity_trajectory": aggregate_dimension_trajectory(results, dimension_key="severity", dimension_values=SEVERITIES),
        "variant_final_delta": aggregate_final_deltas(results, dimension_key="variant", dimension_values=VARIANTS),
        "group_final_delta": aggregate_final_deltas(results, dimension_key="group", dimension_values=GROUPS),
        "severity_final_delta": aggregate_final_deltas(results, dimension_key="severity", dimension_values=SEVERITIES),
        "group_severity_final_delta_matrix": build_group_severity_matrix(results),
        "scale_score_validation": build_repeat_validation(results),
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
        for evaluation in result.get("evaluations", []) or []:
            label = str(evaluation.get("trigger_label", "") or "")
            for scale_name, scale_payload in (evaluation.get("scales", {}) or {}).items():
                original = originals.get((condition_name, label, scale_name), {})
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
                    "repeat_mean": repeat_total,
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
    lines.append("| 条件 | 评估点 | 量表 | 原总分 | 重复均值 | Δ | 原程度 | 重复程度 | 重复标准差 | 缺失重复 |")
    lines.append("|------|--------|------|--------|----------|---|--------|----------|------------|----------|")
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
                mean=format_number(row.get("repeat_mean")),
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


def render_markdown_report(payload: dict[str, Any]) -> str:
    results = payload.get("conditions", [])
    warnings = payload.get("warnings", [])
    lines = []
    lines.append(f"# 存档重复量表评估汇总：{payload['batch_name']}\n")
    lines.append(f"生成时间：{payload['generated_at']}\n")
    lines.append("> 本汇总只统计 PHQ-9 / BDI-II；主分数为重新作答并重新评分后的重复均值。\n")
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
            GROUPS,
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
    lines.extend(render_final_delta_markdown("按 Group 的最终变化对比", payload["group_final_delta"], GROUPS))
    lines.extend(render_final_delta_markdown("按 Severity 的最终变化对比", payload["severity_final_delta"], SEVERITIES))
    lines.extend(render_group_severity_matrix_markdown(payload["group_severity_final_delta_matrix"]))
    lines.extend(render_scale_score_validation_markdown(payload["scale_score_validation"]))
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
    else:
        print("[INFO] report-only: 跳过 worker 调用，只汇总已有重复结果")

    payload = build_summary_payload(cfg, original_summary, warnings)
    _json_path, md_path = write_report_outputs(cfg, payload)
    print(f"\n[Done] 存档重复评估报告: {md_path}")


if __name__ == "__main__":
    main()
