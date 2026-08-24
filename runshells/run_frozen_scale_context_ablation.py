#!/usr/bin/env python3
"""Run a read-only scale-context ablation over frozen staged checkpoints.

This entry point never invokes ``start.py`` and never writes into a source
checkpoint. Each worker copies the archived final storage into its own temp
directory; checkpoint time plus the checkpoint's memory-id lists make the
existing memory loader prune future nodes before retrieval.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import re
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_ARCHIVE_ROOT = BASE_DIR / "results" / "0819-g1-2-4-5-6-9-11"
WORKER_SCRIPT = BASE_DIR / "runshells" / "run_frozen_scale_context_ablation_worker.py"
DEFAULT_GROUPS = ("G5", "G6", "G9", "G11")
DEFAULT_TIMEPOINTS = ("T0", "S4")
DEFAULT_CONDITIONS = ("full", "state_only", "state_complaint_static", "full_no_state")
SCALE_NAMES = ("PHQ-9", "BDI-II")
EXPERT_SCORE_KEYS = {"PHQ-9": "phq9_scores", "BDI-II": "bdi_ii_scores"}
TIMEPOINT_TRIGGER = {
    "T0": "T0",
    "S4": "session_4",
    "S8": "session_8",
    "S12": "session_12",
}
CONDITION_LABELS = {
    "full": "① 当前完整量表输入",
    "state_only": "② 七维持续症状状态 + 必要 persona/题目",
    "state_complaint_static": "③ 七维状态 + current complaint node（动态上下文关闭）",
    "full_no_state": "④ 完整输入但隐藏七维状态",
}
DRIVER_CONTRASTS = {
    "explicit_state_visibility": ("full", "full_no_state"),
    "complaint_node_over_state": ("state_complaint_static", "state_only"),
    "dynamic_context_bundle_over_static_complaint": ("full", "state_complaint_static"),
}
RUN_RE = re.compile(r"batch-\d{4}-(?P<outer>\d+)-Counsel-KBD2-(?P<group>G\d+)-SEV")


@dataclass(frozen=True)
class SourceSnapshot:
    run_name: str
    group: str
    outer_repeat: int
    timepoint: str
    trigger_label: str
    snapshot_path: Path
    storage_root: Path
    staged_metadata_path: Path


@dataclass(frozen=True)
class ReplayTask:
    source: SourceSnapshot
    condition: str
    repeat: int
    output_dir: Path


def parse_csv(raw: str) -> list[str]:
    return [item.strip() for item in str(raw or "").split(",") if item.strip()]


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def resolve_snapshot_from_metadata(checkpoint_dir: Path, metadata: dict[str, Any]) -> Path:
    snapshot_name = str(metadata.get("snapshot_name", "") or "").strip()
    if snapshot_name:
        path = checkpoint_dir / snapshot_name
        if path.is_file():
            return path
    sim_time = str(metadata.get("sim_time", "") or "").strip()
    if sim_time:
        candidate = checkpoint_dir / ("simulate-" + sim_time.replace(":", "") + ".json")
        if candidate.is_file():
            return candidate
    snapshots = sorted(checkpoint_dir.glob("simulate-*.json"))
    if str(metadata.get("trigger_label", "") or "") == "T0" and snapshots:
        return snapshots[0]
    raise FileNotFoundError(
        f"cannot resolve checkpoint snapshot from metadata: {checkpoint_dir} / {metadata.get('trigger_label')}"
    )


def discover_sources(
    archive_root: Path,
    groups: list[str],
    timepoints: list[str],
    outer_repeats: set[int] | None,
) -> list[SourceSnapshot]:
    checkpoint_root = archive_root / "checkpoints"
    experiment_root = archive_root / "experiment_data"
    sources: list[SourceSnapshot] = []
    for checkpoint_dir in sorted(checkpoint_root.iterdir() if checkpoint_root.is_dir() else []):
        if not checkpoint_dir.is_dir():
            continue
        match = RUN_RE.search(checkpoint_dir.name)
        if not match:
            continue
        group = match.group("group")
        outer_repeat = int(match.group("outer"))
        if group not in groups or (outer_repeats is not None and outer_repeat not in outer_repeats):
            continue
        storage_root = checkpoint_dir / "storage"
        if not storage_root.is_dir():
            raise FileNotFoundError(f"checkpoint storage missing: {storage_root}")
        for timepoint in timepoints:
            trigger_label = TIMEPOINT_TRIGGER[timepoint]
            metadata_path = experiment_root / checkpoint_dir.name / "scales" / "staged" / trigger_label / "metadata.json"
            if not metadata_path.is_file():
                raise FileNotFoundError(f"staged metadata missing: {metadata_path}")
            metadata = load_json(metadata_path)
            snapshot_path = resolve_snapshot_from_metadata(checkpoint_dir, metadata)
            sources.append(
                SourceSnapshot(
                    run_name=checkpoint_dir.name,
                    group=group,
                    outer_repeat=outer_repeat,
                    timepoint=timepoint,
                    trigger_label=trigger_label,
                    snapshot_path=snapshot_path.resolve(),
                    storage_root=storage_root.resolve(),
                    staged_metadata_path=metadata_path.resolve(),
                )
            )
    expected_cells = len(groups) * len(timepoints) * (len(outer_repeats) if outer_repeats is not None else 3)
    if not sources:
        raise FileNotFoundError(
            f"no requested group/timepoint sources found under {checkpoint_root}; "
            f"groups={','.join(groups)} timepoints={','.join(timepoints)}"
        )
    observed_groups = {source.group for source in sources}
    missing_groups = sorted(set(groups) - observed_groups)
    if missing_groups:
        raise ValueError(f"requested groups not found: {', '.join(missing_groups)}")
    if outer_repeats is not None and len(sources) != expected_cells:
        raise ValueError(f"incomplete requested source grid: expected={expected_cells} observed={len(sources)}")
    return sources


def build_tasks(
    sources: list[SourceSnapshot],
    conditions: list[str],
    repeats: int,
    output_root: Path,
) -> list[ReplayTask]:
    return [
        ReplayTask(
            source=source,
            condition=condition,
            repeat=repeat_idx,
            output_dir=(
                output_root
                / source.run_name
                / source.timepoint
                / condition
                / f"r{repeat_idx:02d}"
            ),
        )
        for source in sources
        for condition in conditions
        for repeat_idx in range(1, repeats + 1)
    ]


def task_complete(task: ReplayTask, score_mode: str) -> bool:
    metadata_path = task.output_dir / "metadata.json"
    if not metadata_path.is_file():
        return False
    try:
        metadata = load_json(metadata_path)
    except (OSError, json.JSONDecodeError):
        return False
    if (
        metadata.get("status") != "ok"
        or metadata.get("condition") != task.condition
        or metadata.get("score_mode") != score_mode
    ):
        return False
    for scale in SCALE_NAMES:
        required = (
            task.output_dir / f"{scale}_answered.jsonl",
            task.output_dir / f"{scale}_prompt_trace.jsonl",
            task.output_dir / f"{scale}_direct_scored.json",
        )
        if not all(path.is_file() for path in required):
            return False
        if score_mode in {"expert", "both"}:
            expert_required = (
                task.output_dir / f"{scale}_expert_scored.json",
                task.output_dir / f"{scale}_expert_scoring_trace.json",
            )
            if not all(path.is_file() for path in expert_required):
                return False
    return True


def task_job(task: ReplayTask, archive_root: Path, score_mode: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "run_name": task.source.run_name,
        "group": task.source.group,
        "outer_repeat": task.source.outer_repeat,
        "timepoint": task.source.timepoint,
        "source_trigger_label": task.source.trigger_label,
        "snapshot_path": str(task.source.snapshot_path),
        "staged_metadata_path": str(task.source.staged_metadata_path),
        "storage_source_root": str(task.source.storage_root),
        "archive_root": str(archive_root),
        "condition": task.condition,
        "ablation_repeat": task.repeat,
        "target_agent": "卡布达",
        "scales": ["PHQ-9", "BDI-II"],
        "score_mode": score_mode,
        "output_dir": str(task.output_dir),
        "worker_result_path": str(task.output_dir / "worker_result.json"),
    }


def run_task(
    task: ReplayTask,
    archive_root: Path,
    score_mode: str,
    force: bool,
    dry_run: bool,
) -> dict[str, Any]:
    if task_complete(task, score_mode) and not force:
        return {"status": "skipped", "task": task}
    job = task_job(task, archive_root, score_mode)
    if dry_run:
        return {"status": "planned", "task": task, "job": job}
    task.output_dir.mkdir(parents=True, exist_ok=True)
    job_path = task.output_dir / "job.json"
    write_json(job_path, job)
    command = [sys.executable, str(WORKER_SCRIPT), "--job", str(job_path)]
    completed = subprocess.run(command, cwd=BASE_DIR, check=False)
    if completed.returncode != 0:
        result_path = task.output_dir / "worker_result.json"
        detail = load_json(result_path) if result_path.is_file() else {}
        raise RuntimeError(
            f"worker failed returncode={completed.returncode} task={task.source.run_name}/{task.source.timepoint}/{task.condition}/r{task.repeat:02d}: {detail.get('error', '')}"
        )
    return {"status": "ok", "task": task}


def run_tasks(
    tasks: list[ReplayTask],
    archive_root: Path,
    score_mode: str,
    force: bool,
    dry_run: bool,
    max_parallel: int,
) -> list[dict[str, Any]]:
    if dry_run or max_parallel <= 1:
        results = []
        for task in tasks:
            result = run_task(task, archive_root, score_mode, force, dry_run)
            print(
                "[{}] {} {} {} r{:02d}".format(
                    result["status"].upper(),
                    task.source.run_name,
                    task.source.timepoint,
                    task.condition,
                    task.repeat,
                )
            )
            results.append(result)
        return results
    results: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_parallel) as executor:
        future_map = {
            executor.submit(run_task, task, archive_root, score_mode, force, False): task
            for task in tasks
        }
        for future in concurrent.futures.as_completed(future_map):
            task = future_map[future]
            try:
                result = future.result()
                print(f"[{result['status'].upper()}] {task.source.run_name} {task.source.timepoint} {task.condition} r{task.repeat:02d}")
                results.append(result)
            except Exception as exc:
                print(f"[ERROR] {task.source.run_name} {task.source.timepoint} {task.condition} r{task.repeat:02d}: {exc}")
                results.append({"status": "error", "task": task, "error": str(exc)})
    return results


def choose_total(totals: dict[str, Any], score_mode: str) -> tuple[float | None, str]:
    if score_mode in {"expert", "both"}:
        value = totals.get("expert")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value), "expert"
    value = totals.get("direct")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value), "direct"
    return None, "missing"


def extract_expert_total(scale_name: str, payload: dict[str, Any]) -> float | None:
    items = payload.get(EXPERT_SCORE_KEYS[scale_name])
    if isinstance(items, list):
        values: list[int] = []
        for item in items:
            value = item.get("score") if isinstance(item, dict) else None
            if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value <= 3:
                values = []
                break
            values.append(value)
        if values:
            return float(sum(values))
    total = payload.get("total_score")
    if isinstance(total, (int, float)) and not isinstance(total, bool):
        return float(total)
    return None


def load_task_score_totals(task: ReplayTask, scale_name: str) -> tuple[dict[str, Any], str]:
    """Recover totals even when a worker was interrupted after one scale completed."""

    metadata_path = task.output_dir / "metadata.json"
    if metadata_path.is_file():
        try:
            metadata = load_json(metadata_path)
        except (OSError, json.JSONDecodeError):
            metadata = {}
        totals = (metadata.get("score_totals", {}) or {}).get(scale_name, {}) if isinstance(metadata, dict) else {}
        if isinstance(totals, dict):
            return totals, str(metadata.get("status", "unknown") or "unknown")

    totals: dict[str, Any] = {}
    direct_path = task.output_dir / f"{scale_name}_direct_scored.json"
    if direct_path.is_file():
        try:
            direct = load_json(direct_path)
        except (OSError, json.JSONDecodeError):
            direct = {}
        if isinstance(direct, dict) and direct.get("complete") is True:
            totals["direct"] = direct.get("total_score")
    expert_path = task.output_dir / f"{scale_name}_expert_scored.json"
    if expert_path.is_file():
        try:
            expert = load_json(expert_path)
        except (OSError, json.JSONDecodeError):
            expert = {}
        if isinstance(expert, dict):
            totals["expert"] = extract_expert_total(scale_name, expert)
    return totals, "partial_scale_artifact"


def collect_outcomes(
    tasks: list[ReplayTask],
    score_mode: str,
    scale_names: tuple[str, ...] = SCALE_NAMES,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for task in tasks:
        metadata_path = task.output_dir / "metadata.json"
        try:
            metadata = load_json(metadata_path) if metadata_path.is_file() else {}
        except (OSError, json.JSONDecodeError):
            metadata = {}
        for scale_name in scale_names:
            totals, artifact_status = load_task_score_totals(task, scale_name)
            total, source = choose_total(totals, score_mode)
            rows.append(
                {
                    "run_name": task.source.run_name,
                    "group": task.source.group,
                    "outer_repeat": task.source.outer_repeat,
                    "ablation_repeat": task.repeat,
                    "timepoint": task.source.timepoint,
                    "condition": task.condition,
                    "condition_label": CONDITION_LABELS[task.condition],
                    "scale": scale_name,
                    "total_score": total,
                    "score_source": source,
                    "artifact_status": artifact_status,
                    "snapshot_name": str(metadata.get("snapshot_name", "") or task.source.snapshot_path.name),
                    "trace_file": str(task.output_dir / f"{scale_name}_prompt_trace.jsonl")
                    if (task.output_dir / f"{scale_name}_prompt_trace.jsonl").is_file()
                    else "",
                }
            )
    return rows


def build_change_rows(outcomes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Pair every available non-baseline node with T0 within an outer run."""

    cells: dict[tuple[Any, ...], dict[str, float]] = defaultdict(dict)
    for row in outcomes:
        value = row.get("total_score")
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            continue
        key = (
            row["run_name"],
            row["group"],
            row["outer_repeat"],
            row["ablation_repeat"],
            row["condition"],
            row["scale"],
            row["score_source"],
        )
        cells[key][row["timepoint"]] = float(value)
    rows = []
    for key, values in cells.items():
        if "T0" not in values:
            continue
        for timepoint in TIMEPOINT_TRIGGER:
            if timepoint == "T0" or timepoint not in values:
                continue
            run_name, group, outer_repeat, ablation_repeat, condition, scale, score_source = key
            rows.append(
                {
                    "run_name": run_name,
                    "group": group,
                    "outer_repeat": outer_repeat,
                    "ablation_repeat": ablation_repeat,
                    "timepoint": timepoint,
                    "baseline_timepoint": "T0",
                    "condition": condition,
                    "condition_label": CONDITION_LABELS[condition],
                    "scale": scale,
                    "score_source": score_source,
                    "t0_total": values["T0"],
                    "timepoint_total": values[timepoint],
                    "timepoint_minus_t0": values[timepoint] - values["T0"],
                }
            )
    return sorted(
        rows,
        key=lambda row: (
            row["group"], row["scale"], row["timepoint"], row["condition"],
            row["outer_repeat"], row["ablation_repeat"],
        ),
    )


def build_driver_rows(change_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cells: dict[tuple[Any, ...], dict[str, float]] = defaultdict(dict)
    for row in change_rows:
        key = (
            row["run_name"],
            row["group"],
            row["outer_repeat"],
            row["ablation_repeat"],
            row["timepoint"],
            row["scale"],
            row["score_source"],
        )
        cells[key][row["condition"]] = float(row["timepoint_minus_t0"])
    rows: list[dict[str, Any]] = []
    for key, conditions in cells.items():
        run_name, group, outer_repeat, ablation_repeat, timepoint, scale, score_source = key
        for contrast_name, (left, right) in DRIVER_CONTRASTS.items():
            if left not in conditions or right not in conditions:
                continue
            rows.append(
                {
                    "run_name": run_name,
                    "group": group,
                    "outer_repeat": outer_repeat,
                    "ablation_repeat": ablation_repeat,
                    "timepoint": timepoint,
                    "baseline_timepoint": "T0",
                    "scale": scale,
                    "score_source": score_source,
                    "contrast": contrast_name,
                    "left_condition": left,
                    "right_condition": right,
                    "left_change": conditions[left],
                    "right_change": conditions[right],
                    "change_difference": conditions[left] - conditions[right],
                }
            )
    return sorted(
        rows,
        key=lambda row: (
            row["group"], row["scale"], row["timepoint"], row["contrast"],
            row["outer_repeat"], row["ablation_repeat"],
        ),
    )


def aggregate_outer_runs(rows: list[dict[str, Any]], value_key: str, group_keys: tuple[str, ...]) -> list[dict[str, Any]]:
    """Average measurement repeats within each outer run before group summaries."""

    outer_cells: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    for row in rows:
        value = row.get(value_key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            outer_key = tuple(row[key] for key in group_keys) + (row["run_name"], row["outer_repeat"])
            outer_cells[outer_key].append(float(value))
    cells: dict[tuple[Any, ...], list[tuple[float, int]]] = defaultdict(list)
    for outer_key, repeat_values in outer_cells.items():
        cells[outer_key[: len(group_keys)]].append((mean(repeat_values), len(repeat_values)))
    result = []
    for key, outer_values in sorted(cells.items()):
        values = [item[0] for item in outer_values]
        repeat_counts = [item[1] for item in outer_values]
        payload = {name: value for name, value in zip(group_keys, key)}
        payload.update(
            n_outer_runs=len(values),
            mean=mean(values),
            min=min(values),
            max=max(values),
            measurement_repeats_min=min(repeat_counts),
            measurement_repeats_max=max(repeat_counts),
        )
        result.append(payload)
    return result


def rank_diagnostic_drivers(driver_summary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cells: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in driver_summary:
        cells[(str(row["group"]), str(row["scale"]), str(row["timepoint"]))].append(row)
    rankings = []
    for (group, scale, timepoint), rows in sorted(cells.items()):
        ordered = sorted(rows, key=lambda row: abs(float(row["mean"])), reverse=True)
        if not ordered:
            continue
        strongest = ordered[0]
        rankings.append(
            {
                "group": group,
                "scale": scale,
                "timepoint": timepoint,
                "strongest_diagnostic_contrast": strongest["contrast"],
                "signed_mean_difference": strongest["mean"],
                "absolute_mean_difference": abs(float(strongest["mean"])),
                "ranking": [
                    {
                        "contrast": row["contrast"],
                        "signed_mean_difference": row["mean"],
                        "absolute_mean_difference": abs(float(row["mean"])),
                    }
                    for row in ordered
                ],
                "interpretation_limit": "diagnostic sensitivity ranking; not an additive or identified causal decomposition",
            }
        )
    return rankings


def cleanup_worker_artifacts(output_root: Path) -> dict[str, Any]:
    """Remove large, regenerable worker traces while retaining answers and scores."""

    if not output_root.is_dir():
        raise FileNotFoundError(f"report output directory not found: {output_root}")
    patterns = (
        "*_prompt_trace.jsonl",
        "*_expert_scoring_trace.json",
        "job.json",
        "worker_result.json",
    )
    removed: list[dict[str, Any]] = []
    for pattern in patterns:
        for path in sorted(output_root.rglob(pattern)):
            if not path.is_file():
                continue
            size_bytes = path.stat().st_size
            path.unlink()
            removed.append({"path": str(path.relative_to(output_root)), "size_bytes": size_bytes})
    manifest = {
        "schema_version": 1,
        "artifact_kind": "frozen_scale_context_ablation_cleanup",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "policy": "remove_prompt_and_scoring_traces_plus_job_artifacts_only",
        "preserved": ["*_answered.jsonl", "*_direct_scored.json", "*_expert_scored.json", "metadata.json"],
        "removed_file_count": len(removed),
        "removed_bytes": sum(item["size_bytes"] for item in removed),
        "removed": removed,
    }
    write_json(output_root / "artifact_cleanup.json", manifest)
    return manifest


def render_summary_markdown(
    output_root: Path,
    sources: list[SourceSnapshot],
    tasks: list[ReplayTask],
    scale_names: tuple[str, ...],
    change_summary: list[dict[str, Any]],
    driver_summary: list[dict[str, Any]],
    driver_rankings: list[dict[str, Any]],
) -> str:
    lines = [
        "# Frozen checkpoint 量表上下文消融",
        "",
        "本批次只重放量表回答，不调用 `start.py`，不推进小镇，也不写回 source checkpoint。",
        "",
        "## 设计",
        "",
    ]
    for condition in DEFAULT_CONDITIONS:
        if any(task.condition == condition for task in tasks):
            lines.append(f"- {CONDITION_LABELS[condition]}")
    lines.extend(
        [
            "",
            f"- frozen source snapshots: {len(sources)}",
            f"- replay jobs: {len(tasks)}",
            f"- included scales: {','.join(scale_names)}",
            "- 统计单位：outer simulation run；ablation repeat 仅用于测量稳定性。",
            "- `timepoint − T0 < 0` 表示该节点量表分数低于基线。",
            "",
            "## 条件内变化",
            "",
            "| group | scale | timepoint | condition | outer n | mean timepoint−T0 | range |",
            "|---|---|---|---|---:|---:|---:|",
        ]
    )
    for row in change_summary:
        lines.append(
            "| {group} | {scale} | {timepoint} | {condition} | {n_outer_runs} | {mean:.3f} | [{min:.3f}, {max:.3f}] |".format(**row)
        )
    lines.extend(
        [
            "",
            "## 上下文驱动诊断 contrasts",
            "",
            "这些 contrast 是诊断性敏感度比较，不假设三类上下文效应可加。",
            "",
            "| group | scale | timepoint | contrast | outer n | mean change difference | range |",
            "|---|---|---|---|---:|---:|---:|",
        ]
    )
    for row in driver_summary:
        lines.append(
            "| {group} | {scale} | {timepoint} | {contrast} | {n_outer_runs} | {mean:.3f} | [{min:.3f}, {max:.3f}] |".format(**row)
        )
    lines.extend(
        [
            "",
            "## 最强诊断信号",
            "",
            "按 `|mean change difference|` 排序，仅用于回答优先追查哪类上下文，不解释为可加因果贡献。",
            "",
            "| group | scale | timepoint | strongest contrast | signed mean difference |",
            "|---|---|---|---|---:|",
        ]
    )
    for row in driver_rankings:
        lines.append(
            "| {group} | {scale} | {timepoint} | {strongest_diagnostic_contrast} | {signed_mean_difference:.3f} |".format(**row)
        )
    lines.extend(
        [
            "",
            "每个 `Δ` 均为当前节点量表总分减 T0 总分。contrast 定义：",
            "",
            "- `explicit_state_visibility = Δfull − Δfull_no_state`",
            "- `complaint_node_over_state = Δstate_complaint_static − Δstate_only`",
            "- `dynamic_context_bundle_over_static_complaint = Δfull − Δstate_complaint_static`",
            "  （bundle 包含 memory、relation、session scene/history、instant emotion，以及完整 runtime persona/scaffold）",
            "",
            "## Provenance / QC",
            "",
            "- 最终 storage 只作为临时重建载体；加载时由快照时钟和快照内 memory-id 列表裁剪未来节点。",
            "- 每题保存 exact final prompt、所有辅助 relation/emotion prompt 与响应、结构化上下文组件、原始回答和路由信息。",
            "- 本工作流不生成图：当前问题由成对的条件 contrast 表直接回答，避免生成重复或低汇报价值图。",
            f"- 输出根目录：`{output_root}`",
            "",
        ]
    )
    return "\n".join(lines)


def write_reports(
    output_root: Path,
    sources: list[SourceSnapshot],
    tasks: list[ReplayTask],
    score_mode: str,
    scale_names: tuple[str, ...] = SCALE_NAMES,
    cleanup_manifest: dict[str, Any] | None = None,
) -> None:
    outcomes = collect_outcomes(tasks, score_mode, scale_names)
    change_rows = build_change_rows(outcomes)
    driver_rows = build_driver_rows(change_rows)
    change_summary = aggregate_outer_runs(
        change_rows,
        "timepoint_minus_t0",
        ("group", "scale", "timepoint", "condition"),
    )
    driver_summary = aggregate_outer_runs(
        driver_rows,
        "change_difference",
        ("group", "scale", "timepoint", "contrast"),
    )
    driver_rankings = rank_diagnostic_drivers(driver_summary)
    outcome_fields = [
        "run_name", "group", "outer_repeat", "ablation_repeat", "timepoint", "condition",
        "condition_label", "scale", "total_score", "score_source", "artifact_status", "snapshot_name", "trace_file",
    ]
    change_fields = [
        "run_name", "group", "outer_repeat", "ablation_repeat", "timepoint", "baseline_timepoint",
        "condition", "condition_label", "scale", "score_source", "t0_total", "timepoint_total",
        "timepoint_minus_t0",
    ]
    driver_fields = [
        "run_name", "group", "outer_repeat", "ablation_repeat", "timepoint", "baseline_timepoint",
        "scale", "score_source",
        "contrast", "left_condition", "right_condition", "left_change", "right_change", "change_difference",
    ]
    write_csv(output_root / "ablation_outcomes_long.csv", outcomes, outcome_fields)
    write_csv(output_root / "ablation_changes.csv", change_rows, change_fields)
    write_csv(output_root / "ablation_driver_contrasts.csv", driver_rows, driver_fields)
    summary = {
        "schema_version": 1,
        "artifact_kind": "frozen_scale_context_ablation_summary",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "score_mode": score_mode,
        "included_scales": list(scale_names),
        "source_snapshot_count": len(sources),
        "job_count": len(tasks),
        "outcome_count": len(outcomes),
        "scored_outcome_count": sum(1 for row in outcomes if isinstance(row.get("total_score"), (int, float))),
        "artifact_cleanup": cleanup_manifest,
        "change_summary": change_summary,
        "driver_contrast_summary": driver_summary,
        "driver_rankings": driver_rankings,
        "driver_contrast_definitions": {
            name: {"left": left, "right": right, "estimand": "left (timepoint-T0) minus right (timepoint-T0)"}
            for name, (left, right) in DRIVER_CONTRASTS.items()
        },
    }
    write_json(output_root / "summary.json", summary)
    (output_root / "README.md").write_text(
        render_summary_markdown(output_root, sources, tasks, scale_names, change_summary, driver_summary, driver_rankings),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", default=str(DEFAULT_ARCHIVE_ROOT))
    parser.add_argument("--name", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--groups", default=",".join(DEFAULT_GROUPS))
    parser.add_argument("--timepoints", default=",".join(DEFAULT_TIMEPOINTS), help="T0,S4,S8,S12")
    parser.add_argument("--conditions", default=",".join(DEFAULT_CONDITIONS))
    parser.add_argument("--outer-repeats", default="", help="例如 1 或 1,2,3；留空处理全部")
    parser.add_argument("--ablation-repeats", type=int, default=1)
    parser.add_argument("--score-mode", choices=("direct", "expert", "both"), default="both")
    parser.add_argument("--max-parallel", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="不调用模型；仅从已有 answered/scored 文件生成汇总，支持中断后的单量表结果",
    )
    parser.add_argument(
        "--report-scales",
        default=",".join(SCALE_NAMES),
        help="report-only 纳入的量表，例如 PHQ-9；默认 PHQ-9,BDI-II",
    )
    parser.add_argument(
        "--cleanup-artifacts",
        action="store_true",
        help="仅限 report-only：删除该输出批次的 trace、job.json 与 worker_result.json，保留回答和评分",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    archive_root = Path(args.archive_root).expanduser()
    if not archive_root.is_absolute():
        archive_root = (BASE_DIR / archive_root).resolve()
    if not archive_root.is_dir():
        raise FileNotFoundError(f"archive root not found: {archive_root}")
    groups = [item.upper() for item in parse_csv(args.groups)]
    timepoints = [item.upper() for item in parse_csv(args.timepoints)]
    conditions = parse_csv(args.conditions)
    report_scales = tuple(parse_csv(args.report_scales))
    unknown_timepoints = sorted(set(timepoints) - set(TIMEPOINT_TRIGGER))
    unknown_conditions = sorted(set(conditions) - set(DEFAULT_CONDITIONS))
    unknown_report_scales = sorted(set(report_scales) - set(SCALE_NAMES))
    if unknown_timepoints:
        raise ValueError(f"unknown timepoints: {', '.join(unknown_timepoints)}")
    if unknown_conditions:
        raise ValueError(f"unknown conditions: {', '.join(unknown_conditions)}")
    if not report_scales or unknown_report_scales:
        raise ValueError(f"unknown or empty report scales: {', '.join(unknown_report_scales) or 'empty'}")
    if args.cleanup_artifacts and not args.report_only:
        raise ValueError("--cleanup-artifacts requires --report-only so active workers cannot lose their artifacts")
    outer_repeats = {int(item) for item in parse_csv(args.outer_repeats)} if args.outer_repeats else None
    sources = discover_sources(archive_root, groups, timepoints, outer_repeats)
    batch_name = str(args.name or "").strip() or f"frozen-context-ablation-{datetime.now().strftime('%m%d-%H%M%S')}"
    output_root = (
        Path(args.output_dir).expanduser()
        if args.output_dir
        else archive_root / "experiment_data" / "scale_context_ablation" / batch_name
    )
    if not output_root.is_absolute():
        output_root = (BASE_DIR / output_root).resolve()
    try:
        output_root.relative_to((archive_root / "checkpoints").resolve())
    except ValueError:
        pass
    else:
        raise ValueError(f"output directory must not be inside source checkpoints: {output_root}")
    if args.report_only and not output_root.is_dir():
        raise FileNotFoundError(f"report-only output directory not found: {output_root}")
    tasks = build_tasks(sources, conditions, max(1, int(args.ablation_repeats)), output_root)
    print("==========================================")
    print(" Frozen checkpoint scale context ablation")
    print("==========================================")
    print(f"archive:          {archive_root}")
    print(f"output:           {output_root}")
    print(f"groups:           {','.join(groups)}")
    print(f"timepoints:       {','.join(timepoints)}")
    print(f"conditions:       {','.join(conditions)}")
    print(f"source snapshots: {len(sources)}")
    print(f"jobs:             {len(tasks)}")
    print(f"score mode:       {args.score_mode}")
    print(f"report only:      {args.report_only}")
    print(f"report scales:    {','.join(report_scales)}")
    print(f"max parallel:     {max(1, int(args.max_parallel))}")
    print(f"dry run:          {args.dry_run}")
    print("simulation rerun: false")
    print("source writes:    false")
    print("==========================================")
    if args.report_only:
        cleanup_manifest = cleanup_worker_artifacts(output_root) if args.cleanup_artifacts else None
        write_reports(output_root, sources, tasks, args.score_mode, report_scales, cleanup_manifest)
        print(f"[DONE] report-only outputs written to {output_root}")
        return 0
    results = run_tasks(
        tasks,
        archive_root,
        args.score_mode,
        bool(args.force),
        bool(args.dry_run),
        max(1, int(args.max_parallel)),
    )
    errors = [result for result in results if result["status"] == "error"]
    if args.dry_run:
        print(f"[DRY-RUN] planned {len(tasks)} jobs; no files written")
        return 0
    write_reports(output_root, sources, tasks, args.score_mode)
    print(f"[DONE] reports written to {output_root}; errors={len(errors)}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
