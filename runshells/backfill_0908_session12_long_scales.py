#!/usr/bin/env python3
"""补跑 0908 批次 session_12 的 PHQ-9 / BDI-II，并重建复评报告。

脚本只补缺失的长量表文件；已有 T0 和中间短量表会由复评脚本的断点续跑
逻辑直接复用。G2 的旧配置在 72 step 只生成到 session_6，本脚本会先把其
step=72 的最终冻结快照无损别名为 session_12，再执行同一套完整长量表复评。
"""

from __future__ import annotations

import argparse
import concurrent.futures
import copy
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parents[1]
RUNSHELLS_DIR = BASE_DIR / "runshells"
if str(RUNSHELLS_DIR) not in sys.path:
    sys.path.insert(0, str(RUNSHELLS_DIR))

from scale_protocol import LONG_SCALE_NAMES, SCALE_SPECS  # noqa: E402


TARGET_LABEL = "session_12"
TARGET_SESSION_COUNT = 12
EXPECTED_FINAL_STEP = 72


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON 顶层必须是对象: {path}")
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temp_path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    os.replace(temp_path, path)


def trigger_sort_key(evaluation: dict[str, Any]) -> tuple[int, int, str]:
    label = str(evaluation.get("trigger_label", "") or "")
    if label == "T0":
        return (0, 0, label)
    if label.startswith("session_"):
        try:
            return (1, int(label.split("_", 1)[1]), label)
        except ValueError:
            pass
    return (2, int(evaluation.get("completed_session_count", 0) or 0), label)


def latest_snapshot(checkpoint_dir: Path) -> tuple[Path, dict[str, Any]]:
    snapshots = sorted(checkpoint_dir.glob("simulate-*.json"))
    if not snapshots:
        raise FileNotFoundError(f"没有 simulate-*.json: {checkpoint_dir}")
    path = snapshots[-1]
    return path, load_json(path)


def final_staged_source(checkpoint_dir: Path) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    candidates: list[tuple[int, int, Path, dict[str, Any], dict[str, Any]]] = []
    staged_root = checkpoint_dir / "staged_eval"
    for metadata_path in staged_root.glob("*/metadata.json"):
        trigger_dir = metadata_path.parent
        job_path = trigger_dir / "job.json"
        storage_path = trigger_dir / "snapshot_storage"
        if not job_path.is_file() or not storage_path.is_dir():
            continue
        metadata = load_json(metadata_path)
        job = load_json(job_path)
        candidates.append(
            (
                int(metadata.get("step_no", 0) or 0),
                int(metadata.get("completed_session_count", 0) or 0),
                trigger_dir,
                metadata,
                job,
            )
        )
    if not candidates:
        raise FileNotFoundError(
            f"没有保留完整 snapshot_storage 的 staged 快照: {staged_root}"
        )
    _step, _count, trigger_dir, metadata, job = max(candidates)
    return trigger_dir, metadata, job


def validate_staged_source(checkpoint_dir: Path, label: str) -> Path:
    trigger_dir = checkpoint_dir / "staged_eval" / label
    required = [
        trigger_dir / "job.json",
        trigger_dir / "metadata.json",
        trigger_dir / "snapshot_manifest.json",
        trigger_dir / "snapshot_storage",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "补跑需要远程平台保留的完整 staged 冻结快照；缺少: "
            + ", ".join(missing)
        )
    return trigger_dir


def hardlink_or_copy(source: str, destination: str) -> str:
    try:
        os.link(source, destination)
        return destination
    except OSError:
        return shutil.copy2(source, destination)


def relabel_g2_final_snapshot(checkpoint_dir: Path, *, dry_run: bool) -> Path:
    target_dir = checkpoint_dir / "staged_eval" / TARGET_LABEL
    if target_dir.is_dir():
        job = load_json(target_dir / "job.json")
        if job.get("scales") != list(LONG_SCALE_NAMES):
            raise ValueError(f"已有 {target_dir}，但不是 PHQ-9/BDI-II 长量表 job")
        return target_dir

    final_snapshot_path, final_snapshot_payload = latest_snapshot(checkpoint_dir)
    final_step = int(final_snapshot_payload.get("step", 0) or 0)
    source_dir, source_metadata, source_job = final_staged_source(checkpoint_dir)
    source_step = int(source_metadata.get("step_no", 0) or 0)
    if final_step != EXPECTED_FINAL_STEP or source_step != final_step:
        raise ValueError(
            "G2 session_12 只能由 72-step 最终冻结快照严格补建；"
            f"当前 final_step={final_step}, staged_step={source_step}, source={source_dir}"
        )

    print(
        f"[BACKFILL] G2 最终快照别名: {source_dir.name} -> {TARGET_LABEL} "
        f"(step={final_step}, snapshot={final_snapshot_path.name})"
    )
    if dry_run:
        return target_dir

    shutil.copytree(source_dir, target_dir, copy_function=hardlink_or_copy)
    job = copy.deepcopy(source_job)
    job["trigger_label"] = TARGET_LABEL
    job["completed_session_count"] = TARGET_SESSION_COUNT
    job["scales"] = list(LONG_SCALE_NAMES)
    job["scale_question_files"] = {
        scale_name: str(SCALE_SPECS[scale_name]["question_file"])
        for scale_name in LONG_SCALE_NAMES
    }
    job["trigger_dir"] = str(target_dir)
    job["worker_result_path"] = str(target_dir / "worker_result.json")
    job["storage_source_root"] = str(target_dir / "snapshot_storage")
    write_json(target_dir / "job.json", job)

    metadata = copy.deepcopy(source_metadata)
    metadata["trigger_label"] = TARGET_LABEL
    metadata["completed_session_count"] = TARGET_SESSION_COUNT
    metadata["backfilled_from_trigger_label"] = source_dir.name
    metadata["backfill_reason"] = "legacy G2 step alignment used session_6 at final step 72"
    metadata["scales"] = {}
    write_json(target_dir / "metadata.json", metadata)

    manifest_path = target_dir / "snapshot_manifest.json"
    if manifest_path.is_file():
        manifest = load_json(manifest_path)
        manifest["trigger_label"] = TARGET_LABEL
        write_json(manifest_path, manifest)
    return target_dir


def empty_long_scales() -> dict[str, dict[str, Any]]:
    return {
        scale_name: {
            "total_score": None,
            "reported_total_score": None,
            "score_source": "item_sum",
            "severity": None,
            "scored_file": "",
            "delta_from_previous": None,
            "delta_from_baseline": None,
        }
        for scale_name in LONG_SCALE_NAMES
    }


def prepare_source_summary(
    summary_path: Path,
    results_root: Path,
    *,
    selected_conditions: set[str],
    dry_run: bool,
) -> tuple[Path, list[str]]:
    payload = load_json(summary_path)
    conditions = payload.get("conditions", [])
    if not isinstance(conditions, list):
        raise ValueError(f"summary.conditions 必须是列表: {summary_path}")

    handled: list[str] = []
    for condition in conditions:
        if not isinstance(condition, dict):
            continue
        condition_name = str(condition.get("condition_name", "") or "")
        if selected_conditions and condition_name not in selected_conditions:
            continue
        run_name = str(condition.get("run_name", "") or "")
        if not condition_name or not run_name:
            continue
        evaluations = condition.get("evaluations", [])
        if not isinstance(evaluations, list):
            raise ValueError(f"evaluations 必须是列表: {condition_name}")

        endpoint = next(
            (
                evaluation
                for evaluation in evaluations
                if isinstance(evaluation, dict)
                and evaluation.get("trigger_label") == TARGET_LABEL
            ),
            None,
        )
        checkpoint_dir = results_root / "checkpoints" / run_name
        if endpoint is None:
            if "-G2-" not in condition_name:
                raise ValueError(
                    f"{condition_name} 缺少 {TARGET_LABEL}，且不是可由最终 step 快照补建的 G2"
                )
            target_dir = relabel_g2_final_snapshot(checkpoint_dir, dry_run=dry_run)
            if dry_run:
                source_dir, source_metadata, _source_job = final_staged_source(checkpoint_dir)
                backfilled_from = source_dir.name
            else:
                source_metadata = load_json(target_dir / "metadata.json")
                backfilled_from = str(
                    source_metadata.get("backfilled_from_trigger_label", "") or ""
                )
            endpoint = {
                "trigger_label": TARGET_LABEL,
                "completed_session_count": TARGET_SESSION_COUNT,
                "sim_time": str(source_metadata.get("sim_time", "") or ""),
                "snapshot_name": str(source_metadata.get("snapshot_name", "") or ""),
                "source": "staged_eval_snapshot",
                "evaluation_executed": False,
                "metadata_path": str(target_dir / "metadata.json"),
                "scales": empty_long_scales(),
                "backfilled_from_trigger_label": backfilled_from,
            }
            evaluations.append(endpoint)
        else:
            validate_staged_source(checkpoint_dir, TARGET_LABEL)
            endpoint["scales"] = empty_long_scales()
        evaluations.sort(key=trigger_sort_key)
        condition["final_deltas"] = {
            scale_name: None for scale_name in LONG_SCALE_NAMES
        }
        handled.append(condition_name)

    if not handled:
        return summary_path, []
    payload["trigger_labels"] = sorted(
        {
            str(evaluation.get("trigger_label", "") or "")
            for condition in conditions
            if isinstance(condition, dict)
            for evaluation in condition.get("evaluations", []) or []
            if isinstance(evaluation, dict)
        },
        key=lambda label: trigger_sort_key({"trigger_label": label}),
    )
    payload.setdefault("warnings", []).append(
        "session_12 backfill source: boundary scale policy corrected to PHQ-9/BDI-II; "
        "legacy G2 final step snapshot is relabeled from session_6 to session_12."
    )
    output_path = (
        results_root
        / "experiment_data"
        / "reports"
        / "backfill_inputs"
        / f"{summary_path.stem}_{TARGET_LABEL}_long_source.json"
    )
    if dry_run:
        print(f"[DRY-RUN] write backfill source summary: {output_path}")
    else:
        write_json(output_path, payload)
    return output_path, handled


def existing_repeat_name(reports_dir: Path, source_summary: Path, condition: str) -> str | None:
    for report_path in sorted(reports_dir.glob("repeat-*_summary.json")):
        try:
            payload = load_json(report_path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        report_conditions = payload.get("conditions", []) or []
        if not any(
            isinstance(item, dict) and item.get("condition_name") == condition
            for item in report_conditions
        ):
            continue
        source_name = Path(str(payload.get("source_original_summary", "") or "")).name
        if source_name == source_summary.name:
            return str(payload.get("batch_name", "") or "") or None
    return None


def derived_repeat_name(batch_prefix: str, summary_path: Path) -> str:
    stem = summary_path.stem.removesuffix("_summary")
    identity = stem.removeprefix(f"{batch_prefix}-")
    batch_suffix = batch_prefix.removeprefix("batch-")
    return f"repeat-{identity}-{batch_suffix}"


def backup_existing_report(reports_dir: Path, repeat_name: str) -> None:
    for suffix in ("json", "md"):
        source = reports_dir / f"{repeat_name}_summary.{suffix}"
        backup = reports_dir / f"{repeat_name}_summary.pre_session12_long_backfill.{suffix}"
        if source.is_file() and not backup.exists():
            shutil.copy2(source, backup)
            print(f"[BACKUP] {source} -> {backup}")


def verify_report(report_path: Path, conditions: list[str], expected_repeats: int) -> None:
    payload = load_json(report_path)
    by_condition = {
        str(condition.get("condition_name", "") or ""): condition
        for condition in payload.get("conditions", []) or []
        if isinstance(condition, dict)
    }
    for condition_name in conditions:
        condition = by_condition.get(condition_name)
        if condition is None:
            raise ValueError(f"补跑报告缺少 condition: {condition_name}")
        endpoint = next(
            (
                item
                for item in condition.get("evaluations", []) or []
                if isinstance(item, dict) and item.get("trigger_label") == TARGET_LABEL
            ),
            None,
        )
        if endpoint is None:
            raise ValueError(f"补跑报告缺少 {condition_name}/{TARGET_LABEL}")
        scales = endpoint.get("scales", {}) or {}
        if list(scales) != list(LONG_SCALE_NAMES):
            raise ValueError(
                f"{condition_name}/{TARGET_LABEL} 量表错误: {list(scales)}"
            )
        for scale_name in LONG_SCALE_NAMES:
            scale = scales.get(scale_name, {}) or {}
            if (
                scale.get("aggregation_status") != "complete"
                or scale.get("valid_repeats") != expected_repeats
            ):
                raise ValueError(
                    f"{condition_name}/{TARGET_LABEL}/{scale_name} 未完成 "
                    f"{expected_repeats} 次有效复评"
                )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        default=os.getenv("BACKFILL_RESULTS_ROOT", "results"),
        help="远程平台原始 results 根目录，默认 results",
    )
    parser.add_argument("--batch-prefix", default="batch-0908-01")
    parser.add_argument("--condition", action="append", default=[])
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--intermediate-scale-repeats", type=int, default=5)
    parser.add_argument("--max-parallel", type=int, default=6)
    parser.add_argument(
        "--max-condition-parallel",
        type=int,
        default=4,
        help="并发补跑的 condition 数，默认 4；每个 condition 内仍使用 --max-parallel",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_root = Path(args.results_root).expanduser()
    if not results_root.is_absolute():
        results_root = (BASE_DIR / results_root).resolve()
    reports_dir = results_root / "experiment_data" / "reports"
    summaries = sorted(reports_dir.glob(f"{args.batch_prefix}-*_summary.json"))
    if not summaries:
        raise FileNotFoundError(
            f"没有找到 {args.batch_prefix}-*_summary.json: {reports_dir}"
        )
    if args.max_condition_parallel < 1:
        raise ValueError("--max-condition-parallel 必须至少为 1")
    selected = set(args.condition)
    jobs: list[tuple[str, list[str], list[str], Path]] = []
    for summary_path in summaries:
        source_path, conditions = prepare_source_summary(
            summary_path,
            results_root,
            selected_conditions=selected,
            dry_run=args.dry_run,
        )
        if not conditions:
            continue
        if len(conditions) != 1:
            raise ValueError(
                f"临时补跑脚本要求每个源 summary 只有一个目标 condition: {summary_path}"
            )
        condition = conditions[0]
        repeat_name = existing_repeat_name(reports_dir, summary_path, condition)
        repeat_name = repeat_name or derived_repeat_name(args.batch_prefix, summary_path)
        command = [
            sys.executable,
            str(RUNSHELLS_DIR / "run_archived_repeat_scale_eval.py"),
            "--archive-results-root",
            str(results_root),
            "--original-summary",
            str(source_path),
            "--condition",
            condition,
            "--labels",
            "auto",
            "--repeat",
            str(args.repeat),
            "--intermediate-scale-repeats",
            str(args.intermediate_scale_repeats),
            "--name",
            repeat_name,
            "--max-parallel",
            str(args.max_parallel),
            "--require-controller-manifest",
            "--no-cleanup-completed-artifacts",
        ]
        print(f"[PLAN] {condition}: " + " ".join(command))
        if args.dry_run:
            continue
        backup_existing_report(reports_dir, repeat_name)
        report_path = reports_dir / f"{repeat_name}_summary.json"
        jobs.append((condition, conditions, command, report_path))

    if args.dry_run:
        print("[DRY-RUN] 检查完成；未写文件、未调用量表模型。")
        return
    if not jobs:
        raise RuntimeError("没有 condition 被补跑")

    completed_reports: list[Path] = []
    failures: list[tuple[str, Exception]] = []

    def run_job(job: tuple[str, list[str], list[str], Path]) -> Path:
        condition, conditions, command, report_path = job
        print(f"[RUN] {condition}: " + " ".join(command))
        subprocess.run(command, cwd=BASE_DIR, check=True)
        verify_report(report_path, conditions, args.repeat)
        return report_path

    worker_count = min(args.max_condition_parallel, len(jobs))
    print(
        f"[PARALLEL] conditions={len(jobs)}, condition_workers={worker_count}, "
        f"task_workers_per_condition={args.max_parallel}"
    )
    with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
        future_to_condition = {
            executor.submit(run_job, job): job[0]
            for job in jobs
        }
        for future in concurrent.futures.as_completed(future_to_condition):
            condition = future_to_condition[future]
            try:
                report_path = future.result()
            except Exception as exc:
                failures.append((condition, exc))
                print(f"[ERROR] {condition}: {exc}")
            else:
                completed_reports.append(report_path)
                print(f"[DONE] {condition}: {report_path}")

    if failures:
        details = "; ".join(f"{condition}: {exc}" for condition, exc in failures)
        raise RuntimeError(
            "部分 condition 补跑失败；其他 condition 已继续执行。"
            f"直接使用相同命令可断点续跑。失败项: {details}"
        )

    print("[OK] session_12 PHQ-9/BDI-II 补跑并校验完成:")
    for report_path in sorted(completed_reports):
        print(f"  - {report_path}")


if __name__ == "__main__":
    main()
