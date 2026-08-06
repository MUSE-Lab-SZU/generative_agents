#!/usr/bin/env python3
"""Recover staged-eval scores and reports from interrupted checkpoint runs.

Typical usage:
  python runshells/recover_staged_eval_scores.py

By default this only processes checkpoint directories that:
  - have results/checkpoints/<run_name>/staged_eval/
  - do not yet have results/experiment_data/<run_name>/

The script does not start simulation, embedding, or vLLM services. It only copies
existing staged_eval answer files into experiment_data, calls run_score_worker.py
for missing *_scored.json files, and writes a recovered staged-eval summary.
"""

from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent.parent
RUNSHELLS_DIR = BASE_DIR / "runshells"
if str(RUNSHELLS_DIR) not in sys.path:
    sys.path.insert(0, str(RUNSHELLS_DIR))

from run_one_experiment import (  # noqa: E402
    CHECKPOINTS_ROOT,
    EXPERIMENT_DATA_ROOT,
    collect_core_outputs,
)
from run_batch_experiment import (  # noqa: E402
    BatchCondition,
    RuntimeConfig,
    SCALES,
    GROUPS,
    SEVERITIES,
    build_summary_payload,
    ensure_staged_eval_scores,
    load_condition_result,
    render_condition_trajectory_markdown,
    render_dimension_trajectory_markdown,
    render_final_delta_markdown,
    render_group_severity_matrix_markdown,
    write_json_file,
)


REPORTS_DIR = EXPERIMENT_DATA_ROOT / "reports"
DEFAULT_REPORT_NAME = "recovered-staged-eval"
SEVERITY_ALIASES = {
    "MILD": "mild",
    "MOD": "moderate",
    "MODERATE": "moderate",
    "SEV": "severe",
    "SEVERE": "severe",
}
RUN_CONDITION_RE = re.compile(
    r"(Counsel-G(?P<group>\d+)-(?P<severity>MILD|MOD|MODERATE|SEV|SEVERE))",
    re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score and summarize staged_eval outputs from interrupted checkpoint runs."
    )
    parser.add_argument(
        "--run-name",
        action="append",
        default=[],
        help="只处理指定 checkpoint run_name；可重复传入。默认自动找 checkpoint 有但 experiment_data 没有的目录。",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="处理所有带 staged_eval 的 checkpoint，包括已经存在 experiment_data 的目录。",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="重新复制 checkpoint/staged_eval 到 experiment_data/<run>/scales/staged。",
    )
    parser.add_argument(
        "--skip-score",
        action="store_true",
        help="跳过专家评分，只基于已有 *_scored.json 生成报告。",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只打印将要处理的目录和评分命令，不写入文件、不调用专家评分。",
    )
    parser.add_argument(
        "--report-name",
        default=DEFAULT_REPORT_NAME,
        help=f"报告文件名前缀，默认 {DEFAULT_REPORT_NAME}。",
    )
    return parser.parse_args()


def has_staged_eval(checkpoint_dir: Path) -> bool:
    staged_root = checkpoint_dir / "staged_eval"
    if not staged_root.is_dir():
        return False
    return any(staged_root.glob("*/*_answered.jsonl"))


def discover_runs(*, selected_run_names: list[str], include_all: bool) -> list[str]:
    if selected_run_names:
        return sorted(dict.fromkeys(selected_run_names))

    if not CHECKPOINTS_ROOT.is_dir():
        return []

    run_names: list[str] = []
    for checkpoint_dir in sorted(path for path in CHECKPOINTS_ROOT.iterdir() if path.is_dir()):
        if checkpoint_dir.name.startswith("_"):
            continue
        if not has_staged_eval(checkpoint_dir):
            continue
        experiment_dir = EXPERIMENT_DATA_ROOT / checkpoint_dir.name
        if include_all or not experiment_dir.exists():
            run_names.append(checkpoint_dir.name)
    return run_names


def condition_from_run_name(run_name: str) -> BatchCondition:
    match = RUN_CONDITION_RE.search(run_name)
    if not match:
        return BatchCondition(name=run_name, group="unknown", severity="unknown")

    group = f"g{match.group('group')}"
    severity = SEVERITY_ALIASES.get(match.group("severity").upper(), "unknown")
    condition_name = f"Counsel-{group.upper()}-{match.group('severity').upper()}"
    if condition_name.endswith("-MODERATE"):
        condition_name = condition_name.replace("-MODERATE", "-MOD")
    if condition_name.endswith("-SEVERE"):
        condition_name = condition_name.replace("-SEVERE", "-SEV")
    return BatchCondition(name=condition_name, group=group, severity=severity)


def make_runtime_config(report_name: str, conditions: list[BatchCondition], *, dry_run: bool) -> RuntimeConfig:
    return RuntimeConfig(
        name=report_name,
        start="",
        step=0,
        stride=0,
        verbose="",
        log_file="",
        agent="卡布达",
        dry_run=dry_run,
        summary_only=True,
        run_merge=False,
        run_post_scale=False,
        run_compress=False,
        run_agent_memory_vis=False,
        run_external_memory_audit=False,
        max_parallel=1,
        resume_batch=False,
        resume_condition=None,
        skip_completed=True,
        conditions=conditions,
    )


def copy_checkpoint_outputs(run_name: str, *, overwrite: bool, dry_run: bool) -> None:
    experiment_dir = EXPERIMENT_DATA_ROOT / run_name
    staged_dst = experiment_dir / "scales" / "staged"
    if experiment_dir.exists() and staged_dst.exists() and not overwrite:
        print(f"[SKIP-COLLECT] {run_name}: experiment_data/scales/staged 已存在")
        return
    collect_core_outputs(run_name, dry_run=dry_run)


def write_per_run_summary(run_dir: Path, condition: BatchCondition, result: dict[str, Any], *, dry_run: bool) -> Path:
    summary_path = run_dir / "scales" / "staged_eval_summary.json"
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "condition": {
            "name": condition.name,
            "group": condition.group,
            "severity": condition.severity,
        },
        "run_name": run_dir.name,
        "scales": list(SCALES.keys()),
        "evaluations": result.get("evaluations", []),
        "final_deltas": result.get("final_deltas", {}),
    }
    if dry_run:
        print(f"[DRY-RUN] write {summary_path}")
        return summary_path
    write_json_file(summary_path, payload)
    print(f"[WRITE] {summary_path}")
    return summary_path


def render_recovery_report(payload: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append(f"# 中断 checkpoint 的 staged_eval 恢复汇总：{payload['batch_name']}\n")
    lines.append(f"生成时间：{payload['generated_at']}\n")
    lines.append("> 本报告只统计 run_batch_experiment.py 当前配置中的 PHQ-9 / BDI-II。\n")
    lines.append("## 汇总范围\n")
    lines.append(f"- 条件数：{len(payload['conditions'])}")
    lines.append(f"- 评估点：{', '.join(payload['trigger_labels']) if payload['trigger_labels'] else '—'}")
    if payload.get("warnings"):
        lines.append("- 警告：")
        for warning in payload["warnings"]:
            lines.append(f"  - {warning}")
    lines.append("")

    lines.append("## 各 run 量表变化过程\n")
    for result in payload["conditions"]:
        lines.extend(render_condition_trajectory_markdown(result))

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
    lines.extend(render_final_delta_markdown("按 Group 的最终变化对比", payload["group_final_delta"], GROUPS))
    lines.extend(render_final_delta_markdown("按 Severity 的最终变化对比", payload["severity_final_delta"], SEVERITIES))
    lines.extend(render_group_severity_matrix_markdown(payload["group_severity_final_delta_matrix"]))
    return "\n".join(lines)


def write_combined_report(report_name: str, payload: dict[str, Any], *, dry_run: bool) -> tuple[Path, Path]:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    stem = f"{report_name}_{timestamp}"
    json_path = REPORTS_DIR / f"{stem}_summary.json"
    md_path = REPORTS_DIR / f"{stem}_summary.md"
    if dry_run:
        print(f"[DRY-RUN] write {json_path}")
        print(f"[DRY-RUN] write {md_path}")
        return json_path, md_path

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    write_json_file(json_path, payload)
    md_path.write_text(render_recovery_report(payload), encoding="utf-8")
    print(f"[WRITE] {json_path}")
    print(f"[WRITE] {md_path}")
    return json_path, md_path


def recover_run(run_name: str, *, overwrite: bool, skip_score: bool, dry_run: bool) -> dict[str, Any] | None:
    checkpoint_dir = CHECKPOINTS_ROOT / run_name
    if not checkpoint_dir.is_dir():
        print(f"[WARN] checkpoint 不存在，跳过: {checkpoint_dir}")
        return None
    if not has_staged_eval(checkpoint_dir):
        print(f"[WARN] staged_eval 中没有 answered.jsonl，跳过: {checkpoint_dir}")
        return None

    print(f"\n[RUN] {run_name}")
    copy_checkpoint_outputs(run_name, overwrite=overwrite, dry_run=dry_run)

    run_dir = EXPERIMENT_DATA_ROOT / run_name
    if not skip_score:
        ensure_staged_eval_scores(run_dir, dry_run=dry_run)

    condition = condition_from_run_name(run_name)
    result = load_condition_result(run_dir, condition)
    if result is None:
        print(f"[WARN] 无法读取恢复结果，跳过报告汇总: {run_dir}")
        return None

    write_per_run_summary(run_dir, condition, result, dry_run=dry_run)
    return {"condition": condition, "result": result}


def main() -> None:
    args = parse_args()
    report_name = str(args.report_name).strip() or DEFAULT_REPORT_NAME
    run_names = discover_runs(selected_run_names=args.run_name, include_all=args.all)
    if not run_names:
        print("[WARN] 没有找到需要恢复的 checkpoint。")
        return

    print("[INFO] 将处理以下 checkpoint：")
    for run_name in run_names:
        print(f"  - {run_name}")

    recovered: list[dict[str, Any]] = []
    for run_name in run_names:
        item = recover_run(
            run_name,
            overwrite=args.overwrite,
            skip_score=args.skip_score,
            dry_run=args.dry_run,
        )
        if item is not None:
            recovered.append(item)

    if not recovered:
        print("[WARN] 没有可写入报告的恢复结果。")
        return

    conditions = [item["condition"] for item in recovered]
    cfg = make_runtime_config(report_name, conditions, dry_run=args.dry_run)
    payload = build_summary_payload(
        cfg,
        [item["result"] for item in recovered],
        warnings=[],
    )
    payload["recovered_run_names"] = [item["result"]["run_name"] for item in recovered]
    payload["active_scales"] = list(SCALES.keys())

    write_combined_report(cfg.name, payload, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
