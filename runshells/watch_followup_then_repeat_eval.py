#!/usr/bin/env python3
"""持续检测已完成的无干预回访，并自动启动重复量表复评。

完成判定：
    本脚本从 checkpoint 中发现 ``followup_manifest.json``，不依赖日志、进程退出
    或单独一个 ``status=completed``。启动复评前会严格校验最终 full snapshot、全部
    ``followup_step_*`` staged-eval bundle、相对步数元数据、controller identity；
    默认还要求源 run 与 completed ``batch_state`` 的 run/condition/key 精确匹配。

输出隔离与幂等：
    每个回访使用以 ``-followup`` 结尾的独立 repeat batch 名，不覆盖常规仿真的
    repeat report。已有完整报告会跳过；不完整结果使用 ``--resume-partial`` 补齐；
    文件锁会阻止多个 watcher 同时启动同一复评，失败后按 ``--retry-seconds`` 重试。

运行前提：
    从项目根目录运行，并确保重复复评需要的 Qwen/BGE、forced LLM/API 等服务可用。

示例用法：

1. 查看完整命令行帮助：

       python runshells/watch_followup_then_repeat_eval.py --help

2. 只读扫描当前 results 下所有回访，显示将要启动的复评命令：

       python runshells/watch_followup_then_repeat_eval.py \
         --once --dry-run

3. 持续监控所有条件；每 30 秒扫描一次，按默认参数复评：

       python runshells/watch_followup_then_repeat_eval.py

4. 只持续监控一个条件，例如 KBD5-G1-SEV：

       python runshells/watch_followup_then_repeat_eval.py \
         --condition Counsel-KBD5-G1-SEV

5. 同时监控多个精确条件；``--condition`` 可以重复传入：

       python runshells/watch_followup_then_repeat_eval.py \
         --condition Counsel-KBD5-G1-SEV \
         --condition Counsel-KBD6-G1-SEV

6. 用 condition glob 监控一组条件，例如 KBD5-G1 的全部严重度：

       python runshells/watch_followup_then_repeat_eval.py \
         --condition-glob 'Counsel-KBD5-G1-*'

7. 按 follow-up 批次名筛选 rxx 重复，例如所有名称以 r 开头的重复：

       python runshells/watch_followup_then_repeat_eval.py \
         --followup-name-glob 'followup-*-r*'

8. 同时限定 condition 和 follow-up 名称。不同种类的过滤条件按 AND 组合；
   同一种过滤参数重复传入时按 OR 匹配：

       python runshells/watch_followup_then_repeat_eval.py \
         --condition Counsel-KBD5-G1-SEV \
         --followup-name-glob 'followup-*-r*'

9. 只扫描一次并实际补跑当前已经完成、但尚无完整 repeat report 的回访；
   等本次启动的全部复评结束后退出，不继续等待未来回访：

       python runshells/watch_followup_then_repeat_eval.py \
         --condition-glob 'Counsel-KBD5-G1-*' \
         --once

10. 调整扫描间隔、失败重试间隔和每个节点的固定复评次数：

       python runshells/watch_followup_then_repeat_eval.py \
         --poll-seconds 15 \
         --retry-seconds 600 \
         --repeat 10

11. 控制两层并发：同时运行 2 个独立 repeat eval，每个 eval 内部最多 4 个 worker：

       python runshells/watch_followup_then_repeat_eval.py \
         --max-concurrent-evals 2 \
         --eval-max-parallel 4

12. 监控另一个实验存档的 results 根目录：

       python runshells/watch_followup_then_repeat_eval.py \
         --results-root /path/to/archive/results \
         --once --dry-run

13. 只对迁移/导入后确实没有 batch_state 的存档关闭来源 batch_state 校验；
    full snapshot、staged bundle 和 controller 校验仍然保留：

       python runshells/watch_followup_then_repeat_eval.py \
         --results-root /path/to/archive/results \
         --no-require-batch-state

14. 在后台持续运行并把 watcher 自身输出写入日志：

       nohup python runshells/watch_followup_then_repeat_eval.py \
         --condition-glob 'Counsel-KBD5-G1-*' \
         > results/followup-repeat-watcher.log 2>&1 &

15. 常用的保守配置：一次只运行一个独立复评，内部并行 6，30 秒轮询：

       python runshells/watch_followup_then_repeat_eval.py \
         --poll-seconds 30 \
         --max-concurrent-evals 1 \
         --eval-max-parallel 6

生成位置：
    - 正式报告：``results/experiment_data/reports/``
    - repeat 原始结果：``results/experiment_data/repeat_scale_eval/``
    - watcher 单条件源 summary：
      ``results/experiment_data/followup_repeat_watch/sources/``
    - watcher 文件锁：``results/experiment_data/followup_repeat_watch/locks/``
    - 每个 repeat eval 的运行日志：``results/followup_repeat_watch/``
"""

from __future__ import annotations

import argparse
import concurrent.futures
import copy
import datetime as dt
import fnmatch
import fcntl
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from artifact_digest import canonical_json_sha256
from run_post_sim_followup import validate_snapshot_bundle


BASE_DIR = Path(__file__).resolve().parents[1]
FOLLOWUP_MANIFEST_FILENAME = "followup_manifest.json"
CONTROLLER_MANIFEST_FILENAME = "cbt_condition_manifest.json"
IDENTITY_KEYS = (
    "controller_identity",
    "kind",
    "mode",
    "controller_version",
    "progressive_stage",
    "capabilities",
)
SEVERITY_NAMES = {
    "MILD": "mild",
    "MOD": "moderate",
    "SEV": "severe",
}


@dataclass(frozen=True)
class WatchConfig:
    results_root: Path
    conditions: tuple[str, ...]
    condition_globs: tuple[str, ...]
    followup_name_globs: tuple[str, ...]
    poll_seconds: float
    retry_seconds: float
    repeat: int
    eval_max_parallel: int
    max_concurrent_evals: int
    require_batch_state: bool
    once: bool
    dry_run: bool

    @property
    def checkpoints_root(self) -> Path:
        return self.results_root / "checkpoints"

    @property
    def reports_root(self) -> Path:
        return self.results_root / "experiment_data" / "reports"

    @property
    def source_summary_root(self) -> Path:
        return self.results_root / "experiment_data" / "followup_repeat_watch" / "sources"

    @property
    def lock_root(self) -> Path:
        return self.results_root / "experiment_data" / "followup_repeat_watch" / "locks"

    @property
    def log_root(self) -> Path:
        return self.results_root / "followup_repeat_watch"


@dataclass(frozen=True)
class FollowupCandidate:
    manifest_path: Path
    run_dir: Path
    manifest: dict[str, Any]
    condition_name: str
    condition_key: str
    source_label: str
    destination_run: str
    followup_name: str
    repeat_name: str
    labels: tuple[str, ...]


@dataclass(frozen=True)
class EvalResult:
    repeat_name: str
    status: str
    returncode: int
    log_path: Path | None = None
    detail: str = ""


def timestamp() -> str:
    return dt.datetime.now().strftime("%F %T")


def announce(message: str) -> None:
    print(f"[{timestamp()}] {message}", flush=True)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON 顶层必须是对象: {path}")
    return payload


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp_path, path)


def parse_positive_int(raw: str, option: str) -> int:
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(f"{option} 必须是正整数") from exc
    if value <= 0:
        raise argparse.ArgumentTypeError(f"{option} 必须是正整数")
    return value


def parse_positive_float(raw: str, option: str) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(f"{option} 必须是正数") from exc
    if value <= 0:
        raise argparse.ArgumentTypeError(f"{option} 必须是正数")
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="持续检测严格完成的无干预回访，并自动启动独立的重复量表复评",
    )
    parser.add_argument("--results-root", default="results", help="results 根目录，默认 results")
    parser.add_argument(
        "--condition",
        action="append",
        default=[],
        help="只监控指定 condition，可重复传入，例如 Counsel-KBD5-G1-SEV",
    )
    parser.add_argument(
        "--condition-glob",
        action="append",
        default=[],
        help="condition glob，可重复传入，例如 Counsel-KBD5-G1-*",
    )
    parser.add_argument(
        "--followup-name-glob",
        action="append",
        default=[],
        help="follow-up 批次名 glob，可重复传入，例如 followup-*-0804-r*",
    )
    parser.add_argument(
        "--poll-seconds",
        type=lambda value: parse_positive_float(value, "--poll-seconds"),
        default=30.0,
        help="轮询间隔秒数，默认 30",
    )
    parser.add_argument(
        "--retry-seconds",
        type=lambda value: parse_positive_float(value, "--retry-seconds"),
        default=300.0,
        help="repeat eval 失败后的重试间隔，默认 300",
    )
    parser.add_argument(
        "--repeat",
        type=lambda value: parse_positive_int(value, "--repeat"),
        default=10,
        help="每个 agent × 节点 × 量表的固定复评次数，默认 10",
    )
    parser.add_argument(
        "--eval-max-parallel",
        type=lambda value: parse_positive_int(value, "--eval-max-parallel"),
        default=6,
        help="单个 repeat eval 内部并行数，默认 6",
    )
    parser.add_argument(
        "--max-concurrent-evals",
        type=lambda value: parse_positive_int(value, "--max-concurrent-evals"),
        default=1,
        help="同时运行的独立 repeat eval 数，默认 1",
    )
    parser.add_argument(
        "--require-batch-state",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="要求回访源 run 与 batch_state 精确匹配，默认开启",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="只扫描一次；非 dry-run 时等待本次发现的评估结束后退出",
    )
    parser.add_argument("--dry-run", action="store_true", help="只校验并打印待启动命令")
    return parser.parse_args()


def resolve_config(args: argparse.Namespace) -> WatchConfig:
    results_root = Path(args.results_root).expanduser()
    if not results_root.is_absolute():
        results_root = BASE_DIR / results_root
    results_root = results_root.resolve()
    if not results_root.is_dir():
        raise FileNotFoundError(f"results 根目录不存在: {results_root}")
    return WatchConfig(
        results_root=results_root,
        conditions=tuple(str(item).strip() for item in args.condition if str(item).strip()),
        condition_globs=tuple(str(item).strip() for item in args.condition_glob if str(item).strip()),
        followup_name_globs=tuple(
            str(item).strip() for item in args.followup_name_glob if str(item).strip()
        ),
        poll_seconds=float(args.poll_seconds),
        retry_seconds=float(args.retry_seconds),
        repeat=int(args.repeat),
        eval_max_parallel=int(args.eval_max_parallel),
        max_concurrent_evals=int(args.max_concurrent_evals),
        require_batch_state=bool(args.require_batch_state),
        once=bool(args.once),
        dry_run=bool(args.dry_run),
    )


def safe_name(value: str, label: str) -> str:
    token = str(value or "").strip()
    if not token or not re.fullmatch(r"[A-Za-z0-9._-]+", token):
        raise ValueError(f"{label} 含不安全字符: {value!r}")
    return token


def derive_followup_name(
    destination_run: str,
    condition_key: str,
    source_label: str,
) -> str:
    suffix = f"-{condition_key}-from-{source_label}"
    if not destination_run.endswith(suffix):
        raise ValueError(
            "follow-up destination_run 与 condition/source label 不匹配: "
            f"{destination_run}"
        )
    return safe_name(destination_run[: -len(suffix)], "follow-up name")


def derive_repeat_name(followup_name: str, condition_key: str) -> str:
    stem = followup_name.removeprefix("followup-")
    if condition_key not in followup_name:
        stem = f"{stem}--{condition_key}"
    return safe_name(f"repeat-{stem}-followup", "repeat name")


def candidate_from_manifest(path: Path) -> FollowupCandidate:
    manifest = load_json(path)
    if str(manifest.get("artifact_kind", "") or "") != "post_sim_followup":
        raise ValueError(f"follow-up manifest artifact_kind 无效: {path}")
    source = manifest.get("source")
    request = manifest.get("request")
    if not isinstance(source, dict) or not isinstance(request, dict):
        raise ValueError(f"follow-up manifest 缺少 source/request: {path}")
    condition_name = safe_name(str(source.get("condition_name", "") or ""), "condition")
    condition_key = safe_name(str(source.get("condition_key", "") or ""), "condition key")
    source_label = safe_name(str(source.get("label", "") or ""), "source label")
    destination_run = safe_name(
        str(manifest.get("destination_run", "") or ""),
        "destination run",
    )
    if destination_run != path.parent.name:
        raise ValueError(f"follow-up manifest 目录与 destination_run 不一致: {path}")
    if str(request.get("destination_run", "") or "") != destination_run:
        raise ValueError(f"follow-up request destination_run 不一致: {path}")
    labels = tuple(str(item or "").strip() for item in manifest.get("expected_labels", []))
    if not labels or any(not label for label in labels) or len(set(labels)) != len(labels):
        raise ValueError(f"follow-up expected_labels 无效: {path}")
    followup_name = derive_followup_name(destination_run, condition_key, source_label)
    return FollowupCandidate(
        manifest_path=path.resolve(),
        run_dir=path.parent.resolve(),
        manifest=manifest,
        condition_name=condition_name,
        condition_key=condition_key,
        source_label=source_label,
        destination_run=destination_run,
        followup_name=followup_name,
        repeat_name=derive_repeat_name(followup_name, condition_key),
        labels=labels,
    )


def matches_filters(candidate: FollowupCandidate, cfg: WatchConfig) -> bool:
    if cfg.conditions and candidate.condition_name not in cfg.conditions:
        return False
    if cfg.condition_globs and not any(
        fnmatch.fnmatchcase(candidate.condition_name, pattern)
        for pattern in cfg.condition_globs
    ):
        return False
    if cfg.followup_name_globs and not any(
        fnmatch.fnmatchcase(candidate.followup_name, pattern)
        for pattern in cfg.followup_name_globs
    ):
        return False
    return True


def discover_candidates(cfg: WatchConfig) -> list[FollowupCandidate]:
    candidates: list[FollowupCandidate] = []
    for path in sorted(cfg.checkpoints_root.glob(f"*/{FOLLOWUP_MANIFEST_FILENAME}")):
        try:
            candidate = candidate_from_manifest(path)
        except Exception as exc:
            announce(f"[WARN] 忽略无效 manifest {path}: {exc}")
            continue
        if matches_filters(candidate, cfg):
            candidates.append(candidate)
    return candidates


def find_matching_batch_state(candidate: FollowupCandidate, cfg: WatchConfig) -> Path | None:
    source = candidate.manifest["source"]
    source_run = str(source.get("run_name", "") or "")
    state_root = cfg.results_root / "experiment_data" / "batch_state"
    for path in sorted(state_root.glob("*/*.json")):
        try:
            state = load_json(path)
        except Exception:
            continue
        if (
            str(state.get("run_name", "") or "") == source_run
            and str(state.get("condition_name", "") or "") == candidate.condition_name
            and str(state.get("condition_key", "") or "") == candidate.condition_key
            and str(state.get("status", "") or "") == "completed"
        ):
            return path.resolve()
    return None


def validate_completed_candidate(candidate: FollowupCandidate, cfg: WatchConfig) -> None:
    manifest = candidate.manifest
    if str(manifest.get("status", "") or "") != "completed":
        raise ValueError(f"status={manifest.get('status', '') or '(empty)'}")
    request = manifest["request"]
    source = manifest["source"]
    request_labels = tuple(str(item or "") for item in request.get("expected_labels", []))
    actual_labels = tuple(str(item or "") for item in manifest.get("actual_labels", []))
    if request_labels != candidate.labels or actual_labels != candidate.labels:
        raise ValueError(
            "expected/request/actual labels 不一致: "
            f"expected={candidate.labels} request={request_labels} actual={actual_labels}"
        )
    source_step = int(source.get("step", -1))
    followup_steps = int(request.get("followup_steps", -1))
    target_step = int(manifest.get("target_step", -1))
    if source_step < 0 or followup_steps <= 0 or target_step != source_step + followup_steps:
        raise ValueError("source/follow-up/target step 不一致")

    snapshots: list[tuple[int, Path]] = []
    for snapshot_path in candidate.run_dir.glob("simulate-*.json"):
        try:
            snapshots.append((int(load_json(snapshot_path).get("step", -1)), snapshot_path))
        except Exception:
            continue
    if not snapshots or max(step for step, _ in snapshots) != target_step:
        raise ValueError(f"缺少 target_step={target_step} 的最终 full snapshot")

    interval = int(request.get("interval", -1))
    if interval <= 0:
        raise ValueError("follow-up interval 无效")
    observed_nodes: list[dict[str, Any]] = []
    for index, label in enumerate(candidate.labels, start=1):
        relative_step = index * interval
        if label != f"followup_step_{relative_step}":
            raise ValueError(f"label 与 interval 不一致: {label}")
        expected_step = source_step + relative_step
        node_dir = candidate.run_dir / "staged_eval" / label
        job = load_json(node_dir / "job.json")
        metadata = load_json(node_dir / "metadata.json")
        validate_snapshot_bundle(node_dir, job, label)
        runtime = job.get("runtime_config")
        if not isinstance(runtime, dict):
            raise ValueError(f"node runtime_config 无效: {node_dir}")
        observed_steps = {
            int(job.get("step_no", -1)),
            int(metadata.get("step_no", -1)),
            int(runtime.get("step", -1)),
        }
        if observed_steps != {expected_step}:
            raise ValueError(
                f"node step 不一致: label={label} expected={expected_step} actual={observed_steps}"
            )
        expected_metadata = {
            "artifact_kind": "repeat_eval_snapshot",
            "trigger_label": label,
            "step_interval_anchor_step": source_step,
            "step_interval_elapsed_steps": relative_step,
            "step_interval_label_value_mode": "relative_step",
            "followup_phase": True,
        }
        mismatches = {
            key: {"expected": expected, "actual": metadata.get(key)}
            for key, expected in expected_metadata.items()
            if metadata.get(key) != expected
        }
        if mismatches:
            raise ValueError(f"node metadata 不一致: {label} {mismatches}")
        observed_nodes.append(
            {
                "trigger_label": label,
                "step_no": expected_step,
                "snapshot_name": str(job.get("snapshot_name", "") or ""),
            }
        )

    actual_nodes = manifest.get("actual_nodes", [])
    if not isinstance(actual_nodes, list) or [
        str(item.get("trigger_label", "") or "")
        for item in actual_nodes
        if isinstance(item, dict)
    ] != list(candidate.labels):
        raise ValueError("manifest actual_nodes 不完整")

    controller_path = candidate.run_dir / CONTROLLER_MANIFEST_FILENAME
    controller = load_json(controller_path)
    if str(controller.get("artifact_kind", "") or "") != "cbt_experiment_condition":
        raise ValueError(f"controller manifest artifact_kind 无效: {controller_path}")
    if str(controller.get("run_name", "") or "") != candidate.destination_run:
        raise ValueError(f"controller manifest run_name 不一致: {controller_path}")
    if str(controller.get("condition_key", "") or "") != candidate.condition_key:
        raise ValueError(f"controller manifest condition_key 不一致: {controller_path}")
    if cfg.require_batch_state and find_matching_batch_state(candidate, cfg) is None:
        raise ValueError("找不到与 source run/condition/key 精确匹配的 completed batch_state")


def condition_metadata(condition_name: str) -> dict[str, str]:
    match = re.fullmatch(r"Counsel-(KBD\d+)-(G\d+)-([A-Za-z]+)", condition_name)
    if not match:
        return {"variant": "", "group": "", "severity": ""}
    variant, group, severity = match.groups()
    return {
        "variant": variant.lower(),
        "group": group.lower(),
        "severity": SEVERITY_NAMES.get(severity.upper(), severity.lower()),
    }


def source_summary_path(candidate: FollowupCandidate, cfg: WatchConfig) -> Path:
    return cfg.source_summary_root / f"{candidate.repeat_name}_source.json"


def build_source_summary(candidate: FollowupCandidate) -> dict[str, Any]:
    controller_path = candidate.run_dir / CONTROLLER_MANIFEST_FILENAME
    controller = load_json(controller_path)
    metadata = condition_metadata(candidate.condition_name)
    evaluations: list[dict[str, Any]] = []
    for label in candidate.labels:
        node_dir = candidate.run_dir / "staged_eval" / label
        job = load_json(node_dir / "job.json")
        node_metadata = load_json(node_dir / "metadata.json")
        evaluations.append(
            {
                "trigger_label": label,
                "completed_session_count": int(
                    node_metadata.get("completed_session_count", 0) or 0
                ),
                "sim_time": str(job.get("sim_time", "") or ""),
                "snapshot_name": str(job.get("snapshot_name", "") or ""),
                "source": "staged_eval_snapshot",
                "evaluation_executed": False,
                "scales": {},
            }
        )
    condition: dict[str, Any] = {
        "condition_name": candidate.condition_name,
        "run_name": candidate.destination_run,
        "variant": metadata["variant"],
        "group": metadata["group"],
        "severity": metadata["severity"],
        "condition_key": candidate.condition_key,
        "controller_manifest_path": str(controller_path.resolve()),
        "controller_manifest": controller,
        "evaluations": evaluations,
        "trigger_labels": list(candidate.labels),
    }
    for key in IDENTITY_KEYS:
        condition[key] = copy.deepcopy(controller.get(key))
    return {
        "schema_version": 1,
        "artifact_kind": "followup_repeat_watch_source_summary",
        "batch_name": candidate.followup_name,
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "source_followup_manifest": str(candidate.manifest_path),
        "source_followup_manifest_sha256": canonical_json_sha256(candidate.manifest),
        "conditions": [condition],
        "trigger_labels": list(candidate.labels),
        "warnings": [],
    }


def ensure_source_summary(
    candidate: FollowupCandidate,
    cfg: WatchConfig,
) -> Path:
    path = source_summary_path(candidate, cfg)
    payload = build_source_summary(candidate)
    if path.is_file():
        existing = load_json(path)
        if (
            str(existing.get("source_followup_manifest_sha256", "") or "")
            == payload["source_followup_manifest_sha256"]
            and existing.get("trigger_labels") == payload["trigger_labels"]
        ):
            return path.resolve()
    write_json_atomic(path, payload)
    return path.resolve()


def report_path(candidate: FollowupCandidate, cfg: WatchConfig) -> Path:
    return cfg.reports_root / f"{candidate.repeat_name}_summary.json"


def report_complete(candidate: FollowupCandidate, cfg: WatchConfig) -> bool:
    path = report_path(candidate, cfg)
    if not path.is_file():
        return False
    try:
        report = load_json(path)
    except Exception:
        return False
    completion = report.get("completion")
    conditions = report.get("conditions")
    recorded_source = str(report.get("source_original_summary", "") or "").strip()
    expected_source = source_summary_path(candidate, cfg).resolve()
    try:
        source_matches = bool(recorded_source) and Path(recorded_source).resolve() == expected_source
    except (OSError, RuntimeError):
        source_matches = False
    return bool(
        report.get("batch_name") == candidate.repeat_name
        and int(report.get("repeat", -1)) == cfg.repeat
        and report.get("labels") == list(candidate.labels)
        and source_matches
        and isinstance(completion, dict)
        and completion.get("ready_for_final_report") is True
        and isinstance(conditions, list)
        and any(
            isinstance(item, dict)
            and item.get("condition_name") == candidate.condition_name
            and item.get("run_name") == candidate.destination_run
            for item in conditions
        )
    )


def build_eval_command(
    candidate: FollowupCandidate,
    cfg: WatchConfig,
    summary_path: Path,
) -> list[str]:
    return [
        sys.executable,
        str(BASE_DIR / "runshells" / "run_archived_repeat_scale_eval.py"),
        "--archive-results-root",
        str(cfg.results_root),
        "--condition",
        candidate.condition_name,
        "--original-summary",
        str(summary_path),
        "--labels",
        ",".join(candidate.labels),
        "--repeat",
        str(cfg.repeat),
        "--name",
        candidate.repeat_name,
        "--max-parallel",
        str(cfg.eval_max_parallel),
        "--resume-partial",
        "--require-controller-manifest",
    ]


def shell_command(command: Iterable[str]) -> str:
    import shlex

    return " ".join(shlex.quote(str(item)) for item in command)


def run_evaluation(candidate: FollowupCandidate, cfg: WatchConfig) -> EvalResult:
    cfg.lock_root.mkdir(parents=True, exist_ok=True)
    lock_path = cfg.lock_root / f"{candidate.repeat_name}.lock"
    lock_handle = lock_path.open("a+", encoding="utf-8")
    try:
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return EvalResult(candidate.repeat_name, "locked", 0, detail=str(lock_path))
        lock_handle.seek(0)
        lock_handle.truncate()
        lock_handle.write(f"pid={os.getpid()} started_at={timestamp()}\n")
        lock_handle.flush()
        if report_complete(candidate, cfg):
            return EvalResult(candidate.repeat_name, "already_complete", 0)
        summary_path = ensure_source_summary(candidate, cfg)
        command = build_eval_command(candidate, cfg, summary_path)
        cfg.log_root.mkdir(parents=True, exist_ok=True)
        log_path = cfg.log_root / f"{candidate.repeat_name}.log"
        with log_path.open("a", encoding="utf-8") as log_handle:
            log_handle.write(f"\n[{timestamp()}] START {shell_command(command)}\n")
            log_handle.flush()
            result = subprocess.run(
                command,
                cwd=BASE_DIR,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
            log_handle.write(f"[{timestamp()}] EXIT returncode={result.returncode}\n")
        if result.returncode != 0:
            return EvalResult(candidate.repeat_name, "failed", result.returncode, log_path)
        if not report_complete(candidate, cfg):
            return EvalResult(
                candidate.repeat_name,
                "failed",
                1,
                log_path,
                "进程返回 0，但完整 summary 校验未通过",
            )
        return EvalResult(candidate.repeat_name, "completed", 0, log_path)
    except Exception as exc:
        return EvalResult(candidate.repeat_name, "failed", 1, detail=str(exc))
    finally:
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
        finally:
            lock_handle.close()


def eligible_candidates(cfg: WatchConfig) -> list[FollowupCandidate]:
    eligible: list[FollowupCandidate] = []
    for candidate in discover_candidates(cfg):
        status = str(candidate.manifest.get("status", "") or "")
        if status != "completed":
            continue
        try:
            validate_completed_candidate(candidate, cfg)
        except Exception as exc:
            announce(f"[WARN] 回访声称完成但严格校验失败 {candidate.destination_run}: {exc}")
            continue
        eligible.append(candidate)
    return eligible


def dry_run_once(cfg: WatchConfig) -> int:
    candidates = eligible_candidates(cfg)
    if not candidates:
        announce("未发现已严格完成且匹配过滤条件的回访")
        return 0
    for candidate in candidates:
        if report_complete(candidate, cfg):
            announce(f"[SKIP] repeat report 已完整: {report_path(candidate, cfg)}")
            continue
        summary_path = source_summary_path(candidate, cfg).resolve()
        command = build_eval_command(candidate, cfg, summary_path)
        announce(
            f"[DRY-RUN] {candidate.destination_run} -> {candidate.repeat_name}\n"
            f"  {shell_command(command)}"
        )
    return 0


def watch(cfg: WatchConfig) -> int:
    if cfg.dry_run:
        return dry_run_once(cfg)

    announce(
        "开始监控 follow-up manifests: "
        f"root={cfg.checkpoints_root} poll={cfg.poll_seconds:g}s "
        f"repeat={cfg.repeat} eval_parallel={cfg.eval_max_parallel} "
        f"eval_jobs={cfg.max_concurrent_evals}"
    )
    retries: dict[str, float] = {}
    submitted_once: set[str] = set()
    active: dict[concurrent.futures.Future[EvalResult], FollowupCandidate] = {}
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=cfg.max_concurrent_evals)
    try:
        while True:
            for future in list(active):
                if not future.done():
                    continue
                candidate = active.pop(future)
                result = future.result()
                if result.status in {"completed", "already_complete"}:
                    announce(
                        f"[DONE] {candidate.repeat_name}"
                        + (f" log={result.log_path}" if result.log_path else "")
                    )
                    retries.pop(candidate.repeat_name, None)
                elif result.status == "locked":
                    announce(f"[SKIP] 其他监控进程正在处理 {candidate.repeat_name}")
                    retries[candidate.repeat_name] = time.monotonic() + cfg.retry_seconds
                else:
                    announce(
                        f"[ERROR] repeat eval 失败 {candidate.repeat_name} "
                        f"returncode={result.returncode} detail={result.detail or '-'} "
                        f"log={result.log_path or '-'}"
                    )
                    retries[candidate.repeat_name] = time.monotonic() + cfg.retry_seconds

            now = time.monotonic()
            active_names = {item.repeat_name for item in active.values()}
            for candidate in eligible_candidates(cfg):
                if report_complete(candidate, cfg):
                    continue
                if candidate.repeat_name in active_names:
                    continue
                if cfg.once and candidate.repeat_name in submitted_once:
                    continue
                if retries.get(candidate.repeat_name, 0.0) > now:
                    continue
                future = executor.submit(run_evaluation, candidate, cfg)
                active[future] = candidate
                active_names.add(candidate.repeat_name)
                submitted_once.add(candidate.repeat_name)
                announce(
                    f"[START] 回访已严格完成，启动 repeat eval: "
                    f"{candidate.destination_run} -> {candidate.repeat_name}"
                )

            if cfg.once and not active:
                return 0
            time.sleep(min(cfg.poll_seconds, 1.0) if cfg.once else cfg.poll_seconds)
    except KeyboardInterrupt:
        announce("收到中断，等待已启动的 repeat eval 收尾；再次 Ctrl-C 可由终端终止进程")
        return 130
    finally:
        executor.shutdown(wait=True, cancel_futures=False)


def main() -> None:
    try:
        cfg = resolve_config(parse_args())
        raise SystemExit(watch(cfg))
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"错误: {exc}") from exc


if __name__ == "__main__":
    main()
