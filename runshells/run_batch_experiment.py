#!/usr/bin/env python3
"""简化版批量实验脚本。

只保留第一阶段批量实验需要的能力：
- 按 group × severity 替换配置
- 逐条件调用 start.py 跑仿真
- 复用 run_one_experiment.py 的后处理流程
- 导出当前批次的统一结果汇总

直接运行方式：
    python3 runshells/run_batch_experiment.py

也支持按条件筛选，例如：
    python3 runshells/run_batch_experiment.py --condition Counsel-G1-MILD
    python3 runshells/run_batch_experiment.py --condition Counsel-G1-ALL
    python3 runshells/run_batch_experiment.py --condition Counsel-ALL-MOD
"""

from __future__ import annotations

import argparse
import concurrent.futures
import copy
import json
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from run_one_experiment import (
    BASE_DIR,
    COMPRESS_SCRIPT,
    MERGE_SCRIPT,
    POST_EVAL_TIMEOUT_SECONDS,
    SCALE_SCORING_DIR,
    SCORE_WORKER_SCRIPT,
    SCALES,
    START_SCRIPT,
    WORKER_PYTHON,
    collect_core_outputs,
    ensure_checkpoint_state,
    run_agent_memory_visualization,
    run_cmd,
    run_external_memory_audit,
    run_post_scales,
)


# ============================================================
# ↓↓↓ 可调参数：直接修改这里即可（中文注释）↓↓↓
# ============================================================

# 批量实验基础名称；留空则自动生成，例如 sim-batch-0531-1430
RUN_NAME = ""

# 仿真起始时间（对应 start.py 的 --start）
START_TIME = "20260614-09:30"

# 仿真步数（对应 start.py 的 --step）
STEP = 280

# 每步推进的分钟数（对应 start.py 的 --stride）
STRIDE = 720

# 日志详细程度（对应 start.py 的 --verbose）
VERBOSE = "info"

# 日志文件名（对应 start.py 的 --log）；留空表示不额外写文件日志
LOG_FILE = "run_batch_experiment.log"

# 量表评估的目标角色
SCALE_AGENT = "卡布达"

# 是否执行“merge 对话与咨询记录”阶段
RUN_MERGE = True

# 是否执行治疗后量表评估（PHQ-9 / BDI-II）
RUN_POST_SCALE = True

# 是否执行 compress.py 生成回放资源
RUN_COMPRESS = True

# 是否执行角色记忆可视化（visualize_agent_memory.py）
RUN_AGENT_MEMORY_VIS = True

# 是否执行外置记忆审计可视化（visualize_external_memory_audit.py）
RUN_EXTERNAL_MEMORY_AUDIT = True

# 是否只做结果汇总（开启后跳过配置替换与批量仿真，只读取已有结果并生成汇总）
SUMMARY_ONLY = False

# 是否只打印命令，不实际执行
DRY_RUN = False

MAX_PARALLEL = 2

SIMULATION_LOG_POLL_SECONDS = 60
# 首步可能长时间停留在密集 LLM 调用里，此时日志未必持续刷新；
# 10 分钟会把“仍在工作但很慢”的运行误判成卡死并重启。
SIMULATION_LOG_STALE_SECONDS = 15 * 60
SIMULATION_MAX_RESTARTS = 3

# 当前 results 已迁移到数据盘软链接，这里把磁盘保护阈值调低到 50 MiB，
# 作为临时折中，避免仓库根目录所在分区的剩余空间误触发中断。
MIN_FREE_DISK_BYTES_TO_START = 50 * 1024 ** 2
MIN_FREE_DISK_BYTES_TO_CONTINUE = 50 * 1024 ** 2

# ============================================================
# ↑↑↑ 可调参数：直接修改这里即可（中文注释）↑↑↑
# ============================================================


DEPRESSION_CONFIG = BASE_DIR / "frontend" / "static" / "assets" / "village" / "agents" / "卡布达" / "depression_config.json"
GLOBAL_CONFIG = BASE_DIR / "data" / "config.json"
GROUP_OVERLAY_DIR = BASE_DIR / "experiments" / "config" / "groups"
EXPERIMENT_DATA_ROOT = BASE_DIR / "results" / "experiment_data"
REPORTS_DIR = EXPERIMENT_DATA_ROOT / "reports"
BATCH_STATE_ROOT = EXPERIMENT_DATA_ROOT / "batch_state"
PERSISTENT_GLOBAL_CONFIG_BACKUP = BASE_DIR / "experiments" / "config" / "default_config.backup.json"

PERSONAS = [
    "卡布达",
    "金龟次郎",
    "田德莉娜",
    "呱呱蛙",
    "蜻蜓队长",
    "蟑螂恶霸",
]

SEVERITY_CONFIG_FILES = {
    "mild": BASE_DIR / "frontend" / "static" / "assets" / "village" / "agents" / "卡布达" / "depression_config_mild.json",
    "moderate": BASE_DIR / "frontend" / "static" / "assets" / "village" / "agents" / "卡布达" / "depression_config_moderate.json",
    "severe": BASE_DIR / "frontend" / "static" / "assets" / "village" / "agents" / "卡布达" / "depression_config_severe.json",
}

GROUP_OVERLAY_FILES = {
    "g1": GROUP_OVERLAY_DIR / "g1_doctor_intervention.json",
    "g2": GROUP_OVERLAY_DIR / "g2_no_intervention.json",
    "g3": GROUP_OVERLAY_DIR / "g3_random_resident_chat.json",
    "g5": GROUP_OVERLAY_DIR / "g5_negative_resident_chat.json",
}

SEVERITY_SHORT_NAMES = {
    "mild": "MILD",
    "moderate": "MOD",
    "severe": "SEV",
}
SEVERITIES = ["mild", "moderate", "severe"]
GROUPS = ["g1", "g2", "g3", "g5"]
WILDCARD_TOKENS = {"*", "ALL"}
GROUP_SELECTOR_ALIASES = {group.upper(): group for group in GROUPS}
SEVERITY_SELECTOR_ALIASES = {
    "MILD": "mild",
    "MOD": "moderate",
    "MODERATE": "moderate",
    "SEV": "severe",
    "SEVERE": "severe",
}

@dataclass(frozen=True)
class BatchCondition:
    name: str
    group: str
    severity: str


ALL_CONDITIONS = [
    BatchCondition(
        name=f"Counsel-{group.upper()}-{SEVERITY_SHORT_NAMES[severity]}",
        group=group,
        severity=severity,
    )
    for group in GROUPS
    for severity in SEVERITIES
]


@dataclass
class RuntimeConfig:
    name: str
    start: str
    step: int
    stride: int
    verbose: str
    log_file: str
    agent: str
    dry_run: bool
    summary_only: bool
    run_merge: bool
    run_post_scale: bool
    run_compress: bool
    run_agent_memory_vis: bool
    run_external_memory_audit: bool
    max_parallel: int
    resume_batch: bool
    resume_condition: str | None
    skip_completed: bool
    conditions: list[BatchCondition]


@dataclass
class SimulationRunStats:
    attempt_count: int = 0
    restart_count: int = 0
    resumed_attempt_count: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run batch experiments with minimal config replacement flow")
    parser.add_argument("--name", default=None, help="覆盖脚本前面的 RUN_NAME（作为批量实验基础名）")
    parser.add_argument("--start", default=None, help="覆盖脚本前面的 START_TIME")
    parser.add_argument("--step", type=int, default=None, help="覆盖脚本前面的 STEP")
    parser.add_argument("--stride", type=int, default=None, help="覆盖脚本前面的 STRIDE")
    parser.add_argument("--verbose", default=None, help="覆盖脚本前面的 VERBOSE")
    parser.add_argument("--log", default=None, help="覆盖脚本前面的 LOG_FILE")
    parser.add_argument("--agent", default=None, help="覆盖脚本前面的 SCALE_AGENT")
    parser.add_argument(
        "--condition",
        default=None,
        help="只跑指定条件；支持 Counsel-G1-MILD、Counsel-G1-ALL、Counsel-ALL-MOD",
    )
    parser.add_argument("--dry-run", action="store_true", help="覆盖脚本前面的 DRY_RUN=True")
    parser.add_argument("--max-parallel", type=int, default=None, help="并行运行的 condition 数，默认 2")
    parser.add_argument("--resume-batch", action="store_true", help="只续跑当前 batch 中可续的失败项")
    parser.add_argument("--resume-condition", default=None, help="只续跑指定条件；支持和 --condition 相同的选择器")
    parser.add_argument("--skip-completed", dest="skip_completed", action="store_true", default=True, help="默认跳过已完成 condition")
    parser.add_argument("--no-skip-completed", dest="skip_completed", action="store_false", help="即使已完成也重新调度")
    parser.add_argument("--skip-merge", action="store_true", help="跳过 merge 阶段")
    parser.add_argument("--skip-post-scale", action="store_true", help="跳过治疗后 PHQ-9/BDI-II 评估")
    parser.add_argument("--skip-compress", action="store_true", help="跳过 compress.py")
    parser.add_argument("--run-agent-memory-vis", action="store_true", help="显式开启角色记忆可视化")
    parser.add_argument("--skip-agent-memory-vis", action="store_true", help="跳过角色记忆可视化")
    parser.add_argument("--run-external-memory-audit", action="store_true", help="显式开启外置记忆审计可视化")
    parser.add_argument("--skip-external-memory-audit", action="store_true", help="跳过外置记忆审计可视化")
    return parser.parse_args()


def generate_batch_name() -> str:
    return f"sim-batch-{datetime.now().strftime('%m%d-%H%M')}"


def _available_condition_selector_examples() -> str:
    return ", ".join([
        "Counsel-G1-MILD",
        "Counsel-G1-ALL",
        "Counsel-ALL-MOD",
    ])



def resolve_conditions(condition_name: str | None) -> list[BatchCondition]:
    if not condition_name:
        return list(ALL_CONDITIONS)

    normalized_name = str(condition_name or "").strip()
    matched = [condition for condition in ALL_CONDITIONS if condition.name == normalized_name]
    if matched:
        return matched

    tokens = [token.strip().upper() for token in normalized_name.split("-")]
    if len(tokens) == 3 and tokens[0] == "COUNSEL":
        group_token, severity_token = tokens[1], tokens[2]
        valid_group_token = group_token in WILDCARD_TOKENS or group_token in GROUP_SELECTOR_ALIASES
        valid_severity_token = severity_token in WILDCARD_TOKENS or severity_token in SEVERITY_SELECTOR_ALIASES
        if valid_group_token and valid_severity_token:
            target_group = None if group_token in WILDCARD_TOKENS else GROUP_SELECTOR_ALIASES[group_token]
            target_severity = None if severity_token in WILDCARD_TOKENS else SEVERITY_SELECTOR_ALIASES[severity_token]
            return [
                condition
                for condition in ALL_CONDITIONS
                if (target_group is None or condition.group == target_group)
                and (target_severity is None or condition.severity == target_severity)
            ]

    available = ", ".join(condition.name for condition in ALL_CONDITIONS)
    examples = _available_condition_selector_examples()
    raise ValueError(
        f"未找到条件: {normalized_name}\n"
        f"可用精确条件: {available}\n"
        f"也支持选择器: {examples}"
    )


def resolve_runtime_config(args: argparse.Namespace) -> RuntimeConfig:
    summary_only = bool(SUMMARY_ONLY)
    base_name = (args.name if args.name is not None else RUN_NAME).strip()
    if summary_only and not base_name:
        raise ValueError("SUMMARY_ONLY=True 时必须提供 RUN_NAME 或 --name，对应已有批量实验基础名。")
    if not base_name:
        base_name = generate_batch_name()

    return RuntimeConfig(
        name=base_name,
        start=args.start if args.start is not None else START_TIME,
        step=args.step if args.step is not None else STEP,
        stride=args.stride if args.stride is not None else STRIDE,
        verbose=args.verbose if args.verbose is not None else VERBOSE,
        log_file=args.log if args.log is not None else LOG_FILE,
        agent=args.agent if args.agent is not None else SCALE_AGENT,
        dry_run=bool(DRY_RUN or args.dry_run),
        summary_only=summary_only,
        run_merge=bool(RUN_MERGE and not args.skip_merge),
        run_post_scale=bool(RUN_POST_SCALE and not args.skip_post_scale),
        run_compress=bool(RUN_COMPRESS and not args.skip_compress),
        run_agent_memory_vis=bool((RUN_AGENT_MEMORY_VIS or args.run_agent_memory_vis) and not args.skip_agent_memory_vis),
        run_external_memory_audit=bool((RUN_EXTERNAL_MEMORY_AUDIT or args.run_external_memory_audit) and not args.skip_external_memory_audit),
        max_parallel=max(1, int(args.max_parallel if args.max_parallel is not None else MAX_PARALLEL)),
        resume_batch=bool(args.resume_batch),
        resume_condition=str(args.resume_condition or "").strip() or None,
        skip_completed=bool(args.skip_completed),
        conditions=resolve_conditions(args.condition),
    )


def print_effective_config(cfg: RuntimeConfig) -> None:
    print("==========================================")
    print(" 简化版批量实验配置")
    print("==========================================")
    print(f"  基础名称:     {cfg.name}")
    print(f"  起始时间:     {cfg.start}")
    print(f"  步数:         {cfg.step}")
    print(f"  步间隔:       {cfg.stride}")
    print(f"  verbose:      {cfg.verbose}")
    print(f"  log:          {cfg.log_file or '(空)'}")
    print(f"  评估角色:     {cfg.agent}")
    print(f"  仅做汇总:     {cfg.summary_only}")
    print(f"  条件数:       {len(cfg.conditions)}")
    print(f"  跑 merge:     {cfg.run_merge}")
    print(f"  跑治疗后量表: {cfg.run_post_scale}")
    print(f"  跑 compress:  {cfg.run_compress}")
    print(f"  跑记忆可视化: {cfg.run_agent_memory_vis}")
    print(f"  跑外置审计:   {cfg.run_external_memory_audit}")
    print(f"  并行度:       {cfg.max_parallel}")
    print(f"  续跑批次:     {cfg.resume_batch}")
    print(f"  续跑条件:     {cfg.resume_condition or '(空)'}")
    print(f"  跳过已完成:   {cfg.skip_completed}")
    print(f"  dry-run:      {cfg.dry_run}")
    print(f"  启动剩余空间: >= {format_bytes(MIN_FREE_DISK_BYTES_TO_START)}")
    print(f"  运行保底空间: >= {format_bytes(MIN_FREE_DISK_BYTES_TO_CONTINUE)}")
    print("  条件列表:     " + ", ".join(condition.name for condition in cfg.conditions))
    print("==========================================")


def build_trial_run_prefix(batch_name: str, condition_name: str) -> str:
    return f"{batch_name}-{condition_name}"


def build_trial_run_name(batch_name: str, condition_name: str) -> str:
    return f"{build_trial_run_prefix(batch_name, condition_name)}-{datetime.now().strftime('%m%d-%H%M')}"


def slugify_condition_name(value: str) -> str:
    text = str(value or "").strip()
    safe = []
    for ch in text:
        if ch.isalnum() or ch in {"-", "_"}:
            safe.append(ch)
        else:
            safe.append("_")
    return "".join(safe) or "condition"


def batch_state_dir(cfg: RuntimeConfig) -> Path:
    return BATCH_STATE_ROOT / cfg.name


def batch_runtime_config_dir(cfg: RuntimeConfig) -> Path:
    return batch_state_dir(cfg) / "runtime_configs"


def batch_condition_state_path(cfg: RuntimeConfig, condition: BatchCondition) -> Path:
    return batch_state_dir(cfg) / f"{slugify_condition_name(condition.name)}.json"


def batch_condition_timing_path(cfg: RuntimeConfig, condition: BatchCondition) -> Path:
    return batch_state_dir(cfg) / "timings" / f"{slugify_condition_name(condition.name)}.jsonl"


def batch_condition_runtime_config_path(cfg: RuntimeConfig, condition: BatchCondition) -> Path:
    return batch_runtime_config_dir(cfg) / f"{slugify_condition_name(condition.name)}.json"


def format_bytes(num_bytes: int) -> str:
    size = float(max(0, num_bytes))
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}TiB"


def format_duration(seconds: float) -> str:
    total_seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def current_free_disk_bytes() -> int:
    return shutil.disk_usage(BASE_DIR).free


def ensure_free_disk_space(*, min_free_bytes: int, context: str, dry_run: bool) -> int:
    free_bytes = current_free_disk_bytes()
    if free_bytes < min_free_bytes:
        message = (
            f"{context} 前磁盘剩余空间不足：当前 {format_bytes(free_bytes)}，"
            f"低于安全阈值 {format_bytes(min_free_bytes)}。"
            "建议先清理 `results/checkpoints`、`results/experiment_data` "
            "或迁移旧日志/回放文件后再继续。"
        )
        if dry_run:
            print(f"[DRY-RUN] {message}")
        else:
            raise RuntimeError(message)
    return free_bytes


def batch_timing_report_path(cfg: RuntimeConfig) -> Path:
    return REPORTS_DIR / f"{cfg.name}_timings.jsonl"


def append_condition_timing_record(cfg: RuntimeConfig, condition: BatchCondition, payload: dict[str, Any]) -> Path:
    path = batch_condition_timing_path(cfg, condition)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False))
        f.write("\n")
    print(f"[TIMING] {path}")
    return path


def merge_batch_timing_records(cfg: RuntimeConfig) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = batch_timing_report_path(cfg)
    lines: list[str] = []
    for condition in cfg.conditions:
        timing_path = batch_condition_timing_path(cfg, condition)
        if not timing_path.exists():
            continue
        lines.extend(
            line for line in timing_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    if cfg.dry_run:
        print(f"[DRY-RUN] merge timing records -> {path}")
        return path
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    print(f"[TIMING] merged -> {path}")
    return path


def matches_trial_run_name(run_name: str, batch_name: str, condition_name: str) -> bool:
    prefix = build_trial_run_prefix(batch_name, condition_name)
    return run_name == prefix or run_name.startswith(prefix + "-")


def find_latest_trial_run_dir(batch_name: str, condition_name: str) -> Path | None:
    if not EXPERIMENT_DATA_ROOT.is_dir():
        return None
    candidates = sorted(
        path for path in EXPERIMENT_DATA_ROOT.iterdir()
        if path.is_dir() and matches_trial_run_name(path.name, batch_name, condition_name)
    )
    if not candidates:
        return None
    return candidates[-1]


def checkpoint_dir_for_run(run_name: str) -> Path:
    return BASE_DIR / "results" / "checkpoints" / run_name


def archive_blocked_checkpoint_dir(run_name: str) -> None:
    checkpoint_dir = checkpoint_dir_for_run(run_name)
    if not checkpoint_dir.exists():
        return
    blocked_root = checkpoint_dir.parent / "_blocked_attempts"
    blocked_root.mkdir(parents=True, exist_ok=True)
    archived_dir = blocked_root / f"{run_name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    shutil.move(str(checkpoint_dir), str(archived_dir))
    print(f"[ARCHIVE] {checkpoint_dir} -> {archived_dir}")


def has_snapshot(run_name: str) -> bool:
    checkpoint_dir = checkpoint_dir_for_run(run_name)
    if not checkpoint_dir.is_dir():
        return False
    return any(
        path.name.startswith("simulate-") and path.name.endswith(".json")
        for path in checkpoint_dir.iterdir()
    )


def simulation_log_path(run_name: str, cfg: RuntimeConfig) -> Path:
    if not cfg.log_file:
        raise ValueError("日志静默恢复模式要求 start.py 写入日志文件，请设置 log_file。")
    return checkpoint_dir_for_run(run_name) / cfg.log_file


def latest_checkpoint_activity_mtime(run_name: str) -> float | None:
    checkpoint_dir = checkpoint_dir_for_run(run_name)
    if not checkpoint_dir.is_dir():
        return None

    latest_mtime = None
    for path in checkpoint_dir.rglob("*"):
        if not path.is_file():
            continue
        try:
            current_mtime = path.stat().st_mtime
        except FileNotFoundError:
            continue
        if latest_mtime is None or current_mtime > latest_mtime:
            latest_mtime = current_mtime
    return latest_mtime


def build_simulation_cmd(run_name: str, cfg: RuntimeConfig, *, resume: bool, runtime_config_path: Path | None = None) -> list[str]:
    cmd = [
        sys.executable,
        str(START_SCRIPT),
        "--name",
        run_name,
        "--step",
        str(cfg.step),
        "--stride",
        str(cfg.stride),
        "--verbose",
        cfg.verbose,
        "--start",
        cfg.start,
    ]
    if resume:
        cmd.append("--resume")
    elif runtime_config_path is not None:
        cmd.extend(["--runtime-config", str(runtime_config_path)])
    if cfg.log_file:
        cmd.extend(["--log", cfg.log_file])
    return cmd


def terminate_process(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def run_simulation_with_recovery(
    run_name: str,
    cfg: RuntimeConfig,
    *,
    runtime_config_path: Path | None = None,
    resume: bool = False,
    stats: SimulationRunStats | None = None,
) -> None:
    ensure_checkpoint_state(run_name, resume=resume, dry_run=cfg.dry_run)
    if cfg.dry_run:
        print(f"[DRY-RUN] monitored simulation for {run_name}")
        return

    stats = stats or SimulationRunStats()
    restart_count = 0
    resume_next = bool(resume)
    while True:
        ensure_free_disk_space(
            min_free_bytes=MIN_FREE_DISK_BYTES_TO_START,
            context=f"启动仿真子进程 {run_name}",
            dry_run=cfg.dry_run,
        )
        cmd = build_simulation_cmd(run_name, cfg, resume=resume_next, runtime_config_path=runtime_config_path)
        print(f"[RUN] {' '.join(cmd)}")
        stats.attempt_count += 1
        if resume_next:
            stats.resumed_attempt_count += 1
        proc = subprocess.Popen(cmd, cwd=BASE_DIR)
        log_path = simulation_log_path(run_name, cfg)
        last_log_mtime = None
        last_checkpoint_mtime = latest_checkpoint_activity_mtime(run_name)
        stale_since = time.time()

        while True:
            if proc.poll() is not None:
                if proc.returncode != 0:
                    raise subprocess.CalledProcessError(proc.returncode, cmd)
                return

            if log_path.exists():
                current_mtime = log_path.stat().st_mtime
                if last_log_mtime is None or current_mtime > last_log_mtime:
                    last_log_mtime = current_mtime
                    stale_since = time.time()

            current_checkpoint_mtime = latest_checkpoint_activity_mtime(run_name)
            if (
                current_checkpoint_mtime is not None
                and (last_checkpoint_mtime is None or current_checkpoint_mtime > last_checkpoint_mtime)
            ):
                last_checkpoint_mtime = current_checkpoint_mtime
                stale_since = time.time()

            free_bytes = current_free_disk_bytes()
            if free_bytes < MIN_FREE_DISK_BYTES_TO_CONTINUE:
                terminate_process(proc)
                raise RuntimeError(
                    f"仿真运行中磁盘剩余空间过低：当前 {format_bytes(free_bytes)}，"
                    f"已低于保底阈值 {format_bytes(MIN_FREE_DISK_BYTES_TO_CONTINUE)}，"
                    f"已提前停止子进程以保护 checkpoint：{run_name}"
                )

            if time.time() - stale_since >= SIMULATION_LOG_STALE_SECONDS:
                terminate_process(proc)
                restart_count += 1
                stats.restart_count = restart_count
                if restart_count > SIMULATION_MAX_RESTARTS:
                    raise TimeoutError(
                        f"simulation log stayed stale for {SIMULATION_LOG_STALE_SECONDS}s after {SIMULATION_MAX_RESTARTS} restarts: {run_name}"
                    )
                if has_snapshot(run_name):
                    print(f"[RESTART] resume simulation after stale log: {run_name}")
                    resume_next = True
                else:
                    archive_blocked_checkpoint_dir(run_name)
                    print(f"[RESTART] restart simulation from scratch after stale log: {run_name}")
                    resume_next = False
                break

            time.sleep(SIMULATION_LOG_POLL_SECONDS)


def summary_output_stem(cfg: RuntimeConfig) -> str:
    if len(cfg.conditions) == len(ALL_CONDITIONS):
        return cfg.name
    if len(cfg.conditions) == 1:
        return f"{cfg.name}-{cfg.conditions[0].name}"
    return f"{cfg.name}-subset-{len(cfg.conditions)}"


def load_json_file(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_optional_json_file(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        payload = load_json_file(path)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def write_json_file(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def deep_merge_dict(base: dict, overlay: dict) -> dict:
    merged = json.loads(json.dumps(base, ensure_ascii=False))
    for key, value in (overlay or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def latest_snapshot_name_for_run(run_name: str) -> str:
    checkpoint_dir = checkpoint_dir_for_run(run_name)
    if not checkpoint_dir.is_dir():
        return ""
    snapshots = sorted(
        path.name for path in checkpoint_dir.iterdir()
        if path.is_file() and path.name.startswith("simulate-") and path.name.endswith(".json")
    )
    if not snapshots:
        return ""
    return snapshots[-1]


def condition_checkpoint_exists(run_name: str) -> bool:
    return checkpoint_dir_for_run(run_name).is_dir()


def condition_experiment_dir_exists(run_name: str) -> bool:
    return (EXPERIMENT_DATA_ROOT / run_name).is_dir()


def runtime_config_template_path() -> Path:
    if PERSISTENT_GLOBAL_CONFIG_BACKUP.exists():
        return PERSISTENT_GLOBAL_CONFIG_BACKUP
    return GLOBAL_CONFIG


def build_condition_runtime_config_payload(condition: BatchCondition, cfg: RuntimeConfig) -> dict[str, Any]:
    template = load_json_file(runtime_config_template_path())
    overlay = load_json_file(GROUP_OVERLAY_FILES[condition.group])
    merged = deep_merge_dict(template, overlay)
    assets_root = "assets/village"
    payload: dict[str, Any] = {
        "stride": cfg.stride,
        "time": {"start": cfg.start},
        "maze": {"path": f"{assets_root}/maze.json"},
        "agent_base": copy.deepcopy(merged.get("agent", {})),
        "agents": {},
    }
    intervention_config = copy.deepcopy(merged.get("intervention", {}))
    staged_eval_config = copy.deepcopy(merged.get("staged_eval", {}))
    checkpointing_config = copy.deepcopy(merged.get("checkpointing", {}))
    if intervention_config:
        payload["intervention"] = intervention_config
    if staged_eval_config:
        payload["staged_eval"] = staged_eval_config
    if checkpointing_config:
        payload["checkpointing"] = checkpointing_config
    for agent_name in PERSONAS:
        payload["agents"][agent_name] = {
            "config_path": f"{assets_root}/agents/{agent_name.replace(' ', '_')}/agent.json",
        }
    payload["agents"].setdefault("卡布达", {})
    payload["agents"]["卡布达"]["depression_config_path"] = str(SEVERITY_CONFIG_FILES[condition.severity])
    return payload


def ensure_condition_runtime_config(cfg: RuntimeConfig, condition: BatchCondition) -> Path:
    path = batch_condition_runtime_config_path(cfg, condition)
    payload = build_condition_runtime_config_payload(condition, cfg)
    if cfg.dry_run:
        print(f"[DRY-RUN] write runtime config: {path}")
        return path
    write_json_file(path, payload)
    return path


def default_condition_state(cfg: RuntimeConfig, condition: BatchCondition) -> dict[str, Any]:
    return {
        "batch_name": cfg.name,
        "condition_name": condition.name,
        "group": condition.group,
        "severity": condition.severity,
        "run_name": "",
        "status": "queued",
        "resume_allowed": False,
        "last_completed_phase": "",
        "checkpoint_exists": False,
        "experiment_dir_exists": False,
        "latest_snapshot_name": "",
        "error": "",
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }


def read_condition_state(cfg: RuntimeConfig, condition: BatchCondition) -> dict[str, Any]:
    path = batch_condition_state_path(cfg, condition)
    if not path.exists():
        return default_condition_state(cfg, condition)
    payload = load_optional_json_file(path) or {}
    state = default_condition_state(cfg, condition)
    state.update(payload)
    return state


def write_condition_state(cfg: RuntimeConfig, condition: BatchCondition, state: dict[str, Any]) -> Path:
    path = batch_condition_state_path(cfg, condition)
    payload = default_condition_state(cfg, condition)
    payload.update(state or {})
    run_name = str(payload.get("run_name", "") or "").strip()
    if run_name:
        payload["checkpoint_exists"] = condition_checkpoint_exists(run_name)
        payload["experiment_dir_exists"] = condition_experiment_dir_exists(run_name)
        payload["latest_snapshot_name"] = latest_snapshot_name_for_run(run_name)
    payload["updated_at"] = datetime.now().isoformat(timespec="seconds")
    if cfg.dry_run:
        print(f"[DRY-RUN] write condition state: {path}")
        return path
    write_json_file(path, payload)
    return path


def update_condition_state(cfg: RuntimeConfig, condition: BatchCondition, **updates: Any) -> dict[str, Any]:
    state = read_condition_state(cfg, condition)
    state.update(updates)
    write_condition_state(cfg, condition, state)
    return state


def mean(values: list[float]) -> float | None:
    valid = [float(v) for v in values if v is not None]
    if not valid:
        return None
    return sum(valid) / len(valid)


def format_number(value: float | None, digits: int = 1) -> str:
    if value is None:
        return "—"
    return f"{float(value):.{digits}f}"


def format_delta(value: float | None, digits: int = 1) -> str:
    if value is None:
        return "—"
    return f"{float(value):+.{digits}f}"


def extract_scale_total(scored_result: dict) -> float | None:
    if not isinstance(scored_result, dict):
        return None
    for key in ["total_score", "total_score_raw", "standard_score"]:
        value = scored_result.get(key)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def extract_scale_severity(scored_result: dict) -> str:
    if not isinstance(scored_result, dict):
        return "—"
    for key in ["severity", "severity_by_index", "severity_by_standard_score"]:
        value = scored_result.get(key)
        if value:
            return str(value)
    return "—"


def trigger_sort_key(label: str, completed_session_count: int) -> tuple[int, int, str]:
    normalized = str(label or "")
    completed = max(0, int(completed_session_count))
    if normalized == "T0":
        return (0, 0, normalized)
    if normalized.startswith("session_"):
        return (completed, 1, normalized)
    if normalized == "T4":
        return (completed, 2, normalized)
    if normalized == "POST":
        return (completed, 3, normalized)
    return (completed, 4, normalized)


def write_batch_metadata(run_name: str, condition: BatchCondition, cfg: RuntimeConfig) -> None:
    output_dir = EXPERIMENT_DATA_ROOT / run_name
    meta_path = output_dir / "trial_meta.json"
    if not output_dir.exists():
        return

    meta = {}
    if meta_path.exists():
        try:
            meta = load_json_file(meta_path)
        except json.JSONDecodeError:
            meta = {}

    meta.update(
        {
            "trial_name": run_name,
            "condition_name": condition.name,
            "batch_name": cfg.name,
            "group": condition.group,
            "severity": condition.severity,
            "start": cfg.start,
            "step": cfg.step,
            "stride": cfg.stride,
            "verbose": cfg.verbose,
            "log_file": cfg.log_file,
            "scale_agent": cfg.agent,
            "group_overlay_file": GROUP_OVERLAY_FILES[condition.group].name,
            "severity_config_file": SEVERITY_CONFIG_FILES[condition.severity].name,
            "script": "runshells/run_batch_experiment.py",
        }
    )

    write_json_file(meta_path, meta)
    print(f"[WRITE] {meta_path}")


def run_simulation(
    run_name: str,
    cfg: RuntimeConfig,
    *,
    runtime_config_path: Path | None = None,
    resume: bool = False,
    stats: SimulationRunStats | None = None,
) -> None:
    run_simulation_with_recovery(
        run_name,
        cfg,
        runtime_config_path=runtime_config_path,
        resume=resume,
        stats=stats,
    )


def run_merge(run_name: str, cfg: RuntimeConfig) -> None:
    if not cfg.run_merge:
        return
    run_cmd([sys.executable, str(MERGE_SCRIPT), run_name], dry_run=cfg.dry_run, timeout=300)


def run_compress(run_name: str, cfg: RuntimeConfig) -> None:
    if not cfg.run_compress:
        return
    run_cmd([sys.executable, str(COMPRESS_SCRIPT), "--name", run_name], dry_run=cfg.dry_run, timeout=1800)


RESUMEABLE_CONDITION_STATUSES = {
    "simulation_interrupted_disk",
    "failed_after_checkpoint",
    "postprocessing_interrupted",
    "simulation_done",
    "postprocessing",
}


def is_disk_protection_error(exc: Exception) -> bool:
    message = str(exc or "")
    return "磁盘剩余空间过低" in message or "磁盘剩余空间不足" in message


def resolve_condition_run_name(cfg: RuntimeConfig, condition: BatchCondition, state: dict[str, Any]) -> str:
    run_name = str(state.get("run_name", "") or "").strip()
    if run_name:
        return run_name
    return build_trial_run_name(cfg.name, condition.name)


def should_resume_condition(cfg: RuntimeConfig, condition: BatchCondition, state: dict[str, Any]) -> bool:
    if cfg.resume_batch:
        return str(state.get("status", "") or "") in RESUMEABLE_CONDITION_STATUSES
    if cfg.resume_condition:
        resume_targets = {item.name for item in resolve_conditions(cfg.resume_condition)}
        return condition.name in resume_targets
    return False


def ensure_condition_state_initialized(cfg: RuntimeConfig, condition: BatchCondition) -> dict[str, Any]:
    state = read_condition_state(cfg, condition)
    if str(state.get("run_name", "") or "").strip():
        return state
    state["run_name"] = resolve_condition_run_name(cfg, condition, state)
    write_condition_state(cfg, condition, state)
    return state


def run_condition(condition: BatchCondition, cfg: RuntimeConfig) -> str:
    state = ensure_condition_state_initialized(cfg, condition)
    run_name = resolve_condition_run_name(cfg, condition, state)
    simulation_stats = SimulationRunStats()
    runtime_config_path = ensure_condition_runtime_config(cfg, condition)
    checkpoint_exists = condition_checkpoint_exists(run_name)
    resume_requested = should_resume_condition(cfg, condition, state)

    if state.get("status") == "completed" and not resume_requested and not cfg.skip_completed:
        run_name = build_trial_run_name(cfg.name, condition.name)
        state = default_condition_state(cfg, condition)
        state["run_name"] = run_name
        write_condition_state(cfg, condition, state)
        checkpoint_exists = False

    if checkpoint_exists and not resume_requested and str(state.get("status", "") or "") not in {"completed", "simulation_done"}:
        raise RuntimeError(
            f"{condition.name} 已存在未完成 checkpoint（run_name={run_name}），请使用 --resume-batch 或 --resume-condition 续跑。"
        )

    print("==========================================")
    print(f" 条件: {condition.name}")
    print("==========================================")
    print(f"  实际运行名:   {run_name}")
    print(f"  group:        {condition.group}")
    print(f"  severity:     {condition.severity}")
    print(f"  resume:       {resume_requested}")
    print("==========================================")

    update_condition_state(
        cfg,
        condition,
        run_name=run_name,
        status="queued",
        resume_allowed=bool(checkpoint_exists),
        error="",
    )

    should_run_simulation = True
    current_status = str(state.get("status", "") or "")
    if checkpoint_exists and current_status in {"simulation_done", "postprocessing_interrupted", "postprocessing", "completed"}:
        should_run_simulation = False

    if should_run_simulation:
        simulation_started_at = datetime.now()
        simulation_started_perf = time.monotonic()
        simulation_status = "success"
        simulation_error = ""
        update_condition_state(
            cfg,
            condition,
            run_name=run_name,
            status="running_simulation",
            resume_allowed=False,
            error="",
        )
        try:
            ensure_free_disk_space(
                min_free_bytes=MIN_FREE_DISK_BYTES_TO_START,
                context=f"开始条件 {condition.name}",
                dry_run=cfg.dry_run,
            )
            run_simulation(
                run_name,
                cfg,
                runtime_config_path=runtime_config_path,
                resume=resume_requested and checkpoint_exists,
                stats=simulation_stats,
            )
            update_condition_state(
                cfg,
                condition,
                run_name=run_name,
                status="simulation_done",
                resume_allowed=True,
                last_completed_phase="simulation",
                error="",
            )
        except Exception as exc:
            simulation_status = "failed"
            simulation_error = str(exc)
            checkpoint_exists = condition_checkpoint_exists(run_name)
            if is_disk_protection_error(exc):
                update_condition_state(
                    cfg,
                    condition,
                    run_name=run_name,
                    status="simulation_interrupted_disk",
                    resume_allowed=True,
                    error=simulation_error,
                )
            elif checkpoint_exists:
                update_condition_state(
                    cfg,
                    condition,
                    run_name=run_name,
                    status="failed_after_checkpoint",
                    resume_allowed=True,
                    error=simulation_error,
                )
            else:
                update_condition_state(
                    cfg,
                    condition,
                    run_name=run_name,
                    status="failed",
                    resume_allowed=False,
                    error=simulation_error,
                )
            raise
        finally:
            simulation_finished_at = datetime.now()
            simulation_duration_seconds = time.monotonic() - simulation_started_perf
            free_bytes_after_sim = current_free_disk_bytes()
            timing_payload = {
                "recorded_at": datetime.now().isoformat(timespec="seconds"),
                "batch_name": cfg.name,
                "condition_name": condition.name,
                "run_name": run_name,
                "phase": "simulation",
                "status": simulation_status,
                "started_at": simulation_started_at.isoformat(timespec="seconds"),
                "ended_at": simulation_finished_at.isoformat(timespec="seconds"),
                "duration_seconds": round(simulation_duration_seconds, 3),
                "duration_human": format_duration(simulation_duration_seconds),
                "attempt_count": simulation_stats.attempt_count,
                "restart_count": simulation_stats.restart_count,
                "resumed_attempt_count": simulation_stats.resumed_attempt_count,
                "free_disk_bytes_after": free_bytes_after_sim,
                "free_disk_after": format_bytes(free_bytes_after_sim),
                "log_file": cfg.log_file,
                "error": simulation_error,
            }
            if cfg.dry_run:
                print(f"[DRY-RUN] timing record: {json.dumps(timing_payload, ensure_ascii=False)}")
            else:
                append_condition_timing_record(cfg, condition, timing_payload)
            print(
                f"[TIMING] {condition.name} simulation {simulation_status}: "
                f"{format_duration(simulation_duration_seconds)} "
                f"(attempts={simulation_stats.attempt_count}, restarts={simulation_stats.restart_count})"
            )

    update_condition_state(
        cfg,
        condition,
        run_name=run_name,
        status="postprocessing",
        resume_allowed=True,
        error="",
    )
    try:
        ensure_free_disk_space(
            min_free_bytes=MIN_FREE_DISK_BYTES_TO_START,
            context=f"merge 阶段 {run_name}",
            dry_run=cfg.dry_run,
        )
        run_merge(run_name, cfg)
        update_condition_state(cfg, condition, run_name=run_name, last_completed_phase="merge")
        ensure_free_disk_space(
            min_free_bytes=MIN_FREE_DISK_BYTES_TO_START,
            context=f"收集结果 {run_name}",
            dry_run=cfg.dry_run,
        )
        collect_core_outputs(run_name, dry_run=cfg.dry_run)
        update_condition_state(cfg, condition, run_name=run_name, last_completed_phase="collect")

        if not cfg.dry_run:
            write_batch_metadata(run_name, condition, cfg)

        if cfg.run_post_scale:
            ensure_free_disk_space(
                min_free_bytes=MIN_FREE_DISK_BYTES_TO_START,
                context=f"量表评估 {run_name}",
                dry_run=cfg.dry_run,
            )
            run_post_scales(run_name, cfg.agent, dry_run=cfg.dry_run)
            update_condition_state(cfg, condition, run_name=run_name, last_completed_phase="post_scale")
        if cfg.run_compress:
            ensure_free_disk_space(
                min_free_bytes=MIN_FREE_DISK_BYTES_TO_START,
                context=f"compress 阶段 {run_name}",
                dry_run=cfg.dry_run,
            )
            run_compress(run_name, cfg)
            update_condition_state(cfg, condition, run_name=run_name, last_completed_phase="compress")
        if cfg.run_agent_memory_vis:
            ensure_free_disk_space(
                min_free_bytes=MIN_FREE_DISK_BYTES_TO_START,
                context=f"角色记忆可视化 {run_name}",
                dry_run=cfg.dry_run,
            )
            run_agent_memory_visualization(run_name, cfg.agent, dry_run=cfg.dry_run)
            update_condition_state(cfg, condition, run_name=run_name, last_completed_phase="agent_memory_vis")
        if cfg.run_external_memory_audit:
            ensure_free_disk_space(
                min_free_bytes=MIN_FREE_DISK_BYTES_TO_START,
                context=f"外置记忆审计 {run_name}",
                dry_run=cfg.dry_run,
            )
            run_external_memory_audit(run_name, cfg.agent, dry_run=cfg.dry_run)
            update_condition_state(cfg, condition, run_name=run_name, last_completed_phase="external_memory_audit")
    except Exception as exc:
        update_condition_state(
            cfg,
            condition,
            run_name=run_name,
            status="postprocessing_interrupted",
            resume_allowed=True,
            error=str(exc),
        )
        raise

    update_condition_state(
        cfg,
        condition,
        run_name=run_name,
        status="completed",
        resume_allowed=False,
        last_completed_phase="completed",
        error="",
    )

    return run_name


def ensure_staged_eval_scores(run_dir: Path, *, dry_run: bool) -> None:
    staged_root = run_dir / "scales" / "staged"
    if not staged_root.is_dir():
        return

    for trigger_dir in sorted(path for path in staged_root.iterdir() if path.is_dir()):
        for scale_name, scale_cfg in SCALES.items():
            answers_path = trigger_dir / f"{scale_name}_answered.jsonl"
            scored_path = trigger_dir / f"{scale_name}_scored.json"
            if not answers_path.exists() or scored_path.exists():
                continue
            scoring_prompt = SCALE_SCORING_DIR / scale_cfg["scoring_prompt"]
            run_cmd(
                [
                    WORKER_PYTHON,
                    str(SCORE_WORKER_SCRIPT),
                    "--answers",
                    str(answers_path),
                    "--scoring-prompt",
                    str(scoring_prompt),
                    "--output",
                    str(scored_path),
                ],
                dry_run=dry_run,
                timeout=POST_EVAL_TIMEOUT_SECONDS,
            )


def build_scale_snapshot(scored_result: dict | None, scored_path: Path | None = None) -> dict:
    total_score = extract_scale_total(scored_result or {})
    severity = extract_scale_severity(scored_result or {})
    return {
        "total_score": total_score,
        "severity": severity,
        "scored_file": str(scored_path) if scored_path is not None else "",
    }


def append_post_entry(evaluations: list[dict], run_dir: Path) -> None:
    scale_scores_path = run_dir / "scales" / "scale_scores.json"
    if not scale_scores_path.exists():
        return

    scale_scores = load_json_file(scale_scores_path)
    post_payload = scale_scores.get("post", {}) if isinstance(scale_scores, dict) else {}
    if not isinstance(post_payload, dict) or not post_payload:
        return

    completed_session_count = max(
        [int(item.get("completed_session_count", 0) or 0) for item in evaluations] or [0]
    )
    scales_payload = {}
    for scale_name in SCALES:
        scored_result = post_payload.get(scale_name)
        scales_payload[scale_name] = build_scale_snapshot(scored_result, scale_scores_path)

    evaluations.append(
        {
            "trigger_label": "POST",
            "completed_session_count": completed_session_count,
            "sim_time": "",
            "snapshot_name": "",
            "source": "post_scale",
            "metadata_path": str(scale_scores_path),
            "scales": scales_payload,
        }
    )


def apply_trajectory_deltas(evaluations: list[dict]) -> None:
    for scale_name in SCALES:
        baseline = None
        previous = None
        for evaluation in evaluations:
            scale_payload = evaluation["scales"].setdefault(scale_name, {})
            total_score = scale_payload.get("total_score")
            if total_score is None:
                scale_payload["delta_from_previous"] = None
                scale_payload["delta_from_baseline"] = None
                continue
            if baseline is None:
                baseline = float(total_score)
            if previous is None:
                delta_from_previous = 0.0
            else:
                delta_from_previous = float(total_score) - previous
            delta_from_baseline = float(total_score) - baseline
            scale_payload["delta_from_previous"] = delta_from_previous
            scale_payload["delta_from_baseline"] = delta_from_baseline
            previous = float(total_score)


def load_condition_result(run_dir: Path, condition: BatchCondition) -> dict | None:
    if not run_dir.is_dir():
        return None

    run_name = run_dir.name
    evaluations: list[dict] = []
    staged_root = run_dir / "scales" / "staged"
    if staged_root.is_dir():
        for trigger_dir in sorted(path for path in staged_root.iterdir() if path.is_dir()):
            metadata_path = trigger_dir / "metadata.json"
            if not metadata_path.exists():
                continue
            metadata = load_json_file(metadata_path)
            scales_payload = {}
            for scale_name in SCALES:
                scored_path = trigger_dir / f"{scale_name}_scored.json"
                scored_result = load_json_file(scored_path) if scored_path.exists() else {}
                scales_payload[scale_name] = build_scale_snapshot(scored_result, scored_path if scored_path.exists() else None)
            evaluations.append(
                {
                    "trigger_label": str(metadata.get("trigger_label", "") or ""),
                    "completed_session_count": int(metadata.get("completed_session_count", 0) or 0),
                    "sim_time": str(metadata.get("sim_time", "") or ""),
                    "snapshot_name": str(metadata.get("snapshot_name", "") or ""),
                    "source": "staged_eval",
                    "metadata_path": str(metadata_path),
                    "scales": scales_payload,
                }
            )

    append_post_entry(evaluations, run_dir)
    evaluations.sort(key=lambda item: trigger_sort_key(item.get("trigger_label", ""), int(item.get("completed_session_count", 0) or 0)))
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
        "condition_name": condition.name,
        "run_name": run_name,
        "run_dir": str(run_dir),
        "group": condition.group,
        "severity": condition.severity,
        "evaluations": evaluations,
        "final_deltas": final_deltas,
    }


def collect_batch_results(cfg: RuntimeConfig) -> tuple[list[dict], list[str]]:
    results = []
    warnings = []
    for condition in cfg.conditions:
        state = read_condition_state(cfg, condition)
        run_name = str(state.get("run_name", "") or "").strip()
        status = str(state.get("status", "") or "").strip()
        if run_name and status and status != "completed":
            warnings.append(f"跳过未完成条件: {condition.name} (status={status})")
            continue
        run_dir = EXPERIMENT_DATA_ROOT / run_name if run_name else find_latest_trial_run_dir(cfg.name, condition.name)
        if run_dir is None:
            warnings.append(f"缺少结果目录: {build_trial_run_prefix(cfg.name, condition.name)}[-时间后缀]")
            continue
        if not run_dir.is_dir():
            warnings.append(f"缺少结果目录: {run_dir.name}")
            continue
        if not cfg.dry_run:
            ensure_staged_eval_scores(run_dir, dry_run=False)
        result = load_condition_result(run_dir, condition)
        if result is None:
            warnings.append(f"无法读取结果: {run_dir.name}")
            continue
        results.append(result)
    return results, warnings


def ordered_trigger_labels(results: list[dict]) -> list[str]:
    label_keys = {}
    for result in results:
        for evaluation in result.get("evaluations", []):
            label = evaluation.get("trigger_label", "")
            count = int(evaluation.get("completed_session_count", 0) or 0)
            label_keys[label] = trigger_sort_key(label, count)
    return [label for label, _ in sorted(label_keys.items(), key=lambda item: item[1])]


def aggregate_dimension_trajectory(results: list[dict], *, dimension_key: str, dimension_values: list[str]) -> dict:
    labels = ordered_trigger_labels(results)
    payload = {}
    for scale_name in SCALES:
        scale_payload = {}
        for dimension_value in dimension_values:
            row = {}
            for label in labels:
                values = []
                for result in results:
                    if result.get(dimension_key) != dimension_value:
                        continue
                    for evaluation in result.get("evaluations", []):
                        if evaluation.get("trigger_label") != label:
                            continue
                        total_score = evaluation["scales"].get(scale_name, {}).get("total_score")
                        if total_score is not None:
                            values.append(float(total_score))
                row[label] = mean(values)
            scale_payload[dimension_value] = row
        payload[scale_name] = scale_payload
    return payload


def aggregate_final_deltas(results: list[dict], *, dimension_key: str, dimension_values: list[str]) -> dict:
    payload = {}
    for scale_name in SCALES:
        scale_payload = {}
        for dimension_value in dimension_values:
            values = []
            for result in results:
                if result.get(dimension_key) != dimension_value:
                    continue
                delta = result.get("final_deltas", {}).get(scale_name)
                if delta is not None:
                    values.append(float(delta))
            scale_payload[dimension_value] = mean(values)
        payload[scale_name] = scale_payload
    return payload


def build_group_severity_matrix(results: list[dict]) -> dict:
    payload = {}
    for scale_name in SCALES:
        scale_payload = {}
        for severity in SEVERITIES:
            row = {}
            for group in GROUPS:
                values = []
                for result in results:
                    if result.get("severity") != severity or result.get("group") != group:
                        continue
                    delta = result.get("final_deltas", {}).get(scale_name)
                    if delta is not None:
                        values.append(float(delta))
                row[group] = mean(values)
            scale_payload[severity] = row
        payload[scale_name] = scale_payload
    return payload


def build_summary_payload(cfg: RuntimeConfig, results: list[dict], warnings: list[str]) -> dict:
    return {
        "batch_name": cfg.name,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "summary_only": cfg.summary_only,
        "conditions": results,
        "warnings": warnings,
        "trigger_labels": ordered_trigger_labels(results),
        "group_trajectory": aggregate_dimension_trajectory(results, dimension_key="group", dimension_values=GROUPS),
        "severity_trajectory": aggregate_dimension_trajectory(results, dimension_key="severity", dimension_values=SEVERITIES),
        "group_final_delta": aggregate_final_deltas(results, dimension_key="group", dimension_values=GROUPS),
        "severity_final_delta": aggregate_final_deltas(results, dimension_key="severity", dimension_values=SEVERITIES),
        "group_severity_final_delta_matrix": build_group_severity_matrix(results),
    }


def render_condition_trajectory_markdown(result: dict) -> list[str]:
    lines = []
    lines.append(f"### {result['condition_name']}\n")
    lines.append(f"- run_name: `{result['run_name']}`")
    lines.append(f"- group: `{result['group']}`")
    lines.append(f"- severity: `{result['severity']}`")
    lines.append("")
    for scale_name in SCALES:
        lines.append(f"#### {scale_name}\n")
        lines.append("| 评估点 | session完成数 | sim_time | 总分 | 程度 | 相对上次Δ | 相对首次Δ |")
        lines.append("|--------|--------------|----------|------|------|-----------|-----------|")
        for evaluation in result.get("evaluations", []):
            scale_payload = evaluation["scales"].get(scale_name, {})
            lines.append(
                "| {label} | {completed} | {sim_time} | {total} | {severity} | {delta_prev} | {delta_base} |".format(
                    label=evaluation.get("trigger_label", "—"),
                    completed=evaluation.get("completed_session_count", 0),
                    sim_time=evaluation.get("sim_time", "") or "—",
                    total=format_number(scale_payload.get("total_score")),
                    severity=scale_payload.get("severity", "—"),
                    delta_prev=format_delta(scale_payload.get("delta_from_previous")),
                    delta_base=format_delta(scale_payload.get("delta_from_baseline")),
                )
            )
        lines.append("")
    return lines


def render_dimension_trajectory_markdown(title: str, trajectory_payload: dict, dimension_values: list[str], labels: list[str]) -> list[str]:
    lines = []
    lines.append(f"## {title}\n")
    for scale_name in SCALES:
        lines.append(f"### {scale_name}\n")
        header = "| 维度 | " + " | ".join(labels) + " |"
        sep = "|------|" + "|".join(["------"] * len(labels)) + "|"
        lines.append(header)
        lines.append(sep)
        for dimension_value in dimension_values:
            row = trajectory_payload.get(scale_name, {}).get(dimension_value, {})
            rendered = [format_number(row.get(label)) for label in labels]
            lines.append(f"| {dimension_value} | " + " | ".join(rendered) + " |")
        lines.append("")
    return lines


def render_final_delta_markdown(title: str, delta_payload: dict, dimension_values: list[str]) -> list[str]:
    lines = []
    lines.append(f"## {title}\n")
    header = "| 量表 | " + " | ".join(dimension_values) + " |"
    sep = "|------|" + "|".join(["------"] * len(dimension_values)) + "|"
    lines.append(header)
    lines.append(sep)
    for scale_name in SCALES:
        rendered = [format_delta(delta_payload.get(scale_name, {}).get(value)) for value in dimension_values]
        lines.append(f"| {scale_name} | " + " | ".join(rendered) + " |")
    lines.append("")
    return lines


def render_group_severity_matrix_markdown(matrix_payload: dict) -> list[str]:
    lines = []
    lines.append("## Group × Severity 最终变化矩阵\n")
    for scale_name in SCALES:
        lines.append(f"### {scale_name}\n")
        header = "| severity \\ group | " + " | ".join(GROUPS) + " |"
        sep = "|--------------------|" + "|".join(["----"] * len(GROUPS)) + "|"
        lines.append(header)
        lines.append(sep)
        for severity in SEVERITIES:
            row = matrix_payload.get(scale_name, {}).get(severity, {})
            rendered = [format_delta(row.get(group)) for group in GROUPS]
            lines.append(f"| {severity} | " + " | ".join(rendered) + " |")
        lines.append("")
    return lines


def export_batch_summary(cfg: RuntimeConfig) -> tuple[Path, Path] | None:
    results, warnings = collect_batch_results(cfg)
    if not results:
        print("[WARN] 没有找到可汇总的批量结果")
        for warning in warnings:
            print(f"- {warning}")
        return None

    payload = build_summary_payload(cfg, results, warnings)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    stem = summary_output_stem(cfg)
    json_path = REPORTS_DIR / f"{stem}_summary.json"
    md_path = REPORTS_DIR / f"{stem}_summary.md"
    write_json_file(json_path, payload)

    lines = []
    lines.append(f"# 批量实验结果汇总：{cfg.name}\n")
    lines.append(f"生成时间：{payload['generated_at']}\n")
    lines.append("> 本汇总只统计 PHQ-9 / BDI-II；当前不包含 30Q。\n")
    lines.append("## 汇总范围\n")
    lines.append(f"- 条件数：{len(results)}")
    lines.append(f"- 评估点：{', '.join(payload['trigger_labels']) if payload['trigger_labels'] else '—'}")
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

    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[WRITE] {json_path}")
    print(f"[WRITE] {md_path}")
    return json_path, md_path


def prepare_conditions_for_execution(cfg: RuntimeConfig) -> tuple[list[BatchCondition], list[str]]:
    selected: list[BatchCondition] = []
    notes: list[str] = []
    resume_targets = set()
    if cfg.resume_condition:
        resume_targets = {item.name for item in resolve_conditions(cfg.resume_condition)}

    for condition in cfg.conditions:
        state = read_condition_state(cfg, condition)
        status = str(state.get("status", "") or "").strip()
        run_name = str(state.get("run_name", "") or "").strip()
        if cfg.resume_condition and condition.name not in resume_targets:
            notes.append(f"[SKIP] 非指定续跑条件: {condition.name}")
            continue
        if cfg.resume_condition and status == "completed":
            notes.append(f"[SKIP] 指定续跑条件已完成: {condition.name}")
            continue
        if cfg.resume_batch and status not in RESUMEABLE_CONDITION_STATUSES:
            notes.append(f"[SKIP] 不在可续跑状态: {condition.name} (status={status or 'queued'})")
            continue
        if (
            (not cfg.resume_batch)
            and (not cfg.resume_condition)
            and run_name
            and condition_checkpoint_exists(run_name)
            and status
            and status not in {"completed"}
        ):
            notes.append(f"[BLOCKED] {condition.name} 存在未完成 checkpoint，请改用 --resume-batch 或 --resume-condition")
            continue
        if cfg.skip_completed and (not cfg.resume_batch) and (not cfg.resume_condition) and status == "completed":
            notes.append(f"[SKIP] 已完成: {condition.name}")
            continue
        selected.append(condition)
    return selected, notes


def main() -> None:
    args = parse_args()
    cfg = resolve_runtime_config(args)
    print_effective_config(cfg)

    if cfg.summary_only:
        export_batch_summary(cfg)
        return

    if not cfg.dry_run:
        batch_state_dir(cfg).mkdir(parents=True, exist_ok=True)
    conditions_to_run, notes = prepare_conditions_for_execution(cfg)
    for note in notes:
        print(note)

    completed: list[str] = []
    failed: list[str] = []
    if conditions_to_run:
        with concurrent.futures.ThreadPoolExecutor(max_workers=cfg.max_parallel) as executor:
            future_map = {}
            for index, condition in enumerate(conditions_to_run, start=1):
                print(f"\n##### [{index}/{len(conditions_to_run)}] {condition.name} #####")
                future = executor.submit(run_condition, condition, cfg)
                future_map[future] = condition
            for future in concurrent.futures.as_completed(future_map):
                condition = future_map[future]
                try:
                    run_name = future.result()
                    completed.append(run_name)
                except Exception as exc:
                    print(f"[ERROR] {condition.name}: {exc}")
                    failed.append(condition.name)
    else:
        print("[INFO] 当前没有需要执行的 condition。")

    merge_batch_timing_records(cfg)

    print("\n==========================================")
    print(f"批量实验完成: {len(completed)} 成功 / {len(failed)} 失败 / {len(notes)} 跳过")
    if completed:
        print("成功运行名:")
        for run_name in completed:
            print(f"- {run_name}")
    if failed:
        print("失败条件:")
        for condition_name in failed:
            print(f"- {condition_name}")
    print("==========================================")

    if (completed or cfg.summary_only or notes) and not cfg.dry_run:
        export_batch_summary(cfg)


if __name__ == "__main__":
    main()
