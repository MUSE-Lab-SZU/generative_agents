#!/usr/bin/env python3
"""单次实验自动化脚本。

直接运行方式：
    python3 runshells/run_one_experiment.py

也支持用命令行覆盖部分参数，例如：
    python3 runshells/run_one_experiment.py --name my-exp --step 20 --stride 360
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional


# ============================================================
# ↓↓↓ 可调参数：直接修改这里即可（中文注释）↓↓↓
# ============================================================

# 实验名称；留空则自动生成，例如 sim-one-0514-1830
RUN_NAME = "sim-test-0529"

# 是否续跑已有实验（对应 start.py 的 --resume）
# - False：新开一个实验
# - True：基于已有 checkpoint 继续跑
RESUME_RUN = False

# 仿真起始时间（对应 start.py 的 --start）
# 仅在 RESUME_RUN=False 时生效
START_TIME = "20260529-09:30"

# 仿真步数（对应 start.py 的 --step）
STEP = 560

# 每步推进的分钟数（对应 start.py 的 --stride）
STRIDE = 360

# 日志详细程度（对应 start.py 的 --verbose）
VERBOSE = "info"

# 日志文件名（对应 start.py 的 --log）；留空表示不额外写文件日志
LOG_FILE = "sim-test-0529.log"

# 量表评估的目标角色
SCALE_AGENT = "卡布达"

# 是否执行“仿真”阶段
RUN_SIMULATION = True

# 是否执行“merge 对话与咨询记录”阶段
RUN_MERGE = True

# 是否执行治疗后量表评估（PHQ-9 / BDI-II）
RUN_POST_SCALE = True

# 是否执行治疗后 30Q 评估
# 当前仓库里默认未发现 30Q 综合评分 prompt，默认关闭更安全
RUN_30Q = False

# 是否执行 compress.py 生成回放资源
RUN_COMPRESS = True

# 是否执行角色记忆可视化（visualize_agent_memory.py）
RUN_AGENT_MEMORY_VIS = True

# 是否执行外置记忆审计可视化（visualize_external_memory_audit.py）
RUN_EXTERNAL_MEMORY_AUDIT = True

# 是否只打印命令，不实际执行
DRY_RUN = False

# ============================================================
# ↑↑↑ 可调参数：直接修改这里即可（中文注释）↑↑↑
# ============================================================


BASE_DIR = Path(__file__).resolve().parent.parent
CHECKPOINTS_ROOT = BASE_DIR / "results" / "checkpoints"
EXPERIMENT_DATA_ROOT = BASE_DIR / "results" / "experiment_data"
START_SCRIPT = BASE_DIR / "start.py"
MERGE_SCRIPT = BASE_DIR / "merge_consultation_dialogues.py"
COMPRESS_SCRIPT = BASE_DIR / "compress.py"
AGENT_MEMORY_VIS_SCRIPT = BASE_DIR / "visualize_agent_memory.py"
EXTERNAL_MEMORY_AUDIT_SCRIPT = BASE_DIR / "visualize_external_memory_audit.py"
SCALE_WORKER_SCRIPT = BASE_DIR / "runshells" / "run_scale_worker.py"
SCORE_WORKER_SCRIPT = BASE_DIR / "runshells" / "run_score_worker.py"
WORKER_PYTHON = os.environ.get("GA_WORKER_PYTHON") or sys.executable
SCALE_SCORING_DIR = BASE_DIR / "customization" / "depression_scale_agent" / "questions" / "scoring_prompts"
THIRTY_Q_PROMPT = BASE_DIR / "30Q综合评估提示词.md"
POST_EVAL_TIMEOUT_SECONDS = 2 * 60 * 60

SCALES = {
    "PHQ-9": {
        "question_file": "PHQ-9-v2.jsonl",
        "scoring_prompt": "PHQ-9评估提示词.md",
    },
    "BDI-II": {
        "question_file": "BDI-II-v2.jsonl",
        "scoring_prompt": "BDI-II评估提示词.md",
    },
}


@dataclass
class RuntimeConfig:
    name: str
    resume: bool
    start: str
    step: int
    stride: int
    verbose: str
    log_file: str
    agent: str
    dry_run: bool
    run_simulation: bool
    run_merge: bool
    run_post_scale: bool
    run_30q: bool
    run_compress: bool
    run_agent_memory_vis: bool
    run_external_memory_audit: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one experiment with optional stages")
    parser.add_argument("--name", default=None, help="覆盖脚本前面的 RUN_NAME")
    parser.add_argument("--start", default=None, help="覆盖脚本前面的 START_TIME")
    parser.add_argument("--step", type=int, default=None, help="覆盖脚本前面的 STEP")
    parser.add_argument("--stride", type=int, default=None, help="覆盖脚本前面的 STRIDE")
    parser.add_argument("--verbose", default=None, help="覆盖脚本前面的 VERBOSE")
    parser.add_argument("--log", default=None, help="覆盖脚本前面的 LOG_FILE")
    parser.add_argument("--agent", default=None, help="覆盖脚本前面的 SCALE_AGENT")
    parser.add_argument("--resume", action="store_true", help="覆盖脚本前面的 RESUME_RUN=True")
    parser.add_argument("--dry-run", action="store_true", help="覆盖脚本前面的 DRY_RUN=True")
    parser.add_argument("--skip-sim", action="store_true", help="跳过仿真阶段")
    parser.add_argument("--skip-merge", action="store_true", help="跳过 merge 阶段")
    parser.add_argument("--skip-post-scale", action="store_true", help="跳过治疗后 PHQ-9/BDI-II 评估")
    parser.add_argument("--run-30q", action="store_true", help="显式开启治疗后 30Q 评估")
    parser.add_argument("--skip-compress", action="store_true", help="跳过 compress.py")
    parser.add_argument("--run-agent-memory-vis", action="store_true", help="显式开启角色记忆可视化")
    parser.add_argument("--skip-agent-memory-vis", action="store_true", help="跳过角色记忆可视化")
    parser.add_argument("--run-external-memory-audit", action="store_true", help="显式开启外置记忆审计可视化")
    parser.add_argument("--skip-external-memory-audit", action="store_true", help="跳过外置记忆审计可视化")
    return parser.parse_args()


def generate_run_name() -> str:
    return f"sim-one-{datetime.now().strftime('%m%d-%H%M')}"


def resolve_runtime_config(args: argparse.Namespace) -> RuntimeConfig:
    name = (args.name if args.name is not None else RUN_NAME).strip()
    resume = bool(RESUME_RUN or args.resume)

    if not name:
        if resume:
            raise ValueError("RESUME_RUN=True 时必须提供 RUN_NAME 或 --name，不能留空。")
        name = generate_run_name()

    return RuntimeConfig(
        name=name,
        resume=resume,
        start=args.start if args.start is not None else START_TIME,
        step=args.step if args.step is not None else STEP,
        stride=args.stride if args.stride is not None else STRIDE,
        verbose=args.verbose if args.verbose is not None else VERBOSE,
        log_file=args.log if args.log is not None else LOG_FILE,
        agent=args.agent if args.agent is not None else SCALE_AGENT,
        dry_run=bool(DRY_RUN or args.dry_run),
        run_simulation=bool(RUN_SIMULATION and not args.skip_sim),
        run_merge=bool(RUN_MERGE and not args.skip_merge),
        run_post_scale=bool(RUN_POST_SCALE and not args.skip_post_scale),
        run_30q=bool(RUN_30Q or args.run_30q),
        run_compress=bool(RUN_COMPRESS and not args.skip_compress),
        run_agent_memory_vis=bool((RUN_AGENT_MEMORY_VIS or args.run_agent_memory_vis) and not args.skip_agent_memory_vis),
        run_external_memory_audit=bool((RUN_EXTERNAL_MEMORY_AUDIT or args.run_external_memory_audit) and not args.skip_external_memory_audit),
    )


def run_cmd(cmd: list[str], *, dry_run: bool, timeout: Optional[int] = None) -> None:
    print(f"[RUN] {' '.join(cmd)}")
    if dry_run:
        return
    subprocess.run(cmd, cwd=BASE_DIR, check=True, timeout=timeout)


def detect_assets_root(name: str) -> str:
    """从 checkpoint 的首个 simulate-*.json 推断回放所需的 assets 子目录。

    存档里嵌入了运行时配置，``maze.path`` 含 ``counsel_room`` 即咨询室（6×7），
    否则按村庄（50×50）处理。读不到任何存档时安全回退到 ``village``。
    用于给 compress.py 显式传递 ``--assets-root``。
    """
    checkpoint_dir = CHECKPOINTS_ROOT / name
    if not checkpoint_dir.is_dir():
        return "village"
    try:
        names = sorted(
            p for p in checkpoint_dir.iterdir()
            if p.name.startswith("simulate-") and p.name.endswith(".json")
        )
    except OSError:
        return "village"
    for path in names:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return "village"
        maze_field = data.get("maze")
        maze_path = ""
        if isinstance(maze_field, dict):
            maze_path = str(maze_field.get("path", "") or "")
        return "counsel_room" if "counsel_room" in maze_path else "village"
    return "village"


def ensure_checkpoint_state(name: str, *, resume: bool, dry_run: bool) -> None:
    if dry_run:
        return

    checkpoint_dir = CHECKPOINTS_ROOT / name
    if resume:
        if not checkpoint_dir.is_dir():
            raise FileNotFoundError(f"resume 目标不存在: {checkpoint_dir}")
    else:
        if checkpoint_dir.exists():
            raise FileExistsError(
                f"实验目录已存在: {checkpoint_dir}\n"
                f"请更换 RUN_NAME / --name，或开启 RESUME_RUN / --resume。"
            )


def ensure_experiment_dirs(name: str, *, dry_run: bool) -> Path:
    output_dir = EXPERIMENT_DATA_ROOT / name
    traces_dir = output_dir / "traces"
    scales_dir = output_dir / "scales"
    if dry_run:
        print(f"[DRY-RUN] ensure dirs: {output_dir} / {traces_dir} / {scales_dir}")
        return output_dir
    traces_dir.mkdir(parents=True, exist_ok=True)
    scales_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def find_last_snapshot_name(name: str) -> str:
    checkpoint_dir = CHECKPOINTS_ROOT / name
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"checkpoint directory not found: {checkpoint_dir}")
    snapshots = sorted(
        p.name for p in checkpoint_dir.iterdir()
        if p.name.startswith("simulate-") and p.name.endswith(".json")
    )
    if not snapshots:
        raise FileNotFoundError(f"no simulate-*.json found under: {checkpoint_dir}")
    return snapshots[-1]


def collect_core_outputs(name: str, *, dry_run: bool) -> None:
    checkpoint_dir = CHECKPOINTS_ROOT / name
    output_dir = ensure_experiment_dirs(name, dry_run=dry_run)
    traces_dir = output_dir / "traces"
    scales_dir = output_dir / "scales"

    judge_src = checkpoint_dir / "judge_traces" / "judge_conversation.json"
    if judge_src.exists():
        dst = traces_dir / "judge_conversation.json"
        print(f"[COLLECT] {judge_src} -> {dst}")
        if not dry_run:
            shutil.copy2(judge_src, dst)

    forced_src = checkpoint_dir / "forced_prompt_traces"
    if forced_src.is_dir():
        dst = traces_dir / "forced_prompt_traces"
        print(f"[COLLECT] {forced_src} -> {dst}")
        if not dry_run:
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(forced_src, dst)

    conv_src = checkpoint_dir / "conversation.json"
    if conv_src.exists():
        dst = output_dir / "conversation.json"
        print(f"[COLLECT] {conv_src} -> {dst}")
        if not dry_run:
            shutil.copy2(conv_src, dst)

    staged_src = checkpoint_dir / "staged_eval"
    if staged_src.is_dir():
        dst = scales_dir / "staged"
        print(f"[COLLECT] {staged_src} -> {dst}")
        if not dry_run:
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(staged_src, dst)

    merge_dst = traces_dir / "merge_consultation_dialogues.json"
    if merge_dst.exists():
        print(f"[CHECK] merge 结果已存在: {merge_dst}")

    meta = {
        "trial_name": name,
        "mode": "single_run",
        "resume": False,
        "timestamp_source": "manual_or_scripted",
    }
    meta_dst = output_dir / "trial_meta.json"
    print(f"[WRITE] {meta_dst}")
    if not dry_run:
        with meta_dst.open("w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)


def run_post_scales(name: str, agent_name: str, *, dry_run: bool) -> None:
    output_dir = ensure_experiment_dirs(name, dry_run=dry_run)
    scales_dir = output_dir / "scales"
    snapshot_name = "<latest-snapshot>" if dry_run else find_last_snapshot_name(name)
    aggregated: dict[str, dict] = {"post": {}}

    for scale_name, cfg in SCALES.items():
        scoring_prompt = SCALE_SCORING_DIR / cfg["scoring_prompt"]
        answers_path = scales_dir / f"{scale_name}_post_answered.jsonl"
        scored_path = scales_dir / f"{scale_name}_post_scored.json"

        run_cmd(
            [
                WORKER_PYTHON,
                str(SCALE_WORKER_SCRIPT),
                "--cp-name", name,
                "--snapshot", snapshot_name,
                "--agent", agent_name,
                "--question-file", cfg["question_file"],
                "--output", str(answers_path),
            ],
            dry_run=dry_run,
            timeout=POST_EVAL_TIMEOUT_SECONDS,
        )

        run_cmd(
            [
                WORKER_PYTHON,
                str(SCORE_WORKER_SCRIPT),
                "--answers", str(answers_path),
                "--scoring-prompt", str(scoring_prompt),
                "--output", str(scored_path),
            ],
            dry_run=dry_run,
            timeout=POST_EVAL_TIMEOUT_SECONDS,
        )

        if not dry_run and scored_path.exists():
            with scored_path.open("r", encoding="utf-8") as f:
                aggregated["post"][scale_name] = json.load(f)

    aggregated_path = scales_dir / "scale_scores.json"
    print(f"[WRITE] {aggregated_path}")
    if not dry_run:
        with aggregated_path.open("w", encoding="utf-8") as f:
            json.dump(aggregated, f, ensure_ascii=False, indent=2)


def run_post_30q(name: str, agent_name: str, *, dry_run: bool) -> None:
    if not THIRTY_Q_PROMPT.exists():
        raise FileNotFoundError(f"30Q scoring prompt not found: {THIRTY_Q_PROMPT}")

    output_dir = ensure_experiment_dirs(name, dry_run=dry_run)
    scales_dir = output_dir / "scales"
    snapshot_name = "<latest-snapshot>" if dry_run else find_last_snapshot_name(name)
    answers_path = scales_dir / "30Q_post_answered.jsonl"
    scored_path = scales_dir / "30Q_post_scored.json"

    run_cmd(
        [
            WORKER_PYTHON,
            str(SCALE_WORKER_SCRIPT),
            "--cp-name", name,
            "--snapshot", snapshot_name,
            "--agent", agent_name,
            "--question-file", "30Q.jsonl",
            "--output", str(answers_path),
        ],
        dry_run=dry_run,
        timeout=1800,
    )

    run_cmd(
        [
            WORKER_PYTHON,
            str(SCORE_WORKER_SCRIPT),
            "--answers", str(answers_path),
            "--scoring-prompt", str(THIRTY_Q_PROMPT),
            "--output", str(scored_path),
        ],
        dry_run=dry_run,
        timeout=300,
    )


def run_agent_memory_visualization(name: str, agent_name: str, *, dry_run: bool) -> None:
    output_dir = ensure_experiment_dirs(name, dry_run=dry_run) / "visualizations" / "agent_memory"
    run_cmd(
        [
            sys.executable,
            str(AGENT_MEMORY_VIS_SCRIPT),
            "--cp-name", name,
            "--agent", agent_name,
            "--output-dir", str(output_dir),
        ],
        dry_run=dry_run,
        timeout=1800,
    )


def run_external_memory_audit(name: str, agent_name: str, *, dry_run: bool) -> None:
    output_root = ensure_experiment_dirs(name, dry_run=dry_run) / "visualizations" / "external_memory_audit"
    run_cmd(
        [
            sys.executable,
            str(EXTERNAL_MEMORY_AUDIT_SCRIPT),
            "--cp-name", name,
            "--agent", agent_name,
            "--output-root", str(output_root),
        ],
        dry_run=dry_run,
        timeout=1800,
    )


def print_effective_config(cfg: RuntimeConfig) -> None:
    print("==========================================")
    print(" 单次实验配置")
    print("==========================================")
    print(f"  名称:         {cfg.name}")
    print(f"  续跑:         {cfg.resume}")
    print(f"  起始时间:     {cfg.start}")
    print(f"  步数:         {cfg.step}")
    print(f"  步间隔:       {cfg.stride}")
    print(f"  verbose:      {cfg.verbose}")
    print(f"  log:          {cfg.log_file or '(空)'}")
    print(f"  评估角色:     {cfg.agent}")
    print(f"  跑仿真:       {cfg.run_simulation}")
    print(f"  跑 merge:     {cfg.run_merge}")
    print(f"  跑治疗后量表: {cfg.run_post_scale}")
    print(f"  跑 30Q:       {cfg.run_30q}")
    print(f"  跑 compress:  {cfg.run_compress}")
    print(f"  跑记忆可视化: {cfg.run_agent_memory_vis}")
    print(f"  跑外置审计:   {cfg.run_external_memory_audit}")
    print(f"  dry-run:      {cfg.dry_run}")
    print("==========================================")


def main() -> None:
    args = parse_args()
    cfg = resolve_runtime_config(args)
    print_effective_config(cfg)

    if cfg.run_simulation:
        ensure_checkpoint_state(cfg.name, resume=cfg.resume, dry_run=cfg.dry_run)
        cmd = [
            sys.executable,
            str(START_SCRIPT),
            "--name", cfg.name,
            "--step", str(cfg.step),
            "--stride", str(cfg.stride),
            "--verbose", cfg.verbose,
        ]
        if cfg.resume:
            cmd.append("--resume")
        else:
            cmd.extend(["--start", cfg.start])
        if cfg.log_file:
            cmd.extend(["--log", cfg.log_file])
        run_cmd(cmd, dry_run=cfg.dry_run, timeout=3600)

    if cfg.run_merge:
        run_cmd([sys.executable, str(MERGE_SCRIPT), cfg.name], dry_run=cfg.dry_run, timeout=300)

    collect_core_outputs(cfg.name, dry_run=cfg.dry_run)

    if cfg.run_post_scale:
        run_post_scales(cfg.name, cfg.agent, dry_run=cfg.dry_run)

    if cfg.run_30q:
        run_post_30q(cfg.name, cfg.agent, dry_run=cfg.dry_run)

    if cfg.run_compress:
        assets_root = detect_assets_root(cfg.name)
        run_cmd(
            [sys.executable, str(COMPRESS_SCRIPT), "--name", cfg.name, "--assets-root", assets_root],
            dry_run=cfg.dry_run,
            timeout=1800,
        )

    if cfg.run_agent_memory_vis:
        run_agent_memory_visualization(cfg.name, cfg.agent, dry_run=cfg.dry_run)

    if cfg.run_external_memory_audit:
        run_external_memory_audit(cfg.name, cfg.agent, dry_run=cfg.dry_run)

    print("\n[Done] 单次实验流程完成")
    print(f"- 查看状态: bash runshells/sim_status.sh {cfg.name}")
    print(f"- 启动回放: bash runshells/sim_replay.sh {cfg.name}")


if __name__ == "__main__":
    main()
