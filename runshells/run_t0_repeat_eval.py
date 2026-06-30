#!/usr/bin/env python3
"""卡布达变体 T0 重复量表评估脚本。

脚本用途：
- 针对 frontend/static/assets/village/agents 下的“卡布达/卡布达2...卡布达9”人设，
  按 mild / moderate / severe 三种抑郁配置生成初始 T0 checkpoint。
- 对每个 T0 checkpoint 重复执行 PHQ-9 / BDI-II 初始评分，用于观察同一人设、
  同一严重程度下的量表稳定性。
- 使用多进程并行调度不同 persona × severity × repeat 任务，避免完全串行太慢。
- 输出 answered/scored 原始文件、聚合 JSON，以及分类对比 Markdown 报告。

直接运行：
    python3 runshells/run_t0_repeat_eval.py

选择条件示例：
    python3 runshells/run_t0_repeat_eval.py --condition T0-KBD2-MILD --repeat 3 --max-parallel 2
    python3 runshells/run_t0_repeat_eval.py --condition T0-KBD9-ALL --condition T0-ALL-SEV
    python3 runshells/run_t0_repeat_eval.py --persona 卡布达2 --severity MOD
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
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from kabuda_variant_runtime import (
    RUNTIME_AGENT_NAME,
    SEVERITY_CONFIG_NAMES,
    VARIANT_SELECTOR_ALIASES,
    VARIANT_SHORT_NAMES,
    VARIANT_SOURCE_AGENT_NAMES,
    VARIANTS,
    prepare_kabuda_variant_runtime,
)


# ============================================================
# ↓↓↓ 可调参数：直接修改这里即可（中文注释）↓↓↓
# ============================================================

# 批次名称；留空则自动生成，例如 t0-repeat-0630-1530
BATCH_NAME = ""

# 初始 T0 时间；只用于构造初始 checkpoint，不推进仿真
START_TIME = "20260614-09:30"

# checkpoint 中保留的 stride 字段，和正式实验默认保持一致
STRIDE = 720

# 默认评估人设；支持：卡布达、卡布达2...卡布达9、KBD1...KBD9、ALL
PERSONA_SELECTOR = "ALL"

# 默认严重程度；支持：MILD、MOD、SEV、ALL
SEVERITY_SELECTOR = "ALL"

# 每个 persona × severity 重复 T0 评估次数
REPEAT = 3

# 并行执行的 T0 任务数；每个任务内部会依次跑各个量表
MAX_PARALLEL = 2

# 是否跳过已有完整 scored 文件的重复项
SKIP_EXISTING = True

# 是否覆盖已有输出；为 True 时会重新写 T0 checkpoint 和评分文件
FORCE = False

# 是否只打印计划，不实际创建 checkpoint / 调用量表 worker
DRY_RUN = False

# 量表配置；默认与 run_one_experiment.py 的 PHQ-9 / BDI-II 保持一致
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

# ============================================================
# ↑↑↑ 可调参数：直接修改这里即可（中文注释）↑↑↑
# ============================================================


BASE_DIR = Path(__file__).resolve().parents[1]
GLOBAL_CONFIG = BASE_DIR / "data" / "config.json"
STATIC_ROOT = BASE_DIR / "frontend" / "static"
CHECKPOINTS_ROOT = BASE_DIR / "results" / "checkpoints"
EXPERIMENT_DATA_ROOT = BASE_DIR / "results" / "experiment_data"
T0_OUTPUT_ROOT = EXPERIMENT_DATA_ROOT / "t0_repeat_eval"
SCALE_AGENT_DIR = BASE_DIR / "customization" / "depression_scale_agent"
SCALE_QUESTIONS_DIR = SCALE_AGENT_DIR / "questions" / "templates"
SCALE_SCORING_DIR = SCALE_AGENT_DIR / "questions" / "scoring_prompts"
SCALE_WORKER_SCRIPT = BASE_DIR / "runshells" / "run_scale_worker.py"
SCORE_WORKER_SCRIPT = BASE_DIR / "runshells" / "run_score_worker.py"

PERSONAS = [
    "卡布达",
    "金龟次郎",
    "田德莉娜",
    "呱呱蛙",
    "蜻蜓队长",
    "蟑螂恶霸",
]

SEVERITY_SHORT_NAMES = {
    "mild": "MILD",
    "moderate": "MOD",
    "severe": "SEV",
}
SEVERITY_SELECTOR_ALIASES = {
    "MILD": "mild",
    "MOD": "moderate",
    "MODERATE": "moderate",
    "SEV": "severe",
    "SEVERE": "severe",
}
WILDCARD_TOKENS = {"*", "ALL"}

PERSONA_SELECTOR_ALIASES = dict(VARIANT_SELECTOR_ALIASES)
for variant, source_name in VARIANT_SOURCE_AGENT_NAMES.items():
    PERSONA_SELECTOR_ALIASES[source_name.upper()] = variant
    PERSONA_SELECTOR_ALIASES[source_name] = variant
PERSONA_SELECTOR_ALIASES["卡布达1"] = "kbd1"


@dataclass(frozen=True)
class T0Condition:
    name: str
    variant: str
    severity: str


@dataclass(frozen=True)
class T0Task:
    condition: T0Condition
    repeat_idx: int


@dataclass
class RuntimeConfig:
    batch_name: str
    start_time: str
    stride: int
    repeat: int
    max_parallel: int
    skip_existing: bool
    force: bool
    dry_run: bool
    conditions: list[T0Condition]


ALL_CONDITIONS = [
    T0Condition(
        name=f"T0-{VARIANT_SHORT_NAMES[variant]}-{SEVERITY_SHORT_NAMES[severity]}",
        variant=variant,
        severity=severity,
    )
    for variant in VARIANTS
    for severity in SEVERITY_CONFIG_NAMES
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run repeatable T0 scale evaluations for Kabuda variants")
    parser.add_argument("--name", default=None, help="覆盖脚本顶部 BATCH_NAME")
    parser.add_argument("--start", default=None, help="覆盖脚本顶部 START_TIME")
    parser.add_argument("--stride", type=int, default=None, help="覆盖脚本顶部 STRIDE")
    parser.add_argument("--persona", default=None, help="人设选择器：卡布达2 / KBD2 / ALL")
    parser.add_argument("--severity", default=None, help="严重程度选择器：MILD / MOD / SEV / ALL")
    parser.add_argument(
        "--condition",
        action="append",
        default=None,
        help="条件选择器，可重复传入；支持 T0-KBD2-MILD、T0-KBD9-ALL、T0-ALL-SEV",
    )
    parser.add_argument("--repeat", type=int, default=None, help="覆盖脚本顶部 REPEAT")
    parser.add_argument("--max-parallel", type=int, default=None, help="覆盖脚本顶部 MAX_PARALLEL")
    parser.add_argument("--skip-existing", dest="skip_existing", action="store_true", default=None, help="跳过已有完整结果")
    parser.add_argument("--no-skip-existing", dest="skip_existing", action="store_false", help="不跳过已有完整结果")
    parser.add_argument("--force", action="store_true", help="覆盖已有 checkpoint 和评分结果")
    parser.add_argument("--report-only", action="store_true", help="只汇总已有 T0 结果并生成报告，不调用量表/评分 worker")
    parser.add_argument("--results-dir", default=None, help="已有结果目录，可指向 results/T0-KBD-1-6 或具体 t0-repeat-* 批次目录")
    parser.add_argument("--dry-run", action="store_true", help="只打印计划，不实际执行")
    return parser.parse_args()


def generate_batch_name() -> str:
    return f"t0-repeat-{datetime.now().strftime('%m%d-%H%M')}"


def normalize_selector(value: str | None) -> str:
    return str(value or "").strip()


def resolve_persona_selector(selector: str | None) -> list[str]:
    raw = normalize_selector(selector)
    if not raw or raw.upper() in WILDCARD_TOKENS:
        return list(VARIANTS)
    if raw in PERSONA_SELECTOR_ALIASES:
        return [PERSONA_SELECTOR_ALIASES[raw]]
    upper = raw.upper()
    if upper in PERSONA_SELECTOR_ALIASES:
        return [PERSONA_SELECTOR_ALIASES[upper]]
    available = ", ".join(["ALL", *VARIANT_SHORT_NAMES.values(), *VARIANT_SOURCE_AGENT_NAMES.values()])
    raise ValueError(f"未知人设选择器: {selector}; 可用: {available}")


def resolve_severity_selector(selector: str | None) -> list[str]:
    raw = normalize_selector(selector)
    if not raw or raw.upper() in WILDCARD_TOKENS:
        return list(SEVERITY_CONFIG_NAMES.keys())
    upper = raw.upper()
    if upper in SEVERITY_SELECTOR_ALIASES:
        return [SEVERITY_SELECTOR_ALIASES[upper]]
    available = ", ".join(["ALL", *SEVERITY_SELECTOR_ALIASES.keys()])
    raise ValueError(f"未知严重程度选择器: {selector}; 可用: {available}")


def resolve_condition_selector(selector: str | None) -> list[T0Condition]:
    raw = normalize_selector(selector)
    if not raw:
        return list(ALL_CONDITIONS)

    exact = [condition for condition in ALL_CONDITIONS if condition.name == raw]
    if exact:
        return exact

    tokens = [token.strip() for token in raw.split("-")]
    upper_tokens = [token.upper() for token in tokens]
    if len(tokens) == 3 and upper_tokens[0] == "T0":
        variants = resolve_persona_selector(tokens[1])
        severities = resolve_severity_selector(tokens[2])
        return [
            condition
            for condition in ALL_CONDITIONS
            if condition.variant in variants and condition.severity in severities
        ]

    examples = "T0-KBD2-MILD, T0-KBD9-ALL, T0-ALL-SEV"
    available = ", ".join(condition.name for condition in ALL_CONDITIONS)
    raise ValueError(f"未找到条件: {selector}\n可用精确条件: {available}\n也支持选择器: {examples}")


def resolve_conditions(args: argparse.Namespace) -> list[T0Condition]:
    if args.condition:
        conditions: list[T0Condition] = []
        seen = set()
        for selector in args.condition:
            for condition in resolve_condition_selector(selector):
                if condition.name in seen:
                    continue
                seen.add(condition.name)
                conditions.append(condition)
        return conditions

    persona_selector = args.persona if args.persona is not None else PERSONA_SELECTOR
    severity_selector = args.severity if args.severity is not None else SEVERITY_SELECTOR
    variants = set(resolve_persona_selector(persona_selector))
    severities = set(resolve_severity_selector(severity_selector))
    return [
        condition
        for condition in ALL_CONDITIONS
        if condition.variant in variants and condition.severity in severities
    ]


def resolve_runtime_config(args: argparse.Namespace) -> RuntimeConfig:
    batch_name = normalize_selector(args.name if args.name is not None else BATCH_NAME)
    if not batch_name:
        batch_name = generate_batch_name()
    repeat = max(1, int(args.repeat if args.repeat is not None else REPEAT))
    skip_existing = SKIP_EXISTING if args.skip_existing is None else bool(args.skip_existing)
    return RuntimeConfig(
        batch_name=batch_name,
        start_time=normalize_selector(args.start if args.start is not None else START_TIME),
        stride=max(1, int(args.stride if args.stride is not None else STRIDE)),
        repeat=repeat,
        max_parallel=max(1, int(args.max_parallel if args.max_parallel is not None else MAX_PARALLEL)),
        skip_existing=bool(skip_existing),
        force=bool(FORCE or args.force),
        dry_run=bool(DRY_RUN or args.dry_run),
        conditions=resolve_conditions(args),
    )


def print_effective_config(cfg: RuntimeConfig) -> None:
    print("==========================================")
    print(" 卡布达 T0 重复评估配置")
    print("==========================================")
    print(f"  批次名称:     {cfg.batch_name}")
    print(f"  T0 时间:      {cfg.start_time}")
    print(f"  stride:       {cfg.stride}")
    print(f"  条件数:       {len(cfg.conditions)}")
    print(f"  重复次数:     {cfg.repeat}")
    print(f"  并行度:       {cfg.max_parallel}")
    print(f"  跳过已有:     {cfg.skip_existing}")
    print(f"  覆盖重跑:     {cfg.force}")
    print(f"  dry-run:      {cfg.dry_run}")
    print("  条件列表:     " + ", ".join(condition.name for condition in cfg.conditions))
    print("  量表:         " + ", ".join(SCALES.keys()))
    print("==========================================")


def load_json_file(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON payload must be object: {path}")
    return payload


def write_json_file(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def run_cmd(cmd: list[str], *, timeout: int = 1800) -> None:
    print("[RUN] " + " ".join(str(item) for item in cmd))
    subprocess.run(cmd, cwd=BASE_DIR, check=True, timeout=timeout)


def safe_slug(value: str) -> str:
    return "".join(ch if (ch.isalnum() or ch in {"-", "_"}) else "_" for ch in value).strip("_") or "item"


def start_time_to_snapshot_name(start_time: str) -> str:
    safe = start_time.replace(":", "")
    return f"simulate-{safe}.json"


def runtime_persona_dir(cfg: RuntimeConfig, task: T0Task) -> Path:
    return (
        T0_OUTPUT_ROOT
        / cfg.batch_name
        / "runtime_personas"
        / safe_slug(task.condition.name)
        / f"r{task.repeat_idx:02d}"
    )


def condition_output_dir(cfg: RuntimeConfig, condition: T0Condition) -> Path:
    return T0_OUTPUT_ROOT / cfg.batch_name / condition.name


def repeat_output_dir(cfg: RuntimeConfig, task: T0Task) -> Path:
    return condition_output_dir(cfg, task.condition) / f"r{task.repeat_idx:02d}"


def build_run_name(cfg: RuntimeConfig, task: T0Task) -> str:
    return f"{cfg.batch_name}-{task.condition.name}-r{task.repeat_idx:02d}"


def build_runtime_config_payload(cfg: RuntimeConfig, task: T0Task) -> dict[str, Any]:
    global_cfg = load_json_file(GLOBAL_CONFIG)
    variant_runtime = prepare_kabuda_variant_runtime(
        base_dir=BASE_DIR,
        variant=task.condition.variant,
        severity=task.condition.severity,
        output_dir=runtime_persona_dir(cfg, task),
        dry_run=cfg.dry_run,
    )
    assets_root = "assets/village"
    payload: dict[str, Any] = {
        "stride": cfg.stride,
        "time": {"start": cfg.start_time},
        "maze": {"path": f"{assets_root}/maze.json"},
        "agent_base": copy.deepcopy(global_cfg.get("agent", {})),
        "agents": {},
    }
    for agent_name in PERSONAS:
        payload["agents"][agent_name] = {
            "config_path": f"{assets_root}/agents/{agent_name.replace(' ', '_')}/agent.json",
        }

    payload["agents"].setdefault(RUNTIME_AGENT_NAME, {})
    payload["agents"][RUNTIME_AGENT_NAME]["config_path"] = str(variant_runtime.agent_config_path)
    payload["agents"][RUNTIME_AGENT_NAME]["depression_config_path"] = str(variant_runtime.depression_config_path)

    intervention_cfg = copy.deepcopy(global_cfg.get("intervention", {}))
    if intervention_cfg:
        memory_injection_cfg = intervention_cfg.setdefault("memory_injection", {})
        if isinstance(memory_injection_cfg, dict):
            memory_injection_cfg["content_file"] = variant_runtime.memory_injection_config_path
        payload["intervention"] = intervention_cfg

    checkpointing_cfg = copy.deepcopy(global_cfg.get("checkpointing", {}))
    if checkpointing_cfg:
        payload["checkpointing"] = checkpointing_cfg

    return payload


def ensure_t0_checkpoint(cfg: RuntimeConfig, task: T0Task) -> tuple[str, str, Path]:
    run_name = build_run_name(cfg, task)
    checkpoint_dir = CHECKPOINTS_ROOT / run_name
    snapshot_name = start_time_to_snapshot_name(cfg.start_time)
    snapshot_path = checkpoint_dir / snapshot_name

    if cfg.force and checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)

    if cfg.dry_run:
        print(f"[DRY-RUN] write T0 checkpoint: {snapshot_path}")
        return run_name, snapshot_name, snapshot_path

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    runtime_config = build_runtime_config_payload(cfg, task)
    runtime_config["step"] = 0
    write_json_file(snapshot_path, runtime_config)
    conversation_path = checkpoint_dir / "conversation.json"
    if cfg.force or not conversation_path.exists():
        write_json_file(conversation_path, {})
    return run_name, snapshot_name, snapshot_path


def scale_paths(scales_dir: Path, scale_name: str) -> tuple[Path, Path]:
    stem = f"{scale_name}_T0"
    return scales_dir / f"{stem}_answered.jsonl", scales_dir / f"{stem}_scored.json"


def result_complete(scales_dir: Path) -> bool:
    for scale_name in SCALES:
        _answers_path, scored_path = scale_paths(scales_dir, scale_name)
        if not scored_path.is_file():
            return False
    return True


def resolve_scale_question_file(scale_cfg: dict[str, Any]) -> str:
    raw = str(scale_cfg["question_file"])
    path = Path(raw)
    if path.is_absolute():
        return str(path)
    repo_path = BASE_DIR / path
    if repo_path.exists():
        return str(repo_path)
    question_path = SCALE_QUESTIONS_DIR / raw
    if not question_path.is_file():
        raise FileNotFoundError(f"题目文件不存在: {question_path}")
    return question_path.name


def resolve_scoring_prompt(scale_cfg: dict[str, Any]) -> Path:
    raw = str(scale_cfg["scoring_prompt"])
    path = Path(raw).expanduser()
    if path.is_absolute():
        return path
    repo_path = BASE_DIR / path
    if repo_path.exists():
        return repo_path
    return SCALE_SCORING_DIR / raw


def load_optional_scored(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = load_json_file(path)
    except (OSError, json.JSONDecodeError, ValueError):
        return None
    return payload


def extract_total(scored: dict[str, Any] | None) -> float | None:
    if not isinstance(scored, dict):
        return None
    for key in ("total_score", "total_score_raw", "standard_score"):
        value = scored.get(key)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def extract_severity(scored: dict[str, Any] | None) -> str:
    if not isinstance(scored, dict):
        return "—"
    for key in ("severity", "severity_by_index", "severity_by_standard_score"):
        value = scored.get(key)
        if value:
            return str(value)
    return "—"


def extract_safety(scored: dict[str, Any] | None) -> str:
    if not isinstance(scored, dict):
        return "—"
    value = scored.get("safety_risk")
    return str(value) if value else "—"


SCALE_ITEM_SCORE_KEYS = {
    "PHQ-9": "phq9_scores",
    "BDI-II": "bdi_ii_scores",
}

CN_SCORE_VALUES = {
    "0": 0,
    "1": 1,
    "2": 2,
    "3": 3,
    "零": 0,
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
}


def load_jsonl_file(path: Path) -> list[dict[str, Any]]:
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


def extract_item_scores(scored_result: dict[str, Any] | None, scale_name: str) -> list[int] | None:
    item_key = SCALE_ITEM_SCORE_KEYS.get(scale_name)
    if not item_key or not isinstance(scored_result, dict):
        return None
    items = scored_result.get(item_key)
    if not isinstance(items, list):
        return None

    scores: list[int] = []
    for item in items:
        score = item.get("score") if isinstance(item, dict) else None
        if not isinstance(score, int) or score < 0 or score > 3:
            return None
        scores.append(score)
    return scores


def has_ambiguous_score_context(text: str, start: int, end: int) -> bool:
    context = text[max(0, start - 18) : min(len(text), end + 28)]
    return any(
        marker in context
        for marker in [
            "或者",
            "不确定",
            "之间",
            "两三",
            "一两",
            "也可能",
            "选0感觉是骗人的",
        ]
    )


def extract_direct_answer_score(answer: str) -> tuple[int | None, str]:
    patterns = [
        r"(?:我)?\s*(?:会|想|大概|可能|应该|还是|就|其实)?\s*(?:选|选择)\s*(?:了)?\s*([0-3零一二两三])",
        r"([0-3零一二两三])\s*分",
        r"(?:评分的话|打分的话|大概|应该|可能|算是|算|就是|我觉得|我想|我会|应该是|大概是|可能是|差不多)\s*[，,。\.……\s]*([0-3零一二两三])\s*(?:吧|。|，|,|$)",
        r"(?:^|[，,。\.……\s])([0-3零一二两三])\s*吧",
        r"^[\s（\(\）\)……。,.，、嗯唔]*([0-3零一二两三])\s*(?:吧|。|，|,|$)",
    ]
    hits = []
    for pattern in patterns:
        for match in re.finditer(pattern, str(answer or "")):
            context = answer[max(0, match.start() - 6) : match.end() + 6]
            if any(marker in context for marker in ["不想选", "不是选", "不能选", "不敢选"]):
                continue
            score = CN_SCORE_VALUES.get(match.group(1))
            if score is None:
                continue
            hits.append((match.start(), match.end(), score))
    if not hits:
        return None, "none"

    hits.sort(key=lambda item: (item[0], item[1]))
    first_start, first_end, first_score = hits[0]
    if has_ambiguous_score_context(answer, first_start, first_end):
        return first_score, "ambiguous"
    for start, _end, score in hits[1:]:
        if start - first_end < 35 and score != first_score:
            return first_score, "ambiguous"
    return first_score, "direct"


def scale_file_path(result: dict[str, Any], scale_name: str, file_key: str) -> Path:
    scale_payload = result.get("scales", {}).get(scale_name, {})
    if isinstance(scale_payload, dict):
        raw_path = str(scale_payload.get(file_key, "") or "").strip()
        if raw_path:
            return Path(raw_path)
    return Path("")


def scored_payload_for_validation(result: dict[str, Any], scale_name: str) -> dict[str, Any] | None:
    scale_payload = result.get("scales", {}).get(scale_name, {})
    if isinstance(scale_payload, dict):
        scored = scale_payload.get("score")
        if isinstance(scored, dict) and scored:
            return scored
        scored_path = scale_file_path(result, scale_name, "scored_file")
        if scored_path.is_file():
            return load_optional_scored(scored_path)
    return None


def build_corrected_scale_records(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for result in results:
        if result.get("status") not in {"completed", "skipped_existing", "dry_run"}:
            continue
        repeat_value = result.get("repeat")
        repeat_label = f"r{int(repeat_value or 0):02d}" if repeat_value else "—"

        for scale_name in SCALES:
            scored_result = scored_payload_for_validation(result, scale_name)
            if not scored_result:
                continue
            reported_total = extract_total(scored_result)
            item_scores = extract_item_scores(scored_result, scale_name)
            corrected_scores = list(item_scores) if item_scores is not None else []
            override_count = 0
            direct_answer_count = 0

            answer_path = scale_file_path(result, scale_name, "answers_file")
            if item_scores is not None and answer_path.is_file():
                try:
                    answers = load_jsonl_file(answer_path)
                except (OSError, json.JSONDecodeError):
                    answers = []
                for answer_row in answers:
                    item_id = answer_row.get("id")
                    if not isinstance(item_id, int):
                        continue
                    if scale_name == "PHQ-9" and item_id == 10:
                        continue
                    if item_id < 1 or item_id > len(corrected_scores):
                        continue
                    answer_score, status = extract_direct_answer_score(str(answer_row.get("answer", "") or ""))
                    if status != "direct" or answer_score is None:
                        continue
                    direct_answer_count += 1
                    if corrected_scores[item_id - 1] != answer_score:
                        corrected_scores[item_id - 1] = int(answer_score)
                        override_count += 1

            item_sum = sum(item_scores) if item_scores is not None else None
            corrected_total = sum(corrected_scores) if corrected_scores else reported_total
            records.append(
                {
                    "condition_name": result.get("condition_name", ""),
                    "variant": result.get("variant", ""),
                    "variant_short_name": result.get("variant_short_name", ""),
                    "source_agent_name": result.get("source_agent_name", "—"),
                    "severity": result.get("severity", ""),
                    "severity_short_name": result.get("severity_short_name", "—"),
                    "repeat": result.get("repeat"),
                    "repeat_label": repeat_label,
                    "scale": scale_name,
                    "reported_total": reported_total,
                    "item_sum_total": float(item_sum) if item_sum is not None else None,
                    "corrected_total": float(corrected_total) if corrected_total is not None else None,
                    "direct_answer_count": direct_answer_count,
                    "override_count": override_count,
                    "scored_file": str(scale_file_path(result, scale_name, "scored_file") or ""),
                    "answers_file": str(answer_path or ""),
                }
            )
    return records


def build_scale_score_validation(results: list[dict[str, Any]]) -> dict[str, Any]:
    total_mismatches = []
    item_mismatches = []
    checked_scored_files = 0
    checked_answer_files = 0

    for result in results:
        if result.get("status") not in {"completed", "skipped_existing", "dry_run"}:
            continue
        repeat_label = f"r{int(result.get('repeat', 0) or 0):02d}" if result.get("repeat") else "—"
        for scale_name in SCALES:
            scored_result = scored_payload_for_validation(result, scale_name)
            if not scored_result:
                continue
            checked_scored_files += 1

            item_scores = extract_item_scores(scored_result, scale_name)
            reported_total = extract_total(scored_result)
            if item_scores is not None and reported_total is not None:
                item_sum = sum(item_scores)
                if float(item_sum) != float(reported_total):
                    total_mismatches.append(
                        {
                            "condition_name": result.get("condition_name", ""),
                            "repeat": repeat_label,
                            "scale": scale_name,
                            "item_sum": item_sum,
                            "reported_total": reported_total,
                            "delta": float(reported_total) - float(item_sum),
                            "scored_file": str(scale_file_path(result, scale_name, "scored_file") or ""),
                        }
                    )

            answer_path = scale_file_path(result, scale_name, "answers_file")
            if item_scores is None or not answer_path.is_file():
                continue
            checked_answer_files += 1
            try:
                answers = load_jsonl_file(answer_path)
            except (OSError, json.JSONDecodeError):
                continue

            for answer_row in answers:
                item_id = answer_row.get("id")
                if not isinstance(item_id, int):
                    continue
                if scale_name == "PHQ-9" and item_id == 10:
                    continue
                if item_id < 1 or item_id > len(item_scores):
                    continue
                answer_score, status = extract_direct_answer_score(str(answer_row.get("answer", "") or ""))
                if status != "direct" or answer_score is None:
                    continue
                llm_score = item_scores[item_id - 1]
                if answer_score != llm_score:
                    item_mismatches.append(
                        {
                            "condition_name": result.get("condition_name", ""),
                            "repeat": repeat_label,
                            "scale": scale_name,
                            "item_id": item_id,
                            "llm_score": llm_score,
                            "answer_score": answer_score,
                            "answer_file": str(answer_path),
                        }
                    )

    return {
        "checked_scored_files": checked_scored_files,
        "checked_answer_files": checked_answer_files,
        "total_mismatches": total_mismatches,
        "item_mismatches": item_mismatches,
        "corrected_scores": build_corrected_scale_records(results),
    }


def render_validation_number(value: Any) -> str:
    if value is None:
        return "—"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if number.is_integer():
        return str(int(number))
    return f"{number:.1f}"


def group_corrected_records(records: list[dict[str, Any]], *keys: str) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(tuple(record.get(key) for key in keys), []).append(record)
    return grouped


def corrected_record_total(record: dict[str, Any]) -> float | None:
    value = record.get("corrected_total")
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def render_corrected_comparison_markdown(validation_payload: dict[str, Any]) -> list[str]:
    records = validation_payload.get("corrected_scores", [])
    if not isinstance(records, list):
        records = []
    records = [record for record in records if isinstance(record, dict)]

    lines = []
    lines.append("## 复核后对比附录\n")
    lines.append("> 复核后总分计算规则：以 LLM 条目分为底稿；若回答文本中可明确抽取 0-3 分且与 LLM 条目分冲突，则用回答分覆盖该条目；最终总分使用复核后的条目分之和。\n")
    lines.append(f"- 复核后总分记录数：{len(records)}")
    lines.append(f"- 发生条目覆盖的记录数：{sum(1 for record in records if int(record.get('override_count', 0) or 0) > 0)}")
    lines.append("")

    lines.append("### 复核后重复稳定性对比\n")
    for scale_name in SCALES:
        scale_records = [record for record in records if record.get("scale") == scale_name]
        lines.extend(
            [
                f"#### {scale_name}",
                "",
                "| 人设 | 严重程度 | n | 均值 | 标准差 | 最小 | 最大 | 极差 | 覆盖条目数 |",
                "|---|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for (_variant, _severity), group in sorted(group_corrected_records(scale_records, "variant", "severity").items()):
            totals = [corrected_record_total(record) for record in group]
            valid = [float(value) for value in totals if value is not None]
            if not valid:
                continue
            override_count = sum(int(record.get("override_count", 0) or 0) for record in group)
            lines.append(
                "| {persona} | {severity} | {n} | {avg} | {sd} | {minv} | {maxv} | {rangev} | {override_count} |".format(
                    persona=group[0].get("source_agent_name", "—"),
                    severity=group[0].get("severity_short_name", "—"),
                    n=len(valid),
                    avg=fmt(mean(valid)),
                    sd=fmt(stddev(valid)),
                    minv=fmt(min(valid)),
                    maxv=fmt(max(valid)),
                    rangev=fmt(max(valid) - min(valid)),
                    override_count=override_count,
                )
            )
        lines.append("")

    lines.append("### 复核后按严重程度对比人设\n")
    for scale_name in SCALES:
        scale_records = [record for record in records if record.get("scale") == scale_name]
        lines.extend(
            [
                f"#### {scale_name}",
                "",
                "| 严重程度 | 人设 | n | 均值 | 标准差 | 覆盖条目数 |",
                "|---|---|---:|---:|---:|---:|",
            ]
        )
        rows = []
        for (_severity, _variant), group in group_corrected_records(scale_records, "severity", "variant").items():
            totals = [corrected_record_total(record) for record in group]
            valid = [float(value) for value in totals if value is not None]
            if not valid:
                continue
            rows.append(
                (
                    group[0].get("severity_short_name", "—"),
                    group[0].get("source_agent_name", "—"),
                    len(valid),
                    mean(valid),
                    stddev(valid),
                    sum(int(record.get("override_count", 0) or 0) for record in group),
                )
            )
        for severity, persona, n, avg, sd, override_count in sorted(rows):
            lines.append(f"| {severity} | {persona} | {n} | {fmt(avg)} | {fmt(sd)} | {override_count} |")
        lines.append("")

    return lines


def render_scale_score_validation_markdown(validation_payload: dict[str, Any]) -> list[str]:
    total_mismatches = validation_payload.get("total_mismatches", [])
    item_mismatches = validation_payload.get("item_mismatches", [])
    if not isinstance(total_mismatches, list):
        total_mismatches = []
    if not isinstance(item_mismatches, list):
        item_mismatches = []

    lines = []
    lines.append("## 评分结果校验附录\n")
    lines.append("> 仅做结构化复核：总分差异为“条目分之和 vs LLM 总分”；异常条目仅统计回答中可明确抽取 0-3 分、且与 LLM 条目分不同的情况。\n")
    lines.append(f"- 已复核 scored 文件数：{render_validation_number(validation_payload.get('checked_scored_files'))}")
    lines.append(f"- 已复核 answered 文件数：{render_validation_number(validation_payload.get('checked_answer_files'))}")
    lines.append(f"- 总分差异记录数：{len(total_mismatches)}")
    lines.append(f"- 异常评分条目数：{len(item_mismatches)}")
    lines.append("")

    if total_mismatches:
        lines.append("### 总分差异对比\n")
        lines.append("| 条件 | 重复 | 量表 | 条目和 | LLM总分 | 差值 |")
        lines.append("|------|------|------|--------|---------|------|")
        for item in total_mismatches:
            lines.append(
                "| {condition} | {repeat} | {scale} | {item_sum} | {reported_total} | {delta} |".format(
                    condition=item.get("condition_name", "—"),
                    repeat=item.get("repeat", "—"),
                    scale=item.get("scale", "—"),
                    item_sum=render_validation_number(item.get("item_sum")),
                    reported_total=render_validation_number(item.get("reported_total")),
                    delta=render_validation_number(item.get("delta")),
                )
            )
        lines.append("")

    if item_mismatches:
        lines.append("### 异常评分条目数值对比\n")
        lines.append("| 条件 | 重复 | 量表 | 条目 | LLM分 | 回答分 |")
        lines.append("|------|------|------|------|-------|--------|")
        for item in item_mismatches:
            lines.append(
                "| {condition} | {repeat} | {scale} | {item_id} | {llm_score} | {answer_score} |".format(
                    condition=item.get("condition_name", "—"),
                    repeat=item.get("repeat", "—"),
                    scale=item.get("scale", "—"),
                    item_id=render_validation_number(item.get("item_id")),
                    llm_score=render_validation_number(item.get("llm_score")),
                    answer_score=render_validation_number(item.get("answer_score")),
                )
            )
        lines.append("")

    lines.extend(render_corrected_comparison_markdown(validation_payload))

    return lines


def run_task(task_payload: dict[str, Any]) -> dict[str, Any]:
    cfg = RuntimeConfig(
        batch_name=task_payload["batch_name"],
        start_time=task_payload["start_time"],
        stride=int(task_payload["stride"]),
        repeat=int(task_payload["repeat"]),
        max_parallel=int(task_payload["max_parallel"]),
        skip_existing=bool(task_payload["skip_existing"]),
        force=bool(task_payload["force"]),
        dry_run=bool(task_payload["dry_run"]),
        conditions=[],
    )
    condition = T0Condition(**task_payload["condition"])
    task = T0Task(condition=condition, repeat_idx=int(task_payload["repeat_idx"]))
    started_at = time.time()
    scales_dir = repeat_output_dir(cfg, task) / "scales"
    metadata_path = repeat_output_dir(cfg, task) / "metadata.json"

    result: dict[str, Any] = {
        "condition_name": condition.name,
        "variant": condition.variant,
        "variant_short_name": VARIANT_SHORT_NAMES[condition.variant],
        "source_agent_name": VARIANT_SOURCE_AGENT_NAMES[condition.variant],
        "severity": condition.severity,
        "severity_short_name": SEVERITY_SHORT_NAMES[condition.severity],
        "repeat": task.repeat_idx,
        "status": "queued",
        "run_name": build_run_name(cfg, task),
        "scales_dir": str(scales_dir),
        "metadata_path": str(metadata_path),
        "scales": {},
        "error": "",
    }

    try:
        if cfg.skip_existing and not cfg.force and result_complete(scales_dir):
            result["status"] = "skipped_existing"
        else:
            run_name, snapshot_name, snapshot_path = ensure_t0_checkpoint(cfg, task)
            result["run_name"] = run_name
            result["snapshot"] = snapshot_name
            result["snapshot_path"] = str(snapshot_path)
            if not cfg.dry_run:
                scales_dir.mkdir(parents=True, exist_ok=True)

            for scale_name, scale_cfg in SCALES.items():
                answers_path, scored_path = scale_paths(scales_dir, scale_name)
                scoring_prompt = resolve_scoring_prompt(scale_cfg)
                result["scales"][scale_name] = {
                    "answers_file": str(answers_path),
                    "scored_file": str(scored_path),
                    "scoring_prompt": str(scoring_prompt),
                }
                if cfg.dry_run:
                    print(f"[DRY-RUN] {condition.name} r{task.repeat_idx:02d} {scale_name}")
                    continue
                if cfg.force:
                    for path in (answers_path, scored_path):
                        if path.exists():
                            path.unlink()
                if not answers_path.is_file():
                    run_cmd(
                        [
                            sys.executable,
                            str(SCALE_WORKER_SCRIPT),
                            "--cp-name",
                            run_name,
                            "--snapshot",
                            snapshot_name,
                            "--agent",
                            RUNTIME_AGENT_NAME,
                            "--question-file",
                            resolve_scale_question_file(scale_cfg),
                            "--output",
                            str(answers_path),
                        ],
                        timeout=3600,
                    )
                if not scored_path.is_file():
                    if not scoring_prompt.is_file():
                        raise FileNotFoundError(f"评分提示词不存在: {scoring_prompt}")
                    run_cmd(
                        [
                            sys.executable,
                            str(SCORE_WORKER_SCRIPT),
                            "--answers",
                            str(answers_path),
                            "--scoring-prompt",
                            str(scoring_prompt),
                            "--output",
                            str(scored_path),
                        ],
                        timeout=1800,
                    )

            result["status"] = "dry_run" if cfg.dry_run else "completed"

        if not cfg.dry_run:
            for scale_name in SCALES:
                _answers_path, scored_path = scale_paths(scales_dir, scale_name)
                scored = load_optional_scored(scored_path)
                result["scales"].setdefault(scale_name, {})
                result["scales"][scale_name].update(
                    {
                        "total": extract_total(scored),
                        "severity_label": extract_severity(scored),
                        "safety_risk": extract_safety(scored),
                        "score": scored or {},
                    }
                )
            metadata = copy.deepcopy(result)
            metadata["elapsed_seconds"] = round(time.time() - started_at, 3)
            write_json_file(metadata_path, metadata)
    except Exception as exc:
        result["status"] = "failed"
        result["error"] = str(exc)
    result["elapsed_seconds"] = round(time.time() - started_at, 3)
    return result


def task_to_payload(cfg: RuntimeConfig, task: T0Task) -> dict[str, Any]:
    return {
        "batch_name": cfg.batch_name,
        "start_time": cfg.start_time,
        "stride": cfg.stride,
        "repeat": cfg.repeat,
        "max_parallel": cfg.max_parallel,
        "skip_existing": cfg.skip_existing,
        "force": cfg.force,
        "dry_run": cfg.dry_run,
        "condition": {
            "name": task.condition.name,
            "variant": task.condition.variant,
            "severity": task.condition.severity,
        },
        "repeat_idx": task.repeat_idx,
    }


def run_all_tasks(cfg: RuntimeConfig) -> list[dict[str, Any]]:
    tasks = [
        T0Task(condition=condition, repeat_idx=repeat_idx)
        for condition in cfg.conditions
        for repeat_idx in range(1, cfg.repeat + 1)
    ]
    print(f"[PLAN] {len(tasks)} 个 T0 任务，max_parallel={cfg.max_parallel}")
    if cfg.max_parallel <= 1:
        return [run_task(task_to_payload(cfg, task)) for task in tasks]

    results: list[dict[str, Any]] = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=cfg.max_parallel) as executor:
        future_map = {
            executor.submit(run_task, task_to_payload(cfg, task)): task
            for task in tasks
        }
        for future in concurrent.futures.as_completed(future_map):
            task = future_map[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {
                    "condition_name": task.condition.name,
                    "variant": task.condition.variant,
                    "variant_short_name": VARIANT_SHORT_NAMES[task.condition.variant],
                    "source_agent_name": VARIANT_SOURCE_AGENT_NAMES[task.condition.variant],
                    "severity": task.condition.severity,
                    "severity_short_name": SEVERITY_SHORT_NAMES[task.condition.severity],
                    "repeat": task.repeat_idx,
                    "status": "failed",
                    "error": str(exc),
                    "scales": {},
                }
            print(f"[RESULT] {result['condition_name']} r{int(result['repeat']):02d}: {result['status']}")
            results.append(result)
    return sorted(results, key=lambda item: (item.get("condition_name", ""), int(item.get("repeat", 0))))


def mean(values: list[float]) -> float | None:
    valid = [float(v) for v in values if v is not None]
    if not valid:
        return None
    return sum(valid) / len(valid)


def stddev(values: list[float]) -> float | None:
    valid = [float(v) for v in values if v is not None]
    if len(valid) < 2:
        return 0.0 if valid else None
    avg = sum(valid) / len(valid)
    return math.sqrt(sum((value - avg) ** 2 for value in valid) / (len(valid) - 1))


def fmt(value: float | None, digits: int = 1) -> str:
    if value is None:
        return "—"
    return f"{float(value):.{digits}f}"


def md_link(path_value: str) -> str:
    if not path_value:
        return "—"
    path = Path(path_value)
    try:
        rel = path.relative_to(BASE_DIR)
    except ValueError:
        rel = path
    return f"`{rel}`"


def result_total(result: dict[str, Any], scale_name: str) -> float | None:
    scale = result.get("scales", {}).get(scale_name, {})
    if not isinstance(scale, dict):
        return None
    return extract_total(scale.get("score") if "score" in scale else scale)


def result_severity(result: dict[str, Any], scale_name: str) -> str:
    scale = result.get("scales", {}).get(scale_name, {})
    if not isinstance(scale, dict):
        return "—"
    return str(scale.get("severity_label") or extract_severity(scale.get("score")) or "—")


def result_safety(result: dict[str, Any], scale_name: str) -> str:
    scale = result.get("scales", {}).get(scale_name, {})
    if not isinstance(scale, dict):
        return "—"
    return str(scale.get("safety_risk") or extract_safety(scale.get("score")) or "—")


def group_results(results: list[dict[str, Any]], *keys: str) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for result in results:
        grouped.setdefault(tuple(result.get(key) for key in keys), []).append(result)
    return grouped


def render_markdown_report(cfg: RuntimeConfig, results: list[dict[str, Any]]) -> str:
    completed = [item for item in results if item.get("status") in {"completed", "skipped_existing", "dry_run"}]
    failed = [item for item in results if item.get("status") == "failed"]
    lines = [
        f"# 卡布达 T0 重复评估报告",
        "",
        f"- 批次：`{cfg.batch_name}`",
        f"- T0 时间：`{cfg.start_time}`",
        f"- 条件数：{len(cfg.conditions)}",
        f"- 重复次数：{cfg.repeat}",
        f"- 并行度：{cfg.max_parallel}",
        f"- 量表：{', '.join(SCALES.keys())}",
        f"- 完成/跳过：{len(completed)}，失败：{len(failed)}",
        "",
        "## 初始评分 T0 明细",
        "",
        "| 人设 | 严重程度 | 重复 | 量表 | 总分 | 评级 | 安全风险 | scored 文件 |",
        "|---|---|---:|---|---:|---|---|---|",
    ]
    for result in completed:
        for scale_name in SCALES:
            scale = result.get("scales", {}).get(scale_name, {})
            scored_file = scale.get("scored_file", "") if isinstance(scale, dict) else ""
            lines.append(
                "| {persona} | {severity} | {repeat} | {scale} | {total} | {label} | {risk} | {file} |".format(
                    persona=result.get("source_agent_name", "—"),
                    severity=result.get("severity_short_name", "—"),
                    repeat=result.get("repeat", "—"),
                    scale=scale_name,
                    total=fmt(result_total(result, scale_name)),
                    label=result_severity(result, scale_name),
                    risk=result_safety(result, scale_name),
                    file=md_link(str(scored_file or "")),
                )
            )

    lines.extend(["", "## 重复稳定性对比", ""])
    for scale_name in SCALES:
        lines.extend(
            [
                f"### {scale_name}",
                "",
                "| 人设 | 严重程度 | n | 均值 | 标准差 | 最小 | 最大 | 极差 |",
                "|---|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for (_variant, _severity), group in sorted(group_results(completed, "variant", "severity").items()):
            totals = [result_total(item, scale_name) for item in group]
            valid = [float(value) for value in totals if value is not None]
            if not valid:
                continue
            lines.append(
                "| {persona} | {severity} | {n} | {avg} | {sd} | {minv} | {maxv} | {rangev} |".format(
                    persona=group[0].get("source_agent_name", "—"),
                    severity=group[0].get("severity_short_name", "—"),
                    n=len(valid),
                    avg=fmt(mean(valid)),
                    sd=fmt(stddev(valid)),
                    minv=fmt(min(valid)),
                    maxv=fmt(max(valid)),
                    rangev=fmt(max(valid) - min(valid)),
                )
            )
        lines.append("")

    lines.extend(["## 按严重程度对比人设", ""])
    for scale_name in SCALES:
        lines.extend(
            [
                f"### {scale_name}",
                "",
                "| 严重程度 | 人设 | n | 均值 | 标准差 |",
                "|---|---|---:|---:|---:|",
            ]
        )
        rows = []
        for (_severity, _variant), group in group_results(completed, "severity", "variant").items():
            totals = [result_total(item, scale_name) for item in group]
            valid = [float(value) for value in totals if value is not None]
            if not valid:
                continue
            rows.append(
                (
                    group[0].get("severity_short_name", "—"),
                    group[0].get("source_agent_name", "—"),
                    len(valid),
                    mean(valid),
                    stddev(valid),
                )
            )
        for severity, persona, n, avg, sd in sorted(rows):
            lines.append(f"| {severity} | {persona} | {n} | {fmt(avg)} | {fmt(sd)} |")
        lines.append("")

    lines.extend(["## 按人设对比严重程度", ""])
    for scale_name in SCALES:
        lines.extend(
            [
                f"### {scale_name}",
                "",
                "| 人设 | MILD 均值 | MOD 均值 | SEV 均值 | MOD-MILD | SEV-MOD |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        by_variant = group_results(completed, "variant")
        for (_variant,), group in sorted(by_variant.items()):
            per_severity = {}
            for severity in SEVERITY_CONFIG_NAMES:
                values = [
                    result_total(item, scale_name)
                    for item in group
                    if item.get("severity") == severity
                ]
                valid = [float(value) for value in values if value is not None]
                per_severity[severity] = mean(valid)
            mild = per_severity.get("mild")
            mod = per_severity.get("moderate")
            sev = per_severity.get("severe")
            lines.append(
                "| {persona} | {mild} | {mod} | {sev} | {d1} | {d2} |".format(
                    persona=group[0].get("source_agent_name", "—"),
                    mild=fmt(mild),
                    mod=fmt(mod),
                    sev=fmt(sev),
                    d1=fmt((mod - mild) if mod is not None and mild is not None else None),
                    d2=fmt((sev - mod) if sev is not None and mod is not None else None),
                )
            )
        lines.append("")

    if failed:
        lines.extend(["## 失败任务", "", "| 条件 | 重复 | 错误 |", "|---|---:|---|"])
        for item in failed:
            error = str(item.get("error", "") or "").replace("\n", "<br>")
            lines.append(f"| {item.get('condition_name', '—')} | {item.get('repeat', '—')} | {error} |")
        lines.append("")

    lines.extend(render_scale_score_validation_markdown(build_scale_score_validation(results)))

    return "\n".join(lines).rstrip() + "\n"


def resolve_existing_batch_dir(raw_dir: str) -> Path:
    root = (BASE_DIR / raw_dir).resolve() if not Path(raw_dir).is_absolute() else Path(raw_dir).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"已有结果目录不存在: {root}")

    metadata_files = sorted(root.rglob("metadata.json"))
    if not metadata_files:
        raise FileNotFoundError(f"未在目录下找到 metadata.json: {root}")

    batch_dirs = {path.parents[2] for path in metadata_files if len(path.parents) >= 3}
    if len(batch_dirs) == 1:
        return next(iter(batch_dirs))

    direct_batch_dirs = [
        path for path in sorted(batch_dirs)
        if path.name.startswith("t0-repeat-") or path.parent.name == "t0_repeat_eval"
    ]
    if len(direct_batch_dirs) == 1:
        return direct_batch_dirs[0]

    available = "\n".join(str(path) for path in sorted(batch_dirs))
    raise ValueError(f"检测到多个 T0 批次目录，请把 --results-dir 指到其中一个:\n{available}")


def parse_snapshot_start_time(snapshot_name: str) -> str:
    text = str(snapshot_name or "").strip()
    if text.startswith("simulate-") and text.endswith(".json"):
        value = text[len("simulate-") : -len(".json")]
        if len(value) == 13 and value[8] == "-":
            return f"{value[:8]}-{value[9:11]}:{value[11:13]}"
    return ""


def localize_metadata_paths(metadata_path: Path, result: dict[str, Any]) -> dict[str, Any]:
    localized = copy.deepcopy(result)
    repeat_dir = metadata_path.parent
    scales_dir = repeat_dir / "scales"
    localized["metadata_path"] = str(metadata_path)
    localized["scales_dir"] = str(scales_dir)
    for scale_name in SCALES:
        answers_path, scored_path = scale_paths(scales_dir, scale_name)
        scale_payload = localized.setdefault("scales", {}).setdefault(scale_name, {})
        if isinstance(scale_payload, dict):
            scale_payload["answers_file"] = str(answers_path)
            scale_payload["scored_file"] = str(scored_path)
            scored = load_optional_scored(scored_path)
            scale_payload["total"] = extract_total(scored)
            scale_payload["severity_label"] = extract_severity(scored)
            scale_payload["safety_risk"] = extract_safety(scored)
            scale_payload["score"] = scored or {}
    return localized


def load_existing_results(batch_dir: Path) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for metadata_path in sorted(batch_dir.glob("T0-*/r*/metadata.json")):
        try:
            result = load_json_file(metadata_path)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            results.append(
                {
                    "condition_name": metadata_path.parents[1].name,
                    "repeat": metadata_path.parent.name.lstrip("r") or "—",
                    "status": "failed",
                    "error": f"metadata load failed: {exc}",
                    "scales": {},
                }
            )
            continue
        results.append(localize_metadata_paths(metadata_path, result))
    if not results:
        raise FileNotFoundError(f"未在批次目录下找到 T0 metadata: {batch_dir}")
    return results


def build_report_config_from_existing(batch_dir: Path, results: list[dict[str, Any]]) -> RuntimeConfig:
    condition_map: dict[str, T0Condition] = {}
    repeat = 1
    start_time = ""
    for result in results:
        condition_name = str(result.get("condition_name", "") or "")
        variant = str(result.get("variant", "") or "")
        severity = str(result.get("severity", "") or "")
        if condition_name and variant and severity:
            condition_map[condition_name] = T0Condition(condition_name, variant, severity)
        try:
            repeat = max(repeat, int(result.get("repeat", 1) or 1))
        except (TypeError, ValueError):
            pass
        if not start_time:
            start_time = parse_snapshot_start_time(str(result.get("snapshot", "") or ""))

    conditions = sorted(condition_map.values(), key=lambda item: item.name)
    return RuntimeConfig(
        batch_name=batch_dir.name,
        start_time=start_time or "from-existing-results",
        stride=STRIDE,
        repeat=repeat,
        max_parallel=1,
        skip_existing=True,
        force=False,
        dry_run=False,
        conditions=conditions,
    )


def write_existing_report_outputs(batch_dir: Path, cfg: RuntimeConfig, results: list[dict[str, Any]]) -> tuple[Path, Path]:
    aggregate_path = batch_dir / "t0_repeat_results.json"
    report_path = batch_dir / "t0_repeat_report.md"
    validation_payload = build_scale_score_validation(results)
    payload = {
        "batch_name": cfg.batch_name,
        "start_time": cfg.start_time,
        "stride": cfg.stride,
        "repeat": cfg.repeat,
        "max_parallel": cfg.max_parallel,
        "conditions": [
            {"name": c.name, "variant": c.variant, "severity": c.severity}
            for c in cfg.conditions
        ],
        "scales": SCALES,
        "results": results,
        "scale_score_validation": validation_payload,
        "report_only": True,
    }
    write_json_file(aggregate_path, payload)
    report_path.write_text(render_markdown_report(cfg, results), encoding="utf-8")
    print(f"[WRITE] {aggregate_path}")
    print(f"[WRITE] {report_path}")
    return aggregate_path, report_path


def run_report_only(results_dir: str) -> tuple[Path, Path]:
    batch_dir = resolve_existing_batch_dir(results_dir)
    results = load_existing_results(batch_dir)
    cfg = build_report_config_from_existing(batch_dir, results)
    print("==========================================")
    print(" T0 已有结果报告汇总")
    print("==========================================")
    print(f"  批次目录:     {batch_dir}")
    print(f"  批次名称:     {cfg.batch_name}")
    print(f"  条件数:       {len(cfg.conditions)}")
    print(f"  metadata 数:  {len(results)}")
    print(f"  重复次数:     {cfg.repeat}")
    print("==========================================")
    return write_existing_report_outputs(batch_dir, cfg, results)


def write_outputs(cfg: RuntimeConfig, results: list[dict[str, Any]]) -> tuple[Path, Path]:
    batch_dir = T0_OUTPUT_ROOT / cfg.batch_name
    aggregate_path = batch_dir / "t0_repeat_results.json"
    report_path = batch_dir / "t0_repeat_report.md"
    validation_payload = build_scale_score_validation(results)
    payload = {
        "batch_name": cfg.batch_name,
        "start_time": cfg.start_time,
        "stride": cfg.stride,
        "repeat": cfg.repeat,
        "max_parallel": cfg.max_parallel,
        "conditions": [
            {"name": c.name, "variant": c.variant, "severity": c.severity}
            for c in cfg.conditions
        ],
        "scales": SCALES,
        "results": results,
        "scale_score_validation": validation_payload,
    }
    if cfg.dry_run:
        print(f"[DRY-RUN] write aggregate: {aggregate_path}")
        print(f"[DRY-RUN] write report:    {report_path}")
        return aggregate_path, report_path
    write_json_file(aggregate_path, payload)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_markdown_report(cfg, results), encoding="utf-8")
    print(f"[WRITE] {aggregate_path}")
    print(f"[WRITE] {report_path}")
    return aggregate_path, report_path


def main() -> None:
    args = parse_args()
    if args.report_only:
        if not normalize_selector(args.results_dir):
            raise ValueError("--report-only 需要配合 --results-dir 指定已有结果目录。")
        aggregate_path, report_path = run_report_only(str(args.results_dir))
        print(f"\n[Done] 已基于已有结果生成报告: {report_path}")
        print(f"[Done] 聚合数据: {aggregate_path}")
        return

    cfg = resolve_runtime_config(args)
    print_effective_config(cfg)
    results = run_all_tasks(cfg)
    aggregate_path, report_path = write_outputs(cfg, results)
    failed_count = sum(1 for item in results if item.get("status") == "failed")
    if failed_count:
        print(f"\n[Done] T0 评估结束，但有 {failed_count} 个失败任务。报告: {report_path}")
    else:
        print(f"\n[Done] T0 评估完成。报告: {report_path}")
        print(f"[Done] 聚合数据: {aggregate_path}")


if __name__ == "__main__":
    main()
