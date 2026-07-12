#!/usr/bin/env python3
"""Run repeat scale evaluation immediately for an unfinished archived run.

Typical usage:
    python3 runshells/run_immediate_repeat_scale_eval.py \
      --run-name batch-0707-Counsel-KBD2-G1-SEV-0707-2011 \
      --repeat 3 --max-parallel 3

The script prepares a synthetic staged_eval/NOW job from the latest available
checkpoint state, refreshes its runtime config from current storage memory, and
then delegates the actual repeat scoring/reporting to
run_archived_repeat_scale_eval.py.
"""

from __future__ import annotations

import argparse
import copy
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from run_archived_repeat_scale_eval import (
    BASE_DIR,
    SCALES,
    WORKER_PYTHON,
    load_json_file,
    write_json_file,
)
from run_batch_experiment import (
    BATCH_STATE_ROOT,
    trigger_sort_key,
    resolve_conditions,
)


# ============================================================
# ↓↓↓ 可调参数：直接修改这里即可；命令行 --xxx 会覆盖这里 ↓↓↓
# ============================================================

# checkpoint run_name，例如 batch-0707-Counsel-KBD2-G1-SEV-0707-2011。
# 留空时必须通过 --run-name 传入。
RUN_NAME = ""

# results 根目录。
ARCHIVE_RESULTS_ROOT = "results"

# condition 名；留空时从 batch_state 或 run_name 反查。
CONDITION = ""

# 原始 *_summary.json；留空时自动寻找或临时合成。
ORIGINAL_SUMMARY = ""

# 输出批次名；留空时自动生成 immediate-repeat-<condition>-<MMdd-HHmm>。
NAME = ""

# 每个评估点重复次数。
REPEAT = 3

# 逗号分隔 labels；"auto" 表示已完成 staged_eval + NOW。
LABELS = "auto"

# 即时评估 label。
NOW_LABEL = "NOW"

# auto labels 中是否加入 NOW。
INCLUDE_NOW = True

# 传给 archived repeat 的并行任务数。
MAX_PARALLEL = 1

# 是否覆盖已有 repeat 输出。
FORCE = False

# 是否只打印计划，不写入、不调用 worker。
DRY_RUN = False

# 是否只汇总已有 repeat 输出。
REPORT_ONLY = False

# 是否启用条目稳定性补跑。
STABILITY_RERUN = True

# 不稳定条目最多补跑轮数。
MAX_EXTRA_REPEAT = 4

# 条目分数极差阈值。
STABILITY_RANGE_THRESHOLD = 1

# 是否把目标患者动态抑郁状态重置为初始 depression_config 状态后复评。
RESET_TARGET_DEPRESSION_STATE = False

# 复评输出使用的新 group 标签，例如 "G8"；留空则沿用原始 condition。
OUTPUT_GROUP = ""

# 开启后，把即时复评 runtime_config 中的 agent.think.llm 和
# agent.associate.embedding 从存档里的 Ollama 改为 vLLM/OpenAI-compatible。
USE_VLLM_MODELS = False

# 下面四项参照 runshells/vllm_services.sh。
VLLM_THINK_MODEL = "qwen3-8b-vllm"
VLLM_THINK_BASE_URL = "http://127.0.0.1:18000/v1"
VLLM_EMBED_MODEL = "bge-m3-vllm"
VLLM_EMBED_BASE_URL = "http://127.0.0.1:18001/v1"
VLLM_API_KEY = "EMPTY"

# ============================================================
# ↑↑↑ 可调参数：直接修改这里即可；命令行 --xxx 会覆盖这里 ↑↑↑
# ============================================================


IMMEDIATE_SNAPSHOT_NAME = "immediate_latest_snapshot.json"
ARCHIVED_REPEAT_SCRIPT = BASE_DIR / "runshells" / "run_archived_repeat_scale_eval.py"


@dataclass(frozen=True)
class ImmediateConfig:
    archive_results_root: Path
    run_name: str
    condition_name: str
    output_name: str
    repeat: int
    labels: list[str]
    now_label: str
    include_now: bool
    max_parallel: int
    force: bool
    dry_run: bool
    report_only: bool
    stability_rerun: bool
    max_extra_repeat: int
    stability_range_threshold: int
    original_summary: Path | None
    reset_target_depression_state: bool
    output_group: str
    use_vllm_models: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="立即为未完成存档运行重复量表评估")
    run_name_default = str(RUN_NAME or "").strip()
    parser.add_argument("--run-name", default=run_name_default or None, required=not bool(run_name_default), help="checkpoint run_name，例如 batch-0707-Counsel-KBD2-G1-SEV-0707-2011")
    parser.add_argument("--archive-results-root", default=ARCHIVE_RESULTS_ROOT, help="results 根目录，默认 results")
    parser.add_argument("--condition", default=CONDITION, help="condition 名；留空时从 batch_state 或 run_name 反查")
    parser.add_argument("--original-summary", default=ORIGINAL_SUMMARY, help="原始 *_summary.json；留空时交给 archived repeat 脚本自动处理")
    parser.add_argument("--name", default=NAME, help="输出批次名；默认 immediate-repeat-<condition>-<MMdd-HHmm>")
    parser.add_argument("--repeat", type=int, default=REPEAT, help="每个评估点重复次数")
    parser.add_argument("--labels", default=LABELS, help="逗号分隔 labels；默认 auto=已完成 staged_eval + NOW")
    parser.add_argument("--now-label", default=NOW_LABEL, help="即时评估 label，默认 NOW")
    parser.add_argument("--include-now", action=argparse.BooleanOptionalAction, default=INCLUDE_NOW, help="auto labels 中是否加入 NOW")
    parser.add_argument("--max-parallel", type=int, default=MAX_PARALLEL, help="传给 archived repeat 的并行任务数")
    parser.add_argument("--force", action=argparse.BooleanOptionalAction, default=FORCE, help="覆盖已有 repeat 输出")
    parser.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=DRY_RUN, help="只打印计划，不写入、不调用 worker")
    parser.add_argument("--report-only", action=argparse.BooleanOptionalAction, default=REPORT_ONLY, help="只汇总已有 repeat 输出")
    parser.add_argument("--stability-rerun", action=argparse.BooleanOptionalAction, default=STABILITY_RERUN, help="启用条目稳定性补跑")
    parser.add_argument("--max-extra-repeat", type=int, default=MAX_EXTRA_REPEAT, help="不稳定条目最多补跑轮数")
    parser.add_argument("--stability-range-threshold", type=int, default=STABILITY_RANGE_THRESHOLD, help="条目分数极差阈值")
    parser.add_argument("--reset-target-depression-state", action=argparse.BooleanOptionalAction, default=RESET_TARGET_DEPRESSION_STATE, help="沿用 archived repeat 的抑郁状态重置选项")
    parser.add_argument("--output-group", default=OUTPUT_GROUP, help="沿用 archived repeat 的输出 group 重标记选项")
    parser.add_argument("--use-vllm-models", action=argparse.BooleanOptionalAction, default=USE_VLLM_MODELS, help="把即时复评 runtime_config 的 think LLM / embedding 改为 vLLM")
    return parser.parse_args()


def resolve_path(raw: str | Path) -> Path:
    text = str(raw or "").strip()
    if text.startswith("share/"):
        text = "/" + text
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = BASE_DIR / path
    return path.resolve()


def load_json_retry(path: Path, retries: int = 3, sleep_seconds: float = 1.0) -> dict[str, Any]:
    last_exc: Exception | None = None
    for idx in range(max(1, retries)):
        try:
            return load_json_file(path)
        except Exception as exc:
            last_exc = exc
            if idx + 1 < retries:
                time.sleep(sleep_seconds)
    raise last_exc or FileNotFoundError(str(path))


def load_optional_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = load_json_retry(path)
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def checkpoint_dir(cfg: ImmediateConfig) -> Path:
    return cfg.archive_results_root / "checkpoints" / cfg.run_name


def find_batch_state_for_run(run_name: str) -> dict[str, Any]:
    for path in sorted(BATCH_STATE_ROOT.glob("*/*.json")):
        payload = load_optional_json(path)
        if str(payload.get("run_name", "") or "") == run_name:
            return payload
    return {}


def infer_condition_from_run_name(run_name: str) -> str:
    match = re.search(r"(Counsel(?:-[A-Za-z0-9]+){2,3})(?:-\d{4}-\d{4})?$", run_name)
    if match:
        return match.group(1)
    raise ValueError(f"无法从 run_name 推断 condition，请显式传入 --condition: {run_name}")


def resolve_condition_name(args: argparse.Namespace) -> str:
    raw = str(args.condition or "").strip()
    if raw:
        return raw
    state = find_batch_state_for_run(str(args.run_name or "").strip())
    condition_name = str(state.get("condition_name", "") or "").strip()
    if condition_name:
        return condition_name
    return infer_condition_from_run_name(str(args.run_name or "").strip())


def generate_output_name(condition_name: str) -> str:
    safe_condition = condition_name.replace("/", "_").replace("\\", "_")
    return f"immediate-repeat-{safe_condition}-{datetime.now().strftime('%m%d-%H%M')}"


def resolve_runtime_config(args: argparse.Namespace) -> ImmediateConfig:
    archive_results_root = resolve_path(args.archive_results_root)
    run_name = str(args.run_name or "").strip()
    if not run_name:
        raise ValueError("--run-name is required")
    if not (archive_results_root / "checkpoints" / run_name).is_dir():
        raise FileNotFoundError(f"checkpoint 不存在: {archive_results_root / 'checkpoints' / run_name}")

    condition_name = resolve_condition_name(args)
    original_summary = None
    if str(args.original_summary or "").strip():
        original_summary = resolve_path(args.original_summary)
        if not original_summary.is_file():
            raise FileNotFoundError(f"original summary 不存在: {original_summary}")

    return ImmediateConfig(
        archive_results_root=archive_results_root,
        run_name=run_name,
        condition_name=condition_name,
        output_name=str(args.name or "").strip() or generate_output_name(condition_name),
        repeat=max(1, int(args.repeat or 1)),
        labels=[],
        now_label=str(args.now_label or NOW_LABEL).strip() or NOW_LABEL,
        include_now=bool(args.include_now),
        max_parallel=max(1, int(args.max_parallel or 1)),
        force=bool(args.force),
        dry_run=bool(args.dry_run),
        report_only=bool(args.report_only),
        stability_rerun=bool(args.stability_rerun),
        max_extra_repeat=max(0, int(args.max_extra_repeat or 0)),
        stability_range_threshold=max(1, int(args.stability_range_threshold or 1)),
        original_summary=original_summary,
        reset_target_depression_state=bool(args.reset_target_depression_state),
        output_group=str(args.output_group or "").strip(),
        use_vllm_models=bool(args.use_vllm_models),
    )


def timestamped_simulate_files(cp_dir: Path) -> list[Path]:
    files = []
    for path in cp_dir.iterdir():
        if not path.is_file():
            continue
        if re.fullmatch(r"simulate-\d{8}-\d{4}\.json", path.name):
            files.append(path)
    return sorted(files, key=lambda path: path.name)


def latest_source_snapshot(cp_dir: Path) -> Path:
    snapshots = timestamped_simulate_files(cp_dir)
    if not snapshots:
        snapshots = sorted(
            path for path in cp_dir.iterdir()
            if path.is_file() and path.name.startswith("simulate-") and path.name.endswith(".json")
        )
    if not snapshots:
        raise FileNotFoundError(f"未找到 simulate-*.json: {cp_dir}")
    return snapshots[-1]


def parse_dt(value: str) -> datetime | None:
    text = str(value or "").strip()
    for fmt in ("%Y%m%d-%H:%M:%S", "%Y%m%d-%H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def normalize_sim_time(value: str) -> str:
    dt = parse_dt(value)
    if dt:
        return dt.strftime("%Y%m%d-%H:%M")
    text = str(value or "").strip()
    if re.fullmatch(r"\d{8}-\d{4}", text):
        return f"{text[:9]}{text[9:11]}:{text[11:]}"
    return text


def snapshot_time(payload: dict[str, Any]) -> str:
    raw_time = payload.get("time", "")
    if isinstance(raw_time, dict):
        raw_time = raw_time.get("start", "")
    return normalize_sim_time(str(raw_time or ""))


def infer_progress_from_log(cp_dir: Path, fallback_step: int, fallback_time: str) -> tuple[int, str, str]:
    log_path = cp_dir / "run_batch_experiment.log"
    if not log_path.is_file():
        return fallback_step, fallback_time, "snapshot"
    try:
        text = log_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return fallback_step, fallback_time, "snapshot"
    pattern = re.compile(
        r"Simulate Step\[(\d+)/(\d+), time: (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]"
    )
    matches = pattern.findall(text)
    if not matches:
        return fallback_step, fallback_time, "snapshot"
    step, _total, sim_time = matches[-1]
    return int(step), normalize_sim_time(sim_time), "run_log"


def node_id_sort_value(node_id: str) -> int:
    match = re.search(r"(\d+)$", str(node_id or ""))
    return int(match.group(1)) if match else -1


def extract_memory_from_docstore(docstore_path: Path, sim_time: str) -> dict[str, list[str]]:
    payload = load_optional_json(docstore_path)
    data = payload.get("docstore/data", {})
    if not isinstance(data, dict):
        return {"event": [], "thought": [], "chat": []}
    now = parse_dt(sim_time)
    grouped: dict[str, list[tuple[str, str]]] = {"event": [], "thought": [], "chat": []}
    for node_id, raw_node in data.items():
        if not isinstance(raw_node, dict):
            continue
        node_data = raw_node.get("__data__", {})
        if not isinstance(node_data, dict):
            continue
        metadata = node_data.get("metadata", {})
        if not isinstance(metadata, dict):
            continue
        node_type = str(metadata.get("node_type", "") or "")
        if node_type not in grouped:
            continue
        create = str(metadata.get("create", "") or "")
        expire = str(metadata.get("expire", "") or "")
        if now is not None:
            create_dt = parse_dt(create)
            expire_dt = parse_dt(expire)
            if create_dt is not None and create_dt > now:
                continue
            if expire_dt is not None and expire_dt < now:
                continue
        grouped[node_type].append((str(node_id), create))

    result: dict[str, list[str]] = {}
    for node_type, rows in grouped.items():
        rows.sort(key=lambda item: (parse_dt(item[1]) or datetime.min, node_id_sort_value(item[0])), reverse=True)
        result[node_type] = [node_id for node_id, _create in rows]
    return result


def refresh_agent_memories_from_storage(runtime_config: dict[str, Any], cp_dir: Path, sim_time: str) -> dict[str, Any]:
    agents = runtime_config.get("agents", {})
    if not isinstance(agents, dict):
        return {"updated_agents": [], "node_counts": {}}

    updated_agents = []
    node_counts: dict[str, dict[str, int]] = {}
    for agent_name, agent_cfg in agents.items():
        if not isinstance(agent_cfg, dict):
            continue
        docstore_path = cp_dir / "storage" / str(agent_name) / "associate" / "docstore.json"
        if not docstore_path.is_file():
            continue
        memory = extract_memory_from_docstore(docstore_path, sim_time)
        associate = agent_cfg.setdefault("associate", {})
        if not isinstance(associate, dict):
            associate = {}
            agent_cfg["associate"] = associate
        associate["memory"] = memory
        updated_agents.append(str(agent_name))
        node_counts[str(agent_name)] = {key: len(value) for key, value in memory.items()}
    return {"updated_agents": updated_agents, "node_counts": node_counts}


def apply_vllm_runtime_models(runtime_config: dict[str, Any]) -> dict[str, Any]:
    think_payload = {
        "provider": "openai",
        "model": VLLM_THINK_MODEL,
        "base_url": VLLM_THINK_BASE_URL,
        "api_key": VLLM_API_KEY,
    }
    embedding_payload = {
        "provider": "openai",
        "model": VLLM_EMBED_MODEL,
        "base_url": VLLM_EMBED_BASE_URL,
        "api_key": VLLM_API_KEY,
    }
    forced_llm_payload = {
        "enabled": True,
        "provider": "openai",
        "model": VLLM_THINK_MODEL,
        "base_url": VLLM_THINK_BASE_URL,
        "api_key_env": "GA_REPEAT_VLLM_API_KEY",
    }
    patched_paths: list[str] = []

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
    patched_paths.append("agent_base.think.llm")

    associate_cfg = agent_base.setdefault("associate", {})
    if not isinstance(associate_cfg, dict):
        associate_cfg = {}
        agent_base["associate"] = associate_cfg
    embedding_cfg = associate_cfg.setdefault("embedding", {})
    if not isinstance(embedding_cfg, dict):
        embedding_cfg = {}
        associate_cfg["embedding"] = embedding_cfg
    embedding_cfg.update(embedding_payload)
    patched_paths.append("agent_base.associate.embedding")

    agents = runtime_config.get("agents", {})
    if isinstance(agents, dict):
        for agent_name, agent_cfg in agents.items():
            if not isinstance(agent_cfg, dict):
                continue
            local_think = agent_cfg.get("think")
            if isinstance(local_think, dict) and isinstance(local_think.get("llm"), dict):
                local_think["llm"].update(think_payload)
                patched_paths.append(f"agents.{agent_name}.think.llm")
            local_associate = agent_cfg.get("associate")
            if isinstance(local_associate, dict) and isinstance(local_associate.get("embedding"), dict):
                local_associate["embedding"].update(embedding_payload)
                patched_paths.append(f"agents.{agent_name}.associate.embedding")

    intervention_cfg = runtime_config.setdefault("intervention", {})
    if not isinstance(intervention_cfg, dict):
        intervention_cfg = {}
        runtime_config["intervention"] = intervention_cfg
    forced_llm_cfg = intervention_cfg.setdefault("forced_llm", {})
    if not isinstance(forced_llm_cfg, dict):
        forced_llm_cfg = {}
        intervention_cfg["forced_llm"] = forced_llm_cfg
    forced_llm_cfg.update(forced_llm_payload)
    patched_paths.append("intervention.forced_llm")

    return {
        "enabled": True,
        "patched_paths": patched_paths,
        "think_llm": copy.deepcopy(think_payload),
        "embedding": copy.deepcopy(embedding_payload),
        "forced_llm": copy.deepcopy(forced_llm_payload),
    }


def completed_staged_records(cp_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    index = load_optional_json(cp_dir / "staged_eval" / "index.json")
    for record in index.get("records", []) or []:
        if isinstance(record, dict) and record.get("status") == "ok":
            label = str(record.get("trigger_label", "") or "").strip()
            if label:
                records.append(record)

    seen = {str(record.get("trigger_label", "") or "") for record in records}
    staged_root = cp_dir / "staged_eval"
    if staged_root.is_dir():
        for path in staged_root.iterdir():
            if not path.is_dir() or path.name.startswith("_") or path.name in seen:
                continue
            metadata = load_optional_json(path / "metadata.json")
            if metadata.get("status") == "ok" and (path / "job.json").is_file():
                records.append(metadata)
                seen.add(path.name)

    records.sort(
        key=lambda item: trigger_sort_key(
            str(item.get("trigger_label", "") or ""),
            int(item.get("completed_session_count", 0) or 0),
        )
    )
    return records


def auto_labels(cp_dir: Path, now_label: str, include_now: bool) -> list[str]:
    labels = [str(record.get("trigger_label", "") or "") for record in completed_staged_records(cp_dir)]
    labels = [label for label in labels if label and label != now_label]
    if include_now:
        labels.append(now_label)
    return labels


def resolve_labels(args: argparse.Namespace, cp_dir: Path, now_label: str, include_now: bool) -> list[str]:
    raw = str(args.labels or "auto").strip()
    if not raw or raw.lower() == "auto":
        labels = auto_labels(cp_dir, now_label, include_now)
    else:
        labels = [item.strip() for item in raw.split(",") if item.strip()]
    if not labels:
        raise ValueError("没有可评估的 labels：staged_eval 尚无完成记录，且未启用 NOW")
    return labels


def target_agent(cp_dir: Path) -> str:
    for label in ("T0", "session_4", "session_8", "session_12"):
        job = load_optional_json(cp_dir / "staged_eval" / label / "job.json")
        if job.get("target_agent"):
            return str(job["target_agent"])
    for metadata_path in sorted((cp_dir / "staged_eval").glob("*/metadata.json")):
        metadata = load_optional_json(metadata_path)
        if metadata.get("target_agent"):
            return str(metadata["target_agent"])
    return "卡布达"


def completed_session_count(cp_dir: Path) -> int:
    counts = [int(record.get("completed_session_count", 0) or 0) for record in completed_staged_records(cp_dir)]
    manifest_paths = sorted((cp_dir / "consult_history").glob("*/manifest.json"))
    for path in manifest_paths:
        manifest = load_optional_json(path)
        meeting_to_record = manifest.get("meeting_to_record", {})
        if isinstance(meeting_to_record, dict):
            counts.append(len(meeting_to_record))
        try:
            counts.append(int(manifest.get("record_seq", 0) or 0))
        except Exception:
            pass
    judge = load_optional_json(cp_dir / "judge_traces" / "judge_conversation.json")
    sessions = judge.get("sessions", [])
    if isinstance(sessions, list):
        counts.append(len(sessions))
    return max(counts or [0])


def condition_metadata(condition_name: str) -> dict[str, str]:
    try:
        condition = resolve_conditions(condition_name)[0]
    except Exception:
        return {"variant": "", "group": "", "severity": ""}
    return {
        "variant": condition.variant,
        "group": condition.group,
        "severity": condition.severity,
    }


def prepare_immediate_now_job(cfg: ImmediateConfig) -> dict[str, Any]:
    cp_dir = checkpoint_dir(cfg)
    source_snapshot_path = latest_source_snapshot(cp_dir)
    runtime_config = copy.deepcopy(load_json_retry(source_snapshot_path))
    fallback_time = snapshot_time(runtime_config)
    fallback_step = int(runtime_config.get("step", 0) or 0)
    step_no, sim_time, progress_source = infer_progress_from_log(cp_dir, fallback_step, fallback_time)
    if sim_time:
        runtime_config["time"] = sim_time
    if step_no:
        runtime_config["step"] = step_no
    conversation = load_optional_json(cp_dir / "conversation.json")
    vllm_patch = (
        apply_vllm_runtime_models(runtime_config)
        if cfg.use_vllm_models
        else {"enabled": False}
    )
    refresh_info = refresh_agent_memories_from_storage(runtime_config, cp_dir, sim_time or fallback_time)

    runtime_config["_immediate_snapshot"] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source_snapshot": source_snapshot_path.name,
        "progress_source": progress_source,
        "sim_time": sim_time,
        "step": step_no,
        "memory_refresh": refresh_info,
        "vllm_model_patch": vllm_patch,
    }

    snapshot_path = cp_dir / IMMEDIATE_SNAPSHOT_NAME
    now_dir = cp_dir / "staged_eval" / cfg.now_label
    metadata = {
        "status": "ok",
        "trigger_label": cfg.now_label,
        "completed_session_count": completed_session_count(cp_dir),
        "completed_session_count_source": "immediate_consult_history_or_staged_eval",
        "step_no": step_no,
        "sim_time": sim_time,
        "snapshot_name": IMMEDIATE_SNAPSHOT_NAME,
        "target_agent": target_agent(cp_dir),
        "source": "immediate_repeat_scale_eval",
        "source_snapshot_name": source_snapshot_path.name,
        "memory_refresh": refresh_info,
        "vllm_model_patch": vllm_patch,
        "readonly_eval": True,
        "uses_answer_without_memory": True,
        "writes_isolated_via_temp_storage": True,
    }
    job = {
        "run_name": cfg.run_name,
        "trigger_label": cfg.now_label,
        "completed_session_count": int(metadata["completed_session_count"]),
        "step_no": step_no,
        "sim_time": sim_time,
        "snapshot_name": IMMEDIATE_SNAPSHOT_NAME,
        "target_agent": metadata["target_agent"],
        "scales": list(SCALES.keys()),
        "scale_question_files": {
            scale_name: scale_cfg["question_file"]
            for scale_name, scale_cfg in SCALES.items()
        },
        "runtime_config": runtime_config,
        "conversation": conversation,
        "trigger_dir": str(now_dir),
        "worker_result_path": str(now_dir / "worker_result.json"),
        "storage_source_root": str(cp_dir / "storage"),
        "tmp_root_parent": str(now_dir / "_tmp"),
        "cleanup_tmp_storage": True,
    }

    if cfg.dry_run:
        print(f"[DRY-RUN] would write refreshed snapshot: {snapshot_path}")
        print(f"[DRY-RUN] would write NOW job: {now_dir / 'job.json'}")
        print(f"[DRY-RUN] immediate metadata: step={step_no} sim_time={sim_time} completed_sessions={metadata['completed_session_count']}")
        if cfg.use_vllm_models:
            print(f"[DRY-RUN] would patch runtime models to vLLM: llm={VLLM_THINK_MODEL} embed={VLLM_EMBED_MODEL}")
    else:
        write_json_file(snapshot_path, runtime_config)
        write_json_file(now_dir / "job.json", job)
        write_json_file(now_dir / "metadata.json", metadata)
        print(f"[WRITE] {snapshot_path}")
        print(f"[WRITE] {now_dir / 'job.json'}")
        print(f"[WRITE] {now_dir / 'metadata.json'}")
    return metadata


def synthesize_original_summary_if_needed(cfg: ImmediateConfig, labels: list[str]) -> Path | None:
    if cfg.original_summary:
        return cfg.original_summary
    reports_dir = cfg.archive_results_root / "experiment_data" / "reports"
    exact = reports_dir / f"{cfg.run_name}_summary.json"
    if exact.is_file():
        return exact
    candidates = sorted(
        path for path in reports_dir.glob("*_summary.json")
        if not path.name.startswith("repeat-") and not path.name.startswith("immediate-repeat-")
    )
    for path in reversed(candidates):
        payload = load_optional_json(path)
        for condition in payload.get("conditions", []) or []:
            if (
                isinstance(condition, dict)
                and str(condition.get("run_name", "") or "") == cfg.run_name
                and str(condition.get("condition_name", "") or "") == cfg.condition_name
            ):
                return path

    metadata = condition_metadata(cfg.condition_name)
    cp_dir = checkpoint_dir(cfg)
    evaluations = []
    for record in completed_staged_records(cp_dir):
        label = str(record.get("trigger_label", "") or "")
        if label not in labels:
            continue
        evaluations.append(
            {
                "trigger_label": label,
                "completed_session_count": int(record.get("completed_session_count", 0) or 0),
                "sim_time": str(record.get("sim_time", "") or ""),
                "snapshot_name": str(record.get("snapshot_name", "") or ""),
                "source": "immediate_synthesized_original_summary",
                "scales": {},
            }
        )
    summary = {
        "batch_name": "immediate-synthesized-original-summary",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "summary_only": True,
        "conditions": [
            {
                "condition_name": cfg.condition_name,
                "run_name": cfg.run_name,
                "run_dir": str(cfg.archive_results_root / "experiment_data" / cfg.run_name),
                "variant": metadata["variant"],
                "group": metadata["group"],
                "severity": metadata["severity"],
                "evaluations": evaluations,
                "final_deltas": {},
            }
        ],
        "warnings": [
            "此 summary 由 run_immediate_repeat_scale_eval.py 为未完成存档临时合成。",
            "原始量表分数可能缺失，重复评估报告以本次复评轨迹为主。",
        ],
        "trigger_labels": labels,
    }
    out_dir = cfg.archive_results_root / "experiment_data" / "reports"
    out_path = out_dir / f"{cfg.output_name}_original_summary.json"
    if cfg.dry_run:
        print(f"[DRY-RUN] would write synthesized original summary: {out_path}")
        return out_path
    write_json_file(out_path, summary)
    print(f"[WRITE] {out_path}")
    return out_path


def build_archived_repeat_cmd(cfg: ImmediateConfig, labels: list[str], original_summary: Path | None) -> list[str]:
    cmd = [
        WORKER_PYTHON or sys.executable,
        str(ARCHIVED_REPEAT_SCRIPT),
        "--archive-results-root",
        str(cfg.archive_results_root),
        "--condition",
        cfg.condition_name,
        "--labels",
        ",".join(labels),
        "--repeat",
        str(cfg.repeat),
        "--name",
        cfg.output_name,
        "--max-parallel",
        str(cfg.max_parallel),
        "--max-extra-repeat",
        str(cfg.max_extra_repeat),
        "--stability-range-threshold",
        str(cfg.stability_range_threshold),
    ]
    if original_summary:
        cmd.extend(["--original-summary", str(original_summary)])
    if cfg.force:
        cmd.append("--force")
    if cfg.dry_run:
        cmd.append("--dry-run")
    if cfg.report_only:
        cmd.append("--report-only")
    if not cfg.stability_rerun:
        cmd.append("--no-stability-rerun")
    if cfg.reset_target_depression_state:
        cmd.append("--reset-target-depression-state")
    if cfg.output_group:
        cmd.extend(["--output-group", cfg.output_group])
    return cmd


def build_child_env(cfg: ImmediateConfig) -> dict[str, str]:
    env = os.environ.copy()
    if cfg.use_vllm_models:
        env.update(
            {
                "GA_REPEAT_USE_VLLM_MODELS": "1",
                "GA_REPEAT_VLLM_THINK_MODEL": VLLM_THINK_MODEL,
                "GA_REPEAT_VLLM_THINK_BASE_URL": VLLM_THINK_BASE_URL,
                "GA_REPEAT_VLLM_EMBED_MODEL": VLLM_EMBED_MODEL,
                "GA_REPEAT_VLLM_EMBED_BASE_URL": VLLM_EMBED_BASE_URL,
                "GA_REPEAT_VLLM_API_KEY": VLLM_API_KEY,
            }
        )
    return env


def validate_archived_repeat_script(cfg: ImmediateConfig) -> None:
    try:
        script_text = ARCHIVED_REPEAT_SCRIPT.read_text(encoding="utf-8", errors="ignore")
    except Exception as exc:
        raise RuntimeError(f"无法读取 archived repeat 脚本: {ARCHIVED_REPEAT_SCRIPT} ({exc})") from exc

    required_markers = ["worker_failure_detail"]
    if cfg.use_vllm_models:
        required_markers.extend(
            [
                "GA_REPEAT_USE_VLLM_MODELS",
                "apply_runtime_model_env_override",
                "intervention_cfg",
                "forced_llm",
            ]
        )
    missing = [marker for marker in required_markers if marker not in script_text]
    if missing:
        raise RuntimeError(
            "当前 run_archived_repeat_scale_eval.py 不是最新版本，缺少补丁标记: "
            + ", ".join(missing)
            + f"\n请同时更新这两个脚本后再运行:\n  {Path(__file__).resolve()}\n  {ARCHIVED_REPEAT_SCRIPT}"
        )


def print_plan(cfg: ImmediateConfig, labels: list[str], now_metadata: dict[str, Any]) -> None:
    print("==========================================")
    print(" 立即重复量表评估")
    print("==========================================")
    print(f"  run_name:       {cfg.run_name}")
    print(f"  condition:      {cfg.condition_name}")
    print(f"  archive root:   {cfg.archive_results_root}")
    print(f"  output name:    {cfg.output_name}")
    print(f"  labels:         {', '.join(labels)}")
    print(f"  repeat:         {cfg.repeat}")
    print(f"  max_parallel:   {cfg.max_parallel}")
    print(f"  use vLLM:       {cfg.use_vllm_models}")
    if cfg.use_vllm_models:
        print(f"  vLLM llm:       {VLLM_THINK_MODEL} @ {VLLM_THINK_BASE_URL}")
        print(f"  vLLM embed:     {VLLM_EMBED_MODEL} @ {VLLM_EMBED_BASE_URL}")
    if now_metadata:
        print(f"  NOW sim_time:   {now_metadata.get('sim_time', '')}")
        print(f"  NOW sessions:   {now_metadata.get('completed_session_count', 0)}")
    print("==========================================")


def main() -> int:
    args = parse_args()
    cfg_base = resolve_runtime_config(args)
    cp_dir = checkpoint_dir(cfg_base)
    labels = resolve_labels(args, cp_dir, cfg_base.now_label, cfg_base.include_now)
    cfg = ImmediateConfig(**{**cfg_base.__dict__, "labels": labels})

    now_metadata: dict[str, Any] = {}
    if cfg.now_label in labels and not cfg.report_only:
        now_metadata = prepare_immediate_now_job(cfg)

    original_summary = synthesize_original_summary_if_needed(cfg, labels)
    print_plan(cfg, labels, now_metadata)
    cmd = build_archived_repeat_cmd(cfg, labels, original_summary)
    validate_archived_repeat_script(cfg)
    print(f"[RUN] {' '.join(cmd)}")
    if cfg.dry_run:
        return 0
    subprocess.run(cmd, cwd=BASE_DIR, check=True, env=build_child_env(cfg))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
