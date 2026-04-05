#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Analyze dynamic depression persona changes from checkpoint snapshots.

Main capabilities:
1) Quantify dynamic persona state changes over time.
2) Quantify forced intervention conversation effects.
3) Output timeline CSV, event CSV, summary Markdown and plots.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt


TIME_FMT_PRIMARY = "%Y%m%d-%H:%M"
TIME_FMT_ALT = "%Y%m%d-%H:%M:%S"
TASK_DATE_FMT = "%Y-%m-%d"
TASK_TIME_FMT = "%H:%M"

# ========= User-editable defaults =========
# 这些参数是默认注入值：不传终端参数时，脚本直接使用这里的配置。
# DEFAULT_ARCHIVE: 存档目录名（位于 results/checkpoints/<archive>）
# DEFAULT_AGENT: 需要分析的角色名（必须与快照 agents 键完全一致）
# DEFAULT_DOCTOR: 医生角色名；留空则自动从快照 intervention.doctor 读取
# DEFAULT_PAIR_KEY: 医生::患者配对键；留空则按 archive 内容自动推断
DEFAULT_ARCHIVE = "sim-dym-severe-kbd-0404-2"
DEFAULT_AGENT = "卡布达"
DEFAULT_DOCTOR = ""
DEFAULT_PAIR_KEY = ""
# DEFAULT_CHECKPOINTS_ROOT: checkpoints 根目录；留空用 <repo>/results/checkpoints
# DEFAULT_OUTPUT_DIR: 输出目录；留空用 analysis_outputs/<archive>_<agent>
# DEFAULT_NO_PLOT: True=只导出表格/摘要，False=同时生成图表
# DEFAULT_LIST_ARCHIVES: True=仅列出存档并退出
# DEFAULT_LIST_AGENTS: True=仅列出角色并退出（依赖 DEFAULT_ARCHIVE）
DEFAULT_CHECKPOINTS_ROOT = ""
DEFAULT_OUTPUT_DIR = "sim-dym-severe-kbd-0404-2"
DEFAULT_NO_PLOT = False
DEFAULT_LIST_ARCHIVES = False
DEFAULT_LIST_AGENTS = False

TRIGGER_PLOT_ALIASES = {
    "未来焦虑": "future_anxiety",
    "自我价值": "self_worth",
    "自杀意念": "suicidal_ideation",
    "社交压力": "social_pressure",
    "学业失败": "academic_failure",
}


@dataclass
class SnapshotRow:
    file: str
    sim_time: str
    sim_dt: Optional[datetime]
    step: int
    agent: str
    doctor: str
    pair_key: str
    dynamic_enabled: bool
    dynamic_state: str
    dynamic_state_severity_index: int
    dynamic_interaction_count: int
    dynamic_state_history_len: int
    dynamic_overall_stress: Optional[float]
    dynamic_env_stress_level: Optional[float]
    dynamic_env_comfort_level: Optional[float]
    dynamic_env_trigger_potential: Optional[float]
    dynamic_trigger_count: int
    dynamic_trigger_names: str
    dynamic_active_biases_count: int
    dynamic_activated_memory_count: int
    dynamic_accumulated_triggers: Dict[str, float]
    dynamic_last_update_time: str
    intervention_lock_enabled: bool
    intervention_lock_meeting_id: str
    forced_tasks_total: int
    forced_tasks_pending: int
    forced_tasks_scheduled: int
    forced_tasks_done: int
    forced_tasks_other: int
    forced_tasks_overdue_open: int
    session_current_index: int
    session_current_session: str
    session_completed: bool
    session_last_detected_end: bool
    active_meetings_count: int
    doctor_queue_size: int
    consult_records_total: int
    consult_pair_records_total: int
    consult_write_audit_total: int
    consult_write_success_total: int
    consult_write_failed_total: int
    consult_pair_mismatch_total: int


def _safe_get(data: Any, path: Iterable[str], default: Any = None) -> Any:
    cur = data
    for key in path:
        if isinstance(cur, dict):
            cur = cur.get(key, default)
        else:
            cur = getattr(cur, key, default)
        if cur is default:
            return default
    return cur


def _to_dict(obj: Any) -> Dict[str, Any]:
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "__dict__"):
        return dict(obj.__dict__)
    return {}


def _to_list(obj: Any) -> List[Any]:
    if obj is None:
        return []
    if isinstance(obj, list):
        return obj
    if isinstance(obj, tuple):
        return list(obj)
    return []


def _to_float(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)):
        return float(value)
    try:
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            return None
        return float(text)
    except (TypeError, ValueError):
        return None


def _infer_state_severity_index(state: str) -> int:
    s = (state or "").strip().lower()
    if not s:
        return -1
    exact = {
        "normal": 0,
        "stable": 0,
        "baseline": 0,
        "mild_episode": 1,
        "moderate_episode": 2,
        "severe_episode": 3,
        "crisis": 4,
    }
    if s in exact:
        return exact[s]
    if "mild" in s:
        return 1
    if "moderate" in s:
        return 2
    if "severe" in s:
        return 3
    if "crisis" in s:
        return 4
    return -1


def _is_support_key(key: str) -> bool:
    k = (key or "").strip().lower()
    if not k:
        return False
    return ("support" in k) or ("支持" in key)


def _trigger_plot_label(key: str) -> str:
    if key in TRIGGER_PLOT_ALIASES:
        return TRIGGER_PLOT_ALIASES[key]
    return key


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _improvement_status_label(delta: float) -> str:
    if delta >= 15.0:
        return "显著改善"
    if delta >= 5.0:
        return "轻度改善"
    if delta > -5.0:
        return "基本持平"
    if delta > -15.0:
        return "轻度恶化"
    return "显著恶化"


def _parse_sim_dt(value: str) -> Optional[datetime]:
    if not value:
        return None
    for fmt in (TIME_FMT_PRIMARY, TIME_FMT_ALT):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    return None


def _parse_task_due(task: Dict[str, Any]) -> Optional[datetime]:
    date_text = str(task.get("date", "") or "").strip()
    time_text = str(task.get("time", "") or "").strip()
    if not date_text or not time_text:
        return None
    try:
        d = datetime.strptime(date_text, TASK_DATE_FMT).date()
        t = datetime.strptime(time_text, TASK_TIME_FMT).time()
        return datetime.combine(d, t)
    except ValueError:
        return None


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _find_snapshots(archive_dir: Path) -> List[Path]:
    files = [p for p in archive_dir.glob("simulate-*.json") if p.is_file()]
    files.sort(key=lambda p: p.name)
    return files


def _resolve_doctor(snapshot: Dict[str, Any], doctor_arg: str) -> str:
    if doctor_arg:
        return doctor_arg
    return str(_safe_get(snapshot, ["intervention", "doctor"], "") or "").strip()


def _resolve_pair_key(
    snapshot: Dict[str, Any],
    agent: str,
    doctor: str,
    pair_arg: str,
) -> str:
    pairs = _to_dict(_safe_get(snapshot, ["intervention_state", "session_prompt_state", "pairs"], {}))
    pair_keys = list(pairs.keys())
    if pair_arg and pair_arg in pairs:
        return pair_arg
    if doctor:
        exact = f"{doctor}::{agent}"
        if exact in pairs:
            return exact
    if agent:
        with_agent = [k for k in pair_keys if agent in k]
        if len(with_agent) == 1:
            return with_agent[0]
        if doctor:
            with_both = [k for k in with_agent if doctor in k]
            if with_both:
                return with_both[0]
    return pair_keys[0] if pair_keys else ""


def _count_write_audit(write_audit: List[Dict[str, Any]]) -> Tuple[int, int, int]:
    success = 0
    failed = 0
    pair_mismatch = 0
    for entry in write_audit:
        status = str(entry.get("status", "") or "").strip().lower()
        reason = str(entry.get("reason", "") or "").strip().lower()
        if status == "success":
            success += 1
        elif status:
            failed += 1
        if reason == "pair_mismatch":
            pair_mismatch += 1
    return success, failed, pair_mismatch


def _count_pair_records(records_by_id: Dict[str, Any], doctor: str, agent: str) -> int:
    total = 0
    for _, rec in records_by_id.items():
        participants = _to_dict(rec.get("participants", {}))
        d = str(participants.get("doctor", "") or "").strip()
        p = str(participants.get("patient", "") or "").strip()
        if d == doctor and p == agent:
            total += 1
    return total


def _count_queue_size(snapshot: Dict[str, Any], doctor: str) -> int:
    queues = _to_dict(_safe_get(snapshot, ["intervention_state", "doctor_meeting_queues"], {}))
    if doctor and doctor in queues:
        return len(_to_list(queues.get(doctor)))
    total = 0
    for _, val in queues.items():
        total += len(_to_list(val))
    return total


def _state_counts(tasks: List[Dict[str, Any]]) -> Tuple[int, int, int, int]:
    pending = 0
    scheduled = 0
    done = 0
    other = 0
    for task in tasks:
        state = str(task.get("state", "") or "").strip().lower()
        if state == "pending":
            pending += 1
        elif state == "scheduled":
            scheduled += 1
        elif state in {"done", "completed", "finished"}:
            done += 1
        else:
            other += 1
    return pending, scheduled, done, other


def _count_overdue_open_tasks(tasks: List[Dict[str, Any]], snapshot_dt: Optional[datetime]) -> int:
    if snapshot_dt is None:
        return 0
    count = 0
    for task in tasks:
        state = str(task.get("state", "") or "").strip().lower()
        if state not in {"pending", "scheduled"}:
            continue
        due = _parse_task_due(task)
        if due is not None and due < snapshot_dt:
            count += 1
    return count


def _extract_row(
    snapshot: Dict[str, Any],
    file_name: str,
    agent: str,
    doctor: str,
    pair_key: str,
) -> SnapshotRow:
    sim_time = str(snapshot.get("time", "") or "").strip()
    sim_dt = _parse_sim_dt(sim_time)
    step = int(snapshot.get("step", -1) or -1)

    agents = _to_dict(snapshot.get("agents", {}))
    agent_obj = _to_dict(agents.get(agent, {}))
    dynamic_state = _to_dict(agent_obj.get("depression_dynamic_state", {}))
    runtime = _to_dict(dynamic_state.get("runtime", {}))
    state_machine = _to_dict(runtime.get("state_machine", {}))
    context_analyzer = _to_dict(runtime.get("context_analyzer", {}))
    current_context = _to_dict(context_analyzer.get("current_context", {}))
    env = _to_dict(current_context.get("environment", {}))
    bias_injector = _to_dict(runtime.get("bias_injector", {}))
    memory_system = _to_dict(runtime.get("memory_system", {}))
    accumulated_raw = _to_dict(state_machine.get("accumulated_triggers", {}))
    accumulated_triggers: Dict[str, float] = {}
    for k, v in accumulated_raw.items():
        fv = _to_float(v)
        accumulated_triggers[str(k)] = fv if fv is not None else 0.0
    trigger_names = [str(x) for x in _to_list(current_context.get("triggers", [])) if str(x).strip()]
    current_state = str(state_machine.get("current_state", "") or "")

    intervention_status = _to_dict(_safe_get(agent_obj, ["status", "intervention"], {}))
    lock = _to_dict(intervention_status.get("lock", {}))
    forced_tasks = _to_list(intervention_status.get("forced_tasks", []))

    pending, scheduled, done, other = _state_counts([_to_dict(t) for t in forced_tasks])
    overdue_open = _count_overdue_open_tasks([_to_dict(t) for t in forced_tasks], sim_dt)

    active_meetings = _to_dict(_safe_get(snapshot, ["intervention_state", "active_meetings"], {}))
    session_pairs = _to_dict(_safe_get(snapshot, ["intervention_state", "session_prompt_state", "pairs"], {}))
    pair_state = _to_dict(session_pairs.get(pair_key, {}))

    consult_state = _to_dict(_safe_get(snapshot, ["intervention_state", "consult_record_state"], {}))
    records_by_id = _to_dict(consult_state.get("records_by_id", {}))
    write_audit = [_to_dict(x) for x in _to_list(consult_state.get("write_audit", []))]
    write_success, write_failed, pair_mismatch = _count_write_audit(write_audit)

    return SnapshotRow(
        file=file_name,
        sim_time=sim_time,
        sim_dt=sim_dt,
        step=step,
        agent=agent,
        doctor=doctor,
        pair_key=pair_key,
        dynamic_enabled=bool(dynamic_state.get("enabled", False)),
        dynamic_state=current_state,
        dynamic_state_severity_index=_infer_state_severity_index(current_state),
        dynamic_interaction_count=int(runtime.get("interaction_count", 0) or 0),
        dynamic_state_history_len=len(_to_list(state_machine.get("state_history", []))),
        dynamic_overall_stress=_to_float(current_context.get("overall_stress")),
        dynamic_env_stress_level=_to_float(env.get("stress_level")),
        dynamic_env_comfort_level=_to_float(env.get("comfort_level")),
        dynamic_env_trigger_potential=_to_float(env.get("trigger_potential")),
        dynamic_trigger_count=len(_to_list(current_context.get("triggers", []))),
        dynamic_trigger_names="|".join(trigger_names),
        dynamic_active_biases_count=len(_to_list(bias_injector.get("active_biases", []))),
        dynamic_activated_memory_count=len(_to_list(memory_system.get("activated_memory_ids", []))),
        dynamic_accumulated_triggers=accumulated_triggers,
        dynamic_last_update_time=str(runtime.get("last_update_time", "") or ""),
        intervention_lock_enabled=bool(lock.get("enabled", False)),
        intervention_lock_meeting_id=str(lock.get("meeting_id", "") or ""),
        forced_tasks_total=len(forced_tasks),
        forced_tasks_pending=pending,
        forced_tasks_scheduled=scheduled,
        forced_tasks_done=done,
        forced_tasks_other=other,
        forced_tasks_overdue_open=overdue_open,
        session_current_index=int(pair_state.get("current_index", -1) or -1),
        session_current_session=str(pair_state.get("current_session", "") or ""),
        session_completed=bool(pair_state.get("completed", False)),
        session_last_detected_end=bool(pair_state.get("last_detected_end", False)),
        active_meetings_count=len(active_meetings),
        doctor_queue_size=_count_queue_size(snapshot, doctor),
        consult_records_total=len(records_by_id),
        consult_pair_records_total=_count_pair_records(records_by_id, doctor, agent),
        consult_write_audit_total=len(write_audit),
        consult_write_success_total=write_success,
        consult_write_failed_total=write_failed,
        consult_pair_mismatch_total=pair_mismatch,
    )


def _build_events(rows: List[SnapshotRow]) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    for idx in range(1, len(rows)):
        prev = rows[idx - 1]
        cur = rows[idx]

        if cur.dynamic_interaction_count > prev.dynamic_interaction_count:
            events.append(
                {
                    "file": cur.file,
                    "step": cur.step,
                    "sim_time": cur.sim_time,
                    "event": "dynamic_interaction_increment",
                    "from": prev.dynamic_interaction_count,
                    "to": cur.dynamic_interaction_count,
                }
            )
        if cur.session_current_index > prev.session_current_index:
            events.append(
                {
                    "file": cur.file,
                    "step": cur.step,
                    "sim_time": cur.sim_time,
                    "event": "session_index_advanced",
                    "from": prev.session_current_index,
                    "to": cur.session_current_index,
                }
            )
        if (not prev.session_completed) and cur.session_completed:
            events.append(
                {
                    "file": cur.file,
                    "step": cur.step,
                    "sim_time": cur.sim_time,
                    "event": "session_completed",
                    "from": False,
                    "to": True,
                }
            )
        if prev.intervention_lock_enabled and (not cur.intervention_lock_enabled):
            events.append(
                {
                    "file": cur.file,
                    "step": cur.step,
                    "sim_time": cur.sim_time,
                    "event": "lock_released",
                    "from": True,
                    "to": False,
                }
            )
    return events


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with path.open("w", encoding="utf-8", newline="") as f:
            f.write("")
        return
    keys = list(rows[0].keys())
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _collect_trigger_keys(rows: List[SnapshotRow]) -> List[str]:
    keys = set()
    for r in rows:
        keys.update(r.dynamic_accumulated_triggers.keys())
    return sorted(keys)


def _rows_to_dicts(rows: List[SnapshotRow], trigger_keys: List[str]) -> List[Dict[str, Any]]:
    result: List[Dict[str, Any]] = []
    prev_trigger_values = {k: 0.0 for k in trigger_keys}
    start_wellbeing: Optional[float] = None
    prev_wellbeing: Optional[float] = None
    for r in rows:
        d = {
            "file": r.file,
            "sim_time": r.sim_time,
            "step": r.step,
            "agent": r.agent,
            "doctor": r.doctor,
            "pair_key": r.pair_key,
            "dynamic_enabled": r.dynamic_enabled,
            "dynamic_state": r.dynamic_state,
            "dynamic_state_severity_index": r.dynamic_state_severity_index,
            "dynamic_interaction_count": r.dynamic_interaction_count,
            "dynamic_state_history_len": r.dynamic_state_history_len,
            "dynamic_overall_stress": r.dynamic_overall_stress,
            "dynamic_env_stress_level": r.dynamic_env_stress_level,
            "dynamic_env_comfort_level": r.dynamic_env_comfort_level,
            "dynamic_env_trigger_potential": r.dynamic_env_trigger_potential,
            "dynamic_trigger_count": r.dynamic_trigger_count,
            "dynamic_trigger_names": r.dynamic_trigger_names,
            "dynamic_active_biases_count": r.dynamic_active_biases_count,
            "dynamic_activated_memory_count": r.dynamic_activated_memory_count,
            "dynamic_last_update_time": r.dynamic_last_update_time,
            "intervention_lock_enabled": r.intervention_lock_enabled,
            "intervention_lock_meeting_id": r.intervention_lock_meeting_id,
            "forced_tasks_total": r.forced_tasks_total,
            "forced_tasks_pending": r.forced_tasks_pending,
            "forced_tasks_scheduled": r.forced_tasks_scheduled,
            "forced_tasks_done": r.forced_tasks_done,
            "forced_tasks_other": r.forced_tasks_other,
            "forced_tasks_overdue_open": r.forced_tasks_overdue_open,
            "session_current_index": r.session_current_index,
            "session_current_session": r.session_current_session,
            "session_completed": r.session_completed,
            "session_last_detected_end": r.session_last_detected_end,
            "active_meetings_count": r.active_meetings_count,
            "doctor_queue_size": r.doctor_queue_size,
            "consult_records_total": r.consult_records_total,
            "consult_pair_records_total": r.consult_pair_records_total,
            "consult_write_audit_total": r.consult_write_audit_total,
            "consult_write_success_total": r.consult_write_success_total,
            "consult_write_failed_total": r.consult_write_failed_total,
            "consult_pair_mismatch_total": r.consult_pair_mismatch_total,
        }

        trigger_total = 0.0
        trigger_support = 0.0
        support_delta_sum = 0.0
        negative_delta_sum = 0.0
        for key in trigger_keys:
            cur = float(r.dynamic_accumulated_triggers.get(key, 0.0) or 0.0)
            prev = float(prev_trigger_values.get(key, 0.0) or 0.0)
            delta = cur - prev
            d[f"dynamic_trigger_accum__{key}"] = cur
            d[f"dynamic_trigger_delta__{key}"] = delta
            trigger_total += cur
            if _is_support_key(key):
                trigger_support += cur
                support_delta_sum += delta
            else:
                negative_delta_sum += delta
            prev_trigger_values[key] = cur

        trigger_negative = trigger_total - trigger_support
        balance = trigger_support - trigger_negative
        d["dynamic_trigger_accum_total"] = trigger_total
        d["dynamic_trigger_accum_support_total"] = trigger_support
        d["dynamic_trigger_accum_negative_total"] = trigger_negative
        d["dynamic_trigger_support_minus_negative"] = balance
        d["dynamic_trigger_support_delta_sum"] = support_delta_sum
        d["dynamic_trigger_negative_delta_sum"] = negative_delta_sum

        # Intuitive scoring layer for easier trend reading.
        severity_score = 50.0
        if r.dynamic_state_severity_index >= 0:
            severity_score = ((4.0 - float(r.dynamic_state_severity_index)) / 4.0) * 100.0
        stress_score = 50.0
        if r.dynamic_overall_stress is not None:
            stress_score = (1.0 - _clamp(float(r.dynamic_overall_stress), 0.0, 1.0)) * 100.0

        if trigger_total <= 0.0:
            balance_score = 50.0
        else:
            balance_score = _clamp((trigger_support / trigger_total) * 100.0, 0.0, 100.0)

        momentum_total = abs(support_delta_sum) + abs(negative_delta_sum)
        if momentum_total <= 1e-9:
            momentum_score = 50.0
        else:
            momentum_score = _clamp(
                ((support_delta_sum - negative_delta_sum) / momentum_total + 1.0) * 50.0,
                0.0,
                100.0,
            )

        wellbeing = (
            severity_score * 0.35
            + stress_score * 0.30
            + balance_score * 0.25
            + momentum_score * 0.10
        )
        risk = 100.0 - wellbeing

        if start_wellbeing is None:
            start_wellbeing = wellbeing
        improvement_vs_start = wellbeing - start_wellbeing
        change_vs_prev = 0.0 if prev_wellbeing is None else wellbeing - prev_wellbeing
        status_label = _improvement_status_label(improvement_vs_start)
        prev_wellbeing = wellbeing

        d["dynamic_wellbeing_index"] = round(wellbeing, 3)
        d["dynamic_risk_index"] = round(risk, 3)
        d["dynamic_improvement_score_vs_start"] = round(improvement_vs_start, 3)
        d["dynamic_improvement_change_vs_prev"] = round(change_vs_prev, 3)
        d["dynamic_improvement_status"] = status_label
        d["dynamic_score_component_severity"] = round(severity_score, 3)
        d["dynamic_score_component_stress"] = round(stress_score, 3)
        d["dynamic_score_component_balance"] = round(balance_score, 3)
        d["dynamic_score_component_momentum"] = round(momentum_score, 3)
        result.append(d)
    return result


def _rows_to_dynamic_dicts(base_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    keep_prefixes = (
        "dynamic_",
        "file",
        "sim_time",
        "step",
        "agent",
    )
    out: List[Dict[str, Any]] = []
    for r in base_rows:
        d: Dict[str, Any] = {}
        for k, v in r.items():
            if k == "doctor" or k == "pair_key":
                continue
            if k.startswith(keep_prefixes):
                d[k] = v
        out.append(d)
    return out


def _make_plots(rows: List[SnapshotRow], events: List[Dict[str, Any]], out_path: Path) -> None:
    if not rows:
        return

    x = list(range(len(rows)))
    x_labels = [f"{r.step}" for r in rows]
    dyn_count = [r.dynamic_interaction_count for r in rows]
    pair_records = [r.consult_pair_records_total for r in rows]
    session_idx = [r.session_current_index for r in rows]
    session_done = [1 if r.session_completed else 0 for r in rows]
    lock_enabled = [1 if r.intervention_lock_enabled else 0 for r in rows]
    ft_pending = [r.forced_tasks_pending for r in rows]
    ft_scheduled = [r.forced_tasks_scheduled for r in rows]
    ft_done = [r.forced_tasks_done for r in rows]
    ft_other = [r.forced_tasks_other for r in rows]
    stress = [r.dynamic_overall_stress if r.dynamic_overall_stress is not None else float("nan") for r in rows]
    trigger_count = [r.dynamic_trigger_count for r in rows]
    queue_size = [r.doctor_queue_size for r in rows]

    fig, axes = plt.subplots(4, 1, figsize=(14, 14), sharex=True)
    fig.suptitle("Dynamic Persona & Forced Intervention Analysis", fontsize=14)

    ax1 = axes[0]
    ax1.plot(x, dyn_count, marker="o", label="dynamic_interaction_count")
    ax1.set_ylabel("Dynamic Count")
    ax1.grid(alpha=0.2)
    ax1b = ax1.twinx()
    ax1b.plot(x, pair_records, marker="s", color="#d62728", label="consult_pair_records_total")
    ax1b.set_ylabel("Consult Records")
    l1, lb1 = ax1.get_legend_handles_labels()
    l2, lb2 = ax1b.get_legend_handles_labels()
    ax1.legend(l1 + l2, lb1 + lb2, loc="upper left")

    ax2 = axes[1]
    ax2.plot(x, session_idx, marker="o", label="session_current_index")
    ax2.plot(x, session_done, marker="x", label="session_completed(0/1)")
    ax2.plot(x, lock_enabled, marker="^", label="lock_enabled(0/1)")
    ax2.set_ylabel("Session/Lock")
    ax2.grid(alpha=0.2)
    ax2.legend(loc="upper left")

    ax3 = axes[2]
    ax3.bar(x, ft_pending, label="forced_tasks_pending")
    ax3.bar(x, ft_scheduled, bottom=ft_pending, label="forced_tasks_scheduled")
    bottom_2 = [a + b for a, b in zip(ft_pending, ft_scheduled)]
    ax3.bar(x, ft_done, bottom=bottom_2, label="forced_tasks_done")
    bottom_3 = [a + b for a, b in zip(bottom_2, ft_done)]
    ax3.bar(x, ft_other, bottom=bottom_3, label="forced_tasks_other")
    ax3.set_ylabel("Forced Tasks")
    ax3.grid(alpha=0.2)
    ax3.legend(loc="upper left")

    ax4 = axes[3]
    ax4.plot(x, stress, marker="o", label="dynamic_overall_stress")
    ax4.plot(x, trigger_count, marker="s", label="dynamic_trigger_count")
    ax4.plot(x, queue_size, marker="^", label="doctor_queue_size")
    ax4.set_ylabel("Stress/Trigger/Queue")
    ax4.grid(alpha=0.2)
    ax4.legend(loc="upper left")
    ax4.set_xlabel("Snapshot Index (tick label = step)")

    for ev in events:
        if ev["event"] in {"session_index_advanced", "session_completed"}:
            try:
                step = int(ev["step"])
                idx = next(i for i, r in enumerate(rows) if r.step == step and r.sim_time == ev["sim_time"])
                for ax in axes:
                    ax.axvline(idx, color="#999999", alpha=0.15)
            except Exception:
                continue

    plt.xticks(x, x_labels, rotation=45)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160)
    plt.close(fig)


def _make_dynamic_state_plots(
    rows: List[SnapshotRow],
    trigger_keys: List[str],
    dynamic_rows: List[Dict[str, Any]],
    out_path: Path,
) -> None:
    if not rows:
        return

    x = list(range(len(rows)))
    x_labels = [f"{r.step}" for r in rows]

    overall = [r.dynamic_overall_stress if r.dynamic_overall_stress is not None else float("nan") for r in rows]
    env_stress = [r.dynamic_env_stress_level if r.dynamic_env_stress_level is not None else float("nan") for r in rows]
    env_comfort = [r.dynamic_env_comfort_level if r.dynamic_env_comfort_level is not None else float("nan") for r in rows]
    env_trigger = [
        r.dynamic_env_trigger_potential if r.dynamic_env_trigger_potential is not None else float("nan")
        for r in rows
    ]
    wellbeing = [float(d.get("dynamic_wellbeing_index", float("nan")) or float("nan")) for d in dynamic_rows]
    risk = [float(d.get("dynamic_risk_index", float("nan")) or float("nan")) for d in dynamic_rows]
    improvement = [
        float(d.get("dynamic_improvement_score_vs_start", float("nan")) or float("nan")) for d in dynamic_rows
    ]
    support_delta = [
        float(d.get("dynamic_trigger_support_delta_sum", float("nan")) or float("nan")) for d in dynamic_rows
    ]
    negative_delta = [
        float(d.get("dynamic_trigger_negative_delta_sum", float("nan")) or float("nan")) for d in dynamic_rows
    ]

    accum_by_key = {
        key: [float(r.dynamic_accumulated_triggers.get(key, 0.0) or 0.0) for r in rows]
        for key in trigger_keys
    }
    support_total = []
    negative_total = []
    balance = []
    for i in range(len(rows)):
        st = 0.0
        ng = 0.0
        for key in trigger_keys:
            v = accum_by_key[key][i]
            if _is_support_key(key):
                st += v
            else:
                ng += v
        support_total.append(st)
        negative_total.append(ng)
        balance.append(st - ng)

    # Plot top trigger keys by final accumulated value.
    sorted_keys = sorted(
        trigger_keys,
        key=lambda k: accum_by_key.get(k, [0.0])[-1] if accum_by_key.get(k) else 0.0,
        reverse=True,
    )
    top_keys = sorted_keys[:6]

    fig, axes = plt.subplots(4, 1, figsize=(14, 15), sharex=True)
    fig.suptitle("Depression Dynamic State Deep Dive", fontsize=14)

    ax1 = axes[0]
    ax1.plot(x, overall, marker="o", label="overall_stress")
    ax1.plot(x, env_stress, marker="s", label="env.stress_level")
    ax1.plot(x, env_trigger, marker="^", label="env.trigger_potential")
    ax1.plot(x, env_comfort, marker="d", label="env.comfort_level")
    ax1.set_ylabel("Stress/Comfort")
    ax1.grid(alpha=0.2)
    ax1.legend(loc="upper left")

    ax2 = axes[1]
    for key in top_keys:
        ax2.plot(
            x,
            accum_by_key[key],
            marker="o",
            linewidth=1.4,
            label=f"accum:{_trigger_plot_label(key)}",
        )
    ax2.set_ylabel("Accumulated Triggers")
    ax2.grid(alpha=0.2)
    if top_keys:
        ax2.legend(loc="upper left")

    ax3 = axes[2]
    ax3.plot(x, support_total, marker="o", label="support_total")
    ax3.plot(x, negative_total, marker="s", label="negative_total")
    ax3.plot(x, balance, marker="^", label="support_minus_negative")
    ax3.set_ylabel("Support vs Negative")
    ax3.grid(alpha=0.2)
    ax3.legend(loc="upper left")

    ax4 = axes[3]
    ax4.plot(x, wellbeing, marker="o", label="wellbeing_index(0-100)")
    ax4.plot(x, risk, marker="s", label="risk_index(0-100)")
    ax4.plot(x, improvement, marker="^", label="improvement_vs_start")
    ax4.plot(x, support_delta, linestyle="--", label="support_delta_sum")
    ax4.plot(x, negative_delta, linestyle="--", label="negative_delta_sum")
    ax4.axhline(0.0, color="#666666", linewidth=0.8, alpha=0.5)
    ax4.set_ylabel("Intuitive Scores")
    ax4.grid(alpha=0.2)
    ax4.legend(loc="upper left")
    ax4.set_xlabel("Snapshot Index (tick label = step)")

    plt.xticks(x, x_labels, rotation=45)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160)
    plt.close(fig)


def _build_summary_text(
    archive_name: str,
    rows: List[SnapshotRow],
    events: List[Dict[str, Any]],
    trigger_keys: List[str],
    row_dicts: List[Dict[str, Any]],
) -> str:
    if not rows:
        return "# 分析摘要\n\n未发现可用快照。"

    first = rows[0]
    last = rows[-1]
    unique_states = sorted({r.dynamic_state for r in rows if r.dynamic_state})
    state_counter = Counter(r.dynamic_state for r in rows if r.dynamic_state)
    state_counter_text = ", ".join(f"{k}:{v}" for k, v in state_counter.items())

    event_counter = Counter(ev["event"] for ev in events)
    event_text = ", ".join(f"{k}:{v}" for k, v in event_counter.items()) if event_counter else "无"

    first_score = row_dicts[0] if row_dicts else {}
    last_score = row_dicts[-1] if row_dicts else {}
    wellbeing_start = first_score.get("dynamic_wellbeing_index", "")
    wellbeing_end = last_score.get("dynamic_wellbeing_index", "")
    risk_start = first_score.get("dynamic_risk_index", "")
    risk_end = last_score.get("dynamic_risk_index", "")
    improve_end = last_score.get("dynamic_improvement_score_vs_start", "")
    improve_status = str(last_score.get("dynamic_improvement_status", "未知"))

    stress_values = [r.dynamic_overall_stress for r in rows if r.dynamic_overall_stress is not None]
    stress_text = "无有效值"
    if stress_values:
        stress_text = f"min={min(stress_values):.3f}, max={max(stress_values):.3f}, mean={sum(stress_values) / len(stress_values):.3f}"

    first_support_total = 0.0
    first_negative_total = 0.0
    last_support_total = 0.0
    last_negative_total = 0.0
    trigger_change_lines: List[str] = []
    for key in trigger_keys:
        f = float(first.dynamic_accumulated_triggers.get(key, 0.0) or 0.0)
        l = float(last.dynamic_accumulated_triggers.get(key, 0.0) or 0.0)
        delta = l - f
        trigger_change_lines.append(f"- `{key}`：`{f:.3f}` -> `{l:.3f}`（增量 `{delta:.3f}`）")
        if _is_support_key(key):
            first_support_total += f
            last_support_total += l
        else:
            first_negative_total += f
            last_negative_total += l

    first_balance = first_support_total - first_negative_total
    last_balance = last_support_total - last_negative_total

    state_trend_text = "无法判断"
    if first.dynamic_state_severity_index >= 0 and last.dynamic_state_severity_index >= 0:
        if last.dynamic_state_severity_index < first.dynamic_state_severity_index:
            state_trend_text = "严重度下降（可能改善）"
        elif last.dynamic_state_severity_index > first.dynamic_state_severity_index:
            state_trend_text = "严重度上升（可能恶化）"
        else:
            state_trend_text = "严重度未变化"

    lines = [
        "# 动态抑郁人设变化量化摘要",
        "",
        f"- 存档：`{archive_name}`",
        f"- 角色：`{last.agent}`",
        f"- 医生：`{last.doctor}`",
        f"- 配对键：`{last.pair_key}`",
        f"- 快照范围：`{first.file}` -> `{last.file}`",
        f"- 步数范围：`{first.step}` -> `{last.step}`",
        "",
        "## 零、直观评分（优先看这里）",
        f"- wellbeing_index：`{wellbeing_start}` -> `{wellbeing_end}`（0-100，越高越好）",
        f"- risk_index：`{risk_start}` -> `{risk_end}`（0-100，越高风险越大）",
        f"- improvement_vs_start：`{improve_end}`（>0 改善，<0 恶化）",
        f"- 状态判定：`{improve_status}`",
        "",
        "## 一、动态抑郁状态变化",
        f"- dynamic_interaction_count：`{first.dynamic_interaction_count}` -> `{last.dynamic_interaction_count}`（增量 `{last.dynamic_interaction_count - first.dynamic_interaction_count}`）",
        f"- dynamic_state（出现过）：`{', '.join(unique_states) if unique_states else '空'}`",
        f"- dynamic_state 分布：{state_counter_text if state_counter_text else '空'}",
        f"- 状态严重度索引：`{first.dynamic_state_severity_index}` -> `{last.dynamic_state_severity_index}`（{state_trend_text}）",
        f"- dynamic_overall_stress 统计：{stress_text}",
        f"- 环境压力值（末值）：stress=`{last.dynamic_env_stress_level}` / comfort=`{last.dynamic_env_comfort_level}` / trigger_potential=`{last.dynamic_env_trigger_potential}`",
        "",
        "## 二、depression_dynamic_state 数值核心（accumulated_triggers）",
        f"- support 总量：`{first_support_total:.3f}` -> `{last_support_total:.3f}`",
        f"- 负向触发总量：`{first_negative_total:.3f}` -> `{last_negative_total:.3f}`",
        f"- support-负向 平衡值：`{first_balance:.3f}` -> `{last_balance:.3f}`",
        "- 分项变化：",
        *trigger_change_lines,
        "",
        "## 三、强制干预对话效果（可观测）",
        f"- session_current_index：`{first.session_current_index}` -> `{last.session_current_index}`",
        f"- session_current_session：`{first.session_current_session}` -> `{last.session_current_session}`",
        f"- session_completed：`{first.session_completed}` -> `{last.session_completed}`",
        f"- consult_pair_records_total：`{first.consult_pair_records_total}` -> `{last.consult_pair_records_total}`",
        f"- intervention_lock_enabled：`{first.intervention_lock_enabled}` -> `{last.intervention_lock_enabled}`",
        "",
        "## 四、任务与积压",
        f"- forced_tasks_total：`{first.forced_tasks_total}` -> `{last.forced_tasks_total}`",
        f"- forced_tasks_pending（末值）：`{last.forced_tasks_pending}`",
        f"- forced_tasks_scheduled（末值）：`{last.forced_tasks_scheduled}`",
        f"- forced_tasks_overdue_open（末值）：`{last.forced_tasks_overdue_open}`",
        "",
        "## 五、关键事件计数",
        f"- {event_text}",
        "",
        "## 六、建议重点查看",
        "- `dynamic_state_timeline.csv`：专门看 `depression_dynamic_state` 数值变化（包含每个 trigger 的累计值和单步增量）。",
        "- `timeline.csv`：全链路时间线（动态抑郁 + 强制干预）。",
        "- `events.csv`：关键跳变点（会话推进/完成、锁释放、动态互动增长）。",
        "- `depression_state_deep_dive.png`：动态抑郁状态深度图（状态数值变化主图）。",
        "- `persona_effect_plots.png`：原四联图（干预链路效果主图）。",
    ]
    return "\n".join(lines) + "\n"
def analyze_archive(
    checkpoints_root: Path,
    archive_name: str,
    agent: str,
    doctor_arg: str,
    pair_arg: str,
    output_dir: Path,
    no_plot: bool,
) -> Path:
    archive_dir = checkpoints_root / archive_name
    if not archive_dir.exists():
        raise FileNotFoundError(f"存档目录不存在: {archive_dir}")

    snapshots = _find_snapshots(archive_dir)
    if not snapshots:
        raise RuntimeError(f"未找到快照文件 simulate-*.json: {archive_dir}")

    first_snapshot = _load_json(snapshots[0])
    doctor = _resolve_doctor(first_snapshot, doctor_arg)
    pair_key = _resolve_pair_key(first_snapshot, agent, doctor, pair_arg)

    rows: List[SnapshotRow] = []
    for path in snapshots:
        data = _load_json(path)
        rows.append(_extract_row(data, path.name, agent, doctor, pair_key))

    trigger_keys = _collect_trigger_keys(rows)
    events = _build_events(rows)
    row_dicts = _rows_to_dicts(rows, trigger_keys)
    dynamic_rows = _rows_to_dynamic_dicts(row_dicts)
    output_dir.mkdir(parents=True, exist_ok=True)

    timeline_csv = output_dir / "timeline.csv"
    dynamic_timeline_csv = output_dir / "dynamic_state_timeline.csv"
    events_csv = output_dir / "events.csv"
    summary_md = output_dir / "summary.md"
    plot_png = output_dir / "persona_effect_plots.png"
    dynamic_plot_png = output_dir / "depression_state_deep_dive.png"

    _write_csv(timeline_csv, row_dicts)
    _write_csv(dynamic_timeline_csv, dynamic_rows)
    _write_csv(events_csv, events)

    summary_text = _build_summary_text(archive_name, rows, events, trigger_keys, row_dicts)
    summary_md.write_text(summary_text, encoding="utf-8")

    if not no_plot:
        _make_plots(rows, events, plot_png)
        _make_dynamic_state_plots(rows, trigger_keys, dynamic_rows, dynamic_plot_png)

    return output_dir


def _build_parser() -> argparse.ArgumentParser:
    base_dir = Path(__file__).resolve().parents[2]
    default_checkpoints_root = (
        Path(DEFAULT_CHECKPOINTS_ROOT).expanduser().resolve()
        if DEFAULT_CHECKPOINTS_ROOT
        else (base_dir / "results" / "checkpoints")
    )
    default_output_root = Path(__file__).resolve().parent / "analysis_outputs"

    parser = argparse.ArgumentParser(
        description="根据存档与角色，量化分析动态抑郁人设与强制干预对话效果。"
    )
    parser.add_argument("--archive", default=DEFAULT_ARCHIVE, help="存档名，例如 sim-dym-severe-kbd-0404")
    parser.add_argument("--agent", default=DEFAULT_AGENT, help="角色名，例如 卡布达")
    parser.add_argument(
        "--checkpoints-root",
        default=str(default_checkpoints_root),
        help=f"checkpoints 根目录（默认: {default_checkpoints_root}）",
    )
    parser.add_argument("--doctor", default=DEFAULT_DOCTOR, help="医生角色名（默认自动从快照读取）")
    parser.add_argument(
        "--pair-key",
        default=DEFAULT_PAIR_KEY,
        help="session pair 键，例如 蜻蜓队长::卡布达（默认自动推断）",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="输出目录（默认: customization/depression_scale_agent/analysis_outputs/<archive>_<agent>）",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="仅输出表格与摘要，不生成图片",
    )
    parser.add_argument(
        "--list-archives",
        action="store_true",
        help="列出可用存档并退出",
    )
    parser.add_argument(
        "--list-agents",
        action="store_true",
        help="列出指定存档中首个快照的角色并退出（需配合 --archive）",
    )
    parser.set_defaults(
        _default_output_root=str(default_output_root),
        no_plot=DEFAULT_NO_PLOT,
        list_archives=DEFAULT_LIST_ARCHIVES,
        list_agents=DEFAULT_LIST_AGENTS,
    )
    return parser


def _list_archives(checkpoints_root: Path) -> None:
    if not checkpoints_root.exists():
        print(f"[ERROR] checkpoints 根目录不存在: {checkpoints_root}")
        return
    archives = sorted([p.name for p in checkpoints_root.iterdir() if p.is_dir()])
    print("可用存档：")
    for name in archives:
        print(f"- {name}")


def _list_agents(checkpoints_root: Path, archive_name: str) -> None:
    archive_dir = checkpoints_root / archive_name
    snapshots = _find_snapshots(archive_dir)
    if not snapshots:
        print(f"[ERROR] 存档不存在或无快照: {archive_dir}")
        return
    snap = _load_json(snapshots[0])
    agents = list(_to_dict(snap.get("agents", {})).keys())
    print(f"存档 `{archive_name}` 可用角色：")
    for a in agents:
        print(f"- {a}")


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    checkpoints_root = Path(args.checkpoints_root).resolve()

    if args.list_archives:
        _list_archives(checkpoints_root)
        return

    if args.list_agents:
        if not args.archive:
            print("[ERROR] --list-agents 需要提供 --archive")
            return
        _list_agents(checkpoints_root, args.archive)
        return

    if not args.archive or not args.agent:
        print("[ERROR] 分析模式需要同时提供 --archive 与 --agent，或在脚本顶部设置 DEFAULT_ARCHIVE/DEFAULT_AGENT")
        return

    output_dir: Path
    if args.output_dir:
        output_dir = Path(args.output_dir).resolve()
    else:
        default_output_root = Path(args._default_output_root).resolve()
        safe_agent = re.sub(r"[\\/:*?\"<>|]", "_", args.agent)
        output_dir = default_output_root / f"{args.archive}_{safe_agent}"

    out = analyze_archive(
        checkpoints_root=checkpoints_root,
        archive_name=args.archive,
        agent=args.agent,
        doctor_arg=args.doctor,
        pair_arg=args.pair_key,
        output_dir=output_dir,
        no_plot=bool(args.no_plot),
    )

    print("[OK] 分析完成，输出目录：")
    print(out)
    print("- timeline.csv")
    print("- dynamic_state_timeline.csv")
    print("- events.csv")
    print("- summary.md")
    if not args.no_plot:
        print("- persona_effect_plots.png")
        print("- depression_state_deep_dive.png")


if __name__ == "__main__":
    main()
