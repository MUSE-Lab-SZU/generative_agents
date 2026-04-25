#!/usr/bin/env python3
"""
统计模拟快照中指定 agent 的动态抑郁状态变化。

默认目标:
- checkpoint 目录: results/checkpoints/sim0408
- agent: 卡布达

输出:
- 终端摘要
- 可选 CSV/JSON 文件
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Dict, List, Optional, Tuple


STATE_SEVERITY = {
    "remission": 0,
    "mild_episode": 1,
    "moderate_episode": 2,
    "severe_episode": 3,
    "crisis": 4,
}

STATE_LABEL = {
    "remission": "缓解期",
    "mild_episode": "轻度发作",
    "moderate_episode": "中度发作",
    "severe_episode": "重度发作",
    "crisis": "危机期",
}


@dataclass
class SnapshotRow:
    timestamp_raw: str
    timestamp_display: str
    filename: str
    current_state: str
    state_label: str
    state_severity: int
    overall_stress: Optional[float]
    interaction_count: Optional[int]
    trigger_count: int
    triggers: List[str]
    state_history_len: int
    location: str
    social_density: str
    time_of_day: str
    transition_sensitivity: Optional[float]
    minimum_state_duration: Optional[int]
    state_start_time: Optional[str]
    last_update_time: Optional[str]


@dataclass
class TransitionRow:
    at_timestamp: str
    from_state: str
    to_state: str
    from_label: str
    to_label: str
    severity_delta: int


def parse_timestamp_from_name(name: str) -> Tuple[str, str, Optional[datetime]]:
    """
    从 simulate-YYYYMMDD-HHMM.json 提取时间戳。
    返回 (raw, display, datetime_obj)
    """
    match = re.search(r"simulate-(\d{8}-\d{4})\.json$", name)
    if not match:
        return name, name, None

    raw = match.group(1)
    try:
        dt = datetime.strptime(raw, "%Y%m%d-%H%M")
        return raw, dt.strftime("%Y-%m-%d %H:%M"), dt
    except ValueError:
        return raw, raw, None


def normalize_state(state: Optional[str]) -> str:
    if not state:
        return "unknown"
    return str(state)


def state_severity(state: str) -> int:
    return STATE_SEVERITY.get(state, -1)


def state_label(state: str) -> str:
    return STATE_LABEL.get(state, state)


def _as_float(value: object) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_int(value: object) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def collect_rows(checkpoint_dir: Path, agent_name: str) -> Tuple[List[SnapshotRow], List[str]]:
    simulate_files = sorted(checkpoint_dir.glob("simulate-*.json"))
    rows: List[SnapshotRow] = []
    warnings: List[str] = []

    if not simulate_files:
        warnings.append(f"未找到 simulate-*.json 文件: {checkpoint_dir}")
        return rows, warnings

    parsed_files: List[Tuple[Optional[datetime], Path]] = []
    for path in simulate_files:
        _, _, dt = parse_timestamp_from_name(path.name)
        parsed_files.append((dt, path))

    parsed_files.sort(key=lambda item: (item[0] is None, item[0], item[1].name))

    for _, path in parsed_files:
        raw_ts, display_ts, _ = parse_timestamp_from_name(path.name)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"读取失败 {path.name}: {exc}")
            continue

        agent_payload = payload.get("agents", {}).get(agent_name)
        if not isinstance(agent_payload, dict):
            warnings.append(f"{path.name}: 未找到 agent={agent_name}")
            continue

        runtime = (
            agent_payload.get("depression_dynamic_state", {})
            .get("runtime", {})
        )
        if not isinstance(runtime, dict):
            warnings.append(f"{path.name}: depression_dynamic_state.runtime 缺失")
            continue

        sm = runtime.get("state_machine", {})
        if not isinstance(sm, dict):
            sm = {}

        context = runtime.get("context_analyzer", {}).get("current_context", {})
        if not isinstance(context, dict):
            context = {}
        env = context.get("environment", {})
        if not isinstance(env, dict):
            env = {}

        triggers_raw = context.get("triggers", [])
        if isinstance(triggers_raw, list):
            triggers = [str(item) for item in triggers_raw]
        elif triggers_raw is None:
            triggers = []
        else:
            triggers = [str(triggers_raw)]

        current_state = normalize_state(sm.get("current_state"))
        row = SnapshotRow(
            timestamp_raw=raw_ts,
            timestamp_display=display_ts,
            filename=path.name,
            current_state=current_state,
            state_label=state_label(current_state),
            state_severity=state_severity(current_state),
            overall_stress=_as_float(context.get("overall_stress")),
            interaction_count=_as_int(runtime.get("interaction_count")),
            trigger_count=len(triggers),
            triggers=triggers,
            state_history_len=len(sm.get("state_history", []))
            if isinstance(sm.get("state_history"), list)
            else 0,
            location=str(env.get("location", "")),
            social_density=str(env.get("social_density", "")),
            time_of_day=str(env.get("time_of_day", "")),
            transition_sensitivity=_as_float(sm.get("transition_sensitivity")),
            minimum_state_duration=_as_int(sm.get("minimum_state_duration")),
            state_start_time=str(sm.get("state_start_time", "")) or None,
            last_update_time=str(runtime.get("last_update_time", "")) or None,
        )
        rows.append(row)

    return rows, warnings


def compute_transitions(rows: List[SnapshotRow]) -> List[TransitionRow]:
    transitions: List[TransitionRow] = []
    for i in range(1, len(rows)):
        prev = rows[i - 1]
        curr = rows[i]
        if curr.current_state != prev.current_state:
            transitions.append(
                TransitionRow(
                    at_timestamp=curr.timestamp_display,
                    from_state=prev.current_state,
                    to_state=curr.current_state,
                    from_label=prev.state_label,
                    to_label=curr.state_label,
                    severity_delta=curr.state_severity - prev.state_severity,
                )
            )
    return transitions


def compute_stress_change_extremes(
    rows: List[SnapshotRow],
) -> Tuple[Optional[Dict[str, object]], Optional[Dict[str, object]]]:
    max_rise: Optional[Dict[str, object]] = None
    max_drop: Optional[Dict[str, object]] = None

    for i in range(1, len(rows)):
        prev = rows[i - 1]
        curr = rows[i]
        if prev.overall_stress is None or curr.overall_stress is None:
            continue
        delta = curr.overall_stress - prev.overall_stress
        item = {
            "from": prev.timestamp_display,
            "to": curr.timestamp_display,
            "delta": delta,
            "from_value": prev.overall_stress,
            "to_value": curr.overall_stress,
        }
        if max_rise is None or delta > float(max_rise["delta"]):
            max_rise = item
        if max_drop is None or delta < float(max_drop["delta"]):
            max_drop = item

    return max_rise, max_drop


def compute_interaction_changes(rows: List[SnapshotRow]) -> List[Dict[str, object]]:
    points: List[Dict[str, object]] = []
    for i in range(1, len(rows)):
        prev = rows[i - 1].interaction_count
        curr = rows[i].interaction_count
        if prev is None or curr is None:
            continue
        if curr != prev:
            points.append(
                {
                    "at": rows[i].timestamp_display,
                    "from": prev,
                    "to": curr,
                    "delta": curr - prev,
                }
            )
    return points


def summarize(rows: List[SnapshotRow]) -> Dict[str, object]:
    transitions = compute_transitions(rows)
    state_counter = Counter(row.current_state for row in rows)
    trigger_counter = Counter(trigger for row in rows for trigger in row.triggers)

    stress_rows = [row for row in rows if row.overall_stress is not None]
    stress_values = [row.overall_stress for row in stress_rows if row.overall_stress is not None]

    stress_summary: Dict[str, object] = {}
    if stress_values:
        min_idx, min_val = min(
            enumerate(stress_values), key=lambda item: item[1]
        )
        max_idx, max_val = max(
            enumerate(stress_values), key=lambda item: item[1]
        )
        stress_summary = {
            "count": len(stress_values),
            "mean": mean(stress_values),
            "min": min_val,
            "min_at": stress_rows[min_idx].timestamp_display,
            "max": max_val,
            "max_at": stress_rows[max_idx].timestamp_display,
        }

    max_rise, max_drop = compute_stress_change_extremes(rows)
    interaction_changes = compute_interaction_changes(rows)

    interaction_start = rows[0].interaction_count if rows else None
    interaction_end = rows[-1].interaction_count if rows else None
    interaction_delta = (
        interaction_end - interaction_start
        if isinstance(interaction_start, int) and isinstance(interaction_end, int)
        else None
    )

    return {
        "snapshot_count": len(rows),
        "first_timestamp": rows[0].timestamp_display if rows else None,
        "last_timestamp": rows[-1].timestamp_display if rows else None,
        "states": {
            "unique_in_order": list(dict.fromkeys(row.current_state for row in rows)),
            "counts": dict(state_counter),
        },
        "transition_count": len(transitions),
        "transitions": [asdict(item) for item in transitions],
        "stress": {
            **stress_summary,
            "max_rise": max_rise,
            "max_drop": max_drop,
        },
        "triggers": {
            "top": trigger_counter.most_common(20),
            "distinct_count": len(trigger_counter),
        },
        "interaction": {
            "start": interaction_start,
            "end": interaction_end,
            "delta": interaction_delta,
            "change_points": interaction_changes,
        },
    }


def print_report(
    rows: List[SnapshotRow],
    summary_data: Dict[str, object],
    show_timeline: bool,
    agent_name: str,
) -> None:
    if not rows:
        print("没有可用数据，无法生成报告。")
        return

    states = summary_data["states"]  # type: ignore[index]
    stress = summary_data["stress"]  # type: ignore[index]
    triggers = summary_data["triggers"]  # type: ignore[index]
    interaction = summary_data["interaction"]  # type: ignore[index]

    print(f"=== 动态抑郁状态统计: {agent_name} ===")
    print(
        f"时间范围: {summary_data['first_timestamp']} -> {summary_data['last_timestamp']}, "
        f"快照数: {summary_data['snapshot_count']}"
    )
    print(
        f"状态种类: {', '.join(states['unique_in_order']) if states['unique_in_order'] else '无'}; "
        f"状态转换次数: {summary_data['transition_count']}"
    )

    state_count_pairs = states["counts"].items() if isinstance(states.get("counts"), dict) else []
    if state_count_pairs:
        state_count_text = ", ".join(
            f"{state}={count}" for state, count in state_count_pairs
        )
        print(f"各状态出现次数: {state_count_text}")

    if stress.get("count"):
        print(
            "overall_stress: "
            f"均值={stress['mean']:.4f}, 最小={stress['min']:.4f}({stress['min_at']}), "
            f"最大={stress['max']:.4f}({stress['max_at']})"
        )
        if stress.get("max_rise"):
            rise = stress["max_rise"]
            print(
                "最大上升: "
                f"{rise['from']} -> {rise['to']}, "
                f"{rise['from_value']:.4f} -> {rise['to_value']:.4f}, "
                f"delta={rise['delta']:.4f}"
            )
        if stress.get("max_drop"):
            drop = stress["max_drop"]
            print(
                "最大下降: "
                f"{drop['from']} -> {drop['to']}, "
                f"{drop['from_value']:.4f} -> {drop['to_value']:.4f}, "
                f"delta={drop['delta']:.4f}"
            )

    top_triggers = triggers.get("top", [])
    if top_triggers:
        trigger_text = ", ".join(f"{name}:{count}" for name, count in top_triggers[:10])
        print(f"触发词Top10: {trigger_text}")

    print(
        f"interaction_count: {interaction.get('start')} -> {interaction.get('end')}, "
        f"delta={interaction.get('delta')}"
    )
    if interaction.get("change_points"):
        changes = interaction["change_points"]
        print("interaction_count 变化点:")
        for item in changes:
            print(
                f"  - {item['at']}: {item['from']} -> {item['to']} "
                f"(delta={item['delta']:+d})"
            )

    if show_timeline:
        print("\n--- 时间线 ---")
        print("timestamp         state             stress   interaction  trigger_count  triggers")
        for row in rows:
            stress_text = (
                f"{row.overall_stress:.4f}"
                if isinstance(row.overall_stress, float) and not math.isnan(row.overall_stress)
                else "None"
            )
            trigger_text = "|".join(row.triggers) if row.triggers else "-"
            print(
                f"{row.timestamp_display:<16} "
                f"{row.current_state:<17} "
                f"{stress_text:<8} "
                f"{str(row.interaction_count):<12} "
                f"{row.trigger_count:<13} "
                f"{trigger_text}"
            )


def write_csv_report(path: Path, rows: List[SnapshotRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=[
                "timestamp_raw",
                "timestamp_display",
                "filename",
                "current_state",
                "state_label",
                "state_severity",
                "overall_stress",
                "interaction_count",
                "trigger_count",
                "triggers",
                "state_history_len",
                "location",
                "social_density",
                "time_of_day",
                "transition_sensitivity",
                "minimum_state_duration",
                "state_start_time",
                "last_update_time",
            ],
        )
        writer.writeheader()
        for row in rows:
            data = asdict(row)
            data["triggers"] = "|".join(row.triggers)
            writer.writerow(data)


def write_json_report(path: Path, rows: List[SnapshotRow], summary_data: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "summary": summary_data,
        "timeline": [asdict(row) for row in rows],
    }
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="统计模拟运行结果中动态抑郁状态变化（默认 sim0408 / 卡布达）"
    )
    parser.add_argument(
        "--checkpoint-dir",
        default="results/checkpoints/sim-test-kbd-0421-2",
        help="包含 simulate-*.json 的目录",
    )
    parser.add_argument(
        "--agent",
        default="卡布达",
        help="agent 名称（默认: 卡布达）",
    )
    parser.add_argument(
        "--csv",
        default="results/checkpoints/sim-test-kbd-0421-2/judge_traces/depression_state_summary.csv",
        help="可选：输出 CSV 路径",
    )
    parser.add_argument(
        "--json",
        default="results/checkpoints/sim-test-kbd-0421-2/judge_traces/depression_state_summary.json",
        help="可选：输出 JSON 路径",
    )
    parser.add_argument(
        "--no-timeline",
        action="store_true",
        help="只输出摘要，不打印完整时间线",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    checkpoint_dir = Path(args.checkpoint_dir)

    rows, warnings = collect_rows(checkpoint_dir=checkpoint_dir, agent_name=args.agent)
    summary_data = summarize(rows)
    print_report(
        rows,
        summary_data,
        show_timeline=(not args.no_timeline),
        agent_name=args.agent,
    )

    for warn in warnings:
        print(f"[WARN] {warn}")

    if args.csv:
        csv_path = Path(args.csv)
        write_csv_report(csv_path, rows)
        print(f"\nCSV 已写入: {csv_path}")

    if args.json:
        json_path = Path(args.json)
        write_json_report(json_path, rows, summary_data)
        print(f"JSON 已写入: {json_path}")


if __name__ == "__main__":
    main()
