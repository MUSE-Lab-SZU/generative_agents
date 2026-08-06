"""Extract complaint-stage changes only at scale-evaluation session nodes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .schema import ExperimentRecord, label_sort_key, session_index


def _stage(payload: Any) -> dict[str, Any] | None:
    if not isinstance(payload, dict) or not payload.get("stage_id"):
        return None
    return {
        "stage_id": str(payload.get("stage_id")),
        "stage_label": str(payload.get("stage_label") or ""),
        "summary": str(payload.get("summary") or ""),
        "core_belief": str(payload.get("core_belief") or ""),
        "stage_index": payload.get("stage_index"),
    }


def _trace_path(checkpoints_root: Path, record: ExperimentRecord) -> Path:
    return checkpoints_root / record.run_name / "judge_traces" / "judge_conversation.json"


def _action_counts(sessions: list[dict[str, Any]], start: int, stop: int) -> dict[str, int]:
    counts = {
        "session_reflection_advance": 0,
        "session_reflection_hold": 0,
        "turn_advance": 0,
        "turn_hold": 0,
        "turn_unavailable": 0,
    }
    for session in sessions[start:stop]:
        graph = session.get("complaint_graph") or {}
        action = ((graph.get("reflection") or {}).get("transition") or {}).get("action")
        if action == "advance":
            counts["session_reflection_advance"] += 1
        elif action == "hold":
            counts["session_reflection_hold"] += 1
        for turn in session.get("turns") or []:
            turn_graph = turn.get("complaint_graph") or {}
            if turn_graph.get("status") != "available":
                counts["turn_unavailable"] += 1
                continue
            turn_action = (turn_graph.get("transition") or {}).get("action")
            if turn_action == "advance":
                counts["turn_advance"] += 1
            elif turn_action == "hold":
                counts["turn_hold"] += 1
    return counts


def extract_complaint_evaluation_nodes(
    records: list[ExperimentRecord], labels: list[str], checkpoints_root: Path
) -> dict[str, Any]:
    evaluation_labels = sorted(
        [label for label in labels if label == "T0" or session_index(label) is not None],
        key=lambda value: label_sort_key(value, labels),
    )
    node_rows: list[dict[str, Any]] = []
    interval_rows: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    prompt_cases: list[dict[str, Any]] = []
    for record in records:
        trace_path = _trace_path(checkpoints_root, record)
        if not trace_path.is_file():
            missing.append({"stable_id": record.stable_id, "timepoint": "ALL", "reason": "trace_file_missing"})
            continue
        payload = json.loads(trace_path.read_text(encoding="utf-8"))
        sessions = payload.get("sessions") or []
        nodes: list[dict[str, Any]] = []
        for label in evaluation_labels:
            if not sessions:
                missing.append(
                    {
                        "stable_id": record.stable_id,
                        "timepoint": label,
                        "reason": "judge_trace_contains_no_sessions",
                        "source": "judge_conversation.json",
                    }
                )
                continue
            if label == "T0":
                session_position = 0
                graph = (sessions[0].get("complaint_graph") or {}) if sessions else {}
                stage = _stage(graph.get("before_chat"))
                source = "session_1.before_chat"
            else:
                number = session_index(label)
                session_position = int(number or 0)
                graph = (
                    (sessions[session_position - 1].get("complaint_graph") or {})
                    if number and 0 < session_position <= len(sessions)
                    else {}
                )
                stage = _stage(graph.get("after_reflection"))
                source = f"session_{session_position}.after_reflection"
            if stage is None:
                missing.append(
                    {
                        "stable_id": record.stable_id,
                        "timepoint": label,
                        "reason": "complaint_stage_missing_at_exact_evaluation_node",
                        "source": source,
                    }
                )
                continue
            row = {
                "stable_id": record.stable_id,
                "outer_run_id": record.repeat_id,
                "persona": record.kbd,
                "group": record.group,
                "severity": record.severity,
                "run_name": record.run_name,
                "timepoint": label,
                "session_number": session_position,
                "source": source,
                **stage,
            }
            nodes.append(row)
            node_rows.append(row)
        nodes.sort(key=lambda row: label_sort_key(row["timepoint"], evaluation_labels))
        for start_node, end_node in zip(nodes, nodes[1:]):
            start_session = int(start_node["session_number"])
            end_session = int(end_node["session_number"])
            counts = _action_counts(sessions, start_session, end_session)
            start_index, end_index = start_node.get("stage_index"), end_node.get("stage_index")
            index_change = (
                float(end_index) - float(start_index)
                if isinstance(start_index, (int, float)) and isinstance(end_index, (int, float))
                else None
            )
            same_stage = start_node["stage_id"] == end_node["stage_id"]
            if same_stage:
                simple_status = "停滞在同一主诉阶段"
            elif index_change is not None and index_change > 0:
                simple_status = "主诉阶段向前推进"
            elif index_change is not None and index_change < 0:
                simple_status = "主诉阶段回退"
            else:
                simple_status = "主诉内容发生变化"
            reflection_total = counts["session_reflection_advance"] + counts["session_reflection_hold"]
            turn_total = counts["turn_advance"] + counts["turn_hold"]
            interval_rows.append(
                {
                    "stable_id": record.stable_id,
                    "outer_run_id": record.repeat_id,
                    "persona": record.kbd,
                    "group": record.group,
                    "from_timepoint": start_node["timepoint"],
                    "to_timepoint": end_node["timepoint"],
                    "from_stage_id": start_node["stage_id"],
                    "to_stage_id": end_node["stage_id"],
                    "from_stage_label": start_node["stage_label"],
                    "to_stage_label": end_node["stage_label"],
                    "from_summary": start_node["summary"],
                    "to_summary": end_node["summary"],
                    "from_core_belief": start_node["core_belief"],
                    "to_core_belief": end_node["core_belief"],
                    "from_stage_index": start_index,
                    "to_stage_index": end_index,
                    "stage_index_change": index_change,
                    "same_stage_id": same_stage,
                    "simple_status": simple_status,
                    **counts,
                    "session_reflection_advance_rate": (
                        counts["session_reflection_advance"] / reflection_total if reflection_total else None
                    ),
                    "turn_advance_rate": counts["turn_advance"] / turn_total if turn_total else None,
                }
            )
        if len(nodes) >= 2:
            prompt_cases.append(
                {
                    "stable_id": record.stable_id,
                    "persona": record.kbd,
                    "group": record.group,
                    "outer_run_id": record.repeat_id,
                    "nodes": [
                        {
                            key: row[key]
                            for key in (
                                "timepoint",
                                "stage_id",
                                "stage_label",
                                "summary",
                                "core_belief",
                                "stage_index",
                            )
                        }
                        for row in nodes
                    ],
                }
            )
    return {
        "evaluation_labels": evaluation_labels,
        "nodes": node_rows,
        "intervals": interval_rows,
        "missing": missing,
        "prompt_cases": prompt_cases,
    }


SEMANTIC_SYSTEM_PROMPT = """你是仿真实验数据整理员。你只能比较输入中明确给出的主诉阶段文字，不能诊断，不能推断真实患者疗效，不能把阶段序号增大自动解释为病情改善。输出严格 JSON，不要 Markdown。"""

SEMANTIC_USER_PROMPT_TEMPLATE = """请按固定规则比较这个仿真 run 在量表评估节点的主诉内容。

规则：
1. 逐个比较相邻节点，只说明关注点/核心信念如何变化。
2. change_type 只能是：停滞、深化、转向、回退、信息不足。
3. “深化”表示同一主题变得更具体或更深入，不等于症状改善。
4. “转向”表示主诉焦点发生改变，也不自动等于改善。
5. 每条 plain_change 使用普通中文，最多 80 字；evidence 必须直接来自输入摘要。
6. 最后给出 overall_plain_summary，最多 150 字，明确区分“主诉推进”和“量表症状改善”。

输出结构：
{{
  "stable_id": "...",
  "intervals": [
    {{
      "from_timepoint": "...",
      "to_timepoint": "...",
      "change_type": "停滞|深化|转向|回退|信息不足",
      "plain_change": "...",
      "evidence": "..."
    }}
  ],
  "overall_plain_summary": "..."
}}

输入：
{case_json}
"""
