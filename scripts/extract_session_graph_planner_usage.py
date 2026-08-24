#!/usr/bin/env python3
"""导出每个医生 session 的最终主诉图变化及其前置 detector / planner 调用。

数据来源：
* ``judge_traces/judge_conversation.json``：医生 session 的边界和患者轮次；
* ``judge_traces/depression_dynamic_llm_trace.jsonl``：LLM 的原始 prompt/response。

对每个医生 session，报告会列出：
1. 最后一条患者对话的主诉图节点变化；
2. 该变化的前置 ``graph_transition_change`` 检测器调用及其原始输出；
3. 实际传入 ``graph_planner`` 的程序校验后 ``verified_change``；
4. 为该变化的父节点生成候选分支的最近一个 ``graph_planner`` 调用；
5. 会后最后一个反思的同类记录。

提示词均从 JSONL 的 ``prompt`` 字段原样复制，包含模板固定规则和当次实际输入。

示例：
    python scripts/extract_session_graph_planner_usage.py \
      results/checkpoints/batch-0813-01-Counsel-KBD2-G1-SEV--cbt-progressive-d-0813-2103 \
      --agent 卡布达
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Mapping


TRACE_RELATIVE_PATH = Path("judge_traces") / "depression_dynamic_llm_trace.jsonl"
JUDGE_TRACE_RELATIVE_PATH = Path("judge_traces") / "judge_conversation.json"
OUTPUT_FILENAME = "session_final_graph_changes_and_planners.md"
INPUT_MARKER = "\n输入："
MEETING_TIME_RE = re.compile(r"_(\d{8})-(\d{6})(?:_|$)")


class ExtractionError(ValueError):
    """Raised when session and LLM traces cannot be aligned safely."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="导出每个医生 session 的最终节点变化与前置主诉图规划器调用。"
    )
    parser.add_argument("checkpoint", type=Path, help="包含 judge_traces 的 checkpoint 目录")
    parser.add_argument("--agent", help="患者名；默认从医生 session 的 pair_key 自动读取。")
    parser.add_argument("--output", type=Path, help="输出 Markdown 路径。")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ExtractionError("JSON 无法解析: {}".format(path)) from exc
    if not isinstance(value, dict):
        raise ExtractionError("JSON 根节点不是对象: {}".format(path))
    return value


def decode_json_object(raw: Any, field_name: str) -> dict[str, Any]:
    if isinstance(raw, Mapping):
        return dict(raw)
    text = str(raw or "").strip()
    start = text.find("{")
    if start < 0:
        raise ExtractionError("{} 中未找到 JSON 对象".format(field_name))
    try:
        value, _ = json.JSONDecoder().raw_decode(text[start:])
    except json.JSONDecodeError as exc:
        raise ExtractionError("{} 的 JSON 无法解析".format(field_name)) from exc
    if not isinstance(value, dict):
        raise ExtractionError("{} 的 JSON 根节点不是对象".format(field_name))
    return value


def prompt_payload(prompt: Any) -> dict[str, Any]:
    text = str(prompt or "")
    position = text.rfind(INPUT_MARKER)
    if position < 0:
        raise ExtractionError("prompt 中未找到输入 payload")
    return decode_json_object(text[position + len(INPUT_MARKER) :], "prompt 输入")


def meeting_time(meeting_id: str) -> str:
    match = MEETING_TIME_RE.search(str(meeting_id or ""))
    if not match:
        raise ExtractionError("meeting_id 无法解析仿真时间: {}".format(meeting_id))
    date, time = match.groups()
    return "{}-{}-{}T{}:{}:{}".format(date[:4], date[4:6], date[6:8], time[:2], time[2:4], time[4:6])


def patient_from_session(session: Mapping[str, Any], requested_agent: str | None) -> str:
    if requested_agent:
        return requested_agent
    meeting = session.get("meeting", {})
    pair_key = str(meeting.get("pair_key", "") or "") if isinstance(meeting, Mapping) else ""
    names = [part.strip() for part in pair_key.split("::") if part.strip()]
    if len(names) != 2:
        raise ExtractionError("无法从 pair_key 推断患者，请传 --agent: {}".format(pair_key))
    return names[1]


def read_trace_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as source:
        for index, line in enumerate(source):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ExtractionError("trace 第 {} 行不是合法 JSON".format(index + 1)) from exc
            if isinstance(record, dict):
                record["_trace_index"] = index
                records.append(record)
    return records


def transition_details(record: Mapping[str, Any]) -> dict[str, Any]:
    payload = prompt_payload(record.get("prompt"))
    decision = decode_json_object(record.get("response"), "推进器输出")
    current = payload.get("current_stage", {})
    current = dict(current) if isinstance(current, Mapping) else {}
    candidates = payload.get("candidate_stages", [])
    candidates = candidates if isinstance(candidates, list) else []
    action = str(decision.get("action", "") or "")
    target_id = str(decision.get("target_stage_id", "") or "")
    if action == "advance":
        llm_candidate = next(
            (
                dict(candidate)
                for candidate in candidates
                if isinstance(candidate, Mapping) and str(candidate.get("id", "") or "") == target_id
            ),
            {},
        )
    else:
        llm_candidate = dict(current)
    runtime = record.get("_runtime_transition", {})
    runtime = dict(runtime) if isinstance(runtime, Mapping) else {}
    runtime_available = str(runtime.get("status", "") or "") == "available"
    runtime_transition = runtime.get("transition", {})
    runtime_transition = (
        dict(runtime_transition) if isinstance(runtime_transition, Mapping) else {}
    )
    before = runtime.get("before", {}) if runtime_available else current
    after = runtime.get("after", {}) if runtime_available else {}
    return {
        "llm_action": action,
        "llm_candidate": llm_candidate,
        "commit_status": "available" if runtime_available else "unavailable",
        "commit_action": str(runtime_transition.get("action", "") or "") if runtime_available else "",
        "commit_reason": str(runtime.get("reason", "") or "") if not runtime_available else "",
        "before": dict(before) if isinstance(before, Mapping) else {},
        "after": dict(after) if isinstance(after, Mapping) else {},
        "prompt": str(record.get("prompt", "") or ""),
        "response": str(record.get("response", "") or ""),
        "context": record.get("context", {}),
    }


def matching_chat_transitions(
    records: list[dict[str, Any]],
    patient: str,
    session: Mapping[str, Any],
) -> list[dict[str, Any]]:
    meeting = session.get("meeting", {})
    meeting = meeting if isinstance(meeting, Mapping) else {}
    expected_time = meeting_time(str(meeting.get("meeting_id", "") or ""))
    turns = session.get("turns", [])
    turns = turns if isinstance(turns, list) else []
    runtime_by_patient_text: dict[str, list[dict[str, Any]]] = {}
    for turn in turns:
        if not isinstance(turn, Mapping):
            continue
        patient_text = str(turn.get("patient", "") or "").strip()
        if not patient_text:
            continue
        runtime = turn.get("complaint_graph", {})
        runtime_by_patient_text.setdefault(patient_text, []).append(
            dict(runtime) if isinstance(runtime, Mapping) else {}
        )
    matched_per_text: dict[str, int] = {}
    matched = []
    for record in records:
        context = record.get("context", {})
        dialogue = context.get("dialogue", {}) if isinstance(context, Mapping) else {}
        utterance = str(dialogue.get("patient_utterance", "") or "").strip() if isinstance(dialogue, Mapping) else ""
        if (
            record.get("call_type") == "graph_transition"
            and str(record.get("agent", "") or "") == patient
            and isinstance(context, Mapping)
            and str(context.get("source", "") or "") == "chat"
            and str(record.get("sim_time", "") or "") == expected_time
            and utterance in runtime_by_patient_text
        ):
            occurrence = matched_per_text.get(utterance, 0)
            runtime_rows = runtime_by_patient_text[utterance]
            runtime = (
                runtime_rows[occurrence]
                if occurrence < len(runtime_rows)
                else {
                    "status": "unavailable",
                    "reason": "duplicate_patient_text_occurrence_not_aligned",
                }
            )
            annotated = dict(record)
            annotated["_runtime_transition"] = runtime
            matched.append(annotated)
            matched_per_text[utterance] = occurrence + 1
    return matched


def matching_reflection_transition(
    records: list[dict[str, Any]],
    patient: str,
    last_chat: Mapping[str, Any],
    session: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Find the reflection committed after the session's final chat transition."""
    timestamp = str(last_chat.get("sim_time", "") or "")
    last_chat_index = int(last_chat.get("_trace_index", -1))
    candidates = [
        record
        for record in records
        if record.get("call_type") == "graph_transition"
        and str(record.get("agent", "") or "") == patient
        and str(record.get("sim_time", "") or "") == timestamp
        and int(record.get("_trace_index", -1)) > last_chat_index
        and isinstance(record.get("context"), Mapping)
        and str(record["context"].get("source", "") or "") == "reflection"
    ]
    if not candidates:
        return None
    annotated = dict(candidates[0])
    graph_summary = session.get("complaint_graph", {})
    graph_summary = graph_summary if isinstance(graph_summary, Mapping) else {}
    runtime = graph_summary.get("reflection", {})
    annotated["_runtime_transition"] = (
        dict(runtime) if isinstance(runtime, Mapping) else {}
    )
    return annotated


def previous_matching_planner(
    records: list[dict[str, Any]],
    transition: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Find the nearest graph_planner call that planned the transition's parent stage."""
    transition_payload = prompt_payload(transition.get("prompt"))
    current = transition_payload.get("current_stage", {})
    parent_id = str(current.get("id", "") or "") if isinstance(current, Mapping) else ""
    transition_index = int(transition.get("_trace_index", -1))
    agent = str(transition.get("agent", "") or "")
    fallback: dict[str, Any] | None = None
    for record in reversed(records):
        if int(record.get("_trace_index", -1)) >= transition_index:
            continue
        if record.get("call_type") != "graph_planner" or str(record.get("agent", "") or "") != agent:
            continue
        try:
            planner_payload = prompt_payload(record.get("prompt"))
        except ExtractionError:
            continue
        if fallback is None:
            fallback = record
        parent = planner_payload.get("current_stage", {})
        planned_parent_id = str(parent.get("id", "") or "") if isinstance(parent, Mapping) else ""
        if parent_id and planned_parent_id == parent_id:
            return record
    return fallback


def planner_verified_change(planner: Mapping[str, Any]) -> dict[str, Any]:
    """Return the exact verified-change view that was sent to a planner."""
    payload = prompt_payload(planner.get("prompt"))
    value = payload.get("verified_change", {})
    return dict(value) if isinstance(value, Mapping) else {}


def previous_matching_change_detector(
    records: list[dict[str, Any]],
    planner: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Find the detector whose verified result led to the selected planner call.

    ``verified_change`` itself is not the detector's raw LLM output: the state
    machine filters its quotes before constructing the planner prompt.  Match
    by agent, simulation time, source and parent node, then prefer an exact
    ``change`` / evidence-quote match with the verified payload.
    """
    planner_payload = prompt_payload(planner.get("prompt"))
    current = planner_payload.get("current_stage", {})
    parent_id = str(current.get("id", "") or "") if isinstance(current, Mapping) else ""
    verified = planner_verified_change(planner)
    expected_change = str(verified.get("change", "") or "").strip()
    expected_quotes = {
        str(item or "").strip()
        for item in verified.get("evidence_quotes", [])
        if str(item or "").strip()
    }
    planner_index = int(planner.get("_trace_index", -1))
    planner_time = str(planner.get("sim_time", "") or "")
    planner_agent = str(planner.get("agent", "") or "")
    planner_context = planner.get("context", {})
    planner_source = (
        str(planner_context.get("source", "") or "")
        if isinstance(planner_context, Mapping)
        else ""
    )
    fallback: dict[str, Any] | None = None
    for record in reversed(records):
        if int(record.get("_trace_index", -1)) >= planner_index:
            continue
        if record.get("call_type") != "graph_transition_change":
            continue
        if str(record.get("agent", "") or "") != planner_agent:
            continue
        if str(record.get("sim_time", "") or "") != planner_time:
            continue
        context = record.get("context", {})
        if not isinstance(context, Mapping) or str(context.get("source", "") or "") != planner_source:
            continue
        try:
            detector_payload = prompt_payload(record.get("prompt"))
            detector_current = detector_payload.get("current_stage", {})
            detector_parent_id = (
                str(detector_current.get("id", "") or "")
                if isinstance(detector_current, Mapping)
                else ""
            )
            detector_output = decode_json_object(record.get("response"), "变化检测器输出")
        except ExtractionError:
            continue
        if parent_id and detector_parent_id != parent_id:
            continue
        if not detector_output.get("has_new_change", False):
            continue
        if fallback is None:
            fallback = record
        detector_change = str(detector_output.get("change", "") or "").strip()
        detector_quotes = {
            str(item or "").strip()
            for item in detector_output.get("evidence_quotes", [])
            if str(item or "").strip()
        }
        if detector_change == expected_change and detector_quotes == expected_quotes:
            return record
    return fallback


def markdown_block_for_detector(
    detector: Mapping[str, Any] | None,
    planner: Mapping[str, Any],
) -> list[str]:
    """Render detector raw I/O and the verified payload actually sent onward."""
    verified = planner_verified_change(planner)
    lines = [
        "#### 前置变化检测器（graph_transition_change）",
        "",
        "> 检测器原始输出会再经过逐字证据校验；下方 `verified_change` 才是实际传入 planner 的内容。",
        "",
        "##### 实际传入 planner 的 verified_change（程序校验后）",
        "",
        "```json",
        json.dumps(verified, ensure_ascii=False, indent=2),
        "```",
        "",
    ]
    if not detector:
        lines.extend(["未找到可安全对齐的前置 `graph_transition_change` 调用。", ""])
        return lines
    output = decode_json_object(detector.get("response"), "变化检测器输出")
    lines.extend(
        [
            "- 检测器原始结论：`has_new_change={}`".format(output.get("has_new_change", "未知")),
            "- 原始变化描述：{}".format(str(output.get("change", "") or "（无）")),
            "",
            "##### 检测器完整实际提示词（原样）",
            "",
            "```text",
            str(detector.get("prompt", "") or ""),
            "```",
            "",
            "##### 检测器 LLM 输出（原样）",
            "",
            "```text",
            str(detector.get("response", "") or ""),
            "```",
            "",
        ]
    )
    return lines


def markdown_block_for_transition(
    title: str,
    transition: Mapping[str, Any] | None,
    planner: Mapping[str, Any] | None,
    detector: Mapping[str, Any] | None,
) -> list[str]:
    lines = ["### {}".format(title), ""]
    if not transition:
        lines.extend(["未找到可对齐的主诉图推进器记录。", ""])
        return lines
    details = transition_details(transition)
    before = details["before"]
    after = details["after"]
    context = details["context"] if isinstance(details["context"], Mapping) else {}
    dialogue = context.get("dialogue", {}) if isinstance(context, Mapping) else {}
    dialogue = dialogue if isinstance(dialogue, Mapping) else {}
    summary_lines = [
        "- LLM 推进判定：`{}`".format(details["llm_action"] or "未知"),
    ]
    if details["commit_status"] == "available":
        summary_lines.extend(
            [
                "- 真实 commit：`{}`".format(details["commit_action"] or "未知"),
                "- 已提交节点：**{}** → **{}**".format(
                    str(before.get("stage_label", before.get("stage_id", "未知节点")) or "未知节点"),
                    str(after.get("stage_label", after.get("stage_id", "未知节点")) or "未知节点"),
                ),
            ]
        )
    else:
        summary_lines.append(
            "- 真实 commit：无法从 judge trace 安全对齐（`{}`）".format(
                details["commit_reason"] or "runtime_transition_unavailable"
            )
        )
    if details["llm_action"] == "advance":
        candidate = details["llm_candidate"]
        summary_lines.append(
            "- LLM 提议候选：**{}**".format(
                str(candidate.get("label", candidate.get("id", "未知节点")) or "未知节点")
            )
        )
    lines.extend(
        summary_lines
        + [
            "- 患者本轮话语：{}".format(str(dialogue.get("patient_utterance", "") or "") or "（反思触发，不是单句患者对话）"),
            "",
            "#### 推进器完整实际提示词（原样）",
            "",
            "```text",
            details["prompt"],
            "```",
            "",
            "#### 推进器 LLM 输出（原样）",
            "",
            "```text",
            details["response"],
            "```",
            "",
        ]
    )
    if not planner:
        lines.extend(["未找到同父节点的前置 `graph_planner` 调用，因此没有可展示的 verified_change。", ""])
        return lines
    lines.extend(markdown_block_for_detector(detector, planner))
    planner_payload = prompt_payload(planner.get("prompt"))
    lines.extend(
        [
            "#### 前一个 graph_planner 调用",
            "",
            "- planner mode：`{}`".format(planner_payload.get("mode", "未知")),
            "- 被规划的父节点：**{}**".format(
                str((planner_payload.get("current_stage", {}) or {}).get("label", "未知节点"))
            ),
            "",
            "##### planner 完整实际提示词（原样）",
            "",
            "```text",
            str(planner.get("prompt", "") or ""),
            "```",
            "",
            "##### planner LLM 输出（原样）",
            "",
            "```text",
            str(planner.get("response", "") or ""),
            "```",
            "",
        ]
    )
    return lines


def build_markdown(checkpoint: Path, sessions: list[Any], records: list[dict[str, Any]], requested_agent: str | None) -> str:
    lines = [
        "# 医生 session 的最终主诉图变化与前置 Planner",
        "",
        "- checkpoint：`{}`".format(checkpoint),
        "- 对齐规则：LLM 调用按 `meeting_id` 时间 + session 中患者原话定位；真实 chat commit 读取对应 turn 的 `complaint_graph`，真实 reflection commit 读取 session 汇总的 `complaint_graph.reflection`。不得把 LLM 的 `action=advance` 直接当成已提交推进。",
        "- planner 对齐规则：从对应推进器调用向前回溯，选择最近一条 `graph_planner` 且其 `current_stage.id` 与推进器父节点一致的调用。",
        "- detector 对齐规则：从已选 planner 向前回溯，匹配相同患者、仿真时间、source 与父节点的 `graph_transition_change`；若其原始 `change` 和证据引文与 planner 的 `verified_change` 完全一致则优先采用。",
        "- `verified_change` 是程序完成逐字证据校验后传给 planner 的最小变化视图，不必与检测器原始输出逐字段相同。",
        "- 下方 prompt/response 均直接复制自 `depression_dynamic_llm_trace.jsonl`，没有重写。",
        "",
    ]
    for number, raw_session in enumerate(sessions, start=1):
        if not isinstance(raw_session, Mapping):
            continue
        meeting = raw_session.get("meeting", {})
        meeting = meeting if isinstance(meeting, Mapping) else {}
        patient = patient_from_session(raw_session, requested_agent)
        chat_records = matching_chat_transitions(records, patient, raw_session)
        last_chat = chat_records[-1] if chat_records else None
        reflection = (
            matching_reflection_transition(records, patient, last_chat, raw_session)
            if last_chat
            else None
        )
        lines.extend(
            [
                "## Session {}：{}".format(number, str(meeting.get("meeting_id", "") or "未知 meeting")),
                "",
                "- 患者：{}".format(patient),
                "- 医生 session 中可对齐的患者对话判定次数：{}".format(len(chat_records)),
                "",
            ]
        )
        chat_planner = previous_matching_planner(records, last_chat) if last_chat else None
        chat_detector = previous_matching_change_detector(records, chat_planner) if chat_planner else None
        reflection_planner = previous_matching_planner(records, reflection) if reflection else None
        reflection_detector = previous_matching_change_detector(records, reflection_planner) if reflection_planner else None
        lines.extend(markdown_block_for_transition("最后一个对话节点变化", last_chat, chat_planner, chat_detector))
        lines.extend(markdown_block_for_transition("最后一个反思节点变化", reflection, reflection_planner, reflection_detector))
    return "\n".join(lines)


def build_empty_sessions_markdown(checkpoint: Path) -> str:
    """Emit an explicit report for groups without doctor sessions instead of failing."""
    return "\n".join(
        [
            "# 医生 session 的最终主诉图变化与前置 Planner",
            "",
            "- checkpoint：`{}`".format(checkpoint),
            "- 结果：`judge_conversation.json` 中没有医生 session。",
            "- 说明：该实验组没有可按医生 session 对齐的对话/反思节点变化，因此本报告没有 session 级主诉图推进器或 graph planner 记录。",
            "",
        ]
    )


def main() -> int:
    args = parse_args()
    checkpoint = args.checkpoint.expanduser()
    trace_path = checkpoint / TRACE_RELATIVE_PATH
    judge_path = checkpoint / JUDGE_TRACE_RELATIVE_PATH
    if not checkpoint.is_dir():
        raise FileNotFoundError("输入不是 checkpoint 目录: {}".format(checkpoint))
    if not trace_path.is_file() or not judge_path.is_file():
        raise FileNotFoundError("缺少 depression trace 或 judge session trace: {}".format(checkpoint / "judge_traces"))
    records = read_trace_records(trace_path)
    judge_payload = load_json(judge_path)
    sessions = judge_payload.get("sessions", [])
    if not isinstance(sessions, list):
        raise ExtractionError("judge_conversation.json 的 sessions 不是列表")
    output_path = args.output or (checkpoint / "judge_traces" / OUTPUT_FILENAME)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if sessions:
        report = build_markdown(checkpoint, sessions, records, args.agent)
        message = "已导出 {} 个医生 session".format(len(sessions))
    else:
        report = build_empty_sessions_markdown(checkpoint)
        message = "未发现医生 session，已导出说明"
    output_path.write_text(report, encoding="utf-8")
    print("{}到: {}".format(message, output_path))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ExtractionError) as exc:
        print("[ERROR] {}".format(exc), file=sys.stderr)
        raise SystemExit(1)
