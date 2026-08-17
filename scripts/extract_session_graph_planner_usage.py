#!/usr/bin/env python3
"""导出每个医生 session 的最终主诉图变化及其前置 graph planner 调用。

数据来源：
* ``judge_traces/judge_conversation.json``：医生 session 的边界和患者轮次；
* ``judge_traces/depression_dynamic_llm_trace.jsonl``：LLM 的原始 prompt/response。

对每个医生 session，报告会列出：
1. 最后一条患者对话的主诉图节点变化；
2. 为该变化的父节点生成候选分支的最近一个 ``graph_planner`` 调用；
3. 会后最后一个反思的主诉图节点变化；
4. 为该反思变化的父节点生成候选分支的最近一个 ``graph_planner`` 调用。

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
        after = next(
            (
                dict(candidate)
                for candidate in candidates
                if isinstance(candidate, Mapping) and str(candidate.get("id", "") or "") == target_id
            ),
            {},
        )
    else:
        after = dict(current)
    return {
        "action": action,
        "before": current,
        "after": after,
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
    patient_texts = {
        str(turn.get("patient", "") or "").strip()
        for turn in turns
        if isinstance(turn, Mapping) and str(turn.get("patient", "") or "").strip()
    }
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
            and utterance in patient_texts
        ):
            matched.append(record)
    return matched


def matching_reflection_transition(
    records: list[dict[str, Any]],
    patient: str,
    last_chat: Mapping[str, Any],
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
    return candidates[0] if candidates else None


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


def markdown_block_for_transition(title: str, transition: Mapping[str, Any] | None, planner: Mapping[str, Any] | None) -> list[str]:
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
    lines.extend(
        [
            "- 判定：`{}`".format(details["action"] or "未知"),
            "- 节点变化：**{}** → **{}**".format(
                str(before.get("label", before.get("id", "未知节点")) or "未知节点"),
                str(after.get("label", after.get("id", "未知节点")) or "未知节点"),
            ),
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
            "#### 前一个 graph_planner 调用",
            "",
        ]
    )
    if not planner:
        lines.extend(["未找到同父节点的前置 `graph_planner` 调用。", ""])
        return lines
    planner_payload = prompt_payload(planner.get("prompt"))
    lines.extend(
        [
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
        "- 对齐规则：最后对话变化按 `meeting_id` 时间 + session 中患者原话定位；反思变化取该最后对话之后、同一仿真时间的第一条 reflection 推进记录。",
        "- planner 对齐规则：从对应推进器调用向前回溯，选择最近一条 `graph_planner` 且其 `current_stage.id` 与推进器父节点一致的调用。",
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
        reflection = matching_reflection_transition(records, patient, last_chat) if last_chat else None
        lines.extend(
            [
                "## Session {}：{}".format(number, str(meeting.get("meeting_id", "") or "未知 meeting")),
                "",
                "- 患者：{}".format(patient),
                "- 医生 session 中可对齐的患者对话推进次数：{}".format(len(chat_records)),
                "",
            ]
        )
        lines.extend(markdown_block_for_transition("最后一个对话节点变化", last_chat, previous_matching_planner(records, last_chat) if last_chat else None))
        lines.extend(markdown_block_for_transition("最后一个反思节点变化", reflection, previous_matching_planner(records, reflection) if reflection else None))
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
