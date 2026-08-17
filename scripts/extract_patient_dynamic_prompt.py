#!/usr/bin/env python3
"""提取实际注入患者发言 prompt 的动态抑郁模块，并输出为易读 Markdown。

动态抑郁 LLM trace 保存了图规划、推进和情绪推断等内部调用；真正传给
患者发言模型的动态模块，保存在 checkpoint 快照中：
``agents.<患者>.status.intervention.last_generate_chat_prompt.prompt_text``。

本脚本遍历 checkpoint 目录下的 ``simulate-*.json``，只提取其中实际包含
“当前主诉节点层”的患者 prompt，并输出以下三层：

* 当前主诉节点层（含主诉图当前节点与候选分支）；
* 会话上下文层；
* 瞬时情绪与表达指导层。

同时，若存在 ``judge_traces/depression_dynamic_llm_trace.jsonl``，脚本会把
每个快照当前状态对应的最近一次“主诉图推进器”（``graph_transition``）判定，
以及“对话中最后一次主诉图推进”（``source=chat`` 且 ``action=advance``）附在
同一条记录后：每条均提供实际发送给 LLM 的完整 prompt 原文和 LLM JSON 输出。

不输出基础人设、长期记忆、完整对话记录、原始 trace 或模型回复。

示例：
    python scripts/extract_patient_dynamic_prompt.py \
      'results/0808-g1-实验精简存档/checkpoints/batch-0808-01-Counsel-KBD1-G1-SEV--cbt-progressive-d-0808-0821'

默认输出：
    <checkpoint>/judge_traces/patient_dynamic_depression_prompts.md

注意：一个 snapshot 只保存当时该角色最近一次 generate_chat prompt。因此输出是
“每个已保存快照中的最近患者发言 prompt”，而不是存档内每一轮对话的完整历史。
要获得逐轮完整 prompt，需在运行时单独逐轮落盘该字段。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping


OUTPUT_FILENAME = "patient_dynamic_depression_prompts.md"
STAGE_LAYER_HEADER = "=== 当前主诉节点层 ==="
END_INSTRUCTION = "请将以上各层综合起来，真实地表现这个角色在当前主诉节点下的说话方式。"
TRACE_RELATIVE_PATH = Path("judge_traces") / "depression_dynamic_llm_trace.jsonl"


class ExtractionError(ValueError):
    """Raised when the checkpoint structure is not usable for extraction."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="从 checkpoint 快照提取患者发言 prompt 中的动态抑郁模块。"
    )
    parser.add_argument("archive", type=Path, help="包含 simulate-*.json 的 checkpoint 目录")
    parser.add_argument(
        "--agent",
        help="只导出指定患者；默认导出所有含动态抑郁模块的角色。",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="输出 Markdown 文件；默认写到 checkpoint/judge_traces/ 下",
    )
    return parser.parse_args()


def extract_dynamic_module(prompt_text: Any) -> str:
    """Keep only the dynamic layers that control the patient's utterance."""
    text = str(prompt_text or "")
    start = text.find(STAGE_LAYER_HEADER)
    if start < 0:
        return ""

    end = text.find(END_INSTRUCTION, start)
    if end >= 0:
        end += len(END_INSTRUCTION)
        return text[start:end].strip()

    # 兼容旧快照或自定义模板：动态块通常在 consult history 前结束。
    for marker in ("\n<consult_history_memory>", "\n以下是 ", "\n背景："):
        end = text.find(marker, start)
        if end >= 0:
            return text[start:end].strip()
    return text[start:].strip()


def load_snapshot(path: Path) -> Mapping[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ExtractionError("快照不是合法 JSON: {}".format(path)) from exc
    if not isinstance(data, Mapping):
        raise ExtractionError("快照 JSON 根节点不是对象: {}".format(path))
    return data


def decode_json_object(raw: Any, field_name: str) -> dict[str, Any]:
    """Parse a trace JSON object, allowing leading Markdown/code-fence text."""
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


def load_transition_records(checkpoint_dir: Path) -> list[dict[str, Any]]:
    trace_path = checkpoint_dir / TRACE_RELATIVE_PATH
    if not trace_path.is_file():
        return []
    records: list[dict[str, Any]] = []
    with trace_path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ExtractionError("动态抑郁 trace 第 {} 行不是合法 JSON".format(line_number)) from exc
            if isinstance(record, dict) and record.get("call_type") == "graph_transition":
                records.append(record)
    return records


def trace_for_history_row(
    history_row: Mapping[str, Any],
    agent_name: str,
    trace_records: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Align one committed graph-history row to its exact LLM trace record."""
    latest = history_row if isinstance(history_row, Mapping) else {}
    if not latest:
        return None
    timestamp = str(latest.get("timestamp", "") or "")
    source = str(latest.get("source", "") or "")
    action = str(latest.get("action", "") or "")
    target_stage_id = str(latest.get("to_stage_id", "") or "")
    matches: list[dict[str, Any]] = []
    for record in trace_records:
        context = record.get("context", {})
        if (
            record.get("agent") != agent_name
            or not isinstance(context, Mapping)
            or str(context.get("source", "") or "") != source
            or str(record.get("sim_time", "") or "") != timestamp
        ):
            continue
        try:
            decision = decode_json_object(record.get("response"), "推进器输出")
        except ExtractionError:
            continue
        if str(decision.get("action", "") or "") != action:
            continue
        if action == "advance" and str(decision.get("target_stage_id", "") or "") != target_stage_id:
            continue
        matches.append(record)
    if len(matches) != 1:
        return None
    record = matches[0]
    return {
        "timestamp": timestamp,
        "source": source,
        "action": action,
        "from_stage_label": str(latest.get("from_stage_label", "") or ""),
        "to_stage_label": str(latest.get("to_stage_label", "") or ""),
        "prompt": str(record.get("prompt", "") or ""),
        "output": decode_json_object(record.get("response"), "推进器输出"),
    }


def latest_transition_for_snapshot(
    snapshot: Mapping[str, Any],
    agent_name: str,
    trace_records: list[dict[str, Any]],
    source: str | None = None,
    action: str | None = None,
) -> dict[str, Any] | None:
    """Find the latest committed transition, optionally restricted by source/action."""
    agents = snapshot.get("agents", {})
    agent_data = agents.get(agent_name, {}) if isinstance(agents, Mapping) else {}
    state = agent_data.get("depression_dynamic_state", {}) if isinstance(agent_data, Mapping) else {}
    manager = state.get("complaint_graph_manager", {}) if isinstance(state, Mapping) else {}
    history = manager.get("stage_history", []) if isinstance(manager, Mapping) else []
    if not isinstance(history, list) or not history:
        return None
    for history_row in reversed(history):
        if not isinstance(history_row, Mapping):
            continue
        if source is not None and str(history_row.get("source", "") or "") != source:
            continue
        if action is not None and str(history_row.get("action", "") or "") != action:
            continue
        trace = trace_for_history_row(history_row, agent_name, trace_records)
        if trace is not None:
            return trace
    return None


def collect_prompt_entries(checkpoint_dir: Path, agent_filter: str | None) -> tuple[list[dict[str, Any]], int]:
    snapshots = sorted(checkpoint_dir.glob("simulate-*.json"))
    if not snapshots:
        raise FileNotFoundError("未找到快照文件 simulate-*.json: {}".format(checkpoint_dir))

    trace_records = load_transition_records(checkpoint_dir)
    entries: list[dict[str, Any]] = []
    for snapshot_path in snapshots:
        snapshot = load_snapshot(snapshot_path)
        agents = snapshot.get("agents", {})
        if not isinstance(agents, Mapping):
            continue
        for agent_name, agent_data in agents.items():
            agent_name = str(agent_name or "")
            if agent_filter and agent_name != agent_filter:
                continue
            if not isinstance(agent_data, Mapping):
                continue
            status = agent_data.get("status", {})
            intervention = status.get("intervention", {}) if isinstance(status, Mapping) else {}
            last_prompt = (
                intervention.get("last_generate_chat_prompt", {})
                if isinstance(intervention, Mapping)
                else {}
            )
            if not isinstance(last_prompt, Mapping):
                continue
            dynamic_module = extract_dynamic_module(last_prompt.get("prompt_text"))
            if not dynamic_module:
                continue
            entries.append(
                {
                    "snapshot": snapshot_path.name,
                    "sim_time": str(last_prompt.get("ts", "") or snapshot.get("time", "") or ""),
                    "agent": agent_name,
                    "other": str(last_prompt.get("other", "") or ""),
                    "turn_no": str(last_prompt.get("turn_no", "") or ""),
                    "dynamic_module": dynamic_module,
                    "latest_transition": latest_transition_for_snapshot(
                        snapshot, agent_name, trace_records
                    ),
                    "last_chat_advance": latest_transition_for_snapshot(
                        snapshot, agent_name, trace_records, source="chat", action="advance"
                    ),
                }
            )
    return entries, len(snapshots)


def markdown_for_entries(
    entries: list[dict[str, Any]],
    checkpoint_dir: Path,
    snapshot_count: int,
) -> str:
    lines = [
        "# 患者发言提示词中的动态抑郁模块",
        "",
        "来源：`{}`".format(checkpoint_dir),
        "",
        "共扫描 {} 个快照，提取 {} 条含动态抑郁模块的患者发言 prompt。".format(
            snapshot_count, len(entries)
        ),
        "",
        "> 每条记录是对应快照中该患者最近一次发言时实际注入的模块，不是完整对话历史。",
        "> 每条记录同时展示：生成快照最终状态的“最近一次判定”（通常是会后 reflection），以及“对话中最后一次主诉图推进”（仅 `source=chat` 且 `action=advance`）；初始节点来自配置，不经过推进器。",
        "> “本次完整实际提示词（原样）”直接复制自 `depression_dynamic_llm_trace.jsonl` 的 `prompt` 字段，包含 `graph_transition.txt` 固定规则及该次实际上下文，不是脚本重组的 JSON 摘录。",
        "",
    ]
    for index, entry in enumerate(entries, start=1):
        lines.extend(
            [
                "## {}. {} → {}".format(index, entry["agent"], entry["other"] or "未知对象"),
                "",
                "- 快照：`{}`".format(entry["snapshot"]),
                "- 发言时间：{}".format(entry["sim_time"] or "未知"),
                "",
                "```text",
                entry["dynamic_module"],
                "```",
                "",
            ]
        )
        append_transition_section(
            lines,
            "主诉图推进器：最近一次判定",
            entry.get("latest_transition"),
            "该快照仍处于初始主诉节点；该节点来自配置初始化，没有对应的推进器调用。",
        )
        append_transition_section(
            lines,
            "对话中最后一次主诉图推进",
            entry.get("last_chat_advance"),
            "截至该快照，尚未出现由实际对话（`source=chat`）触发且判定为 `advance` 的主诉图推进。",
        )
    return "\n".join(lines)


def append_transition_section(
    lines: list[str],
    title: str,
    transition: Any,
    missing_message: str,
) -> None:
    """Append one trace-aligned transition section in a consistent readable form."""
    lines.extend(["### {}".format(title), ""])
    if not isinstance(transition, Mapping):
        lines.extend([missing_message, ""])
        return
    lines.extend(
        [
            "- 判定时间：{}".format(transition.get("timestamp") or "未知"),
            "- 触发来源：{}".format(transition.get("source") or "未知"),
            "- 判定结果：`{}`（{} → {}）".format(
                transition.get("action") or "未知",
                transition.get("from_stage_label") or "未知节点",
                transition.get("to_stage_label") or "未知节点",
            ),
            "",
            "#### 本次完整实际提示词（原样）",
            "",
            "```text",
            str(transition.get("prompt", "") or ""),
            "```",
            "",
            "#### 推进器 LLM 输出",
            "",
            "```json",
            json.dumps(transition.get("output", {}), ensure_ascii=False, indent=2),
            "```",
            "",
        ]
    )


def main() -> int:
    args = parse_args()
    checkpoint_dir = args.archive.expanduser()
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError("输入不是 checkpoint 目录: {}".format(checkpoint_dir))
    entries, snapshot_count = collect_prompt_entries(checkpoint_dir, args.agent)
    if not entries:
        name_hint = "角色 '{}'".format(args.agent) if args.agent else "任何角色"
        raise ExtractionError("未在快照中找到含动态抑郁模块的 {} 发言 prompt".format(name_hint))

    output_path = args.output or (checkpoint_dir / "judge_traces" / OUTPUT_FILENAME)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        markdown_for_entries(entries, checkpoint_dir, snapshot_count), encoding="utf-8"
    )
    print("已从 {} 个快照提取 {} 条患者动态抑郁模块到: {}".format(snapshot_count, len(entries), output_path))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ExtractionError) as exc:
        print("[ERROR] {}".format(exc), file=sys.stderr)
        raise SystemExit(1)
