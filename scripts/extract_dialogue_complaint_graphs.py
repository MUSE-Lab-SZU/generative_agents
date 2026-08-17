#!/usr/bin/env python3
"""从动态抑郁 JSONL 中提取每轮对话前后的主诉图节点。

输入为某次实验的 checkpoint 目录（或直接指定
``depression_dynamic_llm_trace.jsonl`` 文件）。脚本只处理 ``source=chat``
的主诉图推进判定，因此会同时覆盖强制治疗对话和居民对话，但会排除会后
内在反思、图规划、情绪推断等非对话记录。

示例：
    python scripts/extract_dialogue_complaint_graphs.py \
      'results/0808-g1-实验精简存档/checkpoints/batch-0808-01-Counsel-KBD1-G1-SEV--cbt-progressive-d-0808-0821'

默认输出：
    <checkpoint>/judge_traces/dialogue_complaint_graphs.json

输出不包含原始 prompt、LLM response 或原始 trace；每个 dialogue 仅保留
对话识别信息，以及 ``before_graph`` 和 ``after_graph``。其中推进后的
``candidate_stages`` 在原始判定记录中尚不可得，故为空列表；后续对话的
``before_graph`` 会记录其届时可用的候选节点。
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping


TRACE_RELATIVE_PATH = Path("judge_traces") / "depression_dynamic_llm_trace.jsonl"
OUTPUT_FILENAME = "dialogue_complaint_graphs.json"


class ExtractionError(ValueError):
    """Raised when a graph-transition record cannot be decoded."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="提取强制对话及居民对话前后的主诉图，不保留原始 trace。"
    )
    parser.add_argument(
        "archive",
        type=Path,
        help="checkpoint 目录，或 depression_dynamic_llm_trace.jsonl 文件",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="输出 JSON 文件；默认写到 checkpoint/judge_traces/ 下",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="任一目标对话记录无法解析时立即失败，而非跳过并报告。",
    )
    return parser.parse_args()


def resolve_trace_path(archive: Path) -> tuple[Path, Path]:
    """Return ``(trace_path, checkpoint_dir)`` for a checkpoint dir or JSONL file."""
    archive = archive.expanduser()
    if archive.is_file():
        if archive.name != TRACE_RELATIVE_PATH.name:
            raise FileNotFoundError(
                "输入文件不是 {}: {}".format(TRACE_RELATIVE_PATH.name, archive)
            )
        if archive.parent.name != "judge_traces":
            raise FileNotFoundError(
                "输入 trace 必须位于 checkpoint 的 judge_traces 目录: {}".format(archive)
            )
        return archive, archive.parent.parent

    trace_path = archive / TRACE_RELATIVE_PATH
    if not trace_path.is_file():
        raise FileNotFoundError("未找到动态抑郁记录文档: {}".format(trace_path))
    return trace_path, archive


def decode_json_object(raw: Any, field_name: str) -> dict[str, Any]:
    """Decode an object response, tolerating leading prose or Markdown fences."""
    if isinstance(raw, Mapping):
        return dict(raw)
    text = str(raw or "").strip()
    start = text.find("{")
    if start < 0:
        raise ExtractionError("{} 中没有 JSON 对象".format(field_name))
    try:
        value, _ = json.JSONDecoder().raw_decode(text[start:])
    except json.JSONDecodeError as exc:
        raise ExtractionError("{} 的 JSON 无法解析".format(field_name)) from exc
    if not isinstance(value, dict):
        raise ExtractionError("{} 的 JSON 根节点不是对象".format(field_name))
    return value


def transition_payload_from_prompt(prompt: Any) -> dict[str, Any]:
    """Read the transition input object following the last ``输入：`` marker."""
    text = str(prompt or "")
    marker = "\n输入："
    position = text.rfind(marker)
    if position < 0:
        raise ExtractionError("graph_transition prompt 中未找到输入 payload")
    return decode_json_object(text[position + len(marker) :], "graph_transition 输入")


def stage_copy(value: Any, field_name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ExtractionError("{} 不是主诉图节点对象".format(field_name))
    stage = copy.deepcopy(dict(value))
    if not str(stage.get("id", "") or "").strip():
        raise ExtractionError("{} 缺少节点 id".format(field_name))
    return stage


def candidate_stages(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw_candidates = payload.get("candidate_stages", [])
    if not isinstance(raw_candidates, list):
        raise ExtractionError("graph_transition 输入的 candidate_stages 不是列表")
    return [stage_copy(item, "candidate_stages") for item in raw_candidates]


def build_dialogue_record(record: Mapping[str, Any], order: int) -> dict[str, Any]:
    context = record.get("context", {})
    context = context if isinstance(context, Mapping) else {}
    if str(context.get("source", "") or "") != "chat":
        raise ExtractionError("不是对话记录")

    payload = transition_payload_from_prompt(record.get("prompt"))
    before_stage = stage_copy(payload.get("current_stage"), "current_stage")
    before_candidates = candidate_stages(payload)
    decision = decode_json_object(record.get("response"), "graph_transition 输出")
    action = str(decision.get("action", "") or "").strip()

    if action == "advance":
        target_id = str(decision.get("target_stage_id", "") or "").strip()
        after_stage = next(
            (candidate for candidate in before_candidates if candidate["id"] == target_id),
            None,
        )
        if after_stage is None:
            raise ExtractionError("advance 的 target_stage_id 未出现在 candidate_stages 中")
    elif action == "hold":
        after_stage = copy.deepcopy(before_stage)
    else:
        raise ExtractionError("graph_transition 输出的 action 不是 hold 或 advance")

    dialogue = context.get("dialogue", {})
    dialogue = dialogue if isinstance(dialogue, Mapping) else {}
    return {
        "order": order,
        "sim_time": str(record.get("sim_time", "") or ""),
        "agent": str(record.get("agent", "") or ""),
        "dialogue": {
            "other_agent": str(context.get("other_agent", "") or ""),
            "relationship": str(context.get("relationship", "") or ""),
            "interaction_type": str(context.get("interaction_type", "") or ""),
            "patient_utterance": str(dialogue.get("patient_utterance", "") or ""),
            "counterpart_utterance": str(dialogue.get("counterpart_utterance", "") or ""),
        },
        "before_graph": {
            "current_stage": before_stage,
            "candidate_stages": before_candidates,
        },
        "after_graph": {
            "current_stage": after_stage,
            "candidate_stages": [],
        },
    }


def iter_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ExtractionError("第 {} 行不是合法 JSON".format(line_number)) from exc
            if not isinstance(record, dict):
                raise ExtractionError("第 {} 行 JSON 根节点不是对象".format(line_number))
            yield line_number, record


def extract(trace_path: Path, strict: bool) -> tuple[list[dict[str, Any]], list[str], int]:
    dialogues: list[dict[str, Any]] = []
    warnings: list[str] = []
    skipped_non_dialogues = 0
    for line_number, record in iter_jsonl(trace_path):
        if record.get("call_type") != "graph_transition":
            continue
        context = record.get("context", {})
        source = context.get("source") if isinstance(context, Mapping) else None
        if source != "chat":
            skipped_non_dialogues += 1
            continue
        try:
            dialogues.append(build_dialogue_record(record, len(dialogues) + 1))
        except ExtractionError as exc:
            message = "跳过第 {} 行对话记录：{}".format(line_number, exc)
            if strict:
                raise ExtractionError(message) from exc
            warnings.append(message)
    return dialogues, warnings, skipped_non_dialogues


def main() -> int:
    args = parse_args()
    trace_path, checkpoint_dir = resolve_trace_path(args.archive)
    output_path = args.output or (checkpoint_dir / "judge_traces" / OUTPUT_FILENAME)
    dialogues, warnings, skipped_non_dialogues = extract(trace_path, args.strict)

    payload = {
        "schema_version": 1,
        "source_trace": str(trace_path),
        "dialogue_count": len(dialogues),
        "excluded_non_dialogue_transition_count": skipped_non_dialogues,
        "skipped_malformed_dialogue_count": len(warnings),
        "dialogues": dialogues,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("已提取 {} 条对话主诉图到: {}".format(len(dialogues), output_path))
    print("已排除 {} 条非对话主诉图判定（如内在反思）。".format(skipped_non_dialogues))
    for warning in warnings:
        print("[WARN] {}".format(warning), file=sys.stderr)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ExtractionError) as exc:
        print("[ERROR] {}".format(exc), file=sys.stderr)
        raise SystemExit(1)
