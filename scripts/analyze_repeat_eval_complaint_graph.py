#!/usr/bin/env python3
"""将重复量表评估的逐项得分与同一冻结时点的主诉图状态对齐。

重复评估的 K 次量表回答来自同一份冻结 checkpoint：主诉图在同一评估点内
不变，变化的是量表回答。因此本脚本会保留逐次得分以描述测量波动，同时生成
按“主诉图状态 × 量表条目”聚合的均值、标准差和范围；它不会把 K 次重复误当成
K 个独立主诉图状态。

示例：
    python scripts/analyze_repeat_eval_complaint_graph.py \
      'results/0808-g1-实验精简存档/checkpoints/batch-0808-01-Counsel-KBD1-G1-SEV--cbt-progressive-d-0808-0821'

可选地传入重复评估 summary，避免自动定位：
    --repeat-summary results/.../experiment_data/reports/repeat-..._summary.json

输出目录（默认）：
    <checkpoint>/judge_traces/repeat_eval_complaint_graph_analysis/

其中：
* ``item_repeat_long.csv``：一行一条 ``评估点 × 条目 × repeat`` 记录；
* ``item_summary.csv``：同一主诉图状态下条目得分的均值、标准差、范围；
* ``report.md``：主诉图与总分/高波动条目的可读对照。
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping


OUTPUT_DIRNAME = "repeat_eval_complaint_graph_analysis"

ITEM_LABELS = {
    "PHQ-9": [
        "兴趣或愉悦感缺失",
        "情绪低落、沮丧或绝望",
        "睡眠问题",
        "疲劳或精力不足",
        "食欲或体重变化",
        "自我评价低、失败感",
        "注意力困难",
        "动作迟缓或坐立不安",
        "自伤/不如死去想法",
    ],
    "BDI-II": [
        "悲伤",
        "悲观",
        "过去失败感",
        "快乐丧失",
        "内疚感",
        "受惩罚感",
        "不喜欢自己",
        "自我批评",
        "自杀想法或愿望",
        "哭泣",
        "激越",
        "兴趣丧失",
        "犹豫不决",
        "无价值感",
        "精力丧失",
        "睡眠变化",
        "易激惹",
        "食欲变化",
        "注意力困难",
        "疲倦或乏力",
        "性兴趣丧失",
    ],
}


class AnalysisError(ValueError):
    """Raised when archived repeat results cannot be aligned safely."""


def timepoint_sort_key(row: Mapping[str, Any]) -> tuple[int, str]:
    """Keep T0, session_4, … in their clinical order rather than lexical order."""
    value = str(row.get("timepoint", "") or "")
    if value == "T0":
        return (0, value)
    completed = row.get("completed_session_count", "")
    try:
        return (int(completed), value)
    except (TypeError, ValueError):
        match = re.search(r"(\d+)", value)
        return (int(match.group(1)) if match else 10**9, value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="对齐重复量表评估的条目得分与冻结主诉图状态。"
    )
    parser.add_argument("checkpoint", type=Path, help="包含 simulate-*.json 的 checkpoint 目录")
    parser.add_argument(
        "--repeat-summary",
        type=Path,
        help="repeat-*_summary.json；省略时在同一存档的 experiment_data/reports 自动定位。",
    )
    parser.add_argument("--agent", help="目标患者名；省略时自动识别唯一启用动态抑郁模块的角色。")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="输出目录；默认在 checkpoint/judge_traces/ 下。",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise AnalysisError("JSON 文件无法解析: {}".format(path)) from exc
    if not isinstance(payload, dict):
        raise AnalysisError("JSON 根节点不是对象: {}".format(path))
    return payload


def candidate_summary_paths(checkpoint: Path) -> Iterable[Path]:
    # checkpoint 位于 <archive>/checkpoints/<run>，报告位于 <archive>/experiment_data/reports。
    if checkpoint.parent.name != "checkpoints":
        return []
    reports_dir = checkpoint.parent.parent / "experiment_data" / "reports"
    if not reports_dir.is_dir():
        return []
    return sorted(
        path
        for path in reports_dir.glob("repeat-*_summary.json")
        if not path.name.endswith("_incomplete.json")
    )


def select_condition(summary: Mapping[str, Any], run_name: str) -> dict[str, Any]:
    conditions = summary.get("conditions", [])
    if not isinstance(conditions, list):
        raise AnalysisError("重复评估 summary 的 conditions 不是列表")
    matches = [item for item in conditions if isinstance(item, dict) and item.get("run_name") == run_name]
    if len(matches) != 1:
        raise AnalysisError(
            "summary 中与 checkpoint run_name={} 匹配的 condition 数为 {}，无法唯一对齐".format(
                run_name, len(matches)
            )
        )
    return matches[0]


def resolve_summary(checkpoint: Path, explicit_path: Path | None) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    if explicit_path:
        path = explicit_path.expanduser()
        if not path.is_file():
            raise FileNotFoundError("未找到重复评估 summary: {}".format(path))
        payload = load_json(path)
        return path, payload, select_condition(payload, checkpoint.name)

    matches: list[tuple[Path, dict[str, Any], dict[str, Any]]] = []
    for path in candidate_summary_paths(checkpoint):
        try:
            payload = load_json(path)
            condition = select_condition(payload, checkpoint.name)
        except AnalysisError:
            continue
        matches.append((path, payload, condition))
    if not matches:
        raise FileNotFoundError(
            "未自动定位到 run_name={} 的 repeat-*_summary.json；请用 --repeat-summary 指定。".format(
                checkpoint.name
            )
        )
    if len(matches) > 1:
        paths = ", ".join(str(item[0]) for item in matches)
        raise AnalysisError("定位到多个匹配的重复评估 summary，请用 --repeat-summary 指定其中一个: {}".format(paths))
    return matches[0]


def resolve_snapshot_path(checkpoint: Path, evaluation: Mapping[str, Any]) -> Path:
    snapshot_name = str(evaluation.get("snapshot_name", "") or "").strip()
    if snapshot_name:
        candidate = checkpoint / snapshot_name
        if candidate.is_file():
            return candidate
        raise FileNotFoundError("评估点指定的 snapshot 不存在: {}".format(candidate))

    # T0 等早期评估点可能没有 snapshot_name，但 summary 仍保存 sim_time。
    sim_time = str(evaluation.get("sim_time", "") or "").strip()
    if re.fullmatch(r"\d{8}-\d{2}:\d{2}", sim_time):
        candidate = checkpoint / "simulate-{}.json".format(sim_time.replace(":", ""))
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "评估点 {} 缺少可解析的 snapshot_name，且无法按 sim_time={} 定位快照。".format(
            evaluation.get("trigger_label", ""), sim_time
        )
    )


def choose_agent(snapshot: Mapping[str, Any], requested_agent: str | None) -> str:
    agents = snapshot.get("agents", {})
    if not isinstance(agents, Mapping):
        raise AnalysisError("快照缺少 agents 对象")
    if requested_agent:
        if requested_agent not in agents:
            raise AnalysisError("快照中不存在目标角色: {}".format(requested_agent))
        return requested_agent

    candidates = []
    for name, payload in agents.items():
        if not isinstance(payload, Mapping):
            continue
        state = payload.get("depression_dynamic_state", {})
        if isinstance(state, Mapping) and bool(state.get("enabled", False)):
            candidates.append(str(name))
    if len(candidates) != 1:
        raise AnalysisError("无法唯一识别动态抑郁患者（候选：{}）；请传 --agent。".format("、".join(candidates) or "无"))
    return candidates[0]


def configured_stage(
    state: Mapping[str, Any],
    current_id: str,
    checkpoint: Path,
) -> dict[str, Any]:
    """Load a stage omitted from an early snapshot's ``runtime_stages``.

    T0 snapshots can already point at the configured initial stage while their
    runtime stage cache is still empty.  The archived ``config_reference`` uses
    the original container path, so remap its ``experiment_data`` suffix to the
    current archive before looking up the stage definition.
    """
    reference = state.get("config_reference", {})
    if not isinstance(reference, Mapping):
        return {}
    raw_path = str(reference.get("config_path", "") or "").strip()
    if not raw_path:
        return {}

    referenced_path = Path(raw_path)
    candidates = [referenced_path]
    if "experiment_data" in referenced_path.parts and checkpoint.parent.name == "checkpoints":
        suffix_start = referenced_path.parts.index("experiment_data")
        archive_root = checkpoint.parent.parent
        candidates.append(archive_root.joinpath(*referenced_path.parts[suffix_start:]))

    seen: set[Path] = set()
    for config_path in candidates:
        if config_path in seen or not config_path.is_file():
            continue
        seen.add(config_path)
        config = load_json(config_path)
        complaint_graph = config.get("complaint_graph", {})
        stages = complaint_graph.get("stages", []) if isinstance(complaint_graph, Mapping) else []
        if not isinstance(stages, list):
            continue
        stage = next(
            (
                item
                for item in stages
                if isinstance(item, Mapping)
                and str(item.get("id", "") or "") == current_id
            ),
            {},
        )
        if stage:
            return dict(stage)
    return {}


def graph_features(snapshot: Mapping[str, Any], agent: str, checkpoint: Path) -> dict[str, Any]:
    agent_payload = (snapshot.get("agents", {}) or {}).get(agent, {})
    state = agent_payload.get("depression_dynamic_state", {}) if isinstance(agent_payload, Mapping) else {}
    manager = state.get("complaint_graph_manager", {}) if isinstance(state, Mapping) else {}
    if not isinstance(manager, Mapping):
        raise AnalysisError("角色 {} 的快照没有 complaint_graph_manager".format(agent))

    current_id = str(manager.get("current_stage_id", "") or "")
    stages = manager.get("runtime_stages", [])
    stages = stages if isinstance(stages, list) else []
    current_stage = next(
        (stage for stage in stages if isinstance(stage, Mapping) and str(stage.get("id", "") or "") == current_id),
        {},
    )
    if current_id and not current_stage:
        current_stage = configured_stage(state, current_id, checkpoint)
    if not current_id or not current_stage:
        raise AnalysisError("角色 {} 的快照无法定位当前主诉节点".format(agent))

    history = manager.get("stage_history", [])
    history = history if isinstance(history, list) else []
    advances = [row for row in history if isinstance(row, Mapping) and row.get("action") == "advance"]
    return {
        "graph_current_stage_id": current_id,
        "graph_current_stage_label": str(current_stage.get("label", "") or current_id),
        "graph_current_stage_summary": str(current_stage.get("summary", "") or ""),
        "graph_current_core_belief": str(current_stage.get("core_belief", "") or ""),
        "graph_stage_index": manager.get("stage_index", ""),
        "graph_runtime_stage_count": len(stages),
        "graph_planned_path_length": len(manager.get("planned_graph", []) or []),
        "graph_history_count": len(history),
        "graph_advance_count": len(advances),
        "graph_chat_advance_count": sum(1 for row in advances if row.get("source") == "chat"),
        "graph_reflection_advance_count": sum(1 for row in advances if row.get("source") == "reflection"),
    }


def common_row_fields(condition: Mapping[str, Any], evaluation: Mapping[str, Any], agent: str, snapshot_path: Path, graph: Mapping[str, Any]) -> dict[str, Any]:
    row = {
        "condition_name": str(condition.get("condition_name", "") or ""),
        "run_name": str(condition.get("run_name", "") or ""),
        "timepoint": str(evaluation.get("trigger_label", "") or ""),
        "completed_session_count": evaluation.get("completed_session_count", ""),
        "sim_time": str(evaluation.get("sim_time", "") or ""),
        "snapshot_name": snapshot_path.name,
        "agent": agent,
    }
    row.update(graph)
    return row


def build_rows(condition: Mapping[str, Any], agent_arg: str | None, checkpoint: Path) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    evaluations = condition.get("evaluations", [])
    if not isinstance(evaluations, list) or not evaluations:
        raise AnalysisError("匹配的 condition 没有 evaluations")

    for evaluation in evaluations:
        if not isinstance(evaluation, Mapping):
            warnings.append("跳过非对象 evaluation")
            continue
        snapshot_path = resolve_snapshot_path(checkpoint, evaluation)
        snapshot = load_json(snapshot_path)
        agent = choose_agent(snapshot, agent_arg)
        graph = graph_features(snapshot, agent, checkpoint)
        common = common_row_fields(condition, evaluation, agent, snapshot_path, graph)
        scales = evaluation.get("scales", {})
        if not isinstance(scales, Mapping):
            warnings.append("{} 的 scales 不是对象".format(common["timepoint"]))
            continue
        for scale_name, scale_payload in scales.items():
            if not isinstance(scale_payload, Mapping):
                continue
            item_labels = ITEM_LABELS.get(str(scale_name), [])
            repeat_runs = scale_payload.get("repeat_runs", [])
            if not isinstance(repeat_runs, list):
                warnings.append("{} {} 的 repeat_runs 不是列表".format(common["timepoint"], scale_name))
                continue
            for repeat_run in repeat_runs:
                if not isinstance(repeat_run, Mapping) or repeat_run.get("status") != "ok":
                    continue
                scores = repeat_run.get("item_scores", [])
                if not isinstance(scores, list):
                    warnings.append("{} {} r{} 没有 item_scores".format(common["timepoint"], scale_name, repeat_run.get("repeat", "?")))
                    continue
                for item_id, score in enumerate(scores, start=1):
                    if isinstance(score, bool) or not isinstance(score, (int, float)):
                        warnings.append("{} {} r{} item{} 得分无效".format(common["timepoint"], scale_name, repeat_run.get("repeat", "?"), item_id))
                        continue
                    row = dict(common)
                    row.update(
                        {
                            "scale": str(scale_name),
                            "item_id": item_id,
                            "item_label": item_labels[item_id - 1] if item_id <= len(item_labels) else "条目{}".format(item_id),
                            "repeat": repeat_run.get("repeat", ""),
                            "score": score,
                        }
                    )
                    rows.append(row)
    if not rows:
        raise AnalysisError("没有可用的重复量表条目得分")
    return rows, warnings


def summarize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    key_fields = [
        "condition_name", "run_name", "timepoint", "completed_session_count", "sim_time", "snapshot_name", "agent",
        "graph_current_stage_id", "graph_current_stage_label", "graph_current_stage_summary", "graph_current_core_belief",
        "graph_stage_index", "graph_runtime_stage_count", "graph_planned_path_length", "graph_history_count",
        "graph_advance_count", "graph_chat_advance_count", "graph_reflection_advance_count", "scale", "item_id", "item_label",
    ]
    for row in rows:
        grouped[tuple(row.get(field, "") for field in key_fields)].append(row)

    summaries: list[dict[str, Any]] = []
    for key, members in grouped.items():
        scores = [float(item["score"]) for item in members]
        row = dict(zip(key_fields, key))
        row.update(
            {
                "repeat_count": len(scores),
                "score_mean": round(statistics.mean(scores), 4),
                "score_median": round(statistics.median(scores), 4),
                "score_stddev": round(statistics.stdev(scores), 4) if len(scores) > 1 else 0.0,
                "score_min": min(scores),
                "score_max": max(scores),
                "score_range": max(scores) - min(scores),
            }
        )
        summaries.append(row)
    return sorted(summaries, key=lambda row: (timepoint_sort_key(row), str(row["scale"]), int(row["item_id"])))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0].keys()), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def total_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["timepoint"]), str(row["scale"]))].append(row)
    result = []
    for (timepoint, scale), members in grouped.items():
        by_repeat: dict[Any, float] = defaultdict(float)
        for member in members:
            by_repeat[member["repeat"]] += float(member["score"])
        totals = list(by_repeat.values())
        representative = members[0]
        result.append(
            {
                "timepoint": timepoint,
                "scale": scale,
                "repeat_count": len(totals),
                "total_mean": round(statistics.mean(totals), 2),
                "total_stddev": round(statistics.stdev(totals), 2) if len(totals) > 1 else 0.0,
                "total_min": min(totals),
                "total_max": max(totals),
                "graph_current_stage_label": representative["graph_current_stage_label"],
            }
        )
    return result


def build_report(summary_path: Path, rows: list[dict[str, Any]], item_summaries: list[dict[str, Any]], warnings: list[str]) -> str:
    total_rows = total_summary(rows)
    total_rows.sort(key=lambda row: (timepoint_sort_key(row), str(row["scale"])))
    most_variable = sorted(item_summaries, key=lambda row: (float(row["score_stddev"]), float(row["score_range"])), reverse=True)[:10]
    graph_rows: dict[str, dict[str, Any]] = {}
    for row in rows:
        graph_rows.setdefault(str(row["timepoint"]), row)

    lines = [
        "# 重复评估中的主诉图与量表条目对照",
        "",
        "- 重复评估 summary：`{}`".format(summary_path),
        "- 条目逐次记录数：{}".format(len(rows)),
        "- 说明：同一评估点的 K 次回答共用同一冻结主诉图；K 次重复用于描述量表测量波动，不是 K 个独立主诉图样本。",
        "",
        "## 各评估点的冻结主诉图",
        "",
        "这里的“当前主诉节点”是节点名称；节点概述与核心信念是该节点的完整语义内容，三者应结合阅读。",
        "",
        "| 评估点 | 快照 | 当前节点（名称） | 节点概述 | 核心信念 | 图索引 | 累计推进（对话/反思） |",
        "| --- | --- | --- | --- | --- | ---: | ---: |",
    ]
    for timepoint, row in graph_rows.items():
        lines.append(
            "| {} | {} | {} | {} | {} | {} | {}（{}/{}） |".format(
                timepoint,
                row["snapshot_name"],
                str(row["graph_current_stage_label"]).replace("|", "\\|"),
                str(row["graph_current_stage_summary"]).replace("|", "\\|").replace("\n", "<br>"),
                str(row["graph_current_core_belief"]).replace("|", "\\|").replace("\n", "<br>"),
                row["graph_stage_index"],
                row["graph_advance_count"],
                row["graph_chat_advance_count"],
                row["graph_reflection_advance_count"],
            )
        )

    lines.extend(
        [
            "",
            "## 重复评估总分（同一冻结图下的 K 次波动）",
            "",
            "| 评估点 | 量表 | 总分均值 | 标准差 | 范围 | 对应当前主诉节点 |",
            "| --- | --- | ---: | ---: | --- | --- |",
        ]
    )
    for row in total_rows:
        lines.append(
            "| {} | {} | {} | {} | {}–{} | {} |".format(
                row["timepoint"], row["scale"], row["total_mean"], row["total_stddev"], row["total_min"], row["total_max"],
                str(row["graph_current_stage_label"]).replace("|", "\\|"),
            )
        )

    lines.extend(
        [
            "",
            "## 重复波动最大的条目",
            "",
            "这些条目并不代表主诉图造成了分数变化；它们表示在该冻结图状态下，量表回答最不稳定，适合回看原始回答文本。",
            "",
            "| 评估点 | 量表条目 | 条目均值 | 标准差 | 范围 | 当前主诉节点 |",
            "| --- | --- | ---: | ---: | --- | --- |",
        ]
    )
    for row in most_variable:
        lines.append(
            "| {} | {}-{} {} | {} | {} | {}–{} | {} |".format(
                row["timepoint"], row["scale"], row["item_id"], row["item_label"], row["score_mean"], row["score_stddev"], row["score_min"], row["score_max"],
                str(row["graph_current_stage_label"]).replace("|", "\\|"),
            )
        )
    if warnings:
        lines.extend(["", "## 数据警告", ""])
        lines.extend("- {}".format(item) for item in warnings)
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    checkpoint = args.checkpoint.expanduser()
    if not checkpoint.is_dir():
        raise FileNotFoundError("输入不是 checkpoint 目录: {}".format(checkpoint))
    summary_path, _summary, condition = resolve_summary(checkpoint, args.repeat_summary)
    rows, warnings = build_rows(condition, args.agent, checkpoint)
    item_summaries = summarize_rows(rows)
    output_dir = args.output_dir or (checkpoint / "judge_traces" / OUTPUT_DIRNAME)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "item_repeat_long.csv", rows)
    write_csv(output_dir / "item_summary.csv", item_summaries)
    (output_dir / "report.md").write_text(
        build_report(summary_path, rows, item_summaries, warnings), encoding="utf-8"
    )
    print("已输出 {} 条逐次条目记录和 {} 条聚合条目记录到: {}".format(len(rows), len(item_summaries), output_dir))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, AnalysisError) as exc:
        print("[ERROR] {}".format(exc), file=sys.stderr)
        raise SystemExit(1)
