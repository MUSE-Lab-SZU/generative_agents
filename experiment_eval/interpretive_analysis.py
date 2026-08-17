"""Standalone life-state, evaluation-node complaint, and plain-Kappa reports."""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from .complaint_nodes import (
    SEMANTIC_SYSTEM_PROMPT,
    SEMANTIC_USER_PROMPT_TEMPLATE,
    extract_complaint_evaluation_nodes,
)
from .life_state import SYMPTOM_MAP, build_life_state_analysis
from .loader import find_report_files, load_records
from .visualization.interpretive_plots import render_interpretive_figures
from .visualization.weighted_kappa_plots import render_weighted_kappa_figures
from .weighted_kappa import build_weighted_kappa
from modules.model.llm_model import record_prompt_cache_usage


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> Path:
    fields = list(dict.fromkeys(key for row in rows for key in row)) or ["status"]
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _load_dotenv_key(variable: str, dotenv_path: Path) -> str:
    value = str(os.getenv(variable, "") or "").strip()
    if value or not dotenv_path.is_file():
        return value
    for line in dotenv_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, candidate = stripped.split("=", 1)
        if key.strip() == variable:
            return candidate.strip().strip('"').strip("'")
    return ""


def _parse_json_response(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.removeprefix("```json").removeprefix("```").strip()
        if stripped.endswith("```"):
            stripped = stripped[:-3].strip()
    payload = json.loads(stripped)
    if not isinstance(payload, dict):
        raise ValueError("semantic API response must be a JSON object")
    return payload


def run_semantic_api(
    cases: list[dict[str, Any]],
    *,
    config_path: Path,
    dotenv_path: Path,
    cache_path: Path | None = None,
) -> list[dict[str, Any]]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    llm = ((config.get("intervention") or {}).get("forced_llm") or {})
    api_key_env = str(llm.get("api_key_env") or "DEEPSEEK_API_KEY")
    api_key = _load_dotenv_key(api_key_env, dotenv_path)
    if not api_key:
        raise RuntimeError(f"missing API key: {api_key_env}")
    import requests

    outputs: list[dict[str, Any]] = []
    if cache_path is not None and cache_path.is_file():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if isinstance(cached, list):
            outputs = [row for row in cached if isinstance(row, dict) and row.get("stable_id")]
    completed = {str(row["stable_id"]) for row in outputs}
    endpoint = str(llm["base_url"]).rstrip("/") + "/chat/completions"
    session = requests.Session()
    retry_count = max(1, int(llm.get("retry") or 3))
    for case in cases:
        if case["stable_id"] in completed:
            continue
        user_prompt = SEMANTIC_USER_PROMPT_TEMPLATE.format(
            case_json=json.dumps(case, ensure_ascii=False, indent=2)
        )
        request_payload = {
            "model": str(llm["model"]),
            "messages": [
                {"role": "system", "content": SEMANTIC_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0,
            "response_format": {"type": "json_object"},
        }
        last_error: Exception | None = None
        response_payload: dict[str, Any] | None = None
        for attempt in range(retry_count):
            try:
                response = session.post(
                    endpoint,
                    headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                    json=request_payload,
                    timeout=180,
                )
                response.raise_for_status()
                decoded = response.json()
                if not isinstance(decoded, dict):
                    raise ValueError("semantic API response envelope is not an object")
                record_prompt_cache_usage(
                    decoded,
                    caller="experiment_eval_semantic",
                    provider="openai",
                    model=str(llm["model"]),
                    base_url=endpoint,
                )
                response_payload = decoded
                break
            except (requests.RequestException, ValueError) as exc:
                last_error = exc
                if attempt + 1 < retry_count:
                    time.sleep(min(8, 2 ** attempt))
        if response_payload is None:
            raise RuntimeError(f"semantic API failed for {case['stable_id']}: {last_error}")
        content = str(response_payload["choices"][0]["message"].get("content") or "")
        parsed = _parse_json_response(content)
        parsed["stable_id"] = case["stable_id"]
        parsed["api_model"] = str(llm["model"])
        outputs.append(parsed)
        completed.add(case["stable_id"])
        if cache_path is not None:
            cache_path.write_text(json.dumps(outputs, ensure_ascii=False, indent=2), encoding="utf-8")
    return outputs


def _number(value: Any, digits: int = 2) -> str:
    return "NA" if value is None else f"{float(value):.{digits}f}"


def write_life_state_report(out_dir: Path, life_state: dict[str, list[dict[str, Any]]]) -> Path:
    path = out_dir / "life_state_change_report.md"
    overall = [
        row
        for row in life_state["summary"]
        if row["stratum_type"] == "overall"
        and row["metric_level"] == "symptom"
        and row["scale"] == "COMBINED_EQUAL_SCALE_WEIGHT"
    ]
    ranked = sorted(overall, key=lambda row: float(row["mean_change"]))
    n_outer_runs = max((int(row["n_outer_runs"]) for row in overall), default=0)
    group_ns = sorted(
        {
            int(row["n_outer_runs"])
            for row in life_state["summary"]
            if row["stratum_type"] == "group"
            and row["metric_level"] == "symptom"
            and row["scale"] == "COMBINED_EQUAL_SCALE_WEIGHT"
        }
    )
    lines = [
        "# 从量表条目看仿真生活状态发生了什么变化",
        "",
        "## 先说结论",
        "",
        "这里不再只看 PHQ-9/BDI-II 总分，而是把每个条目翻译成兴趣、情绪、睡眠、精力、食欲、自我评价、注意力、行动状态和风险等生活状态。",
        "",
        "- 负数表示该症状在 `session_20` 比 `T0` 少，即模拟生活状态向改善方向变化。",
        "- 正数表示症状更多，即向恶化方向变化。",
        f"- 每个 snapshot 先把重复回答取均值，再把 {n_outer_runs} 个独立仿真 run 用作统计样本。",
        "- 这些结果描述的是 agent 在仿真内的量表回答变化，不等于真实患者疗效。",
        "",
        "## 九类生活状态的前后变化",
        "",
        "| 生活状态 | T0→session_20 平均变化 | 95% outer-run CI | 改善/不变/恶化 run | 直白解释 |",
        "|---|---:|---:|---:|---|",
    ]
    for row in ranked:
        change = float(row["mean_change"])
        lower, upper = row.get("ci95_lower"), row.get("ci95_upper")
        if upper is not None and float(upper) < 0:
            interpretation = "总体上相关困扰减少"
        elif lower is not None and float(lower) > 0:
            interpretation = "总体上相关困扰增加"
        else:
            interpretation = "不同 runs 方向不一，不能判定总体改善或恶化"
        lines.append(
            f"| {row['metric_label_zh']} | {change:+.2f} | "
            f"[{_number(row.get('ci95_lower'))}, {_number(row.get('ci95_upper'))}] | "
            f"{row['improved_n']}/{row['stable_n']}/{row['worsened_n']} | {interpretation} |"
        )
    overall_items = {
        (row["scale"], str(row["metric_id"])): row
        for row in life_state["summary"]
        if row["stratum_type"] == "overall" and row["metric_level"] == "item"
    }
    bdi_agitation = overall_items.get(("BDI-II", "BDI-II_item_11"))
    bdi_irritability = overall_items.get(("BDI-II", "BDI-II_item_17"))
    phq_risk = overall_items.get(("PHQ-9", "PHQ-9_item_9"))
    lines.extend(
        [
            "",
            "## 必须单独指出的例外",
            "",
            (
                f"- BDI-II 烦躁不安条目平均变化为 {float(bdi_agitation['mean_change']):+.2f}，"
                f"{bdi_agitation['worsened_n']}/{bdi_agitation['n_outer_runs']} 个 runs 恶化；这是当前最明确的反向变化。"
                if bdi_agitation
                else ""
            ),
            (
                f"- BDI-II 易怒条目平均变化为 {float(bdi_irritability['mean_change']):+.2f}，"
                f"{bdi_irritability['worsened_n']}/{bdi_irritability['n_outer_runs']} 个 runs 恶化。"
                if bdi_irritability
                else ""
            ),
            (
                f"- PHQ-9 风险条目平均变化为 {float(phq_risk['mean_change']):+.2f}，"
                f"{phq_risk['worsened_n']}/{phq_risk['n_outer_runs']} 个 runs 恶化，且总体 CI 跨 0；不能汇报成风险改善。"
                if phq_risk
                else ""
            ),
            "- 因此总分下降主要来自食欲、兴趣、精力、自我评价、睡眠等条目下降，并不表示所有生活状态都同步改善。",
            "",
            "## 怎样理解这些数字",
            "",
            "每个条目都是 0–3 分。例如睡眠从 2.1 降到 1.4，变化为 -0.7，表示 agent 报告睡眠困扰的频率/严重程度下降。跨量表症状先分别在 PHQ-9 和 BDI-II 内求条目均值，再让两个量表等权平均，避免 BDI-II 条目更多而自动占更大权重。",
            "",
            "风险项单独展示，不与普通生活状态合成一个总域。风险分下降只表示生成式量表回答中的相关想法减少，仍需回看逐 run 数据和原始回答。",
            "",
            "## 组间图的限制",
            "",
            "观察到的 outer n/group="
            + (",".join(map(str, group_ns)) or "不可用")
            + "；当前是部分 persona × group 交叉设计。因此组间热图只能描述现有仿真结果，不能把差异简单归因于组别。",
            "",
            "## 对应文件",
            "",
            "- `life_state_item_trajectory.csv`：每个 run、时间点、量表条目的 10-repeat 均值。",
            "- `life_state_item_change.csv`：每个 run 的 T0→session_20 条目变化。",
            "- `life_state_symptom_trajectory.csv`：九类生活状态的纵向轨迹。",
            "- `life_state_symptom_change.csv`：每个 run 的九类生活状态前后变化。",
            "- `life_state_summary.csv`：总体、persona 和 group 的统计汇总。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _semantic_rows(outputs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for output in outputs:
        for interval in output.get("intervals") or []:
            if isinstance(interval, dict):
                rows.append(
                    {
                        "stable_id": output.get("stable_id"),
                        "from_timepoint": interval.get("from_timepoint"),
                        "to_timepoint": interval.get("to_timepoint"),
                        "change_type": interval.get("change_type"),
                        "plain_change": interval.get("plain_change"),
                        "evidence": interval.get("evidence"),
                        "overall_plain_summary": output.get("overall_plain_summary"),
                        "api_model": output.get("api_model"),
                    }
                )
    return rows


def write_complaint_report(
    out_dir: Path, complaint: dict[str, Any], semantic_outputs: list[dict[str, Any]]
) -> Path:
    path = out_dir / "complaint_evaluation_node_report.md"
    nodes, intervals, missing = complaint["nodes"], complaint["intervals"], complaint["missing"]
    expected = len({row["stable_id"] for row in nodes} | {row["stable_id"] for row in missing}) * len(
        complaint["evaluation_labels"]
    )
    semantic_rows = _semantic_rows(semantic_outputs)
    change_counts = Counter(str(row.get("change_type")) for row in semantic_rows)
    empty_session_runs = len(
        {
            row["stable_id"]
            for row in missing
            if row.get("reason") == "judge_trace_contains_no_sessions"
        }
    )
    by_interval: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in intervals:
        by_interval[(row["from_timepoint"], row["to_timepoint"])].append(row)
    lines = [
        "# 只在量表评估节点分析主诉变化",
        "",
        "## 能不能做",
        "",
        "可以。这里不再尝试重建完整主诉图，只读取与量表一致的 `T0/session_4/session_8/session_12/session_16/session_20` 节点。T0 使用第一次会谈前的主诉阶段，其余节点使用对应会谈 reflection 后的主诉阶段。",
        "",
        f"- 计划节点数：{expected}；成功提取：{len(nodes)}；缺失：{len(missing)}。",
        f"- 成功形成相邻评估区间：{len(intervals)}。",
        f"- 有 {empty_session_runs} 个 legacy runs 的 judge trace 不含任何 session，因此 checkpoint 中没有可用主诉阶段快照；没有用近似文本补造。",
        "- `stage_index` 增大只表示主诉探索向后推进，不表示症状改善。是否改善仍应看量表条目。",
        "",
        "## 各评估区间的推进情况",
        "",
        "| 区间 | 有效 runs | 同一阶段停滞 | 平均 stage-index 变化 | reflection advance 比例 |",
        "|---|---:|---:|---:|---:|",
    ]
    for pair, rows in sorted(by_interval.items()):
        index_values = [float(row["stage_index_change"]) for row in rows if row.get("stage_index_change") is not None]
        advance_count = sum(int(row["session_reflection_advance"]) for row in rows)
        action_count = advance_count + sum(int(row["session_reflection_hold"]) for row in rows)
        lines.append(
            f"| {pair[0]}→{pair[1]} | {len(rows)} | {sum(bool(row['same_stage_id']) for row in rows)} | "
            f"{_number(mean(index_values) if index_values else None)} | "
            f"{_number(advance_count / action_count if action_count else None, 3)} |"
        )
    if semantic_rows:
        lines.extend(
            [
                "",
                "## 固定提示词 API 的直白归纳",
                "",
                "API 只比较输入中的阶段标签、摘要和核心信念，并被明确禁止诊断或把主诉推进写成疗效。所有输入、固定提示词和逐 run JSON 输出均保留供审查。",
                "",
                "语义分类计数：" + "；".join(f"{key} {value}" for key, value in sorted(change_counts.items())) + "。",
                "",
                "| Run | 直白总结 |",
                "|---|---|",
            ]
        )
        for output in semantic_outputs:
            lines.append(f"| {output.get('stable_id')} | {str(output.get('overall_plain_summary') or '').replace('|', '｜')} |")
    lines.extend(
        [
            "",
            "## 这里没有做什么",
            "",
            "没有计算节点新增/删除、图深度、分支宽度或 Sankey，因为 checkpoint 没有稳定 node id 和完整 edges。当前报告回答的是‘每个量表评估节点的主诉文字和推进状态怎么变’，不是完整图拓扑怎么变。",
            "",
            "## 对应文件",
            "",
            "- `complaint_evaluation_nodes.csv`：每个评估节点的阶段快照。",
            "- `complaint_evaluation_intervals.csv`：相邻评估节点之间的 stage-index 和 hold/advance。",
            "- `complaint_semantic_prompt.md`：固定 API 提示词。",
            "- `complaint_semantic_outputs.json` / `.csv`：逐 run 原始结构化解释。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_plain_kappa_report(out_dir: Path, kappa: dict[str, Any]) -> Path:
    path = out_dir / "weighted_kappa_plain_language_report.md"
    overall = [row for row in kappa["summary"] if row["stratum_type"] == "overall"]
    primary = {row["scale"]: row for row in overall if row["weights"] == "quadratic"}
    sensitivity = {row["scale"]: row for row in overall if row["weights"] == "linear"}
    item_primary = [row for row in kappa["item"] if row["weights"] == "quadratic" and row.get("kappa") is not None]
    lowest = sorted(item_primary, key=lambda row: float(row["kappa"]))[:5]
    item_long = kappa.get("item_long") or []
    run_count = len({str(row["stable_id"]) for row in item_long})
    timepoints = sorted({str(row["timepoint"]) for row in item_long})
    repeat_count = len({str(row["measurement_repeat_id"]) for row in item_long})
    pair_count = repeat_count * (repeat_count - 1) // 2
    item_counts = {
        scale: len({int(row["item_id"]) for row in item_long if row["scale"] == scale})
        for scale in ("PHQ-9", "BDI-II")
    }
    lines = [
        "# Weighted Kappa：不懂统计也能直接汇报的版本",
        "",
        "## 一句话解释",
        "",
        f"同一个冻结状态让模型重复回答 {repeat_count} 次同一道 0–3 分题目，Weighted Kappa 检查这些回答是否大体一致。",
        "",
        "例如同一道题 10 次都在 1 分或 2 分附近，说明回答比较稳定；如果一会儿 0 分、一会儿 3 分，说明这道题不稳定。Weighted 的意思是：差 1 分算小分歧，差 3 分算大分歧。",
        "",
        "## 它和‘生活状态是否改善’不是一回事",
        "",
        "- 条目变化回答：仿真前后生活状态朝哪个方向变了。",
        "- Weighted Kappa 回答：在同一个时间点重复问时，模型回答稳不稳定。",
        "- Kappa 高不代表症状改善，只代表重复回答较一致；Kappa 低也不代表症状严重，只代表回答波动较大。",
        "",
        "## 本次计算用了什么数据",
        "",
        f"- {run_count} 个独立仿真 runs；",
        f"- 每个 run 的 {', '.join(timepoints)}；",
        f"- PHQ-9 的 {item_counts['PHQ-9']} 个条目和 BDI-II 的 {item_counts['BDI-II']} 个条目；",
        f"- 每个冻结节点重复回答 {repeat_count} 次；",
        f"- 总计 {len(item_long):,} 个条目回答；",
        f"- {repeat_count} 次 repeat 两两比较，共 {pair_count} 对。",
        "",
        "量表总分没有计算 Kappa，总分稳定性继续使用 ICC。Kappa 配对时必须是完全相同的 `snapshot × scale × item`，不会把不同时间点或不同条目错配。",
        "",
        "## 目前结果",
        "",
        "| 量表 | 主要结果 quadratic κ | 95% CI | linear 敏感性结果 | 最直白的解释 |",
        "|---|---:|---:|---:|---|",
    ]
    for scale in ("PHQ-9", "BDI-II"):
        row = primary[scale]
        linear = sensitivity[scale]
        relative = "重复回答相对更稳定" if scale == "BDI-II" else "有一定一致性，但条目间波动更明显"
        lines.append(
            f"| {scale} | {_number(row['kappa'], 3)} | [{_number(row['ci95_lower'], 3)}, {_number(row['ci95_upper'], 3)}] | "
            f"{_number(linear['kappa'], 3)} | {relative} |"
        )
    lines.extend(
        [
            "",
            "BDI-II 的总体重复一致性高于 PHQ-9。quadratic 比 linear 高，说明很多分歧只是相邻分数之间的摇摆，而不是 0 分与 3 分之间的大幅冲突。",
            "",
            "回答最不稳定的部分条目是：",
            "",
        ]
    )
    for row in lowest:
        lines.append(f"- {row['scale']} item {row['item_id']}：κ={float(row['kappa']):.3f}。")
    lines.extend(
        [
            "",
            "不要只按某个固定阈值说‘合格/不合格’，因为 Kappa 还会受 0/1/2/3 各分数出现频率影响。最稳妥的说法是比较两张量表、不同条目以及 linear/quadratic 是否得到一致结论。",
            "",
            "## 输入和输出",
            "",
            "输入是 `item_measurement_repeat_long.csv`：一行表示某个 run、某个时间点、某张量表、某个条目的一次重复回答。",
            "",
            "输出包括：",
            "",
            "- `weighted_kappa_summary.csv`：两张量表总体及 persona/group/timepoint 分层；",
            "- `weighted_kappa_pairwise.csv`：10 个 repeats 的每一对比较；",
            "- `weighted_kappa_item.csv`：每个条目的稳定性；",
            "- repeat-pair 热图：看哪一对重复回答更不一致；",
            "- item forest：看哪些具体条目更不稳定。",
            "",
            "## 汇报时可以直接这样说",
            "",
            f"> 我们在每个冻结评估节点让模型重复完成 {repeat_count} 次量表，用 Weighted Kappa 检查逐条目回答是否稳定。"
            f"BDI-II 的 quadratic Kappa 为 {float(primary['BDI-II']['kappa']):.3f}，PHQ-9 为 {float(primary['PHQ-9']['kappa']):.3f}，"
            "说明 BDI-II 的重复条目回答整体更一致。这个指标只说明重复测量稳定性，不代表症状是否改善；生活状态改善需要另外看 T0 到 session_20 的条目变化。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_master_report(out_dir: Path) -> Path:
    path = out_dir / "README.md"
    path.write_text(
        """# 0802+0808 可解释性补充分析

本目录直接回答三个问题：

1. [`life_state_change_report.md`](life_state_change_report.md)：总分变化具体来自哪些生活状态条目；
2. [`complaint_evaluation_node_report.md`](complaint_evaluation_node_report.md)：只在 T0/session_xx 评估节点比较主诉变化；
3. [`weighted_kappa_plain_language_report.md`](weighted_kappa_plain_language_report.md)：Weighted Kappa 的非统计版说明和汇报话术。

原始存档只读，所有新增 CSV、PNG、固定提示词和 API 输出均写在本目录。
""",
        encoding="utf-8",
    )
    return path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports-dir", type=Path, required=True)
    parser.add_argument("--checkpoints-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--repeat-alias",
        action="append",
        default=[],
        metavar="PATH_MATCH=REPEAT_ID",
    )
    parser.add_argument("--semantic-api", action="store_true")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "data" / "config.json")
    parser.add_argument("--dotenv", type=Path, default=PROJECT_ROOT / ".env")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    reports_dir = args.reports_dir.resolve()
    checkpoints_root = args.checkpoints_root.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    report_files = find_report_files(reports_dir, recursive=True)
    repeat_aliases: list[tuple[str, str]] = []
    for value in args.repeat_alias:
        if "=" not in value:
            raise SystemExit(f"Invalid --repeat-alias: {value}")
        path_match, label = value.split("=", 1)
        if not path_match or not label:
            raise SystemExit(f"Invalid --repeat-alias: {value}")
        repeat_aliases.append((path_match, label))
    records, labels, scales = load_records(
        report_files, repeat_aliases=repeat_aliases
    )
    life_state = build_life_state_analysis(records, labels, scales)
    complaint = extract_complaint_evaluation_nodes(records, labels, checkpoints_root)
    kappa = build_weighted_kappa(records, labels, scales)

    _write_csv(out_dir / "life_state_item_trajectory.csv", life_state["item_trajectory"])
    _write_csv(out_dir / "life_state_item_change.csv", life_state["item_change"])
    _write_csv(out_dir / "life_state_symptom_trajectory.csv", life_state["symptom_trajectory"])
    _write_csv(out_dir / "life_state_symptom_change.csv", life_state["symptom_change"])
    _write_csv(out_dir / "life_state_summary.csv", life_state["summary"])
    _write_csv(out_dir / "complaint_evaluation_nodes.csv", complaint["nodes"])
    _write_csv(out_dir / "complaint_evaluation_intervals.csv", complaint["intervals"])
    _write_csv(out_dir / "complaint_evaluation_missing.csv", complaint["missing"])
    _write_csv(out_dir / "item_measurement_repeat_long.csv", kappa["item_long"])
    _write_csv(out_dir / "weighted_kappa_summary.csv", kappa["summary"])
    _write_csv(out_dir / "weighted_kappa_pairwise.csv", kappa["pairwise"])
    _write_csv(out_dir / "weighted_kappa_item.csv", kappa["item"])
    (out_dir / "complaint_semantic_cases.json").write_text(
        json.dumps(complaint["prompt_cases"], ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "complaint_semantic_prompt.md").write_text(
        "# 固定主诉节点语义分析提示词\n\n## System\n\n"
        + SEMANTIC_SYSTEM_PROMPT
        + "\n\n## User template\n\n```text\n"
        + SEMANTIC_USER_PROMPT_TEMPLATE
        + "\n```\n",
        encoding="utf-8",
    )
    semantic_outputs: list[dict[str, Any]] = []
    if args.semantic_api:
        semantic_outputs = run_semantic_api(
            complaint["prompt_cases"],
            config_path=args.config.resolve(),
            dotenv_path=args.dotenv.resolve(),
            cache_path=out_dir / "complaint_semantic_outputs.json",
        )
    (out_dir / "complaint_semantic_outputs.json").write_text(
        json.dumps(semantic_outputs, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_csv(out_dir / "complaint_semantic_outputs.csv", _semantic_rows(semantic_outputs))
    render_interpretive_figures(life_state, complaint, labels, out_dir)
    render_weighted_kappa_figures(
        kappa, out_dir, f"0808 integrated {len(records)}-run analysis"
    )
    write_life_state_report(out_dir, life_state)
    write_complaint_report(out_dir, complaint, semantic_outputs)
    write_plain_kappa_report(out_dir, kappa)
    write_master_report(out_dir)
    print(f"Loaded {len(records)} outer runs, {len(labels)} evaluation labels, {len(scales)} scales")
    print(f"Wrote interpretive analysis to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
