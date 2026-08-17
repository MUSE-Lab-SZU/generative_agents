"""Separate cross-persona G1 and KBD2-condition analyses without pooling outer runs."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from .complaint_nodes import extract_complaint_evaluation_nodes
from .life_state import ITEM_LABELS_ZH, SYMPTOM_MAP, build_life_state_analysis
from .loader import find_report_files, load_records
from .schema import ExperimentRecord, group_sort_key, label_display
from .visualization.stratified_interpretive_plots import render_stratified_figures
from .visualization.weighted_kappa_plots import render_weighted_kappa_figures
from .weighted_kappa import build_weighted_kappa


EPSILON = 0.05


def _record_entity(record: ExperimentRecord, entity_field: str) -> str:
    return record.kbd if entity_field == "persona" else str(getattr(record, entity_field))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> Path:
    fields = list(dict.fromkeys(key for row in rows for key in row)) or ["status"]
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def build_input_inventory(
    records: list[ExperimentRecord], experiment_data_root: Path
) -> list[dict[str, Any]]:
    """Record local/remote provenance from each archived run's immutable metadata."""
    rows: list[dict[str, Any]] = []
    for record in records:
        meta_path = experiment_data_root / record.run_name / "trial_meta.json"
        source_agent_dir = ""
        if meta_path.is_file():
            payload = json.loads(meta_path.read_text(encoding="utf-8"))
            source_agent_dir = str(payload.get("variant_source_agent_dir") or "")
        if source_agent_dir.startswith("/workspace/"):
            data_origin = "remote_workspace"
        elif source_agent_dir.startswith("/home/"):
            data_origin = "local_server"
        else:
            data_origin = "unknown"
        analysis_sets = []
        if record.group == "G1":
            analysis_sets.append("cross_persona_g1")
        if record.kbd == "KBD2":
            analysis_sets.append("kbd2_conditions")
        rows.append(
            {
                "stable_id": record.stable_id,
                "persona": record.kbd,
                "group": record.group,
                "outer_run_id": record.repeat_id,
                "run_name": record.run_name,
                "data_origin": data_origin,
                "source_agent_dir": source_agent_dir,
                "trial_meta_path": str(meta_path),
                "analysis_sets": ";".join(analysis_sets),
            }
        )
    return sorted(
        rows,
        key=lambda row: (
            group_sort_key(str(row["persona"])),
            group_sort_key(str(row["group"])),
            group_sort_key(str(row["outer_run_id"])),
        ),
    )


def _direction(value: float) -> str:
    if value < -EPSILON:
        return "改善"
    if value > EPSILON:
        return "恶化"
    return "稳定"


def build_replicate_agreement(
    symptom_changes: list[dict[str, Any]], entity_field: str
) -> list[dict[str, Any]]:
    rows = [row for row in symptom_changes if row["scale"] == "COMBINED_EQUAL_SCALE_WEIGHT"]
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row[entity_field]), str(row["symptom_id"]))].append(row)
    output: list[dict[str, Any]] = []
    for (entity, symptom_id), values in grouped.items():
        by_repeat = {str(row["outer_run_id"]): row for row in values}
        first, second = by_repeat.get("R01"), by_repeat.get("R02")
        if first is None or second is None:
            agreement = "重复不足"
        else:
            directions = (_direction(float(first["change"])), _direction(float(second["change"])))
            if directions == ("改善", "改善"):
                agreement = "两次均改善"
            elif directions == ("恶化", "恶化"):
                agreement = "两次均恶化"
            elif directions == ("稳定", "稳定"):
                agreement = "两次均稳定"
            else:
                agreement = "方向不一致或含稳定"
        example = first or second or values[0]
        output.append(
            {
                entity_field: entity,
                "symptom_id": symptom_id,
                "symptom_label_zh": example["symptom_label_zh"],
                "symptom_label_en": example["symptom_label_en"],
                "R01_change": float(first["change"]) if first else None,
                "R02_change": float(second["change"]) if second else None,
                "R01_direction": _direction(float(first["change"])) if first else "缺失",
                "R02_direction": _direction(float(second["change"])) if second else "缺失",
                "agreement": agreement,
                "summary_rule": "不跨 outer run 求平均；检查 R01/R02 是否复现同一方向",
            }
        )
    return sorted(output, key=lambda row: (group_sort_key(str(row[entity_field])), str(row["symptom_id"])))


def _alignment_class(advance_rate: float, general_change: float) -> str:
    progressed = advance_rate >= 0.5
    improved = general_change < -EPSILON
    if progressed and improved:
        return "推进且同期症状改善"
    if not progressed and improved:
        return "主诉停滞但同期症状改善"
    if progressed and not improved:
        return "主诉推进但同期症状未改善"
    return "主诉停滞且同期症状未改善"


def build_complaint_outcome_alignment(
    life_state: dict[str, list[dict[str, Any]]], complaint: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    symptom_rows = [
        row
        for row in life_state["symptom_trajectory"]
        if row["scale"] == "COMBINED_EQUAL_SCALE_WEIGHT"
    ]
    snapshot_general: dict[tuple[str, str], float] = {}
    snapshot_risk: dict[tuple[str, str], float] = {}
    grouped_snapshot: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in symptom_rows:
        grouped_snapshot[(row["stable_id"], row["timepoint"])].append(row)
    for key, rows in grouped_snapshot.items():
        general = [float(row["symptom_score"]) for row in rows if row["symptom_id"] != "self_harm_risk"]
        risk = next((float(row["symptom_score"]) for row in rows if row["symptom_id"] == "self_harm_risk"), None)
        if general:
            snapshot_general[key] = mean(general)
        if risk is not None:
            snapshot_risk[key] = risk

    interval_rows: list[dict[str, Any]] = []
    for row in complaint["intervals"]:
        start_key = (row["stable_id"], row["from_timepoint"])
        end_key = (row["stable_id"], row["to_timepoint"])
        if start_key not in snapshot_general or end_key not in snapshot_general:
            continue
        action_total = int(row["session_reflection_advance"]) + int(row["session_reflection_hold"])
        if not action_total:
            continue
        advance_rate = int(row["session_reflection_advance"]) / action_total
        general_change = snapshot_general[end_key] - snapshot_general[start_key]
        risk_change = (
            snapshot_risk[end_key] - snapshot_risk[start_key]
            if start_key in snapshot_risk and end_key in snapshot_risk
            else None
        )
        interval_rows.append(
            row
            | {
                "general_symptom_from": snapshot_general[start_key],
                "general_symptom_to": snapshot_general[end_key],
                "general_symptom_change": general_change,
                "general_symptom_direction": _direction(general_change),
                "risk_change": risk_change,
                "risk_worsened": risk_change is not None and risk_change > EPSILON,
                "complaint_advance_rate": advance_rate,
                "complaint_progress_rule": "advance rate >= 0.5",
                "alignment_class": _alignment_class(advance_rate, general_change),
            }
        )

    combined_changes = [
        row
        for row in life_state["symptom_change"]
        if row["scale"] == "COMBINED_EQUAL_SCALE_WEIGHT"
    ]
    by_run_change: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in combined_changes:
        by_run_change[row["stable_id"]].append(row)
    by_run_intervals: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in interval_rows:
        by_run_intervals[row["stable_id"]].append(row)
    run_rows: list[dict[str, Any]] = []
    for stable_id, changes in by_run_change.items():
        intervals = by_run_intervals.get(stable_id, [])
        if not intervals:
            continue
        general_values = [float(row["change"]) for row in changes if row["symptom_id"] != "self_harm_risk"]
        risk = next((float(row["change"]) for row in changes if row["symptom_id"] == "self_harm_risk"), None)
        advance = sum(int(row["session_reflection_advance"]) for row in intervals)
        hold = sum(int(row["session_reflection_hold"]) for row in intervals)
        rate = advance / (advance + hold) if advance + hold else None
        if not general_values or rate is None:
            continue
        general_change = mean(general_values)
        first = changes[0]
        run_rows.append(
            {
                "stable_id": stable_id,
                "outer_run_id": first["outer_run_id"],
                "persona": first["persona"],
                "group": first["group"],
                "available_evaluation_intervals": len(intervals),
                "complaint_advance": advance,
                "complaint_hold": hold,
                "complaint_advance_rate": rate,
                "general_symptom_change": general_change,
                "general_symptom_direction": _direction(general_change),
                "risk_change": risk,
                "risk_worsened": risk is not None and risk > EPSILON,
                "alignment_class": _alignment_class(rate, general_change),
            }
        )
    return interval_rows, run_rows


def _write_subset_csvs(
    out_dir: Path,
    life_state: dict[str, list[dict[str, Any]]],
    agreement: list[dict[str, Any]],
    complaint: dict[str, Any],
    interval_alignment: list[dict[str, Any]],
    run_alignment: list[dict[str, Any]],
    kappa: dict[str, Any],
) -> None:
    for key, filename in (
        ("item_trajectory", "life_state_item_trajectory.csv"),
        ("item_change", "life_state_item_change.csv"),
        ("symptom_trajectory", "life_state_symptom_trajectory.csv"),
        ("symptom_change", "life_state_symptom_change.csv"),
    ):
        _write_csv(out_dir / filename, life_state[key])
    _write_csv(out_dir / "life_state_r01_r02_reproducibility.csv", agreement)
    _write_csv(out_dir / "complaint_evaluation_nodes.csv", complaint["nodes"])
    _write_csv(out_dir / "complaint_evaluation_intervals.csv", complaint["intervals"])
    _write_csv(out_dir / "complaint_evaluation_missing.csv", complaint["missing"])
    _write_csv(out_dir / "complaint_interval_outcome_alignment.csv", interval_alignment)
    _write_csv(out_dir / "complaint_run_outcome_alignment.csv", run_alignment)
    _write_csv(out_dir / "weighted_kappa_summary.csv", kappa["summary"])
    _write_csv(out_dir / "weighted_kappa_pairwise.csv", kappa["pairwise"])
    _write_csv(out_dir / "weighted_kappa_item.csv", kappa["item"])


def _fmt(value: Any, digits: int = 2) -> str:
    return "NA" if value is None else f"{float(value):.{digits}f}"


def write_subset_report(
    out_dir: Path,
    *,
    title: str,
    records: list[ExperimentRecord],
    entity_field: str,
    entity_type: str,
    agreement: list[dict[str, Any]],
    life_state: dict[str, list[dict[str, Any]]],
    complaint: dict[str, Any],
    interval_alignment: list[dict[str, Any]],
    run_alignment: list[dict[str, Any]],
    kappa: dict[str, Any],
) -> Path:
    path = out_dir / "analysis_report.md"
    entities = sorted({_record_entity(record, entity_field) for record in records}, key=group_sort_key)
    repeat_ids = sorted({record.repeat_id for record in records}, key=group_sort_key)
    later_repeat_text = "、".join(repeat_ids[2:]) or "其余重复"
    agreement_counts: dict[str, Counter[str]] = defaultdict(Counter)
    symptom_labels: dict[str, str] = {}
    for row in agreement:
        agreement_counts[row["symptom_id"]][row["agreement"]] += 1
        symptom_labels[row["symptom_id"]] = row["symptom_label_zh"]
    lines = [
        f"# {title}",
        "",
        "## 分析口径",
        "",
        f"- 包含 {len(entities)} 个{('人设' if entity_field == 'persona' else '实验条件')}、{len(records)} 次独立仿真；观察到的 outer run ID 为 {', '.join(repeat_ids)}。",
        "- 生活状态分析单位是一条独立 outer run，不跨人设或条件先求平均。",
        "- 10 次 measurement repeat 只在同一 snapshot 内取均值，用于降低测量随机性。",
        f"- `R01/R02 reproducibility` 专图只比较前两次预设复现；{later_repeat_text} 仍进入单-run 热图、轨迹、条目、主诉和 Kappa 统计。",
        "",
        "## 一、生活状态：哪些变化能在 R01/R02 复现",
        "",
        "### 每个生活状态看哪些量表条目",
        "",
        "| 生活状态 | PHQ-9 条目 | BDI-II 条目 |",
        "|---|---|---|",
    ]
    for mapping in SYMPTOM_MAP.values():
        phq_items = "；".join(
            f"I{item_id}（{ITEM_LABELS_ZH['PHQ-9'][item_id]}）"
            for item_id in mapping.get("PHQ-9", [])
        )
        bdi_items = "；".join(
            f"I{item_id}（{ITEM_LABELS_ZH['BDI-II'][item_id]}）"
            for item_id in mapping.get("BDI-II", [])
        )
        lines.append(f"| {mapping['zh']} | {phq_items} | {bdi_items} |")
    lines.extend(
        [
            "",
            "**生活状态分数的合成顺序：**",
            "",
            "1. 在同一个 run、同一个评估时点、同一道题内，先把 10 次 measurement repeat 的条目分数取均值。",
            "2. 一个量表内若映射了多道题，先对这些题取均值。例如 BDI-II 的‘精力与疲劳’是 I15 与 I20 的均值。",
            "3. 再把 PHQ-9 状态分数和 BDI-II 状态分数各赋 50% 权重取均值。这样 BDI-II 条目较多的状态不会天然占更大权重。",
            "4. 单次实验的生活状态变化 = 终点评估状态分数 − T0 状态分数。负数表示减轻/改善，正数表示加重。R01/R02 方向复现表使用 ±0.05 容差：小于 −0.05 为改善，大于 +0.05 为恶化，中间为稳定。",
            "",
            "这些对应关系是用于解释量表条目的描述性分组，不代表本研究已经验证了新的量表因子结构或潜变量。风险状态单独展示，不并入一般生活状态均值。",
            "",
            "### R01/R02 方向复现汇总",
            "",
        "| 生活状态 | 两次均改善 | 两次均恶化 | 方向不一致/含稳定 |",
        "|---|---:|---:|---:|",
        ]
    )
    for symptom_id, counts in agreement_counts.items():
        lines.append(
            f"| {symptom_labels[symptom_id]} | {counts['两次均改善']}/{len(entities)} | "
            f"{counts['两次均恶化']}/{len(entities)} | "
            f"{counts['方向不一致或含稳定'] + counts['两次均稳定'] + counts['重复不足']}/{len(entities)} |"
        )
    lines.extend(
        [
            "",
            "### 每个实体的 R01/R02 原始变化",
            "",
            "| 实体 | 生活状态 | R01 | R02 | 可重复性结论 |",
            "|---|---|---:|---:|---|",
        ]
    )
    for row in agreement:
        lines.append(
            f"| {row[entity_field]} | {row['symptom_label_zh']} | {_fmt(row.get('R01_change'))} | "
            f"{_fmt(row.get('R02_change'))} | {row['agreement']} |"
        )
    combined = [row for row in life_state["symptom_change"] if row["scale"] == "COMBINED_EQUAL_SCALE_WEIGHT"]
    by_run: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in combined:
        by_run[row["stable_id"]].append(row)
    lines.extend(
        [
            "",
            "### 单次实验需要关注的反向变化",
            "",
            "| Run | 改善最明显 | 恶化/未改善项目 | 风险变化 |",
            "|---|---|---|---:|",
        ]
    )
    for stable_id, rows in sorted(by_run.items(), key=lambda item: item[0]):
        first = rows[0]
        improvement = sorted((row for row in rows if row["symptom_id"] != "self_harm_risk"), key=lambda row: float(row["change"]))[:2]
        worsening = [row for row in rows if row["symptom_id"] != "self_harm_risk" and float(row["change"]) > EPSILON]
        risk = next((row for row in rows if row["symptom_id"] == "self_harm_risk"), None)
        lines.append(
            f"| {first[entity_field]}-{first['outer_run_id']} | "
            + "；".join(f"{row['symptom_label_zh']} {float(row['change']):+.2f}" for row in improvement)
            + " | "
            + ("；".join(f"{row['symptom_label_zh']} {float(row['change']):+.2f}" for row in worsening) or "无明显恶化域")
            + f" | {_fmt(risk.get('change') if risk else None)} |"
        )

    class_counts = Counter(row["alignment_class"] for row in run_alignment)
    empty_runs = len({row["stable_id"] for row in complaint["missing"] if row.get("reason") == "judge_trace_contains_no_sessions"})
    lines.extend(
        [
            "",
            "## 二、主诉推进怎么看好坏",
            "",
            "不能把 stage index 增大直接叫作好。本报告把两个维度放在一起：横轴是主诉 reflection 中 advance 的比例，纵轴是同期八个非风险生活状态的变化。",
            "",
            "### 图中‘症状改善’具体怎么算",
            "",
            "1. 对每个评估时点，先按上一节的方法算出 9 个跨量表生活状态分数。",
            "2. 将‘自伤或自杀相关想法’作为风险项剔除并单列；其余 8 个生活状态等权平均，得到该时点的**一般生活状态分数**：`G(t) = 8 个非风险生活状态分数之和 / 8`。",
            "3. 节点区间图的纵轴是相邻评估节点之差：`ΔG = G(后一个评估节点) − G(前一个评估节点)`。例如 S4→S8 使用 `G(S8) − G(S4)`。",
            "4. run-level 图使用 T0 到 session 20 的变化：8 个非风险生活状态各自的 `session20 − T0` 变化再等权平均；数据完整时等价于 `G(session20) − G(T0)`。",
            "5. 判定规则：`ΔG < −0.05` 才记为‘症状改善’；`−0.05 ≤ ΔG ≤ +0.05` 记为基本稳定；`ΔG > +0.05` 为症状加重。图中稳定和加重都归入‘同期症状未改善’。",
            "6. 风险变化不参与 `G(t)` 计算。只要风险条目变化大于 +0.05，就给散点加黑色边框，即使一般症状改善也必须单独提示风险。",
            "",
            "横轴的主诉推进率按同一区间内 reflection 记录计算：`advance / (advance + hold)`；推进率不低于 0.5 记为‘主诉推进’，否则记为‘主诉停滞’。0.5 是本报告采用的透明描述性阈值，不是临床阈值。",
            "",
            "- **推进且同期症状改善**：过程与结果方向一致，是四类中最容易汇报的正向证据。",
            "- **主诉停滞但症状改善**：结果变好，但主诉图没有同步反映，可能是图机制不敏感。",
            "- **主诉推进但同期症状未改善**：谈得更深不等于状态变好，需要继续观察。",
            "- **主诉停滞且同期症状未改善**：过程和结果均缺少正向变化，是最需要关注的一类。",
            "- 点的黑边表示风险条目同期上升，即使落在正向象限也要单独标记。",
            "",
            f"当前有 {len(run_alignment)}/{len(records)} 个 runs 能形成 run-level 主诉/量表对照；{empty_runs} 个 runs 的 judge trace 为空。",
            "",
            "### 每个评估节点之间的变化",
            "",
            "下表按 `T0→session_4→…→session_20` 分段。‘推进且改善’才是过程与结果同时为正；‘推进但未改善’不能汇报成有效，只能说主诉图向前走了。",
            "",
            "| 评估区间 | 可比 runs | 推进且改善 | 停滞但改善 | 推进但未改善 | 停滞且未改善 | 风险条目上升 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    interval_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in interval_alignment:
        interval_groups[(row["from_timepoint"], row["to_timepoint"])].append(row)
    for (start, end), values in interval_groups.items():
        counts = Counter(row["alignment_class"] for row in values)
        lines.append(
            f"| {label_display(start)}→{label_display(end)} | {len(values)} | "
            f"{counts['推进且同期症状改善']} | {counts['主诉停滞但同期症状改善']} | "
            f"{counts['主诉推进但同期症状未改善']} | {counts['主诉停滞且同期症状未改善']} | "
            f"{sum(bool(row.get('risk_worsened')) for row in values)} |"
        )
    lines.extend(
        [
            "",
            "### 从 T0 到 session 20 的 run-level 总结",
            "",
            "| Run | 主诉 advance 比例 | 一般生活状态变化 | 风险变化 | 四象限结论 |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for row in sorted(run_alignment, key=lambda value: (group_sort_key(str(value[entity_field])), group_sort_key(str(value["outer_run_id"])))):
        lines.append(
            f"| {row[entity_field]}-{row['outer_run_id']} | {_fmt(row['complaint_advance_rate'], 3)} | "
            f"{_fmt(row['general_symptom_change'])} | {_fmt(row.get('risk_change'))} | {row['alignment_class']} |"
        )
    lines.extend(["", "四象限计数：" + "；".join(f"{key} {value}" for key, value in class_counts.items()) + "。"])

    overall_kappa = [row for row in kappa["summary"] if row["stratum_type"] == "overall"]
    entity_kappa = [
        row
        for row in kappa["summary"]
        if row["stratum_type"] == entity_type and row["weights"] == "quadratic"
    ]
    lines.extend(
        [
            "",
            "## 三、Weighted Kappa（仅在本组内重算）",
            "",
            "### 先用一句话理解",
            "",
            "Weighted Kappa 回答的是：**把同一个人、同一个时点、同一道题重复问 10 次，答案能不能稳定地落在相同或相近的等级上。**它不回答症状是否改善。",
            "",
            "- 输入：同一 `snapshot_id × scale × item` 下的 10 次条目答案；每一对 repeat 都比较一次。",
            "- 配对范围：只比较同一个 snapshot 的同一道题，绝不把不同 run、时点或条目错位配对。",
            "- 主结果：quadratic weights；差 1 级处罚较轻，差得越远处罚越重。linear weights 只作敏感性检查。",
            "- 不计算：PHQ-9/BDI-II 总分不做 Kappa；总分的一致性仍应看 ICC。10 次 repeat 也没有被当成 10 个独立实验。",
            "- 读数：κ 越接近 1，重复答案越稳定；接近 0 表示没有明显超过随机一致；负值表示系统性相反。类别分布会影响 κ，不能只套一个阈值判定好坏。",
            "",
            "| Scale | Quadratic κ | 95% outer-run cluster CI | Linear κ |",
            "|---|---:|---:|---:|",
        ]
    )
    for scale in ("PHQ-9", "BDI-II"):
        quadratic = next(row for row in overall_kappa if row["scale"] == scale and row["weights"] == "quadratic")
        linear = next(row for row in overall_kappa if row["scale"] == scale and row["weights"] == "linear")
        lines.append(
            f"| {scale} | {_fmt(quadratic['kappa'], 3)} | [{_fmt(quadratic.get('ci95_lower'), 3)}, "
            f"{_fmt(quadratic.get('ci95_upper'), 3)}] | {_fmt(linear['kappa'], 3)} |"
        )
    lines.extend(
        [
            "",
            "### 实际计算范围和输出",
            "",
            "| Scale | 独立 outer runs | 同一题同一时点对象数 | repeat 两两组合数 | 有效答案配对 | 缺失 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for scale in ("PHQ-9", "BDI-II"):
        quadratic = next(row for row in overall_kappa if row["scale"] == scale and row["weights"] == "quadratic")
        lines.append(
            f"| {scale} | {quadratic['n_outer_run_clusters']} | {quadratic['n_object_targets']} | "
            f"{quadratic['n_repeat_pairs']} | {quadratic['n_valid']} | {quadratic['n_missing']} |"
        )
    quadratic_by_scale = {
        row["scale"]: row for row in overall_kappa if row["weights"] == "quadratic"
    }
    lines.extend(
        [
            "",
            "**本组的直白结论：**重复测量时，BDI-II 条目答案比 PHQ-9 条目答案更稳定；这个结论只适用于本组仿真及其当前类别分布。",
            "",
            "**汇报建议说法：**“在本组独立仿真中，同一状态下 10 次重复量表回答的条目级 quadratic weighted Kappa，"
            f"PHQ-9 为 {float(quadratic_by_scale['PHQ-9']['kappa']):.3f}，BDI-II 为 {float(quadratic_by_scale['BDI-II']['kappa']):.3f}。"
            "BDI-II 的重复回答相对更稳定；Kappa 衡量重复回答稳定性，不代表症状改善幅度。”",
            "",
            "对应明细文件：`weighted_kappa_summary.csv`（组级和分层汇总）、`weighted_kappa_pairwise.csv`（每一对 repeat）、"
            "`weighted_kappa_item.csv`（逐条目），以及 3 张 `weighted_kappa_*.png`。",
        ]
    )
    lines.extend(
        [
            "",
            "### 各实体描述性 Kappa",
            "",
            "每个实体的独立 outer runs 很少，因此不强行给实体级 CI；下面只用于识别哪些人设/条件的重复量表回答更稳定。",
            "",
            "| 实体 | Scale | Quadratic κ | 有效对象 |",
            "|---|---|---:|---:|",
        ]
    )
    for row in entity_kappa:
        lines.append(
            f"| {row['stratum_value']} | {row['scale']} | {_fmt(row.get('kappa'), 3)} | {row.get('n_valid')} |"
        )
    lines.extend(
        [
            "",
            "## 汇报边界",
            "",
            "- 这里描述的是生成式 agent 的仿真内变化，不是真实患者疗效。",
            f"- R01/R02 同方向比跨实体总平均更有解释力；每个实体当前有 {len(repeat_ids)} 次独立实验，{later_repeat_text} 不进入这张两次复现专图。",
            "- 主诉 advance 阈值 0.5 是透明的描述性规则，不是临床验证阈值。",
            "- 风险条目永远单列，不并入一般生活状态好坏判断。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def analyze_subset(
    records: list[ExperimentRecord],
    labels: list[str],
    scales: list[str],
    checkpoints_root: Path,
    out_dir: Path,
    *,
    title: str,
    entity_field: str,
    entity_type: str,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    life_state = build_life_state_analysis(records, labels, scales)
    complaint = extract_complaint_evaluation_nodes(records, labels, checkpoints_root)
    agreement = build_replicate_agreement(life_state["symptom_change"], entity_field)
    interval_alignment, run_alignment = build_complaint_outcome_alignment(life_state, complaint)
    kappa = build_weighted_kappa(records, labels, scales)
    _write_subset_csvs(out_dir, life_state, agreement, complaint, interval_alignment, run_alignment, kappa)
    chart_paths = render_stratified_figures(
        life_state,
        agreement,
        interval_alignment,
        run_alignment,
        kappa,
        labels,
        entity_field,
        entity_type,
        title,
        out_dir,
    )
    chart_paths.extend(render_weighted_kappa_figures(kappa, out_dir, title))
    report_path = write_subset_report(
        out_dir,
        title=title,
        records=records,
        entity_field=entity_field,
        entity_type=entity_type,
        agreement=agreement,
        life_state=life_state,
        complaint=complaint,
        interval_alignment=interval_alignment,
        run_alignment=run_alignment,
        kappa=kappa,
    )
    return {
        "title": title,
        "record_count": len(records),
        "entities": sorted({_record_entity(record, entity_field) for record in records}, key=group_sort_key),
        "chart_count": len(chart_paths),
        "report": report_path.name,
        "agreement": agreement,
        "run_alignment": run_alignment,
        "kappa": kappa,
    }


def write_master_report(
    out_dir: Path, results: list[dict[str, Any]], input_inventory: list[dict[str, Any]]
) -> Path:
    path = out_dir / "README.md"
    origin_counts = Counter(row["data_origin"] for row in input_inventory)
    lines = [
        "# 0802+0808 分组可解释性论文图与汇报分析",
        "",
        f"按每个 run 的 `trial_meta.json` 复核：远程平台 {origin_counts['remote_workspace']} 个 runs，"
        f"本地服务器 {origin_counts['local_server']} 个 runs，来源未知 {origin_counts['unknown']} 个 runs；"
        "逐 run 证据见 [`input_run_inventory.csv`](input_run_inventory.csv)。",
        "",
        "本目录不把部分交叉设计直接池化成一个组间效应。两套分析完全分开：",
        "",
        "1. [`cross_persona_g1/analysis_report.md`](cross_persona_g1/analysis_report.md)：跨人设 G1；",
        "2. [`kbd2_conditions/analysis_report.md`](kbd2_conditions/analysis_report.md)：KBD2 的8个实验条件。",
        "",
        "KBD2-G1 的 runs 按研究问题同时出现在两套分析中：在第一套代表 KBD2 这一人设，在第二套代表 G1 这一条件；它们没有在任何一套组内重复计数。",
        "",
        "生活状态图中每一行都是单次独立实验。R01/R02 专图只判断前两次方向能否复现；R03–R05 保留在其他分析中。",
        "",
        "主诉图用‘主诉 advance 比例 × 同期非风险生活状态变化’判断过程与结果是否一致；风险条目上升单独加黑边警告。",
        "",
        "Weighted Kappa 也在两套子样本中分别重新计算，未使用跨设计池化结果。",
        "",
    ]
    for result in results:
        primary = [
            row
            for row in result["kappa"]["summary"]
            if row["stratum_type"] == "overall" and row["weights"] == "quadratic"
        ]
        agreement_counts = Counter(row["agreement"] for row in result["agreement"])
        entity_count = len(result["entities"])
        symptom_count = len({row["symptom_id"] for row in result["agreement"]})
        fully_reproduced = [
            next(row["symptom_label_zh"] for row in result["agreement"] if row["symptom_id"] == symptom_id)
            for symptom_id in sorted({row["symptom_id"] for row in result["agreement"]})
            if sum(
                row["agreement"] == "两次均改善"
                for row in result["agreement"]
                if row["symptom_id"] == symptom_id
            )
            == entity_count
        ]
        complaint_counts = Counter(row["alignment_class"] for row in result["run_alignment"])
        lines.extend(
            [
                f"## {result['title']}",
                "",
                f"- 独立 runs：{result['record_count']}；实体：{', '.join(result['entities'])}。",
                f"- 生活状态：{entity_count} 个实体 × {symptom_count} 个状态均保留 R01/R02；"
                f"两次均改善 {agreement_counts['两次均改善']} 格，两次均恶化 {agreement_counts['两次均恶化']} 格，"
                f"其余为不一致或稳定。",
                "- 所有实体都在 R01/R02 复现改善的生活状态：" + ("、".join(fully_reproduced) or "无") + "。",
                f"- 主诉/量表 run-level 对照可用：{len(result['run_alignment'])}/{result['record_count']}。",
                "- 主诉四象限：" + ("；".join(f"{key} {value}" for key, value in complaint_counts.items()) or "无可用数据") + "。",
                "- Kappa：" + "；".join(f"{row['scale']} κ={float(row['kappa']):.3f}" for row in primary) + "。",
                f"- 完整解读与逐 run 表：[打开本组报告]({'cross_persona_g1' if '跨人设' in result['title'] else 'kbd2_conditions'}/analysis_report.md)。",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")
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
    parser.add_argument(
        "--subset",
        choices=("all", "cross-persona-g1", "kbd2-conditions"),
        default="all",
        help="Run both separated analyses or resume one subset only.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report_files = find_report_files(args.reports_dir.resolve(), recursive=True)
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
    g1_records = [record for record in records if record.group == "G1"]
    kbd2_records = [record for record in records if record.kbd == "KBD2"]
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    input_inventory = build_input_inventory(records, args.reports_dir.resolve().parent)
    _write_csv(out_dir / "input_run_inventory.csv", input_inventory)
    results: list[dict[str, Any]] = []
    if args.subset in {"all", "cross-persona-g1"}:
        results.append(
            analyze_subset(
                g1_records,
                labels,
                scales,
                args.checkpoints_root.resolve(),
                out_dir / "cross_persona_g1",
                title="跨人设 G1",
                entity_field="persona",
                entity_type="persona",
            )
        )
    if args.subset in {"all", "kbd2-conditions"}:
        results.append(
            analyze_subset(
                kbd2_records,
                labels,
                scales,
                args.checkpoints_root.resolve(),
                out_dir / "kbd2_conditions",
                title="KBD2 不同实验条件",
                entity_field="group",
                entity_type="group",
            )
        )
    if args.subset == "all":
        write_master_report(out_dir, results, input_inventory)
    (out_dir / "analysis_manifest.json").write_text(
        json.dumps(
            [
                {key: value for key, value in result.items() if key not in {"agreement", "run_alignment", "kappa"}}
                for result in results
            ],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {len(results)} separated analysis set(s) to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
