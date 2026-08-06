"""Interpret ordinal scale-item changes as observable simulated life-state changes."""

from __future__ import annotations

from collections import defaultdict
from statistics import mean, median
from typing import Any

from .schema import ExperimentRecord, group_sort_key, label_sort_key, scale_sort_key
from .statistics import baseline_label_for_record, endpoint_label_for_record, mean_ci95


ITEM_LABELS_ZH = {
    "PHQ-9": {
        1: "兴趣或乐趣",
        2: "低落、沮丧或绝望",
        3: "睡眠",
        4: "疲倦或精力不足",
        5: "食欲",
        6: "自我评价、失败或让人失望",
        7: "注意力",
        8: "动作迟缓或烦躁不安",
        9: "自伤或不想活的想法",
    },
    "BDI-II": {
        1: "悲伤",
        2: "对未来失去信心",
        3: "失败感",
        4: "快乐减少",
        5: "内疚",
        6: "惩罚感",
        7: "对自己失望或厌恶",
        8: "自责或自我批评",
        9: "自杀想法",
        10: "哭泣",
        11: "烦躁不安",
        12: "对人或活动失去兴趣",
        13: "做决定困难",
        14: "无价值感",
        15: "精力不足",
        16: "睡眠变化",
        17: "易怒",
        18: "食欲变化",
        19: "注意力下降",
        20: "疲劳乏力",
        21: "性兴趣下降",
    },
}

ITEM_LABELS_EN = {
    "PHQ-9": {
        1: "Interest/pleasure",
        2: "Low mood/hopelessness",
        3: "Sleep",
        4: "Energy/fatigue",
        5: "Appetite",
        6: "Self-worth/failure",
        7: "Attention",
        8: "Psychomotor/agitation",
        9: "Self-harm/death thoughts",
    },
    "BDI-II": {
        1: "Sadness",
        2: "Pessimism",
        3: "Past failure",
        4: "Loss of pleasure",
        5: "Guilty feelings",
        6: "Punishment feelings",
        7: "Self-dislike",
        8: "Self-criticalness",
        9: "Suicidal thoughts",
        10: "Crying",
        11: "Agitation",
        12: "Loss of interest",
        13: "Indecisiveness",
        14: "Worthlessness",
        15: "Loss of energy",
        16: "Sleep changes",
        17: "Irritability",
        18: "Appetite changes",
        19: "Concentration",
        20: "Tiredness/fatigue",
        21: "Loss of sexual interest",
    },
}

# Multi-item BDI concepts are averaged within scale. PHQ and BDI concept scores
# are then given equal weight so BDI concepts with more items do not dominate.
SYMPTOM_MAP = {
    "interest_pleasure": {"zh": "兴趣、乐趣和参与", "en": "Interest/pleasure", "PHQ-9": [1], "BDI-II": [4, 12]},
    "low_mood": {"zh": "低落与悲伤", "en": "Low mood", "PHQ-9": [2], "BDI-II": [1]},
    "sleep": {"zh": "睡眠", "en": "Sleep", "PHQ-9": [3], "BDI-II": [16]},
    "energy_fatigue": {"zh": "精力与疲劳", "en": "Energy/fatigue", "PHQ-9": [4], "BDI-II": [15, 20]},
    "appetite": {"zh": "食欲", "en": "Appetite", "PHQ-9": [5], "BDI-II": [18]},
    "self_worth_guilt": {
        "zh": "自我价值、内疚与失败感",
        "en": "Self-worth/guilt",
        "PHQ-9": [6],
        "BDI-II": [3, 5, 7, 8, 14],
    },
    "attention_decisions": {"zh": "注意力与决策", "en": "Attention/decisions", "PHQ-9": [7], "BDI-II": [13, 19]},
    "psychomotor_agitation": {
        "zh": "行动迟缓或烦躁不安",
        "en": "Psychomotor/agitation",
        "PHQ-9": [8],
        "BDI-II": [11],
    },
    "self_harm_risk": {"zh": "自伤或自杀相关想法", "en": "Self-harm risk", "PHQ-9": [9], "BDI-II": [9]},
}


def _item_means(stats: dict[str, Any]) -> dict[int, float]:
    values_by_repeat = stats.get("item_values_by_repeat") or {}
    by_item: dict[int, list[float]] = defaultdict(list)
    for values in values_by_repeat.values():
        for item_id, value in enumerate(values, start=1):
            by_item[item_id].append(float(value))
    return {item_id: mean(values) for item_id, values in by_item.items() if values}


def build_item_trajectory_rows(
    records: list[ExperimentRecord], labels: list[str], scales: list[str]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        for scale in sorted(scales, key=scale_sort_key):
            baseline = baseline_label_for_record(record, labels, scale)
            endpoint = endpoint_label_for_record(record, labels, scale)
            for label in sorted(labels, key=lambda value: label_sort_key(value, labels)):
                item_means = _item_means(record.series.get(scale, {}).get(label) or {})
                for item_id, score in sorted(item_means.items()):
                    rows.append(
                        {
                            "stable_id": record.stable_id,
                            "outer_run_id": record.repeat_id,
                            "persona": record.kbd,
                            "group": record.group,
                            "severity": record.severity,
                            "scale": scale,
                            "item_id": item_id,
                            "item_label_zh": ITEM_LABELS_ZH.get(scale, {}).get(item_id, f"条目 {item_id}"),
                            "item_label_en": ITEM_LABELS_EN.get(scale, {}).get(item_id, f"Item {item_id}"),
                            "timepoint": label,
                            "item_repeat_mean": score,
                            "is_baseline": label == baseline,
                            "is_endpoint": label == endpoint,
                        }
                    )
    return rows


def build_item_change_rows(
    records: list[ExperimentRecord], labels: list[str], scales: list[str]
) -> list[dict[str, Any]]:
    trajectories = build_item_trajectory_rows(records, labels, scales)
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in trajectories:
        grouped[(row["stable_id"], row["scale"], int(row["item_id"]))].append(row)
    changes: list[dict[str, Any]] = []
    for rows in grouped.values():
        baseline = next((row for row in rows if row["is_baseline"]), None)
        endpoint = next((row for row in rows if row["is_endpoint"]), None)
        if baseline is None or endpoint is None:
            continue
        change = float(endpoint["item_repeat_mean"]) - float(baseline["item_repeat_mean"])
        changes.append(
            {
                key: baseline[key]
                for key in (
                    "stable_id",
                    "outer_run_id",
                    "persona",
                    "group",
                    "severity",
                    "scale",
                    "item_id",
                    "item_label_zh",
                    "item_label_en",
                )
            }
            | {
                "baseline_timepoint": baseline["timepoint"],
                "endpoint_timepoint": endpoint["timepoint"],
                "baseline_score": baseline["item_repeat_mean"],
                "endpoint_score": endpoint["item_repeat_mean"],
                "change": change,
                "direction": "改善" if change < 0 else "恶化" if change > 0 else "不变",
            }
        )
    return changes


def build_symptom_trajectory_rows(item_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_snapshot: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in item_rows:
        by_snapshot[(row["stable_id"], row["timepoint"])].append(row)
    output: list[dict[str, Any]] = []
    for (_stable_id, _timepoint), rows in by_snapshot.items():
        first = rows[0]
        for symptom_id, mapping in SYMPTOM_MAP.items():
            scale_scores: dict[str, float] = {}
            for scale in ("PHQ-9", "BDI-II"):
                item_ids = set(mapping.get(scale, []))
                values = [
                    float(row["item_repeat_mean"])
                    for row in rows
                    if row["scale"] == scale and int(row["item_id"]) in item_ids
                ]
                if values:
                    scale_scores[scale] = mean(values)
            for scale, score in scale_scores.items():
                output.append(
                    {
                        key: first[key]
                        for key in ("stable_id", "outer_run_id", "persona", "group", "severity", "timepoint", "is_baseline", "is_endpoint")
                    }
                    | {
                        "symptom_id": symptom_id,
                        "symptom_label_zh": mapping["zh"],
                        "symptom_label_en": mapping["en"],
                        "scale": scale,
                        "symptom_score": score,
                    }
                )
            if scale_scores:
                output.append(
                    {
                        key: first[key]
                        for key in ("stable_id", "outer_run_id", "persona", "group", "severity", "timepoint", "is_baseline", "is_endpoint")
                    }
                    | {
                        "symptom_id": symptom_id,
                        "symptom_label_zh": mapping["zh"],
                        "symptom_label_en": mapping["en"],
                        "scale": "COMBINED_EQUAL_SCALE_WEIGHT",
                        "symptom_score": mean(scale_scores.values()),
                    }
                )
    return output


def build_symptom_change_rows(symptom_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in symptom_rows:
        grouped[(row["stable_id"], row["symptom_id"], row["scale"])].append(row)
    changes: list[dict[str, Any]] = []
    for rows in grouped.values():
        baseline = next((row for row in rows if row["is_baseline"]), None)
        endpoint = next((row for row in rows if row["is_endpoint"]), None)
        if baseline is None or endpoint is None:
            continue
        change = float(endpoint["symptom_score"]) - float(baseline["symptom_score"])
        changes.append(
            {
                key: baseline[key]
                for key in (
                    "stable_id",
                    "outer_run_id",
                    "persona",
                    "group",
                    "severity",
                    "symptom_id",
                    "symptom_label_zh",
                    "symptom_label_en",
                    "scale",
                )
            }
            | {
                "baseline_timepoint": baseline["timepoint"],
                "endpoint_timepoint": endpoint["timepoint"],
                "baseline_score": baseline["symptom_score"],
                "endpoint_score": endpoint["symptom_score"],
                "change": change,
                "direction": "改善" if change < 0 else "恶化" if change > 0 else "不变",
            }
        )
    return changes


def summarize_changes(
    item_changes: list[dict[str, Any]], symptom_changes: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    source_rows = [
        row
        | {
            "metric_level": "item",
            "metric_id": f"{row['scale']}_item_{row['item_id']}",
            "metric_label_zh": row["item_label_zh"],
            "metric_label_en": row["item_label_en"],
        }
        for row in item_changes
    ] + [
        row
        | {
            "metric_level": "symptom",
            "metric_id": row["symptom_id"],
            "metric_label_zh": row["symptom_label_zh"],
            "metric_label_en": row["symptom_label_en"],
        }
        for row in symptom_changes
    ]
    output: list[dict[str, Any]] = []
    for stratum_type, field in (("overall", None), ("persona", "persona"), ("group", "group")):
        values = ["ALL"] if field is None else sorted({str(row[field]) for row in source_rows}, key=group_sort_key)
        for stratum_value in values:
            selected = source_rows if field is None else [row for row in source_rows if str(row[field]) == stratum_value]
            grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
            for row in selected:
                grouped[(row["metric_level"], row["metric_id"], row["scale"])].append(row)
            for (metric_level, metric_id, scale), rows in grouped.items():
                changes = [float(row["change"]) for row in rows]
                interval = mean_ci95(changes)
                improved = sum(value < 0 for value in changes)
                stable = sum(value == 0 for value in changes)
                worsened = sum(value > 0 for value in changes)
                output.append(
                    {
                        "stratum_type": stratum_type,
                        "stratum_value": stratum_value,
                        "metric_level": metric_level,
                        "metric_id": metric_id,
                        "metric_label_zh": rows[0]["metric_label_zh"],
                        "metric_label_en": rows[0]["metric_label_en"],
                        "scale": scale,
                        "n_outer_runs": len(changes),
                        "mean_change": mean(changes),
                        "median_change": median(changes),
                        "ci95_lower": interval["lower"],
                        "ci95_upper": interval["upper"],
                        "improved_n": improved,
                        "stable_n": stable,
                        "worsened_n": worsened,
                        "improved_rate": improved / len(changes),
                        "stable_rate": stable / len(changes),
                        "worsened_rate": worsened / len(changes),
                        "interpretation": "负数=症状减轻/生活状态改善；正数=症状加重；0=不变",
                    }
                )
    return output


def build_life_state_analysis(
    records: list[ExperimentRecord], labels: list[str], scales: list[str]
) -> dict[str, list[dict[str, Any]]]:
    item_trajectory = build_item_trajectory_rows(records, labels, scales)
    item_change = build_item_change_rows(records, labels, scales)
    symptom_trajectory = build_symptom_trajectory_rows(item_trajectory)
    symptom_change = build_symptom_change_rows(symptom_trajectory)
    return {
        "item_trajectory": item_trajectory,
        "item_change": item_change,
        "symptom_trajectory": symptom_trajectory,
        "symptom_change": symptom_change,
        "summary": summarize_changes(item_change, symptom_change),
    }

