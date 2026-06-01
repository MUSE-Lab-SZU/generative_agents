#!/usr/bin/env python3
"""对比量表聚合结果，并输出 Markdown 报告。"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, OrderedDict, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev
from typing import Any


# ============================================================
# ↓↓↓ 可调参数：直接修改这里即可（中文注释）↓↓↓
# ============================================================

RESULT_A = "results/experiment_data/sim-init-0523/scales/scale_scores.json"
RESULT_B = "results/experiment_data/sim-test-0523-2/scales/scale_scores.json"

LABEL_A = "初始化存档sim-init-0523"
LABEL_B = "仿真实验后存档sim-test-0523-2"

# 输入来源：
# - "auto": 自动根据 JSON 结构识别（推荐）
# - "extra_scale_eval": runshells/run_extra_scale_eval.py 输出的 scale_scores_*.json
# - "run_one_experiment": runshells/run_one_experiment.py 输出的 scale_scores.json
INPUT_SOURCE = "auto"

# 留空则自动写到 results/experiment_data/reports/
OUTPUT_MD = ""

# ============================================================
# ↑↑↑ 可调参数：直接修改这里即可（中文注释）↑↑↑
# ============================================================


BASE_DIR = Path(__file__).resolve().parent.parent
REPORTS_DIR = BASE_DIR / "results" / "experiment_data" / "reports"
SUPPORTED_INPUT_SOURCES = {"auto", "extra_scale_eval", "run_one_experiment"}

PRIMARY_METRIC_SPECS = {
    "PHQ-9": [("total_score", "总分", 27.0)],
    "BDI-II": [("total_score", "总分", 63.0)],
    "SDS": [
        ("standard_score", "标准分", 100.0),
        ("total_score_raw", "原始分", 80.0),
        ("depression_severity_index", "抑郁指数", 1.0),
    ],
}

EXTRA_METRIC_SPECS = {
    "PHQ-9": [],
    "BDI-II": [],
    "SDS": [
        ("total_score_raw", "原始分", 80.0),
        ("depression_severity_index", "抑郁指数", 1.0),
    ],
}


@dataclass
class RuntimeConfig:
    result_a: Path
    result_b: Path
    label_a: str
    label_b: str
    input_source: str
    output_md: Path


@dataclass
class ScaleSummary:
    scale_name: str
    metric_key: str | None
    metric_label: str
    metric_max: float | None
    values: list[float]
    run_rows: list[dict[str, Any]]
    severity_counts: Counter
    safety_count: int
    function_counts: Counter
    item_means: OrderedDict[str, float]
    extra_metric_means: OrderedDict[str, tuple[str, float | None]]


@dataclass
class NumericStats:
    count: int
    avg: float
    sd: float
    min_value: float
    max_value: float



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two scale_scores_*.json files and write a markdown report")
    parser.add_argument("--a", default=None, help="第一份聚合结果 JSON 路径")
    parser.add_argument("--b", default=None, help="第二份聚合结果 JSON 路径")
    parser.add_argument("--label-a", default=None, help="第一份结果标签")
    parser.add_argument("--label-b", default=None, help="第二份结果标签")
    parser.add_argument("--input-source", default=None, help="输入来源：auto / extra_scale_eval / run_one_experiment")
    parser.add_argument("--output", default=None, help="Markdown 输出路径")
    return parser.parse_args()



def resolve_path(raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path
    return BASE_DIR / path



def slugify_filename(text: str) -> str:
    text = re.sub(r"[\\/:*?\"<>|]+", "_", text.strip())
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_") or "compare"



def auto_output_path(label_a: str, label_b: str, path_a: Path, path_b: Path) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    file_name = (
        f"scale_compare_{slugify_filename(label_a)}_vs_{slugify_filename(label_b)}"
        f"_{slugify_filename(path_a.stem)}.md"
    )
    return REPORTS_DIR / file_name



def resolve_runtime_config(args: argparse.Namespace) -> RuntimeConfig:
    result_a = resolve_path(args.a if args.a is not None else RESULT_A)
    result_b = resolve_path(args.b if args.b is not None else RESULT_B)
    label_a = (args.label_a if args.label_a is not None else LABEL_A).strip()
    label_b = (args.label_b if args.label_b is not None else LABEL_B).strip()
    input_source = (args.input_source if args.input_source is not None else INPUT_SOURCE).strip()
    if not label_a or not label_b:
        raise ValueError("LABEL_A 和 LABEL_B 不能为空。")
    if input_source not in SUPPORTED_INPUT_SOURCES:
        allowed = ", ".join(sorted(SUPPORTED_INPUT_SOURCES))
        raise ValueError(f"INPUT_SOURCE / --input-source 非法：{input_source}。可选值：{allowed}")

    if args.output is not None:
        output_md = resolve_path(args.output)
    elif OUTPUT_MD:
        output_md = resolve_path(OUTPUT_MD)
    else:
        output_md = auto_output_path(label_a, label_b, result_a, result_b)

    return RuntimeConfig(
        result_a=result_a,
        result_b=result_b,
        label_a=label_a,
        label_b=label_b,
        input_source=input_source,
        output_md=output_md,
    )



def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"文件不存在: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)



def detect_input_source(bundle: dict[str, Any]) -> str:
    if isinstance(bundle.get("scales"), dict):
        return "extra_scale_eval"
    if isinstance(bundle.get("post"), dict):
        return "run_one_experiment"
    raise ValueError(
        "无法识别输入来源：既不是 run_extra_scale_eval.py 的 scales 聚合格式，"
        "也不是 run_one_experiment.py 的 post 聚合格式。"
    )



def infer_checkpoint_from_path(path: Path) -> str:
    if path.parent.name == "scales" and path.parent.parent.name:
        return path.parent.parent.name
    return path.stem



def normalize_bundle(bundle: dict[str, Any], path: Path, input_source: str) -> dict[str, Any]:
    source = detect_input_source(bundle) if input_source == "auto" else input_source

    if source == "extra_scale_eval":
        scales = bundle.get("scales")
        if not isinstance(scales, dict):
            raise ValueError(f"{path} 不是 run_extra_scale_eval.py 的聚合格式：缺少 scales 字段。")
        return bundle

    phase_scores = bundle.get("post")
    if not isinstance(phase_scores, dict):
        raise ValueError(f"{path} 不是 run_one_experiment.py 的聚合格式：缺少 post 字段。")

    normalized_scales: dict[str, dict[str, Any]] = {}
    for scale_name, score in phase_scores.items():
        if not isinstance(score, dict):
            continue
        normalized_scales[str(scale_name)] = {
            "repeat": 1,
            "runs": [
                {
                    "repeat": 1,
                    "score": score,
                }
            ],
        }

    if not normalized_scales:
        raise ValueError(f"{path} 的 post 字段中没有可用于对比的量表评分数据。")

    return {
        "phase": "post",
        "suffix": bundle.get("suffix") or "run_one_experiment",
        "checkpoint": bundle.get("checkpoint") or infer_checkpoint_from_path(path),
        "snapshot": bundle.get("snapshot") or "—",
        "agent": bundle.get("agent") or "—",
        "scales": normalized_scales,
    }



def choose_metric_spec(scale_name: str, score: dict[str, Any]) -> tuple[str | None, str, float | None]:
    specs = PRIMARY_METRIC_SPECS.get(scale_name, [])
    for key, label, max_value in specs:
        value = score.get(key)
        if isinstance(value, (int, float)):
            return key, label, max_value

    generic_specs = [
        ("total_score", "总分", None),
        ("standard_score", "标准分", None),
        ("total_score_raw", "原始分", None),
        ("depression_severity_index", "抑郁指数", None),
    ]
    for key, label, max_value in generic_specs:
        value = score.get(key)
        if isinstance(value, (int, float)):
            return key, label, max_value
    return None, "主指标", None



def choose_primary_severity(scale_name: str, score: dict[str, Any]) -> str | None:
    if scale_name == "SDS":
        keys = ["severity_by_standard_score", "severity_by_index", "severity"]
    else:
        keys = ["severity", "severity_by_standard_score", "severity_by_index"]
    for key in keys:
        value = score.get(key)
        if value:
            return str(value)
    return None



def extract_item_scores(score: dict[str, Any]) -> list[tuple[str, float]]:
    for key in ["phq9_scores", "bdi_ii_scores", "sds_scores"]:
        items = score.get(key)
        if not isinstance(items, list):
            continue
        results: list[tuple[str, float]] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            name = item.get("item") or item.get("name")
            value = item.get("score")
            if name is None or not isinstance(value, (int, float)):
                continue
            results.append((str(name), float(value)))
        return results
    return []



def extract_function_level(score: dict[str, Any]) -> str | None:
    value = score.get("functional_impairment")
    if isinstance(value, dict) and value.get("level"):
        return str(value["level"])
    return None



def is_safety_flagged(score: dict[str, Any]) -> bool:
    text = str(score.get("safety_risk") or "").strip()
    if not text:
        return False
    if "未见明确" in text or "无明确" in text:
        return False
    if "需要人工进一步安全评估" in text:
        return True
    if text.startswith("存在"):
        return True
    return False



def to_stats(values: list[float]) -> NumericStats | None:
    clean = [float(v) for v in values if isinstance(v, (int, float))]
    if not clean:
        return None
    return NumericStats(
        count=len(clean),
        avg=mean(clean),
        sd=stdev(clean) if len(clean) > 1 else 0.0,
        min_value=min(clean),
        max_value=max(clean),
    )



def format_number(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "—"
    return f"{value:.{digits}f}"



def format_stats(stats: NumericStats | None, *, digits: int = 2) -> str:
    if stats is None:
        return "—"
    return (
        f"{stats.avg:.{digits}f} ± {stats.sd:.{digits}f} "
        f"(n={stats.count}, {stats.min_value:.{digits}f}-{stats.max_value:.{digits}f})"
    )



def format_counter(counter: Counter) -> str:
    if not counter:
        return "—"
    ordered = sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    return "，".join(f"{label}×{count}" for label, count in ordered)



def dominant_label(counter: Counter) -> str:
    if not counter:
        return "—"
    return sorted(counter.items(), key=lambda item: (-item[1], item[0]))[0][0]



def relative_percentage(value: float | None, max_value: float | None) -> float | None:
    if value is None or max_value in (None, 0):
        return None
    return value / max_value * 100.0



def change_label(delta: float | None, eps: float = 1e-9) -> str:
    if delta is None:
        return "—"
    if delta < -eps:
        return "改善"
    if delta > eps:
        return "加重"
    return "持平"



def relative_change(base_value: float | None, delta: float | None) -> float | None:
    if base_value in (None, 0) or delta is None:
        return None
    return delta / base_value * 100.0



def stability_label(sd_a: float | None, sd_b: float | None, label_a: str, label_b: str) -> str:
    if sd_a is None or sd_b is None:
        return "—"
    if abs(sd_a - sd_b) < 1e-9:
        return "接近"
    return label_a if sd_a < sd_b else label_b



def cohen_d(values_a: list[float], values_b: list[float]) -> float | None:
    if not values_a or not values_b:
        return None
    mean_a = mean(values_a)
    mean_b = mean(values_b)
    if len(values_a) < 2 and len(values_b) < 2:
        return None

    var_a = stdev(values_a) ** 2 if len(values_a) > 1 else 0.0
    var_b = stdev(values_b) ** 2 if len(values_b) > 1 else 0.0
    denom = len(values_a) + len(values_b) - 2
    if denom <= 0:
        return None
    pooled_var = ((len(values_a) - 1) * var_a + (len(values_b) - 1) * var_b) / denom
    if pooled_var <= 0:
        return None
    return (mean_b - mean_a) / (pooled_var ** 0.5)



def summarize_scale(scale_name: str, scale_data: dict[str, Any]) -> ScaleSummary:
    runs = sorted(scale_data.get("runs", []), key=lambda item: item.get("repeat", 0))
    metric_key: str | None = None
    metric_label = "主指标"
    metric_max: float | None = None
    values: list[float] = []
    run_rows: list[dict[str, Any]] = []
    severity_counts: Counter = Counter()
    safety_count = 0
    function_counts: Counter = Counter()
    item_values: defaultdict[str, list[float]] = defaultdict(list)
    item_order: list[str] = []

    extra_specs = EXTRA_METRIC_SPECS.get(scale_name, [])
    extra_metric_values: OrderedDict[str, list[float]] = OrderedDict()
    for key, _, _ in extra_specs:
        extra_metric_values[key] = []

    for run in runs:
        score = run.get("score", {}) if isinstance(run.get("score"), dict) else {}
        repeat = run.get("repeat")

        local_metric_key, local_metric_label, local_metric_max = choose_metric_spec(scale_name, score)
        if metric_key is None:
            metric_key = local_metric_key
            metric_label = local_metric_label
            metric_max = local_metric_max

        metric_value = score.get(metric_key) if metric_key else None
        if isinstance(metric_value, (int, float)):
            values.append(float(metric_value))
            metric_value = float(metric_value)
        else:
            metric_value = None

        severity = choose_primary_severity(scale_name, score)
        if severity:
            severity_counts[severity] += 1

        function_level = extract_function_level(score)
        if function_level:
            function_counts[function_level] += 1

        flagged = is_safety_flagged(score)
        if flagged:
            safety_count += 1

        for item_name, item_score in extract_item_scores(score):
            if item_name not in item_values:
                item_order.append(item_name)
            item_values[item_name].append(item_score)

        for key, _, _ in extra_specs:
            value = score.get(key)
            if isinstance(value, (int, float)):
                extra_metric_values[key].append(float(value))

        run_rows.append(
            {
                "repeat": repeat,
                "score": metric_value,
                "severity": severity,
                "safety": score.get("safety_risk"),
                "flagged": flagged,
                "function_level": function_level,
            }
        )

    item_means: OrderedDict[str, float] = OrderedDict()
    for item_name in item_order:
        item_means[item_name] = mean(item_values[item_name])

    extra_metric_means: OrderedDict[str, tuple[str, float | None]] = OrderedDict()
    for key, label, _ in extra_specs:
        values_for_key = extra_metric_values.get(key, [])
        extra_metric_means[key] = (label, mean(values_for_key) if values_for_key else None)

    return ScaleSummary(
        scale_name=scale_name,
        metric_key=metric_key,
        metric_label=metric_label,
        metric_max=metric_max,
        values=values,
        run_rows=run_rows,
        severity_counts=severity_counts,
        safety_count=safety_count,
        function_counts=function_counts,
        item_means=item_means,
        extra_metric_means=extra_metric_means,
    )



def build_scale_map(bundle: dict[str, Any]) -> dict[str, ScaleSummary]:
    scales = bundle.get("scales", {})
    result: dict[str, ScaleSummary] = {}
    if not isinstance(scales, dict):
        return result
    for scale_name, scale_data in scales.items():
        if isinstance(scale_data, dict):
            result[str(scale_name)] = summarize_scale(str(scale_name), scale_data)
    return result



def ordered_scale_names(scale_map_a: dict[str, ScaleSummary], scale_map_b: dict[str, ScaleSummary]) -> list[str]:
    names = list(scale_map_a.keys())
    for name in scale_map_b:
        if name not in names:
            names.append(name)
    return names



def overall_normalized_burden(scale_names: list[str], scale_map: dict[str, ScaleSummary]) -> float | None:
    values: list[float] = []
    for scale_name in scale_names:
        summary = scale_map.get(scale_name)
        if summary is None:
            continue
        stats = to_stats(summary.values)
        percent = relative_percentage(stats.avg if stats else None, summary.metric_max)
        if percent is not None:
            values.append(percent)
    return mean(values) if values else None



def build_core_findings(scale_names: list[str], scale_map_a: dict[str, ScaleSummary], scale_map_b: dict[str, ScaleSummary], cfg: RuntimeConfig) -> list[str]:
    findings: list[str] = []
    better_count = 0
    worse_count = 0
    flat_count = 0
    biggest_change: tuple[str, float] | None = None

    for scale_name in scale_names:
        summary_a = scale_map_a.get(scale_name)
        summary_b = scale_map_b.get(scale_name)
        if summary_a is None or summary_b is None:
            continue
        stats_a = to_stats(summary_a.values)
        stats_b = to_stats(summary_b.values)
        if stats_a is None or stats_b is None:
            continue
        delta = stats_b.avg - stats_a.avg
        label = change_label(delta)
        if label == "改善":
            better_count += 1
        elif label == "加重":
            worse_count += 1
        else:
            flat_count += 1
        if biggest_change is None or abs(delta) > abs(biggest_change[1]):
            biggest_change = (scale_name, delta)

    findings.append(
        f"以各量表主指标均值看，{cfg.label_b} 相对 {cfg.label_a}：改善 {better_count} 个量表，加重 {worse_count} 个量表，持平 {flat_count} 个量表。"
    )

    burden_a = overall_normalized_burden(scale_names, scale_map_a)
    burden_b = overall_normalized_burden(scale_names, scale_map_b)
    if burden_a is not None and burden_b is not None:
        delta_burden = burden_b - burden_a
        findings.append(
            f"按“均值 ÷ 量表满分”的归一化症状负担估算，{cfg.label_a} 为 {burden_a:.2f}%，{cfg.label_b} 为 {burden_b:.2f}%，变化 {delta_burden:+.2f} 个百分点。"
        )

    if biggest_change is not None:
        scale_name, delta = biggest_change
        findings.append(
            f"绝对变化最大的量表是 {scale_name}，{cfg.label_b} 相比 {cfg.label_a} 的主指标变化为 {delta:+.2f}。"
        )

    total_risk_a = sum(summary.safety_count for summary in scale_map_a.values())
    total_runs_a = sum(len(summary.run_rows) for summary in scale_map_a.values())
    total_risk_b = sum(summary.safety_count for summary in scale_map_b.values())
    total_runs_b = sum(len(summary.run_rows) for summary in scale_map_b.values())
    findings.append(
        f"安全风险标记次数：{cfg.label_a} 为 {total_risk_a}/{total_runs_a}，{cfg.label_b} 为 {total_risk_b}/{total_runs_b}。"
    )
    return findings



def relative_path_for_md(path: Path) -> str:
    try:
        return str(path.relative_to(BASE_DIR))
    except ValueError:
        return str(path)



def render_report(bundle_a: dict[str, Any], bundle_b: dict[str, Any], cfg: RuntimeConfig) -> str:
    scale_map_a = build_scale_map(bundle_a)
    scale_map_b = build_scale_map(bundle_b)
    scale_names = ordered_scale_names(scale_map_a, scale_map_b)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines: list[str] = []
    lines.append("# 量表评估结果对比报告")
    lines.append("")
    lines.append(f"生成时间：{now}")
    lines.append("")
    lines.append("## 1. 对比对象")
    lines.append("")
    lines.append(f"> 所有量表默认按“分数越低越轻”解释；下文的 Δ 均指 **{cfg.label_b} - {cfg.label_a}**。")
    lines.append("")
    lines.append(f"- {cfg.label_a}: [{relative_path_for_md(cfg.result_a)}]({relative_path_for_md(cfg.result_a)})")
    lines.append(f"- {cfg.label_b}: [{relative_path_for_md(cfg.result_b)}]({relative_path_for_md(cfg.result_b)})")
    lines.append("")
    lines.append(f"| 项目 | {cfg.label_a} | {cfg.label_b} |")
    lines.append("|---|---|---|")
    lines.append(f"| checkpoint | {bundle_a.get('checkpoint', '—')} | {bundle_b.get('checkpoint', '—')} |")
    lines.append(f"| snapshot | {bundle_a.get('snapshot', '—')} | {bundle_b.get('snapshot', '—')} |")
    lines.append(f"| phase | {bundle_a.get('phase', '—')} | {bundle_b.get('phase', '—')} |")
    lines.append(f"| suffix | {bundle_a.get('suffix', '—')} | {bundle_b.get('suffix', '—')} |")
    lines.append(f"| agent | {bundle_a.get('agent', '—')} | {bundle_b.get('agent', '—')} |")
    lines.append(f"| 量表数 | {len(scale_map_a)} | {len(scale_map_b)} |")
    lines.append("")

    lines.append("## 2. 核心结论")
    lines.append("")
    for finding in build_core_findings(scale_names, scale_map_a, scale_map_b, cfg):
        lines.append(f"- {finding}")
    lines.append("")

    lines.append("## 3. 各量表汇总")
    lines.append("")
    lines.append(
        f"| 量表 | 主指标 | {cfg.label_a} | {cfg.label_b} | Δ | 相对变化 | 结论 | Cohen's d | 严重程度主峰 | 风险标记 |"
    )
    lines.append("|---|---|---|---|---:|---:|---|---:|---|---|")
    for scale_name in scale_names:
        summary_a = scale_map_a.get(scale_name)
        summary_b = scale_map_b.get(scale_name)
        stats_a = to_stats(summary_a.values) if summary_a else None
        stats_b = to_stats(summary_b.values) if summary_b else None
        metric_label = summary_a.metric_label if summary_a else (summary_b.metric_label if summary_b else "主指标")

        avg_a = stats_a.avg if stats_a else None
        avg_b = stats_b.avg if stats_b else None
        delta = (avg_b - avg_a) if (avg_a is not None and avg_b is not None) else None
        pct = relative_change(avg_a, delta)
        d_value = cohen_d(summary_a.values, summary_b.values) if summary_a and summary_b else None

        norm_a = relative_percentage(avg_a, summary_a.metric_max if summary_a else None)
        norm_b = relative_percentage(avg_b, summary_b.metric_max if summary_b else None)
        stats_a_text = format_stats(stats_a, digits=2)
        stats_b_text = format_stats(stats_b, digits=2)
        if norm_a is not None:
            stats_a_text += f"<br>≈满分 {norm_a:.1f}%"
        if norm_b is not None:
            stats_b_text += f"<br>≈满分 {norm_b:.1f}%"

        severity_text = (
            f"{cfg.label_a}: {dominant_label(summary_a.severity_counts) if summary_a else '—'}"
            f"<br>{cfg.label_b}: {dominant_label(summary_b.severity_counts) if summary_b else '—'}"
        )
        risk_text = (
            f"{cfg.label_a}: {summary_a.safety_count}/{len(summary_a.run_rows)}"
            if summary_a else f"{cfg.label_a}: —"
        )
        risk_text += "<br>"
        risk_text += (
            f"{cfg.label_b}: {summary_b.safety_count}/{len(summary_b.run_rows)}"
            if summary_b else f"{cfg.label_b}: —"
        )

        lines.append(
            f"| {scale_name} | {metric_label} | {stats_a_text} | {stats_b_text} | "
            f"{format_number(delta, 2)} | {format_number(pct, 2)}% | {change_label(delta)} | "
            f"{format_number(d_value, 2)} | {severity_text} | {risk_text} |"
        )
    lines.append("")

    lines.append("## 4. 严重程度分布与稳定性")
    lines.append("")
    lines.append(
        f"| 量表 | {cfg.label_a} 严重程度分布 | {cfg.label_b} 严重程度分布 | {cfg.label_a} SD | {cfg.label_b} SD | 更稳定的一侧 |"
    )
    lines.append("|---|---|---|---:|---:|---|")
    for scale_name in scale_names:
        summary_a = scale_map_a.get(scale_name)
        summary_b = scale_map_b.get(scale_name)
        stats_a = to_stats(summary_a.values) if summary_a else None
        stats_b = to_stats(summary_b.values) if summary_b else None
        sd_a = stats_a.sd if stats_a else None
        sd_b = stats_b.sd if stats_b else None
        lines.append(
            f"| {scale_name} | {format_counter(summary_a.severity_counts) if summary_a else '—'} | "
            f"{format_counter(summary_b.severity_counts) if summary_b else '—'} | "
            f"{format_number(sd_a, 2)} | {format_number(sd_b, 2)} | "
            f"{stability_label(sd_a, sd_b, cfg.label_a, cfg.label_b)} |"
        )
    lines.append("")

    lines.append("## 5. 分量表明细")
    lines.append("")
    for scale_name in scale_names:
        summary_a = scale_map_a.get(scale_name)
        summary_b = scale_map_b.get(scale_name)
        lines.append(f"### 5.{scale_names.index(scale_name) + 1} {scale_name}")
        lines.append("")

        if summary_a and summary_b:
            stats_a = to_stats(summary_a.values)
            stats_b = to_stats(summary_b.values)
            delta = (stats_b.avg - stats_a.avg) if stats_a and stats_b else None
            lines.append(f"- 主指标：{summary_a.metric_label}")
            lines.append(f"- {cfg.label_a}：{format_stats(stats_a, digits=2)}")
            lines.append(f"- {cfg.label_b}：{format_stats(stats_b, digits=2)}")
            lines.append(f"- 平均变化：{format_number(delta, 2)}（{change_label(delta)}）")
            lines.append(f"- {cfg.label_a} 风险标记：{summary_a.safety_count}/{len(summary_a.run_rows)}")
            lines.append(f"- {cfg.label_b} 风险标记：{summary_b.safety_count}/{len(summary_b.run_rows)}")
            if summary_a.function_counts or summary_b.function_counts:
                lines.append(f"- {cfg.label_a} 功能损害分布：{format_counter(summary_a.function_counts)}")
                lines.append(f"- {cfg.label_b} 功能损害分布：{format_counter(summary_b.function_counts)}")
        lines.append("")

        lines.append("#### 重复次对比")
        lines.append("")
        lines.append(
            f"| repeat | {cfg.label_a} 分数 | {cfg.label_a} 严重程度 | {cfg.label_b} 分数 | {cfg.label_b} 严重程度 | Δ | 风险变化 |"
        )
        lines.append("|---:|---:|---|---:|---|---:|---|")

        rows_a = {row.get('repeat'): row for row in summary_a.run_rows} if summary_a else {}
        rows_b = {row.get('repeat'): row for row in summary_b.run_rows} if summary_b else {}
        repeat_ids = sorted({*rows_a.keys(), *rows_b.keys()}, key=lambda value: (value is None, value))
        for repeat_id in repeat_ids:
            row_a = rows_a.get(repeat_id, {})
            row_b = rows_b.get(repeat_id, {})
            score_a = row_a.get("score")
            score_b = row_b.get("score")
            delta = (score_b - score_a) if isinstance(score_a, (int, float)) and isinstance(score_b, (int, float)) else None
            risk_change = f"{('是' if row_a.get('flagged') else '否')} → {('是' if row_b.get('flagged') else '否')}"
            lines.append(
                f"| {repeat_id if repeat_id is not None else '—'} | {format_number(score_a, 2)} | {row_a.get('severity') or '—'} | "
                f"{format_number(score_b, 2)} | {row_b.get('severity') or '—'} | {format_number(delta, 2)} | {risk_change} |"
            )
        lines.append("")

        extra_metric_rows: list[tuple[str, str]] = []
        if summary_a:
            for _, (label, value_a) in summary_a.extra_metric_means.items():
                value_b = summary_b.extra_metric_means.get(_, (label, None))[1] if summary_b else None
                delta = (value_b - value_a) if value_a is not None and value_b is not None else None
                extra_metric_rows.append((label, f"| {label} | {format_number(value_a, 3)} | {format_number(value_b, 3)} | {format_number(delta, 3)} |"))
        if extra_metric_rows:
            lines.append("#### 额外数值字段")
            lines.append("")
            lines.append(f"| 字段 | {cfg.label_a} | {cfg.label_b} | Δ |")
            lines.append("|---|---:|---:|---:|")
            for _, row_text in extra_metric_rows:
                lines.append(row_text)
            lines.append("")

        lines.append("#### 条目均值差异")
        lines.append("")
        lines.append(f"| 条目 | {cfg.label_a} 均值 | {cfg.label_b} 均值 | Δ |")
        lines.append("|---|---:|---:|---:|")

        item_names = list(summary_a.item_means.keys()) if summary_a else []
        if summary_b:
            for item_name in summary_b.item_means.keys():
                if item_name not in item_names:
                    item_names.append(item_name)

        for item_name in item_names:
            value_a = summary_a.item_means.get(item_name) if summary_a else None
            value_b = summary_b.item_means.get(item_name) if summary_b else None
            delta = (value_b - value_a) if value_a is not None and value_b is not None else None
            lines.append(
                f"| {item_name} | {format_number(value_a, 3)} | {format_number(value_b, 3)} | {format_number(delta, 3)} |"
            )
        lines.append("")

    return "\n".join(lines) + "\n"



def main() -> None:
    args = parse_args()
    cfg = resolve_runtime_config(args)

    bundle_a = normalize_bundle(load_json(cfg.result_a), cfg.result_a, cfg.input_source)
    bundle_b = normalize_bundle(load_json(cfg.result_b), cfg.result_b, cfg.input_source)
    report = render_report(bundle_a, bundle_b, cfg)

    cfg.output_md.parent.mkdir(parents=True, exist_ok=True)
    with cfg.output_md.open("w", encoding="utf-8") as f:
        f.write(report)

    print("[DONE] 对比报告已生成")
    print(f"- A: {cfg.result_a}")
    print(f"- B: {cfg.result_b}")
    print(f"- 输出: {cfg.output_md}")


if __name__ == "__main__":
    main()
