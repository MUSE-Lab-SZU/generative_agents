"""Machine-readable metrics, CSV exports, and human-readable report indexes."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .schema import EXPECTED_AGGREGATION_METHOD_VERSION, ExperimentRecord
from .statistics import (
    baseline_label_for_record,
    delta_ci,
    endpoint_label_for_record,
    score_at,
    trajectory_metrics,
)


def _display_path(path: Path, project_root: Path) -> str:
    try:
        return str(path.relative_to(project_root))
    except ValueError:
        return str(path)


def _number(value: Any, digits: int = 2) -> str:
    return "NA" if value is None else f"{float(value):.{digits}f}"


def _interval(payload: dict[str, Any]) -> str:
    lower, upper = payload.get("lower"), payload.get("upper")
    return "NA" if lower is None or upper is None else f"[{lower:.2f}, {upper:.2f}]"


def build_metrics(
    records: list[ExperimentRecord],
    labels: list[str],
    scales: list[str],
    *,
    batch_kind: str,
    report_mode: str,
    error_bar: str,
    y_axis: str,
    outer_summary: str,
    project_root: Path,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "metadata": {
            "input_schema": EXPECTED_AGGREGATION_METHOD_VERSION,
            "primary_score": "mean_of_complete_reviewed_scale_totals",
            "repeat_unit": "same agent and frozen snapshot; Monte Carlo measurement repeat, not independent patient",
            "outer_repeat_unit": "independent complete simulation experiment",
            "default_delta_ci_method": "welch_difference_t",
            "delta_ci_pairing_rule": "paired_difference_t only when metadata explicitly declares cross-timepoint pairing",
            "delta_ci_uncertainty_scope": "Monte Carlo measurement uncertainty only",
            "endpoint_rule": "POST, else NOW, else last available timepoint",
            "delta_auc_rule": "trapezoidal AUC over ordered observed timepoints with unit spacing",
            "report_mode": report_mode,
            "error_bar": error_bar,
            "y_axis": y_axis,
            "outer_summary": outer_summary,
        },
        "labels": labels,
        "scales": scales,
        "batch_kind": batch_kind,
        "condition_order": [record.stable_id for record in records],
        "records": {},
        "data_files": {},
        "timepoint_scores": {},
        "timepoint_statistics": {},
        "measurement_metrics": {},
        "trajectory_metrics": {},
        "delta_ci": {},
        "final_delta": {},
        "stability_metrics": {},
        "legacy_aliases": {"temporal_sd": "trajectory_volatility"},
    }
    for record in records:
        key = record.stable_id
        payload["records"][key] = {
            "stable_id": key,
            "kbd": record.kbd,
            "condition_name": record.condition_name,
            "group": record.group,
            "source_group": record.source_group,
            "severity": record.severity,
            "repeat_id": record.repeat_id,
            "plot_label": record.plot_label,
            "batch_name": record.batch_name,
            "run_name": record.run_name,
            "expected_repeats": record.expected_repeats,
            "repeats_paired_across_timepoints": record.repeats_paired_across_timepoints,
            "pairing_evidence": record.pairing_evidence,
        }
        payload["data_files"][key] = _display_path(record.path, project_root)
        payload["timepoint_scores"][key] = {}
        payload["timepoint_statistics"][key] = {}
        payload["measurement_metrics"][key] = {}
        payload["trajectory_metrics"][key] = {}
        payload["delta_ci"][key] = {}
        payload["final_delta"][key] = {}
        payload["stability_metrics"][key] = {}
        for scale in scales:
            payload["timepoint_scores"][key][scale] = {}
            payload["timepoint_statistics"][key][scale] = {}
            payload["measurement_metrics"][key][scale] = {}
            for label in labels:
                stats = record.series.get(scale, {}).get(label, {})
                payload["timepoint_scores"][key][scale][label] = stats.get("score")
                payload["timepoint_statistics"][key][scale][label] = {
                    field: stats.get(field)
                    for field in (
                        "score",
                        "score_semantics",
                        "sample_sd",
                        "ci95_lower",
                        "ci95_upper",
                        "ci95_width",
                        "median",
                        "q1",
                        "q3",
                        "iqr",
                        "trimmed_mean_drop_one_each",
                        "min",
                        "max",
                        "n",
                        "expected_repeats",
                        "valid_repeats",
                        "aggregation_status",
                        "values",
                        "severity",
                    )
                }
                payload["measurement_metrics"][key][scale][label] = {
                    field: stats.get(field)
                    for field in (
                        "sample_sd",
                        "ci95_width",
                        "iqr",
                        "modal_confidence",
                        "category_pairwise_flip_rate",
                        "any_category_flip",
                    )
                }
            trajectory = trajectory_metrics(record, labels, scale)
            interval = delta_ci(record, labels, scale)
            payload["trajectory_metrics"][key][scale] = trajectory
            payload["delta_ci"][key][scale] = interval
            payload["final_delta"][key][scale] = trajectory.get("endpoint_change")
            payload["stability_metrics"][key][scale] = {
                "trajectory_volatility": trajectory.get("trajectory_volatility"),
                "temporal_sd": trajectory.get("trajectory_volatility"),
                "temporal_sd_deprecated": True,
                "max_upward_step": trajectory.get("max_upward_step"),
            }
    return payload


def write_item_statistics_csv(out_dir: Path, records: list[ExperimentRecord], labels: list[str], scales: list[str]) -> Path:
    path = out_dir / "item_statistics.csv"
    rows: list[dict[str, Any]] = []
    for record in records:
        for scale in scales:
            for label in labels:
                for stats in (record.series.get(scale, {}).get(label, {}) or {}).get("item_statistics", []):
                    rows.append(
                        {
                            "stable_id": record.stable_id,
                            "kbd": record.kbd,
                            "condition_name": record.condition_name,
                            "group": record.group,
                            "severity": record.severity,
                            "outer_repeat": record.repeat_id,
                            "timepoint": label,
                            "scale": scale,
                            "item_id": stats["item_id"],
                            "n": stats["n"],
                            "mean": stats["mean"],
                            "sample_sd": stats["sample_sd"],
                            "exact_agreement_rate": stats["exact_agreement_rate"],
                            "score_entropy_bits": stats["score_entropy_bits"],
                        }
                    )
    fieldnames = list(rows[0]) if rows else ["stable_id"]
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def write_statistics_detail_csv(out_dir: Path, metrics: dict[str, Any]) -> Path:
    path = out_dir / "statistics_detail.csv"
    rows: list[dict[str, Any]] = []
    for key in metrics["condition_order"]:
        record = metrics["records"][key]
        for scale in metrics["scales"]:
            for label in metrics["labels"]:
                stats = metrics["timepoint_statistics"][key][scale][label]
                if stats.get("score") is None:
                    continue
                measurement = metrics["measurement_metrics"][key][scale][label]
                rows.append(
                    {
                        "stable_id": key,
                        "kbd": record["kbd"],
                        "condition_name": record["condition_name"],
                        "group": record["group"],
                        "severity": record["severity"],
                        "outer_repeat": record["repeat_id"],
                        "timepoint": label,
                        "scale": scale,
                        "primary_score": stats["score"],
                        "sample_sd": measurement["sample_sd"],
                        "ci95_lower": stats["ci95_lower"],
                        "ci95_upper": stats["ci95_upper"],
                        "ci95_width": measurement["ci95_width"],
                        "median": stats["median"],
                        "iqr": measurement["iqr"],
                        "modal_confidence": measurement["modal_confidence"],
                        "category_pairwise_flip_rate": measurement["category_pairwise_flip_rate"],
                        "any_category_flip": measurement["any_category_flip"],
                        "valid_repeats": stats["valid_repeats"],
                        "expected_repeats": stats["expected_repeats"],
                    }
                )
    fieldnames = list(rows[0]) if rows else ["stable_id"]
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def write_statistics_report(out_dir: Path, metrics: dict[str, Any]) -> Path:
    path = out_dir / "statistics_report.md"
    lines = [
        "# KBD 重复评估统计报告",
        "",
        "## 统计口径",
        "",
        "- 主分数：固定 K 次完整复核量表总分的均值。",
        "- 内层 repeat：同一 Agent、同一冻结快照的 Monte Carlo 测量重复，不是独立患者。",
        "- 外层 R01/R02/R03：相互独立的完整仿真实验，默认不汇总。",
        "- 差值 CI：默认 Welch 非配对 t 区间；只有 metadata 明确声明跨时间点配对时才使用 paired t 区间。",
        "- 中位数与去除一个最高/最低值后的均值只作为敏感性统计。",
        "",
        "## 基线—终点与轨迹指标",
        "",
        "| Stable ID | Scale | Baseline | Endpoint | Endpoint change | Difference 95% CI | Method | Best change | Best time | Rebound | Trajectory volatility | Delta AUC |",
        "|---|---|---|---|---:|---:|---|---:|---|---:|---:|---:|",
    ]
    for key in metrics["condition_order"]:
        for scale in metrics["scales"]:
            trajectory = metrics["trajectory_metrics"][key][scale]
            interval = metrics["delta_ci"][key][scale]
            if trajectory.get("endpoint_change") is None:
                continue
            lines.append(
                f"| {key} | {scale} | {trajectory.get('baseline_label')} | {trajectory.get('endpoint_label')} | "
                f"{_number(trajectory.get('endpoint_change'))} | {_interval(interval)} | {interval.get('method')} | "
                f"{_number(trajectory.get('best_change_from_baseline'))} | {trajectory.get('best_timepoint')} | "
                f"{_number(trajectory.get('rebound_from_nadir'))} | {_number(trajectory.get('trajectory_volatility'))} | "
                f"{_number(trajectory.get('delta_auc'))} |"
            )
    lines.extend(
        [
            "",
            "## 输出文件",
            "",
            "- `statistics_detail.csv`：每个时间点的主分数与测量指标。",
            "- `item_statistics.csv`：条目级 mean、sample SD、两两 exact agreement rate、Shannon entropy (bits)。",
            "- `chart_metrics_summary.json`：完整机器可读统计、CI 方法和假设。",
            "",
            "本报告为生成式 Agent 仿真的描述统计，不代表真实患者临床疗效。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_chart_index(out_dir: Path, chart_paths: list[Path], metrics: dict[str, Any]) -> Path:
    path = out_dir / "chart_index.md"
    lines = [
        "# KBD 重复量表实验图表说明",
        "",
        f"- 模式：`{metrics['metadata']['report_mode']}`",
        f"- 误差：`{metrics['metadata']['error_bar']}`",
        f"- y 轴：`{metrics['metadata']['y_axis']}`",
        f"- 终点规则：{metrics['metadata']['endpoint_rule']}",
        "- 差值 CI 只表示 Monte Carlo 测量不确定性。",
        "- 每张图同时提供 PNG（300 DPI）、SVG 和 PDF。",
        "- 明细：[`statistics_report.md`](statistics_report.md)、[`statistics_detail.csv`](statistics_detail.csv)、[`item_statistics.csv`](item_statistics.csv)。",
        "",
        "## 图表",
        "",
    ]
    for chart_path in chart_paths:
        lines.extend([f"### {chart_path.stem.replace('_', ' ')}", "", f"![{chart_path.stem}]({chart_path.name})", ""])
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_batch_reports(
    out_dir: Path,
    records: list[ExperimentRecord],
    labels: list[str],
    scales: list[str],
    chart_paths: list[Path],
    *,
    batch_kind: str,
    report_mode: str,
    error_bar: str,
    y_axis: str,
    outer_summary: str,
    project_root: Path,
) -> dict[str, Any]:
    metrics = build_metrics(
        records,
        labels,
        scales,
        batch_kind=batch_kind,
        report_mode=report_mode,
        error_bar=error_bar,
        y_axis=y_axis,
        outer_summary=outer_summary,
        project_root=project_root,
    )
    metrics_path = out_dir / "chart_metrics_summary.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    detail_path = write_statistics_detail_csv(out_dir, metrics)
    item_path = write_item_statistics_csv(out_dir, records, labels, scales)
    report_path = write_statistics_report(out_dir, metrics)
    index_path = write_chart_index(out_dir, chart_paths, metrics)
    return {
        "batch_kind": batch_kind,
        "out_dir": _display_path(out_dir, project_root),
        "record_keys": [record.stable_id for record in records],
        "chart_count": len(chart_paths),
        "format_count": len(chart_paths) * 3,
        "metrics_path": _display_path(metrics_path, project_root),
        "statistics_report_path": _display_path(report_path, project_root),
        "statistics_csv_path": _display_path(detail_path, project_root),
        "item_statistics_csv_path": _display_path(item_path, project_root),
        "index_path": _display_path(index_path, project_root),
    }
