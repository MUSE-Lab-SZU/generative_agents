"""Machine-readable metrics, CSV exports, and human-readable report indexes."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .schema import (
    EXPECTED_AGGREGATION_METHOD_VERSION,
    EXPERIMENT_EVAL_METRICS_SCHEMA_VERSION,
    ExperimentRecord,
)
from .statistics import (
    baseline_adjusted_endpoint_contrasts,
    cross_scale_concurrent_validity,
    cross_scale_convergence,
    delta_ci,
    group_endpoint_contrasts,
    group_time_contrasts,
    measurement_error_summary,
    measurement_icc,
    safety_proxy_metrics,
    trajectory_metrics,
)
from .weighted_kappa import build_weighted_kappa, item_repeat_long_rows


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
    process_rows: list[dict[str, Any]],
    stage_rows: list[dict[str, Any]],
    dataset_label: str,
    project_root: Path,
) -> dict[str, Any]:
    unique_outer_runs = {record.repeat_id for record in records}
    unique_personas = {record.kbd for record in records}
    unique_snapshots = {
        (record.stable_id, scale, label)
        for record in records
        for scale in scales
        for label in labels
        if (record.series.get(scale, {}).get(label) or {}).get("score") is not None
    }
    weighted_kappa = build_weighted_kappa(records, labels, scales)
    payload: dict[str, Any] = {
        "metadata": {
            "output_schema_version": EXPERIMENT_EVAL_METRICS_SCHEMA_VERSION,
            "dataset_label": dataset_label,
            "input_schema": EXPECTED_AGGREGATION_METHOD_VERSION,
            "primary_score": "mean_of_complete_reviewed_scale_totals",
            "repeat_unit": "same agent and frozen snapshot; Monte Carlo measurement repeat, not independent patient",
            "outer_repeat_unit": "independent complete simulation experiment",
            "default_delta_ci_method": "welch_difference_t",
            "delta_ci_pairing_rule": "paired_difference_t only when metadata explicitly declares cross-timepoint pairing",
            "delta_ci_uncertainty_scope": "Monte Carlo measurement uncertainty only",
            "endpoint_rule": "POST, else NOW, else last available observed timepoint; endpoint is not inferred as follow-up",
            "delta_auc_rule": "trapezoidal AUC over sim_time days when available; otherwise ordered observed timepoints",
            "true_delayed_followup": False,
            "followup_note": (
                "No no-intervention delayed follow-up is present. Meetings after CBT prompt completion, when present, "
                "use a fixed follow-up consultation prompt and remain forced consultations."
            ),
            "n_persona": len(unique_personas),
            "n_outer_run_ids": len(unique_outer_runs),
            "n_outer_run_records": len(records),
            "n_snapshot_scale_targets": len(unique_snapshots),
            "k_measurement_values": sorted({record.expected_repeats for record in records}),
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
        "measurement_reliability": {},
        "measurement_error": {},
        "weighted_kappa": {key: value for key, value in weighted_kappa.items() if key != "item_long"},
        "cross_scale_concurrent_validity": cross_scale_concurrent_validity(records, labels),
        "cross_scale_convergence": cross_scale_convergence(records, labels),
        "group_endpoint_contrasts": group_endpoint_contrasts(records, labels, scales),
        "baseline_adjusted_endpoint_contrasts": baseline_adjusted_endpoint_contrasts(records, labels, scales),
        "group_time_contrasts": group_time_contrasts(records, labels, scales),
        "safety_proxy_metrics": {},
        "process_metrics": process_rows,
        "stage_metrics": stage_rows,
    }
    for scale in scales:
        payload["measurement_reliability"][scale] = measurement_icc(records, labels, scale)
        payload["measurement_error"][scale] = measurement_error_summary(records, labels, scale)
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
        payload["safety_proxy_metrics"][key] = safety_proxy_metrics(record, labels)
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
                "max_upward_step": trajectory.get("max_upward_step"),
            }
    return payload


def write_measurement_repeat_long_csv(
    out_dir: Path, records: list[ExperimentRecord], labels: list[str], scales: list[str]
) -> Path:
    path = out_dir / "measurement_repeat_long.csv"
    fieldnames = [
        "stable_id",
        "outer_run_id",
        "persona",
        "group",
        "severity",
        "snapshot_id",
        "timepoint",
        "scale",
        "measurement_repeat_id",
        "reviewed_total_score",
        "sim_time",
        "completed_session_count",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            for scale in scales:
                for label in labels:
                    stats = record.series.get(scale, {}).get(label) or {}
                    for repeat_id, value in (stats.get("values_by_repeat") or {}).items():
                        writer.writerow(
                            {
                                "stable_id": record.stable_id,
                                "outer_run_id": record.repeat_id,
                                "persona": record.kbd,
                                "group": record.group,
                                "severity": record.severity,
                                "snapshot_id": f"{record.stable_id}::{label}",
                                "timepoint": label,
                                "scale": scale,
                                "measurement_repeat_id": repeat_id,
                                "reviewed_total_score": value,
                                "sim_time": stats.get("sim_time"),
                                "completed_session_count": stats.get("completed_session_count"),
                            }
                        )
    return path


def write_item_measurement_repeat_long_csv(
    out_dir: Path, records: list[ExperimentRecord], labels: list[str], scales: list[str]
) -> Path:
    path = out_dir / "item_measurement_repeat_long.csv"
    return _write_rows_csv(path, item_repeat_long_rows(records, labels, scales), drop={"cluster_id"})


def write_outer_run_metrics_csv(out_dir: Path, metrics: dict[str, Any]) -> Path:
    path = out_dir / "outer_run_metrics.csv"
    fieldnames = [
        "stable_id",
        "persona",
        "group",
        "severity",
        "outer_run_id",
        "scale",
        "baseline_label",
        "observed_endpoint_label",
        "endpoint_change",
        "normalized_delta",
        "best_change_from_baseline",
        "best_timepoint",
        "rebound_from_nadir",
        "delta_auc",
        "delta_auc_unit",
        "time_basis",
        "elapsed_simulation_days",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for key in metrics["condition_order"]:
            record = metrics["records"][key]
            for scale in metrics["scales"]:
                trajectory = metrics["trajectory_metrics"][key][scale]
                writer.writerow(
                    {
                        "stable_id": key,
                        "persona": record["kbd"],
                        "group": record["group"],
                        "severity": record["severity"],
                        "outer_run_id": record["repeat_id"],
                        "scale": scale,
                        "baseline_label": trajectory.get("baseline_label"),
                        "observed_endpoint_label": trajectory.get("endpoint_label"),
                        "endpoint_change": trajectory.get("endpoint_change"),
                        "normalized_delta": trajectory.get("normalized_delta"),
                        "best_change_from_baseline": trajectory.get("best_change_from_baseline"),
                        "best_timepoint": trajectory.get("best_timepoint"),
                        "rebound_from_nadir": trajectory.get("rebound_from_nadir"),
                        "delta_auc": trajectory.get("delta_auc"),
                        "delta_auc_unit": trajectory.get("delta_auc_unit"),
                        "time_basis": trajectory.get("time_basis"),
                        "elapsed_simulation_days": trajectory.get("elapsed_simulation_days"),
                    }
                )
    return path


def write_group_contrasts_csv(out_dir: Path, metrics: dict[str, Any]) -> Path:
    path = out_dir / "group_endpoint_contrasts.csv"
    fieldnames = [
        "scale",
        "contrast",
        "n_first",
        "n_second",
        "mean_change_first",
        "mean_change_second",
        "difference",
        "ci95_lower",
        "ci95_upper",
        "hedges_g",
        "inference_scope",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in metrics["group_endpoint_contrasts"]:
            difference = row["difference"]
            effect = row["hedges_g"]
            writer.writerow(
                {
                    "scale": row["scale"],
                    "contrast": row["contrast"],
                    "n_first": difference.get("n_endpoint"),
                    "n_second": difference.get("n_baseline"),
                    "mean_change_first": row["mean_change_first"],
                    "mean_change_second": row["mean_change_second"],
                    "difference": difference.get("estimate"),
                    "ci95_lower": difference.get("lower"),
                    "ci95_upper": difference.get("upper"),
                    "hedges_g": effect.get("estimate"),
                    "inference_scope": row["inference_scope"],
                }
            )
    return path


def write_group_time_contrasts_csv(out_dir: Path, metrics: dict[str, Any]) -> Path:
    path = out_dir / "group_time_contrasts.csv"
    rows: list[dict[str, Any]] = []
    for row in metrics["group_time_contrasts"]:
        interval = row["difference_in_change"]
        rows.append(
            {
                "scale": row["scale"],
                "timepoint": row["timepoint"],
                "contrast": row["contrast"],
                "mean_change_first": row["mean_change_first"],
                "mean_change_second": row["mean_change_second"],
                "difference_in_change": interval.get("estimate"),
                "ci95_lower": interval.get("lower"),
                "ci95_upper": interval.get("upper"),
                "n_first": interval.get("n_endpoint"),
                "n_second": interval.get("n_baseline"),
                "inference_scope": row["inference_scope"],
            }
        )
    return _write_rows_csv(path, rows)


def write_adjusted_contrasts_csv(out_dir: Path, metrics: dict[str, Any]) -> Path:
    path = out_dir / "baseline_adjusted_endpoint_contrasts.csv"
    return _write_rows_csv(path, metrics["baseline_adjusted_endpoint_contrasts"])


def write_safety_proxy_csv(out_dir: Path, metrics: dict[str, Any]) -> Path:
    path = out_dir / "safety_proxy_metrics.csv"
    rows: list[dict[str, Any]] = []
    for key in metrics["condition_order"]:
        record = metrics["records"][key]
        rows.append(
            {
                "stable_id": key,
                "persona": record["kbd"],
                "group": record["group"],
                "severity": record["severity"],
                "outer_run_id": record["repeat_id"],
                **metrics["safety_proxy_metrics"][key],
            }
        )
    return _write_rows_csv(path, rows)


def write_process_balance_csv(out_dir: Path, process_rows: list[dict[str, Any]]) -> Path:
    path = out_dir / "process_group_balance.csv"
    fields = [
        "completed_consultation_meetings",
        "cbt_prompt_completion_meeting",
        "fixed_prompt_post_completion_meetings",
        "total_turns",
        "total_chars",
        "mean_turns_per_meeting",
        "environment_tasks",
    ]
    groups = sorted({row["group"] for row in process_rows if row.get("raw_artifacts_available")})
    rows: list[dict[str, Any]] = []
    from .statistics import hedges_g

    for index, first in enumerate(groups):
        for second in groups[index + 1 :]:
            for field in fields:
                first_values = [
                    float(row[field])
                    for row in process_rows
                    if row.get("raw_artifacts_available") and row["group"] == first and row.get(field) is not None
                ]
                second_values = [
                    float(row[field])
                    for row in process_rows
                    if row.get("raw_artifacts_available") and row["group"] == second and row.get(field) is not None
                ]
                effect = hedges_g(first_values, second_values)
                rows.append(
                    {
                        "contrast": f"{first} - {second}",
                        "metric": field,
                        "n_first": len(first_values),
                        "n_second": len(second_values),
                        "mean_first": sum(first_values) / len(first_values) if first_values else None,
                        "mean_second": sum(second_values) / len(second_values) if second_values else None,
                        "standardized_mean_difference_hedges_g": effect.get("estimate"),
                        "balance_target_note": "|SMD| < 0.1 is a design target, not a post-hoc proof of no confounding",
                    }
                )
    return _write_rows_csv(path, rows)


def _write_rows_csv(path: Path, rows: list[dict[str, Any]], *, drop: set[str] | None = None) -> Path:
    drop = drop or set()
    flattened = [{key: value for key, value in row.items() if key not in drop} for row in rows]
    fieldnames = list(dict.fromkeys(key for row in flattened for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames or ["stable_id"])
        writer.writeheader()
        writer.writerows(flattened)
    return path


def write_data_dictionary(out_dir: Path) -> Path:
    path = out_dir / "data_dictionary.md"
    path.write_text(
        """# 评估数据字典

| 文件 | 统计单位 | 关键字段 | 用途 |
|---|---|---|---|
| `measurement_repeat_long.csv` | 同一冻结快照的一次量表生成 | `snapshot_id`, `measurement_repeat_id` | 测量噪声、ICC；K 不能当外层样本量 |
| `item_measurement_repeat_long.csv` | snapshot × scale × item × measurement repeat | 精确对象键、条目有序分数 | Weighted Kappa 的可追溯输入；K 仍不是外层样本量 |
| `statistics_detail.csv` | snapshot × scale | `primary_score`, `sample_sd`, measurement CI | 冻结点汇总轨迹 |
| `outer_run_metrics.csv` | 独立仿真 run × scale | `outer_run_id`, endpoint delta, AUC, nadir/rebound | 外层轨迹与效应量 |
| `group_endpoint_contrasts.csv` | group contrast × scale | outer-run difference, Hedges' g | 探索性组间比较 |
| `baseline_adjusted_endpoint_contrasts.csv` | group contrast × scale | endpoint ~ group + baseline | 探索性基线调整终点差 |
| `group_time_contrasts.csv` | group contrast × scale × time | baseline-referenced difference in change | 描述性 group×time analogue |
| `process_metrics.csv` | 独立仿真 run | turns/chars/stage completion/fixed-prompt meetings | 剂量与 CBT 交付 |
| `process_group_balance.csv` | group contrast × dose metric | outer-run Hedges' g | 注意力/暴露平衡审计 |
| `stage_metrics.csv` | run × CBT prompt meeting | stage/subgoal/transition | 阶段停留与推进审计 |
| `change_score_rows.csv` | outer run × outcome × timepoint | baseline/current/change/QC/replicate | Change+CI 的规范输入层 |
| `change_score_summary.csv` | panel × outcome × timepoint × group | mean/adjusted change、outer-run CI | Change+CI 柱和误差线 |
| `change_score_contrasts.csv` | panel × outcome × timepoint × group pair | Holm-adjusted p/star | 显著性括号的完整统计来源 |
| `engagement_events.csv` | 一次 conversation | timestamp/participants/turns/type/duration proxy | 过程图事件层；duration proxy 不是真实 elapsed duration |
| `engagement_daily_summary.csv` | group × simulation day | frequency、secondary metric、outer-run CI | Engagement Panel A |
| `engagement_24h_summary.csv` | group × fixed time bin | interactions/day、outer-run CI | Engagement Panel B |
| `engagement_session_outcome_summary.csv` | group × CBT session × metric | 正式 session turn 数、相对 T0 评分变化、outer-run CI | Engagement Panel A |
| `paper_figure_data_availability.json` | batch | 字段来源、缺失字段、空 panel、解析告警 | 禁止把缺失数据静默当 0 |
| `item_statistics.csv` | snapshot × scale item | item repeat mean/SD/agreement/entropy | 条目信度与安全代理复核 |
| `weighted_kappa_summary.csv` | scale × 分层 × 权重 | repeat-pair κ 的均值、覆盖、cluster CI/原因 | 总体及 persona/group/timepoint 一致性 |
| `weighted_kappa_pairwise.csv` | scale × 分层 × repeat pair × 权重 | Cohen weighted κ、有效/缺失对象数、NA 原因 | repeat 矩阵与边际分布敏感性检查 |
| `weighted_kappa_item.csv` | scale × item × 权重 | 条目 κ、outer-run cluster bootstrap CI | PHQ-9/BDI-II 条目级一致性 |
| `measurement_reliability_summary.csv` | scale | ICC(A,1)/(A,K)、cluster bootstrap CI、SEM、MDC95 | 总分重复可靠性与测量误差 |
| `measurement_error_snapshot.csv` | snapshot × scale | 同一冻结快照 K 次总分的 SD | 误差分布图输入 |
| `cross_scale_concurrent_validity.csv` | outer run × timepoint | 对齐的 PHQ-9/BDI-II 总分 | 同期收敛效度 |
| `cross_scale_change_agreement.csv` | outer run | 统一 baseline/endpoint 的两量表变化 | 治疗变化一致性 |
| `safety_proxy_metrics.csv` | 独立仿真 run | item 9、恶化与 response-like flags | 自动筛查；所有 flag 须人工复核 |

`session_20` 表示第 20 次咨询后的观察点，不自动等于 POST。固定回访提示词阶段仍有强制医患会谈，因此不是无干预延迟随访。

Weighted Kappa 的对象严格定义为同一 `snapshot_id × scale × item`，不同 `measurement_repeat_id` 是重复评分者；主结果用 quadratic weights，linear weights 仅作敏感性分析。量表总分不计算 Kappa，继续使用 ICC。若期望分歧为 0（常见于所有评分均为同一类别），κ 为 NA，绝不替换成 1。κ 会受类别流行率和两位评分者边际分布影响，不能只按单一阈值解释。
""",
        encoding="utf-8",
    )
    return path


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


def write_weighted_kappa_csvs(out_dir: Path, metrics: dict[str, Any]) -> tuple[Path, Path, Path]:
    payload = metrics["weighted_kappa"]
    summary_path = _write_rows_csv(out_dir / "weighted_kappa_summary.csv", payload.get("summary") or [])
    pairwise_path = _write_rows_csv(out_dir / "weighted_kappa_pairwise.csv", payload.get("pairwise") or [])
    item_path = _write_rows_csv(out_dir / "weighted_kappa_item.csv", payload.get("item") or [])
    return summary_path, pairwise_path, item_path


def write_scale_credibility_csvs(out_dir: Path, metrics: dict[str, Any]) -> dict[str, Path]:
    summary_rows: list[dict[str, Any]] = []
    snapshot_rows: list[dict[str, Any]] = []
    for scale in metrics["scales"]:
        reliability = metrics["measurement_reliability"].get(scale, {})
        error = metrics["measurement_error"].get(scale, {})
        summary_rows.append(
            {
                "scale": scale,
                "n_snapshot_targets": reliability.get("n_snapshot_targets"),
                "n_outer_run_clusters": reliability.get("n_outer_run_clusters"),
                "k_measurement": reliability.get("k_measurement"),
                "icc_a_1": reliability.get("icc_a_1"),
                "icc_a_1_ci95_lower": reliability.get("icc_a_1_ci95_lower"),
                "icc_a_1_ci95_upper": reliability.get("icc_a_1_ci95_upper"),
                "icc_a_k": reliability.get("icc_a_k"),
                "icc_a_k_ci95_lower": reliability.get("icc_a_k_ci95_lower"),
                "icc_a_k_ci95_upper": reliability.get("icc_a_k_ci95_upper"),
                "icc_definition": reliability.get("icc_definition"),
                "ci_method": reliability.get("ci_method"),
                "sem": error.get("sem"),
                "mdc95": error.get("mdc95"),
                "measurement_error_definition": error.get("definition"),
                "status": reliability.get("status"),
            }
        )
        snapshot_rows.extend(error.get("snapshot_rows") or [])
    return {
        "summary": _write_rows_csv(out_dir / "measurement_reliability_summary.csv", summary_rows),
        "snapshot": _write_rows_csv(out_dir / "measurement_error_snapshot.csv", snapshot_rows),
        "concurrent": _write_rows_csv(
            out_dir / "cross_scale_concurrent_validity.csv",
            metrics["cross_scale_concurrent_validity"].get("rows") or [],
        ),
        "change": _write_rows_csv(
            out_dir / "cross_scale_change_agreement.csv",
            metrics["cross_scale_convergence"].get("rows") or [],
        ),
    }


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
        "# 实验指标统计报告",
        "",
        "## 统计口径",
        "",
        f"- 数据集：`{metrics['metadata']['dataset_label']}`。",
        "- 主分数：固定 K 次完整复核量表总分的均值。",
        "- 内层 repeat：同一 Agent、同一冻结快照的 Monte Carlo 测量重复，不是独立患者。",
        "- 外层 R01/R02/...：相互独立的完整仿真实验，是组间描述和效应量的统计单位。",
        "- 差值 CI：默认 Welch 非配对 t 区间；只有 metadata 明确声明跨时间点配对时才使用 paired t 区间。",
        "- 中位数与去除一个最高/最低值后的均值只作为敏感性统计。",
        "- 当前没有无干预延迟随访；`session_20` 仅是第 20 次咨询后的观察点，不能自动写成 POST/随访。",
        "- CBT prompt 完成后的固定提示词会谈仍是强制医患咨询，单独计数但不作为 true follow-up。",
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
            "## 测量可靠性",
            "",
            "- 主结果为 two-way random absolute agreement：ICC(A,1)/ICC(2,1) 表示单次生成，ICC(A,K)/ICC(2,K) 表示 K 次均值；CI 按独立 outer run 聚类 bootstrap。",
            "",
            "| Scale | Snapshot targets | outer clusters | K | ICC(A,1) [95% CI] | ICC(A,K) [95% CI] | SEM | MDC95 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for scale, reliability in metrics["measurement_reliability"].items():
        error = metrics["measurement_error"].get(scale, {})
        single_interval = (
            "NA" if reliability.get("icc_a_1_ci95_lower") is None else
            f"{reliability['icc_a_1']:.3f} [{reliability['icc_a_1_ci95_lower']:.3f}, {reliability['icc_a_1_ci95_upper']:.3f}]"
        )
        average_interval = (
            "NA" if reliability.get("icc_a_k_ci95_lower") is None else
            f"{reliability['icc_a_k']:.3f} [{reliability['icc_a_k_ci95_lower']:.3f}, {reliability['icc_a_k_ci95_upper']:.3f}]"
        )
        lines.append(
            f"| {scale} | {reliability.get('n_snapshot_targets')} | {reliability.get('n_outer_run_clusters')} | "
            f"{reliability.get('k_measurement')} | {single_interval} | {average_interval} | "
            f"{_number(error.get('sem'), 3)} | {_number(error.get('mdc95'), 3)} |"
        )
    lines.extend(
        [
            "",
            "## 条目级 Weighted Kappa",
            "",
            "- 对象严格按同一 `snapshot_id × scale × item` 配对；measurement repeat 是评分者，不是独立 outer sample。",
            "- quadratic weights 为主结果，linear weights 为敏感性结果；量表总分不计算 Kappa，总分可靠性见上方 ICC。",
            "- κ 为 repeat-pair κ 的等权均值。CI 仅在至少 5 个独立 outer-run cluster 时用 cluster percentile bootstrap。",
            "- 单一类别等导致期望分歧为 0 时输出 NA；κ 受类别分布和边际分布影响，不能仅按单一阈值定性。",
            "",
            "| Scale | Weights | Pairwise mean κ | 95% cluster CI | Valid repeat pairs | Valid objects | Missing | CI/NA reason |",
            "|---|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    overall_kappa = [
        row
        for row in metrics["weighted_kappa"]["summary"]
        if row["stratum_type"] == "overall" and row["stratum_value"] == "ALL"
    ]
    for row in overall_kappa:
        interval = (
            "NA"
            if row.get("ci95_lower") is None or row.get("ci95_upper") is None
            else f"[{row['ci95_lower']:.3f}, {row['ci95_upper']:.3f}]"
        )
        reason = row.get("reason") or row.get("ci_reason") or ""
        lines.append(
            f"| {row['scale']} | {row['weights']} | {_number(row.get('kappa'), 3)} | {interval} | "
            f"{row.get('n_valid_repeat_pairs')}/{row.get('n_repeat_pairs')} | {row.get('n_valid')} | "
            f"{row.get('n_missing')} | {reason} |"
        )
    concurrent = metrics["cross_scale_concurrent_validity"]
    convergence = metrics["cross_scale_convergence"]
    lines.extend(
        [
            "",
            "## PHQ-9 / BDI-II 跨量表效度与变化一致性",
            "",
            f"- 同期收敛效度：对齐 {concurrent.get('n_aligned_snapshots')} 个 persona/condition/timepoint；Spearman ρ={_number(concurrent.get('spearman_rho'), 3)}，cluster 95% CI [{_number(concurrent.get('ci95_lower'), 3)}, {_number(concurrent.get('ci95_upper'), 3)}]。",
            "- 同期数据包含同一 outer run 的重复时间点，因此以 outer-run cluster bootstrap CI 为主，不把朴素 p 值作为显著性结论。",
            "",
            f"- 治疗变化一致性外层 run 数：{convergence.get('n_outer_runs')}；两量表使用同一 baseline 与 endpoint。",
            f"- 终点变化方向一致率：{_number(convergence.get('direction_agreement_rate') * 100 if convergence.get('direction_agreement_rate') is not None else None)}%。",
            f"- ΔPHQ-9 与 ΔBDI-II Spearman ρ：{_number(convergence.get('spearman_rho'), 3)}，95% CI [{_number(convergence.get('ci95_lower'), 3)}, {_number(convergence.get('ci95_upper'), 3)}]（p={_number(convergence.get('spearman_p'), 3)}）。负值表示改善，正值表示恶化。",
            "",
            "## 外层组间对照（探索性）",
            "",
            "| Scale | Contrast | Mean change first | Mean change second | Difference 95% CI | Hedges' g |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in metrics["group_endpoint_contrasts"]:
        lines.append(
            f"| {row['scale']} | {row['contrast']} | {_number(row.get('mean_change_first'))} | "
            f"{_number(row.get('mean_change_second'))} | {_interval(row['difference'])} | "
            f"{_number(row['hedges_g'].get('estimate'))} |"
        )
    lines.extend(
        [
            "",
            "### 基线调整终点差（探索性 ANCOVA）",
            "",
            "| Scale | Contrast | Adjusted endpoint difference | 95% CI | n outer runs | Status |",
            "|---|---|---:|---:|---:|---|",
        ]
    )
    for row in metrics["baseline_adjusted_endpoint_contrasts"]:
        lines.append(
            f"| {row['scale']} | {row['contrast']} | {_number(row.get('estimate'))} | {_interval(row)} | "
            f"{row.get('n_outer_runs')} | {row.get('status')} |"
        )
    process_rows = [row for row in metrics["process_metrics"] if row.get("raw_artifacts_available")]
    if process_rows:
        lines.extend(
            [
                "",
                "## CBT 交付与对话剂量",
                "",
                "| Run | Consultations | Prompt complete at | Fixed-prompt meetings | Turns | Characters | Environment tasks | True follow-up |",
                "|---|---:|---:|---:|---:|---:|---:|---|",
            ]
        )
        for row in process_rows:
            lines.append(
                f"| {row['group']}-{row['outer_run_id']} | {row.get('completed_consultation_meetings')} | "
                f"{row.get('cbt_prompt_completion_meeting')} | {row.get('fixed_prompt_post_completion_meetings')} | "
                f"{row.get('total_turns')} | {row.get('total_chars')} | {row.get('environment_tasks')} | "
                f"{row.get('true_no_intervention_followup')} |"
            )
    lines.extend(
        [
            "",
            "## 自动安全代理（须人工复核）",
            "",
            "| Run | PHQ item 9 max | BDI item 9 max | PHQ worsening ≥5 | PHQ worsening ≥6 | PHQ 50% response-like |",
            "|---|---:|---:|---|---|---|",
        ]
    )
    for key in metrics["condition_order"]:
        record = metrics["records"][key]
        safety = metrics["safety_proxy_metrics"][key]
        lines.append(
            f"| {record['group']}-{record['repeat_id']} | {_number(safety.get('phq_item9_max_repeat_mean'))} | "
            f"{_number(safety.get('bdi_item9_max_repeat_mean'))} | {safety.get('phq_worsening_ge_5')} | "
            f"{safety.get('phq_worsening_ge_6')} | {safety.get('phq_response_like_50pct')} |"
        )
    lines.extend(
        [
            "",
            "## 输出文件",
            "",
            "- `statistics_detail.csv`：每个时间点的主分数与测量指标。",
            "- `measurement_repeat_long.csv`：规范化的 snapshot × scale × K 测量长表。",
            "- `item_measurement_repeat_long.csv`：保留精确 item 键的 Kappa 可追溯输入长表。",
            "- `outer_run_metrics.csv`：以独立仿真 run 为单位的变化、AUC、nadir/rebound。",
            "- `group_endpoint_contrasts.csv`：外层 run 组间差和 Hedges’ g。",
            "- `baseline_adjusted_endpoint_contrasts.csv`：探索性 ANCOVA 基线调整终点差。",
            "- `group_time_contrasts.csv`：各时间点相对基线变化的组间差（描述性 group×time analogue）。",
            "- `process_metrics.csv` / `stage_metrics.csv`：会谈剂量、prompt 完成、固定回访提示词会谈和 CBT 阶段。",
            "- `process_group_balance.csv`：组间剂量标准化差；`safety_proxy_metrics.csv`：需人工复核的量表代理标记。",
            "- `item_statistics.csv`：条目级 mean、sample SD、两两 exact agreement rate、Shannon entropy (bits)。",
            "- `weighted_kappa_summary.csv` / `weighted_kappa_pairwise.csv` / `weighted_kappa_item.csv`：总体、分层、repeat-pair 与条目级有序一致性。",
            "- `measurement_reliability_summary.csv` / `measurement_error_snapshot.csv`：absolute-agreement ICC、SEM/MDC95 与快照内 repeat SD。",
            "- `cross_scale_concurrent_validity.csv` / `cross_scale_change_agreement.csv`：对齐总分与统一 baseline/endpoint 的跨量表变化。",
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
        "# 实验指标与图表索引",
        "",
        f"- 模式：`{metrics['metadata']['report_mode']}`",
        f"- 误差：`{metrics['metadata']['error_bar']}`",
        f"- y 轴：`{metrics['metadata']['y_axis']}`",
        f"- 数据集：`{metrics['metadata']['dataset_label']}`",
        f"- 观察终点规则：{metrics['metadata']['endpoint_rule']}",
        "- `session_20` 不自动解释为 POST；当前 true delayed follow-up = false。",
        "- 轨迹点/单次快照的误差条可表示 Monte Carlo 测量不确定性；核心组间对比图的 CI 明确按外层运行计算，具体以图注为准。",
        "- 每张图默认提供 PNG（300 DPI）。",
        "- 明细：[`statistics_report.md`](statistics_report.md)、[`data_dictionary.md`](data_dictionary.md)、[`outer_run_metrics.csv`](outer_run_metrics.csv)、[`process_metrics.csv`](process_metrics.csv)。",
        "",
        "## 图表",
        "",
    ]
    for chart_path in chart_paths:
        lines.extend([f"### {chart_path.stem.replace('_', ' ')}", "", f"![{chart_path.stem}]({chart_path.name})", ""])
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_paper_figure_data(
    out_dir: Path,
    *,
    change_rows: list[dict[str, Any]],
    change_estimates: list[dict[str, Any]],
    change_contrasts: list[dict[str, Any]],
    engagement_events: list[dict[str, Any]],
    engagement_coverage: list[dict[str, Any]],
    engagement_daily: list[dict[str, Any]],
    engagement_time_bins: list[dict[str, Any]],
    engagement_sessions: list[dict[str, Any]],
    diagnostics: dict[str, Any],
) -> dict[str, Path]:
    """Write the extraction/statistics layers consumed by the new paper figures."""
    paths = {
        "change_score_rows": _write_rows_csv(out_dir / "change_score_rows.csv", change_rows),
        "change_score_summary": _write_rows_csv(out_dir / "change_score_summary.csv", change_estimates),
        "change_score_contrasts": _write_rows_csv(out_dir / "change_score_contrasts.csv", change_contrasts),
        "engagement_events": _write_rows_csv(out_dir / "engagement_events.csv", engagement_events),
        "engagement_run_coverage": _write_rows_csv(out_dir / "engagement_run_coverage.csv", engagement_coverage),
        "engagement_daily_summary": _write_rows_csv(out_dir / "engagement_daily_summary.csv", engagement_daily),
        "engagement_24h_summary": _write_rows_csv(out_dir / "engagement_24h_summary.csv", engagement_time_bins),
        "engagement_session_outcome_summary": _write_rows_csv(
            out_dir / "engagement_session_outcome_summary.csv", engagement_sessions
        ),
    }
    diagnostics_path = out_dir / "paper_figure_data_availability.json"
    diagnostics_path.write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8")
    paths["data_availability"] = diagnostics_path
    return paths


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
    process_rows: list[dict[str, Any]],
    stage_rows: list[dict[str, Any]],
    dataset_label: str,
    project_root: Path,
    title_prefix: str,
    render_weighted_kappa: bool,
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
        process_rows=process_rows,
        stage_rows=stage_rows,
        dataset_label=dataset_label,
        project_root=project_root,
    )
    chart_paths = list(chart_paths)
    if render_weighted_kappa:
        from .visualization.weighted_kappa_plots import render_weighted_kappa_figures

        chart_paths.extend(render_weighted_kappa_figures(metrics["weighted_kappa"], out_dir, title_prefix))
    metrics_path = out_dir / "chart_metrics_summary.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    detail_path = write_statistics_detail_csv(out_dir, metrics)
    item_path = write_item_statistics_csv(out_dir, records, labels, scales)
    measurement_long_path = write_measurement_repeat_long_csv(out_dir, records, labels, scales)
    item_measurement_long_path = write_item_measurement_repeat_long_csv(out_dir, records, labels, scales)
    kappa_summary_path, kappa_pairwise_path, kappa_item_path = write_weighted_kappa_csvs(out_dir, metrics)
    credibility_paths = write_scale_credibility_csvs(out_dir, metrics)
    outer_metrics_path = write_outer_run_metrics_csv(out_dir, metrics)
    contrasts_path = write_group_contrasts_csv(out_dir, metrics)
    adjusted_contrasts_path = write_adjusted_contrasts_csv(out_dir, metrics)
    time_contrasts_path = write_group_time_contrasts_csv(out_dir, metrics)
    process_path = _write_rows_csv(out_dir / "process_metrics.csv", process_rows, drop={"meeting_rows"})
    process_balance_path = write_process_balance_csv(out_dir, process_rows)
    stage_path = _write_rows_csv(out_dir / "stage_metrics.csv", stage_rows)
    safety_path = write_safety_proxy_csv(out_dir, metrics)
    dictionary_path = write_data_dictionary(out_dir)
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
        "measurement_repeat_long_csv_path": _display_path(measurement_long_path, project_root),
        "item_measurement_repeat_long_csv_path": _display_path(item_measurement_long_path, project_root),
        "weighted_kappa_summary_csv_path": _display_path(kappa_summary_path, project_root),
        "weighted_kappa_pairwise_csv_path": _display_path(kappa_pairwise_path, project_root),
        "weighted_kappa_item_csv_path": _display_path(kappa_item_path, project_root),
        "measurement_reliability_summary_csv_path": _display_path(credibility_paths["summary"], project_root),
        "measurement_error_snapshot_csv_path": _display_path(credibility_paths["snapshot"], project_root),
        "cross_scale_concurrent_validity_csv_path": _display_path(credibility_paths["concurrent"], project_root),
        "cross_scale_change_agreement_csv_path": _display_path(credibility_paths["change"], project_root),
        "outer_run_metrics_csv_path": _display_path(outer_metrics_path, project_root),
        "group_contrasts_csv_path": _display_path(contrasts_path, project_root),
        "baseline_adjusted_contrasts_csv_path": _display_path(adjusted_contrasts_path, project_root),
        "group_time_contrasts_csv_path": _display_path(time_contrasts_path, project_root),
        "process_metrics_csv_path": _display_path(process_path, project_root),
        "process_group_balance_csv_path": _display_path(process_balance_path, project_root),
        "stage_metrics_csv_path": _display_path(stage_path, project_root),
        "safety_proxy_metrics_csv_path": _display_path(safety_path, project_root),
        "data_dictionary_path": _display_path(dictionary_path, project_root),
        "index_path": _display_path(index_path, project_root),
    }
