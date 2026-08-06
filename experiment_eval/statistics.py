"""Statistical definitions for outcomes and Monte Carlo measurement repeats."""

from __future__ import annotations

import math
from collections import Counter
from datetime import datetime
from statistics import mean, median, pstdev, stdev
from typing import Any, Iterable

import numpy as np
from scipy.optimize import brentq
from scipy.stats import nct, spearmanr, t

from .schema import ENDPOINT_LABEL_PRIORITY, ExperimentRecord, SCALE_RANGES


def _parse_sim_time(value: Any) -> datetime | None:
    if not value:
        return None
    text = str(value).strip()
    for fmt in ("%Y%m%d-%H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def quantiles(values: list[float]) -> tuple[float, float]:
    if not values:
        return (math.nan, math.nan)
    q1, q3 = np.quantile(np.asarray(values, dtype=float), [0.25, 0.75], method="linear")
    return float(q1), float(q3)


def mean_ci95(values: Iterable[float]) -> dict[str, Any]:
    clean = [float(value) for value in values]
    n = len(clean)
    center = mean(clean) if clean else None
    if n < 2:
        return {"lower": None, "upper": None, "method": "student_t", "df": None, "n": n}
    sd = stdev(clean)
    half = float(t.ppf(0.975, n - 1)) * sd / math.sqrt(n)
    return {"lower": center - half, "upper": center + half, "method": "student_t", "df": n - 1, "n": n}


def descriptive_statistics(values: Iterable[float]) -> dict[str, Any]:
    clean = [float(value) for value in values]
    if not clean:
        return {
            "mean": None,
            "sample_sd": None,
            "median": None,
            "q1": None,
            "q3": None,
            "iqr": None,
            "ci95": mean_ci95([]),
        }
    q1, q3 = quantiles(clean)
    return {
        "mean": mean(clean),
        "sample_sd": stdev(clean) if len(clean) > 1 else None,
        "median": median(clean),
        "q1": q1,
        "q3": q3,
        "iqr": q3 - q1,
        "ci95": mean_ci95(clean),
    }


def category_metrics(categories: Iterable[str]) -> dict[str, Any]:
    clean = [str(value) for value in categories if str(value).strip()]
    counts = Counter(clean)
    n = len(clean)
    pair_count = n * (n - 1) // 2
    same_pairs = sum(count * (count - 1) // 2 for count in counts.values())
    flip_rate = (pair_count - same_pairs) / pair_count if pair_count else 0.0
    modal = max(counts.values()) / n if n else None
    return {
        "category_counts": dict(counts),
        "modal_confidence": modal,
        "category_pairwise_flip_rate": flip_rate if n else None,
        "any_category_flip": len(counts) > 1,
    }


def exact_pairwise_agreement(values: Iterable[float]) -> float | None:
    clean = [float(value) for value in values]
    n = len(clean)
    pairs = n * (n - 1) // 2
    if not pairs:
        return None
    counts = Counter(clean)
    return sum(count * (count - 1) // 2 for count in counts.values()) / pairs


def shannon_entropy(values: Iterable[float]) -> float | None:
    clean = [float(value) for value in values]
    if not clean:
        return None
    counts = Counter(clean)
    n = len(clean)
    return -sum((count / n) * math.log2(count / n) for count in counts.values())


def item_statistics(item_score_rows: list[list[float]]) -> list[dict[str, Any]]:
    if not item_score_rows:
        return []
    lengths = {len(row) for row in item_score_rows}
    if len(lengths) != 1:
        raise ValueError(f"Inconsistent item_scores lengths across repeats: {sorted(lengths)}")
    rows: list[dict[str, Any]] = []
    for index in range(next(iter(lengths))):
        values = [float(row[index]) for row in item_score_rows]
        rows.append(
            {
                "item_id": index + 1,
                "n": len(values),
                "mean": mean(values),
                "sample_sd": stdev(values) if len(values) > 1 else None,
                "exact_agreement_rate": exact_pairwise_agreement(values),
                "score_entropy_bits": shannon_entropy(values),
                "scores": values,
            }
        )
    return rows


def welch_difference_ci(endpoint: Iterable[float], baseline: Iterable[float]) -> dict[str, Any]:
    end = [float(value) for value in endpoint]
    base = [float(value) for value in baseline]
    estimate = mean(end) - mean(base) if end and base else None
    result = {
        "estimate": estimate,
        "lower": None,
        "upper": None,
        "method": "welch_difference_t",
        "assumption": "Monte Carlo repeats at baseline and endpoint are independent",
        "n_baseline": len(base),
        "n_endpoint": len(end),
        "df": None,
    }
    if len(end) < 2 or len(base) < 2:
        return result
    v_end = stdev(end) ** 2 / len(end)
    v_base = stdev(base) ** 2 / len(base)
    se2 = v_end + v_base
    if se2 == 0:
        result.update({"lower": estimate, "upper": estimate, "df": math.inf})
        return result
    df = se2**2 / (v_end**2 / (len(end) - 1) + v_base**2 / (len(base) - 1))
    half = float(t.ppf(0.975, df)) * math.sqrt(se2)
    result.update({"lower": estimate - half, "upper": estimate + half, "df": df})
    return result


def paired_difference_ci(
    endpoint_by_repeat: dict[str, float], baseline_by_repeat: dict[str, float]
) -> dict[str, Any]:
    shared = sorted(set(endpoint_by_repeat) & set(baseline_by_repeat))
    differences = [float(endpoint_by_repeat[key]) - float(baseline_by_repeat[key]) for key in shared]
    estimate = mean(differences) if differences else None
    result = {
        "estimate": estimate,
        "lower": None,
        "upper": None,
        "method": "paired_difference_t",
        "assumption": "Metadata explicitly declares repeats paired across timepoints",
        "n_pairs": len(differences),
        "repeat_ids": shared,
        "df": None,
    }
    if len(differences) < 2:
        return result
    sd = stdev(differences)
    half = float(t.ppf(0.975, len(differences) - 1)) * sd / math.sqrt(len(differences))
    result.update({"lower": estimate - half, "upper": estimate + half, "df": len(differences) - 1})
    return result


def labels_with_score(record: ExperimentRecord, labels: list[str], scale: str) -> list[str]:
    return [label for label in labels if (record.series.get(scale, {}).get(label) or {}).get("score") is not None]


def baseline_label_for_record(record: ExperimentRecord, labels: list[str], scale: str) -> str | None:
    available = labels_with_score(record, labels, scale)
    if not available:
        return None
    return "T0" if "T0" in available else available[0]


def endpoint_label_for_record(record: ExperimentRecord, labels: list[str], scale: str) -> str | None:
    available = labels_with_score(record, labels, scale)
    if not available:
        return None
    for label in ENDPOINT_LABEL_PRIORITY:
        if label in available:
            return label
    return available[-1]


def stat_at(record: ExperimentRecord, scale: str, label: str | None, field: str) -> Any:
    if label is None:
        return None
    return (record.series.get(scale, {}).get(label) or {}).get(field)


def score_at(record: ExperimentRecord, scale: str, label: str | None) -> float | None:
    value = stat_at(record, scale, label, "score")
    return float(value) if value is not None else None


def delta_ci(record: ExperimentRecord, labels: list[str], scale: str) -> dict[str, Any]:
    baseline = baseline_label_for_record(record, labels, scale)
    endpoint = endpoint_label_for_record(record, labels, scale)
    baseline_stats = record.series.get(scale, {}).get(baseline or "", {})
    endpoint_stats = record.series.get(scale, {}).get(endpoint or "", {})
    if record.repeats_paired_across_timepoints:
        result = paired_difference_ci(
            endpoint_stats.get("values_by_repeat", {}), baseline_stats.get("values_by_repeat", {})
        )
    else:
        result = welch_difference_ci(endpoint_stats.get("values", []), baseline_stats.get("values", []))
    result.update(
        {
            "baseline_label": baseline,
            "endpoint_label": endpoint,
            "pairing_evidence": record.pairing_evidence,
            "uncertainty_scope": "Monte Carlo measurement uncertainty; not patient or outer-experiment uncertainty",
        }
    )
    return result


def trajectory_metrics(record: ExperimentRecord, labels: list[str], scale: str) -> dict[str, Any]:
    available = labels_with_score(record, labels, scale)
    if not available:
        return {
            "trajectory_volatility": None,
            "max_upward_step": None,
            "best_change_from_baseline": None,
            "best_timepoint": None,
            "rebound_from_nadir": None,
            "endpoint_change": None,
            "delta_auc": None,
            "normalized_delta": None,
        }
    scores = [float(score_at(record, scale, label)) for label in available]
    baseline = scores[0]
    endpoint = scores[-1]
    nadir = min(scores)
    nadir_index = scores.index(nadir)
    steps = [scores[index] - scores[index - 1] for index in range(1, len(scores))]
    deltas = [score - baseline for score in scores]
    scale_range = SCALE_RANGES.get(scale)
    endpoint_change = endpoint - baseline
    parsed_times = [_parse_sim_time(stat_at(record, scale, label, "sim_time")) for label in available]
    if all(value is not None for value in parsed_times):
        origin = parsed_times[0]
        x_values = [(value - origin).total_seconds() / 86400.0 for value in parsed_times]
        auc_unit = "score x simulation-day"
        time_basis = "sim_time"
    else:
        x_values = [float(index) for index in range(len(available))]
        auc_unit = "score x observed-timepoint interval"
        time_basis = "ordered_timepoint_fallback"
    return {
        "trajectory_volatility": pstdev(scores) if len(scores) > 1 else 0.0,
        "max_upward_step": max([step for step in steps if step > 0], default=0.0) if steps else None,
        "best_change_from_baseline": nadir - baseline,
        "best_timepoint": available[nadir_index],
        "rebound_from_nadir": endpoint - nadir,
        "endpoint_change": endpoint_change,
        "delta_auc": float(np.trapezoid(deltas, x=x_values)) if len(deltas) > 1 else 0.0,
        "delta_auc_unit": auc_unit,
        "time_basis": time_basis,
        "elapsed_simulation_days": x_values[-1] if x_values else None,
        "normalized_delta": endpoint_change / (scale_range[1] - scale_range[0]) if scale_range else None,
        "baseline_label": available[0],
        "endpoint_label": available[-1],
        "n_observed_timepoints": len(available),
    }


def measurement_icc(
    records: list[ExperimentRecord], labels: list[str], scale: str
) -> dict[str, Any]:
    """One-way random-effects ICC across frozen snapshot targets.

    Rows are independent snapshot targets and columns are repeated measurements of
    the same target. This estimates measurement repeat reliability, not outer-run
    reliability or clinical test-retest reliability.
    """
    rows: list[list[float]] = []
    target_ids: list[str] = []
    for record in records:
        for label in labels:
            values = (record.series.get(scale, {}).get(label) or {}).get("values") or []
            if values:
                rows.append([float(value) for value in values])
                target_ids.append(f"{record.stable_id}::{label}")
    if not rows:
        return {
            "icc_1_1": None,
            "icc_1_k": None,
            "n_snapshot_targets": 0,
            "k_measurement": None,
            "status": "unavailable",
        }
    k_values = {len(row) for row in rows}
    if len(k_values) != 1 or next(iter(k_values)) < 2 or len(rows) < 2:
        return {
            "icc_1_1": None,
            "icc_1_k": None,
            "n_snapshot_targets": len(rows),
            "k_measurement": next(iter(k_values)) if len(k_values) == 1 else None,
            "status": "requires_at_least_two_targets_with_equal_k",
        }
    matrix = np.asarray(rows, dtype=float)
    n_targets, k = matrix.shape
    target_means = matrix.mean(axis=1)
    grand_mean = float(matrix.mean())
    between_ms = float(k * np.square(target_means - grand_mean).sum() / (n_targets - 1))
    within_ms = float(np.square(matrix - target_means[:, None]).sum() / (n_targets * (k - 1)))
    denominator = between_ms + (k - 1) * within_ms
    icc_single = (between_ms - within_ms) / denominator if denominator else 1.0
    icc_average = (between_ms - within_ms) / between_ms if between_ms else 1.0
    return {
        "icc_1_1": icc_single,
        "icc_1_k": icc_average,
        "between_target_mean_square": between_ms,
        "within_target_mean_square": within_ms,
        "n_snapshot_targets": n_targets,
        "k_measurement": k,
        "status": "ok",
        "scope": "measurement repeats nested in frozen snapshot targets",
        "target_ids": target_ids,
    }


def cross_scale_convergence(
    records: list[ExperimentRecord], labels: list[str], first_scale: str = "PHQ-9", second_scale: str = "BDI-II"
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for record in records:
        first = trajectory_metrics(record, labels, first_scale).get("normalized_delta")
        second = trajectory_metrics(record, labels, second_scale).get("normalized_delta")
        if first is None or second is None:
            continue
        rows.append(
            {
                "stable_id": record.stable_id,
                "group": record.group,
                "kbd": record.kbd,
                "repeat_id": record.repeat_id,
                "first_normalized_delta": first,
                "second_normalized_delta": second,
                "direction_agrees": (first == 0 and second == 0) or (first * second > 0),
            }
        )
    x = [row["first_normalized_delta"] for row in rows]
    y = [row["second_normalized_delta"] for row in rows]
    if len(rows) >= 2 and len(set(x)) > 1 and len(set(y)) > 1:
        correlation = spearmanr(x, y)
        rho, p_value = float(correlation.statistic), float(correlation.pvalue)
    else:
        rho, p_value = None, None
    return {
        "first_scale": first_scale,
        "second_scale": second_scale,
        "n_outer_runs": len(rows),
        "direction_agreement_rate": mean(row["direction_agrees"] for row in rows) if rows else None,
        "spearman_rho": rho,
        "spearman_p_exploratory": p_value,
        "rows": rows,
    }


def hedges_g(first: Iterable[float], second: Iterable[float]) -> dict[str, Any]:
    a, b = [float(value) for value in first], [float(value) for value in second]
    if len(a) < 2 or len(b) < 2:
        return {
            "estimate": None,
            "lower": None,
            "upper": None,
            "ci_method": "noncentral_t",
            "n_first": len(a),
            "n_second": len(b),
            "status": "insufficient_outer_runs",
        }
    degrees = len(a) + len(b) - 2
    pooled_variance = ((len(a) - 1) * stdev(a) ** 2 + (len(b) - 1) * stdev(b) ** 2) / degrees
    if pooled_variance == 0:
        estimate = 0.0 if math.isclose(mean(a), mean(b)) else math.copysign(math.inf, mean(a) - mean(b))
        lower = upper = estimate if estimate == 0.0 else None
        ci_status = "degenerate_zero_variance"
    else:
        correction = 1.0 - 3.0 / (4.0 * degrees - 1.0)
        standardized = (mean(a) - mean(b)) / math.sqrt(pooled_variance)
        estimate = correction * standardized
        t_statistic = standardized / math.sqrt(1.0 / len(a) + 1.0 / len(b))

        def solve_noncentrality(target_cdf: float) -> float:
            def objective(noncentrality: float) -> float:
                return float(nct.cdf(t_statistic, degrees, noncentrality)) - target_cdf

            grid = [
                -100.0,
                -64.0,
                -48.0,
                -32.0,
                -24.0,
                -16.0,
                -12.0,
                -10.0,
                -8.0,
                -6.0,
                -4.0,
                -2.0,
                0.0,
                2.0,
                4.0,
                6.0,
                8.0,
                10.0,
                12.0,
                16.0,
                24.0,
                32.0,
                48.0,
                64.0,
                100.0,
            ]
            previous: tuple[float, float] | None = None
            for point in grid:
                value = objective(point)
                if not math.isfinite(value):
                    continue
                if value == 0:
                    return point
                if previous is not None and previous[1] * value < 0:
                    return float(brentq(objective, previous[0], point, maxiter=1000))
                previous = (point, value)
            raise ValueError("Unable to bracket noncentral-t confidence bound")

        try:
            lower_ncp = solve_noncentrality(0.975)
            upper_ncp = solve_noncentrality(0.025)
            multiplier = correction * math.sqrt(1.0 / len(a) + 1.0 / len(b))
            lower, upper = lower_ncp * multiplier, upper_ncp * multiplier
            ci_status = "ok"
        except (ValueError, RuntimeError):
            lower, upper = None, None
            ci_status = "noncentral_t_failed"
    return {
        "estimate": estimate,
        "lower": lower,
        "upper": upper,
        "ci_method": "noncentral_t",
        "ci_status": ci_status,
        "df": degrees,
        "n_first": len(a),
        "n_second": len(b),
        "status": "ok",
        "scale": "outer-run endpoint-change pooled SD",
    }


def group_endpoint_contrasts(
    records: list[ExperimentRecord], labels: list[str], scales: list[str]
) -> list[dict[str, Any]]:
    """Exploratory unadjusted contrasts using outer simulation runs as n."""
    groups = sorted({record.group for record in records})
    rows: list[dict[str, Any]] = []
    for scale in scales:
        by_group: dict[str, list[float]] = {}
        for group in groups:
            values = [
                trajectory_metrics(record, labels, scale).get("endpoint_change")
                for record in records
                if record.group == group
            ]
            by_group[group] = [float(value) for value in values if value is not None]
        for index, first in enumerate(groups):
            for second in groups[index + 1 :]:
                first_values, second_values = by_group[first], by_group[second]
                difference = welch_difference_ci(first_values, second_values)
                effect = hedges_g(first_values, second_values)
                rows.append(
                    {
                        "scale": scale,
                        "contrast": f"{first} - {second}",
                        "first_group": first,
                        "second_group": second,
                        "mean_change_first": mean(first_values) if first_values else None,
                        "mean_change_second": mean(second_values) if second_values else None,
                        "difference": difference,
                        "hedges_g": effect,
                        "interpretation": "negative difference favors the first group for symptom scores",
                        "inference_scope": "exploratory outer-run contrast; unadjusted for baseline/persona/severity",
                    }
                )
    return rows


def baseline_adjusted_endpoint_contrasts(
    records: list[ExperimentRecord], labels: list[str], scales: list[str]
) -> list[dict[str, Any]]:
    """Exploratory pairwise ANCOVA: observed endpoint ~ group + baseline.

    This is intentionally small and transparent. It does not add persona/severity
    covariates and returns unavailable when the design matrix is not estimable.
    """
    groups = sorted({record.group for record in records})
    rows: list[dict[str, Any]] = []
    for scale in scales:
        for first_index, first in enumerate(groups):
            for second in groups[first_index + 1 :]:
                observations: list[tuple[str, float, float]] = []
                for record in records:
                    if record.group not in {first, second}:
                        continue
                    baseline_label = baseline_label_for_record(record, labels, scale)
                    endpoint_label = endpoint_label_for_record(record, labels, scale)
                    baseline = score_at(record, scale, baseline_label)
                    endpoint = score_at(record, scale, endpoint_label)
                    if baseline is not None and endpoint is not None:
                        observations.append((record.group, baseline, endpoint))
                result: dict[str, Any] = {
                    "scale": scale,
                    "contrast": f"{first} - {second}",
                    "first_group": first,
                    "second_group": second,
                    "n_outer_runs": len(observations),
                    "estimate": None,
                    "lower": None,
                    "upper": None,
                    "df": None,
                    "method": "pairwise_ancova_endpoint_on_group_and_baseline",
                    "status": "insufficient_or_singular",
                    "inference_scope": "exploratory baseline-adjusted outer-run endpoint contrast",
                }
                if len(observations) >= 4:
                    design = np.asarray(
                        [[1.0, 1.0 if group == first else 0.0, baseline] for group, baseline, _ in observations],
                        dtype=float,
                    )
                    outcome = np.asarray([endpoint for _, _, endpoint in observations], dtype=float)
                    degrees = len(outcome) - design.shape[1]
                    if degrees > 0 and np.linalg.matrix_rank(design) == design.shape[1]:
                        coefficients = np.linalg.lstsq(design, outcome, rcond=None)[0]
                        residuals = outcome - design @ coefficients
                        residual_variance = float(residuals @ residuals / degrees)
                        covariance = residual_variance * np.linalg.inv(design.T @ design)
                        standard_error = math.sqrt(max(float(covariance[1, 1]), 0.0))
                        half = float(t.ppf(0.975, degrees)) * standard_error
                        estimate = float(coefficients[1])
                        result.update(
                            {
                                "estimate": estimate,
                                "lower": estimate - half,
                                "upper": estimate + half,
                                "df": degrees,
                                "baseline_coefficient": float(coefficients[2]),
                                "status": "ok",
                            }
                        )
                rows.append(result)
    return rows


def group_time_contrasts(
    records: list[ExperimentRecord], labels: list[str], scales: list[str]
) -> list[dict[str, Any]]:
    """Pairwise differences in baseline-referenced outer-run change at each timepoint."""
    groups = sorted({record.group for record in records})
    rows: list[dict[str, Any]] = []
    for scale in scales:
        for label in labels:
            by_group: dict[str, list[float]] = {}
            for group in groups:
                changes: list[float] = []
                for record in records:
                    if record.group != group:
                        continue
                    baseline = baseline_label_for_record(record, labels, scale)
                    current = score_at(record, scale, label)
                    baseline_score = score_at(record, scale, baseline)
                    if current is not None and baseline_score is not None:
                        changes.append(current - baseline_score)
                by_group[group] = changes
            for index, first in enumerate(groups):
                for second in groups[index + 1 :]:
                    interval = welch_difference_ci(by_group[first], by_group[second])
                    rows.append(
                        {
                            "scale": scale,
                            "timepoint": label,
                            "contrast": f"{first} - {second}",
                            "first_group": first,
                            "second_group": second,
                            "mean_change_first": mean(by_group[first]) if by_group[first] else None,
                            "mean_change_second": mean(by_group[second]) if by_group[second] else None,
                            "difference_in_change": interval,
                            "inference_scope": "exploratory outer-run difference in baseline-referenced change",
                        }
                    )
    return rows


def _item_mean(record: ExperimentRecord, scale: str, label: str, item_id: int) -> float | None:
    items = (record.series.get(scale, {}).get(label) or {}).get("item_statistics") or []
    for item in items:
        if item.get("item_id") == item_id:
            return float(item["mean"])
    return None


def safety_proxy_metrics(record: ExperimentRecord, labels: list[str]) -> dict[str, Any]:
    """Automated scale-item and worsening flags requiring human review."""
    phq_labels = labels_with_score(record, labels, "PHQ-9")
    bdi_labels = labels_with_score(record, labels, "BDI-II")
    phq_trajectory = trajectory_metrics(record, labels, "PHQ-9")
    bdi_trajectory = trajectory_metrics(record, labels, "BDI-II")
    phq_item9 = [_item_mean(record, "PHQ-9", label, 9) for label in phq_labels]
    bdi_item9 = [_item_mean(record, "BDI-II", label, 9) for label in bdi_labels]
    phq_change = phq_trajectory.get("endpoint_change")
    bdi_change = bdi_trajectory.get("endpoint_change")
    return {
        "phq_item9_max_repeat_mean": max((value for value in phq_item9 if value is not None), default=None),
        "phq_item9_endpoint_repeat_mean": phq_item9[-1] if phq_item9 else None,
        "bdi_item9_max_repeat_mean": max((value for value in bdi_item9 if value is not None), default=None),
        "bdi_item9_endpoint_repeat_mean": bdi_item9[-1] if bdi_item9 else None,
        "phq_worsening_ge_5": phq_change is not None and phq_change >= 5,
        "phq_worsening_ge_6": phq_change is not None and phq_change >= 6,
        "phq_response_like_50pct": (
            phq_change is not None
            and score_at(record, "PHQ-9", phq_trajectory.get("baseline_label")) not in (None, 0)
            and phq_change <= -0.5 * score_at(record, "PHQ-9", phq_trajectory.get("baseline_label"))
        ),
        "bdi_endpoint_worsened": bdi_change is not None and bdi_change > 0,
        "requires_human_review": True,
        "scope": "Agent safety proxy; not human clinical risk or harm",
    }
