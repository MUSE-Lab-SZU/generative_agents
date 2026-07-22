"""Statistical definitions for Monte Carlo scale-measurement repeats."""

from __future__ import annotations

import math
from collections import Counter
from statistics import mean, median, pstdev, stdev
from typing import Any, Iterable

import numpy as np
from scipy.stats import t

from .schema import ENDPOINT_LABEL_PRIORITY, ExperimentRecord, SCALE_RANGES


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
    return {
        "trajectory_volatility": pstdev(scores) if len(scores) > 1 else 0.0,
        "max_upward_step": max([step for step in steps if step > 0], default=0.0) if steps else None,
        "best_change_from_baseline": nadir - baseline,
        "best_timepoint": available[nadir_index],
        "rebound_from_nadir": endpoint - nadir,
        "endpoint_change": endpoint_change,
        "delta_auc": float(np.trapezoid(deltas, dx=1.0)) if len(deltas) > 1 else 0.0,
        "delta_auc_unit": "score x observed-timepoint interval",
        "normalized_delta": endpoint_change / (scale_range[1] - scale_range[0]) if scale_range else None,
        "baseline_label": available[0],
        "endpoint_label": available[-1],
        "n_observed_timepoints": len(available),
    }
