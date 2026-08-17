"""Outer-run estimates, confidence intervals and contrasts for change figures."""

from __future__ import annotations

import math
from collections import defaultdict
from statistics import mean, stdev
from typing import Any

import numpy as np
from scipy.stats import t, ttest_ind

from ..schema import group_sort_key, label_sort_key, scale_sort_key
from ..statistics import mean_ci95


PANEL_LABELS = {
    "complete-case": "Complete-case sensitivity",
    "qc-passed": "QC-passed runs",
    "replication": "Replication",
}


def _clean_zero(value: float, tolerance: float = 1e-12) -> float:
    return 0.0 if abs(value) < tolerance else value


def _usable(row: dict[str, Any]) -> bool:
    return (
        row.get("baseline_score") is not None and row.get("current_score") is not None
    )


def _panel_rows(
    rows: list[dict[str, Any]], panel_b: str
) -> list[tuple[str, str, list[dict[str, Any]]]]:
    usable = [row for row in rows if _usable(row)]
    if panel_b == "complete-case":
        sensitivity = [row for row in usable if row.get("complete_case") is True]
    elif panel_b == "qc-passed":
        sensitivity = [row for row in usable if row.get("qc_passed") is True]
    elif panel_b == "replication":
        sensitivity = [
            row
            for row in usable
            if str(row.get("analysis_set") or "").lower() == "replication"
        ]
    else:
        raise ValueError(f"Unsupported panel B selection: {panel_b}")
    return [
        ("A", "All available runs", usable),
        ("B", PANEL_LABELS[panel_b], sensitivity),
    ]


def _holm_adjust(rows: list[dict[str, Any]]) -> None:
    valid = [
        (index, float(row["p_value_raw"]))
        for index, row in enumerate(rows)
        if row.get("p_value_raw") is not None
    ]
    valid.sort(key=lambda item: item[1])
    running = 0.0
    total = len(valid)
    for rank, (index, value) in enumerate(valid):
        adjusted = min(1.0, value * (total - rank))
        running = max(running, adjusted)
        rows[index]["p_value_adjusted"] = running
        rows[index]["significance"] = p_value_stars(running)


def p_value_stars(value: float | None) -> str:
    if value is None or not math.isfinite(value):
        return ""
    if value < 0.001:
        return "***"
    if value < 0.01:
        return "**"
    if value < 0.05:
        return "*"
    return "ns"


def _mean_analysis(
    panel: str,
    panel_title: str,
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    cells: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        cells[(row["outcome"], row["timepoint"], row["group"])].append(row)
    estimates: list[dict[str, Any]] = []
    for (outcome, timepoint, group), values in cells.items():
        changes = [
            float(row["current_score"]) - float(row["baseline_score"]) for row in values
        ]
        interval = mean_ci95(changes)
        estimates.append(
            {
                "panel": panel,
                "panel_title": panel_title,
                "outcome": outcome,
                "timepoint": timepoint,
                "group": group,
                "estimate": mean(changes),
                "ci95_lower": interval["lower"],
                "ci95_upper": interval["upper"],
                "n_outer_runs": len(changes),
                "estimator": "outer_run_mean_change",
                "status": "ok",
            }
        )
    contrasts: list[dict[str, Any]] = []
    by_time: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        by_time[(row["outcome"], row["timepoint"])][row["group"]].append(
            float(row["current_score"]) - float(row["baseline_score"])
        )
    for (outcome, timepoint), groups in by_time.items():
        names = sorted(groups, key=group_sort_key)
        family: list[dict[str, Any]] = []
        for first_index, first in enumerate(names):
            for second in names[first_index + 1 :]:
                first_values, second_values = groups[first], groups[second]
                p_value = None
                if (
                    len(first_values) >= 2
                    and len(second_values) >= 2
                    and (stdev(first_values) > 0 or stdev(second_values) > 0)
                ):
                    result = ttest_ind(first_values, second_values, equal_var=False)
                    if math.isfinite(float(result.pvalue)):
                        p_value = float(result.pvalue)
                family.append(
                    {
                        "panel": panel,
                        "panel_title": panel_title,
                        "outcome": outcome,
                        "timepoint": timepoint,
                        "first_group": first,
                        "second_group": second,
                        "difference": mean(first_values) - mean(second_values),
                        "p_value_raw": p_value,
                        "p_value_adjusted": None,
                        "significance": "",
                        "method": "welch_t_outer_run_change_holm",
                        "n_first": len(first_values),
                        "n_second": len(second_values),
                        "status": (
                            "ok"
                            if p_value is not None
                            else "insufficient_or_zero_variance"
                        ),
                    }
                )
        _holm_adjust(family)
        contrasts.extend(family)
    return estimates, contrasts


def _ancova_analysis(
    panel: str,
    panel_title: str,
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Estimate endpoint score by group at the pooled mean baseline, then subtract it."""
    estimates: list[dict[str, Any]] = []
    contrasts: list[dict[str, Any]] = []
    by_time: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_time[(row["outcome"], row["timepoint"])].append(row)
    for (outcome, timepoint), values in by_time.items():
        groups = sorted({row["group"] for row in values}, key=group_sort_key)
        baseline_reference = mean(float(row["baseline_score"]) for row in values)
        design_rows: list[list[float]] = []
        outcome_values: list[float] = []
        for row in values:
            design_rows.append(
                [1.0]
                + [1.0 if row["group"] == group else 0.0 for group in groups[1:]]
                + [float(row["baseline_score"]) - baseline_reference]
            )
            outcome_values.append(float(row["current_score"]))
        design = np.asarray(design_rows, dtype=float)
        response = np.asarray(outcome_values, dtype=float)
        degrees = len(response) - design.shape[1]
        estimable = degrees > 0 and np.linalg.matrix_rank(design) == design.shape[1]
        if not estimable:
            for group in groups:
                estimates.append(
                    {
                        "panel": panel,
                        "panel_title": panel_title,
                        "outcome": outcome,
                        "timepoint": timepoint,
                        "group": group,
                        "estimate": None,
                        "ci95_lower": None,
                        "ci95_upper": None,
                        "n_outer_runs": sum(row["group"] == group for row in values),
                        "estimator": "ancova_endpoint_at_pooled_mean_baseline",
                        "status": "insufficient_or_singular",
                    }
                )
            continue
        coefficients = np.linalg.lstsq(design, response, rcond=None)[0]
        residuals = response - design @ coefficients
        residual_variance = float(residuals @ residuals / degrees)
        covariance = residual_variance * np.linalg.inv(design.T @ design)
        vectors: dict[str, np.ndarray] = {}
        for group_index, group in enumerate(groups):
            vector = np.zeros(design.shape[1], dtype=float)
            vector[0] = 1.0
            if group_index:
                vector[group_index] = 1.0
            vectors[group] = vector
            predicted = float(vector @ coefficients)
            standard_error = math.sqrt(max(float(vector @ covariance @ vector), 0.0))
            half = float(t.ppf(0.975, degrees)) * standard_error
            adjusted_change = _clean_zero(predicted - baseline_reference)
            estimates.append(
                {
                    "panel": panel,
                    "panel_title": panel_title,
                    "outcome": outcome,
                    "timepoint": timepoint,
                    "group": group,
                    "estimate": adjusted_change,
                    "ci95_lower": _clean_zero(adjusted_change - half),
                    "ci95_upper": _clean_zero(adjusted_change + half),
                    "n_outer_runs": sum(row["group"] == group for row in values),
                    "estimator": "ancova_endpoint_at_pooled_mean_baseline",
                    "baseline_reference": baseline_reference,
                    "status": "ok",
                }
            )
        family: list[dict[str, Any]] = []
        for first_index, first in enumerate(groups):
            for second in groups[first_index + 1 :]:
                contrast_vector = vectors[first] - vectors[second]
                estimate = _clean_zero(float(contrast_vector @ coefficients))
                standard_error = math.sqrt(
                    max(float(contrast_vector @ covariance @ contrast_vector), 0.0)
                )
                statistic = (
                    estimate / standard_error if standard_error > 1e-12 else None
                )
                p_value = (
                    float(2.0 * t.sf(abs(statistic), degrees))
                    if statistic is not None
                    else None
                )
                family.append(
                    {
                        "panel": panel,
                        "panel_title": panel_title,
                        "outcome": outcome,
                        "timepoint": timepoint,
                        "first_group": first,
                        "second_group": second,
                        "difference": estimate,
                        "p_value_raw": p_value,
                        "p_value_adjusted": None,
                        "significance": "",
                        "method": "ancova_group_contrast_holm",
                        "n_first": sum(row["group"] == first for row in values),
                        "n_second": sum(row["group"] == second for row in values),
                        "df": degrees,
                        "status": (
                            "ok" if p_value is not None else "zero_residual_variance"
                        ),
                    }
                )
        _holm_adjust(family)
        contrasts.extend(family)
    return estimates, contrasts


def build_change_ci_analysis(
    rows: list[dict[str, Any]],
    *,
    panel_b: str = "complete-case",
    estimator: str = "mean",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    estimates: list[dict[str, Any]] = []
    contrasts: list[dict[str, Any]] = []
    panel_counts: dict[str, int] = {}
    for panel, title, selected in _panel_rows(rows, panel_b):
        panel_counts[panel] = len({row["stable_id"] for row in selected})
        if estimator == "mean":
            panel_estimates, panel_contrasts = _mean_analysis(panel, title, selected)
        elif estimator == "ancova":
            panel_estimates, panel_contrasts = _ancova_analysis(panel, title, selected)
        else:
            raise ValueError(f"Unsupported change estimator: {estimator}")
        estimates.extend(panel_estimates)
        contrasts.extend(panel_contrasts)
    estimates.sort(
        key=lambda row: (
            row["panel"],
            scale_sort_key(row["outcome"]),
            label_sort_key(row["timepoint"]),
            group_sort_key(row["group"]),
        )
    )
    diagnostics = {
        "panel_b_selection": panel_b,
        "estimator": estimator,
        "panel_outer_run_counts": panel_counts,
        "empty_panels": [panel for panel, count in panel_counts.items() if count == 0],
        "multiple_testing": "Holm adjustment within each panel x outcome x timepoint family",
        "inference_unit": "independent outer simulation run",
    }
    return estimates, contrasts, diagnostics
