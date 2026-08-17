"""Repeated-measures and standardized item effects for symptom forest plots."""

from __future__ import annotations

import math
import warnings
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm, ttest_ind

from ..data.longitudinal import ordered_timepoints


EFFECT_TERMS = {
    "time": "Time",
    "treatment": "Group",
    "time:treatment": "Group × Time",
}


def benjamini_hochberg(values: list[float | None]) -> list[float | None]:
    adjusted: list[float | None] = [None] * len(values)
    valid = [
        (index, float(value))
        for index, value in enumerate(values)
        if value is not None and math.isfinite(float(value))
    ]
    if not valid:
        return adjusted
    ordered = sorted(valid, key=lambda pair: pair[1])
    running = 1.0
    total = len(ordered)
    for rank_index in range(total - 1, -1, -1):
        original_index, p_value = ordered[rank_index]
        rank = rank_index + 1
        running = min(running, p_value * total / rank)
        adjusted[original_index] = min(1.0, running)
    return adjusted


def _fit_item_model(
    part: pd.DataFrame,
) -> tuple[dict[str, dict[str, float]], str, str | None]:
    import statsmodels.formula.api as smf

    formula = "score ~ time + treatment + time:treatment"
    failure: str | None = None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fitted = smf.mixedlm(formula, part, groups=part["run"]).fit(
                reml=False, method="lbfgs", maxiter=500, disp=False
            )
        if not bool(getattr(fitted, "converged", True)):
            raise RuntimeError("mixed model did not converge")
        term_values = [float(fitted.params[term]) for term in EFFECT_TERMS]
        term_errors = [float(fitted.bse[term]) for term in EFFECT_TERMS]
        random_variance = float(np.asarray(fitted.cov_re)[0, 0])
        if (
            not all(
                math.isfinite(value)
                for value in [*term_values, *term_errors, random_variance]
            )
            or any(abs(value) > 10 for value in term_values)
            or any(value <= 0 or value > 10 for value in term_errors)
            or random_variance <= 1e-8
        ):
            raise RuntimeError(
                "mixed model produced a singular or implausibly scaled covariance estimate"
            )
        estimator = "mixedlm_random_intercept_ml"
    except Exception as exc:
        failure = str(exc)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fitted = smf.ols(formula, part).fit(
                cov_type="cluster", cov_kwds={"groups": part["run"]}
            )
        estimator = "ols_cluster_run_fallback"
    output: dict[str, dict[str, float]] = {}
    for term in EFFECT_TERMS:
        estimate = float(fitted.params[term])
        se = float(fitted.bse[term])
        output[term] = {
            "estimate": estimate,
            "se": se,
            "ci95_lower": estimate - 1.96 * se,
            "ci95_upper": estimate + 1.96 * se,
            "p_value": float(fitted.pvalues[term]),
        }
    return output, estimator, failure


def build_model_effects(
    frame: pd.DataFrame,
    *,
    scale: str,
    cbt_group: str,
    control_group: str,
    fdr_alpha: float = 0.05,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    selected = frame.loc[
        (frame["scale"] == scale) & frame["group"].isin([cbt_group, control_group])
    ].copy()
    if selected.empty:
        return pd.DataFrame(), {
            "status": "unavailable",
            "reason": "No rows for selected scale/groups",
        }
    group_runs = selected.groupby("group", observed=True)["run"].nunique().to_dict()
    missing_groups = [
        group for group in (cbt_group, control_group) if group_runs.get(group, 0) < 2
    ]
    if missing_groups:
        return pd.DataFrame(), {
            "status": "unavailable",
            "reason": f"Need at least 2 independent runs in each arm; insufficient: {missing_groups}",
            "runs_by_group": {
                str(key): int(value) for key, value in group_runs.items()
            },
        }
    timepoints_by_group = {
        group: set(part["timepoint"].astype(str))
        for group, part in selected.groupby("group", observed=True)
    }
    common_timepoints = set.intersection(
        *(timepoints_by_group[group] for group in (cbt_group, control_group))
    )
    timepoints = [
        value for value in ordered_timepoints(selected) if value in common_timepoints
    ]
    if len(timepoints) < 2:
        return pd.DataFrame(), {
            "status": "unavailable",
            "reason": "Selected arms do not share at least two assessment timepoints",
            "timepoints_by_group": {
                key: sorted(value) for key, value in timepoints_by_group.items()
            },
        }
    selected = selected.loc[selected["timepoint"].isin(timepoints)].copy()
    time_map = {
        value: index / max(1, len(timepoints) - 1)
        for index, value in enumerate(timepoints)
    }
    selected["time"] = selected["timepoint"].map(time_map).astype(float)
    selected["treatment"] = (selected["group"] == cbt_group).astype(float)
    selected = selected.rename(columns={"item_score": "score"})
    rows: list[dict[str, Any]] = []
    failures: dict[str, str] = {}
    for item, part in selected.groupby("item", observed=True):
        try:
            estimates, estimator, failure = _fit_item_model(part)
        except Exception as exc:
            failures[str(int(item))] = str(exc)
            continue
        if failure:
            failures[str(int(item))] = failure
        for term, values in estimates.items():
            rows.append(
                {
                    "scale": scale,
                    "item": int(item),
                    "effect": EFFECT_TERMS[term],
                    "term": term,
                    "estimate": values["estimate"],
                    "se": values["se"],
                    "ci95_lower": values["ci95_lower"],
                    "ci95_upper": values["ci95_upper"],
                    "p_value": values["p_value"],
                    "estimator": estimator,
                    "time_coding": "0=baseline, 1=last selected assessment",
                    "cbt_group": cbt_group,
                    "control_group": control_group,
                }
            )
    result = pd.DataFrame(rows)
    if result.empty:
        return result, {
            "status": "unavailable",
            "reason": "All item models failed",
            "failures": failures,
        }
    result["p_fdr"] = benjamini_hochberg(result["p_value"].tolist())
    result["significant_fdr"] = result["p_fdr"].astype(float) <= fdr_alpha
    minimum_n = min(int(value) for value in group_runs.values())
    warnings_list = []
    if minimum_n < 5:
        warnings_list.append(
            "Fewer than 5 independent runs in at least one arm; model CIs and FDR results are exploratory"
        )
    if failures:
        warnings_list.append(
            "One or more mixed models required cluster-robust OLS fallback or failed"
        )
    return result, {
        "status": "available" if not warnings_list else "available_with_warnings",
        "runs_by_group": {str(key): int(value) for key, value in group_runs.items()},
        "timepoints": timepoints,
        "timepoint_rule": "intersection of observed CBT and control assessment labels",
        "fdr_method": "Benjamini-Hochberg across item × effect tests",
        "fdr_alpha": fdr_alpha,
        "model": "item_score ~ normalized_time + CBT + normalized_time:CBT; random intercept for run",
        "fallbacks": failures,
        "warnings": warnings_list,
    }


def _cohen_d(
    part: pd.DataFrame, cbt_group: str, control_group: str
) -> dict[str, float] | None:
    cbt = part.loc[part["group"] == cbt_group, "value"].astype(float).to_numpy()
    control = part.loc[part["group"] == control_group, "value"].astype(float).to_numpy()
    n_cbt, n_control = len(cbt), len(control)
    if n_cbt < 2 or n_control < 2:
        return None
    df = n_cbt + n_control - 2
    pooled_variance = (
        (n_cbt - 1) * np.var(cbt, ddof=1) + (n_control - 1) * np.var(control, ddof=1)
    ) / df
    if pooled_variance <= 0 or not math.isfinite(float(pooled_variance)):
        return None
    effect = (float(np.mean(cbt)) - float(np.mean(control))) / math.sqrt(
        float(pooled_variance)
    )
    se = math.sqrt((n_cbt + n_control) / (n_cbt * n_control) + effect**2 / (2 * df))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        test = ttest_ind(cbt, control, equal_var=False)
    return {
        "cohen_d": effect,
        "se": se,
        "ci95_lower": effect - float(norm.ppf(0.975)) * se,
        "ci95_upper": effect + float(norm.ppf(0.975)) * se,
        "p_value": float(test.pvalue),
        "n_cbt": n_cbt,
        "n_control": n_control,
    }


def build_timepoint_effects(
    frame: pd.DataFrame,
    *,
    scale: str,
    cbt_group: str,
    control_group: str,
    timepoints: list[str] | None = None,
    fdr_alpha: float = 0.05,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    selected = frame.loc[
        (frame["scale"] == scale) & frame["group"].isin([cbt_group, control_group])
    ].copy()
    available = ordered_timepoints(selected)
    baseline = available[0] if available else None
    if timepoints is None:
        post = [value for value in available if value != baseline]
        timepoints = post[-3:]
    missing = [value for value in timepoints if value not in available]
    if missing:
        return (
            pd.DataFrame(),
            pd.DataFrame(),
            {
                "status": "unavailable",
                "reason": f"Requested timepoints not found: {missing}",
                "available_timepoints": available,
            },
        )
    selected = selected.loc[selected["timepoint"].isin(timepoints)].copy()
    time_order = {value: index for index, value in enumerate(timepoints)}
    selected["time_order"] = selected["timepoint"].map(time_order)
    item_values = selected.rename(columns={"item_score": "value"})
    rows: list[dict[str, Any]] = []
    for (timepoint, item), part in item_values.groupby(
        ["timepoint", "item"], observed=True, sort=False
    ):
        effect = _cohen_d(part, cbt_group, control_group)
        if effect:
            rows.append(
                {
                    "scale": scale,
                    "timepoint": str(timepoint),
                    "time_order": time_order[str(timepoint)],
                    "item": int(item),
                    **effect,
                }
            )
    item_effects = pd.DataFrame(rows)
    total_values = (
        selected.groupby(["run", "persona", "group", "timepoint"], observed=True)[
            "item_score"
        ]
        .sum()
        .reset_index(name="value")
    )
    total_rows: list[dict[str, Any]] = []
    for timepoint, part in total_values.groupby("timepoint", observed=True, sort=False):
        effect = _cohen_d(part, cbt_group, control_group)
        if effect:
            total_rows.append(
                {
                    "scale": scale,
                    "timepoint": str(timepoint),
                    "time_order": time_order[str(timepoint)],
                    **effect,
                }
            )
    total_effects = pd.DataFrame(total_rows)
    if not item_effects.empty:
        item_effects = item_effects.sort_values(["time_order", "item"]).reset_index(
            drop=True
        )
        item_effects["p_fdr"] = benjamini_hochberg(item_effects["p_value"].tolist())
        item_effects["significant_fdr"] = (
            item_effects["p_fdr"].astype(float) <= fdr_alpha
        )
        reference = (
            total_effects.set_index("timepoint")["cohen_d"].to_dict()
            if not total_effects.empty
            else {}
        )
        item_effects["total_score_reference_d"] = item_effects["timepoint"].map(
            reference
        )
    if not total_effects.empty:
        total_effects = total_effects.sort_values("time_order").reset_index(drop=True)
    minimum_n = None
    if not item_effects.empty:
        minimum_n = int(
            min(item_effects["n_cbt"].min(), item_effects["n_control"].min())
        )
    warnings_list = []
    if minimum_n is not None and minimum_n < 5:
        warnings_list.append(
            "Fewer than 5 independent runs in at least one arm; Cohen's d CIs are highly unstable"
        )
    return (
        item_effects,
        total_effects,
        {
            "status": (
                "available"
                if not warnings_list and not item_effects.empty
                else (
                    "available_with_warnings"
                    if not item_effects.empty
                    else "unavailable"
                )
            ),
            "timepoints": timepoints,
            "effect_direction": "negative Cohen's d favors CBT because higher symptom scores are worse",
            "fdr_method": "Benjamini-Hochberg across item × timepoint tests",
            "fdr_alpha": fdr_alpha,
            "minimum_runs_per_arm": minimum_n,
            "warnings": warnings_list,
        },
    )
