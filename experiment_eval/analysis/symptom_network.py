"""Regularized symptom-network estimation from independent-run item rows.

This module contains no plotting code.  PHQ-9/BDI-II item means are bounded,
ordinal-derived variables, so the default estimator first applies a rank-based
Gaussian-copula transform and then fits an EBIC-selected graphical lasso.  The
reported edges are partial correlations derived from the selected precision
matrix, not raw correlations.
"""

from __future__ import annotations

import math
import warnings
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata
from sklearn.covariance import GraphicalLasso

from ..data.longitudinal import ordered_timepoints
from ..schema import group_sort_key, slugify
from .symptom_trajectory import EXPECTED_ITEMS


METHOD_NAME = "ordinal_gaussian_copula_ebic_graphical_lasso"
EDGE_COLUMNS = [
    "figure_key",
    "scale",
    "group",
    "timepoint",
    "n",
    "item_a",
    "item_b",
    "partial_correlation",
    "sign",
    "plotted",
    "selected_alpha",
    "method",
]


def _copula_transform(values: np.ndarray) -> np.ndarray:
    transformed = np.empty_like(values, dtype=float)
    n = values.shape[0]
    for column in range(values.shape[1]):
        ranks = rankdata(values[:, column], method="average")
        probabilities = np.clip((ranks - 0.5) / n, 1e-6, 1.0 - 1e-6)
        transformed[:, column] = norm.ppf(probabilities)
    means = transformed.mean(axis=0)
    scales = transformed.std(axis=0, ddof=0)
    return (transformed - means) / scales


def _partial_correlations(precision: np.ndarray) -> np.ndarray:
    diagonal = np.sqrt(np.diag(precision))
    partial = -precision / np.outer(diagonal, diagonal)
    np.fill_diagonal(partial, 1.0)
    return partial


def _fit_ebic_glasso(
    values: np.ndarray,
    *,
    gamma: float,
    alpha_min_ratio: float,
    n_alphas: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    transformed = _copula_transform(values)
    empirical_covariance = np.cov(transformed, rowvar=False, bias=True)
    off_diagonal = empirical_covariance.copy()
    np.fill_diagonal(off_diagonal, 0.0)
    alpha_max = float(np.max(np.abs(off_diagonal)))
    if not math.isfinite(alpha_max) or alpha_max <= 1e-12:
        raise ValueError("All transformed item covariances are zero")
    alpha_min = max(alpha_max * alpha_min_ratio, 1e-4)
    alphas = np.geomspace(alpha_max, alpha_min, num=n_alphas)
    n, p = transformed.shape
    candidates: list[tuple[float, float, np.ndarray, int]] = []
    failures = 0
    for alpha in alphas:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = GraphicalLasso(alpha=float(alpha), max_iter=500, tol=1e-4).fit(
                    transformed
                )
            precision = np.asarray(model.precision_, dtype=float)
            sign, log_determinant = np.linalg.slogdet(precision)
            if sign <= 0 or not np.isfinite(precision).all():
                failures += 1
                continue
            edge_count = int(np.count_nonzero(np.triu(np.abs(precision) > 1e-8, k=1)))
            log_likelihood_without_constant = (
                0.5
                * n
                * (log_determinant - float(np.trace(empirical_covariance @ precision)))
            )
            ebic = (
                -2.0 * log_likelihood_without_constant
                + edge_count * math.log(n)
                + 4.0 * gamma * edge_count * math.log(p)
            )
            candidates.append((float(ebic), float(alpha), precision, edge_count))
        except (FloatingPointError, ValueError):
            failures += 1
    if not candidates:
        raise ValueError("Graphical lasso failed for every alpha candidate")
    ebic, alpha, precision, edge_count = min(candidates, key=lambda row: row[0])
    return _partial_correlations(precision), {
        "selected_alpha": alpha,
        "ebic": ebic,
        "nonzero_edges": edge_count,
        "alpha_max": alpha_max,
        "alpha_min": alpha_min,
        "alpha_candidates": n_alphas,
        "failed_candidates": failures,
    }


def _panel_specs(
    frame: pd.DataFrame,
    *,
    mode: str,
    groups: list[str] | None,
    timepoints: list[str] | None,
) -> list[dict[str, str]]:
    available_groups = sorted(frame["group"].astype(str).unique(), key=group_sort_key)
    selected_groups = groups or available_groups
    missing_groups = sorted(
        set(selected_groups) - set(available_groups), key=group_sort_key
    )
    if missing_groups:
        raise ValueError(
            f"Symptom-network groups not found: {', '.join(missing_groups)}"
        )
    available_timepoints = ordered_timepoints(frame)
    if timepoints:
        missing_timepoints = [
            value for value in timepoints if value not in available_timepoints
        ]
        if missing_timepoints:
            raise ValueError(
                f"Symptom-network timepoints not found: {', '.join(missing_timepoints)}"
            )
        selected_timepoints = timepoints
    elif len(available_timepoints) <= 3:
        selected_timepoints = available_timepoints
    else:
        selected_timepoints = [
            available_timepoints[0],
            available_timepoints[len(available_timepoints) // 2],
            available_timepoints[-1],
        ]
    specs: list[dict[str, str]] = []
    if mode == "longitudinal":
        for group in selected_groups:
            figure_key = f"longitudinal_{slugify(group)}"
            for index, timepoint in enumerate(selected_timepoints):
                role = (
                    "Baseline"
                    if index == 0
                    else "Final" if index == len(selected_timepoints) - 1 else "Mid"
                )
                specs.append(
                    {
                        "figure_key": figure_key,
                        "figure_title": f"{group}: symptom network over time",
                        "group": group,
                        "timepoint": timepoint,
                        "panel_label": f"{role} · {timepoint}",
                    }
                )
        return specs
    if mode != "group-comparison":
        raise ValueError(f"Unsupported symptom-network mode: {mode}")
    if len(selected_groups) != 2:
        raise ValueError("group-comparison mode requires exactly two groups")
    comparison_timepoints = (
        [selected_timepoints[0], selected_timepoints[-1]]
        if len(selected_timepoints) > 1
        else selected_timepoints
    )
    first, second = selected_groups
    figure_key = f"comparison_{slugify(first)}_vs_{slugify(second)}"
    for group in selected_groups:
        for index, timepoint in enumerate(comparison_timepoints):
            role = "Baseline" if index == 0 else "Final"
            specs.append(
                {
                    "figure_key": figure_key,
                    "figure_title": f"{first} vs {second}: baseline to final symptom networks",
                    "group": group,
                    "timepoint": timepoint,
                    "panel_label": f"{group} · {role} · {timepoint}",
                }
            )
    return specs


def build_symptom_network_analysis(
    frame: pd.DataFrame,
    *,
    scale: str = "PHQ-9",
    mode: str = "longitudinal",
    groups: list[str] | None = None,
    timepoints: list[str] | None = None,
    min_n: int | None = None,
    ebic_gamma: float = 0.5,
    alpha_min_ratio: float = 0.01,
    n_alphas: int = 30,
    edge_threshold: float = 0.05,
) -> tuple[list[dict[str, Any]], pd.DataFrame, dict[str, Any]]:
    """Estimate requested panels and return models, edge rows, and diagnostics.

    Every panel is gated independently, and a figure is renderable only when
    all of its panels pass the gate.  This prevents an underpowered panel from
    being presented as a formal network result.
    """
    selected = frame.loc[frame["scale"] == scale].copy()
    if selected.empty:
        return (
            [],
            pd.DataFrame(),
            {
                "status": "unavailable",
                "reason": f"No rows for scale {scale}",
                "method": METHOD_NAME,
            },
        )
    expected_count = EXPECTED_ITEMS.get(scale)
    items = sorted(int(value) for value in selected["item"].unique())
    expected_items = list(range(1, expected_count + 1)) if expected_count else items
    automatic_required_n = max(30, 5 * len(expected_items))
    required_n = (
        max(automatic_required_n, int(min_n))
        if min_n is not None
        else automatic_required_n
    )
    if required_n < 2:
        raise ValueError("min_n must be at least 2")
    if not 0.0 <= ebic_gamma <= 1.0:
        raise ValueError("ebic_gamma must be between 0 and 1")
    if not 0.0 < alpha_min_ratio <= 1.0:
        raise ValueError("alpha_min_ratio must be in (0, 1]")
    if n_alphas < 2:
        raise ValueError("n_alphas must be at least 2")
    specs = _panel_specs(selected, mode=mode, groups=groups, timepoints=timepoints)
    panels: list[dict[str, Any]] = []
    edge_rows: list[dict[str, Any]] = []
    for spec in specs:
        cell = selected.loc[
            (selected["group"].astype(str) == spec["group"])
            & (selected["timepoint"].astype(str) == spec["timepoint"])
        ]
        pivot = cell.pivot(index="run", columns="item", values="item_score")
        pivot = pivot.reindex(columns=expected_items)
        raw_n = int(pivot.shape[0])
        complete = pivot.dropna(axis=0, how="any")
        n = int(complete.shape[0])
        personas = sorted(
            cell.loc[cell["run"].isin(complete.index), "persona"].astype(str).unique()
        )
        panel: dict[str, Any] = {
            **spec,
            "scale": scale,
            "items": expected_items,
            "n": n,
            "raw_n": raw_n,
            "n_personas": len(personas),
            "personas": personas,
            "required_n": required_n,
            "method": METHOD_NAME,
            "status": "estimable",
        }
        reasons: list[str] = []
        if items != expected_items:
            reasons.append(
                f"incomplete item set: expected {expected_items}, found {items}"
            )
        if n < required_n:
            reasons.append(
                f"n={n} is below the pre-specified rendering gate n>={required_n}"
            )
        if n and complete.nunique(dropna=False).min() <= 1:
            constant_items = [
                int(column)
                for column in complete.columns
                if complete[column].nunique() <= 1
            ]
            reasons.append(f"zero-variance items: {constant_items}")
        if reasons:
            panel["status"] = "insufficient"
            panel["reason"] = "; ".join(reasons)
            panels.append(panel)
            continue
        try:
            partial, fit = _fit_ebic_glasso(
                complete.to_numpy(dtype=float),
                gamma=ebic_gamma,
                alpha_min_ratio=alpha_min_ratio,
                n_alphas=n_alphas,
            )
        except ValueError as exc:
            panel["status"] = "estimation_failed"
            panel["reason"] = str(exc)
            panels.append(panel)
            continue
        panel.update(fit)
        panel["partial_correlation"] = partial.tolist()
        panel["plotted_edges"] = int(
            np.count_nonzero(np.triu(np.abs(partial) >= edge_threshold, k=1))
        )
        panels.append(panel)
        for left_index, left_item in enumerate(expected_items):
            for right_index in range(left_index + 1, len(expected_items)):
                value = float(partial[left_index, right_index])
                edge_rows.append(
                    {
                        "figure_key": spec["figure_key"],
                        "scale": scale,
                        "group": spec["group"],
                        "timepoint": spec["timepoint"],
                        "n": n,
                        "item_a": left_item,
                        "item_b": expected_items[right_index],
                        "partial_correlation": value,
                        "sign": (
                            "positive"
                            if value > 0
                            else "negative" if value < 0 else "zero"
                        ),
                        "plotted": abs(value) >= edge_threshold,
                        "selected_alpha": fit["selected_alpha"],
                        "method": METHOD_NAME,
                    }
                )
    figure_status: dict[str, dict[str, Any]] = {}
    for figure_key in dict.fromkeys(panel["figure_key"] for panel in panels):
        figure_panels = [panel for panel in panels if panel["figure_key"] == figure_key]
        renderable = bool(figure_panels) and all(
            panel["status"] == "estimable" for panel in figure_panels
        )
        figure_status[figure_key] = {
            "status": "renderable" if renderable else "insufficient",
            "panel_count": len(figure_panels),
            "reasons": [
                panel.get("reason") for panel in figure_panels if panel.get("reason")
            ],
        }
    diagnostics = {
        "status": (
            "renderable"
            if figure_status
            and all(v["status"] == "renderable" for v in figure_status.values())
            else "insufficient"
        ),
        "scale": scale,
        "mode": mode,
        "method": METHOD_NAME,
        "inference_unit": "independent outer simulation run; frozen-snapshot repeats pre-averaged",
        "item_type": "ordinal-derived bounded item means (0-3)",
        "raw_correlations_used_as_edges": False,
        "required_n_per_panel": required_n,
        "required_n_rule": "max(user-specified min_n, 30, 5 * node_count); the CLI cannot lower the automatic gate",
        "ebic_gamma": ebic_gamma,
        "alpha_min_ratio": alpha_min_ratio,
        "alpha_candidates": n_alphas,
        "edge_plot_threshold": edge_threshold,
        "group_node_included": False,
        "figure_status": figure_status,
        "missing_conditions": [
            reason for value in figure_status.values() for reason in value["reasons"]
        ],
        "panels": [
            {key: value for key, value in panel.items() if key != "partial_correlation"}
            for panel in panels
        ],
        "recommendation": (
            "Do not use as a formal result; collect more independent runs/personas per group-time cell."
            if any(value["status"] != "renderable" for value in figure_status.values())
            else "Eligible for exploratory/supplementary rendering; bootstrap edge-stability analysis is still recommended."
        ),
    }
    return panels, pd.DataFrame(edge_rows, columns=EDGE_COLUMNS), diagnostics
