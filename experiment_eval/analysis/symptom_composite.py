"""Two-scale total trajectories and run-level item-change heatmap data."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import t

from ..data.longitudinal import ordered_timepoints, timepoint_sort_key
from ..schema import group_sort_key, scale_sort_key


def _mean_ci(values: pd.Series) -> tuple[float, float, float, int]:
    array = values.astype(float).to_numpy()
    n = len(array)
    center = float(np.mean(array))
    if n < 2:
        return center, math.nan, math.nan, n
    se = float(np.std(array, ddof=1)) / math.sqrt(n)
    critical = float(t.ppf(0.975, n - 1))
    return center, center - critical * se, center + critical * se, n


def _intervention_endpoint(timepoints: list[str]) -> str:
    intervention = [
        value for value in timepoints if timepoint_sort_key(value)[0] == 1
    ]
    return intervention[-1] if intervention else timepoints[-1]


def build_symptom_composite_analysis(
    frame: pd.DataFrame,
    *,
    scales: list[str],
    groups: list[str],
    baseline: str | None = None,
    endpoint: str | None = None,
    common_personas_only: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Build panel-A total trajectories and panel-B/C item changes.

    The heatmap column is one independent outer run. Frozen-snapshot
    measurement repeats must already be averaged in the canonical long input.
    """
    selected_scales = sorted(set(scales), key=scale_sort_key)
    selected_groups = sorted(set(groups), key=group_sort_key)
    if len(selected_scales) != 2:
        raise ValueError("Symptom composite requires exactly two scales")
    if len(selected_groups) != 2:
        raise ValueError("Symptom composite requires exactly two groups")

    selected = frame.loc[
        frame["scale"].isin(selected_scales)
        & frame["group"].isin(selected_groups)
    ].copy()
    if selected.empty:
        return pd.DataFrame(), pd.DataFrame(), {
            "status": "unavailable",
            "reason": "No rows for the selected scales and groups",
        }

    personas_by_group = {
        group: set(part["persona"].astype(str))
        for group, part in selected.groupby("group", observed=True)
    }
    common_personas = set.intersection(
        *(personas_by_group.get(group, set()) for group in selected_groups)
    )
    if common_personas_only:
        if not common_personas:
            return pd.DataFrame(), pd.DataFrame(), {
                "status": "unavailable",
                "reason": "Selected groups have no common persona",
                "personas_by_group": {
                    key: sorted(value) for key, value in personas_by_group.items()
                },
            }
        selected = selected.loc[
            selected["persona"].astype(str).isin(common_personas)
        ].copy()

    available_by_cell = {
        (str(scale), str(group)): set(part["timepoint"].astype(str))
        for (scale, group), part in selected.groupby(
            ["scale", "group"], observed=True
        )
    }
    common_timepoints = set.intersection(
        *(
            available_by_cell.get((scale, group), set())
            for scale in selected_scales
            for group in selected_groups
        )
    )
    ordered = [
        value for value in ordered_timepoints(selected) if value in common_timepoints
    ]
    if len(ordered) < 2:
        return pd.DataFrame(), pd.DataFrame(), {
            "status": "unavailable",
            "reason": "Selected scale/group cells do not share two timepoints",
        }
    resolved_baseline = baseline or ordered[0]
    resolved_endpoint = endpoint or _intervention_endpoint(ordered)
    missing_timepoints = [
        value
        for value in (resolved_baseline, resolved_endpoint)
        if value not in common_timepoints
    ]
    if missing_timepoints:
        raise ValueError(
            f"Composite baseline/endpoint not shared by all cells: {missing_timepoints}"
        )
    endpoint_order = ordered.index(resolved_endpoint)
    trajectory_timepoints = ordered[: endpoint_order + 1]
    selected = selected.loc[
        selected["timepoint"].astype(str).isin(trajectory_timepoints)
    ].copy()

    run_totals = (
        selected.groupby(
            ["run", "persona", "group", "scale", "timepoint"], observed=True
        )["item_score"]
        .sum()
        .reset_index(name="total_score")
    )
    time_order = {value: index for index, value in enumerate(trajectory_timepoints)}
    trajectory_rows: list[dict[str, Any]] = []
    for (scale, group, timepoint), part in run_totals.groupby(
        ["scale", "group", "timepoint"], observed=True, sort=False
    ):
        center, lower, upper, n = _mean_ci(part["total_score"])
        trajectory_rows.append(
            {
                "scale": str(scale),
                "group": str(group),
                "timepoint": str(timepoint),
                "time_order": time_order[str(timepoint)],
                "mean": center,
                "ci95_lower": lower,
                "ci95_upper": upper,
                "n_outer_runs": n,
            }
        )
    trajectory = pd.DataFrame(trajectory_rows).sort_values(
        ["scale", "group", "time_order"]
    )

    item_key = ["run", "persona", "group", "scale", "item"]
    baseline_rows = selected.loc[
        selected["timepoint"].astype(str) == resolved_baseline,
        [*item_key, "item_score"],
    ].rename(columns={"item_score": "baseline_score"})
    endpoint_rows = selected.loc[
        selected["timepoint"].astype(str) == resolved_endpoint,
        [*item_key, "item_score"],
    ].rename(columns={"item_score": "endpoint_score"})
    changes = baseline_rows.merge(endpoint_rows, on=item_key, how="inner")
    changes["score_change"] = changes["endpoint_score"] - changes["baseline_score"]
    run_order = (
        changes[["run", "group"]]
        .drop_duplicates()
        .sort_values(["group", "run"], key=lambda column: column.map(
            group_sort_key if column.name == "group" else str
        ))
        .reset_index(drop=True)
    )
    run_order["within_group_index"] = (
        run_order.groupby("group", observed=True).cumcount() + 1
    )
    run_order["run_order"] = range(len(run_order))
    run_order["run_label"] = run_order.apply(
        lambda row: f"{row['group']}-{int(row['within_group_index']):02d}", axis=1
    )
    changes = changes.merge(run_order, on=["run", "group"], how="left")
    changes = changes.sort_values(["scale", "item", "run_order"]).reset_index(
        drop=True
    )

    run_counts = {
        str(group): int(part["run"].nunique())
        for group, part in changes.groupby("group", observed=True)
    }
    warnings: list[str] = []
    if min(run_counts.values(), default=0) < 5:
        warnings.append(
            "At least one group has fewer than 5 independent outer runs; trajectory CIs are unstable"
        )
    return trajectory, changes, {
        "status": "available" if not warnings else "available_with_warnings",
        "scales": selected_scales,
        "groups": selected_groups,
        "baseline": resolved_baseline,
        "endpoint": resolved_endpoint,
        "trajectory_timepoints": trajectory_timepoints,
        "common_personas_only": common_personas_only,
        "included_personas": sorted(common_personas) if common_personas_only else sorted(
            selected["persona"].astype(str).unique()
        ),
        "runs_by_group": run_counts,
        "heatmap_unit": "independent outer run",
        "heatmap_value": "endpoint item score minus baseline item score",
        "warnings": warnings,
    }
