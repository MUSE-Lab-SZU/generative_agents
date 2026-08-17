"""Descriptive item-trajectory statistics from canonical long data."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import t

from ..data.longitudinal import ordered_timepoints


EXPECTED_ITEMS = {"PHQ-9": 9, "BDI-II": 21}


def build_symptom_trajectory_summary(
    frame: pd.DataFrame, scale: str
) -> tuple[pd.DataFrame, dict[str, Any]]:
    selected = frame.loc[frame["scale"] == scale].copy()
    if selected.empty:
        return pd.DataFrame(), {
            "status": "unavailable",
            "reason": f"No rows for scale {scale}",
        }
    timepoints = ordered_timepoints(selected)
    selected["time_order"] = selected["timepoint"].map(
        {value: index for index, value in enumerate(timepoints)}
    )
    rows: list[dict[str, Any]] = []
    for (group, item, timepoint, time_order), part in selected.groupby(
        ["group", "item", "timepoint", "time_order"], observed=True, sort=False
    ):
        values = part["item_score"].astype(float).to_numpy()
        n = len(values)
        center = float(np.mean(values))
        sd = float(np.std(values, ddof=1)) if n > 1 else math.nan
        se = sd / math.sqrt(n) if n > 1 else math.nan
        critical = float(t.ppf(0.975, n - 1)) if n > 1 else math.nan
        rows.append(
            {
                "scale": scale,
                "group": str(group),
                "item": int(item),
                "timepoint": str(timepoint),
                "time_order": int(time_order),
                "n_runs": n,
                "mean": center,
                "sd": sd,
                "se": se,
                "ci95_lower": center - critical * se if n > 1 else math.nan,
                "ci95_upper": center + critical * se if n > 1 else math.nan,
            }
        )
    summary = (
        pd.DataFrame(rows)
        .sort_values(["item", "group", "time_order"])
        .reset_index(drop=True)
    )
    found_items = sorted(int(value) for value in selected["item"].unique())
    expected_count = EXPECTED_ITEMS.get(scale)
    complete_item_set = expected_count is None or found_items == list(
        range(1, expected_count + 1)
    )
    minimum_n = int(summary["n_runs"].min())
    status = (
        "available"
        if complete_item_set and minimum_n >= 2
        else "available_with_warnings"
    )
    warnings: list[str] = []
    if not complete_item_set:
        warnings.append(f"Expected items 1–{expected_count}; found {found_items}")
    if minimum_n < 2:
        warnings.append(
            "At least one group × item × timepoint cell has fewer than 2 independent runs; CI is unavailable"
        )
    elif minimum_n < 5:
        warnings.append(
            "Some cells have fewer than 5 independent runs; t intervals are very unstable"
        )
    return summary, {
        "status": status,
        "scale": scale,
        "items": found_items,
        "timepoints": timepoints,
        "groups": sorted(selected["group"].astype(str).unique()),
        "minimum_runs_per_cell": minimum_n,
        "warnings": warnings,
    }
