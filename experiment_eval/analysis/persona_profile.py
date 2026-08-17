"""Run-level change aggregation and explicit Z-score definitions for persona profiles."""

from __future__ import annotations

import math
import re
from statistics import mean, stdev
from typing import Any

import pandas as pd

from ..data.longitudinal import timepoint_sort_key
from ..statistics import mean_ci95


ZSCORE_METHODS = {"relative-profile", "common-reference", "standardized-change"}
BASELINE_PRIORITY = ("baseline", "t0", "pre")
POST_PRIORITY = ("post", "endpoint", "now")


def _natural_key(value: str) -> tuple[Any, ...]:
    return tuple(
        int(part) if part.isdigit() else part.casefold()
        for part in re.split(r"(\d+)", str(value))
    )


def _ordered_values(
    available: list[str], requested: list[str] | None, label: str
) -> list[str]:
    unique = sorted(set(available), key=_natural_key)
    if not requested:
        return unique
    if len(requested) != len(set(requested)):
        raise ValueError(f"{label} order contains duplicate values")
    missing = [value for value in requested if value not in unique]
    if missing:
        raise ValueError(
            f"{label} order contains values absent from data: {', '.join(missing)}"
        )
    return list(requested)


def _select_timepoint(
    values: list[str], explicit: str | None, *, baseline: bool
) -> str | None:
    if not values:
        return None
    by_lower = {value.casefold(): value for value in values}
    if explicit:
        selected = by_lower.get(explicit.casefold())
        if selected is None:
            return None
        return selected
    priorities = BASELINE_PRIORITY if baseline else POST_PRIORITY
    for candidate in priorities:
        if candidate in by_lower:
            return by_lower[candidate]
    ordered = sorted(values, key=timepoint_sort_key)
    return ordered[0] if baseline else ordered[-1]


def _select_post_timepoint(
    panel_frame: pd.DataFrame,
    values: list[str],
    baseline: str | None,
    explicit: str | None,
) -> tuple[str | None, str]:
    if explicit:
        return _select_timepoint(values, explicit, baseline=False), "explicit"
    by_lower = {value.casefold(): value for value in values}
    for candidate in POST_PRIORITY:
        if candidate in by_lower:
            return by_lower[candidate], f"semantic_priority:{candidate}"
    candidates = [value for value in values if value != baseline]
    if not candidates or baseline is None:
        return None, "unavailable"
    keys = ["run_id", "persona_id", "group", "setting", "scale"]
    timepoint_sets = [
        set(part["timepoint"].astype(str))
        for _, part in panel_frame.groupby(keys, observed=True, sort=False)
    ]
    coverage = {
        candidate: sum(
            baseline in available and candidate in available
            for available in timepoint_sets
        )
        for candidate in candidates
    }
    selected = max(
        candidates, key=lambda value: (coverage[value], timepoint_sort_key(value))
    )
    return selected, f"maximum_paired_run_coverage:{coverage[selected]}"


def _panel_column(frame: pd.DataFrame, panel_by: str) -> pd.Series:
    if panel_by == "scale":
        return frame["scale"].astype(str)
    if panel_by == "setting":
        return frame["setting"].astype(str)
    raise ValueError(f"Unsupported persona-profile panel definition: {panel_by}")


def _extract_changes(
    frame: pd.DataFrame,
    *,
    panel_by: str,
    baseline_timepoint: str | None,
    post_timepoint: str | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    work = frame.copy()
    work["panel"] = _panel_column(work, panel_by)
    if panel_by == "setting":
        scales_per_panel = work.groupby("panel", observed=True)["scale"].nunique()
        ambiguous = scales_per_panel[scales_per_panel != 1]
        if not ambiguous.empty:
            names = ", ".join(str(value) for value in ambiguous.index)
            raise ValueError(
                "--profile-panel-by setting requires exactly one scale within each setting; "
                f"ambiguous settings: {names}"
            )

    rows: list[dict[str, Any]] = []
    panel_audit: dict[str, Any] = {}
    for panel, panel_frame in work.groupby("panel", observed=True, sort=False):
        timepoints = sorted(
            panel_frame["timepoint"].astype(str).unique(), key=timepoint_sort_key
        )
        baseline = _select_timepoint(timepoints, baseline_timepoint, baseline=True)
        post, post_selection = _select_post_timepoint(
            panel_frame, timepoints, baseline, post_timepoint
        )
        missing_pairs: list[dict[str, str]] = []
        keys = ["run_id", "persona_id", "group", "setting", "scale"]
        for key, run_frame in panel_frame.groupby(keys, observed=True, sort=False):
            lookup = dict(
                zip(
                    run_frame["timepoint"].astype(str), run_frame["score"].astype(float)
                )
            )
            if (
                baseline is None
                or post is None
                or baseline not in lookup
                or post not in lookup
            ):
                missing_pairs.append(
                    {
                        "run_id": str(key[0]),
                        "persona_id": str(key[1]),
                        "group": str(key[2]),
                    }
                )
                continue
            baseline_score = float(lookup[baseline])
            post_score = float(lookup[post])
            rows.append(
                {
                    "run_id": str(key[0]),
                    "persona_id": str(key[1]),
                    "group": str(key[2]),
                    "setting": str(key[3]),
                    "scale": str(key[4]),
                    "panel": str(panel),
                    "baseline_timepoint": baseline,
                    "post_timepoint": post,
                    "baseline_score": baseline_score,
                    "post_score": post_score,
                    "change": post_score - baseline_score,
                }
            )
        panel_audit[str(panel)] = {
            "scale": sorted(panel_frame["scale"].astype(str).unique()),
            "baseline_timepoint": baseline,
            "post_timepoint": post,
            "post_selection": post_selection,
            "available_timepoints": timepoints,
            "complete_change_rows": sum(row["panel"] == str(panel) for row in rows),
            "missing_pair_count": len(missing_pairs),
            "missing_pair_examples": missing_pairs[:20],
        }
    return pd.DataFrame(rows), panel_audit


def _raw_cell_summary(changes: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    keys = ["panel", "scale", "persona_id", "group"]
    for key, cell in changes.groupby(keys, observed=True, sort=False):
        values = [float(value) for value in cell["change"]]
        interval = mean_ci95(values)
        sd = stdev(values) if len(values) > 1 else None
        rows.append(
            {
                "panel": str(key[0]),
                "scale": str(key[1]),
                "persona_id": str(key[2]),
                "group": str(key[3]),
                "n": len(values),
                "mean_change": mean(values),
                "sd_change": sd,
                "se_change": sd / math.sqrt(len(values)) if sd is not None else None,
                "ci95_lower_change": interval["lower"],
                "ci95_upper_change": interval["upper"],
            }
        )
    return pd.DataFrame(rows)


def _references(
    changes: pd.DataFrame, summary: pd.DataFrame, method: str
) -> tuple[dict[tuple[str, str | None], tuple[float, float]], list[dict[str, Any]]]:
    lookup: dict[tuple[str, str | None], tuple[float, float]] = {}
    rows: list[dict[str, Any]] = []
    if method == "relative-profile":
        for (panel, group), part in summary.groupby(
            ["panel", "group"], observed=True, sort=False
        ):
            values = [float(value) for value in part["mean_change"]]
            center = mean(values) if values else math.nan
            spread = stdev(values) if len(values) > 1 else math.nan
            lookup[(str(panel), str(group))] = (center, spread)
            rows.append(
                {
                    "panel": str(panel),
                    "group": str(group),
                    "method": method,
                    "reference_center": center,
                    "reference_sd": spread,
                    "reference_n": len(values),
                    "reference_unit": "persona_by_group_cell_means",
                }
            )
    else:
        for panel, part in changes.groupby("panel", observed=True, sort=False):
            if method == "common-reference":
                values = [float(value) for value in part["change"]]
                center = mean(values) if values else math.nan
                spread = stdev(values) if len(values) > 1 else math.nan
                unit = "complete_independent_run_changes"
            else:
                values = [float(value) for value in part["baseline_score"]]
                center = 0.0
                spread = stdev(values) if len(values) > 1 else math.nan
                unit = "complete_independent_run_baseline_scores"
            lookup[(str(panel), None)] = (center, spread)
            rows.append(
                {
                    "panel": str(panel),
                    "group": None,
                    "method": method,
                    "reference_center": center,
                    "reference_sd": spread,
                    "reference_n": len(values),
                    "reference_unit": unit,
                }
            )
    return lookup, rows


def build_persona_profile_analysis(
    frame: pd.DataFrame,
    *,
    method: str = "relative-profile",
    scales: list[str] | None = None,
    persona_order: list[str] | None = None,
    group_order: list[str] | None = None,
    panel_by: str = "scale",
    setting_order: list[str] | None = None,
    baseline_timepoint: str | None = None,
    post_timepoint: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Pair baseline/post within run, aggregate cells, then apply one declared Z transform."""
    if method not in ZSCORE_METHODS:
        raise ValueError(f"Unsupported persona-profile Z-score method: {method}")
    selected_scales = _ordered_values(
        frame["scale"].astype(str).tolist(), scales, "scale"
    )
    work = frame.loc[frame["scale"].astype(str).isin(selected_scales)].copy()
    personas = _ordered_values(
        work["persona_id"].astype(str).tolist(), persona_order, "persona"
    )
    groups = _ordered_values(work["group"].astype(str).tolist(), group_order, "group")
    work = work.loc[
        work["persona_id"].astype(str).isin(personas)
        & work["group"].astype(str).isin(groups)
    ]
    if panel_by == "setting":
        settings = _ordered_values(
            work["setting"].astype(str).tolist(), setting_order, "setting"
        )
        work = work.loc[work["setting"].astype(str).isin(settings)]
    else:
        settings = sorted(work["setting"].astype(str).unique(), key=_natural_key)

    changes, panel_audit = _extract_changes(
        work,
        panel_by=panel_by,
        baseline_timepoint=baseline_timepoint,
        post_timepoint=post_timepoint,
    )
    if changes.empty:
        summary = pd.DataFrame()
        reference_frame = pd.DataFrame()
    else:
        summary = _raw_cell_summary(changes)
        reference_lookup, reference_rows = _references(changes, summary, method)
        reference_frame = pd.DataFrame(reference_rows)
        z_columns = {
            "mean_change": "z_mean",
            "se_change": "z_se",
            "ci95_lower_change": "z_ci95_lower",
            "ci95_upper_change": "z_ci95_upper",
        }
        for source, target in z_columns.items():
            transformed = []
            for row in summary.to_dict("records"):
                key = (
                    row["panel"],
                    row["group"] if method == "relative-profile" else None,
                )
                center, spread = reference_lookup[key]
                value = row[source]
                if value is None or not math.isfinite(spread) or spread <= 0:
                    transformed.append(None)
                elif source == "mean_change" or source.startswith("ci95_"):
                    transformed.append((float(value) - center) / spread)
                else:
                    transformed.append(float(value) / spread)
            summary[target] = transformed

        change_z = []
        for row in changes.to_dict("records"):
            key = (row["panel"], row["group"] if method == "relative-profile" else None)
            center, spread = reference_lookup[key]
            change_z.append(
                (float(row["change"]) - center) / spread
                if math.isfinite(spread) and spread > 0
                else None
            )
        changes["z_change"] = change_z

    panel_order = selected_scales if panel_by == "scale" else settings
    warnings: list[str] = []
    if len(personas) < 2:
        warnings.append("At least two personas are required for a comparative profile.")
    if len(groups) < 2:
        warnings.append(
            "Only one condition is present; between-condition profile comparison is unavailable."
        )
    if not summary.empty and summary["z_mean"].isna().any():
        warnings.append(
            "Some Z-scores are unavailable because their declared reference SD is zero or undefined."
        )
    expected_cells = len(panel_order) * len(personas) * len(groups)
    observed_cells = 0 if summary.empty else len(summary)
    if observed_cells < expected_cells:
        warnings.append(
            f"Missing persona × group cells: {expected_cells - observed_cells} of {expected_cells}."
        )
    status = (
        "unavailable"
        if len(personas) < 2
        or summary.empty
        or summary.get("z_mean", pd.Series(dtype=float)).notna().sum() == 0
        else ("available_with_warnings" if warnings else "available")
    )
    audit = {
        "status": status,
        "zscore_method": method,
        "panel_by": panel_by,
        "panel_order": panel_order,
        "persona_order": personas,
        "group_order": groups,
        "setting_order": settings,
        "independent_run_count": (
            int(changes["run_id"].nunique()) if not changes.empty else 0
        ),
        "complete_change_rows": int(len(changes)),
        "observed_cells": observed_cells,
        "expected_cells": expected_cells,
        "panels": panel_audit,
        "warnings": warnings,
        "change_definition": "post_score - baseline_score; negative values mean symptom reduction for PHQ-9/BDI-II",
        "inference_unit": "independent outer run; frozen-snapshot measurement repeats must be pre-aggregated",
    }
    return changes, summary, reference_frame, audit
