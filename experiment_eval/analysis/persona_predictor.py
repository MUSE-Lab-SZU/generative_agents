"""Feasibility audit and model statistics for the persona/SHAP figure."""

from __future__ import annotations

import json
import math
import re
from importlib.util import find_spec
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from ..data.longitudinal import ordered_timepoints


FEATURE_PATTERN = re.compile(
    r"^(baseline|persona|behavior)(?:__|_)(.+)$", re.IGNORECASE
)


def read_feature_data(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        frame = pd.read_csv(path)
    elif suffix in {".jsonl", ".ndjson"}:
        frame = pd.read_json(path, lines=True)
    elif suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            payload = payload.get("rows")
        if not isinstance(payload, list):
            raise ValueError(
                "Feature JSON must be a row list or {'rows': [...]} object"
            )
        frame = pd.DataFrame(payload)
    else:
        raise ValueError(
            f"Unsupported feature format {suffix!r}; use CSV, JSON, or JSONL"
        )
    aliases = {
        "stable_id": "run",
        "outer_run_id": "run",
        "sample_id": "run",
        "subject_id": "run",
    }
    for alias, canonical in aliases.items():
        if canonical not in frame.columns and alias in frame.columns:
            frame = frame.rename(columns={alias: canonical})
    if "run" not in frame:
        raise ValueError("Feature data requires a run column")
    frame["run"] = frame["run"].astype("string").str.strip()
    if frame["run"].isna().any() or (frame["run"] == "").any():
        raise ValueError("Feature data run values must be non-empty")
    if frame["run"].duplicated().any():
        raise ValueError(
            "Feature data must contain exactly one row per independent run"
        )
    return frame


def derive_cbt_relative_benefit(
    long_frame: pd.DataFrame,
    *,
    scale: str,
    cbt_group: str,
    control_group: str,
    endpoint: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Compute CBT gain minus the mean control gain for the same persona.

    Higher target values indicate greater symptom reduction under CBT relative
    to the observed control runs.  This is descriptive, not a causal CATE.
    """
    selected = long_frame.loc[
        (long_frame["scale"] == scale)
        & long_frame["group"].isin([cbt_group, control_group])
    ].copy()
    if selected.empty:
        return pd.DataFrame(), {
            "status": "unavailable",
            "reason": "No scale rows for selected arms",
        }
    timepoints = ordered_timepoints(selected)
    if len(timepoints) < 2:
        return pd.DataFrame(), {
            "status": "unavailable",
            "reason": "Need baseline and endpoint assessments",
        }
    baseline = timepoints[0]
    endpoint = endpoint or timepoints[-1]
    if endpoint not in timepoints:
        return pd.DataFrame(), {
            "status": "unavailable",
            "reason": f"Endpoint {endpoint!r} not found",
            "available_timepoints": timepoints,
        }
    expected_items = int(selected["item"].nunique())
    totals = (
        selected.loc[selected["timepoint"].isin([baseline, endpoint])]
        .groupby(["run", "persona", "group", "timepoint"], observed=True)
        .agg(total_score=("item_score", "sum"), item_count=("item", "nunique"))
        .reset_index()
    )
    totals = totals.loc[totals["item_count"] == expected_items]
    wide = totals.pivot(
        index=["run", "persona", "group"], columns="timepoint", values="total_score"
    ).reset_index()
    if baseline not in wide or endpoint not in wide:
        return pd.DataFrame(), {
            "status": "unavailable",
            "reason": "No complete baseline/endpoint totals",
        }
    wide = wide.dropna(subset=[baseline, endpoint]).copy()
    wide["gain"] = wide[baseline] - wide[endpoint]
    controls = (
        wide.loc[wide["group"] == control_group]
        .groupby("persona", observed=True)["gain"]
        .agg(control_gain_mean="mean", matched_control_runs="size")
        .reset_index()
    )
    targets = wide.loc[wide["group"] == cbt_group].merge(
        controls, on="persona", how="left"
    )
    targets["cbt_relative_benefit"] = targets["gain"] - targets["control_gain_mean"]
    targets = targets.dropna(subset=["cbt_relative_benefit"])
    return targets[
        ["run", "persona", "cbt_relative_benefit", "matched_control_runs"]
    ], {
        "status": "available" if not targets.empty else "unavailable",
        "definition": "CBT baseline-to-endpoint symptom reduction minus mean control reduction within persona",
        "scale": scale,
        "baseline": baseline,
        "endpoint": endpoint,
        "matched_cbt_runs": int(len(targets)),
        "matched_personas": (
            int(targets["persona"].nunique()) if not targets.empty else 0
        ),
        "causal_warning": "Descriptive matched contrast; not an identified individual treatment effect",
    }


def _feature_columns(
    frame: pd.DataFrame, target: str
) -> tuple[list[str], dict[str, str], list[str]]:
    columns: list[str] = []
    categories: dict[str, str] = {}
    unsupported: list[str] = []
    for column in frame.columns:
        match = FEATURE_PATTERN.match(str(column))
        if not match:
            continue
        if pd.api.types.is_numeric_dtype(frame[column]):
            columns.append(str(column))
            categories[str(column)] = match.group(1).lower()
        else:
            unsupported.append(str(column))
    return columns, categories, unsupported


def build_persona_shap_analysis(
    long_frame: pd.DataFrame,
    feature_frame: pd.DataFrame | None,
    *,
    scale: str,
    cbt_group: str | None,
    control_group: str | None,
    target_column: str = "cbt_relative_benefit",
    endpoint: str | None = None,
    min_samples: int = 40,
    random_state: int = 17,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    if feature_frame is None:
        run_count = int(long_frame["run"].nunique())
        missing = [
            "feature data with one row per run and baseline/persona/behavior columns"
        ]
        if run_count < min_samples:
            missing.append(
                f"at least {min_samples} independent runs before applying the feature-count rule (found {run_count})"
            )
        if find_spec("shap") is None:
            missing.append("optional Python package 'shap'")
        return None, {
            "status": "unavailable",
            "independent_runs_in_long_data": run_count,
            "personas_in_long_data": int(long_frame["persona"].nunique()),
            "missing_conditions": missing,
        }
    targets_audit: dict[str, Any]
    if target_column in feature_frame.columns:
        model_frame = feature_frame.loc[
            feature_frame["run"].isin(long_frame["run"].astype(str))
        ].copy()
        targets_audit = {
            "status": "available",
            "definition": f"Explicit feature-data target: {target_column}",
        }
    else:
        if not cbt_group or not control_group:
            return None, {
                "status": "unavailable",
                "missing_conditions": [
                    f"target column {target_column!r}, or both --cbt-group and --control-group to derive it"
                ],
            }
        targets, targets_audit = derive_cbt_relative_benefit(
            long_frame,
            scale=scale,
            cbt_group=cbt_group,
            control_group=control_group,
            endpoint=endpoint,
        )
        model_frame = feature_frame.merge(
            targets[["run", "cbt_relative_benefit"]], on="run", how="inner"
        )
        if (
            target_column != "cbt_relative_benefit"
            and "cbt_relative_benefit" in model_frame
        ):
            model_frame = model_frame.rename(
                columns={"cbt_relative_benefit": target_column}
            )
    candidate_features, categories, unsupported = _feature_columns(
        model_frame, target_column
    )
    missing_conditions: list[str] = []
    clean = (
        model_frame.dropna(subset=[target_column]).copy()
        if target_column in model_frame
        else model_frame.iloc[0:0]
    )
    features = [
        column
        for column in candidate_features
        if clean[column].notna().mean() >= 0.8
        and clean[column].nunique(dropna=True) >= 2
    ]
    if not features:
        missing_conditions.append(
            "at least one varying numeric baseline__/persona__/behavior__ feature with ≤20% missing values"
        )
    required_n = max(int(min_samples), 5 * len(features))
    if len(clean) < required_n:
        missing_conditions.append(
            f"at least {required_n} usable runs (found {len(clean)}; rule=max(min_samples, 5×features))"
        )
    if target_column not in clean or clean[target_column].nunique() < 2:
        missing_conditions.append("a non-constant CBT relative-benefit target")
    if find_spec("shap") is None:
        missing_conditions.append("optional Python package 'shap'")
    audit = {
        "status": "unavailable" if missing_conditions else "available",
        "usable_runs": int(len(clean)),
        "feature_count": len(features),
        "candidate_feature_count": len(candidate_features),
        "required_runs": required_n,
        "feature_categories": {
            category: sum(categories.get(feature) == category for feature in features)
            for category in sorted({categories.get(feature) for feature in features})
        },
        "dropped_features": sorted(set(candidate_features) - set(features)),
        "unsupported_nonnumeric_features": unsupported,
        "target": targets_audit,
        "missing_conditions": missing_conditions,
    }
    if missing_conditions:
        return None, audit

    from sklearn.ensemble import RandomForestRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.model_selection import KFold, cross_val_score
    import shap

    x_raw = clean[features].astype(float)
    retained = features
    imputer = SimpleImputer(strategy="median")
    x = imputer.fit_transform(x_raw[retained])
    y = clean[target_column].astype(float).to_numpy()
    model = RandomForestRegressor(
        n_estimators=500,
        min_samples_leaf=max(2, len(clean) // 20),
        max_features="sqrt",
        random_state=random_state,
        n_jobs=-1,
    )
    folds = min(5, len(clean))
    cv = KFold(n_splits=folds, shuffle=True, random_state=random_state)
    cv_r2 = (
        cross_val_score(model, x, y, cv=cv, scoring="r2")
        if len(clean) >= 5
        else np.asarray([])
    )
    model.fit(x, y)
    explanation = shap.TreeExplainer(model)(x)
    shap_values = np.asarray(explanation.values, dtype=float)
    mean_abs = np.mean(np.abs(shap_values), axis=0)
    total = float(np.sum(mean_abs))
    feature_rows = []
    for index, feature in enumerate(retained):
        correlation = spearmanr(x[:, index], shap_values[:, index]).statistic
        match = FEATURE_PATTERN.match(feature)
        feature_rows.append(
            {
                "feature": feature,
                "feature_label": (
                    match.group(2).replace("__", " ").replace("_", " ")
                    if match
                    else feature
                ),
                "category": categories[feature],
                "direction": (
                    "+" if correlation > 0 else "−" if correlation < 0 else "0"
                ),
                "direction_spearman": (
                    float(correlation) if math.isfinite(float(correlation)) else 0.0
                ),
                "mean_abs_shap": float(mean_abs[index]),
                "proportional_importance_pct": (
                    float(100 * mean_abs[index] / total) if total else 0.0
                ),
                "column_index": index,
            }
        )
    category_order = {"baseline": 0, "persona": 1, "behavior": 2}
    feature_stats = pd.DataFrame(feature_rows)
    feature_stats["_category_order"] = (
        feature_stats["category"].map(category_order).fillna(99)
    )
    feature_stats = (
        feature_stats.sort_values(
            ["_category_order", "mean_abs_shap"], ascending=[True, False]
        )
        .drop(columns=["_category_order"])
        .reset_index(drop=True)
    )
    audit.update(
        {
            "status": "available",
            "retained_features": len(retained),
            "model": "RandomForestRegressor",
            "cross_validated_r2_mean": float(np.mean(cv_r2)) if cv_r2.size else None,
            "cross_validated_r2_values": cv_r2.tolist(),
        }
    )
    return {
        "feature_stats": feature_stats,
        "shap_values": shap_values,
        "feature_values": x,
        "feature_names": retained,
        "target_values": y,
    }, audit
