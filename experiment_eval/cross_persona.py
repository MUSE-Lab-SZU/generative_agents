"""Cross-persona descriptive analyses built on the unified experiment schema.

The functions in this module keep the independent outer simulation run as the
analysis unit. The K repeated scale generations are used only for frozen-
snapshot means and the measurement-noise component.
"""

from __future__ import annotations

import math
from collections import defaultdict
from statistics import mean
from typing import Any, Iterable

import numpy as np
from scipy.stats import t

from .schema import ExperimentRecord, group_sort_key, scale_sort_key
from .statistics import (
    baseline_adjusted_endpoint_contrasts,
    baseline_label_for_record,
    endpoint_label_for_record,
    group_endpoint_contrasts,
    mean_ci95,
    score_at,
    trajectory_metrics,
)


def _finite(values: Iterable[Any]) -> list[float]:
    clean: list[float] = []
    for value in values:
        if value is None:
            continue
        number = float(value)
        if math.isfinite(number):
            clean.append(number)
    return clean


def group_time_persona_summary(
    records: list[ExperimentRecord], labels: list[str], scales: list[str]
) -> list[dict[str, Any]]:
    """Return outer-run marginal means for every persona × group × time cell."""
    rows: list[dict[str, Any]] = []
    personas = sorted({record.kbd for record in records}, key=group_sort_key)
    groups = sorted({record.group for record in records}, key=group_sort_key)
    for persona in personas:
        for group in groups:
            cell = [record for record in records if record.kbd == persona and record.group == group]
            if not cell:
                continue
            for scale in sorted(scales, key=scale_sort_key):
                for label in labels:
                    values = _finite(score_at(record, scale, label) for record in cell)
                    if not values:
                        continue
                    interval = mean_ci95(values)
                    rows.append(
                        {
                            "persona": persona,
                            "group": group,
                            "scale": scale,
                            "timepoint": label,
                            "n_outer_runs": len(values),
                            "mean_score": mean(values),
                            "sample_sd": float(np.std(values, ddof=1)) if len(values) > 1 else None,
                            "ci95_lower": interval.get("lower"),
                            "ci95_upper": interval.get("upper"),
                            "ci_method": interval.get("method"),
                            "analysis_unit": "independent outer simulation run",
                        }
                    )
    return rows


def persona_group_summary(
    records: list[ExperimentRecord],
    labels: list[str],
    scales: list[str],
    process_rows: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Summarize outcome and available process metrics by persona × group."""
    process_by_id = {row["stable_id"]: row for row in (process_rows or [])}
    rows: list[dict[str, Any]] = []
    cells: dict[tuple[str, str], list[ExperimentRecord]] = defaultdict(list)
    for record in records:
        cells[(record.kbd, record.group)].append(record)

    for (persona, group), cell in sorted(
        cells.items(), key=lambda item: (group_sort_key(item[0][0]), group_sort_key(item[0][1]))
    ):
        row: dict[str, Any] = {
            "persona": persona,
            "group": group,
            "n_outer_runs": len(cell),
        }
        direction_flags: list[bool] = []
        for record in cell:
            phq = trajectory_metrics(record, labels, "PHQ-9").get("endpoint_change")
            bdi = trajectory_metrics(record, labels, "BDI-II").get("endpoint_change")
            if phq is None or bdi is None:
                continue
            direction_flags.append((phq == 0 and bdi == 0) or (phq * bdi > 0))
        row["phq_bdi_direction_agreement_rate"] = (
            mean(direction_flags) if direction_flags else None
        )
        row["n_direction_pairs"] = len(direction_flags)

        for scale in sorted(scales, key=scale_sort_key):
            prefix = "phq9" if scale == "PHQ-9" else "bdi2" if scale == "BDI-II" else scale.lower()
            deltas = _finite(trajectory_metrics(record, labels, scale).get("endpoint_change") for record in cell)
            interval = mean_ci95(deltas)
            row[f"{prefix}_mean_endpoint_change"] = mean(deltas) if deltas else None
            row[f"{prefix}_endpoint_change_ci95_lower"] = interval.get("lower")
            row[f"{prefix}_endpoint_change_ci95_upper"] = interval.get("upper")

        process = [process_by_id[record.stable_id] for record in cell if record.stable_id in process_by_id]
        available = [entry for entry in process if entry.get("raw_artifacts_available")]
        row["process_raw_available_rate"] = len(available) / len(cell) if cell else None
        for field in (
            "completed_consultation_meetings",
            "total_turns",
            "total_chars",
            "environment_tasks",
        ):
            values = _finite(entry.get(field) for entry in available)
            row[f"mean_{field}"] = mean(values) if values else None
            row[f"n_{field}"] = len(values)
        rows.append(row)
    return rows


def persona_group_contrasts(
    records: list[ExperimentRecord],
    labels: list[str],
    scales: list[str],
    *,
    first_group: str,
    second_group: str,
) -> list[dict[str, Any]]:
    """Return per-persona adjusted differences and Hedges' g with 95% CIs."""
    rows: list[dict[str, Any]] = []
    personas = sorted({record.kbd for record in records}, key=group_sort_key)
    for persona in personas:
        cell = [
            record
            for record in records
            if record.kbd == persona and record.group in {first_group, second_group}
        ]
        adjusted_rows = baseline_adjusted_endpoint_contrasts(cell, labels, scales)
        change_rows = group_endpoint_contrasts(cell, labels, scales)
        for scale in sorted(scales, key=scale_sort_key):
            adjusted = next(
                (
                    row
                    for row in adjusted_rows
                    if row["scale"] == scale
                    and {row["first_group"], row["second_group"]} == {first_group, second_group}
                ),
                None,
            )
            change = next(
                (
                    row
                    for row in change_rows
                    if row["scale"] == scale
                    and {row["first_group"], row["second_group"]} == {first_group, second_group}
                ),
                None,
            )
            if adjusted is None or change is None:
                continue
            orientation = 1.0 if adjusted["first_group"] == first_group else -1.0
            effect = change["hedges_g"]
            effect_orientation = 1.0 if change["first_group"] == first_group else -1.0
            rows.append(
                {
                    "persona": persona,
                    "scale": scale,
                    "contrast": f"{first_group} - {second_group}",
                    "n_first": sum(record.group == first_group for record in cell),
                    "n_second": sum(record.group == second_group for record in cell),
                    "adjusted_endpoint_difference": (
                        orientation * adjusted["estimate"] if adjusted.get("estimate") is not None else None
                    ),
                    "adjusted_ci95_lower": (
                        orientation * adjusted["upper"] if orientation < 0 and adjusted.get("upper") is not None
                        else adjusted.get("lower")
                    ),
                    "adjusted_ci95_upper": (
                        orientation * adjusted["lower"] if orientation < 0 and adjusted.get("lower") is not None
                        else adjusted.get("upper")
                    ),
                    "adjusted_df": adjusted.get("df"),
                    "adjusted_status": adjusted.get("status"),
                    "hedges_g": (
                        effect_orientation * effect["estimate"] if effect.get("estimate") is not None else None
                    ),
                    "hedges_g_ci95_lower": (
                        effect_orientation * effect["upper"]
                        if effect_orientation < 0 and effect.get("upper") is not None
                        else effect.get("lower")
                    ),
                    "hedges_g_ci95_upper": (
                        effect_orientation * effect["lower"]
                        if effect_orientation < 0 and effect.get("lower") is not None
                        else effect.get("upper")
                    ),
                    "hedges_g_ci_method": effect.get("ci_method"),
                    "unadjusted_change_difference": (
                        effect_orientation * change["difference"]["estimate"]
                        if change["difference"].get("estimate") is not None
                        else None
                    ),
                    "analysis_unit": "independent outer simulation run",
                    "inference_scope": (
                        "exploratory persona-specific contrast; baseline-adjusted CI has very low residual df "
                        "when each persona-group cell has only two outer runs"
                    ),
                }
            )
    return rows


def _fixed_persona_adjusted_contrast(
    records: list[ExperimentRecord],
    labels: list[str],
    scale: str,
    first_group: str,
    second_group: str,
) -> dict[str, Any]:
    observations: list[tuple[str, str, float, float]] = []
    for record in records:
        if record.group not in {first_group, second_group}:
            continue
        baseline = score_at(record, scale, baseline_label_for_record(record, labels, scale))
        endpoint = score_at(record, scale, endpoint_label_for_record(record, labels, scale))
        if baseline is not None and endpoint is not None:
            observations.append((record.group, record.kbd, baseline, endpoint))
    personas = sorted({persona for _, persona, _, _ in observations}, key=group_sort_key)
    result: dict[str, Any] = {
        "estimate": None,
        "lower": None,
        "upper": None,
        "df": None,
        "n_outer_runs": len(observations),
        "personas": personas,
        "status": "insufficient_or_singular",
        "method": "endpoint_on_group_baseline_and_persona_fixed_effects",
    }
    if len(personas) < 2:
        return result
    reference_persona = personas[0]
    design = np.asarray(
        [
            [
                1.0,
                1.0 if group == first_group else 0.0,
                baseline,
                *[1.0 if persona == value else 0.0 for value in personas if value != reference_persona],
            ]
            for group, persona, baseline, _ in observations
        ],
        dtype=float,
    )
    outcome = np.asarray([endpoint for _, _, _, endpoint in observations], dtype=float)
    degrees = len(outcome) - design.shape[1]
    if degrees <= 0 or np.linalg.matrix_rank(design) != design.shape[1]:
        return result
    coefficients = np.linalg.lstsq(design, outcome, rcond=None)[0]
    residuals = outcome - design @ coefficients
    residual_variance = float(residuals @ residuals / degrees)
    covariance = residual_variance * np.linalg.inv(design.T @ design)
    standard_error = math.sqrt(max(float(covariance[1, 1]), 0.0))
    estimate = float(coefficients[1])
    half = float(t.ppf(0.975, degrees)) * standard_error
    result.update(
        {
            "estimate": estimate,
            "lower": estimate - half,
            "upper": estimate + half,
            "df": degrees,
            "status": "ok",
        }
    )
    return result


def leave_one_persona_out(
    records: list[ExperimentRecord],
    labels: list[str],
    scales: list[str],
    *,
    first_group: str,
    second_group: str,
) -> list[dict[str, Any]]:
    """Fixed-case leave-one-persona-out sensitivity analysis.

    This checks dependence on the selected persona cases. It is not a
    jackknife estimate for a sampled persona population.
    """
    selected = [record for record in records if record.group in {first_group, second_group}]
    personas = sorted({record.kbd for record in selected}, key=group_sort_key)
    rows: list[dict[str, Any]] = []
    for scale in sorted(scales, key=scale_sort_key):
        for omitted in [None, *personas]:
            subset = [record for record in selected if omitted is None or record.kbd != omitted]
            result = _fixed_persona_adjusted_contrast(
                subset, labels, scale, first_group, second_group
            )
            rows.append(
                {
                    "scale": scale,
                    "contrast": f"{first_group} - {second_group}",
                    "omitted_persona": omitted or "NONE",
                    "included_personas": ",".join(result["personas"]),
                    "n_outer_runs": result["n_outer_runs"],
                    "adjusted_endpoint_difference": result["estimate"],
                    "ci95_lower": result["lower"],
                    "ci95_upper": result["upper"],
                    "df": result["df"],
                    "status": result["status"],
                    "method": result["method"],
                    "inference_scope": "exploratory fixed-case influence analysis",
                }
            )
    return rows


VARIANCE_SOURCE_LABELS = {
    "group_bundle": "Group bundle",
    "persona_within_group": "Persona within group",
    "outer_run_within_cell": "Outer run within persona × group",
    "common_time_profile": "Common time profile",
    "run_by_time_trajectory": "Run × time trajectory",
    "measurement_noise": "K-repeat measurement noise",
}


def hierarchical_variance_partition(
    records: list[ExperimentRecord], labels: list[str], scales: list[str]
) -> list[dict[str, Any]]:
    """Descriptive sums-of-squares partition for the observed balanced hierarchy.

    The hierarchy is group → persona → outer run, with time crossed within run
    and K measurements nested within snapshot. It is an accounting identity,
    not a REML population variance-component model.
    """
    output: list[dict[str, Any]] = []
    for scale in sorted(scales, key=scale_sort_key):
        raw: list[dict[str, Any]] = []
        for record in records:
            for label in labels:
                stats = record.series.get(scale, {}).get(label) or {}
                for repeat_id, value in (stats.get("values_by_repeat") or {}).items():
                    raw.append(
                        {
                            "group": record.group,
                            "persona": record.kbd,
                            "run": record.stable_id,
                            "time": label,
                            "snapshot": f"{record.stable_id}::{label}",
                            "measurement_repeat_id": repeat_id,
                            "value": float(value),
                        }
                    )
        if not raw:
            continue

        def grouped_mean(keys: tuple[str, ...]) -> tuple[dict[tuple[str, ...], float], dict[tuple[str, ...], int]]:
            values: dict[tuple[str, ...], list[float]] = defaultdict(list)
            for item in raw:
                values[tuple(str(item[key]) for key in keys)].append(item["value"])
            return (
                {key: mean(group_values) for key, group_values in values.items()},
                {key: len(group_values) for key, group_values in values.items()},
            )

        grand = mean(item["value"] for item in raw)
        total_ss = sum((item["value"] - grand) ** 2 for item in raw)
        group_mean, group_n = grouped_mean(("group",))
        cell_mean, cell_n = grouped_mean(("group", "persona"))
        run_mean, run_n = grouped_mean(("group", "persona", "run"))
        snapshot_mean, snapshot_n = grouped_mean(("group", "persona", "run", "time"))

        group_ss = sum(
            group_n[key] * (value - grand) ** 2 for key, value in group_mean.items()
        )
        persona_ss = sum(
            cell_n[key] * (value - group_mean[(key[0],)]) ** 2
            for key, value in cell_mean.items()
        )
        run_ss = sum(
            run_n[key] * (value - cell_mean[(key[0], key[1])]) ** 2
            for key, value in run_mean.items()
        )
        measurement_ss = sum(
            (
                item["value"]
                - snapshot_mean[(item["group"], item["persona"], item["run"], item["time"])]
            )
            ** 2
            for item in raw
        )

        snapshot_deviation: dict[tuple[str, str, str, str], float] = {}
        time_values: dict[str, list[tuple[float, int]]] = defaultdict(list)
        for key, value in snapshot_mean.items():
            run_key = key[:3]
            deviation = value - run_mean[run_key]
            snapshot_deviation[key] = deviation
            time_values[key[3]].append((deviation, snapshot_n[key]))
        time_effect = {
            label: sum(value * weight for value, weight in values)
            / sum(weight for _, weight in values)
            for label, values in time_values.items()
        }
        time_ss = sum(
            snapshot_n[key] * time_effect[key[3]] ** 2 for key in snapshot_mean
        )
        run_time_ss = sum(
            snapshot_n[key] * (snapshot_deviation[key] - time_effect[key[3]]) ** 2
            for key in snapshot_mean
        )

        components = {
            "group_bundle": group_ss,
            "persona_within_group": persona_ss,
            "outer_run_within_cell": run_ss,
            "common_time_profile": time_ss,
            "run_by_time_trajectory": run_time_ss,
            "measurement_noise": measurement_ss,
        }
        component_total = sum(components.values())
        closure_error = component_total - total_ss
        k_values = sorted(set(snapshot_n.values()))
        common_k = k_values[0] if len(k_values) == 1 else None
        single_generation_variances = {
            source: sum_squares / len(raw) for source, sum_squares in components.items()
        }
        k_mean_variances = {
            source: (
                variance / common_k
                if source == "measurement_noise" and common_k is not None
                else variance
            )
            for source, variance in single_generation_variances.items()
        }
        k_mean_total = sum(k_mean_variances.values())
        for source, sum_squares in components.items():
            output.append(
                {
                    "scale": scale,
                    "source": source,
                    "source_label": VARIANCE_SOURCE_LABELS[source],
                    "sum_squares": sum_squares,
                    "descriptive_variance": single_generation_variances[source],
                    "proportion_of_observed_variance": (
                        sum_squares / total_ss if total_ss else None
                    ),
                    "k_mean_descriptive_variance": k_mean_variances[source],
                    "proportion_of_k_mean_variance": (
                        k_mean_variances[source] / k_mean_total if k_mean_total else None
                    ),
                    "n_measurements": len(raw),
                    "n_personas": len({item["persona"] for item in raw}),
                    "n_outer_runs": len({item["run"] for item in raw}),
                    "n_snapshots": len({item["snapshot"] for item in raw}),
                    "k_values": ",".join(map(str, k_values)),
                    "closure_error": closure_error,
                    "method": "descriptive hierarchical sums-of-squares partition",
                    "inference_scope": (
                        "exploratory; outer-run component is a stochastic-realization/seed proxy, "
                        "run×time includes trajectory-specific stochasticity, and K-mean measurement "
                        "variance assumes independent repeat generations within each frozen snapshot"
                    ),
                }
            )
    return output
