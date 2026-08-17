"""Pairwise weighted Cohen's kappa for repeated ordinal item ratings.

The rated object is always an exact ``snapshot_id × scale × item`` tuple.
Measurement repeats are raters; they are never treated as independent outer runs.
"""

from __future__ import annotations

import hashlib
import math
import random
from collections import defaultdict
from itertools import combinations, zip_longest
from statistics import mean, median
from typing import Any, Iterable, Sequence

import numpy as np

from .schema import ExperimentRecord, group_sort_key, label_sort_key, scale_sort_key


ITEM_CATEGORIES = (0, 1, 2, 3)
MIN_BOOTSTRAP_CLUSTERS = 5
DEFAULT_BOOTSTRAP_REPLICATES = 300


def _percentile(values: Sequence[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def cohen_weighted_kappa(
    rater_a: Iterable[float | int | None],
    rater_b: Iterable[float | int | None],
    *,
    weights: str = "quadratic",
    categories: Sequence[int] = ITEM_CATEGORIES,
) -> dict[str, Any]:
    """Compute weighted Cohen's kappa with pairwise deletion.

    Agreement weights follow ``1 - distance`` for linear and
    ``1 - distance²`` for quadratic weighting. A zero expected-disagreement
    denominator is reported as undefined rather than changed to perfect agreement.
    """
    if weights not in {"quadratic", "linear"}:
        raise ValueError("weights must be 'quadratic' or 'linear'")
    ordered_categories = tuple(categories)
    if len(ordered_categories) < 2 or len(set(ordered_categories)) != len(ordered_categories):
        raise ValueError("categories must contain at least two unique ordered values")
    category_index = {float(value): index for index, value in enumerate(ordered_categories)}
    pairs: list[tuple[int, int]] = []
    n_possible = 0
    n_missing = 0
    for first, second in zip_longest(rater_a, rater_b, fillvalue=None):
        n_possible += 1
        if first is None or second is None:
            n_missing += 1
            continue
        first_index = category_index.get(float(first))
        second_index = category_index.get(float(second))
        if first_index is None or second_index is None:
            n_missing += 1
            continue
        pairs.append((first_index, second_index))
    n_valid = len(pairs)
    base = {
        "weights": weights,
        "kappa": None,
        "n_possible": n_possible,
        "n_valid": n_valid,
        "n_missing": n_missing,
        "reason": None,
    }
    if n_valid < 2:
        base["reason"] = "fewer_than_two_valid_objects"
        return base

    size = len(ordered_categories)
    first_counts = [0] * size
    second_counts = [0] * size
    observed_disagreement = 0.0
    maximum_distance = size - 1
    for first_index, second_index in pairs:
        first_counts[first_index] += 1
        second_counts[second_index] += 1
        distance = abs(first_index - second_index) / maximum_distance
        observed_disagreement += distance**2 if weights == "quadratic" else distance
    observed_disagreement /= n_valid

    expected_disagreement = 0.0
    for first_index, first_count in enumerate(first_counts):
        for second_index, second_count in enumerate(second_counts):
            distance = abs(first_index - second_index) / maximum_distance
            penalty = distance**2 if weights == "quadratic" else distance
            expected_disagreement += penalty * first_count * second_count / (n_valid * n_valid)
    if math.isclose(expected_disagreement, 0.0, abs_tol=1e-15):
        base["reason"] = "zero_expected_disagreement_single_category"
        return base
    base["kappa"] = 1.0 - observed_disagreement / expected_disagreement
    return base


def item_repeat_long_rows(
    records: list[ExperimentRecord], labels: list[str], scales: list[str]
) -> list[dict[str, Any]]:
    """Return traceable ordinal-item rows retained by the validated loader."""
    rows: list[dict[str, Any]] = []
    for record in records:
        for scale in scales:
            for label in labels:
                stats = record.series.get(scale, {}).get(label) or {}
                for repeat_id, item_values in sorted((stats.get("item_values_by_repeat") or {}).items()):
                    for item_index, score in enumerate(item_values, start=1):
                        rows.append(
                            {
                                "stable_id": record.stable_id,
                                "outer_run_id": record.repeat_id,
                                "persona": record.kbd,
                                "group": record.group,
                                "severity": record.severity,
                                "snapshot_id": f"{record.stable_id}::{label}",
                                "timepoint": label,
                                "scale": scale,
                                "item_id": item_index,
                                "measurement_repeat_id": str(repeat_id),
                                "item_score": score,
                                "cluster_id": record.stable_id,
                            }
                        )
    return rows


def _pairwise_rows(
    observations: list[dict[str, Any]],
    *,
    scale: str,
    weights: str,
    stratum_type: str,
    stratum_value: str,
) -> list[dict[str, Any]]:
    repeat_ids = sorted({str(row["measurement_repeat_id"]) for row in observations}, key=group_sort_key)
    by_repeat: dict[str, dict[tuple[Any, ...], float]] = defaultdict(dict)
    for row in observations:
        object_key = row.get("bootstrap_object_id") or (
            row["snapshot_id"],
            row["scale"],
            int(row["item_id"]),
        )
        by_repeat[str(row["measurement_repeat_id"])][object_key] = float(row["item_score"])
    results: list[dict[str, Any]] = []
    for repeat_a, repeat_b in combinations(repeat_ids, 2):
        first = by_repeat[repeat_a]
        second = by_repeat[repeat_b]
        object_keys = sorted(set(first) | set(second), key=str)
        estimate = cohen_weighted_kappa(
            [first.get(key) for key in object_keys],
            [second.get(key) for key in object_keys],
            weights=weights,
        )
        results.append(
            {
                "stratum_type": stratum_type,
                "stratum_value": stratum_value,
                "scale": scale,
                "weights": weights,
                "repeat_a": repeat_a,
                "repeat_b": repeat_b,
                **estimate,
            }
        )
    return results


def _aggregate_pairwise(pair_rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [float(row["kappa"]) for row in pair_rows if row.get("kappa") is not None]
    reasons = sorted({str(row["reason"]) for row in pair_rows if row.get("reason")})
    return {
        "kappa": mean(valid) if valid else None,
        "median_kappa": median(valid) if valid else None,
        "min_kappa": min(valid) if valid else None,
        "max_kappa": max(valid) if valid else None,
        "n_repeat_pairs": len(pair_rows),
        "n_valid_repeat_pairs": len(valid),
        "n_possible": sum(int(row["n_possible"]) for row in pair_rows),
        "n_valid": sum(int(row["n_valid"]) for row in pair_rows),
        "n_missing": sum(int(row["n_missing"]) for row in pair_rows),
        "reason": None if valid else (";".join(reasons) if reasons else "fewer_than_two_repeats"),
    }


def _bootstrap_interval(
    observations: list[dict[str, Any]],
    *,
    scale: str,
    weights: str,
    scope_key: str,
    replicates: int,
) -> dict[str, Any]:
    clusters: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in observations:
        clusters[str(row["cluster_id"])].append(row)
    cluster_ids = sorted(clusters)
    if len(cluster_ids) < MIN_BOOTSTRAP_CLUSTERS:
        return {
            "ci95_lower": None,
            "ci95_upper": None,
            "ci_method": None,
            "bootstrap_valid_replicates": 0,
            "ci_reason": f"fewer_than_{MIN_BOOTSTRAP_CLUSTERS}_independent_outer_run_clusters",
        }
    repeat_ids = sorted({str(row["measurement_repeat_id"]) for row in observations}, key=group_sort_key)
    repeat_pairs = list(combinations(repeat_ids, 2))
    if not repeat_pairs:
        return {
            "ci95_lower": None,
            "ci95_upper": None,
            "ci_method": None,
            "bootstrap_valid_replicates": 0,
            "ci_reason": "fewer_than_two_repeats",
        }
    cubes = np.zeros((len(cluster_ids), len(repeat_pairs), len(ITEM_CATEGORIES), len(ITEM_CATEGORIES)), dtype=float)
    for cluster_index, cluster_id in enumerate(cluster_ids):
        by_repeat: dict[str, dict[tuple[Any, ...], int]] = defaultdict(dict)
        for row in clusters[cluster_id]:
            object_key = (row["snapshot_id"], row["scale"], int(row["item_id"]))
            score = int(float(row["item_score"]))
            if score in ITEM_CATEGORIES:
                by_repeat[str(row["measurement_repeat_id"])][object_key] = score
        for pair_index, (repeat_a, repeat_b) in enumerate(repeat_pairs):
            first, second = by_repeat[repeat_a], by_repeat[repeat_b]
            for object_key in set(first) & set(second):
                cubes[cluster_index, pair_index, first[object_key], second[object_key]] += 1

    maximum_distance = len(ITEM_CATEGORIES) - 1
    disagreement = np.fromfunction(
        lambda first, second: np.abs(first - second) / maximum_distance,
        (len(ITEM_CATEGORIES), len(ITEM_CATEGORIES)),
        dtype=float,
    )
    if weights == "quadratic":
        disagreement **= 2

    def aggregate_from_matrices(matrices: np.ndarray) -> float | None:
        pair_estimates: list[float] = []
        for matrix in matrices:
            total = float(matrix.sum())
            if total < 2:
                continue
            observed = float((matrix * disagreement).sum() / total)
            expected = float(
                (np.outer(matrix.sum(axis=1), matrix.sum(axis=0)) * disagreement).sum() / (total * total)
            )
            if math.isclose(expected, 0.0, abs_tol=1e-15):
                continue
            pair_estimates.append(1.0 - observed / expected)
        return mean(pair_estimates) if pair_estimates else None

    digest = hashlib.sha256(scope_key.encode("utf-8")).digest()
    rng = random.Random(int.from_bytes(digest[:8], "big"))
    estimates: list[float] = []
    for _ in range(replicates):
        counts = np.bincount(
            [rng.randrange(len(cluster_ids)) for _ in cluster_ids],
            minlength=len(cluster_ids),
        )
        estimate = aggregate_from_matrices(np.tensordot(counts, cubes, axes=(0, 0)))
        if estimate is not None:
            estimates.append(float(estimate))
    minimum_valid = max(30, replicates // 2)
    if len(estimates) < minimum_valid:
        return {
            "ci95_lower": None,
            "ci95_upper": None,
            "ci_method": None,
            "bootstrap_valid_replicates": len(estimates),
            "ci_reason": "too_few_valid_cluster_bootstrap_replicates",
        }
    return {
        "ci95_lower": _percentile(estimates, 0.025),
        "ci95_upper": _percentile(estimates, 0.975),
        "ci_method": "outer_run_cluster_percentile_bootstrap",
        "bootstrap_valid_replicates": len(estimates),
        "ci_reason": None,
    }


def _summary_row(
    observations: list[dict[str, Any]],
    *,
    scale: str,
    weights: str,
    stratum_type: str,
    stratum_value: str,
    item_id: int | None,
    bootstrap_replicates: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    pair_rows = _pairwise_rows(
        observations,
        scale=scale,
        weights=weights,
        stratum_type=stratum_type,
        stratum_value=stratum_value,
    )
    aggregate = _aggregate_pairwise(pair_rows)
    cluster_count = len({row["cluster_id"] for row in observations})
    object_count = len({(row["snapshot_id"], row["scale"], row["item_id"]) for row in observations})
    interval = _bootstrap_interval(
        observations,
        scale=scale,
        weights=weights,
        scope_key=f"{stratum_type}|{stratum_value}|{scale}|{item_id}|{weights}",
        replicates=bootstrap_replicates,
    )
    row = {
        "stratum_type": stratum_type,
        "stratum_value": stratum_value,
        "scale": scale,
        "item_id": item_id,
        "weights": weights,
        "aggregation": "unweighted_mean_of_pairwise_cohen_weighted_kappa",
        "n_outer_run_clusters": cluster_count,
        "n_object_targets": object_count,
        **aggregate,
        **interval,
    }
    return row, pair_rows


def build_weighted_kappa(
    records: list[ExperimentRecord],
    labels: list[str],
    scales: list[str],
    *,
    bootstrap_replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
) -> dict[str, Any]:
    """Build scale, item, pair, matrix, and supported stratified Kappa outputs."""
    observations = item_repeat_long_rows(records, labels, scales)
    summary_rows: list[dict[str, Any]] = []
    item_rows: list[dict[str, Any]] = []
    pairwise_rows: list[dict[str, Any]] = []
    for scale in sorted(scales, key=scale_sort_key):
        scale_rows = [row for row in observations if row["scale"] == scale]
        strata: list[tuple[str, str, list[dict[str, Any]]]] = [("overall", "ALL", scale_rows)]
        for field, name, sorter in (
            ("persona", "persona", group_sort_key),
            ("group", "group", group_sort_key),
            ("timepoint", "timepoint", lambda value: label_sort_key(value, labels)),
        ):
            values = sorted({str(row[field]) for row in scale_rows}, key=sorter)
            strata.extend((name, value, [row for row in scale_rows if str(row[field]) == value]) for value in values)
        for weights in ("quadratic", "linear"):
            for stratum_type, stratum_value, selected in strata:
                summary, pairs = _summary_row(
                    selected,
                    scale=scale,
                    weights=weights,
                    stratum_type=stratum_type,
                    stratum_value=stratum_value,
                    item_id=None,
                    bootstrap_replicates=bootstrap_replicates,
                )
                summary_rows.append(summary)
                pairwise_rows.extend(pairs)
            item_ids = sorted({int(row["item_id"]) for row in scale_rows})
            for item_id in item_ids:
                selected = [row for row in scale_rows if int(row["item_id"]) == item_id]
                summary, _ = _summary_row(
                    selected,
                    scale=scale,
                    weights=weights,
                    stratum_type="overall",
                    stratum_value="ALL",
                    item_id=item_id,
                    bootstrap_replicates=bootstrap_replicates,
                )
                item_rows.append(summary)

    matrix_rows = [
        row
        for row in pairwise_rows
        if row["stratum_type"] == "overall" and row["weights"] == "quadratic"
    ]
    return {
        "metadata": {
            "rated_object": "snapshot_id × scale × item",
            "rater": "measurement_repeat_id",
            "primary_weights": "quadratic",
            "sensitivity_weights": "linear",
            "pairing": "exact object-key intersection; no cross-snapshot or cross-item alignment",
            "scale_total_excluded": True,
            "aggregate": "unweighted mean of valid repeat-pair kappas",
            "ci_method": "outer-run cluster percentile bootstrap when at least 5 independent clusters",
            "bootstrap_replicates": bootstrap_replicates,
            "distribution_warning": "Kappa depends on category prevalence and marginal distributions.",
        },
        "summary": summary_rows,
        "pairwise": pairwise_rows,
        "item": item_rows,
        "matrix": matrix_rows,
        "item_long": observations,
    }


def build_item_weighted_kappa_by_entity(
    records: list[ExperimentRecord],
    labels: list[str],
    scales: list[str],
    *,
    entity_field: str,
) -> dict[str, Any]:
    """Reuse the canonical item κ estimator separately for each group/persona."""
    if entity_field not in {"group", "kbd"}:
        raise ValueError("entity_field must be 'group' or 'kbd'")
    entities = sorted(
        {str(getattr(record, entity_field)) for record in records},
        key=group_sort_key,
    )
    rows: list[dict[str, Any]] = []
    for entity in entities:
        selected = [record for record in records if str(getattr(record, entity_field)) == entity]
        payload = build_weighted_kappa(
            selected,
            labels,
            scales,
            bootstrap_replicates=0,
        )
        for row in payload.get("item") or []:
            if row.get("weights") != "quadratic":
                continue
            rows.append(
                {
                    "entity_type": "persona" if entity_field == "kbd" else "condition",
                    "entity": entity,
                    **row,
                }
            )
    return {
        "metadata": {
            "rated_object": "snapshot_id × scale × item within entity",
            "rater": "measurement_repeat_id",
            "weights": "quadratic",
            "aggregate": "unweighted mean of valid repeat-pair kappas",
            "ci_note": "heatmap shows point estimates; the canonical overall item table retains cluster-bootstrap CI",
        },
        "entities": entities,
        "item": rows,
    }
