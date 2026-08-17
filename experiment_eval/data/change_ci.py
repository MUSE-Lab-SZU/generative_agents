"""Canonical run-level data extraction for baseline-change figures.

This module knows about :class:`ExperimentRecord`, but it does not calculate
confidence intervals and it never imports Matplotlib.  One row represents one
outer simulation run at one outcome/timepoint; Monte Carlo scale generations
remain collapsed into the validated snapshot mean loaded by ``loader.py``.
"""

from __future__ import annotations

from typing import Any

from ..schema import ExperimentRecord
from ..statistics import baseline_label_for_record, score_at


def extract_change_score_rows(
    records: list[ExperimentRecord],
    labels: list[str],
    outcomes: list[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Adapt validated repeat-summary records to a reusable change-score table."""
    rows: list[dict[str, Any]] = []
    missing: list[dict[str, str]] = []
    for record in records:
        for outcome in outcomes:
            baseline_timepoint = baseline_label_for_record(record, labels, outcome)
            baseline_score = score_at(record, outcome, baseline_timepoint)
            outcome_timepoints = [
                label
                for label in labels
                if any(
                    score_at(candidate, outcome, label) is not None
                    for candidate in records
                )
            ]
            complete_case = (
                bool(outcome_timepoints)
                and baseline_score is not None
                and all(
                    score_at(record, outcome, label) is not None
                    for label in outcome_timepoints
                )
            )
            if baseline_timepoint is None or baseline_score is None:
                missing.append(
                    {
                        "stable_id": record.stable_id,
                        "outcome": outcome,
                        "field": "baseline_score",
                    }
                )
            for timepoint in outcome_timepoints:
                current_score = score_at(record, outcome, timepoint)
                if current_score is None:
                    missing.append(
                        {
                            "stable_id": record.stable_id,
                            "outcome": outcome,
                            "field": f"current_score:{timepoint}",
                        }
                    )
                rows.append(
                    {
                        "stable_id": record.stable_id,
                        "run_name": record.run_name,
                        "persona": record.kbd,
                        "group": record.group,
                        "replicate_id": record.repeat_id,
                        "analysis_set": "main",
                        "outcome": outcome,
                        "timepoint": timepoint,
                        "baseline_timepoint": baseline_timepoint,
                        "baseline_score": baseline_score,
                        "current_score": current_score,
                        "score_change": (
                            current_score - baseline_score
                            if current_score is not None and baseline_score is not None
                            else None
                        ),
                        "complete_case": complete_case,
                        # load_records accepts only strict-complete, recomputed and
                        # internally consistent scale summaries. This flag refers
                        # to that scale-data QC, not to an unrecorded simulation QC.
                        "qc_passed": True,
                        "qc_source": "strict_repeat_summary_validation",
                    }
                )
    diagnostics = {
        "row_count": len(rows),
        "outer_run_count": len({row["stable_id"] for row in rows}),
        "outcomes_requested": outcomes,
        "missing_values": missing,
        "qc_scope": (
            "qc_passed means the repeat-summary passed loader integrity and complete-K validation; "
            "no separate simulation-run QC flag is present in the current schema"
        ),
        "replication_scope": (
            "replicate_id is the independent outer-run identifier; the current schema has no explicit "
            "main-versus-replication study role"
        ),
    }
    return rows, diagnostics
