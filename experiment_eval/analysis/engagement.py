"""Outer-run aggregation for daily engagement and 24-hour interaction profiles."""

from __future__ import annotations

from collections import defaultdict
import math
import re
from statistics import mean
from typing import Any

from ..schema import group_sort_key
from ..statistics import mean_ci95


_SESSION_LABEL = re.compile(r"session[_ -]?(\d+)$", re.IGNORECASE)


def _summary(
    values: list[float],
) -> tuple[float | None, float | None, float | None, int]:
    if not values:
        return None, None, None, 0
    interval = mean_ci95(values)
    return mean(values), interval["lower"], interval["upper"], len(values)


def _is_true(value: Any) -> bool:
    return value is True or str(value).strip().lower() == "true"


def _session_number(value: Any) -> int | None:
    match = _SESSION_LABEL.fullmatch(str(value or "").strip())
    return int(match.group(1)) if match else None


def _scheduled_session_events(
    rows: list[dict[str, Any]], max_session: int
) -> dict[int, dict[str, Any]]:
    """Identify the regular doctor-target CBT sequence within one outer run.

    Archived event rows often lack ``session_id``. In that case the planned
    CBT conversations form a complete arithmetic progression in simulation
    day, while incidental doctor-target chats fall outside that progression.
    """
    explicit: dict[int, dict[str, Any]] = {}
    candidates = [
        row for row in rows if _is_true(row.get("doctor_target_interaction"))
    ]
    for row in candidates:
        number = _session_number(row.get("session_id"))
        if number is not None and 1 <= number <= max_session:
            explicit[number] = row
    if len(explicit) == max_session:
        return explicit

    by_day: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        try:
            day = int(row["simulation_day"])
        except (KeyError, TypeError, ValueError):
            continue
        by_day[day].append(row)
    if not by_day:
        return explicit

    days = sorted(by_day)
    best: tuple[int, int, int] | None = None
    for start in days[: min(4, len(days))]:
        for cadence in range(1, 15):
            support = sum(
                start + cadence * index in by_day for index in range(max_session)
            )
            score = (support, cadence, -start)
            if best is None or score > best:
                best = score
    if best is not None and best[0] >= max(2, max_session // 2):
        _, cadence, negative_start = best
        start = -negative_start
        selected: dict[int, dict[str, Any]] = {}
        for number in range(1, max_session + 1):
            matches = by_day.get(start + cadence * (number - 1), [])
            if not matches:
                continue
            selected[number] = sorted(
                matches,
                key=lambda row: (
                    row.get("interaction_type") != "doctor_consult",
                    row.get("timestamp") or "",
                    int(row.get("conversation_index") or 0),
                ),
            )[0]
        if len(selected) >= len(explicit):
            return selected

    chronological = sorted(
        candidates,
        key=lambda row: (
            row.get("timestamp") or "",
            int(row.get("conversation_index") or 0),
        ),
    )
    return {
        number: row
        for number, row in enumerate(chronological[:max_session], start=1)
    }


def build_session_turn_score_analysis(
    events: list[dict[str, Any]],
    change_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Summarize formal CBT-session turns and assessment changes by group."""
    session_numbers = [
        number
        for row in change_rows
        if (number := _session_number(row.get("timepoint"))) is not None
    ]
    max_session = max(session_numbers, default=0)
    if max_session <= 0:
        return [], {
            "status": "unavailable",
            "reason": "No session-numbered outcome assessments were available",
        }

    events_by_run: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in events:
        events_by_run[str(row.get("stable_id") or "")].append(row)

    run_turn_rows: list[dict[str, Any]] = []
    session_detection: dict[str, int] = {}
    for stable_id, rows in events_by_run.items():
        selected = _scheduled_session_events(rows, max_session)
        session_detection[stable_id] = len(selected)
        for session, row in selected.items():
            turns = row.get("turn_count")
            if turns is None or not math.isfinite(float(turns)):
                continue
            run_turn_rows.append(
                {
                    "stable_id": stable_id,
                    "group": str(row.get("group") or ""),
                    "session": session,
                    "value": float(turns),
                }
            )

    output: list[dict[str, Any]] = []
    grouped_turns: dict[tuple[str, int], list[float]] = defaultdict(list)
    for row in run_turn_rows:
        grouped_turns[(row["group"], row["session"])].append(row["value"])
    for (group, session), values in grouped_turns.items():
        center, lower, upper, n = _summary(values)
        output.append(
            {
                "group": group,
                "session": session,
                "metric": "dialogue_turns",
                "outcome": None,
                "mean": center,
                "ci95_lower": lower,
                "ci95_upper": upper,
                "n_outer_runs": n,
            }
        )

    grouped_changes: dict[tuple[str, int, str], list[float]] = defaultdict(list)
    for row in change_rows:
        timepoint = str(row.get("timepoint") or "")
        session = 0 if timepoint == "T0" else _session_number(timepoint)
        value = row.get("score_change")
        complete_case = row.get("complete_case")
        if (
            session is None
            or value is None
            or not math.isfinite(float(value))
            or (complete_case is not None and not _is_true(complete_case))
        ):
            continue
        grouped_changes[
            (str(row.get("group") or ""), session, str(row.get("outcome") or ""))
        ].append(float(value))
    for (group, session, outcome), values in grouped_changes.items():
        center, lower, upper, n = _summary(values)
        output.append(
            {
                "group": group,
                "session": session,
                "metric": "score_change",
                "outcome": outcome,
                "mean": center,
                "ci95_lower": lower,
                "ci95_upper": upper,
                "n_outer_runs": n,
            }
        )

    output.sort(
        key=lambda row: (
            group_sort_key(row["group"]),
            row["metric"] != "dialogue_turns",
            str(row.get("outcome") or ""),
            row["session"],
        )
    )
    complete_runs = sum(count >= max_session for count in session_detection.values())
    return output, {
        "status": "available" if output else "unavailable",
        "max_session": max_session,
        "session_definition": (
            "doctor-target CBT conversations on the complete regular simulation-day sequence; "
            "incidental doctor-target chats are excluded"
        ),
        "runs_with_detected_sessions": sum(count > 0 for count in session_detection.values()),
        "runs_with_complete_session_sequence": complete_runs,
        "turn_inference_unit": "independent outer simulation run",
        "score_change_reference": "T0",
    }


def build_engagement_analysis(
    events: list[dict[str, Any]],
    coverage: list[dict[str, Any]],
    *,
    bin_hours: int = 2,
    secondary_metric: str = "turns",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if bin_hours <= 0 or 24 % bin_hours:
        raise ValueError("bin_hours must be a positive divisor of 24")
    if secondary_metric not in {"turns", "duration-proxy"}:
        raise ValueError(f"Unsupported engagement secondary metric: {secondary_metric}")

    events_by_run_day: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    events_by_run_bin: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in events:
        events_by_run_day[(row["stable_id"], int(row["simulation_day"]))].append(row)
        bin_start = int(float(row["hour"]) // bin_hours) * bin_hours
        events_by_run_bin[(row["stable_id"], bin_start)].append(row)

    run_daily: list[dict[str, Any]] = []
    run_bins: list[dict[str, Any]] = []
    for run in coverage:
        days = int(run.get("observation_days") or 0)
        if days <= 0:
            continue
        for day in range(1, days + 1):
            values = events_by_run_day.get((run["stable_id"], day), [])
            turns = [
                float(row["turn_count"])
                for row in values
                if row.get("turn_count") is not None
            ]
            durations = [
                float(row["duration_minutes_proxy"])
                for row in values
                if row.get("duration_minutes_proxy") is not None
            ]
            secondary_values = turns if secondary_metric == "turns" else durations
            run_daily.append(
                {
                    **{
                        key: run[key]
                        for key in ("stable_id", "persona", "group", "replicate_id")
                    },
                    "simulation_day": day,
                    "interaction_frequency": len(values),
                    "secondary_value": (
                        mean(secondary_values) if secondary_values else None
                    ),
                    "secondary_metric": secondary_metric,
                }
            )
        for bin_start in range(0, 24, bin_hours):
            count = len(events_by_run_bin.get((run["stable_id"], bin_start), []))
            run_bins.append(
                {
                    **{
                        key: run[key]
                        for key in ("stable_id", "persona", "group", "replicate_id")
                    },
                    "bin_start_hour": bin_start,
                    "bin_end_hour": bin_start + bin_hours,
                    "interaction_frequency_per_day": count / days,
                }
            )

    daily: list[dict[str, Any]] = []
    grouped_daily: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in run_daily:
        grouped_daily[(row["group"], row["simulation_day"])].append(row)
    for (group, day), rows in grouped_daily.items():
        frequency = [float(row["interaction_frequency"]) for row in rows]
        secondary = [
            float(row["secondary_value"])
            for row in rows
            if row["secondary_value"] is not None
        ]
        f_mean, f_lower, f_upper, f_n = _summary(frequency)
        s_mean, s_lower, s_upper, s_n = _summary(secondary)
        daily.append(
            {
                "group": group,
                "simulation_day": day,
                "frequency_mean": f_mean,
                "frequency_ci95_lower": f_lower,
                "frequency_ci95_upper": f_upper,
                "frequency_n_outer_runs": f_n,
                "secondary_metric": secondary_metric,
                "secondary_mean": s_mean,
                "secondary_ci95_lower": s_lower,
                "secondary_ci95_upper": s_upper,
                "secondary_n_outer_runs": s_n,
            }
        )

    time_bins: list[dict[str, Any]] = []
    grouped_bins: dict[tuple[str, int, int], list[float]] = defaultdict(list)
    for row in run_bins:
        grouped_bins[(row["group"], row["bin_start_hour"], row["bin_end_hour"])].append(
            float(row["interaction_frequency_per_day"])
        )
    for (group, start, end), values in grouped_bins.items():
        center, lower, upper, n = _summary(values)
        time_bins.append(
            {
                "group": group,
                "bin_start_hour": start,
                "bin_end_hour": end,
                "time_bin": f"{start:02d}-{end:02d}",
                "frequency_mean_per_day": center,
                "frequency_ci95_lower": lower,
                "frequency_ci95_upper": upper,
                "n_outer_runs": n,
            }
        )
    daily.sort(key=lambda row: (group_sort_key(row["group"]), row["simulation_day"]))
    time_bins.sort(
        key=lambda row: (group_sort_key(row["group"]), row["bin_start_hour"])
    )
    diagnostics = {
        "bin_hours": bin_hours,
        "secondary_metric": secondary_metric,
        "inference_unit": "independent outer simulation run",
        "daily_zero_handling": "zero-frequency covered run-days are retained; secondary means are missing when no session occurred",
        "duration_semantics": (
            "planned action-duration proxy"
            if secondary_metric == "duration-proxy"
            else "not used"
        ),
        "groups": sorted({row["group"] for row in daily}, key=group_sort_key),
    }
    return daily, time_bins, diagnostics
