"""Canonical long-format item data for paper-oriented symptom figures.

The independent sampling unit is ``run``.  Measurement repeats from one
frozen snapshot must be averaged before they enter this table.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from ..life_state import build_item_trajectory_rows
from ..schema import ExperimentRecord


REQUIRED_COLUMNS = (
    "run",
    "persona",
    "group",
    "timepoint",
    "scale",
    "item",
    "item_score",
)
ALIASES = {
    "outer_run_id": "run",
    "stable_id": "run",
    "sample_id": "run",
    "subject_id": "run",
    "kbd": "persona",
    "condition": "group",
    "time": "timepoint",
    "outcome": "scale",
    "item_id": "item",
    "item_repeat_mean": "item_score",
    "score": "item_score",
    "value": "item_score",
}


def _read_json(path: Path) -> pd.DataFrame:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("rows")
    if not isinstance(payload, list):
        raise ValueError("JSON long data must be a row list or {'rows': [...]} object")
    return pd.DataFrame(payload)


def read_long_data(path: Path) -> pd.DataFrame:
    """Read CSV, JSON, or JSONL and normalize to the canonical schema."""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        frame = pd.read_csv(path)
    elif suffix in {".jsonl", ".ndjson"}:
        frame = pd.read_json(path, lines=True)
    elif suffix == ".json":
        frame = _read_json(path)
    else:
        raise ValueError(
            f"Unsupported long-data format {suffix!r}; use CSV, JSON, or JSONL"
        )
    return normalize_long_data(frame)


def normalize_long_data(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate item observations without treating measurement repeats as runs."""
    renamed = frame.copy()
    for alias, canonical in ALIASES.items():
        if canonical not in renamed.columns and alias in renamed.columns:
            renamed = renamed.rename(columns={alias: canonical})
    missing = [column for column in REQUIRED_COLUMNS if column not in renamed.columns]
    if missing:
        raise ValueError(f"Long data missing required columns: {', '.join(missing)}")
    normalized = renamed.loc[
        :, list(dict.fromkeys([*REQUIRED_COLUMNS, *renamed.columns]))
    ].copy()
    for column in ("run", "persona", "group", "timepoint", "scale"):
        normalized[column] = normalized[column].astype("string").str.strip()
        if normalized[column].isna().any() or (normalized[column] == "").any():
            raise ValueError(f"Long data column {column!r} contains empty values")
    normalized["item"] = pd.to_numeric(normalized["item"], errors="coerce")
    normalized["item_score"] = pd.to_numeric(normalized["item_score"], errors="coerce")
    if normalized[["item", "item_score"]].isna().any().any():
        raise ValueError(
            "Long data item and item_score must be numeric and non-missing"
        )
    if (normalized["item"] % 1 != 0).any() or (normalized["item"] < 1).any():
        raise ValueError("Long data item values must be positive integers")
    normalized["item"] = normalized["item"].astype(int)
    if ((normalized["item_score"] < 0) | (normalized["item_score"] > 3)).any():
        raise ValueError("PHQ-9/BDI-II item_score values must be within 0–3")
    key = ["run", "group", "timepoint", "scale", "item"]
    duplicates = normalized.duplicated(key, keep=False)
    if duplicates.any():
        example = normalized.loc[duplicates, key].head(3).to_dict("records")
        raise ValueError(
            "Long data contains duplicate run/group/timepoint/scale/item rows; "
            f"average frozen-snapshot measurement repeats first. Examples: {example}"
        )
    return normalized.reset_index(drop=True)


def long_data_from_records(
    records: list[ExperimentRecord], labels: list[str], scales: list[str]
) -> pd.DataFrame:
    """Adapt validated repeat summaries to long format using outer-run item means."""
    rows = build_item_trajectory_rows(records, labels, scales)
    frame = pd.DataFrame(
        {
            # The actual simulation run remains the same when a later
            # follow-up summary is added.  Archived batches can reuse R01/R02
            # stable IDs, so prefer run_name when it is available.
            "run": row.get("run_name") or row["stable_id"],
            "persona": row["persona"],
            "group": row["group"],
            "timepoint": row["timepoint"],
            "scale": row["scale"],
            "item": row["item_id"],
            "item_score": row["item_repeat_mean"],
        }
        for row in rows
    )
    return normalize_long_data(frame)


def timepoint_sort_key(value: str) -> tuple[int, int, str]:
    text = str(value).strip()
    if text.lower() in {"t0", "baseline", "base", "pre"}:
        return (0, 0, text)
    session = re.fullmatch(r"(?:session|s)[_ -]?(\d+)", text, re.IGNORECASE)
    if session:
        return (1, int(session.group(1)), text)
    scheduled = re.fullmatch(
        r"(?:week|wk|w|day|d|month|mo|t)[_ -]?(\d+)", text, re.IGNORECASE
    )
    if scheduled:
        return (1, int(scheduled.group(1)), text)
    followup = re.fullmatch(
        r"(?:followup|follow-up)[_ -]?(?:step[_ -]?)?(\d+)", text, re.IGNORECASE
    )
    if followup:
        return (2, int(followup.group(1)), text)
    if text.upper() in {"NOW", "POST", "ENDPOINT"}:
        return (3, {"NOW": 0, "POST": 1, "ENDPOINT": 2}[text.upper()], text)
    return (4, 999, text)


def ordered_timepoints(frame: pd.DataFrame) -> list[str]:
    return sorted(
        frame["timepoint"].dropna().astype(str).unique(), key=timepoint_sort_key
    )


def dataset_audit(frame: pd.DataFrame) -> dict[str, Any]:
    """Return coverage information used by every new figure family."""
    cell_runs = (
        frame.groupby(["scale", "group", "timepoint"], observed=True)["run"]
        .nunique()
        .reset_index(name="n_runs")
    )
    return {
        "rows": int(len(frame)),
        "independent_runs": int(frame["run"].nunique()),
        "personas": sorted(frame["persona"].astype(str).unique()),
        "groups": {
            str(group): int(part["run"].nunique())
            for group, part in frame.groupby("group", observed=True)
        },
        "scales": {
            str(scale): {
                "items": sorted(int(value) for value in part["item"].unique()),
                "timepoints": ordered_timepoints(part),
                "runs": int(part["run"].nunique()),
                "minimum_runs_per_group_time_cell": int(
                    cell_runs.loc[cell_runs["scale"] == scale, "n_runs"].min()
                ),
            }
            for scale, part in frame.groupby("scale", observed=True)
        },
    }
