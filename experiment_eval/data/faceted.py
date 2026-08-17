"""Input adapters and validation for faceted boxplot tidy data.

The plotting module only consumes the canonical long table produced here.  In
particular, the repeat-summary adapter emits one row per *outer simulation run*
and snapshot.  Its within-snapshot Monte Carlo measurement repeats are not
treated as independent boxplot samples.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from ..loader import load_records
from ..statistics import baseline_label_for_record, endpoint_label_for_record


CANONICAL_COLUMNS = [
    "sample_id",
    "study",
    "group",
    "severity_group",
    "time",
    "outcome",
    "value",
]

_COLUMN_ALIASES = {
    "id": "sample_id",
    "subject_id": "sample_id",
    "outer_run_id": "sample_id",
    "run_id": "sample_id",
    "experiment": "study",
    "replicate": "study",
    "condition": "group",
    "severity": "severity_group",
    "persona_category": "severity_group",
    "persona_id": "severity_group",
    "timepoint": "time",
    "scale": "outcome",
    "score": "value",
    "total_score": "value",
    "setting": "study",
}

_TIME_ALIASES = {
    "pre": "pre",
    "baseline": "pre",
    "t0": "pre",
    "before": "pre",
    "post": "post",
    "endpoint": "post",
    "now": "post",
    "after": "post",
}

_SEVERITY_ALIASES = {
    "minimal": "minimal",
    "min": "minimal",
    "mild": "mild",
    "moderate": "moderate",
    "mod": "moderate",
    "mod-severe": "mod-severe",
    "mod_severe": "mod-severe",
    "moderately severe": "mod-severe",
    "moderately-severe": "mod-severe",
    "severe": "severe",
    "sev": "severe",
}


def _read_json_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict) and isinstance(payload.get("rows"), list):
        rows = payload["rows"]
    elif isinstance(payload, dict) and isinstance(payload.get("data"), list):
        rows = payload["data"]
    else:
        raise ValueError(
            f"{path}: long JSON must be a row list or an object containing 'rows'/'data'"
        )
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"{path}: every long JSON row must be an object")
    return rows


def _read_long_file(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".json":
        return pd.DataFrame.from_records(_read_json_rows(path))
    if suffix in {".jsonl", ".ndjson"}:
        return pd.read_json(path, lines=True)
    raise ValueError(f"Unsupported long-table file type: {path}")


def _rename_aliases(frame: pd.DataFrame) -> pd.DataFrame:
    lowered = {str(column).strip().lower(): column for column in frame.columns}
    rename: dict[Any, str] = {}
    for alias, canonical in _COLUMN_ALIASES.items():
        if canonical not in lowered and alias in lowered:
            rename[lowered[alias]] = canonical
    frame = frame.rename(columns=rename)
    frame.columns = [str(column).strip().lower() for column in frame.columns]
    return frame


def normalize_long_data(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a validated canonical long table while preserving extra columns."""
    frame = _rename_aliases(frame.copy())
    required = {"sample_id", "time", "outcome", "value"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError("Long data is missing required columns: " + ", ".join(missing))

    for column in ("study", "group", "severity_group"):
        if column not in frame:
            frame[column] = "unspecified"

    for column in ("sample_id", "study", "group", "severity_group", "time", "outcome"):
        if frame[column].isna().any():
            raise ValueError(f"Long data column {column!r} contains missing values")
        frame[column] = frame[column].astype(str).str.strip()
        if (frame[column] == "").any():
            raise ValueError(f"Long data column {column!r} contains empty values")

    numeric = pd.to_numeric(frame["value"], errors="coerce")
    invalid = numeric.isna()
    if invalid.any():
        examples = frame.loc[invalid, "value"].astype(str).head(3).tolist()
        raise ValueError(
            f"Long data contains non-numeric value entries, e.g. {examples}"
        )
    frame["value"] = numeric.astype(float)
    if (~frame["value"].map(math.isfinite)).any():
        raise ValueError("Long data contains non-finite value entries")

    frame["time"] = frame["time"].map(
        lambda value: _TIME_ALIASES.get(value.lower(), value.lower())
    )
    frame["severity_group"] = frame["severity_group"].map(
        lambda value: _SEVERITY_ALIASES.get(value.lower(), value)
    )

    duplicate_key = ["sample_id", "study", "group", "severity_group", "time", "outcome"]
    duplicates = frame.duplicated(duplicate_key, keep=False)
    if duplicates.any():
        example = frame.loc[duplicates, duplicate_key].iloc[0].to_dict()
        raise ValueError(
            f"Long data contains duplicate sample/time/outcome rows, e.g. {example}"
        )

    extras = [column for column in frame.columns if column not in CANONICAL_COLUMNS]
    return frame[CANONICAL_COLUMNS + extras].reset_index(drop=True)


def load_long_data(paths: Iterable[Path]) -> pd.DataFrame:
    """Read one or more CSV/JSON/JSONL long tables and normalize their schema."""
    path_list = list(paths)
    if not path_list:
        raise ValueError("No long-table input files were provided")
    frames = [_read_long_file(path) for path in path_list]
    return normalize_long_data(pd.concat(frames, ignore_index=True, sort=False))


def _study_value(record: Any, source: str) -> str:
    if source == "kbd":
        return str(record.kbd)
    if source == "repeat_id":
        return str(record.repeat_id)
    if source == "condition_name":
        return str(record.condition_name)
    if source == "batch_name":
        return str(record.batch_name)
    raise ValueError(f"Unsupported repeat-summary study source: {source}")


def load_repeat_summary_data(
    paths: Iterable[Path],
    *,
    pre_label: str = "auto",
    post_label: str = "auto",
    study_source: str = "kbd",
) -> pd.DataFrame:
    """Adapt validated repeat-summary JSON files to the canonical long table."""
    path_list = list(paths)
    if not path_list:
        raise ValueError("No repeat-summary files were provided")
    records, labels, scales = load_records(path_list)
    rows: list[dict[str, Any]] = []
    for record in records:
        for outcome in scales:
            available = record.series.get(outcome, {})
            before = (
                baseline_label_for_record(record, labels, outcome)
                if pre_label == "auto"
                else pre_label
            )
            after = (
                endpoint_label_for_record(record, labels, outcome)
                if post_label == "auto"
                else post_label
            )
            for canonical_time, source_label in (("pre", before), ("post", after)):
                stats = available.get(source_label or "")
                if not stats or stats.get("score") is None:
                    continue
                rows.append(
                    {
                        "sample_id": record.stable_id,
                        "study": _study_value(record, study_source),
                        "group": record.group,
                        "severity_group": record.severity,
                        "time": canonical_time,
                        "outcome": outcome,
                        "value": float(stats["score"]),
                        "source_time": source_label,
                        "source_file": str(record.path),
                    }
                )
    if not rows:
        raise ValueError("Repeat-summary inputs contain no usable pre/post scores")
    return normalize_long_data(pd.DataFrame.from_records(rows))


def quality_summary(frame: pd.DataFrame, time_order: list[str]) -> dict[str, Any]:
    """Describe coverage and pre/post pairing without changing the data."""
    id_columns = ["sample_id", "study", "group", "severity_group", "outcome"]
    expected = set(time_order)
    grouped = frame.groupby(id_columns, dropna=False)["time"].agg(
        lambda values: set(values)
    )
    incomplete = int(sum(not expected.issubset(values) for values in grouped))
    return {
        "rows": int(len(frame)),
        "samples": int(frame[["study", "sample_id"]].drop_duplicates().shape[0]),
        "studies": sorted(frame["study"].unique().tolist()),
        "groups": sorted(frame["group"].unique().tolist()),
        "severity_groups": sorted(frame["severity_group"].unique().tolist()),
        "times": sorted(frame["time"].unique().tolist()),
        "outcomes": sorted(frame["outcome"].unique().tolist()),
        "incomplete_pre_post_cells": incomplete,
    }
