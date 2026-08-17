"""Canonical total-score data for persona treatment-response profiles.

One row is one independent outer run at one timepoint/scale.  Frozen-snapshot
measurement repeats must already be collapsed before entering this layer.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ..schema import ExperimentRecord
from ..statistics import score_at


REQUIRED_COLUMNS = ("run_id", "persona_id", "group", "timepoint", "scale", "score")
OPTIONAL_COLUMNS = ("setting",)
ALIASES = {
    "run": "run_id",
    "outer_run_id": "run_id",
    "stable_id": "run_id",
    "sample_id": "run_id",
    "persona": "persona_id",
    "kbd": "persona_id",
    "condition": "group",
    "time": "timepoint",
    "outcome": "scale",
    "value": "score",
    "study": "setting",
    "experiment": "setting",
    "panel": "setting",
}


def _read_json(path: Path) -> pd.DataFrame:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("rows")
    if not isinstance(payload, list):
        raise ValueError(
            "JSON profile data must be a row list or {'rows': [...]} object"
        )
    return pd.DataFrame(payload)


def read_persona_profile_data(path: Path) -> pd.DataFrame:
    """Read CSV, JSON, or JSONL total-score observations."""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        frame = pd.read_csv(path)
    elif suffix in {".jsonl", ".ndjson"}:
        frame = pd.read_json(path, lines=True)
    elif suffix == ".json":
        frame = _read_json(path)
    else:
        raise ValueError(
            f"Unsupported persona-profile format {suffix!r}; use CSV, JSON, or JSONL"
        )
    return normalize_persona_profile_data(frame)


def normalize_persona_profile_data(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate the run-level input and reject pseudo-replicated duplicate cells."""
    rename: dict[str, str] = {}
    for source, target in ALIASES.items():
        if source in frame.columns and target not in frame.columns:
            rename[source] = target
    normalized = frame.rename(columns=rename).copy()
    missing = [
        column for column in REQUIRED_COLUMNS if column not in normalized.columns
    ]
    if missing:
        raise ValueError(
            f"Persona-profile data missing required columns: {', '.join(missing)}"
        )
    if "setting" not in normalized.columns:
        normalized["setting"] = "unspecified"
    columns = list(
        dict.fromkeys([*REQUIRED_COLUMNS, *OPTIONAL_COLUMNS, *normalized.columns])
    )
    normalized = normalized.loc[:, columns].copy()
    for column in ("run_id", "persona_id", "group", "timepoint", "scale", "setting"):
        normalized[column] = normalized[column].astype("string").str.strip()
        if normalized[column].isna().any() or (normalized[column] == "").any():
            raise ValueError(f"Persona-profile column {column!r} contains empty values")
    normalized["score"] = pd.to_numeric(normalized["score"], errors="coerce")
    if normalized["score"].isna().any():
        raise ValueError("Persona-profile score must be numeric and non-missing")
    duplicate_key = ["run_id", "persona_id", "group", "setting", "timepoint", "scale"]
    duplicates = normalized.duplicated(duplicate_key, keep=False)
    if duplicates.any():
        examples = normalized.loc[duplicates, duplicate_key].head(3).to_dict("records")
        raise ValueError(
            "Persona-profile data contains duplicate run/persona/group/setting/timepoint/scale rows; "
            f"collapse frozen-snapshot measurement repeats first. Examples: {examples}"
        )
    return normalized.reset_index(drop=True)


def persona_profile_data_from_records(
    records: list[ExperimentRecord],
    labels: list[str],
    scales: list[str],
    *,
    setting: str = "unspecified",
) -> pd.DataFrame:
    """Adapt validated repeat summaries without inventing a severity setting.

    Persona and experimental condition are the default grouping dimensions.
    Callers may provide a real archived protocol/study label through
    ``setting``; the condition severity token is retained on
    :class:`ExperimentRecord` for compatibility but is not a study dimension.
    """
    rows = []
    for record in records:
        for scale in scales:
            for timepoint in labels:
                score = score_at(record, scale, timepoint)
                if score is None:
                    continue
                rows.append(
                    {
                        "run_id": record.stable_id,
                        "persona_id": record.kbd,
                        "group": record.group,
                        "timepoint": timepoint,
                        "scale": scale,
                        "score": score,
                        "setting": setting,
                    }
                )
    return normalize_persona_profile_data(pd.DataFrame(rows))
