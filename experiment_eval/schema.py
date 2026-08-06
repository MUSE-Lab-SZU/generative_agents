"""Shared schema objects, constants, and naming helpers."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any


EXPECTED_AGGREGATION_METHOD_VERSION = "fixed_complete_scale_reviewed_v2"
EXPECTED_PRIMARY_SCORE = "mean_of_complete_reviewed_scale_totals"
EXPERIMENT_EVAL_METRICS_SCHEMA_VERSION = "experiment_eval_metrics_v2"
SCALE_ORDER = ["PHQ-9", "BDI-II"]
SCALE_RANGES = {"PHQ-9": (0.0, 27.0), "BDI-II": (0.0, 63.0)}
DEFAULT_LABEL_ORDER = [
    "T0",
    "session_4",
    "session_8",
    "session_12",
    "session_16",
    "session_20",
    "NOW",
    "POST",
]
ENDPOINT_LABEL_PRIORITY = ["POST", "NOW"]
SEVERITY_ORDER = ["MILD", "MOD", "SEV"]
SEVERITY_ALIASES = {
    "mild": "MILD",
    "moderate": "MOD",
    "mod": "MOD",
    "severe": "SEV",
    "sev": "SEV",
}
GROUP_COLORS = {
    "G1": "#2563eb",
    "G2": "#16a34a",
    "G3": "#dc2626",
    "G4": "#9333ea",
    "G5": "#b45309",
    "G6": "#0f766e",
    "G7": "#db2777",
    "G8": "#475569",
    "G9": "#0891b2",
}
KBD_COLORS = {
    "KBD2": "#2563eb",
    "KBD4": "#16a34a",
    "KBD6": "#9333ea",
}
FALLBACK_COLORS = ["#65a30d", "#7c3aed", "#0369a1", "#be123c", "#4d7c0f"]
REPEAT_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "<", ">", "h"]
REPEAT_LINESTYLES = ["-", "--", "-.", ":"]


@dataclass
class ExperimentRecord:
    stable_id: str
    key: str
    kbd: str
    group: str
    source_group: str
    severity: str
    repeat_id: str
    plot_label: str
    batch_name: str
    path: Path
    condition_name: str
    run_name: str
    series: dict[str, dict[str, dict[str, Any]]]
    expected_repeats: int
    repeats_paired_across_timepoints: bool
    pairing_evidence: str

    def with_plot_label(self, label: str) -> "ExperimentRecord":
        return replace(self, plot_label=label)


def as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(number) else number


def slugify(value: str) -> str:
    known = {"PHQ-9": "phq9", "BDI-II": "bdi2"}
    if value in known:
        return known[value]
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip()).strip("_").lower()
    return slug or "unknown"


def stable_component(value: str) -> str:
    cleaned = re.sub(r"[^\w.-]+", "-", value.strip(), flags=re.UNICODE).strip("-")
    return cleaned or "UNKNOWN"


def normalize_group(value: str) -> str:
    match = re.search(r"G\s*0*(\d+)", value.strip(), re.IGNORECASE)
    return f"G{int(match.group(1))}" if match else value.strip().upper()


def normalize_repeat_id(value: str) -> str:
    match = re.search(r"(?:R)?0*(\d+)$", value.strip(), re.IGNORECASE)
    return f"R{int(match.group(1)):02d}" if match else value.strip().upper()


def normalize_kbd(value: str) -> str:
    match = re.search(r"KBD\s*0*(\d+)", value.strip(), re.IGNORECASE)
    return f"KBD{int(match.group(1))}" if match else value.strip().upper()


def session_index(label: str) -> int | None:
    match = re.fullmatch(r"session[_ -]?(\d+)", label.strip(), re.IGNORECASE)
    return int(match.group(1)) if match else None


def label_sort_key(label: str, discovered_order: list[str] | None = None) -> tuple[int, int, str]:
    if label == "T0":
        return (0, 0, label)
    number = session_index(label)
    if number is not None:
        return (1, number, label)
    if label in {"NOW": 0, "POST": 1}:
        return (2, {"NOW": 0, "POST": 1}[label], label)
    try:
        return (3, (discovered_order or DEFAULT_LABEL_ORDER).index(label), label)
    except ValueError:
        return (4, 999, label)


def label_display(label: str) -> str:
    number = session_index(label)
    return f"S{number}" if number is not None else label.replace("_", " ")


def group_sort_key(value: str) -> tuple[int, str]:
    match = re.search(r"(\d+)", value)
    return (int(match.group(1)) if match else 999, value)


def severity_sort_key(value: str) -> tuple[int, str]:
    try:
        return (SEVERITY_ORDER.index(value), value)
    except ValueError:
        return (len(SEVERITY_ORDER), value)


def scale_sort_key(value: str) -> tuple[int, str]:
    try:
        return (SCALE_ORDER.index(value), value)
    except ValueError:
        return (len(SCALE_ORDER), value)


def group_color(group: str) -> str:
    normalized = normalize_group(group)
    if normalized in GROUP_COLORS:
        return GROUP_COLORS[normalized]
    index = sum(ord(char) for char in normalized) % len(FALLBACK_COLORS)
    return FALLBACK_COLORS[index]


def kbd_color(kbd: str) -> str:
    normalized = normalize_kbd(kbd)
    if normalized in KBD_COLORS:
        return KBD_COLORS[normalized]
    index = sum(ord(char) for char in normalized) % len(FALLBACK_COLORS)
    return FALLBACK_COLORS[index]


def repeat_style(repeat_id: str) -> tuple[str, str]:
    match = re.search(r"(\d+)", repeat_id)
    index = (int(match.group(1)) - 1) if match else sum(ord(c) for c in repeat_id)
    return REPEAT_MARKERS[index % len(REPEAT_MARKERS)], REPEAT_LINESTYLES[index % len(REPEAT_LINESTYLES)]
