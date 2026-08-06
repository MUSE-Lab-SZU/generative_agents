"""Small public API used by historical, bespoke analysis scripts.

The main CLI should be preferred for new analyses.  This module keeps older
0712/0716 scripts working without depending on the removed monolithic plotting
entry point.
"""

from __future__ import annotations

from typing import Any

from .loader import display_path, find_report_files, load_records, read_json
from .schema import SCALE_RANGES, ExperimentRecord, label_display
from .statistics import baseline_label_for_record, endpoint_label_for_record, score_at


def axis_label_display(label: str) -> str:
    """Return the compact display label used on trajectory x axes."""
    return label_display(label)


def normalized_improvement(
    record: ExperimentRecord,
    labels: list[str],
    scales: list[str],
) -> dict[str, Any]:
    """Average baseline-to-endpoint improvement normalized by scale range."""
    components: dict[str, float] = {}
    for scale in scales:
        baseline_label = baseline_label_for_record(record, labels, scale)
        endpoint_label = endpoint_label_for_record(record, labels, scale)
        baseline = score_at(record, scale, baseline_label)
        endpoint = score_at(record, scale, endpoint_label)
        scale_range = SCALE_RANGES.get(scale)
        if baseline is None or endpoint is None or scale_range is None:
            continue
        lower, upper = scale_range
        width = upper - lower
        if width <= 0:
            continue
        components[scale] = (baseline - endpoint) / width
    return {
        "score": sum(components.values()) / len(components) if components else None,
        "components": components,
    }


__all__ = [
    "ExperimentRecord",
    "axis_label_display",
    "baseline_label_for_record",
    "display_path",
    "endpoint_label_for_record",
    "find_report_files",
    "label_display",
    "load_records",
    "normalized_improvement",
    "read_json",
    "score_at",
]
