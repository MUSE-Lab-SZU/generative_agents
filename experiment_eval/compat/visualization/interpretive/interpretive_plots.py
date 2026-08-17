"""Compatibility facade and renderer for interpretive charts."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from ....charts.interpretive.complaint_evaluation_changes import (
    plot_complaint_evaluation_changes,
)
from ....charts.interpretive.group_symptom_heatmap import plot_group_symptom_heatmap
from ....charts.interpretive.item_life_state_change import plot_item_life_state_change
from ....charts.interpretive.life_state_symptom_trajectory import (
    plot_symptom_trajectory,
)


def render_interpretive_figures(
    life_state: dict[str, list[dict[str, Any]]],
    complaint: dict[str, Any],
    labels: list[str],
    out_dir: Path,
) -> list[Path]:
    paths = [
        plot_item_life_state_change(life_state["summary"], out_dir),
        plot_symptom_trajectory(life_state["symptom_trajectory"], labels, out_dir),
        plot_group_symptom_heatmap(life_state["summary"], out_dir),
        plot_complaint_evaluation_changes(
            complaint["intervals"], complaint["evaluation_labels"], out_dir
        ),
    ]
    return [path for path in paths if path is not None]
