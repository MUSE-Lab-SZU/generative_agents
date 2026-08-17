"""Compatibility facade and renderer for stratified interpretive charts."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from ....charts.interpretive.complaint_interval_alignment import (
    plot_complaint_interval_alignment,
)
from ....charts.interpretive.complaint_run_alignment import plot_complaint_run_alignment
from ....charts.kappa.entity_heatmap import plot_entity_kappa_heatmap
from ....charts.interpretive.stratified_item_heatmap import plot_item_heatmap
from ....charts.interpretive.stratified_replicate_agreement import (
    plot_replicate_agreement,
)
from ....charts.interpretive.stratified_run_symptom_heatmap.plot import (
    _run_order,
    plot_run_symptom_heatmap,
)


def render_stratified_figures(
    life_state: dict[str, list[dict[str, Any]]],
    agreement_rows: list[dict[str, Any]],
    interval_alignment: list[dict[str, Any]],
    run_alignment: list[dict[str, Any]],
    kappa: dict[str, Any],
    labels: list[str],
    entity_field: str,
    entity_type: str,
    title: str,
    out_dir: Path,
) -> list[Path]:
    paths = [
        plot_run_symptom_heatmap(
            life_state["symptom_change"], entity_field, title, out_dir
        ),
        plot_replicate_agreement(agreement_rows, entity_field, title, out_dir),
        plot_item_heatmap(
            life_state["item_change"], "PHQ-9", entity_field, title, out_dir
        ),
        plot_item_heatmap(
            life_state["item_change"], "BDI-II", entity_field, title, out_dir
        ),
        plot_complaint_interval_alignment(
            interval_alignment, labels, entity_field, title, out_dir
        ),
        plot_complaint_run_alignment(run_alignment, entity_field, title, out_dir),
        plot_entity_kappa_heatmap(kappa["summary"], entity_type, title, out_dir),
    ]
    return [path for path in paths if path is not None]
