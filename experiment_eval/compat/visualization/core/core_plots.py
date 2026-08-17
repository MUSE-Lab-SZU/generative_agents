"""Compatibility facade for paper-oriented core charts."""

from pathlib import Path
from typing import Any

from ....charts.process import delivery as _process_api
from ....charts.process.delivery import plot as _process_plot
from ....charts.shared.plotting import plt, save_figure
from ....workflows.figure_families import render_core_figures
from .outcome_plots import (
    plot_cross_scale_convergence,
    plot_endpoint_waterfall,
    plot_group_time_contrasts,
    plot_outer_contrast_forest,
)
from .reliability_plots import plot_measurement_icc


def plot_process_delivery(
    process_rows: list[dict[str, Any]],
    stage_rows: list[dict[str, Any]],
    out_dir: Path,
    title_prefix: str,
) -> list[Path]:
    """Preserve historical module-level Matplotlib monkeypatch points."""
    _process_plot.plt = plt
    _process_plot.save_figure = save_figure
    return _process_api.plot_process_delivery(
        process_rows, stage_rows, out_dir, title_prefix
    )
