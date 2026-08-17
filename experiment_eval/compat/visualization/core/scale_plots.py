"""Compatibility facade for the historical scale plotting API."""

from ....charts.shared.plotting import plt, save_figure, series_color as _series_color
from ....workflows.figure_families import (
    render_appendix,
    render_lines_only,
    render_presentation,
)
from .outcome_plots import plot_best_change_and_rebound, plot_endpoint_delta_with_ci
from .reliability_plots import (
    MEASUREMENT_METRICS,
    plot_measurement_reliability_heatmaps,
)
from .trajectory_plots import plot_trajectory_lines_only, plot_trajectory_with_ci
