"""Compatibility facade for trajectory charts."""

from ....charts.outcomes.trajectory_ci.plot import (
    _draw_outer_summary,
    plot_trajectory_with_ci,
)
from ....charts.outcomes.trajectory_lines import plot_trajectory_lines_only
from ....workflows.figure_families import render_lines_only
