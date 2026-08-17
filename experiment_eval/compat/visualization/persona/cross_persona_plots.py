"""Compatibility facade for cross-persona charts."""

from __future__ import annotations

from ....charts.persona.contrast_forest import plot_persona_contrast_forest
from ....charts.persona.leave_one_out import plot_leave_one_persona_out
from ....charts.persona.outcome_heatmap import plot_outcome_heatmap
from ....charts.persona.process_heatmap import plot_process_heatmap
from ....charts.persona.scale_trajectories import plot_persona_scale_trajectories
from ....charts.persona.variance_partition import plot_variance_partition

__all__ = [
    "plot_leave_one_persona_out",
    "plot_outcome_heatmap",
    "plot_persona_contrast_forest",
    "plot_persona_scale_trajectories",
    "plot_process_heatmap",
    "plot_variance_partition",
]
