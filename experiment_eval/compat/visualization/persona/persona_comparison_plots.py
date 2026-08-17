"""Compatibility facade for persona-comparison charts."""

from ....charts.shared.persona import _draw_clipped_interval, _interval_text
from ....charts.persona.contrast_forest import plot_persona_contrast_forest
from ....charts.persona.leave_one_out import plot_leave_one_persona_out
from ....charts.persona.variance_partition import plot_variance_partition
