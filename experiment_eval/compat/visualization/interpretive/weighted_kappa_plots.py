"""Compatibility facade and renderer for weighted-kappa charts."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from ....charts.kappa.item_forest import plot_item_heatmap
from ....charts.kappa.repeat_pair_heatmap import plot_repeat_pair_heatmap
from ....charts.kappa.stratum_heatmap import plot_stratum_heatmap


def render_weighted_kappa_figures(
    payload: dict[str, Any], out_dir: Path, title_prefix: str
) -> list[Path]:
    """Render one compact figure set; skip sparse persona/group panels."""
    paths = [
        plot_repeat_pair_heatmap(payload, out_dir, title_prefix),
        plot_item_heatmap(payload, out_dir, title_prefix),
        plot_stratum_heatmap(payload, out_dir, title_prefix),
    ]
    return [path for path in paths if path is not None]
