"""Backward-compatible appendix figures."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ....loader import available_severities

from ....schema import ExperimentRecord, label_display, repeat_style, slugify

from ....statistics import (
    endpoint_label_for_record,
    labels_with_score,
    score_at,
    stat_at,
    trajectory_metrics,
)

from ...shared.plotting import (
    add_measurement_footer as _footer,
    figure_grid as _grid,
    heatmap_text_color as _text_color,
    Line2D,
    plt,
    save_figure,
    series_color as _series_color,
    series_legend as _series_legend,
    wrapped as _wrapped,
)


def _appendix_delta_heatmap(
    records: list[ExperimentRecord],
    labels: list[str],
    scales: list[str],
    out_dir: Path,
    title_prefix: str,
    index: int,
) -> Path:
    fig, axes = _grid(
        1, len(scales), width=4.8, height=max(4.0, 0.5 * len(records) + 2.0)
    )
    for column, scale in enumerate(scales):
        ax = axes[0][column]
        values = np.asarray(
            [
                [trajectory_metrics(record, labels, scale).get("endpoint_change")]
                for record in records
            ],
            dtype=float,
        )
        finite = values[np.isfinite(values)]
        max_abs = (
            max(abs(float(finite.min())), abs(float(finite.max())))
            if finite.size
            else 1.0
        )
        if max_abs == 0:
            max_abs = 1.0
        im = ax.imshow(
            values, cmap="RdBu_r", vmin=-max_abs, vmax=max_abs, aspect="auto"
        )
        ax.set_xticks([0], [scale])
        ax.set_yticks(range(len(records)), [record.plot_label for record in records])
        ax.set_title(f"{scale} raw delta\n(independent color range)")
        for row, value in enumerate(values[:, 0]):
            if np.isfinite(value):
                ax.text(
                    0,
                    row,
                    f"{value:.1f}",
                    ha="center",
                    va="center",
                    color=_text_color(im.cmap, im.norm, value),
                )
        fig.colorbar(im, ax=ax, label="Endpoint − baseline")
    fig.suptitle(
        _wrapped(f"{title_prefix}: final delta heatmap (scale-specific raw ranges)"),
        fontsize=13,
    )
    _footer(fig, records, "none", labels)
    return save_figure(fig, out_dir / f"{index:02d}_final_delta_heatmap.png")
