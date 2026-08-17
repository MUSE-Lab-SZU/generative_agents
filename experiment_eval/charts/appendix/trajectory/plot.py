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


def _appendix_trajectory(
    records: list[ExperimentRecord],
    labels: list[str],
    scale: str,
    out_dir: Path,
    title_prefix: str,
    index: int,
) -> Path:
    severities = available_severities(records)
    fig, axes = _grid(1, len(severities), height=4.3)
    for column, severity in enumerate(severities):
        ax = axes[0][column]
        for record in [item for item in records if item.severity == severity]:
            available = labels_with_score(record, labels, scale)
            xs = [labels.index(label) for label in available]
            ys = [score_at(record, scale, label) for label in available]
            marker, linestyle = repeat_style(record.repeat_id)
            ax.plot(
                xs,
                ys,
                color=_series_color(record, records),
                marker=marker,
                linestyle=linestyle,
                linewidth=2,
            )
        ax.set_xticks(
            range(len(labels)),
            [label_display(label) for label in labels],
            rotation=25 if len(labels) > 7 else 0,
        )
        ax.set_title(severity)
        ax.set_ylabel("Primary score")
        ax.grid(alpha=0.22)
    handles = _series_legend(records)
    if handles:
        fig.legend(
            handles=handles,
            loc="center left",
            bbox_to_anchor=(1.0, 0.5),
            ncol=1,
            frameon=False,
            fontsize=8,
        )
    fig.suptitle(
        _wrapped(f"{title_prefix}: {scale} trajectory by severity"), fontsize=13
    )
    _footer(fig, records, "none", labels)
    return save_figure(
        fig, out_dir / f"{index:02d}_{slugify(scale)}_trajectory_by_severity.png"
    )
