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


def _appendix_volatility(
    records: list[ExperimentRecord],
    labels: list[str],
    scales: list[str],
    out_dir: Path,
    title_prefix: str,
    index: int,
) -> Path:
    severities = available_severities(records)
    fig, axes = _grid(len(scales), len(severities), height=3.7)
    definitions = [
        ("trajectory_volatility", "Trajectory volatility", "#64748b"),
        ("max_upward_step", "Max upward step", "#ef4444"),
    ]
    for row, scale in enumerate(scales):
        for column, severity in enumerate(severities):
            ax = axes[row][column]
            panel = [record for record in records if record.severity == severity]
            x = np.arange(len(panel))
            for metric_index, (field, label, color) in enumerate(definitions):
                values = [
                    trajectory_metrics(record, labels, scale).get(field)
                    for record in panel
                ]
                ax.bar(
                    x + (metric_index - 0.5) * 0.35,
                    values,
                    width=0.35,
                    color=color,
                    label=label,
                    alpha=0.85,
                )
            ax.set_xticks(
                x, [record.plot_label for record in panel], rotation=25, ha="right"
            )
            ax.set_title(f"{scale} — {severity}")
            ax.set_ylabel("Score units")
            ax.grid(axis="y", alpha=0.22)
    fig.legend(
        handles=[
            Line2D([0], [0], color=color, linewidth=8, label=label)
            for _, label, color in definitions
        ],
        loc="center left",
        bbox_to_anchor=(1.0, 0.5),
        ncol=1,
        frameon=False,
    )
    fig.suptitle(
        _wrapped(f"{title_prefix}: trajectory volatility and upward steps"), fontsize=13
    )
    _footer(fig, records, "none", labels)
    return save_figure(fig, out_dir / f"{index:02d}_stability_rebound_by_severity.png")
