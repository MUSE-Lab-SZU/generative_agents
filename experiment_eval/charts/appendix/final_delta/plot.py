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


def _appendix_final_delta(
    records: list[ExperimentRecord],
    labels: list[str],
    scales: list[str],
    out_dir: Path,
    title_prefix: str,
) -> Path:
    severities = available_severities(records)
    fig, axes = _grid(len(scales), len(severities), height=3.7)
    for row, scale in enumerate(scales):
        for column, severity in enumerate(severities):
            ax = axes[row][column]
            panel = [record for record in records if record.severity == severity]
            values = [
                trajectory_metrics(record, labels, scale).get("endpoint_change")
                for record in panel
            ]
            bars = ax.bar(
                range(len(panel)),
                values,
                color=[_series_color(record, panel) for record in panel],
                alpha=0.86,
            )
            ax.axhline(0, color="#111827", linewidth=0.8)
            ax.set_xticks(
                range(len(panel)),
                [record.plot_label for record in panel],
                rotation=25,
                ha="right",
            )
            ax.set_title(f"{scale} final delta — {severity}")
            ax.set_ylabel("Observed endpoint − baseline")
            ax.grid(axis="y", alpha=0.22)
            for bar, value in zip(bars, values):
                if value is not None:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        value,
                        f"{value:.1f}",
                        ha="center",
                        va="bottom" if value >= 0 else "top",
                        fontsize=8,
                    )
    fig.suptitle(_wrapped(f"{title_prefix}: final change by severity"), fontsize=13)
    _footer(fig, records, "none", labels)
    return save_figure(fig, out_dir / "01_final_delta_bars_by_severity.png")
