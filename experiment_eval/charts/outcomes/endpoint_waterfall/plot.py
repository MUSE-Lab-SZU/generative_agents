"""Endpoint-change, rebound, and outer-run group-comparison figures."""

from __future__ import annotations

import math

from pathlib import Path

import numpy as np

from ....loader import available_severities

from ....schema import ExperimentRecord, group_color, label_display, slugify

from ....statistics import (
    baseline_adjusted_endpoint_contrasts,
    cross_scale_convergence,
    delta_ci,
    group_endpoint_contrasts,
    group_time_contrasts,
    trajectory_metrics,
)

from ...shared.plotting import (
    add_measurement_footer as _footer,
    figure_grid as _grid,
    Line2D,
    plt,
    save_figure,
    series_color as _series_color,
    wrapped as _wrapped,
)


def plot_endpoint_waterfall(
    records: list[ExperimentRecord],
    labels: list[str],
    scales: list[str],
    out_dir: Path,
    title_prefix: str,
) -> list[Path]:
    paths: list[Path] = []
    include_kbd = len({record.kbd for record in records}) > 1
    for scale in scales:
        rows = []
        for record in records:
            change = trajectory_metrics(record, labels, scale).get("endpoint_change")
            if change is not None:
                rows.append((record, float(change)))
        if not rows:
            continue
        rows.sort(key=lambda item: item[1])
        fig, ax = plt.subplots(
            figsize=(max(7.4, len(rows) * 0.65), 4.8), constrained_layout=True
        )
        bars = ax.bar(
            range(len(rows)),
            [value for _, value in rows],
            color=[group_color(record.group) for record, _ in rows],
            alpha=0.85,
        )
        ax.axhline(0, color="#111827", linewidth=0.9)
        if scale == "PHQ-9":
            ax.axhline(
                5,
                color="#dc2626",
                linewidth=0.9,
                linestyle="--",
                label="worsening proxy +5",
            )
            ax.axhline(
                -5,
                color="#2563eb",
                linewidth=0.9,
                linestyle="--",
                label="improvement proxy −5",
            )
            ax.legend(frameon=False, fontsize=8)
        ax.set_xticks(
            range(len(rows)),
            [
                "-".join(
                    part
                    for part in (
                        record.kbd if include_kbd else None,
                        record.group,
                        record.repeat_id,
                    )
                    if part
                )
                for record, _ in rows
            ],
            rotation=35,
            ha="right",
        )
        ax.set_ylabel("Observed endpoint − baseline")
        ax.set_title(f"{title_prefix}: {scale} outer-run waterfall")
        ax.grid(axis="y", alpha=0.2)
        for bar, (_, value) in zip(bars, rows):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value,
                f"{value:.1f}",
                ha="center",
                va="bottom" if value >= 0 else "top",
                fontsize=8,
            )
        fig.text(
            0.5,
            -0.02,
            "Threshold lines are transferred human-scale conventions used only as Agent simulation proxies. "
            "They do not indicate clinical response, remission, or harm.",
            ha="center",
            fontsize=8,
            color="#475569",
        )
        paths.append(
            save_figure(
                fig, out_dir / f"core_04_{slugify(scale)}_endpoint_waterfall.png"
            )
        )
    return paths
