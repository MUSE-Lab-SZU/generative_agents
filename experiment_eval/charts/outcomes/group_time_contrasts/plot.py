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


def plot_group_time_contrasts(
    records: list[ExperimentRecord],
    labels: list[str],
    scales: list[str],
    out_dir: Path,
    title_prefix: str,
) -> Path | None:
    rows = group_time_contrasts(records, labels, scales)
    contrasts = sorted({row["contrast"] for row in rows})
    if not contrasts:
        return None
    fig, axes = plt.subplots(
        len(scales),
        len(contrasts),
        figsize=(max(7.5, 5.3 * len(contrasts)), max(4.2, 3.8 * len(scales))),
        squeeze=False,
        constrained_layout=True,
    )
    plotted = False
    for scale_index, scale in enumerate(scales):
        for contrast_index, contrast in enumerate(contrasts):
            ax = axes[scale_index][contrast_index]
            panel = [
                row
                for row in rows
                if row["scale"] == scale and row["contrast"] == contrast
            ]
            x_values, centers, lower, upper = [], [], [], []
            for label_index, label in enumerate(labels):
                row = next(
                    (value for value in panel if value["timepoint"] == label), None
                )
                if row is None:
                    continue
                interval = row["difference_in_change"]
                if interval.get("estimate") is None:
                    continue
                plotted = True
                x_values.append(label_index)
                centers.append(interval["estimate"])
                lower.append(interval.get("lower"))
                upper.append(interval.get("upper"))
            if centers:
                ax.plot(x_values, centers, marker="o", color="#334155", linewidth=1.8)
                if all(value is not None for value in lower + upper):
                    ax.fill_between(x_values, lower, upper, color="#64748b", alpha=0.18)
            ax.axhline(0, color="#111827", linewidth=0.8)
            ax.set_xticks(
                range(len(labels)),
                [label_display(label) for label in labels],
                rotation=25,
            )
            ax.set_ylabel("Difference in change")
            ax.set_title(f"{scale}: {contrast}")
            ax.grid(alpha=0.2)
    if not plotted:
        plt.close(fig)
        return None
    fig.suptitle(f"{title_prefix}: outer-run baseline-referenced time contrasts")
    fig.text(
        0.5,
        -0.018,
        "Welch 95% CI across outer simulation runs at each observed timepoint. "
        "This is a descriptive group×time analogue, not a fitted confirmatory mixed model.",
        ha="center",
        fontsize=8,
        color="#475569",
    )
    return save_figure(fig, out_dir / "core_00_group_time_contrasts.png")
