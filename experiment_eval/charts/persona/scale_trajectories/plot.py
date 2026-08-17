"""Cross-persona longitudinal trajectory figures."""

from __future__ import annotations

import math

from pathlib import Path

from statistics import mean, stdev

import numpy as np

from scipy.stats import t

from ....schema import (
    SCALE_RANGES,
    ExperimentRecord,
    group_color,
    group_sort_key,
    label_display,
    scale_sort_key,
)

from ....statistics import labels_with_score, score_at

from ...shared.plotting import Line2D, plt, save_figure


def _outer_mean_ci(values: list[float]) -> tuple[float, float, float]:
    center = mean(values)
    if len(values) < 2:
        return center, center, center
    half = float(t.ppf(0.975, len(values) - 1)) * stdev(values) / math.sqrt(len(values))
    return center, center - half, center + half


def plot_persona_scale_trajectories(
    records: list[ExperimentRecord],
    labels: list[str],
    scales: list[str],
    out_dir: Path,
) -> Path:
    """KBD2/KBD4/KBD6 × PHQ-9/BDI-II faceted outer-run trajectories."""
    personas = sorted({record.kbd for record in records}, key=group_sort_key)
    ordered_scales = sorted(scales, key=scale_sort_key)
    groups = sorted({record.group for record in records}, key=group_sort_key)
    fig, axes = plt.subplots(
        len(personas),
        len(ordered_scales),
        figsize=(12.8, 10.2),
        sharex=True,
        squeeze=False,
        constrained_layout=True,
    )
    x_all = np.arange(len(labels))
    for row_index, persona in enumerate(personas):
        for column_index, scale in enumerate(ordered_scales):
            ax = axes[row_index][column_index]
            panel = [record for record in records if record.kbd == persona]
            counts: list[str] = []
            for group in groups:
                group_records = [record for record in panel if record.group == group]
                if not group_records:
                    continue
                counts.append(f"{group} n={len(group_records)}")
                for record in group_records:
                    available = labels_with_score(record, labels, scale)
                    xs = [labels.index(label) for label in available]
                    ys = [float(score_at(record, scale, label)) for label in available]
                    ax.plot(xs, ys, color=group_color(group), alpha=0.16, linewidth=1.0)

                xs: list[int] = []
                centers: list[float] = []
                lowers: list[float] = []
                uppers: list[float] = []
                for label_index, label in enumerate(labels):
                    values = [
                        float(value)
                        for value in (
                            score_at(record, scale, label) for record in group_records
                        )
                        if value is not None
                    ]
                    if not values:
                        continue
                    center, lower, upper = _outer_mean_ci(values)
                    xs.append(label_index)
                    centers.append(center)
                    lowers.append(lower)
                    uppers.append(upper)
                ax.plot(
                    xs,
                    centers,
                    color=group_color(group),
                    marker="o",
                    markersize=4.5,
                    linewidth=2.3,
                    zorder=3,
                )
                ax.fill_between(
                    xs,
                    lowers,
                    uppers,
                    color=group_color(group),
                    alpha=0.13,
                    linewidth=0,
                    zorder=2,
                )
            if row_index == 0:
                ax.set_title(scale, fontsize=12)
            if column_index == 0:
                ax.set_ylabel(f"{persona}\nAgent score")
            else:
                ax.set_ylabel("Agent score")
            ax.set_ylim(*SCALE_RANGES[scale])
            ax.set_xticks(x_all, [label_display(label) for label in labels])
            ax.grid(axis="y", alpha=0.18)
            ax.text(
                0.98,
                0.95,
                ", ".join(counts),
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=7.5,
                color="#475569",
            )
    handles = [
        Line2D(
            [0],
            [0],
            color=group_color(group),
            marker="o",
            linewidth=2.4,
            label=group,
        )
        for group in groups
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.966),
        ncol=max(1, len(handles)),
        frameon=False,
    )
    fig.suptitle(
        "Longitudinal Agent symptom-scale trajectories by persona and setting",
        fontsize=14,
        y=1.015,
    )
    fig.text(
        0.5,
        -0.012,
        "Thin lines: independent outer runs. Thick lines and bands: outer-run mean and t 95% CI. "
        "K=10 repeated generations estimate each frozen-snapshot mean only.",
        ha="center",
        fontsize=8,
        color="#475569",
    )
    return save_figure(fig, out_dir / "figure_01_0727_persona_scale_trajectory.png")
