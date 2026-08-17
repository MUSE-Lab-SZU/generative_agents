"""Endpoint-change, rebound, and outer-run group-comparison figures."""

from __future__ import annotations

import math

from pathlib import Path

import numpy as np

from ....loader import available_severities

from ....schema import ExperimentRecord, group_color, group_sort_key, kbd_color, label_display, slugify

from ....statistics import (
    baseline_adjusted_endpoint_contrasts,
    cross_scale_concurrent_validity,
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


def plot_cross_scale_convergence(
    records: list[ExperimentRecord],
    labels: list[str],
    out_dir: Path,
    title_prefix: str,
    *,
    entity_field: str = "group",
    dimension_label: str | None = None,
    filename_suffix: str | None = None,
    fit_lines: bool = False,
) -> Path | None:
    result = cross_scale_convergence(records, labels)
    rows = result["rows"]
    if not rows:
        return None
    fig, ax = plt.subplots(figsize=(7.4, 6.2), constrained_layout=True)
    if entity_field not in {"group", "persona"}:
        raise ValueError("entity_field must be 'group' or 'persona'")
    entities = sorted({str(row[entity_field]) for row in rows}, key=group_sort_key)
    handles: list[Line2D] = []
    for entity in entities:
        subset = [row for row in rows if str(row[entity_field]) == entity]
        color = group_color(entity) if entity_field == "group" else kbd_color(entity)
        agreed = [row for row in subset if row["direction_agrees"]]
        discordant = [row for row in subset if not row["direction_agrees"]]
        if agreed:
            ax.scatter(
                [row["first_delta"] for row in agreed],
                [row["second_delta"] for row in agreed],
                color=color,
                marker="o",
                s=52,
                alpha=0.82,
            )
        if discordant:
            ax.scatter(
                [row["first_delta"] for row in discordant],
                [row["second_delta"] for row in discordant],
                color=color,
                marker="x",
                s=60,
                linewidth=1.6,
                alpha=0.9,
            )
        if fit_lines:
            x = np.asarray([row["first_delta"] for row in subset], dtype=float)
            y = np.asarray([row["second_delta"] for row in subset], dtype=float)
            if len(x) >= 2 and len(set(x)) > 1:
                slope, intercept = np.polyfit(x, y, 1)
                line_x = np.linspace(float(x.min()), float(x.max()), 60)
                ax.plot(line_x, slope * line_x + intercept, color=color, linewidth=1.8, alpha=0.9)
        handles.append(Line2D([0], [0], marker="o", color=color, linewidth=1.8 if fit_lines else 0, label=f"{entity} (n={len(subset)})", markersize=6))
    if any(not row["direction_agrees"] for row in rows):
        handles.append(Line2D([0], [0], marker="x", color="#475569", linewidth=0, label="discordant direction", markersize=7))
    ax.axhline(0, color="#64748b", linewidth=0.8)
    ax.axvline(0, color="#64748b", linewidth=0.8)
    ax.set_xlabel("ΔPHQ-9 (common endpoint − common baseline)")
    ax.set_ylabel("ΔBDI-II (common endpoint − common baseline)")
    rho = result["spearman_rho"]
    agreement = result["direction_agreement_rate"]
    lower, upper = result.get("ci95_lower"), result.get("ci95_upper")
    interval = f" [{lower:.2f}, {upper:.2f}]" if lower is not None and upper is not None else ""
    subtitle = (
        f"direction agreement={agreement:.1%}; Spearman ρ={rho:.2f}{interval}; p={result['spearman_p']:.3g}"
        if agreement is not None and rho is not None
        else (
            f"direction agreement={agreement:.1%}"
            if agreement is not None
            else "insufficient paired scales"
        )
    )
    heading = dimension_label or title_prefix
    ax.set_title(f"{heading}: treatment-change agreement\n{subtitle}")
    ax.grid(alpha=0.18)
    ax.legend(handles=handles, frameon=False, fontsize=8, ncol=2)
    fig.text(
        0.5,
        -0.015,
        "Each point is one independent outer run using the same baseline and endpoint for both scales. "
        "Negative=improvement; positive=worsening; ×=discordant direction. "
        + ("Lines are descriptive within-group OLS fits." if fit_lines else ""),
        ha="center",
        fontsize=8,
        color="#475569",
    )
    filename = "core_02b_cross_scale_change_agreement"
    if filename_suffix:
        filename += f"_{filename_suffix}"
    return save_figure(fig, out_dir / f"{filename}.png")


def plot_cross_scale_concurrent_validity(
    records: list[ExperimentRecord],
    labels: list[str],
    out_dir: Path,
    title_prefix: str,
    *,
    entity_field: str = "group",
    dimension_label: str | None = None,
    filename_suffix: str | None = None,
    fit_lines: bool = False,
) -> Path | None:
    result = cross_scale_concurrent_validity(records, labels)
    rows = result["rows"]
    if not rows:
        return None
    fig, ax = plt.subplots(figsize=(7.4, 6.2), constrained_layout=True)
    if entity_field not in {"group", "persona"}:
        raise ValueError("entity_field must be 'group' or 'persona'")
    for entity in sorted({str(row[entity_field]) for row in rows}, key=group_sort_key):
        subset = [row for row in rows if str(row[entity_field]) == entity]
        color = group_color(entity) if entity_field == "group" else kbd_color(entity)
        ax.scatter(
            [row["first_total"] for row in subset],
            [row["second_total"] for row in subset],
            color=color,
            s=30,
            alpha=0.55,
            label=f"{entity} (n={len(subset)})",
        )
        if fit_lines:
            x = np.asarray([row["first_total"] for row in subset], dtype=float)
            y = np.asarray([row["second_total"] for row in subset], dtype=float)
            if len(x) >= 2 and len(set(x)) > 1:
                slope, intercept = np.polyfit(x, y, 1)
                line_x = np.linspace(float(x.min()), float(x.max()), 80)
                ax.plot(line_x, slope * line_x + intercept, color=color, linewidth=1.9, alpha=0.92)
    rho = result["spearman_rho"]
    lower, upper = result.get("ci95_lower"), result.get("ci95_upper")
    subtitle = f"Spearman ρ={rho:.2f}" if rho is not None else "insufficient variation"
    if lower is not None and upper is not None:
        subtitle += f"; cluster 95% CI [{lower:.2f}, {upper:.2f}]"
    ax.set_xlabel("PHQ-9 total score")
    ax.set_ylabel("BDI-II total score")
    heading = dimension_label or title_prefix
    ax.set_title(f"{heading}: cross-scale convergent validity\n{subtitle}")
    ax.grid(alpha=0.18)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.text(
        0.5,
        -0.015,
        f"Aligned same persona/condition/timepoint (n={result['n_aligned_snapshots']}); CI resamples outer runs. "
        "Naive p is omitted because timepoints within a run are repeated observations. "
        + ("Lines are descriptive within-group OLS fits." if fit_lines else ""),
        ha="center",
        fontsize=8,
        color="#475569",
    )
    filename = "core_02a_cross_scale_concurrent_validity"
    if filename_suffix:
        filename += f"_{filename_suffix}"
    return save_figure(fig, out_dir / f"{filename}.png")
