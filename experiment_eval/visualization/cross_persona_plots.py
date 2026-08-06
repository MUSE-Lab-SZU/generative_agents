"""Paper-oriented cross-persona figures."""

from __future__ import annotations

import math
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import numpy as np
from scipy.stats import t

from .scale_plots import plt, save_figure
from matplotlib.lines import Line2D

from ..schema import (
    SCALE_RANGES,
    ExperimentRecord,
    group_color,
    group_sort_key,
    kbd_color,
    label_display,
    scale_sort_key,
)
from ..statistics import labels_with_score, score_at


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
                        for value in (score_at(record, scale, label) for record in group_records)
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


def _draw_clipped_interval(
    ax: Any,
    *,
    estimate: float,
    lower: float | None,
    upper: float | None,
    y: float,
    color: str,
    limits: tuple[float, float],
) -> None:
    left, right = limits
    if lower is not None and upper is not None:
        ax.hlines(y, max(lower, left), min(upper, right), color=color, linewidth=1.8)
        if lower < left:
            ax.scatter(left, y, marker="<", color=color, s=32, clip_on=False)
        if upper > right:
            ax.scatter(right, y, marker=">", color=color, s=32, clip_on=False)
    ax.scatter(estimate, y, color=color, edgecolor="white", linewidth=0.6, s=48, zorder=3)


def _interval_text(estimate: Any, lower: Any, upper: Any) -> str:
    if estimate is None:
        return "NA"
    if lower is None or upper is None:
        return f"{estimate:.2f} [NA]"
    return f"{estimate:.2f} [{lower:.2f}, {upper:.2f}]"


def plot_persona_contrast_forest(
    rows: list[dict[str, Any]],
    out_dir: Path,
    *,
    first_group: str,
    second_group: str,
) -> Path:
    """Per-persona G1−G9 adjusted differences and standardized effects."""
    scales = sorted({row["scale"] for row in rows}, key=scale_sort_key)
    personas = sorted({row["persona"] for row in rows}, key=group_sort_key)
    fig, axes = plt.subplots(
        len(scales),
        2,
        figsize=(13.8, max(7.2, 3.8 * len(scales))),
        squeeze=False,
        constrained_layout=True,
    )
    for scale_index, scale in enumerate(scales):
        scale_rows = {row["persona"]: row for row in rows if row["scale"] == scale}
        y_positions = np.arange(len(personas))
        adjusted_limits = (-25.0, 25.0) if scale == "PHQ-9" else (-65.0, 65.0)
        specifications = [
            (
                axes[scale_index][0],
                "adjusted_endpoint_difference",
                "adjusted_ci95_lower",
                "adjusted_ci95_upper",
                adjusted_limits,
                f"{scale}: baseline-adjusted endpoint difference",
                "Agent score difference",
            ),
            (
                axes[scale_index][1],
                "hedges_g",
                "hedges_g_ci95_lower",
                "hedges_g_ci95_upper",
                (-4.0, 4.0),
                f"{scale}: Hedges' g of endpoint change",
                "Standardized mean difference",
            ),
        ]
        for ax, estimate_key, lower_key, upper_key, limits, title, xlabel in specifications:
            for y, persona in zip(y_positions, personas):
                row = scale_rows.get(persona)
                if row is None or row.get(estimate_key) is None:
                    continue
                estimate = float(row[estimate_key])
                lower = float(row[lower_key]) if row.get(lower_key) is not None else None
                upper = float(row[upper_key]) if row.get(upper_key) is not None else None
                _draw_clipped_interval(
                    ax,
                    estimate=estimate,
                    lower=lower,
                    upper=upper,
                    y=float(y),
                    color=kbd_color(persona),
                    limits=limits,
                )
                ax.text(
                    1.02,
                    y,
                    _interval_text(estimate, lower, upper),
                    transform=ax.get_yaxis_transform(),
                    ha="left",
                    va="center",
                    fontsize=7.5,
                )
            ax.axvline(0, color="#111827", linewidth=0.9)
            ax.set_xlim(*limits)
            ax.set_yticks(y_positions, personas)
            ax.invert_yaxis()
            ax.set_title(title, fontsize=10.5)
            ax.set_xlabel(xlabel)
            ax.grid(axis="x", alpha=0.2)
    fig.suptitle(f"Persona-specific exploratory contrast: {first_group} − {second_group}", fontsize=14)
    fig.text(
        0.5,
        -0.012,
        "Negative values favor the first group for symptom scores. Arrows denote 95% CIs extending beyond the "
        "plot window; full intervals are printed and exported. Adjusted CI: ANCOVA t interval; g CI: noncentral t. "
        "The 0718 endpoint is legacy immediate POST, not delayed follow-up.",
        ha="center",
        fontsize=8,
        color="#475569",
    )
    return save_figure(fig, out_dir / "figure_02_0718_g1_vs_g9_persona_forest.png")


def _heatmap(
    ax: Any,
    matrix: np.ndarray,
    *,
    personas: list[str],
    groups: list[str],
    title: str,
    cmap_name: str,
    vmin: float | None,
    vmax: float | None,
    formatter: Any,
    colorbar_label: str,
) -> None:
    masked = np.ma.masked_invalid(matrix)
    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad("#e5e7eb")
    image = ax.imshow(masked, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(groups)), groups)
    ax.set_yticks(range(len(personas)), personas)
    ax.set_title(title, fontsize=10.5)
    for row_index in range(len(personas)):
        for column_index in range(len(groups)):
            value = matrix[row_index, column_index]
            if not np.isfinite(value):
                text, color = "NA", "#475569"
            else:
                text = formatter(float(value), personas[row_index], groups[column_index])
                normalized = image.norm(float(value))
                red, green, blue, _ = image.cmap(normalized)
                luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
                color = "#111827" if luminance > 0.52 else "white"
            ax.text(column_index, row_index, text, ha="center", va="center", fontsize=7.3, color=color)
    plt.colorbar(image, ax=ax, shrink=0.78, label=colorbar_label)


def plot_outcome_heatmap(summary_rows: list[dict[str, Any]], out_dir: Path) -> Path:
    personas = sorted({row["persona"] for row in summary_rows}, key=group_sort_key)
    groups = sorted({row["group"] for row in summary_rows}, key=group_sort_key)
    by_cell = {(row["persona"], row["group"]): row for row in summary_rows}
    definitions = [
        ("phq9_mean_endpoint_change", "PHQ-9 endpoint change", "RdBu_r", "Agent score change"),
        ("bdi2_mean_endpoint_change", "BDI-II endpoint change", "RdBu_r", "Agent score change"),
        (
            "phq_bdi_direction_agreement_rate",
            "PHQ-9 / BDI-II direction agreement",
            "Blues",
            "Proportion",
        ),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(16.2, 4.6), constrained_layout=True)
    for ax, (field, title, cmap, colorbar_label) in zip(axes, definitions):
        matrix = np.asarray(
            [
                [
                    (by_cell.get((persona, group)) or {}).get(field, np.nan)
                    if (by_cell.get((persona, group)) or {}).get(field) is not None
                    else np.nan
                    for group in groups
                ]
                for persona in personas
            ],
            dtype=float,
        )
        if field.endswith("agreement_rate"):
            vmin, vmax = 0.0, 1.0

            def formatter(value: float, persona: str, group: str) -> str:
                row = by_cell[(persona, group)]
                return f"{value:.0%}\nn={row['n_direction_pairs']}"

        else:
            finite = np.abs(matrix[np.isfinite(matrix)])
            bound = max(float(finite.max()), 1.0) if finite.size else 1.0
            vmin, vmax = -bound, bound

            def formatter(value: float, persona: str, group: str) -> str:
                row = by_cell[(persona, group)]
                return f"{value:+.1f}\nn={row['n_outer_runs']}"

        _heatmap(
            ax,
            matrix,
            personas=personas,
            groups=groups,
            title=title,
            cmap_name=cmap,
            vmin=vmin,
            vmax=vmax,
            formatter=formatter,
            colorbar_label=colorbar_label,
        )
    fig.suptitle("Persona × group outcome summary (0718 protocol)", fontsize=14)
    fig.text(
        0.5,
        -0.02,
        "Cells are outer-run means; negative score change indicates improvement. Direction agreement is descriptive "
        "and has only n=2 paired outer runs per cell.",
        ha="center",
        fontsize=8,
        color="#475569",
    )
    return save_figure(fig, out_dir / "figure_03a_0718_persona_group_outcome_heatmap.png")


def plot_process_heatmap(summary_rows: list[dict[str, Any]], out_dir: Path) -> Path:
    personas = sorted({row["persona"] for row in summary_rows}, key=group_sort_key)
    groups = sorted({row["group"] for row in summary_rows}, key=group_sort_key)
    by_cell = {(row["persona"], row["group"]): row for row in summary_rows}
    definitions = [
        ("mean_completed_consultation_meetings", "Doctor consultations", "meetings"),
        ("mean_total_turns", "Recorded doctor–patient turns", "turns"),
        ("mean_environment_tasks", "Environment tasks", "tasks"),
        ("process_raw_available_rate", "Raw process availability", "proportion"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13.8, 8.0), squeeze=False, constrained_layout=True)
    for ax, (field, title, colorbar_label) in zip(axes.flat, definitions):
        matrix = np.asarray(
            [
                [
                    (by_cell.get((persona, group)) or {}).get(field, np.nan)
                    if (by_cell.get((persona, group)) or {}).get(field) is not None
                    else np.nan
                    for group in groups
                ]
                for persona in personas
            ],
            dtype=float,
        )
        finite = matrix[np.isfinite(matrix)]
        vmin = 0.0
        vmax = 1.0 if field.endswith("_rate") else (float(finite.max()) if finite.size else 1.0)

        def formatter(value: float, persona: str, group: str) -> str:
            if field.endswith("_rate"):
                return f"{value:.0%}"
            return f"{value:.1f}"

        _heatmap(
            ax,
            matrix,
            personas=personas,
            groups=groups,
            title=title,
            cmap_name="YlGnBu",
            vmin=vmin,
            vmax=max(vmax, 1e-9),
            formatter=formatter,
            colorbar_label=colorbar_label,
        )
    fig.suptitle("Persona × group process-data summary (0718 protocol)", fontsize=14)
    fig.text(
        0.5,
        -0.02,
        "NA means the archived extractor input is absent, not zero. Zero consultations for resident-chat groups "
        "describes the group design and does not mean zero social interaction; resident-chat dose is not yet normalized.",
        ha="center",
        fontsize=8,
        color="#475569",
    )
    return save_figure(fig, out_dir / "figure_03b_0718_persona_group_process_heatmap.png")


def plot_leave_one_persona_out(
    rows: list[dict[str, Any]],
    out_dir: Path,
    *,
    first_group: str,
    second_group: str,
) -> Path:
    scales = sorted({row["scale"] for row in rows}, key=scale_sort_key)
    order = ["NONE", *sorted({row["omitted_persona"] for row in rows if row["omitted_persona"] != "NONE"}, key=group_sort_key)]
    labels = ["All personas", *[f"Without {persona}" for persona in order[1:]]]
    fig, axes = plt.subplots(1, len(scales), figsize=(12.8, 4.8), squeeze=False, constrained_layout=True)
    for column, scale in enumerate(scales):
        ax = axes[0][column]
        scale_rows = {row["omitted_persona"]: row for row in rows if row["scale"] == scale}
        finite_bounds = [
            abs(float(value))
            for row in scale_rows.values()
            for value in (row.get("ci95_lower"), row.get("ci95_upper"))
            if value is not None and math.isfinite(float(value))
        ]
        bound = max(finite_bounds, default=1.0) * 1.08
        limits = (-bound, bound)
        for y, omitted in enumerate(order):
            row = scale_rows.get(omitted)
            if row is None or row.get("adjusted_endpoint_difference") is None:
                continue
            estimate = float(row["adjusted_endpoint_difference"])
            lower = float(row["ci95_lower"]) if row.get("ci95_lower") is not None else None
            upper = float(row["ci95_upper"]) if row.get("ci95_upper") is not None else None
            color = "#111827" if omitted == "NONE" else kbd_color(omitted)
            _draw_clipped_interval(
                ax,
                estimate=estimate,
                lower=lower,
                upper=upper,
                y=float(y),
                color=color,
                limits=limits,
            )
        ax.axvline(0, color="#111827", linewidth=0.9)
        ax.set_xlim(*limits)
        ax.set_yticks(range(len(labels)), labels)
        ax.invert_yaxis()
        ax.set_title(scale)
        ax.set_xlabel("Baseline-adjusted endpoint difference")
        ax.grid(axis="x", alpha=0.2)
    fig.suptitle(f"Leave-one-persona-out sensitivity: {first_group} − {second_group}", fontsize=14)
    fig.text(
        0.5,
        -0.015,
        "Endpoint ~ group + baseline + persona fixed effects. This checks influence of three selected cases; "
        "it is not a jackknife estimate for a sampled persona population.",
        ha="center",
        fontsize=8,
        color="#475569",
    )
    return save_figure(fig, out_dir / "figure_s01_0718_g1_vs_g9_leave_one_persona_out.png")


def plot_variance_partition(rows: list[dict[str, Any]], out_dir: Path) -> Path:
    scales = sorted({row["scale"] for row in rows}, key=scale_sort_key)
    sources = [
        "group_bundle",
        "persona_within_group",
        "outer_run_within_cell",
        "common_time_profile",
        "run_by_time_trajectory",
        "measurement_noise",
    ]
    labels = {
        row["source"]: row["source_label"]
        for row in rows
    }
    colors = {
        "group_bundle": "#64748b",
        "persona_within_group": "#7c3aed",
        "outer_run_within_cell": "#0f766e",
        "common_time_profile": "#2563eb",
        "run_by_time_trajectory": "#f59e0b",
        "measurement_noise": "#dc2626",
    }
    by_key = {(row["scale"], row["source"]): row for row in rows}
    scenarios = [
        (scale, "single", "proportion_of_observed_variance")
        for scale in scales
    ] + [
        (scale, "K mean", "proportion_of_k_mean_variance")
        for scale in scales
    ]
    scenarios.sort(key=lambda item: (scale_sort_key(item[0]), item[1] != "single"))
    x = np.asarray(
        [
            scale_index * 2.6 + scenario_index * 0.9
            for scale_index, scale in enumerate(scales)
            for scenario_index in range(2)
        ],
        dtype=float,
    )
    fig, ax = plt.subplots(figsize=(11.4, 5.7), constrained_layout=True)
    bottom = np.zeros(len(scenarios), dtype=float)
    for source in sources:
        values = np.asarray(
            [
                float((by_key.get((scale, source)) or {}).get(field) or 0.0)
                for scale, _, field in scenarios
            ],
            dtype=float,
        )
        bars = ax.bar(
            x,
            values,
            width=0.72,
            bottom=bottom,
            color=colors[source],
            label=labels.get(source, source),
        )
        for bar, value, base in zip(bars, values, bottom):
            if value >= 0.04:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    base + value / 2,
                    f"{value:.0%}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if source not in {"common_time_profile"} else "#111827",
                )
        bottom += values
    tick_labels = []
    for scale, scenario, _ in scenarios:
        k_value = (by_key.get((scale, "measurement_noise")) or {}).get("k_values", "K")
        tick_labels.append(
            f"{scale}\nsingle generation"
            if scenario == "single"
            else f"{scale}\nK={k_value} mean"
        )
    ax.set_xticks(x, tick_labels)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Share of observed variance")
    ax.set_title("Descriptive hierarchical variance partition (0727 protocol)")
    ax.grid(axis="y", alpha=0.18)
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False, fontsize=8)
    fig.text(
        0.5,
        -0.015,
        "Single-generation bars exactly partition raw observed variance. K-mean bars divide only within-snapshot "
        "measurement variance by K (independent-repeat assumption). Not REML population variance components.",
        ha="center",
        fontsize=8,
        color="#475569",
    )
    return save_figure(fig, out_dir / "figure_s02_0727_variance_decomposition.png")
