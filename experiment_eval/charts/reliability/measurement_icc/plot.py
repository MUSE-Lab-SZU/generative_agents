"""Frozen-snapshot repeat reliability figures."""

from __future__ import annotations

import math

from pathlib import Path

from typing import Any

import numpy as np

from ....schema import ExperimentRecord, group_sort_key, label_display, slugify

from ....statistics import measurement_error_summary, measurement_icc, stat_at

from ...shared.plotting import (
    heatmap_text_color as _text_color,
    plt,
    save_figure,
    wrapped as _wrapped,
)

MEASUREMENT_METRICS = {
    "sample_sd": "Sample SD",
    "ci95_width": "95% CI width",
    "category_pairwise_flip_rate": "Category pairwise flip rate",
}


def plot_measurement_icc(
    records: list[ExperimentRecord],
    labels: list[str],
    scales: list[str],
    out_dir: Path,
    title_prefix: str,
) -> Path | None:
    rows = [(scale, measurement_icc(records, labels, scale)) for scale in scales]
    rows = [(scale, result) for scale, result in rows if result["status"] == "ok"]
    if not rows:
        return None
    fig, ax = plt.subplots(figsize=(8.4, max(4.0, len(rows) * 1.35 + 2.0)), constrained_layout=True)
    y = np.arange(len(rows))
    offset = 0.13
    for index, (key, label, color, marker) in enumerate(
        (("icc_a_1", "ICC(A,1): single repeat", "#2563eb", "o"),
         ("icc_a_k", "ICC(A,K): mean of K repeats", "#b45309", "s"))
    ):
        positions = y + (-offset if index == 0 else offset)
        estimates = np.asarray([result[key] for _, result in rows], dtype=float)
        lower = np.asarray([
            result[key] if result.get(f"{key}_ci95_lower") is None else result[f"{key}_ci95_lower"]
            for _, result in rows
        ], dtype=float)
        upper = np.asarray([
            result[key] if result.get(f"{key}_ci95_upper") is None else result[f"{key}_ci95_upper"]
            for _, result in rows
        ], dtype=float)
        ax.errorbar(
            estimates,
            positions,
            xerr=np.vstack((estimates - lower, upper - estimates)),
            fmt=marker,
            color=color,
            ecolor=color,
            capsize=3,
            linewidth=1.2,
            label=label,
        )
    ax.set_yticks(
        y,
        [
            f"{scale} (targets={result['n_snapshot_targets']}, K={result['k_measurement']})"
            for scale, result in rows
        ],
    )
    finite_bounds = [
        float(value)
        for _, result in rows
        for value in (result.get("icc_a_1_ci95_lower"), result.get("icc_a_k_ci95_lower"))
        if value is not None
    ]
    ax.set_xlim(min(-0.2, min(finite_bounds, default=0.0) - 0.05), 1.02)
    ax.axvline(0, color="#64748b", linewidth=0.8)
    ax.set_xlabel("Intraclass correlation")
    ax.set_title(f"{title_prefix}: total-score repeat reliability")
    ax.legend(frameon=False)
    ax.grid(axis="x", alpha=0.2)
    fig.text(
        0.5,
        -0.015,
        "Two-way random absolute agreement (ICC(A,1)/ICC(2,1); ICC(A,K)/ICC(2,K)); "
        "95% CI: outer-run cluster bootstrap. Frozen-snapshot generation reliability, not clinical test-retest reliability.",
        ha="center",
        fontsize=8,
        color="#475569",
    )
    return save_figure(fig, out_dir / "core_03_measurement_icc.png")


def plot_measurement_icc_by_entity(
    records: list[ExperimentRecord],
    labels: list[str],
    scales: list[str],
    out_dir: Path,
    title_prefix: str,
    *,
    entity_field: str,
    dimension_label: str,
    filename_suffix: str,
) -> tuple[Path | None, list[dict[str, Any]]]:
    """Reference-style ICC(A,1) forest with entity and overall rows."""
    if entity_field not in {"group", "kbd"}:
        raise ValueError("entity_field must be 'group' or 'kbd'")
    entities = sorted(
        {str(getattr(record, entity_field)) for record in records},
        key=group_sort_key,
    )
    result_rows: list[dict[str, Any]] = []
    display_rows: list[dict[str, Any]] = []
    for scale in scales:
        display_rows.append({"kind": "header", "label": scale})
        for entity in entities:
            selected = [record for record in records if str(getattr(record, entity_field)) == entity]
            result = measurement_icc(selected, labels, scale)
            row = {
                "scale": scale,
                "entity": entity,
                "is_overall": False,
                **{key: value for key, value in result.items() if key != "target_ids"},
            }
            result_rows.append(row)
            display_rows.append({"kind": "estimate", **row})
        overall = measurement_icc(records, labels, scale)
        row = {
            "scale": scale,
            "entity": "Overall",
            "is_overall": True,
            **{key: value for key, value in overall.items() if key != "target_ids"},
        }
        result_rows.append(row)
        display_rows.append({"kind": "estimate", **row})
        if scale != scales[-1]:
            display_rows.append({"kind": "spacer", "label": ""})
    valid = [row for row in result_rows if row.get("status") == "ok" and row.get("icc_a_1") is not None]
    if not valid:
        return None, result_rows

    fig = plt.figure(figsize=(11.8, max(6.2, 0.44 * len(display_rows) + 1.7)), constrained_layout=True)
    grid = fig.add_gridspec(1, 3, width_ratios=(2.25, 4.7, 2.45), wspace=0.02)
    label_ax = fig.add_subplot(grid[0, 0])
    forest_ax = fig.add_subplot(grid[0, 1])
    value_ax = fig.add_subplot(grid[0, 2])
    y_positions = list(reversed(range(len(display_rows))))
    for axis in (label_ax, forest_ax, value_ax):
        axis.set_ylim(-0.7, len(display_rows) - 0.3)
    label_ax.axis("off")
    value_ax.axis("off")
    value_ax.text(0.02, len(display_rows) - 0.05, "ICC (95% CI)", fontsize=10, fontweight="bold", va="top")

    lower_values = [float(row["icc_a_1_ci95_lower"]) for row in valid if row.get("icc_a_1_ci95_lower") is not None]
    x_min = min(0.45, min(lower_values, default=0.5) - 0.04)
    forest_ax.set_xlim(x_min, 1.01)
    forest_ax.set_xticks([tick for tick in (0.5, 0.75, 1.0) if tick >= x_min])
    forest_ax.axvline(0.75, color="#cbd5e1", linewidth=0.9, linestyle="--", zorder=0)
    forest_ax.grid(axis="x", alpha=0.16)
    forest_ax.set_xlabel("Absolute-agreement ICC(A,1)")
    forest_ax.set_yticks([])
    for spine in ("top", "left", "right"):
        forest_ax.spines[spine].set_visible(False)

    for y, row in zip(y_positions, display_rows):
        kind = row["kind"]
        if kind == "header":
            label_ax.text(0.0, y, row["label"], fontsize=11, fontweight="bold", va="center")
            continue
        if kind == "spacer":
            continue
        is_overall = bool(row["is_overall"])
        label_ax.text(
            0.08 if not is_overall else 0.04,
            y,
            row["entity"],
            fontsize=9.5,
            fontweight="bold" if is_overall else "normal",
            va="center",
        )
        estimate = row.get("icc_a_1")
        lower = row.get("icc_a_1_ci95_lower")
        upper = row.get("icc_a_1_ci95_upper")
        if estimate is None:
            value_ax.text(0.02, y, "NA", fontsize=9, va="center", color="#64748b")
            continue
        if lower is not None and upper is not None:
            forest_ax.errorbar(
                float(estimate),
                y,
                xerr=[[float(estimate) - float(lower)], [float(upper) - float(estimate)]],
                fmt="D" if is_overall else "o",
                markersize=7.2 if is_overall else 4.8,
                color="#111827" if is_overall else "#2563eb",
                ecolor="#111827" if is_overall else "#64748b",
                capsize=2.5,
                linewidth=1.15,
                zorder=3,
            )
            text_value = f"{float(estimate):.3f} [{float(lower):.3f}, {float(upper):.3f}]"
        else:
            forest_ax.scatter(float(estimate), y, marker="D" if is_overall else "o", color="#111827")
            text_value = f"{float(estimate):.3f} [CI unavailable]"
        value_ax.text(0.02, y, text_value, fontsize=9, fontweight="bold" if is_overall else "normal", va="center")

    fig.suptitle(f"A  Total-score reliability — {dimension_label}", fontsize=15, fontweight="bold", x=0.02, ha="left")
    fig.text(
        0.5,
        -0.012,
        "ICC(A,1)/ICC(2,1): two-way random, absolute agreement, single generation; 95% CI uses outer-run cluster bootstrap. ◆ Overall.",
        ha="center",
        fontsize=8,
        color="#475569",
    )
    path = save_figure(fig, out_dir / f"core_03_measurement_icc_{filename_suffix}.png")
    return path, result_rows


def plot_measurement_error(
    records: list[ExperimentRecord],
    labels: list[str],
    scales: list[str],
    out_dir: Path,
    title_prefix: str,
) -> Path | None:
    summaries = [measurement_error_summary(records, labels, scale) for scale in scales]
    summaries = [summary for summary in summaries if summary["status"] == "ok" and summary["snapshot_rows"]]
    if not summaries:
        return None
    fig, axes = plt.subplots(1, len(summaries), figsize=(max(8.0, 4.4 * len(summaries)), 5.3), squeeze=False, constrained_layout=True)
    for index, summary in enumerate(summaries):
        ax = axes[0][index]
        values = [float(row["repeat_sd"]) for row in summary["snapshot_rows"]]
        parts = ax.violinplot(values, positions=[1], widths=0.62, showmedians=True, showextrema=False)
        for body in parts["bodies"]:
            body.set_facecolor("#93c5fd")
            body.set_edgecolor("#2563eb")
            body.set_alpha(0.72)
        parts["cmedians"].set_color("#1e3a8a")
        ax.scatter(np.ones(len(values)), values, s=9, color="#1d4ed8", alpha=0.18, zorder=2)
        ax.axhline(summary["sem"], color="#b45309", linewidth=1.8, label=f"SEM={summary['sem']:.2f}")
        ax.axhline(summary["mdc95"], color="#b91c1c", linewidth=1.8, linestyle="--", label=f"MDC95={summary['mdc95']:.2f}")
        ax.set_xticks([1], [f"snapshot repeat SD\n(n={len(values)})"])
        ax.set_ylabel("Total-score points" if index == 0 else "")
        ax.set_title(summary["scale"])
        ax.grid(axis="y", alpha=0.18)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle(f"{title_prefix}: measurement error")
    fig.text(
        0.5,
        -0.015,
        "Violin: distribution of within-snapshot repeat SD. SEM uses the absolute-agreement ANOVA error component; MDC95=1.96×√2×SEM.",
        ha="center",
        fontsize=8,
        color="#475569",
    )
    return save_figure(fig, out_dir / "core_04_measurement_error.png")
