"""Run-preserving figures for cross-persona and cross-condition interpretation."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from ..schema import group_sort_key, label_display, scale_sort_key
from .scale_plots import Line2D, plt, save_figure


def _run_order(rows: list[dict[str, Any]], entity_field: str) -> list[tuple[str, str, str]]:
    return sorted(
        {(str(row[entity_field]), str(row["outer_run_id"]), str(row["stable_id"])) for row in rows},
        key=lambda value: (group_sort_key(value[0]), group_sort_key(value[1])),
    )


def plot_run_symptom_heatmap(
    symptom_changes: list[dict[str, Any]], entity_field: str, title: str, out_dir: Path
) -> Path | None:
    rows = [row for row in symptom_changes if row["scale"] == "COMBINED_EQUAL_SCALE_WEIGHT"]
    run_order = _run_order(rows, entity_field)
    symptoms = list(dict.fromkeys(str(row["symptom_id"]) for row in rows))
    if not run_order or not symptoms:
        return None
    matrix = np.full((len(run_order), len(symptoms)), np.nan)
    labels: dict[str, str] = {}
    stable_to_row = {stable_id: index for index, (_entity, _repeat, stable_id) in enumerate(run_order)}
    for row in rows:
        matrix[stable_to_row[row["stable_id"]], symptoms.index(row["symptom_id"])] = float(row["change"])
        labels[row["symptom_id"]] = row["symptom_label_en"]
    limit = max(0.5, float(np.nanmax(np.abs(matrix))))
    fig, ax = plt.subplots(figsize=(15, max(6.5, len(run_order) * 0.45 + 2.5)), constrained_layout=True)
    image = ax.imshow(matrix, vmin=-limit, vmax=limit, cmap="RdYlGn_r", aspect="auto")
    for row_index in range(len(run_order)):
        for column_index in range(len(symptoms)):
            value = matrix[row_index, column_index]
            if not math.isnan(value):
                ax.text(column_index, row_index, f"{value:+.2f}", ha="center", va="center", fontsize=7)
    ax.set_xticks(range(len(symptoms)), [labels[value] for value in symptoms], rotation=30, ha="right")
    ax.set_yticks(
        range(len(run_order)),
        [f"{entity}-{repeat}" for entity, repeat, _stable_id in run_order],
        fontsize=8,
    )
    ax.set_title(f"{title}: each row is one independent simulation run")
    fig.colorbar(image, ax=ax, label="session 20 − T0 (negative/green = improvement)")
    fig.text(
        0.5,
        -0.018,
        "No averaging across personas or conditions. R01 and R02 remain separate; each snapshot averages only its 10 measurement repeats.",
        ha="center",
        fontsize=8,
        color="#475569",
    )
    return save_figure(fig, out_dir / "life_state_01_single_run_symptom_heatmap.png")


AGREEMENT_CODES = {
    "两次均改善": -1.0,
    "两次均恶化": 1.0,
    "两次均稳定": 0.0,
    "方向不一致或含稳定": 0.45,
    "重复不足": np.nan,
}


def plot_replicate_agreement(
    agreement_rows: list[dict[str, Any]], entity_field: str, title: str, out_dir: Path
) -> Path | None:
    entities = sorted({str(row[entity_field]) for row in agreement_rows}, key=group_sort_key)
    symptoms = list(dict.fromkeys(str(row["symptom_id"]) for row in agreement_rows))
    if not entities or not symptoms:
        return None
    matrix = np.full((len(entities), len(symptoms)), np.nan)
    annotations = [["" for _ in symptoms] for _ in entities]
    labels: dict[str, str] = {}
    for row in agreement_rows:
        row_index = entities.index(str(row[entity_field]))
        column_index = symptoms.index(str(row["symptom_id"]))
        matrix[row_index, column_index] = AGREEMENT_CODES[row["agreement"]]
        annotations[row_index][column_index] = f"{row.get('R01_change', 'NA'):+.2f}/{row.get('R02_change', 'NA'):+.2f}" if row.get("R01_change") is not None and row.get("R02_change") is not None else "NA"
        labels[row["symptom_id"]] = row["symptom_label_en"]
    from matplotlib.colors import BoundaryNorm, ListedColormap

    cmap = ListedColormap(["#16a34a", "#eab308", "#f59e0b", "#dc2626"])
    norm = BoundaryNorm([-1.1, -0.5, 0.2, 0.7, 1.1], cmap.N)
    fig, ax = plt.subplots(figsize=(15, max(5.0, len(entities) * 0.65 + 2.5)), constrained_layout=True)
    ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto")
    for row_index in range(len(entities)):
        for column_index in range(len(symptoms)):
            if annotations[row_index][column_index]:
                ax.text(column_index, row_index, annotations[row_index][column_index], ha="center", va="center", fontsize=7)
    ax.set_xticks(range(len(symptoms)), [labels[value] for value in symptoms], rotation=30, ha="right")
    ax.set_yticks(range(len(entities)), entities)
    ax.set_title(f"{title}: can the R01 finding be reproduced in R02?")
    handles = [
        Line2D([0], [0], marker="s", linestyle="", color=color, markersize=9, label=label)
        for color, label in (
            ("#16a34a", "both improve"),
            ("#dc2626", "both worsen"),
            ("#eab308", "both stable"),
            ("#f59e0b", "mixed/one stable"),
        )
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=4, fontsize=8)
    fig.text(0.5, -0.015, "Cell text = R01 change / R02 change. This is reproducibility, not an average effect.", ha="center", fontsize=8, color="#475569")
    return save_figure(fig, out_dir / "life_state_02_r01_r02_reproducibility.png")


def plot_item_heatmap(
    item_changes: list[dict[str, Any]], scale: str, entity_field: str, title: str, out_dir: Path
) -> Path | None:
    rows = [row for row in item_changes if row["scale"] == scale]
    run_order = _run_order(rows, entity_field)
    item_ids = sorted({int(row["item_id"]) for row in rows})
    if not run_order or not item_ids:
        return None
    matrix = np.full((len(run_order), len(item_ids)), np.nan)
    stable_to_row = {stable_id: index for index, (_entity, _repeat, stable_id) in enumerate(run_order)}
    for row in rows:
        matrix[stable_to_row[row["stable_id"]], item_ids.index(int(row["item_id"]))] = float(row["change"])
    limit = max(0.5, float(np.nanmax(np.abs(matrix))))
    fig, ax = plt.subplots(figsize=(max(11, len(item_ids) * 0.68), max(6, len(run_order) * 0.42 + 2.2)), constrained_layout=True)
    image = ax.imshow(matrix, vmin=-limit, vmax=limit, cmap="RdYlGn_r", aspect="auto")
    ax.set_xticks(range(len(item_ids)), [f"I{value}" for value in item_ids])
    ax.set_yticks(range(len(run_order)), [f"{entity}-{repeat}" for entity, repeat, _stable_id in run_order], fontsize=8)
    ax.set_title(f"{title}: {scale} item changes by independent run")
    fig.colorbar(image, ax=ax, label="session 20 − T0")
    slug = "phq9" if scale == "PHQ-9" else "bdi2"
    return save_figure(fig, out_dir / f"life_state_{'03' if scale == 'PHQ-9' else '04'}_{slug}_single_run_items.png")


CLASS_COLORS = {
    "推进且同期症状改善": "#16a34a",
    "主诉停滞但同期症状改善": "#2563eb",
    "主诉推进但同期症状未改善": "#f59e0b",
    "主诉停滞且同期症状未改善": "#dc2626",
}


def plot_complaint_interval_alignment(
    rows: list[dict[str, Any]], labels: list[str], entity_field: str, title: str, out_dir: Path
) -> Path | None:
    if not rows:
        return None
    intervals = [
        (first, second)
        for first, second in zip(labels, labels[1:])
        if any(row["from_timepoint"] == first and row["to_timepoint"] == second for row in rows)
    ]
    column_count = min(3, len(intervals))
    row_count = math.ceil(len(intervals) / column_count)
    fig, axes = plt.subplots(
        row_count,
        column_count,
        figsize=(13, max(5.8, 4.2 * row_count)),
        squeeze=False,
        constrained_layout=True,
    )
    flat_axes = list(axes.flat)
    for ax, pair in zip(flat_axes, intervals):
        selected = [row for row in rows if (row["from_timepoint"], row["to_timepoint"]) == pair]
        ax.axvspan(-0.03, 0.5, color="#2563eb", alpha=0.035)
        ax.axvspan(0.5, 1.03, color="#16a34a", alpha=0.035)
        for row in selected:
            x = float(row["complaint_advance_rate"])
            y = float(row["general_symptom_change"])
            ax.scatter(
                x,
                y,
                color=CLASS_COLORS[row["alignment_class"]],
                edgecolor="#111827" if row.get("risk_worsened") else "white",
                linewidth=1.3,
                s=48,
            )
            ax.annotate(
                f"{row[entity_field]}-{row['outer_run_id']}",
                (x, y),
                xytext=(3, 3),
                textcoords="offset points",
                fontsize=5.5,
                alpha=0.82,
            )
        ax.axvline(0.5, color="#64748b", linestyle="--", linewidth=0.8)
        ax.axhline(0, color="#111827", linewidth=0.8)
        ax.set_xlim(-0.03, 1.03)
        ax.set_title(f"{label_display(pair[0])}→{label_display(pair[1])}\nn={len(selected)}")
        ax.set_xlabel("Complaint advance rate")
        ax.grid(alpha=0.16)
    unused_axes = flat_axes[len(intervals):]
    for ax in unused_axes:
        ax.axis("off")
    for ax in axes[:, 0]:
        ax.set_ylabel("General life-state change\n(negative = improvement)")
    handles = [
        Line2D([0], [0], marker="o", linestyle="", color=color, label=label)
        for label, color in CLASS_COLORS.items()
    ]
    if unused_axes:
        legend_ax = unused_axes[0]
        legend_ax.legend(handles=handles, loc="center", ncol=1, fontsize=9, frameon=False)
        legend_ax.text(
            0.5,
            0.18,
            "Upper-left = clearest concern\nLower-right = aligned progress\nBlack outline = risk item worsened",
            transform=legend_ax.transAxes,
            ha="center",
            va="center",
            fontsize=8,
            color="#475569",
        )
    else:
        fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=8)
    fig.suptitle(f"{title}: is complaint progress accompanied by symptom improvement?")
    return save_figure(fig, out_dir / "complaint_01_interval_progress_outcome_alignment.png")


def plot_complaint_run_alignment(
    rows: list[dict[str, Any]], entity_field: str, title: str, out_dir: Path
) -> Path | None:
    if not rows:
        return None
    fig, ax = plt.subplots(figsize=(9, 7), constrained_layout=True)
    ax.axvspan(-0.03, 0.5, color="#2563eb", alpha=0.035)
    ax.axvspan(0.5, 1.03, color="#16a34a", alpha=0.035)
    for row in rows:
        x, y = float(row["complaint_advance_rate"]), float(row["general_symptom_change"])
        ax.scatter(
            x,
            y,
            color=CLASS_COLORS[row["alignment_class"]],
            edgecolor="#111827" if row.get("risk_worsened") else "white",
            linewidth=1.4,
            s=72,
        )
        ax.annotate(f"{row[entity_field]}-{row['outer_run_id']}", (x, y), xytext=(4, 4), textcoords="offset points", fontsize=8)
    ax.axvline(0.5, color="#64748b", linestyle="--", linewidth=0.9)
    ax.axhline(0, color="#111827", linewidth=0.9)
    ax.set_xlim(-0.03, 1.03)
    ax.set_xlabel("Complaint reflection advance rate across available intervals")
    ax.set_ylabel("T0→session 20 general life-state change\n(negative = improvement)")
    ax.set_title(f"{title}: run-level complaint/outcome alignment")
    ax.grid(alpha=0.18)
    return save_figure(fig, out_dir / "complaint_02_run_progress_outcome_alignment.png")


def plot_entity_kappa_heatmap(
    summary_rows: list[dict[str, Any]], entity_type: str, title: str, out_dir: Path
) -> Path | None:
    rows = [
        row
        for row in summary_rows
        if row["stratum_type"] == entity_type and row["weights"] == "quadratic" and row.get("kappa") is not None
    ]
    entities = sorted({str(row["stratum_value"]) for row in rows}, key=group_sort_key)
    scales = sorted({str(row["scale"]) for row in rows}, key=scale_sort_key)
    if not entities or not scales:
        return None
    matrix = np.full((len(entities), len(scales)), np.nan)
    for row in rows:
        matrix[entities.index(row["stratum_value"]), scales.index(row["scale"])] = float(row["kappa"])
    fig, ax = plt.subplots(figsize=(6.5, max(4.5, len(entities) * 0.65 + 2)), constrained_layout=True)
    image = ax.imshow(matrix, vmin=0, vmax=1, cmap="YlGnBu", aspect="auto")
    for row_index in range(len(entities)):
        for column_index in range(len(scales)):
            value = matrix[row_index, column_index]
            if not math.isnan(value):
                ax.text(column_index, row_index, f"{value:.3f}", ha="center", va="center")
    ax.set_xticks(range(len(scales)), scales)
    ax.set_yticks(range(len(entities)), entities)
    ax.set_title(f"{title}: entity-specific quadratic weighted Kappa")
    fig.colorbar(image, ax=ax, label="Weighted κ")
    fig.text(0.5, -0.02, "Each entity has two independent outer runs; values are descriptive and no entity-level CI is forced.", ha="center", fontsize=8, color="#475569")
    return save_figure(fig, out_dir / "weighted_kappa_03_entity_scale_heatmap.png")


def render_stratified_figures(
    life_state: dict[str, list[dict[str, Any]]],
    agreement_rows: list[dict[str, Any]],
    interval_alignment: list[dict[str, Any]],
    run_alignment: list[dict[str, Any]],
    kappa: dict[str, Any],
    labels: list[str],
    entity_field: str,
    entity_type: str,
    title: str,
    out_dir: Path,
) -> list[Path]:
    paths = [
        plot_run_symptom_heatmap(life_state["symptom_change"], entity_field, title, out_dir),
        plot_replicate_agreement(agreement_rows, entity_field, title, out_dir),
        plot_item_heatmap(life_state["item_change"], "PHQ-9", entity_field, title, out_dir),
        plot_item_heatmap(life_state["item_change"], "BDI-II", entity_field, title, out_dir),
        plot_complaint_interval_alignment(interval_alignment, labels, entity_field, title, out_dir),
        plot_complaint_run_alignment(run_alignment, entity_field, title, out_dir),
        plot_entity_kappa_heatmap(kappa["summary"], entity_type, title, out_dir),
    ]
    return [path for path in paths if path is not None]
