"""Figures that translate item and complaint-stage changes into readable outcomes."""

from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np

from ..schema import group_sort_key, label_display, label_sort_key, scale_sort_key
from ..statistics import mean_ci95
from .scale_plots import plt, save_figure


def plot_item_life_state_change(summary_rows: list[dict[str, Any]], out_dir: Path) -> Path | None:
    rows = [
        row
        for row in summary_rows
        if row["stratum_type"] == "overall" and row["metric_level"] == "item"
    ]
    scales = sorted({row["scale"] for row in rows}, key=scale_sort_key)
    if not rows:
        return None
    fig, axes = plt.subplots(1, len(scales), figsize=(14, 8.5), squeeze=False, constrained_layout=True)
    for scale_index, scale in enumerate(scales):
        ax = axes[0][scale_index]
        selected = sorted(
            (row for row in rows if row["scale"] == scale),
            key=lambda row: int(str(row["metric_id"]).rsplit("_", 1)[-1]),
        )
        y = np.arange(len(selected))
        values = [float(row["mean_change"]) for row in selected]
        lower = [row.get("ci95_lower") for row in selected]
        upper = [row.get("ci95_upper") for row in selected]
        errors = np.asarray(
            [
                [value - float(lo) if lo is not None else 0 for value, lo in zip(values, lower)],
                [float(hi) - value if hi is not None else 0 for value, hi in zip(values, upper)],
            ]
        )
        colors = ["#16a34a" if value < 0 else "#dc2626" if value > 0 else "#64748b" for value in values]
        ax.errorbar(values, y, xerr=errors, fmt="none", ecolor="#94a3b8", capsize=2, linewidth=1)
        ax.scatter(values, y, c=colors, s=36, zorder=3)
        ax.axvline(0, color="#111827", linewidth=0.8)
        ax.set_yticks(y, [f"I{str(row['metric_id']).rsplit('_', 1)[-1]} {row['metric_label_en']}" for row in selected])
        ax.invert_yaxis()
        ax.set_title(scale)
        ax.set_xlabel("Endpoint − T0 item score")
        ax.grid(axis="x", alpha=0.2)
    fig.suptitle("Which simulated life-state items changed from T0 to session 20?")
    fig.text(
        0.5,
        -0.012,
        "Green/negative = fewer reported symptoms (improvement); red/positive = worsening. "
        "Points are means across 26 independent outer runs; bars are outer-run t intervals.",
        ha="center",
        fontsize=8,
        color="#475569",
    )
    return save_figure(fig, out_dir / "life_state_01_item_change_forest.png")


def plot_symptom_trajectory(symptom_rows: list[dict[str, Any]], labels: list[str], out_dir: Path) -> Path | None:
    rows = [row for row in symptom_rows if row["scale"] == "COMBINED_EQUAL_SCALE_WEIGHT"]
    symptoms = list(dict.fromkeys(row["symptom_id"] for row in rows))
    if not rows:
        return None
    fig, axes = plt.subplots(3, 3, figsize=(14, 10), squeeze=False, constrained_layout=True)
    for ax, symptom_id in zip(axes.ravel(), symptoms):
        selected = [row for row in rows if row["symptom_id"] == symptom_id]
        centers, lower, upper, x_values = [], [], [], []
        for label in sorted(labels, key=lambda value: label_sort_key(value, labels)):
            values = [float(row["symptom_score"]) for row in selected if row["timepoint"] == label]
            if not values:
                continue
            interval = mean_ci95(values)
            x_values.append(labels.index(label))
            centers.append(mean(values))
            lower.append(interval["lower"])
            upper.append(interval["upper"])
        ax.plot(x_values, centers, marker="o", color="#2563eb", linewidth=1.8)
        if all(value is not None for value in lower + upper):
            ax.fill_between(x_values, lower, upper, color="#93c5fd", alpha=0.28)
        ax.set_xticks(range(len(labels)), [label_display(label) for label in labels], rotation=30)
        ax.set_ylim(0, 3)
        ax.set_title(selected[0]["symptom_label_en"])
        ax.set_ylabel("Mean symptom score (0–3)")
        ax.grid(alpha=0.2)
    for ax in axes.ravel()[len(symptoms) :]:
        ax.axis("off")
    fig.suptitle("Simulated life-state symptom trajectories")
    fig.text(
        0.5,
        -0.012,
        "Each snapshot first averages 10 measurement repeats. PHQ and BDI concept scores receive equal weight. "
        "Risk is shown separately and is not averaged into a general domain.",
        ha="center",
        fontsize=8,
        color="#475569",
    )
    return save_figure(fig, out_dir / "life_state_02_symptom_trajectory.png")


def plot_group_symptom_heatmap(summary_rows: list[dict[str, Any]], out_dir: Path) -> Path | None:
    rows = [
        row
        for row in summary_rows
        if row["stratum_type"] == "group"
        and row["metric_level"] == "symptom"
        and row["scale"] == "COMBINED_EQUAL_SCALE_WEIGHT"
    ]
    groups = sorted({row["stratum_value"] for row in rows}, key=group_sort_key)
    symptoms = list(dict.fromkeys(row["metric_id"] for row in rows))
    if not groups or not symptoms:
        return None
    matrix = np.full((len(groups), len(symptoms)), np.nan)
    labels = {}
    for row in rows:
        matrix[groups.index(row["stratum_value"]), symptoms.index(row["metric_id"])] = float(row["mean_change"])
        labels[row["metric_id"]] = row["metric_label_en"]
    limit = max(0.5, float(np.nanmax(np.abs(matrix))))
    fig, ax = plt.subplots(figsize=(15, 6.5), constrained_layout=True)
    image = ax.imshow(matrix, vmin=-limit, vmax=limit, cmap="RdYlGn_r", aspect="auto")
    for row_index in range(len(groups)):
        for column_index in range(len(symptoms)):
            value = matrix[row_index, column_index]
            if not math.isnan(value):
                ax.text(column_index, row_index, f"{value:+.2f}", ha="center", va="center", fontsize=8)
    ax.set_xticks(range(len(symptoms)), [labels[value] for value in symptoms], rotation=30, ha="right")
    ax.set_yticks(range(len(groups)), groups)
    ax.set_title("Group-level simulated life-state change: session 20 − T0")
    fig.colorbar(image, ax=ax, label="Change (negative/green = improvement)")
    fig.text(
        0.5,
        -0.02,
        "Descriptive only. Most groups have n=2 outer runs, and G1 contains multiple personas while other groups are mainly KBD2.",
        ha="center",
        fontsize=8,
        color="#475569",
    )
    return save_figure(fig, out_dir / "life_state_03_group_symptom_change_heatmap.png")


def plot_complaint_evaluation_changes(
    interval_rows: list[dict[str, Any]], evaluation_labels: list[str], out_dir: Path
) -> Path | None:
    if not interval_rows:
        return None
    run_ids = sorted(
        {row["stable_id"] for row in interval_rows},
        key=lambda value: tuple(group_sort_key(part) for part in value.split("::")[:2] + value.split("::")[-1:]),
    )
    intervals = [
        (first, second)
        for first, second in zip(evaluation_labels, evaluation_labels[1:])
        if any(row["from_timepoint"] == first and row["to_timepoint"] == second for row in interval_rows)
    ]
    if not intervals:
        return None
    delta = np.full((len(run_ids), len(intervals)), np.nan)
    advance = np.full((len(run_ids), len(intervals)), np.nan)
    for row in interval_rows:
        pair = (row["from_timepoint"], row["to_timepoint"])
        if pair not in intervals:
            continue
        row_index, column_index = run_ids.index(row["stable_id"]), intervals.index(pair)
        if row.get("stage_index_change") is not None:
            delta[row_index, column_index] = float(row["stage_index_change"])
        if row.get("session_reflection_advance_rate") is not None:
            advance[row_index, column_index] = float(row["session_reflection_advance_rate"])
    fig, axes = plt.subplots(1, 2, figsize=(14, 10), constrained_layout=True)
    first_image = axes[0].imshow(delta, cmap="PuOr", aspect="auto")
    axes[0].set_title("Stage-index change between evaluation nodes")
    fig.colorbar(first_image, ax=axes[0], label="End index − start index")
    second_image = axes[1].imshow(advance, vmin=0, vmax=1, cmap="YlGnBu", aspect="auto")
    axes[1].set_title("Reflection advance rate within interval")
    fig.colorbar(second_image, ax=axes[1], label="Advance proportion")
    xlabels = [f"{label_display(first)}→{label_display(second)}" for first, second in intervals]
    short_ids = [f"{value.split('::')[0]}-{value.split('::')[2]}-{value.split('::')[-1]}" for value in run_ids]
    for ax in axes:
        ax.set_xticks(range(len(intervals)), xlabels, rotation=30, ha="right")
        ax.set_yticks(range(len(run_ids)), short_ids, fontsize=7)
    fig.suptitle("Complaint-stage changes at scale-evaluation nodes")
    fig.text(
        0.5,
        -0.012,
        "Stage-index movement means complaint progression, not symptom improvement. "
        "Only exact T0/session_4/... snapshots and the sessions between them are used.",
        ha="center",
        fontsize=8,
        color="#475569",
    )
    return save_figure(fig, out_dir / "complaint_01_evaluation_node_changes.png")


def render_interpretive_figures(
    life_state: dict[str, list[dict[str, Any]]],
    complaint: dict[str, Any],
    labels: list[str],
    out_dir: Path,
) -> list[Path]:
    paths = [
        plot_item_life_state_change(life_state["summary"], out_dir),
        plot_symptom_trajectory(life_state["symptom_trajectory"], labels, out_dir),
        plot_group_symptom_heatmap(life_state["summary"], out_dir),
        plot_complaint_evaluation_changes(complaint["intervals"], complaint["evaluation_labels"], out_dir),
    ]
    return [path for path in paths if path is not None]

