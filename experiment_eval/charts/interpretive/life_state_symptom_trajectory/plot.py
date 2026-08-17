"""Item-level and symptom-domain figures."""

from __future__ import annotations

import math

from pathlib import Path

from statistics import mean

from typing import Any

import numpy as np

from ....schema import group_sort_key, label_display, label_sort_key, scale_sort_key

from ....statistics import mean_ci95

from ...shared.plotting import plt, save_figure


def plot_symptom_trajectory(
    symptom_rows: list[dict[str, Any]], labels: list[str], out_dir: Path
) -> Path | None:
    rows = [
        row for row in symptom_rows if row["scale"] == "COMBINED_EQUAL_SCALE_WEIGHT"
    ]
    symptoms = list(dict.fromkeys(row["symptom_id"] for row in rows))
    if not rows:
        return None
    fig, axes = plt.subplots(
        3, 3, figsize=(14, 10), squeeze=False, constrained_layout=True
    )
    for ax, symptom_id in zip(axes.ravel(), symptoms):
        selected = [row for row in rows if row["symptom_id"] == symptom_id]
        centers, lower, upper, x_values = [], [], [], []
        for label in sorted(labels, key=lambda value: label_sort_key(value, labels)):
            values = [
                float(row["symptom_score"])
                for row in selected
                if row["timepoint"] == label
            ]
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
        ax.set_xticks(
            range(len(labels)), [label_display(label) for label in labels], rotation=30
        )
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
