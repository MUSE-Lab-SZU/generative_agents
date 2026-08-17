"""Cross-persona contrast, sensitivity, and variance figures."""

from __future__ import annotations

import math

from pathlib import Path

from typing import Any

import numpy as np

from ....schema import group_sort_key, kbd_color, scale_sort_key

from ...shared.plotting import plt, save_figure


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
    labels = {row["source"]: row["source_label"] for row in rows}
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
        (scale, "single", "proportion_of_observed_variance") for scale in scales
    ] + [(scale, "K mean", "proportion_of_k_mean_variance") for scale in scales]
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
                    color=(
                        "white" if source not in {"common_time_profile"} else "#111827"
                    ),
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
    ax.set_title("Descriptive hierarchical variance partition")
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
