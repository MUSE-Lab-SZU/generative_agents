"""Two-panel baseline-change bar figures with outer-run CIs and contrasts."""

from __future__ import annotations

from collections import defaultdict

from pathlib import Path

from typing import Any

import numpy as np

from ....schema import (
    group_color,
    group_sort_key,
    label_display,
    label_sort_key,
    scale_sort_key,
    slugify,
)

from ...shared.plotting import plt, save_figure


def _significance_brackets(
    ax: Any,
    contrasts: list[dict[str, Any]],
    positions: dict[tuple[str, str], float],
    bottom: float,
    span: float,
    significance_label: str,
) -> None:
    selected = [
        row
        for row in contrasts
        if row.get("p_value_adjusted") is not None
        and float(row["p_value_adjusted"]) < 0.05
        and (row["timepoint"], row["first_group"]) in positions
        and (row["timepoint"], row["second_group"]) in positions
    ]
    by_time: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in selected:
        by_time[row["timepoint"]].append(row)
    level = 0
    for timepoint in sorted(by_time, key=label_sort_key):
        for row in sorted(
            by_time[timepoint], key=lambda item: item["p_value_adjusted"]
        )[:3]:
            x1 = positions[(timepoint, row["first_group"])]
            x2 = positions[(timepoint, row["second_group"])]
            y = bottom - span * (0.07 + level * 0.075)
            cap = span * 0.025
            ax.plot(
                [x1, x1, x2, x2],
                [y + cap, y, y, y + cap],
                color="#1f2937",
                linewidth=0.9,
            )
            stars = row.get("significance") or ""
            p_label = f"p={row['p_value_adjusted']:.3g}"
            label = {
                "stars": stars,
                "p-value": p_label,
                "both": f"{stars} {p_label}".strip(),
            }[significance_label]
            ax.text(
                (x1 + x2) / 2,
                y - span * 0.012,
                label,
                ha="center",
                va="top",
                fontsize=8,
            )
            level += 1


def plot_change_ci_comparison(
    estimates: list[dict[str, Any]],
    contrasts: list[dict[str, Any]],
    out_dir: Path,
    title_prefix: str,
    *,
    significance_label: str = "stars",
) -> list[Path]:
    """Render one vertically stacked A/B figure per outcome."""
    paths: list[Path] = []
    outcomes = sorted({row["outcome"] for row in estimates}, key=scale_sort_key)
    for outcome in outcomes:
        outcome_rows = [row for row in estimates if row["outcome"] == outcome]
        timepoints = sorted(
            {row["timepoint"] for row in outcome_rows}, key=label_sort_key
        )
        groups = sorted({row["group"] for row in outcome_rows}, key=group_sort_key)
        if not timepoints or not groups:
            continue
        values = [
            float(value)
            for row in outcome_rows
            for value in (
                row.get("estimate"),
                row.get("ci95_lower"),
                row.get("ci95_upper"),
            )
            if value is not None
        ]
        if not values:
            continue
        low, high = min(values + [0.0]), max(values + [0.0])
        span = max(high - low, 1.0)
        bracket_depth = span * 0.34
        y_limits = (low - bracket_depth, high + span * 0.18)
        cluster_gap = 1.0
        width = min(0.74, 2.7 / max(len(groups), 1))
        cluster_width = width * len(groups)
        centers = np.arange(len(timepoints), dtype=float) * (
            cluster_width + cluster_gap
        )

        fig, axes = plt.subplots(
            2,
            1,
            figsize=(max(8.2, len(timepoints) * max(2.1, len(groups) * 0.72)), 10.2),
            sharex=True,
            sharey=True,
            constrained_layout=True,
        )
        for panel_index, (ax, panel) in enumerate(zip(np.atleast_1d(axes), ("A", "B"))):
            rows = [
                row
                for row in outcome_rows
                if row["panel"] == panel and row.get("estimate") is not None
            ]
            panel_title = next(
                (row["panel_title"] for row in outcome_rows if row["panel"] == panel),
                "Sensitivity analysis",
            )
            if not rows:
                ax.text(
                    0.5,
                    0.5,
                    "Required panel data unavailable",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                )
                ax.set_ylim(*y_limits)
                ax.set_title(panel_title, loc="left")
                ax.text(-0.08, 1.03, f"({panel})", transform=ax.transAxes, fontsize=12)
                continue
            lookup = {(row["timepoint"], row["group"]): row for row in rows}
            positions: dict[tuple[str, str], float] = {}
            for time_index, timepoint in enumerate(timepoints):
                for group_index, group in enumerate(groups):
                    x = (
                        centers[time_index]
                        + (group_index - (len(groups) - 1) / 2) * width
                    )
                    positions[(timepoint, group)] = x
                    row = lookup.get((timepoint, group))
                    if row is None:
                        continue
                    estimate = float(row["estimate"])
                    lower, upper = row.get("ci95_lower"), row.get("ci95_upper")
                    yerr = None
                    if lower is not None and upper is not None:
                        yerr = [[estimate - float(lower)], [float(upper) - estimate]]
                    ax.bar(
                        [x],
                        [estimate],
                        width=width * 0.86,
                        color=group_color(group),
                        alpha=0.78,
                        edgecolor="#374151",
                        linewidth=0.8,
                        yerr=yerr,
                        capsize=3,
                        error_kw={"elinewidth": 1.0, "ecolor": "#111827"},
                    )
                    offset = span * 0.025
                    ax.text(
                        x + width * 0.1,
                        estimate - offset if estimate < 0 else estimate + offset,
                        f"{estimate:+.2f}",
                        ha="left",
                        va="top" if estimate < 0 else "bottom",
                        fontsize=8,
                    )
            panel_contrasts = [
                row
                for row in contrasts
                if row["panel"] == panel and row["outcome"] == outcome
            ]
            _significance_brackets(
                ax,
                panel_contrasts,
                positions,
                low,
                span,
                significance_label,
            )
            ax.axhline(0, color="#111827", linewidth=1.0)
            for boundary in (centers[:-1] + centers[1:]) / 2:
                ax.axvline(
                    boundary, color="#cbd5e1", linestyle=(0, (4, 4)), linewidth=0.9
                )
            ax.set_ylim(*y_limits)
            ax.set_ylabel(f"Change from baseline in {outcome}\n(mean, 95% CI)")
            ax.set_title(panel_title, loc="left", fontsize=10)
            ax.text(-0.08, 1.03, f"({panel})", transform=ax.transAxes, fontsize=12)
            ax.grid(axis="y", alpha=0.18)
            if panel_index == 0:
                handles = [
                    plt.Rectangle((0, 0), 1, 1, color=group_color(group), alpha=0.78)
                    for group in groups
                ]
                ax.legend(
                    handles,
                    groups,
                    title="Group",
                    frameon=False,
                    ncol=min(4, len(groups)),
                    loc="best",
                )

        tick_positions = [
            centers[time_index] + (group_index - (len(groups) - 1) / 2) * width
            for time_index in range(len(timepoints))
            for group_index in range(len(groups))
        ]
        axes[-1].set_xticks(tick_positions, groups * len(timepoints), rotation=0)
        for center, timepoint in zip(centers, timepoints):
            axes[-1].text(
                center,
                -0.14,
                label_display(timepoint),
                transform=axes[-1].get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=10,
            )
        fig.suptitle(f"{title_prefix}: {outcome} change from baseline")
        fig.text(
            0.5,
            -0.01,
            "Bars and 95% CIs use independent outer simulation runs. Brackets show Holm-adjusted pairwise tests; "
            "negative change indicates symptom reduction.",
            ha="center",
            fontsize=8,
            color="#475569",
        )
        paths.append(
            save_figure(
                fig, out_dir / f"paper_figure_02_{slugify(outcome)}_change_ci.png"
            )
        )
    return paths
