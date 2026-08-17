"""Two-panel session-dose/outcome and interaction-context figures."""

from __future__ import annotations

from pathlib import Path

from typing import Any

import numpy as np

from ....schema import group_color, group_sort_key, slugify

from ...shared.plotting import plt, save_figure


def _errors(rows: list[dict[str, Any]], prefix: str) -> np.ndarray:
    lower: list[float] = []
    upper: list[float] = []
    stem = f"{prefix}_" if prefix else ""
    for row in rows:
        center = row.get(f"{stem}mean")
        low = row.get(f"{stem}ci95_lower")
        high = row.get(f"{stem}ci95_upper")
        lower.append(
            float(center) - float(low)
            if center is not None and low is not None
            else 0.0
        )
        upper.append(
            float(high) - float(center)
            if center is not None and high is not None
            else 0.0
        )
    return np.asarray([lower, upper])


def plot_engagement_process(
    daily_rows: list[dict[str, Any]],
    time_bin_rows: list[dict[str, Any]],
    out_dir: Path,
    title_prefix: str,
    session_rows: list[dict[str, Any]] | None = None,
    resident_interval_rows: list[dict[str, Any]] | None = None,
) -> list[Path]:
    """Render one A/B engagement figure per experimental group."""
    paths: list[Path] = []
    groups = sorted({row["group"] for row in daily_rows}, key=group_sort_key)
    for group in groups:
        daily = [row for row in daily_rows if row["group"] == group]
        bins = [row for row in time_bin_rows if row["group"] == group]
        resident_intervals = [
            row
            for row in (resident_interval_rows or [])
            if row.get("group") == group
        ]
        if not daily or (not bins and not resident_intervals):
            continue
        secondary_metric = str(daily[0].get("secondary_metric") or "turns")
        secondary_label = (
            "Mean turns / interaction"
            if secondary_metric == "turns"
            else "Mean planned duration proxy (minutes / interaction)"
        )
        fig = plt.figure(figsize=(10.0, 10.0), constrained_layout=True)
        grid = fig.add_gridspec(2, 1, height_ratios=[1.08, 1.0])
        ax_frequency = fig.add_subplot(grid[0, 0])
        ax_secondary = ax_frequency.twinx()
        ax_bins = fig.add_subplot(grid[1, 0])

        group_sessions = [
            row for row in (session_rows or []) if row.get("group") == group
        ]
        turn_rows = [
            row for row in group_sessions if row.get("metric") == "dialogue_turns"
        ]
        handles = []
        if turn_rows:
            turn_line = ax_frequency.errorbar(
                [row["session"] for row in turn_rows],
                [row["mean"] for row in turn_rows],
                yerr=_errors(turn_rows, ""),
                color="#1d4ed8",
                marker="o",
                linewidth=2.0,
                capsize=3,
                label="CBT dialogue turns",
            )
            handles.append(turn_line)
            outcome_colors = {"PHQ-9": "#f59e0b", "BDI-II": "#0f766e"}
            outcome_markers = {"PHQ-9": "s", "BDI-II": "D"}
            outcomes = sorted(
                {
                    str(row.get("outcome"))
                    for row in group_sessions
                    if row.get("metric") == "score_change"
                },
                key=lambda value: ({"PHQ-9": 0, "BDI-II": 1}.get(value, 2), value),
            )
            for outcome in outcomes:
                selected = [
                    row
                    for row in group_sessions
                    if row.get("metric") == "score_change"
                    and str(row.get("outcome")) == outcome
                ]
                line = ax_secondary.errorbar(
                    [row["session"] for row in selected],
                    [row["mean"] for row in selected],
                    yerr=_errors(selected, ""),
                    color=outcome_colors.get(outcome, "#475569"),
                    marker=outcome_markers.get(outcome, "^"),
                    linewidth=1.8,
                    capsize=3,
                    label=f"{outcome} change from T0",
                )
                handles.append(line)
            max_session = max(int(row["session"]) for row in turn_rows)
            ticks = list(range(0, max_session + 1))
            ax_frequency.set_xticks(
                ticks,
                ["T0" if value == 0 else f"S{value}" for value in ticks],
                fontsize=8,
            )
            ax_frequency.set_xlim(-0.4, max_session + 0.4)
            ax_frequency.set_xlabel("CBT session")
            ax_frequency.set_ylabel("Mean dialogue turns, 95% CI", color="#1d4ed8")
            ax_secondary.set_ylabel("Mean score change from T0, 95% CI")
            ax_secondary.axhline(0, color="#64748b", linestyle="--", linewidth=0.9)
            score_lower, score_upper = ax_secondary.get_ylim()
            ax_secondary.set_ylim(score_lower, max(1.0, score_upper))
            ax_frequency.tick_params(axis="y", colors="#1d4ed8")
        else:
            days = np.asarray([row["simulation_day"] for row in daily], dtype=float)
            frequency = np.asarray([row["frequency_mean"] for row in daily], dtype=float)
            frequency_line = ax_frequency.errorbar(
                days,
                frequency,
                yerr=_errors(daily, "frequency"),
                color="#d62728",
                marker="o",
                linewidth=2.0,
                capsize=3,
                label="Interaction frequency",
            )
            handles.append(frequency_line)
            secondary_rows = [
                row for row in daily if row.get("secondary_mean") is not None
            ]
            if secondary_rows:
                secondary_line = ax_secondary.errorbar(
                    [row["simulation_day"] for row in secondary_rows],
                    [row["secondary_mean"] for row in secondary_rows],
                    yerr=_errors(secondary_rows, "secondary"),
                    color="#1d4ed8",
                    marker="o",
                    linewidth=2.0,
                    capsize=3,
                    label=secondary_label,
                )
                handles.append(secondary_line)
            ax_frequency.set_xlabel("Simulation day")
            ax_frequency.set_ylabel("Mean interactions / day, 95% CI", color="#d62728")
            ax_secondary.set_ylabel(f"{secondary_label}, 95% CI", color="#1d4ed8")
            ax_frequency.tick_params(axis="y", colors="#d62728")
            ax_secondary.tick_params(axis="y", colors="#1d4ed8")
        ax_frequency.grid(axis="y", alpha=0.18)
        ax_frequency.legend(
            handles,
            [handle.get_label() for handle in handles],
            frameon=False,
            loc="best",
        )
        ax_frequency.text(
            -0.07, 1.02, "(A)", transform=ax_frequency.transAxes, fontsize=12
        )

        lower_rows = resident_intervals or bins
        if resident_intervals:
            mean_field = "mean_turns"
            lower_field = "ci95_lower"
            upper_field = "ci95_upper"
            label_field = "session_interval"
        else:
            mean_field = "frequency_mean_per_day"
            lower_field = "frequency_ci95_lower"
            upper_field = "frequency_ci95_upper"
            label_field = "time_bin"
        x = np.arange(len(lower_rows), dtype=float)
        means = np.asarray([row[mean_field] for row in lower_rows], dtype=float)
        bin_errors = np.asarray(
            [
                [
                    (
                        float(row[mean_field]) - float(row[lower_field])
                        if row.get(lower_field) is not None
                        else 0.0
                    )
                    for row in lower_rows
                ],
                [
                    (
                        float(row[upper_field]) - float(row[mean_field])
                        if row.get(upper_field) is not None
                        else 0.0
                    )
                    for row in lower_rows
                ],
            ]
        )
        color = group_color(group)
        ax_bins.bar(
            x,
            means,
            width=0.84,
            color=color,
            alpha=0.32,
            edgecolor="#374151",
            linewidth=0.8,
            yerr=bin_errors,
            capsize=3,
        )
        ax_bins.plot(x, means, color=color, marker="s", linewidth=1.35, markersize=4)
        ax_bins.set_xticks(
            x, [row[label_field] for row in lower_rows], rotation=35, ha="right"
        )
        if resident_intervals:
            ax_bins.set_xlabel("相邻 CBT session 区间")
            ax_bins.set_ylabel("居民对话 turn 数均值（95% CI）")
        else:
            ax_bins.set_xlabel("Simulation time (24 h)")
            ax_bins.set_ylabel("Mean interaction frequency / day, 95% CI")
        ax_bins.grid(axis="y", alpha=0.18)
        ax_bins.text(-0.07, 1.02, "(B)", transform=ax_bins.transAxes, fontsize=12)
        suffix = (
            "CBT session 对话 turn、量表变化与相邻 session 间居民对话 turn"
            if resident_intervals
            else "CBT session dialogue dose, score change, and interaction timing"
        )
        fig.suptitle(f"{title_prefix}: {suffix} ({group})")
        fig.text(
            0.5,
            -0.01,
            (
                "置信区间以独立 outer simulation run 为统计单位。上图为正式医患 CBT session；量表变化相对 T0。"
                "下图统计相邻 session 之间目标患者与居民对话的总 turn 数，并保留零对话区间。"
                if resident_intervals
                else "CIs use independent outer simulation runs. Panel A uses formal doctor-target CBT sessions; "
                "score change is relative to T0. Panel B retains zero-interaction covered days."
            ),
            ha="center",
            fontsize=8,
            color="#475569",
        )
        paths.append(
            save_figure(
                fig,
                out_dir / f"paper_figure_03_engagement_process_{slugify(group)}.png",
            )
        )
    return paths
