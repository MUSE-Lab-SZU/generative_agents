"""Intervention-dose, prompt-completion, and CBT-stage figures."""

from __future__ import annotations

from collections import defaultdict

from pathlib import Path

from statistics import mean

from typing import Any

import numpy as np

from ....schema import group_color, group_sort_key, kbd_color, label_display

from ...shared.plotting import plt, save_figure


def plot_process_delivery(
    process_rows: list[dict[str, Any]],
    stage_rows: list[dict[str, Any]],
    out_dir: Path,
    title_prefix: str,
) -> list[Path]:
    available = [row for row in process_rows if row.get("raw_artifacts_available")]
    if not available:
        return []
    paths: list[Path] = []
    include_kbd = len({row["kbd"] for row in available}) > 1
    available.sort(
        key=lambda row: (
            group_sort_key(row["kbd"]),
            group_sort_key(row["group"]),
            row["outer_run_id"],
        )
    )
    labels = [
        "-".join(
            part
            for part in (
                row["kbd"] if include_kbd else None,
                row["group"],
                row["outer_run_id"],
            )
            if part
        )
        for row in available
    ]
    x = np.arange(len(available))

    fig, ax = plt.subplots(
        figsize=(max(8.0, len(available) * 0.75), 5.0), constrained_layout=True
    )
    prompt = [row.get("cbt_prompt_completion_meeting") or 0 for row in available]
    fixed = [row.get("fixed_prompt_post_completion_meetings") or 0 for row in available]
    ax.bar(x, prompt, color="#2563eb", label="CBT session-prompt phase")
    ax.bar(
        x,
        fixed,
        bottom=prompt,
        color="#f59e0b",
        label="fixed follow-up-prompt consultations",
    )
    ax.set_xticks(x, labels, rotation=35, ha="right")
    ax.set_ylabel("Consultation meetings")
    ax.set_title(f"{title_prefix}: CBT prompt completion and continued consultations")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)
    fig.text(
        0.5,
        -0.02,
        "Orange meetings still use a fixed follow-up prompt and forced doctor consultation. "
        "They are not a no-intervention delayed follow-up; true_followup=false.",
        ha="center",
        fontsize=8,
        color="#475569",
    )
    paths.append(
        save_figure(fig, out_dir / "core_05_cbt_prompt_and_fixed_followup_meetings.png")
    )

    metrics = [
        ("total_turns", "Total dialogue turns"),
        ("total_chars", "Total dialogue characters"),
        ("mean_turns_per_meeting", "Mean turns / meeting"),
        ("environment_tasks", "Environment tasks"),
    ]
    fig, axes = plt.subplots(
        2, 2, figsize=(10.5, 8.0), squeeze=False, constrained_layout=True
    )
    compare_personas = len({row["group"] for row in available}) == 1 and include_kbd
    category_field = "kbd" if compare_personas else "group"
    for ax, (field, title) in zip(axes.flat, metrics):
        grouped: dict[str, list[float]] = defaultdict(list)
        for row in available:
            value = row.get(field)
            if value is not None:
                grouped[row[category_field]].append(float(value))
        categories = sorted(grouped, key=group_sort_key)
        for index, category in enumerate(categories):
            values = grouped[category]
            jitter = np.linspace(-0.08, 0.08, len(values)) if len(values) > 1 else [0.0]
            ax.scatter(
                np.asarray([index] * len(values)) + jitter,
                values,
                color=(
                    kbd_color(category) if compare_personas else group_color(category)
                ),
                alpha=0.75,
            )
            ax.scatter(index, mean(values), color="#111827", marker="D", s=38, zorder=3)
        ax.set_xticks(range(len(categories)), categories)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle(f"{title_prefix}: intervention dose and delivery")
    fig.text(
        0.5,
        -0.015,
        "Points are outer simulation runs; diamonds are category means. Character counts are exposure proxies, not tokens.",
        ha="center",
        fontsize=8,
        color="#475569",
    )
    paths.append(save_figure(fig, out_dir / "core_06_intervention_dose_delivery.png"))

    if stage_rows:
        stage_names: list[str] = []
        by_run: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in stage_rows:
            name = str(row.get("legacy_session") or row.get("macro_stage") or "unknown")
            if name not in stage_names:
                stage_names.append(name)
            by_run[row["stable_id"]].append(row)
        run_ids = [row["stable_id"] for row in available if row["stable_id"] in by_run]
        if run_ids and stage_names:
            matrix = np.zeros((len(run_ids), len(stage_names)), dtype=float)
            for row_index, stable_id in enumerate(run_ids):
                for entry in by_run[stable_id]:
                    name = str(
                        entry.get("legacy_session")
                        or entry.get("macro_stage")
                        or "unknown"
                    )
                    matrix[row_index, stage_names.index(name)] += 1
            fig, ax = plt.subplots(
                figsize=(
                    max(9.0, len(stage_names) * 0.85),
                    max(4.0, len(run_ids) * 0.48 + 2.0),
                ),
                constrained_layout=True,
            )
            image = ax.imshow(matrix, cmap="Blues", aspect="auto", vmin=0)
            run_label_map = {
                row["stable_id"]: "-".join(
                    part
                    for part in (
                        row["kbd"] if include_kbd else None,
                        row["group"],
                        row["outer_run_id"],
                    )
                    if part
                )
                for row in available
            }
            ax.set_xticks(
                range(len(stage_names)),
                [label_display(name) for name in stage_names],
                rotation=35,
                ha="right",
            )
            ax.set_yticks(
                range(len(run_ids)), [run_label_map[value] for value in run_ids]
            )
            ax.set_title(f"{title_prefix}: meetings spent in each CBT prompt")
            for row_index in range(matrix.shape[0]):
                for column_index in range(matrix.shape[1]):
                    if matrix[row_index, column_index]:
                        ax.text(
                            column_index,
                            row_index,
                            f"{matrix[row_index, column_index]:.0f}",
                            ha="center",
                            va="center",
                            fontsize=8,
                        )
            fig.colorbar(image, ax=ax, label="meetings")
            paths.append(
                save_figure(fig, out_dir / "core_07_cbt_stage_dwell_heatmap.png")
            )
    return paths
