"""Case-study figures for complaint-stage evaluation nodes."""

from __future__ import annotations

from pathlib import Path

from typing import Any

import numpy as np

from ....schema import group_sort_key, label_display

from ...shared.plotting import plt, save_figure


def plot_complaint_evaluation_changes(
    interval_rows: list[dict[str, Any]], evaluation_labels: list[str], out_dir: Path
) -> Path | None:
    if not interval_rows:
        return None
    run_ids = sorted(
        {row["stable_id"] for row in interval_rows},
        key=lambda value: tuple(
            group_sort_key(part)
            for part in value.split("::")[:2] + value.split("::")[-1:]
        ),
    )
    intervals = [
        (first, second)
        for first, second in zip(evaluation_labels, evaluation_labels[1:])
        if any(
            row["from_timepoint"] == first and row["to_timepoint"] == second
            for row in interval_rows
        )
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
            advance[row_index, column_index] = float(
                row["session_reflection_advance_rate"]
            )
    fig, axes = plt.subplots(1, 2, figsize=(14, 10), constrained_layout=True)
    first_image = axes[0].imshow(delta, cmap="PuOr", aspect="auto")
    axes[0].set_title("Stage-index change between evaluation nodes")
    fig.colorbar(first_image, ax=axes[0], label="End index − start index")
    second_image = axes[1].imshow(advance, vmin=0, vmax=1, cmap="YlGnBu", aspect="auto")
    axes[1].set_title("Reflection advance rate within interval")
    fig.colorbar(second_image, ax=axes[1], label="Advance proportion")
    xlabels = [
        f"{label_display(first)}→{label_display(second)}" for first, second in intervals
    ]
    short_ids = [
        f"{value.split('::')[0]}-{value.split('::')[2]}-{value.split('::')[-1]}"
        for value in run_ids
    ]
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
