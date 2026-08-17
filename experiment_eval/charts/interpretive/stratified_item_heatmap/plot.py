"""Single-run item heatmap for stratified interpretation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ....schema import group_sort_key, scale_sort_key
from ...shared.plotting import plt, save_figure
from ...shared.stratified import _run_order


def plot_item_heatmap(
    item_changes: list[dict[str, Any]],
    scale: str,
    entity_field: str,
    title: str,
    out_dir: Path,
) -> Path | None:
    rows = [row for row in item_changes if row["scale"] == scale]
    run_order = _run_order(rows, entity_field)
    item_ids = sorted({int(row["item_id"]) for row in rows})
    if not run_order or not item_ids:
        return None
    matrix = np.full((len(run_order), len(item_ids)), np.nan)
    stable_to_row = {
        stable_id: index
        for index, (_entity, _repeat, stable_id) in enumerate(run_order)
    }
    for row in rows:
        matrix[stable_to_row[row["stable_id"]], item_ids.index(int(row["item_id"]))] = (
            float(row["change"])
        )
    limit = max(0.5, float(np.nanmax(np.abs(matrix))))
    fig, ax = plt.subplots(
        figsize=(max(11, len(item_ids) * 0.68), max(6, len(run_order) * 0.42 + 2.2)),
        constrained_layout=True,
    )
    image = ax.imshow(matrix, vmin=-limit, vmax=limit, cmap="RdYlGn_r", aspect="auto")
    ax.set_xticks(range(len(item_ids)), [f"I{value}" for value in item_ids])
    ax.set_yticks(
        range(len(run_order)),
        [f"{entity}-{repeat}" for entity, repeat, _stable_id in run_order],
        fontsize=8,
    )
    ax.set_title(f"{title}: {scale} item changes by independent run")
    fig.colorbar(image, ax=ax, label="session 20 − T0")
    slug = "phq9" if scale == "PHQ-9" else "bdi2"
    return save_figure(
        fig,
        out_dir
        / f"life_state_{'03' if scale == 'PHQ-9' else '04'}_{slug}_single_run_items.png",
    )
