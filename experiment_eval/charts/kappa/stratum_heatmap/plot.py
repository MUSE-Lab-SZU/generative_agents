"""Compact weighted-kappa figures for ordinal PHQ-9/BDI-II items."""

from __future__ import annotations

import math

from pathlib import Path

from typing import Any

import numpy as np

from ....schema import group_sort_key, scale_sort_key

from ...shared.plotting import plt, save_figure


def plot_stratum_heatmap(
    payload: dict[str, Any], out_dir: Path, title_prefix: str
) -> Path | None:
    rows = [
        row
        for row in (payload.get("summary") or [])
        if row["weights"] == "quadratic"
        and row["stratum_type"] in {"persona", "group"}
        and row.get("kappa") is not None
        and int(row.get("n_outer_run_clusters") or 0) >= 5
    ]
    eligible_type = next(
        (
            stratum_type
            for stratum_type in ("persona", "group")
            if len(
                {
                    row["stratum_value"]
                    for row in rows
                    if row["stratum_type"] == stratum_type
                }
            )
            >= 2
        ),
        None,
    )
    if eligible_type is None:
        return None
    selected = [row for row in rows if row["stratum_type"] == eligible_type]
    strata = sorted({row["stratum_value"] for row in selected}, key=group_sort_key)
    scales = sorted({row["scale"] for row in selected}, key=scale_sort_key)
    matrix = np.full((len(strata), len(scales)), np.nan)
    for row in selected:
        matrix[strata.index(row["stratum_value"]), scales.index(row["scale"])] = float(
            row["kappa"]
        )
    fig, ax = plt.subplots(
        figsize=(max(5.5, 2.1 * len(scales)), max(4.0, 0.55 * len(strata) + 2.2)),
        constrained_layout=True,
    )
    image = ax.imshow(matrix, vmin=-1, vmax=1, cmap="RdYlBu", aspect="auto")
    for row_index in range(len(strata)):
        for column_index in range(len(scales)):
            value = matrix[row_index, column_index]
            if not math.isnan(value):
                ax.text(
                    column_index,
                    row_index,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                )
    ax.set_xticks(range(len(scales)), scales)
    ax.set_yticks(range(len(strata)), strata)
    ax.set_title(f"{title_prefix}: {eligible_type} × scale quadratic weighted Kappa")
    fig.colorbar(image, ax=ax, label="Quadratic weighted κ")
    return save_figure(
        fig, out_dir / f"weighted_kappa_03_{eligible_type}_scale_heatmap.png"
    )
