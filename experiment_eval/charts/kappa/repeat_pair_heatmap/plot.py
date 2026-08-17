"""Compact weighted-kappa figures for ordinal PHQ-9/BDI-II items."""

from __future__ import annotations

import math

from pathlib import Path

from typing import Any

import numpy as np

from ....schema import group_sort_key, scale_sort_key

from ...shared.plotting import plt, save_figure


def plot_repeat_pair_heatmap(
    payload: dict[str, Any], out_dir: Path, title_prefix: str
) -> Path | None:
    rows = payload.get("matrix") or []
    scales = sorted({row["scale"] for row in rows}, key=scale_sort_key)
    repeats = sorted(
        {row["repeat_a"] for row in rows} | {row["repeat_b"] for row in rows},
        key=group_sort_key,
    )
    if not scales or not repeats:
        return None
    fig, axes = plt.subplots(
        1,
        len(scales),
        figsize=(max(7.2, 5.5 * len(scales)), 5.0),
        squeeze=False,
        constrained_layout=True,
    )
    image = None
    for scale_index, scale in enumerate(scales):
        ax = axes[0][scale_index]
        matrix = np.full((len(repeats), len(repeats)), np.nan)
        for row in rows:
            if row["scale"] != scale or row.get("kappa") is None:
                continue
            first = repeats.index(row["repeat_a"])
            second = repeats.index(row["repeat_b"])
            matrix[first, second] = matrix[second, first] = float(row["kappa"])
        image = ax.imshow(matrix, vmin=-1, vmax=1, cmap="RdYlBu", aspect="equal")
        for row_index in range(len(repeats)):
            for column_index in range(len(repeats)):
                value = matrix[row_index, column_index]
                if not math.isnan(value):
                    ax.text(
                        column_index,
                        row_index,
                        f"{value:.2f}",
                        ha="center",
                        va="center",
                        fontsize=7,
                    )
        ax.set_xticks(range(len(repeats)), repeats, rotation=45, ha="right")
        ax.set_yticks(range(len(repeats)), repeats)
        ax.set_xlabel("Measurement repeat")
        ax.set_ylabel("Measurement repeat")
        ax.set_title(scale)
    if image is not None:
        fig.colorbar(
            image, ax=axes.ravel().tolist(), label="Quadratic weighted κ", shrink=0.82
        )
    fig.suptitle(f"{title_prefix}: repeat-pair ordinal item agreement")
    fig.text(
        0.5,
        -0.015,
        "Objects are exact snapshot × scale × item matches. Blank diagonal is not estimated. "
        "Kappa is prevalence/marginal-distribution dependent.",
        ha="center",
        fontsize=8,
        color="#475569",
    )
    return save_figure(fig, out_dir / "weighted_kappa_01_repeat_pair_heatmap.png")
