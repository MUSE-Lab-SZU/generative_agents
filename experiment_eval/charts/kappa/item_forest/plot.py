"""Compact weighted-kappa figures for ordinal PHQ-9/BDI-II items."""

from __future__ import annotations

import math

from pathlib import Path

from typing import Any

import numpy as np

from ....schema import group_sort_key, scale_sort_key

from ...shared.plotting import heatmap_text_color, plt, save_figure


def plot_item_forest(
    payload: dict[str, Any], out_dir: Path, title_prefix: str
) -> Path | None:
    rows = payload.get("item") or []
    scales = sorted({row["scale"] for row in rows}, key=scale_sort_key)
    if not scales:
        return None
    fig, axes = plt.subplots(
        1,
        len(scales),
        figsize=(max(8.0, 5.8 * len(scales)), 8.0),
        squeeze=False,
        constrained_layout=True,
    )
    plotted = False
    for scale_index, scale in enumerate(scales):
        ax = axes[0][scale_index]
        quadratic = sorted(
            (
                row
                for row in rows
                if row["scale"] == scale and row["weights"] == "quadratic"
            ),
            key=lambda row: int(row["item_id"]),
        )
        linear_by_item = {
            int(row["item_id"]): row
            for row in rows
            if row["scale"] == scale and row["weights"] == "linear"
        }
        y_positions = np.arange(len(quadratic))
        for y_position, row in zip(y_positions, quadratic):
            estimate = row.get("kappa")
            if estimate is None:
                ax.text(
                    -0.96,
                    y_position,
                    f"I{row['item_id']}: NA ({row.get('reason')})",
                    fontsize=7,
                    va="center",
                )
                continue
            plotted = True
            lower, upper = row.get("ci95_lower"), row.get("ci95_upper")
            xerr = None
            if lower is not None and upper is not None:
                xerr = [
                    [float(estimate) - float(lower)],
                    [float(upper) - float(estimate)],
                ]
            ax.errorbar(
                float(estimate),
                y_position,
                xerr=xerr,
                fmt="o",
                color="#2563eb",
                ecolor="#93c5fd",
                capsize=2,
                label="quadratic" if y_position == 0 else None,
            )
            sensitivity = linear_by_item.get(int(row["item_id"]), {}).get("kappa")
            if sensitivity is not None:
                ax.scatter(
                    float(sensitivity),
                    y_position,
                    marker="x",
                    color="#b45309",
                    s=30,
                    label="linear" if y_position == 0 else None,
                )
        ax.axvline(0, color="#111827", linewidth=0.8)
        ax.set_xlim(-1, 1)
        ax.set_yticks(y_positions, [f"Item {row['item_id']}" for row in quadratic])
        ax.invert_yaxis()
        ax.set_xlabel("Pairwise weighted κ (mean across repeat pairs)")
        ax.set_title(scale)
        ax.grid(axis="x", alpha=0.2)
        ax.legend(loc="lower right", fontsize=8)
    if not plotted:
        plt.close(fig)
        return None
    fig.suptitle(f"{title_prefix}: item-level weighted Kappa")
    fig.text(
        0.5,
        -0.015,
        "Circles: quadratic (primary); crosses: linear sensitivity. Error bars use outer-run cluster bootstrap when ≥5 clusters.",
        ha="center",
        fontsize=8,
        color="#475569",
    )
    return save_figure(fig, out_dir / "weighted_kappa_02_item_forest.png")


def plot_item_heatmap(
    payload: dict[str, Any], out_dir: Path, title_prefix: str
) -> Path | None:
    """Compact primary (quadratic) weighted-kappa heatmap by scale and item."""
    rows = [row for row in (payload.get("item") or []) if row.get("weights") == "quadratic"]
    scales = sorted({row["scale"] for row in rows}, key=scale_sort_key)
    item_ids = sorted({int(row["item_id"]) for row in rows})
    if not scales or not item_ids:
        return None
    lookup = {(row["scale"], int(row["item_id"])): row for row in rows}
    matrix = np.full((len(item_ids), len(scales)), np.nan)
    for row_index, item_id in enumerate(item_ids):
        for column_index, scale in enumerate(scales):
            estimate = lookup.get((scale, item_id), {}).get("kappa")
            if estimate is not None:
                matrix[row_index, column_index] = float(estimate)
    fig, ax = plt.subplots(figsize=(5.0, max(5.6, 0.31 * len(item_ids) + 1.8)), constrained_layout=True)
    image = ax.imshow(matrix, cmap="RdYlBu", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(len(scales)), scales)
    ax.set_yticks(np.arange(len(item_ids)), [f"Item {item_id}" for item_id in item_ids])
    ax.set_xticks(np.arange(-0.5, len(scales), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(item_ids), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2)
    ax.tick_params(which="minor", bottom=False, left=False)
    for row_index in range(len(item_ids)):
        for column_index in range(len(scales)):
            value = matrix[row_index, column_index]
            if math.isfinite(value):
                ax.text(
                    column_index,
                    row_index,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7.5,
                    color=heatmap_text_color(image.cmap, image.norm, value),
                )
            else:
                ax.text(column_index, row_index, "—", ha="center", va="center", fontsize=7, color="#94a3b8")
    colorbar = fig.colorbar(image, ax=ax, shrink=0.8, pad=0.04)
    colorbar.set_label("Quadratic weighted κ")
    ax.set_title(f"{title_prefix}: item-level repeat reliability")
    fig.text(
        0.5,
        -0.012,
        "Each cell is the mean pairwise quadratic weighted κ across measurement-repeat pairs; blank cells are not items in that scale.",
        ha="center",
        fontsize=8,
        color="#475569",
    )
    return save_figure(fig, out_dir / "weighted_kappa_02_item_heatmap.png")


def plot_item_heatmap_by_entity(
    payload: dict[str, Any],
    out_dir: Path,
    *,
    dimension_label: str,
    filename_suffix: str,
) -> Path | None:
    """Two-panel PHQ-9/BDI-II item κ heatmap with entity columns."""
    rows = payload.get("item") or []
    entities = list(payload.get("entities") or [])
    scales = sorted({str(row["scale"]) for row in rows}, key=scale_sort_key)
    if not rows or not entities or not scales:
        return None
    fig, axes = plt.subplots(
        1,
        len(scales),
        figsize=(max(12.0, 1.05 * len(entities) * len(scales)), 9.0),
        squeeze=False,
        constrained_layout=True,
    )
    image = None
    for scale_index, scale in enumerate(scales):
        ax = axes[0][scale_index]
        scale_rows = [row for row in rows if row["scale"] == scale]
        item_ids = sorted({int(row["item_id"]) for row in scale_rows})
        lookup = {(str(row["entity"]), int(row["item_id"])): row for row in scale_rows}
        matrix = np.full((len(item_ids), len(entities)), np.nan)
        for row_index, item_id in enumerate(item_ids):
            for column_index, entity in enumerate(entities):
                estimate = lookup.get((entity, item_id), {}).get("kappa")
                if estimate is not None:
                    matrix[row_index, column_index] = float(estimate)
        image = ax.imshow(matrix, cmap="RdYlBu", vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(np.arange(len(entities)), entities, rotation=40, ha="right")
        ax.set_yticks(np.arange(len(item_ids)), [f"Item {item_id}" for item_id in item_ids])
        ax.set_xticks(np.arange(-0.5, len(entities), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(item_ids), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.05)
        ax.tick_params(which="minor", bottom=False, left=False)
        for row_index in range(len(item_ids)):
            for column_index in range(len(entities)):
                value = matrix[row_index, column_index]
                if math.isfinite(value):
                    ax.text(
                        column_index,
                        row_index,
                        f"{value:.2f}",
                        ha="center",
                        va="center",
                        fontsize=7.2,
                        color=heatmap_text_color(image.cmap, image.norm, float(value)),
                    )
                else:
                    ax.text(column_index, row_index, "NA", ha="center", va="center", fontsize=6.5, color="#94a3b8")
        ax.set_title(f"{'B' if scale_index == 0 else 'C'}  {scale}", loc="left", fontweight="bold")
        ax.set_xlabel("Persona" if rows[0].get("entity_type") == "persona" else "Experimental condition")
    if image is not None:
        colorbar = fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.72, pad=0.02)
        colorbar.set_label("Quadratic weighted κ")
    fig.suptitle(f"Item-level repeat reliability — {dimension_label}", fontsize=15, fontweight="bold")
    fig.text(
        0.5,
        -0.012,
        "Each cell is the mean pairwise quadratic weighted κ across measurement-repeat pairs within that persona/condition; higher values indicate more stable item ratings.",
        ha="center",
        fontsize=8,
        color="#475569",
    )
    return save_figure(fig, out_dir / f"weighted_kappa_02_item_heatmap_{filename_suffix}.png")
