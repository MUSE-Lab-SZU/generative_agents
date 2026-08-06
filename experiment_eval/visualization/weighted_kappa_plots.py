"""Compact weighted-kappa figures for ordinal PHQ-9/BDI-II items."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from ..schema import group_sort_key, scale_sort_key
from .scale_plots import plt, save_figure


def plot_repeat_pair_heatmap(payload: dict[str, Any], out_dir: Path, title_prefix: str) -> Path | None:
    rows = payload.get("matrix") or []
    scales = sorted({row["scale"] for row in rows}, key=scale_sort_key)
    repeats = sorted({row["repeat_a"] for row in rows} | {row["repeat_b"] for row in rows}, key=group_sort_key)
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
                    ax.text(column_index, row_index, f"{value:.2f}", ha="center", va="center", fontsize=7)
        ax.set_xticks(range(len(repeats)), repeats, rotation=45, ha="right")
        ax.set_yticks(range(len(repeats)), repeats)
        ax.set_xlabel("Measurement repeat")
        ax.set_ylabel("Measurement repeat")
        ax.set_title(scale)
    if image is not None:
        fig.colorbar(image, ax=axes.ravel().tolist(), label="Quadratic weighted κ", shrink=0.82)
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


def plot_item_forest(payload: dict[str, Any], out_dir: Path, title_prefix: str) -> Path | None:
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
            (row for row in rows if row["scale"] == scale and row["weights"] == "quadratic"),
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
                ax.text(-0.96, y_position, f"I{row['item_id']}: NA ({row.get('reason')})", fontsize=7, va="center")
                continue
            plotted = True
            lower, upper = row.get("ci95_lower"), row.get("ci95_upper")
            xerr = None
            if lower is not None and upper is not None:
                xerr = [[float(estimate) - float(lower)], [float(upper) - float(estimate)]]
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
                ax.scatter(float(sensitivity), y_position, marker="x", color="#b45309", s=30, label="linear" if y_position == 0 else None)
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


def plot_stratum_heatmap(payload: dict[str, Any], out_dir: Path, title_prefix: str) -> Path | None:
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
            if len({row["stratum_value"] for row in rows if row["stratum_type"] == stratum_type}) >= 2
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
        matrix[strata.index(row["stratum_value"]), scales.index(row["scale"])] = float(row["kappa"])
    fig, ax = plt.subplots(figsize=(max(5.5, 2.1 * len(scales)), max(4.0, 0.55 * len(strata) + 2.2)), constrained_layout=True)
    image = ax.imshow(matrix, vmin=-1, vmax=1, cmap="RdYlBu", aspect="auto")
    for row_index in range(len(strata)):
        for column_index in range(len(scales)):
            value = matrix[row_index, column_index]
            if not math.isnan(value):
                ax.text(column_index, row_index, f"{value:.2f}", ha="center", va="center", fontsize=8)
    ax.set_xticks(range(len(scales)), scales)
    ax.set_yticks(range(len(strata)), strata)
    ax.set_title(f"{title_prefix}: {eligible_type} × scale quadratic weighted Kappa")
    fig.colorbar(image, ax=ax, label="Quadratic weighted κ")
    return save_figure(fig, out_dir / f"weighted_kappa_03_{eligible_type}_scale_heatmap.png")


def render_weighted_kappa_figures(payload: dict[str, Any], out_dir: Path, title_prefix: str) -> list[Path]:
    """Render one compact figure set; skip sparse persona/group panels."""
    paths = [
        plot_repeat_pair_heatmap(payload, out_dir, title_prefix),
        plot_item_forest(payload, out_dir, title_prefix),
        plot_stratum_heatmap(payload, out_dir, title_prefix),
    ]
    return [path for path in paths if path is not None]
