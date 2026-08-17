"""Multi-column symptom effect forest figure (model effects + Cohen's d)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

import pandas as pd

from ....life_state import ITEM_LABELS_EN

from ....schema import slugify

from ...shared.plotting import plt, save_figure

EFFECT_COLORS = {"Time": "#2563eb", "Group": "#059669", "Group × Time": "#f59e0b"}


def _item_labels(scale: str, items: list[int]) -> list[str]:
    return [
        f"I{item:02d}  {ITEM_LABELS_EN.get(scale, {}).get(item, f'Item {item}')}"
        for item in items
    ]


def _symmetric_limit(lower: pd.Series, upper: pd.Series, floor: float = 0.25) -> float:
    values = np.concatenate([lower.to_numpy(dtype=float), upper.to_numpy(dtype=float)])
    finite = np.abs(values[np.isfinite(values)])
    return max(floor, float(np.max(finite)) * 1.08) if finite.size else floor


def plot_symptom_effect_forest(
    model_effects: pd.DataFrame,
    timepoint_effects: pd.DataFrame,
    out_dir: Path,
    *,
    scale: str,
    total_reference: bool = False,
    title: str | None = None,
) -> Path | None:
    if model_effects.empty or timepoint_effects.empty:
        return None
    effects = [
        value
        for value in ("Time", "Group", "Group × Time")
        if value in set(model_effects["effect"])
    ]
    timepoints = (
        timepoint_effects[["timepoint", "time_order"]]
        .drop_duplicates()
        .sort_values("time_order")["timepoint"]
        .astype(str)
        .tolist()
    )
    items = sorted(
        set(model_effects["item"].astype(int))
        & set(timepoint_effects["item"].astype(int))
    )
    y = np.arange(len(items))
    columns = len(effects) + len(timepoints)
    fig, axes = plt.subplots(
        1,
        columns,
        figsize=(3.05 * columns + 1.9, max(6.0, 0.43 * len(items) + 2.0)),
        sharey=True,
        squeeze=False,
        constrained_layout=True,
    )
    axes_flat = axes.ravel()
    for index, effect in enumerate(effects):
        ax = axes_flat[index]
        selected = (
            model_effects.loc[model_effects["effect"] == effect]
            .set_index("item")
            .reindex(items)
        )
        center = selected["estimate"].to_numpy(dtype=float)
        lower = selected["ci95_lower"].to_numpy(dtype=float)
        upper = selected["ci95_upper"].to_numpy(dtype=float)
        errors = np.vstack([center - lower, upper - center])
        color = EFFECT_COLORS[effect]
        ax.errorbar(
            center,
            y,
            xerr=errors,
            fmt="o",
            color=color,
            ecolor=color,
            capsize=2,
            markersize=4,
        )
        ax.axvline(0, color="#64748b", linestyle="--", linewidth=0.9)
        limit = _symmetric_limit(selected["ci95_lower"], selected["ci95_upper"])
        ax.set_xlim(-limit, limit)
        ax.set_title(
            f"{'A · ' if index == 0 else ''}{effect}",
            fontsize=10,
            color=color,
            fontweight="bold",
        )
        ax.set_xlabel("Model coefficient (95% CI)", fontsize=8)
        significant = selected["significant_fdr"].fillna(False).to_numpy(dtype=bool)
        for row_index, is_significant in enumerate(significant):
            if is_significant and np.isfinite(upper[row_index]):
                ax.text(
                    upper[row_index],
                    row_index,
                    " *",
                    va="center",
                    fontsize=9,
                    color=color,
                )
        ax.grid(axis="x", alpha=0.18)
    for offset, timepoint in enumerate(timepoints):
        ax = axes_flat[len(effects) + offset]
        selected = (
            timepoint_effects.loc[timepoint_effects["timepoint"] == timepoint]
            .set_index("item")
            .reindex(items)
        )
        center = selected["cohen_d"].to_numpy(dtype=float)
        lower = selected["ci95_lower"].to_numpy(dtype=float)
        upper = selected["ci95_upper"].to_numpy(dtype=float)
        errors = np.vstack([center - lower, upper - center])
        ax.errorbar(
            center,
            y,
            xerr=errors,
            fmt="o",
            color="#111827",
            ecolor="#334155",
            capsize=2,
            markersize=4,
        )
        ax.axvline(0, color="#64748b", linestyle="--", linewidth=0.9)
        if total_reference:
            references = selected["total_score_reference_d"].dropna().unique()
            if len(references):
                ax.axvline(
                    float(references[0]), color="#dc2626", linewidth=1.15, alpha=0.85
                )
        limit = _symmetric_limit(
            selected["ci95_lower"], selected["ci95_upper"], floor=0.5
        )
        if total_reference and "total_score_reference_d" in selected:
            references = selected["total_score_reference_d"].dropna().abs()
            if not references.empty:
                limit = max(limit, float(references.max()) * 1.08)
        ax.set_xlim(-limit, limit)
        ax.set_title(
            f"{'B · ' if offset == 0 else ''}{timepoint}",
            fontsize=10,
            fontweight="bold",
        )
        ax.set_xlabel("CBT vs control Cohen's d", fontsize=8)
        significant = selected["significant_fdr"].fillna(False).to_numpy(dtype=bool)
        for row_index, is_significant in enumerate(significant):
            if is_significant and np.isfinite(upper[row_index]):
                ax.text(upper[row_index], row_index, " *", va="center", fontsize=9)
        ax.grid(axis="x", alpha=0.18)
    axes_flat[0].set_yticks(y, _item_labels(scale, items), fontsize=8)
    axes_flat[0].invert_yaxis()
    if len(timepoints):
        axes_flat[len(effects)].tick_params(labelleft=True)
    fig.text(0.02, 0.98, "A", fontsize=13, fontweight="bold", va="top")
    panel_b_x = (len(effects) + 0.1) / columns
    fig.text(panel_b_x, 0.98, "B", fontsize=13, fontweight="bold", va="top")
    fig.suptitle(
        title or f"{scale} item-level symptom effects", fontsize=13, fontweight="bold"
    )
    reference_note = "; red line = total-score d" if total_reference else ""
    fig.text(
        0.5,
        -0.015,
        "A: time is scaled from baseline=0 to last assessment=1; Group is the baseline CBT−control difference. "
        "B: negative d favors CBT; * Benjamini–Hochberg FDR q≤.05"
        + reference_note
        + ".",
        ha="center",
        fontsize=8,
        color="#475569",
    )
    return save_figure(fig, out_dir / f"symptom_effect_forest_{slugify(scale)}.png")
