"""Predictor category/importance table plus SHAP beeswarm."""

from __future__ import annotations

from pathlib import Path

from typing import Any

import numpy as np

from ...shared.plotting import plt, save_figure

CATEGORY_LABELS = {"baseline": "Baseline", "persona": "Persona", "behavior": "Behavior"}


def plot_persona_shap(
    analysis: dict[str, Any], out_dir: Path, *, max_features: int = 20
) -> Path:
    stats = analysis["feature_stats"].head(max_features).copy()
    original_indices = stats["column_index"].astype(int).tolist()
    shap_values = np.asarray(analysis["shap_values"])[:, original_indices]
    feature_values = np.asarray(analysis["feature_values"])[:, original_indices]
    row_count = len(stats)
    fig, (table_ax, swarm_ax) = plt.subplots(
        1,
        2,
        figsize=(13.8, max(6.0, 0.42 * row_count + 2.2)),
        gridspec_kw={"width_ratios": [1.45, 1.0]},
        sharey=True,
        constrained_layout=True,
    )
    y = np.arange(row_count)
    table_ax.set_xlim(0, 1)
    table_ax.set_ylim(row_count - 0.5, -1.5)
    table_ax.axis("off")
    table_ax.text(
        0.00, -0.9, "Feature category / feature", fontweight="bold", fontsize=9
    )
    table_ax.text(0.77, -0.9, "Direction", ha="center", fontweight="bold", fontsize=9)
    table_ax.text(0.98, -0.9, "Importance", ha="right", fontweight="bold", fontsize=9)
    previous_category = None
    for row_index, row in stats.iterrows():
        category = str(row["category"])
        if category != previous_category:
            table_ax.text(
                0.0,
                row_index,
                CATEGORY_LABELS.get(category, category.title()),
                fontweight="bold",
                va="center",
            )
        table_ax.text(
            0.22, row_index, str(row["feature_label"]), va="center", fontsize=8.5
        )
        table_ax.text(
            0.77, row_index, str(row["direction"]), ha="center", va="center", fontsize=9
        )
        table_ax.text(
            0.98,
            row_index,
            f"{float(row['proportional_importance_pct']):.1f}%",
            ha="right",
            va="center",
            fontsize=8.5,
        )
        table_ax.hlines(row_index + 0.48, 0, 1, color="#e2e8f0", linewidth=0.7)
        previous_category = category
    rng = np.random.default_rng(23)
    for row_index in range(row_count):
        values = feature_values[:, row_index]
        low, high = np.nanpercentile(values, [5, 95])
        normalized = (
            np.full_like(values, 0.5, dtype=float)
            if high <= low
            else np.clip((values - low) / (high - low), 0, 1)
        )
        jitter = rng.normal(0, 0.10, len(values))
        swarm_ax.scatter(
            shap_values[:, row_index],
            row_index + jitter,
            c=normalized,
            cmap="coolwarm",
            vmin=0,
            vmax=1,
            s=15,
            alpha=0.78,
            edgecolors="none",
        )
    swarm_ax.axvline(0, color="#64748b", linewidth=0.9)
    swarm_ax.set_ylim(row_count - 0.5, -1.5)
    swarm_ax.set_yticks([])
    swarm_ax.set_xlabel("SHAP value for predicted CBT relative benefit")
    swarm_ax.grid(axis="x", alpha=0.18)
    colorbar = fig.colorbar(
        plt.cm.ScalarMappable(cmap="coolwarm", norm=plt.Normalize(0, 1)),
        ax=swarm_ax,
        orientation="horizontal",
        fraction=0.05,
        pad=0.08,
    )
    colorbar.set_ticks([0, 1])
    colorbar.set_ticklabels(["Low", "High"])
    colorbar.set_label("Feature value", fontsize=8)
    fig.suptitle(
        "Baseline, persona, and behavior predictors of CBT relative benefit",
        fontsize=13,
        fontweight="bold",
    )
    return save_figure(fig, out_dir / "persona_predictor_shap.png")
