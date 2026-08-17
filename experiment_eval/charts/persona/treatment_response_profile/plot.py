"""Paper-style vertically stacked persona treatment-response profiles."""

from __future__ import annotations

import string

from pathlib import Path

from typing import Any

import numpy as np

import pandas as pd

from ....schema import group_color

from ...shared.plotting import plt, save_figure

Y_LABELS = {
    "relative-profile": "Relative-profile Z-score",
    "common-reference": "Common-reference Z-score",
    "standardized-change": "Standardized change",
}


def plot_persona_treatment_response_profile(
    summary: pd.DataFrame,
    out_dir: Path,
    *,
    method: str,
    panel_order: list[str],
    persona_order: list[str],
    group_order: list[str],
    uncertainty: str = "none",
    title: str = "Persona treatment-response profile",
) -> Path | None:
    if summary.empty or summary["z_mean"].notna().sum() == 0:
        return None
    panels = [
        value for value in panel_order if value in set(summary["panel"].astype(str))
    ]
    if not panels:
        return None
    x = np.arange(len(persona_order), dtype=float)
    fig, axes = plt.subplots(
        len(panels),
        1,
        figsize=(
            max(7.2, 0.9 * len(persona_order) + 2.8),
            max(4.0, 3.45 * len(panels)),
        ),
        sharex=True,
        constrained_layout=True,
        squeeze=False,
    )
    layout_engine = fig.get_layout_engine()
    if layout_engine is not None:
        layout_engine.set(h_pad=0.12, hspace=0.08)
    marker_cycle = ["o", "s", "^", "D", "v", "P", "X"]
    for panel_index, panel in enumerate(panels):
        ax = axes[panel_index, 0]
        panel_rows = summary.loc[summary["panel"].astype(str) == panel]
        for group_index, group in enumerate(group_order):
            group_rows = panel_rows.loc[panel_rows["group"].astype(str) == group]
            lookup = {
                str(row["persona_id"]): row for row in group_rows.to_dict("records")
            }
            centers = np.asarray(
                [
                    (
                        float(lookup[value]["z_mean"])
                        if value in lookup and pd.notna(lookup[value]["z_mean"])
                        else np.nan
                    )
                    for value in persona_order
                ]
            )
            color = group_color(group)
            ax.plot(
                x,
                centers,
                color=color,
                marker=marker_cycle[group_index % len(marker_cycle)],
                markersize=4.2,
                linewidth=1.35,
                label=group,
                zorder=3,
            )
            if uncertainty == "ci95-band":
                lower = np.asarray(
                    [
                        (
                            float(lookup[value]["z_ci95_lower"])
                            if value in lookup
                            and pd.notna(lookup[value]["z_ci95_lower"])
                            else np.nan
                        )
                        for value in persona_order
                    ]
                )
                upper = np.asarray(
                    [
                        (
                            float(lookup[value]["z_ci95_upper"])
                            if value in lookup
                            and pd.notna(lookup[value]["z_ci95_upper"])
                            else np.nan
                        )
                        for value in persona_order
                    ]
                )
                ax.fill_between(
                    x, lower, upper, color=color, alpha=0.12, linewidth=0, zorder=1
                )
            elif uncertainty in {"ci95-bars", "se-bars"}:
                if uncertainty == "se-bars":
                    errors = np.asarray(
                        [
                            (
                                float(lookup[value]["z_se"])
                                if value in lookup and pd.notna(lookup[value]["z_se"])
                                else np.nan
                            )
                            for value in persona_order
                        ]
                    )
                    yerr: Any = errors
                else:
                    lower = np.asarray(
                        [
                            (
                                float(lookup[value]["z_ci95_lower"])
                                if value in lookup
                                and pd.notna(lookup[value]["z_ci95_lower"])
                                else np.nan
                            )
                            for value in persona_order
                        ]
                    )
                    upper = np.asarray(
                        [
                            (
                                float(lookup[value]["z_ci95_upper"])
                                if value in lookup
                                and pd.notna(lookup[value]["z_ci95_upper"])
                                else np.nan
                            )
                            for value in persona_order
                        ]
                    )
                    yerr = np.vstack([centers - lower, upper - centers])
                ax.errorbar(
                    x,
                    centers,
                    yerr=yerr,
                    fmt="none",
                    ecolor=color,
                    elinewidth=0.9,
                    capsize=2,
                    alpha=0.75,
                )
        scale_labels = sorted(panel_rows["scale"].astype(str).unique())
        panel_title = (
            panel if panel in scale_labels else f"{panel} ({', '.join(scale_labels)})"
        )
        ax.axhline(0, color="#64748b", linewidth=0.9, alpha=0.85, zorder=0)
        ax.grid(axis="both", color="#cbd5e1", alpha=0.28, linewidth=0.7)
        ax.set_ylabel(Y_LABELS[method])
        ax.set_title(panel_title, fontsize=10, pad=5)
        panel_letter = (
            string.ascii_uppercase[panel_index]
            if panel_index < 26
            else str(panel_index + 1)
        )
        ax.text(
            -0.115,
            1.025,
            panel_letter,
            transform=ax.transAxes,
            fontsize=12,
            va="bottom",
        )
        if panel_index == 0:
            ax.legend(
                frameon=False,
                loc="best",
                ncol=min(4, len(group_order)),
                title="Condition",
            )
    axes[-1, 0].set_xticks(x, persona_order, rotation=0)
    axes[-1, 0].set_xlabel("Persona")
    fig.suptitle(title, fontsize=12)
    fig.text(
        0.5,
        -0.012,
        "Change = post − baseline; negative values indicate symptom reduction for PHQ-9/BDI-II. "
        "Lines compare profiles across a single fixed persona order.",
        ha="center",
        va="top",
        fontsize=8,
        color="#475569",
    )
    return save_figure(
        fig,
        out_dir / f"persona_treatment_response_profile_{method.replace('-', '_')}.png",
    )
