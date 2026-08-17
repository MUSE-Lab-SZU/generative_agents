"""Paper-style 1x2 outcome-by-group/time boxplot."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ...shared.faceted import (
    PlotOptions,
    _draw_box_panel,
    _legend,
    _ordered_present,
    _save,
    _validate_exact,
)
from ...shared.plotting import plt


def plot_figure2(
    frame: pd.DataFrame,
    output_base: Path,
    options: PlotOptions,
    *,
    outcome_order: list[str] | None = None,
    group_order: list[str] | None = None,
) -> list[Path]:
    """Render the Figure-2-like 1x2 outcome x group/time boxplot."""
    outcomes = _ordered_present(frame, "outcome", outcome_order)
    groups = _ordered_present(frame, "group", group_order)
    studies = frame["study"].unique().tolist()
    _validate_exact(outcomes, 2, "outcome", "figure2")
    if len(studies) > 1:
        raise ValueError(
            f"figure2 does not pool studies; filter to one study first (found {studies})"
        )
    if not groups or groups == ["unspecified"]:
        raise ValueError("figure2 requires a populated group column")

    fig, axes = plt.subplots(
        1, 2, figsize=(11.2, 4.8), squeeze=False, constrained_layout=True
    )
    for index, outcome in enumerate(outcomes):
        ax = axes[0][index]
        panel = frame[frame["outcome"] == outcome]
        _draw_box_panel(ax, panel, groups, "group", options, seed=941 + index)
        letter = chr(ord("A") + index)
        ax.set_title(
            f"{letter}  {outcome}",
            loc="left",
            fontsize=12.5,
            fontweight="semibold",
            pad=34 if options.category_labels == "top" else 12,
        )
        ax.set_ylabel(outcome, fontsize=10)
        ax.set_xlabel("Group" if options.category_labels == "bottom" else "")
    _legend(fig, options)
    if options.title:
        fig.suptitle(options.title, fontsize=14, fontweight="semibold")
    fig.get_layout_engine().set(w_pad=5 / 72, h_pad=5 / 72, wspace=0.08, hspace=0.08)
    return _save(fig, output_base, options)
