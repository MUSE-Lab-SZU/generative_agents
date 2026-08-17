"""Paper-style 2x2 outcome-by-study/severity/time boxplot."""

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
    DEFAULT_SEVERITY_ORDER,
)
from ...shared.plotting import plt


def plot_figure4(
    frame: pd.DataFrame,
    output_base: Path,
    options: PlotOptions,
    *,
    outcome_order: list[str] | None = None,
    study_order: list[str] | None = None,
    severity_order: list[str] | None = None,
) -> list[Path]:
    """Render the Figure-4-like 2x2 outcome x study/severity/time boxplot."""
    outcomes = _ordered_present(frame, "outcome", outcome_order)
    studies = _ordered_present(frame, "study", study_order)
    if severity_order:
        severities = _ordered_present(frame, "severity_group", severity_order)
    else:
        present_severities = _ordered_present(frame, "severity_group", None)
        severities = [
            value for value in DEFAULT_SEVERITY_ORDER if value in present_severities
        ]
        severities.extend(
            value for value in present_severities if value not in severities
        )
    _validate_exact(outcomes, 2, "outcome", "figure4")
    _validate_exact(studies, 2, "study", "figure4")
    if not severities or severities == ["unspecified"]:
        raise ValueError(
            "figure4 requires a populated severity_group/persona category column"
        )

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(12.4, 9.0),
        squeeze=False,
        sharey="row",
        constrained_layout=True,
    )
    panel_index = 0
    for row_index, outcome in enumerate(outcomes):
        for column_index, study in enumerate(studies):
            ax = axes[row_index][column_index]
            panel = frame[(frame["outcome"] == outcome) & (frame["study"] == study)]
            _draw_box_panel(
                ax,
                panel,
                severities,
                "severity_group",
                options,
                seed=1941 + panel_index,
            )
            letter = chr(ord("A") + panel_index)
            ax.set_title(
                f"{letter}  {study}: {outcome}",
                loc="left",
                fontsize=12,
                fontweight="semibold",
                pad=34 if options.category_labels == "top" else 12,
            )
            ax.set_ylabel(outcome if column_index == 0 else "", fontsize=10)
            ax.set_xlabel(
                "Severity group" if options.category_labels == "bottom" else ""
            )
            panel_index += 1
    _legend(fig, options)
    if options.title:
        fig.suptitle(options.title, fontsize=14, fontweight="semibold")
    fig.get_layout_engine().set(w_pad=5 / 72, h_pad=5 / 72, wspace=0.06, hspace=0.12)
    return _save(fig, output_base, options)
