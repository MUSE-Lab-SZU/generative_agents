"""Shared options and drawing helpers for faceted boxplots."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from .plotting import plt

DEFAULT_TIME_COLORS = {"pre": "#D55E00", "post": "#56B4E9"}
DEFAULT_SEVERITY_ORDER = ["minimal", "mild", "moderate", "mod-severe", "severe"]


@dataclass
class PlotOptions:
    title: str | None = None
    time_order: list[str] = field(default_factory=lambda: ["pre", "post"])
    time_colors: dict[str, str] = field(
        default_factory=lambda: dict(DEFAULT_TIME_COLORS)
    )
    show_points: bool = False
    max_points: int = 24
    annotate_n: bool = False
    show_fliers: bool = False
    category_labels: str = "top"
    formats: list[str] = field(default_factory=lambda: ["png"])
    dpi: int = 300


def _validate_exact(
    values: list[str], expected: int, dimension: str, layout: str
) -> None:
    if len(values) != expected:
        raise ValueError(
            f"{layout} requires exactly {expected} {dimension} values after filtering; "
            f"found {len(values)}: {values}"
        )


def _save(fig: Any, output_base: Path, options: PlotOptions) -> list[Path]:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for extension in options.formats:
        path = output_base.with_suffix(f".{extension}")
        kwargs = {"bbox_inches": "tight"}
        if extension.lower() in {"png", "jpg", "jpeg", "tif", "tiff"}:
            kwargs["dpi"] = options.dpi
        fig.savefig(path, **kwargs)
        paths.append(path)
    plt.close(fig)
    return paths


def _ordered_present(
    frame: pd.DataFrame, column: str, requested: list[str] | None
) -> list[str]:
    present = list(dict.fromkeys(frame[column].astype(str).tolist()))
    if requested:
        return [value for value in requested if value in set(present)]
    return present


def _style_axis(ax: Any, categories: list[str], category_labels: str) -> None:
    centers = np.arange(len(categories), dtype=float)
    ax.set_xticks(centers, categories)
    if category_labels == "top":
        ax.tick_params(
            axis="x",
            top=True,
            labeltop=True,
            bottom=False,
            labelbottom=False,
            pad=4,
            length=0,
        )
        for tick in ax.get_xticklabels():
            tick.set_bbox(
                {
                    "facecolor": "white",
                    "edgecolor": "#64748B",
                    "linewidth": 0.8,
                    "pad": 2.5,
                }
            )
    else:
        ax.tick_params(
            axis="x", top=False, labeltop=False, bottom=True, labelbottom=True, length=0
        )
    ax.grid(axis="y", color="#CBD5E1", linewidth=0.65, alpha=0.6)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#334155")
    ax.spines["bottom"].set_color("#334155")
    ax.tick_params(axis="y", colors="#334155", labelsize=9)
    ax.tick_params(axis="x", labelsize=8.5)


def _draw_box_panel(
    ax: Any,
    frame: pd.DataFrame,
    categories: list[str],
    category_column: str,
    options: PlotOptions,
    *,
    seed: int,
) -> None:
    centers = np.arange(len(categories), dtype=float)
    width = (
        0.30 if len(options.time_order) == 2 else 0.62 / max(len(options.time_order), 1)
    )
    offsets = (
        np.linspace(-0.18, 0.18, num=len(options.time_order))
        if len(options.time_order) > 1
        else [0.0]
    )
    rng = np.random.default_rng(seed)
    counts: list[tuple[float, int]] = []

    for time_index, time_value in enumerate(options.time_order):
        arrays: list[np.ndarray] = []
        positions: list[float] = []
        for category_index, category in enumerate(categories):
            values = frame.loc[
                (frame[category_column] == category) & (frame["time"] == time_value),
                "value",
            ].to_numpy(dtype=float)
            if not len(values):
                continue
            position = float(centers[category_index] + offsets[time_index])
            arrays.append(values)
            positions.append(position)
            counts.append((position, len(values)))
            if options.show_points:
                point_values = values
                if len(point_values) > options.max_points:
                    selected = rng.choice(
                        len(point_values), size=options.max_points, replace=False
                    )
                    point_values = point_values[selected]
                jitter = rng.uniform(
                    -width * 0.27, width * 0.27, size=len(point_values)
                )
                ax.scatter(
                    position + jitter,
                    point_values,
                    s=11,
                    color="#0F172A",
                    alpha=0.42,
                    linewidths=0,
                    zorder=3,
                )
        if not arrays:
            continue
        artists = ax.boxplot(
            arrays,
            positions=positions,
            widths=width,
            patch_artist=True,
            whis=1.5,
            showfliers=options.show_fliers,
            manage_ticks=False,
            medianprops={"color": "#111827", "linewidth": 1.8},
            boxprops={"edgecolor": "#111827", "linewidth": 0.9},
            whiskerprops={"color": "#111827", "linewidth": 0.9},
            capprops={"color": "#111827", "linewidth": 0.9},
            flierprops={
                "marker": "o",
                "markersize": 2.5,
                "alpha": 0.35,
                "markeredgewidth": 0,
            },
        )
        color = options.time_colors[time_value]
        for box in artists["boxes"]:
            box.set_facecolor(color)
            box.set_alpha(0.88)

    _style_axis(ax, categories, options.category_labels)
    ax.margins(x=0.08)
    if options.annotate_n and counts:
        annotation_y = -0.035 if options.category_labels == "top" else 1.01
        vertical_alignment = "top" if options.category_labels == "top" else "bottom"
        for position, count in counts:
            ax.text(
                position,
                annotation_y,
                f"n={count}",
                transform=ax.get_xaxis_transform(),
                ha="center",
                va=vertical_alignment,
                fontsize=6.8,
                color="#475569",
                clip_on=False,
            )


def _legend(fig: Any, options: PlotOptions) -> None:
    handles = [
        Patch(
            facecolor=options.time_colors[value],
            edgecolor="#111827",
            linewidth=0.8,
            label=value.title(),
        )
        for value in options.time_order
    ]
    fig.legend(
        handles=handles,
        title="Time",
        loc="center left",
        bbox_to_anchor=(1.0, 0.5),
        frameon=False,
        borderaxespad=0.8,
    )
