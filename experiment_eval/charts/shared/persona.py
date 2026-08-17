"""Shared interval and heatmap primitives for persona charts."""

from __future__ import annotations

from typing import Any

import numpy as np

from .plotting import plt


def _draw_clipped_interval(
    ax: Any,
    *,
    estimate: float,
    lower: float | None,
    upper: float | None,
    y: float,
    color: str,
    limits: tuple[float, float],
) -> None:
    left, right = limits
    if lower is not None and upper is not None:
        ax.hlines(y, max(lower, left), min(upper, right), color=color, linewidth=1.8)
        if lower < left:
            ax.scatter(left, y, marker="<", color=color, s=32, clip_on=False)
        if upper > right:
            ax.scatter(right, y, marker=">", color=color, s=32, clip_on=False)
    ax.scatter(
        estimate, y, color=color, edgecolor="white", linewidth=0.6, s=48, zorder=3
    )


def _interval_text(estimate: Any, lower: Any, upper: Any) -> str:
    if estimate is None:
        return "NA"
    if lower is None or upper is None:
        return f"{estimate:.2f} [NA]"
    return f"{estimate:.2f} [{lower:.2f}, {upper:.2f}]"


def _heatmap(
    ax: Any,
    matrix: np.ndarray,
    *,
    personas: list[str],
    groups: list[str],
    title: str,
    cmap_name: str,
    vmin: float | None,
    vmax: float | None,
    formatter: Any,
    colorbar_label: str,
) -> None:
    masked = np.ma.masked_invalid(matrix)
    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad("#e5e7eb")
    image = ax.imshow(masked, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(groups)), groups)
    ax.set_yticks(range(len(personas)), personas)
    ax.set_title(title, fontsize=10.5)
    for row_index in range(len(personas)):
        for column_index in range(len(groups)):
            value = matrix[row_index, column_index]
            if not np.isfinite(value):
                text, color = "NA", "#475569"
            else:
                text = formatter(
                    float(value), personas[row_index], groups[column_index]
                )
                normalized = image.norm(float(value))
                red, green, blue, _ = image.cmap(normalized)
                luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
                color = "#111827" if luminance > 0.52 else "white"
            ax.text(
                column_index,
                row_index,
                text,
                ha="center",
                va="center",
                fontsize=7.3,
                color=color,
            )
    plt.colorbar(image, ax=ax, shrink=0.78, label=colorbar_label)
