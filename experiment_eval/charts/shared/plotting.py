"""Shared Matplotlib setup, styling, layout, and save helpers.

All plotting modules depend on this file instead of importing one another for
side effects.  Keeping the headless backend and font setup here also makes a
new figure module cheap to add and consistent with the existing outputs.
"""

from __future__ import annotations

import math
import os
import tempfile
import textwrap
from pathlib import Path
from typing import Any

CACHE_ROOT = Path(tempfile.gettempdir()) / "experiment_eval_chart_cache"
CACHE_ROOT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_ROOT / "mpl"))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_ROOT / "xdg"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D

from ...schema import (
    SCALE_RANGES,
    ExperimentRecord,
    group_color,
    kbd_color,
    repeat_style,
)

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = [
    "Noto Sans CJK JP",
    "Droid Sans Fallback",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False


def configure_simplified_chinese_font() -> str:
    """Register an explicit Simplified-Chinese face for reproducible figures.

    Noto's Linux CJK fonts are commonly installed as TrueType collections.
    Matplotlib otherwise selects the first face (usually Japanese), even when a
    Simplified-Chinese family is requested.  Extracting the SC face into the
    chart cache makes the selected glyph forms and bold weight deterministic.
    """
    font_cache = CACHE_ROOT / "fonts"
    font_cache.mkdir(parents=True, exist_ok=True)
    faces = [
        (
            Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
            font_cache / "NotoSansCJK-SC-Regular.ttf",
            2,
        ),
        (
            Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"),
            font_cache / "NotoSansCJK-SC-Bold.ttf",
            2,
        ),
    ]
    try:
        from fontTools.ttLib import TTCollection

        for collection_path, extracted_path, face_index in faces:
            if not collection_path.is_file():
                raise FileNotFoundError(collection_path)
            if not extracted_path.is_file():
                collection = TTCollection(collection_path)
                collection.fonts[face_index].save(extracted_path)
                collection.close()
            font_manager.fontManager.addfont(extracted_path)
        family = "Noto Sans CJK SC"
    except (ImportError, IndexError, OSError):
        fallback = Path("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf")
        if fallback.is_file():
            font_manager.fontManager.addfont(fallback)
            family = "Droid Sans Fallback"
        else:
            family = "DejaVu Sans"

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = [family, "Droid Sans Fallback", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    return family


def wrapped(value: str, width: int = 72) -> str:
    return textwrap.fill(
        value, width=width, break_long_words=False, break_on_hyphens=False
    )


def figure_grid(rows: int, cols: int, width: float = 5.5, height: float = 4.1):
    return plt.subplots(
        max(1, rows),
        max(1, cols),
        figsize=(max(7.0, width * max(cols, 1)), max(4.2, height * max(rows, 1))),
        squeeze=False,
        constrained_layout=True,
    )


def save_figure(fig: Any, png_path: Path) -> Path:
    """Save a chart as a 300-DPI PNG and release its Matplotlib resources."""
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    return png_path


def add_measurement_footer(
    fig: Any,
    records: list[ExperimentRecord],
    error_bar: str,
    labels: list[str],
    outer_summary: str = "none",
) -> None:
    ns = sorted({record.expected_repeats for record in records})
    endpoint_rule = (
        "observed endpoint: POST > NOW > last available; not inferred as follow-up"
    )
    outer_counts = sorted(
        {
            sum(1 for record in records if record.group == group)
            for group in {record.group for record in records}
        }
    )
    outer_note = (
        f"; outer n/group={','.join(map(str, outer_counts))}"
        if outer_summary == "mean-ci"
        else ""
    )
    fig.text(
        0.5,
        -0.025,
        f"n={','.join(map(str, ns))} Monte Carlo repeats/snapshot{outer_note}; error={error_bar}; {endpoint_rule}",
        ha="center",
        va="top",
        fontsize=8,
        color="#475569",
    )


def apply_score_y_axis(ax: Any, scale: str, y_axis: str, values: list[float]) -> None:
    if y_axis == "full" and scale in SCALE_RANGES:
        ax.set_ylim(*SCALE_RANGES[scale])
        return
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        return
    low, high = min(finite), max(finite)
    pad = max((high - low) * 0.12, 1.0)
    ax.set_ylim(low - pad, high + pad)


def measurement_error(stats: dict[str, Any], error_bar: str) -> tuple[float, float]:
    center = stats.get("score")
    if center is None or error_bar == "none":
        return (0.0, 0.0)
    if error_bar == "sd":
        sd = stats.get("sample_sd") or 0.0
        return (float(sd), float(sd))
    lower, upper = stats.get("ci95_lower"), stats.get("ci95_upper")
    if lower is None or upper is None:
        return (0.0, 0.0)
    return (float(center) - float(lower), float(upper) - float(center))


def series_color(record: ExperimentRecord, records: list[ExperimentRecord]) -> str:
    groups = {item.group for item in records}
    kbds = {item.kbd for item in records}
    return (
        kbd_color(record.kbd)
        if len(groups) == 1 and len(kbds) > 1
        else group_color(record.group)
    )


def series_legend(records: list[ExperimentRecord]) -> list[Line2D]:
    handles: list[Line2D] = []
    seen: set[str] = set()
    for record in records:
        if record.plot_label in seen:
            continue
        seen.add(record.plot_label)
        marker, linestyle = repeat_style(record.repeat_id)
        handles.append(
            Line2D(
                [0],
                [0],
                color=series_color(record, records),
                marker=marker,
                linestyle=linestyle,
                linewidth=2,
                label=record.plot_label,
            )
        )
    return handles


def lines_only_legend(records: list[ExperimentRecord]) -> list[Line2D]:
    handles: list[Line2D] = []
    cross_persona = (
        len({record.group for record in records}) == 1
        and len({record.kbd for record in records}) > 1
    )
    color_values = [record.kbd if cross_persona else record.group for record in records]
    seen_colors: set[str] = set()
    for record, value in zip(records, color_values):
        if value in seen_colors:
            continue
        seen_colors.add(value)
        handles.append(
            Line2D(
                [0],
                [0],
                color=series_color(record, records),
                linewidth=2.4,
                label=value,
            )
        )
    seen_repeats: set[str] = set()
    for record in records:
        if record.repeat_id in seen_repeats:
            continue
        seen_repeats.add(record.repeat_id)
        _, linestyle = repeat_style(record.repeat_id)
        handles.append(
            Line2D(
                [0],
                [0],
                color="#334155",
                linestyle=linestyle,
                linewidth=2.4,
                label=record.repeat_id,
            )
        )
    return handles


def heatmap_text_color(cmap: Any, norm: Any, value: float) -> str:
    red, green, blue, _ = cmap(norm(value))
    luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    return "#111827" if luminance > 0.56 else "white"
