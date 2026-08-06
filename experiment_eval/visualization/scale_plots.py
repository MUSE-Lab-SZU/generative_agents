"""Presentation and backward-compatible appendix chart rendering."""

from __future__ import annotations

import math
import os
import tempfile
import textwrap
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any

CACHE_ROOT = Path(tempfile.gettempdir()) / "experiment_eval_chart_cache"
CACHE_ROOT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_ROOT / "mpl"))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_ROOT / "xdg"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy.stats import t

# The report titles contain Chinese.  The server ships Noto CJK but matplotlib's
# default DejaVu font does not cover those glyphs, which otherwise produces boxes.
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "Droid Sans Fallback", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

from ..loader import available_severities
from ..schema import SCALE_RANGES, ExperimentRecord, group_color, kbd_color, label_display, repeat_style, slugify
from ..statistics import (
    delta_ci,
    endpoint_label_for_record,
    labels_with_score,
    score_at,
    stat_at,
    trajectory_metrics,
)


MEASUREMENT_METRICS = {
    "sample_sd": "Sample SD",
    "ci95_width": "95% CI width",
    "category_pairwise_flip_rate": "Category pairwise flip rate",
}


def _wrapped(value: str, width: int = 72) -> str:
    return textwrap.fill(value, width=width, break_long_words=False, break_on_hyphens=False)


def _grid(rows: int, cols: int, width: float = 5.5, height: float = 4.1):
    return plt.subplots(
        max(1, rows),
        max(1, cols),
        figsize=(max(7.0, width * max(cols, 1)), max(4.2, height * max(rows, 1))),
        squeeze=False,
        constrained_layout=True,
    )


def save_figure(fig: Any, png_path: Path) -> Path:
    """Save every chart as a 300-DPI PNG."""
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    return png_path


def _footer(
    fig: Any,
    records: list[ExperimentRecord],
    error_bar: str,
    labels: list[str],
    outer_summary: str = "none",
) -> None:
    ns = sorted({record.expected_repeats for record in records})
    endpoint_rule = "observed endpoint: POST > NOW > last available; not inferred as follow-up"
    outer_counts = sorted(
        {sum(1 for record in records if record.group == group) for group in {record.group for record in records}}
    )
    outer_note = f"; outer n/group={','.join(map(str, outer_counts))}" if outer_summary == "mean-ci" else ""
    fig.text(
        0.5,
        -0.025,
        f"n={','.join(map(str, ns))} Monte Carlo repeats/snapshot{outer_note}; error={error_bar}; {endpoint_rule}",
        ha="center",
        va="top",
        fontsize=8,
        color="#475569",
    )


def _apply_y_axis(ax: Any, scale: str, y_axis: str, values: list[float]) -> None:
    if y_axis == "full" and scale in SCALE_RANGES:
        ax.set_ylim(*SCALE_RANGES[scale])
        return
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        return
    low, high = min(finite), max(finite)
    pad = max((high - low) * 0.12, 1.0)
    ax.set_ylim(low - pad, high + pad)


def _measurement_error(stats: dict[str, Any], error_bar: str) -> tuple[float, float]:
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


def _series_color(record: ExperimentRecord, records: list[ExperimentRecord]) -> str:
    """Use KBD colors when a same-group batch compares multiple personas."""
    groups = {item.group for item in records}
    kbds = {item.kbd for item in records}
    if len(groups) == 1 and len(kbds) > 1:
        return kbd_color(record.kbd)
    return group_color(record.group)


def _series_legend(records: list[ExperimentRecord]) -> list[Line2D]:
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
                color=_series_color(record, records),
                marker=marker,
                linestyle=linestyle,
                linewidth=2,
                label=record.plot_label,
            )
        )
    return handles


def _lines_only_legend(records: list[ExperimentRecord]) -> list[Line2D]:
    """Build a compact color-key plus repeat-line-style legend."""
    handles: list[Line2D] = []
    same_group_cross_kbd = len({record.group for record in records}) == 1 and len(
        {record.kbd for record in records}
    ) > 1
    color_values = [record.kbd if same_group_cross_kbd else record.group for record in records]
    seen_colors: set[str] = set()
    for record, value in zip(records, color_values):
        if value in seen_colors:
            continue
        seen_colors.add(value)
        handles.append(
            Line2D([0], [0], color=_series_color(record, records), linewidth=2.4, label=value)
        )
    seen_repeats: set[str] = set()
    for record in records:
        if record.repeat_id in seen_repeats:
            continue
        seen_repeats.add(record.repeat_id)
        _, linestyle = repeat_style(record.repeat_id)
        handles.append(
            Line2D([0], [0], color="#334155", linestyle=linestyle, linewidth=2.4, label=record.repeat_id)
        )
    return handles


def plot_trajectory_lines_only(
    records: list[ExperimentRecord],
    labels: list[str],
    scale: str,
    out_dir: Path,
    title_prefix: str,
    *,
    y_axis: str,
) -> Path:
    """Plot only mean trajectories: no markers, error bars, or inner-repeat points."""
    severities = available_severities(records)
    fig, axes = _grid(1, len(severities), height=4.8)
    x_ticks = np.arange(len(labels))
    for column, severity in enumerate(severities):
        ax = axes[0][column]
        panel_records = [record for record in records if record.severity == severity]
        panel_values: list[float] = []
        for record in panel_records:
            available = labels_with_score(record, labels, scale)
            if not available:
                continue
            xs = [labels.index(label) for label in available]
            ys = [float(score_at(record, scale, label)) for label in available]
            _, linestyle = repeat_style(record.repeat_id)
            ax.plot(
                xs,
                ys,
                color=_series_color(record, panel_records),
                linestyle=linestyle,
                linewidth=2.2,
                alpha=0.94,
            )
            panel_values.extend(ys)
        ax.set_title(severity)
        ax.set_xticks(x_ticks, [label_display(label) for label in labels], rotation=25 if len(labels) > 7 else 0)
        ax.set_ylabel(f"{scale} primary score")
        ax.grid(alpha=0.20)
        _apply_y_axis(ax, scale, y_axis, panel_values)
    handles = _lines_only_legend(records)
    if handles:
        fig.legend(handles=handles, loc="center left", bbox_to_anchor=(1.0, 0.5), ncol=1, frameon=False, fontsize=8)
    fig.suptitle(_wrapped(f"{title_prefix}: {scale} trajectory — lines only"), fontsize=13)
    _footer(fig, records, "none (lines only)", labels)
    return save_figure(fig, out_dir / f"presentation_00_{slugify(scale)}_trajectory_lines_only.png")


def render_lines_only(
    records: list[ExperimentRecord],
    labels: list[str],
    scales: list[str],
    out_dir: Path,
    title_prefix: str,
    *,
    y_axis: str,
) -> list[Path]:
    return [
        plot_trajectory_lines_only(records, labels, scale, out_dir, title_prefix, y_axis=y_axis)
        for scale in scales
    ]


def _draw_outer_summary(ax: Any, records: list[ExperimentRecord], labels: list[str], scale: str) -> None:
    grouped: dict[str, list[ExperimentRecord]] = defaultdict(list)
    for record in records:
        grouped[record.group].append(record)
    for group, group_records in grouped.items():
        xs: list[int] = []
        centers: list[float] = []
        lowers: list[float] = []
        uppers: list[float] = []
        for index, label in enumerate(labels):
            values = [score_at(record, scale, label) for record in group_records]
            clean = [float(value) for value in values if value is not None]
            if not clean:
                continue
            center = mean(clean)
            if len(clean) > 1:
                half = float(t.ppf(0.975, len(clean) - 1)) * stdev(clean) / math.sqrt(len(clean))
            else:
                half = 0.0
            xs.append(index)
            centers.append(center)
            lowers.append(center - half)
            uppers.append(center + half)
        if centers:
            color = group_color(group)
            ax.plot(xs, centers, color=color, linewidth=3.2, alpha=0.95, label=f"{group} outer mean")
            ax.fill_between(xs, lowers, uppers, color=color, alpha=0.12)


def plot_trajectory_with_ci(
    records: list[ExperimentRecord],
    labels: list[str],
    scale: str,
    out_dir: Path,
    title_prefix: str,
    *,
    error_bar: str,
    y_axis: str,
    show_repeat_points: bool,
    outer_summary: str,
) -> Path:
    severities = available_severities(records)
    fig, axes = _grid(1, len(severities), height=4.8)
    x_ticks = np.arange(len(labels))
    all_values: list[float] = []
    for column, severity in enumerate(severities):
        ax = axes[0][column]
        panel_records = [record for record in records if record.severity == severity]
        for record in panel_records:
            marker, linestyle = repeat_style(record.repeat_id)
            xs: list[int] = []
            ys: list[float] = []
            lower_error: list[float] = []
            upper_error: list[float] = []
            for index, label in enumerate(labels):
                stats = record.series.get(scale, {}).get(label)
                if not stats or stats.get("score") is None:
                    continue
                value = float(stats["score"])
                low, high = _measurement_error(stats, error_bar)
                xs.append(index)
                ys.append(value)
                lower_error.append(low)
                upper_error.append(high)
                all_values.extend([value - low, value + high])
                if show_repeat_points:
                    points = stats.get("values", [])
                    jitter = np.linspace(-0.10, 0.10, len(points)) if points else []
                    ax.scatter(
                        np.asarray([index] * len(points)) + jitter,
                        points,
                        s=12,
                        color=_series_color(record, panel_records),
                        alpha=0.18,
                        linewidths=0,
                        zorder=1,
                    )
            if not ys:
                continue
            alpha = 0.45 if outer_summary == "mean-ci" else 0.92
            ax.errorbar(
                xs,
                ys,
                yerr=np.asarray([lower_error, upper_error]) if error_bar != "none" else None,
                color=_series_color(record, panel_records),
                marker=marker,
                linestyle=linestyle,
                linewidth=1.8,
                markersize=5,
                capsize=3,
                alpha=alpha,
                zorder=2,
            )
        if outer_summary == "mean-ci":
            _draw_outer_summary(ax, panel_records, labels, scale)
        ax.set_title(severity)
        ax.set_xticks(x_ticks, [label_display(label) for label in labels], rotation=25 if len(labels) > 7 else 0)
        ax.set_ylabel(f"{scale} primary score")
        ax.grid(alpha=0.22)
        _apply_y_axis(ax, scale, y_axis, all_values)
    handles = _series_legend(records)
    if outer_summary == "mean-ci":
        for group in sorted({record.group for record in records}):
            handles.append(Line2D([0], [0], color=group_color(group), linewidth=3.2, label=f"{group} outer mean ± 95% CI"))
    if handles:
        fig.legend(handles=handles, loc="center left", bbox_to_anchor=(1.0, 0.5), ncol=1, frameon=False, fontsize=8)
    error_label = {"ci95": "95% CI", "sd": "SD", "none": "no error bars"}[error_bar]
    fig.suptitle(_wrapped(f"{title_prefix}: {scale} trajectory with {error_label}"), fontsize=13)
    _footer(fig, records, error_bar, labels, outer_summary)
    return save_figure(fig, out_dir / f"presentation_01_{slugify(scale)}_trajectory_with_ci.png")


def plot_endpoint_delta_with_ci(
    records: list[ExperimentRecord], labels: list[str], scales: list[str], out_dir: Path, title_prefix: str
) -> Path:
    severities = available_severities(records)
    fig, axes = _grid(len(scales), len(severities), height=3.9)
    for row, scale in enumerate(scales):
        for column, severity in enumerate(severities):
            ax = axes[row][column]
            panel_records = [record for record in records if record.severity == severity]
            for index, record in enumerate(panel_records):
                interval = delta_ci(record, labels, scale)
                estimate = interval.get("estimate")
                if estimate is None:
                    continue
                lower, upper = interval.get("lower"), interval.get("upper")
                crosses = lower is not None and upper is not None and lower <= 0 <= upper
                yerr = None
                if lower is not None and upper is not None:
                    yerr = np.asarray([[estimate - lower], [upper - estimate]])
                ax.errorbar(
                    [index],
                    [estimate],
                    yerr=yerr,
                    marker="o",
                    markersize=7,
                    markerfacecolor="white" if crosses else _series_color(record, panel_records),
                    markeredgecolor=_series_color(record, panel_records),
                    color=_series_color(record, panel_records),
                    capsize=4,
                    linewidth=1.6,
                )
                if crosses:
                    ax.annotate("CI crosses 0", (index, estimate), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=7)
            ax.axhline(0, color="#111827", linewidth=0.9)
            ax.set_xticks(range(len(panel_records)), [record.plot_label for record in panel_records], rotation=25, ha="right")
            ax.set_title(f"{scale} — {severity}")
            ax.set_ylabel("Observed endpoint − baseline")
            ax.grid(axis="y", alpha=0.22)
    fig.suptitle(_wrapped(f"{title_prefix}: observed-endpoint change with 95% difference CI"), fontsize=13)
    ns = sorted({record.expected_repeats for record in records})
    fig.text(
        0.5,
        -0.025,
        f"n={','.join(map(str, ns))} repeats/snapshot; observed endpoint: POST > NOW > last available (not inferred follow-up). "
        "Error=95% difference CI; Monte Carlo measurement uncertainty only, not patient or outer-experiment uncertainty. "
        "Default Welch unless metadata explicitly declares pairing.",
        ha="center",
        va="top",
        fontsize=8,
        color="#475569",
    )
    return save_figure(fig, out_dir / "presentation_02_endpoint_delta_with_ci.png")


def plot_best_change_and_rebound(
    records: list[ExperimentRecord], labels: list[str], scales: list[str], out_dir: Path, title_prefix: str
) -> Path:
    severities = available_severities(records)
    fig, axes = _grid(len(scales), len(severities), height=4.0)
    definitions = [
        ("best_change_from_baseline", "Baseline→nadir", "#2563eb"),
        ("rebound_from_nadir", "Nadir→endpoint", "#dc2626"),
        ("endpoint_change", "Baseline→endpoint", "#64748b"),
    ]
    for row, scale in enumerate(scales):
        for column, severity in enumerate(severities):
            ax = axes[row][column]
            panel_records = [record for record in records if record.severity == severity]
            x = np.arange(len(panel_records))
            width = 0.24
            for metric_index, (field, label, color) in enumerate(definitions):
                values = [trajectory_metrics(record, labels, scale).get(field) for record in panel_records]
                ax.bar(x + (metric_index - 1) * width, values, width=width, label=label, color=color, alpha=0.86)
            for index, record in enumerate(panel_records):
                metrics = trajectory_metrics(record, labels, scale)
                best = metrics.get("best_change_from_baseline")
                rebound = metrics.get("rebound_from_nadir")
                if best is not None and best < 0 and rebound is not None and rebound > 0:
                    ax.annotate(
                        f"rebound after {label_display(metrics['best_timepoint'])}",
                        (index, rebound),
                        xytext=(0, 6),
                        textcoords="offset points",
                        ha="center",
                        va="bottom",
                        fontsize=7,
                    )
            ax.axhline(0, color="#111827", linewidth=0.8)
            ax.set_xticks(x, [record.plot_label for record in panel_records], rotation=25, ha="right")
            ax.set_title(f"{scale} — {severity}")
            ax.set_ylabel("Score change")
            ax.grid(axis="y", alpha=0.22)
            ax.margins(y=0.18)
    handles = [Line2D([0], [0], color=color, linewidth=8, label=label) for _, label, color in definitions]
    fig.legend(handles=handles, loc="center left", bbox_to_anchor=(1.0, 0.5), ncol=1, frameon=False)
    fig.suptitle(_wrapped(f"{title_prefix}: best change, rebound, and endpoint change"), fontsize=13)
    _footer(fig, records, "none (point summaries)", labels)
    return save_figure(fig, out_dir / "presentation_03_best_change_and_rebound.png")


def _text_color(cmap: Any, norm: Any, value: float) -> str:
    red, green, blue, _ = cmap(norm(value))
    luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    return "#111827" if luminance > 0.56 else "white"


def plot_measurement_reliability_heatmaps(
    records: list[ExperimentRecord], labels: list[str], scales: list[str], out_dir: Path, title_prefix: str
) -> list[Path]:
    paths: list[Path] = []
    for scale in scales:
        for metric, metric_label in MEASUREMENT_METRICS.items():
            matrix = np.asarray(
                [
                    [
                        stat_at(record, scale, label, metric)
                        if stat_at(record, scale, label, metric) is not None
                        else np.nan
                        for label in labels
                    ]
                    for record in records
                ],
                dtype=float,
            )
            fig, ax = plt.subplots(
                figsize=(max(7.0, 1.05 * len(labels) + 3.0), max(3.5, 0.55 * len(records) + 2.0)),
                constrained_layout=True,
            )
            cmap_name = "viridis" if metric != "category_pairwise_flip_rate" else "magma"
            finite = matrix[np.isfinite(matrix)]
            vmin = 0.0 if metric == "category_pairwise_flip_rate" else (float(finite.min()) if finite.size else 0.0)
            vmax = 1.0 if metric == "category_pairwise_flip_rate" else (float(finite.max()) if finite.size else 1.0)
            if math.isclose(vmin, vmax):
                vmax = vmin + 1.0
            im = ax.imshow(matrix, aspect="auto", cmap=cmap_name, vmin=vmin, vmax=vmax)
            ax.set_xticks(range(len(labels)), [label_display(label) for label in labels])
            ax.set_yticks(range(len(records)), [record.plot_label for record in records])
            ax.set_title(_wrapped(f"{title_prefix}: {scale} measurement reliability — {metric_label}"), fontsize=12)
            for y, record in enumerate(records):
                for x, label in enumerate(labels):
                    value = matrix[y, x]
                    if not np.isfinite(value):
                        continue
                    modal = stat_at(record, scale, label, "modal_confidence")
                    text = f"{value:.2f}" + (f"\nm={modal:.2f}" if modal is not None else "")
                    ax.text(x, y, text, ha="center", va="center", fontsize=7, color=_text_color(im.cmap, im.norm, value))
            fig.colorbar(im, ax=ax, label=metric_label)
            ns = sorted({record.expected_repeats for record in records})
            fig.text(
                0.5,
                -0.015,
                f"n={','.join(map(str, ns))} repeats/cell; error=N/A; endpoint rule for derived metrics: "
                "POST > NOW > last available; not inferred follow-up. m=modal confidence; rows=independent outer experiments.",
                ha="center",
                fontsize=8,
            )
            path = out_dir / f"presentation_04_{slugify(scale)}_{metric}_measurement_reliability_heatmap.png"
            paths.append(save_figure(fig, path))
    return paths


def _appendix_final_delta(
    records: list[ExperimentRecord], labels: list[str], scales: list[str], out_dir: Path, title_prefix: str
) -> Path:
    severities = available_severities(records)
    fig, axes = _grid(len(scales), len(severities), height=3.7)
    for row, scale in enumerate(scales):
        for column, severity in enumerate(severities):
            ax = axes[row][column]
            panel = [record for record in records if record.severity == severity]
            values = [trajectory_metrics(record, labels, scale).get("endpoint_change") for record in panel]
            bars = ax.bar(range(len(panel)), values, color=[_series_color(record, panel) for record in panel], alpha=0.86)
            ax.axhline(0, color="#111827", linewidth=0.8)
            ax.set_xticks(range(len(panel)), [record.plot_label for record in panel], rotation=25, ha="right")
            ax.set_title(f"{scale} final delta — {severity}")
            ax.set_ylabel("Observed endpoint − baseline")
            ax.grid(axis="y", alpha=0.22)
            for bar, value in zip(bars, values):
                if value is not None:
                    ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.1f}", ha="center", va="bottom" if value >= 0 else "top", fontsize=8)
    fig.suptitle(_wrapped(f"{title_prefix}: final change by severity"), fontsize=13)
    _footer(fig, records, "none", labels)
    return save_figure(fig, out_dir / "01_final_delta_bars_by_severity.png")


def _appendix_trajectory(
    records: list[ExperimentRecord], labels: list[str], scale: str, out_dir: Path, title_prefix: str, index: int
) -> Path:
    severities = available_severities(records)
    fig, axes = _grid(1, len(severities), height=4.3)
    for column, severity in enumerate(severities):
        ax = axes[0][column]
        for record in [item for item in records if item.severity == severity]:
            available = labels_with_score(record, labels, scale)
            xs = [labels.index(label) for label in available]
            ys = [score_at(record, scale, label) for label in available]
            marker, linestyle = repeat_style(record.repeat_id)
            ax.plot(xs, ys, color=_series_color(record, records), marker=marker, linestyle=linestyle, linewidth=2)
        ax.set_xticks(range(len(labels)), [label_display(label) for label in labels], rotation=25 if len(labels) > 7 else 0)
        ax.set_title(severity)
        ax.set_ylabel("Primary score")
        ax.grid(alpha=0.22)
    handles = _series_legend(records)
    if handles:
        fig.legend(handles=handles, loc="center left", bbox_to_anchor=(1.0, 0.5), ncol=1, frameon=False, fontsize=8)
    fig.suptitle(_wrapped(f"{title_prefix}: {scale} trajectory by severity"), fontsize=13)
    _footer(fig, records, "none", labels)
    return save_figure(fig, out_dir / f"{index:02d}_{slugify(scale)}_trajectory_by_severity.png")


def _appendix_delta_trajectory(
    records: list[ExperimentRecord], labels: list[str], scales: list[str], out_dir: Path, title_prefix: str, index: int
) -> Path:
    severities = available_severities(records)
    fig, axes = _grid(len(scales), len(severities), height=3.7)
    for row, scale in enumerate(scales):
        for column, severity in enumerate(severities):
            ax = axes[row][column]
            for record in [item for item in records if item.severity == severity]:
                available = labels_with_score(record, labels, scale)
                if not available:
                    continue
                baseline = score_at(record, scale, available[0])
                ys = [score_at(record, scale, label) - baseline for label in available]
                marker, linestyle = repeat_style(record.repeat_id)
                ax.plot(
                    [labels.index(label) for label in available],
                    ys,
                    color=_series_color(record, records),
                    marker=marker,
                    linestyle=linestyle,
                )
            ax.axhline(0, color="#111827", linewidth=0.8)
            ax.set_xticks(range(len(labels)), [label_display(label) for label in labels], rotation=25 if len(labels) > 7 else 0)
            ax.set_title(f"{scale} — {severity}")
            ax.set_ylabel("Primary score − baseline")
            ax.grid(alpha=0.22)
    handles = _series_legend(records)
    if handles:
        fig.legend(handles=handles, loc="center left", bbox_to_anchor=(1.0, 0.5), ncol=1, frameon=False, fontsize=8)
    fig.suptitle(_wrapped(f"{title_prefix}: delta from baseline"), fontsize=13)
    _footer(fig, records, "none", labels)
    return save_figure(fig, out_dir / f"{index:02d}_delta_from_baseline_by_severity.png")


def _appendix_volatility(
    records: list[ExperimentRecord], labels: list[str], scales: list[str], out_dir: Path, title_prefix: str, index: int
) -> Path:
    severities = available_severities(records)
    fig, axes = _grid(len(scales), len(severities), height=3.7)
    definitions = [("trajectory_volatility", "Trajectory volatility", "#64748b"), ("max_upward_step", "Max upward step", "#ef4444")]
    for row, scale in enumerate(scales):
        for column, severity in enumerate(severities):
            ax = axes[row][column]
            panel = [record for record in records if record.severity == severity]
            x = np.arange(len(panel))
            for metric_index, (field, label, color) in enumerate(definitions):
                values = [trajectory_metrics(record, labels, scale).get(field) for record in panel]
                ax.bar(x + (metric_index - 0.5) * 0.35, values, width=0.35, color=color, label=label, alpha=0.85)
            ax.set_xticks(x, [record.plot_label for record in panel], rotation=25, ha="right")
            ax.set_title(f"{scale} — {severity}")
            ax.set_ylabel("Score units")
            ax.grid(axis="y", alpha=0.22)
    fig.legend(
        handles=[Line2D([0], [0], color=color, linewidth=8, label=label) for _, label, color in definitions],
        loc="center left",
        bbox_to_anchor=(1.0, 0.5),
        ncol=1,
        frameon=False,
    )
    fig.suptitle(_wrapped(f"{title_prefix}: trajectory volatility and upward steps"), fontsize=13)
    _footer(fig, records, "none", labels)
    return save_figure(fig, out_dir / f"{index:02d}_stability_rebound_by_severity.png")


def _appendix_endpoint_uncertainty(
    records: list[ExperimentRecord], labels: list[str], scales: list[str], out_dir: Path, title_prefix: str, index: int
) -> Path:
    severities = available_severities(records)
    fig, axes = _grid(len(scales), len(severities), height=3.7)
    for row, scale in enumerate(scales):
        for column, severity in enumerate(severities):
            ax = axes[row][column]
            panel = [record for record in records if record.severity == severity]
            endpoints = [endpoint_label_for_record(record, labels, scale) for record in panel]
            scores = [score_at(record, scale, endpoint) for record, endpoint in zip(panel, endpoints)]
            sds = [stat_at(record, scale, endpoint, "sample_sd") for record, endpoint in zip(panel, endpoints)]
            ax.bar(
                range(len(panel)),
                scores,
                yerr=sds,
                capsize=4,
                color=[_series_color(record, panel) for record in panel],
                alpha=0.82,
            )
            ax.set_xticks(range(len(panel)), [record.plot_label for record in panel], rotation=25, ha="right")
            ax.set_title(f"{scale} endpoint — {severity}")
            ax.set_ylabel("Primary score ± within-snapshot SD")
            ax.grid(axis="y", alpha=0.22)
    fig.suptitle(_wrapped(f"{title_prefix}: endpoint repeat uncertainty"), fontsize=13)
    _footer(fig, records, "sd", labels)
    return save_figure(fig, out_dir / f"{index:02d}_endpoint_repeat_uncertainty_by_severity.png")


def _appendix_delta_heatmap(
    records: list[ExperimentRecord], labels: list[str], scales: list[str], out_dir: Path, title_prefix: str, index: int
) -> Path:
    fig, axes = _grid(1, len(scales), width=4.8, height=max(4.0, 0.5 * len(records) + 2.0))
    for column, scale in enumerate(scales):
        ax = axes[0][column]
        values = np.asarray([[trajectory_metrics(record, labels, scale).get("endpoint_change")] for record in records], dtype=float)
        finite = values[np.isfinite(values)]
        max_abs = max(abs(float(finite.min())), abs(float(finite.max()))) if finite.size else 1.0
        if max_abs == 0:
            max_abs = 1.0
        im = ax.imshow(values, cmap="RdBu_r", vmin=-max_abs, vmax=max_abs, aspect="auto")
        ax.set_xticks([0], [scale])
        ax.set_yticks(range(len(records)), [record.plot_label for record in records])
        ax.set_title(f"{scale} raw delta\n(independent color range)")
        for row, value in enumerate(values[:, 0]):
            if np.isfinite(value):
                ax.text(0, row, f"{value:.1f}", ha="center", va="center", color=_text_color(im.cmap, im.norm, value))
        fig.colorbar(im, ax=ax, label="Endpoint − baseline")
    fig.suptitle(_wrapped(f"{title_prefix}: final delta heatmap (scale-specific raw ranges)"), fontsize=13)
    _footer(fig, records, "none", labels)
    return save_figure(fig, out_dir / f"{index:02d}_final_delta_heatmap.png")


def render_presentation(
    records: list[ExperimentRecord],
    labels: list[str],
    scales: list[str],
    out_dir: Path,
    title_prefix: str,
    *,
    error_bar: str,
    y_axis: str,
    show_repeat_points: bool,
    outer_summary: str,
) -> list[Path]:
    paths = render_lines_only(records, labels, scales, out_dir, title_prefix, y_axis=y_axis)
    paths.extend(
        [
        plot_trajectory_with_ci(
            records,
            labels,
            scale,
            out_dir,
            title_prefix,
            error_bar=error_bar,
            y_axis=y_axis,
            show_repeat_points=show_repeat_points,
            outer_summary=outer_summary,
        )
        for scale in scales
        ]
    )
    paths.append(plot_endpoint_delta_with_ci(records, labels, scales, out_dir, title_prefix))
    paths.append(plot_best_change_and_rebound(records, labels, scales, out_dir, title_prefix))
    paths.extend(plot_measurement_reliability_heatmaps(records, labels, scales, out_dir, title_prefix))
    return paths


def render_appendix(
    records: list[ExperimentRecord], labels: list[str], scales: list[str], out_dir: Path, title_prefix: str
) -> list[Path]:
    paths = [_appendix_final_delta(records, labels, scales, out_dir, title_prefix)]
    index = 2
    for scale in scales:
        paths.append(_appendix_trajectory(records, labels, scale, out_dir, title_prefix, index))
        index += 1
    paths.append(_appendix_delta_trajectory(records, labels, scales, out_dir, title_prefix, index))
    index += 1
    paths.append(_appendix_volatility(records, labels, scales, out_dir, title_prefix, index))
    index += 1
    paths.append(_appendix_endpoint_uncertainty(records, labels, scales, out_dir, title_prefix, index))
    index += 1
    paths.append(_appendix_delta_heatmap(records, labels, scales, out_dir, title_prefix, index))
    return paths
