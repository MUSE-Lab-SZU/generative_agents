"""Render the requested PHQ-9 scale-context ablation trajectory comparison.

The input summary pools frozen evaluation nodes.  It is intentionally a
descriptive diagnostic plot: the spread band is one node-score standard
deviation, not an outer-run confidence interval.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .charts.shared.plotting import (
    configure_simplified_chinese_font,
    plt,
    save_figure,
)


DEFAULT_INPUT = Path(
    "results/scale_context_ablation/"
    "0821-all-groups-all-nodes-domain-ablation-r10/summary.json"
)
DEFAULT_OUTPUT_DIR = Path(
    "docs/experiment_evaluation/0821_scale_context_ablation"
)
TIMEPOINTS = ["T0", "S4", "S8", "S12"]
GROUPS = ["G1", "G2", "G4", "G5", "G6", "G9", "G11"]
CONDITIONS = [
    ("full", "消融前：完整量表上下文", "#7C3AED"),
    ("full_no_state", "消融后：移除状态上下文", "#DB2777"),
]
SCALE_LIMITS = (0, 27)
OUTPUT_NAME = "figure_01_0821_phq9_scale_context_ablation_trajectory.png"


def _as_float(value: Any) -> float:
    if value is None:
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def load_plot_summary(path: Path) -> dict[tuple[str, str, str], dict[str, Any]]:
    """Read usable rows only, retaining missing cells for plotting gaps."""
    with path.open(encoding="utf-8") as stream:
        payload = json.load(stream)
    if payload.get("artifact_kind") != "frozen_scale_context_ablation_phq9_summary":
        raise ValueError(f"{path} 不是 PHQ-9 量表上下文消融汇总")
    rows = payload.get("plot_summary")
    if not isinstance(rows, list):
        raise ValueError(f"{path} 缺少 plot_summary")

    selected: dict[tuple[str, str, str], dict[str, Any]] = {}
    valid_conditions = {condition for condition, _, _ in CONDITIONS}
    for row in rows:
        if not isinstance(row, dict):
            continue
        group = str(row.get("group") or "")
        timepoint = str(row.get("timepoint") or "")
        condition = str(row.get("condition") or "")
        if (
            group not in GROUPS
            or timepoint not in TIMEPOINTS
            or condition not in valid_conditions
        ):
            continue
        key = (group, condition, timepoint)
        if key in selected:
            raise ValueError(f"{path} 的 plot_summary 存在重复单元: {key}")
        selected[key] = row
    if not selected:
        raise ValueError(f"{path} 没有可绘制的 PHQ-9 消融汇总行")
    return selected


def _cell_values(
    rows: dict[tuple[str, str, str], dict[str, Any]], group: str, condition: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    means: list[float] = []
    stds: list[float] = []
    counts: list[float] = []
    for timepoint in TIMEPOINTS:
        row = rows.get((group, condition, timepoint), {})
        mean = _as_float(row.get("mean_total_score"))
        count = _as_float(row.get("n_plottable"))
        std = _as_float(row.get("std_total_score"))
        if not np.isfinite(mean) or not np.isfinite(count) or count <= 0:
            mean = float("nan")
            std = float("nan")
            count = 0.0
        means.append(mean)
        stds.append(std if np.isfinite(std) else 0.0)
        counts.append(count)
    return np.asarray(means), np.asarray(stds), np.asarray(counts)


def render(rows: dict[tuple[str, str, str], dict[str, Any]], output_path: Path) -> Path:
    """Render one group-by-condition PHQ-9 trajectory figure from summary rows."""
    x_values = np.arange(len(TIMEPOINTS))
    height = 2.15 * len(GROUPS) + 1.7
    fig, axes = plt.subplots(
        len(GROUPS),
        len(CONDITIONS),
        figsize=(15.2, height),
        squeeze=False,
        sharex=True,
        sharey=True,
    )
    for row_index, group in enumerate(GROUPS):
        for column_index, (condition, title, color) in enumerate(CONDITIONS):
            ax = axes[row_index, column_index]
            means, stds, counts = _cell_values(rows, group, condition)
            lower = means - stds
            upper = means + stds
            ax.fill_between(x_values, lower, upper, color=color, alpha=0.16, linewidth=0)
            ax.plot(
                x_values,
                means,
                color=color,
                marker="o",
                markersize=4.2,
                linewidth=2.3,
                zorder=3,
            )
            for index, value in enumerate(means):
                if not np.isfinite(value):
                    ax.scatter(
                        index,
                        1.1,
                        marker="x",
                        color="#DC2626",
                        s=24,
                        linewidths=1.1,
                        zorder=4,
                    )
            count_text = "/".join(str(int(value)) for value in counts)
            ax.text(
                0.985,
                0.91,
                f"有效节点 n：{count_text}",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=8.4,
                color="#475569",
            )
            ax.set_ylim(*SCALE_LIMITS)
            ax.grid(axis="y", color="#CBD5E1", linewidth=0.65, alpha=0.6)
            ax.set_axisbelow(True)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            if row_index == 0:
                ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
            if column_index == 0:
                ax.set_ylabel(f"{group}\nPHQ-9 分数")
            else:
                ax.set_ylabel("PHQ-9 分数")
            ax.set_xticks(x_values, TIMEPOINTS)

    fig.suptitle(
        "0821全条件：PHQ-9 量表上下文消融前后轨迹",
        fontsize=18,
        fontweight="bold",
        y=0.985,
    )
    fig.text(
        0.5,
        0.018,
        "数据来自 summary.json 的 plot_summary；实线为可用冻结评估节点的均值，阴影为 ±1 SD（不是 outer-run 置信区间）。"
        "红色 × 表示该节点无可用评分；节点不完整时仍保留其余轨迹用于描述性对比。",
        ha="center",
        fontsize=9.3,
        color="#475569",
    )
    fig.subplots_adjust(
        left=0.08, right=0.985, bottom=0.075, top=0.925, wspace=0.12, hspace=0.20
    )
    return save_figure(fig, output_path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)
    configure_simplified_chinese_font()
    rows = load_plot_summary(args.summary)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    output = render(rows, args.out_dir / OUTPUT_NAME)
    manifest = {
        "schema_version": "scale_context_ablation_plot_v1",
        "source_summary": str(args.summary),
        "source_table": "plot_summary",
        "scale": "PHQ-9",
        "conditions": {
            "before_ablation": "full",
            "after_ablation": "full_no_state",
        },
        "groups": GROUPS,
        "timepoints": TIMEPOINTS,
        "missing_data_policy": "retain gaps and plot all available summary cells",
        "spread_band": "mean plus or minus one node-score standard deviation",
        "png_file": output.name,
    }
    (args.out_dir / "plot_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
