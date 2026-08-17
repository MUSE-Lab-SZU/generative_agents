"""Render explicitly selected third-batch KBD2 symptom composite figures.

Panel A keeps the independent-outer-run trajectory analysis used by the
original composite.  Panels B/C collapse R01--R05 within each experiment
condition, so every heatmap column is a condition rather than a single run.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from scipy import stats

from .charts.shared.plotting import (
    configure_simplified_chinese_font,
    heatmap_text_color,
    plt,
    save_figure,
)
from .life_state import ITEM_LABELS_ZH
from .schema import SCALE_RANGES, group_color


RESULT_ROOT = Path("results/0802-0808联合实验结果图")
DEFAULT_INPUT = (
    RESULT_ROOT
    / "01_KBD2跨实验组_五次重复/long_format_paper_figures/symptom_long_data.csv"
)
DEFAULT_OUTPUT_DIR = RESULT_ROOT / "09_按需重绘图/第三批"

SCALES = ["PHQ-9", "BDI-II"]
TIMEPOINTS = [
    "T0",
    "session_4",
    "session_8",
    "session_12",
    "session_16",
    "session_20",
]
TIMEPOINT_LABELS = ["T0", "S4", "S8", "S12", "S16", "S20"]
COMPARISONS: dict[str, list[str]] = {
    "g1-vs-g4": ["G1", "G4"],
    "g1-vs-g6": ["G1", "G6"],
    "g1-vs-g7": ["G1", "G7"],
    "g3-vs-g5-vs-g9": ["G3", "G5", "G9"],
}
FIGURE_FILES = {
    key: f"symptom_trajectory_item_change_composite_{key.replace('-', '_')}.png"
    for key in COMPARISONS
}


def _mean_ci(values: pd.Series) -> tuple[float, float, float, int]:
    clean = pd.to_numeric(values, errors="coerce").dropna().to_numpy(float)
    n = len(clean)
    if not n:
        return np.nan, np.nan, np.nan, 0
    center = float(np.mean(clean))
    if n < 2:
        return center, center, center, n
    half_width = float(stats.t.ppf(0.975, n - 1) * stats.sem(clean))
    return center, center - half_width, center + half_width, n


def load_kbd2_root_data(path: Path, groups: Sequence[str]) -> pd.DataFrame:
    """Load complete root-experiment symptom rows for selected KBD2 groups."""
    frame = pd.read_csv(path)
    required = {
        "run",
        "persona",
        "group",
        "timepoint",
        "scale",
        "item",
        "item_score",
        "source_kind",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path} 缺少必要列: {', '.join(missing)}")

    selected = frame[
        (frame["persona"] == "KBD2")
        & frame["group"].isin(groups)
        & (frame["source_kind"] == "root_repeat_summary")
        & frame["scale"].isin(SCALES)
        & frame["timepoint"].isin(TIMEPOINTS)
    ].copy()
    selected["item"] = pd.to_numeric(selected["item"], errors="raise").astype(int)
    selected["item_score"] = pd.to_numeric(
        selected["item_score"], errors="raise"
    ).astype(float)

    duplicates = selected.duplicated(["run", "scale", "timepoint", "item"])
    if duplicates.any():
        raise ValueError("KBD2 root data contains duplicate run/scale/timepoint/item rows")

    counts = selected.groupby(["group", "scale"])["run"].nunique()
    expected_index = pd.MultiIndex.from_product(
        [list(groups), SCALES], names=["group", "scale"]
    )
    counts = counts.reindex(expected_index)
    if counts.isna().any() or not (counts == 5).all():
        raise ValueError(
            "Each selected KBD2 group/scale must contain exactly five outer runs; "
            f"observed={counts.to_dict()}"
        )

    expected_items = {"PHQ-9": 9, "BDI-II": 21}
    for (run, scale), rows in selected.groupby(["run", "scale"], sort=False):
        expected = len(TIMEPOINTS) * expected_items[str(scale)]
        if len(rows) != expected:
            raise ValueError(
                f"Incomplete KBD2 root run {run}/{scale}: {len(rows)} != {expected}"
            )
    return selected


def build_composite_summaries(
    frame: pd.DataFrame, groups: Sequence[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build trajectory CIs and group-level endpoint-minus-baseline item means."""
    run_totals = (
        frame.groupby(["run", "group", "scale", "timepoint"], observed=True)[
            "item_score"
        ]
        .sum()
        .rename("total_score")
        .reset_index()
    )
    trajectory_rows: list[dict[str, object]] = []
    for (group, scale, timepoint), rows in run_totals.groupby(
        ["group", "scale", "timepoint"], sort=False, observed=True
    ):
        center, lower, upper, n = _mean_ci(rows["total_score"])
        trajectory_rows.append(
            {
                "group": str(group),
                "scale": str(scale),
                "timepoint": str(timepoint),
                "time_order": TIMEPOINTS.index(str(timepoint)),
                "mean": center,
                "ci95_lower": lower,
                "ci95_upper": upper,
                "n_outer_runs": n,
            }
        )
    trajectory = pd.DataFrame(trajectory_rows)

    endpoints = frame[frame["timepoint"].isin(["T0", "session_20"])].pivot(
        index=["run", "group", "scale", "item"],
        columns="timepoint",
        values="item_score",
    )
    if endpoints[["T0", "session_20"]].isna().any().any():
        raise ValueError("Some KBD2 item rows cannot be paired between T0 and S20")
    run_changes = endpoints.reset_index()
    run_changes["score_change"] = run_changes["session_20"] - run_changes["T0"]

    change_rows: list[dict[str, object]] = []
    for (group, scale, item), rows in run_changes.groupby(
        ["group", "scale", "item"], sort=False, observed=True
    ):
        center, lower, upper, n = _mean_ci(rows["score_change"])
        change_rows.append(
            {
                "group": str(group),
                "scale": str(scale),
                "item": int(item),
                "mean_change": center,
                "ci95_lower": lower,
                "ci95_upper": upper,
                "n_outer_runs": n,
            }
        )
    item_changes = pd.DataFrame(change_rows)

    for table_name, summary in (
        ("trajectory", trajectory),
        ("item change", item_changes),
    ):
        if summary.empty or not (summary["n_outer_runs"] == 5).all():
            raise ValueError(f"{table_name} summary is missing five-run estimates")

    group_rank = {group: index for index, group in enumerate(groups)}
    scale_rank = {scale: index for index, scale in enumerate(SCALES)}
    trajectory = trajectory.sort_values(
        ["scale", "group", "time_order"],
        key=lambda values: values.map(
            scale_rank if values.name == "scale" else group_rank
        )
        if values.name in {"scale", "group"}
        else values,
    ).reset_index(drop=True)
    item_changes = item_changes.assign(
        _scale_order=item_changes["scale"].map(scale_rank),
        _group_order=item_changes["group"].map(group_rank),
    ).sort_values(["_scale_order", "_group_order", "item"])
    item_changes = item_changes.drop(columns=["_scale_order", "_group_order"]).reset_index(
        drop=True
    )
    return trajectory, item_changes


def _plot_group_heatmap(
    ax: Any,
    rows: pd.DataFrame,
    *,
    scale: str,
    groups: Sequence[str],
    norm: TwoSlopeNorm,
    cmap: Any,
) -> Any:
    items = list(range(1, 10 if scale == "PHQ-9" else 22))
    matrix = (
        rows.pivot(index="item", columns="group", values="mean_change")
        .reindex(index=items, columns=list(groups))
        .to_numpy(float)
    )
    image = ax.imshow(matrix, aspect="auto", cmap=cmap, norm=norm)
    ax.set_xticks(np.arange(len(groups)), list(groups), fontsize=10, fontweight="bold")
    labels = [f"I{item:02d}  {ITEM_LABELS_ZH[scale][item]}" for item in items]
    ax.set_yticks(np.arange(len(items)), labels, fontsize=7.2)
    ax.set_title(
        f"{scale} 条目变化（每组 5 次 Rxx 均值）",
        fontsize=11,
        fontweight="bold",
        pad=10,
    )
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            value = matrix[row_index, column_index]
            if np.isfinite(value):
                ax.text(
                    column_index,
                    row_index,
                    f"{value:+.2f}",
                    ha="center",
                    va="center",
                    fontsize=7.1,
                    fontweight="medium",
                    color=heatmap_text_color(cmap, norm, float(value)),
                )
    ax.set_xlabel("实验条件（Rxx 已整合）", fontsize=9)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_color("#CBD5E1")
    return image


def plot_composite(
    trajectory: pd.DataFrame,
    item_changes: pd.DataFrame,
    groups: Sequence[str],
    output_path: Path,
) -> Path:
    """Plot one requested comparison with group-aggregated B/C heatmaps."""
    fig = plt.figure(figsize=(16.8, 11.2), constrained_layout=True)
    grid = fig.add_gridspec(2, 3, width_ratios=[1.18, 0.92, 1.10])
    trajectory_axes = [fig.add_subplot(grid[index, 0]) for index in range(2)]
    heatmap_axes = [fig.add_subplot(grid[:, 1]), fig.add_subplot(grid[:, 2])]

    for ax, scale in zip(trajectory_axes, SCALES):
        scale_rows = trajectory[trajectory["scale"] == scale]
        for group in groups:
            selected = scale_rows[scale_rows["group"] == group].sort_values(
                "time_order"
            )
            x = selected["time_order"].to_numpy(float)
            center = selected["mean"].to_numpy(float)
            lower = selected["ci95_lower"].to_numpy(float)
            upper = selected["ci95_upper"].to_numpy(float)
            ax.errorbar(
                x,
                center,
                yerr=np.vstack([center - lower, upper - center]),
                color=group_color(group),
                marker="o",
                markersize=4.8,
                linewidth=1.9,
                capsize=3,
                label=group,
            )
        ax.set_xticks(np.arange(len(TIMEPOINTS)), TIMEPOINT_LABELS, fontsize=9)
        ax.set_ylabel(f"{scale} 总分", fontsize=10)
        ax.set_title(f"{scale} 总分轨迹", fontsize=11, fontweight="bold")
        ax.set_ylim(*SCALE_RANGES[scale])
        ax.grid(axis="y", color="#CBD5E1", linewidth=0.7, alpha=0.65)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    trajectory_axes[-1].set_xlabel("评估时间点", fontsize=10)
    handles = [
        Line2D(
            [0],
            [0],
            color=group_color(group),
            marker="o",
            linewidth=1.9,
            label=group,
        )
        for group in groups
    ]
    trajectory_axes[0].legend(
        handles=handles, title="实验条件", frameon=False, loc="best", ncol=len(groups)
    )
    trajectory_axes[0].text(
        -0.15,
        1.08,
        "A",
        transform=trajectory_axes[0].transAxes,
        fontsize=15,
        fontweight="bold",
    )

    values = pd.to_numeric(item_changes["mean_change"], errors="coerce").to_numpy()
    limit = max(0.5, float(np.nanmax(np.abs(values))))
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
    cmap = plt.get_cmap("RdBu_r")
    images = []
    for panel, ax, scale in zip(("B", "C"), heatmap_axes, SCALES):
        image = _plot_group_heatmap(
            ax,
            item_changes[item_changes["scale"] == scale],
            scale=scale,
            groups=groups,
            norm=norm,
            cmap=cmap,
        )
        images.append(image)
        ax.text(
            -0.16,
            1.035,
            panel,
            transform=ax.transAxes,
            fontsize=15,
            fontweight="bold",
        )
    colorbar = fig.colorbar(images[-1], ax=heatmap_axes, fraction=0.025, pad=0.025)
    colorbar.set_label("条目分数变化（S20 − T0）", fontsize=9)

    comparison = " vs ".join(groups)
    fig.suptitle(
        f"KBD2：{comparison} 的 PHQ-9 / BDI-II 总分轨迹与条目变化",
        fontsize=15,
        fontweight="bold",
    )
    fig.text(
        0.5,
        -0.012,
        "A：各条件 5 次独立外层重复的总分均值与 95% t 置信区间。"
        "B/C：每列为一个实验条件，单元格为该条件 5 次 Rxx 的平均条目变化；"
        "负值（蓝）表示症状改善，正值（红）表示症状加重。",
        ha="center",
        fontsize=8.5,
        color="#475569",
    )
    return save_figure(fig, output_path)


def _write_readme(output_dir: Path, selected: Sequence[str]) -> None:
    command_lines = [
        "python -m experiment_eval.targeted_redraw_third_batch \\",
        *[
            f"  --figure {key}" + (" \\" if index < len(selected) - 1 else "")
            for index, key in enumerate(selected)
        ],
    ]
    outputs = "\n".join(
        f"- `{FIGURE_FILES[key]}`：KBD2 {' vs '.join(COMPARISONS[key])}。"
        for key in selected
    )
    content = f"""# 第三批按需重绘图

本目录只生成显式选择的 {len(selected)} 张 KBD2 条件对比图。子图 A 以独立外层重复为统计单位；子图 B/C 将每个条件的 5 次 Rxx 汇总为一列（S20−T0 的均值），不把 Rxx 分列展示。

## 运行命令

```bash
{chr(10).join(command_lines)}
```

## 图文件

{outputs}

## 配套文件

- `symptom_trajectory_statistics.csv`：子图 A 的均值与 95% t 置信区间。
- `symptom_item_change_summary_by_group.csv`：子图 B/C 的组均值、95% t 置信区间与独立重复数。
- `manifest.json`：输入、选择项、聚合口径与输出清单。

未生成逐 Rxx 热图、其他条件组合、随访图、PDF 或 SVG。
"""
    (output_dir / "README.md").write_text(content, encoding="utf-8")


def render_selected(
    selected: Sequence[str],
    *,
    input_path: Path = DEFAULT_INPUT,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> list[Path]:
    """Render only explicitly selected comparison keys."""
    if not selected:
        raise ValueError("At least one --figure selection is required")
    unknown = sorted(set(selected) - set(COMPARISONS))
    if unknown:
        raise ValueError(f"Unknown third-batch figure selection: {', '.join(unknown)}")
    ordered = list(dict.fromkeys(selected))
    configure_simplified_chinese_font()
    output_dir.mkdir(parents=True, exist_ok=True)

    paths: list[Path] = []
    trajectory_tables: list[pd.DataFrame] = []
    change_tables: list[pd.DataFrame] = []
    for key in ordered:
        groups = COMPARISONS[key]
        frame = load_kbd2_root_data(input_path, groups)
        trajectory, item_changes = build_composite_summaries(frame, groups)
        trajectory.insert(0, "comparison", key)
        item_changes.insert(0, "comparison", key)
        trajectory_tables.append(trajectory)
        change_tables.append(item_changes)
        paths.append(
            plot_composite(
                trajectory,
                item_changes,
                groups,
                output_dir / FIGURE_FILES[key],
            )
        )

    pd.concat(trajectory_tables, ignore_index=True).to_csv(
        output_dir / "symptom_trajectory_statistics.csv", index=False
    )
    pd.concat(change_tables, ignore_index=True).to_csv(
        output_dir / "symptom_item_change_summary_by_group.csv", index=False
    )
    _write_readme(output_dir, ordered)
    manifest = {
        "schema_version": "1.0",
        "input": str(input_path),
        "persona": "KBD2",
        "selected_figures": ordered,
        "comparisons": {key: COMPARISONS[key] for key in ordered},
        "analysis_unit": "independent outer run",
        "outer_runs_per_condition": 5,
        "trajectory_summary": "mean and 95% t CI across outer runs",
        "item_change": "S20 minus T0",
        "heatmap_aggregation": "mean item change across the five Rxx runs within each condition",
        "png_files": [path.name for path in paths],
        "supporting_files": [
            "symptom_trajectory_statistics.csv",
            "symptom_item_change_summary_by_group.csv",
            "README.md",
        ],
        "intentionally_not_generated": [
            "per-Rxx heatmaps",
            "unrequested condition comparisons",
            "follow-up panels",
            "PDF or SVG variants",
        ],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return paths


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render explicitly selected third-batch KBD2 symptom composites."
    )
    parser.add_argument(
        "--figure",
        action="append",
        choices=list(COMPARISONS),
        required=True,
        help="Figure key; repeat this option to render multiple requested comparisons.",
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    paths = render_selected(
        args.figure,
        input_path=args.input,
        output_dir=args.output_dir,
    )
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
