#!/usr/bin/env python3
"""Render 0922 KBD1-G1 as two separate experimental conditions (R01/R02)."""
from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

SCRIPT_PATH = Path(__file__).absolute()
PROJECT_ROOT = SCRIPT_PATH.parents[2] if SCRIPT_PATH.parent.name == "0922画图" else SCRIPT_PATH.parents[1]
REFERENCE_SCRIPT = PROJECT_ROOT / "results/0918-g1-24step/0918画图/render_0918_24step.py"
REPORTS_DIR = PROJECT_ROOT / "results/experiment_data/reports"
OUT_DIR = PROJECT_ROOT / "results/0922画图"
RUN_ORDER = ["R01", "R02"]
LONG_SCALES = ["PHQ-9", "BDI-II"]
SHORT_SCALES = ["总体抑郁水平及干扰程度量表", "总体焦虑水平及干扰程度量表"]
TIMEPOINTS = ["T0", "session_2", "session_4", "session_6", "session_8", "session_10", "session_12"]
SHORT_TIMEPOINTS = TIMEPOINTS[1:-1]
LONG_TIMEPOINTS = ["T0", "session_12"]
TICKS = ["T0", "S2", "S4", "S6", "S8", "S10", "S12"]
BLUE, ORANGE = "#0072B2", "#D55E00"


def load_reference():
    spec = importlib.util.spec_from_file_location("renderer_0918", REFERENCE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load reference renderer: {REFERENCE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def draw_endpoints(renderer, records):
    fig, axes = renderer.plt.subplots(2, 2, figsize=(12.8, 9.0), squeeze=False)
    for row, scale in enumerate(LONG_SCALES):
        for col, record in enumerate(records):
            ax = axes[row, col]
            color = BLUE if scale == "PHQ-9" else ORANGE
            scores = []
            for x, label in enumerate(LONG_TIMEPOINTS):
                stats = record.series[scale][label]
                center = float(stats["score"])
                scores.append(center)
                ax.errorbar(x, center, yerr=[[center - stats["ci95_lower"]], [stats["ci95_upper"] - center]],
                            color=color, marker="o", markersize=7, capsize=5, linewidth=2)
                ax.annotate(f"{center:.1f}", (x, center), xytext=(9, 7), textcoords="offset points", fontsize=9)
            ax.plot([0, 1], scores, color=color, alpha=0.55, linewidth=1.4)
            ax.set_xlim(-0.35, 1.35)
            ax.set_ylim(0, 27 if scale == "PHQ-9" else 63)
            ax.set_xticks([0, 1], ["T0", "S12"])
            ax.set_ylabel(f"{scale} 分数")
            ax.grid(axis="y", alpha=0.22)
            ax.spines[["top", "right"]].set_visible(False)
            if row == 0:
                ax.set_title(f"条件 {record.repeat_id}", fontsize=13, fontweight="bold")
    fig.suptitle("0922实验：两个条件各自的 PHQ-9 / BDI-II 前后测", fontsize=16, fontweight="bold")
    fig.text(0.5, 0.02, "每列是一次独立条件实验。点和误差线为该条件快照内 K=10 次生成的均值及 95% t CI；不是条件间置信区间。", ha="center", fontsize=9)
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.09, top=0.89, hspace=0.30, wspace=0.19)
    return renderer.save_figure(fig, OUT_DIR / "figure1_0922_condition_endpoint_scores.png")


def draw_item_changes(renderer, records, changes):
    fig, axes = renderer.plt.subplots(2, 2, figsize=(15.5, 11.2), squeeze=False)
    limit = max(1.0, float(changes["mean_change"].abs().max()))
    norm = renderer.TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
    image = None
    for row, scale in enumerate(LONG_SCALES):
        for col, record in enumerate(records):
            ax = axes[row, col]
            selected = changes.loc[(changes["scale"] == scale) & (changes["group"] == record.repeat_id)].sort_values("item_id")
            values = selected["mean_change"].to_numpy(float)
            image = ax.imshow(values[:, None], cmap="RdBu_r", norm=norm, aspect="auto")
            ax.set_xticks([0], [record.repeat_id])
            ax.set_yticks(range(len(values)), [f"I{int(item):02d}  {renderer.ITEM_LABELS_ZH[scale][int(item)]}" for item in selected["item_id"]], fontsize=8)
            ax.set_title(f"{record.repeat_id} · {scale}", fontsize=12, fontweight="bold")
            for i, value in enumerate(values):
                ax.text(0, i, f"{value:+.2f}", ha="center", va="center", color="white" if abs(value) > limit * 0.6 else "#111827", fontsize=9)
            ax.tick_params(length=0)
    fig.colorbar(image, ax=axes, fraction=0.016, pad=0.02, label="条目变化（S12 − T0）")
    fig.suptitle("0922实验：两个条件各自的 PHQ-9 / BDI-II 条目变化", fontsize=16, fontweight="bold")
    fig.text(0.5, 0.02, "每格为该条件内 K=10 次生成先取均值后的前后变化；负值（蓝）表示症状减轻，正值（红）表示加重。", ha="center", fontsize=9)
    fig.subplots_adjust(left=0.26, right=0.86, bottom=0.08, top=0.90, hspace=0.24, wspace=0.28)
    return renderer.save_figure(fig, OUT_DIR / "figure3_0922_condition_item_changes.png")


def draw_trajectory(renderer, records):
    x_map = {label: i for i, label in enumerate(TIMEPOINTS)}
    fig, axes = renderer.plt.subplots(1, 2, figsize=(17.5, 6.9), sharex=True, sharey=True)
    for ax_short, record in zip(axes, records):
        ax_long = ax_short.twinx()
        for scale, linestyle, marker in zip(SHORT_SCALES, ("-", "--"), ("o", "^")):
            xs = [x_map[label] for label in SHORT_TIMEPOINTS]
            values = [record.series[scale][label]["score"] for label in SHORT_TIMEPOINTS]
            ax_short.plot(xs, values, color=BLUE, linestyle=linestyle, marker=marker, linewidth=2, markersize=5)
        for scale, marker in zip(LONG_SCALES, ("o", "s")):
            for label in LONG_TIMEPOINTS:
                stats = record.series[scale][label]
                center = float(stats["score"])
                ax_long.errorbar(x_map[label], center,
                    yerr=[[center - stats["ci95_lower"]], [stats["ci95_upper"] - center]],
                    color=ORANGE, linestyle="None", marker=marker, markersize=6, capsize=3, elinewidth=1.5)
        ax_short.set_xlim(-0.35, len(TIMEPOINTS) - 0.65)
        ax_short.set_ylim(0, 15.5)
        ax_long.set_ylim(0, 63)
        ax_short.set_xticks(range(len(TIMEPOINTS)), TICKS)
        ax_short.set_xlabel("评估时间点")
        ax_short.set_ylabel("中间总体量表分数", color=BLUE)
        ax_long.set_ylabel("PHQ-9 / BDI-II 分数", color=ORANGE)
        ax_short.tick_params(axis="y", colors=BLUE)
        ax_long.tick_params(axis="y", colors=ORANGE)
        ax_short.grid(axis="y", color="#CBD5E1", alpha=0.7)
        ax_short.set_axisbelow(True)
        ax_short.spines[["top"]].set_visible(False)
        ax_long.spines[["top"]].set_visible(False)
        ax_short.set_title(f"条件 {record.repeat_id}", fontsize=13, fontweight="bold")
    fig.suptitle("0922实验：两个条件各自的量表轨迹", fontsize=16.5, fontweight="bold", y=0.98)
    fig.legend(handles=[
        renderer.Line2D([], [], color=BLUE, linestyle="-", marker="o", label="总体抑郁（左轴）"),
        renderer.Line2D([], [], color=BLUE, linestyle="--", marker="^", label="总体焦虑（左轴）"),
        renderer.Line2D([], [], color=ORANGE, linestyle="None", marker="o", label="PHQ-9（右轴）"),
        renderer.Line2D([], [], color=ORANGE, linestyle="None", marker="s", label="BDI-II（右轴）"),
    ], loc="upper center", bbox_to_anchor=(0.5, 0.91), ncol=4, frameon=False, fontsize=9)
    fig.text(0.5, 0.02, "R01/R02 分列，不计算条件间均值。首尾长量表误差线为各条件快照内 K=10 次生成的 95% t CI；中间线连接各快照 K=5 次生成的均值。", ha="center", fontsize=8.8)
    fig.subplots_adjust(left=0.07, right=0.93, bottom=0.13, top=0.80, wspace=0.26)
    return renderer.save_figure(fig, OUT_DIR / "figure4_0922_condition_dual_axis_trajectory.png")


def main():
    reference = load_reference()
    source = reference.load_reference()
    renderer = source.load_renderer()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    reference.OUT_DIR = source.OUT_DIR = renderer.OUT_DIR = OUT_DIR
    reference.RUN_ORDER = source.ENTITY_ORDER = RUN_ORDER
    reference.LONG_SCALES = source.LONG_SCALES = LONG_SCALES
    reference.INTERMEDIATE_SCALES = source.INTERMEDIATE_SCALES = SHORT_SCALES
    source.ALL_TIMEPOINTS = TIMEPOINTS
    renderer.GROUPS = RUN_ORDER
    renderer.GROUP_COLORS = dict(zip(RUN_ORDER, [BLUE, ORANGE]))
    renderer.EXPERIMENT_LABEL = "0922实验"
    renderer.COMPARISON_LABEL = "两个独立条件"
    renderer.OUTER_RUNS_PER_ENTITY = 1
    renderer.TIMEPOINTS = LONG_TIMEPOINTS
    renderer.TIME_TICKS = ["T0", "S12"]
    renderer.ENDPOINT_LABEL = "session_12"
    renderer.ENDPOINT_DISPLAY = "S12"
    renderer.TIME_COLORS = {"T0": ORANGE, "session_12": "#56B4E9"}
    renderer.configure_simplified_chinese_font()
    renderer.plt.rcParams["font.family"] = ["DejaVu Sans", "Droid Sans Fallback"]

    paths = sorted(REPORTS_DIR.glob("repeat-Counsel-KBD1-G1-MOD--cbt-progressive-d--pilot-session12-0922-*_summary.json"))
    if len(paths) != 2:
        raise ValueError(f"Expected two 0922 reports, found {len(paths)}")
    records, labels, scales = renderer.load_records(paths)
    if labels != TIMEPOINTS or set(scales) != set([*LONG_SCALES, *SHORT_SCALES]):
        raise ValueError(f"Unexpected design: labels={labels}, scales={scales}")
    if [record.repeat_id for record in records] != RUN_ORDER:
        raise ValueError("Expected conditions R01/R02")
    for record in records:
        for scale, points, repeats in [
            *((scale, LONG_TIMEPOINTS, 10) for scale in LONG_SCALES),
            *((scale, SHORT_TIMEPOINTS, 5) for scale in SHORT_SCALES),
        ]:
            if any((record.series.get(scale, {}).get(label) or {}).get("n") != repeats for label in points):
                raise ValueError(f"Incomplete measurement coverage: {record.repeat_id}, {scale}")
    by_condition = [replace(record, kbd="KBD1", group=record.repeat_id) for record in records]
    measurement_path = renderer.write_measurement_repeat_long_csv(OUT_DIR, by_condition, LONG_TIMEPOINTS, LONG_SCALES)
    item_path = renderer.write_item_measurement_repeat_long_csv(OUT_DIR, by_condition, LONG_TIMEPOINTS, LONG_SCALES)
    scores = renderer.score_long(by_condition, LONG_TIMEPOINTS)
    scores.to_csv(OUT_DIR / "condition_score_long.csv", index=False, encoding="utf-8-sig")
    changes = renderer.item_change_summary(by_condition, LONG_TIMEPOINTS)
    changes.to_csv(OUT_DIR / "item_change_by_condition.csv", index=False, encoding="utf-8-sig")
    trajectory_path = OUT_DIR / "trajectory_measurement_summary.csv"
    source.trajectory_measurement_summary(by_condition).to_csv(trajectory_path, index=False, encoding="utf-8-sig")

    outputs = [
        draw_endpoints(renderer, by_condition),
        reference.draw_repeat_stability(renderer, by_condition, scales=LONG_SCALES, timepoints=LONG_TIMEPOINTS, time_ticks=["T0", "S12"], ranges={"PHQ-9": (0, 27), "BDI-II": (0, 63)}, filename="figure2a_0922_condition_long_scale_stability.png", title="0922实验：两个条件的长量表快照内重复生成", repeats_note="T0/S12 的长量表每快照各生成 K=10 次。"),
        reference.draw_repeat_stability(renderer, by_condition, scales=SHORT_SCALES, timepoints=SHORT_TIMEPOINTS, time_ticks=TICKS[1:-1], ranges={scale: (0, 15) for scale in SHORT_SCALES}, filename="figure2b_0922_condition_short_scale_stability.png", title="0922实验：两个条件的中间短量表快照内重复生成", repeats_note="S2–S10 的中间短量表每快照各生成 K=5 次。"),
        draw_item_changes(renderer, by_condition, changes),
        draw_trajectory(renderer, by_condition),
    ]
    manifest = {
        "source_reports_dir": str(REPORTS_DIR.relative_to(PROJECT_ROOT)),
        "report_files": [str(path.relative_to(PROJECT_ROOT)) for path in paths],
        "conditions": RUN_ORDER,
        "independent_experiments_per_condition": 1,
        "long_scale_timepoints": LONG_TIMEPOINTS,
        "intermediate_scale_timepoints": SHORT_TIMEPOINTS,
        "measurement_repeats": {"PHQ-9/BDI-II": 10, "intermediate short scales": 5},
        "statistical_scope": "all scores, item changes and snapshot CIs computed within each condition; no between-condition pooling or CI",
        "outputs": [path.name for path in outputs],
        "data_exports": [measurement_path.name, item_path.name, "condition_score_long.csv", "item_change_by_condition.csv", trajectory_path.name],
        "not_generated": ["condition-level ICC: each condition has only one independent experiment", "n=1 endpoint boxplots: replaced by endpoint means and within-snapshot CIs", "by-repeat expansion: duplicates stability panels"],
    }
    (OUT_DIR / "plot_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT_DIR / "README.md").write_text(
        "# 0922 KBD1-G1：R01/R02 分条件绘图\n\n"
        "R01、R02 作为两个不同条件实验，所有图均分列呈现，不计算跨条件均值或置信区间。每个条件只有一次独立实验。"
        "图1为前后测分数；图2a/2b为快照内量表重复生成；图3为各条件条目变化；图4为共享时间横轴、左右双纵轴的量表轨迹。"
        "误差线仅表示同一冻结快照内 K 次生成的 95% t CI，不代表实验重复性。\n\n"
        "未生成条件级 ICC（每条件只有一次独立实验）和 n=1 箱线图。\n\n"
        "重绘：`python render_0922_g1.py`（使用 llm-depression conda 环境）。\n",
        encoding="utf-8",
    )
    print("Generated:")
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
