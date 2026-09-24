#!/usr/bin/env python3
"""Render 0915 persona × condition figures from validated repeat summaries."""

from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np


SCRIPT_PATH = Path(__file__).absolute()
PROJECT_ROOT = SCRIPT_PATH.parents[3] if SCRIPT_PATH.parent.name.endswith("画图") else SCRIPT_PATH.parents[1]
REFERENCE_SCRIPT = PROJECT_ROOT / "results/0908-g1-2-5/0908画图/render_0908_all_groups.py"
ROOT = PROJECT_ROOT / "results/0915-g1-2-5"
REPORTS_DIR = ROOT / "experiment_data/reports"
OUT_DIR = ROOT / "0915画图"
ENTITY_ORDER = ["LRN-G1", "LRN-G2", "LRN-G5", "SQL-G1", "TW-G1", "ZYH-G1"]
LONG_SCALES = ["PHQ-9", "BDI-II"]
INTERMEDIATE_SCALES = ["总体抑郁水平及干扰程度量表", "总体焦虑水平及干扰程度量表"]
ALL_TIMEPOINTS = ["T0", "session_2", "session_4", "session_6", "session_8", "session_10", "session_12"]
LONG_TIMEPOINTS = ["T0", "session_12"]
TIME_TICKS = ["T0", "S2", "S4", "S6", "S8", "S10", "S12"]
MIDDLE_COLOR = "#0072B2"
ENDPOINT_COLOR = "#D55E00"


def load_reference():
    spec = importlib.util.spec_from_file_location("renderer_0908", REFERENCE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load reference renderer: {REFERENCE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def draw_dual_axis_trajectory(renderer, records: list[object]) -> Path:
    x_by_label = {label: index for index, label in enumerate(ALL_TIMEPOINTS)}
    fig, axes = renderer.plt.subplots(
        len(ENTITY_ORDER), 1, figsize=(15.4, 2.72 * len(ENTITY_ORDER) + 1.5), sharex=True
    )
    axes = np.atleast_1d(axes)
    for row, entity in enumerate(ENTITY_ORDER):
        ax_left = axes[row]
        ax_right = ax_left.twinx()
        record = next(record for record in records if record.group == entity)
        for scale, style, marker in zip(INTERMEDIATE_SCALES, ("-", "--"), ("o", "^")):
            points = [
                (x_by_label[label], stats)
                for label, stats in record.series.get(scale, {}).items()
                if label in x_by_label and stats.get("score") is not None
            ]
            points.sort(key=lambda item: item[0])
            ax_left.plot(
                [point[0] for point in points],
                [float(point[1]["score"]) for point in points],
                color=MIDDLE_COLOR,
                linestyle=style,
                marker=marker,
                markersize=4.7,
                linewidth=2.1,
                zorder=3,
            )
        for scale, marker in zip(LONG_SCALES, ("o", "s")):
            points = [
                (x_by_label[label], stats)
                for label, stats in record.series.get(scale, {}).items()
                if label in x_by_label and stats.get("score") is not None
            ]
            points.sort(key=lambda item: item[0])
            centers = np.asarray([float(point[1]["score"]) for point in points])
            lows = np.asarray([float(point[1]["ci95_lower"]) for point in points])
            highs = np.asarray([float(point[1]["ci95_upper"]) for point in points])
            ax_right.errorbar(
                [point[0] for point in points],
                centers,
                yerr=np.vstack((centers - lows, highs - centers)),
                color=ENDPOINT_COLOR,
                marker=marker,
                linestyle="None",
                markersize=6.1,
                capsize=3.4,
                elinewidth=1.4,
                zorder=5,
            )
        ax_left.set_ylim(0, 15)
        ax_right.set_ylim(0, 63)
        ax_left.set_xlim(-0.35, len(ALL_TIMEPOINTS) - 0.65)
        ax_left.grid(axis="y", color="#CBD5E1", linewidth=0.65, alpha=0.75)
        ax_left.set_axisbelow(True)
        ax_left.spines[["top"]].set_visible(False)
        ax_right.spines[["top"]].set_visible(False)
        ax_left.tick_params(axis="y", colors=MIDDLE_COLOR)
        ax_right.tick_params(axis="y", colors=ENDPOINT_COLOR)
        ax_left.set_ylabel(f"{entity}\n中间总体量表分数", color=MIDDLE_COLOR)
        ax_right.set_ylabel("PHQ-9 / BDI-II 分数", color=ENDPOINT_COLOR)
        ax_left.text(0.985, 0.92, "outer run n=1", transform=ax_left.transAxes, ha="right", va="top", fontsize=8.3, color="#475569")
    axes[-1].set_xticks(range(len(ALL_TIMEPOINTS)), TIME_TICKS)
    axes[-1].set_xlabel("评估时间点")
    fig.suptitle("0915实验：中间总体量表轨迹与首尾 PHQ-9 / BDI-II 评分", fontsize=17, fontweight="bold", y=0.992)
    fig.legend(
        handles=[
            renderer.Line2D([], [], color=MIDDLE_COLOR, linestyle="-", marker="o", label="总体抑郁水平及干扰程度量表（左轴）"),
            renderer.Line2D([], [], color=MIDDLE_COLOR, linestyle="--", marker="^", label="总体焦虑水平及干扰程度量表（左轴）"),
            renderer.Line2D([], [], color=ENDPOINT_COLOR, linestyle="None", marker="o", label="PHQ-9（右轴，点+95% CI）"),
            renderer.Line2D([], [], color=ENDPOINT_COLOR, linestyle="None", marker="s", label="BDI-II（右轴，点+95% CI）"),
        ],
        loc="upper center", bbox_to_anchor=(0.5, 0.965), ncol=2, frameon=False, fontsize=9,
    )
    fig.text(
        0.5, 0.012,
        "蓝色左轴：中间短量表的观测轨迹（实线=抑郁，虚线=焦虑）。橙色右轴：T0/S12 的 PHQ-9 与 BDI-II 均值及同一冻结快照内量表生成的 95% t 置信区间（PHQ-9/BDI-II 各 K=10；中间量表 K=5）。",
        ha="center", fontsize=8.6, color="#475569",
    )
    fig.subplots_adjust(left=0.10, right=0.90, bottom=0.052, top=0.91, hspace=0.23)
    return renderer.save_figure(fig, OUT_DIR / "figure4_0915_persona_group_dual_axis_scale_trajectory.png")


def main() -> None:
    source = load_reference()
    # Reuse the 0908 data exports and snapshot-stability figures, but update
    # their module-level configuration to this six-entity 0915 design.
    source.ROOT = ROOT
    source.REPORTS_DIR = REPORTS_DIR
    source.OUT_DIR = OUT_DIR
    source.ENTITY_ORDER = ENTITY_ORDER
    source.LONG_SCALES = LONG_SCALES
    source.INTERMEDIATE_SCALES = INTERMEDIATE_SCALES
    source.ALL_TIMEPOINTS = ALL_TIMEPOINTS
    source.LONG_TIMEPOINTS = LONG_TIMEPOINTS
    source.TIME_TICKS = TIME_TICKS
    source.MIDDLE_COLOR = MIDDLE_COLOR
    source.ENDPOINT_COLOR = ENDPOINT_COLOR

    renderer = source.load_renderer()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    renderer.OUT_DIR = OUT_DIR
    renderer.GROUPS = ENTITY_ORDER
    renderer.GROUP_COLORS = {
        "LRN-G1": "#E41A1C", "LRN-G2": "#377EB8", "LRN-G5": "#FF7F00",
        "SQL-G1": "#984EA3", "TW-G1": "#00A6C8", "ZYH-G1": "#A65628",
    }
    renderer.EXPERIMENT_LABEL = "0915实验"
    renderer.COMPARISON_LABEL = "LRN各实验组、SQL-G1、TW-G1与ZYH-G1"
    renderer.OUTER_RUNS_PER_ENTITY = 1
    renderer.TIMEPOINTS = LONG_TIMEPOINTS
    renderer.TIME_TICKS = ["T0", "S12"]
    renderer.ENDPOINT_LABEL = "session_12"
    renderer.ENDPOINT_DISPLAY = "S12"
    renderer.TIME_COLORS = {"T0": "#D55E00", "session_12": "#56B4E9"}
    renderer.configure_simplified_chinese_font()

    paths = renderer.find_report_files(REPORTS_DIR, recursive=True)
    records, labels, scales = renderer.load_records(paths)
    if labels != ALL_TIMEPOINTS or not set([*LONG_SCALES, *INTERMEDIATE_SCALES]).issubset(scales):
        raise ValueError(f"Unexpected 0915 design: labels={labels}, scales={scales}")
    selected = []
    for record in records:
        persona, entity = source.entity_for(record)
        if entity in ENTITY_ORDER:
            selected.append(replace(record, kbd=persona, group=entity))
    counts = {entity: sum(record.group == entity for record in selected) for entity in ENTITY_ORDER}
    if counts != {entity: 1 for entity in ENTITY_ORDER}:
        raise ValueError(f"Expected one outer run per entity; got {counts}")

    measurement_path = renderer.write_measurement_repeat_long_csv(OUT_DIR, selected, LONG_TIMEPOINTS, LONG_SCALES)
    item_measurement_path = renderer.write_item_measurement_repeat_long_csv(OUT_DIR, selected, LONG_TIMEPOINTS, LONG_SCALES)
    scores = renderer.score_long(selected, LONG_TIMEPOINTS)
    scores.to_csv(OUT_DIR / "outer_run_score_long.csv", index=False, encoding="utf-8-sig")
    changes = renderer.item_change_summary(selected, LONG_TIMEPOINTS)
    changes.to_csv(OUT_DIR / "item_change_entity_mean.csv", index=False, encoding="utf-8-sig")
    trajectory_path = OUT_DIR / "trajectory_measurement_summary.csv"
    source.trajectory_measurement_summary(selected).to_csv(trajectory_path, index=False, encoding="utf-8-sig")

    generated = [
        renderer.draw_boxplot(scores),
        source.draw_repeat_stability(
            renderer, selected, scales=LONG_SCALES, timepoints=LONG_TIMEPOINTS,
            time_ticks=["T0", "S12"], ranges={"PHQ-9": (0, 27), "BDI-II": (0, 63)},
            filename="figure2a_0915_long_scale_repeat_stability.png",
            title="0915实验：PHQ-9 / BDI-II 的快照内重复生成稳定性",
            repeats_note="PHQ-9 与 BDI-II 在每个 T0/S12 冻结快照各生成 K=10 次。",
        ),
        source.draw_repeat_stability(
            renderer, selected, scales=INTERMEDIATE_SCALES, timepoints=ALL_TIMEPOINTS[1:-1],
            time_ticks=TIME_TICKS[1:-1],
            ranges={"总体抑郁水平及干扰程度量表": (0, 15), "总体焦虑水平及干扰程度量表": (0, 15)},
            filename="figure2b_0915_intermediate_scale_repeat_stability.png",
            title="0915实验：中间短量表的快照内重复生成稳定性",
            repeats_note="中间总体抑郁/焦虑量表在每个 S2–S10 冻结快照各生成 K=5 次。",
        ),
        renderer.draw_item_change(scores, changes),
        draw_dual_axis_trajectory(renderer, selected),
    ]
    output_names = [
        "figure1_0915_persona_group_boxplots.png",
        "figure2a_0915_long_scale_repeat_stability.png",
        "figure2b_0915_intermediate_scale_repeat_stability.png",
        "figure3_0915_persona_group_item_change_composite.png",
        "figure4_0915_persona_group_dual_axis_scale_trajectory.png",
    ]
    outputs = []
    for path, name in zip(generated, output_names):
        target = path.with_name(name)
        if path != target:
            path.replace(target)
        outputs.append(target)

    manifest = {
        "source_reports_dir": str(REPORTS_DIR.relative_to(PROJECT_ROOT)),
        "report_files": [str(path.relative_to(PROJECT_ROOT)) for path in paths],
        "comparison_entities": ENTITY_ORDER,
        "entity_definition": "persona × experimental group; LRN has G1/G2/G5 and SQL/TW/ZYH have G1",
        "outer_runs_per_entity": counts,
        "long_scale_timepoints": LONG_TIMEPOINTS,
        "intermediate_scale_timepoints": ALL_TIMEPOINTS[1:-1],
        "long_scales": LONG_SCALES,
        "intermediate_scales": INTERMEDIATE_SCALES,
        "reliability_figures": "snapshot-level repeat-score clouds, means, and within-snapshot CIs; no pooled group ICC",
        "figure4_design": "dual y axes with intermediate short-scale trajectories on the left and endpoint PHQ-9/BDI-II measurement-CI points on the right",
        "figure4_ci_scope": "within frozen snapshot measurement repeats; not outer-run uncertainty",
        "measurement_repeats": {"PHQ-9/BDI-II": 10, "intermediate short scales": 5},
        "outputs": [path.name for path in outputs],
        "data_exports": [measurement_path.name, item_measurement_path.name, "outer_run_score_long.csv", "item_change_entity_mean.csv", trajectory_path.name],
    }
    (OUT_DIR / "plot_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT_DIR / "README.md").write_text(
        "# 0915 人设×实验组绘图\n\n"
        "比较实体为 LRN-G1、LRN-G2、LRN-G5、SQL-G1、TW-G1 与 ZYH-G1；每个实体仅有 1 次独立 outer run。图1、图3沿用首尾 PHQ-9 / BDI-II（T0、S12）的口径。图2a/2b 为快照内重复生成稳定性图：每个分面直接显示 K 次量表生成、均值和快照内 95% CI；不同实验组不合并为 outer-run 样本，也不报告组级 ICC。图4将中间总体抑郁/焦虑量表置于左轴，以同一蓝色的实线/虚线画轨迹；将 T0/S12 的 PHQ-9 / BDI-II 置于右轴，以同一橙色的不同点型和 95% CI 绘制。图中 CI 来自同一冻结快照内的量表生成（长量表 K=10；中间短量表 K=5），不是 outer-run 间 CI。\n\n"
        "重绘：`python render_0915_all_groups.py`。\n",
        encoding="utf-8",
    )
    print("Generated:")
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
