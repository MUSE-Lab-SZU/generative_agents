#!/usr/bin/env python3
"""Render the 0918-g1-24step pilot figures from three KBD1-G1 outer runs."""

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
ROOT = PROJECT_ROOT / "results/0918-g1-24step"
REPORTS_DIR = ROOT / "experiment_data/reports"
OUT_DIR = ROOT / "0918画图"
RUN_ORDER = ["R01", "R02", "R03"]
LONG_SCALES = ["PHQ-9", "BDI-II"]
INTERMEDIATE_SCALES = ["总体抑郁水平及干扰程度量表", "总体焦虑水平及干扰程度量表"]
ALL_TIMEPOINTS = ["T0", "session_2", "session_4"]
LONG_TIMEPOINTS = ["T0", "session_4"]
TIME_TICKS = ["T0", "S2", "S4"]
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


def draw_repeat_stability(
    renderer,
    records: list[object],
    *,
    scales: list[str],
    timepoints: list[str],
    time_ticks: list[str],
    ranges: dict[str, tuple[float, float]],
    filename: str,
    title: str,
    repeats_note: str,
) -> Path:
    """Draw one stability panel per independent outer run."""
    fig, axes = renderer.plt.subplots(
        len(scales), len(RUN_ORDER), figsize=(4.25 * len(RUN_ORDER), 3.45 * len(scales) + 1.45),
        sharex="row", sharey="row", squeeze=False,
    )
    x = np.arange(len(timepoints))
    for row, scale in enumerate(scales):
        for col, run_id in enumerate(RUN_ORDER):
            ax = axes[row, col]
            record = next(record for record in records if record.group == run_id)
            color = "#0072B2" if scale in {"PHQ-9", INTERMEDIATE_SCALES[0]} else "#D55E00" if scale == "BDI-II" else "#009E73"
            for time_index, timepoint in enumerate(timepoints):
                stats = record.series.get(scale, {}).get(timepoint) or {}
                values = np.asarray(stats["values"], dtype=float)
                jitter = np.linspace(-0.11, 0.11, len(values))
                ax.scatter(np.repeat(time_index, len(values)) + jitter, values, s=20, color="#334155", alpha=0.62, linewidths=0, zorder=2)
                mean_value = float(stats["score"])
                lower, upper = float(stats["ci95_lower"]), float(stats["ci95_upper"])
                ax.errorbar(time_index, mean_value, yerr=[[mean_value - lower], [upper - mean_value]], color=color, marker="o", markersize=5.4, capsize=3, linewidth=1.7, elinewidth=1.4, zorder=4)
            ax.set_ylim(*ranges[scale])
            ax.set_xticks(x, time_ticks)
            ax.grid(axis="y", color="#CBD5E1", linewidth=0.65, alpha=0.72)
            ax.set_axisbelow(True)
            ax.spines[["top", "right"]].set_visible(False)
            if row == 0:
                ax.set_title(run_id, fontsize=12, fontweight="bold", pad=9)
            if col == 0:
                ax.set_ylabel(f"{scale} 分数")
            ax.text(0.98, 0.93, f"outer run=1；K={len(values)}；快照={len(timepoints)}", transform=ax.transAxes, ha="right", va="top", fontsize=7.7, color="#64748B")
    for ax in axes[-1]:
        ax.set_xlabel("评估时间点")
    fig.suptitle(title, fontsize=16.5, fontweight="bold", y=0.99)
    fig.legend(
        handles=[
            renderer.Line2D([], [], color="#334155", marker="o", linestyle="None", alpha=0.7, label="同一冻结快照的单次量表生成"),
            renderer.Line2D([], [], color="#111827", marker="o", linewidth=1.5, label="均值及快照内 95% t CI"),
        ],
        loc="upper center", bbox_to_anchor=(0.5, 0.95), ncol=2, frameon=False, fontsize=9,
    )
    fig.text(0.5, 0.012, f"{repeats_note} 每个分面为 1 次独立 outer run；图中展示的是快照内生成稳定性，不将量表生成重复当成独立 outer run。", ha="center", fontsize=8.5, color="#475569")
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.075, top=0.88, wspace=0.20, hspace=0.25)
    return renderer.save_figure(fig, OUT_DIR / filename)


def draw_group_dual_axis_trajectory(renderer, records: list[object]) -> Path:
    """Show n=3 outer-run summaries without mixing snapshot and outer CIs."""
    x_by_label = {label: index for index, label in enumerate(ALL_TIMEPOINTS)}
    fig, ax_left = renderer.plt.subplots(figsize=(13.2, 6.8))
    ax_right = ax_left.twinx()
    rng = np.random.default_rng(918)

    for scale, marker in zip(INTERMEDIATE_SCALES, ("o", "^")):
        values = np.asarray(
            [record.series[scale]["session_2"]["score"] for record in records], dtype=float
        )
        center, low, high = renderer.mean_ci(values)
        jitter = rng.uniform(-0.055, 0.055, len(values))
        ax_left.scatter(np.repeat(x_by_label["session_2"], len(values)) + jitter, values, color=MIDDLE_COLOR, marker=marker, s=36, alpha=0.38, zorder=2)
        ax_left.errorbar(x_by_label["session_2"], center, yerr=[[center - low], [high - center]], color=MIDDLE_COLOR, marker=marker, markersize=7, capsize=4, linewidth=2.0, zorder=5)

    for scale, marker in zip(LONG_SCALES, ("o", "s")):
        matrix = np.asarray(
            [[record.series[scale][label]["score"] for label in LONG_TIMEPOINTS] for record in records],
            dtype=float,
        )
        for run_values in matrix:
            ax_right.plot([x_by_label[label] for label in LONG_TIMEPOINTS], run_values, color=ENDPOINT_COLOR, linestyle="--", marker=marker, markersize=3.6, linewidth=1.0, alpha=0.22, zorder=1)
        centers, lows, highs = [], [], []
        for index in range(matrix.shape[1]):
            center, low, high = renderer.mean_ci(matrix[:, index])
            centers.append(center); lows.append(low); highs.append(high)
        centers = np.asarray(centers); lows = np.asarray(lows); highs = np.asarray(highs)
        ax_right.errorbar([x_by_label[label] for label in LONG_TIMEPOINTS], centers, yerr=np.vstack((centers - lows, highs - centers)), color=ENDPOINT_COLOR, marker=marker, linestyle="None", markersize=7, capsize=4, elinewidth=1.8, zorder=6)

    ax_left.set_xlim(-0.35, len(ALL_TIMEPOINTS) - 0.65)
    ax_left.set_ylim(0, 15)
    ax_right.set_ylim(0, 63)
    ax_left.set_xticks(range(len(ALL_TIMEPOINTS)), TIME_TICKS)
    ax_left.set_xlabel("评估时间点")
    ax_left.set_ylabel("中间总体量表分数", color=MIDDLE_COLOR)
    ax_right.set_ylabel("PHQ-9 / BDI-II 分数", color=ENDPOINT_COLOR)
    ax_left.tick_params(axis="y", colors=MIDDLE_COLOR)
    ax_right.tick_params(axis="y", colors=ENDPOINT_COLOR)
    ax_left.grid(axis="y", color="#CBD5E1", linewidth=0.7, alpha=0.75)
    ax_left.set_axisbelow(True)
    ax_left.spines[["top"]].set_visible(False)
    ax_right.spines[["top"]].set_visible(False)
    fig.suptitle("0918-g1-24step：KBD1-G1 量表轨迹", fontsize=17, fontweight="bold", y=0.98)
    fig.legend(
        handles=[
            renderer.Line2D([], [], color=MIDDLE_COLOR, marker="o", linestyle="None", label="总体抑郁水平及干扰程度量表（左轴）"),
            renderer.Line2D([], [], color=MIDDLE_COLOR, marker="^", linestyle="None", label="总体焦虑水平及干扰程度量表（左轴）"),
            renderer.Line2D([], [], color=ENDPOINT_COLOR, marker="o", linestyle="None", label="PHQ-9（右轴）"),
            renderer.Line2D([], [], color=ENDPOINT_COLOR, marker="s", linestyle="None", label="BDI-II（右轴）"),
        ],
        loc="upper center", bbox_to_anchor=(0.5, 0.93), ncol=2, frameon=False, fontsize=9,
    )
    fig.text(
        0.5, 0.02,
        "浅色虚线/散点为 3 次独立 outer run；深色点及误差线为 run 间均值与 95% t CI。中间短量表仅在 S2 观测；首尾 PHQ-9/BDI-II 仅在 T0/S4 观测。",
        ha="center", fontsize=9, color="#475569",
    )
    fig.subplots_adjust(left=0.10, right=0.90, bottom=0.12, top=0.84)
    return renderer.save_figure(fig, OUT_DIR / "figure4_0918_kbd1_g1_dual_axis_scale_trajectory.png")


def main() -> None:
    source = load_reference()
    source.ROOT = ROOT
    source.REPORTS_DIR = REPORTS_DIR
    source.OUT_DIR = OUT_DIR
    source.ENTITY_ORDER = RUN_ORDER
    source.LONG_SCALES = LONG_SCALES
    source.INTERMEDIATE_SCALES = INTERMEDIATE_SCALES
    source.ALL_TIMEPOINTS = ALL_TIMEPOINTS
    source.LONG_TIMEPOINTS = LONG_TIMEPOINTS
    source.TIME_TICKS = TIME_TICKS

    renderer = source.load_renderer()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    renderer.OUT_DIR = OUT_DIR
    renderer.GROUPS = ["G1"]
    renderer.GROUP_COLORS = {"G1": "#0072B2"}
    renderer.EXPERIMENT_LABEL = "0918-g1-24step"
    renderer.COMPARISON_LABEL = "KBD1-G1（3 次独立 outer run）"
    renderer.OUTER_RUNS_PER_ENTITY = 3
    renderer.TIMEPOINTS = LONG_TIMEPOINTS
    renderer.TIME_TICKS = ["T0", "S4"]
    renderer.ENDPOINT_LABEL = "session_4"
    renderer.ENDPOINT_DISPLAY = "S4"
    renderer.TIME_COLORS = {"T0": "#D55E00", "session_4": "#56B4E9"}
    renderer.configure_simplified_chinese_font()

    paths = renderer.find_report_files(REPORTS_DIR, recursive=True)
    records, labels, scales = renderer.load_records(paths)
    if labels != ALL_TIMEPOINTS or not set([*LONG_SCALES, *INTERMEDIATE_SCALES]).issubset(scales):
        raise ValueError(f"Unexpected pilot design: labels={labels}, scales={scales}")
    if len(records) != 3 or [record.repeat_id for record in records] != RUN_ORDER:
        raise ValueError(f"Expected three KBD1-G1 runs {RUN_ORDER}; got {[record.repeat_id for record in records]}")
    grouped = [replace(record, kbd="KBD1", group="G1") for record in records]
    by_run = [replace(record, kbd="KBD1", group=record.repeat_id) for record in records]

    measurement_path = renderer.write_measurement_repeat_long_csv(OUT_DIR, grouped, LONG_TIMEPOINTS, LONG_SCALES)
    item_measurement_path = renderer.write_item_measurement_repeat_long_csv(OUT_DIR, grouped, LONG_TIMEPOINTS, LONG_SCALES)
    scores = renderer.score_long(grouped, LONG_TIMEPOINTS)
    scores.to_csv(OUT_DIR / "outer_run_score_long.csv", index=False, encoding="utf-8-sig")
    changes = renderer.item_change_summary(grouped, LONG_TIMEPOINTS)
    changes.to_csv(OUT_DIR / "item_change_group_mean.csv", index=False, encoding="utf-8-sig")
    trajectory_path = OUT_DIR / "trajectory_measurement_summary.csv"
    source.trajectory_measurement_summary(by_run).to_csv(trajectory_path, index=False, encoding="utf-8-sig")

    generated = [
        renderer.draw_boxplot(scores),
        draw_repeat_stability(renderer, by_run, scales=LONG_SCALES, timepoints=LONG_TIMEPOINTS, time_ticks=["T0", "S4"], ranges={"PHQ-9": (0, 27), "BDI-II": (0, 63)}, filename="figure2a_0918_long_scale_repeat_stability.png", title="0918-g1-24step：PHQ-9 / BDI-II 的快照内重复生成稳定性", repeats_note="PHQ-9 与 BDI-II 在每个 T0/S4 冻结快照各生成 K=10 次。"),
        draw_repeat_stability(renderer, by_run, scales=INTERMEDIATE_SCALES, timepoints=["session_2"], time_ticks=["S2"], ranges={"总体抑郁水平及干扰程度量表": (0, 15), "总体焦虑水平及干扰程度量表": (0, 15)}, filename="figure2b_0918_intermediate_scale_repeat_stability.png", title="0918-g1-24step：中间短量表的快照内重复生成稳定性", repeats_note="中间总体抑郁/焦虑量表仅在 S2 冻结快照各生成 K=5 次。"),
        renderer.draw_item_change(scores, changes),
        draw_group_dual_axis_trajectory(renderer, grouped),
    ]
    output_names = [
        "figure1_0918_kbd1_g1_boxplots.png",
        "figure2a_0918_long_scale_repeat_stability.png",
        "figure2b_0918_intermediate_scale_repeat_stability.png",
        "figure3_0918_kbd1_g1_item_change_composite.png",
        "figure4_0918_kbd1_g1_dual_axis_scale_trajectory.png",
    ]
    outputs = []
    for path, name in zip(generated, output_names):
        target = path.with_name(name)
        if path != target:
            # Compatibility renderers return their historical 0819 filename
            # even when the shared save helper has already written the
            # requested target name.  Keep reruns idempotent in either case.
            if path.exists():
                path.replace(target)
            elif not target.exists():
                raise FileNotFoundError(f"Renderer produced neither {path} nor {target}")
        outputs.append(target)
    manifest = {
        "source_reports_dir": str(REPORTS_DIR.relative_to(PROJECT_ROOT)),
        "report_files": [str(path.relative_to(PROJECT_ROOT)) for path in paths],
        "entity_definition": "KBD1-G1 with three independent outer runs (R01–R03)",
        "outer_runs": RUN_ORDER,
        "timepoints": ALL_TIMEPOINTS,
        "long_scales": LONG_SCALES,
        "intermediate_scales": INTERMEDIATE_SCALES,
        "measurement_repeats": {"PHQ-9/BDI-II": 10, "intermediate short scales": 5},
        "figure2_scope": "snapshot-level repeated-generation stability by outer run; not a pooled group ICC",
        "figure4_scope": "outer-run mean and 95% t CI (n=3); long scales at T0/S4 and short scales at S2",
        "outputs": [path.name for path in outputs],
        "data_exports": [measurement_path.name, item_measurement_path.name, "outer_run_score_long.csv", "item_change_group_mean.csv", trajectory_path.name],
    }
    (OUT_DIR / "plot_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT_DIR / "README.md").write_text(
        "# 0918-g1-24step KBD1-G1 绘图\n\n"
        "输入目录名为 0918-g1-24step，但 3 份 repeat summary 的批次名为 0919；图题保留输入目录的实验标识。KBD1-G1 有 3 次独立 outer run（R01–R03）。图1/3/4以 outer run 为统计单位；图2a/2b展示每个 frozen snapshot 内 K 次量表生成的稳定性，不把生成重复当成独立 outer run。长量表仅在 T0/S4 观测（K=10），中间短量表仅在 S2 观测（K=5）。\n\n"
        "重绘：`python render_0918_24step.py`。\n",
        encoding="utf-8",
    )
    print("Generated:")
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
