"""Render only explicitly selected 0802+0808 report figures.

This module intentionally has no default figure selection.  Callers must pass
one or more ``--figure`` values, which prevents an incremental redraw request
from accidentally expanding into the repository's full chart suite.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from .charts.shared.faceted import PlotOptions, _draw_box_panel
from .charts.shared.plotting import (
    configure_simplified_chinese_font,
    plt,
    save_figure,
)
from .loader import load_records
from matplotlib.patches import Patch


RESULT_ROOT = Path("results/0802-0808联合实验结果图")
DEFAULT_G1_DATA = RESULT_ROOT / "02_G1跨人设_五次重复/measurement_repeat_long.csv"
DEFAULT_KBD2_DATA = RESULT_ROOT / "01_KBD2跨实验组_五次重复/measurement_repeat_long.csv"
DEFAULT_ARCHIVE_ITEMS = RESULT_ROOT / "00_规范数据接口/archive_items_long.csv"
DEFAULT_OUTPUT_DIR = RESULT_ROOT / "09_按需重绘图/第一批"
DEFAULT_0813_REPORTS_DIR = Path("results/experiment_data/reports")
DEFAULT_0813_OUTPUT_DIR = Path("docs/experiment_evaluation/0813_kbd2_cross_condition")
DEFAULT_0814_REPORTS_DIR = Path("results/experiment_data/reports")
DEFAULT_0814_OUTPUT_DIR = Path("docs/experiment_evaluation/0814_kbd2_cross_condition")

SCALES = ["PHQ-9", "BDI-II"]
TIMEPOINTS = ["T0", "session_4", "session_8", "session_12", "session_16", "session_20"]
TIME_TICKS = ["T0", "S4", "S8", "S12", "S16", "S20"]
TRAJECTORY_TIMEPOINTS = TIMEPOINTS + [
    "followup_step_30",
    "followup_step_60",
    "followup_step_90",
    "followup_step_120",
]
TRAJECTORY_TICKS = TIME_TICKS + ["FU30", "FU60", "FU90", "FU120"]
PERSONA_ORDER = ["KBD1", "KBD2", "KBD3", "KBD5", "KBD6", "KBD7"]
GROUP_ORDER = ["G1", "G2", "G3", "G4", "G5", "G6", "G7", "G9"]
GROUP_ORDER_0813 = ["G1", "G4", "G10", "G11", "G12"]
GROUP_ORDER_0814 = ["G1", "G2", "G4", "G5", "G11"]
TRAJECTORY_0813_TIMEPOINTS = ["T0", "session_4", "session_8", "session_12"]
TRAJECTORY_0813_TICKS = ["T0", "S4", "S8", "S12"]
SCALE_LIMITS = {"PHQ-9": (0, 27), "BDI-II": (0, 63)}

MAIN_TITLE = "0802+0808共5次重复实验的前后量表变化"
TIME_COLORS = {"pre": "#D55E00", "post": "#56B4E9"}
BOX_OPTIONS = PlotOptions(
    time_colors=TIME_COLORS,
    show_points=True,
    annotate_n=True,
    category_labels="bottom",
    formats=["png"],
    dpi=300,
)

FIGURE_FILES = {
    "g1-boxplots": "figure2_g1_persona_five_repeat_boxplots.png",
    "kbd2-boxplots": "figure2_kbd2_group_five_repeat_boxplots.png",
    "combined-boxplots": "figure4_g1_batch_by_persona_boxplots.png",
    "g1-trajectories": "figure_01_g1_cross_persona_scale_trajectory.png",
    "kbd2-trajectories": "figure_02_kbd2_cross_condition_scale_trajectory.png",
    "0813-kbd2-trajectories": "figure_01_0813_kbd2_cross_condition_scale_trajectory.png",
    "0814-kbd2-trajectories": "figure_01_0814_kbd2_cross_condition_scale_trajectory.png",
}


def _load_outer_run_scores(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {
        "stable_id",
        "outer_run_id",
        "persona",
        "group",
        "timepoint",
        "scale",
        "measurement_repeat_id",
        "reviewed_total_score",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path} 缺少必要列: {', '.join(missing)}")

    frame = frame[
        frame["timepoint"].isin(TIMEPOINTS) & frame["scale"].isin(SCALES)
    ].copy()
    frame["reviewed_total_score"] = pd.to_numeric(
        frame["reviewed_total_score"], errors="coerce"
    )
    if frame["reviewed_total_score"].isna().any():
        raise ValueError(f"{path} 存在无法解析的量表总分")

    repeat_counts = frame.groupby(["stable_id", "timepoint", "scale"], observed=True)[
        "measurement_repeat_id"
    ].nunique()
    if repeat_counts.empty or repeat_counts.nunique() != 1:
        raise ValueError(f"{path} 各 frozen snapshot 的测量重复数不一致")

    return (
        frame.groupby(
            ["stable_id", "outer_run_id", "persona", "group", "timepoint", "scale"],
            as_index=False,
            observed=True,
        )["reviewed_total_score"]
        .mean()
        .rename(columns={"reviewed_total_score": "score"})
    )


def _load_archive_outer_run_scores(path: Path) -> pd.DataFrame:
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

    frame = frame[
        frame["timepoint"].isin(TRAJECTORY_TIMEPOINTS)
        & frame["scale"].isin(SCALES)
        & frame["source_kind"].isin(
            ["root_repeat_summary", "linked_followup_repeat_summary"]
        )
    ].copy()
    frame["item_score"] = pd.to_numeric(frame["item_score"], errors="coerce")
    if frame["item_score"].isna().any():
        raise ValueError(f"{path} 存在无法解析的量表条目分数")
    duplicate_columns = ["run", "timepoint", "scale", "item"]
    if frame.duplicated(duplicate_columns).any():
        raise ValueError(f"{path} 存在重复的 run/timepoint/scale/item 行")

    scores = (
        frame.groupby(
            ["run", "persona", "group", "timepoint", "scale"],
            as_index=False,
            observed=True,
        )["item_score"]
        .sum()
        .rename(columns={"run": "stable_id", "item_score": "score"})
    )
    scores["outer_run_id"] = scores["stable_id"]
    return scores


def _validate_five_runs(
    frame: pd.DataFrame,
    entity_column: str,
    entities: Sequence[str],
    *,
    timepoints: Sequence[str] = TIMEPOINTS,
) -> None:
    expected = pd.MultiIndex.from_product(
        [entities, timepoints, SCALES], names=[entity_column, "timepoint", "scale"]
    )
    counts = frame.groupby([entity_column, "timepoint", "scale"], observed=True)[
        "stable_id"
    ].nunique()
    counts = counts.reindex(expected)
    invalid = counts[counts != 5]
    if not invalid.empty:
        preview = ", ".join(
            f"{index}={value}" for index, value in invalid.head(6).items()
        )
        raise ValueError(f"目标设计不是每格5次独立 outer runs: {preview}")


def _box_rows(frame: pd.DataFrame, entity_column: str) -> pd.DataFrame:
    endpoint = frame[frame["timepoint"].isin(["T0", "session_20"])].copy()
    endpoint["time"] = endpoint["timepoint"].map({"T0": "pre", "session_20": "post"})
    return endpoint.rename(columns={entity_column: "entity", "score": "value"})


def _time_legend(fig: Any, *, anchor: tuple[float, float]) -> None:
    handles = [
        Patch(
            facecolor=TIME_COLORS[value],
            edgecolor="#111827",
            linewidth=0.8,
            label={"pre": "前测（T0）", "post": "后测（session_20）"}[value],
        )
        for value in ("pre", "post")
    ]
    fig.legend(
        handles=handles,
        title="时间",
        loc="upper center",
        bbox_to_anchor=anchor,
        frameon=False,
        ncol=2,
    )


def _draw_named_box_panel(
    ax: Any,
    frame: pd.DataFrame,
    entities: Sequence[str],
    scale: str,
    panel_title: str,
    x_label: str,
    seed: int,
) -> None:
    panel = frame[frame["scale"] == scale]
    _draw_box_panel(
        ax,
        panel,
        list(entities),
        "entity",
        BOX_OPTIONS,
        seed=seed,
    )
    ax.set_title(panel_title, loc="left", fontsize=13, fontweight="bold", pad=34)
    ax.set_ylabel(f"{scale} 分数")
    ax.set_xlabel(x_label)


def _render_figure2(
    frame: pd.DataFrame,
    entities: Sequence[str],
    entity_column: str,
    x_label: str,
    output_path: Path,
) -> Path:
    data = _box_rows(frame, entity_column)
    fig, axes = plt.subplots(1, 2, figsize=(15.8, 6.2), squeeze=False)
    for index, scale in enumerate(SCALES):
        _draw_named_box_panel(
            axes[0, index],
            data,
            entities,
            scale,
            f"{chr(ord('A') + index)}  {scale}",
            x_label,
            821 + index,
        )
    fig.suptitle(MAIN_TITLE, fontsize=18, fontweight="bold", y=0.965)
    _time_legend(fig, anchor=(0.5, 0.91))
    fig.subplots_adjust(left=0.065, right=0.985, bottom=0.12, top=0.76, wspace=0.18)
    return save_figure(fig, output_path)


def _render_combined_boxplots(
    g1: pd.DataFrame, kbd2: pd.DataFrame, output_path: Path
) -> Path:
    panels = [
        (_box_rows(g1, "persona"), PERSONA_ORDER, "G1跨人设", "人设"),
        (_box_rows(kbd2, "group"), GROUP_ORDER, "KBD2跨实验条件", "实验条件"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(16.0, 11.3), squeeze=False, sharey="col")
    panel_index = 0
    for row_index, (data, entities, design, x_label) in enumerate(panels):
        for column_index, scale in enumerate(SCALES):
            letter = chr(ord("A") + panel_index)
            _draw_named_box_panel(
                axes[row_index, column_index],
                data,
                entities,
                scale,
                f"{letter}  {design} · {scale}",
                x_label,
                1281 + panel_index,
            )
            panel_index += 1
    fig.suptitle(MAIN_TITLE, fontsize=18, fontweight="bold", y=0.977)
    _time_legend(fig, anchor=(0.5, 0.94))
    fig.subplots_adjust(
        left=0.065, right=0.985, bottom=0.075, top=0.82, wspace=0.18, hspace=0.42
    )
    return save_figure(fig, output_path)


def _mean_ci(values: np.ndarray) -> tuple[float, float, float]:
    center = float(np.mean(values))
    if len(values) < 2:
        return center, center, center
    half = float(stats.t.ppf(0.975, len(values) - 1) * stats.sem(values))
    return center, center - half, center + half


def _load_kbd2_trajectory_records(
    reports_dir: Path, experiment_date: str, groups: Sequence[str]
) -> list[Any]:
    """Load completed KBD2 repeat summaries for one dated trajectory figure."""
    paths = sorted(reports_dir.glob(f"**/repeat-*{experiment_date}-*_summary.json"))
    if not paths:
        raise ValueError(f"{reports_dir} 中未找到 {experiment_date} 的完整复评汇总文件")
    records, labels, scales = load_records(paths)
    if labels != TRAJECTORY_0813_TIMEPOINTS:
        raise ValueError(
            f"{experiment_date} 评估节点与预期不一致: " + ", ".join(labels)
        )
    if scales != SCALES:
        raise ValueError(f"{experiment_date} 量表与预期不一致: " + ", ".join(scales))
    selected = [
        record
        for record in records
        if record.kbd == "KBD2" and record.group in groups
    ]
    counts = {group: sum(record.group == group for record in selected) for group in groups}
    invalid = {group: count for group, count in counts.items() if count != 2}
    if invalid:
        detail = ", ".join(f"{group}={count}" for group, count in invalid.items())
        raise ValueError(
            f"{experiment_date} 每个实验条件应有 2 次独立 outer runs，实际为: {detail}"
        )
    return selected


def _render_kbd2_trajectory(
    records: list[Any],
    experiment_date: str,
    groups: Sequence[str],
    output_path: Path,
) -> Path:
    """Render one dated KBD2 condition-by-scale trajectory figure."""
    x_values = np.arange(len(TRAJECTORY_0813_TIMEPOINTS))
    height = 2.15 * len(groups) + 1.7
    fig, axes = plt.subplots(
        len(groups),
        len(SCALES),
        figsize=(15.2, height),
        squeeze=False,
        sharex=True,
        sharey="col",
    )
    for row_index, group in enumerate(groups):
        group_records = [record for record in records if record.group == group]
        for column_index, scale in enumerate(SCALES):
            ax = axes[row_index, column_index]
            trajectories = []
            for record in group_records:
                trajectory = np.array(
                    [record.series[scale][timepoint]["score"] for timepoint in TRAJECTORY_0813_TIMEPOINTS],
                    dtype=float,
                )
                trajectories.append(trajectory)
                ax.plot(
                    x_values,
                    trajectory,
                    color="#7C3AED",
                    alpha=0.18,
                    linewidth=1.15,
                )
            values = np.asarray(trajectories, dtype=float)
            centers = np.mean(values, axis=0)
            if len(values) < 2:
                lowers = centers
                uppers = centers
            else:
                half_widths = stats.t.ppf(0.975, len(values) - 1) * stats.sem(values, axis=0)
                lowers = centers - half_widths
                uppers = centers + half_widths
            ax.fill_between(
                x_values, lowers, uppers, color="#7C3AED", alpha=0.14, linewidth=0
            )
            ax.plot(
                x_values,
                centers,
                color="#7C3AED",
                marker="o",
                markersize=4.2,
                linewidth=2.3,
                zorder=3,
            )
            ax.set_ylim(*SCALE_LIMITS[scale])
            ax.grid(axis="y", color="#CBD5E1", linewidth=0.65, alpha=0.6)
            ax.set_axisbelow(True)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.text(
                0.985,
                0.91,
                "n=2",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=9,
                color="#475569",
            )
            if row_index == 0:
                ax.set_title(scale, fontsize=13, fontweight="bold", pad=10)
            if column_index == 0:
                ax.set_ylabel(f"{group}\n量表分数")
            else:
                ax.set_ylabel("量表分数")
            ax.set_xticks(x_values, TRAJECTORY_0813_TICKS)

    fig.suptitle(
        f"{experiment_date}实验：KBD2跨实验条件量表轨迹",
        fontsize=18,
        fontweight="bold",
        y=0.985,
    )
    fig.text(
        0.5,
        0.018,
        "细线表示独立 outer run；粗线与阴影表示每组 2 次 outer runs 的均值及 t 分布95%置信区间。"
        f"每个 frozen snapshot 的10次量表生成仅用于估计该节点均值；{experiment_date}实验当前仅有 T0–S12 节点。",
        ha="center",
        fontsize=9.5,
        color="#475569",
    )
    fig.subplots_adjust(
        left=0.085, right=0.985, bottom=0.075, top=0.925, wspace=0.13, hspace=0.20
    )
    return save_figure(fig, output_path)


def _render_trajectories(
    frame: pd.DataFrame,
    entities: Sequence[str],
    entity_column: str,
    title: str,
    color: str,
    output_path: Path,
) -> Path:
    x_values = np.arange(len(TRAJECTORY_TIMEPOINTS))
    height = 2.15 * len(entities) + 2.0
    fig, axes = plt.subplots(
        len(entities),
        2,
        figsize=(15.2, height),
        squeeze=False,
        sharex=True,
        sharey="col",
    )
    for row_index, entity in enumerate(entities):
        for column_index, scale in enumerate(SCALES):
            ax = axes[row_index, column_index]
            cell = frame[(frame[entity_column] == entity) & (frame["scale"] == scale)]
            pivot = cell.pivot(
                index="stable_id", columns="timepoint", values="score"
            ).reindex(columns=TRAJECTORY_TIMEPOINTS)
            ax.axvspan(5.5, 9.35, color="#F1F5F9", alpha=0.8, zorder=0)
            ax.axvline(5.5, color="#64748B", linewidth=0.85, linestyle="--")
            for _, run in pivot.iterrows():
                ax.plot(
                    x_values,
                    run.to_numpy(dtype=float),
                    color=color,
                    alpha=0.18,
                    linewidth=1.15,
                )

            centers: list[float] = []
            lowers: list[float] = []
            uppers: list[float] = []
            for timepoint in TRAJECTORY_TIMEPOINTS:
                values = pivot[timepoint].dropna().to_numpy(dtype=float)
                center, lower, upper = _mean_ci(values)
                centers.append(center)
                lowers.append(lower)
                uppers.append(upper)
            ax.fill_between(
                x_values, lowers, uppers, color=color, alpha=0.14, linewidth=0
            )
            ax.plot(
                x_values,
                centers,
                color=color,
                marker="o",
                markersize=4.2,
                linewidth=2.3,
                zorder=3,
            )
            ax.set_ylim(*SCALE_LIMITS[scale])
            ax.grid(axis="y", color="#CBD5E1", linewidth=0.65, alpha=0.6)
            ax.set_axisbelow(True)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.text(
                0.985,
                0.91,
                "n=5",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=9,
                color="#475569",
            )
            if row_index == 0:
                ax.set_title(scale, fontsize=13, fontweight="bold", pad=10)
            if column_index == 0:
                ax.set_ylabel(f"{entity}\n量表分数")
            else:
                ax.set_ylabel("量表分数")
            ax.set_xticks(x_values, TRAJECTORY_TICKS)
            if row_index == 0:
                ax.text(
                    3.0,
                    1.085,
                    "仿真阶段",
                    transform=ax.get_xaxis_transform(),
                    ha="center",
                    va="bottom",
                    fontsize=9.5,
                    color="#475569",
                )
                ax.text(
                    7.5,
                    1.085,
                    "无干预回访",
                    transform=ax.get_xaxis_transform(),
                    ha="center",
                    va="bottom",
                    fontsize=9.5,
                    color="#475569",
                )

    fig.suptitle(title, fontsize=18, fontweight="bold", y=0.985)
    fig.text(
        0.5,
        0.018,
        "细线表示独立 outer run；粗线与阴影表示5次 outer runs 的均值及 t 分布95%置信区间。"
        "每个 frozen snapshot 的10次量表生成仅用于估计该节点均值；FU30–FU120为无干预回访。",
        ha="center",
        fontsize=9.5,
        color="#475569",
    )
    fig.subplots_adjust(
        left=0.085, right=0.985, bottom=0.065, top=0.935, wspace=0.13, hspace=0.20
    )
    return save_figure(fig, output_path)


def run(
    g1_data_path: Path,
    kbd2_data_path: Path,
    output_dir: Path,
    figures: Sequence[str],
    *,
    archive_items_path: Path = DEFAULT_ARCHIVE_ITEMS,
    reports_0813_dir: Path = DEFAULT_0813_REPORTS_DIR,
    output_0813_dir: Path = DEFAULT_0813_OUTPUT_DIR,
    reports_0814_dir: Path = DEFAULT_0814_REPORTS_DIR,
    output_0814_dir: Path = DEFAULT_0814_OUTPUT_DIR,
) -> list[Path]:
    requested = list(dict.fromkeys(figures))
    unknown = sorted(set(requested) - set(FIGURE_FILES))
    if unknown:
        raise ValueError(f"未知图表键: {', '.join(unknown)}")
    if not requested:
        raise ValueError("必须至少显式选择一个图表键")

    configure_simplified_chinese_font()
    needs_g1 = bool(set(requested) & {"g1-boxplots", "combined-boxplots"})
    needs_kbd2 = bool(set(requested) & {"kbd2-boxplots", "combined-boxplots"})
    g1 = pd.DataFrame()
    kbd2 = pd.DataFrame()
    if needs_g1:
        g1 = _load_outer_run_scores(g1_data_path)
        g1 = g1[g1["group"] == "G1"].copy()
        _validate_five_runs(g1, "persona", PERSONA_ORDER)
    if needs_kbd2:
        kbd2 = _load_outer_run_scores(kbd2_data_path)
        kbd2 = kbd2[kbd2["persona"] == "KBD2"].copy()
        _validate_five_runs(kbd2, "group", GROUP_ORDER)
    trajectory_data = pd.DataFrame(
        columns=["stable_id", "persona", "group", "timepoint", "scale", "score"]
    )
    if set(requested) & {"g1-trajectories", "kbd2-trajectories"}:
        trajectory_data = _load_archive_outer_run_scores(archive_items_path)
    g1_trajectory = trajectory_data[trajectory_data["group"] == "G1"].copy()
    kbd2_trajectory = trajectory_data[trajectory_data["persona"] == "KBD2"].copy()
    if "g1-trajectories" in requested:
        _validate_five_runs(
            g1_trajectory,
            "persona",
            PERSONA_ORDER,
            timepoints=TRAJECTORY_TIMEPOINTS,
        )
    if "kbd2-trajectories" in requested:
        _validate_five_runs(
            kbd2_trajectory,
            "group",
            GROUP_ORDER,
            timepoints=TRAJECTORY_TIMEPOINTS,
        )
    trajectory_specs = {
        "0813-kbd2-trajectories": (
            "0813",
            reports_0813_dir,
            output_0813_dir,
            GROUP_ORDER_0813,
        ),
        "0814-kbd2-trajectories": (
            "0814",
            reports_0814_dir,
            output_0814_dir,
            GROUP_ORDER_0814,
        ),
    }
    needs_standard_output = any(figure not in trajectory_specs for figure in requested)
    if needs_standard_output:
        output_dir.mkdir(parents=True, exist_ok=True)
    trajectory_records: dict[str, list[Any]] = {}
    for figure, (experiment_date, reports_dir, figure_dir, groups) in trajectory_specs.items():
        if figure in requested:
            trajectory_records[figure] = _load_kbd2_trajectory_records(
                reports_dir, experiment_date, groups
            )
            figure_dir.mkdir(parents=True, exist_ok=True)

    renderers: dict[str, Callable[[Path], Path]] = {
        "g1-boxplots": lambda path: _render_figure2(
            g1, PERSONA_ORDER, "persona", "人设", path
        ),
        "kbd2-boxplots": lambda path: _render_figure2(
            kbd2, GROUP_ORDER, "group", "实验条件", path
        ),
        "combined-boxplots": lambda path: _render_combined_boxplots(g1, kbd2, path),
        "g1-trajectories": lambda path: _render_trajectories(
            g1_trajectory,
            PERSONA_ORDER,
            "persona",
            "0802+0808共5次重复实验：G1跨人设量表轨迹（含无干预回访）",
            "#2563EB",
            path,
        ),
        "kbd2-trajectories": lambda path: _render_trajectories(
            kbd2_trajectory,
            GROUP_ORDER,
            "group",
            "0802+0808共5次重复实验：KBD2跨实验条件量表轨迹（含无干预回访）",
            "#7C3AED",
            path,
        ),
    }
    for figure, (experiment_date, _, _, groups) in trajectory_specs.items():
        renderers[figure] = lambda path, figure=figure, experiment_date=experiment_date, groups=groups: _render_kbd2_trajectory(
            trajectory_records[figure], experiment_date, groups, path
        )
    outputs = []
    for figure in requested:
        figure_dir = trajectory_specs[figure][2] if figure in trajectory_specs else output_dir
        outputs.append(renderers[figure](figure_dir / FIGURE_FILES[figure]))
    manifest = {
        "schema_version": "targeted_redraw_v1",
        "font_family": plt.rcParams["font.sans-serif"][0],
        "analysis_unit": "independent outer simulation run",
        "measurement_repeat_role": "10 frozen-snapshot generations collapsed to one outer-run mean",
        "selected_figures": requested,
        "png_files": [path.name for path in outputs],
    }
    for figure, (experiment_date, reports_dir, _, groups) in trajectory_specs.items():
        if figure not in requested:
            continue
        manifest[f"{experiment_date}_reports_dir"] = str(reports_dir)
        manifest[f"{experiment_date}_trajectory"] = {
            "timepoints": TRAJECTORY_0813_TIMEPOINTS,
            "groups": list(groups),
            "outer_runs_per_group": 2,
            "followup": "not available; not plotted",
        }
    if needs_standard_output:
        manifest["followup_timepoints"] = TRAJECTORY_TIMEPOINTS[len(TIMEPOINTS) :]
        manifest["followup_source"] = str(archive_items_path)
    manifest_dir = output_dir if needs_standard_output else outputs[0].parent
    (manifest_dir / "redraw_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return outputs


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--g1-data", type=Path, default=DEFAULT_G1_DATA)
    parser.add_argument("--kbd2-data", type=Path, default=DEFAULT_KBD2_DATA)
    parser.add_argument("--archive-items", type=Path, default=DEFAULT_ARCHIVE_ITEMS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--0813-reports-dir", type=Path, default=DEFAULT_0813_REPORTS_DIR
    )
    parser.add_argument(
        "--0813-out-dir", type=Path, default=DEFAULT_0813_OUTPUT_DIR
    )
    parser.add_argument(
        "--0814-reports-dir", type=Path, default=DEFAULT_0814_REPORTS_DIR
    )
    parser.add_argument(
        "--0814-out-dir", type=Path, default=DEFAULT_0814_OUTPUT_DIR
    )
    parser.add_argument(
        "--figure",
        action="append",
        choices=sorted(FIGURE_FILES),
        required=True,
        help="可重复传入；只渲染显式选择的图。",
    )
    args = parser.parse_args(argv)
    outputs = run(
        args.g1_data,
        args.kbd2_data,
        args.out_dir,
        args.figure,
        archive_items_path=args.archive_items,
        reports_0813_dir=args.__dict__["0813_reports_dir"],
        output_0813_dir=args.__dict__["0813_out_dir"],
        reports_0814_dir=args.__dict__["0814_reports_dir"],
        output_0814_dir=args.__dict__["0814_out_dir"],
    )
    print(f"已生成 {len(outputs)} 张指定图：")
    for output in outputs:
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
