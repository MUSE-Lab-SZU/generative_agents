"""Redraw explicitly selected scale-credibility figures with clear units."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .charts.shared.plotting import (
    Line2D,
    configure_simplified_chinese_font,
    heatmap_text_color,
    plt,
    save_figure,
)
from .life_state import ITEM_LABELS_ZH
from .schema import group_color, kbd_color


RESULT_ROOT = Path("results/0802-0808联合实验结果图")
SOURCE_DIR = RESULT_ROOT / "10_量表可信度分析"
DEFAULT_LONG_DATA = (
    RESULT_ROOT
    / "01_KBD2跨实验组_五次重复/long_format_paper_figures/symptom_long_data.csv"
)
DEFAULT_OUTPUT_DIR = RESULT_ROOT / "09_按需重绘图/第四批"
TIMEPOINTS = ["T0", "session_4", "session_8", "session_12", "session_16", "session_20"]
SCALES = ["PHQ-9", "BDI-II"]

DIMENSIONS = {
    "g1_by_persona": {
        "title": "G1 跨人设",
        "entity_column": "persona",
        "entities": ["KBD1", "KBD2", "KBD3", "KBD5", "KBD6", "KBD7"],
        "clusters": 30,
        "snapshots": 180,
        "entity_name": "人设",
    },
    "kbd2_by_condition": {
        "title": "KBD2 跨实验条件",
        "entity_column": "group",
        "entities": ["G1", "G2", "G3", "G4", "G5", "G6", "G7", "G9"],
        "clusters": 40,
        "snapshots": 240,
        "entity_name": "实验条件",
    },
}

FIGURES = {
    "core-02a-g1": ("core_02a", "g1_by_persona"),
    "core-02a-kbd2": ("core_02a", "kbd2_by_condition"),
    "core-02b-g1": ("core_02b", "g1_by_persona"),
    "core-02b-kbd2": ("core_02b", "kbd2_by_condition"),
    "core-03-g1": ("core_03", "g1_by_persona"),
    "core-03-kbd2": ("core_03", "kbd2_by_condition"),
    "weighted-kappa-02-g1": ("weighted_kappa_02", "g1_by_persona"),
    "weighted-kappa-02-kbd2": ("weighted_kappa_02", "kbd2_by_condition"),
}


def _output_name(kind: str, dimension: str) -> str:
    if kind == "core_02a":
        stem = "core_02a_cross_scale_concurrent_validity"
    elif kind == "core_02b":
        stem = "core_02b_cross_scale_change_agreement"
    elif kind == "core_03":
        stem = "core_03_measurement_icc"
    else:
        stem = "weighted_kappa_02_item_heatmap"
    return f"{stem}_{dimension}.png"


def _entity_color(entity: str) -> str:
    return group_color(entity) if entity.startswith("G") else kbd_color(entity)


def _statistics_box(ax: Any, text: str) -> None:
    ax.text(
        0.018,
        0.982,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.8,
        color="#0F172A",
        bbox={
            "boxstyle": "round,pad=0.38",
            "facecolor": "#F8FAFC",
            "edgecolor": "#94A3B8",
            "linewidth": 0.8,
            "alpha": 0.94,
        },
        zorder=8,
    )


def _load_cross_scale_rows(path: Path, dimension: str) -> pd.DataFrame:
    config = DIMENSIONS[dimension]
    frame = pd.read_csv(path)
    required = {
        "run", "persona", "group", "timepoint", "scale", "item",
        "item_score", "source_kind",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path} 缺少必要列: {', '.join(missing)}")
    selected = frame[
        (frame["source_kind"] == "root_repeat_summary")
        & frame["timepoint"].isin(TIMEPOINTS)
        & frame["scale"].isin(SCALES)
    ].copy()
    if dimension == "g1_by_persona":
        selected = selected[
            (selected["group"] == "G1")
            & selected["persona"].isin(config["entities"])
        ]
    else:
        selected = selected[
            (selected["persona"] == "KBD2")
            & selected["group"].isin(config["entities"])
        ]
    totals = (
        selected.groupby(["run", "persona", "group", "timepoint", "scale"], observed=True)[
            "item_score"
        ]
        .sum()
        .rename("score")
        .reset_index()
    )
    rows = totals.pivot(
        index=["run", "persona", "group", "timepoint"],
        columns="scale",
        values="score",
    ).reset_index()
    if rows[SCALES].isna().any().any():
        raise ValueError(f"{dimension} 存在无法对齐的 PHQ-9/BDI-II 快照")
    counts = rows.groupby(config["entity_column"])["run"].nunique().reindex(config["entities"])
    if counts.isna().any() or not (counts == 5).all():
        raise ValueError(f"{dimension} 必须为每个实体5次独立外层重复: {counts.to_dict()}")
    if rows["run"].nunique() != config["clusters"] or len(rows) != config["snapshots"]:
        raise ValueError(f"{dimension} 的run数或快照数与设计不一致")
    return rows


def _cluster_spearman(
    rows: pd.DataFrame, x: str, y: str, *, seed: int
) -> tuple[float, float, float, float]:
    result = spearmanr(rows[x], rows[y])
    clusters = {key: part for key, part in rows.groupby("run", sort=True)}
    cluster_ids = sorted(clusters)
    rng = np.random.default_rng(seed)
    estimates: list[float] = []
    for _ in range(300):
        sampled = rng.choice(cluster_ids, size=len(cluster_ids), replace=True)
        boot = pd.concat([clusters[str(cluster)] for cluster in sampled], ignore_index=True)
        value = float(spearmanr(boot[x], boot[y]).statistic)
        if math.isfinite(value):
            estimates.append(value)
    return (
        float(result.statistic),
        float(result.pvalue),
        float(np.quantile(estimates, 0.025)),
        float(np.quantile(estimates, 0.975)),
    )


def _plot_cross_scale(
    rows: pd.DataFrame,
    dimension: str,
    *,
    change: bool,
    output_path: Path,
) -> Path:
    config = DIMENSIONS[dimension]
    entity_column = str(config["entity_column"])
    if change:
        endpoints = rows[rows["timepoint"].isin(["T0", "session_20"])].pivot(
            index=["run", "persona", "group"],
            columns="timepoint",
            values=SCALES,
        )
        change_rows = pd.DataFrame(
            {
                "run": endpoints.index.get_level_values("run"),
                "persona": endpoints.index.get_level_values("persona"),
                "group": endpoints.index.get_level_values("group"),
                "x": endpoints[("PHQ-9", "session_20")]
                - endpoints[("PHQ-9", "T0")],
                "y": endpoints[("BDI-II", "session_20")]
                - endpoints[("BDI-II", "T0")],
            }
        ).reset_index(drop=True)
        plot_rows = change_rows
        rho, p_value, lower, upper = _cluster_spearman(
            plot_rows, "x", "y", seed=20260813
        )
        direction = float(
            np.mean(
                ((plot_rows["x"] == 0) & (plot_rows["y"] == 0))
                | (plot_rows["x"] * plot_rows["y"] > 0)
            )
        )
        title = (
            f"{config['title']}：PHQ-9 / BDI-II 前后变化一致性\n"
            f"方向一致率={direction:.1%}；Spearman ρ={rho:.2f} "
            f"[{lower:.2f}, {upper:.2f}]；p={p_value:.3g}"
        )
        x_label = "ΔPHQ-9（S20 − T0）"
        y_label = "ΔBDI-II（S20 − T0）"
        unit_text = (
            f"统计单位：1个点 = 1次独立外层run的T0→S20配对变化\n"
            f"样本量：{config['clusters']}个独立外层run（每{config['entity_name']}5个）"
        )
    else:
        plot_rows = rows.rename(columns={"PHQ-9": "x", "BDI-II": "y"})
        rho, _p_value, lower, upper = _cluster_spearman(
            plot_rows, "x", "y", seed=20260812
        )
        title = (
            f"{config['title']}：PHQ-9 / BDI-II 同期总分收敛效度\n"
            f"Spearman ρ={rho:.2f}；外层run聚类95% CI [{lower:.2f}, {upper:.2f}]"
        )
        x_label = "PHQ-9 总分"
        y_label = "BDI-II 总分"
        unit_text = (
            "观察单位：1个点 = 同一外层run在同一时间点的两量表对齐快照\n"
            f"观察数={config['snapshots']}；推断/聚类单位=独立外层run（{config['clusters']}个）"
        )

    fig, ax = plt.subplots(figsize=(10.4, 8.3), constrained_layout=True)
    handles: list[Line2D] = []
    for entity in config["entities"]:
        subset = plot_rows[plot_rows[entity_column] == entity]
        color = _entity_color(str(entity))
        if change:
            agreement = ((subset["x"] == 0) & (subset["y"] == 0)) | (
                subset["x"] * subset["y"] > 0
            )
            ax.scatter(
                subset.loc[agreement, "x"], subset.loc[agreement, "y"],
                color=color, s=58, alpha=0.82, marker="o",
            )
            ax.scatter(
                subset.loc[~agreement, "x"], subset.loc[~agreement, "y"],
                color=color, s=72, marker="x", linewidth=2.0,
            )
        else:
            ax.scatter(subset["x"], subset["y"], color=color, s=38, alpha=0.58)
        x_values = subset["x"].to_numpy(float)
        y_values = subset["y"].to_numpy(float)
        if len(x_values) >= 2 and len(np.unique(x_values)) > 1:
            slope, intercept = np.polyfit(x_values, y_values, 1)
            line_x = np.linspace(float(x_values.min()), float(x_values.max()), 80)
            ax.plot(line_x, slope * line_x + intercept, color=color, linewidth=1.9)
        handles.append(
            Line2D(
                [0], [0], color=color, marker="o", linewidth=1.8,
                label=f"{entity}（n={len(subset)}）", markersize=6,
            )
        )
    if change:
        ax.axhline(0, color="#64748B", linewidth=0.9)
        ax.axvline(0, color="#64748B", linewidth=0.9)
        handles.append(
            Line2D([0], [0], marker="x", color="#475569", linewidth=0,
                   label="两量表变化方向不一致", markersize=7)
        )
    ax.set_xlabel(x_label, fontsize=11)
    ax.set_ylabel(y_label, fontsize=11)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.grid(color="#CBD5E1", linewidth=0.65, alpha=0.6)
    ax.legend(handles=handles, frameon=False, fontsize=8.2, ncol=2, loc="lower right")
    _statistics_box(ax, unit_text)
    footer = (
        "每个实体的实线仅为描述性OLS拟合；主相关统计为整套数据的Spearman ρ。"
        + ("负值表示改善，正值表示加重。" if change else "同一run内的6个时间点不视为6个独立推断单位。")
    )
    fig.text(0.5, -0.012, footer, ha="center", fontsize=8.2, color="#475569")
    return save_figure(fig, output_path)


def _plot_icc(source_dir: Path, dimension: str, output_path: Path) -> Path:
    config = DIMENSIONS[dimension]
    frame = pd.read_csv(source_dir / f"measurement_icc_{dimension}.csv")
    frame = frame[(frame["status"] == "ok") & frame["scale"].isin(SCALES)].copy()
    rows: list[dict[str, Any]] = []
    for scale in SCALES:
        rows.append({"kind": "header", "label": scale})
        scale_rows = frame[frame["scale"] == scale].set_index("entity")
        for entity in [*config["entities"], "Overall"]:
            row = scale_rows.loc[entity].to_dict()
            rows.append({"kind": "estimate", "entity": entity, **row})
        if scale != SCALES[-1]:
            rows.append({"kind": "spacer"})

    fig = plt.figure(figsize=(12.8, max(7.4, 0.48 * len(rows) + 2.0)), constrained_layout=True)
    grid = fig.add_gridspec(1, 3, width_ratios=(2.75, 4.65, 3.0), wspace=0.02)
    label_ax, forest_ax, value_ax = [fig.add_subplot(grid[0, i]) for i in range(3)]
    y_values = list(reversed(range(len(rows))))
    for axis in (label_ax, forest_ax, value_ax):
        axis.set_ylim(-0.7, len(rows) - 0.3)
    label_ax.axis("off")
    value_ax.axis("off")
    value_ax.text(0.02, len(rows) - 0.05, "ICC(A,1)（95% CI）", fontsize=10,
                  fontweight="bold", va="top")
    lower_min = float(frame["icc_a_1_ci95_lower"].min())
    x_min = min(0.28, lower_min - 0.04)
    forest_ax.set_xlim(x_min, 1.01)
    forest_ax.axvline(0.75, color="#94A3B8", linewidth=0.9, linestyle="--")
    forest_ax.grid(axis="x", color="#CBD5E1", alpha=0.55)
    forest_ax.set_xlabel("绝对一致性 ICC(A,1)")
    forest_ax.set_yticks([])
    for spine in ("top", "left", "right"):
        forest_ax.spines[spine].set_visible(False)

    for y, row in zip(y_values, rows):
        if row["kind"] == "header":
            label_ax.text(0.0, y, row["label"], fontsize=11, fontweight="bold", va="center")
            continue
        if row["kind"] == "spacer":
            continue
        overall = row["entity"] == "Overall"
        entity_label = "总体" if overall else str(row["entity"])
        label_ax.text(
            0.04 if overall else 0.08, y,
            f"{entity_label}（快照={int(row['n_snapshot_targets'])}；run={int(row['n_outer_run_clusters'])}）",
            fontsize=8.7, fontweight="bold" if overall else "normal", va="center",
        )
        estimate = float(row["icc_a_1"])
        lower = float(row["icc_a_1_ci95_lower"])
        upper = float(row["icc_a_1_ci95_upper"])
        forest_ax.errorbar(
            estimate, y,
            xerr=[[estimate - lower], [upper - estimate]],
            fmt="D" if overall else "o", markersize=7 if overall else 4.8,
            color="#0F172A" if overall else "#2563EB",
            ecolor="#0F172A" if overall else "#64748B", capsize=2.5,
        )
        value_ax.text(
            0.02, y, f"{estimate:.3f} [{lower:.3f}, {upper:.3f}]",
            fontsize=8.8, fontweight="bold" if overall else "normal", va="center",
        )

    fig.suptitle(
        f"量表总分的生成重复信度：{config['title']}",
        fontsize=15, fontweight="bold", x=0.02, ha="left",
    )
    fig.text(
        0.02, 0.955,
        "统计单位：目标=1个冻结快照×1张量表；K=10次量表生成作为重复测量。"
        f" 95% CI按独立外层run聚类bootstrap（每{config['entity_name']}5个；总体{config['clusters']}个）。",
        ha="left", va="top", fontsize=9.2, color="#0F172A",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#F8FAFC", "edgecolor": "#94A3B8"},
    )
    fig.text(
        0.5, -0.012,
        "ICC(A,1)/ICC(2,1)：two-way random、absolute agreement、single generation；◆为总体。"
        "这是冻结快照下的生成信度，不是临床重测信度。",
        ha="center", fontsize=8.2, color="#475569",
    )
    return save_figure(fig, output_path)


def _plot_kappa(source_dir: Path, dimension: str, output_path: Path) -> Path:
    config = DIMENSIONS[dimension]
    frame = pd.read_csv(source_dir / f"weighted_kappa_item_{dimension}.csv")
    frame = frame[(frame["weights"] == "quadratic") & frame["scale"].isin(SCALES)].copy()
    fig, axes = plt.subplots(
        1, 2, figsize=(18.0 if dimension == "kbd2_by_condition" else 15.5, 10.4),
        constrained_layout=True,
    )
    image = None
    for panel, ax, scale in zip(("A", "B"), axes, SCALES):
        items = list(range(1, 10 if scale == "PHQ-9" else 22))
        subset = frame[frame["scale"] == scale]
        matrix = (
            subset.pivot(index="item_id", columns="entity", values="kappa")
            .reindex(index=items, columns=config["entities"])
            .to_numpy(float)
        )
        image = ax.imshow(matrix, cmap="RdYlBu", vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(np.arange(len(config["entities"])), config["entities"],
                      rotation=35, ha="right", fontsize=9)
        labels = [f"I{item:02d}  {ITEM_LABELS_ZH[scale][item]}" for item in items]
        ax.set_yticks(np.arange(len(items)), labels, fontsize=7.2)
        ax.set_xticks(np.arange(-0.5, len(config["entities"]), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(items), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.05)
        ax.tick_params(which="minor", bottom=False, left=False)
        for row_index in range(matrix.shape[0]):
            for column_index in range(matrix.shape[1]):
                value = matrix[row_index, column_index]
                if math.isfinite(value):
                    ax.text(
                        column_index, row_index, f"{value:.2f}",
                        ha="center", va="center", fontsize=7.0,
                        color=heatmap_text_color(image.cmap, image.norm, float(value)),
                    )
        ax.set_title(f"{panel}  {scale}", loc="left", fontsize=12, fontweight="bold")
        ax.set_xlabel(str(config["entity_name"]), fontsize=10)
    if image is not None:
        colorbar = fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.72, pad=0.02)
        colorbar.set_label("二次加权 κ", fontsize=10)
    fig.suptitle(f"条目级生成重复信度：{config['title']}", fontsize=15, fontweight="bold")
    fig.text(
        0.5, 0.965,
        "统计单位：被评分对象=1个冻结快照×1个量表条目；评分者=10次measurement repeat。"
        "每格汇总5个独立外层run×6个时间点=30个对象，并取45个repeat-pair二次加权κ的非加权均值。",
        ha="center", va="top", fontsize=9.1, color="#0F172A",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#F8FAFC", "edgecolor": "#94A3B8"},
    )
    fig.text(
        0.5, -0.012,
        "单元格为点估计；κ越高表示同一冻结快照的条目评分在10次生成之间越稳定。",
        ha="center", fontsize=8.3, color="#475569",
    )
    return save_figure(fig, output_path)


def _write_documentation(
    output_dir: Path,
    selected: Sequence[str],
    paths: Sequence[Path],
    *,
    source_dir: Path,
    long_data: Path,
) -> None:
    command = [
        "python -m experiment_eval.targeted_redraw_scale_credibility \\",
        *[
            f"  --figure {key}" + (" \\" if index < len(selected) - 1 else "")
            for index, key in enumerate(selected)
        ],
    ]
    readme = f"""# 第四批：量表可信度按需重绘

本目录仅包含下列 {len(paths)} 张可信度图。每张图均在图内明确写出观察/统计单位、独立外层 run 数，以及 measurement repeat 的角色；“第三批”文件保持独立，不混放。

## 运行命令

```bash
{chr(10).join(command)}
```

## 本次新增图

{chr(10).join(f'- `{path.name}`' for path in paths)}

未生成 `core_04`、legacy overall、repeat-pair heatmap、PDF 或 SVG。
"""
    (output_dir / "README_量表可信度重绘.md").write_text(readme, encoding="utf-8")
    manifest = {
        "schema_version": "1.0",
        "selected_figures": list(selected),
        "source_directory": str(source_dir),
        "long_data": str(long_data),
        "png_files": [path.name for path in paths],
        "statistical_units": {
            "core_02a": "aligned outer-run × timepoint snapshot; inference clustered by independent outer run",
            "core_02b": "one T0-to-S20 paired change per independent outer run",
            "core_03": "frozen snapshot × scale target; 10 generations are repeat measurements; CI clustered by outer run",
            "weighted_kappa_02": "frozen snapshot × scale × item rated object; 10 generation repeats form 45 rater pairs",
        },
        "intentionally_not_generated": [
            "core_04", "legacy overall figures", "repeat-pair heatmaps", "PDF", "SVG",
        ],
    }
    (output_dir / "scale_credibility_redraw_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def render_selected(
    selected: Sequence[str],
    *,
    source_dir: Path = SOURCE_DIR,
    long_data: Path = DEFAULT_LONG_DATA,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> list[Path]:
    if not selected:
        raise ValueError("At least one --figure selection is required")
    unknown = sorted(set(selected) - set(FIGURES))
    if unknown:
        raise ValueError(f"Unknown figure selection: {', '.join(unknown)}")
    ordered = list(dict.fromkeys(selected))
    configure_simplified_chinese_font()
    output_dir.mkdir(parents=True, exist_ok=True)
    cached_rows: dict[str, pd.DataFrame] = {}
    paths: list[Path] = []
    for key in ordered:
        kind, dimension = FIGURES[key]
        output_path = output_dir / _output_name(kind, dimension)
        if kind in {"core_02a", "core_02b"}:
            if dimension not in cached_rows:
                cached_rows[dimension] = _load_cross_scale_rows(long_data, dimension)
            rows = cached_rows[dimension]
            path = _plot_cross_scale(
                rows, dimension, change=kind == "core_02b", output_path=output_path
            )
        elif kind == "core_03":
            path = _plot_icc(source_dir, dimension, output_path)
        else:
            path = _plot_kappa(source_dir, dimension, output_path)
        paths.append(path)
    _write_documentation(
        output_dir,
        ordered,
        paths,
        source_dir=source_dir,
        long_data=long_data,
    )
    return paths


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--figure", action="append", choices=list(FIGURES), required=True,
        help="Explicit figure key; repeat to render multiple figures.",
    )
    parser.add_argument("--source-dir", type=Path, default=SOURCE_DIR)
    parser.add_argument("--long-data", type=Path, default=DEFAULT_LONG_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    paths = render_selected(
        args.figure,
        source_dir=args.source_dir,
        long_data=args.long_data,
        output_dir=args.output_dir,
    )
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
