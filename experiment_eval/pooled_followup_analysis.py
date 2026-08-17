"""Cross-batch pooling diagnostics and lineage-safe 0802 follow-up figures."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from .charts.faceted.figure2 import plot_figure2
from .charts.faceted.figure4 import plot_figure4
from .charts.shared.faceted import PlotOptions
from .charts.shared.plotting import plt, save_figure


SCALES = ["PHQ-9", "BDI-II"]
FOLLOWUP_LABELS = [
    "session_20",
    "followup_step_30",
    "followup_step_60",
    "followup_step_90",
    "followup_step_120",
]
FOLLOWUP_TICKS = ["S20", "FU30", "FU60", "FU90", "FU120"]
COLORS = list(plt.get_cmap("tab10").colors)


def _entity_sort_key(value: str) -> tuple[int, int]:
    match = re.search(r"\d+", value)
    return (0 if value.startswith("KBD") else 1, int(match.group()) if match else 999)


def _mean_ci(values: pd.Series) -> tuple[float, float, float, int]:
    clean = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    n = len(clean)
    if not n:
        return math.nan, math.nan, math.nan, 0
    center = float(np.mean(clean))
    if n < 2:
        return center, center, center, n
    half = float(stats.t.ppf(0.975, n - 1) * stats.sem(clean))
    return center, center - half, center + half, n


def _batch(value: str) -> str:
    match = re.search(r"batch-(0802|0808)-", str(value))
    return match.group(1) if match else "unknown"


def _root_changes(frame: pd.DataFrame) -> pd.DataFrame:
    roots = frame[frame["source_kind"] == "root_repeat_summary"].copy()
    roots["source_batch"] = roots["run_id"].map(_batch)
    pivot = roots.pivot_table(
        index=["run_id", "persona_id", "group", "repeat_id", "source_batch", "scale"],
        columns="timepoint",
        values="score",
        aggfunc="first",
    ).reset_index()
    required = {"T0", "session_20"}
    if not required <= set(pivot.columns):
        raise ValueError("Root outcomes require T0 and session_20")
    pivot["endpoint_delta"] = pivot["session_20"] - pivot["T0"]
    return pivot


def _design_views(changes: pd.DataFrame) -> list[tuple[str, str, pd.DataFrame]]:
    return [
        ("KBD2 实验组", "group", changes[changes["persona_id"] == "KBD2"].copy()),
        ("G1 人设", "persona_id", changes[changes["group"] == "G1"].copy()),
    ]


def _batch_summary(changes: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for design, entity_col, subset in _design_views(changes):
        for keys, values in subset.groupby([entity_col, "source_batch", "scale"], sort=True):
            entity, batch, scale = keys
            mean, low, high, n = _mean_ci(values["endpoint_delta"])
            rows.append(
                {
                    "design": design,
                    "entity": entity,
                    "source_batch": batch,
                    "scale": scale,
                    "n_outer_runs": n,
                    "mean_endpoint_delta": mean,
                    "ci95_lower": low,
                    "ci95_upper": high,
                }
            )
    return pd.DataFrame(rows)


def _welch_difference(left: np.ndarray, right: np.ndarray) -> tuple[float, float, float]:
    estimate = float(np.mean(right) - np.mean(left))
    v1 = float(np.var(left, ddof=1) / len(left))
    v2 = float(np.var(right, ddof=1) / len(right))
    se = math.sqrt(v1 + v2)
    if not se:
        return estimate, estimate, estimate
    df = (v1 + v2) ** 2 / (
        (v1**2 / (len(left) - 1)) + (v2**2 / (len(right) - 1))
    )
    half = float(stats.t.ppf(0.975, df) * se)
    return estimate, estimate - half, estimate + half


def _batch_differences(changes: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for design, entity_col, subset in _design_views(changes):
        for (entity, scale), cell in subset.groupby([entity_col, "scale"], sort=True):
            old = cell.loc[cell["source_batch"] == "0802", "endpoint_delta"].to_numpy(float)
            new = cell.loc[cell["source_batch"] == "0808", "endpoint_delta"].to_numpy(float)
            if len(old) < 2 or len(new) < 2:
                continue
            estimate, low, high = _welch_difference(old, new)
            rows.append(
                {
                    "design": design,
                    "entity": entity,
                    "scale": scale,
                    "n_0802": len(old),
                    "n_0808": len(new),
                    "difference_0808_minus_0802": estimate,
                    "ci95_lower": low,
                    "ci95_upper": high,
                }
            )
    return pd.DataFrame(rows)


def _plot_batch_means(
    changes: pd.DataFrame, design: str, entity_col: str, out_path: Path
) -> Path:
    subset = changes[
        (changes["persona_id"] == "KBD2") if entity_col == "group" else (changes["group"] == "G1")
    ]
    entities = sorted(subset[entity_col].unique(), key=_entity_sort_key)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), constrained_layout=True)
    offsets = {"0802": -0.14, "0808": 0.14}
    for ax, scale in zip(axes, SCALES):
        panel = subset[subset["scale"] == scale]
        for batch, color in zip(["0802", "0808"], ["#2563EB", "#E11D48"]):
            xs: list[float] = []
            means: list[float] = []
            lows: list[float] = []
            highs: list[float] = []
            for index, entity in enumerate(entities):
                values = panel.loc[
                    (panel[entity_col] == entity) & (panel["source_batch"] == batch),
                    "endpoint_delta",
                ]
                mean, low, high, _ = _mean_ci(values)
                x = index + offsets[batch]
                xs.append(x); means.append(mean); lows.append(mean - low); highs.append(high - mean)
                ax.scatter(
                    np.full(len(values), x) + np.linspace(-0.035, 0.035, len(values)),
                    values,
                    color=color,
                    s=24,
                    alpha=0.55,
                    zorder=3,
                )
            ax.errorbar(xs, means, yerr=[lows, highs], fmt="o", color=color, capsize=3, label=f"{batch}（n={'2' if batch == '0802' else '3'}/格）")
        ax.axhline(0, color="#64748B", linewidth=1)
        ax.set_xticks(range(len(entities)), entities)
        ax.set_ylabel("session_20 − T0（负值=改善）")
        ax.set_title(scale)
        ax.grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False)
    fig.suptitle(f"{design}：0802 与 0808 批次一致性")
    return save_figure(fig, out_path)


def _plot_pooled_distribution(
    changes: pd.DataFrame, design: str, entity_col: str, out_path: Path
) -> Path:
    subset = changes[
        (changes["persona_id"] == "KBD2") if entity_col == "group" else (changes["group"] == "G1")
    ]
    entities = sorted(subset[entity_col].unique(), key=_entity_sort_key)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), constrained_layout=True)
    for ax, scale in zip(axes, SCALES):
        panel = subset[subset["scale"] == scale]
        arrays = [panel.loc[panel[entity_col] == entity, "endpoint_delta"].to_numpy(float) for entity in entities]
        ax.boxplot(arrays, tick_labels=entities, showfliers=False, widths=0.55)
        for index, entity in enumerate(entities, start=1):
            cell = panel[panel[entity_col] == entity]
            for batch, color, marker in [("0802", "#2563EB", "o"), ("0808", "#E11D48", "s")]:
                values = cell.loc[cell["source_batch"] == batch, "endpoint_delta"].to_numpy(float)
                ax.scatter(
                    np.full(len(values), index) + np.linspace(-0.08, 0.08, len(values)),
                    values,
                    color=color,
                    marker=marker,
                    s=25,
                    alpha=0.7,
                )
        ax.axhline(0, color="#64748B", linewidth=1)
        ax.set_title(scale)
        ax.set_ylabel("session_20 − T0（负值=改善）")
        ax.grid(axis="y", alpha=0.25)
    axes[0].scatter([], [], color="#2563EB", marker="o", label="0802")
    axes[0].scatter([], [], color="#E11D48", marker="s", label="0808")
    axes[0].legend(frameon=False)
    fig.suptitle(f"{design}：合并后的五次 outer-run 分布")
    return save_figure(fig, out_path)


def _plot_batch_forest(differences: pd.DataFrame, out_path: Path) -> Path:
    designs = ["KBD2 实验组", "G1 人设"]
    fig, axes = plt.subplots(1, 2, figsize=(15, 7.5), constrained_layout=True)
    for ax, design in zip(axes, designs):
        panel = differences[differences["design"] == design].copy()
        panel["label"] = panel["entity"] + " / " + panel["scale"]
        panel = panel.sort_values(["entity", "scale"])
        y = np.arange(len(panel))
        x = panel["difference_0808_minus_0802"].to_numpy(float)
        low = panel["ci95_lower"].to_numpy(float)
        high = panel["ci95_upper"].to_numpy(float)
        ax.errorbar(x, y, xerr=[x - low, high - x], fmt="o", color="#334155", capsize=3)
        ax.axvline(0, color="#DC2626", linewidth=1)
        ax.set_yticks(y, panel["label"])
        ax.invert_yaxis()
        ax.set_xlabel("0808 − 0802 的终点变化差")
        ax.set_title(design)
        ax.grid(axis="x", alpha=0.25)
    fig.suptitle("批次差异 forest（CI 跨 0 表示方向不确定）")
    return save_figure(fig, out_path)


def _followup_tables(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data = frame.copy()
    data["source_batch"] = data["run_id"].map(_batch)
    data = data[data["source_batch"] == "0802"].copy()
    selected = data[data["timepoint"].isin(["T0", *FOLLOWUP_LABELS])]
    pivot = selected.pivot_table(
        index=["run_id", "persona_id", "group", "repeat_id", "scale"],
        columns="timepoint",
        values="score",
        aggfunc="first",
    ).reset_index()
    for label in ["T0", *FOLLOWUP_LABELS]:
        if label not in pivot:
            raise ValueError(f"0802 follow-up is missing {label}")
    pivot["treatment_delta"] = pivot["session_20"] - pivot["T0"]
    pivot["maintenance_delta_120"] = pivot["followup_step_120"] - pivot["session_20"]
    pivot["total_delta_120"] = pivot["followup_step_120"] - pivot["T0"]
    pivot["rebounded_at_120"] = pivot["maintenance_delta_120"] > 0
    pivot["improvement_retained_at_120"] = pivot["total_delta_120"] < 0

    rows: list[dict[str, object]] = []
    for design, entity_col, subset in _design_views(pivot):
        for (entity, scale, timepoint), cell in (
            selected[
                (selected["persona_id"] == "KBD2")
                if entity_col == "group"
                else (selected["group"] == "G1")
            ]
            .groupby([entity_col, "scale", "timepoint"], sort=True)
        ):
            mean, low, high, n = _mean_ci(cell["score"])
            rows.append(
                {
                    "design": design,
                    "entity": entity,
                    "scale": scale,
                    "timepoint": timepoint,
                    "n_outer_runs": n,
                    "mean_score": mean,
                    "ci95_lower": low,
                    "ci95_upper": high,
                }
            )
    trajectory = pd.DataFrame(rows)

    maintenance_rows: list[dict[str, object]] = []
    for design, entity_col, subset in _design_views(pivot):
        for entity in sorted(subset[entity_col].unique()):
            for scale in SCALES:
                cell = subset[(subset[entity_col] == entity) & (subset["scale"] == scale)]
                for label in FOLLOWUP_LABELS[1:]:
                    values = cell[label] - cell["session_20"]
                    mean, low, high, n = _mean_ci(values)
                    maintenance_rows.append(
                        {
                            "design": design,
                            "entity": entity,
                            "scale": scale,
                            "timepoint": label,
                            "n_outer_runs": n,
                            "mean_change_from_session20": mean,
                            "ci95_lower": low,
                            "ci95_upper": high,
                        }
                    )
    return pivot, trajectory, pd.DataFrame(maintenance_rows)


def _plot_followup_trajectory(
    trajectory: pd.DataFrame, design: str, out_path: Path
) -> Path:
    panel_data = trajectory[trajectory["design"] == design]
    entities = sorted(panel_data["entity"].unique(), key=_entity_sort_key)
    order = ["T0", *FOLLOWUP_LABELS]
    ticks = ["T0", *FOLLOWUP_TICKS]
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.8), constrained_layout=True)
    for ax, scale in zip(axes, SCALES):
        for index, entity in enumerate(entities):
            cell = panel_data[(panel_data["entity"] == entity) & (panel_data["scale"] == scale)].set_index("timepoint")
            means = [cell.loc[label, "mean_score"] for label in order]
            x = np.arange(len(order))
            ax.plot(x, means, marker="o", color=COLORS[index % len(COLORS)], label=entity)
        ax.axvline(1, color="#64748B", linestyle="--", linewidth=1)
        ax.set_xticks(range(len(order)), ticks)
        ax.set_ylabel("量表总分")
        ax.set_ylim((0, 27) if scale == "PHQ-9" else (0, 63))
        ax.set_title(scale)
        ax.grid(axis="y", alpha=0.22)
    axes[1].legend(frameon=False, ncol=2, fontsize=8)
    fig.suptitle(f"0802 {design}：治疗期与无干预回访均值轨迹（每格 n=2）")
    return save_figure(fig, out_path)


def _plot_maintenance(
    maintenance: pd.DataFrame, design: str, out_path: Path
) -> Path:
    panel_data = maintenance[maintenance["design"] == design]
    entities = sorted(panel_data["entity"].unique(), key=_entity_sort_key)
    followups = FOLLOWUP_LABELS[1:]
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), constrained_layout=True)
    for ax, scale in zip(axes, SCALES):
        for index, entity in enumerate(entities):
            cell = panel_data[(panel_data["entity"] == entity) & (panel_data["scale"] == scale)].set_index("timepoint")
            means = [cell.loc[label, "mean_change_from_session20"] for label in followups]
            ax.plot(range(4), means, marker="o", color=COLORS[index % len(COLORS)], label=entity)
        ax.axhline(0, color="#64748B", linewidth=1)
        ax.set_xticks(range(4), ["FU30", "FU60", "FU90", "FU120"])
        ax.set_ylabel("回访 − session_20（正值=反弹）")
        ax.set_title(scale)
        ax.grid(axis="y", alpha=0.22)
    axes[1].legend(frameon=False, ncol=2, fontsize=8)
    fig.suptitle(f"0802 {design}：无干预期维持/反弹")
    return save_figure(fig, out_path)


def _entity_label(row: pd.Series) -> str:
    return str(row["persona_id"] if row["group"] == "G1" else row["group"])


def _plot_followup_heatmap(pivot: pd.DataFrame, out_path: Path) -> Path:
    data = pivot.copy()
    data["entity"] = data.apply(_entity_label, axis=1)
    table = data.pivot_table(index="entity", columns="scale", values="maintenance_delta_120", aggfunc="mean").reindex(columns=SCALES)
    table = table.loc[sorted(table.index, key=_entity_sort_key)]
    fig, ax = plt.subplots(figsize=(7.2, 8), constrained_layout=True)
    limit = max(1.0, float(np.nanmax(np.abs(table.to_numpy(float)))))
    image = ax.imshow(table, cmap="RdYlGn_r", vmin=-limit, vmax=limit, aspect="auto")
    ax.set_xticks(range(len(SCALES)), SCALES)
    ax.set_yticks(range(len(table)), table.index)
    persona_count = sum(str(value).startswith("KBD") for value in table.index)
    ax.axhline(persona_count - 0.5, color="#334155", linewidth=1.5)
    for i in range(len(table)):
        for j in range(len(SCALES)):
            ax.text(j, i, f"{table.iloc[i, j]:+.1f}", ha="center", va="center", fontsize=9)
    ax.set_title("0802 FU120 − session_20 平均变化（正值=反弹）")
    fig.colorbar(image, ax=ax, label="分数变化")
    return save_figure(fig, out_path)


def _plot_followup_rates(pivot: pd.DataFrame, out_path: Path) -> Path:
    data = pivot.copy()
    data["entity"] = data.apply(_entity_label, axis=1)
    rates = data.groupby(["entity", "scale"]).agg(
        rebound_rate=("rebounded_at_120", "mean"),
        retained_rate=("improvement_retained_at_120", "mean"),
    ).reset_index()
    entities = sorted(rates["entity"].unique(), key=_entity_sort_key)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)
    width = 0.34
    for ax, scale in zip(axes, SCALES):
        panel = rates[rates["scale"] == scale].set_index("entity").reindex(entities)
        x = np.arange(len(entities))
        ax.bar(x - width / 2, panel["rebound_rate"] * 100, width, label="FU120 高于 S20（反弹）", color="#E11D48")
        ax.bar(x + width / 2, panel["retained_rate"] * 100, width, label="FU120 仍低于 T0", color="#059669")
        ax.set_xticks(x, entities, rotation=45, ha="right")
        ax.set_ylim(0, 105)
        ax.set_ylabel("outer runs（%）")
        ax.set_title(scale)
        ax.grid(axis="y", alpha=0.22)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("0802 FU120：反弹率与总体改善保留率")
    return save_figure(fig, out_path)


def _plot_spaghetti(pivot: pd.DataFrame, out_path: Path) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    for row, (design, entity_col, subset) in enumerate(_design_views(pivot)):
        for col, scale in enumerate(SCALES):
            ax = axes[row, col]
            panel = subset[subset["scale"] == scale]
            entities = sorted(panel[entity_col].unique(), key=_entity_sort_key)
            colors = {entity: COLORS[i % len(COLORS)] for i, entity in enumerate(entities)}
            for _, record in panel.iterrows():
                ax.plot(range(5), [record[label] for label in FOLLOWUP_LABELS], color=colors[record[entity_col]], alpha=0.48, linewidth=1.4)
            for entity in entities:
                cell = panel[panel[entity_col] == entity]
                ax.plot(range(5), [cell[label].mean() for label in FOLLOWUP_LABELS], color=colors[entity], linewidth=2.8, label=entity)
            ax.set_xticks(range(5), FOLLOWUP_TICKS)
            ax.set_title(f"{design} / {scale}")
            ax.set_ylabel("量表总分")
            ax.grid(axis="y", alpha=0.2)
            ax.legend(frameon=False, ncol=2, fontsize=7)
    fig.suptitle("0802 无干预回访：逐 outer-run 轨迹（粗线为单元均值）")
    return save_figure(fig, out_path)


def _plot_gain_erosion(pivot: pd.DataFrame, out_path: Path) -> Path:
    data = pivot.copy()
    data["entity"] = data.apply(_entity_label, axis=1)
    entities = sorted(data["entity"].unique(), key=_entity_sort_key)
    color_map = {entity: COLORS[i % len(COLORS)] for i, entity in enumerate(entities)}
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    for ax, scale in zip(axes, SCALES):
        panel = data[data["scale"] == scale]
        for entity in entities:
            cell = panel[panel["entity"] == entity]
            ax.scatter(cell["treatment_delta"], cell["maintenance_delta_120"], color=color_map[entity], label=entity, s=42, alpha=0.75)
        ax.axhline(0, color="#64748B", linewidth=1)
        ax.axvline(0, color="#64748B", linewidth=1)
        ax.set_xlabel("session_20 − T0（治疗期变化）")
        ax.set_ylabel("FU120 − session_20（维持期变化）")
        ax.set_title(scale)
        ax.grid(alpha=0.2)
    axes[1].legend(frameon=False, ncol=2, fontsize=7)
    fig.suptitle("治疗期改善与 FU120 反弹的 run-level 关系")
    return save_figure(fig, out_path)


def _plot_measurement_sd(frame: pd.DataFrame, out_path: Path) -> Path:
    data = frame.copy()
    data["source_batch"] = data["run_id"].map(_batch)
    data = data[(data["source_batch"] == "0802") & data["timepoint"].isin(FOLLOWUP_LABELS)]
    fig, ax = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
    for scale, color in zip(SCALES, ["#2563EB", "#E11D48"]):
        means = [data.loc[(data["scale"] == scale) & (data["timepoint"] == label), "measurement_sd"].mean() for label in FOLLOWUP_LABELS]
        ax.plot(range(5), means, marker="o", linewidth=2.2, color=color, label=scale)
    ax.set_xticks(range(5), FOLLOWUP_TICKS)
    ax.set_ylabel("10 次 frozen-snapshot 测量的平均 SD")
    ax.set_title("0802 回访节点的测量生成稳定性")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    return save_figure(fig, out_path)


def _plot_elapsed_days(frame: pd.DataFrame, out_path: Path) -> Path:
    data = frame.copy()
    data["source_batch"] = data["run_id"].map(_batch)
    data = data[(data["source_batch"] == "0802") & data["timepoint"].isin(FOLLOWUP_LABELS)]
    unique = data.drop_duplicates(["run_id", "timepoint"])
    session_days = unique[unique["timepoint"] == "session_20"].set_index("run_id")["elapsed_days"]
    rows = []
    for _, row in unique[unique["timepoint"] != "session_20"].iterrows():
        rows.append({"timepoint": row["timepoint"], "days_after_session20": row["elapsed_days"] - session_days[row["run_id"]]})
    values = pd.DataFrame(rows)
    arrays = [values.loc[values["timepoint"] == label, "days_after_session20"].to_numpy(float) for label in FOLLOWUP_LABELS[1:]]
    fig, ax = plt.subplots(figsize=(8.5, 5.2), constrained_layout=True)
    ax.boxplot(
        arrays,
        tick_labels=["FU30", "FU60", "FU90", "FU120"],
        showfliers=False,
    )
    for index, array in enumerate(arrays, start=1):
        ax.scatter(np.full(len(array), index) + np.linspace(-0.08, 0.08, len(array)), array, s=16, alpha=0.4, color="#334155")
    ax.set_ylabel("距 session_20 的实际模拟天数")
    ax.set_title("回访 step 标签与实际模拟时间间隔")
    ax.grid(axis="y", alpha=0.25)
    return save_figure(fig, out_path)


def _write_faceted(changes: pd.DataFrame, out_dir: Path) -> list[Path]:
    paths: list[Path] = []
    options = PlotOptions(
        title="0802+0808 五次重复：T0 与 session_20",
        show_points=True,
        annotate_n=True,
        category_labels="bottom",
        formats=["png", "pdf"],
    )
    for name, entity_col, subset in [
        ("figure2_kbd2_group_five_repeat_boxplots", "group", changes[changes["persona_id"] == "KBD2"]),
        ("figure2_g1_persona_five_repeat_boxplots", "persona_id", changes[changes["group"] == "G1"]),
    ]:
        rows = []
        for _, row in subset.iterrows():
            for time, value in [("pre", row["T0"]), ("post", row["session_20"])]:
                rows.append(
                    {
                        "sample_id": row["run_id"],
                        "study": "0802+0808 pooled",
                        "group": row[entity_col],
                        "severity_group": "not-a-design-factor",
                        "time": time,
                        "outcome": row["scale"],
                        "value": value,
                    }
                )
        paths.extend(plot_figure2(pd.DataFrame(rows), out_dir / name, options, outcome_order=SCALES))

    for name, entity_col, subset, title in [
        (
            "figure4_kbd2_batch_by_group_boxplots",
            "group",
            changes[changes["persona_id"] == "KBD2"],
            "0802/0808 批次 × KBD2 实验条件：T0 与 session_20",
        ),
        (
            "figure4_g1_batch_by_persona_boxplots",
            "persona_id",
            changes[changes["group"] == "G1"],
            "0802/0808 批次 × G1 人设：T0 与 session_20",
        ),
    ]:
        rows = []
        for _, row in subset.iterrows():
            for time, value in [("pre", row["T0"]), ("post", row["session_20"])]:
                rows.append(
                    {
                        "sample_id": row["run_id"],
                        "study": row["source_batch"],
                        "group": row[entity_col],
                        "severity_group": row[entity_col],
                        "time": time,
                        "outcome": row["scale"],
                        "value": value,
                    }
                )
        figure4_options = PlotOptions(
            title=title,
            show_points=True,
            annotate_n=True,
            category_labels="bottom",
            formats=["png", "pdf"],
        )
        paths.extend(
            plot_figure4(
                pd.DataFrame(rows),
                out_dir / name,
                figure4_options,
                outcome_order=SCALES,
                study_order=["0802", "0808"],
                severity_order=sorted(subset[entity_col].unique(), key=_entity_sort_key),
            )
        )
    return paths


def run(outcomes_path: Path, out_dir: Path) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    batch_dir = out_dir / "06_批次一致性与合并诊断"
    followup_dir = out_dir / "07_0802无干预回访"
    faceted_dir = out_dir / "08_分面箱线图"
    for path in [batch_dir, followup_dir, faceted_dir]:
        path.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(outcomes_path)
    changes = _root_changes(frame)
    cell_counts = changes.groupby(["persona_id", "group", "scale"])['run_id'].nunique()
    if not (
        (cell_counts.loc[("KBD2", slice(None), slice(None))] == 5).all()
        and (cell_counts.loc[(slice(None), "G1", slice(None))] == 5).all()
    ):
        raise ValueError("The pooled design does not contain exactly five root runs per target cell")

    batch_summary = _batch_summary(changes)
    differences = _batch_differences(changes)
    batch_summary.to_csv(batch_dir / "batch_cell_summary.csv", index=False, encoding="utf-8-sig")
    differences.to_csv(batch_dir / "batch_difference_0808_minus_0802.csv", index=False, encoding="utf-8-sig")

    images = [
        _plot_batch_means(changes, "KBD2 实验组", "group", batch_dir / "batch_01_kbd2_endpoint_change_by_source_batch.png"),
        _plot_batch_means(changes, "G1 人设", "persona_id", batch_dir / "batch_02_g1_persona_endpoint_change_by_source_batch.png"),
        _plot_pooled_distribution(changes, "KBD2 实验组", "group", batch_dir / "batch_03_kbd2_combined_five_run_distribution.png"),
        _plot_pooled_distribution(changes, "G1 人设", "persona_id", batch_dir / "batch_04_g1_combined_five_run_distribution.png"),
        _plot_batch_forest(differences, batch_dir / "batch_05_source_batch_difference_forest.png"),
    ]

    followup_runs, trajectory, maintenance = _followup_tables(frame)
    followup_runs.to_csv(followup_dir / "followup_run_metrics.csv", index=False, encoding="utf-8-sig")
    trajectory.to_csv(followup_dir / "followup_trajectory_summary.csv", index=False, encoding="utf-8-sig")
    maintenance.to_csv(followup_dir / "followup_maintenance_summary.csv", index=False, encoding="utf-8-sig")
    images.extend(
        [
            _plot_followup_trajectory(trajectory, "KBD2 实验组", followup_dir / "followup_01_kbd2_group_trajectory.png"),
            _plot_followup_trajectory(trajectory, "G1 人设", followup_dir / "followup_02_g1_persona_trajectory.png"),
            _plot_maintenance(maintenance, "KBD2 实验组", followup_dir / "followup_03_kbd2_maintenance_change.png"),
            _plot_maintenance(maintenance, "G1 人设", followup_dir / "followup_04_g1_persona_maintenance_change.png"),
            _plot_followup_heatmap(followup_runs, followup_dir / "followup_05_fu120_maintenance_heatmap.png"),
            _plot_followup_rates(followup_runs, followup_dir / "followup_06_fu120_rebound_and_retention_rates.png"),
            _plot_spaghetti(followup_runs, followup_dir / "followup_07_individual_spaghetti.png"),
            _plot_gain_erosion(followup_runs, followup_dir / "followup_08_treatment_gain_vs_maintenance.png"),
            _plot_measurement_sd(frame, followup_dir / "followup_09_measurement_sd.png"),
            _plot_elapsed_days(frame, followup_dir / "followup_10_actual_elapsed_days.png"),
        ]
    )
    faceted = _write_faceted(changes, faceted_dir)

    pooled = changes.groupby(["scale"])["endpoint_delta"].agg(["mean", "std", "count"])
    fu = followup_runs.groupby("scale").agg(
        mean_treatment_delta=("treatment_delta", "mean"),
        mean_fu120_maintenance_delta=("maintenance_delta_120", "mean"),
        rebound_rate=("rebounded_at_120", "mean"),
        retained_rate=("improvement_retained_at_120", "mean"),
    )
    lines = [
        "# 0802+0808 批次合并与 0802 followup 分析",
        "",
        "- 主分析统计单位为独立根 outer run；0802 两次与 0808 三次映射为 R01–R05。",
        "- followup 通过 parent lineage 接回 0802 根 run，不增加独立样本量。",
        "- 分数变化为后时点减前时点；负值表示症状分数下降。",
        "",
        "## 联合终点概览",
        "",
    ]
    for scale, row in pooled.iterrows():
        lines.append(f"- {scale}: 65 个设计视图记录中的根 run 变化均值 {row['mean']:.2f}（SD {row['std']:.2f}）。")
    lines.extend(["", "## 0802 FU120 概览", ""])
    for scale, row in fu.iterrows():
        lines.append(
            f"- {scale}: session_20→FU120 平均 {row['mean_fu120_maintenance_delta']:+.2f}；"
            f"反弹率 {row['rebound_rate']:.1%}；仍低于 T0 的比例 {row['retained_rate']:.1%}。"
        )
    lines.extend(
        [
            "",
            "## 解释边界",
            "",
            "- 每个 persona×condition 批次内只有 0802 n=2、0808 n=3；批次差 forest 的 CI 很宽，只用于检查明显的不一致方向。",
            "- followup 是无干预模拟期的 Agent 指标，不代表真人临床随访或复发率。",
            "- `followup_step_30/60/90/120` 是模拟 step 标签；实际时间间隔另见 `followup_10_actual_elapsed_days.png`。",
        ]
    )
    (out_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")
    manifest = {
        "schema_version": "pooled_followup_analysis_v1",
        "input": str(outcomes_path),
        "root_outer_runs": int(changes["run_id"].nunique()),
        "followup_root_runs_0802": int(followup_runs["run_id"].nunique()),
        "png_files": [str(path.relative_to(out_dir)) for path in images],
        "faceted_exports": [str(path.relative_to(out_dir)) for path in faceted],
    }
    (out_dir / "pooled_followup_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outcomes", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run(args.outcomes.resolve(), args.out_dir.resolve())
    print(
        f"Wrote {len(result['png_files'])} pooled/follow-up PNG figures for "
        f"{result['root_outer_runs']} root runs and "
        f"{result['followup_root_runs_0802']} linked 0802 follow-up roots."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
