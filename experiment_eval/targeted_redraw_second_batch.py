"""Render only the explicitly selected second-batch 0802+0808 figures."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from .charts.process.engagement import plot_engagement_process
from .charts.shared.plotting import configure_simplified_chinese_font, plt, save_figure
from .life_state import ITEM_LABELS_ZH, SYMPTOM_MAP


RESULT_ROOT = Path("results/0802-0808联合实验结果图")
DEFAULT_OUTPUT_DIR = RESULT_ROOT / "09_按需重绘图/第二批"
DEFAULT_G1_ITEM_CHANGE = (
    RESULT_ROOT / "04_分层解释分析/cross_persona_g1/life_state_item_change.csv"
)
DEFAULT_KBD2_ITEM_CHANGE = (
    RESULT_ROOT / "04_分层解释分析/kbd2_conditions/life_state_item_change.csv"
)
DEFAULT_G1_STATE_CHANGE = (
    RESULT_ROOT / "04_分层解释分析/cross_persona_g1/life_state_symptom_change.csv"
)
DEFAULT_KBD2_STATE_CHANGE = (
    RESULT_ROOT / "04_分层解释分析/kbd2_conditions/life_state_symptom_change.csv"
)
DEFAULT_ARCHIVE_ITEMS = RESULT_ROOT / "00_规范数据接口/archive_items_long.csv"
DEFAULT_ENGAGEMENT_EVENTS = RESULT_ROOT / "02_G1跨人设_五次重复/engagement_events.csv"
DEFAULT_ENGAGEMENT_DAILY = (
    RESULT_ROOT / "02_G1跨人设_五次重复/engagement_daily_summary.csv"
)
DEFAULT_ENGAGEMENT_BINS = (
    RESULT_ROOT / "02_G1跨人设_五次重复/engagement_24h_summary.csv"
)
DEFAULT_ENGAGEMENT_SESSIONS = (
    RESULT_ROOT / "02_G1跨人设_五次重复/engagement_session_outcome_summary.csv"
)

SCALES = ["PHQ-9", "BDI-II"]
PERSONAS = ["KBD1", "KBD2", "KBD3", "KBD5", "KBD6", "KBD7"]
GROUPS = ["G1", "G2", "G3", "G4", "G5", "G6", "G7", "G9"]
COLORS = {
    "g1": "#2563EB",
    "kbd2": "#7C3AED",
    "followup": "#0F766E",
    "negative": "#16A34A",
    "positive": "#DC2626",
    "neutral": "#64748B",
}

FIGURE_FILES = {
    "symptom-forest-bdi2": "symptom_effect_forest_bdi2.png",
    "symptom-forest-phq9": "symptom_effect_forest_phq9.png",
    "g1-item-forest": "life_state_01_g1_persona_item_change_forest.png",
    "kbd2-item-forest": "life_state_02_kbd2_condition_item_change_forest.png",
    "g1-state-forest": "life_state_03_g1_persona_nine_state_change_forest.png",
    "kbd2-state-forest": "life_state_04_kbd2_condition_nine_state_change_forest.png",
    "engagement-g1": "paper_figure_03_engagement_process_g1.png",
}


def _read_csv(path: Path, required: set[str]) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path} 缺少必要列: {', '.join(missing)}")
    return frame


def _mean_ci(values: pd.Series | np.ndarray) -> tuple[float, float, float, int]:
    clean = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(float)
    n = len(clean)
    if not n:
        return np.nan, np.nan, np.nan, 0
    center = float(np.mean(clean))
    if n < 2:
        return center, center, center, n
    half = float(stats.t.ppf(0.975, n - 1) * stats.sem(clean))
    return center, center - half, center + half, n


def _summarize(
    frame: pd.DataFrame,
    entity_column: str,
    metric_column: str,
    label_column: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    group_columns = [entity_column, metric_column, label_column]
    for keys, values in frame.groupby(group_columns, sort=False, observed=True):
        entity, metric, label = keys
        center, lower, upper, n = _mean_ci(values["change"])
        rows.append(
            {
                "entity": str(entity),
                "metric": metric,
                "label": str(label),
                "mean_change": center,
                "ci95_lower": lower,
                "ci95_upper": upper,
                "n_outer_runs": n,
            }
        )
    return pd.DataFrame(rows)


def _followup_item_changes(path: Path) -> pd.DataFrame:
    frame = _read_csv(
        path,
        {
            "run",
            "persona",
            "group",
            "timepoint",
            "scale",
            "item",
            "item_score",
            "source_kind",
        },
    )
    frame = frame[
        frame["timepoint"].isin(["session_20", "followup_step_120"])
        & frame["source_kind"].isin(
            ["root_repeat_summary", "linked_followup_repeat_summary"]
        )
    ].copy()

    def build_design(selected: pd.DataFrame, design: str) -> pd.DataFrame:
        duplicate_columns = ["run", "scale", "item", "timepoint"]
        if selected.duplicated(duplicate_columns).any():
            raise ValueError(
                f"{design} follow-up item data contains duplicate run/item/timepoint rows"
            )
        pivot = selected.pivot(
            index=["run", "persona", "group", "scale", "item"],
            columns="timepoint",
            values="item_score",
        ).reset_index()
        if pivot[["session_20", "followup_step_120"]].isna().any().any():
            raise ValueError(
                f"{design} follow-up item data is not fully linked to session_20"
            )
        pivot["change"] = pivot["followup_step_120"] - pivot["session_20"]
        pivot["item_label_zh"] = pivot.apply(
            lambda row: ITEM_LABELS_ZH[str(row["scale"])][int(row["item"])],
            axis=1,
        )
        pivot["design"] = design
        return pivot.rename(columns={"run": "stable_id", "item": "item_id"})

    g1 = build_design(
        frame[(frame["group"] == "G1") & frame["persona"].isin(PERSONAS)],
        "G1跨人设组",
    )
    kbd2 = build_design(
        frame[(frame["persona"] == "KBD2") & frame["group"].isin(GROUPS)],
        "KBD2跨实验条件组",
    )
    g1_counts = g1.groupby(["persona", "scale"])["stable_id"].nunique()
    if len(g1_counts) != len(PERSONAS) * len(SCALES) or not (g1_counts == 5).all():
        raise ValueError(
            "G1 FU120 does not contain five linked outer runs per persona/scale"
        )
    kbd2_counts = kbd2.groupby(["group", "scale"])["stable_id"].nunique()
    if len(kbd2_counts) != len(GROUPS) * len(SCALES) or not (kbd2_counts == 5).all():
        raise ValueError(
            "KBD2 FU120 does not contain five linked outer runs per group/scale"
        )
    return pd.concat([g1, kbd2], ignore_index=True)


def _item_order(scale: str) -> list[int]:
    return list(range(1, 10 if scale == "PHQ-9" else 22))


def _item_tick_labels(scale: str, items: Sequence[int]) -> list[str]:
    return [f"I{item:02d}  {ITEM_LABELS_ZH[scale][item]}" for item in items]


def _symmetric_limit(summaries: Sequence[pd.DataFrame], floor: float = 0.35) -> float:
    values: list[float] = []
    for summary in summaries:
        for column in ("ci95_lower", "ci95_upper"):
            values.extend(
                pd.to_numeric(summary[column], errors="coerce").dropna().tolist()
            )
    return max(floor, max((abs(value) for value in values), default=floor) * 1.08)


def _forest_marks(
    ax: Any,
    summary: pd.DataFrame,
    metric_order: Sequence[Any],
    color: str,
    *,
    labels: Sequence[str] | None,
    x_limit: float,
) -> None:
    selected = summary.set_index("metric").reindex(metric_order)
    centers = selected["mean_change"].to_numpy(float)
    lower = selected["ci95_lower"].to_numpy(float)
    upper = selected["ci95_upper"].to_numpy(float)
    y = np.arange(len(metric_order))
    errors = np.vstack([np.maximum(0, centers - lower), np.maximum(0, upper - centers)])
    ax.errorbar(
        centers,
        y,
        xerr=errors,
        fmt="o",
        color=color,
        ecolor=color,
        capsize=2.5,
        markersize=4.8,
        linewidth=1.2,
    )
    ax.axvline(0, color="#334155", linewidth=0.9)
    ax.set_xlim(-x_limit, x_limit)
    ax.set_yticks(y)
    if labels is not None:
        ax.set_yticklabels(labels)
    ax.set_ylim(len(metric_order) - 0.5, -0.5)
    ax.grid(axis="x", color="#CBD5E1", linewidth=0.65, alpha=0.65)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _plot_overall_item_forest(
    scale: str,
    g1_items: pd.DataFrame,
    kbd2_items: pd.DataFrame,
    followup_items: pd.DataFrame,
    output_path: Path,
) -> Path:
    items = _item_order(scale)
    g1 = g1_items[g1_items["scale"] == scale].copy()
    g1["design"] = "G1跨人设组"
    kbd2 = kbd2_items[kbd2_items["scale"] == scale].copy()
    kbd2["design"] = "KBD2跨实验条件组"
    followup = followup_items[followup_items["scale"] == scale].copy()
    g1_followup = followup[followup["design"] == "G1跨人设组"]
    kbd2_followup = followup[followup["design"] == "KBD2跨实验条件组"]
    summaries = [
        _summarize(g1, "design", "item_id", "item_label_zh"),
        _summarize(kbd2, "design", "item_id", "item_label_zh"),
        _summarize(g1_followup, "design", "item_id", "item_label_zh"),
        _summarize(kbd2_followup, "design", "item_id", "item_label_zh"),
    ]
    expected_ns = [30, 40, 30, 40]
    for summary, expected_n in zip(summaries, expected_ns):
        if not (summary["n_outer_runs"] == expected_n).all():
            raise ValueError(f"{scale} overall/follow-up forest has unexpected outer n")
    limit = _symmetric_limit(summaries)
    height = max(7.0, len(items) * 0.46 + 2.5)
    fig, axes = plt.subplots(1, 4, figsize=(20.5, height), sharey=True)
    titles = [
        "G1跨人设组\nT0→session_20（n=30）",
        "KBD2跨实验条件组\nT0→session_20（n=40）",
        "G1跨人设组\nsession_20→FU120（n=30）",
        "KBD2跨实验条件组\nsession_20→FU120（n=40）",
    ]
    colors = [COLORS["g1"], COLORS["kbd2"], COLORS["g1"], COLORS["kbd2"]]
    tick_labels = _item_tick_labels(scale, items)
    for index, (ax, summary, title, color) in enumerate(
        zip(axes, summaries, titles, colors)
    ):
        _forest_marks(
            ax,
            summary,
            items,
            color,
            labels=tick_labels if index in (0, 2) else None,
            x_limit=limit,
        )
        ax.set_title(title, fontsize=11, fontweight="bold", pad=10)
        ax.set_xlabel("后时点 − 前时点条目分数")
        if index == 2:
            ax.tick_params(labelleft=True)
    fig.suptitle(
        f"{scale}条目评分变化：仿真前后与无干预回访",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )
    fig.text(0.29, 0.945, "仿真前后", ha="center", fontsize=12, color="#334155")
    fig.text(0.75, 0.945, "无干预回访环节", ha="center", fontsize=12, color="#334155")
    fig.text(
        0.5,
        0.012,
        "负值表示条目评分下降。点和横线为独立 outer runs 的均值及 t 分布95%置信区间；"
        "回访变化为同一 outer run 的 session_20→FU120 配对差值，回访分支不作为新增独立样本。",
        ha="center",
        fontsize=9,
        color="#475569",
    )
    fig.subplots_adjust(left=0.105, right=0.99, bottom=0.09, top=0.84, wspace=0.19)
    return save_figure(fig, output_path)


def _plot_item_entity_grid(
    frame: pd.DataFrame,
    entity_column: str,
    entities: Sequence[str],
    design_title: str,
    output_path: Path,
) -> Path:
    summaries = _summarize(frame, entity_column, "item_id", "item_label_zh")
    if not (summaries["n_outer_runs"] == 5).all():
        raise ValueError(f"{design_title} item forest requires n=5 per entity/item")
    fig = plt.figure(figsize=(max(19.0, 2.8 * len(entities) + 4.0), 18.5))
    outer = fig.add_gridspec(2, 1, height_ratios=[10, 22], hspace=0.18)
    for row_index, scale in enumerate(SCALES):
        scale_summary = _summarize(
            frame[frame["scale"] == scale], entity_column, "item_id", "item_label_zh"
        )
        items = _item_order(scale)
        limit = _symmetric_limit([scale_summary])
        inner = outer[row_index].subgridspec(1, len(entities), wspace=0.08)
        axes: list[Any] = []
        for column_index, entity in enumerate(entities):
            ax = fig.add_subplot(
                inner[0, column_index], sharey=axes[0] if axes else None
            )
            selected = scale_summary[scale_summary["entity"] == entity]
            _forest_marks(
                ax,
                selected,
                items,
                COLORS["g1"] if entity_column == "persona" else COLORS["kbd2"],
                labels=_item_tick_labels(scale, items) if column_index == 0 else None,
                x_limit=limit,
            )
            ax.set_title(str(entity), fontsize=10.5, fontweight="bold")
            ax.text(
                0.97,
                0.02,
                "n=5",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=8,
                color="#475569",
            )
            if column_index:
                ax.tick_params(labelleft=False)
            if row_index == 1:
                ax.set_xlabel("session_20 − T0")
            axes.append(ax)
    fig.suptitle(design_title, fontsize=17, fontweight="bold", y=0.985)
    fig.text(
        0.012,
        0.77,
        "PHQ-9条目",
        rotation=90,
        va="center",
        fontsize=12,
        fontweight="bold",
    )
    fig.text(
        0.012,
        0.34,
        "BDI-II条目",
        rotation=90,
        va="center",
        fontsize=12,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.008,
        "负值表示条目评分下降；所有分面使用同一量表内的共享横轴范围。点和横线为5次独立 outer runs 的均值及 t 分布95%置信区间。",
        ha="center",
        fontsize=9,
        color="#475569",
    )
    fig.subplots_adjust(left=0.15, right=0.995, bottom=0.045, top=0.95)
    return save_figure(fig, output_path)


def _plot_state_entity_grid(
    frame: pd.DataFrame,
    entity_column: str,
    entities: Sequence[str],
    design_title: str,
    output_path: Path,
) -> Path:
    combined = frame[frame["scale"] == "COMBINED_EQUAL_SCALE_WEIGHT"].copy()
    summaries = _summarize(combined, entity_column, "symptom_id", "symptom_label_zh")
    if not (summaries["n_outer_runs"] == 5).all():
        raise ValueError(f"{design_title} state forest requires n=5 per entity/state")
    state_order = list(SYMPTOM_MAP)
    state_labels = [str(SYMPTOM_MAP[state]["zh"]) for state in state_order]
    limit = _symmetric_limit([summaries])
    fig, axes = plt.subplots(
        1,
        len(entities),
        figsize=(max(19.0, 2.8 * len(entities) + 4.0), 7.2),
        sharey=True,
    )
    axes_flat = np.atleast_1d(axes).ravel()
    for index, (ax, entity) in enumerate(zip(axes_flat, entities)):
        selected = summaries[summaries["entity"] == entity]
        _forest_marks(
            ax,
            selected,
            state_order,
            COLORS["g1"] if entity_column == "persona" else COLORS["kbd2"],
            labels=state_labels if index == 0 else None,
            x_limit=limit,
        )
        ax.set_title(str(entity), fontsize=10.5, fontweight="bold")
        ax.set_xlabel("session_20 − T0")
        ax.text(
            0.97,
            0.02,
            "n=5",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
            color="#475569",
        )
        if index:
            ax.tick_params(labelleft=False)
    fig.suptitle(design_title, fontsize=17, fontweight="bold", y=0.97)
    fig.text(
        0.5,
        0.012,
        "9类生活状态先在量表内等权汇总相关条目，再对 PHQ-9 与 BDI-II 等权联合。"
        "负值表示症状减轻；风险状态单独保留。",
        ha="center",
        fontsize=9,
        color="#475569",
    )
    fig.subplots_adjust(left=0.14, right=0.995, bottom=0.12, top=0.88, wspace=0.08)
    return save_figure(fig, output_path)


def _resident_interval_summary(events_path: Path) -> pd.DataFrame:
    events = _read_csv(
        events_path,
        {
            "stable_id",
            "group",
            "timestamp",
            "turn_count",
            "target_involved",
            "doctor_target_interaction",
        },
    )
    events = events[events["group"] == "G1"].copy()
    events["timestamp"] = pd.to_datetime(events["timestamp"], errors="raise")
    events["target_involved"] = events["target_involved"].astype(bool)
    events["doctor_target_interaction"] = events["doctor_target_interaction"].astype(
        bool
    )
    doctor = events[events["target_involved"] & events["doctor_target_interaction"]]
    starts = doctor.groupby("stable_id")["timestamp"].min()
    run_ids = sorted(events["stable_id"].unique())
    if len(run_ids) != 30 or set(run_ids) != set(starts.index):
        raise ValueError(
            "G1 engagement data must contain 30 runs with a first CBT session"
        )
    resident = events[events["target_involved"] & ~events["doctor_target_interaction"]]
    run_rows: list[dict[str, object]] = []
    for run_id in run_ids:
        first_session = starts.loc[run_id]
        run_events = resident[resident["stable_id"] == run_id]
        for session in range(1, 20):
            lower = first_session + pd.Timedelta(days=3 * (session - 1))
            upper = first_session + pd.Timedelta(days=3 * session)
            turns = run_events.loc[
                (run_events["timestamp"] > lower) & (run_events["timestamp"] < upper),
                "turn_count",
            ]
            run_rows.append(
                {
                    "stable_id": run_id,
                    "group": "G1",
                    "session_start": session,
                    "session_interval": f"S{session}–S{session + 1}",
                    "resident_dialogue_turns": (
                        float(turns.sum()) if len(turns) else 0.0
                    ),
                }
            )
    run_frame = pd.DataFrame(run_rows)
    rows: list[dict[str, object]] = []
    for (session, label), values in run_frame.groupby(
        ["session_start", "session_interval"], sort=True
    ):
        center, lower, upper, n = _mean_ci(values["resident_dialogue_turns"])
        rows.append(
            {
                "group": "G1",
                "session_start": int(session),
                "session_interval": label,
                "mean_turns": center,
                "ci95_lower": lower,
                "ci95_upper": upper,
                "n_outer_runs": n,
            }
        )
    result = pd.DataFrame(rows).sort_values("session_start")
    if len(result) != 19 or not (result["n_outer_runs"] == 30).all():
        raise ValueError(
            "Resident interval summary must contain S1–S2 through S19–S20 at n=30"
        )
    return result


def _plot_engagement(
    events_path: Path,
    daily_path: Path,
    bins_path: Path,
    sessions_path: Path,
    output_dir: Path,
) -> Path:
    resident_intervals = _resident_interval_summary(events_path)
    resident_intervals.to_csv(
        output_dir / "resident_turns_between_sessions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    daily = pd.read_csv(daily_path).to_dict("records")
    bins = pd.read_csv(bins_path).to_dict("records")
    sessions = pd.read_csv(sessions_path).to_dict("records")
    paths = plot_engagement_process(
        daily,
        bins,
        output_dir,
        "0802+0808 G1六人设五次重复联合分析",
        sessions,
        resident_intervals.to_dict("records"),
    )
    if len(paths) != 1:
        raise ValueError("Expected exactly one G1 engagement figure")
    return paths[0]


def run(
    output_dir: Path,
    figures: Sequence[str],
    *,
    g1_item_change_path: Path = DEFAULT_G1_ITEM_CHANGE,
    kbd2_item_change_path: Path = DEFAULT_KBD2_ITEM_CHANGE,
    g1_state_change_path: Path = DEFAULT_G1_STATE_CHANGE,
    kbd2_state_change_path: Path = DEFAULT_KBD2_STATE_CHANGE,
    archive_items_path: Path = DEFAULT_ARCHIVE_ITEMS,
    engagement_events_path: Path = DEFAULT_ENGAGEMENT_EVENTS,
    engagement_daily_path: Path = DEFAULT_ENGAGEMENT_DAILY,
    engagement_bins_path: Path = DEFAULT_ENGAGEMENT_BINS,
    engagement_sessions_path: Path = DEFAULT_ENGAGEMENT_SESSIONS,
) -> list[Path]:
    requested = list(dict.fromkeys(figures))
    unknown = sorted(set(requested) - set(FIGURE_FILES))
    if unknown:
        raise ValueError(f"未知图表键: {', '.join(unknown)}")
    if not requested:
        raise ValueError("必须至少显式选择一个图表键")
    configure_simplified_chinese_font()
    output_dir.mkdir(parents=True, exist_ok=True)

    item_keys = {
        "symptom-forest-bdi2",
        "symptom-forest-phq9",
        "g1-item-forest",
        "kbd2-item-forest",
    }
    state_keys = {"g1-state-forest", "kbd2-state-forest"}
    g1_items = pd.DataFrame()
    kbd2_items = pd.DataFrame()
    followup_items = pd.DataFrame()
    g1_states = pd.DataFrame()
    kbd2_states = pd.DataFrame()
    if set(requested) & item_keys:
        g1_items = pd.read_csv(g1_item_change_path)
        kbd2_items = pd.read_csv(kbd2_item_change_path)
    if set(requested) & {"symptom-forest-bdi2", "symptom-forest-phq9"}:
        followup_items = _followup_item_changes(archive_items_path)
    if set(requested) & state_keys:
        g1_states = pd.read_csv(g1_state_change_path)
        kbd2_states = pd.read_csv(kbd2_state_change_path)

    renderers: dict[str, Callable[[Path], Path]] = {
        "symptom-forest-bdi2": lambda path: _plot_overall_item_forest(
            "BDI-II", g1_items, kbd2_items, followup_items, path
        ),
        "symptom-forest-phq9": lambda path: _plot_overall_item_forest(
            "PHQ-9", g1_items, kbd2_items, followup_items, path
        ),
        "g1-item-forest": lambda path: _plot_item_entity_grid(
            g1_items,
            "persona",
            PERSONAS,
            "G1跨人设实验组：各人设在各量表条目上的分数变化",
            path,
        ),
        "kbd2-item-forest": lambda path: _plot_item_entity_grid(
            kbd2_items,
            "group",
            GROUPS,
            "KBD2跨实验条件组：各实验条件在各量表条目上的分数变化",
            path,
        ),
        "g1-state-forest": lambda path: _plot_state_entity_grid(
            g1_states,
            "persona",
            PERSONAS,
            "G1跨人设实验组：各人设的9类生活状态变化",
            path,
        ),
        "kbd2-state-forest": lambda path: _plot_state_entity_grid(
            kbd2_states,
            "group",
            GROUPS,
            "KBD2跨实验条件组：各实验条件的9类生活状态变化",
            path,
        ),
        "engagement-g1": lambda path: _plot_engagement(
            engagement_events_path,
            engagement_daily_path,
            engagement_bins_path,
            engagement_sessions_path,
            path.parent,
        ),
    }
    outputs = [
        renderers[figure](output_dir / FIGURE_FILES[figure]) for figure in requested
    ]
    manifest = {
        "schema_version": "targeted_redraw_second_batch_v1",
        "font_family": plt.rcParams["font.sans-serif"][0],
        "analysis_unit": "independent outer simulation run",
        "selected_figures": requested,
        "png_files": [path.name for path in outputs],
        "followup_outer_runs": {"G1_cross_persona": 30, "KBD2_cross_condition": 40},
    }
    (output_dir / "redraw_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return outputs


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--figure",
        action="append",
        choices=sorted(FIGURE_FILES),
        required=True,
        help="可重复传入；只渲染显式选择的图。",
    )
    args = parser.parse_args(argv)
    outputs = run(args.out_dir, args.figure)
    print(f"已生成 {len(outputs)} 张第二批指定图：")
    for output in outputs:
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
