"""Paper-oriented outcome, reliability, safety-proxy, and process figures."""

from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np

from ..schema import ExperimentRecord, group_color, group_sort_key, kbd_color, label_display, slugify
from ..statistics import (
    baseline_adjusted_endpoint_contrasts,
    cross_scale_convergence,
    group_endpoint_contrasts,
    group_time_contrasts,
    measurement_icc,
    trajectory_metrics,
)
from .scale_plots import plt, save_figure


def plot_group_time_contrasts(
    records: list[ExperimentRecord],
    labels: list[str],
    scales: list[str],
    out_dir: Path,
    title_prefix: str,
) -> Path | None:
    rows = group_time_contrasts(records, labels, scales)
    contrasts = sorted({row["contrast"] for row in rows})
    if not contrasts:
        return None
    fig, axes = plt.subplots(
        len(scales),
        len(contrasts),
        figsize=(max(7.5, 5.3 * len(contrasts)), max(4.2, 3.8 * len(scales))),
        squeeze=False,
        constrained_layout=True,
    )
    plotted = False
    for scale_index, scale in enumerate(scales):
        for contrast_index, contrast in enumerate(contrasts):
            ax = axes[scale_index][contrast_index]
            panel = [row for row in rows if row["scale"] == scale and row["contrast"] == contrast]
            x_values, centers, lower, upper = [], [], [], []
            for label_index, label in enumerate(labels):
                row = next((value for value in panel if value["timepoint"] == label), None)
                if row is None:
                    continue
                interval = row["difference_in_change"]
                if interval.get("estimate") is None:
                    continue
                plotted = True
                x_values.append(label_index)
                centers.append(interval["estimate"])
                lower.append(interval.get("lower"))
                upper.append(interval.get("upper"))
            if centers:
                ax.plot(x_values, centers, marker="o", color="#334155", linewidth=1.8)
                if all(value is not None for value in lower + upper):
                    ax.fill_between(x_values, lower, upper, color="#64748b", alpha=0.18)
            ax.axhline(0, color="#111827", linewidth=0.8)
            ax.set_xticks(range(len(labels)), [label_display(label) for label in labels], rotation=25)
            ax.set_ylabel("Difference in change")
            ax.set_title(f"{scale}: {contrast}")
            ax.grid(alpha=0.2)
    if not plotted:
        plt.close(fig)
        return None
    fig.suptitle(f"{title_prefix}: outer-run baseline-referenced time contrasts")
    fig.text(
        0.5,
        -0.018,
        "Welch 95% CI across outer simulation runs at each observed timepoint. "
        "This is a descriptive group×time analogue, not a fitted confirmatory mixed model.",
        ha="center",
        fontsize=8,
        color="#475569",
    )
    return save_figure(fig, out_dir / "core_00_group_time_contrasts.png")


def plot_outer_contrast_forest(
    records: list[ExperimentRecord],
    labels: list[str],
    scales: list[str],
    out_dir: Path,
    title_prefix: str,
) -> Path | None:
    adjusted_rows = baseline_adjusted_endpoint_contrasts(records, labels, scales)
    change_rows = group_endpoint_contrasts(records, labels, scales)
    rows = []
    for adjusted in adjusted_rows:
        if adjusted.get("estimate") is None:
            continue
        change = next(
            (
                row
                for row in change_rows
                if row["scale"] == adjusted["scale"] and row["contrast"] == adjusted["contrast"]
            ),
            None,
        )
        rows.append((adjusted, change))
    if not rows:
        return None
    fig, ax = plt.subplots(figsize=(8.6, max(3.8, 0.75 * len(rows) + 2.2)), constrained_layout=True)
    y_positions = np.arange(len(rows))
    for y, (row, change) in zip(y_positions, rows):
        estimate = row["estimate"]
        lower, upper = row.get("lower"), row.get("upper")
        error = None
        if lower is not None and upper is not None:
            error = np.asarray([[estimate - lower], [upper - estimate]])
        ax.errorbar(
            estimate,
            y,
            xerr=error,
            fmt="o",
            color=group_color(row["first_group"]),
            capsize=4,
            markersize=7,
        )
        effect = change["hedges_g"].get("estimate") if change else None
        effect_text = "NA" if effect is None or not math.isfinite(effect) else f"{effect:.2f}"
        ax.annotate(f"g={effect_text}", (estimate, y), xytext=(7, 6), textcoords="offset points", fontsize=8)
    ax.axvline(0, color="#111827", linewidth=0.9)
    ax.set_yticks(y_positions, [f"{row['scale']}: {row['contrast']}" for row, _ in rows])
    ax.invert_yaxis()
    ax.set_xlabel("Baseline-adjusted observed endpoint difference (first − second)")
    ax.set_title(f"{title_prefix}: exploratory baseline-adjusted endpoint contrasts")
    ax.grid(axis="x", alpha=0.22)
    fig.text(
        0.5,
        -0.02,
        "Pairwise ANCOVA 95% CI: endpoint ~ group + baseline. Hedges' g annotation uses unadjusted outer-run change SD. "
        "Exploratory small-n comparison; negative adjusted difference favors the first group for symptom scores.",
        ha="center",
        fontsize=8,
        color="#475569",
    )
    return save_figure(fig, out_dir / "core_01_outer_run_contrast_forest.png")


def plot_cross_scale_convergence(
    records: list[ExperimentRecord],
    labels: list[str],
    out_dir: Path,
    title_prefix: str,
) -> Path | None:
    result = cross_scale_convergence(records, labels)
    rows = result["rows"]
    if not rows:
        return None
    fig, ax = plt.subplots(figsize=(7.4, 6.2), constrained_layout=True)
    for row in rows:
        ax.scatter(
            row["first_normalized_delta"],
            row["second_normalized_delta"],
            color=group_color(row["group"]),
            s=52,
            alpha=0.85,
        )
        ax.annotate(
            f"{row['group']}-{row['repeat_id']}",
            (row["first_normalized_delta"], row["second_normalized_delta"]),
            xytext=(5, 4),
            textcoords="offset points",
            fontsize=7,
        )
    ax.axhline(0, color="#64748b", linewidth=0.8)
    ax.axvline(0, color="#64748b", linewidth=0.8)
    ax.set_xlabel("PHQ-9 endpoint change / 27")
    ax.set_ylabel("BDI-II endpoint change / 63")
    rho = result["spearman_rho"]
    agreement = result["direction_agreement_rate"]
    subtitle = (
        f"direction agreement={agreement:.1%}; Spearman ρ={rho:.2f}"
        if agreement is not None and rho is not None
        else f"direction agreement={agreement:.1%}" if agreement is not None else "insufficient paired scales"
    )
    ax.set_title(f"{title_prefix}: PHQ-9 / BDI-II convergence\n{subtitle}")
    ax.grid(alpha=0.18)
    fig.text(
        0.5,
        -0.015,
        "Each point is one outer simulation run. Normalization only aligns plotting scales; "
        "it does not create a combined depression index.",
        ha="center",
        fontsize=8,
        color="#475569",
    )
    return save_figure(fig, out_dir / "core_02_phq_bdi_convergence.png")


def plot_measurement_icc(
    records: list[ExperimentRecord],
    labels: list[str],
    scales: list[str],
    out_dir: Path,
    title_prefix: str,
) -> Path | None:
    rows = [(scale, measurement_icc(records, labels, scale)) for scale in scales]
    rows = [(scale, result) for scale, result in rows if result["status"] == "ok"]
    if not rows:
        return None
    fig, ax = plt.subplots(figsize=(8.2, max(3.8, len(rows) * 1.2 + 2.0)), constrained_layout=True)
    y = np.arange(len(rows))
    width = 0.32
    ax.barh(y - width / 2, [row[1]["icc_1_1"] for row in rows], height=width, label="ICC(1,1)")
    ax.barh(y + width / 2, [row[1]["icc_1_k"] for row in rows], height=width, label="ICC(1,K)")
    ax.set_yticks(y, [f"{scale} (targets={result['n_snapshot_targets']}, K={result['k_measurement']})" for scale, result in rows])
    ax.set_xlim(min(-0.2, min(result["icc_1_1"] for _, result in rows) - 0.05), 1.02)
    ax.axvline(0, color="#64748b", linewidth=0.8)
    ax.set_xlabel("Intraclass correlation")
    ax.set_title(f"{title_prefix}: frozen-snapshot measurement reliability")
    ax.legend(frameon=False)
    ax.grid(axis="x", alpha=0.2)
    fig.text(
        0.5,
        -0.015,
        "ICC is pooled across frozen snapshot targets. It measures repeated scale-generation stability, "
        "not outer-run reproducibility or clinical test-retest reliability.",
        ha="center",
        fontsize=8,
        color="#475569",
    )
    return save_figure(fig, out_dir / "core_03_measurement_icc.png")


def plot_endpoint_waterfall(
    records: list[ExperimentRecord],
    labels: list[str],
    scales: list[str],
    out_dir: Path,
    title_prefix: str,
) -> list[Path]:
    paths: list[Path] = []
    include_kbd = len({record.kbd for record in records}) > 1
    for scale in scales:
        rows = []
        for record in records:
            change = trajectory_metrics(record, labels, scale).get("endpoint_change")
            if change is not None:
                rows.append((record, float(change)))
        if not rows:
            continue
        rows.sort(key=lambda item: item[1])
        fig, ax = plt.subplots(figsize=(max(7.4, len(rows) * 0.65), 4.8), constrained_layout=True)
        bars = ax.bar(
            range(len(rows)),
            [value for _, value in rows],
            color=[group_color(record.group) for record, _ in rows],
            alpha=0.85,
        )
        ax.axhline(0, color="#111827", linewidth=0.9)
        if scale == "PHQ-9":
            ax.axhline(5, color="#dc2626", linewidth=0.9, linestyle="--", label="worsening proxy +5")
            ax.axhline(-5, color="#2563eb", linewidth=0.9, linestyle="--", label="improvement proxy −5")
            ax.legend(frameon=False, fontsize=8)
        ax.set_xticks(
            range(len(rows)),
            [
                "-".join(
                    part
                    for part in (
                        record.kbd if include_kbd else None,
                        record.group,
                        record.repeat_id,
                    )
                    if part
                )
                for record, _ in rows
            ],
            rotation=35,
            ha="right",
        )
        ax.set_ylabel("Observed endpoint − baseline")
        ax.set_title(f"{title_prefix}: {scale} outer-run waterfall")
        ax.grid(axis="y", alpha=0.2)
        for bar, (_, value) in zip(bars, rows):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value,
                f"{value:.1f}",
                ha="center",
                va="bottom" if value >= 0 else "top",
                fontsize=8,
            )
        fig.text(
            0.5,
            -0.02,
            "Threshold lines are transferred human-scale conventions used only as Agent simulation proxies. "
            "They do not indicate clinical response, remission, or harm.",
            ha="center",
            fontsize=8,
            color="#475569",
        )
        paths.append(save_figure(fig, out_dir / f"core_04_{slugify(scale)}_endpoint_waterfall.png"))
    return paths


def plot_process_delivery(
    process_rows: list[dict[str, Any]],
    stage_rows: list[dict[str, Any]],
    out_dir: Path,
    title_prefix: str,
) -> list[Path]:
    available = [row for row in process_rows if row.get("raw_artifacts_available")]
    if not available:
        return []
    paths: list[Path] = []
    include_kbd = len({row["kbd"] for row in available}) > 1
    available.sort(key=lambda row: (group_sort_key(row["kbd"]), group_sort_key(row["group"]), row["outer_run_id"]))
    labels = [
        "-".join(
            part
            for part in (
                row["kbd"] if include_kbd else None,
                row["group"],
                row["outer_run_id"],
            )
            if part
        )
        for row in available
    ]
    x = np.arange(len(available))

    fig, ax = plt.subplots(figsize=(max(8.0, len(available) * 0.75), 5.0), constrained_layout=True)
    prompt = [row.get("cbt_prompt_completion_meeting") or 0 for row in available]
    fixed = [row.get("fixed_prompt_post_completion_meetings") or 0 for row in available]
    ax.bar(x, prompt, color="#2563eb", label="CBT session-prompt phase")
    ax.bar(x, fixed, bottom=prompt, color="#f59e0b", label="fixed follow-up-prompt consultations")
    ax.set_xticks(x, labels, rotation=35, ha="right")
    ax.set_ylabel("Consultation meetings")
    ax.set_title(f"{title_prefix}: CBT prompt completion and continued consultations")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)
    fig.text(
        0.5,
        -0.02,
        "Orange meetings still use a fixed follow-up prompt and forced doctor consultation. "
        "They are not a no-intervention delayed follow-up; true_followup=false.",
        ha="center",
        fontsize=8,
        color="#475569",
    )
    paths.append(save_figure(fig, out_dir / "core_05_cbt_prompt_and_fixed_followup_meetings.png"))

    metrics = [
        ("total_turns", "Total dialogue turns"),
        ("total_chars", "Total dialogue characters"),
        ("mean_turns_per_meeting", "Mean turns / meeting"),
        ("environment_tasks", "Environment tasks"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.0), squeeze=False, constrained_layout=True)
    compare_personas = len({row["group"] for row in available}) == 1 and include_kbd
    category_field = "kbd" if compare_personas else "group"
    for ax, (field, title) in zip(axes.flat, metrics):
        grouped: dict[str, list[float]] = defaultdict(list)
        for row in available:
            value = row.get(field)
            if value is not None:
                grouped[row[category_field]].append(float(value))
        categories = sorted(grouped, key=group_sort_key)
        for index, category in enumerate(categories):
            values = grouped[category]
            jitter = np.linspace(-0.08, 0.08, len(values)) if len(values) > 1 else [0.0]
            ax.scatter(
                np.asarray([index] * len(values)) + jitter,
                values,
                color=kbd_color(category) if compare_personas else group_color(category),
                alpha=0.75,
            )
            ax.scatter(index, mean(values), color="#111827", marker="D", s=38, zorder=3)
        ax.set_xticks(range(len(categories)), categories)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle(f"{title_prefix}: intervention dose and delivery")
    fig.text(
        0.5,
        -0.015,
        "Points are outer simulation runs; diamonds are category means. Character counts are exposure proxies, not tokens.",
        ha="center",
        fontsize=8,
        color="#475569",
    )
    paths.append(save_figure(fig, out_dir / "core_06_intervention_dose_delivery.png"))

    if stage_rows:
        stage_names: list[str] = []
        by_run: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in stage_rows:
            name = str(row.get("legacy_session") or row.get("macro_stage") or "unknown")
            if name not in stage_names:
                stage_names.append(name)
            by_run[row["stable_id"]].append(row)
        run_ids = [row["stable_id"] for row in available if row["stable_id"] in by_run]
        if run_ids and stage_names:
            matrix = np.zeros((len(run_ids), len(stage_names)), dtype=float)
            for row_index, stable_id in enumerate(run_ids):
                for entry in by_run[stable_id]:
                    name = str(entry.get("legacy_session") or entry.get("macro_stage") or "unknown")
                    matrix[row_index, stage_names.index(name)] += 1
            fig, ax = plt.subplots(
                figsize=(max(9.0, len(stage_names) * 0.85), max(4.0, len(run_ids) * 0.48 + 2.0)),
                constrained_layout=True,
            )
            image = ax.imshow(matrix, cmap="Blues", aspect="auto", vmin=0)
            run_label_map = {
                row["stable_id"]: "-".join(
                    part
                    for part in (
                        row["kbd"] if include_kbd else None,
                        row["group"],
                        row["outer_run_id"],
                    )
                    if part
                )
                for row in available
            }
            ax.set_xticks(range(len(stage_names)), [label_display(name) for name in stage_names], rotation=35, ha="right")
            ax.set_yticks(range(len(run_ids)), [run_label_map[value] for value in run_ids])
            ax.set_title(f"{title_prefix}: meetings spent in each CBT prompt")
            for row_index in range(matrix.shape[0]):
                for column_index in range(matrix.shape[1]):
                    if matrix[row_index, column_index]:
                        ax.text(column_index, row_index, f"{matrix[row_index, column_index]:.0f}", ha="center", va="center", fontsize=8)
            fig.colorbar(image, ax=ax, label="meetings")
            paths.append(save_figure(fig, out_dir / "core_07_cbt_stage_dwell_heatmap.png"))
    return paths


def render_core_figures(
    records: list[ExperimentRecord],
    labels: list[str],
    scales: list[str],
    out_dir: Path,
    title_prefix: str,
    *,
    process_rows: list[dict[str, Any]] | None = None,
    stage_rows: list[dict[str, Any]] | None = None,
) -> list[Path]:
    paths: list[Path] = []
    for path in (
        plot_group_time_contrasts(records, labels, scales, out_dir, title_prefix),
        plot_outer_contrast_forest(records, labels, scales, out_dir, title_prefix),
        plot_cross_scale_convergence(records, labels, out_dir, title_prefix),
        plot_measurement_icc(records, labels, scales, out_dir, title_prefix),
    ):
        if path is not None:
            paths.append(path)
    paths.extend(plot_endpoint_waterfall(records, labels, scales, out_dir, title_prefix))
    paths.extend(plot_process_delivery(process_rows or [], stage_rows or [], out_dir, title_prefix))
    return paths
