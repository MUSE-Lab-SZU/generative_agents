"""CLI for paper-oriented cross-persona analyses and figures."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .cross_persona import (
    group_time_persona_summary,
    hierarchical_variance_partition,
    leave_one_persona_out,
    persona_group_contrasts,
    persona_group_summary,
)
from .loader import find_report_files, load_records
from .process import extract_process_metrics
from .schema import ExperimentRecord, group_sort_key, normalize_group, normalize_kbd
from .visualization.cross_persona_plots import (
    plot_leave_one_persona_out,
    plot_outcome_heatmap,
    plot_persona_contrast_forest,
    plot_persona_scale_trajectories,
    plot_process_heatmap,
    plot_variance_partition,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAJECTORY_DIRS = [
    PROJECT_ROOT / "results" / "experiment_data-0727-KBD2" / "experiment_data" / "reports",
    PROJECT_ROOT / "results" / "experiment_data-0727-KBD4" / "experiment_data" / "reports",
    PROJECT_ROOT / "results" / "experiment_data-0727-KBD6" / "experiment_data" / "reports",
]
DEFAULT_HETEROGENEITY_DIR = PROJECT_ROOT / "results" / "experiment_data" / "reports" / "0718"
DEFAULT_PROCESS_DATA_ROOT = PROJECT_ROOT / "results" / "experiment_data" / "0718"
DEFAULT_PROCESS_CHECKPOINTS_ROOT = PROJECT_ROOT / "results" / "checkpoints" / "0718"
DEFAULT_OUT_DIR = (
    PROJECT_ROOT / "docs" / "experiment_evaluation_0727" / "cross_persona_paper_figures"
)


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _display(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> Path:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or ["status"])
        writer.writeheader()
        writer.writerows(rows)
    return path


def _discover(directories: list[Path]) -> list[Path]:
    paths: list[Path] = []
    for directory in directories:
        if not directory.is_dir():
            raise SystemExit(f"Report directory not found: {directory}")
        paths.extend(find_report_files(directory, False))
    return sorted(set(paths))


def _inventory_rows(
    records: list[ExperimentRecord],
    labels: list[str],
    scales: list[str],
    dataset: str,
) -> list[dict[str, Any]]:
    counts = Counter((record.kbd, record.group) for record in records)
    rows: list[dict[str, Any]] = []
    for (persona, group), count in sorted(
        counts.items(), key=lambda item: (group_sort_key(item[0][0]), group_sort_key(item[0][1]))
    ):
        cell = [record for record in records if record.kbd == persona and record.group == group]
        rows.append(
            {
                "dataset": dataset,
                "persona": persona,
                "group": group,
                "n_outer_runs": count,
                "outer_run_ids": ",".join(
                    sorted({record.repeat_id for record in cell}, key=group_sort_key)
                ),
                "timepoints": ",".join(labels),
                "scales": ",".join(scales),
                "k_measurement": ",".join(
                    map(str, sorted({record.expected_repeats for record in cell}))
                ),
                "analysis_unit": "independent outer simulation run",
            }
        )
    return rows


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate cross-persona paper figures from existing experiment_eval report summaries."
    )
    parser.add_argument(
        "--trajectory-reports-dir",
        type=Path,
        action="append",
        default=None,
        help="0727-style report directory; repeat for multiple persona archives.",
    )
    parser.add_argument(
        "--heterogeneity-reports-dir",
        type=Path,
        default=DEFAULT_HETEROGENEITY_DIR,
        help="0718-style report directory containing the G1/G9 and persona×group matrix.",
    )
    parser.add_argument(
        "--process-experiment-data-root",
        type=Path,
        default=DEFAULT_PROCESS_DATA_ROOT,
    )
    parser.add_argument(
        "--process-checkpoints-root",
        type=Path,
        default=DEFAULT_PROCESS_CHECKPOINTS_ROOT,
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--persona", action="append", default=None)
    parser.add_argument("--trajectory-group", action="append", default=None)
    parser.add_argument("--first-group", default="G1")
    parser.add_argument("--second-group", default="G9")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    trajectory_dirs = [
        _resolve(path) for path in (args.trajectory_reports_dir or DEFAULT_TRAJECTORY_DIRS)
    ]
    heterogeneity_dir = _resolve(args.heterogeneity_reports_dir)
    out_dir = _resolve(args.out_dir)
    personas = [
        normalize_kbd(value) for value in (args.persona or ["KBD2", "KBD4", "KBD6"])
    ]
    trajectory_groups = [
        normalize_group(value) for value in (args.trajectory_group or ["G1", "G4"])
    ]
    first_group = normalize_group(args.first_group)
    second_group = normalize_group(args.second_group)

    trajectory_files = _discover(trajectory_dirs)
    heterogeneity_files = _discover([heterogeneity_dir])
    trajectory_records, trajectory_labels, trajectory_scales = load_records(trajectory_files)
    heterogeneity_records, heterogeneity_labels, heterogeneity_scales = load_records(
        heterogeneity_files
    )
    trajectory_records = [
        record
        for record in trajectory_records
        if record.kbd in personas and record.group in trajectory_groups
    ]
    heterogeneity_records = [
        record for record in heterogeneity_records if record.kbd in personas
    ]
    if not trajectory_records:
        raise SystemExit("No trajectory records remain after persona/group filtering.")
    if not heterogeneity_records:
        raise SystemExit("No heterogeneity records remain after persona filtering.")
    if {first_group, second_group} - {record.group for record in heterogeneity_records}:
        raise SystemExit(
            f"Contrast groups not both present: requested {first_group}, {second_group}; "
            f"available {sorted({record.group for record in heterogeneity_records}, key=group_sort_key)}"
        )

    process_rows, _ = extract_process_metrics(
        heterogeneity_records,
        experiment_data_root=_resolve(args.process_experiment_data_root),
        checkpoints_root=_resolve(args.process_checkpoints_root),
        project_root=PROJECT_ROOT,
    )
    time_summary = group_time_persona_summary(
        trajectory_records, trajectory_labels, trajectory_scales
    )
    cell_summary = persona_group_summary(
        heterogeneity_records,
        heterogeneity_labels,
        heterogeneity_scales,
        process_rows,
    )
    contrast_rows = persona_group_contrasts(
        heterogeneity_records,
        heterogeneity_labels,
        heterogeneity_scales,
        first_group=first_group,
        second_group=second_group,
    )
    lopo_rows = leave_one_persona_out(
        heterogeneity_records,
        heterogeneity_labels,
        heterogeneity_scales,
        first_group=first_group,
        second_group=second_group,
    )
    variance_rows = hierarchical_variance_partition(
        trajectory_records, trajectory_labels, trajectory_scales
    )
    inventory_rows = [
        *_inventory_rows(
            trajectory_records, trajectory_labels, trajectory_scales, "0727 trajectory"
        ),
        *_inventory_rows(
            heterogeneity_records,
            heterogeneity_labels,
            heterogeneity_scales,
            "0718 heterogeneity",
        ),
    ]

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_paths = [
        _write_csv(out_dir / "data_inventory.csv", inventory_rows),
        _write_csv(out_dir / "group_time_persona_summary.csv", time_summary),
        _write_csv(out_dir / "persona_group_summary.csv", cell_summary),
        _write_csv(out_dir / "persona_g1_vs_g9_contrasts.csv", contrast_rows),
        _write_csv(out_dir / "leave_one_persona_out.csv", lopo_rows),
        _write_csv(out_dir / "variance_decomposition.csv", variance_rows),
    ]
    chart_paths = [
        plot_persona_scale_trajectories(
            trajectory_records, trajectory_labels, trajectory_scales, out_dir
        ),
        plot_persona_contrast_forest(
            contrast_rows,
            out_dir,
            first_group=first_group,
            second_group=second_group,
        ),
        plot_outcome_heatmap(cell_summary, out_dir),
        plot_process_heatmap(cell_summary, out_dir),
        plot_leave_one_persona_out(
            lopo_rows,
            out_dir,
            first_group=first_group,
            second_group=second_group,
        ),
        plot_variance_partition(variance_rows, out_dir),
    ]
    manifest = {
        "trajectory_dataset": {
            "report_directories": [_display(path) for path in trajectory_dirs],
            "report_count": len(trajectory_files),
            "record_count": len(trajectory_records),
            "personas": personas,
            "groups": trajectory_groups,
            "timepoints": trajectory_labels,
            "scales": trajectory_scales,
        },
        "heterogeneity_dataset": {
            "report_directory": _display(heterogeneity_dir),
            "report_count": len(heterogeneity_files),
            "record_count": len(heterogeneity_records),
            "groups": sorted(
                {record.group for record in heterogeneity_records}, key=group_sort_key
            ),
            "timepoints": heterogeneity_labels,
            "scales": heterogeneity_scales,
        },
        "contrast": f"{first_group} - {second_group}",
        "analysis_scope": (
            "in-silico exploratory cross-persona analysis; K repeats are measurement repeats, "
            "not independent outer runs"
        ),
        "process_experiment_data_root": _display(_resolve(args.process_experiment_data_root)),
        "process_checkpoints_root": _display(_resolve(args.process_checkpoints_root)),
        "csv_files": [_display(path) for path in csv_paths],
        "charts": [{"png": _display(path)} for path in chart_paths],
    }
    manifest_path = out_dir / "analysis_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    index_lines = [
        "# 跨人设论文图示例",
        "",
        "- 0727：KBD2/KBD4/KBD6 × G1/G4；同协议轨迹和描述性方差分解。",
        "- 0718：KBD2/KBD4/KBD6 × G1/G3/G4/G5/G6/G7/G9；G1−G9 forest、热图和 LOPO。",
        "- 所有区间的统计单位均为 outer run；K=10 只用于冻结快照均值和测量噪声。",
        "- 0718 每个人设×组只有 2 个 outer run，相关结果全部是探索性。",
        "",
        "- [分析方案](../../../cross_persona_analysis_plan.md)",
        "- [逐图结果说明](../../../cross_persona_results_notes.md)",
        "",
        "## Figures",
        "",
    ]
    for path in chart_paths:
        index_lines.extend([f"### {path.stem}", "", f"![{path.stem}]({path.name})", ""])
    index_lines.extend(
        [
            "## Tables",
            "",
            *[f"- [`{path.name}`]({path.name})" for path in csv_paths],
            f"- [`{manifest_path.name}`]({manifest_path.name})",
            "",
        ]
    )
    (out_dir / "README.md").write_text("\n".join(index_lines), encoding="utf-8")

    print(f"Loaded {len(trajectory_records)} trajectory outer-run records from 0727 archives")
    print(f"Loaded {len(heterogeneity_records)} heterogeneity outer-run records from 0718")
    print(f"Wrote {len(chart_paths)} PNG figures and {len(csv_paths)} CSV files to {_display(out_dir)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
