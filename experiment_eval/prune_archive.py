"""Safely curate analysis-ready artifacts from experiment archives.

The tool is intentionally coupled to :mod:`experiment_eval`: report loading and
process-metric extraction are executed against the staged copy before any
source directory is replaced or any backup is published.  The default mode is
read-only; ``--apply`` is required for either prune or non-destructive backup.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from .loader import find_report_files, load_records, resolve_followup_source_summary
from .process import extract_process_metrics


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# 用户默认配置区
#
# 配好这里以后，可以直接运行：
#     python -m experiment_eval.prune_archive
#
# 两种存档定位方式二选一：
# 1. 合并存档：填写 RESULTS_ARCHIVE，存档内须有 checkpoints/ 和 experiment_data/。
# 2. 分离存档：RESULTS_ARCHIVE 留空，同时填写 CHECKPOINTS_ARCHIVE 和
#    EXPERIMENT_DATA_ARCHIVE。
#
# 相对名称会从 RESULTS_ROOT 下解析，也可以填写绝对路径。命令行参数仍可临时
# 覆盖这里的值。OPERATION_MODE="prune" 会用精简副本替换原存档；"backup"
# 会保持源存档不变，把必要内容复制到 BACKUP_OUTPUT_DIR。APPLY_CHANGES 默认必须
# 保持 False；先检查 dry-run 输出，确认 report/run 数量、救援目录和空间估算后，
# 再手动改为 True 或传入 --apply。
# ---------------------------------------------------------------------------
RESULTS_ROOT: str | Path = PROJECT_ROOT / "results"
RESULTS_ARCHIVE = ""  # 例："0630/results-batch-0628-KBD2-G1-SEV/results"
CHECKPOINTS_ARCHIVE = ""  # 例："checkpoints-0727-KBD4"
EXPERIMENT_DATA_ARCHIVE = ""  # 例："experiment_data-0727-KBD4"
OPERATION_MODE = "backup"  # "prune"=替换并清理源目录；"backup"=复制必要内容且不改源目录
BACKUP_OUTPUT_DIR: str | Path = "draw_data"  # backup 模式必填；相对路径从 RESULTS_ROOT 解析
BACKUP_COMPLETED_ONLY = False  # 仅 backup：只保留已有 complete repeat summary 的画图输入
BACKUP_INCLUDE_RUN_PREFIXES: tuple[str, ...] = ()  # 例：("followup-",)
INCLUDE_RAW_SCALE_TRACES = False  # trace/job/snapshot 副本通常占绝大多数空间
INCLUDE_MERGED_CONSULTATION_DIALOGUES = True
INCLUDE_DYNAMIC_LLM_TRACES = True
APPLY_CHANGES = False  # False=只读计划；True=校验通过后执行所选模式
KEEP_BACKUP = False  # 仅 prune：True=替换后保留旧完整目录，因此暂时不释放其空间
STAGING_PARENT: str | Path = ""  # 留空使用系统临时目录；也可填写空间充足的目录

RAW_EVALUATION_PATTERNS = ("*_answered.jsonl", "*_scored.json", "*_trace.json")
SMALL_EVALUATION_METADATA = {
    "metadata.json",
    "snapshot_manifest.json",
    "worker_result.json",
    "index.json",
    "state.json",
}
CURATED_RUN_FILES = {
    "cbt_condition_manifest.json",
    "followup_manifest.json",
    "trial_meta.json",
    "conversation.json",
    "simulation_events.jsonl",
}
PATH_ONLY_PROCESS_FIELDS = {"judge_trace_path", "final_checkpoint_path"}


@dataclass(frozen=True)
class ArchiveLayout:
    """Physical archive roots and their normalized data containers."""

    mode: str
    results_root: Path
    checkpoint_target: Path
    checkpoint_container: Path
    experiment_target: Path
    experiment_container: Path

    @property
    def reports_dir(self) -> Path:
        return self.experiment_container / "reports"


@dataclass(frozen=True)
class CopyItem:
    source: Path
    destination_root: str
    destination_relative: Path
    reason: str


@dataclass
class PrunePlan:
    layout: ArchiveLayout
    complete_report_paths: list[Path]
    complete_run_names: list[str]
    excluded_complete_run_names: list[str]
    incomplete_report_paths: list[Path]
    incomplete_run_names: list[str]
    superseded_incomplete_run_names: list[str]
    unreported_run_names: list[str]
    followup_run_names: list[str]
    report_only_run_names: list[str]
    items: list[CopyItem]
    source_bytes: int
    retained_bytes: int
    raw_evaluation_file_count: int
    completed_only: bool
    include_run_prefixes: tuple[str, ...]
    include_raw_scale_traces: bool
    include_merged_consultation_dialogues: bool
    include_dynamic_llm_traces: bool

    @property
    def estimated_reclaimed_bytes(self) -> int:
        return max(0, self.source_bytes - self.retained_bytes)


def _resolve_path(raw: str | Path, results_root: Path) -> Path:
    candidate = Path(raw).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (results_root / candidate).resolve()


def _resolve_new_path(raw: str | Path, results_root: Path) -> Path:
    """Resolve a not-yet-created output path consistently below results_root."""

    candidate = Path(raw).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (results_root / candidate).resolve()


def _container(target: Path, child: str) -> Path:
    nested = target / child
    return nested if nested.is_dir() else target


def resolve_layout(
    *,
    results_root: Path,
    results_archive: str | None,
    checkpoints_archive: str | None,
    experiment_data_archive: str | None,
) -> ArchiveLayout:
    results_root = results_root.expanduser().resolve()
    if results_archive:
        if checkpoints_archive or experiment_data_archive:
            raise ValueError(
                "--results-archive cannot be combined with "
                "--checkpoints-archive/--experiment-data-archive"
            )
        archive = _resolve_path(results_archive, results_root)
        checkpoint_target = archive / "checkpoints"
        experiment_target = archive / "experiment_data"
        mode = "combined"
    else:
        if not checkpoints_archive or not experiment_data_archive:
            raise ValueError(
                "Use either --results-archive, or provide both "
                "--checkpoints-archive and --experiment-data-archive"
            )
        checkpoint_target = _resolve_path(checkpoints_archive, results_root)
        experiment_target = _resolve_path(experiment_data_archive, results_root)
        mode = "split"

    checkpoint_target = checkpoint_target.resolve()
    experiment_target = experiment_target.resolve()
    if checkpoint_target == experiment_target:
        raise ValueError("Checkpoint and experiment-data targets must be different")
    if (
        checkpoint_target in experiment_target.parents
        or experiment_target in checkpoint_target.parents
    ):
        raise ValueError("Checkpoint and experiment-data targets must not overlap")
    for label, target in (
        ("checkpoint archive", checkpoint_target),
        ("experiment-data archive", experiment_target),
    ):
        if not target.is_dir():
            raise FileNotFoundError(f"{label} not found: {target}")
        if target in {Path("/"), Path.home().resolve(), PROJECT_ROOT.resolve(), results_root}:
            raise ValueError(f"Refusing broad destructive target for {label}: {target}")

    checkpoint_container = _container(checkpoint_target, "checkpoints")
    experiment_container = _container(experiment_target, "experiment_data")
    reports_dir = experiment_container / "reports"
    if not reports_dir.is_dir():
        raise FileNotFoundError(f"Repeat-report directory not found: {reports_dir}")
    if not any(checkpoint_container.glob("*/simulate-*.json")):
        raise FileNotFoundError(f"No run checkpoints found below: {checkpoint_container}")

    return ArchiveLayout(
        mode=mode,
        results_root=results_root,
        checkpoint_target=checkpoint_target,
        checkpoint_container=checkpoint_container,
        experiment_target=experiment_target,
        experiment_container=experiment_container,
    )


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid JSON: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _incomplete_runs(paths: Iterable[Path]) -> list[str]:
    run_names: set[str] = set()
    for path in paths:
        payload = _read_json(path)
        completion = payload.get("completion") or {}
        if completion.get("ready_for_final_report") is True:
            raise ValueError(f"File is named incomplete but declares final readiness: {path}")
        for condition in payload.get("conditions") or []:
            if not isinstance(condition, dict):
                continue
            run_name = str(condition.get("run_name") or "").strip()
            if run_name:
                run_names.add(run_name)
    return sorted(run_names)


def _tree_size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(value.stat().st_size for value in path.rglob("*") if value.is_file())


class _PlanBuilder:
    def __init__(self, layout: ArchiveLayout) -> None:
        self.layout = layout
        self._items: dict[tuple[str, str], CopyItem] = {}

    def _relative_destination(self, source: Path, source_target: Path) -> Path:
        try:
            return source.relative_to(source_target)
        except ValueError as exc:
            raise ValueError(f"Preserved file is outside archive target: {source}") from exc

    def add_file(
        self,
        source: Path,
        *,
        destination_root: str,
        source_target: Path,
        reason: str,
    ) -> None:
        if not source.is_file():
            raise FileNotFoundError(f"Required {reason} file not found: {source}")
        relative = self._relative_destination(source, source_target)
        key = (destination_root, relative.as_posix())
        existing = self._items.get(key)
        item = CopyItem(source, destination_root, relative, reason)
        if existing is not None and existing.source.resolve() != source.resolve():
            raise ValueError(f"Two source files map to the same retained path: {existing.source}, {source}")
        self._items[key] = item

    def add_tree(
        self,
        source: Path,
        *,
        destination_root: str,
        source_target: Path,
        reason: str,
    ) -> None:
        if not source.is_dir():
            raise FileNotFoundError(f"Required {reason} directory not found: {source}")
        for path in source.rglob("*"):
            if path.is_file():
                self.add_file(
                    path,
                    destination_root=destination_root,
                    source_target=source_target,
                    reason=reason,
                )

    def add_outside_container(
        self,
        target: Path,
        container: Path,
        *,
        destination_root: str,
    ) -> None:
        if target == container:
            return
        try:
            first_container_part = container.relative_to(target).parts[0]
        except (ValueError, IndexError) as exc:
            raise ValueError(f"Container is not nested below archive target: {container}") from exc
        for child in target.iterdir():
            if child.name == first_container_part:
                continue
            if child.is_dir():
                self.add_tree(
                    child,
                    destination_root=destination_root,
                    source_target=target,
                    reason="archive-level metadata",
                )
            elif child.is_file():
                self.add_file(
                    child,
                    destination_root=destination_root,
                    source_target=target,
                    reason="archive-level metadata",
                )

    def items(self) -> list[CopyItem]:
        return sorted(
            self._items.values(),
            key=lambda item: (item.destination_root, item.destination_relative.as_posix()),
        )


def _followup_run_names(reports_dir: Path) -> list[str]:
    """Return branches explicitly identified by archived follow-up manifests."""

    names: set[str] = set()
    for path in reports_dir.glob("**/*_summary.json"):
        payload = _read_json(path)
        if payload.get("artifact_kind") != "post_sim_followup_summary":
            continue
        for condition in payload.get("conditions") or []:
            if isinstance(condition, dict) and condition.get("run_name"):
                names.add(str(condition["run_name"]))
    return sorted(names)


def _add_first_existing(
    builder: _PlanBuilder,
    candidates: list[tuple[Path, str, Path]],
    *,
    reason: str,
    required: bool = False,
) -> None:
    """Keep one canonical copy when checkpoint/experiment copies are duplicates."""

    for source, destination_root, source_target in candidates:
        if source.is_file():
            builder.add_file(
                source,
                destination_root=destination_root,
                source_target=source_target,
                reason=reason,
            )
            return
    if required:
        raise FileNotFoundError(
            f"No {reason} found in: " + ", ".join(str(value[0]) for value in candidates)
        )


def _add_filtered_tree(
    builder: _PlanBuilder,
    source: Path,
    *,
    destination_root: str,
    source_target: Path,
    reason: str,
    include_raw_scale_traces: bool,
) -> None:
    """Keep analysis artifacts while dropping embedded snapshot/work copies."""

    if not source.is_dir():
        return
    for path in source.rglob("*"):
        if not path.is_file():
            continue
        relative_parts = path.relative_to(source).parts
        if any(part in {"_tmp", "snapshot_storage"} for part in relative_parts):
            continue
        if path.name == "job.json":
            continue
        if (
            not include_raw_scale_traces
            and (path.name.endswith("_trace.json") or path.name.endswith("_trace.jsonl"))
        ):
            continue
        builder.add_file(
            path,
            destination_root=destination_root,
            source_target=source_target,
            reason=reason,
        )


def _add_curated_run(
    builder: _PlanBuilder,
    layout: ArchiveLayout,
    run_name: str,
    *,
    completed_only: bool,
    include_raw_scale_traces: bool,
    include_merged_consultation_dialogues: bool,
    include_dynamic_llm_traces: bool,
) -> None:
    checkpoint_run = layout.checkpoint_container / run_name
    experiment_run = layout.experiment_container / run_name
    checkpoints = sorted(checkpoint_run.glob("simulate-*.json"))
    if not checkpoints:
        raise FileNotFoundError(f"No simulate checkpoint found: {checkpoint_run}")
    for checkpoint in ([checkpoints[-1]] if completed_only else checkpoints):
        builder.add_file(
            checkpoint,
            destination_root="checkpoint",
            source_target=layout.checkpoint_target,
            reason=(
                "final simulation checkpoint"
                if completed_only
                else "simulation checkpoint timeline"
            ),
        )

    for name in CURATED_RUN_FILES:
        _add_first_existing(
            builder,
            [
                (experiment_run / name, "experiment", layout.experiment_target),
                (checkpoint_run / name, "checkpoint", layout.checkpoint_target),
            ],
            reason="run provenance/process artifact",
            required=name == "cbt_condition_manifest.json",
        )
    _add_first_existing(
        builder,
        [
            (
                experiment_run / "traces" / "judge_conversation.json",
                "experiment",
                layout.experiment_target,
            ),
            (
                checkpoint_run / "judge_traces" / "judge_conversation.json",
                "checkpoint",
                layout.checkpoint_target,
            ),
        ],
        reason="judge conversation",
        required=True,
    )
    checkpoint_judge = checkpoint_run / "judge_traces" / "judge_conversation.json"
    if checkpoint_judge.is_file():
        builder.add_file(
            checkpoint_judge,
            destination_root="checkpoint",
            source_target=layout.checkpoint_target,
            reason="interpretive judge-conversation path compatibility",
        )

    # A completed run's storage is the only durable copy of the agents' raw
    # memory/index state.  It is needed for resume, exact downstream
    # re-evaluation and forensic inspection, so it is retained even in a
    # completed-only analysis backup.
    storage_tree = checkpoint_run / "storage"
    if storage_tree.is_dir():
        builder.add_tree(
            storage_tree,
            destination_root="checkpoint",
            source_target=layout.checkpoint_target,
            reason="completed-run storage",
        )

    # Dynamic traces are run-specific audit evidence.  Selecting a single
    # archive-wide example silently discarded other groups/personas (for
    # example non-G1 runs), so retain the trace belonging to every included
    # curated run.
    dynamic_trace = (
        checkpoint_run / "judge_traces" / "depression_dynamic_llm_trace.jsonl"
    )
    if include_dynamic_llm_traces and dynamic_trace.is_file():
        builder.add_file(
            dynamic_trace,
            destination_root="checkpoint",
            source_target=layout.checkpoint_target,
            reason="per-run dynamic LLM audit trace",
        )
    if completed_only:
        return

    merged_dialogues = experiment_run / "traces" / "merge_consultation_dialogues.json"
    if include_merged_consultation_dialogues and merged_dialogues.is_file():
        builder.add_file(
            merged_dialogues,
            destination_root="experiment",
            source_target=layout.experiment_target,
            reason="merged consultation dialogues",
        )

    for tree, destination_root, target, reason in (
        (
            experiment_run / "visualizations",
            "experiment",
            layout.experiment_target,
            "run visualizations",
        ),
        (
            experiment_run / "traces" / "forced_prompt_traces",
            "experiment",
            layout.experiment_target,
            "intervention prompt traces",
        ),
        (
            checkpoint_run / "consult_history",
            "checkpoint",
            layout.checkpoint_target,
            "consultation history",
        ),
    ):
        if tree.is_dir():
            builder.add_tree(
                tree,
                destination_root=destination_root,
                source_target=target,
                reason=reason,
            )

    # Prefer experiment_data copies of staged scales and prompt traces.  Follow-up
    # branches often have only checkpoint copies, so fall back to those.
    prompt_tree = experiment_run / "traces" / "forced_prompt_traces"
    if not prompt_tree.is_dir():
        checkpoint_prompts = checkpoint_run / "forced_prompt_traces"
        if checkpoint_prompts.is_dir():
            builder.add_tree(
                checkpoint_prompts,
                destination_root="checkpoint",
                source_target=layout.checkpoint_target,
                reason="intervention prompt traces",
            )
    scale_tree = experiment_run / "scales" / "staged"
    if scale_tree.is_dir():
        _add_filtered_tree(
            builder,
            scale_tree,
            destination_root="experiment",
            source_target=layout.experiment_target,
            reason="curated staged scale artifacts",
            include_raw_scale_traces=include_raw_scale_traces,
        )
    else:
        _add_filtered_tree(
            builder,
            checkpoint_run / "staged_eval",
            destination_root="checkpoint",
            source_target=layout.checkpoint_target,
            reason="curated staged evaluation artifacts",
            include_raw_scale_traces=include_raw_scale_traces,
        )


def _select_complete_reports_by_prefix(
    report_paths: list[Path],
    prefixes: tuple[str, ...],
) -> list[Path]:
    if not prefixes:
        return report_paths
    selected: list[Path] = []
    for report_path in report_paths:
        report_records, _, _ = load_records([report_path])
        matches = [
            any(record.run_name.startswith(prefix) for prefix in prefixes)
            for record in report_records
        ]
        if any(matches) and not all(matches):
            raise ValueError(
                "Run-prefix filtering cannot split a report containing both included "
                f"and excluded runs: {report_path}"
            )
        if matches and all(matches):
            selected.append(report_path)
    return selected


def _referenced_original_report(reports_dir: Path, report_path: Path) -> Path | None:
    """Resolve a repeat summary's archived source report by unique basename.

    Repeat summaries commonly store an absolute path from the machine that
    created them.  A relocated archive therefore cannot rely on that path,
    while :mod:`experiment_eval.archive_data` still needs the source report
    for controller and provenance metadata.  Resolve within the archive just
    as the archive interface does.
    """

    payload = _read_json(report_path)
    declared = str(payload.get("source_original_summary") or "").strip()
    if not declared:
        return None
    basename = Path(declared).name
    matches = sorted(
        path for path in reports_dir.glob(f"**/{basename}") if path.is_file()
    )
    if not matches and "-followup_summary" in report_path.name:
        followup_path, _ = resolve_followup_source_summary(report_path, payload)
        if followup_path is not None:
            return followup_path
    if len(matches) != 1:
        raise ValueError(
            "Cannot uniquely resolve source_original_summary for completed-only "
            f"backup: {report_path}: {declared!r}"
        )
    return matches[0]


def build_plan(
    layout: ArchiveLayout,
    *,
    completed_only: bool = False,
    include_run_prefixes: tuple[str, ...] = (),
    include_raw_scale_traces: bool = False,
    include_merged_consultation_dialogues: bool = True,
    include_dynamic_llm_traces: bool = True,
) -> PrunePlan:
    # Pruning must see completed follow-up repeat summaries as well as root
    # summaries; plotting discovery excludes them by default.
    all_complete_report_paths = find_report_files(
        layout.reports_dir, recursive=False, include_followup=True
    )
    if not all_complete_report_paths:
        raise ValueError(f"No complete repeat summaries found: {layout.reports_dir}")
    all_records, _, _ = load_records(all_complete_report_paths)
    all_complete_run_names = sorted({record.run_name for record in all_records if record.run_name})
    if len(all_complete_run_names) != len(all_records):
        raise ValueError("Every complete report record must declare a unique non-empty run_name")

    complete_report_paths = _select_complete_reports_by_prefix(
        all_complete_report_paths,
        include_run_prefixes,
    )
    if not complete_report_paths:
        prefixes = ", ".join(repr(prefix) for prefix in include_run_prefixes)
        raise ValueError(f"No complete repeat summaries match run prefixes: {prefixes}")
    records, _, _ = load_records(complete_report_paths)
    complete_run_names = sorted({record.run_name for record in records if record.run_name})
    if len(complete_run_names) != len(records):
        raise ValueError("Every complete report record must declare a unique non-empty run_name")
    excluded_complete_run_names = sorted(set(all_complete_run_names) - set(complete_run_names))

    incomplete_report_paths = sorted(layout.reports_dir.glob("repeat-*_incomplete.json"))
    incomplete_run_name_set = set(_incomplete_runs(incomplete_report_paths))
    superseded_incomplete_run_names = sorted(
        incomplete_run_name_set & set(all_complete_run_names)
    )
    incomplete_run_names = sorted(incomplete_run_name_set - set(all_complete_run_names))
    followup_run_names = _followup_run_names(layout.reports_dir)
    reported_run_names = (
        set(all_complete_run_names)
        | set(incomplete_run_names)
        | set(followup_run_names)
    )
    checkpoint_run_names = {
        path.name
        for path in layout.checkpoint_container.iterdir()
        if path.is_dir() and any(path.glob("simulate-*.json"))
    }
    experiment_run_names = {
        path.name
        for path in layout.experiment_container.iterdir()
        if path.is_dir()
        and path.name not in {"reports", "batch_state", "repeat_scale_eval"}
        and (
            (path / "cbt_condition_manifest.json").is_file()
            or (path / "trial_meta.json").is_file()
            or (path / "traces" / "judge_conversation.json").is_file()
        )
    }
    unreported_run_names = sorted((checkpoint_run_names | experiment_run_names) - reported_run_names)
    physical_run_names = checkpoint_run_names | experiment_run_names
    report_only_run_names = sorted(reported_run_names - physical_run_names)
    all_run_names = (
        complete_run_names
        if completed_only
        else sorted(reported_run_names | set(unreported_run_names))
    )
    rescue_runs = (
        set()
        if completed_only
        else set(incomplete_run_names) | set(unreported_run_names)
    )
    builder = _PlanBuilder(layout)

    if completed_only:
        for report_path in complete_report_paths:
            builder.add_file(
                report_path,
                destination_root="experiment",
                source_target=layout.experiment_target,
                reason="complete repeat summary",
            )
            original_report = _referenced_original_report(
                layout.reports_dir, report_path
            )
            if original_report is not None:
                builder.add_file(
                    original_report,
                    destination_root="experiment",
                    source_target=layout.experiment_target,
                    reason="repeat summary source report",
                )
                original_markdown = original_report.with_suffix(".md")
                if original_markdown.is_file():
                    builder.add_file(
                        original_markdown,
                        destination_root="experiment",
                        source_target=layout.experiment_target,
                        reason="repeat summary source report markdown",
                    )
    else:
        builder.add_tree(
            layout.reports_dir,
            destination_root="experiment",
            source_target=layout.experiment_target,
            reason="reports",
        )
    batch_state = layout.experiment_container / "batch_state"
    if not completed_only:
        if batch_state.is_dir():
            builder.add_tree(
                batch_state,
                destination_root="experiment",
                source_target=layout.experiment_target,
                reason="batch-state metadata",
            )
        builder.add_outside_container(
            layout.experiment_target,
            layout.experiment_container,
            destination_root="experiment",
        )
        builder.add_outside_container(
            layout.checkpoint_target,
            layout.checkpoint_container,
            destination_root="checkpoint",
        )

    for run_name in all_run_names:
        if run_name in report_only_run_names:
            continue
        if run_name in rescue_runs:
            checkpoint_run = layout.checkpoint_container / run_name
            experiment_run = layout.experiment_container / run_name
            rescue_reason = "incomplete-run rescue" if run_name in incomplete_run_names else "unreported-run rescue"
            if checkpoint_run.is_dir():
                builder.add_tree(
                    checkpoint_run,
                    destination_root="checkpoint",
                    source_target=layout.checkpoint_target,
                    reason=f"{rescue_reason} checkpoint",
                )
            if experiment_run.is_dir():
                builder.add_tree(
                    experiment_run,
                    destination_root="experiment",
                    source_target=layout.experiment_target,
                    reason=f"{rescue_reason} experiment data",
                )
            continue
        _add_curated_run(
            builder,
            layout,
            run_name,
            completed_only=completed_only,
            include_raw_scale_traces=include_raw_scale_traces,
            include_merged_consultation_dialogues=include_merged_consultation_dialogues,
            include_dynamic_llm_traces=include_dynamic_llm_traces,
        )

    raw_paths: set[Path] = set()
    if not completed_only:
        for pattern in RAW_EVALUATION_PATTERNS[:2]:
            raw_paths.update(
                path for path in layout.experiment_container.rglob(pattern) if path.is_file()
            )
        raw_parent_dirs = {path.parent for path in raw_paths}
        for parent in raw_parent_dirs:
            for name in SMALL_EVALUATION_METADATA:
                path = parent / name
                if path.is_file():
                    raw_paths.add(path)
            if include_raw_scale_traces:
                for path in parent.glob(RAW_EVALUATION_PATTERNS[2]):
                    if path.is_file():
                        raw_paths.add(path)
        for path in sorted(raw_paths):
            builder.add_file(
                path,
                destination_root="experiment",
                source_target=layout.experiment_target,
                reason="raw scale answer/score/trace",
            )

    items = builder.items()
    retained_bytes = sum(item.source.stat().st_size for item in items)
    source_bytes = _tree_size(layout.checkpoint_target) + _tree_size(layout.experiment_target)
    return PrunePlan(
        layout=layout,
        complete_report_paths=complete_report_paths,
        complete_run_names=complete_run_names,
        excluded_complete_run_names=excluded_complete_run_names,
        incomplete_report_paths=incomplete_report_paths,
        incomplete_run_names=incomplete_run_names,
        superseded_incomplete_run_names=superseded_incomplete_run_names,
        unreported_run_names=unreported_run_names,
        followup_run_names=followup_run_names,
        report_only_run_names=report_only_run_names,
        items=items,
        source_bytes=source_bytes,
        retained_bytes=retained_bytes,
        raw_evaluation_file_count=len(raw_paths),
        completed_only=completed_only,
        include_run_prefixes=include_run_prefixes,
        include_raw_scale_traces=include_raw_scale_traces,
        include_merged_consultation_dialogues=include_merged_consultation_dialogues,
        include_dynamic_llm_traces=include_dynamic_llm_traces,
    )


def _format_bytes(value: int) -> str:
    amount = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if amount < 1024 or unit == "TiB":
            return f"{amount:.1f} {unit}"
        amount /= 1024
    return f"{amount:.1f} TiB"


def print_plan(plan: PrunePlan, *, operation_mode: str = "prune") -> None:
    print("Experiment-evaluation archive retention plan")
    print(f"  layout:                    {plan.layout.mode}")
    print(f"  checkpoint target:         {plan.layout.checkpoint_target}")
    print(f"  experiment-data target:    {plan.layout.experiment_target}")
    print(f"  complete report records:   {len(plan.complete_report_paths)}")
    print(f"  complete report runs:       {len(plan.complete_run_names)}")
    print(f"  incomplete report files:   {len(plan.incomplete_report_paths)}")
    print(f"  active incomplete runs:    {len(plan.incomplete_run_names)}")
    print(f"  superseded incomplete runs: {len(plan.superseded_incomplete_run_names)}")
    print(f"  unreported rescue runs:    {len(plan.unreported_run_names)}")
    print(f"  linked follow-up branches: {len(plan.followup_run_names)}")
    print(f"  report-only runs:          {len(plan.report_only_run_names)}")
    print(f"  retained raw scale files:  {plan.raw_evaluation_file_count}")
    print(f"  current archive size:       {_format_bytes(plan.source_bytes)}")
    print(f"  retained size estimate:     {_format_bytes(plan.retained_bytes)}")
    if operation_mode == "prune":
        print(f"  reclaimed size estimate:    {_format_bytes(plan.estimated_reclaimed_bytes)}")
    else:
        print("  source space reclaimed:     0 B (backup mode leaves sources unchanged)")
    if plan.completed_only:
        print("  retention scope:           completed repeat summaries only")
        print("  raw scale audit artifacts: skipped")
    elif plan.include_raw_scale_traces:
        print("  raw scale traces:          retained (large opt-in artifacts)")
    else:
        print("  raw scale traces:          omitted; answers/scores/metadata retained")
    merged_dialogue_count = sum(
        item.destination_relative.name == "merge_consultation_dialogues.json"
        for item in plan.items
    )
    dynamic_trace_count = sum(
        item.destination_relative.name == "depression_dynamic_llm_trace.jsonl"
        for item in plan.items
    )
    completed_storage_run_count = len(
        {
            item.source.relative_to(plan.layout.checkpoint_container).parts[0]
            for item in plan.items
            if item.reason == "completed-run storage"
        }
    )
    print(f"  merged consultation dialogues: {merged_dialogue_count} retained")
    print(f"  dynamic LLM traces:        {dynamic_trace_count} retained")
    print(f"  completed-run storage:     {completed_storage_run_count} runs retained")
    if plan.include_run_prefixes:
        print(f"  included run prefixes:     {', '.join(plan.include_run_prefixes)}")
        print(f"  excluded complete runs:    {len(plan.excluded_complete_run_names)}")
        print("  selected complete runs:")
        for run_name in plan.complete_run_names:
            print(f"    - {run_name}")
    if plan.incomplete_run_names:
        label = "incomplete skipped runs" if plan.completed_only else "incomplete full rescue runs"
        print(f"  {label}:")
        for run_name in plan.incomplete_run_names:
            print(f"    - {run_name}")
    if plan.unreported_run_names:
        label = "unreported skipped runs" if plan.completed_only else "unreported full rescue runs"
        print(f"  {label}:")
        for run_name in plan.unreported_run_names:
            print(f"    - {run_name}")
    if plan.superseded_incomplete_run_names:
        print("  stale incomplete entries superseded by complete summaries:")
        for run_name in plan.superseded_incomplete_run_names:
            print(f"    - {run_name}")


def resolve_backup_output(
    raw: str | Path,
    *,
    results_root: Path,
    layout: ArchiveLayout,
) -> Path:
    if not str(raw).strip():
        raise ValueError("Backup mode requires --backup-output")
    output = _resolve_new_path(raw, results_root.expanduser().resolve())
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite backup output: {output}")

    source_targets = (layout.checkpoint_target, layout.experiment_target)
    for source in source_targets:
        if output == source or output in source.parents or source in output.parents:
            raise ValueError(f"Backup output must not overlap a source target: {output}, {source}")
    if (
        layout.mode == "combined"
        and layout.checkpoint_target.parent == layout.experiment_target.parent
        and layout.checkpoint_target.parent in output.parents
    ):
        raise ValueError(
            "Backup output must be outside the combined source archive: "
            f"{layout.checkpoint_target.parent}"
        )
    if layout.checkpoint_target.name == layout.experiment_target.name:
        raise ValueError(
            "Backup mode requires checkpoint and experiment-data targets with different names"
        )
    return output


def _backup_layout(source: ArchiveLayout, backup_root: Path) -> ArchiveLayout:
    checkpoint_target = backup_root / source.checkpoint_target.name
    experiment_target = backup_root / source.experiment_target.name
    return ArchiveLayout(
        mode=source.mode,
        results_root=backup_root,
        checkpoint_target=checkpoint_target,
        checkpoint_container=(
            checkpoint_target
            / source.checkpoint_container.relative_to(source.checkpoint_target)
        ),
        experiment_target=experiment_target,
        experiment_container=(
            experiment_target
            / source.experiment_container.relative_to(source.experiment_target)
        ),
    )


def _copy_destination(item: CopyItem, checkpoint_stage: Path, experiment_stage: Path) -> Path:
    root = checkpoint_stage if item.destination_root == "checkpoint" else experiment_stage
    return root / item.destination_relative


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stage_plan(plan: PrunePlan, staging_parent: Path | None) -> tuple[Path, Path, Path]:
    parent = staging_parent.expanduser().resolve() if staging_parent else Path(tempfile.gettempdir())
    parent.mkdir(parents=True, exist_ok=True)
    free_bytes = shutil.disk_usage(parent).free
    if free_bytes < plan.retained_bytes * 1.05:
        raise OSError(
            f"Insufficient staging space below {parent}: "
            f"need about {_format_bytes(plan.retained_bytes)}, have {_format_bytes(free_bytes)}"
        )
    staging_root = Path(tempfile.mkdtemp(prefix="experiment-eval-prune-", dir=parent))
    checkpoint_stage = staging_root / "checkpoint-target"
    experiment_stage = staging_root / "experiment-target"
    checkpoint_stage.mkdir()
    experiment_stage.mkdir()
    try:
        for item in plan.items:
            destination = _copy_destination(item, checkpoint_stage, experiment_stage)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item.source, destination)
        for item in plan.items:
            destination = _copy_destination(item, checkpoint_stage, experiment_stage)
            if item.source.stat().st_size != destination.stat().st_size:
                raise ValueError(f"Staged file size mismatch: {destination}")
            if _sha256(item.source) != _sha256(destination):
                raise ValueError(f"Staged file checksum mismatch: {destination}")
    except BaseException:
        shutil.rmtree(staging_root, ignore_errors=True)
        raise
    return staging_root, checkpoint_stage, experiment_stage


def _staged_layout(
    source: ArchiveLayout,
    checkpoint_stage: Path,
    experiment_stage: Path,
) -> ArchiveLayout:
    checkpoint_relative = source.checkpoint_container.relative_to(source.checkpoint_target)
    experiment_relative = source.experiment_container.relative_to(source.experiment_target)
    return ArchiveLayout(
        mode=source.mode,
        results_root=source.results_root,
        checkpoint_target=checkpoint_stage,
        checkpoint_container=checkpoint_stage / checkpoint_relative,
        experiment_target=experiment_stage,
        experiment_container=experiment_stage / experiment_relative,
    )


def _normalized_process(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {key: value for key, value in row.items() if key not in PATH_ONLY_PROCESS_FIELDS}
        for row in rows
    ]


def _reports_for_expected_runs(
    report_paths: list[Path],
    expected_run_names: set[str],
) -> list[Path]:
    selected: list[Path] = []
    found: set[str] = set()
    for report_path in report_paths:
        report_records, _, _ = load_records([report_path])
        report_run_names = {record.run_name for record in report_records}
        matching = report_run_names & expected_run_names
        if matching and not report_run_names <= expected_run_names:
            raise ValueError(
                "Selected runs share a report with excluded runs, so the report cannot be "
                f"copied without rewriting it: {report_path}"
            )
        if matching:
            selected.append(report_path)
            found.update(matching)
    missing = expected_run_names - found
    if missing:
        raise ValueError("Complete reports not found for selected runs: " + ", ".join(sorted(missing)))
    return selected


def validate_retained_layout(
    source: ArchiveLayout,
    retained: ArchiveLayout,
    *,
    expected_run_names: set[str] | None = None,
) -> None:
    source_reports = find_report_files(
        source.reports_dir, recursive=False, include_followup=True
    )
    retained_reports = find_report_files(
        retained.reports_dir, recursive=False, include_followup=True
    )
    if expected_run_names is not None:
        source_reports = _reports_for_expected_runs(source_reports, expected_run_names)
    source_records, source_labels, source_scales = load_records(source_reports)
    retained_records, retained_labels, retained_scales = load_records(retained_reports)
    if expected_run_names is not None:
        retained_run_names = {record.run_name for record in retained_records}
        if retained_run_names != expected_run_names:
            raise ValueError("Retained reports changed the selected run set")
    if [record.stable_id for record in source_records] != [record.stable_id for record in retained_records]:
        raise ValueError("Retained reports changed the stable experiment record set")
    if source_labels != retained_labels or source_scales != retained_scales:
        raise ValueError("Retained reports changed labels or scales")

    source_process, source_stages = extract_process_metrics(
        source_records,
        experiment_data_root=source.experiment_target,
        checkpoints_root=source.checkpoint_target,
        project_root=PROJECT_ROOT,
    )
    retained_process, retained_stages = extract_process_metrics(
        retained_records,
        experiment_data_root=retained.experiment_target,
        checkpoints_root=retained.checkpoint_target,
        project_root=PROJECT_ROOT,
    )
    if _normalized_process(source_process) != _normalized_process(retained_process):
        raise ValueError("Retained files changed process/dose metrics")
    if source_stages != retained_stages:
        raise ValueError("Retained files changed progressive-stage metrics")
    source_by_id = {str(row["stable_id"]): row for row in source_process}
    regressions = [
        str(row["stable_id"])
        for row in retained_process
        if source_by_id.get(str(row["stable_id"]), {}).get("raw_artifacts_available")
        and not row.get("raw_artifacts_available")
    ]
    if regressions:
        raise ValueError(
            "Retained layout lost raw process inputs that existed in the source: "
            + ", ".join(regressions)
        )


def _backup_path(target: Path, stamp: str) -> Path:
    return target.with_name(f"{target.name}.prune-backup-{stamp}")


def install_staged(
    plan: PrunePlan,
    checkpoint_stage: Path,
    experiment_stage: Path,
    *,
    keep_backup: bool,
) -> list[Path]:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    pairs = [
        (plan.layout.checkpoint_target, checkpoint_stage),
        (plan.layout.experiment_target, experiment_stage),
    ]
    backups = [(target, _backup_path(target, stamp)) for target, _ in pairs]
    for target, backup in backups:
        if backup.exists():
            raise FileExistsError(f"Refusing to overwrite backup: {backup}")

    installed: list[Path] = []
    moved: list[tuple[Path, Path]] = []
    try:
        for target, backup in backups:
            os.replace(target, backup)
            moved.append((target, backup))
        for target, staged in pairs:
            shutil.copytree(staged, target)
            installed.append(target)

        installed_layout = ArchiveLayout(
            mode=plan.layout.mode,
            results_root=plan.layout.results_root,
            checkpoint_target=plan.layout.checkpoint_target,
            checkpoint_container=(
                plan.layout.checkpoint_target
                / plan.layout.checkpoint_container.relative_to(plan.layout.checkpoint_target)
            ),
            experiment_target=plan.layout.experiment_target,
            experiment_container=(
                plan.layout.experiment_target
                / plan.layout.experiment_container.relative_to(plan.layout.experiment_target)
            ),
        )
        for item in plan.items:
            staged_file = _copy_destination(item, checkpoint_stage, experiment_stage)
            installed_root = (
                installed_layout.checkpoint_target
                if item.destination_root == "checkpoint"
                else installed_layout.experiment_target
            )
            installed_file = installed_root / item.destination_relative
            if (
                not installed_file.is_file()
                or staged_file.stat().st_size != installed_file.stat().st_size
                or _sha256(staged_file) != _sha256(installed_file)
            ):
                raise ValueError(f"Installed file checksum mismatch: {installed_file}")
        validate_retained_layout(
            _staged_layout(plan.layout, checkpoint_stage, experiment_stage),
            installed_layout,
            expected_run_names=set(plan.complete_run_names),
        )
    except BaseException:
        for target in reversed(installed):
            shutil.rmtree(target, ignore_errors=True)
        for target, backup in reversed(moved):
            if backup.exists() and not target.exists():
                os.replace(backup, target)
        raise

    if keep_backup:
        return [backup for _, backup in backups]
    for _, backup in backups:
        shutil.rmtree(backup)
    return []


def install_backup(
    plan: PrunePlan,
    checkpoint_stage: Path,
    experiment_stage: Path,
    backup_output: Path,
) -> ArchiveLayout:
    """Install a validated retained copy without changing either source target."""

    if backup_output.exists():
        raise FileExistsError(f"Refusing to overwrite backup output: {backup_output}")
    backup_output.parent.mkdir(parents=True, exist_ok=True)
    can_move_staged = (
        checkpoint_stage.stat().st_dev == backup_output.parent.stat().st_dev
        and experiment_stage.stat().st_dev == backup_output.parent.stat().st_dev
    )
    free_bytes = shutil.disk_usage(backup_output.parent).free
    if not can_move_staged and free_bytes < plan.retained_bytes * 1.05:
        raise OSError(
            f"Insufficient backup space below {backup_output.parent}: "
            f"need about {_format_bytes(plan.retained_bytes)}, have {_format_bytes(free_bytes)}"
        )

    temporary_root = Path(
        tempfile.mkdtemp(
            prefix=f".{backup_output.name}.partial-",
            dir=backup_output.parent,
        )
    )
    temporary_layout = _backup_layout(plan.layout, temporary_root)
    final_layout = _backup_layout(plan.layout, backup_output)
    published = False
    try:
        if can_move_staged:
            # Staging already contains checksum-verified copies.  On the same
            # filesystem, move those trees into the publication directory so
            # backup mode does not need a second full copy of large storage
            # and audit traces.  Source archives remain untouched.
            os.replace(checkpoint_stage, temporary_layout.checkpoint_target)
            os.replace(experiment_stage, temporary_layout.experiment_target)
        else:
            shutil.copytree(checkpoint_stage, temporary_layout.checkpoint_target)
            shutil.copytree(experiment_stage, temporary_layout.experiment_target)
        for item in plan.items:
            installed_root = (
                temporary_layout.checkpoint_target
                if item.destination_root == "checkpoint"
                else temporary_layout.experiment_target
            )
            installed_file = installed_root / item.destination_relative
            if (
                not installed_file.is_file()
                or item.source.stat().st_size != installed_file.stat().st_size
                or _sha256(item.source) != _sha256(installed_file)
            ):
                raise ValueError(f"Backup file checksum mismatch: {installed_file}")
        validate_retained_layout(
            plan.layout,
            temporary_layout,
            expected_run_names=set(plan.complete_run_names),
        )
        if backup_output.exists():
            raise FileExistsError(f"Refusing to overwrite backup output: {backup_output}")
        os.replace(temporary_root, backup_output)
        published = True
        validate_retained_layout(
            plan.layout,
            final_layout,
            expected_run_names=set(plan.complete_run_names),
        )
    except BaseException:
        shutil.rmtree(temporary_root, ignore_errors=True)
        if published and backup_output.exists():
            shutil.rmtree(backup_output, ignore_errors=True)
        raise
    return final_layout


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Retain artifacts required by experiment_eval, either by pruning "
            "the source or creating a non-destructive backup. Default: read-only plan."
        )
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path(RESULTS_ROOT),
        help="Base results directory used to resolve relative archive names (default: top config)",
    )
    parser.add_argument(
        "--results-archive",
        default=RESULTS_ARCHIVE or None,
        help="Combined archive name/path containing checkpoints/ and experiment_data/ (default: top config)",
    )
    parser.add_argument(
        "--checkpoints-archive",
        default=CHECKPOINTS_ARCHIVE or None,
        help="Split checkpoint archive name/path; use with --experiment-data-archive (default: top config)",
    )
    parser.add_argument(
        "--experiment-data-archive",
        default=EXPERIMENT_DATA_ARCHIVE or None,
        help="Split experiment-data archive name/path; use with --checkpoints-archive (default: top config)",
    )
    parser.add_argument(
        "--mode",
        choices=("prune", "backup"),
        default=OPERATION_MODE,
        help="prune replaces source archives; backup copies retained files without changing sources",
    )
    parser.add_argument(
        "--backup-output",
        type=Path,
        default=Path(BACKUP_OUTPUT_DIR) if str(BACKUP_OUTPUT_DIR).strip() else None,
        help="New output directory for backup mode; relative paths resolve below results-root",
    )
    parser.add_argument(
        "--completed-only",
        action=argparse.BooleanOptionalAction,
        default=BACKUP_COMPLETED_ONLY,
        help=(
            "Backup only complete repeat-summary plot inputs; skip incomplete/unreported "
            "runs and raw scale audit artifacts (backup mode only)"
        ),
    )
    parser.add_argument(
        "--include-run-prefix",
        action="append",
        default=list(BACKUP_INCLUDE_RUN_PREFIXES),
        help=(
            "Include only complete runs whose run_name starts with this prefix; repeatable "
            "and allowed only with backup --completed-only"
        ),
    )
    parser.add_argument(
        "--include-raw-scale-traces",
        action=argparse.BooleanOptionalAction,
        default=INCLUDE_RAW_SCALE_TRACES,
        help=(
            "Retain large *_trace.json files from repeated/staged scale jobs. "
            "Answers, scores and metadata are retained without this opt-in."
        ),
    )
    parser.add_argument(
        "--include-merged-consultation-dialogues",
        action=argparse.BooleanOptionalAction,
        default=INCLUDE_MERGED_CONSULTATION_DIALOGUES,
        help="Retain each run's traces/merge_consultation_dialogues.json (default: enabled).",
    )
    parser.add_argument(
        "--include-dynamic-llm-traces",
        "--include-one-dynamic-llm-trace",
        dest="include_dynamic_llm_traces",
        action=argparse.BooleanOptionalAction,
        default=INCLUDE_DYNAMIC_LLM_TRACES,
        help=(
            "Retain each included run's depression_dynamic_llm_trace.jsonl "
            "(default: enabled; the old --include-one-* spelling is a compatibility alias)."
        ),
    )
    parser.add_argument(
        "--apply",
        action=argparse.BooleanOptionalAction,
        default=APPLY_CHANGES,
        help="Build, checksum, validate, and execute the selected mode (default: top config)",
    )
    parser.add_argument(
        "--keep-backup",
        action=argparse.BooleanOptionalAction,
        default=KEEP_BACKUP,
        help="Keep renamed full archives after successful replacement (default: top config)",
    )
    parser.add_argument(
        "--staging-parent",
        type=Path,
        default=Path(STAGING_PARENT) if str(STAGING_PARENT).strip() else None,
        help="Temporary staging parent (default: top config or system temporary directory)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        layout = resolve_layout(
            results_root=args.results_root,
            results_archive=args.results_archive,
            checkpoints_archive=args.checkpoints_archive,
            experiment_data_archive=args.experiment_data_archive,
        )
        include_run_prefixes = tuple(args.include_run_prefix)
        if any(not prefix for prefix in include_run_prefixes):
            raise ValueError("--include-run-prefix cannot be empty")
        if args.completed_only and args.mode != "backup":
            raise ValueError("--completed-only is only allowed with --mode backup")
        if include_run_prefixes and not (args.mode == "backup" and args.completed_only):
            raise ValueError(
                "--include-run-prefix requires --mode backup and --completed-only"
            )
        plan = build_plan(
            layout,
            completed_only=args.completed_only,
            include_run_prefixes=include_run_prefixes,
            include_raw_scale_traces=args.include_raw_scale_traces,
            include_merged_consultation_dialogues=args.include_merged_consultation_dialogues,
            include_dynamic_llm_traces=args.include_dynamic_llm_traces,
        )
        backup_output = None
        if args.mode == "backup":
            if args.keep_backup:
                raise ValueError("--keep-backup only applies to prune mode")
            backup_output = resolve_backup_output(
                args.backup_output or "",
                results_root=args.results_root,
                layout=layout,
            )
        else:
            # A configured backup output is harmless when a caller temporarily
            # overrides the top-level default with --mode prune.
            backup_output = None
        print_plan(plan, operation_mode=args.mode)
        if args.mode == "backup":
            print("  operation mode:            backup (source archives remain unchanged)")
            print(f"  backup output:             {backup_output}")
        else:
            print("  operation mode:            prune (source archives will be replaced)")
        if not args.apply:
            action = "backup copy" if args.mode == "backup" else "cleanup"
            print(f"\nDry-run only. Re-run with --apply to perform validated {action}.")
            return 0

        staging_root, checkpoint_stage, experiment_stage = stage_plan(plan, args.staging_parent)
        try:
            staged_layout = _staged_layout(layout, checkpoint_stage, experiment_stage)
            validate_retained_layout(
                layout,
                staged_layout,
                expected_run_names=set(plan.complete_run_names),
            )
            if args.mode == "backup":
                assert backup_output is not None
                backup_layout = install_backup(
                    plan,
                    checkpoint_stage,
                    experiment_stage,
                    backup_output,
                )
                backups: list[Path] = []
            else:
                backup_layout = None
                backups = install_staged(
                    plan,
                    checkpoint_stage,
                    experiment_stage,
                    keep_backup=args.keep_backup,
                )
        finally:
            shutil.rmtree(staging_root, ignore_errors=True)

        if backup_layout is not None:
            print("\nBackup completed and retained artifacts passed experiment_eval validation.")
            print(f"  source checkpoint archive:     {layout.checkpoint_target} (unchanged)")
            print(f"  source experiment-data archive: {layout.experiment_target} (unchanged)")
            print(f"  backup root:                   {backup_output}")
            print(f"  backup checkpoint archive:     {backup_layout.checkpoint_target}")
            print(f"  backup experiment-data archive: {backup_layout.experiment_target}")
            return 0

        print("\nCleanup completed and retained artifacts passed experiment_eval validation.")
        print(f"  final checkpoint archive:      {layout.checkpoint_target}")
        print(f"  final experiment-data archive: {layout.experiment_target}")
        if backups:
            print("  retained backups:")
            for backup in backups:
                print(f"    - {backup}")
        else:
            print("  superseded full archives: deleted (not locally recoverable)")
        return 0
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise SystemExit(f"Archive retention failed: {exc}") from exc


if __name__ == "__main__":
    raise SystemExit(main())
