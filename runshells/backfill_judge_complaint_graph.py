#!/usr/bin/env python3
"""Backfill compact complaint-graph changes into archived judge traces."""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.complaint_graph_trace import (
    ComplaintGraphTraceAlignmentError,
    enrich_archived_judge_trace,
)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=".{}.".format(path.name),
        suffix=".tmp",
        dir=str(path.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def infer_patient_names(trace_payload: Mapping[str, Any]) -> Sequence[str]:
    names = []
    seen = set()
    sessions = trace_payload.get("sessions", [])
    if not isinstance(sessions, list):
        return names
    for session in sessions:
        if not isinstance(session, Mapping):
            continue
        meeting = session.get("meeting", {})
        meeting = meeting if isinstance(meeting, Mapping) else {}
        pair_key = str(meeting.get("pair_key", "") or "")
        parts = pair_key.split("::", 1)
        if len(parts) != 2:
            continue
        patient = parts[1].strip()
        if patient and patient not in seen:
            seen.add(patient)
            names.append(patient)
    return names


def select_graph_snapshot(
    checkpoint_dir: Path,
    patient_names: Sequence[str],
) -> Tuple[Path, str, Dict[str, Any]]:
    best: Optional[Tuple[int, str, Path, str, Dict[str, Any]]] = None
    for path in sorted(checkpoint_dir.glob("simulate-*.json")):
        payload = read_json(path)
        if not isinstance(payload, Mapping):
            continue
        agents = payload.get("agents", {})
        agents = agents if isinstance(agents, Mapping) else {}
        for patient in patient_names:
            agent = agents.get(patient, {})
            agent = agent if isinstance(agent, Mapping) else {}
            dynamic = agent.get("depression_dynamic_state", {})
            dynamic = dynamic if isinstance(dynamic, Mapping) else {}
            graph = dynamic.get("complaint_graph_manager", {})
            graph = graph if isinstance(graph, Mapping) else {}
            history = graph.get("stage_history", [])
            if not isinstance(history, list):
                continue
            candidate = (
                len(history),
                path.name,
                path,
                patient,
                copy.deepcopy(dict(graph)),
            )
            if best is None or candidate[:2] > best[:2]:
                best = candidate
    if best is None or best[0] <= 0:
        raise ComplaintGraphTraceAlignmentError(
            "no checkpoint snapshot contains complaint graph stage_history"
        )
    return best[2], best[3], best[4]


def select_forced_prompt_sidecar(
    checkpoint_dir: Path,
    source_snapshot: Path,
) -> Path:
    sidecar_dir = (
        checkpoint_dir
        / "trace_state_sidecars"
        / "forced_prompt_trace_state"
    )
    exact = sidecar_dir / "{}.json".format(source_snapshot.stem)
    if exact.is_file():
        return exact
    candidates = sorted(sidecar_dir.glob("simulate-*.json"))
    if not candidates:
        raise FileNotFoundError(
            "forced prompt trace sidecar not found under: {}".format(sidecar_dir)
        )
    best: Optional[Tuple[int, str, Path]] = None
    for path in candidates:
        payload = read_json(path)
        sessions = payload.get("sessions", []) if isinstance(payload, Mapping) else []
        count = len(sessions) if isinstance(sessions, list) else 0
        candidate = (count, path.name, path)
        if best is None or candidate[:2] > best[:2]:
            best = candidate
    assert best is not None
    return best[2]


def default_experiment_trace(checkpoint_dir: Path) -> Optional[Path]:
    run_name = checkpoint_dir.name
    candidate = (
        PROJECT_ROOT
        / "results"
        / "experiment_data"
        / run_name
        / "traces"
        / "judge_conversation.json"
    )
    return candidate if candidate.is_file() else None


def prepare_enriched_targets(
    checkpoint_dir: Path,
    experiment_trace: Optional[Path] = None,
) -> Tuple[Dict[Path, Any], Dict[str, Any]]:
    main_trace = checkpoint_dir / "judge_traces" / "judge_conversation.json"
    if not main_trace.is_file():
        raise FileNotFoundError("judge trace not found: {}".format(main_trace))
    main_payload = read_json(main_trace)
    if not isinstance(main_payload, Mapping):
        raise TypeError("judge trace root must be an object")
    patient_names = infer_patient_names(main_payload)
    if not patient_names:
        raise ComplaintGraphTraceAlignmentError(
            "could not infer patient from judge trace pair_key"
        )
    source_snapshot, patient_name, graph_state = select_graph_snapshot(
        checkpoint_dir,
        patient_names,
    )
    forced_sidecar = select_forced_prompt_sidecar(
        checkpoint_dir,
        source_snapshot,
    )
    forced_payload = read_json(forced_sidecar)
    if not isinstance(forced_payload, Mapping):
        raise TypeError("forced prompt sidecar root must be an object")

    enriched_main, stats = enrich_archived_judge_trace(
        main_payload,
        graph_state,
        forced_payload,
        source_snapshot.name,
    )
    targets: Dict[Path, Any] = {main_trace: enriched_main}

    dialog_sidecar_dir = (
        checkpoint_dir
        / "trace_state_sidecars"
        / "dialog_judge_trace_state"
    )
    dialog_sidecars = sorted(dialog_sidecar_dir.glob("simulate-*.json"))
    if not dialog_sidecars:
        raise FileNotFoundError(
            "dialog judge sidecars not found under: {}".format(
                dialog_sidecar_dir
            )
        )
    for sidecar in dialog_sidecars:
        payload = read_json(sidecar)
        if not isinstance(payload, Mapping):
            raise TypeError("{} root must be an object".format(sidecar))
        enriched_sidecar, _ = enrich_archived_judge_trace(
            payload,
            graph_state,
            forced_payload,
            source_snapshot.name,
        )
        targets[sidecar] = enriched_sidecar

    experiment_path = experiment_trace or default_experiment_trace(checkpoint_dir)
    if experiment_path is not None:
        if not experiment_path.is_file():
            raise FileNotFoundError(
                "experiment judge trace not found: {}".format(experiment_path)
            )
        experiment_payload = read_json(experiment_path)
        if experiment_payload != main_payload:
            raise ComplaintGraphTraceAlignmentError(
                "experiment judge trace differs from checkpoint judge trace; "
                "refusing to overwrite it"
            )
        targets[experiment_path] = copy.deepcopy(enriched_main)

    report = {
        "checkpoint_dir": str(checkpoint_dir),
        "patient": patient_name,
        "source_snapshot": source_snapshot.name,
        "forced_prompt_sidecar": str(forced_sidecar),
        "dialog_sidecars": len(dialog_sidecars),
        "experiment_trace": str(experiment_path) if experiment_path else "",
        **stats,
    }
    return targets, report


def backup_targets(
    targets: Iterable[Path],
    backup_root: Path,
    checkpoint_dir: Path,
) -> Tuple[Path, Dict[Path, Path]]:
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    backup_dir = (
        backup_root
        / "{}-complaint-graph-backfill-{}".format(checkpoint_dir.name, stamp)
    )
    if backup_dir.exists():
        raise FileExistsError("backup directory already exists: {}".format(backup_dir))
    backup_map: Dict[Path, Path] = {}
    project_results_root = (PROJECT_ROOT / "results").resolve()
    for target in targets:
        try:
            relative = Path("results") / target.resolve().relative_to(
                project_results_root
            )
        except ValueError:
            try:
                relative = target.resolve().relative_to(PROJECT_ROOT.resolve())
            except ValueError:
                resolved_parts = target.resolve().parts
                relative = Path("external").joinpath(*resolved_parts[1:])
        destination = backup_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(target, destination)
        backup_map[target] = destination
    return backup_dir, backup_map


def write_targets_with_rollback(
    payloads: Mapping[Path, Any],
    backup_map: Mapping[Path, Path],
) -> None:
    written = []
    try:
        for target, payload in payloads.items():
            atomic_write_json(target, payload)
            written.append(target)
    except BaseException:
        for target in reversed(written):
            backup_payload = read_json(backup_map[target])
            atomic_write_json(target, backup_payload)
        raise


def run_backfill(
    checkpoint_dir: Path,
    *,
    write: bool,
    experiment_trace: Optional[Path] = None,
    backup_root: Optional[Path] = None,
) -> Dict[str, Any]:
    checkpoint_dir = checkpoint_dir.resolve()
    targets, report = prepare_enriched_targets(
        checkpoint_dir,
        experiment_trace=experiment_trace.resolve() if experiment_trace else None,
    )
    changed = {
        path: payload
        for path, payload in targets.items()
        if read_json(path) != payload
    }
    report["mode"] = "write" if write else "dry-run"
    report["target_files"] = len(targets)
    report["changed_files"] = len(changed)
    report["backup_dir"] = ""
    if not write or not changed:
        return report

    resolved_backup_root = (
        backup_root.resolve()
        if backup_root
        else (PROJECT_ROOT / "results" / "recovery_backups").resolve()
    )
    backup_dir, backup_map = backup_targets(
        changed.keys(),
        resolved_backup_root,
        checkpoint_dir,
    )
    write_targets_with_rollback(changed, backup_map)
    report["backup_dir"] = str(backup_dir)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Backfill per-patient-turn and post-reflection complaint graph "
            "changes into judge traces."
        )
    )
    parser.add_argument(
        "checkpoint_dir",
        type=Path,
        help="Path to results/checkpoints/<run>.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print counts without writing (default).",
    )
    mode.add_argument(
        "--write",
        action="store_true",
        help="Back up and atomically update all trace copies.",
    )
    parser.add_argument(
        "--experiment-trace",
        type=Path,
        default=None,
        help="Optional experiment_data judge trace; auto-detected by run name.",
    )
    parser.add_argument(
        "--backup-root",
        type=Path,
        default=None,
        help="Backup root; defaults to results/recovery_backups.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = run_backfill(
        args.checkpoint_dir,
        write=bool(args.write),
        experiment_trace=args.experiment_trace,
        backup_root=args.backup_root,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
