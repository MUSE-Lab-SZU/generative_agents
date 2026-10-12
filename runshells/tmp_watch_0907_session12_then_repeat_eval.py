#!/usr/bin/env python3
"""Monitor the three unfinished 0907 simulations and relay at session_12.

This is intentionally a narrow, temporary operations script.  It only knows
about the three run names listed in ``TARGETS`` below.  A target becomes ready
only after every capture-only staged node from T0 through session_12 passes the
same snapshot-bundle validation used by the follow-up tooling.  In execute
mode the script then:

1. sends SIGTERM only to ``start.py --name <exact-run-name>``;
2. waits for that exact simulation process to exit (SIGKILL after a timeout);
3. writes an isolated source summary with the controller manifest embedded;
4. runs the normal archived repeat evaluation with the original 10/5 repeat
   counts, controller-manifest validation, partial resume, and cleanup policy.

Without ``--execute`` this performs one read-only scan and prints the action
that would be taken.  This guard is deliberate because execute mode terminates
live simulations.

Remote-platform usage from the repository root::

    python runshells/tmp_watch_0907_session12_then_repeat_eval.py

    nohup python runshells/tmp_watch_0907_session12_then_repeat_eval.py \
      --execute > results/session12-repeat-watch-0907.log 2>&1 &

The watcher exits successfully only after all three formal repeat reports are
complete.  Failed repeat evaluations are retried with ``--resume-partial``.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import copy
import datetime as dt
import fcntl
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from artifact_digest import canonical_json_sha256
from run_post_sim_followup import validate_snapshot_bundle


CONTROLLER_MANIFEST_FILENAME = "cbt_condition_manifest.json"
LABELS = ("T0", "session_2", "session_4", "session_6", "session_8", "session_10", "session_12")
IDENTITY_KEYS = (
    "controller_identity",
    "kind",
    "mode",
    "controller_version",
    "progressive_stage",
    "capabilities",
)


@dataclass(frozen=True)
class Target:
    condition_name: str
    condition_key: str
    run_name: str
    repeat_name: str
    variant: str
    group: str
    severity: str


TARGETS = (
    Target(
        condition_name="Counsel-KBD2-G4-MOD",
        condition_key="Counsel-KBD2-G4-MOD--cbt-progressive-d",
        run_name="batch-0907-01-Counsel-KBD2-G4-MOD--cbt-progressive-d-0907-1356",
        repeat_name="repeat-Counsel-KBD2-G4-MOD--cbt-progressive-d-0907-01",
        variant="kbd2",
        group="g4",
        severity="moderate",
    ),
    Target(
        condition_name="Counsel-LRN-G1-MOD",
        condition_key="Counsel-LRN-G1-MOD--cbt-progressive-d",
        run_name="batch-0907-01-Counsel-LRN-G1-MOD--cbt-progressive-d-0907-1355",
        repeat_name="repeat-Counsel-LRN-G1-MOD--cbt-progressive-d-0907-01",
        variant="lrn",
        group="g1",
        severity="moderate",
    ),
    Target(
        condition_name="Counsel-SQL-G1-MOD",
        condition_key="Counsel-SQL-G1-MOD--cbt-progressive-d",
        run_name="batch-0907-01-Counsel-SQL-G1-MOD--cbt-progressive-d-0907-1357",
        repeat_name="repeat-Counsel-SQL-G1-MOD--cbt-progressive-d-0907-01",
        variant="sql",
        group="g1",
        severity="moderate",
    ),
)


@dataclass(frozen=True)
class Config:
    results_root: Path
    poll_seconds: float
    retry_seconds: float
    term_timeout_seconds: float
    repeat: int
    intermediate_scale_repeats: int
    eval_max_parallel: int
    max_concurrent_evals: int
    cleanup_completed_artifacts: bool
    execute: bool

    @property
    def checkpoints_root(self) -> Path:
        return self.results_root / "checkpoints"

    @property
    def reports_root(self) -> Path:
        return self.results_root / "experiment_data" / "reports"

    @property
    def state_root(self) -> Path:
        return self.results_root / "experiment_data" / "session12_repeat_watch_0907"

    @property
    def source_root(self) -> Path:
        return self.state_root / "sources"

    @property
    def lock_root(self) -> Path:
        return self.state_root / "locks"


@dataclass(frozen=True)
class EvalResult:
    target: Target
    status: str
    returncode: int
    log_path: Path | None = None
    detail: str = ""


def timestamp() -> str:
    return dt.datetime.now().strftime("%F %T")


def announce(message: str) -> None:
    print(f"[{timestamp()}] {message}", flush=True)


def positive_int(raw: str) -> int:
    value = int(raw)
    if value <= 0:
        raise argparse.ArgumentTypeError("必须是正整数")
    return value


def positive_float(raw: str) -> float:
    value = float(raw)
    if value <= 0:
        raise argparse.ArgumentTypeError("必须是正数")
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="监控 0907 三个未完成实验，在 session_12 快照完整后停仿真并接力复评",
    )
    parser.add_argument("--results-root", default="results", help="远程实验的 results 根目录")
    parser.add_argument("--poll-seconds", type=positive_float, default=15.0, help="轮询间隔，默认 15 秒")
    parser.add_argument("--retry-seconds", type=positive_float, default=300.0, help="复评失败重试间隔，默认 300 秒")
    parser.add_argument("--term-timeout-seconds", type=positive_float, default=30.0, help="SIGTERM 后等待时限，默认 30 秒")
    parser.add_argument("--repeat", type=positive_int, default=10, help="长量表完整复评次数，默认 10")
    parser.add_argument(
        "--intermediate-scale-repeats",
        type=positive_int,
        default=5,
        help="中间短量表完整复评次数，默认 5",
    )
    parser.add_argument("--eval-max-parallel", type=positive_int, default=6, help="单组复评内部并行数，默认 6")
    parser.add_argument(
        "--max-concurrent-evals",
        type=positive_int,
        default=3,
        help="同时进行的独立复评组数，默认 3",
    )
    parser.add_argument(
        "--keep-raw-artifacts",
        action="store_true",
        help="保留复评 job、trace 和 experiment_data 阶段快照副本",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="实际终止进程并启动复评；省略时只读扫描一次",
    )
    return parser.parse_args()


def resolve_config(args: argparse.Namespace) -> Config:
    results_root = Path(args.results_root).expanduser()
    if not results_root.is_absolute():
        results_root = BASE_DIR / results_root
    return Config(
        results_root=results_root.resolve(),
        poll_seconds=float(args.poll_seconds),
        retry_seconds=float(args.retry_seconds),
        term_timeout_seconds=float(args.term_timeout_seconds),
        repeat=int(args.repeat),
        intermediate_scale_repeats=int(args.intermediate_scale_repeats),
        eval_max_parallel=int(args.eval_max_parallel),
        max_concurrent_evals=min(len(TARGETS), int(args.max_concurrent_evals)),
        cleanup_completed_artifacts=not bool(args.keep_raw_artifacts),
        execute=bool(args.execute),
    )


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON 顶层不是对象: {path}")
    return payload


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp_path, path)


def checkpoint_dir(target: Target, cfg: Config) -> Path:
    return cfg.checkpoints_root / target.run_name


def validate_controller(
    target: Target,
    cfg: Config,
    batch_state_path: Path,
) -> tuple[Path, dict[str, Any]]:
    """Load the exact controller manifest, tolerating old live checkpoints.

    Current batch runs copy the manifest to the checkpoint before simulation,
    but a few already-running 0907 jobs were started before that safeguard.
    Their batch-state manifest is the authoritative source and has the same
    identity fields required by archived repeat evaluation.
    """
    state = load_json(batch_state_path)
    candidates = [checkpoint_dir(target, cfg) / CONTROLLER_MANIFEST_FILENAME]
    manifest_path = str(state.get("manifest_path", "") or "").strip()
    if manifest_path:
        candidate = Path(manifest_path).expanduser()
        if not candidate.is_absolute():
            candidate = BASE_DIR / candidate
        candidates.append(candidate)
    candidates.append(batch_state_path.parent / "manifests" / f"{target.condition_key}.json")

    path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if path is None:
        raise FileNotFoundError(
            "找不到 controller manifest（已检查 checkpoint 与 batch_state/manifests）: "
            + "; ".join(str(candidate) for candidate in candidates)
        )
    manifest = load_json(path)
    expected = {
        "artifact_kind": "cbt_experiment_condition",
        "condition_name": target.condition_name,
        "condition_key": target.condition_key,
        "run_name": target.run_name,
    }
    mismatches = {
        key: {"expected": value, "actual": manifest.get(key)}
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    if mismatches:
        raise ValueError(f"controller manifest identity 不一致: {mismatches}")
    return path.resolve(), manifest


def validate_batch_state(target: Target, cfg: Config) -> Path:
    root = cfg.results_root / "experiment_data" / "batch_state"
    matches: list[Path] = []
    for path in root.glob("*/*.json"):
        try:
            payload = load_json(path)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
            continue
        if (
            str(payload.get("run_name", "") or "") == target.run_name
            and str(payload.get("condition_name", "") or "") == target.condition_name
            and str(payload.get("condition_key", "") or "") == target.condition_key
        ):
            matches.append(path.resolve())
    if len(matches) != 1:
        raise ValueError(f"要求唯一匹配 batch_state，实际 {len(matches)} 个")
    return matches[0]


def validate_staged_node(target: Target, cfg: Config, label: str) -> dict[str, Any]:
    node_dir = checkpoint_dir(target, cfg) / "staged_eval" / label
    job = load_json(node_dir / "job.json")
    metadata = load_json(node_dir / "metadata.json")
    expected_count = 0 if label == "T0" else int(label.removeprefix("session_"))
    expected_job = {
        "run_name": target.run_name,
        "trigger_label": label,
        "completed_session_count": expected_count,
    }
    expected_metadata = {
        "status": "ok",
        "trigger_label": label,
        "completed_session_count": expected_count,
        "execution_mode": "capture_only",
        "artifact_kind": "repeat_eval_snapshot",
        "evaluation_executed": False,
    }
    mismatches: dict[str, Any] = {}
    for prefix, payload, expected in (
        ("job", job, expected_job),
        ("metadata", metadata, expected_metadata),
    ):
        for key, value in expected.items():
            if payload.get(key) != value:
                mismatches[f"{prefix}.{key}"] = {"expected": value, "actual": payload.get(key)}
    if mismatches:
        raise ValueError(f"{label} 节点不一致: {mismatches}")
    validate_snapshot_bundle(node_dir, job, label)
    return {
        "trigger_label": label,
        "completed_session_count": expected_count,
        "sim_time": str(job.get("sim_time", "") or ""),
        "snapshot_name": str(job.get("snapshot_name", "") or ""),
        "source": "staged_eval_snapshot",
        "evaluation_executed": False,
        "scales": {},
    }


def validate_ready(target: Target, cfg: Config) -> tuple[list[dict[str, Any]], Path, dict[str, Any], Path]:
    cp_dir = checkpoint_dir(target, cfg)
    if not cp_dir.is_dir():
        raise FileNotFoundError(f"checkpoint 尚未出现: {cp_dir}")
    # Keep the frequent polling path cheap.  Full bundle verification hashes
    # every captured storage file, so do it only after session_12 is visibly
    # complete and the index contains the whole requested trajectory.
    session12_dir = cp_dir / "staged_eval" / "session_12"
    session12_metadata = load_json(session12_dir / "metadata.json")
    if (
        session12_metadata.get("status") != "ok"
        or int(session12_metadata.get("completed_session_count", -1)) != 12
    ):
        raise ValueError("session_12 metadata 尚未完成")
    index = load_json(cp_dir / "staged_eval" / "index.json")
    ok_labels = {
        str(item.get("trigger_label", "") or "")
        for item in index.get("records", [])
        if isinstance(item, dict) and item.get("status") == "ok"
    }
    missing = [label for label in LABELS if label not in ok_labels]
    if missing:
        raise ValueError(f"staged_eval/index.json 尚未登记: {', '.join(missing)}")
    batch_state_path = validate_batch_state(target, cfg)
    controller_path, controller = validate_controller(target, cfg, batch_state_path)
    evaluations = [validate_staged_node(target, cfg, label) for label in LABELS]
    return evaluations, controller_path, controller, batch_state_path


def proc_cmdline(pid: int) -> list[str]:
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return []
    return [part.decode("utf-8", errors="replace") for part in raw.split(b"\0") if part]


def process_exists(pid: int) -> bool:
    # A SIGTERM/SIGKILL-ed child can remain in /proc as state Z until its
    # run_batch_experiment parent calls wait().  It no longer executes and
    # must not block the hand-off to repeat evaluation.
    try:
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        closing_paren = stat.rfind(")")
        if closing_paren >= 0 and stat[closing_paren + 2 : closing_paren + 3] == "Z":
            return False
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return False
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, FileNotFoundError):
        return False
    except PermissionError:
        return True
    return True


def command_option(args: list[str], option: str) -> str:
    try:
        index = args.index(option)
    except ValueError:
        return ""
    return args[index + 1] if index + 1 < len(args) else ""


def python_script_matches(args: list[str], script_name: str) -> bool:
    return any(Path(arg).name == script_name for arg in args[1:])


def matching_simulation_pids(target: Target) -> list[int]:
    matches: list[int] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if pid == os.getpid():
            continue
        args = proc_cmdline(pid)
        if (
            args
            and python_script_matches(args, "start.py")
            and command_option(args, "--name") == target.run_name
        ):
            matches.append(pid)
    return sorted(matches)


def matching_eval_pids(target: Target) -> list[int]:
    matches: list[int] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if pid == os.getpid():
            continue
        args = proc_cmdline(pid)
        if (
            args
            and python_script_matches(args, "run_archived_repeat_scale_eval.py")
            and command_option(args, "--name") == target.repeat_name
        ):
            matches.append(pid)
    return sorted(matches)


def terminate_exact_simulation(target: Target, cfg: Config) -> list[int]:
    pids = matching_simulation_pids(target)
    if not pids:
        announce(f"[INFO] {target.condition_name}: 未发现仍存活的精确 start.py 进程，直接接力复评")
        return []
    announce(f"[STOP] {target.condition_name}: SIGTERM exact start.py pid={','.join(map(str, pids))}")
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + cfg.term_timeout_seconds
    remaining = [pid for pid in pids if process_exists(pid)]
    while remaining and time.monotonic() < deadline:
        time.sleep(0.25)
        remaining = [pid for pid in remaining if process_exists(pid)]
    if remaining:
        announce(f"[STOP] {target.condition_name}: SIGKILL timeout pid={','.join(map(str, remaining))}")
        for pid in remaining:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        hard_deadline = time.monotonic() + 5.0
        while remaining and time.monotonic() < hard_deadline:
            time.sleep(0.25)
            remaining = [pid for pid in remaining if process_exists(pid)]
    if remaining:
        raise RuntimeError(f"精确仿真进程未退出: {remaining}")
    return pids


def source_summary_path(target: Target, cfg: Config) -> Path:
    return cfg.source_root / f"{target.repeat_name}_source.json"


def build_source_summary(
    target: Target,
    evaluations: list[dict[str, Any]],
    controller_path: Path,
    controller: dict[str, Any],
    batch_state_path: Path,
) -> dict[str, Any]:
    condition: dict[str, Any] = {
        "condition_name": target.condition_name,
        "condition_key": target.condition_key,
        "run_name": target.run_name,
        "run_dir": "",
        "variant": target.variant,
        "group": target.group,
        "severity": target.severity,
        "controller_manifest_path": str(controller_path),
        "controller_manifest": controller,
        "evaluations": evaluations,
        "final_deltas": {},
    }
    for key in IDENTITY_KEYS:
        condition[key] = copy.deepcopy(controller.get(key))
    return {
        "schema_version": 1,
        "artifact_kind": "session12_early_stop_repeat_source",
        "batch_name": "batch-0907-01",
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "source_batch_state": str(batch_state_path),
        "controller_manifest_sha256": canonical_json_sha256(controller),
        "conditions": [condition],
        "trigger_labels": list(LABELS),
        "warnings": [
            "主仿真在 session_12 capture-only 快照完整后按运维请求提前终止。",
            "未运行原批处理的 merge/compress/memory visualization/external-memory audit；本报告直接从严格历史快照复评。",
        ],
    }


def ensure_source_summary(
    target: Target,
    cfg: Config,
    evaluations: list[dict[str, Any]],
    controller_path: Path,
    controller: dict[str, Any],
    batch_state_path: Path,
) -> Path:
    path = source_summary_path(target, cfg)
    payload = build_source_summary(
        target,
        evaluations,
        controller_path,
        controller,
        batch_state_path,
    )
    write_json_atomic(path, payload)
    return path.resolve()


def report_path(target: Target, cfg: Config) -> Path:
    return cfg.reports_root / f"{target.repeat_name}_summary.json"


def report_complete(target: Target, cfg: Config) -> bool:
    path = report_path(target, cfg)
    if not path.is_file():
        return False
    try:
        report = load_json(path)
        completion = report.get("completion")
        conditions = report.get("conditions")
        protocol = report.get("evaluation_protocol")
        report_labels = report.get("labels")
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    return bool(
        report.get("batch_name") == target.repeat_name
        and int(report.get("repeat", -1)) == cfg.repeat
        and isinstance(protocol, dict)
        and int(protocol.get("intermediate_scale_repeats", -1)) == cfg.intermediate_scale_repeats
        and isinstance(report_labels, list)
        and set(LABELS).issubset(str(label) for label in report_labels)
        and isinstance(completion, dict)
        and completion.get("ready_for_final_report") is True
        and isinstance(conditions, list)
        and any(
            isinstance(item, dict)
            and item.get("condition_name") == target.condition_name
            and item.get("run_name") == target.run_name
            for item in conditions
        )
    )


def build_eval_command(target: Target, cfg: Config, summary_path: Path) -> list[str]:
    command = [
        sys.executable,
        str(BASE_DIR / "runshells" / "run_archived_repeat_scale_eval.py"),
        "--archive-results-root",
        str(cfg.results_root),
        "--condition",
        target.condition_name,
        "--original-summary",
        str(summary_path),
        "--labels",
        ",".join(LABELS),
        "--repeat",
        str(cfg.repeat),
        "--intermediate-scale-repeats",
        str(cfg.intermediate_scale_repeats),
        "--name",
        target.repeat_name,
        "--max-parallel",
        str(cfg.eval_max_parallel),
        "--resume-partial",
        "--require-controller-manifest",
    ]
    if cfg.cleanup_completed_artifacts:
        command.append("--cleanup-completed-artifacts")
    return command


def shell_command(command: Iterable[str]) -> str:
    return " ".join(shlex.quote(str(item)) for item in command)


def write_stop_record(target: Target, cfg: Config, pids: list[int], summary_path: Path) -> None:
    payload = {
        "schema_version": 1,
        "artifact_kind": "session12_early_stop",
        "stopped_at": dt.datetime.now().isoformat(timespec="seconds"),
        "condition_name": target.condition_name,
        "condition_key": target.condition_key,
        "run_name": target.run_name,
        "signal": "SIGTERM_then_SIGKILL_on_timeout",
        "matched_start_pids": pids,
        "labels": list(LABELS),
        "repeat_name": target.repeat_name,
        "source_summary": str(summary_path),
    }
    write_json_atomic(cfg.state_root / f"{target.run_name}_early_stop.json", payload)


def run_evaluation(target: Target, cfg: Config, summary_path: Path) -> EvalResult:
    cfg.lock_root.mkdir(parents=True, exist_ok=True)
    lock_path = cfg.lock_root / f"{target.repeat_name}.lock"
    with lock_path.open("a+", encoding="utf-8") as lock_handle:
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return EvalResult(target, "locked", 0, detail=str(lock_path))
        if report_complete(target, cfg):
            return EvalResult(target, "already_complete", 0)
        command = build_eval_command(target, cfg, summary_path)
        log_path = cfg.results_root / f"{target.repeat_name}.log"
        try:
            with log_path.open("a", encoding="utf-8") as log_handle:
                log_handle.write(f"\n[{timestamp()}] START {shell_command(command)}\n")
                log_handle.flush()
                result = subprocess.run(
                    command,
                    cwd=BASE_DIR,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=False,
                )
                log_handle.write(f"[{timestamp()}] EXIT returncode={result.returncode}\n")
            if result.returncode != 0:
                return EvalResult(target, "failed", result.returncode, log_path)
            if not report_complete(target, cfg):
                return EvalResult(target, "failed", 1, log_path, "返回 0，但正式报告完整性校验失败")
            return EvalResult(target, "completed", 0, log_path)
        finally:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)


def readonly_scan(cfg: Config) -> int:
    announce("未提供 --execute：只读扫描一次，不会发送信号或启动复评")
    for target in TARGETS:
        if report_complete(target, cfg):
            announce(f"[DONE] {target.condition_name}: repeat report 已完整")
            continue
        try:
            evaluations, controller_path, controller, batch_state_path = validate_ready(target, cfg)
        except Exception as exc:
            announce(f"[WAIT] {target.condition_name}: {exc}")
            continue
        summary = source_summary_path(target, cfg).resolve()
        command = build_eval_command(target, cfg, summary)
        announce(
            f"[READY] {target.condition_name}: session_12 严格快照完整；"
            f"start_pids={matching_simulation_pids(target) or '-'}\n"
            f"  would write {summary}\n"
            f"  would run {shell_command(command)}"
        )
        del evaluations, controller_path, controller, batch_state_path
    return 0


def watch(cfg: Config) -> int:
    if not cfg.execute:
        return readonly_scan(cfg)
    if not cfg.results_root.is_dir():
        raise FileNotFoundError(f"results 根目录不存在: {cfg.results_root}")

    cfg.state_root.mkdir(parents=True, exist_ok=True)
    watcher_lock_path = cfg.state_root / "watcher.lock"
    watcher_lock = watcher_lock_path.open("a+", encoding="utf-8")
    try:
        try:
            fcntl.flock(watcher_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"已有另一个 0907 session12 watcher 在运行: {watcher_lock_path}") from exc
        watcher_lock.seek(0)
        watcher_lock.truncate()
        watcher_lock.write(f"pid={os.getpid()} started_at={timestamp()}\n")
        watcher_lock.flush()

        announce(
            "开始监控 0907 三组实验: "
            f"poll={cfg.poll_seconds:g}s repeat={cfg.repeat}/{cfg.intermediate_scale_repeats} "
            f"eval_parallel={cfg.eval_max_parallel} concurrent_evals={cfg.max_concurrent_evals}"
        )
        completed: set[str] = set()
        submitted: set[str] = set()
        last_wait_reason: dict[str, str] = {}
        retry_after: dict[str, float] = {}
        active: dict[concurrent.futures.Future[EvalResult], Target] = {}
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=cfg.max_concurrent_evals)
        try:
            while len(completed) < len(TARGETS):
                for future in list(active):
                    if not future.done():
                        continue
                    target = active.pop(future)
                    try:
                        result = future.result()
                    except Exception as exc:
                        result = EvalResult(target, "failed", 1, detail=str(exc))
                    if result.status in {"completed", "already_complete"}:
                        completed.add(target.run_name)
                        announce(f"[DONE] {target.condition_name}: {report_path(target, cfg)}")
                    else:
                        submitted.discard(target.run_name)
                        retry_after[target.run_name] = time.monotonic() + cfg.retry_seconds
                        announce(
                            f"[ERROR] {target.condition_name}: repeat eval {result.status} "
                            f"returncode={result.returncode} detail={result.detail or '-'} "
                            f"log={result.log_path or '-'}"
                        )

                active_names = {target.run_name for target in active.values()}
                for target in TARGETS:
                    if target.run_name in completed or target.run_name in active_names:
                        continue
                    if report_complete(target, cfg):
                        completed.add(target.run_name)
                        announce(f"[DONE] {target.condition_name}: 已有完整 repeat report")
                        continue
                    external_eval_pids = matching_eval_pids(target)
                    if external_eval_pids:
                        reason = f"已有 repeat eval pid={','.join(map(str, external_eval_pids))}，等待其完成"
                        if last_wait_reason.get(target.run_name) != reason:
                            announce(f"[WAIT] {target.condition_name}: {reason}")
                            last_wait_reason[target.run_name] = reason
                        continue
                    if retry_after.get(target.run_name, 0.0) > time.monotonic():
                        continue
                    try:
                        evaluations, controller_path, controller, batch_state_path = validate_ready(target, cfg)
                    except Exception as exc:
                        reason = str(exc)
                        if last_wait_reason.get(target.run_name) != reason:
                            announce(f"[WAIT] {target.condition_name}: {reason}")
                            last_wait_reason[target.run_name] = reason
                        continue
                    last_wait_reason.pop(target.run_name, None)
                    if target.run_name in submitted:
                        continue
                    pids = terminate_exact_simulation(target, cfg)
                    summary_path = ensure_source_summary(
                        target,
                        cfg,
                        evaluations,
                        controller_path,
                        controller,
                        batch_state_path,
                    )
                    write_stop_record(target, cfg, pids, summary_path)
                    future = executor.submit(run_evaluation, target, cfg, summary_path)
                    active[future] = target
                    submitted.add(target.run_name)
                    announce(
                        f"[START] {target.condition_name}: session_12 完整，已停止主仿真并启动复评 "
                        f"labels={','.join(LABELS)} log={cfg.results_root / (target.repeat_name + '.log')}"
                    )
                if len(completed) < len(TARGETS):
                    time.sleep(cfg.poll_seconds)
        finally:
            executor.shutdown(wait=True, cancel_futures=False)
        announce("三个 0907 实验均已生成并通过正式 repeat report 完整性校验")
        return 0
    finally:
        try:
            fcntl.flock(watcher_lock.fileno(), fcntl.LOCK_UN)
        finally:
            watcher_lock.close()


def main() -> None:
    try:
        raise SystemExit(watch(resolve_config(parse_args())))
    except KeyboardInterrupt:
        raise SystemExit("收到中断；已启动的复评子进程将由当前 Python 退出流程等待收尾。")
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"错误: {exc}") from exc


if __name__ == "__main__":
    main()
