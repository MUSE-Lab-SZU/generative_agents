from __future__ import annotations

import copy
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any, Dict, List

from runshells.artifact_digest import canonical_json_sha256, file_sha256
from runshells.scale_protocol import LONG_SCALE_NAMES, SCALE_SPECS, SHORT_SCALE_NAMES


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
QUESTIONS_ROOT = os.path.join(
    BASE_DIR,
    "customization",
    "depression_scale_agent",
    "questions",
    "templates",
)
DEFAULT_WORKER_SCRIPT = "runshells/run_staged_eval_worker.py"
SCALE_QUESTION_FILES = {
    scale_name: str(scale_spec["question_file"])
    for scale_name, scale_spec in SCALE_SPECS.items()
}
SNAPSHOT_BUNDLE_SCHEMA_VERSION = 1
SNAPSHOT_STORAGE_DIRNAME = "snapshot_storage"
SNAPSHOT_MANIFEST_FILENAME = "snapshot_manifest.json"
ROLLING_RESUME_DIRNAME = "recovery_checkpoint"
ROLLING_RESUME_LATEST_DIRNAME = "latest"
ROLLING_RESUME_ARTIFACT_KIND = "rolling_resume_checkpoint"


class StagedEvalManager:
    def __init__(self, run_name: str, checkpoints_folder: str, config: Dict[str, Any], logger: Any = None):
        self.run_name = str(run_name or "").strip()
        self.checkpoints_folder = checkpoints_folder
        self.config = config if isinstance(config, dict) else {}
        self.logger = logger
        self.output_dir = os.path.join(self.checkpoints_folder, "staged_eval")
        self.state = self.config.setdefault("staged_eval_state", {})
        self._running_proc: subprocess.Popen | None = None
        self._running_log_handle = None
        self._step_trigger_enqueued = False
        self._ensure_state_schema()
        self._load_state_file()
        self._recover_inflight_state()

    def begin_step(self) -> None:
        self._step_trigger_enqueued = False

    def maybe_run_t0(
        self,
        runtime_config: Dict[str, Any],
        conversation: Dict[str, Any],
        step_no: int,
        sim_time: str,
    ) -> Dict[str, Any] | None:
        self.poll()
        if not self._enabled():
            return None
        if not bool(self._cfg().get("include_t0", False)):
            return None
        if bool(self.state.get("t0_done", False)):
            return None
        try:
            return self._enqueue_trigger(
                trigger_label="T0",
                completed_session_count=0,
                runtime_config=runtime_config,
                conversation=conversation,
                step_no=step_no,
                sim_time=sim_time,
                snapshot_name="",
            )
        except Exception as exc:
            if self._capture_only():
                raise
            self._log("warning", f"[STAGED_EVAL] T0 failed: {exc}")
            return None

    def maybe_run_post_step_eval(
        self,
        runtime_config: Dict[str, Any],
        conversation: Dict[str, Any],
        step_no: int,
        sim_time: str,
        snapshot_name: str,
    ) -> Dict[str, Any] | None:
        self.poll()
        if not self._enabled():
            return None

        last_result = None
        step_result = self._maybe_run_step_interval_trigger(
            runtime_config=runtime_config,
            conversation=conversation,
            step_no=step_no,
            sim_time=sim_time,
            snapshot_name=snapshot_name,
        )
        if step_result is not None:
            last_result = step_result

        session_result = self._maybe_run_session_interval_trigger(
            runtime_config=runtime_config,
            conversation=conversation,
            step_no=step_no,
            sim_time=sim_time,
            snapshot_name=snapshot_name,
        )
        if session_result is not None:
            last_result = session_result

        t4_result = self._maybe_run_t4_trigger(
            runtime_config=runtime_config,
            conversation=conversation,
            step_no=step_no,
            sim_time=sim_time,
            snapshot_name=snapshot_name,
        )
        if t4_result is not None:
            last_result = t4_result

        return last_result

    def poll(self) -> Dict[str, Any] | None:
        result = self._poll_worker()
        self._maybe_start_next_job()
        return result

    def drain(self) -> List[Dict[str, Any]]:
        completed: List[Dict[str, Any]] = []
        while True:
            result = self.poll()
            if result is not None:
                completed.append(result)
            if self._running_proc is None and not self.state.get("pending_jobs"):
                break
            time.sleep(1)
        return completed

    def was_trigger_enqueued_this_step(self) -> bool:
        return bool(self._step_trigger_enqueued)

    def should_capture_rolling_resume_checkpoint(
        self,
        runtime_config: Dict[str, Any],
    ) -> bool:
        cfg = self._rolling_resume_cfg()
        if not bool(cfg.get("enabled", False)):
            return False
        completed_session_count = self._compute_completed_session_count(runtime_config)
        if completed_session_count <= 0:
            return False
        latest_job_path = os.path.join(
            self.checkpoints_folder,
            ROLLING_RESUME_DIRNAME,
            ROLLING_RESUME_LATEST_DIRNAME,
            "job.json",
        )
        try:
            latest_job = self._load_worker_result(latest_job_path)
            latest_count = self._safe_int(
                latest_job.get("completed_session_count", 0),
                0,
            )
        except Exception:
            latest_count = 0
        return completed_session_count > latest_count

    def capture_rolling_resume_checkpoint(
        self,
        runtime_config: Dict[str, Any],
        conversation: Dict[str, Any],
        step_no: int,
        sim_time: str,
        snapshot_name: str,
    ) -> Dict[str, Any] | None:
        if not self.should_capture_rolling_resume_checkpoint(runtime_config):
            return None

        completed_session_count = self._compute_completed_session_count(runtime_config)
        resume_root = os.path.join(self.checkpoints_folder, ROLLING_RESUME_DIRNAME)
        os.makedirs(resume_root, exist_ok=True)
        for entry_name in os.listdir(resume_root):
            if entry_name.startswith(".latest-"):
                shutil.rmtree(
                    os.path.join(resume_root, entry_name),
                    ignore_errors=True,
                )
        temp_dir = tempfile.mkdtemp(prefix=".latest-", dir=resume_root)
        latest_dir = os.path.join(resume_root, ROLLING_RESUME_LATEST_DIRNAME)
        previous_dir = os.path.join(resume_root, ".previous")
        promoted = False
        try:
            snapshot_bundle, _storage_root = self._capture_snapshot_bundle(
                trigger_label=f"session_{completed_session_count}",
                trigger_dir=temp_dir,
                runtime_config=runtime_config,
                conversation=conversation,
                step_no=step_no,
                sim_time=sim_time,
                snapshot_name=snapshot_name,
                artifact_kind=ROLLING_RESUME_ARTIFACT_KIND,
            )
            job = {
                "schema_version": 1,
                "artifact_kind": ROLLING_RESUME_ARTIFACT_KIND,
                "run_name": self.run_name,
                "trigger_label": f"session_{completed_session_count}",
                "completed_session_count": int(completed_session_count),
                "step_no": int(step_no),
                "sim_time": str(sim_time or ""),
                "snapshot_name": str(snapshot_name or ""),
                "runtime_config": copy.deepcopy(runtime_config),
                "conversation": copy.deepcopy(conversation or {}),
                "snapshot_bundle": snapshot_bundle,
            }
            self._write_json_atomic(os.path.join(temp_dir, "job.json"), job)

            if os.path.exists(previous_dir):
                shutil.rmtree(previous_dir)
            if os.path.exists(latest_dir):
                os.replace(latest_dir, previous_dir)
            try:
                os.replace(temp_dir, latest_dir)
                promoted = True
            except BaseException:
                if os.path.exists(previous_dir) and not os.path.exists(latest_dir):
                    os.replace(previous_dir, latest_dir)
                raise
            if os.path.exists(previous_dir):
                shutil.rmtree(previous_dir)
            self._log(
                "info",
                "[ROLLING_RESUME_CHECKPOINT] session_count={} step={} snapshot={} path={}".format(
                    completed_session_count,
                    step_no,
                    snapshot_name,
                    latest_dir,
                ),
            )
            return job
        finally:
            if not promoted and os.path.isdir(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)

    def is_t4_done(self) -> bool:
        return bool(self.state.get("t4_done", False))

    def should_auto_stop_after_t4_done(self) -> bool:
        return bool(
            self._enabled()
            and self._cfg().get("auto_stop_after_t4_done", False)
            and self.is_t4_done()
        )

    def abort_pending_work(self, reason: str = "") -> None:
        self._log("warning", "[STAGED_EVAL] abort_pending_work reason={}".format(str(reason or "")))
        self._terminate_running_worker()
        running_job = self.state.get("running_job")
        if isinstance(running_job, dict) and running_job:
            self._requeue_running_job(running_job)
        self.state["running_job"] = None
        self._persist_state()

    def _enqueue_trigger(
        self,
        trigger_label: str,
        completed_session_count: int,
        runtime_config: Dict[str, Any],
        conversation: Dict[str, Any],
        step_no: int,
        sim_time: str,
        snapshot_name: str,
        extra_metadata: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        target_agent = self._target_agent()
        if not target_agent:
            raise ValueError("staged_eval.target_agent is required")
        if target_agent not in (runtime_config.get("agents", {}) or {}):
            raise ValueError(f"target agent not found in runtime config: {target_agent}")
        if self._is_label_finished_or_pending(trigger_label):
            return self._load_existing_metadata(trigger_label)

        os.makedirs(self.output_dir, exist_ok=True)
        trigger_dir = self._build_trigger_dir(trigger_label)
        worker_paths = self._build_worker_paths(trigger_dir)
        worker_cfg = self._worker_cfg()
        scales = self._scales_for_trigger(
            trigger_label,
            runtime_config,
            extra_metadata=extra_metadata,
        )
        scale_question_files = self._resolve_scale_question_files(scales)

        snapshot_bundle = None
        storage_source_root = os.path.join(self.checkpoints_folder, "storage")
        if self._capture_only():
            self._cleanup_capture_only_eval_artifacts(trigger_dir)
            snapshot_bundle, storage_source_root = self._capture_snapshot_bundle(
                trigger_label=trigger_label,
                trigger_dir=trigger_dir,
                runtime_config=runtime_config,
                conversation=conversation,
                step_no=step_no,
                sim_time=sim_time,
                snapshot_name=snapshot_name,
            )

        job_payload = self._build_worker_job_payload(
            trigger_label=trigger_label,
            completed_session_count=completed_session_count,
            runtime_config=runtime_config,
            conversation=conversation,
            step_no=step_no,
            sim_time=sim_time,
            snapshot_name=snapshot_name,
            target_agent=target_agent,
            trigger_dir=trigger_dir,
            worker_result_path=worker_paths["result"],
            scales=scales,
            scale_question_files=scale_question_files,
            worker_cfg=worker_cfg,
            storage_source_root=storage_source_root,
            snapshot_bundle=snapshot_bundle,
        )
        self._write_json(worker_paths["job"], job_payload)
        metadata = self._build_trigger_queue_metadata(
            trigger_label=trigger_label,
            completed_session_count=completed_session_count,
            runtime_config=runtime_config,
            step_no=step_no,
            sim_time=sim_time,
            snapshot_name=snapshot_name,
            worker_paths=worker_paths,
            worker_cfg=worker_cfg,
            extra_metadata=extra_metadata,
        )
        if self._capture_only():
            metadata.update(
                {
                    "status": "ok",
                    "execution_mode": "capture_only",
                    "artifact_kind": "repeat_eval_snapshot",
                    "evaluation_executed": False,
                    "worker_mode": "capture_only",
                    "worker_status": "not_started",
                    "worker_script": "",
                    "worker_log_file": "",
                    "worker_result_file": "",
                    "snapshot_bundle": copy.deepcopy(snapshot_bundle or {}),
                    "scales": {},
                }
            )
            if trigger_label == "T0":
                metadata["snapshot_phase"] = "after_initial_injection_before_first_think"
            self._write_json(os.path.join(trigger_dir, "metadata.json"), metadata)
            self._step_trigger_enqueued = True
            self._mark_trigger_done(metadata, trigger_dir)
            self._persist_state()
            self._log(
                "info",
                "[STAGED_EVAL_CAPTURE] trigger={} target={} completed_sessions={} step={} sim_time={}".format(
                    trigger_label,
                    target_agent,
                    completed_session_count,
                    step_no,
                    sim_time,
                ),
            )
            return metadata

        self._write_json(os.path.join(trigger_dir, "metadata.json"), metadata)
        self._enqueue_job_record(
            trigger_label=trigger_label,
            trigger_dir=trigger_dir,
            worker_paths=worker_paths,
            worker_cfg=worker_cfg,
        )
        self._step_trigger_enqueued = True
        self._persist_state()
        self._maybe_start_next_job()
        self._log(
            "info",
            "[STAGED_EVAL] trigger={} queued target={} completed_sessions={} step={} sim_time={}".format(
                trigger_label,
                target_agent,
                completed_session_count,
                step_no,
                sim_time,
            ),
        )
        return metadata

    def _compute_completed_session_count(self, runtime_config: Dict[str, Any]) -> int:
        if not isinstance(runtime_config, dict):
            return 0
        count, _ = self._resolve_completed_session_count(runtime_config)
        return count

    def _resolve_completed_session_count(self, runtime_config: Dict[str, Any]) -> tuple[int, str]:
        if not isinstance(runtime_config, dict):
            return 0, "none"

        resident_count = self._compute_resident_chat_completed_count(runtime_config)
        if resident_count > 0:
            return resident_count, "resident_chat"

        completed_meeting_count = self._compute_intervention_completed_meeting_count(runtime_config)
        if completed_meeting_count > 0:
            return completed_meeting_count, "intervention_completed_meeting"

        # completed_meeting_state counts every finished doctor consultation,
        # including post-treatment follow-ups that intentionally skip session
        # evaluation. Keep session_eval_state only as a compatibility fallback
        # for older checkpoints created before completed_meeting_state existed.
        doctor_count = self._compute_doctor_completed_meeting_count(runtime_config)
        if doctor_count > 0:
            return doctor_count, "doctor_completed_meeting"

        return 0, "none"

    def _compute_resident_chat_completed_count(self, runtime_config: Dict[str, Any]) -> int:
        if not isinstance(runtime_config, dict):
            return 0
        state = runtime_config.get("intervention_state", {}) or {}
        if not isinstance(state, dict):
            return 0
        resident_chat_state = state.get("resident_chat_state", {}) or {}
        if not isinstance(resident_chat_state, dict):
            return 0
        completed_counts = resident_chat_state.get("completed_counts_by_patient", {}) or {}
        if not isinstance(completed_counts, dict):
            return 0
        target_agent = self._target_agent()
        if not target_agent:
            return 0
        count = self._safe_int(completed_counts.get(target_agent, 0), 0)
        return max(0, count)

    def _compute_doctor_completed_meeting_count(self, runtime_config: Dict[str, Any]) -> int:
        if not isinstance(runtime_config, dict):
            return 0

        state = runtime_config.get("intervention_state", {}) or {}
        if not isinstance(state, dict):
            return 0

        session_eval_state = state.get("session_eval_state", {}) or {}
        if not isinstance(session_eval_state, dict):
            return 0

        history_by_pair = session_eval_state.get("history_by_pair", {}) or {}
        if not isinstance(history_by_pair, dict):
            return 0

        doctor_name = self._doctor_name(runtime_config)
        target_agent = self._target_agent()
        if not doctor_name or not target_agent:
            return 0

        pair_key = f"{doctor_name}::{target_agent}"
        history = history_by_pair.get(pair_key, [])
        if not isinstance(history, list) or not history:
            return 0

        completed_meeting_ids = []
        seen_meeting_ids = set()
        fallback_history_count = 0
        for item in history:
            if not isinstance(item, dict):
                continue
            fallback_history_count += 1
            meeting_id = str(item.get("meeting_id", "") or "").strip()
            if (not meeting_id) or (meeting_id in seen_meeting_ids):
                continue
            seen_meeting_ids.add(meeting_id)
            completed_meeting_ids.append(meeting_id)

        if completed_meeting_ids:
            return len(completed_meeting_ids)
        return fallback_history_count

    def _compute_intervention_completed_meeting_count(self, runtime_config: Dict[str, Any]) -> int:
        if not isinstance(runtime_config, dict):
            return 0

        state = runtime_config.get("intervention_state", {}) or {}
        if not isinstance(state, dict):
            return 0

        completed_state = state.get("completed_meeting_state", {}) or {}
        if not isinstance(completed_state, dict):
            return 0

        doctor_name = self._doctor_name(runtime_config)
        target_agent = self._target_agent()
        if not doctor_name or not target_agent:
            return 0

        pair_key = f"{doctor_name}::{target_agent}"
        meeting_ids_by_pair = completed_state.get("meeting_ids_by_pair", {}) or {}
        if isinstance(meeting_ids_by_pair, dict):
            meeting_ids = meeting_ids_by_pair.get(pair_key, [])
            if isinstance(meeting_ids, list):
                unique_ids = {
                    str(item or "").strip()
                    for item in meeting_ids
                    if str(item or "").strip()
                }
                if unique_ids:
                    return len(unique_ids)

        counts_by_pair = completed_state.get("counts_by_pair", {}) or {}
        if not isinstance(counts_by_pair, dict):
            return 0
        return max(0, self._safe_int(counts_by_pair.get(pair_key, 0), 0))

    def _maybe_run_session_interval_trigger(
        self,
        runtime_config: Dict[str, Any],
        conversation: Dict[str, Any],
        step_no: int,
        sim_time: str,
        snapshot_name: str,
    ) -> Dict[str, Any] | None:
        interval = self._safe_int(self._cfg().get("session_interval", 0), 0)
        if interval <= 0:
            return None

        completed_conversation_count = self._compute_completed_session_count(runtime_config)
        if completed_conversation_count <= 0:
            return None

        max_completed_sessions = self._safe_int(
            self._cfg().get("max_completed_sessions", 0),
            0,
        )
        if (
            max_completed_sessions > 0
            and completed_conversation_count > max_completed_sessions
            and not self._is_target_pair_completed(runtime_config)
        ):
            return None

        # Keep regular intermediate nodes on the configured interval, but do
        # not lose the actual treatment endpoint when its final session is
        # not an exact interval multiple.
        if (
            completed_conversation_count % interval != 0
            and not self._is_target_pair_completed(runtime_config)
        ):
            return None

        trigger_label = f"session_{completed_conversation_count}"
        if self._is_label_finished_or_pending(trigger_label):
            return None

        try:
            return self._enqueue_trigger(
                trigger_label=trigger_label,
                completed_session_count=completed_conversation_count,
                runtime_config=runtime_config,
                conversation=conversation,
                step_no=step_no,
                sim_time=sim_time,
                snapshot_name=snapshot_name,
            )
        except Exception as exc:
            if self._capture_only():
                raise
            self._log("warning", f"[STAGED_EVAL] {trigger_label} failed: {exc}")
            return None

    def _maybe_run_step_interval_trigger(
        self,
        runtime_config: Dict[str, Any],
        conversation: Dict[str, Any],
        step_no: int,
        sim_time: str,
        snapshot_name: str,
    ) -> Dict[str, Any] | None:
        step_cfg = self._step_interval_cfg()
        if not bool(step_cfg.get("enabled", False)):
            return None

        every_steps = max(0, self._safe_int(step_cfg.get("every_steps", 0), 0))
        if every_steps <= 0:
            return None

        current_step = max(0, self._safe_int(step_no, 0))
        anchor_step = max(0, self._safe_int(step_cfg.get("anchor_step", 0), 0))
        elapsed_steps = current_step - anchor_step
        if elapsed_steps <= 0 or elapsed_steps % every_steps != 0:
            return None

        trigger_index = elapsed_steps // every_steps
        max_triggers = max(0, self._safe_int(step_cfg.get("max_triggers", 0), 0))
        if max_triggers > 0 and trigger_index > max_triggers:
            return None

        virtual_session_interval = max(
            1,
            self._safe_int(step_cfg.get("virtual_session_interval", self._cfg().get("session_interval", 1)), 1),
        )
        label_value_mode = str(
            step_cfg.get("label_value_mode", "virtual_session") or "virtual_session"
        ).strip().lower()
        if label_value_mode not in {"virtual_session", "relative_step"}:
            raise ValueError(
                "unsupported staged_eval.step_interval.label_value_mode: {}".format(
                    label_value_mode
                )
            )
        if label_value_mode == "relative_step":
            completed_session_count = self._compute_completed_session_count(runtime_config)
            label_value = elapsed_steps
        else:
            completed_session_count = int(trigger_index * virtual_session_interval)
            label_value = completed_session_count
        label_prefix = str(step_cfg.get("label_prefix", "session") or "session").strip() or "session"
        trigger_label = f"{label_prefix}_{label_value}"
        if self._is_label_finished_or_pending(trigger_label):
            return None

        try:
            extra_metadata = {
                "trigger_mode": "step_interval",
                "step_interval_every_steps": every_steps,
                "step_interval_anchor_step": anchor_step,
                "step_interval_elapsed_steps": elapsed_steps,
                "step_interval_virtual_session_interval": virtual_session_interval,
                "step_interval_trigger_index": int(trigger_index),
                "step_interval_label_prefix": label_prefix,
                "step_interval_label_value_mode": label_value_mode,
                "followup_phase": label_value_mode == "relative_step",
            }
            if label_value_mode == "virtual_session":
                extra_metadata["completed_session_count_source"] = "step_interval"
            return self._enqueue_trigger(
                trigger_label=trigger_label,
                completed_session_count=completed_session_count,
                runtime_config=runtime_config,
                conversation=conversation,
                step_no=current_step,
                sim_time=sim_time,
                snapshot_name=snapshot_name,
                extra_metadata=extra_metadata,
            )
        except Exception as exc:
            if self._capture_only():
                raise
            self._log("warning", f"[STAGED_EVAL] {trigger_label} failed: {exc}")
            return None

    def _maybe_run_t4_trigger(
        self,
        runtime_config: Dict[str, Any],
        conversation: Dict[str, Any],
        step_no: int,
        sim_time: str,
        snapshot_name: str,
    ) -> Dict[str, Any] | None:
        if not bool(self._cfg().get("t4_enabled", False)):
            return None
        if bool(self.state.get("t4_done", False)) or self._is_label_finished_or_pending("T4"):
            self.state["t4_done"] = True
            self._persist_state()
            return None
        if not self._is_target_pair_completed(runtime_config):
            return None

        completion_step_no = self.state.get("t4_completion_step_no")
        if completion_step_no is None:
            completion_step_no = int(step_no)
            self.state["t4_completion_step_no"] = completion_step_no
            self.state["t4_completion_snapshot_name"] = str(snapshot_name or "")
            self._persist_state()

        after_steps = max(0, self._safe_int(self._cfg().get("t4_after_steps", 0), 0))
        if int(step_no) - int(completion_step_no) < after_steps:
            return None

        completed_session_count = self._compute_completed_session_count(runtime_config)
        try:
            return self._enqueue_trigger(
                trigger_label="T4",
                completed_session_count=completed_session_count,
                runtime_config=runtime_config,
                conversation=conversation,
                step_no=step_no,
                sim_time=sim_time,
                snapshot_name=snapshot_name,
                extra_metadata={
                    "completion_detected_step_no": int(completion_step_no),
                    "completion_detected_snapshot_name": str(self.state.get("t4_completion_snapshot_name", "") or ""),
                    "observation_steps_after_completion": int(step_no) - int(completion_step_no),
                },
            )
        except Exception as exc:
            if self._capture_only():
                raise
            self._log("warning", f"[STAGED_EVAL] T4 failed: {exc}")
            return None

    def _resolve_target_pair_state(self, runtime_config: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(runtime_config, dict):
            return {}
        state = runtime_config.get("intervention_state", {}) or {}
        session_prompt_state = state.get("session_prompt_state", {}) or {}
        pairs = session_prompt_state.get("pairs", {}) or {}
        if not isinstance(pairs, dict):
            return {}

        doctor_name = self._doctor_name(runtime_config)
        target_agent = self._target_agent()
        if not doctor_name or not target_agent:
            return {}

        pair_state = pairs.get(f"{doctor_name}::{target_agent}", {})
        return pair_state if isinstance(pair_state, dict) else {}

    def _is_target_pair_completed(self, runtime_config: Dict[str, Any]) -> bool:
        pair_state = self._resolve_target_pair_state(runtime_config)
        return bool(pair_state.get("completed", False))

    def _enabled(self) -> bool:
        return bool(self._cfg().get("enabled", False))

    def _execution_mode(self) -> str:
        mode = str(self._cfg().get("execution_mode", "evaluate") or "evaluate").strip().lower()
        if mode not in {"evaluate", "capture_only"}:
            raise ValueError("unsupported staged_eval.execution_mode: {}".format(mode))
        return mode

    def _capture_only(self) -> bool:
        return self._execution_mode() == "capture_only"

    def _cfg(self) -> Dict[str, Any]:
        cfg = self.config.get("staged_eval", {}) or {}
        return cfg if isinstance(cfg, dict) else {}

    def _rolling_resume_cfg(self) -> Dict[str, Any]:
        checkpointing = self.config.get("checkpointing", {}) or {}
        if not isinstance(checkpointing, dict):
            return {}
        cfg = checkpointing.get("rolling_resume", {}) or {}
        return cfg if isinstance(cfg, dict) else {}

    def _worker_cfg(self) -> Dict[str, Any]:
        cfg = self._cfg()
        return {
            "worker_script": str(cfg.get("worker_script", DEFAULT_WORKER_SCRIPT) or DEFAULT_WORKER_SCRIPT).strip(),
            "worker_timeout_seconds": max(
                1,
                self._safe_int(cfg.get("worker_timeout_seconds", 1800), 1800),
            ),
            "cleanup_tmp_storage": bool(cfg.get("cleanup_tmp_storage", True)),
        }

    def _step_interval_cfg(self) -> Dict[str, Any]:
        cfg = self._cfg().get("step_interval", {}) or {}
        return cfg if isinstance(cfg, dict) else {}

    def _target_agent(self) -> str:
        return str(self._cfg().get("target_agent", "") or "").strip()

    def _scales(self) -> List[str]:
        scales = self._cfg().get("scales", []) or []
        if not isinstance(scales, list):
            return []
        return [str(item or "").strip() for item in scales if str(item or "").strip()]

    def _scales_for_trigger(
        self,
        trigger_label: str,
        runtime_config: Dict[str, Any],
        *,
        extra_metadata: Dict[str, Any] | None = None,
    ) -> List[str]:
        if not bool(self._cfg().get("use_short_scales_for_intermediate_eval", False)):
            return self._scales()
        if trigger_label == "T0" or self._is_final_trigger(
            trigger_label,
            runtime_config,
            extra_metadata=extra_metadata,
        ):
            return list(LONG_SCALE_NAMES)
        return list(SHORT_SCALE_NAMES)

    def _is_final_trigger(
        self,
        trigger_label: str,
        runtime_config: Dict[str, Any],
        *,
        extra_metadata: Dict[str, Any] | None = None,
    ) -> bool:
        if trigger_label in {"T4", "POST"}:
            return True
        if self._is_target_pair_completed(runtime_config):
            return True
        completed_session_count = self._compute_completed_session_count(runtime_config)
        max_completed_sessions = self._safe_int(
            self._cfg().get("max_completed_sessions", 0),
            0,
        )
        if max_completed_sessions > 0 and completed_session_count == max_completed_sessions:
            return True
        metadata = extra_metadata if isinstance(extra_metadata, dict) else {}
        trigger_index = self._safe_int(metadata.get("step_interval_trigger_index", 0), 0)
        max_triggers = self._safe_int(self._step_interval_cfg().get("max_triggers", 0), 0)
        return max_triggers > 0 and trigger_index == max_triggers

    def _doctor_name(self, runtime_config: Dict[str, Any]) -> str:
        intervention_cfg = runtime_config.get("intervention", {}) or {}
        return str(intervention_cfg.get("doctor", "") or "").strip()

    def _build_trigger_dir(self, trigger_label: str) -> str:
        trigger_dir = os.path.join(self.output_dir, trigger_label)
        os.makedirs(trigger_dir, exist_ok=True)
        return trigger_dir

    def _build_worker_paths(self, trigger_dir: str) -> Dict[str, str]:
        return {
            "job": os.path.join(trigger_dir, "job.json"),
            "log": os.path.join(trigger_dir, "worker.log"),
            "result": os.path.join(trigger_dir, "worker_result.json"),
        }

    def _resolve_scale_question_files(self, scales: List[str] | None = None) -> Dict[str, str]:
        question_files: Dict[str, str] = {}
        for scale_name in self._scales() if scales is None else scales:
            question_file = SCALE_QUESTION_FILES.get(scale_name)
            if not question_file:
                raise ValueError(f"unsupported scale: {scale_name}")
            question_path = os.path.join(QUESTIONS_ROOT, question_file)
            if not os.path.exists(question_path):
                raise FileNotFoundError(f"question file not found: {question_path}")
            question_files[scale_name] = question_file
        return question_files

    def _build_worker_job_payload(
        self,
        trigger_label: str,
        completed_session_count: int,
        runtime_config: Dict[str, Any],
        conversation: Dict[str, Any],
        step_no: int,
        sim_time: str,
        snapshot_name: str,
        target_agent: str,
        trigger_dir: str,
        worker_result_path: str,
        scales: List[str],
        scale_question_files: Dict[str, str],
        worker_cfg: Dict[str, Any],
        storage_source_root: str,
        snapshot_bundle: Dict[str, Any] | None,
    ) -> Dict[str, Any]:
        payload = {
            "run_name": self.run_name,
            "api_cost": {
                "run_name": self.run_name,
                "phase": "simulation",
                "experiment_data_root": os.path.join(BASE_DIR, "results", "experiment_data"),
                "evaluation_id": "simulation:staged:{}:{}".format(self.run_name, trigger_label),
            },
            "trigger_label": trigger_label,
            "completed_session_count": int(completed_session_count),
            "step_no": int(step_no),
            "sim_time": str(sim_time or ""),
            "snapshot_name": str(snapshot_name or ""),
            "target_agent": target_agent,
            "scales": list(scales),
            "scale_question_files": copy.deepcopy(scale_question_files),
            "runtime_config": copy.deepcopy(runtime_config),
            "conversation": copy.deepcopy(conversation or {}),
            "trigger_dir": trigger_dir,
            "worker_result_path": worker_result_path,
            "storage_source_root": str(storage_source_root or ""),
            "tmp_root_parent": os.path.join(self.output_dir, "_tmp"),
            "cleanup_tmp_storage": bool(worker_cfg.get("cleanup_tmp_storage", True)),
        }
        if isinstance(snapshot_bundle, dict) and snapshot_bundle:
            payload["snapshot_bundle"] = copy.deepcopy(snapshot_bundle)
        return payload

    @staticmethod
    def _canonical_json_sha256(payload: Any) -> str:
        return canonical_json_sha256(payload)

    @staticmethod
    def _file_sha256(path: str) -> str:
        return file_sha256(path)

    def _storage_file_manifest(self, storage_root: str) -> tuple[List[Dict[str, Any]], int]:
        files: List[Dict[str, Any]] = []
        total_size = 0
        for current_root, dirnames, filenames in os.walk(storage_root):
            dirnames.sort()
            filenames.sort()
            for filename in filenames:
                path = os.path.join(current_root, filename)
                relative_path = os.path.relpath(path, storage_root).replace(os.sep, "/")
                size = os.path.getsize(path)
                files.append(
                    {
                        "path": relative_path,
                        "size": int(size),
                        "sha256": self._file_sha256(path),
                    }
                )
                total_size += int(size)
        return files, total_size

    def _capture_snapshot_bundle(
        self,
        trigger_label: str,
        trigger_dir: str,
        runtime_config: Dict[str, Any],
        conversation: Dict[str, Any],
        step_no: int,
        sim_time: str,
        snapshot_name: str,
        artifact_kind: str = "repeat_eval_snapshot",
    ) -> tuple[Dict[str, Any], str]:
        source_storage = os.path.join(self.checkpoints_folder, "storage")
        if not os.path.isdir(source_storage):
            raise FileNotFoundError("staged_eval capture source storage not found: {}".format(source_storage))

        snapshot_storage = os.path.join(trigger_dir, SNAPSHOT_STORAGE_DIRNAME)
        manifest_path = os.path.join(trigger_dir, SNAPSHOT_MANIFEST_FILENAME)
        temp_storage = tempfile.mkdtemp(prefix=".snapshot_storage-", dir=trigger_dir)
        try:
            shutil.copytree(source_storage, temp_storage, dirs_exist_ok=True)
            files, total_size = self._storage_file_manifest(temp_storage)
            manifest = {
                "schema_version": SNAPSHOT_BUNDLE_SCHEMA_VERSION,
                "artifact_kind": str(artifact_kind or "repeat_eval_snapshot"),
                "storage_scope": "full",
                "trigger_label": str(trigger_label or ""),
                "step_no": int(step_no),
                "sim_time": str(sim_time or ""),
                "snapshot_name": str(snapshot_name or ""),
                "runtime_config_sha256": self._canonical_json_sha256(runtime_config),
                "conversation_sha256": self._canonical_json_sha256(conversation or {}),
                "file_count": len(files),
                "total_size": int(total_size),
                "files": files,
            }
            if os.path.exists(snapshot_storage):
                shutil.rmtree(snapshot_storage)
            os.replace(temp_storage, snapshot_storage)
            temp_storage = ""
            self._write_json_atomic(manifest_path, manifest)
        except Exception:
            if temp_storage and os.path.isdir(temp_storage):
                shutil.rmtree(temp_storage, ignore_errors=True)
            raise

        bundle = {
            "schema_version": SNAPSHOT_BUNDLE_SCHEMA_VERSION,
            "artifact_kind": str(artifact_kind or "repeat_eval_snapshot"),
            "storage_scope": "full",
            "storage_relpath": SNAPSHOT_STORAGE_DIRNAME,
            "manifest_relpath": SNAPSHOT_MANIFEST_FILENAME,
            "runtime_config_sha256": manifest["runtime_config_sha256"],
            "conversation_sha256": manifest["conversation_sha256"],
        }
        return bundle, snapshot_storage

    def _cleanup_capture_only_eval_artifacts(self, trigger_dir: str) -> None:
        for filename in os.listdir(trigger_dir):
            path = os.path.join(trigger_dir, filename)
            if not os.path.isfile(path):
                continue
            is_worker_artifact = filename in {"worker.log", "worker_result.json"}
            is_scale_artifact = filename.endswith(
                ("_answered.jsonl", "_trace.json", "_scored.json", "_item_scored.json")
            )
            if is_worker_artifact or is_scale_artifact:
                os.unlink(path)

    def _resolve_worker_script_path(self, worker_script: str) -> str:
        path = str(worker_script or "").strip()
        if not path:
            path = DEFAULT_WORKER_SCRIPT
        if not os.path.isabs(path):
            path = os.path.join(BASE_DIR, path)
        return os.path.abspath(path)

    def _state_file_path(self) -> str:
        return os.path.join(self.output_dir, "state.json")

    def _persist_state(self) -> None:
        self.config["staged_eval_state"] = self.state
        os.makedirs(self.output_dir, exist_ok=True)
        self._write_json(self._state_file_path(), self.state)
        self._write_index()

    def _load_state_file(self) -> None:
        path = self._state_file_path()
        if not os.path.exists(path):
            return
        try:
            payload = self._load_worker_result(path)
        except Exception:
            return
        if not isinstance(payload, dict):
            return
        self.state.update(copy.deepcopy(payload))
        self.config["staged_eval_state"] = self.state
        self._ensure_state_schema()

    def _recover_inflight_state(self) -> None:
        running_job = self.state.get("running_job")
        if isinstance(running_job, dict) and running_job:
            self._requeue_running_job(running_job)
        self.state["running_job"] = None
        self._persist_state()

    def _build_trigger_queue_metadata(
        self,
        trigger_label: str,
        completed_session_count: int,
        runtime_config: Dict[str, Any],
        step_no: int,
        sim_time: str,
        snapshot_name: str,
        worker_paths: Dict[str, str],
        worker_cfg: Dict[str, Any],
        extra_metadata: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        _, completed_session_count_source = self._resolve_completed_session_count(runtime_config)
        trigger_dir = os.path.dirname(worker_paths["job"])
        metadata = {
            "status": "queued",
            "trigger_label": trigger_label,
            "completed_session_count": int(completed_session_count),
            "completed_session_count_source": completed_session_count_source,
            "step_no": int(step_no),
            "sim_time": str(sim_time or ""),
            "snapshot_name": str(snapshot_name or ""),
            "target_agent": self._target_agent(),
            "doctor_name": self._doctor_name(runtime_config),
            "session_interval": self._safe_int(self._cfg().get("session_interval", 0), 0),
            "max_completed_sessions": self._safe_int(self._cfg().get("max_completed_sessions", 0), 0),
            "step_interval": copy.deepcopy(self._step_interval_cfg()),
            "t4_enabled": bool(self._cfg().get("t4_enabled", False)),
            "t4_after_steps": max(0, self._safe_int(self._cfg().get("t4_after_steps", 0), 0)),
            "readonly_eval": True,
            "external_memory_read": False,
            "uses_answer_without_memory": True,
            "writes_blocked_by_design": False,
            "writes_isolated_via_temp_storage": True,
            "storage_isolation_enabled": True,
            "local_write_attempt_count": 0,
            "external_write_attempt_count": 0,
            "worker_mode": "subprocess_async",
            "worker_script": str(worker_cfg.get("worker_script", DEFAULT_WORKER_SCRIPT) or DEFAULT_WORKER_SCRIPT),
            "worker_timeout_seconds": self._safe_int(worker_cfg.get("worker_timeout_seconds", 1800), 1800),
            "cleanup_tmp_storage": bool(worker_cfg.get("cleanup_tmp_storage", True)),
            "worker_log_file": os.path.relpath(worker_paths["log"], trigger_dir),
            "worker_job_file": os.path.relpath(worker_paths["job"], trigger_dir),
            "worker_result_file": os.path.relpath(worker_paths["result"], trigger_dir),
            "worker_returncode": None,
            "worker_timed_out": False,
            "worker_duration_seconds": 0.0,
            "worker_status": "",
            "worker_error": "",
            "tmp_root": "",
            "tmp_storage_path": "",
            "tmp_storage_deleted": False,
            "cleanup_error": "",
            "scales": {},
        }
        if isinstance(extra_metadata, dict) and extra_metadata:
            metadata.update(extra_metadata)
        return metadata

    def _enqueue_job_record(
        self,
        trigger_label: str,
        trigger_dir: str,
        worker_paths: Dict[str, str],
        worker_cfg: Dict[str, Any],
    ) -> None:
        pending_jobs = self.state.setdefault("pending_jobs", [])
        if not isinstance(pending_jobs, list):
            pending_jobs = []
            self.state["pending_jobs"] = pending_jobs
        if self._find_pending_job(trigger_label) is not None:
            return
        pending_jobs.append(
            {
                "trigger_label": trigger_label,
                "trigger_dir": trigger_dir,
                "job_path": worker_paths["job"],
                "log_path": worker_paths["log"],
                "result_path": worker_paths["result"],
                "worker_cfg": copy.deepcopy(worker_cfg),
                "queued_at": round(time.time(), 3),
            }
        )
        queued_labels = self.state.setdefault("queued_labels", [])
        if trigger_label not in queued_labels:
            queued_labels.append(trigger_label)

    def _find_pending_job(self, trigger_label: str) -> Dict[str, Any] | None:
        pending_jobs = self.state.get("pending_jobs", [])
        if not isinstance(pending_jobs, list):
            return None
        for item in pending_jobs:
            if str((item or {}).get("trigger_label", "") or "") == trigger_label:
                return item if isinstance(item, dict) else None
        return None

    def _requeue_running_job(self, job_info: Dict[str, Any]) -> None:
        if not isinstance(job_info, dict) or not job_info:
            return
        label = str(job_info.get("trigger_label", "") or "").strip()
        if not label:
            return
        if self._find_pending_job(label) is None:
            pending_jobs = self.state.setdefault("pending_jobs", [])
            pending_jobs.insert(0, copy.deepcopy(job_info))
        queued_labels = self.state.setdefault("queued_labels", [])
        if label not in queued_labels:
            queued_labels.append(label)

    def _remove_pending_job(self, trigger_label: str) -> None:
        pending_jobs = self.state.get("pending_jobs", [])
        if isinstance(pending_jobs, list):
            self.state["pending_jobs"] = [
                item for item in pending_jobs
                if str((item or {}).get("trigger_label", "") or "") != trigger_label
            ]
        queued_labels = self.state.get("queued_labels", [])
        if isinstance(queued_labels, list):
            self.state["queued_labels"] = [label for label in queued_labels if str(label or "") != trigger_label]

    def _start_worker_process(
        self,
        job_info: Dict[str, Any],
    ) -> subprocess.Popen:
        worker_cfg = job_info.get("worker_cfg", {}) or {}
        worker_script_path = self._resolve_worker_script_path(worker_cfg.get("worker_script", DEFAULT_WORKER_SCRIPT))
        if not os.path.isfile(worker_script_path):
            raise FileNotFoundError("staged_eval worker script not found: {}".format(worker_script_path))

        cmd = [sys.executable, worker_script_path, "--job", str(job_info.get("job_path", "") or "")]
        worker_log_path = str(job_info.get("log_path", "") or "")
        os.makedirs(os.path.dirname(worker_log_path), exist_ok=True)
        log_file = open(worker_log_path, "w", encoding="utf-8")
        log_file.write("[STAGED_EVAL_WORKER_CMD] {}\n".format(" ".join(cmd)))
        log_file.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=BASE_DIR,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
        self._running_log_handle = log_file
        return proc

    def _maybe_start_next_job(self) -> None:
        if self._running_proc is not None:
            return
        pending_jobs = self.state.get("pending_jobs", [])
        if not isinstance(pending_jobs, list) or not pending_jobs:
            return

        job_info = copy.deepcopy(pending_jobs[0])
        try:
            proc = self._start_worker_process(job_info)
        except Exception as exc:
            label = str(job_info.get("trigger_label", "") or "").strip()
            self._remove_pending_job(label)
            failed_labels = self.state.setdefault("failed_labels", [])
            if label and label not in failed_labels:
                failed_labels.append(label)
            self.state["running_job"] = None
            self._persist_state()
            self._log("warning", "[STAGED_EVAL] failed_to_start trigger={} error={}".format(label, exc))
            return

        started_at = time.time()
        timeout_seconds = max(
            1,
            self._safe_int((job_info.get("worker_cfg", {}) or {}).get("worker_timeout_seconds", 1800), 1800),
        )
        job_info["started_at"] = round(started_at, 3)
        job_info["worker_timeout_seconds"] = timeout_seconds
        job_info["pid"] = int(proc.pid)
        self.state["running_job"] = job_info
        self._running_proc = proc
        self._persist_state()

    def _terminate_running_worker(self) -> None:
        proc = self._running_proc
        if proc is None:
            return
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        if self._running_log_handle is not None:
            self._running_log_handle.flush()
            self._running_log_handle.close()
            self._running_log_handle = None
        self._running_proc = None

    def _poll_worker(self) -> Dict[str, Any] | None:
        proc = self._running_proc
        running_job = self.state.get("running_job")
        if proc is None or not isinstance(running_job, dict) or not running_job:
            return None

        started_at = float(running_job.get("started_at", time.time()) or time.time())
        timeout_seconds = max(1, self._safe_int(running_job.get("worker_timeout_seconds", 1800), 1800))
        returncode = proc.poll()
        if returncode is None and (time.time() - started_at) < timeout_seconds:
            return None

        timed_out = False
        error = ""
        if returncode is None:
            timed_out = True
            error = "worker timed out after {}s".format(timeout_seconds)
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            returncode = None

        if self._running_log_handle is not None:
            if timed_out:
                self._running_log_handle.write(
                    "[STAGED_EVAL_WORKER_TIMEOUT] timeout_seconds={} error={}\n".format(timeout_seconds, error)
                )
            self._running_log_handle.flush()
            self._running_log_handle.close()
            self._running_log_handle = None

        self._running_proc = None
        worker_run = {
            "returncode": returncode,
            "timed_out": timed_out,
            "duration_seconds": round(time.time() - started_at, 3),
            "error": error,
        }
        metadata = self._finalize_running_job(running_job, worker_run)
        self.state["running_job"] = None
        self._persist_state()
        return metadata

    def _finalize_running_job(self, running_job: Dict[str, Any], worker_run: Dict[str, Any]) -> Dict[str, Any]:
        job_path = str(running_job.get("job_path", "") or "")
        trigger_dir = str(running_job.get("trigger_dir", "") or "")
        queued_metadata = self._load_existing_metadata(str(running_job.get("trigger_label", "") or ""))
        worker_paths = {
            "job": job_path,
            "log": str(running_job.get("log_path", "") or ""),
            "result": str(running_job.get("result_path", "") or ""),
        }
        worker_cfg = running_job.get("worker_cfg", {}) or {}
        worker_result = self._load_worker_result(worker_paths["result"])
        worker_success = self._worker_succeeded(worker_run, worker_result)
        job_payload = self._load_worker_result(job_path)
        runtime_config = job_payload.get("runtime_config", {}) if isinstance(job_payload.get("runtime_config", {}), dict) else {}
        metadata = self._build_trigger_metadata(
            trigger_label=str(job_payload.get("trigger_label", "") or ""),
            completed_session_count=self._safe_int(job_payload.get("completed_session_count", 0), 0),
            runtime_config=runtime_config,
            step_no=self._safe_int(job_payload.get("step_no", 0), 0),
            sim_time=str(job_payload.get("sim_time", "") or ""),
            snapshot_name=str(job_payload.get("snapshot_name", "") or ""),
            trigger_dir=trigger_dir,
            worker_paths=worker_paths,
            worker_cfg=worker_cfg,
            worker_run=worker_run,
            worker_result=worker_result,
            worker_success=worker_success,
        )
        if isinstance(queued_metadata, dict):
            preserved_queued_keys = {
                "trigger_mode",
                "completed_session_count_source",
                "step_interval_every_steps",
                "step_interval_anchor_step",
                "step_interval_elapsed_steps",
                "step_interval_virtual_session_interval",
                "step_interval_trigger_index",
                "step_interval_label_prefix",
                "step_interval_label_value_mode",
                "followup_phase",
            }
            for key, value in queued_metadata.items():
                if key not in metadata or key in preserved_queued_keys:
                    metadata[key] = value
        self._write_json(os.path.join(trigger_dir, "metadata.json"), metadata)
        label = str(metadata.get("trigger_label", "") or "").strip()
        self._remove_pending_job(label)

        if worker_success:
            self._mark_trigger_done(metadata, trigger_dir)
            self._log(
                "info",
                "[STAGED_EVAL] trigger={} target={} completed_sessions={} step={} sim_time={} worker_duration_seconds={}".format(
                    label,
                    metadata.get("target_agent", ""),
                    metadata.get("completed_session_count", 0),
                    metadata.get("step_no", 0),
                    metadata.get("sim_time", ""),
                    metadata.get("worker_duration_seconds", 0.0),
                ),
            )
        else:
            failed_labels = self.state.setdefault("failed_labels", [])
            if label and label not in failed_labels:
                failed_labels.append(label)
            self._log(
                "warning",
                "[STAGED_EVAL] trigger={} worker_failed returncode={} timed_out={} error={}".format(
                    label,
                    worker_run.get("returncode"),
                    bool(worker_run.get("timed_out", False)),
                    metadata.get("worker_error", ""),
                ),
            )
        return metadata

    def _load_worker_result(self, result_path: str) -> Dict[str, Any]:
        if not os.path.exists(result_path):
            return {}
        with open(result_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        return payload if isinstance(payload, dict) else {}

    def _worker_succeeded(self, worker_run: Dict[str, Any], worker_result: Dict[str, Any]) -> bool:
        if bool(worker_run.get("timed_out", False)):
            return False
        if worker_run.get("returncode", None) != 0:
            return False
        if not isinstance(worker_result, dict):
            return False
        return str(worker_result.get("status", "") or "").strip().lower() == "ok"

    def _build_trigger_metadata(
        self,
        trigger_label: str,
        completed_session_count: int,
        runtime_config: Dict[str, Any],
        step_no: int,
        sim_time: str,
        snapshot_name: str,
        trigger_dir: str,
        worker_paths: Dict[str, str],
        worker_cfg: Dict[str, Any],
        worker_run: Dict[str, Any],
        worker_result: Dict[str, Any],
        worker_success: bool,
        extra_metadata: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        scales = worker_result.get("scale_summaries", {}) if isinstance(worker_result.get("scale_summaries", {}), dict) else {}
        worker_error = str(worker_result.get("error", "") or worker_run.get("error", "") or "").strip()
        _, completed_session_count_source = self._resolve_completed_session_count(runtime_config)
        metadata = {
            "status": "ok" if worker_success else "worker_failed",
            "trigger_label": trigger_label,
            "completed_session_count": int(completed_session_count),
            "completed_session_count_source": completed_session_count_source,
            "step_no": int(step_no),
            "sim_time": str(sim_time or ""),
            "snapshot_name": str(snapshot_name or ""),
            "target_agent": self._target_agent(),
            "doctor_name": self._doctor_name(runtime_config),
            "session_interval": self._safe_int(self._cfg().get("session_interval", 0), 0),
            "max_completed_sessions": self._safe_int(self._cfg().get("max_completed_sessions", 0), 0),
            "step_interval": copy.deepcopy(self._step_interval_cfg()),
            "t4_enabled": bool(self._cfg().get("t4_enabled", False)),
            "t4_after_steps": max(0, self._safe_int(self._cfg().get("t4_after_steps", 0), 0)),
            "readonly_eval": True,
            "external_memory_read": bool(worker_result.get("external_memory_read", False)),
            "uses_answer_without_memory": True,
            "writes_blocked_by_design": False,
            "writes_isolated_via_temp_storage": True,
            "storage_isolation_enabled": True,
            "local_write_attempt_count": 0,
            "external_write_attempt_count": 0,
            "worker_mode": "subprocess",
            "worker_script": str(worker_cfg.get("worker_script", DEFAULT_WORKER_SCRIPT) or DEFAULT_WORKER_SCRIPT),
            "worker_timeout_seconds": self._safe_int(worker_cfg.get("worker_timeout_seconds", 1800), 1800),
            "cleanup_tmp_storage": bool(worker_cfg.get("cleanup_tmp_storage", True)),
            "worker_log_file": os.path.relpath(worker_paths["log"], trigger_dir),
            "worker_job_file": os.path.relpath(worker_paths["job"], trigger_dir),
            "worker_result_file": os.path.relpath(worker_paths["result"], trigger_dir),
            "worker_returncode": worker_run.get("returncode"),
            "worker_timed_out": bool(worker_run.get("timed_out", False)),
            "worker_duration_seconds": worker_run.get("duration_seconds", 0.0),
            "worker_status": str(worker_result.get("status", "") or "").strip(),
            "worker_error": worker_error,
            "tmp_root": str(worker_result.get("tmp_root", "") or ""),
            "tmp_storage_path": str(worker_result.get("tmp_storage_path", "") or ""),
            "tmp_storage_deleted": bool(worker_result.get("tmp_storage_deleted", False)),
            "cleanup_error": str(worker_result.get("cleanup_error", "") or ""),
            "scales": scales,
        }
        if isinstance(extra_metadata, dict) and extra_metadata:
            metadata.update(extra_metadata)
        return metadata

    def _mark_trigger_done(self, metadata: Dict[str, Any], trigger_dir: str) -> None:
        label = str(metadata.get("trigger_label", "") or "").strip()
        labels = self.state.setdefault("triggered_labels", [])
        if label and label not in labels:
            labels.append(label)
        self._remove_pending_job(label)
        failed_labels = self.state.setdefault("failed_labels", [])
        if label in failed_labels:
            failed_labels[:] = [item for item in failed_labels if str(item or "") != label]
        if label == "T0":
            self.state["t0_done"] = True
        if label == "T4":
            self.state["t4_done"] = True
        completed_counts = self.state.setdefault("triggered_completed_session_counts", [])
        completed_session_count = int(metadata.get("completed_session_count", 0) or 0)
        if completed_session_count not in completed_counts:
            completed_counts.append(completed_session_count)
        records = self.state.setdefault("records", [])
        if not isinstance(records, list):
            records = []
            self.state["records"] = records
        records[:] = [r for r in records if str((r or {}).get("trigger_label", "") or "") != label]
        record = dict(metadata)
        record["output_dir"] = os.path.relpath(trigger_dir, self.checkpoints_folder)
        records.append(record)

    def _write_index(self) -> None:
        os.makedirs(self.output_dir, exist_ok=True)
        payload = {
            "run_name": self.run_name,
            "triggered_labels": list(self.state.get("triggered_labels", []) or []),
            "queued_labels": list(self.state.get("queued_labels", []) or []),
            "failed_labels": list(self.state.get("failed_labels", []) or []),
            "pending_jobs": list(self.state.get("pending_jobs", []) or []),
            "running_job": copy.deepcopy(self.state.get("running_job", None)),
            "records": list(self.state.get("records", []) or []),
        }
        self._write_json(os.path.join(self.output_dir, "index.json"), payload)

    def _ensure_state_schema(self) -> None:
        if not isinstance(self.state, dict):
            self.state = {}
            self.config["staged_eval_state"] = self.state
        if not isinstance(self.state.get("triggered_labels"), list):
            self.state["triggered_labels"] = []
        if not isinstance(self.state.get("triggered_completed_session_counts"), list):
            self.state["triggered_completed_session_counts"] = []
        if not isinstance(self.state.get("records"), list):
            self.state["records"] = []
        if not isinstance(self.state.get("queued_labels"), list):
            self.state["queued_labels"] = []
        if not isinstance(self.state.get("failed_labels"), list):
            self.state["failed_labels"] = []
        if not isinstance(self.state.get("pending_jobs"), list):
            self.state["pending_jobs"] = []
        if not isinstance(self.state.get("running_job"), dict):
            self.state["running_job"] = None
        self.state["t0_done"] = bool(self.state.get("t0_done", False))
        self.state["t4_done"] = bool(self.state.get("t4_done", False))
        completion_step_no = self.state.get("t4_completion_step_no")
        try:
            self.state["t4_completion_step_no"] = (
                int(completion_step_no)
                if completion_step_no is not None and str(completion_step_no).strip() != ""
                else None
            )
        except Exception:
            self.state["t4_completion_step_no"] = None
        self.state["t4_completion_snapshot_name"] = str(self.state.get("t4_completion_snapshot_name", "") or "")

    def _is_label_finished_or_pending(self, trigger_label: str) -> bool:
        label = str(trigger_label or "").strip()
        if not label:
            return False
        if label in self.state.get("triggered_labels", []):
            return True
        if label in self.state.get("queued_labels", []):
            return True
        if label in self.state.get("failed_labels", []):
            return True
        running_job = self.state.get("running_job")
        if isinstance(running_job, dict) and str(running_job.get("trigger_label", "") or "") == label:
            return True
        return self._find_pending_job(label) is not None

    def _load_existing_metadata(self, trigger_label: str) -> Dict[str, Any] | None:
        trigger_dir = os.path.join(self.output_dir, str(trigger_label or "").strip())
        metadata_path = os.path.join(trigger_dir, "metadata.json")
        if not os.path.exists(metadata_path):
            return None
        try:
            payload = self._load_worker_result(metadata_path)
        except Exception:
            return None
        return payload if isinstance(payload, dict) else None

    def _safe_int(self, value: Any, default: int) -> int:
        try:
            return int(value)
        except Exception:
            return int(default)

    def _write_json(self, path: str, payload: Any) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def _write_json_atomic(self, path: str, payload: Any) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fd, temp_path = tempfile.mkstemp(prefix=".{}-".format(os.path.basename(path)), dir=os.path.dirname(path))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
            os.replace(temp_path, path)
        except Exception:
            try:
                os.unlink(temp_path)
            except FileNotFoundError:
                pass
            raise

    def _log(self, level: str, message: str) -> None:
        logger = self.logger
        if logger is None:
            return
        func = getattr(logger, str(level or "info"), None)
        if callable(func):
            func(message)
