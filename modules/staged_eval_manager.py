from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
import time
from typing import Any, Dict, List


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
    "PHQ-9": "PHQ-9-v2.jsonl",
    "BDI-II": "BDI-II-v2.jsonl",
    "SDS": "SDS-v2.jsonl",
}


class StagedEvalManager:
    def __init__(self, run_name: str, checkpoints_folder: str, config: Dict[str, Any], logger: Any = None):
        self.run_name = str(run_name or "").strip()
        self.checkpoints_folder = checkpoints_folder
        self.config = config if isinstance(config, dict) else {}
        self.logger = logger
        self.output_dir = os.path.join(self.checkpoints_folder, "staged_eval")
        self.state = self.config.setdefault("staged_eval_state", {})
        self._ensure_state_schema()

    def maybe_run_t0(
        self,
        runtime_config: Dict[str, Any],
        conversation: Dict[str, Any],
        step_no: int,
        sim_time: str,
    ) -> Dict[str, Any] | None:
        if not self._enabled():
            return None
        if not bool(self._cfg().get("include_t0", False)):
            return None
        if bool(self.state.get("t0_done", False)):
            return None
        try:
            return self._run_trigger(
                trigger_label="T0",
                completed_session_count=0,
                runtime_config=runtime_config,
                conversation=conversation,
                step_no=step_no,
                sim_time=sim_time,
                snapshot_name="",
            )
        except Exception as exc:
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
        if not self._enabled():
            return None

        last_result = None
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

    def _run_trigger(
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

        os.makedirs(self.output_dir, exist_ok=True)
        trigger_dir = self._build_trigger_dir(trigger_label)
        worker_paths = self._build_worker_paths(trigger_dir)
        worker_cfg = self._worker_cfg()
        scale_question_files = self._resolve_scale_question_files()

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
            scale_question_files=scale_question_files,
            worker_cfg=worker_cfg,
        )
        self._write_json(worker_paths["job"], job_payload)

        worker_run = self._run_worker_process(
            job_path=worker_paths["job"],
            worker_log_path=worker_paths["log"],
            worker_cfg=worker_cfg,
        )
        worker_result = self._load_worker_result(worker_paths["result"])
        worker_success = self._worker_succeeded(worker_run, worker_result)

        metadata = self._build_trigger_metadata(
            trigger_label=trigger_label,
            completed_session_count=completed_session_count,
            runtime_config=runtime_config,
            step_no=step_no,
            sim_time=sim_time,
            snapshot_name=snapshot_name,
            trigger_dir=trigger_dir,
            worker_paths=worker_paths,
            worker_cfg=worker_cfg,
            worker_run=worker_run,
            worker_result=worker_result,
            worker_success=worker_success,
            extra_metadata=extra_metadata,
        )
        self._write_json(os.path.join(trigger_dir, "metadata.json"), metadata)

        if not worker_success:
            self._log(
                "warning",
                "[STAGED_EVAL] trigger={} worker_failed returncode={} timed_out={} error={}".format(
                    trigger_label,
                    worker_run.get("returncode"),
                    bool(worker_run.get("timed_out", False)),
                    metadata.get("worker_error", ""),
                ),
            )
            raise RuntimeError(
                "staged_eval worker failed for {}: {}".format(
                    trigger_label,
                    metadata.get("worker_error", "unknown error"),
                )
            )

        self._mark_trigger_done(metadata, trigger_dir)
        self._write_index()
        self._log(
            "info",
            "[STAGED_EVAL] trigger={} target={} completed_sessions={} step={} sim_time={} worker_duration_seconds={}".format(
                trigger_label,
                target_agent,
                completed_session_count,
                step_no,
                sim_time,
                metadata.get("worker_duration_seconds", 0.0),
            ),
        )
        return metadata

    def _compute_completed_session_count(self, runtime_config: Dict[str, Any]) -> int:
        if not isinstance(runtime_config, dict):
            return 0
        return self._compute_resident_chat_completed_count(runtime_config)

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
        if max_completed_sessions > 0 and completed_conversation_count > max_completed_sessions:
            return None

        if completed_conversation_count % interval != 0:
            return None

        trigger_label = f"session_{completed_conversation_count}"
        if trigger_label in self.state.get("triggered_labels", []):
            return None

        try:
            return self._run_trigger(
                trigger_label=trigger_label,
                completed_session_count=completed_conversation_count,
                runtime_config=runtime_config,
                conversation=conversation,
                step_no=step_no,
                sim_time=sim_time,
                snapshot_name=snapshot_name,
            )
        except Exception as exc:
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
        if bool(self.state.get("t4_done", False)) or ("T4" in self.state.get("triggered_labels", [])):
            self.state["t4_done"] = True
            return None
        if not self._is_target_pair_completed(runtime_config):
            return None

        completion_step_no = self.state.get("t4_completion_step_no")
        if completion_step_no is None:
            completion_step_no = int(step_no)
            self.state["t4_completion_step_no"] = completion_step_no
            self.state["t4_completion_snapshot_name"] = str(snapshot_name or "")

        after_steps = max(0, self._safe_int(self._cfg().get("t4_after_steps", 0), 0))
        if int(step_no) - int(completion_step_no) < after_steps:
            return None

        completed_session_count = self._compute_completed_session_count(runtime_config)
        try:
            return self._run_trigger(
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

    def _cfg(self) -> Dict[str, Any]:
        cfg = self.config.get("staged_eval", {}) or {}
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

    def _target_agent(self) -> str:
        return str(self._cfg().get("target_agent", "") or "").strip()

    def _scales(self) -> List[str]:
        scales = self._cfg().get("scales", []) or []
        if not isinstance(scales, list):
            return []
        return [str(item or "").strip() for item in scales if str(item or "").strip()]

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

    def _resolve_scale_question_files(self) -> Dict[str, str]:
        question_files: Dict[str, str] = {}
        for scale_name in self._scales():
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
        scale_question_files: Dict[str, str],
        worker_cfg: Dict[str, Any],
    ) -> Dict[str, Any]:
        return {
            "run_name": self.run_name,
            "trigger_label": trigger_label,
            "completed_session_count": int(completed_session_count),
            "step_no": int(step_no),
            "sim_time": str(sim_time or ""),
            "snapshot_name": str(snapshot_name or ""),
            "target_agent": target_agent,
            "scales": list(self._scales()),
            "scale_question_files": copy.deepcopy(scale_question_files),
            "runtime_config": copy.deepcopy(runtime_config),
            "conversation": copy.deepcopy(conversation or {}),
            "trigger_dir": trigger_dir,
            "worker_result_path": worker_result_path,
            "storage_source_root": os.path.join(self.checkpoints_folder, "storage"),
            "tmp_root_parent": os.path.join(self.output_dir, "_tmp"),
            "cleanup_tmp_storage": bool(worker_cfg.get("cleanup_tmp_storage", True)),
        }

    def _resolve_worker_script_path(self, worker_script: str) -> str:
        path = str(worker_script or "").strip()
        if not path:
            path = DEFAULT_WORKER_SCRIPT
        if not os.path.isabs(path):
            path = os.path.join(BASE_DIR, path)
        return os.path.abspath(path)

    def _run_worker_process(
        self,
        job_path: str,
        worker_log_path: str,
        worker_cfg: Dict[str, Any],
    ) -> Dict[str, Any]:
        worker_script_path = self._resolve_worker_script_path(worker_cfg.get("worker_script", DEFAULT_WORKER_SCRIPT))
        if not os.path.isfile(worker_script_path):
            raise FileNotFoundError("staged_eval worker script not found: {}".format(worker_script_path))

        timeout_seconds = max(1, self._safe_int(worker_cfg.get("worker_timeout_seconds", 1800), 1800))
        cmd = [sys.executable, worker_script_path, "--job", job_path]
        started_at = time.time()
        os.makedirs(os.path.dirname(worker_log_path), exist_ok=True)
        with open(worker_log_path, "w", encoding="utf-8") as log_file:
            log_file.write("[STAGED_EVAL_WORKER_CMD] {}\n".format(" ".join(cmd)))
            log_file.flush()
            try:
                completed = subprocess.run(
                    cmd,
                    cwd=BASE_DIR,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    text=True,
                    timeout=timeout_seconds,
                    check=False,
                )
                return {
                    "returncode": completed.returncode,
                    "timed_out": False,
                    "duration_seconds": round(time.time() - started_at, 3),
                    "error": "",
                }
            except subprocess.TimeoutExpired as exc:
                log_file.write(
                    "[STAGED_EVAL_WORKER_TIMEOUT] timeout_seconds={} error={}\n".format(
                        timeout_seconds,
                        exc,
                    )
                )
                log_file.flush()
                return {
                    "returncode": None,
                    "timed_out": True,
                    "duration_seconds": round(time.time() - started_at, 3),
                    "error": str(exc),
                }
            except Exception as exc:
                log_file.write("[STAGED_EVAL_WORKER_SPAWN_ERROR] error={}\n".format(exc))
                log_file.flush()
                return {
                    "returncode": None,
                    "timed_out": False,
                    "duration_seconds": round(time.time() - started_at, 3),
                    "error": str(exc),
                }

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
        metadata = {
            "status": "ok" if worker_success else "worker_failed",
            "trigger_label": trigger_label,
            "completed_session_count": int(completed_session_count),
            "step_no": int(step_no),
            "sim_time": str(sim_time or ""),
            "snapshot_name": str(snapshot_name or ""),
            "target_agent": self._target_agent(),
            "doctor_name": self._doctor_name(runtime_config),
            "session_interval": self._safe_int(self._cfg().get("session_interval", 0), 0),
            "max_completed_sessions": self._safe_int(self._cfg().get("max_completed_sessions", 0), 0),
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

    def _safe_int(self, value: Any, default: int) -> int:
        try:
            return int(value)
        except Exception:
            return int(default)

    def _write_json(self, path: str, payload: Any) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def _log(self, level: str, message: str) -> None:
        logger = self.logger
        if logger is None:
            return
        func = getattr(logger, str(level or "info"), None)
        if callable(func):
            func(message)
