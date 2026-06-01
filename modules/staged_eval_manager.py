from __future__ import annotations

import copy
import json
import os
from typing import Any, Dict, List

from customization.depression_scale_agent.app import ChatSession, iter_jsonl, resolve_question


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
QUESTIONS_ROOT = os.path.join(
    BASE_DIR,
    "customization",
    "depression_scale_agent",
    "questions",
    "templates",
)
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
        trigger_dir = os.path.join(self.output_dir, trigger_label)
        os.makedirs(trigger_dir, exist_ok=True)

        session = ChatSession(
            self.run_name,
            snapshot_file=snapshot_name or trigger_label,
            runtime_config=copy.deepcopy(runtime_config),
            conversation=copy.deepcopy(conversation or {}),
        )
        session.set_agent(target_agent)

        scale_summaries: Dict[str, Any] = {}
        external_memory_read = False
        for scale_name in self._scales():
            question_file = SCALE_QUESTION_FILES.get(scale_name)
            if not question_file:
                raise ValueError(f"unsupported scale: {scale_name}")
            question_path = os.path.join(QUESTIONS_ROOT, question_file)
            if not os.path.exists(question_path):
                raise FileNotFoundError(f"question file not found: {question_path}")

            answered_rows: List[Dict[str, Any]] = []
            trace_rows: List[Dict[str, Any]] = []
            scale_external_memory_read = False
            for index, item in enumerate(iter_jsonl(question_path), start=1):
                question = resolve_question(item)
                answer = session.answer_without_memory(question)
                trace_payload = session.get_last_answer_trace()
                row = dict(item) if isinstance(item, dict) else {"question": question}
                row["answer"] = answer
                answered_rows.append(row)
                trace_rows.append(
                    {
                        "index": index,
                        "question": question,
                        "answer": answer,
                        "trace": trace_payload,
                    }
                )
                if self._trace_used_external_memory(trace_payload):
                    scale_external_memory_read = True
                    external_memory_read = True

            answers_file = f"{self._safe_file_stem(scale_name)}_answered.jsonl"
            trace_file = f"{self._safe_file_stem(scale_name)}_trace.json"
            self._write_jsonl(os.path.join(trigger_dir, answers_file), answered_rows)
            self._write_json(os.path.join(trigger_dir, trace_file), trace_rows)
            scale_summaries[scale_name] = {
                "question_file": question_file,
                "question_count": len(answered_rows),
                "answers_file": answers_file,
                "trace_file": trace_file,
                "external_memory_read": scale_external_memory_read,
            }

        metadata = {
            "trigger_label": trigger_label,
            "completed_session_count": int(completed_session_count),
            "step_no": int(step_no),
            "sim_time": str(sim_time or ""),
            "snapshot_name": str(snapshot_name or ""),
            "target_agent": target_agent,
            "doctor_name": self._doctor_name(runtime_config),
            "session_interval": self._safe_int(self._cfg().get("session_interval", 0), 0),
            "max_completed_sessions": self._safe_int(self._cfg().get("max_completed_sessions", 0), 0),
            "t4_enabled": bool(self._cfg().get("t4_enabled", False)),
            "t4_after_steps": max(0, self._safe_int(self._cfg().get("t4_after_steps", 0), 0)),
            "readonly_eval": True,
            "external_memory_read": external_memory_read,
            "uses_answer_without_memory": True,
            "writes_blocked_by_design": True,
            "local_write_attempt_count": 0,
            "external_write_attempt_count": 0,
            "scales": scale_summaries,
        }
        if isinstance(extra_metadata, dict) and extra_metadata:
            metadata.update(extra_metadata)
        self._write_json(os.path.join(trigger_dir, "metadata.json"), metadata)
        self._mark_trigger_done(metadata, trigger_dir)
        self._write_index()
        self._log(
            "info",
            "[STAGED_EVAL] trigger={} target={} completed_sessions={} step={} sim_time={}".format(
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
        intervention_cfg = runtime_config.get("intervention", {}) or {}
        pair_state = self._resolve_target_pair_state(runtime_config)
        if not pair_state:
            return 0

        session_cfg = intervention_cfg.get("session_prompt_injection", {}) or {}
        order = session_cfg.get("order", []) or []
        current_index = self._safe_int(pair_state.get("current_index", 0), 0)
        if current_index < 0:
            current_index = 0

        if bool(pair_state.get("completed", False)):
            if isinstance(order, list) and order:
                return len(order)
            return current_index
        return current_index

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

        completed_session_count = self._compute_completed_session_count(runtime_config)
        if completed_session_count <= 0:
            return None

        max_completed_sessions = self._safe_int(
            self._cfg().get("max_completed_sessions", 0),
            0,
        )
        if max_completed_sessions > 0 and completed_session_count > max_completed_sessions:
            return None

        if completed_session_count % interval != 0:
            return None

        trigger_label = f"session_{completed_session_count}"
        if trigger_label in self.state.get("triggered_labels", []):
            return None

        try:
            return self._run_trigger(
                trigger_label=trigger_label,
                completed_session_count=completed_session_count,
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

    def _trace_used_external_memory(self, trace_payload: Dict[str, Any]) -> bool:
        if not isinstance(trace_payload, dict):
            return False
        retrieval = trace_payload.get("external_memory_retrieval", {}) or {}
        if not isinstance(retrieval, dict):
            return False
        return bool(retrieval.get("ok", False)) or bool(str(retrieval.get("scoped_user_id", "") or "").strip())

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

    def _safe_file_stem(self, value: str) -> str:
        return str(value or "").replace("/", "_").replace("\\", "_").replace(" ", "_")

    def _write_json(self, path: str, payload: Any) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def _write_jsonl(self, path: str, rows: List[Dict[str, Any]]) -> None:
        with open(path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _log(self, level: str, message: str) -> None:
        logger = self.logger
        if logger is None:
            return
        func = getattr(logger, str(level or "info"), None)
        if callable(func):
            func(message)
