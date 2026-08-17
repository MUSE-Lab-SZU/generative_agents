"""Append-only structured records for zero-LLM simulation observability."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Iterable, Optional


class SimulationEventRecorder:
    """Write compact JSONL events without participating in simulation logic.

    Recording is deliberately best-effort: serialization or filesystem failures
    are reported once and never propagate into agent decisions or control flow.
    """

    SCHEMA_VERSION = "simulation_events_v1"

    def __init__(
        self,
        checkpoints_folder: str,
        config: Dict[str, Any],
        logger: Optional[Any] = None,
    ) -> None:
        checkpointing = config.get("checkpointing", {}) or {}
        event_cfg = (
            checkpointing.get("event_log", {}) or {}
            if isinstance(checkpointing, dict)
            else {}
        )
        if not isinstance(event_cfg, dict):
            event_cfg = {}
        self.enabled = bool(event_cfg.get("enabled", True))
        self.record_interactions = bool(event_cfg.get("record_interactions", True))
        self.record_agent_states = bool(event_cfg.get("record_agent_states", True))
        filename = str(event_cfg.get("filename", "simulation_events.jsonl") or "").strip()
        self.path = os.path.join(checkpoints_folder, filename or "simulation_events.jsonl")
        self.logger = logger
        self._step_no: Optional[int] = None
        self._simulation_time = ""
        self._write_warned = False

    def begin_step(self, step_no: int, simulation_time: Any) -> None:
        self._step_no = int(step_no)
        self._simulation_time = self._format_time(simulation_time)

    def append_interaction(self, record: Dict[str, Any]) -> None:
        if self.record_interactions:
            self._append_many([record])

    def append_agent_states(self, records: Iterable[Dict[str, Any]]) -> None:
        if self.record_agent_states:
            self._append_many(records)

    def _append_many(self, records: Iterable[Dict[str, Any]]) -> None:
        if not self.enabled:
            return
        try:
            lines = []
            for raw in records:
                if not isinstance(raw, dict):
                    continue
                payload = {
                    "schema_version": self.SCHEMA_VERSION,
                    "simulation_time": self._simulation_time,
                    "step": self._step_no,
                }
                payload.update(raw)
                if not payload.get("simulation_time"):
                    payload["simulation_time"] = self._simulation_time
                lines.append(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
            if not lines:
                return
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as handle:
                handle.write("\n".join(lines) + "\n")
        except Exception as exc:
            if self._write_warned:
                return
            self._write_warned = True
            if self.logger and hasattr(self.logger, "warning"):
                self.logger.warning(
                    "[SIMULATION_EVENT_WRITE_FAIL] path={} error={}".format(
                        self.path,
                        str(exc),
                    )
                )

    @staticmethod
    def _format_time(value: Any) -> str:
        if hasattr(value, "strftime"):
            try:
                return value.strftime("%Y%m%d-%H:%M:%S")
            except Exception:
                pass
        return str(value or "")
