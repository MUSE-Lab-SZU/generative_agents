from __future__ import annotations

import random
from typing import Any, Callable, Dict, List


class ResidentChatScheduler:
    def __init__(self, config: Dict[str, Any], state: Dict[str, Any], logger: Any = None):
        self.config = config
        self.state = state
        self.logger = logger
        self.enabled = False
        self.rules: List[Dict[str, Any]] = []
        self.refresh()

    def refresh(self) -> None:
        intervention_cfg = self.config.get("intervention", {}) or {}
        scheduler_cfg = intervention_cfg.get("resident_chat_scheduler", {}) or {}
        self.enabled = bool(scheduler_cfg.get("enabled", False))
        self.rules = scheduler_cfg.get("rules", []) or []

    def ensure_state_schema(self) -> None:
        if not isinstance(self.state, dict):
            return
        resident_state = self.state.setdefault("resident_chat_state", {})
        if not isinstance(resident_state, dict):
            resident_state = {}
            self.state["resident_chat_state"] = resident_state
        if not isinstance(resident_state.get("rules"), dict):
            resident_state["rules"] = {}

    def collect_triggered_meetings(
        self,
        now: Any,
        agents: Dict[str, Any],
        should_trigger: Callable[[str, Dict[str, Any], Any], bool],
    ) -> List[Dict[str, Any]]:
        self.refresh()
        self.ensure_state_schema()
        if not self.enabled:
            return []

        agent_names = set(agents.keys()) if isinstance(agents, dict) else set()
        payloads: List[Dict[str, Any]] = []
        for idx, raw_rule in enumerate(self.rules):
            if not isinstance(raw_rule, dict):
                continue
            rule = dict(raw_rule)
            rule_id = str(rule.get("rule_id", f"resident_chat_rule_{idx}") or f"resident_chat_rule_{idx}").strip()
            if not rule.get("enabled", True):
                continue
            target_patient = str(
                rule.get("target_patient", "") or rule.get("patient", "") or ""
            ).strip()
            if not target_patient:
                self._log(f"[RESIDENT_CHAT] skip rule={rule_id} reason=target_patient_missing")
                continue
            if agent_names and target_patient not in agent_names:
                self._log(
                    f"[RESIDENT_CHAT] skip rule={rule_id} patient={target_patient} reason=patient_not_found"
                )
                continue
            if not should_trigger(rule_id, rule, now):
                continue
            doctor = self._select_candidate(rule_id, rule, agent_names)
            if not doctor:
                self._log(f"[RESIDENT_CHAT] skip rule={rule_id} reason=no_candidate_selected")
                continue
            payload = {
                "rule_id": rule_id,
                "type": rule.get("type", ""),
                "interval_steps": rule.get("interval_steps", 0),
                "interval_minutes": rule.get("interval_minutes", 0),
                "interval_days": rule.get("interval_days", 0),
                "weekday": rule.get("weekday", 0),
                "time": rule.get("time", ""),
                "step_phase": rule.get("step_phase", ""),
                "duration": rule.get("duration", 30),
                "doctor": doctor,
                "patient": target_patient,
                "meeting_kind": "resident_chat",
                "prompt_file": str(rule.get("prompt_file", "") or "").strip(),
                # Resident-chat prompts describe how the selected resident should
                # respond.  In meeting terminology that resident occupies the
                # `doctor` role, while the target agent occupies `patient`.
                "prompt_target": str(rule.get("prompt_target", "doctor") or "doctor").strip(),
                "meeting_source": "resident_chat_scheduler",
            }
            payloads.append(payload)
            self._log(
                f"[RESIDENT_CHAT] trigger rule={rule_id} doctor={doctor} patient={target_patient}"
            )
        return payloads

    def _select_candidate(
        self,
        rule_id: str,
        rule: Dict[str, Any],
        agent_names: set[str],
    ) -> str:
        selection_strategy = str(
            rule.get("selection_strategy", "random_no_repeat") or "random_no_repeat"
        ).strip().lower()
        candidates = self._normalize_candidates(rule.get("resident_candidates", []) or [], agent_names)
        if not candidates:
            return ""
        if selection_strategy != "random_no_repeat":
            self._log(
                f"[RESIDENT_CHAT] rule={rule_id} unsupported_strategy={selection_strategy} fallback=random_no_repeat"
            )
        rule_state = self._get_rule_state(rule_id)
        remaining = [
            name
            for name in (rule_state.get("remaining_candidates", []) or [])
            if name in candidates
        ]
        last_selected = str(rule_state.get("last_selected", "") or "").strip()
        if not remaining:
            remaining = self._reshuffle_candidates(candidates, last_selected)
        selected = str(remaining.pop(0) or "").strip()
        if not selected:
            return ""
        rule_state["last_selected"] = selected
        rule_state["remaining_candidates"] = remaining
        return selected

    def _normalize_candidates(self, raw_candidates: Any, agent_names: set[str]) -> List[str]:
        items = raw_candidates if isinstance(raw_candidates, list) else []
        normalized: List[str] = []
        seen = set()
        for item in items:
            name = str(item or "").strip()
            if not name or name in seen:
                continue
            if agent_names and name not in agent_names:
                continue
            seen.add(name)
            normalized.append(name)
        return normalized

    def _reshuffle_candidates(self, candidates: List[str], last_selected: str) -> List[str]:
        pool = list(candidates)
        random.shuffle(pool)
        if len(pool) > 1 and last_selected and pool[0] == last_selected:
            swap_idx = random.randrange(1, len(pool))
            pool[0], pool[swap_idx] = pool[swap_idx], pool[0]
        return pool

    def _get_rule_state(self, rule_id: str) -> Dict[str, Any]:
        self.ensure_state_schema()
        resident_state = self.state.setdefault("resident_chat_state", {})
        rules_state = resident_state.setdefault("rules", {})
        rule_state = rules_state.setdefault(rule_id, {})
        if not isinstance(rule_state, dict):
            rule_state = {}
            rules_state[rule_id] = rule_state
        if not isinstance(rule_state.get("remaining_candidates"), list):
            rule_state["remaining_candidates"] = []
        return rule_state

    def _log(self, message: str) -> None:
        if self.logger and hasattr(self.logger, "info"):
            self.logger.info(message)
