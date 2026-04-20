"""Session prompt injection manager for forced doctor-patient chats."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from modules import utils


class SessionPromptInjectionManager:
    """Manage doctor-side CBT session prompt injection and session advancement."""

    def __init__(self, config: Dict[str, Any], state: Dict[str, Any], logger: Any = None):
        self.config = config
        self.state = state if isinstance(state, dict) else {}
        self.logger = logger

        intervention_cfg = self.config.get("intervention", {}) or {}
        self.injection_cfg = intervention_cfg.get("session_prompt_injection", {}) or {}

        self.enabled = bool(self.injection_cfg.get("enabled", False))
        self.prompt_file = str(
            self.injection_cfg.get("prompt_file", "data/prompts/intervention_prompts.json") or ""
        ).strip()
        self.namespace = str(self.injection_cfg.get("namespace", "CBT") or "CBT").strip() or "CBT"
        self.advance_marker = str(self.injection_cfg.get("advance_marker", "[SESSION_END]") or "[SESSION_END]").strip()

        session_prompt_state = self.state.setdefault("session_prompt_state", {})
        if not isinstance(session_prompt_state, dict):
            session_prompt_state = {}
            self.state["session_prompt_state"] = session_prompt_state
        pairs = session_prompt_state.setdefault("pairs", {})
        if not isinstance(pairs, dict):
            session_prompt_state["pairs"] = {}

        self._log(
            "initialized enabled={} namespace={} order_size={}".format(
                self.enabled,
                self.namespace,
                len(self._order()),
            )
        )

    def get_doctor_injection_block(
        self,
        doctor_name: str,
        patient_name: str,
        forced: bool,
        meeting_id: str = "",
    ) -> str:
        if not self.enabled:
            return ""

        if not bool(forced):
            self._log("skip inject reason=non_forced pair={}::{}".format(doctor_name, patient_name))
            return ""

        order = self._order()
        if not order:
            self._log("skip inject reason=empty_order pair={}::{}".format(doctor_name, patient_name))
            return ""

        namespace_map = self.load_session_prompt_map()
        if not namespace_map:
            self._log("skip inject reason=empty_prompt_map pair={}::{}".format(doctor_name, patient_name))
            return ""

        state = self.resolve_current_session(doctor_name, patient_name)
        if bool(state.get("completed", False)):
            self._log("skip inject reason=completed pair={}::{}".format(doctor_name, patient_name))
            return ""

        session_id = str(state.get("current_session", "") or "")
        prompt_text = str(namespace_map.get(session_id, "") or "").strip()

        if not prompt_text:
            fallback = self._find_next_available_session(order, namespace_map, int(state.get("current_index", 0) or 0))
            if not fallback:
                state["completed"] = True
                state["updated_at"] = self._now_str()
                self._log(
                    "skip inject reason=no_available_session pair={}::{} namespace={}".format(
                        doctor_name,
                        patient_name,
                        self.namespace,
                    )
                )
                return ""

            next_index, next_session = fallback
            state["current_index"] = next_index
            state["current_session"] = next_session
            state["completed"] = False
            state["updated_at"] = self._now_str()
            prompt_text = str(namespace_map.get(next_session, "") or "").strip()

        if not prompt_text:
            return ""

        block = (
            "<DOCTOR_SESSION_PROMPT_INJECTION>\n"
            + prompt_text
            + "\n</DOCTOR_SESSION_PROMPT_INJECTION>"
        )
        state["last_meeting_id"] = str(meeting_id or state.get("last_meeting_id", ""))
        self._log_highlight(
            "inject pair={}::{} session={} forced={} meeting_id={}".format(
                doctor_name,
                patient_name,
                state.get("current_session", ""),
                bool(forced),
                meeting_id,
            )
        )
        return block

    def on_chat_finished(
        self,
        doctor_name: str,
        patient_name: str,
        chats: List[Any],
        meeting_id: str,
        now_str: str,
        session_eval_end: Optional[bool] = None,
    ) -> Dict[str, Any]:
        if not self.enabled:
            return {"applied": False, "reason": "disabled"}

        state = self.resolve_current_session(doctor_name, patient_name)
        session_before = str(state.get("current_session", "") or "")
        if session_eval_end is None:
            detected_end = self.detect_session_end(chats or [], doctor_name)
        else:
            detected_end = bool(session_eval_end is True)
        pair_key = self._pair_key(doctor_name, patient_name)
        audit = self.advance_session_if_needed(pair_key, detected_end, meeting_id, now_str)
        audit["applied"] = True
        audit["pair_key"] = pair_key
        audit["session_before"] = session_before
        self._log_highlight(
            "after_chat pair={} detected_end={} action={} current_session={} completed={}".format(
                pair_key,
                bool(detected_end),
                audit.get("action", ""),
                audit.get("current_session", ""),
                bool(audit.get("completed", False)),
            )
        )
        return audit

    def detect_session_end(self, chats: List[Any], doctor_name: str) -> bool:
        marker = self.advance_marker
        if not marker:
            return False
        for item in chats:
            if not isinstance(item, (tuple, list)) or len(item) < 2:
                continue
            speaker = str(item[0])
            text = str(item[1])
            if speaker == doctor_name and marker in text:
                return True
        return False

    def resolve_current_session(self, doctor_name: str, patient_name: str) -> Dict[str, Any]:
        pairs = self._pairs_state()
        pair_key = self._pair_key(doctor_name, patient_name)
        state = pairs.setdefault(
            pair_key,
            {
                "current_index": 0,
                "current_session": "",
                "completed": False,
                "last_meeting_id": "",
                "last_detected_end": False,
                "last_advance_meeting_id": "",
                "updated_at": "",
            },
        )

        order = self._order()
        if not order:
            state["current_index"] = 0
            state["current_session"] = ""
            state["completed"] = True
            state["updated_at"] = self._now_str()
            return state

        current_session = str(state.get("current_session", "") or "")
        if current_session not in order:
            state["current_index"] = 0
            state["current_session"] = order[0]
            state["completed"] = False
            state["updated_at"] = self._now_str()
            return state

        state["current_index"] = order.index(current_session)
        return state

    def advance_session_if_needed(
        self,
        pair_key: str,
        detected_end: bool,
        meeting_id: str,
        now_str: str,
    ) -> Dict[str, Any]:
        pairs = self._pairs_state()
        state = pairs.get(pair_key, {})
        if not isinstance(state, dict):
            return {
                "action": "skip_invalid_state",
                "current_session": "",
                "completed": False,
            }

        order = self._order()
        if not order:
            state["completed"] = True
            state["updated_at"] = now_str or self._now_str()
            return {
                "action": "skip_empty_order",
                "current_session": "",
                "completed": True,
            }

        current_idx = int(state.get("current_index", 0) or 0)
        current_idx = max(0, min(current_idx, len(order) - 1))
        state["current_index"] = current_idx
        state["current_session"] = order[current_idx]

        if not bool(detected_end):
            state["last_detected_end"] = False
            state["last_meeting_id"] = str(meeting_id or state.get("last_meeting_id", ""))
            state["updated_at"] = now_str or self._now_str()
            return {
                "action": "keep_current",
                "current_session": state.get("current_session", ""),
                "completed": bool(state.get("completed", False)),
            }

        if meeting_id and str(state.get("last_advance_meeting_id", "")) == str(meeting_id):
            state["last_detected_end"] = True
            state["updated_at"] = now_str or self._now_str()
            return {
                "action": "skip_duplicate_meeting",
                "current_session": state.get("current_session", ""),
                "completed": bool(state.get("completed", False)),
            }

        if current_idx < len(order) - 1:
            next_idx = current_idx + 1
            state["current_index"] = next_idx
            state["current_session"] = order[next_idx]
            state["completed"] = False
            action = "advanced"
        else:
            state["completed"] = True
            action = "completed_stop_injection"

        state["last_detected_end"] = True
        state["last_meeting_id"] = str(meeting_id or state.get("last_meeting_id", ""))
        state["last_advance_meeting_id"] = str(meeting_id or state.get("last_advance_meeting_id", ""))
        state["updated_at"] = now_str or self._now_str()
        return {
            "action": action,
            "current_session": state.get("current_session", ""),
            "completed": bool(state.get("completed", False)),
        }

    def load_session_prompt_map(self) -> Dict[str, str]:
        if not self.prompt_file:
            return {}
        try:
            payload = utils.load_dict(self.prompt_file)
        except Exception as err:
            self._log("load prompt file failed path={} err={}".format(self.prompt_file, err))
            return {}

        if not isinstance(payload, dict):
            self._log("invalid prompt payload type={} path={}".format(type(payload), self.prompt_file))
            return {}

        namespace_map = payload.get(self.namespace, {})
        if not isinstance(namespace_map, dict):
            self._log("invalid namespace map namespace={} path={}".format(self.namespace, self.prompt_file))
            return {}
        return namespace_map

    def _find_next_available_session(
        self,
        order: List[str],
        namespace_map: Dict[str, str],
        current_index: int,
    ):
        for idx in range(max(0, int(current_index or 0)), len(order)):
            sid = order[idx]
            if str(namespace_map.get(sid, "") or "").strip():
                return idx, sid
        for idx, sid in enumerate(order):
            if str(namespace_map.get(sid, "") or "").strip():
                return idx, sid
        return None

    def _order(self) -> List[str]:
        raw = self.injection_cfg.get("order", []) or []
        if not isinstance(raw, list):
            return []
        result = []
        for item in raw:
            sid = str(item or "").strip()
            if not sid:
                continue
            if sid not in result:
                result.append(sid)
        return result

    def _pairs_state(self) -> Dict[str, Any]:
        session_prompt_state = self.state.setdefault("session_prompt_state", {})
        if not isinstance(session_prompt_state, dict):
            session_prompt_state = {}
            self.state["session_prompt_state"] = session_prompt_state
        pairs = session_prompt_state.setdefault("pairs", {})
        if not isinstance(pairs, dict):
            pairs = {}
            session_prompt_state["pairs"] = pairs
        return pairs

    def _pair_key(self, doctor_name: str, patient_name: str) -> str:
        return "{}::{}".format(str(doctor_name or "").strip(), str(patient_name or "").strip())

    def _now_str(self) -> str:
        try:
            return utils.get_timer().get_date("%Y%m%d-%H:%M:%S")
        except Exception:
            return ""

    def _log(self, message: str) -> None:
        if self.logger:
            self.logger.info("[SESSION_PROMPT_INJECTION] {}".format(message))

    def _log_highlight(self, message: str) -> None:
        text = "========== [SESSION_PROMPT_INJECTION] {} ==========".format(message)
        print(text)
        self._log(message)
