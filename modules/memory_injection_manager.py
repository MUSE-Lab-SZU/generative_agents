"""Memory injection interface manager."""

from __future__ import annotations

import copy
import datetime
import hashlib
import json
import uuid
from typing import Any, Dict, List, Optional, Tuple

from modules import utils
from modules.memory.event import Event


class MemoryInjectionManager:
    _TIME_FMT = "%Y%m%d-%H:%M:%S"
    _VALID_NODE_TYPES = {"event", "thought", "chat"}
    _VALID_DEDUP_MODES = {"none", "request", "global"}
    _CURRENT_TILE_SENTINEL = "<current_tile>"
    _INIT_RULE_WHITELIST = {"init_kabuda_depression_stress"}
    _SESSION_RULE_WHITELIST = {"session_completed_memory_rule_kbd"}

    def __init__(self, config: Dict[str, Any], state: Dict[str, Any], logger: Any = None):
        self.config = config if isinstance(config, dict) else {}
        self.state = state if isinstance(state, dict) else {}
        self.logger = logger
        self._cfg = self._resolve_runtime_config()
        self._memory_state = self._ensure_state_schema()
        self._content_cache: Dict[str, Any] = {}
        self._content_cache_file: str = ""
        self._log_info(
            "INIT enabled={} content_file={} rules={} defaults={} limits={}".format(
                self._cfg["enabled"],
                self._cfg["content_file"],
                len(self._resolve_rules_config()),
                self._cfg["defaults"],
                self._cfg["limits"],
            )
        )

    def validate_request(self, payload: Dict[str, Any], now: Any = None) -> Dict[str, Any]:
        now_dt = self._resolve_now(now)
        req_id = self._safe_str((payload or {}).get("request_id")) if isinstance(payload, dict) else ""
        req_id = req_id or self._build_request_id()
        result = {
            "status": "ok",
            "reason": "",
            "request_id": req_id,
            "target_agent": "",
            "source": "",
            "dry_run": False,
            "accepted": 0,
            "skipped": 0,
            "errors": [],
            "records": [],
            "normalized": {
                "request_id": req_id,
                "target_agent": "",
                "source": "",
                "options": {},
                "meta": {},
                "items": [],
            },
        }
        if not self._cfg["enabled"]:
            result["status"] = "disabled"
            result["reason"] = "memory_injection_disabled"
            return result
        if not isinstance(payload, dict):
            result["status"] = "invalid"
            result["reason"] = "payload_must_be_dict"
            result["errors"].append(
                {"item_id": "", "code": "payload_type_invalid", "message": "payload must be dict"}
            )
            return result

        target_agent = self._safe_str(payload.get("target_agent"))
        if not target_agent:
            result["status"] = "invalid"
            result["reason"] = "target_agent_required"
            result["errors"].append(
                {"item_id": "", "code": "target_agent_required", "message": "target_agent is required"}
            )
            return result

        source = self._safe_str(payload.get("source"))
        options = self._merge_options(payload.get("options"))
        meta = payload.get("meta") if isinstance(payload.get("meta"), dict) else {}
        items_raw = payload.get("items")
        result["target_agent"] = target_agent
        result["source"] = source
        result["dry_run"] = bool(options.get("dry_run", False))
        result["normalized"]["target_agent"] = target_agent
        result["normalized"]["source"] = source
        result["normalized"]["options"] = options
        result["normalized"]["meta"] = meta

        if not isinstance(items_raw, list) or len(items_raw) < 1:
            result["status"] = "invalid"
            result["reason"] = "items_required"
            result["errors"].append(
                {"item_id": "", "code": "items_required", "message": "items must be a non-empty list"}
            )
            return result

        items = list(items_raw)
        max_items = self._cfg["limits"]["max_items_per_request"]
        if len(items) > max_items:
            result["errors"].append(
                {
                    "item_id": "",
                    "code": "items_truncated",
                    "message": "items truncated to max_items_per_request={}".format(max_items),
                }
            )
            items = items[:max_items]

        step_no = self._resolve_step_no(meta)
        request_seen = set()
        for idx, item in enumerate(items):
            normalized, item_errors = self._normalize_item(item, idx, target_agent, now_dt)
            if item_errors:
                result["skipped"] += 1
                result["errors"].extend(item_errors)
                result["records"].append(
                    {
                        "item_id": self._safe_str((item or {}).get("item_id")) if isinstance(item, dict) else "",
                        "node_type": self._safe_str((item or {}).get("node_type")) if isinstance(item, dict) else "",
                        "status": "skipped",
                        "reason": "validation_failed",
                    }
                )
                continue

            dedup_reason, fp = self._check_dedup(target_agent, normalized, options["dedup_mode"], request_seen, step_no)
            if dedup_reason:
                result["skipped"] += 1
                result["records"].append(
                    {
                        "item_id": normalized["item_id"],
                        "node_type": normalized["node_type"],
                        "status": "skipped",
                        "reason": dedup_reason,
                    }
                )
                continue
            normalized["_fingerprint"] = fp
            result["normalized"]["items"].append(normalized)
            result["accepted"] += 1

        if result["accepted"] < 1 and result["status"] == "ok":
            result["status"] = "invalid"
            result["reason"] = "no_valid_items"
        return result

    def inject(self, agent: Any, payload: Dict[str, Any], now: Any = None) -> Dict[str, Any]:
        validation = self.validate_request(payload, now=now)
        options = validation["normalized"].get("options", {}) if isinstance(validation.get("normalized"), dict) else {}
        result = {
            "status": validation["status"],
            "reason": validation.get("reason", ""),
            "request_id": validation.get("request_id", ""),
            "target_agent": validation.get("target_agent", ""),
            "dry_run": bool(options.get("dry_run", False)),
            "accepted": int(validation.get("accepted", 0) or 0),
            "injected": 0,
            "skipped": int(validation.get("skipped", 0) or 0),
            "errors": list(validation.get("errors", [])),
            "records": list(validation.get("records", [])),
        }
        if validation["status"] in {"disabled", "invalid"}:
            self._append_request_audit(result)
            return result
        if agent is None:
            result["status"] = "invalid"
            result["reason"] = "agent_required"
            result["errors"].append(
                {"item_id": "", "code": "agent_required", "message": "agent is required for inject"}
            )
            self._append_request_audit(result)
            return result

        allow_partial = bool(options.get("allow_partial_success", True))
        items = validation["normalized"].get("items", [])
        if result["dry_run"]:
            for item in items:
                result["records"].append(
                    {
                        "item_id": item["item_id"],
                        "node_type": item["node_type"],
                        "status": "dry_run",
                        "poignancy": item["poignancy"],
                        "create": item["create"],
                        "expire": item["expire"],
                    }
                )
            self._append_request_audit(result)
            return result

        if (not allow_partial) and result["errors"]:
            result["status"] = "failed"
            result["reason"] = "validation_error_abort"
            result["skipped"] += len(items)
            for item in items:
                result["records"].append(
                    {
                        "item_id": item["item_id"],
                        "node_type": item["node_type"],
                        "status": "skipped",
                        "reason": "request_aborted_by_validation",
                    }
                )
            self._append_request_audit(result)
            return result

        dedup_mode = options.get("dedup_mode", "global")
        step_no = self._resolve_step_no(validation["normalized"].get("meta", {}))
        written = []
        for item in items:
            try:
                event = Event(
                    subject=item["subject"],
                    predicate=item["predicate"],
                    object=item["object"],
                    address=self._resolve_runtime_address(item["address"], agent),
                    describe=item["describe"],
                )
                concept = agent.associate.add_node(
                    item["node_type"],
                    event,
                    item["poignancy"],
                    create=item["_create_dt"],
                    expire=item["_expire_dt"],
                )
                node_id = str(getattr(concept, "node_id", "") or "")
                written.append({"node_id": node_id})
                result["injected"] += 1
                result["records"].append(
                    {
                        "item_id": item["item_id"],
                        "node_type": item["node_type"],
                        "status": "ok",
                        "node_id": node_id,
                        "poignancy": item["poignancy"],
                        "create": item["create"],
                        "expire": item["expire"],
                    }
                )
                if dedup_mode == "global":
                    self._register_fingerprint(item["_fingerprint"], result["request_id"], item["item_id"], step_no)
            except Exception as exc:
                result["errors"].append(
                    {"item_id": item["item_id"], "code": "inject_failed", "message": str(exc)}
                )
                result["records"].append(
                    {
                        "item_id": item["item_id"],
                        "node_type": item["node_type"],
                        "status": "skipped",
                        "reason": "inject_failed",
                    }
                )
                result["skipped"] += 1
                self._log_warn(
                    "INJECT_SKIP request_id={} item_id={} reason=inject_failed detail={}".format(
                        result["request_id"], item["item_id"], str(exc)
                    )
                )
                if not allow_partial:
                    rolled_back = self._rollback_written(agent, written)
                    if rolled_back:
                        result["injected"] = 0
                        for rec in result["records"]:
                            if rec.get("status") == "ok":
                                rec["status"] = "rolled_back"
                    result["status"] = "failed"
                    result["reason"] = "inject_runtime_error_abort"
                    break

        if result["status"] == "ok":
            result["reason"] = "partial_success" if result["errors"] else "success"
            if result["errors"]:
                result["status"] = "partial"
        self._append_request_audit(result)
        self._log_info(
            "REQUEST_SUMMARY request_id={} status={} accepted={} injected={} skipped={} errors={}".format(
                result["request_id"], result["status"], result["accepted"], result["injected"], result["skipped"], len(result["errors"])
            )
        )
        return result

    def inject_many(self, agents: Dict[str, Any], payloads: List[Dict[str, Any]], now: Any = None) -> Dict[str, Any]:
        summary = {"status": "ok", "requests": 0, "succeeded": 0, "failed": 0, "results": []}
        if not isinstance(payloads, list):
            summary["status"] = "invalid"
            summary["failed"] = 1
            summary["results"].append(
                {
                    "status": "invalid",
                    "reason": "payloads_must_be_list",
                    "request_id": "",
                    "target_agent": "",
                    "accepted": 0,
                    "injected": 0,
                    "skipped": 0,
                    "errors": [{"item_id": "", "code": "payloads_type_invalid", "message": "payloads must be list"}],
                    "records": [],
                }
            )
            return summary
        for payload in payloads:
            summary["requests"] += 1
            target = self._safe_str((payload or {}).get("target_agent")) if isinstance(payload, dict) else ""
            agent = agents.get(target) if isinstance(agents, dict) else None
            item_result = self.inject(agent=agent, payload=payload if isinstance(payload, dict) else {}, now=now)
            summary["results"].append(item_result)
            if item_result["status"] in {"ok", "partial"}:
                summary["succeeded"] += 1
            else:
                summary["failed"] += 1
        if summary["failed"] > 0:
            summary["status"] = "partial" if summary["succeeded"] > 0 else "failed"
        return summary

    def apply_rules_once(self, agents: Dict[str, Any], now: Any = None, step_no: Optional[int] = None) -> Dict[str, Any]:
        now_dt = self._resolve_now(now)
        resolved_step = self._resolve_apply_step_no(step_no)
        summary: Dict[str, Any] = {
            "status": "ok",
            "reason": "",
            "step": resolved_step,
            "executed": 0,
            "skipped": 0,
            "failed": 0,
            "results": [],
        }
        if not self._cfg["enabled"]:
            summary["status"] = "disabled"
            summary["reason"] = "memory_injection_disabled"
            return summary
        if not self._is_init_step(resolved_step):
            summary["status"] = "skipped"
            summary["reason"] = "non_init_step"
            self._log_info("RULE_SKIP reason=non_init_step step={}".format(resolved_step))
            return summary

        rules = self._resolve_rules_config()
        if not rules:
            summary["status"] = "skipped"
            summary["reason"] = "rules_empty"
            self._log_info("RULE_SKIP reason=rules_empty")
            return summary

        payload, load_reason = self._load_content_payload()
        if load_reason:
            summary["status"] = "failed"
            summary["reason"] = load_reason
            summary["failed"] = len(rules)
            return summary
        entries_by_rule = self._map_entries_by_rule(payload)
        applied_rules = self._memory_state.setdefault("applied_rules", {})
        if not isinstance(applied_rules, dict):
            applied_rules = {}
            self._memory_state["applied_rules"] = applied_rules
        agents_map = agents if isinstance(agents, dict) else {}

        self._log_info(
            "RULE_SCAN start step={} rules={} content_file={}".format(
                resolved_step,
                len(rules),
                self._cfg["content_file"],
            )
        )
        for rule_id, rule_cfg in rules.items():
            if not self._safe_bool(rule_cfg.get("enabled"), True):
                summary["skipped"] += 1
                summary["results"].append(
                    {"rule_id": rule_id, "status": "skipped", "reason": "disabled", "request_id": ""}
                )
                self._log_info("RULE_SKIP rule={} reason=disabled".format(rule_id))
                continue
            if rule_id not in self._INIT_RULE_WHITELIST:
                summary["skipped"] += 1
                summary["results"].append(
                    {
                        "rule_id": rule_id,
                        "status": "skipped",
                        "reason": "not_init_rule_whitelist",
                        "request_id": "",
                    }
                )
                self._log_info("RULE_SKIP rule={} reason=not_init_rule_whitelist".format(rule_id))
                continue
            if rule_id in applied_rules:
                summary["skipped"] += 1
                summary["results"].append(
                    {"rule_id": rule_id, "status": "skipped", "reason": "already_applied", "request_id": ""}
                )
                self._log_info("RULE_SKIP rule={} reason=already_applied".format(rule_id))
                continue

            items = entries_by_rule.get(rule_id, [])
            if not items:
                summary["skipped"] += 1
                summary["results"].append(
                    {"rule_id": rule_id, "status": "skipped", "reason": "content_missing", "request_id": ""}
                )
                self._log_warn("RULE_SKIP rule={} reason=content_missing".format(rule_id))
                continue

            payload_req, build_reason = self._build_payload_from_rule(
                rule_id=rule_id,
                rule_cfg=rule_cfg,
                items=items,
                step_no=resolved_step,
            )
            if build_reason:
                summary["failed"] += 1
                summary["results"].append(
                    {"rule_id": rule_id, "status": "failed", "reason": build_reason, "request_id": ""}
                )
                self._log_warn("RULE_ERROR rule={} detail={}".format(rule_id, build_reason))
                continue

            target_agent = self._safe_str(payload_req.get("target_agent"))
            source = self._safe_str(payload_req.get("source"))
            self._log_info(
                "RULE_APPLY rule={} target={} items={} source={}".format(
                    rule_id,
                    target_agent,
                    len(payload_req.get("items", []) or []),
                    source,
                )
            )
            result = self.inject(
                agent=agents_map.get(target_agent),
                payload=payload_req,
                now=now_dt,
            )
            status = self._safe_str(result.get("status"))
            reason = self._safe_str(result.get("reason"))
            injected = int(result.get("injected", 0) or 0)
            skipped = int(result.get("skipped", 0) or 0)
            errors_n = len(result.get("errors", []))
            req_id = self._safe_str(result.get("request_id"))
            summary["results"].append(
                {
                    "rule_id": rule_id,
                    "request_id": req_id,
                    "target_agent": target_agent,
                    "status": status,
                    "reason": reason,
                    "injected": injected,
                    "skipped": skipped,
                    "errors": errors_n,
                }
            )
            self._log_info(
                "RULE_DONE rule={} status={} injected={} skipped={} errors={}".format(
                    rule_id,
                    status,
                    injected,
                    skipped,
                    errors_n,
                )
            )
            if status in {"ok", "partial"}:
                summary["executed"] += 1
                applied_rules[rule_id] = {
                    "applied_at": now_dt.strftime(self._TIME_FMT),
                    "step": resolved_step,
                    "request_id": req_id,
                    "source": source,
                    "target_agent": target_agent,
                    "status": status,
                    "injected": injected,
                }
                self._log_info(
                    "RULE_MARKED_APPLIED rule={} at_step={}".format(
                        rule_id,
                        resolved_step,
                    )
                )
            else:
                summary["failed"] += 1

        if summary["executed"] > 0 and summary["failed"] > 0:
            summary["status"] = "partial"
            summary["reason"] = "partial_success"
        elif summary["executed"] > 0:
            summary["status"] = "ok"
            summary["reason"] = "success"
        elif summary["failed"] > 0:
            summary["status"] = "failed"
            summary["reason"] = "all_failed"
        else:
            summary["status"] = "skipped"
            summary["reason"] = "no_rule_executed"
        return summary

    def apply_rules_on_session_completed(
        self,
        agents: Dict[str, Any],
        doctor_name: str,
        patient_name: str,
        audit: Dict[str, Any],
        meeting_id: str,
        now: Any = None,
        step_no: Optional[int] = None,
    ) -> Dict[str, Any]:
        now_dt = self._resolve_now(now)
        resolved_step = self._resolve_apply_step_no(step_no)
        action = self._safe_str((audit or {}).get("action")) if isinstance(audit, dict) else ""
        session_before = self._safe_str((audit or {}).get("session_before")) if isinstance(audit, dict) else ""
        doctor = self._safe_str(doctor_name)
        patient = self._safe_str(patient_name)
        meeting = self._safe_str(meeting_id)
        summary: Dict[str, Any] = {
            "status": "ok",
            "reason": "",
            "step": resolved_step,
            "action": action,
            "session": session_before,
            "doctor": doctor,
            "patient": patient,
            "meeting_id": meeting,
            "executed": 0,
            "skipped": 0,
            "failed": 0,
            "results": [],
        }
        if not self._cfg["enabled"]:
            summary["status"] = "disabled"
            summary["reason"] = "memory_injection_disabled"
            return summary
        if not isinstance(audit, dict):
            summary["status"] = "skipped"
            summary["reason"] = "invalid_audit"
            return summary
        if not self._is_session_completion_action(action):
            summary["status"] = "skipped"
            summary["reason"] = "non_completion_action"
            self._log_info("SESSION_RULE_SKIP reason=non_completion_action action={}".format(action or "<empty>"))
            return summary
        if not session_before:
            summary["status"] = "skipped"
            summary["reason"] = "session_before_missing"
            self._log_info("SESSION_RULE_SKIP reason=session_before_missing")
            return summary
        if not patient:
            summary["status"] = "skipped"
            summary["reason"] = "patient_missing"
            self._log_info("SESSION_RULE_SKIP reason=patient_missing")
            return summary
        if not meeting:
            summary["status"] = "skipped"
            summary["reason"] = "meeting_id_missing"
            self._log_info(
                "SESSION_RULE_SKIP reason=meeting_id_missing patient={} session={}".format(
                    patient,
                    session_before,
                )
            )
            return summary

        rules = self._resolve_rules_config()
        if not rules:
            summary["status"] = "skipped"
            summary["reason"] = "rules_empty"
            self._log_info("SESSION_RULE_SKIP reason=rules_empty")
            return summary

        payload, load_reason = self._load_content_payload()
        if load_reason:
            summary["status"] = "failed"
            summary["reason"] = load_reason
            summary["failed"] = len(rules)
            return summary
        entries_by_rule = self._map_entries_by_rule(payload)
        applied_session_rules = self._memory_state.setdefault("applied_session_rules", {})
        if not isinstance(applied_session_rules, dict):
            applied_session_rules = {}
            self._memory_state["applied_session_rules"] = applied_session_rules
        agents_map = agents if isinstance(agents, dict) else {}

        self._log_info(
            "SESSION_RULE_SCAN patient={} doctor={} session={} action={} meeting_id={} rules={}".format(
                patient,
                doctor,
                session_before,
                action,
                meeting,
                len(rules),
            )
        )
        for rule_id, rule_cfg in rules.items():
            if not self._safe_bool(rule_cfg.get("enabled"), True):
                summary["skipped"] += 1
                summary["results"].append(
                    {"rule_id": rule_id, "status": "skipped", "reason": "disabled", "request_id": ""}
                )
                self._log_info("SESSION_RULE_SKIP rule={} reason=disabled".format(rule_id))
                continue
            if rule_id not in self._SESSION_RULE_WHITELIST:
                summary["skipped"] += 1
                summary["results"].append(
                    {
                        "rule_id": rule_id,
                        "status": "skipped",
                        "reason": "not_session_rule_whitelist",
                        "request_id": "",
                    }
                )
                self._log_info("SESSION_RULE_SKIP rule={} reason=not_session_rule_whitelist".format(rule_id))
                continue

            target_agent = self._safe_str(rule_cfg.get("target_agent"))
            if not target_agent:
                summary["failed"] += 1
                summary["results"].append(
                    {"rule_id": rule_id, "status": "failed", "reason": "target_agent_missing", "request_id": ""}
                )
                self._log_warn("SESSION_RULE_ERROR rule={} detail=target_agent_missing".format(rule_id))
                continue
            if target_agent != patient:
                summary["skipped"] += 1
                summary["results"].append(
                    {
                        "rule_id": rule_id,
                        "status": "skipped",
                        "reason": "patient_mismatch",
                        "request_id": "",
                        "target_agent": target_agent,
                    }
                )
                self._log_info(
                    "SESSION_RULE_SKIP rule={} reason=patient_mismatch target={} patient={}".format(
                        rule_id,
                        target_agent,
                        patient,
                    )
                )
                continue

            trigger_cfg = rule_cfg.get("trigger", {}) if isinstance(rule_cfg.get("trigger"), dict) else {}
            trigger_sessions = self._normalize_sessions(trigger_cfg.get("sessions", []))
            if not trigger_sessions:
                summary["skipped"] += 1
                summary["results"].append(
                    {
                        "rule_id": rule_id,
                        "status": "skipped",
                        "reason": "trigger_sessions_empty",
                        "request_id": "",
                    }
                )
                self._log_warn("SESSION_RULE_SKIP rule={} reason=trigger_sessions_empty".format(rule_id))
                continue
            if session_before not in trigger_sessions:
                summary["skipped"] += 1
                summary["results"].append(
                    {
                        "rule_id": rule_id,
                        "status": "skipped",
                        "reason": "session_not_matched",
                        "request_id": "",
                    }
                )
                self._log_info(
                    "SESSION_RULE_SKIP rule={} reason=session_not_matched session={} configured={}".format(
                        rule_id,
                        session_before,
                        ",".join(trigger_sessions) or "-",
                    )
                )
                continue

            items = entries_by_rule.get(rule_id, [])
            if not items:
                summary["skipped"] += 1
                summary["results"].append(
                    {"rule_id": rule_id, "status": "skipped", "reason": "content_missing", "request_id": ""}
                )
                self._log_warn("SESSION_RULE_SKIP rule={} reason=content_missing".format(rule_id))
                continue

            matched_items = [
                copy.deepcopy(item)
                for item in items
                if isinstance(item, dict) and self._item_matches_session(item, session_before)
            ]
            if not matched_items:
                summary["skipped"] += 1
                summary["results"].append(
                    {
                        "rule_id": rule_id,
                        "status": "skipped",
                        "reason": "no_items_for_session",
                        "request_id": "",
                    }
                )
                self._log_info(
                    "SESSION_RULE_SKIP rule={} reason=no_items_for_session session={}".format(
                        rule_id,
                        session_before,
                    )
                )
                continue

            max_items = self._resolve_max_items_per_session(rule_cfg)
            if len(matched_items) > max_items:
                matched_items = sorted(
                    matched_items,
                    key=lambda x: self._safe_int(
                        x.get("poignancy"),
                        self._cfg["defaults"]["poignancy"],
                        minimum=1,
                    ),
                    reverse=True,
                )[:max_items]

            matched_item_ids = []
            for item in matched_items:
                item_id = self._safe_str(item.get("item_id"))
                if item_id:
                    matched_item_ids.append(item_id)
            self._log_info(
                "SESSION_ITEM_MATCH rule={} session={} matched_item_ids={} total={}".format(
                    rule_id,
                    session_before,
                    ",".join(matched_item_ids) or "-",
                    len(matched_items),
                )
            )

            event_key = self._build_session_event_key(
                rule_id=rule_id,
                target_agent=target_agent,
                session_before=session_before,
                meeting_id=meeting,
            )
            if event_key in applied_session_rules:
                summary["skipped"] += 1
                summary["results"].append(
                    {
                        "rule_id": rule_id,
                        "status": "skipped",
                        "reason": "already_applied_event",
                        "request_id": "",
                    }
                )
                self._log_info(
                    "SESSION_RULE_SKIP rule={} reason=already_applied_event event_key={}".format(
                        rule_id,
                        event_key,
                    )
                )
                continue

            payload_req, build_reason = self._build_payload_from_rule(
                rule_id=rule_id,
                rule_cfg=rule_cfg,
                items=matched_items,
                step_no=resolved_step,
                scene="session_completed",
                extra_meta={
                    "session": session_before,
                    "meeting_id": meeting,
                    "action": action,
                    "doctor": doctor,
                    "patient": patient,
                },
            )
            if build_reason:
                summary["failed"] += 1
                summary["results"].append(
                    {"rule_id": rule_id, "status": "failed", "reason": build_reason, "request_id": ""}
                )
                self._log_warn("SESSION_RULE_ERROR rule={} detail={}".format(rule_id, build_reason))
                continue

            source = self._safe_str(payload_req.get("source"))
            self._log_info(
                "SESSION_RULE_APPLY rule={} target={} session={} meeting_id={} items={} source={}".format(
                    rule_id,
                    target_agent,
                    session_before,
                    meeting,
                    len(payload_req.get("items", []) or []),
                    source,
                )
            )
            result = self.inject(
                agent=agents_map.get(target_agent),
                payload=payload_req,
                now=now_dt,
            )
            status = self._safe_str(result.get("status"))
            reason = self._safe_str(result.get("reason"))
            injected = int(result.get("injected", 0) or 0)
            skipped = int(result.get("skipped", 0) or 0)
            errors_n = len(result.get("errors", []))
            req_id = self._safe_str(result.get("request_id"))
            summary["results"].append(
                {
                    "rule_id": rule_id,
                    "request_id": req_id,
                    "target_agent": target_agent,
                    "status": status,
                    "reason": reason,
                    "injected": injected,
                    "skipped": skipped,
                    "errors": errors_n,
                }
            )
            self._log_info(
                "SESSION_RULE_DONE rule={} status={} injected={} skipped={} errors={}".format(
                    rule_id,
                    status,
                    injected,
                    skipped,
                    errors_n,
                )
            )
            if status in {"ok", "partial"}:
                summary["executed"] += 1
                applied_session_rules[event_key] = {
                    "applied_at": now_dt.strftime(self._TIME_FMT),
                    "step": resolved_step,
                    "request_id": req_id,
                    "source": source,
                    "target_agent": target_agent,
                    "doctor": doctor,
                    "patient": patient,
                    "meeting_id": meeting,
                    "session": session_before,
                    "action": action,
                    "status": status,
                    "injected": injected,
                    "item_ids": matched_item_ids,
                }
                self._log_info(
                    "SESSION_RULE_MARKED_APPLIED rule={} event_key={}".format(
                        rule_id,
                        event_key,
                    )
                )
            else:
                summary["failed"] += 1

        if summary["executed"] > 0 and summary["failed"] > 0:
            summary["status"] = "partial"
            summary["reason"] = "partial_success"
        elif summary["executed"] > 0:
            summary["status"] = "ok"
            summary["reason"] = "success"
        elif summary["failed"] > 0:
            summary["status"] = "failed"
            summary["reason"] = "all_failed"
        else:
            summary["status"] = "skipped"
            summary["reason"] = "no_rule_executed"
        return summary

    def _resolve_runtime_config(self) -> Dict[str, Any]:
        intervention_cfg = self.config.get("intervention", {}) or {}
        mi_cfg = intervention_cfg.get("memory_injection", {}) or {}
        defaults_cfg = mi_cfg.get("defaults", {}) or {}
        limits_cfg = mi_cfg.get("limits", {}) or {}
        content_file = self._safe_str(mi_cfg.get("content_file")) or "data/intervention/memory_injections.json"
        rules = mi_cfg.get("rules", {}) or {}
        if not isinstance(rules, dict):
            rules = {}
        addr_mode = self._safe_str(defaults_cfg.get("address_mode")).lower() or "current_tile"
        addr_mode = addr_mode if addr_mode in {"current_tile", "empty"} else "current_tile"
        dedup_mode = self._safe_str(defaults_cfg.get("dedup_mode")).lower() or "global"
        dedup_mode = dedup_mode if dedup_mode in self._VALID_DEDUP_MODES else "global"
        defaults = {
            "poignancy": self._normalize_poignancy(defaults_cfg.get("poignancy"), 5)[0],
            "expire_days": self._safe_int(defaults_cfg.get("expire_days"), 30, minimum=1),
            "address_mode": addr_mode,
            "allow_partial_success": self._safe_bool(defaults_cfg.get("allow_partial_success"), True),
            "dedup_mode": dedup_mode,
        }
        limits = {
            "max_items_per_request": self._safe_int(limits_cfg.get("max_items_per_request"), 20, minimum=1),
            "max_describe_chars": self._safe_int(limits_cfg.get("max_describe_chars"), 1000, minimum=1),
            "max_audit_entries": self._safe_int(limits_cfg.get("max_audit_entries"), 2000, minimum=10),
            "max_fingerprints": self._safe_int(limits_cfg.get("max_fingerprints"), 20000, minimum=100),
        }
        return {
            "enabled": self._safe_bool(mi_cfg.get("enabled"), False),
            "content_file": content_file,
            "rules": rules,
            "defaults": defaults,
            "limits": limits,
        }

    def _ensure_state_schema(self) -> Dict[str, Any]:
        if not isinstance(self.state, dict):
            self.state = {}
        memory_state = self.state.setdefault("memory_injection_state", {})
        if not isinstance(memory_state, dict):
            memory_state = {}
            self.state["memory_injection_state"] = memory_state
        if not isinstance(memory_state.get("seen_fingerprints"), dict):
            memory_state["seen_fingerprints"] = {}
        if not isinstance(memory_state.get("request_audit"), list):
            memory_state["request_audit"] = []
        if not isinstance(memory_state.get("applied_rules"), dict):
            memory_state["applied_rules"] = {}
        if not isinstance(memory_state.get("applied_session_rules"), dict):
            memory_state["applied_session_rules"] = {}
        return memory_state

    def _resolve_rules_config(self) -> Dict[str, Dict[str, Any]]:
        raw = self._cfg.get("rules", {}) or {}
        if not isinstance(raw, dict):
            return {}
        out: Dict[str, Dict[str, Any]] = {}
        for key, value in raw.items():
            rule_id = self._safe_str(key)
            if (not rule_id) or (not isinstance(value, dict)):
                continue
            out[rule_id] = value
        return out

    def _load_content_payload(self) -> Tuple[Dict[str, Any], str]:
        path = self._safe_str(self._cfg.get("content_file"))
        if not path:
            self._log_warn("CONTENT_LOAD_FAIL file=<empty> detail=content_file_empty")
            return {}, "content_file_empty"
        if self._content_cache_file == path and isinstance(self._content_cache, dict) and self._content_cache:
            return self._content_cache, ""
        try:
            payload = utils.load_dict(path)
        except Exception as exc:
            self._log_warn("CONTENT_LOAD_FAIL file={} detail={}".format(path, str(exc)))
            return {}, "content_load_failed"
        if not isinstance(payload, dict):
            self._log_warn("CONTENT_LOAD_FAIL file={} detail=payload_not_dict".format(path))
            return {}, "content_payload_not_dict"
        self._content_cache_file = path
        self._content_cache = payload
        return payload, ""

    def _map_entries_by_rule(self, content_payload: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
        mapped: Dict[str, List[Dict[str, Any]]] = {}
        entries = content_payload.get("entries", []) if isinstance(content_payload, dict) else []
        if not isinstance(entries, list):
            return mapped
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            rule_id = self._safe_str(entry.get("rule"))
            if not rule_id:
                continue
            items = entry.get("items", [])
            if not isinstance(items, list):
                continue
            bucket = mapped.setdefault(rule_id, [])
            for item in items:
                if isinstance(item, dict):
                    bucket.append(item)
        return mapped

    def _build_payload_from_rule(
        self,
        rule_id: str,
        rule_cfg: Dict[str, Any],
        items: List[Dict[str, Any]],
        step_no: int,
        scene: str = "town_init",
        extra_meta: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, Any], str]:
        target_agent = self._safe_str(rule_cfg.get("target_agent"))
        if not target_agent:
            return {}, "target_agent_missing"
        source = self._safe_str(rule_cfg.get("source")) or "初始化规则注入_{}".format(rule_id)
        options = rule_cfg.get("options", {})
        if not isinstance(options, dict):
            options = {}
        normalized_items = [copy.deepcopy(item) for item in items if isinstance(item, dict)]
        if not normalized_items:
            return {}, "items_empty"
        request_id = "rule_{}_{}".format(rule_id, self._build_request_id())
        meta = {
            "step": step_no,
            "rule_id": rule_id,
            "scene": self._safe_str(scene) or "town_init",
        }
        if isinstance(extra_meta, dict):
            for key, value in extra_meta.items():
                normalized_key = self._safe_str(key)
                if normalized_key:
                    meta[normalized_key] = value
        payload = {
            "request_id": request_id,
            "source": source,
            "target_agent": target_agent,
            "options": copy.deepcopy(options),
            "meta": meta,
            "items": normalized_items,
        }
        return payload, ""

    def _resolve_apply_step_no(self, step_no: Optional[int]) -> int:
        try:
            if step_no is not None:
                return int(step_no)
        except Exception:
            pass
        try:
            return int(self.config.get("step", 0) or 0) + 1
        except Exception:
            return -1

    def _is_init_step(self, step_no: int) -> bool:
        return isinstance(step_no, int) and step_no == 1

    def _is_session_completion_action(self, action: str) -> bool:
        return self._safe_str(action) in {"advanced", "completed_stop_injection"}

    def _normalize_sessions(self, value: Any) -> List[str]:
        if isinstance(value, (tuple, list, set)):
            out: List[str] = []
            for item in value:
                session_id = self._safe_str(item)
                if session_id and session_id not in out:
                    out.append(session_id)
            return out
        if isinstance(value, str):
            session_id = self._safe_str(value)
            return [session_id] if session_id else []
        return []

    def _item_matches_session(self, item: Dict[str, Any], session_id: str) -> bool:
        if not isinstance(item, dict):
            return False
        sessions = self._normalize_sessions(item.get("sessions", []))
        if not sessions:
            return True
        return self._safe_str(session_id) in sessions

    def _resolve_max_items_per_session(self, rule_cfg: Dict[str, Any]) -> int:
        if not isinstance(rule_cfg, dict):
            return self._cfg["limits"]["max_items_per_request"]
        options = rule_cfg.get("options", {})
        if not isinstance(options, dict):
            return self._cfg["limits"]["max_items_per_request"]
        limit = self._safe_int(
            options.get("max_items_per_session"),
            self._cfg["limits"]["max_items_per_request"],
            minimum=1,
        )
        if limit > self._cfg["limits"]["max_items_per_request"]:
            return self._cfg["limits"]["max_items_per_request"]
        return limit

    def _build_session_event_key(
        self,
        rule_id: str,
        target_agent: str,
        session_before: str,
        meeting_id: str,
    ) -> str:
        return "{}::{}::{}::{}".format(
            self._safe_str(rule_id),
            self._safe_str(target_agent),
            self._safe_str(session_before),
            self._safe_str(meeting_id),
        )

    def _normalize_item(
        self, item: Any, index: int, target_agent: str, now: datetime.datetime
    ) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, str]]]:
        errors: List[Dict[str, str]] = []
        if not isinstance(item, dict):
            errors.append({"item_id": "", "code": "item_type_invalid", "message": "item must be dict"})
            return None, errors
        item_id = self._safe_str(item.get("item_id")) or "item_{:03d}".format(index + 1)
        node_type = self._safe_str(item.get("node_type")).lower()
        if node_type not in self._VALID_NODE_TYPES:
            errors.append({"item_id": item_id, "code": "invalid_node_type", "message": "node_type must be event/thought/chat"})
            return None, errors
        describe = self._safe_str(item.get("describe"))
        if not describe:
            errors.append({"item_id": item_id, "code": "describe_required", "message": "describe is required"})
            return None, errors
        if len(describe) > self._cfg["limits"]["max_describe_chars"]:
            errors.append(
                {
                    "item_id": item_id,
                    "code": "describe_too_long",
                    "message": "describe exceeds max_describe_chars={}".format(self._cfg["limits"]["max_describe_chars"]),
                }
            )
            return None, errors
        subject = self._safe_str(item.get("subject")) or target_agent
        predicate = self._safe_str(item.get("predicate")) or ("对话" if node_type == "chat" else "此时")
        obj = self._safe_str(item.get("object")) or ("unknown_peer" if node_type == "chat" else "空闲")
        create_dt, create_err = self._parse_time(item.get("create"), default=now)
        if create_err:
            errors.append({"item_id": item_id, "code": "create_time_invalid", "message": create_err})
            return None, errors

        expire_value = item.get("expire")
        if expire_value is not None and str(expire_value).strip() != "":
            expire_dt, expire_err = self._parse_time(expire_value, default=None)
            if expire_err or expire_dt is None:
                errors.append({"item_id": item_id, "code": "expire_time_invalid", "message": expire_err or "expire parse failed"})
                return None, errors
        else:
            expire_days = item.get("expire_days")
            if expire_days is None:
                expire_days = self._cfg["defaults"]["expire_days"]
            expire_days = self._safe_int(expire_days, self._cfg["defaults"]["expire_days"], minimum=1)
            expire_dt = create_dt + datetime.timedelta(days=expire_days)

        if expire_dt < create_dt:
            errors.append({"item_id": item_id, "code": "expire_before_create", "message": "expire must be >= create"})
            return None, errors
        poignancy, fallback_used = self._normalize_poignancy(item.get("poignancy"), self._cfg["defaults"]["poignancy"])
        if fallback_used:
            self._log_warn("VALIDATE_WARN item_id={} field=poignancy fallback={}".format(item_id, self._cfg["defaults"]["poignancy"]))
        tags = item.get("tags") if isinstance(item.get("tags"), list) else []
        meta = item.get("meta") if isinstance(item.get("meta"), dict) else {}
        normalized = {
            "item_id": item_id,
            "node_type": node_type,
            "subject": subject,
            "predicate": predicate,
            "object": obj,
            "describe": describe,
            "address": self._normalize_address(item.get("address")),
            "poignancy": poignancy,
            "create": create_dt.strftime(self._TIME_FMT),
            "expire": expire_dt.strftime(self._TIME_FMT),
            "tags": tags,
            "meta": meta,
            "_create_dt": create_dt,
            "_expire_dt": expire_dt,
        }
        return normalized, errors

    def _normalize_address(self, address: Any) -> List[str]:
        if isinstance(address, list):
            return [self._safe_str(a) for a in address if self._safe_str(a)]
        if isinstance(address, str):
            text = address.strip()
            if not text:
                return self._default_address()
            if ":" in text:
                return [seg.strip() for seg in text.split(":") if seg.strip()]
            return [text]
        return self._default_address()

    def _default_address(self) -> List[str]:
        if self._cfg["defaults"]["address_mode"] == "empty":
            return []
        return [self._CURRENT_TILE_SENTINEL]

    def _resolve_runtime_address(self, address: List[str], agent: Any) -> List[str]:
        if address == [self._CURRENT_TILE_SENTINEL]:
            try:
                tile = agent.get_tile()
                resolved = tile.get_address() if tile and hasattr(tile, "get_address") else []
                if isinstance(resolved, list):
                    return [self._safe_str(a) for a in resolved if self._safe_str(a)]
                if isinstance(resolved, str):
                    return [seg.strip() for seg in resolved.split(":") if seg.strip()]
            except Exception:
                pass
            return []
        return [self._safe_str(a) for a in (address or []) if self._safe_str(a)]

    def _merge_options(self, options: Any) -> Dict[str, Any]:
        options = options if isinstance(options, dict) else {}
        dedup_mode = self._safe_str(options.get("dedup_mode")).lower() or self._cfg["defaults"]["dedup_mode"]
        if dedup_mode not in self._VALID_DEDUP_MODES:
            dedup_mode = self._cfg["defaults"]["dedup_mode"]
        return {
            "dry_run": self._safe_bool(options.get("dry_run"), False),
            "allow_partial_success": self._safe_bool(
                options.get("allow_partial_success"), self._cfg["defaults"]["allow_partial_success"]
            ),
            "dedup_mode": dedup_mode,
        }

    def _check_dedup(
        self,
        target_agent: str,
        normalized_item: Dict[str, Any],
        dedup_mode: str,
        request_seen: set,
        step_no: Optional[int],
    ) -> Tuple[str, str]:
        fp = self._build_fingerprint(target_agent, normalized_item, step_no)
        if dedup_mode == "none":
            return "", fp
        if fp in request_seen:
            return "duplicate_in_request", fp
        request_seen.add(fp)
        if dedup_mode == "global":
            seen = self._memory_state.get("seen_fingerprints", {})
            if isinstance(seen, dict) and fp in seen:
                return "duplicate_in_global", fp
        return "", fp

    def _build_fingerprint(self, target_agent: str, item: Dict[str, Any], step_no: Optional[int]) -> str:
        describe_norm = " ".join(str(item.get("describe", "")).split())
        address_norm = ":".join(item.get("address", []))
        if step_no is not None and step_no >= 0:
            time_bucket = "step:{}".format(step_no)
        else:
            create_dt = item.get("_create_dt")
            if isinstance(create_dt, datetime.datetime):
                time_bucket = create_dt.strftime("%Y%m%d-%H:%M")
            else:
                time_bucket = self._resolve_now(None).strftime("%Y%m%d-%H:%M")
        key = {
            "target_agent": target_agent,
            "node_type": item.get("node_type", ""),
            "subject": item.get("subject", ""),
            "predicate": item.get("predicate", ""),
            "object": item.get("object", ""),
            "describe_norm": describe_norm,
            "address_norm": address_norm,
            "time_bucket": time_bucket,
        }
        return hashlib.sha1(json.dumps(key, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()

    def _register_fingerprint(self, fingerprint: str, request_id: str, item_id: str, step_no: Optional[int]) -> None:
        seen = self._memory_state.setdefault("seen_fingerprints", {})
        if not isinstance(seen, dict):
            seen = {}
            self._memory_state["seen_fingerprints"] = seen
        if fingerprint not in seen:
            seen[fingerprint] = {
                "first_seen_at": self._resolve_now(None).strftime(self._TIME_FMT),
                "step": int(step_no) if isinstance(step_no, int) else -1,
                "request_id": request_id,
                "item_id": item_id,
            }
        limit = self._cfg["limits"]["max_fingerprints"]
        while len(seen) > limit:
            try:
                oldest = next(iter(seen))
            except StopIteration:
                break
            seen.pop(oldest, None)

    def _append_request_audit(self, result: Dict[str, Any]) -> None:
        audit = self._memory_state.setdefault("request_audit", [])
        if not isinstance(audit, list):
            audit = []
            self._memory_state["request_audit"] = audit
        audit.append(
            {
                "timestamp": self._resolve_now(None).strftime(self._TIME_FMT),
                "step": self._resolve_step_no({}),
                "request_id": result.get("request_id", ""),
                "target_agent": result.get("target_agent", ""),
                "status": result.get("status", ""),
                "reason": result.get("reason", ""),
                "accepted": int(result.get("accepted", 0) or 0),
                "injected": int(result.get("injected", 0) or 0),
                "skipped": int(result.get("skipped", 0) or 0),
                "errors": len(result.get("errors", [])),
            }
        )
        max_entries = self._cfg["limits"]["max_audit_entries"]
        if len(audit) > max_entries:
            self._memory_state["request_audit"] = audit[-max_entries:]

    def _rollback_written(self, agent: Any, written: List[Dict[str, str]]) -> int:
        node_ids = [w.get("node_id", "") for w in written if w.get("node_id")]
        if not node_ids:
            return 0
        try:
            for node_type, node_list in (agent.associate.memory or {}).items():
                if isinstance(node_list, list):
                    agent.associate.memory[node_type] = [nid for nid in node_list if nid not in node_ids]
            agent.associate.index.remove_nodes(node_ids)
            return len(node_ids)
        except Exception:
            return 0

    def _parse_time(self, value: Any, default: Optional[datetime.datetime]) -> Tuple[Optional[datetime.datetime], str]:
        if value is None or (isinstance(value, str) and not value.strip()):
            return default, ""
        if isinstance(value, datetime.datetime):
            return value, ""
        if isinstance(value, str):
            try:
                return utils.to_date(value.strip(), self._TIME_FMT), ""
            except Exception:
                return None, "time must match {}".format(self._TIME_FMT)
        return None, "time must be string or datetime"

    def _resolve_now(self, now: Any) -> datetime.datetime:
        if isinstance(now, datetime.datetime):
            return now
        return utils.get_timer().get_date()

    def _resolve_step_no(self, meta: Dict[str, Any]) -> int:
        if isinstance(meta, dict) and "step" in meta:
            try:
                return int(meta.get("step"))
            except Exception:
                pass
        try:
            return int(self.config.get("step", -1))
        except Exception:
            return -1

    def _build_request_id(self) -> str:
        stamp = self._resolve_now(None).strftime("%Y%m%d_%H%M%S")
        return "inj_{}_{}".format(stamp, uuid.uuid4().hex[:8])

    def _normalize_poignancy(self, value: Any, fallback: int) -> Tuple[int, bool]:
        try:
            val = int(value)
        except Exception:
            return int(fallback), True
        if val < 1 or val > 10:
            return int(fallback), True
        return val, False

    def _safe_int(self, value: Any, default: int, minimum: Optional[int] = None) -> int:
        try:
            out = int(value)
        except Exception:
            out = int(default)
        if minimum is not None and out < minimum:
            return int(default)
        return out

    def _safe_bool(self, value: Any, default: bool) -> bool:
        if isinstance(value, bool):
            return value
        if value is None:
            return default
        if isinstance(value, str):
            text = value.strip().lower()
            if text in {"true", "1", "yes", "on"}:
                return True
            if text in {"false", "0", "no", "off"}:
                return False
        return default

    def _safe_str(self, value: Any) -> str:
        if value is None:
            return ""
        return str(value).strip()

    def _log_info(self, message: str) -> None:
        if self.logger:
            self.logger.info("[MEMORY_INJECTION] {}".format(message))

    def _log_warn(self, message: str) -> None:
        if self.logger:
            self.logger.warning("[MEMORY_INJECTION] {}".format(message))
