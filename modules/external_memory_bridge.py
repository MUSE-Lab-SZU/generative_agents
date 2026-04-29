"""Bridge module for EC-Doll external memory integration."""

from __future__ import annotations

import datetime
import os
import re
import time
from string import Template
from typing import Any, Dict, List, Optional, Tuple

from modules import utils
from modules.ec_doll_memory_service_client import ECDollMemoryServiceClient


class ExternalMemoryBridge:
    """Encapsulate external memory read/write behaviors for a single agent."""

    MAP_FILENAME = "external_memory_node_map.json"
    RECENT_RAW_SECTION_TITLE = "【近期原文（本服务入库）】"

    def __init__(
        self,
        agent_name: str,
        cfg: Dict[str, Any],
        storage_dir: str,
        logger: Any = None,
    ):
        self.agent_name = str(agent_name or "").strip()
        self.cfg = cfg if isinstance(cfg, dict) else {}
        self.storage_dir = os.path.normpath(str(storage_dir or ""))
        self.logger = logger
        self._project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        self.enabled = self._safe_bool(self.cfg.get("enabled", False), False)
        self.base_url = str(self.cfg.get("base_url", "http://localhost:8031") or "http://localhost:8031").strip()
        self.timeout = self._safe_float(self.cfg.get("timeout", 10.0), 10.0)
        self.read_mode = str(self.cfg.get("read_mode", "chat_only") or "chat_only").strip().lower()
        self.fallback_to_local = self._safe_bool(self.cfg.get("fallback_to_local", True), True)
        self.short_term_recent_n = self._safe_int(
            self.cfg.get("short_term_recent_n", 40),
            40,
        )
        if self.short_term_recent_n < 0:
            self._log(
                "warning",
                "[EXT_MEMORY_CFG] agent={} invalid_short_term_recent_n={} fallback=40".format(
                    self.agent_name,
                    self.short_term_recent_n,
                ),
            )
            self.short_term_recent_n = 40
        self.user_id = str(self.cfg.get("user_id", "") or "").strip()
        self.save_name = self.resolve_save_name_from_storage_dir()
        self.scoped_user_id = self.build_scoped_user_id(self.user_id, self.save_name)
        self.ingest_template_file = self._resolve_path(self.cfg.get("ingest_template_file", ""))
        self.map_path = os.path.join(self.storage_dir, self.MAP_FILENAME)
        self.updata_to_l2_apply_scenes = self._normalize_apply_scenes(
            self.cfg.get("updata_to_L2_apply_scenes", [])
        )
        self.updata_to_milestone_apply_scenes = self._normalize_apply_scenes(
            self.cfg.get("updata_to_milestone_apply_scenes", [])
        )

        self._remote_map_cache = None
        self.client: Optional[ECDollMemoryServiceClient] = None

        self._log(
            "info",
            "[EXT_MEMORY_CFG] agent={} enabled={} user_id={} scoped_user_id={} save_name={} read_mode={} fallback_to_local={} short_term_recent_n={} base_url={} timeout={} ingest_template_file={} updata_to_L2_apply_scenes={} updata_to_milestone_apply_scenes={}".format(
                self.agent_name,
                self.enabled,
                self.user_id,
                self.scoped_user_id,
                self.save_name,
                self.read_mode,
                self.fallback_to_local,
                self.short_term_recent_n,
                self.base_url,
                self.timeout,
                self.ingest_template_file,
                self.updata_to_l2_apply_scenes,
                self.updata_to_milestone_apply_scenes,
            ),
        )

        if self.enabled:
            try:
                self.client = ECDollMemoryServiceClient(
                    base_url=self.base_url,
                    timeout=self.timeout,
                )
                self._log(
                    "info",
                    "[EXT_MEMORY_CLIENT_READY] agent={} base_url={}".format(
                        self.agent_name,
                        self.base_url,
                    ),
                )
            except Exception as exc:
                self.client = None
                self._log(
                    "warning",
                    "[EXT_MEMORY_CLIENT_FAIL] agent={} error={}".format(
                        self.agent_name,
                        exc,
                    ),
                )

    @classmethod
    def from_agent_config(
        cls,
        agent_name: str,
        global_cfg: Any,
        local_cfg: Any,
        storage_dir: str,
        logger: Any = None,
    ) -> "ExternalMemoryBridge":
        merged: Dict[str, Any] = {}
        if isinstance(global_cfg, dict):
            merged.update(global_cfg)
        if isinstance(local_cfg, dict):
            merged.update(local_cfg)
        return cls(
            agent_name=agent_name,
            cfg=merged,
            storage_dir=storage_dir,
            logger=logger,
        )

    def enabled_for_ingest(self) -> bool:
        return bool(self.enabled and self.client and self.get_scoped_user_id())

    def enabled_for_chat_read(self) -> bool:
        if not (self.enabled and self.client and self.get_scoped_user_id()):
            return False
        return self.read_mode == "chat_only"

    def resolve_save_name_from_storage_dir(self) -> str:
        normalized = os.path.normpath(str(self.storage_dir or ""))
        if not normalized:
            return ""
        parts = [part for part in normalized.split(os.sep) if part]
        for idx, part in enumerate(parts):
            if str(part).strip().lower() == "checkpoints" and idx + 1 < len(parts):
                return str(parts[idx + 1]).strip()
        return ""

    @staticmethod
    def build_scoped_user_id(raw_user_id: str, save_name: str, sep: str = "+") -> str:
        base_id = str(raw_user_id or "").strip()
        archive_name = str(save_name or "").strip()
        if not base_id:
            return ""
        if not archive_name:
            return base_id
        return "{}{}{}".format(base_id, sep, archive_name)

    def get_scoped_user_id(self) -> str:
        scoped_user_id = str(getattr(self, "scoped_user_id", "") or "").strip()
        if scoped_user_id:
            return scoped_user_id
        scoped_user_id = self.build_scoped_user_id(
            str(getattr(self, "user_id", "") or ""),
            self.resolve_save_name_from_storage_dir(),
        )
        self.scoped_user_id = scoped_user_id
        return scoped_user_id

    def _normalize_apply_scenes(self, value: Any) -> List[str]:
        if isinstance(value, str):
            raw_items = [value]
        elif isinstance(value, (list, tuple, set)):
            raw_items = list(value)
        elif value is None:
            raw_items = []
        else:
            raw_items = []
            self._log(
                "warning",
                "[EXT_MEMORY_CFG_WARN] agent={} field=apply_scenes invalid_type={}".format(
                    self.agent_name,
                    type(value).__name__,
                ),
            )
        out: List[str] = []
        seen = set()
        for item in raw_items:
            token = str(item or "").strip().lower()
            if (not token) or (token in seen):
                continue
            seen.add(token)
            out.append(token)
        return out

    def _resolve_session_adjust_modes(self, session_before: str, force_all_session_l2: bool = False) -> List[str]:
        session_token = str(session_before or "").strip().lower()
        if not session_token:
            return []
        scene_tag = (
            session_token
            if session_token.endswith("_completed")
            else "{}_completed".format(session_token)
        )
        in_l2 = scene_tag in self.updata_to_l2_apply_scenes
        in_milestone = scene_tag in self.updata_to_milestone_apply_scenes
        all_session_l2 = "all_session_chat" in self.updata_to_l2_apply_scenes

        modes: List[str] = []
        if force_all_session_l2 or all_session_l2 or in_l2 or in_milestone:
            modes.append("update_to_L2")
        if in_milestone:
            modes.append("update_to_milestone")
        return modes

    @staticmethod
    def _format_policy_modes(modes: List[str]) -> str:
        if not isinstance(modes, list) or not modes:
            return "none"
        out: List[str] = []
        seen = set()
        for mode in modes:
            token = str(mode or "").strip()
            if (not token) or (token in seen):
                continue
            seen.add(token)
            out.append(token)
        if not out:
            return "none"
        return "+".join(out)

    def _resolve_session_adjust_policy(self, session_before: str) -> str:
        modes = self._resolve_session_adjust_modes(session_before=session_before, force_all_session_l2=False)
        if not modes:
            return "none"
        if len(modes) > 1:
            self._log(
                "warning",
                "[EXT_MEMORY_LEVEL_POLICY_CONFLICT] agent={} session={} resolved_modes={} preferred={}".format(
                    self.agent_name,
                    str(session_before or "").strip(),
                    ",".join(modes),
                    modes[-1],
                ),
            )
        return modes[-1]

    def get_session_adjust_policy(self, session_before: str) -> str:
        return self._resolve_session_adjust_policy(session_before)

    def resolve_remote_id_by_node_id(self, node_id: str) -> str:
        node_id = str(node_id or "").strip()
        if not node_id:
            return ""
        data = self._load_remote_map()
        node_to_remote = data.get("node_to_remote", {}) if isinstance(data, dict) else {}
        if not isinstance(node_to_remote, dict):
            return ""
        mapped = node_to_remote.get(node_id, {})
        if isinstance(mapped, dict):
            return str(mapped.get("remote_id", "") or "").strip()
        if isinstance(mapped, str):
            return str(mapped or "").strip()
        return ""

    def _build_actions_by_policy(self, policy_mode: str) -> List[Dict[str, Any]]:
        mode = str(policy_mode or "").strip()
        if mode == "update_to_L2":
            return [{"op": "promote"}, {"op": "milestone", "is_milestone": False}]
        if mode == "update_to_milestone":
            return [{"op": "milestone", "is_milestone": True}]
        return []

    @staticmethod
    def _is_action_response_ok(response: Any) -> bool:
        if not isinstance(response, dict):
            return True
        status = str(response.get("status", "") or "").strip().lower()
        if not status:
            return True
        return status not in {"error", "failed"}

    @staticmethod
    def _normalize_node_id_list(node_ids: Any) -> List[str]:
        if not isinstance(node_ids, (list, tuple, set)):
            return []
        out: List[str] = []
        seen = set()
        for node_id in node_ids:
            value = str(node_id or "").strip()
            if (not value) or (value in seen):
                continue
            seen.add(value)
            out.append(value)
        return out

    @staticmethod
    def _format_trace(trace: Optional[Dict[str, Any]]) -> str:
        if not isinstance(trace, dict) or not trace:
            return ""
        kvs = []
        for key in sorted(trace.keys()):
            value = trace.get(key)
            if value is None:
                continue
            text = str(value).replace("\n", " ").strip()
            if not text:
                continue
            kvs.append("{}={}".format(key, text))
        if not kvs:
            return ""
        return " " + " ".join(kvs)

    @staticmethod
    def _extract_http_status_from_exc(exc: Exception) -> Optional[int]:
        response = getattr(exc, "response", None)
        status = getattr(response, "status_code", None)
        try:
            return int(status) if status is not None else None
        except Exception:
            return None

    def _promote_with_retry(self, remote_id: str) -> Dict[str, Any]:
        delays_s = [0.1, 0.3, 0.8]
        last_exc: Optional[Exception] = None
        for idx, delay_s in enumerate(delays_s, start=1):
            try:
                return self.client.promote_memory(remote_id)  # type: ignore[union-attr]
            except Exception as exc:
                status = self._extract_http_status_from_exc(exc)
                if status != 404:
                    raise
                last_exc = exc
                self._log(
                    "warning",
                    "[PROMOTE_RETRY] agent={} remote_id={} attempt={} status=404".format(
                        self.agent_name,
                        remote_id,
                        idx,
                    ),
                )
                if idx < len(delays_s):
                    time.sleep(delay_s)
        if last_exc is not None:
            raise last_exc
        raise RuntimeError("promote_retry_unexpected_empty: remote_id={}".format(remote_id))

    def _call_level_action(self, remote_id: str, action: Dict[str, Any]) -> Dict[str, Any]:
        op = str((action or {}).get("op", "") or "").strip()
        if op == "promote":
            return self._promote_with_retry(remote_id)
        if op == "milestone":
            is_milestone = bool((action or {}).get("is_milestone", False))
            return self.client.set_memory_milestone(remote_id, is_milestone)  # type: ignore[union-attr]
        raise ValueError("unsupported_level_action: {}".format(op or "<empty>"))

    def apply_session_chat_memory_level_adjustment(
        self,
        session_before: str,
        session_chat_node_id: str,
        trace: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        session_before = str(session_before or "").strip()
        node_id = str(session_chat_node_id or "").strip()
        trace_dict = trace if isinstance(trace, dict) else {}
        force_all_session_l2 = (
            self._safe_bool(trace_dict.get("all_session_chat_enabled"), False)
            and ("all_session_chat" in self.updata_to_l2_apply_scenes)
        )
        modes = self._resolve_session_adjust_modes(
            session_before=session_before,
            force_all_session_l2=force_all_session_l2,
        )
        policy = self._format_policy_modes(modes)
        trace = dict(trace_dict)
        trace["all_session_chat_enabled"] = force_all_session_l2
        trace["resolved_modes"] = ",".join(modes) if modes else "none"
        summary: Dict[str, Any] = {
            "status": "skipped",
            "reason": "",
            "session_before": session_before,
            "policy": policy,
            "resolved_modes": list(modes),
            "node_id": node_id,
            "success": 0,
            "failed": 0,
            "details": [],
        }
        trace_text = self._format_trace(trace)
        if not modes:
            summary["reason"] = "policy_not_matched"
            self._log(
                "info",
                "[EXT_MEMORY_LEVEL_ADJUST_SKIP] agent={} session={} policy={} reason=policy_not_matched{}".format(
                    self.agent_name,
                    session_before,
                    policy,
                    trace_text,
                ),
            )
            return summary
        if not (self.enabled and self.client):
            summary["reason"] = "disabled_or_not_ready"
            self._log(
                "warning",
                "[EXT_MEMORY_LEVEL_ADJUST_SKIP] agent={} session={} policy={} reason=disabled_or_not_ready{}".format(
                    self.agent_name,
                    session_before,
                    policy,
                    trace_text,
                ),
            )
            return summary
        if not node_id:
            summary["reason"] = "session_chat_node_id_missing"
            self._log(
                "warning",
                "[EXT_MEMORY_LEVEL_ADJUST_SKIP] agent={} session={} policy={} reason=session_chat_node_id_missing{}".format(
                    self.agent_name,
                    session_before,
                    policy,
                    trace_text,
                ),
            )
            return summary

        remote_id = self.resolve_remote_id_by_node_id(node_id)
        if not remote_id:
            summary["reason"] = "remote_id_not_found"
            self._log(
                "warning",
                "[EXT_MEMORY_LEVEL_ADJUST_SKIP] agent={} session={} policy={} reason=remote_id_not_found node_id={}{}".format(
                    self.agent_name,
                    session_before,
                    policy,
                    node_id,
                    trace_text,
                ),
            )
            return summary

        actions: List[Dict[str, Any]] = []
        for mode in modes:
            actions.extend(self._build_actions_by_policy(mode))
        self._log(
            "info",
            "[EXT_MEMORY_LEVEL_ADJUST_APPLY] agent={} session={} policy={} session_chat_node_id={} remote_id={} action_count={}{}".format(
                self.agent_name,
                session_before,
                policy,
                node_id,
                remote_id,
                len(actions),
                trace_text,
            ),
        )

        node_detail: Dict[str, Any] = {
            "node_id": node_id,
            "remote_id": remote_id,
            "status": "ok",
            "actions": [],
        }
        for action in actions:
            op = str(action.get("op", "") or "")
            self._log(
                "info",
                "[EXT_MEMORY_LEVEL_ACTION_CALL] agent={} node_id={} remote_id={} op={}{}".format(
                    self.agent_name,
                    node_id,
                    remote_id,
                    op,
                    trace_text,
                ),
            )
            try:
                response = self._call_level_action(remote_id=remote_id, action=action)
                ok = self._is_action_response_ok(response)
                action_status = "ok" if ok else "failed"
                node_detail["actions"].append(
                    {
                        "op": op,
                        "status": action_status,
                        "response_status": str((response or {}).get("status", "") or ""),
                        "message": str((response or {}).get("message", "") or ""),
                    }
                )
                self._log(
                    "info",
                    "[EXT_MEMORY_LEVEL_ACTION_DONE] agent={} node_id={} remote_id={} op={} status={}{}".format(
                        self.agent_name,
                        node_id,
                        remote_id,
                        op,
                        action_status,
                        trace_text,
                    ),
                )
                if not ok:
                    node_detail["status"] = "failed"
                    break
            except Exception as exc:
                node_detail["status"] = "failed"
                node_detail["actions"].append(
                    {
                        "op": op,
                        "status": "failed",
                        "error": str(exc),
                    }
                )
                self._log(
                    "warning",
                    "[EXT_MEMORY_LEVEL_ACTION_DONE] agent={} node_id={} remote_id={} op={} status=failed error={}{}".format(
                        self.agent_name,
                        node_id,
                        remote_id,
                        op,
                        exc,
                        trace_text,
                    ),
                )
                break

        summary["details"].append(node_detail)
        if node_detail["status"] == "ok":
            summary["status"] = "ok"
            summary["reason"] = "success"
            summary["success"] = 1
            summary["failed"] = 0
        else:
            summary["status"] = "failed"
            summary["reason"] = "action_failed"
            summary["success"] = 0
            summary["failed"] = 1
        self._log(
            "info",
            "[EXT_MEMORY_LEVEL_ADJUST_SUMMARY] agent={} session={} policy={} success={} failed={} status={}{}".format(
                self.agent_name,
                session_before,
                policy,
                summary["success"],
                summary["failed"],
                summary["status"],
                trace_text,
            ),
        )
        return summary

    def apply_injected_memory_level_adjustments(
        self,
        node_ids: List[str],
        policy_mode: str,
        trace: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        mode = str(policy_mode or "").strip()
        normalized_node_ids = self._normalize_node_id_list(node_ids)
        summary: Dict[str, Any] = {
            "status": "skipped",
            "reason": "",
            "policy": mode,
            "success": 0,
            "failed": 0,
            "details": [],
        }
        trace_text = self._format_trace(trace)
        if mode not in {"update_to_L2", "update_to_milestone"}:
            summary["reason"] = "policy_mode_invalid"
            self._log(
                "warning",
                "[EXT_MEMORY_LEVEL_ADJUST_SKIP] agent={} policy={} reason=policy_mode_invalid{}".format(
                    self.agent_name,
                    mode,
                    trace_text,
                ),
            )
            return summary
        if not (self.enabled and self.client):
            summary["reason"] = "disabled_or_not_ready"
            self._log(
                "warning",
                "[EXT_MEMORY_LEVEL_ADJUST_SKIP] agent={} policy={} reason=disabled_or_not_ready{}".format(
                    self.agent_name,
                    mode,
                    trace_text,
                ),
            )
            return summary
        if not normalized_node_ids:
            summary["reason"] = "node_ids_empty"
            self._log(
                "info",
                "[EXT_MEMORY_LEVEL_ADJUST_SKIP] agent={} policy={} reason=node_ids_empty{}".format(
                    self.agent_name,
                    mode,
                    trace_text,
                ),
            )
            return summary

        actions = self._build_actions_by_policy(mode)
        self._log(
            "info",
            "[EXT_MEMORY_LEVEL_ADJUST_APPLY] agent={} policy={} node_count={} action_count={}{}".format(
                self.agent_name,
                mode,
                len(normalized_node_ids),
                len(actions),
                trace_text,
            ),
        )

        for node_id in normalized_node_ids:
            remote_id = self.resolve_remote_id_by_node_id(node_id)
            node_detail: Dict[str, Any] = {
                "node_id": node_id,
                "remote_id": remote_id,
                "status": "ok",
                "actions": [],
            }
            if not remote_id:
                node_detail["status"] = "skipped"
                node_detail["reason"] = "remote_id_not_found"
                summary["details"].append(node_detail)
                summary["failed"] += 1
                self._log(
                    "warning",
                    "[EXT_MEMORY_LEVEL_ADJUST_SKIP] agent={} policy={} reason=remote_id_not_found node_id={}{}".format(
                        self.agent_name,
                        mode,
                        node_id,
                        trace_text,
                    ),
                )
                continue

            for action in actions:
                op = str(action.get("op", "") or "")
                self._log(
                    "info",
                    "[EXT_MEMORY_LEVEL_ACTION_CALL] agent={} node_id={} remote_id={} op={}{}".format(
                        self.agent_name,
                        node_id,
                        remote_id,
                        op,
                        trace_text,
                    ),
                )
                try:
                    response = self._call_level_action(remote_id=remote_id, action=action)
                    ok = self._is_action_response_ok(response)
                    action_status = "ok" if ok else "failed"
                    node_detail["actions"].append(
                        {
                            "op": op,
                            "status": action_status,
                            "response_status": str((response or {}).get("status", "") or ""),
                            "message": str((response or {}).get("message", "") or ""),
                        }
                    )
                    self._log(
                        "info",
                        "[EXT_MEMORY_LEVEL_ACTION_DONE] agent={} node_id={} remote_id={} op={} status={}{}".format(
                            self.agent_name,
                            node_id,
                            remote_id,
                            op,
                            action_status,
                            trace_text,
                        ),
                    )
                    if not ok:
                        node_detail["status"] = "failed"
                        break
                except Exception as exc:
                    node_detail["status"] = "failed"
                    node_detail["actions"].append(
                        {
                            "op": op,
                            "status": "failed",
                            "error": str(exc),
                        }
                    )
                    self._log(
                        "warning",
                        "[EXT_MEMORY_LEVEL_ACTION_DONE] agent={} node_id={} remote_id={} op={} status=failed error={}{}".format(
                            self.agent_name,
                            node_id,
                            remote_id,
                            op,
                            exc,
                            trace_text,
                        ),
                    )
                    break

            summary["details"].append(node_detail)
            if node_detail["status"] == "ok":
                summary["success"] += 1
            else:
                summary["failed"] += 1

        if summary["success"] > 0 and summary["failed"] > 0:
            summary["status"] = "partial"
            summary["reason"] = "partial_success"
        elif summary["success"] > 0:
            summary["status"] = "ok"
            summary["reason"] = "success"
        elif summary["failed"] > 0:
            summary["status"] = "failed"
            summary["reason"] = "all_failed"
        else:
            summary["status"] = "skipped"
            summary["reason"] = "no_effective_node"
        self._log(
            "info",
            "[EXT_MEMORY_LEVEL_ADJUST_SUMMARY] agent={} policy={} success={} failed={} status={}{}".format(
                self.agent_name,
                mode,
                summary["success"],
                summary["failed"],
                summary["status"],
                trace_text,
            ),
        )
        return summary

    def render_ingest_text(
        self,
        node_type: str,
        event: Any,
        create_time: Any,
    ) -> str:
        create_time_str = self._format_time(create_time)
        describe = ""
        address = ""
        if event is not None:
            try:
                describe = str(event.get_describe() or "")
            except Exception:
                describe = str(getattr(event, "describe", "") or "")
            address_value = getattr(event, "address", [])
            if isinstance(address_value, (list, tuple)):
                address = ":".join([str(x) for x in address_value])
            else:
                address = str(address_value or "")

        data = {
            "create_time": create_time_str,
            "memory_type": str(node_type or ""),
            "address": address,
            "describe": describe,
        }
        template_text = self._load_template(self.ingest_template_file)
        if template_text:
            try:
                rendered = Template(template_text).safe_substitute(data)
                if str(rendered or "").strip():
                    return str(rendered).strip()
            except Exception as exc:
                self._log(
                    "warning",
                    "[EXT_MEMORY_INGEST_TEMPLATE_FAIL] agent={} error={}".format(
                        self.agent_name,
                        exc,
                    ),
                )
        fallback_lines = [
            "create={}".format(create_time_str),
            "type={}".format(node_type),
            "address={}".format(address),
            describe,
        ]
        return "\n".join([line for line in fallback_lines if str(line or "").strip()]).strip()

    def ingest_from_local_node(
        self,
        node_id: str,
        node_type: str,
        event: Any,
        create_time: Any,
    ) -> Dict[str, Any]:
        result = {
            "ok": False,
            "node_id": str(node_id or ""),
            "remote_id": "",
            "reason": "",
            "status": "",
            "message": "",
            "long_term_pending": None,
            "analysis_ok": None,
            "mws_score": None,
            "stored_raw_fallback": None,
            "scoped_user_id": self.get_scoped_user_id(),
        }
        if not self.enabled_for_ingest():
            result["reason"] = "disabled_or_not_ready"
            return result

        content = self.render_ingest_text(node_type=node_type, event=event, create_time=create_time)
        if not content:
            result["reason"] = "empty_content"
            self._log(
                "warning",
                "[EXT_MEMORY_INGEST_FAIL] agent={} node_id={} reason=empty_content".format(
                    self.agent_name,
                    node_id,
                ),
            )
            return result

        scoped_user_id = self.get_scoped_user_id()
        start_ts = time.perf_counter()
        try:
            response = self.client.ingest_memory(  # type: ignore[union-attr]
                content=content,
                user_id=scoped_user_id,
                remote_id=None,
            )
            elapsed_ms = int((time.perf_counter() - start_ts) * 1000)
            response = response if isinstance(response, dict) else {}
            remote_id = str(response.get("remote_id", "") or "").strip()
            status = str(response.get("status", "") or "").strip()
            message = str(response.get("message", "") or "")
            result["remote_id"] = remote_id
            result["status"] = status
            result["message"] = message
            result["long_term_pending"] = response.get("long_term_pending")
            result["analysis_ok"] = response.get("analysis_ok")
            result["mws_score"] = response.get("mws_score")
            result["stored_raw_fallback"] = response.get("stored_raw_fallback")

            if status not in {"completed", "stored_raw"}:
                if status == "error":
                    result["reason"] = "ingest_status_error"
                elif status:
                    result["reason"] = "unexpected_status"
                else:
                    result["reason"] = "missing_status"
                self._log(
                    "warning",
                    "[EXT_MEMORY_INGEST_FAIL] agent={} user_id={} node_id={} node_type={} status={} reason={} message={} long_term_pending={} stored_raw_fallback={} elapsed_ms={}".format(
                        self.agent_name,
                        scoped_user_id,
                        node_id,
                        node_type,
                        status,
                        result["reason"],
                        self._trim_for_log(message, max_len=200),
                        result["long_term_pending"],
                        result["stored_raw_fallback"],
                        elapsed_ms,
                    ),
                )
                return result

            if not remote_id:
                result["reason"] = "missing_remote_id"
                self._log(
                    "warning",
                    "[EXT_MEMORY_INGEST_FAIL] agent={} user_id={} node_id={} node_type={} status={} reason=missing_remote_id message={} long_term_pending={} stored_raw_fallback={} elapsed_ms={}".format(
                        self.agent_name,
                        scoped_user_id,
                        node_id,
                        node_type,
                        status,
                        self._trim_for_log(message, max_len=200),
                        result["long_term_pending"],
                        result["stored_raw_fallback"],
                        elapsed_ms,
                    ),
                )
                return result

            result["ok"] = True
            result["reason"] = "ok"
            self._log(
                "info",
                "[EXT_MEMORY_INGEST] agent={} user_id={} node_id={} node_type={} remote_id={} status={} content_len={} message={} long_term_pending={} stored_raw_fallback={} elapsed_ms={}".format(
                    self.agent_name,
                    scoped_user_id,
                    node_id,
                    node_type,
                    remote_id,
                    status,
                    len(content),
                    self._trim_for_log(message, max_len=200),
                    result["long_term_pending"],
                    result["stored_raw_fallback"],
                    elapsed_ms,
                ),
            )
            self._bind_remote_map(
                node_id=node_id,
                remote_id=remote_id,
                node_type=node_type,
                create_time=create_time,
            )
            return result
        except Exception as exc:
            elapsed_ms = int((time.perf_counter() - start_ts) * 1000)
            result["reason"] = "request_failed"
            self._log(
                "warning",
                "[EXT_MEMORY_INGEST_FAIL] agent={} node_id={} error={} elapsed_ms={}".format(
                    self.agent_name,
                    node_id,
                    exc,
                    elapsed_ms,
                ),
            )
            return result

    def build_chat_query_from_conversation(
        self,
        chats: Any,
        other_name: str,
        is_initiator: bool = False,
        turn_no: int = 1,
    ) -> str:
        other_name = str(other_name or "").strip()
        if bool(is_initiator) and int(turn_no or 1) <= 1:
            return other_name

        for item in reversed(list(chats or [])):
            if not isinstance(item, (list, tuple)) or len(item) < 2:
                continue
            speaker = str(item[0] or "").strip()
            text = str(item[1] or "").strip()
            if speaker == other_name and text:
                return text
        return ""

    def retrieve_chat_context(
        self,
        chats: Any,
        other_name: str,
        is_initiator: bool = False,
        turn_no: int = 1,
    ) -> Dict[str, Any]:
        result = {
            "ok": False,
            "query": "",
            "context": "",
            "reason": "",
            "details": {},
            "scoped_user_id": self.get_scoped_user_id(),
        }
        if not self.enabled_for_chat_read():
            result["reason"] = "disabled_or_not_ready"
            return result

        query = self.build_chat_query_from_conversation(
            chats=chats,
            other_name=other_name,
            is_initiator=is_initiator,
            turn_no=turn_no,
        )
        result["query"] = query
        if not query:
            result["reason"] = "empty_query"
            self._log(
                "warning",
                "[EXT_MEMORY_RETRIEVE_FAIL] agent={} other={} reason=empty_query is_initiator={} turn_no={}".format(
                    self.agent_name,
                    other_name,
                    bool(is_initiator),
                    turn_no,
                ),
            )
            return result

        scoped_user_id = self.get_scoped_user_id()
        start_ts = time.perf_counter()
        try:
            response = self.client.retrieve_memory_context(  # type: ignore[union-attr]
                query=query,
                user_id=scoped_user_id,
            )
            elapsed_ms = int((time.perf_counter() - start_ts) * 1000)
            response = response if isinstance(response, dict) else {}
            details = response.get("details", {})
            if not isinstance(details, dict):
                details = {}
            result["details"] = details
            hit_remote_ids = self._extract_ranked_memory_remote_ids(details=details)
            hit_remote_ids_text = ",".join(hit_remote_ids) if hit_remote_ids else "-"
            raw_context = str((response or {}).get("formatted_prompt", "") or "").strip()
            context = self._postprocess_retrieved_formatted_prompt(raw_context)
            if context:
                result["ok"] = True
                result["reason"] = "ok"
                result["context"] = context
                self._log(
                    "info",
                    "[EXT_MEMORY_RETRIEVE] agent={} user_id={} other={} query={} context_len={} hit_remote_count={} hit_remote_ids={} elapsed_ms={} details_keys={}".format(
                        self.agent_name,
                        scoped_user_id,
                        other_name,
                        self._trim_for_log(query),
                        len(context),
                        len(hit_remote_ids),
                        hit_remote_ids_text,
                        elapsed_ms,
                        ",".join(sorted(details.keys())) if details else "",
                    ),
                )
                return result

            result["reason"] = "empty_formatted_prompt"
            self._log(
                "warning",
                "[EXT_MEMORY_RETRIEVE_FAIL] agent={} other={} query={} reason=empty_formatted_prompt hit_remote_count={} hit_remote_ids={} elapsed_ms={}".format(
                    self.agent_name,
                    other_name,
                    self._trim_for_log(query),
                    len(hit_remote_ids),
                    hit_remote_ids_text,
                    elapsed_ms,
                ),
            )
            return result
        except Exception as exc:
            elapsed_ms = int((time.perf_counter() - start_ts) * 1000)
            result["reason"] = "request_failed"
            self._log(
                "warning",
                "[EXT_MEMORY_RETRIEVE_FAIL] agent={} other={} query={} error={} elapsed_ms={}".format(
                    self.agent_name,
                    other_name,
                    self._trim_for_log(query),
                    exc,
                    elapsed_ms,
                ),
            )
            return result

    def _postprocess_retrieved_formatted_prompt(self, context: str) -> str:
        text = str(context or "")
        if not text:
            return text
        try:
            before_count = self._count_recent_raw_entries(text)
            processed = self._trim_recent_raw_section(text, int(self.short_term_recent_n))
            after_count = self._count_recent_raw_entries(processed)
            if processed != text:
                self._log(
                    "info",
                    "[EXT_MEMORY_RETRIEVE_POSTPROCESS] agent={} recent_n={} recent_before={} recent_after={} context_len_before={} context_len_after={}".format(
                        self.agent_name,
                        int(self.short_term_recent_n),
                        before_count,
                        after_count,
                        len(text),
                        len(processed),
                    ),
                )
            return processed
        except Exception as exc:
            self._log(
                "warning",
                "[EXT_MEMORY_RETRIEVE_POSTPROCESS_FAIL] agent={} error={}".format(
                    self.agent_name,
                    exc,
                ),
            )
            return text

    def _trim_recent_raw_section(self, context: str, n: int) -> str:
        text = str(context or "")
        lines = text.splitlines()
        if not lines:
            return text
        start_idx, end_idx = self._find_recent_raw_section_range(lines)
        if start_idx < 0:
            return text
        section_body = lines[start_idx + 1: end_idx]
        leading_lines, entries = self._split_recent_raw_entries(section_body)
        if not entries:
            return text
        if n <= 0:
            kept_entries: List[List[str]] = []
        elif len(entries) <= n:
            kept_entries = entries
        else:
            kept_entries = entries[-n:]

        rebuilt_section: List[str] = list(leading_lines)
        for entry in kept_entries:
            if not entry:
                continue
            rebuilt_section.append(self._strip_recent_raw_timestamp(entry[0]))
            if len(entry) > 1:
                rebuilt_section.extend(entry[1:])

        updated_lines = lines[: start_idx + 1] + rebuilt_section + lines[end_idx:]
        return "\n".join(updated_lines)

    def _count_recent_raw_entries(self, context: str) -> int:
        text = str(context or "")
        lines = text.splitlines()
        if not lines:
            return 0
        start_idx, end_idx = self._find_recent_raw_section_range(lines)
        if start_idx < 0:
            return 0
        count = 0
        for line in lines[start_idx + 1: end_idx]:
            if str(line).startswith("- "):
                count += 1
        return count

    def _find_recent_raw_section_range(self, lines: List[str]) -> Tuple[int, int]:
        start_idx = -1
        for idx, line in enumerate(lines):
            if str(line).strip().startswith(self.RECENT_RAW_SECTION_TITLE):
                start_idx = idx
                break
        if start_idx < 0:
            return -1, -1

        end_idx = len(lines)
        for idx in range(start_idx + 1, len(lines)):
            if self._is_prompt_section_header(lines[idx]):
                end_idx = idx
                break
        return start_idx, end_idx

    @staticmethod
    def _split_recent_raw_entries(lines: List[str]) -> Tuple[List[str], List[List[str]]]:
        leading_lines: List[str] = []
        entries: List[List[str]] = []
        current_entry: List[str] = []
        seen_entry = False
        for line in lines:
            if str(line).startswith("- "):
                seen_entry = True
                if current_entry:
                    entries.append(current_entry)
                current_entry = [line]
                continue
            if seen_entry and current_entry:
                current_entry.append(line)
            else:
                leading_lines.append(line)
        if current_entry:
            entries.append(current_entry)
        return leading_lines, entries

    @staticmethod
    def _strip_recent_raw_timestamp(line: str) -> str:
        text = str(line or "")
        match = re.match(
            r"^(\s*-\s*)\[\d{4}-\d{2}-\d{2}T[^\]]+\]\s*(.*)$",
            text,
        )
        if not match:
            return text
        prefix = str(match.group(1) or "")
        remainder = str(match.group(2) or "")
        if not remainder:
            return text
        return "{}{}".format(prefix, remainder)

    @staticmethod
    def _is_prompt_section_header(line: str) -> bool:
        return bool(re.match(r"^【[^】]+】[:：]?$", str(line or "").strip()))

    def _load_template(self, path: str) -> str:
        if not path:
            return ""
        if not os.path.isfile(path):
            self._log(
                "warning",
                "[EXT_MEMORY_TEMPLATE_MISSING] agent={} path={}".format(
                    self.agent_name,
                    path,
                ),
            )
            return ""
        try:
            with open(path, "r", encoding="utf-8") as file_obj:
                return file_obj.read()
        except Exception as exc:
            self._log(
                "warning",
                "[EXT_MEMORY_TEMPLATE_LOAD_FAIL] agent={} path={} error={}".format(
                    self.agent_name,
                    path,
                    exc,
                ),
            )
            return ""

    def _empty_remote_map(self) -> Dict[str, Any]:
        return {
            "updated_at": "",
            "node_to_remote": {},
            "remote_to_node": {},
        }

    def _ensure_remote_map_schema(self, data: Any) -> Dict[str, Any]:
        if not isinstance(data, dict):
            return self._empty_remote_map()
        node_to_remote = data.get("node_to_remote", {})
        remote_to_node = data.get("remote_to_node", {})
        if not isinstance(node_to_remote, dict):
            node_to_remote = {}
        if not isinstance(remote_to_node, dict):
            remote_to_node = {}
        return {
            "updated_at": str(data.get("updated_at", "") or ""),
            "node_to_remote": node_to_remote,
            "remote_to_node": remote_to_node,
        }

    def _load_remote_map(self) -> Dict[str, Any]:
        if isinstance(self._remote_map_cache, dict):
            return self._remote_map_cache
        data = self._empty_remote_map()
        if os.path.isfile(self.map_path):
            try:
                data = self._ensure_remote_map_schema(utils.load_dict(self.map_path))
            except Exception as exc:
                self._log(
                    "warning",
                    "[EXT_MEMORY_MAP_LOAD_FAIL] agent={} path={} error={}".format(
                        self.agent_name,
                        self.map_path,
                        exc,
                    ),
                )
                data = self._empty_remote_map()
        self._remote_map_cache = data
        return data

    def _save_remote_map(self, data: Dict[str, Any]) -> bool:
        if not isinstance(data, dict):
            return False
        try:
            os.makedirs(self.storage_dir, exist_ok=True)
            data["updated_at"] = datetime.datetime.now().strftime("%Y%m%d-%H:%M:%S")
            utils.save_dict(data, self.map_path, indent=2)
            self._remote_map_cache = data
            return True
        except Exception as exc:
            self._log(
                "warning",
                "[EXT_MEMORY_MAP_SAVE_FAIL] agent={} path={} error={}".format(
                    self.agent_name,
                    self.map_path,
                    exc,
                ),
            )
            return False

    def _bind_remote_map(
        self,
        node_id: str,
        remote_id: str,
        node_type: str,
        create_time: Any,
    ) -> bool:
        node_id = str(node_id or "").strip()
        remote_id = str(remote_id or "").strip()
        if not node_id or not remote_id:
            return False
        data = self._load_remote_map()
        node_to_remote = data.setdefault("node_to_remote", {})
        remote_to_node = data.setdefault("remote_to_node", {})
        if not isinstance(node_to_remote, dict):
            node_to_remote = {}
            data["node_to_remote"] = node_to_remote
        if not isinstance(remote_to_node, dict):
            remote_to_node = {}
            data["remote_to_node"] = remote_to_node

        old_remote = node_to_remote.get(node_id, {})
        if isinstance(old_remote, dict):
            old_remote_id = str(old_remote.get("remote_id", "") or "").strip()
            if old_remote_id and old_remote_id != remote_id:
                remote_to_node.pop(old_remote_id, None)

        conflict = remote_to_node.get(remote_id, {})
        if isinstance(conflict, dict):
            old_node_id = str(conflict.get("node_id", "") or "").strip()
            if old_node_id and old_node_id != node_id:
                self._log(
                    "warning",
                    "[EXT_MEMORY_MAP_CONFLICT] agent={} remote_id={} old_node_id={} new_node_id={}".format(
                        self.agent_name,
                        remote_id,
                        old_node_id,
                        node_id,
                    ),
                )

        create_time_str = self._format_time(create_time)
        node_to_remote[node_id] = {
            "remote_id": remote_id,
            "node_type": str(node_type or ""),
            "create_time": create_time_str,
        }
        remote_to_node[remote_id] = {
            "node_id": node_id,
            "node_type": str(node_type or ""),
            "create_time": create_time_str,
        }
        saved = self._save_remote_map(data)
        if saved:
            self._log(
                "info",
                "[EXT_MEMORY_MAP_BIND] agent={} node_id={} remote_id={} node_type={}".format(
                    self.agent_name,
                    node_id,
                    remote_id,
                    node_type,
                ),
            )
        return saved

    def _resolve_path(self, raw_path: Any) -> str:
        text = str(raw_path or "").strip()
        if not text:
            return ""
        if os.path.isabs(text):
            return os.path.normpath(text)
        return os.path.normpath(os.path.join(self._project_root, text))

    def _format_time(self, value: Any) -> str:
        if isinstance(value, datetime.datetime):
            return value.strftime("%Y%m%d-%H:%M:%S")
        if isinstance(value, str):
            return value
        return utils.get_timer().get_date("%Y%m%d-%H:%M:%S")

    @staticmethod
    def _safe_bool(value: Any, default: bool) -> bool:
        if isinstance(value, bool):
            return value
        if value is None:
            return default
        if isinstance(value, str):
            text = value.strip().lower()
            if text in {"1", "true", "yes", "on"}:
                return True
            if text in {"0", "false", "no", "off"}:
                return False
        return bool(value)

    @staticmethod
    def _safe_float(value: Any, default: float) -> float:
        if isinstance(value, bool):
            return float(default)
        try:
            return float(value)
        except Exception:
            return float(default)

    @staticmethod
    def _safe_int(value: Any, default: int) -> int:
        if isinstance(value, bool):
            return int(default)
        try:
            return int(value)
        except Exception:
            return int(default)

    @staticmethod
    def _trim_for_log(text: str, max_len: int = 120) -> str:
        text = str(text or "").replace("\n", " ").strip()
        if len(text) <= max_len:
            return text
        return text[:max_len] + "..."

    @staticmethod
    def _extract_ranked_memory_remote_ids(details: Dict[str, Any], max_items: int = 10):
        if not isinstance(details, dict):
            return []
        ranked_memories = details.get("ranked_memories", [])
        if not isinstance(ranked_memories, list):
            return []
        out = []
        seen = set()
        for item in ranked_memories:
            if not isinstance(item, dict):
                continue
            remote_id = str(item.get("remote_id", "") or "").strip()
            if not remote_id or remote_id in seen:
                continue
            seen.add(remote_id)
            out.append(remote_id)
            if len(out) >= int(max_items):
                break
        return out

    def _log(self, level: str, message: str):
        logger = self.logger
        if logger is None:
            return
        fn = getattr(logger, level, None)
        if not callable(fn):
            fn = getattr(logger, "info", None)
        if callable(fn):
            fn(message)
