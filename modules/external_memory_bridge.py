"""Bridge module for EC-Doll external memory integration."""

from __future__ import annotations

import datetime
import os
from string import Template
from typing import Any, Dict, Optional

from modules import utils
from modules.ec_doll_memory_service_client import ECDollMemoryServiceClient


class ExternalMemoryBridge:
    """Encapsulate external memory read/write behaviors for a single agent."""

    MAP_FILENAME = "external_memory_node_map.json"

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
        self.user_id = str(self.cfg.get("user_id", "") or "").strip()
        self.ingest_template_file = self._resolve_path(self.cfg.get("ingest_template_file", ""))
        self.chat_prompt_file = self._resolve_path(self.cfg.get("chat_prompt_file", ""))
        self.map_path = os.path.join(self.storage_dir, self.MAP_FILENAME)

        self._remote_map_cache = None
        self.client: Optional[ECDollMemoryServiceClient] = None

        self._log(
            "info",
            "[EXT_MEMORY_CFG] agent={} enabled={} user_id={} read_mode={} fallback_to_local={} base_url={} timeout={} ingest_template_file={} chat_prompt_file={}".format(
                self.agent_name,
                self.enabled,
                self.user_id,
                self.read_mode,
                self.fallback_to_local,
                self.base_url,
                self.timeout,
                self.ingest_template_file,
                self.chat_prompt_file,
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
        return bool(self.enabled and self.client and self.user_id)

    def enabled_for_chat_read(self) -> bool:
        if not (self.enabled and self.client and self.user_id):
            return False
        return self.read_mode == "chat_only"

    def resolve_chat_prompt_file(self) -> str:
        return self.chat_prompt_file

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

        try:
            response = self.client.ingest_memory(  # type: ignore[union-attr]
                content=content,
                user_id=self.user_id,
                remote_id=None,
            )
            remote_id = str((response or {}).get("remote_id", "") or "").strip()
            result["ok"] = True
            result["remote_id"] = remote_id
            result["reason"] = "ok"
            self._log(
                "info",
                "[EXT_MEMORY_INGEST] agent={} user_id={} node_id={} node_type={} remote_id={} content_len={}".format(
                    self.agent_name,
                    self.user_id,
                    node_id,
                    node_type,
                    remote_id,
                    len(content),
                ),
            )
            if remote_id:
                self._bind_remote_map(
                    node_id=node_id,
                    remote_id=remote_id,
                    node_type=node_type,
                    create_time=create_time,
                )
            else:
                self._log(
                    "warning",
                    "[EXT_MEMORY_INGEST] agent={} node_id={} reason=missing_remote_id".format(
                        self.agent_name,
                        node_id,
                    ),
                )
            return result
        except Exception as exc:
            result["reason"] = "request_failed"
            self._log(
                "warning",
                "[EXT_MEMORY_INGEST_FAIL] agent={} node_id={} error={}".format(
                    self.agent_name,
                    node_id,
                    exc,
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

        try:
            response = self.client.retrieve_memory_context(  # type: ignore[union-attr]
                query=query,
                user_id=self.user_id,
            )
            context = str((response or {}).get("formatted_prompt", "") or "").strip()
            if context:
                result["ok"] = True
                result["reason"] = "ok"
                result["context"] = context
                self._log(
                    "info",
                    "[EXT_MEMORY_RETRIEVE] agent={} user_id={} other={} query={} context_len={}".format(
                        self.agent_name,
                        self.user_id,
                        other_name,
                        self._trim_for_log(query),
                        len(context),
                    ),
                )
                return result

            result["reason"] = "empty_formatted_prompt"
            self._log(
                "warning",
                "[EXT_MEMORY_RETRIEVE_FAIL] agent={} other={} query={} reason=empty_formatted_prompt".format(
                    self.agent_name,
                    other_name,
                    self._trim_for_log(query),
                ),
            )
            return result
        except Exception as exc:
            result["reason"] = "request_failed"
            self._log(
                "warning",
                "[EXT_MEMORY_RETRIEVE_FAIL] agent={} other={} query={} error={}".format(
                    self.agent_name,
                    other_name,
                    self._trim_for_log(query),
                    exc,
                ),
            )
            return result

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
    def _trim_for_log(text: str, max_len: int = 120) -> str:
        text = str(text or "").replace("\n", " ").strip()
        if len(text) <= max_len:
            return text
        return text[:max_len] + "..."

    def _log(self, level: str, message: str):
        logger = self.logger
        if logger is None:
            return
        fn = getattr(logger, level, None)
        if not callable(fn):
            fn = getattr(logger, "info", None)
        if callable(fn):
            fn(message)
