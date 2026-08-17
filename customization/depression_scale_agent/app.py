import os
import sys
import json
from typing import Any

import gradio as gr


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from modules.memory import Event  # noqa: E402
from modules import utils  # noqa: E402
from modules.intervention_manager import InterventionManager  # noqa: E402
from modules.model.llm_model import create_llm_model  # noqa: E402
from customization.depression_scale_agent.ExpertLLM import ExpertLLM  # type: ignore
from customization.snapshot_ui_service import (  # noqa: E402
    UserStub,
    create_checkpoint_game,
    list_agents as _list_agents,
    list_simulations as _list_simulations,
    list_snapshot_files as _list_snapshot_files,
    load_config as _load_snapshot_config,
    load_conversation as _load_conversation,
    resolve_snapshot_file as _resolve_snapshot_file,
)


CHECKPOINTS_ROOT = os.path.join(BASE_DIR, "results", "checkpoints")
STATIC_ROOT = os.path.join(BASE_DIR, "frontend", "static")
DEFAULT_USER_NAME = "用户"
QUESTIONS_ROOT = os.path.join(BASE_DIR, "customization", "depression_scale_agent", "questions")
OUTPUT_ROOT = os.path.join(QUESTIONS_ROOT, "adhoc")
ENV_FILE_PATH = os.path.join(BASE_DIR, ".env")
TRACE_JSON_SUFFIX = "_prompt_trace.json"
TRACE_MD_SUFFIX = "_prompt_trace.md"


_ENV_LOADED = False

CHAIN_MODE_AUTO = "auto"
CHAIN_MODE_DYNAMIC = "depression_dynamic"

CHAIN_MODE_CHOICES = {
    CHAIN_MODE_AUTO,
    CHAIN_MODE_DYNAMIC,
}

DEFAULT_CHAIN_MODE = str(
    os.getenv("DEPR_SCALE_CHAIN_MODE", CHAIN_MODE_AUTO) or CHAIN_MODE_AUTO
).strip().lower()
if DEFAULT_CHAIN_MODE not in CHAIN_MODE_CHOICES:
    DEFAULT_CHAIN_MODE = CHAIN_MODE_AUTO


def _load_env_file_once(path=ENV_FILE_PATH):
    global _ENV_LOADED
    if _ENV_LOADED:
        return
    _ENV_LOADED = True

    if not os.path.exists(path):
        return

    try:
        with open(path, "r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("export "):
                    line = line[len("export "):].strip()
                if "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip()
                if not key:
                    continue
                if (
                    len(value) >= 2
                    and ((value[0] == '"' and value[-1] == '"') or (value[0] == "'" and value[-1] == "'"))
                ):
                    value = value[1:-1]
                if os.getenv(key) is None:
                    os.environ[key] = value
    except Exception:
        return


def _json_safe(value: Any):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    return str(value)


def _clone_json_safe(value: Any):
    return json.loads(json.dumps(_json_safe(value), ensure_ascii=False))


class ChatSession:
    def __init__(self, sim_name, snapshot_file=None, runtime_config=None, conversation=None):
        _load_env_file_once()
        self.sim_name = sim_name
        if runtime_config is not None:
            self.snapshot_file = str(snapshot_file or "")
            self.config = _clone_json_safe(runtime_config)
            self.conversation = _clone_json_safe(conversation or {})
        else:
            self.snapshot_file = resolve_snapshot_file(sim_name, snapshot_file)
            self.config = load_config(sim_name, self.snapshot_file)
            self.conversation = load_conversation(sim_name)
        if self.config is None:
            raise ValueError(f"找不到存档: {sim_name}")
        if isinstance(self.config.get("time"), str):
            self.config["time"] = {"start": self.config.get("time")}
        self.logger, self.game = create_checkpoint_game(
            sim_name,
            STATIC_ROOT,
            self.config,
            self.conversation,
        )
        self.intervention = InterventionManager(config=self.config, logger=self.logger)
        self.game.set_intervention_manager(self.intervention)
        self._forced_llm = None
        self._forced_llm_key = ""
        self.agent_name = ""
        self.agent = None
        self.chats = []
        self.doctor_name = self._resolve_doctor_name()
        self.chain_mode = self._resolve_chain_mode()
        self._last_generate_route = "default"
        self._last_answer_trace = {}
        if self.snapshot_file:
            self.logger.info(
                "[DEPR_SCALE][SNAPSHOT] sim={} snapshot={}".format(
                    self.sim_name,
                    self.snapshot_file,
                )
            )
        if self.doctor_name:
            self.logger.info(
                "[DEPR_SCALE][DOCTOR_NAME] sim={} doctor_name={}".format(
                    self.sim_name,
                    self.doctor_name,
                )
            )
        self.logger.info(
            "[DEPR_SCALE][CHAIN_MODE] mode={} legacy_update_chain=false dynamic_chain={}".format(
                self.chain_mode,
                self._chain_uses_dynamic(),
            )
        )

    def _resolve_doctor_name(self):
        if getattr(self, "intervention", None):
            doctor_name = str(getattr(self.intervention, "doctor", "") or "").strip()
            if doctor_name:
                return doctor_name
        intervention_cfg = (self.config or {}).get("intervention", {}) or {}
        return str(intervention_cfg.get("doctor", "") or "").strip()

    def _resolve_chain_mode(self):
        intervention_cfg = (self.config or {}).get("intervention", {}) or {}
        cfg_mode = str(
            intervention_cfg.get("depression_scale_chain_mode", "") or ""
        ).strip().lower()
        env_mode = str(
            os.getenv("DEPR_SCALE_CHAIN_MODE", DEFAULT_CHAIN_MODE) or DEFAULT_CHAIN_MODE
        ).strip().lower()
        requested_mode = cfg_mode or env_mode or CHAIN_MODE_AUTO
        if requested_mode not in CHAIN_MODE_CHOICES:
            self.logger.warning(
                "[DEPR_SCALE][CHAIN_MODE] invalid_mode={} fallback=auto".format(
                    requested_mode
                )
            )
            requested_mode = CHAIN_MODE_AUTO
        if requested_mode != CHAIN_MODE_AUTO:
            return requested_mode
        return CHAIN_MODE_DYNAMIC

    def _chain_uses_dynamic(self):
        return self.chain_mode == CHAIN_MODE_DYNAMIC

    def _sync_agent_chain_switch(self):
        if not self.agent:
            return
        self.logger.info(
            "[DEPR_SCALE][CHAIN_SWITCH] agent={} mode={} dynamic_chain={} dynamic_runtime_enabled={}".format(
                self.agent.name,
                self.chain_mode,
                self._chain_uses_dynamic(),
                bool(getattr(self.agent, "depression_dynamic", None)),
            )
        )

    def _dump_dynamic_state_for_trace(self):
        if not self.agent:
            return {}
        state_payload = {}
        engine = getattr(self.agent, "depression_dynamic", None)
        try:
            if engine is not None and callable(getattr(engine, "to_dict", None)):
                state_payload = engine.to_dict()
        except Exception as exc:
            state_payload = {"error": "dump_failed", "detail": str(exc)}
        trace_payload = {
            "enabled_runtime": bool(engine),
            "state_dump": state_payload if isinstance(state_payload, dict) else {},
            "chain_mode": self.chain_mode,
            "chain_uses_dynamic": self._chain_uses_dynamic(),
        }
        return _clone_json_safe(trace_payload)

    def _normalize_external_retrieval_trace(self, retrieval, query_text=""):
        payload = retrieval if isinstance(retrieval, dict) else {}
        context = str(payload.get("context", "") or "")
        details = payload.get("details", {})
        if not isinstance(details, dict):
            details = {}
        return {
            "ok": bool(payload.get("ok", False)),
            "query": str(payload.get("query", "") or query_text or ""),
            "reason": str(payload.get("reason", "") or ""),
            "context": context,
            "context_len": len(context),
            "details": _clone_json_safe(details),
            "scoped_user_id": str(payload.get("scoped_user_id", "") or ""),
        }

    def _build_prompt_payload_meta(self, payload):
        if not isinstance(payload, dict):
            return {}
        prompt_text = str(payload.get("prompt", "") or "")
        return {
            "keys": sorted([str(k) for k in payload.keys()]),
            "prompt_len": len(prompt_text),
            "retry": payload.get("retry"),
            "temperature": payload.get("temperature"),
            "has_callback": "callback" in payload,
            "failsafe_type": type(payload.get("failsafe")).__name__ if ("failsafe" in payload) else "",
        }

    def _build_forced_alignment_trace(self, prompt_route):
        return {
            "forced_context_enabled": True,
            "uses_same_direct_engine_helpers": True,
            "uses_same_prompt_templates": True,
            "prompt_route": str(prompt_route or ""),
            "known_differences": [
                "depression_scale_app 直接调用 scratch 构建 prompt，不经过 Agent.completion 包装层。",
                "depression_scale_app 通过 Agent 的 direct-engine helper 预先生成 depression_chat_block。",
            ],
        }

    def get_last_answer_trace(self):
        return _clone_json_safe(self._last_answer_trace)

    def _commit_dynamic_chat_event(self, user, relation, chats_before_reply, reply_text):
        if (not self.agent) or (not self._chain_uses_dynamic()):
            return
        if not bool(getattr(self.agent, "depression_dynamic", None)):
            return
        try:
            context = self.agent._build_depression_chat_context(
                other=user,
                relation_summary=relation,
                chats=chats_before_reply,
            )
            self.agent._commit_depression_generate_chat(context, reply_text)
        except Exception as exc:
            self.logger.warning("[DEPR_SCALE][DYNAMIC_COMMIT] commit_error={}".format(exc))

    def _relation_text(self, relation):
        return str(relation or "").strip()

    def _get_runtime_emotion_snapshot(self):
        if not self.agent:
            return {}
        engine = getattr(self.agent, "depression_dynamic", None)
        if engine is None or not callable(getattr(engine, "get_current_state_info", None)):
            return {}
        try:
            info = engine.get_current_state_info()
        except Exception:
            return {}
        if not isinstance(info, dict):
            return {}
        emotion = info.get("emotion", {}) if isinstance(info.get("emotion", {}), dict) else {}
        return _clone_json_safe(emotion)

    def _build_depression_chat_block(self, user, relation, chats):
        emotion_trace = {
            "applied": False,
            "reason": "",
            "relationship": self._relation_text(relation),
            "emotion_before": {},
            "emotion_after": {},
        }
        if not self.agent:
            emotion_trace["reason"] = "agent_missing"
            return "", emotion_trace
        emotion_trace["emotion_before"] = self._get_runtime_emotion_snapshot()
        dynamic_runtime_enabled = bool(
            self._chain_uses_dynamic()
            and bool(getattr(self.agent, "depression_dynamic", None))
        )
        if not dynamic_runtime_enabled:
            emotion_trace["reason"] = "dynamic_runtime_disabled"
            emotion_trace["emotion_after"] = self._get_runtime_emotion_snapshot()
            return "", emotion_trace
        try:
            prompt_kwargs, _ = self.agent._prepare_depression_generate_chat(
                args=(self.agent, user, relation, chats),
                kwargs={},
            )
            depression_chat_block = str(
                prompt_kwargs.get("depression_chat_block", "") or ""
            )
            emotion_trace["applied"] = bool(depression_chat_block)
            emotion_trace["reason"] = (
                "direct_engine_preview_ok" if depression_chat_block else "direct_engine_preview_empty"
            )
            emotion_trace["emotion_after"] = self._get_runtime_emotion_snapshot()
            return depression_chat_block, emotion_trace
        except Exception as exc:
            emotion_trace["reason"] = "direct_engine_preview_error:{}".format(exc)
            emotion_trace["emotion_after"] = self._get_runtime_emotion_snapshot()
            return "", emotion_trace

    def _get_forced_llm_runtime(self):
        if not getattr(self, "intervention", None):
            return None, "intervention_missing"
        try:
            forced_cfg = self.intervention.get_forced_llm_runtime_config()
        except Exception as exc:
            self.logger.warning("[DEPR_SCALE][FORCED_LLM] cfg_error={}".format(exc))
            return None, "cfg_error:{}".format(exc)
        if not forced_cfg:
            return None, "cfg_unavailable"

        cache_key = "{}|{}|{}|{}".format(
            forced_cfg.get("provider", ""),
            forced_cfg.get("model", ""),
            forced_cfg.get("base_url", ""),
            forced_cfg.get("api_key", ""),
        )
        if self._forced_llm is None or self._forced_llm_key != cache_key:
            self._forced_llm = create_llm_model(forced_cfg)
            self._forced_llm_key = cache_key
        return (self._forced_llm, forced_cfg), "ok"

    def _resolve_local_retrieval_profile_when_external_disabled(self):
        bridge = getattr(getattr(self, "agent", None), "external_memory_bridge", None)
        if bridge is not None and bool(getattr(bridge, "enabled", False)):
            return {}
        intervention_cfg = (self.config or {}).get("intervention", {}) or {}
        memory_policy = intervention_cfg.get("memory_policy", {}) or {}
        if (not isinstance(memory_policy, dict)) or (not bool(memory_policy.get("enabled", False))):
            return {}
        profile = memory_policy.get("retrieve_profile", {}) or {}
        if not isinstance(profile, dict):
            return {}
        return _clone_json_safe(profile)

    def _build_prompt_payload_before_answer(self, user, relation, chats, question_text):
        depression_chat_block, emotion_trace = self._build_depression_chat_block(
            user, relation, chats
        )
        prompt_kwargs = {
            "depression_chat_block": depression_chat_block,
            "chat_history_target_name": self.doctor_name or user.name,
        }
        local_retrieval_profile = self._resolve_local_retrieval_profile_when_external_disabled()
        if local_retrieval_profile:
            prompt_kwargs["retrieval_profile"] = local_retrieval_profile
        route = "local"
        trace_data = {
            "question": str(question_text or ""),
            "prompt_route": route,
            "depression_chat_block": depression_chat_block,
            "emotion_inference": _clone_json_safe(emotion_trace),
            "external_memory_retrieval": {
                "ok": False,
                "query": str(question_text or ""),
                "reason": "bridge_disabled_or_not_ready",
                "context": "",
                "context_len": 0,
                "details": {},
                "scoped_user_id": "",
            },
        }
        bridge = getattr(self.agent, "external_memory_bridge", None)
        if bridge and bridge.enabled_for_chat_read():
            query_text = str(question_text or "").strip()
            retrieval = bridge.retrieve_chat_context(
                chats=[(user.name, query_text)],
                other_name=user.name,
                is_initiator=False,
                turn_no=1,
            )
            trace_data["external_memory_retrieval"] = self._normalize_external_retrieval_trace(
                retrieval,
                query_text=query_text,
            )
            query = str(retrieval.get("query", "") or "").replace("\n", " ").strip()
            if len(query) > 120:
                query = query[:120] + "..."
            retrieval_ok = bool(retrieval.get("ok", False))
            route_reason = str(retrieval.get("reason", "") or "")
            if retrieval_ok or (not bridge.fallback_to_local):
                external_memory_context = str(retrieval.get("context", "") or "")
                if retrieval_ok:
                    route_reason = "retrieve_ok"
                elif not route_reason:
                    route_reason = "retrieve_failed_no_fallback"
                self.logger.info(
                    "[DEPR_SCALE][EXT_MEMORY_ROUTE] agent={} route=external reason={} query={} context_len={}".format(
                        self.agent.name,
                        route_reason,
                        query,
                        len(external_memory_context),
                    )
                )
                route = "external"
                trace_data["prompt_route"] = route
                prompt_payload = self.agent.scratch.prompt_generate_chat(
                    self.agent,
                    user,
                    relation,
                    chats,
                    memory_source="external",
                    external_memory_context=external_memory_context,
                    **prompt_kwargs,
                )
                return prompt_payload, route, trace_data
            self.logger.info(
                "[DEPR_SCALE][EXT_MEMORY_ROUTE] agent={} route=local reason={} query={}".format(
                    self.agent.name,
                    route_reason or "retrieve_failed_fallback",
                    query,
                )
            )
            trace_data["external_memory_retrieval"]["reason"] = route_reason or "retrieve_failed_fallback"
        prompt_payload = self.agent.scratch.prompt_generate_chat(
            self.agent,
            user,
            relation,
            chats,
            **prompt_kwargs,
        )
        trace_data["prompt_route"] = route
        return prompt_payload, route, trace_data

    def _generate_chat_forced_then_fallback(
        self,
        user,
        relation,
        chats,
        question_text="",
        capture_trace=False,
    ):
        prompt_payload, prompt_route, prompt_trace = self._build_prompt_payload_before_answer(
            user, relation, chats, question_text
        )
        trace = {
            "question": str(question_text or ""),
            "prompt_route": str(prompt_route or ""),
            "external_memory_retrieval": _clone_json_safe(
                (prompt_trace or {}).get("external_memory_retrieval", {})
            ),
            "depression_chat_block": str((prompt_trace or {}).get("depression_chat_block", "") or ""),
            "emotion_inference": _clone_json_safe((prompt_trace or {}).get("emotion_inference", {})),
            "dynamic_state_before_answer": self._dump_dynamic_state_for_trace(),
            "forced_chain_alignment": self._build_forced_alignment_trace(prompt_route),
            "llm_route": "",
            "llm_route_reason": "",
            "prompt_before_dynamic": "",
            "prompt_before_dynamic_meta": self._build_prompt_payload_meta(prompt_payload),
            "full_injected_prompt": "",
            "final_prompt_payload_meta": {},
        }
        if isinstance(prompt_payload, dict):
            trace["prompt_before_dynamic"] = str(prompt_payload.get("prompt", "") or "")
        prev_ctx = getattr(self.agent, "_chat_route_ctx", None)
        try:
            # Keep generate_chat in forced context so dynamic depression uses forced_chain rules.
            self.agent._chat_route_ctx = {
                "forced": True,
                "peer_name": getattr(user, "name", ""),
                "peer_agent": user,
            }
            runtime, reason = self._get_forced_llm_runtime()
            if runtime is not None:
                forced_llm, forced_cfg = runtime
                forced_prompt = dict(prompt_payload)
                forced_prompt["failsafe"] = None
                forced_prompt["retry"] = max(
                    1,
                    int(forced_cfg.get("retry", forced_prompt.get("retry", 2)) or 2),
                )
                forced_prompt["temperature"] = float(
                    forced_cfg.get("temperature", forced_prompt.get("temperature", 0.5))
                )
                forced_route_reason = "success"
                try:
                    output = forced_llm.completion(
                        **forced_prompt,
                        caller="depression_scale_generate_chat_forced",
                    )
                    if output is not None:
                        self._last_generate_route = "forced_llm"
                        self.logger.info(
                            "[DEPR_SCALE][CHAT_ROUTE] route=forced_llm reason=success prompt_route={}".format(
                                prompt_route
                            )
                        )
                        trace["llm_route"] = "forced_llm"
                        trace["llm_route_reason"] = "success"
                        trace["full_injected_prompt"] = str(forced_prompt.get("prompt", "") or "")
                        trace["final_prompt_payload_meta"] = self._build_prompt_payload_meta(forced_prompt)
                        if capture_trace:
                            return output, _clone_json_safe(trace)
                        return output
                    self.logger.warning("[DEPR_SCALE][CHAT_ROUTE] route=forced_llm reason=empty_fallback")
                    forced_route_reason = "empty_fallback"
                except Exception as exc:
                    self.logger.warning(
                        "[DEPR_SCALE][CHAT_ROUTE] route=forced_llm reason=error:{} fallback=default".format(exc)
                    )
                    forced_route_reason = "error:{}".format(exc)
                trace["llm_route"] = "forced_llm_fallback_default"
                trace["llm_route_reason"] = forced_route_reason
            else:
                self.logger.info(
                    "[DEPR_SCALE][CHAT_ROUTE] route=forced_llm reason={} fallback=default".format(reason)
                )
                trace["llm_route"] = "forced_cfg_unavailable_fallback_default"
                trace["llm_route_reason"] = str(reason or "cfg_unavailable")

            self.logger.info(
                "[DEPR_SCALE][CHAT_ROUTE] route=default reason=fallback prompt_route={}".format(prompt_route)
            )
            self._last_generate_route = "default"
            default_prompt = dict(prompt_payload)
            trace["full_injected_prompt"] = str(default_prompt.get("prompt", "") or "")
            trace["final_prompt_payload_meta"] = self._build_prompt_payload_meta(default_prompt)
            llm = getattr(self.agent, "_llm", None)
            if llm is None:
                output = default_prompt.get("failsafe")
                if trace["llm_route"] == "":
                    trace["llm_route"] = "default_no_llm"
                    trace["llm_route_reason"] = "llm_missing_use_failsafe"
                if capture_trace:
                    return output, _clone_json_safe(trace)
                return output
            output = llm.completion(
                **default_prompt,
                caller="depression_scale_generate_chat_default",
            )
            if trace["llm_route"] == "":
                trace["llm_route"] = "default"
                trace["llm_route_reason"] = "fallback"
            if capture_trace:
                return output, _clone_json_safe(trace)
            return output
        finally:
            self.agent._chat_route_ctx = prev_ctx

    def set_agent(self, agent_name):
        self.agent_name = agent_name
        self.agent = self.game.get_agent(agent_name)
        self.agent.reset()
        self._sync_agent_chain_switch()
        self.chats = []

    def clear_history(self):
        self.chats = []

    def chat(self, user_text, user_name=DEFAULT_USER_NAME):
        if not self.agent:
            return "请先选择一个角色。"
        if not user_text.strip():
            return "请先输入内容。"

        user = UserStub(user_name, self.agent.get_tile())
        relation = self.agent.completion("summarize_relation", self.agent, user_name)

        chats = list(self.chats)
        chats.append((user_name, user_text))
        reply = self._generate_chat_forced_then_fallback(
            user, relation, chats, question_text=user_text
        )
        chats.append((self.agent.name, reply))
        self.chats = chats

        summary = self.agent.completion("summarize_chats", chats)
        event = Event(
            self.agent.name,
            "对话",
            user_name,
            describe=summary,
            address=self.agent.get_tile().get_address(),
        )
        self.agent._add_concept("chat", event)
        self._commit_dynamic_chat_event(user, relation, chats[:-1], reply)
        self._append_conversation(chats, user_name)
        return reply

    def answer_without_memory(self, user_text, user_name=DEFAULT_USER_NAME):
        self._last_answer_trace = {}
        if not self.agent:
            return "请先选择一个角色。"
        if not user_text.strip():
            return "请先输入内容。"

        user = UserStub(user_name, self.agent.get_tile())
        relation = self.agent.completion("summarize_relation", self.agent, user_name)
        chats = [(user_name, user_text)]
        reply, trace = self._generate_chat_forced_then_fallback(
            user,
            relation,
            chats,
            question_text=user_text,
            capture_trace=True,
        )
        trace = trace if isinstance(trace, dict) else {}
        trace["question"] = str(user_text or "")
        trace["answer"] = str(reply or "")
        self._last_answer_trace = _clone_json_safe(trace)
        return reply

    def _append_conversation(self, chats, user_name):
        key = utils.get_timer().get_date("%Y%m%d-%H:%M")
        address = "，".join(self.agent.get_event().address)
        entry = {f"{user_name} -> {self.agent.name} @ {address}": chats}
        self.conversation.setdefault(key, []).append(entry)
        conversation_path = os.path.join(
            CHECKPOINTS_ROOT, self.sim_name, "conversation.json"
        )
        with open(conversation_path, "w", encoding="utf-8") as f:
            json.dump(self.conversation, f, indent=2, ensure_ascii=False)


def list_simulations():
    return _list_simulations(CHECKPOINTS_ROOT)


def list_question_jsonl_files():
    templates_dir = os.path.join(QUESTIONS_ROOT, "templates")
    if not os.path.isdir(templates_dir):
        return []
    return sorted(
        f
        for f in os.listdir(templates_dir)
        if f.endswith(".jsonl") and os.path.isfile(os.path.join(templates_dir, f))
    )


def resolve_question_file_path(jsonl_name):
    name = os.path.basename(str(jsonl_name or "").strip())
    if not name:
        return None
    path = os.path.join(QUESTIONS_ROOT, "templates", name)
    if not os.path.isfile(path):
        return None
    return path


def list_sim_json_files(sim_name):
    return _list_snapshot_files(CHECKPOINTS_ROOT, sim_name)


def resolve_snapshot_file(sim_name, snapshot_file=None):
    return _resolve_snapshot_file(CHECKPOINTS_ROOT, sim_name, snapshot_file)


def load_config(sim_name, snapshot_file=None):
    return _load_snapshot_config(
        CHECKPOINTS_ROOT,
        sim_name,
        snapshot_file,
        overwrite_agent_config_paths=False,
    )


def load_latest_config(sim_name):
    return load_config(sim_name)


def load_conversation(sim_name):
    return _load_conversation(CHECKPOINTS_ROOT, sim_name)


def list_agents(sim_name, snapshot_file=None):
    return _list_agents(
        CHECKPOINTS_ROOT,
        sim_name,
        snapshot_file,
        overwrite_agent_config_paths=False,
    )


def ensure_session(sim_name, snapshot_file, session):
    resolved_snapshot = resolve_snapshot_file(sim_name, snapshot_file)
    if not resolved_snapshot:
        return None
    if (
        session
        and session.sim_name == sim_name
        and getattr(session, "snapshot_file", None) == resolved_snapshot
    ):
        return session
    return ChatSession(sim_name, resolved_snapshot)


def iter_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            yield json.loads(line)


def resolve_question(item):
    if isinstance(item, str):
        return item
    if not isinstance(item, dict):
        raise ValueError(f"Unsupported jsonl item: {item}")
    for key in ("question", "prompt", "text", "content"):
        if key in item and str(item[key]).strip():
            return str(item[key]).strip()
    if len(item) == 1:
        return str(next(iter(item.values()))).strip()
    raise ValueError(f"Missing question text in item: {item}")


def default_output_path(input_path):
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    filename = os.path.basename(input_path)
    base, ext = os.path.splitext(filename)
    suffix = "_answered"
    if ext.lower() == ".jsonl":
        output_name = f"{base}{suffix}.jsonl"
    else:
        output_name = f"{filename}{suffix}.jsonl"
    return os.path.join(OUTPUT_ROOT, output_name)


def write_jsonl_line(handle, data):
    handle.write(json.dumps(data, ensure_ascii=False) + "\n")


def default_trace_paths(answered_output_path):
    output_dir = os.path.dirname(answered_output_path)
    base_name = os.path.basename(answered_output_path)
    name_root, _ = os.path.splitext(base_name)
    return (
        os.path.join(output_dir, f"{name_root}{TRACE_JSON_SUFFIX}"),
        os.path.join(output_dir, f"{name_root}{TRACE_MD_SUFFIX}"),
    )


def build_answer_trace_record(index, question, answer, trace_payload):
    trace = trace_payload if isinstance(trace_payload, dict) else {}
    return {
        "index": int(index),
        "question": str(question or ""),
         "depression_chat_block": str(trace.get("depression_chat_block", "") or ""), # <--- 新增这一行
        "full_injected_prompt": str(trace.get("full_injected_prompt", "") or ""),
        "answer": str(answer or ""),
        "llm_route": str(trace.get("llm_route", "") or ""),
        "llm_route_reason": str(trace.get("llm_route_reason", "") or ""),
        "prompt_route": str(trace.get("prompt_route", "") or ""),
        "external_memory_retrieval": _clone_json_safe(trace.get("external_memory_retrieval", {})),
        "emotion_inference": _clone_json_safe(trace.get("emotion_inference", {})),
        "dynamic_state_before_answer": _clone_json_safe(trace.get("dynamic_state_before_answer", {})),
        "forced_chain_alignment": _clone_json_safe(trace.get("forced_chain_alignment", {})),
        "prompt_meta": _clone_json_safe(trace.get("final_prompt_payload_meta", {})),
        "prompt_before_dynamic_meta": _clone_json_safe(trace.get("prompt_before_dynamic_meta", {})),
        "prompt_before_dynamic": str(trace.get("prompt_before_dynamic", "") or ""),
    }


def write_prompt_trace_json(path, records):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_clone_json_safe(records), f, ensure_ascii=False, indent=2)


def write_prompt_trace_markdown(path, records):
    lines = [
        "# 量表问答完整注入Prompt追踪",
        "",
        "以下为每道题的循环记录：量表问题 -> 完整注入Prompt -> 患者回答。",
        "",
    ]
    for record in records:
        idx = int(record.get("index", 0) or 0)
        lines.append(f"## 第{idx}题")
        lines.append("")
        lines.append("### 量表问题")
        lines.append(str(record.get("question", "") or ""))
        lines.append("")
        lines.append("### 完整注入Prompt")
        lines.append("```text")
        lines.append(str(record.get("full_injected_prompt", "") or ""))
        lines.append("```")
        lines.append("")
        lines.append("### 患者回答")
        lines.append(str(record.get("answer", "") or ""))
        lines.append("")
        lines.append("### 外置记忆查询返回")
        lines.append("```json")
        lines.append(json.dumps(record.get("external_memory_retrieval", {}), ensure_ascii=False, indent=2))
        lines.append("```")
        lines.append("")
        lines.append("### Emotion 推导结果")
        lines.append("```json")
        lines.append(json.dumps(record.get("emotion_inference", {}), ensure_ascii=False, indent=2))
        lines.append("```")
        lines.append("")
        lines.append("### 动态抑郁人设状态（回答前）")
        lines.append("```json")
        lines.append(json.dumps(record.get("dynamic_state_before_answer", {}), ensure_ascii=False, indent=2))
        lines.append("```")
        lines.append("")
        lines.append("### 强制干预链路对齐检查")
        lines.append("```json")
        lines.append(json.dumps(record.get("forced_chain_alignment", {}), ensure_ascii=False, indent=2))
        lines.append("```")
        lines.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def on_refresh():
    sims = list_simulations()
    default_sim = sims[0] if sims else None
    snapshots = list_sim_json_files(default_sim) if default_sim else []
    default_snapshot = snapshots[-1] if snapshots else None
    agents = list_agents(default_sim, default_snapshot) if default_sim and default_snapshot else []
    default_agent = agents[0] if agents else None
    session = ChatSession(default_sim, default_snapshot) if default_sim and default_snapshot else None
    return (
        gr.Dropdown(choices=sims, value=default_sim),
        gr.Dropdown(choices=snapshots, value=default_snapshot),
        gr.Dropdown(choices=agents, value=default_agent),
        session,
    )


def on_sim_change(sim_name):
    if not sim_name:
        return gr.Dropdown(choices=[], value=None), gr.Dropdown(choices=[], value=None), None
    snapshots = list_sim_json_files(sim_name)
    default_snapshot = snapshots[-1] if snapshots else None
    session = ChatSession(sim_name, default_snapshot) if default_snapshot else None
    agents = list_agents(sim_name, default_snapshot) if default_snapshot else []
    return (
        gr.Dropdown(choices=snapshots, value=default_snapshot),
        gr.Dropdown(choices=agents, value=(agents[0] if agents else None)),
        session,
    )


def on_snapshot_change(sim_name, snapshot_file):
    if not sim_name:
        return gr.Dropdown(choices=[], value=None), None
    resolved_snapshot = resolve_snapshot_file(sim_name, snapshot_file)
    if not resolved_snapshot:
        return gr.Dropdown(choices=[], value=None), None
    session = ChatSession(sim_name, resolved_snapshot)
    agents = list_agents(sim_name, resolved_snapshot)
    return gr.Dropdown(choices=agents, value=(agents[0] if agents else None)), session


def on_agent_change(agent_name, session):
    if not session or not agent_name:
        return session
    session.set_agent(agent_name)
    return session


def on_jsonl_change(jsonl_file):
    if not jsonl_file:
        return "", None, [], ""
    output_path = default_output_path(str(jsonl_file))
    return output_path, None, [], ""


def on_batch_run(jsonl_file, sim_name, snapshot_file, agent_name, session):
    if not jsonl_file:
        yield [], "请先选择 jsonl 文件。", None, session
        return
    input_path = resolve_question_file_path(jsonl_file)
    if not input_path:
        yield [], "所选 jsonl 文件不存在。", None, session
        return
    if not sim_name or not snapshot_file or not agent_name:
        yield [], "请先选择仿真、快照和患者。", None, session
        return

    session = ensure_session(sim_name, snapshot_file, session)
    if not session:
        yield [], "无法初始化会话。", None, session
        return
    if session.agent_name != agent_name:
        session.set_agent(agent_name)

    output_path = default_output_path(str(jsonl_file))
    rows = []
    trace_records = []
    trace_json_path, trace_md_path = default_trace_paths(output_path)
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            for idx, item in enumerate(iter_jsonl(input_path)):
                question = resolve_question(item)
                answer = session.answer_without_memory(question)
                trace_payload = session.get_last_answer_trace() if session else {}
                trace_records.append(
                    build_answer_trace_record(
                        index=idx + 1,
                        question=question,
                        answer=answer,
                        trace_payload=trace_payload,
                    )
                )
                result = dict(item) if isinstance(item, dict) else {"question": question}
                result["answer"] = answer
                write_jsonl_line(f, result)
                rows.append([idx + 1, question, answer])
                yield rows, f"已完成 {idx + 1} 条", output_path, session
    except Exception as exc:
        yield rows, f"执行失败: {exc}", None, session
        return

    try:
        write_prompt_trace_json(trace_json_path, trace_records)
        write_prompt_trace_markdown(trace_md_path, trace_records)
    except Exception as exc:
        yield rows, f"执行失败(trace): {exc}", output_path, session
        return

    yield rows, (
        "完成，共 {} 条 | Prompt追踪(JSON): {} | Prompt追踪(MD): {}".format(
            len(rows),
            trace_json_path,
            trace_md_path,
        )
    ), output_path, session


def on_expert_review(answered_file, system_prompt, model_name):
    if not answered_file:
        return "请先生成 answered 的 jsonl 文件。"
    if not os.path.exists(answered_file):
        return f"文件不存在: {answered_file}"
    try:
        with open(answered_file, "r", encoding="utf-8") as f:
            content = f.read().strip()
    except Exception as exc:
        return f"读取文件失败: {exc}"
    if not content:
        return "answered 文件内容为空。"

    try:
        llm = ExpertLLM(model=model_name)
        reply = llm.generate(
            content,
            system_prompt=system_prompt or None,
            caller="depression_scale_expert_review",
        )
        return reply or ""
    except Exception as exc:
        return f"专家模型调用失败: {exc}"


def build_ui():
    sims = list_simulations()
    default_sim = sims[0] if sims else None
    snapshots = list_sim_json_files(default_sim) if default_sim else []
    default_snapshot = snapshots[-1] if snapshots else None
    agents = list_agents(default_sim, default_snapshot) if default_sim and default_snapshot else []
    default_agent = agents[0] if agents else None
    question_files = list_question_jsonl_files()
    default_question_file = question_files[0] if question_files else None

    with gr.Blocks(title="AI 小镇 - 角色对话") as demo:
        gr.Markdown("# AI 小镇 - 角色对话")
        with gr.Row():
            sim_dropdown = gr.Dropdown(
                label="选择模拟存档",
                choices=sims,
                value=default_sim,
                interactive=True,
            )
            snapshot_dropdown = gr.Dropdown(
                label="选择配置快照(JSON)",
                choices=snapshots,
                value=default_snapshot,
                interactive=True,
            )
            refresh_button = gr.Button("刷新存档")
        agent_dropdown = gr.Dropdown(
            label="选择角色",
            choices=agents,
            value=default_agent,
            interactive=True,
        )

        gr.Markdown("## 批量问答（jsonl）")
        with gr.Row():
            jsonl_file = gr.Dropdown(
                label="选择 jsonl 文件",
                choices=question_files,
                value=default_question_file,
                interactive=True,
            )
            output_path = gr.Textbox(
                label="输出文件路径",
                interactive=False,
            )
        batch_button = gr.Button("开始批量问答")
        batch_status = gr.Textbox(label="批量状态", interactive=False)
        batch_table = gr.Dataframe(
            headers=["序号", "问题", "回复"],
            datatype=["number", "str", "str"],
            row_count=0,
            col_count=(3, "fixed"),
            interactive=False,
        )
        batch_output = gr.File(label="输出文件", interactive=False)
        gr.Markdown("## 专家模型总结")
        with gr.Row():
            expert_system_prompt = gr.Textbox(
                label="System Prompt",
                placeholder="输入专家模型的系统提示词（可选）",
                lines=2,
            )
            expert_model = gr.Textbox(
                label="模型名称",
                value="deepseek-chat",
                interactive=True,
            )
        expert_button = gr.Button("调用专家模型")
        expert_result = gr.Textbox(label="专家模型输出", lines=8, interactive=False)

        session_state = gr.State(None)

        refresh_button.click(
            on_refresh,
            inputs=[],
            outputs=[sim_dropdown, snapshot_dropdown, agent_dropdown, session_state],
        )
        sim_dropdown.change(
            on_sim_change,
            inputs=[sim_dropdown],
            outputs=[snapshot_dropdown, agent_dropdown, session_state],
        )
        snapshot_dropdown.change(
            on_snapshot_change,
            inputs=[sim_dropdown, snapshot_dropdown],
            outputs=[agent_dropdown, session_state],
        )
        agent_dropdown.change(
            on_agent_change,
            inputs=[agent_dropdown, session_state],
            outputs=[session_state],
        )
        jsonl_file.change(
            on_jsonl_change,
            inputs=[jsonl_file],
            outputs=[output_path, batch_output, batch_table, batch_status],
        )
        batch_button.click(
            on_batch_run,
            inputs=[jsonl_file, sim_dropdown, snapshot_dropdown, agent_dropdown, session_state],
            outputs=[batch_table, batch_status, batch_output, session_state],
        )
        expert_button.click(
            on_expert_review,
            inputs=[batch_output, expert_system_prompt, expert_model],
            outputs=[expert_result],
        )

    return demo


if __name__ == "__main__":
    import tempfile
    _tmp_dir = os.path.join(tempfile.gettempdir(), "gradio_" + os.getenv("USER", "default"))
    os.makedirs(_tmp_dir, exist_ok=True)
    os.environ["GRADIO_TEMP_DIR"] = _tmp_dir
    ui = build_ui()
    ui.launch()
