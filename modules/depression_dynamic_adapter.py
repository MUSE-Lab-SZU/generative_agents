"""Dynamic depression integration adapter for Agent/Game."""

import datetime
import json
import os
from typing import Any, Dict, Iterable, List, Optional, Set

from modules import utils
from modules.depression import DepressionSimulationEngine
from modules import depression_dynamic_codec as dynamic_codec


DEFAULT_DEPRESSION_TARGET_HINTS = {
    "decide_chat",
    "generate_chat",
    "decide_chat_terminate",
    "decide_wait",
    "summarize_relation",
    "summarize_chats",
    "determine_sector",
    "determine_arena",
    "determine_object",
    "describe_object",
}

DEFAULT_DEPRESSION_PROMPT_CONTEXT_MAPPING = {
    "decide_chat": "闲聊",
    "generate_chat": "深度交流",
    "decide_chat_terminate": "深度交流",
    "decide_wait": "资源等待",
    "summarize_relation": "关系回顾",
    "summarize_chats": "深度交流",
    "determine_sector": "日常活动",
    "determine_arena": "日常活动",
    "determine_object": "日常活动",
    "describe_object": "日常活动",
}

DEFAULT_DEPRESSION_EVENT_INTERACTION_MAPPING = {
    "chat_event": "深度交流",
    "wait_event": "资源等待",
}

KNOWN_DEPRESSION_RELATIONSHIPS = {
    "亲密朋友",
    "普通朋友",
    "治疗师",
    "陌生人",
    "冲突关系",
}

DEFAULT_GLOBAL_CFG = {
    "enabled": False,
    "normal_chain_enabled": True,
    "forced_chain_enabled": True,
    "prompt_injection_enabled": True,
    "event_commit_enabled": True,
    "target_hints": sorted(list(DEFAULT_DEPRESSION_TARGET_HINTS)),
    "on_missing_agent_config": "warn_and_disable",
    "log_enabled": True,
    "persist_enabled": True,
}


def init_runtime(agent: Any, config: Dict[str, Any]) -> None:
    """Initialize dynamic depression runtime for one agent."""
    cfg = _normalize_global_cfg((config or {}).get("depression_dynamic_global", {}))
    agent.depression_dynamic_cfg = cfg
    agent.depression_dynamic_enabled = False
    agent.depression_dynamic_engine = None
    agent.depression_dynamic_target_hints = set()
    agent.depression_dynamic_relationship_mapping = {}
    agent.depression_dynamic_prompt_interaction_mapping = dict(
        DEFAULT_DEPRESSION_PROMPT_CONTEXT_MAPPING
    )
    agent.depression_dynamic_event_interaction_mapping = dict(
        DEFAULT_DEPRESSION_EVENT_INTERACTION_MAPPING
    )

    if not cfg.get("enabled", False):
        _log(agent, "info", "[DEPR_DYNAMIC][INIT] agent={} enabled=false reason=global_disabled".format(agent.name))
        return

    agent_dir = str((config or {}).get("agent_dir", "") or "").strip()
    config_filename = "depression_config.json"
    config_path = os.path.join(agent_dir, config_filename) if agent_dir else ""
    if not config_path or not os.path.exists(config_path):
        _handle_missing_config(agent, cfg, config_path or "<empty>")
        return

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            depression_config = json.load(f)
    except Exception as e:
        _log(
            agent,
            "warning",
            "[DEPR_DYNAMIC][INIT] agent={} enabled=false reason=config_load_failed path={} error={}".format(
                agent.name,
                config_path,
                e,
            ),
        )
        return

    simulation_config = depression_config.get("depression_simulation", {})
    if not isinstance(simulation_config, dict):
        _log(
            agent,
            "warning",
            "[DEPR_DYNAMIC][INIT] agent={} enabled=false reason=invalid_simulation_config path={}".format(
                agent.name,
                config_path,
            ),
        )
        return

    if not bool(simulation_config.get("enabled", False)):
        _log(
            agent,
            "info",
            "[DEPR_DYNAMIC][INIT] agent={} enabled=false reason=simulation_disabled path={}".format(
                agent.name,
                config_path,
            ),
        )
        return

    engine = DepressionSimulationEngine(simulation_config)
    engine.set_clock_provider(_now_from_timer)
    engine.set_base_prompt(_build_base_prompt(agent))

    prompt_map, event_map = _build_interaction_mappings(simulation_config)
    relationship_mapping = _build_relationship_mapping(simulation_config.get("relationship_mapping", {}))
    target_hints = _resolve_target_hints(cfg, simulation_config)

    agent.depression_dynamic_engine = engine
    agent.depression_dynamic_enabled = True
    agent.depression_dynamic_target_hints = target_hints
    agent.depression_dynamic_relationship_mapping = relationship_mapping
    agent.depression_dynamic_prompt_interaction_mapping = prompt_map
    agent.depression_dynamic_event_interaction_mapping = event_map

    load_result = {"loaded": False, "reason": "persist_disabled_or_empty"}
    if cfg.get("persist_enabled", True):
        load_result = dynamic_codec.load_state(
            engine,
            (config or {}).get("depression_dynamic_state"),
            logger=getattr(agent, "logger", None),
        )

    _log(
        agent,
        "info",
        "[DEPR_DYNAMIC][INIT] agent={} enabled=true path={} hints={} loaded={} load_reason={}".format(
            agent.name,
            config_path,
            len(target_hints),
            bool(load_result.get("loaded", False)),
            load_result.get("reason", ""),
        ),
    )


def patch_prompt(
    agent: Any,
    func_hint: str,
    prompt: Any,
    args: Iterable[Any],
    kwargs: Dict[str, Any],
) -> Any:
    """Patch prompt with dynamic depression layer when enabled."""
    if not _runtime_ready(agent):
        return prompt
    cfg = _cfg(agent)
    if not cfg.get("prompt_injection_enabled", True):
        return prompt

    forced = _is_forced_chain(agent)
    if not _chain_enabled(cfg, forced):
        return prompt
    if func_hint not in getattr(agent, "depression_dynamic_target_hints", set()):
        return prompt
    if not isinstance(prompt, dict):
        return prompt

    original_prompt = prompt.get("prompt")
    if not isinstance(original_prompt, str) or not original_prompt.strip():
        return prompt

    try:
        context_info = _build_context(agent, func_hint, args, kwargs)
        layer_text = agent.depression_dynamic_engine.preview_interaction_prompt(
            location=_resolve_current_location(agent),
            time_of_day=_resolve_time_of_day(agent),
            other_agent=context_info.get("other_agent"),
            relationship=context_info.get("relationship"),
            interaction_type=context_info.get("interaction_type"),
            conversation_content=context_info.get("conversation_content", ""),
        )
        if not isinstance(layer_text, str) or not layer_text.strip():
            _log(
                agent,
                "debug",
                "[DEPR_DYNAMIC][PROMPT] agent={} func_hint={} forced={} applied=false reason=empty_layer".format(
                    agent.name,
                    func_hint,
                    forced,
                ),
            )
            return prompt

        patched_prompt = dict(prompt)
        patched_prompt["prompt"] = (
            f"{layer_text}\n\n"
            f"{'=' * 50}\n\n"
            f"=== 当前任务指令层 ===\n{original_prompt}"
        )
        _log(
            agent,
            "info",
            "[DEPR_DYNAMIC][PROMPT] agent={} func_hint={} forced={} applied=true layer_len={} prompt_len_before={} prompt_len_after={}".format(
                agent.name,
                func_hint,
                forced,
                len(layer_text),
                len(original_prompt),
                len(patched_prompt.get("prompt", "")),
            ),
        )
        return patched_prompt
    except Exception as e:
        _log(
            agent,
            "warning",
            "[DEPR_DYNAMIC][ERROR] agent={} phase=patch_prompt func_hint={} error={} fallback=legacy".format(
                agent.name,
                func_hint,
                e,
            ),
        )
        return prompt


def commit_event(
    agent: Any,
    event_key: str,
    forced: bool = False,
    fallback_hint: Optional[str] = None,
    other_agent: Optional[str] = None,
    relationship: Optional[str] = None,
    conversation_content: str = "",
) -> Optional[Dict[str, Any]]:
    """Commit runtime interaction event for dynamic engine."""
    if not _runtime_ready(agent):
        return None
    cfg = _cfg(agent)
    if not cfg.get("event_commit_enabled", True):
        return None
    if not _chain_enabled(cfg, bool(forced)):
        return None

    interaction_type = _get_event_interaction_type(agent, event_key, fallback_hint)
    if not interaction_type:
        return None

    if other_agent and not relationship:
        relationship = _resolve_relationship(agent, "commit_interaction", (), other_agent)

    try:
        result = agent.depression_dynamic_engine.commit_interaction(
            location=_resolve_current_location(agent),
            time_of_day=_resolve_time_of_day(agent),
            other_agent=other_agent,
            relationship=relationship,
            interaction_type=interaction_type,
            conversation_content=conversation_content or "",
        )
        _log(
            agent,
            "info",
            "[DEPR_DYNAMIC][EVENT] agent={} key={} forced={} committed=true transitioned={} state={}".format(
                agent.name,
                event_key,
                bool(forced),
                bool((result or {}).get("transitioned", False)),
                (result or {}).get("current_state", ""),
            ),
        )
        return result
    except Exception as e:
        _log(
            agent,
            "warning",
            "[DEPR_DYNAMIC][ERROR] agent={} phase=commit_event key={} forced={} error={} fallback=legacy".format(
                agent.name,
                event_key,
                bool(forced),
                e,
            ),
        )
        return None


def dump_state(agent: Any) -> Dict[str, Any]:
    """Dump runtime state for checkpoint."""
    cfg = _cfg(agent)
    if not cfg.get("persist_enabled", True):
        return {}
    payload = dynamic_codec.dump_state(
        getattr(agent, "depression_dynamic_engine", None),
        bool(getattr(agent, "depression_dynamic_enabled", False)),
    )
    _log(
        agent,
        "debug",
        "[DEPR_DYNAMIC][SAVE] agent={} enabled={} runtime_keys={}".format(
            getattr(agent, "name", ""),
            bool(payload.get("enabled", False)),
            list((payload.get("runtime", {}) or {}).keys()),
        ),
    )
    return payload


def serialize_conversation(conversation: Any, max_items: int = 8) -> str:
    if not conversation:
        return ""
    try:
        items = list(conversation)[-max_items:]
    except Exception:
        return str(conversation)
    lines: List[str] = []
    for item in items:
        if isinstance(item, tuple) and len(item) >= 2:
            lines.append(f"{item[0]}: {item[1]}")
            continue
        if hasattr(item, "describe"):
            lines.append(str(item.describe))
            continue
        lines.append(str(item))
    return "\n".join([line for line in lines if line])


def serialize_focus(focus: Any, max_items: int = 6) -> str:
    if not isinstance(focus, dict):
        return ""
    lines: List[str] = []
    for key in ("events", "thoughts"):
        items = focus.get(key, [])
        if not items:
            continue
        for item in list(items)[-max_items:]:
            if hasattr(item, "describe"):
                lines.append(str(item.describe))
            else:
                lines.append(str(item))
    return "\n".join([line for line in lines if line])


def build_wait_content(agent: Any, other: Any, focus: Any) -> str:
    other_name = getattr(other, "name", "")
    other_event_desc = ""
    try:
        other_event_desc = other.get_event().get_describe(False)
    except Exception:
        other_event_desc = ""
    return "\n".join(
        [
            item
            for item in [
                serialize_focus(focus),
                "{} 在等待 {} 完成 {}".format(
                    getattr(agent, "name", ""),
                    other_name,
                    other_event_desc,
                ),
            ]
            if item
        ]
    )


def _runtime_ready(agent: Any) -> bool:
    return bool(
        getattr(agent, "depression_dynamic_enabled", False)
        and getattr(agent, "depression_dynamic_engine", None) is not None
    )


def _cfg(agent: Any) -> Dict[str, Any]:
    raw = getattr(agent, "depression_dynamic_cfg", {})
    return raw if isinstance(raw, dict) else dict(DEFAULT_GLOBAL_CFG)


def _normalize_global_cfg(raw_cfg: Any) -> Dict[str, Any]:
    cfg = dict(DEFAULT_GLOBAL_CFG)
    if not isinstance(raw_cfg, dict):
        return cfg
    for key in (
        "enabled",
        "normal_chain_enabled",
        "forced_chain_enabled",
        "prompt_injection_enabled",
        "event_commit_enabled",
        "log_enabled",
        "persist_enabled",
    ):
        if key in raw_cfg:
            cfg[key] = bool(raw_cfg.get(key))
    strategy = str(raw_cfg.get("on_missing_agent_config", cfg["on_missing_agent_config"]) or "").strip()
    if strategy:
        cfg["on_missing_agent_config"] = strategy

    target_hints = raw_cfg.get("target_hints", cfg["target_hints"])
    cleaned_hints: List[str] = []
    if isinstance(target_hints, list):
        for hint in target_hints:
            text = str(hint).strip()
            if text:
                cleaned_hints.append(text)
    if cleaned_hints:
        cfg["target_hints"] = cleaned_hints
    return cfg


def _resolve_target_hints(cfg: Dict[str, Any], simulation_config: Dict[str, Any]) -> Set[str]:
    simulation_hints_raw = simulation_config.get("target_hints", [])
    simulation_hints = []
    if isinstance(simulation_hints_raw, list):
        simulation_hints = [str(item).strip() for item in simulation_hints_raw if str(item).strip()]
    if simulation_hints:
        return set(simulation_hints)

    global_hints_raw = cfg.get("target_hints", [])
    global_hints = []
    if isinstance(global_hints_raw, list):
        global_hints = [str(item).strip() for item in global_hints_raw if str(item).strip()]
    if global_hints:
        return set(global_hints)

    return set(DEFAULT_DEPRESSION_TARGET_HINTS)


def _build_interaction_mappings(simulation_config: Dict[str, Any]) -> (Dict[str, str], Dict[str, str]):
    prompt_mapping = dict(DEFAULT_DEPRESSION_PROMPT_CONTEXT_MAPPING)
    event_mapping = dict(DEFAULT_DEPRESSION_EVENT_INTERACTION_MAPPING)

    _merge_mapping(simulation_config.get("interaction_type_mapping", {}), prompt_mapping)
    _merge_mapping(simulation_config.get("prompt_context_mapping", {}), prompt_mapping)
    _merge_mapping(simulation_config.get("event_interaction_mapping", {}), event_mapping)

    derived_chat_interaction = prompt_mapping.get("generate_chat")
    if derived_chat_interaction:
        event_mapping["chat_event"] = derived_chat_interaction
    derived_wait_interaction = prompt_mapping.get("decide_wait")
    if derived_wait_interaction:
        event_mapping["wait_event"] = derived_wait_interaction
    return prompt_mapping, event_mapping


def _build_relationship_mapping(raw_mapping: Any) -> Dict[str, str]:
    if not isinstance(raw_mapping, dict):
        return {}
    return {str(k): str(v) for k, v in raw_mapping.items()}


def _merge_mapping(raw_mapping: Any, target_mapping: Dict[str, str]) -> None:
    if not isinstance(raw_mapping, dict):
        return
    for key, value in raw_mapping.items():
        normalized_key = str(key).strip()
        normalized_value = str(value).strip()
        if normalized_key and normalized_value:
            target_mapping[normalized_key] = normalized_value


def _handle_missing_config(agent: Any, cfg: Dict[str, Any], config_path: str) -> None:
    strategy = str(cfg.get("on_missing_agent_config", "warn_and_disable") or "")
    if strategy != "warn_and_disable":
        _log(
            agent,
            "warning",
            "[DEPR_DYNAMIC][CFG_MISSING] agent={} path={} strategy={} fallback=warn_and_disable".format(
                agent.name,
                config_path,
                strategy,
            ),
        )
    _log(
        agent,
        "warning",
        "[DEPR_DYNAMIC][CFG_MISSING] agent={} path={} strategy=warn_and_disable dynamic_enabled=false".format(
            agent.name,
            config_path,
        ),
    )


def _build_base_prompt(agent: Any) -> str:
    parts = [f"你是{getattr(agent, 'name', '')}"]
    scratch = getattr(agent, "scratch", None)
    if scratch is None:
        return "".join(parts)

    age = getattr(scratch, "age", None)
    if age is not None:
        parts.append(f"，{age}岁")
    innate = getattr(scratch, "innate", "")
    if innate:
        parts.append(f"。性格特征：{innate}")
    learned = getattr(scratch, "learned", "")
    if learned:
        parts.append(f"\n背景经历：{learned}")
    lifestyle = getattr(scratch, "lifestyle", "")
    if lifestyle:
        parts.append(f"\n生活方式：{lifestyle}")
    return "".join(parts)


def _build_context(agent: Any, func_hint: str, args: Iterable[Any], kwargs: Dict[str, Any]) -> Dict[str, Any]:
    context_info = {
        "interaction_type": _get_prompt_interaction_type(agent, func_hint),
        "conversation_content": _extract_content(agent, func_hint, args, kwargs),
    }
    other_agent = _extract_other_agent(agent, func_hint, args, kwargs)
    if other_agent:
        context_info["other_agent"] = other_agent
        context_info["relationship"] = _resolve_relationship(agent, func_hint, args, other_agent)
    return context_info


def _get_prompt_interaction_type(agent: Any, func_hint: str) -> str:
    mapping = getattr(agent, "depression_dynamic_prompt_interaction_mapping", {})
    if isinstance(mapping, dict):
        text = str(mapping.get(func_hint, "") or "").strip()
        if text:
            return text
    return "日常活动"


def _get_event_interaction_type(agent: Any, event_key: str, fallback_hint: Optional[str]) -> str:
    mapping = getattr(agent, "depression_dynamic_event_interaction_mapping", {})
    if isinstance(mapping, dict):
        text = str(mapping.get(event_key, "") or "").strip()
        if text:
            return text
    if fallback_hint:
        return _get_prompt_interaction_type(agent, fallback_hint)
    return "日常活动"


def _extract_other_agent(agent: Any, func_hint: str, args: Iterable[Any], kwargs: Dict[str, Any]) -> Optional[str]:
    del kwargs
    args = list(args)
    if func_hint in {"generate_chat", "decide_chat", "decide_chat_terminate", "decide_wait"}:
        if len(args) >= 2:
            return getattr(args[1], "name", str(args[1]))
        return None
    if func_hint == "summarize_relation":
        if len(args) >= 2:
            return str(args[1])
        return None
    if func_hint == "summarize_chats":
        if len(args) >= 1:
            return _extract_other_from_chats(agent, args[0])
        return None
    return None


def _extract_content(agent: Any, func_hint: str, args: Iterable[Any], kwargs: Dict[str, Any]) -> str:
    args = list(args)
    kwargs = kwargs if isinstance(kwargs, dict) else {}
    if func_hint == "generate_chat":
        if len(args) >= 4:
            return serialize_conversation(args[3])
        return ""
    if func_hint == "decide_chat":
        parts: List[str] = []
        if len(args) >= 3:
            parts.append(serialize_focus(args[2]))
        if len(args) >= 4:
            parts.append(serialize_conversation(args[3]))
        return "\n".join([item for item in parts if item])
    if func_hint == "decide_chat_terminate":
        if len(args) >= 3:
            return serialize_conversation(args[2])
        return ""
    if func_hint == "summarize_chats":
        if len(args) >= 1:
            return serialize_conversation(args[0])
        return ""
    if func_hint in {"determine_sector", "determine_arena", "determine_object"}:
        descriptions = []
        if len(args) >= 1 and isinstance(args[0], list):
            descriptions = args[0]
        elif isinstance(kwargs.get("describes"), list):
            descriptions = kwargs.get("describes", [])
        return "；".join([str(item) for item in descriptions if item])
    if func_hint == "describe_object":
        if len(args) >= 2:
            return str(args[1])
        return ""
    return ""


def _resolve_relationship(agent: Any, func_hint: str, args: Iterable[Any], other_agent: str) -> str:
    args = list(args)
    explicit_relationship = None
    if func_hint == "generate_chat" and len(args) >= 3:
        explicit_relationship = _normalize_relationship(args[2])
    if explicit_relationship:
        return explicit_relationship

    mapping = getattr(agent, "depression_dynamic_relationship_mapping", {})
    if isinstance(mapping, dict):
        mapped = _normalize_relationship(mapping.get(other_agent))
        if mapped:
            return mapped
    return "普通朋友"


def _normalize_relationship(relationship: Any) -> Optional[str]:
    if relationship is None:
        return None
    rel = str(relationship).strip()
    if not rel:
        return None
    if rel in KNOWN_DEPRESSION_RELATIONSHIPS:
        return rel
    for known in KNOWN_DEPRESSION_RELATIONSHIPS:
        if known in rel:
            return known
    return None


def _extract_other_from_chats(agent: Any, conversation: Any) -> Optional[str]:
    if not isinstance(conversation, list):
        return None
    self_name = getattr(agent, "name", "")
    for item in conversation:
        if isinstance(item, tuple) and len(item) >= 2:
            speaker = item[0]
            if speaker != self_name:
                return str(speaker)
    return None


def _resolve_current_location(agent: Any) -> str:
    tile = None
    try:
        tile = agent.get_tile()
    except Exception:
        tile = None
    if tile is None:
        return "未知"
    try:
        address = tile.get_address()
        if isinstance(address, list) and address:
            return str(address[-1])
    except Exception:
        pass
    return "未知"


def _resolve_time_of_day(agent: Any) -> str:
    del agent
    hour = _now_from_timer().hour
    if 5 <= hour < 12:
        return "morning"
    if 12 <= hour < 18:
        return "afternoon"
    if 18 <= hour < 22:
        return "evening"
    return "night"


def _is_forced_chain(agent: Any) -> bool:
    ctx = getattr(agent, "_chat_route_ctx", None)
    return bool(isinstance(ctx, dict) and ctx.get("forced", False))


def _chain_enabled(cfg: Dict[str, Any], forced: bool) -> bool:
    if forced:
        return bool(cfg.get("forced_chain_enabled", False))
    return bool(cfg.get("normal_chain_enabled", True))


def _now_from_timer() -> datetime.datetime:
    try:
        now_obj = utils.get_timer().get_date()
        if isinstance(now_obj, datetime.datetime):
            return now_obj
    except Exception:
        pass
    return datetime.datetime.now()


def _log(agent: Any, level: str, message: str) -> None:
    cfg = _cfg(agent)
    if not bool(cfg.get("log_enabled", True)):
        return
    logger = getattr(agent, "logger", None)
    if logger is None:
        return
    fn = getattr(logger, level, None)
    if callable(fn):
        try:
            fn(message)
        except Exception:
            return
