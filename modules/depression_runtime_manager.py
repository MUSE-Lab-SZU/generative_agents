"""Depression runtime manager.

低侵入的抑郁画像运行态管理模块：
- 默认结构与校验
- 中间态视图构建与 Prompt 渲染
- 医患会话后的三层更新（current_event/short_term/progress_state）
- 全链路执行 LLM patch -> trim -> apply（失败自动回退规则 patch）
"""

from __future__ import annotations

import copy
import json
import os
import re
from string import Template
from typing import Any, Callable, Dict, List, Optional, Tuple

from modules.model.llm_model import create_llm_model
from modules.depression.emotion_inferencer import EmotionInferencer


_PROFILE_TOP_KEYS = {
    "schema_version",
    "enabled",
    "severity",
    "case_config",
    "dialogue_protocol",
    "important_notice",
    "render_order",
    "runtime",
}


_TOP_PATCH_KEYS = {"current_event_patch", "active_overrides_patch", "rationale", "audit"}
_CURRENT_EVENT_PATCH_KEYS = {"wording", "noop", "source", "source_detail"}
_SHORT_TERM_PATCH_KEYS = {"path", "text", "source"}
_PROGRESS_PATCH_KEYS = {"path", "delta", "source"}

_PROMPT_TEMPLATE_DIR = os.path.join("data", "prompts")
_FORCED_LLM_CACHE: Dict[Tuple[str, str, str], Any] = {}
_PROMPT_FILE_MAP = {
    "depr_init_current_event": "depr_init_current_event.txt",
    "depr_current_event": "depr_current_event.txt",
    "depr_short_term": "depr_short_term.txt",
    "depr_long_term": "depr_long_term.txt",
    "depr_chat_block_shell": "depr_chat_block_shell.txt",
    "depr_reflect_block_shell": "depr_reflect_block_shell.txt",
}
_PROMPT_CACHE: Dict[str, Dict[str, Any]] = {}
_EMOTION_INFERENCER = EmotionInferencer()
_PROMPT_DEFAULTS = {
    "depr_init_current_event": (
        "你是抑郁运行态初始化器。仅输出 JSON：{\"wording\":\"...\"}。\n"
        "agent_persona=${agent_persona_json}\n"
        "case_config=${case_config_json}\n"
        "current_event_seed=${current_event_seed_json}\n"
        "要求：只生成 wording，不输出 topic，不输出解释文本。"
    ),
    "depr_current_event": (
        "你是抑郁运行态更新器。仅输出 JSON：{\"wording\":\"...\",\"noop\":false}（其中 noop 为可选字段）。\n"
        "current_event=${current_event_json}\n"
        "active_overrides_summary=${active_overrides_summary_json}\n"
        "chat_summary=${chat_summary_json}\n"
        "chat_transcript=${chat_transcript_json}\n"
        "要求：只更新 wording；必须对 current_event.wording 做微调且保持核心语义一致；与 current_event.topic 同轴并保留其核心对象/事件线索；与原 wording 保持高重叠；不得新增超过一个新事实片段；不得写成明显康复；"
        "禁止输出 topic/dominance/confidence 及其他无关字段。若新旧表达近似一致，可设置 noop=true。"
    ),
    "depr_short_term": (
        "你是 short_term 更新器。仅输出 JSON 数组：[{\"path\":\"...\",\"text\":\"...\"}]。\n"
        "allowed_paths=${allowed_paths_json}\n"
        "merged_case_config=${merged_case_config_json}\n"
        "consult_record=${consult_record_json}\n"
        "chat_transcript=${chat_transcript_json}\n"
        "min_items=${min_items_json}\n"
        "max_items=${max_items_json}\n"
        "要求：每个元素只允许包含 path 与 text；path 必须在 allowed_paths 内；"
        "merged_case_config 为当前有效画像参考；若证据覆盖多个域，尽量覆盖不同 path；"
        "文本必须保守表达变化，不得写成明显康复；证据不足可返回 []；不得输出解释文本或其他字段。"
    ),
    "depr_long_term": (
        "你是 long_term 更新器。仅输出 JSON 数组：[{\"path\":\"...\",\"delta\":0.0}]。\n"
        "trigger_reason=${trigger_reason_json}\n"
        "short_term_recent_window=${short_term_recent_window_json}\n"
        "progress_state_current=${progress_state_current_json}\n"
        "baseline=${baseline_json}\n"
        "max_delta=${max_delta_json}\n"
        "要求：证据不足返回 []；delta 必须在 [-max_delta,+max_delta]。方向语义：progress_state.score 越高代表状态越好，delta>0 表示改善，delta<0 表示恶化。"
    ),
    "depr_chat_block_shell": (
        "<抑郁中间态视图>\n"
        "${static_profile_section}"
        "【病例要点】\n"
        "${merged_case_config_block}\n\n"
        "【当前主导事件】\n"
        "- ${current_event_text}\n\n"
        "${emotion_section}"
        "【对话约束】\n"
        "${dialogue_protocol_block}\n"
        "- 干预可能有效，但变化应缓慢、可反复、非线性\n"
        "- 不要在一次对话中表现为明显康复${important_notice_section}\n"
        "</抑郁中间态视图>"
    ),
    "depr_reflect_block_shell": (
        "<反思约束>\n"
        "- 反思需与病例配置一致\n"
        "- 可体现小幅波动，不写成单调持续改善\n"
        "- 若出现积极变化，需保留残余困难描述\n"
        "- 不得输出与当前主导事件完全无关的核心结论\n"
        "- 当前主导事件：${current_event_text}${important_notice_line}\n"
        "</反思约束>"
    ),
}


def default_profile() -> dict:
    """返回可运行的默认 depression_profile。"""
    return {
        "schema_version": "1.2.0",
        "enabled": False,
        "severity": "mild",
        "case_config": {
            "emotion_experience": {},
            "cognitive_pattern": {},
            "somatic_symptoms": {},
            "social_function": {},
        },
        "dialogue_protocol": [],
        "important_notice": "",
        "render_order": {
            "emotion_experience": [],
            "cognitive_pattern": [],
            "somatic_symptoms": [],
            "social_function": [],
        },
        "runtime": {
            "current_event": {
                "topic": "",
                "wording": "",
                "updated_step": -1,
                "source": "fallback",
            },
            "emotion": {
                "label": "",
                "style": "",
                "intensity": 0.0,
                "volatility_note": "",
                "updated_step": -1,
                "source": "",
                "other_agent": "",
                "relationship": "",
                "applies_to": "chat_only",
            },
            "active_overrides": {
                "short_term": {},
                "progress_state": {},
            },
            "update_meta": {
                "last_eval_step": -1,
                "last_apply_step": -1,
                "last_throttle_reason": "",
                "short_term_changed": False,
                "long_term_changed": False,
                "doctor_chat_count": 0,
                "last_long_term_chat_index": 0,
                "long_term_evidence_score": 0.0,
                "current_event_initialized": False,
                "changed_paths": [],
            },
            "rebase": {
                "pending": False,
                "requested_step": None,
                "applied_step": None,
            },
        },
    }


def ensure_profile(agent_or_cfg: dict) -> dict:
    """确保配置中存在合法 depression_profile，并返回补齐后的 profile。"""
    base = default_profile()
    incoming: Dict[str, Any] = {}

    if isinstance(agent_or_cfg, dict):
        if isinstance(agent_or_cfg.get("depression_profile"), dict):
            incoming = copy.deepcopy(agent_or_cfg["depression_profile"])
        elif any(k in agent_or_cfg for k in _PROFILE_TOP_KEYS):
            incoming = copy.deepcopy(agent_or_cfg)

    profile = _deep_merge(base, incoming)
    _normalize_profile(profile)
    validate_profile_or_raise(profile)

    if isinstance(agent_or_cfg, dict) and "depression_profile" in agent_or_cfg:
        agent_or_cfg["depression_profile"] = profile
    return profile


def _domain_order_from_maps(render_order: Any, case_config: Any) -> List[str]:
    domains: List[str] = []
    if isinstance(render_order, dict):
        for domain in render_order.keys():
            if isinstance(domain, str) and domain.strip():
                d = domain.strip()
                if d not in domains:
                    domains.append(d)
    if isinstance(case_config, dict):
        for domain in case_config.keys():
            if isinstance(domain, str) and domain.strip():
                d = domain.strip()
                if d not in domains:
                    domains.append(d)
    return domains


def validate_profile_or_raise(profile: dict, source: str = "") -> None:
    """校验 depression_profile，不合法时抛 ValueError。"""
    if not isinstance(profile, dict):
        _raise(source, "depression_profile must be dict")

    unknown = set(profile.keys()) - _PROFILE_TOP_KEYS
    if unknown:
        _raise(source, f"unknown top-level keys: {sorted(list(unknown))}")

    if not isinstance(profile.get("schema_version", ""), str):
        _raise(source, "schema_version must be string")
    if not isinstance(profile.get("enabled", False), bool):
        _raise(source, "enabled must be bool")
    if not isinstance(profile.get("severity", ""), str):
        _raise(source, "severity must be string")

    case_config = profile.get("case_config")
    if not isinstance(case_config, dict):
        _raise(source, "case_config must be dict")
    for domain, domain_map in case_config.items():
        if not isinstance(domain, str) or not domain.strip():
            _raise(source, "case_config domain must be non-empty string")
        if not isinstance(domain_map, dict):
            _raise(source, f"case_config.{domain} must be dict")
        for sub_key, sub_val in domain_map.items():
            if not isinstance(sub_key, str):
                _raise(source, f"case_config.{domain} sub-key must be string")
            if not isinstance(sub_val, str):
                _raise(source, f"case_config.{domain}.{sub_key} must be string")

    dialogue_protocol = profile.get("dialogue_protocol", [])
    if not isinstance(dialogue_protocol, list) or not all(isinstance(x, str) for x in dialogue_protocol):
        _raise(source, "dialogue_protocol must be list[str]")

    important_notice = profile.get("important_notice", "")
    if not isinstance(important_notice, str):
        _raise(source, "important_notice must be string")

    render_order = profile.get("render_order")
    if not isinstance(render_order, dict):
        _raise(source, "render_order must be dict")
    for domain, ordered_keys in render_order.items():
        if not isinstance(domain, str) or not domain.strip():
            _raise(source, "render_order domain must be non-empty string")
        if not isinstance(ordered_keys, list) or not all(isinstance(x, str) for x in ordered_keys):
            _raise(source, f"render_order.{domain} must be list[str]")
        if domain not in case_config:
            _raise(source, f"render_order.{domain} missing in case_config")
    for domain in case_config.keys():
        if domain not in render_order:
            _raise(source, f"case_config.{domain} missing in render_order")

    runtime = profile.get("runtime")
    if not isinstance(runtime, dict):
        _raise(source, "runtime must be dict")

    current_event = runtime.get("current_event")
    if not isinstance(current_event, dict):
        _raise(source, "runtime.current_event must be dict")
    if not isinstance(current_event.get("topic", ""), str):
        _raise(source, "runtime.current_event.topic must be string")
    if not isinstance(current_event.get("wording", ""), str):
        _raise(source, "runtime.current_event.wording must be string")
    if not isinstance(current_event.get("updated_step", -1), int):
        _raise(source, "runtime.current_event.updated_step must be int")
    if not isinstance(current_event.get("source", "fallback"), str):
        _raise(source, "runtime.current_event.source must be string")

    emotion = runtime.get("emotion", {})
    if not isinstance(emotion, dict):
        _raise(source, "runtime.emotion must be dict")
    if not isinstance(emotion.get("label", ""), str):
        _raise(source, "runtime.emotion.label must be string")
    if not isinstance(emotion.get("style", ""), str):
        _raise(source, "runtime.emotion.style must be string")
    if not isinstance(emotion.get("intensity", 0.0), (int, float)):
        _raise(source, "runtime.emotion.intensity must be number")
    if not isinstance(emotion.get("volatility_note", ""), str):
        _raise(source, "runtime.emotion.volatility_note must be string")
    if not isinstance(emotion.get("updated_step", -1), int):
        _raise(source, "runtime.emotion.updated_step must be int")
    if not isinstance(emotion.get("source", ""), str):
        _raise(source, "runtime.emotion.source must be string")
    if not isinstance(emotion.get("other_agent", ""), str):
        _raise(source, "runtime.emotion.other_agent must be string")
    if not isinstance(emotion.get("relationship", ""), str):
        _raise(source, "runtime.emotion.relationship must be string")
    if not isinstance(emotion.get("applies_to", "chat_only"), str):
        _raise(source, "runtime.emotion.applies_to must be string")

    active_overrides = runtime.get("active_overrides")
    if not isinstance(active_overrides, dict):
        _raise(source, "runtime.active_overrides must be dict")
    if not isinstance(active_overrides.get("short_term", {}), dict):
        _raise(source, "runtime.active_overrides.short_term must be dict")
    if not isinstance(active_overrides.get("progress_state", {}), dict):
        _raise(source, "runtime.active_overrides.progress_state must be dict")

    for path, item in active_overrides.get("short_term", {}).items():
        if not isinstance(path, str) or not isinstance(item, dict):
            _raise(source, "short_term item must be dict by string path")
        if not isinstance(item.get("text", ""), str):
            _raise(source, f"short_term[{path}].text must be string")
        if not isinstance(item.get("ttl_steps", 0), int):
            _raise(source, f"short_term[{path}].ttl_steps must be int")
        if not isinstance(item.get("updated_step", -1), int):
            _raise(source, f"short_term[{path}].updated_step must be int")
        if not isinstance(item.get("source", "fallback"), str):
            _raise(source, f"short_term[{path}].source must be string")

    for path, item in active_overrides.get("progress_state", {}).items():
        if not isinstance(path, str) or not isinstance(item, dict):
            _raise(source, "progress_state item must be dict by string path")
        if not isinstance(item.get("score", 0.0), (int, float)):
            _raise(source, f"progress_state[{path}].score must be number")
        if not isinstance(item.get("baseline", 0.0), (int, float)):
            _raise(source, f"progress_state[{path}].baseline must be number")
        if not isinstance(item.get("updated_step", -1), int):
            _raise(source, f"progress_state[{path}].updated_step must be int")
        if not isinstance(item.get("source", "fallback"), str):
            _raise(source, f"progress_state[{path}].source must be string")

    update_meta = runtime.get("update_meta")
    if not isinstance(update_meta, dict):
        _raise(source, "runtime.update_meta must be dict")
    if not isinstance(update_meta.get("last_eval_step", -1), int):
        _raise(source, "runtime.update_meta.last_eval_step must be int")
    if not isinstance(update_meta.get("last_apply_step", -1), int):
        _raise(source, "runtime.update_meta.last_apply_step must be int")
    if not isinstance(update_meta.get("last_throttle_reason", ""), str):
        _raise(source, "runtime.update_meta.last_throttle_reason must be string")
    if not isinstance(update_meta.get("doctor_chat_count", 0), int):
        _raise(source, "runtime.update_meta.doctor_chat_count must be int")
    if not isinstance(update_meta.get("last_long_term_chat_index", 0), int):
        _raise(source, "runtime.update_meta.last_long_term_chat_index must be int")
    if not isinstance(update_meta.get("long_term_evidence_score", 0.0), (int, float)):
        _raise(source, "runtime.update_meta.long_term_evidence_score must be number")
    if not isinstance(update_meta.get("current_event_initialized", False), bool):
        _raise(source, "runtime.update_meta.current_event_initialized must be bool")

    rebase = runtime.get("rebase")
    if not isinstance(rebase, dict):
        _raise(source, "runtime.rebase must be dict")
    if not isinstance(rebase.get("pending", False), bool):
        _raise(source, "runtime.rebase.pending must be bool")


def validate_all_agents_or_raise(config: dict, static_loader: Callable[[str], dict]) -> None:
    """启动阶段对所有 agent 做配置级校验，异常直接抛出。"""
    if not isinstance(config, dict):
        raise ValueError("config must be dict")

    base = copy.deepcopy(config.get("agent_base", {}) or {})
    agents = config.get("agents", {}) or {}
    if not isinstance(agents, dict):
        raise ValueError("config.agents must be dict")

    for name, agent_cfg in agents.items():
        merged = copy.deepcopy(base)
        if not isinstance(agent_cfg, dict):
            raise ValueError(f"agent {name} config must be dict")
        cfg_path = agent_cfg.get("config_path", "")
        static_cfg = static_loader(cfg_path) if cfg_path else {}
        merged.update(static_cfg or {})
        merged.update(agent_cfg)
        profile = ensure_profile(merged)
        validate_profile_or_raise(profile, source=str(name))


def render_case_lines(
    case_config: dict,
    render_order: dict,
    source_map: Optional[Dict[str, str]] = None,
) -> List[dict]:
    """把 case_config 依据 render_order 渲染成路径列表。"""
    lines: List[dict] = []
    source_map = source_map or {}
    domains = _domain_order_from_maps(render_order, case_config)
    for domain in domains:
        domain_map = case_config.get(domain, {}) or {}
        if not isinstance(domain_map, dict):
            continue
        ordered_keys = list(render_order.get(domain, []) or []) if isinstance(render_order, dict) else []
        known = set()
        for key in ordered_keys:
            if key in domain_map and isinstance(domain_map.get(key), str):
                path = f"{domain}.{key}"
                lines.append(
                    {
                        "path": path,
                        "text": domain_map[key],
                        "source": source_map.get(path, "case"),
                    }
                )
                known.add(key)
        for key, val in domain_map.items():
            if key in known or not isinstance(val, str):
                continue
            path = f"{domain}.{key}"
            lines.append(
                {
                    "path": path,
                    "text": val,
                    "source": source_map.get(path, "case"),
                }
            )
    return lines


def merge_case_and_runtime(case_config: dict, short_term: dict) -> dict:
    """将 short_term 的文本覆盖映射到 case_config 视图，并保留来源。"""
    merged = copy.deepcopy(case_config)
    source_map: Dict[str, str] = {}

    if isinstance(merged, dict):
        for domain, domain_map in merged.items():
            if not isinstance(domain, str) or not domain.strip() or not isinstance(domain_map, dict):
                continue
            for sub_key in domain_map.keys():
                if isinstance(sub_key, str) and sub_key.strip():
                    source_map[f"{domain}.{sub_key}"] = "case"

    if not isinstance(short_term, dict):
        return {
            "case_config": merged,
            "source_map": source_map,
        }

    for path, item in short_term.items():
        domain, sub_key = _split_path(path)
        if not domain or not sub_key:
            continue
        if domain not in merged or not isinstance(merged.get(domain), dict):
            continue
        text = item.get("text", "") if isinstance(item, dict) else ""
        if isinstance(text, str) and text.strip():
            merged[domain][sub_key] = text.strip()
            source_map[path] = "short_term"
    return {
        "case_config": merged,
        "source_map": source_map,
    }


def build_intermediate_view(
    profile: dict,
    now_step: int,
    stage: str,
    update_cfg: Optional[dict] = None,
    static_profile: Optional[dict] = None,
) -> dict:
    """构建 prompt 注入所需中间态视图。"""
    ensured = ensure_profile(profile)
    runtime = ensured["runtime"]
    switches = _resolve_feature_switch(update_cfg or {})
    short_term = runtime["active_overrides"]["short_term"] if switches["short_term_enabled"] else {}
    merged_payload = merge_case_and_runtime(ensured["case_config"], short_term)
    merged_case = merged_payload.get("case_config", {})
    source_map = merged_payload.get("source_map", {})
    merged_lines = render_case_lines(
        merged_case,
        ensured["render_order"],
        source_map=source_map,
    )
    case_lines = [item for item in merged_lines if item.get("source") == "case"]
    short_term_lines = [item for item in merged_lines if item.get("source") == "short_term"]

    current_event = runtime.get("current_event", {}) or {}
    current_event_text = current_event.get("wording", "")
    current_event_topic = current_event.get("topic", "")
    if not current_event_text:
        current_event_text = "暂无显著主导事件"
    current_emotion = runtime.get("emotion", {}) if isinstance(runtime.get("emotion", {}), dict) else {}

    view = {
        "stage": stage,
        "now_step": int(now_step),
        "enabled": bool(ensured.get("enabled", False)),
        "current_event_topic": current_event_topic,
        "current_event_text": current_event_text,
        "current_event_source": current_event.get("source", "fallback"),
        "merged_case_config": merged_case,
        "merged_case_lines": merged_lines,
        "case_lines": case_lines,
        "short_term_lines": short_term_lines,
        "dialogue_protocol": list(ensured.get("dialogue_protocol", []) or []),
        "important_notice": ensured.get("important_notice", "") or "",
        "static_profile": static_profile if isinstance(static_profile, dict) else {},
        "current_emotion": current_emotion,
        "guardrails": {
            "keep_json_protocol": True,
            "no_severity_in_prompt": True,
            "keep_hard_task_objective": True,
        },
    }
    view["chat_block"] = render_chat_prompt_block(view)
    view["reflect_block"] = render_reflect_prompt_block(view)
    return view


def render_chat_prompt_block(view: dict) -> str:
    """渲染对话链路注入块。"""
    if not bool(view.get("enabled", False)):
        return ""

    merged_case_config = view.get("merged_case_config", {})
    if not isinstance(merged_case_config, dict):
        merged_case_config = {}
    merged_case_config_block = json.dumps(merged_case_config, ensure_ascii=False)
    static_profile_section = _render_static_profile_section(view.get("static_profile", {}))
    emotion_section = _render_emotion_section(view.get("current_emotion", {}))

    dialogue_rows: List[str] = []
    for rule in view.get("dialogue_protocol", []) or []:
        if isinstance(rule, str) and rule.strip():
            dialogue_rows.append(f"- {rule.strip()}")
    dialogue_protocol_block = "\n".join(dialogue_rows).strip()

    notice = view.get("important_notice", "")
    important_notice_section = ""
    if isinstance(notice, str) and notice.strip():
        important_notice_section = "\n\n【重要提示】\n- " + notice.strip()

    prompt = _render_prompt_template(
        "depr_chat_block_shell",
        {
            "static_profile_section": static_profile_section,
            "merged_case_config_block": merged_case_config_block,
            "current_event_text": str(view.get("current_event_text", "暂无显著主导事件") or "暂无显著主导事件"),
            "emotion_section": emotion_section,
            "dialogue_protocol_block": dialogue_protocol_block,
            "important_notice_section": important_notice_section,
        },
    )
    content = str(prompt or "").strip()
    if not content:
        return ""
    return content


def infer_chat_emotion(
    patient_agent: Any,
    profile: dict,
    now_step: int,
    static_profile: Optional[dict] = None,
    other_agent: str = "",
    relationship: str = "",
    conversation_content: str = "",
) -> dict:
    """推断当前对话轮次的说话情绪，并写回 runtime.emotion。"""
    ensured = ensure_profile(profile)
    if not bool(ensured.get("enabled", False)):
        return ensured

    runtime = ensured.get("runtime", {}) if isinstance(ensured.get("runtime", {}), dict) else {}
    short_term = (
        runtime.get("active_overrides", {}).get("short_term", {})
        if isinstance(runtime.get("active_overrides", {}), dict)
        else {}
    )
    if not isinstance(short_term, dict):
        short_term = {}
    merged_case_payload = merge_case_and_runtime(ensured.get("case_config", {}), short_term)
    previous_emotion = runtime.get("emotion", {}) if isinstance(runtime.get("emotion", {}), dict) else {}

    payload = {
        "static_profile": static_profile if isinstance(static_profile, dict) else {},
        "current_event": runtime.get("current_event", {}) if isinstance(runtime.get("current_event", {}), dict) else {},
        "merged_case_config": merged_case_payload.get("case_config", {}),
        "previous_emotion": previous_emotion,
        "other_agent": str(other_agent or ""),
        "relationship": str(relationship or ""),
        "conversation_content": str(conversation_content or ""),
        "now_step": int(now_step),
    }

    inferred = _EMOTION_INFERENCER.infer(
        payload,
        completion_func=lambda prompt: _safe_llm_completion(
            patient_agent,
            {"prompt": prompt},
            caller="depr_emotion_infer",
        ),
    )
    emotion = _normalize_runtime_emotion(inferred)
    emotion["updated_step"] = int(now_step)
    emotion["other_agent"] = str(other_agent or "")
    emotion["relationship"] = str(relationship or "")
    emotion["applies_to"] = "chat_only"

    runtime["emotion"] = emotion
    ensured["runtime"] = runtime
    return ensured


def render_reflect_prompt_block(view: dict) -> str:
    """渲染反思链路注入块。"""
    if not bool(view.get("enabled", False)):
        return ""

    current_event_text = str(view.get("current_event_text", "暂无显著主导事件") or "暂无显著主导事件")
    notice = view.get("important_notice", "")
    important_notice_line = ""
    if isinstance(notice, str) and notice.strip():
        important_notice_line = f"\n- 重要提示：{notice.strip()}"

    prompt = _render_prompt_template(
        "depr_reflect_block_shell",
        {
            "current_event_text": current_event_text,
            "important_notice_line": important_notice_line,
        },
    )
    content = str(prompt or "").strip()
    if not content:
        return ""
    return content


def initialize_current_event_if_needed(
    patient_agent: Any,
    profile: dict,
    now_step: int,
    update_cfg: dict,
) -> dict:
    """初始化 current_event：topic 固化 + wording 首次生成。"""
    ensured = ensure_profile(profile)
    switches = _resolve_feature_switch(update_cfg or {})
    if not bool(switches.get("current_event_update_enabled", True)):
        return ensured

    runtime = ensured["runtime"]
    meta = _ensure_runtime_counters(runtime.get("update_meta", {}))
    runtime["update_meta"] = meta
    current_event = runtime.get("current_event", {}) or {}

    if bool(meta.get("current_event_initialized", False)) and str(current_event.get("topic", "")).strip():
        return ensured

    topic = _derive_topic_from_case_config(
        ensured.get("case_config", {}),
        render_order=ensured.get("render_order", {}),
    )
    payload = {
        "agent_persona": getattr(patient_agent, "name", ""),
        "case_config": ensured.get("case_config", {}),
        "current_event_seed": current_event.get("wording", "") or "近期状态存在波动",
    }
    prompt_payload = _build_init_current_event_prompt(payload)
    llm_resp = _safe_llm_completion(patient_agent, prompt_payload, caller="depr_init_current_event")
    parsed = _parse_llm_patch_json(llm_resp)

    wording = ""
    source = "fallback"
    if isinstance(parsed, dict):
        wording = str(parsed.get("wording", "") or "").strip()
    if wording:
        source = "llm"
    else:
        wording = _fallback_init_wording(ensured)

    current_event["topic"] = topic
    current_event["wording"] = wording or "近期状态存在波动"
    current_event["updated_step"] = int(now_step)
    current_event["source"] = source
    runtime["current_event"] = current_event

    meta["current_event_initialized"] = True
    runtime["update_meta"] = meta
    ensured["runtime"] = runtime
    return ensured


def evaluate_post_chat_update(
    patient_agent: Any,
    chats: list,
    summary: str,
    now_step: int,
    now_time: str,
    update_cfg: dict,
) -> dict:
    """医患会话后的运行态评估入口。"""
    effective_cfg, cfg_adjustments = _normalize_update_cfg_for_acceptance(update_cfg)

    if not bool((effective_cfg or {}).get("enabled", True)):
        return {
            "throttled": True,
            "reason": "depression_update_disabled",
            "audit": {"changed_paths": []},
            "cfg_adjustments": cfg_adjustments,
            "doctor_chat_count": 0,
            "long_term_triggered": False,
            "current_event_wording": "",
        }

    switches = _resolve_feature_switch(effective_cfg)
    short_term_enabled = bool(switches.get("short_term_enabled", True))
    progress_state_enabled = bool(switches.get("progress_state_enabled", True))
    current_event_update_enabled = bool(switches.get("current_event_update_enabled", True))

    profile = ensure_profile(getattr(patient_agent, "depression_profile", {}))
    now_step = int(now_step)
    profile = initialize_current_event_if_needed(
        patient_agent=patient_agent,
        profile=profile,
        now_step=now_step,
        update_cfg=effective_cfg,
    )

    runtime = profile["runtime"]
    meta = _ensure_runtime_counters(runtime.get("update_meta", {}))
    runtime["update_meta"] = meta

    meta["last_eval_step"] = now_step
    meta["doctor_chat_count"] = int(meta.get("doctor_chat_count", 0) or 0) + 1
    doctor_chat_count = int(meta.get("doctor_chat_count", 0) or 0)

    min_interval = int(effective_cfg.get("min_update_interval_steps", 3) or 3)
    last_apply = int(meta.get("last_apply_step", -1) or -1)
    slow_layer_blocked = bool(last_apply >= 0 and (now_step - last_apply) < min_interval)

    if progress_state_enabled:
        _accumulate_long_term_evidence(meta, chats, summary, effective_cfg)
    long_term_triggered = _should_trigger_long_term(meta, effective_cfg) if progress_state_enabled else False
    if slow_layer_blocked:
        long_term_triggered = False

    view = build_intermediate_view(profile, now_step=now_step, stage="chat", update_cfg=effective_cfg)
    allowed_paths = _allowed_paths_from_profile(profile)
    _log_depr(
        patient_agent,
        "[DEPR][ALLOWED_PATHS] patient={} step={} source=render_order count={} sample={}".format(
            getattr(patient_agent, "name", ""),
            int(now_step),
            len(allowed_paths),
            allowed_paths[:8],
        ),
    )
    short_term_current = runtime.get("active_overrides", {}).get("short_term", {}) if short_term_enabled else {}
    if not isinstance(short_term_current, dict):
        short_term_current = {}
    merged_case_payload = merge_case_and_runtime(profile.get("case_config", {}), short_term_current)
    merged_case_config = merged_case_payload.get("case_config", {})
    if not isinstance(merged_case_config, dict):
        merged_case_config = {}
    trigger_ctx = {
        "allowed_paths": allowed_paths,
        "short_term_current": short_term_current,
        "merged_case_config": merged_case_config,
        "progress_state_current": (
            runtime.get("active_overrides", {}).get("progress_state", {}) if progress_state_enabled else {}
        ),
        "doctor_chat_count": doctor_chat_count,
        "long_term_trigger": long_term_triggered,
        "short_term_enabled": short_term_enabled,
        "progress_state_enabled": progress_state_enabled,
        "current_event_update_enabled": current_event_update_enabled,
    }

    llm_patch_attempted = bool(effective_cfg.get("llm_patch_required_on_doctor_chat", True))
    llm_patch = {}
    if llm_patch_attempted:
        llm_patch = propose_patch_with_llm(
            patient_agent=patient_agent,
            view=view,
            chats=chats,
            summary=summary,
            update_cfg=effective_cfg,
            trigger_ctx=trigger_ctx,
        )

    rule_patch = _rule_based_patch(
        view=view,
        chats=chats,
        summary=summary,
        allowed_paths=allowed_paths,
        trigger_long=long_term_triggered,
        update_cfg=effective_cfg,
        short_term_enabled=short_term_enabled,
        progress_state_enabled=progress_state_enabled,
    )
    merged = _merge_patch_candidates(llm_patch, rule_patch)
    merged_active = merged.get("active_overrides_patch", {}) if isinstance(merged.get("active_overrides_patch", {}), dict) else {}
    llm_active = llm_patch.get("active_overrides_patch", {}) if isinstance(llm_patch.get("active_overrides_patch", {}), dict) else {}
    rule_active = rule_patch.get("active_overrides_patch", {}) if isinstance(rule_patch.get("active_overrides_patch", {}), dict) else {}
    llm_short_count = len(llm_active.get("short_term_updates", [])) if isinstance(llm_active.get("short_term_updates", []), list) else 0
    rule_short_count = len(rule_active.get("short_term_updates", [])) if isinstance(rule_active.get("short_term_updates", []), list) else 0
    merged_short_before_trim = merged_active.get("short_term_updates", []) if isinstance(merged_active.get("short_term_updates", []), list) else []
    _log_depr(
        patient_agent,
        "[DEPR][SHORT_TERM_MERGE] patient={} step={} llm_short_count={} rule_short_count={} merged_short_count={} merge_policy=llm_only".format(
            getattr(patient_agent, "name", ""),
            int(now_step),
            int(llm_short_count),
            int(rule_short_count),
            int(len(merged_short_before_trim)),
        ),
    )
    trimmed = clamp_and_validate_patch(
        merged,
        profile,
        effective_cfg,
        trigger_long=long_term_triggered,
        now_step=now_step,
        short_term_enabled=short_term_enabled,
        progress_state_enabled=progress_state_enabled,
    )

    required_short = int(effective_cfg.get("short_term_updates_per_chat", 1) or 1) if short_term_enabled else 0
    trimmed_short_updates = trimmed.get("short_term_updates", []) if isinstance(trimmed.get("short_term_updates", []), list) else []
    if short_term_enabled and len(merged_short_before_trim) > 0 and len(trimmed_short_updates) == 0:
        pre_paths: List[str] = []
        for item in merged_short_before_trim:
            if not isinstance(item, dict):
                continue
            path = str(item.get("path", "") or "").strip()
            if path:
                pre_paths.append(path)
            if len(pre_paths) >= 5:
                break
        _log_depr(
            patient_agent,
            "[DEPR][SHORT_TERM_DIAG] patient={} step={} stage=trim diagnosis=raw_non_empty_json_parsed_but_trimmed_to_zero pre_trim={} post_trim={} pre_paths={}".format(
                getattr(patient_agent, "name", ""),
                int(now_step),
                len(merged_short_before_trim),
                len(trimmed_short_updates),
                pre_paths,
            ),
        )

    short_term_missing_llm = len(trimmed_short_updates) < required_short
    if short_term_missing_llm:
        _log_depr(
            patient_agent,
            "[DEPR][SHORT_TERM_MISSING] patient={} step={} required={} actual={} note=llm_empty_no_autofill".format(
                getattr(patient_agent, "name", ""),
                int(now_step),
                required_short,
                len(trimmed_short_updates),
            ),
        )

    before = copy.deepcopy(profile)
    apply_update(
        profile,
        trimmed,
        now_step,
        now_time,
        effective_cfg,
        trigger_long=long_term_triggered,
    )
    audit = diff_runtime(before, profile)

    meta = _ensure_runtime_counters(profile["runtime"].get("update_meta", {}))
    meta["short_term_enabled"] = bool(short_term_enabled)
    meta["progress_state_enabled"] = bool(progress_state_enabled)
    meta["last_apply_step"] = now_step
    meta["last_throttle_reason"] = "slow_layer_interval" if slow_layer_blocked else ""
    meta["short_term_changed"] = bool(audit.get("short_term_changed", False))
    meta["long_term_changed"] = bool(audit.get("long_term_changed", False))
    meta["changed_paths"] = list(audit.get("changed_paths", []))
    meta["short_term_missing_llm"] = bool(short_term_missing_llm)
    merged_audit = trimmed.get("audit", {}) if isinstance(trimmed.get("audit", {}), dict) else {}
    if "current_event_source_detail" in merged_audit:
        meta["current_event_source_detail"] = str(merged_audit.get("current_event_source_detail", "") or "")
    if long_term_triggered and progress_state_enabled:
        meta["last_long_term_chat_index"] = int(meta.get("doctor_chat_count", 0) or 0)
        meta["long_term_evidence_score"] = 0.0
    profile["runtime"]["update_meta"] = meta

    patient_agent.depression_profile = profile
    return {
        "throttled": False,
        "reason": "",
        "audit": audit,
        "cfg_adjustments": cfg_adjustments,
        "doctor_chat_count": doctor_chat_count,
        "long_term_triggered": bool(long_term_triggered and progress_state_enabled),
        "current_event_wording": (
            profile.get("runtime", {})
            .get("current_event", {})
            .get("wording", "")
        ),
        "llm_patch_attempted": llm_patch_attempted,
        "llm_patch_used": bool(llm_patch),
        "short_term_missing_llm": bool(short_term_missing_llm),
    }


def apply_update(
    profile: dict,
    patch: dict,
    now_step: int,
    now_time: str,
    update_cfg: dict,
    trigger_long: bool = False,
) -> dict:
    """应用裁剪后的 patch 到 runtime。"""
    runtime = profile.get("runtime", {}) or {}
    runtime.setdefault("active_overrides", {})
    runtime["active_overrides"].setdefault("short_term", {})
    runtime["active_overrides"].setdefault("progress_state", {})

    switches = _resolve_feature_switch(update_cfg or {})
    if bool(switches.get("short_term_enabled", True)):
        _apply_short_term_updates(
            runtime,
            patch.get("short_term_updates", []) or [],
            now_step=now_step,
            update_cfg=update_cfg,
        )
    if trigger_long and bool(switches.get("progress_state_enabled", True)):
        _apply_progress_updates(
            runtime,
            patch.get("progress_updates", []) or [],
            now_step=now_step,
            update_cfg=update_cfg,
        )

    if bool(switches.get("current_event_update_enabled", True)):
        _apply_current_event_wording(
            runtime,
            patch.get("current_event_patch", {}) or {},
            now_step=now_step,
        )

    runtime.setdefault("update_meta", {})
    runtime["update_meta"]["last_eval_step"] = int(now_step)
    runtime["update_meta"]["updated_time"] = str(now_time)
    profile["runtime"] = runtime
    return profile


def decay_short_term(profile: dict, now_step: int, update_cfg: Optional[dict] = None) -> None:
    """按 step 衰减 short_term，过期条目自动清理。"""
    switches = _resolve_feature_switch(update_cfg or {})
    if not bool(switches.get("short_term_enabled", True)):
        return

    ensured = ensure_profile(profile)
    short_term = ensured["runtime"]["active_overrides"]["short_term"]
    remove_keys: List[str] = []

    for path, item in short_term.items():
        if not isinstance(item, dict):
            remove_keys.append(path)
            continue
        ttl = int(item.get("ttl_steps", 0) or 0)
        last_decay = int(item.get("last_decay_step", item.get("updated_step", now_step)) or now_step)
        elapsed = max(0, int(now_step) - last_decay)
        next_ttl = ttl - elapsed
        item["ttl_steps"] = next_ttl
        item["last_decay_step"] = int(now_step)
        if next_ttl <= 0:
            remove_keys.append(path)

    for key in remove_keys:
        short_term.pop(key, None)

    if isinstance(profile, dict):
        profile.clear()
        profile.update(ensured)


def propose_patch_with_llm(
    patient_agent: Any,
    view: dict,
    chats: list,
    summary: str,
    update_cfg: Optional[dict] = None,
    trigger_ctx: Optional[dict] = None,
) -> dict:
    """LLM 候选 patch 入口：输出最小 patch。"""
    trigger_ctx = trigger_ctx or {}
    update_cfg = update_cfg or {}
    if not getattr(patient_agent, "llm_available", lambda: False)():
        return {}

    allowed_paths = list(trigger_ctx.get("allowed_paths", []) or [])
    short_term_enabled = bool(trigger_ctx.get("short_term_enabled", True))
    progress_state_enabled = bool(trigger_ctx.get("progress_state_enabled", True))
    current_event_update_enabled = bool(trigger_ctx.get("current_event_update_enabled", True))
    chat_text = _format_chats(chats)
    current_parsed: Any = {}
    if current_event_update_enabled:
        current_event = {
            "topic": view.get("current_event_topic", ""),
            "wording": view.get("current_event_text", ""),
        }
        current_payload = _build_current_event_prompt(
            {
                "current_event": current_event,
                "active_overrides_summary": trigger_ctx.get("short_term_current", {}),
                "chat_summary": summary or "",
                "chat_transcript": chat_text,
            }
        )
        current_raw = _safe_llm_completion(patient_agent, current_payload, caller="depr_current_event")
        current_parsed = _parse_llm_patch_json(current_raw)

    patch: Dict[str, Any] = {
        "current_event_patch": {},
        "active_overrides_patch": {
            "short_term_updates": [],
            "progress_updates": [],
        },
        "rationale": "llm_patch",
    }
    if isinstance(current_parsed, dict):
        if "current_event_patch" in current_parsed and isinstance(current_parsed.get("current_event_patch"), dict):
            patch["current_event_patch"] = current_parsed.get("current_event_patch", {})
        elif "wording" in current_parsed:
            patch["current_event_patch"] = {"wording": current_parsed.get("wording", "")}

    if short_term_enabled:
        short_payload = _build_short_term_prompt(
            {
                "allowed_paths": allowed_paths,
                "merged_case_config": trigger_ctx.get("merged_case_config", {}),
                "consult_record": summary or "",
                "chat_transcript": chat_text,
                "min_items": int(update_cfg.get("short_term_updates_per_chat", 1) or 1),
                "max_items": int(update_cfg.get("short_term_max_items_per_chat", 2) or 2),
            }
        )
        short_raw = _safe_llm_completion(patient_agent, short_payload, caller="depr_short_term")
        short_parsed, short_parse_meta = _parse_llm_patch_json(short_raw, return_meta=True)
        short_candidates: List[dict] = []
        if isinstance(short_parsed, list):
            patch["active_overrides_patch"]["short_term_updates"] = short_parsed
            short_candidates = short_parsed
        elif isinstance(short_parsed, dict):
            patch["active_overrides_patch"]["short_term_updates"] = short_parsed.get("short_term_updates", []) or []
            if isinstance(short_parsed.get("short_term_updates", []), list):
                short_candidates = short_parsed.get("short_term_updates", [])

        short_raw_text = str(short_raw or "")
        short_raw_non_empty = bool(short_raw_text.strip())
        short_parse_meta_text = str(short_parse_meta or "")
        diagnosis = "raw_empty"
        if short_raw_non_empty and short_parse_meta_text == "parse_failed":
            diagnosis = "raw_non_empty_json_parse_failed"
        elif short_raw_non_empty and isinstance(short_parsed, list) and len(short_candidates) == 0:
            diagnosis = "llm_returned_empty_array"
        elif short_raw_non_empty and len(short_candidates) > 0:
            diagnosis = "raw_non_empty_json_parsed_candidates_ready"
        elif short_raw_non_empty:
            diagnosis = "raw_non_empty_json_parsed_candidates_empty"

        _log_depr(
            patient_agent,
            "[DEPR][SHORT_TERM_DIAG] patient={} step={} stage=llm_parse diagnosis={} raw_non_empty={} parse_meta={} parsed_type={} candidates={}".format(
                getattr(patient_agent, "name", ""),
                int(view.get("now_step", 0) or 0),
                diagnosis,
                short_raw_non_empty,
                short_parse_meta_text,
                type(short_parsed).__name__,
                len(short_candidates),
            ),
        )

    if bool(trigger_ctx.get("long_term_trigger", False)) and progress_state_enabled:
        long_payload = _build_long_term_prompt(
            {
                "trigger_reason": "interval_or_threshold",
                "short_term_recent_window": trigger_ctx.get("short_term_current", {}),
                "progress_state_current": trigger_ctx.get("progress_state_current", {}),
                "baseline": 0.5,
                "max_delta": float(update_cfg.get("max_progress_delta_per_update", 0.06) or 0.06),
            }
        )
        long_raw = _safe_llm_completion(patient_agent, long_payload, caller="depr_long_term")
        long_parsed = _parse_llm_patch_json(long_raw)
        if isinstance(long_parsed, list):
            patch["active_overrides_patch"]["progress_updates"] = long_parsed
        elif isinstance(long_parsed, dict):
            patch["active_overrides_patch"]["progress_updates"] = long_parsed.get("progress_updates", []) or []

    return patch


def clamp_and_validate_patch(
    patch: dict,
    profile: dict,
    update_cfg: dict,
    trigger_long: bool = False,
    now_step: int = 0,
    short_term_enabled: bool = True,
    progress_state_enabled: bool = True,
) -> dict:
    """按规则裁剪 patch：白名单、长度、范围、触发门控。"""
    p = _trim_top_keys(patch)
    current_event_patch = p.get("current_event_patch", {}) if isinstance(p.get("current_event_patch", {}), dict) else {}

    active = p.get("active_overrides_patch", {})
    if not isinstance(active, dict):
        active = {}

    short_term_updates = (
        _trim_short_term_updates(
            active.get("short_term_updates", []),
            profile,
            update_cfg,
            now_step=now_step,
        )
        if short_term_enabled
        else []
    )
    progress_updates = (
        _trim_progress_updates(
            active.get("progress_updates", []),
            profile,
            update_cfg,
            now_step=now_step,
        )
        if progress_state_enabled
        else []
    )

    if (not trigger_long) or (not progress_state_enabled):
        progress_updates = []

    return {
        "current_event_patch": current_event_patch,
        "short_term_updates": short_term_updates,
        "progress_updates": progress_updates,
        "rationale": str(p.get("rationale", "") or ""),
        "audit": p.get("audit", {}) if isinstance(p.get("audit", {}), dict) else {},
    }


def try_rebase_case_config(profile: dict, now_step: int, update_cfg: dict) -> Tuple[bool, dict]:
    """rebase 接口预留：默认关闭，满足条件后才执行。"""
    ensured = ensure_profile(profile)
    rebase = ensured["runtime"]["rebase"]
    if not bool(rebase.get("pending", False)):
        return False, ensured
    # Rebase flow is intentionally kept disabled after slimming.
    return False, ensured

def diff_runtime(before_profile: dict, after_profile: dict) -> dict:
    """计算 runtime 层差异摘要。"""
    before = ensure_profile(before_profile)
    after = ensure_profile(after_profile)
    before_rt = before.get("runtime", {})
    after_rt = after.get("runtime", {})

    changed_paths = _diff_paths(before_rt, after_rt, prefix="runtime")
    return {
        "changed_paths": changed_paths,
        "short_term_changed": any("runtime.active_overrides.short_term" in p for p in changed_paths),
        "long_term_changed": any("runtime.active_overrides.progress_state" in p for p in changed_paths),
        "current_event_changed": any("runtime.current_event" in p for p in changed_paths),
    }


def _rule_based_patch(
    view: dict,
    chats: list,
    summary: str,
    allowed_paths: Optional[List[str]] = None,
    trigger_long: bool = False,
    update_cfg: Optional[dict] = None,
    short_term_enabled: bool = True,
    progress_state_enabled: bool = True,
) -> dict:
    """规则 patch：输出最小 patch。"""
    update_cfg = update_cfg or {}
    conversation = "\n".join([f"{name}: {text}" for name, text in (chats or [])])
    context = f"{summary or ''}\n{conversation}"

    base_wording = str(view.get("current_event_text", "") or "").strip()
    if not base_wording:
        base_wording = "近期状态存在波动"
    hint = ""
    if any(k in context for k in ["失眠", "疲惫", "低落", "焦虑", "自责", "回避"]):
        hint = "近期仍有轻微起伏"
    elif any(k in context for k in ["缓和", "稳定", "好转", "轻松"]):
        hint = "较前略稳但仍有波动"

    wording = base_wording
    if hint and hint not in base_wording:
        wording = f"{base_wording}，{hint}"

    targets = _pick_rule_update_targets(context)
    if isinstance(allowed_paths, list) and allowed_paths:
        targets = [x for x in targets if x.get("path", "") in set(allowed_paths)] or targets

    short_term_updates = []
    progress_updates = []
    short_max = int(update_cfg.get("short_term_max_items_per_chat", 2) or 2)
    for target in targets[: max(1, short_max)]:
        if short_term_enabled:
            short_term_updates.append(
                {
                    "path": target["path"],
                    "text": target["text"],
                    "source": "rule",
                }
            )
        if trigger_long and progress_state_enabled:
            progress_updates.append(
                {
                    "path": target["path"],
                    "delta": target["delta"],
                    "source": "rule",
                }
            )

    return {
        "current_event_patch": {
            "wording": wording,
        },
        "active_overrides_patch": {
            "short_term_updates": short_term_updates,
            "progress_updates": progress_updates,
        },
        "rationale": "rule_based_minimal",
    }


def _pick_rule_update_targets(context: str) -> List[dict]:
    text = str(context or "")
    candidates: List[dict] = []

    if any(k in text for k in ["睡", "失眠", "疲惫", "食欲", "入睡"]):
        candidates.append(
            {
                "path": "somatic_symptoms.sleep_appetite",
                "text": "最近睡眠与精力仍有波动，偶尔会感到疲惫。",
                "delta": -0.02,
            }
        )
    if any(k in text for k in ["还好", "不想说", "回避", "掩饰", "谨慎"]):
        candidates.append(
            {
                "path": "emotion_experience.masking",
                "text": "回答时更谨慎，常先说还好再补充细节。",
                "delta": -0.02,
            }
        )
    if any(k in text for k in ["挫折", "自责", "否定", "失败", "没用"]):
        candidates.append(
            {
                "path": "cognitive_pattern.self_evaluation",
                "text": "遇到挫折时仍容易先否定自己，需要更久才能缓过来。",
                "delta": -0.02,
            }
        )
    if any(k in text for k in ["社交", "回避", "见人", "联系"]):
        candidates.append(
            {
                "path": "social_function.interpersonal_withdrawal",
                "text": "社交意愿仍偏低，但会在熟悉关系里维持有限联系。",
                "delta": -0.01,
            }
        )
    if not candidates:
        candidates.append(
            {
                "path": "emotion_experience.masking",
                "text": "状态有小幅波动，表达时仍偏谨慎。",
                "delta": -0.01,
            }
        )

    deduped: List[dict] = []
    seen = set()
    for item in candidates:
        path = item.get("path", "")
        if not path or path in seen:
            continue
        seen.add(path)
        deduped.append(item)
    return deduped


def _merge_patch_candidates(llm_patch: dict, rule_patch: dict) -> dict:
    """合并 LLM 与规则 patch（short_term 仅保留 LLM；current_event 仅保留 LLM）。"""
    llm = _trim_top_keys(llm_patch)
    rule = _trim_top_keys(rule_patch)

    llm_ce = llm.get("current_event_patch", {}) if isinstance(llm.get("current_event_patch"), dict) else {}
    current_event_patch = llm_ce if llm_ce else {}
    current_event_source_detail = ""
    if llm_ce:
        current_event_source_detail = "llm"
    else:
        current_event_source_detail = "llm_empty_no_autofill"
    if isinstance(current_event_patch, dict) and current_event_source_detail:
        current_event_patch = dict(current_event_patch)
        current_event_patch.setdefault("source_detail", current_event_source_detail)

    llm_ao = llm.get("active_overrides_patch", {}) if isinstance(llm.get("active_overrides_patch"), dict) else {}
    rule_ao = rule.get("active_overrides_patch", {}) if isinstance(rule.get("active_overrides_patch"), dict) else {}
    llm_short = llm_ao.get("short_term_updates", []) or []
    llm_progress = llm_ao.get("progress_updates", []) or []
    rule_progress = rule_ao.get("progress_updates", []) or []

    # short_term 禁用规则自动补全，仅保留 LLM 结果（空则本轮不更新）。
    short_updates = list(llm_short)

    progress_updates = list(llm_progress) if llm_progress else list(rule_progress)

    return {
        "current_event_patch": current_event_patch,
        "active_overrides_patch": {
            "short_term_updates": short_updates,
            "progress_updates": progress_updates,
        },
        "rationale": str(llm.get("rationale", "") or rule.get("rationale", "") or ""),
        "audit": {
            "current_event_source_detail": current_event_source_detail,
        },
    }


def _wording_overlap_ratio(old: str, new: str) -> float:
    old_chars = {c for c in str(old or "") if not str(c).isspace()}
    new_chars = {c for c in str(new or "") if not str(c).isspace()}
    if not old_chars:
        return 0.0
    return float(len(old_chars & new_chars)) / float(max(1, len(old_chars)))


def _trim_short_term_updates(raw: Any, profile: dict, update_cfg: dict, now_step: int = 0) -> List[dict]:
    updates = raw if isinstance(raw, list) else []
    allowed_paths = _allowed_paths_from_profile(profile)
    allowed = set(allowed_paths)
    max_items = int(update_cfg.get("short_term_max_items_per_chat", 2) or 2)
    legacy_max = int(update_cfg.get("max_changed_subitems_per_chat", max_items) or max_items)
    max_items = max(1, min(max_items, max(1, legacy_max)))
    text_max_len = int(update_cfg.get("short_term_text_max_len", 120) or 120)
    out: List[dict] = []
    seen = set()
    filtered: List[str] = []

    def _mark(reason: str, path: str = "") -> None:
        if len(filtered) >= 12:
            return
        p = path if path else "<empty>"
        filtered.append(f"{reason}:{p}")

    for idx, item in enumerate(updates):
        if len(out) >= max_items:
            remain = len(updates) - idx
            if remain > 0:
                _mark(f"drop_over_max_items(+{remain})")
            break
        if not isinstance(item, dict):
            _mark("drop_not_dict")
            continue

        path = str(item.get("path", "") or "").strip()
        domain, sub_key = _split_path(path)
        if not domain or not sub_key:
            _mark("drop_invalid_path", path)
            continue
        if allowed and path not in allowed:
            _mark("drop_not_in_allowed", path)
            continue
        if path in seen:
            _mark("drop_duplicate_path", path)
            continue

        text = str(item.get("text", "") or "").strip()
        if not text:
            _mark("drop_empty_text", path)
            continue
        if text_max_len > 0:
            text = text[:text_max_len]
        out.append(
            {
                "path": path,
                "text": text,
                "source": str(item.get("source", "llm") or "llm"),
            }
        )
        seen.add(path)

    validation_passed = True
    for item in out:
        path = str(item.get("path", "") or "").strip()
        if not path or "." not in path:
            validation_passed = False
            break
        if allowed and path not in allowed:
            validation_passed = False
            break

    print(
        "[DEPR][SHORT_TERM_TRIM_CHECK] step={} input={} output={} allowed_count={} max_items={} text_max_len={} validation_passed={}".format(
            int(now_step),
            len(updates),
            len(out),
            len(allowed_paths),
            int(max_items),
            int(text_max_len),
            bool(validation_passed),
        )
    )
    if filtered:
        print(
            "[DEPR][SHORT_TERM_TRIM_FILTERED] step={} details={}".format(
                int(now_step),
                filtered,
            )
        )
    return out


def _trim_progress_updates(raw: Any, profile: dict, update_cfg: dict, now_step: int = 0) -> List[dict]:
    updates = raw if isinstance(raw, list) else []
    allowed_paths = _allowed_paths_from_profile(profile)
    allowed = set(allowed_paths)
    max_items = int(update_cfg.get("short_term_max_items_per_chat", 2) or 2)
    legacy_max = int(update_cfg.get("max_changed_subitems_per_chat", max_items) or max_items)
    max_items = max(1, min(max_items, max(1, legacy_max)))
    max_delta = float(update_cfg.get("max_progress_delta_per_update", 0.06) or 0.06)

    out: List[dict] = []
    seen = set()
    filtered: List[str] = []

    def _mark(reason: str, path: str = "") -> None:
        if len(filtered) >= 12:
            return
        p = path if path else "<empty>"
        filtered.append(f"{reason}:{p}")

    for idx, item in enumerate(updates):
        if len(out) >= max_items:
            remain = len(updates) - idx
            if remain > 0:
                _mark(f"drop_over_max_items(+{remain})")
            break
        if not isinstance(item, dict):
            _mark("drop_not_dict")
            continue

        path = str(item.get("path", "") or "").strip()
        domain, sub_key = _split_path(path)
        if not domain or not sub_key:
            _mark("drop_invalid_path", path)
            continue
        if allowed and path not in allowed:
            _mark("drop_not_in_allowed", path)
            continue
        if path in seen:
            _mark("drop_duplicate_path", path)
            continue

        delta = float(item.get("delta", 0.0) or 0.0)
        delta = max(-max_delta, min(max_delta, delta))
        out.append(
            {
                "path": path,
                "delta": delta,
                "source": str(item.get("source", "llm") or "llm"),
            }
        )
        seen.add(path)

    validation_passed = True
    for item in out:
        path = str(item.get("path", "") or "").strip()
        if not path or "." not in path:
            validation_passed = False
            break
        if allowed and path not in allowed:
            validation_passed = False
            break

    print(
        "[DEPR][PROGRESS_TRIM_CHECK] step={} input={} output={} allowed_count={} max_items={} max_delta={} validation_passed={}".format(
            int(now_step),
            len(updates),
            len(out),
            len(allowed_paths),
            int(max_items),
            float(max_delta),
            bool(validation_passed),
        )
    )
    if filtered:
        print(
            "[DEPR][PROGRESS_TRIM_FILTERED] step={} details={}".format(
                int(now_step),
                filtered,
            )
        )
    return out


def _apply_short_term_updates(runtime: dict, short_term_updates: List[dict], now_step: int, update_cfg: dict) -> None:
    short_term = runtime.get("active_overrides", {}).get("short_term", {})
    if not isinstance(short_term, dict):
        short_term = {}
    default_ttl = int(update_cfg.get("short_term_ttl_steps", 12) or 12)

    for item in short_term_updates or []:
        if not isinstance(item, dict):
            continue
        path = str(item.get("path", "") or "").strip()
        if not path:
            continue
        short_term[path] = {
            "text": str(item.get("text", "") or "").strip(),
            "ttl_steps": int(default_ttl),
            "updated_step": int(now_step),
            "last_decay_step": int(now_step),
            "source": str(item.get("source", "intervention") or "intervention"),
        }

    runtime.setdefault("active_overrides", {})
    runtime["active_overrides"]["short_term"] = short_term


def _smooth_progress_score(old_score: float, delta: float, alpha: float) -> float:
    candidate = old_score + delta
    smoothed = alpha * old_score + (1.0 - alpha) * candidate
    return max(0.0, min(1.0, smoothed))


def _apply_progress_with_baseline(old_score: float, baseline: float, delta: float, alpha: float) -> float:
    score = _smooth_progress_score(old_score, delta, alpha)
    if (old_score - baseline) * delta < 0:
        score = baseline + 0.7 * (score - baseline)
    return max(0.0, min(1.0, score))


def _apply_progress_updates(runtime: dict, progress_updates: List[dict], now_step: int, update_cfg: dict) -> None:
    progress_state = runtime.get("active_overrides", {}).get("progress_state", {})
    if not isinstance(progress_state, dict):
        progress_state = {}
    alpha = float(update_cfg.get("score_smoothing_alpha", 0.7) or 0.7)
    alpha = max(0.0, min(1.0, alpha))
    max_delta = float(update_cfg.get("max_progress_delta_per_update", 0.06) or 0.06)

    for item in progress_updates or []:
        if not isinstance(item, dict):
            continue
        path = str(item.get("path", "") or "").strip()
        if not path:
            continue
        old = progress_state.get(path, {}) if isinstance(progress_state.get(path), dict) else {}
        baseline = float(old.get("baseline", 0.5) or 0.5)
        old_score = float(old.get("score", baseline) or baseline)

        delta = float(item.get("delta", 0.0) or 0.0)
        delta = max(-max_delta, min(max_delta, delta))
        next_score = _apply_progress_with_baseline(old_score, baseline, delta, alpha)

        progress_state[path] = {
            "score": next_score,
            "baseline": baseline,
            "updated_step": int(now_step),
            "source": str(item.get("source", "intervention") or "intervention"),
        }

    runtime.setdefault("active_overrides", {})
    runtime["active_overrides"]["progress_state"] = progress_state


def _apply_current_event_wording(runtime: dict, patch: dict, now_step: int) -> None:
    current = runtime.get("current_event", {})
    if not isinstance(current, dict):
        current = {}
    runtime.setdefault("update_meta", {})
    meta = runtime.get("update_meta", {}) if isinstance(runtime.get("update_meta", {}), dict) else {}
    runtime["update_meta"] = meta

    current.setdefault("topic", "")
    current.setdefault("wording", "")
    current.setdefault("updated_step", -1)
    current.setdefault("source", "fallback")

    old_wording = str(current.get("wording", "") or "")
    wording_raw = (patch or {}).get("wording", "") if isinstance(patch, dict) else ""
    wording = wording_raw if isinstance(wording_raw, str) else ""
    if wording != "":
        explicit_noop = patch.get("noop") if isinstance(patch, dict) else None
        semantic_noop = _is_semantic_noop(old_wording, wording)
        noop = bool(explicit_noop) if isinstance(explicit_noop, bool) else bool(semantic_noop)
        current["wording"] = wording
        current["updated_step"] = int(now_step)
        current["source"] = str((patch or {}).get("source", "intervention") or "intervention")
        current["noop"] = noop
        source_detail = str((patch or {}).get("source_detail", "") or "")
        if source_detail:
            current["source_detail"] = source_detail
        meta["current_event_noop"] = bool(noop)
        if source_detail:
            meta["current_event_source_detail"] = source_detail

    runtime["current_event"] = current


def _ensure_runtime_counters(meta: dict) -> dict:
    if not isinstance(meta, dict):
        meta = {}
    meta.setdefault("last_eval_step", -1)
    meta.setdefault("last_apply_step", -1)
    meta.setdefault("last_throttle_reason", "")
    meta.setdefault("short_term_changed", False)
    meta.setdefault("long_term_changed", False)
    meta.setdefault("doctor_chat_count", 0)
    meta.setdefault("last_long_term_chat_index", 0)
    meta.setdefault("long_term_evidence_score", 0.0)
    meta.setdefault("current_event_initialized", False)
    meta.setdefault("changed_paths", [])
    meta.setdefault("short_term_missing_llm", False)
    meta.setdefault("current_event_noop", False)
    meta.setdefault("current_event_source_detail", "")
    meta.setdefault("short_term_enabled", True)
    meta.setdefault("progress_state_enabled", True)
    return meta


def _should_trigger_long_term(meta: dict, update_cfg: dict) -> bool:
    k = int(update_cfg.get("long_term_update_interval_chats", 3) or 3)
    th = float(update_cfg.get("long_term_update_threshold", 1.0) or 1.0)
    c = int(meta.get("doctor_chat_count", 0) or 0)
    score = float(meta.get("long_term_evidence_score", 0.0) or 0.0)
    last_idx = int(meta.get("last_long_term_chat_index", 0) or 0)
    by_interval = bool(k > 0 and c > 0 and (c - last_idx) >= k)
    by_threshold = bool(score >= th)
    return bool(by_interval or by_threshold)


def _accumulate_long_term_evidence(meta: dict, chats: list, summary: str, update_cfg: dict) -> float:
    text_parts = [str(summary or "")]
    for pair in chats or []:
        if isinstance(pair, (list, tuple)) and len(pair) >= 2:
            text_parts.append(str(pair[1] or ""))
    text = "\n".join(text_parts)

    neg_kws = ["失眠", "疲惫", "自责", "回避", "低落", "焦虑", "无助", "没用"]
    pos_kws = ["好转", "愿意", "支持", "稳定", "缓和"]
    neg_hit = sum(1 for k in neg_kws if k in text)
    pos_hit = sum(1 for k in pos_kws if k in text)
    signal = min(2.0, 0.25 * float(neg_hit) + 0.10 * float(pos_hit) + (0.30 if text.strip() else 0.0))

    alpha = float(update_cfg.get("score_smoothing_alpha", 0.7) or 0.7)
    alpha = max(0.0, min(1.0, alpha))
    old = float(meta.get("long_term_evidence_score", 0.0) or 0.0)
    new_score = alpha * old + (1.0 - alpha) * signal
    meta["long_term_evidence_score"] = float(max(0.0, new_score))
    return float(meta["long_term_evidence_score"])


def _normalize_profile(profile: dict) -> None:
    """补齐缺失字段并修复明显类型问题。"""
    base = default_profile()
    merged = _deep_merge(base, profile)
    profile.clear()
    profile.update(merged)

    case_config = profile.get("case_config", {})
    render_order = profile.get("render_order", {})
    if not isinstance(case_config, dict):
        case_config = {}
    if not isinstance(render_order, dict):
        render_order = {}

    normalized_case: Dict[str, dict] = {}
    for domain, domain_map in case_config.items():
        if not isinstance(domain, str) or not domain.strip():
            continue
        d = domain.strip()
        normalized_case[d] = domain_map if isinstance(domain_map, dict) else {}

    normalized_order: Dict[str, List[str]] = {}
    for domain, ordered_keys in render_order.items():
        if not isinstance(domain, str) or not domain.strip():
            continue
        d = domain.strip()
        if isinstance(ordered_keys, list):
            normalized_order[d] = [str(x).strip() for x in ordered_keys if isinstance(x, str) and str(x).strip()]
        else:
            normalized_order[d] = []

    for domain in list(normalized_order.keys()):
        if domain not in normalized_case:
            normalized_case[domain] = {}
    for domain in list(normalized_case.keys()):
        if domain not in normalized_order:
            normalized_order[domain] = []

    profile["case_config"] = normalized_case
    profile["render_order"] = normalized_order

    runtime = profile.setdefault("runtime", {})
    current_event = runtime.setdefault("current_event", {})
    current_event.setdefault("topic", "")
    current_event.setdefault("wording", "")
    current_event.setdefault("updated_step", -1)
    current_event.setdefault("source", "fallback")
    current_event.pop("dominance", None)
    current_event.pop("confidence", None)

    emotion = runtime.setdefault("emotion", {})
    normalized_emotion = _normalize_runtime_emotion(emotion)
    emotion.clear()
    emotion.update(normalized_emotion)

    active_overrides = runtime.setdefault("active_overrides", {})
    short_term = active_overrides.setdefault("short_term", {})
    progress_state = active_overrides.setdefault("progress_state", {})

    for path in list(short_term.keys()):
        item = short_term.get(path)
        if not isinstance(item, dict):
            short_term.pop(path, None)
            continue
        item["text"] = str(item.get("text", "") or "")
        item.pop("intensity_delta", None)
        item["ttl_steps"] = int(item.get("ttl_steps", 0) or 0)
        item["updated_step"] = int(item.get("updated_step", -1) or -1)
        item["last_decay_step"] = int(item.get("last_decay_step", item.get("updated_step", -1)) or -1)
        item["source"] = str(item.get("source", "fallback") or "fallback")
        item.pop("noop", None)
        item.pop("confidence", None)

    for path in list(progress_state.keys()):
        item = progress_state.get(path)
        if not isinstance(item, dict):
            progress_state.pop(path, None)
            continue
        item["score"] = float(item.get("score", 0.5) or 0.5)
        item["baseline"] = float(item.get("baseline", 0.5) or 0.5)
        item["updated_step"] = int(item.get("updated_step", -1) or -1)
        item["source"] = str(item.get("source", "fallback") or "fallback")
        item.pop("confidence", None)

    meta = _ensure_runtime_counters(runtime.setdefault("update_meta", {}))
    if str(current_event.get("topic", "")).strip() and not meta.get("current_event_initialized", False):
        meta["current_event_initialized"] = True
    runtime["update_meta"] = meta


def _trim_top_keys(patch: dict) -> dict:
    """顶层白名单裁剪，并兼容旧 patch 结构。"""
    out = patch if isinstance(patch, dict) else {}
    normalized = {
        "current_event_patch": {},
        "active_overrides_patch": {
            "short_term_updates": [],
            "progress_updates": [],
        },
        "rationale": "",
        "audit": {},
    }

    if isinstance(out.get("current_event_patch"), dict):
        normalized["current_event_patch"] = {
            k: out["current_event_patch"].get(k)
            for k in _CURRENT_EVENT_PATCH_KEYS
            if k in out["current_event_patch"]
        }
    elif isinstance(out.get("event_candidate"), dict):
        wording = out["event_candidate"].get("wording", "")
        normalized["current_event_patch"] = {"wording": wording}
    elif "wording" in out:
        normalized["current_event_patch"] = {"wording": out.get("wording", "")}

    active = out.get("active_overrides_patch", {})
    if not isinstance(active, dict):
        active = {}
    short_raw = active.get("short_term_updates", out.get("short_term_updates", []))
    progress_raw = active.get("progress_updates", out.get("progress_updates", []))

    if isinstance(short_raw, list):
        normalized["active_overrides_patch"]["short_term_updates"] = short_raw
    if isinstance(progress_raw, list):
        normalized["active_overrides_patch"]["progress_updates"] = progress_raw

    normalized["rationale"] = str(out.get("rationale", "") or "")
    if isinstance(out.get("audit"), dict):
        normalized["audit"] = dict(out.get("audit", {}))
    return normalized


def _is_semantic_noop(old_wording: str, new_wording: str) -> bool:
    old = str(old_wording or "").strip()
    new = str(new_wording or "").strip()
    if not old or not new:
        return False
    if old == new:
        return True
    old_norm = re.sub(r"[\s，,。；;！？!?.]", "", old)
    new_norm = re.sub(r"[\s，,。；;！？!?.]", "", new)
    if old_norm and old_norm == new_norm:
        return True
    ratio = _wording_overlap_ratio(old, new)
    return ratio >= 0.95


def _log_depr(patient_agent: Any, message: str) -> None:
    text = f"========== {message} =========="
    print(text)
    logger = getattr(patient_agent, "logger", None)
    if logger:
        try:
            logger.info(message)
        except Exception:
            pass


def _safe_llm_completion(patient_agent: Any, prompt_payload: dict, caller: str = "depr_patch") -> str:
    llm = None
    route_source = "think_fallback"
    route_reason = "default"
    retry = 2

    route_policy_map = {
        "depr_short_term": {
            "policy_getter": "get_depr_short_term_runtime_policy",
            "default_retry": 2,
            "default_force_forced_llm": True,
            "route_log_tag": "SHORT_TERM",
        },
        "depr_current_event": {
            "policy_getter": "get_depr_current_event_runtime_policy",
            "default_retry": 2,
            "default_force_forced_llm": False,
            "route_log_tag": "CURRENT_EVENT",
        },
        "depr_emotion_infer": {
            "policy_getter": "",
            "default_retry": 1,
            "default_force_forced_llm": False,
            "route_log_tag": "EMOTION",
        },
    }
    route_policy = route_policy_map.get(caller)
    route_log_tag = str((route_policy or {}).get("route_log_tag", "") or "")

    if isinstance(route_policy, dict):
        intervention = getattr(patient_agent, "intervention", None)
        runtime_policy = {
            "retry": int(route_policy.get("default_retry", 2) or 2),
            "force_forced_llm": bool(route_policy.get("default_force_forced_llm", False)),
            "source": "defaults",
        }

        policy_getter = str(route_policy.get("policy_getter", "") or "")
        if intervention and policy_getter and hasattr(intervention, policy_getter):
            try:
                policy = getattr(intervention, policy_getter)()
                if isinstance(policy, dict):
                    runtime_policy.update(policy)
            except Exception as err:
                route_reason = "{}_policy_error:{}".format(caller, err)
        elif route_reason == "default":
            route_reason = "{}_policy_missing".format(caller)

        try:
            retry = max(1, int(runtime_policy.get("retry", route_policy.get("default_retry", 2)) or 2))
        except Exception:
            retry = int(route_policy.get("default_retry", 2) or 2)

        default_force = bool(route_policy.get("default_force_forced_llm", False))
        force_forced_llm = bool(runtime_policy.get("force_forced_llm", default_force))

        forced_cfg = None
        if force_forced_llm:
            if intervention and hasattr(intervention, "get_forced_llm_runtime_config"):
                try:
                    forced_cfg = intervention.get_forced_llm_runtime_config()
                except Exception as err:
                    forced_cfg = None
                    route_reason = f"forced_cfg_error:{err}"

            if isinstance(forced_cfg, dict) and forced_cfg:
                cache_key = (
                    str(forced_cfg.get("provider", "") or "").strip(),
                    str(forced_cfg.get("model", "") or "").strip(),
                    str(forced_cfg.get("base_url", "") or "").strip(),
                )
                try:
                    llm = _FORCED_LLM_CACHE.get(cache_key)
                    if llm is None:
                        llm = create_llm_model(forced_cfg)
                        _FORCED_LLM_CACHE[cache_key] = llm
                    route_source = "forced"
                    route_reason = "forced_cfg_ready"
                except Exception as err:
                    llm = None
                    route_reason = f"forced_init_error:{err}"
            elif route_reason == "default":
                route_reason = "forced_cfg_unavailable"
        else:
            route_reason = "{}_policy_force_default".format(caller)

        route_reason = "{}|{}_retry={}|{}_force_forced_llm={}|{}_policy_source={}".format(
            route_reason,
            caller,
            retry,
            caller,
            force_forced_llm,
            caller,
            runtime_policy.get("source", "defaults"),
        )

    if llm is None:
        llm = getattr(patient_agent, "_llm", None)
        route_source = "think_fallback"
        if route_reason == "default":
            route_reason = "default"
    if llm is None:
        if route_log_tag:
            _log_depr(
                patient_agent,
                "[DEPR][{}_LLM_ROUTE] source=unavailable reason={} caller={}".format(
                    route_log_tag,
                    route_reason,
                    caller,
                ),
            )
        return ""
    prompt = str((prompt_payload or {}).get("prompt", "") or "")
    if not prompt:
        return ""
    if route_log_tag:
        _log_depr(
            patient_agent,
            "[DEPR][{}_LLM_ROUTE] source={} reason={} caller={}".format(
                route_log_tag,
                route_source,
                route_reason,
                caller,
            ),
        )
    try:
        ret = llm.completion(prompt=prompt, retry=retry, caller=caller, failsafe="")
        return str(ret or "")
    except Exception:
        return ""


def _parse_llm_patch_json(raw: str, return_meta: bool = False) -> Any:
    def _ret(value: Any, meta: str) -> Any:
        if return_meta:
            return value, meta
        return value

    text = str(raw or "").strip()
    if not text:
        return _ret({}, "empty_input")

    fenced = re.findall(r"```(?:json)?\s*([\s\S]*?)\s*```", text, flags=re.IGNORECASE)
    candidates = [text] + fenced
    for idx, cand in enumerate(candidates):
        s = str(cand or "").strip()
        if not s:
            continue
        try:
            meta = "full_text_json" if idx == 0 else "fenced_json"
            return _ret(json.loads(s), meta)
        except Exception:
            pass

    left = text.find("{")
    right = text.rfind("}")
    if left >= 0 and right > left:
        snippet = text[left : right + 1]
        try:
            return _ret(json.loads(snippet), "object_snippet_json")
        except Exception:
            pass

    left = text.find("[")
    right = text.rfind("]")
    if left >= 0 and right > left:
        snippet = text[left : right + 1]
        try:
            return _ret(json.loads(snippet), "array_snippet_json")
        except Exception:
            pass
    return _ret({}, "parse_failed")


def _load_prompt_template(name: str) -> str:
    default_text = str(_PROMPT_DEFAULTS.get(name, "") or "")
    filename = str(_PROMPT_FILE_MAP.get(name, "") or "").strip()
    if not filename:
        return default_text

    path = os.path.join(_PROMPT_TEMPLATE_DIR, filename)
    try:
        stat = os.stat(path)
        mtime_ns = int(getattr(stat, "st_mtime_ns", 0) or 0)
        cached = _PROMPT_CACHE.get(path, {}) if isinstance(_PROMPT_CACHE.get(path, {}), dict) else {}
        if (
            cached
            and int(cached.get("mtime_ns", -1) or -1) == mtime_ns
            and isinstance(cached.get("content", ""), str)
            and str(cached.get("content", "") or "").strip()
        ):
            print(f"[DEPR][PROMPT_TEMPLATE_CACHE_HIT] name={name} path={path} mtime_ns={mtime_ns}")
            return str(cached.get("content", "") or "")

        with open(path, "r", encoding="utf-8") as f:
            content = str(f.read() or "")
        print(f"[DEPR][PROMPT_TEMPLATE_RELOAD] name={name} path={path} mtime_ns={mtime_ns}")
        if not content.strip():
            print(f"[DEPR][PROMPT_TEMPLATE_EMPTY] name={name} path={path} fallback=default")
            content = default_text
        _PROMPT_CACHE[path] = {"mtime_ns": mtime_ns, "content": content}
        return content
    except FileNotFoundError:
        print(f"[DEPR][PROMPT_TEMPLATE_MISSING] name={name} path={path} fallback=default")
        return default_text
    except UnicodeDecodeError:
        print(f"[DEPR][PROMPT_TEMPLATE_DECODE_ERROR] name={name} path={path} encoding=utf-8 fallback=default")
        return default_text
    except OSError:
        print(f"[DEPR][PROMPT_TEMPLATE_READ_ERROR] name={name} path={path} fallback=default")
        return default_text


def _render_prompt_template(name: str, variables: Optional[dict] = None) -> str:
    data = variables if isinstance(variables, dict) else {}
    content = _load_prompt_template(name)

    placeholders = set()
    for braced, plain in re.findall(r"\$\{([_a-zA-Z][_a-zA-Z0-9]*)\}|\$([_a-zA-Z][_a-zA-Z0-9]*)", content):
        key = braced or plain
        if key:
            placeholders.add(key)
    missing = [k for k in sorted(placeholders) if k not in data]
    if missing:
        print(f"[DEPR][PROMPT_TEMPLATE_VAR_MISSING] name={name} missing={missing} policy=safe_substitute")

    try:
        return Template(content).safe_substitute(data)
    except Exception:
        fallback = str(_PROMPT_DEFAULTS.get(name, "") or "")
        try:
            return Template(fallback).safe_substitute(data)
        except Exception:
            return fallback


def _build_init_current_event_prompt(payload: dict) -> dict:
    input_contract = {
        "agent_persona": {
            "meaning": "角色稳定人设与行为边界",
            "source": "agent.name / static persona",
            "effect": "决定 wording 语气与表达风格",
            "constraint": "不能为空",
            "on_missing": "fallback_default_persona",
        },
        "case_config": {
            "meaning": "干预目标与阶段限制",
            "source": "profile.case_config",
            "effect": "限定 wording 问题域",
            "constraint": "需为可序列化对象",
            "on_missing": "fallback_default_case_config",
        },
        "current_event_seed": {
            "meaning": "初始化锚点句",
            "source": "profile/runtime seed",
            "effect": "降低首轮漂移",
            "constraint": "短文本",
            "on_missing": "derive_from_persona_and_case",
        },
    }
    prompt = _render_prompt_template(
        "depr_init_current_event",
        {
            "agent_persona_json": json.dumps(payload.get("agent_persona", ""), ensure_ascii=False),
            "case_config_json": json.dumps(payload.get("case_config", {}), ensure_ascii=False),
            "current_event_seed_json": json.dumps(payload.get("current_event_seed", ""), ensure_ascii=False),
        },
    )
    return {
        "prompt": prompt,
        "input_contract": input_contract,
        "missing_strategy": _derive_missing_strategy(input_contract),
    }


def _build_current_event_prompt(payload: dict) -> dict:
    input_contract = {
        "current_event": {
            "meaning": "当前运行态事件快照",
            "source": "runtime.current_event",
            "effect": "决定 wording 延续/修正",
            "constraint": "topic 只读",
            "on_missing": "fallback_current_event_snapshot",
        },
        "active_overrides_summary": {
            "meaning": "短期/长期信号摘要",
            "source": "runtime.active_overrides",
            "effect": "约束 wording 与主信号同向",
            "constraint": "需包含主路径",
            "on_missing": "fallback_short_term_window",
        },
        "chat_summary": {
            "meaning": "本轮摘要",
            "source": "after_chat summary",
            "effect": "确定 wording 焦点",
            "constraint": "覆盖关键事实",
            "on_missing": "extract_from_transcript",
        },
        "chat_transcript": {
            "meaning": "本轮完整原文",
            "source": "after_chat chats",
            "effect": "最终事实依据",
            "constraint": "时序完整",
            "on_missing": "only_conservative_wording_update",
        },
    }
    prompt = _render_prompt_template(
        "depr_current_event",
        {
            "current_event_json": json.dumps(payload.get("current_event", {}), ensure_ascii=False),
            "active_overrides_summary_json": json.dumps(payload.get("active_overrides_summary", {}), ensure_ascii=False),
            "chat_summary_json": json.dumps(payload.get("chat_summary", ""), ensure_ascii=False),
            "chat_transcript_json": json.dumps(payload.get("chat_transcript", ""), ensure_ascii=False),
        },
    )
    return {
        "prompt": prompt,
        "input_contract": input_contract,
        "missing_strategy": _derive_missing_strategy(input_contract),
    }


def _build_short_term_prompt(payload: dict) -> dict:
    input_contract = {
        "allowed_paths": {
            "meaning": "short_term 白名单路径",
            "source": "runtime whitelist",
            "effect": "限制输出 path",
            "constraint": "非空数组",
            "on_missing": "fallback_refresh_existing",
        },
        "merged_case_config": {
            "meaning": "被 short_term 覆盖后的当前有效 case 视图",
            "source": "merge_case_and_runtime(case_config, short_term).case_config",
            "effect": "为 text 生成提供当前锚点",
            "constraint": "可序列化对象",
            "on_missing": "fallback_case_config_only",
        },
        "consult_record": {
            "meaning": "最近咨询记录扁平文本",
            "source": "consult_record",
            "effect": "决定更新主轴",
            "constraint": "可验证事实",
            "on_missing": "derive_from_transcript",
        },
        "chat_transcript": {
            "meaning": "会话原文",
            "source": "chats",
            "effect": "事实依据",
            "constraint": "角色可区分",
            "on_missing": "conservative_update",
        },
        "min_items": {
            "meaning": "最小输出条数",
            "source": "cfg.short_term_updates_per_chat",
            "effect": "每轮必更下限",
            "constraint": ">=1",
            "on_missing": "use_1",
        },
        "max_items": {
            "meaning": "最大输出条数",
            "source": "cfg.short_term_max_items_per_chat",
            "effect": "控制膨胀",
            "constraint": ">=min_items",
            "on_missing": "same_as_min_items",
        },
    }
    prompt = _render_prompt_template(
        "depr_short_term",
        {
            "allowed_paths_json": json.dumps(payload.get("allowed_paths", []), ensure_ascii=False),
            "merged_case_config_json": json.dumps(payload.get("merged_case_config", {}), ensure_ascii=False),
            "consult_record_json": json.dumps(payload.get("consult_record", ""), ensure_ascii=False),
            "chat_transcript_json": json.dumps(payload.get("chat_transcript", ""), ensure_ascii=False),
            "min_items_json": json.dumps(payload.get("min_items", 1), ensure_ascii=False),
            "max_items_json": json.dumps(payload.get("max_items", 2), ensure_ascii=False),
        },
    )
    return {
        "prompt": prompt,
        "input_contract": input_contract,
        "missing_strategy": _derive_missing_strategy(input_contract),
    }


def _build_long_term_prompt(payload: dict) -> dict:
    input_contract = {
        "trigger_reason": {
            "meaning": "long_term 触发原因",
            "source": "trigger calc",
            "effect": "决定增量解释方向",
            "constraint": "枚举文本",
            "on_missing": "return_empty",
        },
        "short_term_recent_window": {
            "meaning": "最近窗口 short_term 证据",
            "source": "runtime.short_term",
            "effect": "决定是否有持续证据",
            "constraint": "时间顺序",
            "on_missing": "return_empty",
        },
        "progress_state_current": {
            "meaning": "当前长期状态",
            "source": "runtime.progress_state",
            "effect": "定义 delta 基准",
            "constraint": "score 可解析",
            "on_missing": "fallback_baseline_view",
        },
        "baseline": {
            "meaning": "长期基线",
            "source": "config/runtime",
            "effect": "回调稳定器",
            "constraint": "[0,1]",
            "on_missing": "use_default_baseline",
        },
        "max_delta": {
            "meaning": "单轮最大变化幅度",
            "source": "cfg.max_progress_delta_per_update",
            "effect": "决定 delta 限幅",
            "constraint": "正数",
            "on_missing": "use_default_max_delta",
        },
    }
    prompt = _render_prompt_template(
        "depr_long_term",
        {
            "trigger_reason_json": json.dumps(payload.get("trigger_reason", ""), ensure_ascii=False),
            "short_term_recent_window_json": json.dumps(payload.get("short_term_recent_window", {}), ensure_ascii=False),
            "progress_state_current_json": json.dumps(payload.get("progress_state_current", {}), ensure_ascii=False),
            "baseline_json": json.dumps(payload.get("baseline", 0.5), ensure_ascii=False),
            "max_delta_json": json.dumps(payload.get("max_delta", 0.06), ensure_ascii=False),
        },
    )
    return {
        "prompt": prompt,
        "input_contract": input_contract,
        "missing_strategy": _derive_missing_strategy(input_contract),
    }


def _derive_missing_strategy(input_contract: dict) -> dict:
    out = {}
    for k, v in (input_contract or {}).items():
        if isinstance(v, dict):
            out[k] = v.get("on_missing", "fallback")
    return out


def _resolve_feature_switch(update_cfg: Optional[dict]) -> dict:
    cfg, _ = _normalize_update_cfg_for_acceptance(update_cfg if isinstance(update_cfg, dict) else {})
    return {
        "short_term_enabled": bool(cfg.get("short_term_enabled", True)),
        "progress_state_enabled": bool(cfg.get("progress_state_enabled", True)),
        "current_event_update_enabled": bool(cfg.get("current_event_update_enabled", True)),
    }


def _allowed_paths_from_profile(profile: dict) -> List[str]:
    case_config = profile.get("case_config", {}) if isinstance(profile, dict) else {}
    render_order = profile.get("render_order", {}) if isinstance(profile, dict) else {}
    paths: List[str] = []
    seen = set()

    if isinstance(render_order, dict):
        for domain, ordered_keys in render_order.items():
            if not isinstance(domain, str) or not domain.strip():
                continue
            domain_key = domain.strip()
            domain_map = case_config.get(domain_key, {}) if isinstance(case_config, dict) else {}
            if not isinstance(domain_map, dict):
                continue
            if not isinstance(ordered_keys, list):
                ordered_keys = []
            for sub_key in ordered_keys:
                if not isinstance(sub_key, str) or not sub_key.strip():
                    continue
                sub_key_norm = sub_key.strip()
                if not isinstance(domain_map.get(sub_key_norm), str):
                    continue
                path = f"{domain_key}.{sub_key_norm}"
                if path not in seen:
                    seen.add(path)
                    paths.append(path)

    if isinstance(case_config, dict):
        for domain, domain_map in case_config.items():
            if not isinstance(domain, str) or not domain.strip() or not isinstance(domain_map, dict):
                continue
            domain_key = domain.strip()
            for sub_key, sub_val in domain_map.items():
                if not isinstance(sub_key, str) or not sub_key.strip():
                    continue
                if not isinstance(sub_val, str):
                    continue
                path = f"{domain_key}.{sub_key.strip()}"
                if path not in seen:
                    seen.add(path)
                    paths.append(path)
    return paths


def _derive_topic_from_case_config(case_config: dict, render_order: Optional[dict] = None) -> str:
    if not isinstance(case_config, dict):
        return "case_overview"
    for domain in _domain_order_from_maps(render_order if isinstance(render_order, dict) else {}, case_config):
        domain_map = case_config.get(domain, {})
        if not isinstance(domain_map, dict):
            continue
        for sub_key, text in domain_map.items():
            if isinstance(sub_key, str) and str(text or "").strip():
                return f"{domain}.{sub_key}"
    return "case_overview"


def _fallback_init_wording(profile: dict) -> str:
    topic = _derive_topic_from_case_config(
        profile.get("case_config", {}),
        render_order=profile.get("render_order", {}),
    )
    text = _path_text_from_case_config(profile.get("case_config", {}), topic)
    if text:
        return text
    return "近期状态存在波动"


def _path_text_from_case_config(case_config: dict, path: str) -> str:
    if not isinstance(case_config, dict):
        return ""
    domain, sub_key = _split_path(path)
    if not domain or not sub_key:
        return ""
    domain_map = case_config.get(domain, {}) if isinstance(case_config.get(domain), dict) else {}
    val = domain_map.get(sub_key, "")
    return str(val or "").strip() if isinstance(val, str) else ""


def _format_chats(chats: list) -> str:
    lines = []
    for item in chats or []:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            lines.append(f"{item[0]}: {item[1]}")
    return "\n".join(lines)


def _stringify_profile_value(value: Any) -> str:
    if isinstance(value, list):
        return "、".join([str(v).strip() for v in value if str(v).strip()])
    if isinstance(value, bool):
        return "是" if value else "否"
    if value is None:
        return ""
    return str(value).strip()


def _render_static_profile_section(static_profile: Any) -> str:
    profile = static_profile if isinstance(static_profile, dict) else {}
    if not profile:
        return ""

    preferred_keys = [
        ("age", "年龄"),
        ("gender", "性别"),
        ("occupation", "职业/身份"),
        ("education", "教育状态"),
        ("residence", "居住情况"),
        ("personality", "性格特征"),
        ("speaking_habit", "说话习惯"),
    ]
    lines: List[str] = []
    for key, label in preferred_keys:
        value = profile.get(key, "")
        if key == "age" and isinstance(value, (int, float)):
            text = "{}岁".format(int(value))
        else:
            text = _stringify_profile_value(value)
        if text:
            lines.append(f"- {label}：{text}")

    if not lines:
        for key, value in profile.items():
            text = _stringify_profile_value(value)
            if text:
                lines.append(f"- {str(key)}：{text}")

    if not lines:
        return ""
    return "【Profile（静态）】\n" + "\n".join(lines) + "\n\n"


def _normalize_runtime_emotion(raw: Any) -> Dict[str, Any]:
    emotion = raw if isinstance(raw, dict) else {}
    intensity = emotion.get("intensity", 0.0)
    try:
        intensity_val = float(intensity)
    except Exception:
        intensity_val = 0.0
    intensity_val = max(0.0, min(1.0, intensity_val))

    return {
        "label": str(emotion.get("label", "") or "").strip(),
        "style": str(emotion.get("style", "") or "").strip(),
        "intensity": round(float(intensity_val), 4),
        "volatility_note": str(emotion.get("volatility_note", "") or "").strip(),
        "updated_step": int(emotion.get("updated_step", -1) or -1),
        "source": str(emotion.get("source", "") or "").strip(),
        "other_agent": str(emotion.get("other_agent", "") or "").strip(),
        "relationship": str(emotion.get("relationship", "") or "").strip(),
        "applies_to": str(emotion.get("applies_to", "chat_only") or "chat_only").strip(),
    }


def _render_emotion_section(raw_emotion: Any) -> str:
    emotion = _normalize_runtime_emotion(raw_emotion)
    if not any(
        [
            emotion.get("label"),
            emotion.get("style"),
            emotion.get("volatility_note"),
        ]
    ):
        return ""

    lines = ["【Emotion（当前说话）】"]
    if emotion.get("label", ""):
        lines.append(f"- 当前情绪：{emotion['label']}")
    if emotion.get("style", ""):
        lines.append(f"- 说话风格：{emotion['style']}")
    lines.append(f"- 情绪强度：{emotion.get('intensity', 0.0):.2f}")
    if emotion.get("volatility_note", ""):
        lines.append(f"- 波动说明：{emotion['volatility_note']}")
    lines.append("- 该 Emotion 只用于当前这轮交流，不代表永久性人格变化。")
    return "\n".join(lines) + "\n\n"


def _deep_merge(dst: Any, src: Any) -> Any:
    if isinstance(dst, dict) and isinstance(src, dict):
        out = copy.deepcopy(dst)
        for key, val in src.items():
            if key in out:
                out[key] = _deep_merge(out[key], val)
            else:
                out[key] = copy.deepcopy(val)
        return out
    return copy.deepcopy(src)


def _split_path(path: str) -> Tuple[str, str]:
    if not isinstance(path, str) or "." not in path:
        return "", ""
    parts = path.split(".", 1)
    return parts[0].strip(), parts[1].strip()


def _diff_paths(before: Any, after: Any, prefix: str = "") -> List[str]:
    if type(before) != type(after):
        return [prefix] if prefix else ["<root>"]

    if isinstance(before, dict):
        changed: List[str] = []
        keys = set(before.keys()) | set(after.keys())
        for key in sorted(keys):
            p = f"{prefix}.{key}" if prefix else str(key)
            if key not in before or key not in after:
                changed.append(p)
                continue
            changed.extend(_diff_paths(before[key], after[key], p))
        return changed

    if isinstance(before, list):
        if before == after:
            return []
        return [prefix] if prefix else ["<root>"]

    if before != after:
        return [prefix] if prefix else ["<root>"]
    return []


def _raise(source: str, reason: str) -> None:
    prefix = f"[{source}] " if source else ""
    raise ValueError(prefix + reason)


def _normalize_update_cfg_for_acceptance(update_cfg: dict) -> Tuple[dict, Dict[str, Dict[str, float]]]:
    """按运行约束归一化参数，保持更新开关与阈值语义稳定。"""
    cfg = copy.deepcopy(update_cfg if isinstance(update_cfg, dict) else {})
    adjustments: Dict[str, Dict[str, float]] = {}

    if "llm_patch_required_on_doctor_chat" not in cfg:
        cfg["llm_patch_required_on_doctor_chat"] = True
    else:
        cfg["llm_patch_required_on_doctor_chat"] = bool(cfg.get("llm_patch_required_on_doctor_chat", True))

    cfg["enabled"] = bool(cfg.get("enabled", True))
    cfg["short_term_enabled"] = bool(cfg.get("short_term_enabled", True))
    cfg["progress_state_enabled"] = bool(cfg.get("progress_state_enabled", True))
    cfg["current_event_update_enabled"] = bool(cfg.get("current_event_update_enabled", True))

    min_interval = int(cfg.get("min_update_interval_steps", 3) or 3)
    cfg["min_update_interval_steps"] = max(0, min_interval)

    short_min = int(cfg.get("short_term_updates_per_chat", 1) or 1)
    cfg["short_term_updates_per_chat"] = max(1, short_min)

    short_max = int(cfg.get("short_term_max_items_per_chat", max(2, cfg["short_term_updates_per_chat"])) or max(2, cfg["short_term_updates_per_chat"]))
    if short_max < cfg["short_term_updates_per_chat"]:
        adjustments["short_term_max_items_per_chat"] = {
            "from": float(short_max),
            "to": float(cfg["short_term_updates_per_chat"]),
        }
        short_max = cfg["short_term_updates_per_chat"]
    cfg["short_term_max_items_per_chat"] = short_max

    long_interval = int(cfg.get("long_term_update_interval_chats", 3) or 3)
    cfg["long_term_update_interval_chats"] = max(1, long_interval)

    long_threshold = float(cfg.get("long_term_update_threshold", 1.0) or 1.0)
    cfg["long_term_update_threshold"] = max(0.0, long_threshold)

    max_delta = float(cfg.get("max_progress_delta_per_update", 0.06) or 0.06)
    cfg["max_progress_delta_per_update"] = max(0.001, abs(max_delta))

    alpha = float(cfg.get("score_smoothing_alpha", 0.7) or 0.7)
    cfg["score_smoothing_alpha"] = max(0.0, min(1.0, alpha))

    cfg["short_term_text_max_len"] = max(20, int(cfg.get("short_term_text_max_len", 120) or 120))

    cfg["short_term_ttl_steps"] = max(1, int(cfg.get("short_term_ttl_steps", 12) or 12))
    cfg["max_changed_subitems_per_chat"] = max(1, int(cfg.get("max_changed_subitems_per_chat", 2) or 2))

    return cfg, adjustments
