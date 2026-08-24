#!/usr/bin/env python3
"""Execute one read-only frozen-checkpoint scale context-ablation job."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
import time
import traceback
import types
from pathlib import Path
from string import Template
from typing import Any, Callable


BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
QUESTIONS_DIR = BASE_DIR / "customization" / "depression_scale_agent" / "questions" / "templates"
SCORING_DIR = BASE_DIR / "customization" / "depression_scale_agent" / "questions" / "scoring_prompts"
MINIMAL_PROMPT_PATH = BASE_DIR / "data" / "prompts" / "depression" / "scale_context_ablation_answer.txt"
DOMAIN_MARKER = "=== 持续症状状态 ==="
STAGE_MARKER = "=== 当前主诉节点层 ==="

CONDITION_SPECS: dict[str, dict[str, Any]] = {
    "full": {
        "label": "当前完整量表输入",
        "persona": True,
        "persona_scope": "current_full_base_desc",
        "prompt_scaffold": "current_generate_chat",
        "domains": True,
        "complaint_node": True,
        "memory": True,
        "relation": True,
        "session_context": True,
        "instant_emotion": True,
    },
    "state_only": {
        "label": "七维持续症状状态 + 必要 persona/题目",
        "persona": True,
        "persona_scope": "stable_required_fields_without_date_plan_currently",
        "prompt_scaffold": "minimal_scale_ablation",
        "domains": True,
        "complaint_node": False,
        "memory": False,
        "relation": False,
        "session_context": False,
        "instant_emotion": False,
    },
    "state_complaint_static": {
        "label": "七维状态 + current complaint node（关闭动态上下文）",
        "persona": True,
        "persona_scope": "stable_required_fields_without_date_plan_currently",
        "prompt_scaffold": "minimal_scale_ablation",
        "domains": True,
        "complaint_node": True,
        "memory": False,
        "relation": False,
        "session_context": False,
        "instant_emotion": False,
    },
    "full_no_state": {
        "label": "完整输入但隐藏七维状态",
        "persona": True,
        "persona_scope": "current_full_base_desc",
        "prompt_scaffold": "current_generate_chat",
        "domains": False,
        "complaint_node": True,
        "memory": True,
        "relation": True,
        "session_context": True,
        "instant_emotion": True,
    },
}

SCALE_SPECS = {
    "PHQ-9": {
        "question_file": "PHQ-9-v2.jsonl",
        "scoring_prompt": "PHQ-9评估提示词.md",
        "score_key": "phq9_scores",
        "total_item_ids": list(range(1, 10)),
    },
    "BDI-II": {
        "question_file": "BDI-II-v2.jsonl",
        "scoring_prompt": "BDI-II评估提示词.md",
        "score_key": "bdi_ii_scores",
        "total_item_ids": list(range(1, 22)),
    },
}


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    return str(value)


def load_questions(scale_name: str) -> list[dict[str, Any]]:
    spec = SCALE_SPECS[scale_name]
    rows: list[dict[str, Any]] = []
    with (QUESTIONS_DIR / spec["question_file"]).open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if line:
                item = json.loads(line)
                if isinstance(item, dict):
                    rows.append(item)
    return rows


def disable_all_writes_and_external_memory(config: dict[str, Any]) -> dict[str, Any]:
    """Disable non-storage side effects; storage itself is redirected to a temp copy."""

    payload = copy.deepcopy(config)
    checkpointing = payload.get("checkpointing")
    if not isinstance(checkpointing, dict):
        checkpointing = {}
        payload["checkpointing"] = checkpointing
    checkpointing["enabled"] = False
    staged_eval = payload.get("staged_eval")
    if not isinstance(staged_eval, dict):
        staged_eval = {}
        payload["staged_eval"] = staged_eval
    staged_eval["enabled"] = False
    agent_base = payload.setdefault("agent_base", {})
    if isinstance(agent_base, dict):
        for key in ("external_memory", "memory_write_control"):
            value = agent_base.get(key)
            if not isinstance(value, dict):
                value = {}
                agent_base[key] = value
            value["enabled"] = False
    agents = payload.get("agents", {})
    if isinstance(agents, dict):
        for agent_config in agents.values():
            if not isinstance(agent_config, dict):
                continue
            for key in ("external_memory", "memory_write_control"):
                value = agent_config.get(key)
                if not isinstance(value, dict):
                    value = {}
                    agent_config[key] = value
                value["enabled"] = False
    intervention = payload.setdefault("intervention", {})
    if isinstance(intervention, dict):
        dynamic = intervention.setdefault("depression_dynamic", {})
        if isinstance(dynamic, dict):
            dynamic["log_enabled"] = False
        for key in ("consult_record", "consult_history"):
            value = intervention.get(key)
            if isinstance(value, dict):
                value["write_enabled"] = False
    payload["_frozen_scale_context_ablation"] = {
        "readonly": True,
        "simulation_disabled": True,
        "external_memory_disabled": True,
    }
    return payload


def rewrite_archive_paths(value: Any, archive_root: Path) -> Any:
    """Relocate absolute `/workspace/.../results/...` references into this archive."""

    if isinstance(value, dict):
        return {key: rewrite_archive_paths(item, archive_root) for key, item in value.items()}
    if isinstance(value, list):
        return [rewrite_archive_paths(item, archive_root) for item in value]
    if not isinstance(value, str) or value.startswith(("http://", "https://")):
        return value
    text = value
    marker = "/results/"
    if marker in text:
        suffix = text.split(marker, 1)[1]
        return str(archive_root / suffix)
    if text == "results":
        return str(archive_root)
    if text.startswith("results/"):
        return str(archive_root / text[len("results/") :])
    return text


def _chat_callback(agent_name: str) -> Callable[[str], str]:
    from modules import utils

    def callback(response: str) -> str:
        text = str(response or "")
        assert "{" in text and "}" in text
        parsed = utils.load_dict("{" + text.split("{", 1)[1].split("}", 1)[0] + "}")
        return str(parsed[agent_name]).replace("\n\n", "\n").strip(" \n\"'“”‘’")

    return callback


def _domain_values(graph_snapshot: dict[str, Any]) -> dict[str, Any]:
    domain_state = graph_snapshot.get("domain_state", {}) if isinstance(graph_snapshot, dict) else {}
    domains = domain_state.get("domains", {}) if isinstance(domain_state, dict) else {}
    result: dict[str, Any] = {}
    for key, value in domains.items() if isinstance(domains, dict) else []:
        if isinstance(value, dict):
            result[str(key)] = {
                field: copy.deepcopy(value.get(field))
                for field in ("initial_value", "value", "last_change", "last_observed_at", "evidence")
                if field in value
            }
    return result


def build_minimal_persona(agent: Any) -> str:
    """Keep stable identity/personality fields, excluding date, plan and ``currently``."""

    config = getattr(getattr(agent, "scratch", None), "config", {})
    config = config if isinstance(config, dict) else {}
    fields = (
        ("姓名", getattr(agent, "name", "")),
        ("年龄", config.get("age", "")),
        ("先天特质", config.get("innate", "")),
        ("后天特质", config.get("learned", "")),
        ("生活习惯", config.get("lifestyle", "")),
    )
    return "\n".join(f"{label}：{value}" for label, value in fields if str(value or "").strip())


def build_static_context_block(agent: Any, condition: str) -> tuple[str, dict[str, Any]]:
    if condition not in {"state_only", "state_complaint_static"}:
        raise ValueError(f"not a static-context condition: {condition}")
    engine = getattr(agent, "depression_dynamic", None)
    if engine is None:
        raise ValueError("target agent has no depression_dynamic engine")
    engine.set_base_prompt(build_minimal_persona(agent))
    current_stage = engine.graph_manager.get_current_stage()
    graph_snapshot = engine.graph_manager.get_graph_snapshot()
    builder = engine.prompt_builder
    base_layer = builder._build_base_layer(engine.base_prompt)
    if condition == "state_only":
        domain_layer = builder._build_domain_state_section(graph_snapshot)
        context_block = builder._combine_layers([base_layer, domain_layer])
    else:
        stage_layer = builder._build_stage_layer(current_stage, graph_snapshot)
        context_block = builder._combine_layers([base_layer, stage_layer])
    components = {
        "persona": engine.base_prompt,
        "domain_state": _domain_values(graph_snapshot),
        "current_complaint_node": copy.deepcopy(current_stage) if condition == "state_complaint_static" else {},
        "relation": "",
        "session_context": {},
        "instant_emotion": {},
        "activated_dynamic_memories": [],
    }
    return context_block, components


def build_minimal_prompt_payload(agent: Any, condition: str, question: str) -> tuple[dict[str, Any], dict[str, Any]]:
    context_block, components = build_static_context_block(agent, condition)
    template = Template(MINIMAL_PROMPT_PATH.read_text(encoding="utf-8"))
    prompt_text = template.safe_substitute(
        agent=agent.name,
        context_block=context_block.strip(),
        question=question.strip(),
    )
    return {
        "prompt": prompt_text,
        "callback": _chat_callback(agent.name),
        "failsafe": "嗯",
    }, components


def _capturing_dynamic_completion(agent: Any, calls: list[dict[str, Any]]) -> Callable[[str], str]:
    def complete(prompt: str) -> str:
        started = time.monotonic()
        output = agent._depression_llm_completion(prompt)
        calls.append(
            {
                "call_type": "instant_emotion_inference",
                "prompt": str(prompt or ""),
                "response": str(output or ""),
                "duration_seconds": round(time.monotonic() - started, 3),
            }
        )
        return output

    return complete


def build_relation(agent: Any, other_name: str) -> tuple[str, dict[str, Any]]:
    payload = agent.scratch.prompt_summarize_relation(agent, other_name)
    prompt_text = str(payload.get("prompt", "") or "")
    started = time.monotonic()
    if not agent.llm_available():
        output = payload.get("failsafe", "")
        raw_responses: list[str] = []
        route = "failsafe"
    else:
        output = agent._llm.completion(**payload, caller="scale_ablation_summarize_relation")
        raw_responses = [str(item) for item in getattr(agent._llm, "meta_responses", [])]
        route = "default_llm"
    return str(output or ""), {
        "call_type": "relation_summary",
        "prompt": prompt_text,
        "raw_responses": raw_responses,
        "output": str(output or ""),
        "route": route,
        "duration_seconds": round(time.monotonic() - started, 3),
    }


def extract_full_prompt_segments(prompt: str, agent_name: str) -> dict[str, str]:
    """Expose the exact local-memory and scene/history text already in the final prompt."""

    text = str(prompt or "")
    memory_marker = f"以下是 {agent_name} 的记忆："
    location_marker = "\n\n当前位置："
    memory_block = ""
    if memory_marker in text:
        remainder = text.split(memory_marker, 1)[1]
        memory_block = remainder.split(location_marker, 1)[0].strip()
    scene_block = ""
    if location_marker in text:
        scene_block = text.split(location_marker, 1)[1].strip()
    return {
        "local_associate_memory_block": memory_block,
        "location_time_history_and_current_scene_block": scene_block,
    }


def build_full_prompt_payload(
    session: Any,
    condition: str,
    question: str,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    from customization.snapshot_ui_service import UserStub

    if condition not in {"full", "full_no_state"}:
        raise ValueError(f"not a full-context condition: {condition}")
    agent = session.agent
    engine = getattr(agent, "depression_dynamic", None)
    if engine is None:
        raise ValueError("target agent has no depression_dynamic engine")
    user = UserStub("用户", agent.get_tile())
    relation, relation_trace = build_relation(agent, user.name)
    chats = [(user.name, question)]
    dynamic_context = agent._build_depression_chat_context(
        other=user,
        relation_summary=relation,
        chats=chats,
    )
    engine.set_base_prompt(agent.scratch._base_desc())
    session_context = engine.context_builder.build_context(
        location=dynamic_context["location"],
        time_of_day=dynamic_context["time_of_day"],
        other_agent=dynamic_context["other_agent"],
        relationship=dynamic_context["relationship"],
        interaction_type=dynamic_context["interaction_type"],
        conversation_content=dynamic_context["conversation_content"],
    )
    current_stage = engine.graph_manager.get_current_stage()
    graph_snapshot = engine.graph_manager.get_graph_snapshot()
    rendered_graph = copy.deepcopy(graph_snapshot)
    if condition == "full_no_state":
        rendered_graph.pop("domain_state", None)
    dynamic_memories = engine.memory_system.prepare_memory_context(
        current_stage,
        session_context,
        dynamic_context["conversation_content"],
    )
    auxiliary_calls = [relation_trace]
    emotion = engine._infer_emotion(
        current_stage=current_stage,
        graph_snapshot=rendered_graph,
        session_context=session_context,
        conversation_content=dynamic_context["conversation_content"],
        completion_func=_capturing_dynamic_completion(agent, auxiliary_calls),
    )
    depression_block = engine.prompt_builder.build_prompt(
        base_prompt=engine.base_prompt,
        current_stage=current_stage,
        graph_snapshot=rendered_graph,
        session_context=session_context,
        activated_memories=dynamic_memories,
        emotion=emotion,
    )
    prompt_payload = agent.scratch.prompt_generate_chat(
        agent,
        user,
        relation,
        chats,
        depression_chat_block=depression_block,
        chat_history_target_name=session.doctor_name or user.name,
        retrieval_profile=session._resolve_local_retrieval_profile_when_external_disabled(),
    )
    final_prompt = str(prompt_payload.get("prompt", "") or "")
    components = {
        "persona": engine.base_prompt,
        "domain_state": _domain_values(graph_snapshot) if condition == "full" else {},
        "hidden_domain_state": _domain_values(graph_snapshot) if condition == "full_no_state" else {},
        "current_complaint_node": copy.deepcopy(current_stage),
        "relation": relation,
        "session_context": copy.deepcopy(session_context),
        "instant_emotion": copy.deepcopy(emotion),
        "activated_dynamic_memories": copy.deepcopy(dynamic_memories),
        "depression_chat_block": depression_block,
        "full_prompt_segments": extract_full_prompt_segments(final_prompt, agent.name),
    }
    return prompt_payload, components, auxiliary_calls


def validate_condition_prompt(condition: str, prompt: str) -> dict[str, Any]:
    has_domains = DOMAIN_MARKER in prompt
    has_stage = STAGE_MARKER in prompt
    expected = CONDITION_SPECS[condition]
    errors: list[str] = []
    if has_domains != bool(expected["domains"]):
        errors.append(f"domain marker expected={expected['domains']} observed={has_domains}")
    if has_stage != bool(expected["complaint_node"]):
        errors.append(f"stage marker expected={expected['complaint_node']} observed={has_stage}")
    if errors:
        raise ValueError(f"condition prompt validation failed ({condition}): {'; '.join(errors)}")
    return {
        "has_domain_marker": has_domains,
        "has_complaint_stage_marker": has_stage,
        "prompt_chars": len(prompt),
        "sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
    }


def complete_answer(session: Any, prompt_payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    trace: dict[str, Any] = {
        "route": "",
        "route_reason": "",
        "raw_responses": [],
        "attempts": [],
        "errors": [],
    }
    started = time.monotonic()
    runtime, reason = session._get_forced_llm_runtime()
    if runtime is not None:
        forced_llm, forced_cfg = runtime
        forced_prompt = dict(prompt_payload)
        forced_prompt["failsafe"] = None
        forced_prompt["retry"] = max(1, int(forced_cfg.get("retry", forced_prompt.get("retry", 2)) or 2))
        forced_prompt["temperature"] = float(forced_cfg.get("temperature", forced_prompt.get("temperature", 0.5)))
        attempt_started = time.monotonic()
        try:
            output = forced_llm.completion(**forced_prompt, caller="frozen_scale_context_ablation")
            forced_raw = [str(item) for item in getattr(forced_llm, "meta_responses", [])]
            trace["raw_responses"] = forced_raw
            trace["attempts"].append(
                {
                    "route": "forced_llm",
                    "raw_responses": forced_raw,
                    "output": str(output or ""),
                    "duration_seconds": round(time.monotonic() - attempt_started, 3),
                }
            )
            if output is not None:
                trace.update(route="forced_llm", route_reason="success")
                trace["duration_seconds"] = round(time.monotonic() - started, 3)
                return str(output), trace
            trace["errors"].append("forced_llm_empty")
        except Exception as exc:
            trace["errors"].append(f"forced_llm_error:{type(exc).__name__}:{exc}")
            trace["attempts"].append(
                {
                    "route": "forced_llm",
                    "error": f"{type(exc).__name__}:{exc}",
                    "duration_seconds": round(time.monotonic() - attempt_started, 3),
                }
            )
    else:
        trace["errors"].append(f"forced_llm_unavailable:{reason}")

    default_prompt = dict(prompt_payload)
    attempt_started = time.monotonic()
    if not session.agent.llm_available():
        output = default_prompt.get("failsafe", "")
        trace.update(route="default_failsafe", route_reason="llm_unavailable")
        default_raw: list[str] = []
    else:
        output = session.agent._llm.completion(
            **default_prompt,
            caller="frozen_scale_context_ablation_default",
        )
        default_raw = [str(item) for item in getattr(session.agent._llm, "meta_responses", [])]
        trace["raw_responses"] = default_raw
        trace.update(route="default_llm", route_reason="forced_fallback")
    trace["attempts"].append(
        {
            "route": trace["route"],
            "raw_responses": default_raw,
            "output": str(output or ""),
            "duration_seconds": round(time.monotonic() - attempt_started, 3),
        }
    )
    trace["duration_seconds"] = round(time.monotonic() - started, 3)
    return str(output or ""), trace


def direct_score(scale_name: str, answered_rows: list[dict[str, Any]]) -> dict[str, Any]:
    from scale_protocol import extract_direct_answer_score

    included = set(SCALE_SPECS[scale_name]["total_item_ids"])
    items: list[dict[str, Any]] = []
    total = 0
    complete = True
    for row in answered_rows:
        item_id = int(row.get("id", len(items) + 1))
        score, status = extract_direct_answer_score(str(row.get("answer", "") or ""))
        in_total = item_id in included
        if in_total and (score is None or status != "direct"):
            complete = False
        if in_total and score is not None and status == "direct":
            total += score
        items.append(
            {
                "item_id": item_id,
                "score": score,
                "parse_status": status,
                "included_in_total": in_total,
                "answer": str(row.get("answer", "") or ""),
            }
        )
    return {
        "scale": scale_name,
        "method": "direct_explicit_score_parser",
        "items": items,
        "total_score": total if complete else None,
        "complete": complete,
    }


def _parse_json_reply(reply: str) -> Any:
    text = str(reply or "").strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    return json.loads(text)


def expert_score(scale_name: str, answered_rows: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    from customization.depression_scale_agent.ExpertLLM import ExpertLLM

    spec = SCALE_SPECS[scale_name]
    system_prompt = (SCORING_DIR / spec["scoring_prompt"]).read_text(encoding="utf-8")
    qa_blocks = []
    for row in answered_rows:
        qa_blocks.append(
            "Q{}: {}\n回答: {}".format(
                row.get("id", "?"),
                row.get("question", ""),
                row.get("answer", ""),
            )
        )
    user_prompt = "\n\n".join(qa_blocks)
    started = time.monotonic()
    reply = ExpertLLM().generate(
        user_prompt,
        system_prompt=system_prompt,
        caller="frozen_scale_context_ablation_score",
    )
    trace = {
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "raw_response": str(reply or ""),
        "duration_seconds": round(time.monotonic() - started, 3),
    }
    try:
        scored = _parse_json_reply(str(reply or ""))
        if not isinstance(scored, dict):
            raise ValueError("expert score response is not an object")
    except Exception as exc:
        scored = {"raw_reply": str(reply or ""), "parse_error": str(exc)}
    return scored, trace


def extract_expert_total(scale_name: str, scored: dict[str, Any]) -> float | None:
    spec = SCALE_SPECS[scale_name]
    items = scored.get(spec["score_key"])
    if isinstance(items, list):
        values = []
        for item in items:
            value = item.get("score") if isinstance(item, dict) else None
            if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value <= 3:
                values = []
                break
            values.append(value)
        if len(values) == len(spec["total_item_ids"]):
            return float(sum(values))
    total = scored.get("total_score")
    if isinstance(total, (int, float)) and not isinstance(total, bool):
        return float(total)
    return None


def public_model_provenance(runtime_config: dict[str, Any], target_agent: str) -> dict[str, Any]:
    allowed = ("provider", "model", "base_url", "temperature", "top_p", "seed", "retry")

    def public(value: Any) -> dict[str, Any]:
        return {
            key: copy.deepcopy(value.get(key))
            for key in allowed
            if isinstance(value, dict) and key in value
        }

    agent_base = runtime_config.get("agent_base", {}) if isinstance(runtime_config, dict) else {}
    agents = runtime_config.get("agents", {}) if isinstance(runtime_config, dict) else {}
    target = agents.get(target_agent, {}) if isinstance(agents, dict) else {}
    base_think = agent_base.get("think", {}) if isinstance(agent_base, dict) else {}
    target_think = target.get("think", {}) if isinstance(target, dict) else {}
    intervention = runtime_config.get("intervention", {}) if isinstance(runtime_config, dict) else {}
    return {
        "agent_base_llm": public(base_think.get("llm", {}) if isinstance(base_think, dict) else {}),
        "target_agent_llm_override": public(target_think.get("llm", {}) if isinstance(target_think, dict) else {}),
        "forced_llm": public(intervention.get("forced_llm", {}) if isinstance(intervention, dict) else {}),
    }


def initialize_session(runtime_config: dict[str, Any], run_name: str, snapshot_name: str) -> Any:
    if "gradio" not in sys.modules:
        sys.modules["gradio"] = types.ModuleType("gradio")
    from customization.depression_scale_agent.app import ChatSession
    from modules.model.llm_model import create_llm_model

    session = ChatSession(
        run_name,
        snapshot_file=snapshot_name,
        runtime_config=runtime_config,
        conversation={},
    )
    target_agent = str(runtime_config.get("_ablation_target_agent", "卡布达") or "卡布达")
    session.agent_name = target_agent
    session.agent = session.game.get_agent(target_agent)
    if session.agent is None:
        raise ValueError(f"target agent not found: {target_agent}")
    if session.agent._llm is None:
        session.agent._llm = create_llm_model(session.agent.think_config["llm"])
    session.agent.set_depression_trace_path("")
    session._sync_agent_chain_switch()
    return session


def run_job(job: dict[str, Any]) -> dict[str, Any]:
    output_dir = Path(str(job["output_dir"])).resolve()
    snapshot_path = Path(str(job["snapshot_path"])).resolve()
    storage_source = Path(str(job["storage_source_root"])).resolve()
    archive_root = Path(str(job["archive_root"])).resolve()
    try:
        output_dir.relative_to((archive_root / "checkpoints").resolve())
    except ValueError:
        pass
    else:
        raise ValueError(f"output_dir must not be inside source checkpoints: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    condition = str(job["condition"])
    if condition not in CONDITION_SPECS:
        raise ValueError(f"unknown condition: {condition}")
    score_mode = str(job.get("score_mode", "both") or "both")
    target_agent = str(job.get("target_agent", "卡布达") or "卡布达")
    snapshot_config = load_json(snapshot_path)
    if not isinstance(snapshot_config, dict):
        raise ValueError(f"snapshot is not an object: {snapshot_path}")
    snapshot_config = rewrite_archive_paths(snapshot_config, archive_root)
    runtime_config = disable_all_writes_and_external_memory(snapshot_config)
    runtime_config["_ablation_target_agent"] = target_agent
    metadata = {
        "schema_version": 1,
        "artifact_kind": "frozen_scale_context_ablation",
        "readonly_replay": True,
        "simulation_rerun": False,
        "run_name": str(job["run_name"]),
        "group": str(job["group"]),
        "outer_repeat": int(job.get("outer_repeat", 0) or 0),
        "ablation_repeat": int(job.get("ablation_repeat", 1) or 1),
        "timepoint": str(job["timepoint"]),
        "source_trigger_label": str(job["source_trigger_label"]),
        "staged_metadata_path": str(job.get("staged_metadata_path", "") or ""),
        "snapshot_name": snapshot_path.name,
        "snapshot_path": str(snapshot_path),
        "snapshot_sha256": sha256_file(snapshot_path),
        "storage_source_root": str(storage_source),
        "storage_reconstruction": "final_storage_temp_copy_pruned_by_snapshot_clock_and_snapshot_memory_ids",
        "target_agent": target_agent,
        "condition": condition,
        "condition_spec": copy.deepcopy(CONDITION_SPECS[condition]),
        "score_mode": score_mode,
        "model_provenance": public_model_provenance(runtime_config, target_agent),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    started = time.monotonic()
    score_totals: dict[str, Any] = {}
    with tempfile.TemporaryDirectory(prefix="storage_", dir=str(output_dir)) as temp_dir:
        temp_storage = Path(temp_dir) / "storage"
        shutil.copytree(storage_source, temp_storage)
        runtime_config["storage_root_override"] = str(temp_storage)
        session = initialize_session(runtime_config, str(job["run_name"]), snapshot_path.name)
        for scale_name in job.get("scales", list(SCALE_SPECS)):
            answered_rows: list[dict[str, Any]] = []
            trace_rows: list[dict[str, Any]] = []
            for question_index, item in enumerate(load_questions(scale_name), start=1):
                question = str(item.get("question", "") or "").strip()
                if condition in {"full", "full_no_state"}:
                    prompt_payload, components, auxiliary_calls = build_full_prompt_payload(
                        session,
                        condition,
                        question,
                    )
                else:
                    prompt_payload, components = build_minimal_prompt_payload(
                        session.agent,
                        condition,
                        question,
                    )
                    auxiliary_calls = []
                final_prompt = str(prompt_payload.get("prompt", "") or "")
                audit = validate_condition_prompt(condition, final_prompt)
                answer, completion_trace = complete_answer(session, prompt_payload)
                answer_row = dict(item)
                answer_row["answer"] = answer
                answered_rows.append(answer_row)
                trace_rows.append(
                    {
                        "schema_version": 1,
                        "scale": scale_name,
                        "index": question_index,
                        "item": copy.deepcopy(item),
                        "question": question,
                        "condition": condition,
                        "condition_spec": copy.deepcopy(CONDITION_SPECS[condition]),
                        "source": {
                            "run_name": str(job["run_name"]),
                            "group": str(job["group"]),
                            "timepoint": str(job["timepoint"]),
                            "snapshot_name": snapshot_path.name,
                            "snapshot_sha256": metadata["snapshot_sha256"],
                        },
                        "components": json_safe(components),
                        "auxiliary_llm_calls": json_safe(auxiliary_calls),
                        "final_prompt": final_prompt,
                        "prompt_audit": audit,
                        "answer": answer,
                        "answer_completion": json_safe(completion_trace),
                    }
                )
            write_jsonl(output_dir / f"{scale_name}_answered.jsonl", answered_rows)
            write_jsonl(output_dir / f"{scale_name}_prompt_trace.jsonl", trace_rows)
            direct = direct_score(scale_name, answered_rows)
            write_json(output_dir / f"{scale_name}_direct_scored.json", direct)
            totals: dict[str, Any] = {"direct": direct.get("total_score")}
            if score_mode in {"expert", "both"}:
                scored, scoring_trace = expert_score(scale_name, answered_rows)
                write_json(output_dir / f"{scale_name}_expert_scored.json", scored)
                write_json(output_dir / f"{scale_name}_expert_scoring_trace.json", scoring_trace)
                totals["expert"] = extract_expert_total(scale_name, scored)
            score_totals[scale_name] = totals
    metadata["score_totals"] = score_totals
    metadata["duration_seconds"] = round(time.monotonic() - started, 3)
    metadata["status"] = "ok"
    metadata["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    write_json(output_dir / "metadata.json", metadata)
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", required=True)
    args = parser.parse_args()
    job_path = Path(args.job).resolve()
    job = load_json(job_path)
    result_path = Path(str(job.get("worker_result_path", job_path.with_name("worker_result.json"))))
    try:
        result = run_job(job)
        write_json(result_path, {"status": "ok", "metadata": result})
        return 0
    except Exception as exc:
        write_json(
            result_path,
            {
                "status": "error",
                "error": str(exc),
                "error_type": type(exc).__name__,
                "traceback": traceback.format_exc(),
            },
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
