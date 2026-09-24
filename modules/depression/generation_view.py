"""Pure, explicit projections. Audit metadata never enters generation payloads."""

import copy
import re
from .evidence import digest

STYLE_KEYS = ("tempo", "disclosure", "tone", "repair_pattern")
EMOTION_KEYS = ("valence", "arousal", "defensiveness", "shame", "hopelessness", "trust")
EXPRESSION_KEYS = (
    "label",
    "style",
    "intensity",
    "disclosure_level",
    "defensiveness",
    "volatility_note",
)
FLAGS = (
    "is_help_seeking_frame",
    "is_evaluative_frame",
    "is_close_relationship",
    "is_professional_frame",
    "is_minimizing",
    "is_withdrawing",
)


def project(value, keys):
    return {
        k: copy.deepcopy(value[k])
        for k in keys
        if isinstance(value, dict) and k in value
    }


def context_view(context):
    context = context if isinstance(context, dict) else {}
    return {
        "scene": project(
            context.get("scene"), ("location", "time_of_day", "interaction_type")
        ),
        "participants": project(
            context.get("participants"), ("self_name", "other_agent", "relationship")
        ),
        "conversation": project(context.get("conversation"), ("content",)),
        "session_flags": {
            k: context.get("session_flags", {}).get(k) is True for k in FLAGS
        },
    }


def select_relevant_claims(
    task, active_claims, observations=None, proposal=None, context=None
):
    """Read-only selection; unselected heads are always inherited by code commit."""
    topic = (context or {}).get("topic_id")
    refs = set()
    if proposal:
        raw_refs = list(proposal.get("admission", {}).get("baseline_claim_refs", []))
        raw_refs += [
            d["previous_claim_ref"]
            for d in proposal.get("claim_diffs", [])
            if d.get("previous_claim_ref")
        ]
        raw_refs += proposal.get("focus_claim_refs", [])
        refs = {(r.get("claim_id"), r.get("revision")) for r in raw_refs}
    query = " ".join(o.get("reported_content", "") for o in observations or [])
    query += " " + str((context or {}).get("query", ""))
    def units(text):
        for phrase in ("睡不着", "失眠", "入睡困难", "难以入睡"):
            text = text.replace(phrase, "睡眠障碍")
        text = re.sub(r"[^\w\u4e00-\u9fff]", "", text.lower())
        return {text[i:i+2] for i in range(len(text)-1)} - {
            "就是", "然后", "觉得", "有点", "没有", "不是", "这个", "那个",
            "一下", "还是", "可能", "现在", "刚才", "时候", "一会", "会儿", "一点", "自己"}
    q = units(query)
    ranked = []
    subjects = {o.get("subject") for o in observations or []}
    for index, c in enumerate(active_claims):
        score = len(q & units(c["text"]))
        if task in {"planner", "validator"}:
            symptom_match = ("睡眠" in query and "睡眠障碍" in
                             re.sub("睡不着|失眠|入睡困难|难以入睡", "睡眠障碍", c["text"]))
            eligible = (score >= 2 or ("纠正" in query and score >= 1) or symptom_match)
            eligible = eligible and c["subject"] in subjects
        else:
            eligible = (score >= 2 or not q) and (not topic or c["topic_id"] == topic
                                                   or c.get("actuality") == "ongoing")
        if eligible or (c["claim_id"], c["revision"]) in refs:
            ranked.append((score, index, c))
    limit = 6 if task in {"planner", "validator"} else 3
    chosen = sorted(ranked, key=lambda row: (row[0], row[1]), reverse=True)[:limit]
    # Required revision/baseline/focus references cannot be dropped by the cap.
    indexes = {row[1] for row in chosen}
    chosen.extend(row for row in ranked if row[1] not in indexes
                  and (row[2]["claim_id"], row[2]["revision"]) in refs)
    return copy.deepcopy([row[2] for row in sorted(chosen, key=lambda row: row[1])])


def validator_source_evidence(ledger, claims):
    """Project configured sources; keep the full raw configuration audit-only."""
    records = []
    for ref in sorted({r for c in claims for r in c["evidence_refs"]}):
        source = ledger[ref]
        if source.get("record_kind") != "persona_fragment":
            records.append(copy.deepcopy(source))
            continue
        record = project(
            source,
            ("evidence_id", "record_kind", "case_id", "config_fingerprint",
             "config_path", "json_pointer", "captured_at"),
        )
        record["raw_fragment"] = {
            "accepted_claims": [
                {**claim_view(c), **project(c, ("claim_id", "revision", "topic_id"))}
                for c in claims
                if ref in c["evidence_refs"] and c.get("kind") != "unknown"
            ]
        }
        records.append(record)
    return records


def semantic_claim_view(c):
    result = project(
        c,
        (
            "text",
            "kind",
            "subject",
            "assertion_status",
            "actuality",
            "acceptance_basis",
            "report_context",
        ),
    )
    result["time"] = project(
        c.get("time"),
        ("reported_time_text", "effective_start", "effective_end", "precision"),
    )
    return result


def claim_view(c):
    result = project(
        c,
        (
            "text",
            "kind",
            "subject",
            "assertion_status",
            "actuality",
            "acceptance_basis",
        ),
    )
    result["time"] = project(
        c.get("time"),
        ("reported_time_text", "effective_start", "effective_end", "precision"),
    )
    result["report_context"] = (
        "显式初始化设定"
        if c["acceptance_basis"] == "configured"
        else "此前报告，保留原时点，不代表本轮再次确认"
    )
    return result


def expression_stage(stage):
    return {
        "speaking_style": project(stage.get("speaking_style"), STYLE_KEYS),
        "emotion_vector": project(stage.get("emotion_vector"), EMOTION_KEYS),
    }


def build_patient_generation_view(manager, context=None):
    node = manager.get_current_stage()
    resolved = manager.resolve_active_claims()
    claims = []
    stage = expression_stage(node)
    stage.update(
        topic=manager.topic_registry[node["topic_id"]]["neutral_name"],
        current_claims=[claim_view(c) for c in claims],
        core_belief=manager.core_belief,
        initialization_background=[
            c["text"] for c in resolved["initialization_background"]
        ],
    )
    stage.update(manager.complaint_stage_view())
    stage["root_complaint_anchor"] = manager.root_complaint_anchor
    recent = []
    for event in reversed(manager.stage_history):
        trace = manager.transition_traces.get(event.get("event_id"), {})
        if trace.get("gate_result") == "commit" and trace.get("verified_change"):
            recent.append(trace["verified_change"])
        if len(recent) == 3:
            break
    stage["recent_experiences"] = recent
    # Only the currently applicable relationship adjustment crosses this boundary.
    relationship = (context or {}).get("participants", {}).get("relationship", "")
    stage["relation_modifiers"] = {
        relationship: project(
            node.get("relation_modifiers", {}).get(relationship),
            ("trust_delta", "defensiveness_delta", "disclosure_delta"),
        )
    }
    payload = {
        "current_stage": stage,
        "session_context": context_view(context),
        "root_complaint_anchor": manager.root_complaint_anchor,
    }
    return {
        "payload": payload,
        "snapshot": {
            "stage_id": node["id"],
            "graph_state_version": manager.graph_state_version,
            "claim_refs": [
                {"claim_id": c["claim_id"], "revision": c["revision"]} for c in claims
            ],
            "initialization_refs": [
                r
                for c in resolved["initialization_background"]
                for r in c["evidence_refs"]
            ],
            "initialization_background": copy.deepcopy(
                stage["initialization_background"]
            ),
            "view_hash": digest(payload),
            "coverage": {
                "active_typed": len(resolved["active_claims"]),
                "selected_typed": len(claims),
                "initialization_opaque": len(resolved["initialization_background"]),
            },
        },
    }


def render_claim(c):
    """Render report scope; never infer a psychological trajectory from facts."""
    time = c.get("time", {}).get("reported_time_text") or "发生时间未明确"
    certainty = {"uncertain": "不确定报告", "denied": "否认性报告",
                 "unknown": "立场未明"}.get(c.get("assertion_status"), "报告")
    actuality = {"intended": "计划，尚未实施", "hypothetical": "假设，非既成事实",
                 "unknown": "实际性未明"}.get(c.get("actuality"), "保留原时点")
    source = "初始化设定" if c.get("acceptance_basis") == "configured" else "此前报告，不代表本轮再次确认"
    return f"{source}（{time}；{certainty}；{actuality}）：{c['text']}"


def build_emotion_input_view(stage, context=None):
    result = expression_stage(stage if isinstance(stage, dict) else {})
    stage = stage if isinstance(stage, dict) else {}
    # Only the generation projection carries authorized stage semantics. Raw
    # node display fields are intentionally not a shortcut around that boundary.
    if "accepted_claims" not in stage:
        result.update(project(stage, ("label", "summary", "narrative_focus", "recent_experiences")))
    result["topic"] = stage.get("topic", "初始主诉")
    result["current_claims"] = [
        semantic_claim_view(c) for c in stage.get("current_claims", [])
    ]
    relationship = (context or {}).get("participants", {}).get("relationship", "")
    result["relation_modifiers"] = {
        relationship: project(
            stage.get("relation_modifiers", {}).get(relationship),
            ("trust_delta", "defensiveness_delta", "disclosure_delta"),
        )
    }
    return result
