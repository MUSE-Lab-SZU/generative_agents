"""Pure, explicit projections. Audit metadata never enters generation payloads."""

import copy
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
    pairs = {(o.get("subject"), o.get("kind")) for o in observations or []}
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
    return copy.deepcopy(
        [
            c
            for c in active_claims
            if (c["claim_id"], c["revision"]) in refs
            or (task != "validator" and (not topic or c["topic_id"] == topic))
            or (c["subject"], c["kind"]) in pairs
        ]
    )


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
    claims = select_relevant_claims(
        "patient", resolved["active_claims"], context={"topic_id": node["topic_id"]}
    )
    stage = expression_stage(node)
    stage.update(
        topic=manager.topic_registry[node["topic_id"]]["neutral_name"],
        current_claims=[claim_view(c) for c in claims],
        core_belief=manager.core_belief,
        initialization_background=[
            c["text"] for c in resolved["initialization_background"]
        ],
    )
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


def build_emotion_input_view(stage, context=None):
    result = expression_stage(stage if isinstance(stage, dict) else {})
    stage = stage if isinstance(stage, dict) else {}
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
