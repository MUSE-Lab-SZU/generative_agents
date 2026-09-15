#!/usr/bin/env python3
"""Small fixed-corpus and feedback replay, with inspectable raw evidence and traces.

No semantic scorer or extra verifier. Reviewers annotate audit fields from sources.
Only synthetic fixtures are sent to the explicitly selected model endpoint.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import sys
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from modules.depression.state_machine import ComplaintGraphManager
from modules.depression.evidence import new_id
from modules.depression.generation_view import build_patient_generation_view

CORPUS = [
    (
        "appraisal",
        "以前把没用当成事实，现在知道那是念头，但还是觉得自己没用。",
        "仅确认评价关系变化，保留自我否定",
    ),
    ("plan", "打算明天出去走走，今天还是没出门。", "计划和未出门；无改善"),
    ("sleep_detail", "昨晚睡了五小时。", "没有可比睡眠时长基线，observation-only"),
    (
        "worsening",
        "上周每晚要半小时才能睡着，这周每晚要两个小时，比上周更睡不着了。",
        "仅入睡潜伏期的明确前后变化",
    ),
    ("negation", "没好转，只是不想再解释。", "不能从披露变化推疗效"),
    ("assent", "对。", "简单同意不支持复杂心理机制"),
    (
        "enacted",
        "之前一直躲着朋友，今天终于实际和朋友见了一面，但仍很紧张。",
        "仅一次实际行动",
    ),
    (
        "correction",
        "纠正上次的报告，上周不是两晚，是三晚难以入睡。",
        "修订旧报告而非疗效",
    ),
    (
        "focus",
        "工作先不谈，现在最困扰我的是照顾父亲，他最近住院了。",
        "明确关注转移；优先复用既有topic",
    ),
    (
        "independent",
        "昨晚睡五小时，今天收到快递；以前把没用当事实，现在知道它是念头但还是觉得没用。",
        "只选评价变化，其他独立观察保留",
    ),
]


def fixture(protocol):
    return {
        "complaint_graph": {
            "pipeline_version": protocol,
            "initial_stage_id": "initial",
            "planner": {"llm_enabled": True},
            "stages": [
                {
                    "id": "initial",
                    "label": "工作自我否定",
                    "summary": "失业后认为自己没有价值。",
                    "core_belief": "我没有价值。",
                    "accepted_claims": [
                        {
                            "kind": "symptom_report",
                            "subject": "patient",
                            "text": "上周有两晚难以入睡。",
                            "assertion_status": "affirmed",
                            "actuality": "occurred",
                            "time": {
                                "reported_time_text": "上周",
                                "effective_start": None,
                                "effective_end": None,
                                "precision": "unknown",
                            },
                        }
                    ],
                }
            ],
        }
    }


def accepted(manager, text, snapshot=None):
    sid = manager.start_session()
    metadata = {}
    if snapshot:
        snapshot = copy.deepcopy(snapshot)
        ref, attempt = new_id("snapshot"), new_id("attempt")
        snapshot.update(generation_snapshot_ref=ref, generation_attempt_id=attempt)
        manager.generation_snapshots[ref] = snapshot
        metadata.update(generation_snapshot_ref=ref, generation_attempt_id=attempt)
    mid = manager.register_accepted_message(text, "synthetic_patient", sid, **metadata)
    return {
        "runtime_event": {
            "source": "chat",
            "metadata": dict(
                metadata,
                event_id=new_id("event"),
                message_refs=[mid],
                session_instance_id=sid,
            ),
        }
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="http://127.0.0.1:18000/v1")
    parser.add_argument("--model", default="qwen3-8b-vllm")
    parser.add_argument("--api-key-env", default="COMPLAINT_REPLAY_API_KEY")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--feedback-turns", type=int, default=3)
    args = parser.parse_args()
    report = dict(
        endpoint=args.endpoint,
        model=args.model,
        status="running",
        corpus=[],
        feedback=[],
        completed_events=0,
        audit_note="人工仅依据原话及前态填写 human_audit；不从advance率计算疗效或漂移改善。",
    )
    headers = {"Content-Type": "application/json"}
    token = os.environ.get(args.api_key_env)
    if token:
        headers["Authorization"] = "Bearer " + token

    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def completion(prompt):
        req = urllib.request.Request(
            args.endpoint.rstrip("/") + "/chat/completions",
            data=json.dumps(
                dict(
                    model=args.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    max_tokens=4096,
                )
            ).encode(),
            headers=headers,
        )
        with urllib.request.urlopen(req, timeout=120) as response:
            return json.load(response)["choices"][0]["message"]["content"]

    try:
        req = urllib.request.Request(
            args.endpoint.rstrip("/") + "/models", headers=headers
        )
        with urllib.request.urlopen(req, timeout=5) as response:
            report["available_models"] = [
                x["id"] for x in json.load(response).get("data", [])
            ]
    except Exception as exc:
        report.update(status="blocked_model_unavailable", error=str(exc))
        save()
        print(
            json.dumps(
                {k: report[k] for k in ("status", "error", "completed_events")},
                ensure_ascii=False,
            )
        )
        return 2
    for protocol in ("foundation_legacy", "v3"):
        for name, text, scope in CORPUS:
            manager = ComplaintGraphManager(fixture(protocol))
            context = accepted(manager, text)
            e = manager.evaluate_turn(
                context,
                text,
                counterpart_utterance="你愿意具体说说吗？",
                completion_func=completion,
            )
            result = manager.commit_turn(e)
            report["corpus"].append(
                dict(
                    protocol=protocol,
                    case=name,
                    source_text=text,
                    support_scope=scope,
                    before=fixture(protocol),
                    trace=manager.transition_traces[e["event_id"]],
                    result=result,
                    human_audit=dict(
                        unsupported_assertions=None,
                        abstract_expansions=None,
                        synonym_nodes=None,
                        wrong_kind_or_time=None,
                        false_rejection=None,
                    ),
                )
            )
            report["completed_events"] += 1
            save()
        manager = ComplaintGraphManager(fixture(protocol))
        for turn in range(args.feedback_turns):
            doctor = "请再说一遍你刚才提到的情况，不需要提出新的变化。"
            view = build_patient_generation_view(manager)
            prompt = (
                "你是模拟病例患者，只据以下病例背景及报告作答，保留时间和否定，不编造变化。仅输出患者原话。\n"
                + json.dumps(view["payload"], ensure_ascii=False)
                + "\n医生："
                + doctor
            )
            try:
                text = completion(prompt)
            except Exception as exc:
                report["feedback"].append(
                    dict(protocol=protocol, turn=turn, error=str(exc))
                )
                break
            ctx = accepted(manager, text, view["snapshot"])
            e = manager.evaluate_turn(
                ctx, text, counterpart_utterance=doctor, completion_func=completion
            )
            result = manager.commit_turn(e)
            report["feedback"].append(
                dict(
                    protocol=protocol,
                    turn=turn,
                    doctor=doctor,
                    generation_snapshot=view["snapshot"],
                    patient_text=text,
                    trace=manager.transition_traces[e["event_id"]],
                    result=result,
                    human_audit=None,
                )
            )
            report["completed_events"] += 1
            save()
    report["status"] = "replayed_requires_human_audit"
    save()
    print(
        json.dumps(
            {
                "status": report["status"],
                "completed_events": report["completed_events"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
