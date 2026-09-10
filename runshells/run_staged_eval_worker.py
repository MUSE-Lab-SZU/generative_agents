#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import sys
import time
import traceback
import types
from typing import Any, Dict, List


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SCALE_AGENT_DIR = os.path.join(BASE_DIR, "customization", "depression_scale_agent")
QUESTIONS_ROOT = os.path.join(SCALE_AGENT_DIR, "questions", "templates")

if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
if "gradio" not in sys.modules:
    sys.modules["gradio"] = types.ModuleType("gradio")

from modules.model.api_cost import configure_tracking, rebuild_summary


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def _write_json(path: str, payload: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _safe_file_stem(value: str) -> str:
    return str(value or "").replace("/", "_").replace("\\", "_").replace(" ", "_")


def _normalize_scale_item_ids(raw_value: Any) -> set[int] | None:
    if raw_value is None:
        return None
    if not isinstance(raw_value, list):
        raise ValueError("scale_item_ids values must be lists")
    item_ids: set[int] = set()
    for value in raw_value:
        try:
            item_id = int(value)
        except (TypeError, ValueError):
            raise ValueError("scale_item_ids contains non-integer item id: {}".format(value))
        if item_id <= 0:
            raise ValueError("scale_item_ids contains non-positive item id: {}".format(value))
        item_ids.add(item_id)
    return item_ids


def _trace_used_external_memory(trace_payload: Dict[str, Any]) -> bool:
    if not isinstance(trace_payload, dict):
        return False
    retrieval = trace_payload.get("external_memory_retrieval", {}) or {}
    if not isinstance(retrieval, dict):
        return False
    return bool(retrieval.get("ok", False)) or bool(str(retrieval.get("scoped_user_id", "") or "").strip())


def _create_tmp_storage(job: Dict[str, Any]) -> tuple[str, str]:
    tmp_root_parent = os.path.abspath(str(job.get("tmp_root_parent", "") or "").strip())
    if not tmp_root_parent:
        raise ValueError("tmp_root_parent is required")
    os.makedirs(tmp_root_parent, exist_ok=True)

    trigger_label = _safe_file_stem(str(job.get("trigger_label", "") or "trigger"))
    unique_suffix = "{}_{}".format(os.getpid(), int(time.time() * 1000))
    tmp_root = os.path.join(tmp_root_parent, "{}_{}".format(trigger_label, unique_suffix))
    tmp_storage_root = os.path.join(tmp_root, "storage")

    storage_source_root = os.path.abspath(str(job.get("storage_source_root", "") or "").strip())
    if storage_source_root and os.path.isdir(storage_source_root):
        shutil.copytree(storage_source_root, tmp_storage_root)
    else:
        os.makedirs(tmp_storage_root, exist_ok=True)
    return tmp_root, tmp_storage_root


def _load_scale_chat_components():
    from customization.depression_scale_agent.app import ChatSession, iter_jsonl, resolve_question

    return ChatSession, iter_jsonl, resolve_question


def _api_cost_context(job: Dict[str, Any]) -> Dict[str, str] | None:
    raw_context = job.get("api_cost")
    if raw_context is None:
        return None
    if not isinstance(raw_context, dict):
        raise ValueError("api_cost must be an object when provided")
    context = {
        "run_name": str(raw_context.get("run_name", "") or "").strip(),
        "phase": str(raw_context.get("phase", "") or "").strip(),
        "experiment_data_root": str(raw_context.get("experiment_data_root", "") or "").strip(),
        "evaluation_id": str(raw_context.get("evaluation_id", "") or "").strip(),
    }
    missing = [
        field
        for field in ("run_name", "phase", "experiment_data_root", "evaluation_id")
        if not context[field]
    ]
    if missing:
        raise ValueError("api_cost missing required fields: {}".format(", ".join(missing)))
    return context


def _evaluate(
    job: Dict[str, Any],
    tmp_storage_root: str,
    *,
    api_cost: Dict[str, str] | None = None,
) -> Dict[str, Any]:
    ChatSession, iter_jsonl, resolve_question = _load_scale_chat_components()
    runtime_config = copy.deepcopy(job.get("runtime_config", {})) if isinstance(job.get("runtime_config", {}), dict) else {}
    runtime_config["storage_root_override"] = tmp_storage_root

    conversation = copy.deepcopy(job.get("conversation", {})) if isinstance(job.get("conversation", {}), dict) else {}
    run_name = str(job.get("run_name", "") or "").strip()
    snapshot_name = str(job.get("snapshot_name", "") or "").strip()
    trigger_label = str(job.get("trigger_label", "") or "").strip()
    target_agent = str(job.get("target_agent", "") or "").strip()
    trigger_dir = os.path.abspath(str(job.get("trigger_dir", "") or "").strip())
    scale_question_files = job.get("scale_question_files", {}) or {}
    scale_item_ids_raw = job.get("scale_item_ids", {}) or {}
    scales = job.get("scales", []) or []
    if not isinstance(scale_question_files, dict):
        raise ValueError("scale_question_files must be a dict")
    if not isinstance(scale_item_ids_raw, dict):
        raise ValueError("scale_item_ids must be a dict")
    if not isinstance(scales, list):
        raise ValueError("scales must be a list")
    if not target_agent:
        raise ValueError("target_agent is required")
    if not trigger_dir:
        raise ValueError("trigger_dir is required")

    os.makedirs(trigger_dir, exist_ok=True)

    session = ChatSession(
        run_name,
        snapshot_file=snapshot_name or trigger_label,
        runtime_config=runtime_config,
        conversation=conversation,
    )
    session.set_agent(target_agent)

    scale_summaries: Dict[str, Any] = {}
    external_memory_read = False
    for scale_name in scales:
        if api_cost is not None:
            configure_tracking(
                api_cost["run_name"],
                api_cost["phase"],
                experiment_data_root=api_cost["experiment_data_root"],
                evaluation_id=api_cost["evaluation_id"],
                scale_name=str(scale_name),
                rebuild=False,
            )
        question_file = str(scale_question_files.get(scale_name, "") or "").strip()
        if not question_file:
            raise ValueError("missing question file for scale: {}".format(scale_name))
        question_path = os.path.join(QUESTIONS_ROOT, question_file)
        if not os.path.exists(question_path):
            raise FileNotFoundError("question file not found: {}".format(question_path))
        requested_item_ids = _normalize_scale_item_ids(scale_item_ids_raw.get(scale_name))

        answered_rows: List[Dict[str, Any]] = []
        trace_rows: List[Dict[str, Any]] = []
        scale_external_memory_read = False
        for index, item in enumerate(iter_jsonl(question_path), start=1):
            item_id = item.get("id") if isinstance(item, dict) else index
            try:
                normalized_item_id = int(item_id)
            except (TypeError, ValueError):
                normalized_item_id = index
            if requested_item_ids is not None and normalized_item_id not in requested_item_ids:
                continue
            question = resolve_question(item)
            answer = session.answer_without_memory(question)
            trace_payload = session.get_last_answer_trace()
            row = dict(item) if isinstance(item, dict) else {"question": question}
            row["answer"] = answer
            answered_rows.append(row)
            trace_rows.append(
                {
                    "index": index,
                    "question": question,
                    "answer": answer,
                    "trace": trace_payload,
                }
            )
            if _trace_used_external_memory(trace_payload):
                scale_external_memory_read = True
                external_memory_read = True

        answers_file = "{}_answered.jsonl".format(_safe_file_stem(scale_name))
        trace_file = "{}_trace.json".format(_safe_file_stem(scale_name))
        _write_jsonl(os.path.join(trigger_dir, answers_file), answered_rows)
        _write_json(os.path.join(trigger_dir, trace_file), trace_rows)
        scale_summaries[scale_name] = {
            "question_file": question_file,
            "question_count": len(answered_rows),
            "requested_item_ids": sorted(requested_item_ids) if requested_item_ids is not None else [],
            "answers_file": answers_file,
            "trace_file": trace_file,
            "external_memory_read": scale_external_memory_read,
        }

    return {
        "scale_summaries": scale_summaries,
        "external_memory_read": external_memory_read,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a single staged-eval trigger in an isolated worker process")
    parser.add_argument("--job", required=True, help="Path to staged-eval job json")
    args = parser.parse_args()

    job_path = os.path.abspath(str(args.job or "").strip())
    job = _load_json(job_path)
    api_cost = _api_cost_context(job)

    worker_result_path = os.path.abspath(
        str(job.get("worker_result_path", "") or os.path.join(os.path.dirname(job_path), "worker_result.json")).strip()
    )
    result: Dict[str, Any] = {
        "status": "error",
        "job_file": job_path,
        "trigger_label": str(job.get("trigger_label", "") or ""),
        "tmp_root": "",
        "tmp_storage_path": "",
        "tmp_storage_deleted": False,
        "cleanup_error": "",
        "external_memory_read": False,
        "scale_summaries": {},
        "duration_seconds": 0.0,
        "error": "",
        "traceback": "",
    }

    start_ts = time.time()
    tmp_root = ""
    cleanup_tmp_storage = bool(job.get("cleanup_tmp_storage", True))
    try:
        tmp_root, tmp_storage_root = _create_tmp_storage(job)
        result["tmp_root"] = tmp_root
        result["tmp_storage_path"] = tmp_storage_root

        eval_result = _evaluate(job, tmp_storage_root, api_cost=api_cost)
        result["status"] = "ok"
        result["external_memory_read"] = bool(eval_result.get("external_memory_read", False))
        result["scale_summaries"] = eval_result.get("scale_summaries", {})
    except Exception as exc:
        result["error"] = str(exc)
        result["traceback"] = traceback.format_exc()
    finally:
        result["duration_seconds"] = round(time.time() - start_ts, 3)
        if cleanup_tmp_storage and tmp_root:
            try:
                shutil.rmtree(tmp_root)
                result["tmp_storage_deleted"] = True
            except Exception as exc:
                result["tmp_storage_deleted"] = False
                result["cleanup_error"] = str(exc)
        elif not cleanup_tmp_storage and tmp_root:
            result["cleanup_error"] = "cleanup_disabled"

        if api_cost is not None:
            try:
                rebuild_summary(
                    api_cost["run_name"],
                    experiment_data_root=api_cost["experiment_data_root"],
                )
            except Exception as exc:
                # Usage persistence must never make an otherwise successful
                # evaluation fail, matching the LLM-layer ledger policy.
                print("[API_COST_SUMMARY_ERROR] {}".format(exc), flush=True)

        _write_json(worker_result_path, result)

    return 0 if result.get("status") == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
