#!/usr/bin/env python3
"""Backfill untracked DeepSeek scale-answer calls from preserved worker logs.

The tool intentionally only accepts ``[LLM_PROMPT_CACHE]`` records for the
forced scale-answer caller.  It is therefore suitable for the six recoverable
G2/G4/G5 runs in the 0901 audit, but must not be used to turn proxy estimates
or incomplete logs into apparently recorded usage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.model.api_cost import (
    DEFAULT_EXPERIMENT_DATA_ROOT,
    append_event,
    calculate_cost,
    is_deepseek_call,
    load_pricing_config,
    pricing_tier,
    rebuild_summary,
)


MARKER = "[LLM_PROMPT_CACHE]"
ANSWER_CALLER = "depression_scale_generate_chat_forced"
BACKFILL_NAMESPACE = uuid.UUID("75ab4c48-fc47-5e4a-bd2c-1be4655cdf08")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _nonnegative_int(value: Any) -> int | None:
    try:
        result = int(value)
    except (TypeError, ValueError):
        return None
    return result if result >= 0 else None


def _usage_from_payload(payload: Mapping[str, Any]) -> dict[str, int] | None:
    request = payload.get("request")
    if not isinstance(request, Mapping):
        return None
    hit = _nonnegative_int(request.get("prompt_cache_hit_tokens"))
    miss = _nonnegative_int(request.get("prompt_cache_miss_tokens"))
    completion = _nonnegative_int(request.get("completion_tokens"))
    if hit is None or miss is None or completion is None:
        return None
    return {
        "prompt_cache_hit": hit,
        "prompt_cache_miss": miss,
        "completion": completion,
        "total": hit + miss + completion,
    }


def iter_untracked_answer_records(log_path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    """Yield valid, previously-untracked forced scale-answer cache records."""
    with log_path.open("r", encoding="utf-8") as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            if MARKER not in raw_line:
                continue
            try:
                payload = json.loads(raw_line.split(MARKER, 1)[1].strip())
            except json.JSONDecodeError:
                continue
            if not isinstance(payload, dict):
                continue
            if str(payload.get("caller", "") or "") != ANSWER_CALLER:
                continue
            # A current log can legitimately contain this caller.  Its event
            # already reached the ledger and must never be backfilled again.
            if isinstance(payload.get("accounting"), Mapping):
                continue
            usage = _usage_from_payload(payload)
            if usage is None:
                continue
            model = str(payload.get("model", "") or "")
            endpoint = str(payload.get("endpoint", "") or "")
            if not is_deepseek_call(model, endpoint):
                continue
            yield line_number, {
                "usage": usage,
                "model": model,
                "endpoint": endpoint,
                "provider": str(payload.get("provider", "") or ""),
            }


def existing_event_ids(run_name: str, experiment_data_root: Path) -> set[str]:
    calls_path = experiment_data_root / "api_cost" / run_name / "calls.jsonl"
    if not calls_path.is_file():
        return set()
    event_ids: set[str] = set()
    with calls_path.open("r", encoding="utf-8") as stream:
        for raw_line in stream:
            try:
                event = json.loads(raw_line)
            except json.JSONDecodeError:
                continue
            if isinstance(event, dict) and event.get("event_id"):
                event_ids.add(str(event["event_id"]))
    return event_ids


def _parse_inferred_timestamp(value: datetime | str) -> datetime:
    parsed = value if isinstance(value, datetime) else datetime.fromisoformat(str(value))
    if parsed.tzinfo is None:
        raise ValueError("--inferred-request-started-at must include a timezone offset")
    return parsed


def build_backfill_event(
    *,
    run_name: str,
    evaluation_id: str,
    scale_name: str,
    source_path: Path,
    source_sha256: str,
    source_line: int,
    record: Mapping[str, Any],
    inferred_request_started_at: datetime | str,
    assumed_tier: str,
) -> dict[str, Any]:
    usage = record["usage"]
    if not isinstance(usage, Mapping):
        raise ValueError("record has no usage")
    inferred_timestamp = _parse_inferred_timestamp(inferred_request_started_at)
    pricing = load_pricing_config()
    actual_tier = pricing_tier(inferred_timestamp, pricing)
    if actual_tier != assumed_tier:
        raise ValueError(
            "inferred timestamp tier {} does not match --assumed-tier {}".format(
                actual_tier, assumed_tier
            )
        )
    model = str(record.get("model", "") or "")
    cost = calculate_cost(
        model=model,
        request_started_at=inferred_timestamp,
        hit_tokens=int(usage["prompt_cache_hit"]),
        miss_tokens=int(usage["prompt_cache_miss"]),
        completion_tokens=int(usage["completion"]),
        pricing=pricing,
    )
    timestamp = inferred_timestamp.isoformat(timespec="milliseconds")
    event_id = str(
        uuid.uuid5(
            BACKFILL_NAMESPACE,
            "{}:{}:{}:{}:{}".format(
                run_name,
                evaluation_id,
                source_sha256,
                source_line,
                ANSWER_CALLER,
            ),
        )
    )
    return {
        "schema_version": 1,
        "event_id": event_id,
        "run_name": run_name,
        "phase": "repeat_evaluation",
        "caller": ANSWER_CALLER,
        "module": ANSWER_CALLER,
        "status": "success",
        "error_category": "",
        "request_started_at": timestamp,
        "response_received_at": timestamp,
        "duration_ms": 0,
        "provider": str(record.get("provider", "") or ""),
        "requested_model": model,
        "response_model": "",
        "endpoint": str(record.get("endpoint", "") or ""),
        "tokens": dict(usage),
        "input_cache_rate": round(
            int(usage["prompt_cache_hit"])
            / (int(usage["prompt_cache_hit"]) + int(usage["prompt_cache_miss"])),
            6,
        )
        if int(usage["prompt_cache_hit"]) + int(usage["prompt_cache_miss"])
        else 0.0,
        "pricing": cost,
        "cost_cny": float(cost["total_cost_cny"]),
        "evaluation_id": evaluation_id,
        "scale_name": scale_name,
        "provenance": {
            "kind": "historical_llm_prompt_cache_backfill",
            "evidence": {
                "tokens": "recorded",
                "request_started_at": "inferred",
                "pricing_tier": "inferred",
            },
            "source_file": str(source_path),
            "source_sha256": source_sha256,
            "source_line": source_line,
            "assumed_pricing_tier": assumed_tier,
        },
    }


def backfill_logs(
    *,
    run_name: str,
    evaluation_id: str,
    scale_name: str,
    log_paths: Iterable[Path],
    experiment_data_root: Path,
    inferred_request_started_at: datetime | str,
    assumed_tier: str,
    write: bool,
) -> dict[str, int]:
    known_event_ids = existing_event_ids(run_name, experiment_data_root)
    stats = {"matched": 0, "already_present": 0, "appended": 0}
    for log_path in log_paths:
        source_sha = file_sha256(log_path)
        for source_line, record in iter_untracked_answer_records(log_path):
            stats["matched"] += 1
            event = build_backfill_event(
                run_name=run_name,
                evaluation_id=evaluation_id,
                scale_name=scale_name,
                source_path=log_path,
                source_sha256=source_sha,
                source_line=source_line,
                record=record,
                inferred_request_started_at=inferred_request_started_at,
                assumed_tier=assumed_tier,
            )
            event_id = event["event_id"]
            if event_id in known_event_ids:
                stats["already_present"] += 1
                continue
            if write:
                append_event(event, experiment_data_root=experiment_data_root)
                known_event_ids.add(event_id)
                stats["appended"] += 1
    if write:
        rebuild_summary(run_name, experiment_data_root=experiment_data_root)
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--evaluation-id", required=True)
    parser.add_argument("--log", action="append", required=True, help="Preserved worker log; repeat for each source file")
    parser.add_argument("--experiment-data-root", default=str(DEFAULT_EXPERIMENT_DATA_ROOT))
    parser.add_argument("--scale-name", default="", help="Only set when the source log maps unambiguously to one scale")
    parser.add_argument("--inferred-request-started-at", required=True, help="ISO-8601 timestamp used only for the documented pricing-tier inference")
    parser.add_argument("--assumed-tier", choices=("peak", "off_peak"), required=True)
    parser.add_argument("--write", action="store_true", help="Append events; default is dry-run")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    log_paths = [Path(value).resolve() for value in args.log]
    missing = [str(path) for path in log_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("worker log(s) not found: {}".format(", ".join(missing)))
    stats = backfill_logs(
        run_name=str(args.run_name),
        evaluation_id=str(args.evaluation_id),
        scale_name=str(args.scale_name),
        log_paths=log_paths,
        experiment_data_root=Path(args.experiment_data_root).resolve(),
        inferred_request_started_at=args.inferred_request_started_at,
        assumed_tier=str(args.assumed_tier),
        write=bool(args.write),
    )
    mode = "write" if args.write else "dry-run"
    print("[{}] matched={matched} already_present={already_present} appended={appended}".format(mode, **stats))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
