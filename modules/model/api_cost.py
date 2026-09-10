"""Per-experiment DeepSeek usage ledger and cost summaries."""

from __future__ import annotations

import fcntl
import json
import os
import tempfile
import uuid
from collections import defaultdict
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Mapping
from zoneinfo import ZoneInfo


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PRICING_PATH = PROJECT_ROOT / "data" / "deepseek_pricing.v1.json"
DEFAULT_EXPERIMENT_DATA_ROOT = PROJECT_ROOT / "results" / "experiment_data"

ENV_RUN_NAME = "GA_API_COST_RUN_NAME"
ENV_PHASE = "GA_API_COST_PHASE"
ENV_EXPERIMENT_DATA_ROOT = "GA_API_COST_EXPERIMENT_DATA_ROOT"
ENV_EVALUATION_ID = "GA_API_COST_EVALUATION_ID"
ENV_SCALE_NAME = "GA_API_COST_SCALE_NAME"

VALID_PHASES = {"simulation", "repeat_evaluation"}


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return payload


def load_pricing_config(path: str | Path | None = None) -> dict[str, Any]:
    return _read_json(Path(path).resolve() if path else DEFAULT_PRICING_PATH)


def _safe_run_name(value: Any) -> str:
    run_name = str(value or "").strip()
    if not run_name or run_name in {".", ".."} or "/" in run_name or "\\" in run_name:
        return ""
    return run_name


def configure_tracking(
    run_name: str,
    phase: str,
    *,
    experiment_data_root: str | Path = DEFAULT_EXPERIMENT_DATA_ROOT,
    evaluation_id: str = "",
    scale_name: str = "",
    rebuild: bool = True,
) -> dict[str, Any] | None:
    """Configure this process and its children for one accounting scope."""
    normalized_run = _safe_run_name(run_name)
    normalized_phase = str(phase or "").strip()
    if not normalized_run:
        raise ValueError("A safe non-empty run_name is required for API cost tracking")
    if normalized_phase not in VALID_PHASES:
        raise ValueError(f"Invalid API cost phase: {phase!r}")
    os.environ[ENV_RUN_NAME] = normalized_run
    os.environ[ENV_PHASE] = normalized_phase
    os.environ[ENV_EXPERIMENT_DATA_ROOT] = str(Path(experiment_data_root).resolve())
    if evaluation_id:
        os.environ[ENV_EVALUATION_ID] = str(evaluation_id)
    else:
        os.environ.pop(ENV_EVALUATION_ID, None)
    if scale_name:
        os.environ[ENV_SCALE_NAME] = str(scale_name)
    else:
        os.environ.pop(ENV_SCALE_NAME, None)
    return rebuild_summary(normalized_run, experiment_data_root=experiment_data_root) if rebuild else None


def _tracking_context() -> dict[str, str] | None:
    run_name = _safe_run_name(os.getenv(ENV_RUN_NAME))
    phase = str(os.getenv(ENV_PHASE, "") or "").strip()
    if not run_name or phase not in VALID_PHASES:
        return None
    return {
        "run_name": run_name,
        "phase": phase,
        "experiment_data_root": str(
            Path(os.getenv(ENV_EXPERIMENT_DATA_ROOT) or DEFAULT_EXPERIMENT_DATA_ROOT).resolve()
        ),
        "evaluation_id": str(os.getenv(ENV_EVALUATION_ID, "") or ""),
        "scale_name": str(os.getenv(ENV_SCALE_NAME, "") or ""),
    }


def _ledger_paths(run_name: str, experiment_data_root: str | Path) -> tuple[Path, Path]:
    safe_name = _safe_run_name(run_name)
    if not safe_name:
        raise ValueError(f"Unsafe run_name: {run_name!r}")
    directory = Path(experiment_data_root).resolve() / "api_cost" / safe_name
    return directory / "calls.jsonl", directory / "summary.json"


def is_deepseek_call(model: Any, base_url: Any) -> bool:
    return "deepseek" in str(model or "").lower() or "deepseek" in str(base_url or "").lower()


def _as_datetime(value: datetime | str | None, timezone: ZoneInfo) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    elif value:
        parsed = datetime.fromisoformat(str(value))
    else:
        parsed = datetime.now(timezone)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone)
    return parsed.astimezone(timezone)


def _time_to_minutes(value: str) -> int:
    hour, minute = (int(part) for part in value.split(":", 1))
    return hour * 60 + minute


def pricing_tier(request_started_at: datetime | str, pricing: Mapping[str, Any]) -> str:
    timezone = ZoneInfo(str(pricing.get("timezone") or "Asia/Shanghai"))
    request_time = _as_datetime(request_started_at, timezone)
    peak = pricing.get("peak_windows", {}) or {}
    weekdays = {int(value) for value in peak.get("weekdays", [])}
    if request_time.isoweekday() not in weekdays:
        return "off_peak"
    minute_of_day = request_time.hour * 60 + request_time.minute
    for interval in peak.get("intervals", []) or []:
        if not isinstance(interval, list) or len(interval) != 2:
            continue
        if _time_to_minutes(interval[0]) <= minute_of_day < _time_to_minutes(interval[1]):
            return "peak"
    return "off_peak"


def _resolve_model_price(model: str, pricing: Mapping[str, Any]) -> tuple[str, Mapping[str, Any] | None]:
    models = pricing.get("models", {}) or {}
    normalized = str(model or "").strip().lower()
    if normalized in models:
        return normalized, models[normalized]
    prefix_matches = [name for name in models if normalized.startswith(name + "-")]
    if prefix_matches:
        matched = max(prefix_matches, key=len)
        return matched, models[matched]
    return normalized, None


def calculate_cost(
    *,
    model: str,
    request_started_at: datetime | str,
    hit_tokens: int,
    miss_tokens: int,
    completion_tokens: int,
    reasoning_tokens: int | None = None,
    pricing: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    config = dict(pricing or load_pricing_config())
    tier = pricing_tier(request_started_at, config)
    priced_model, model_config = _resolve_model_price(model, config)
    result: dict[str, Any] = {
        "pricing_version": str(config.get("pricing_version", "")),
        "pricing_model": priced_model,
        "tier": tier,
        "currency": str(config.get("currency", "CNY")),
        "unit_tokens": int(config.get("unit_tokens", 1_000_000) or 1_000_000),
        "known": model_config is not None,
        "rates_cny_per_million": {},
        "hit_cost_cny": 0.0,
        "miss_cost_cny": 0.0,
        "output_cost_cny": 0.0,
        "reasoning_cost_cny": None,
        "visible_output_cost_cny": None,
        "total_cost_cny": 0.0,
    }
    if model_config is None:
        return result
    rates = model_config.get(tier, {}) or {}
    result["rates_cny_per_million"] = {
        "prompt_cache_hit": float(rates.get("prompt_cache_hit", 0) or 0),
        "prompt_cache_miss": float(rates.get("prompt_cache_miss", 0) or 0),
        "completion": float(rates.get("completion", 0) or 0),
    }
    unit = Decimal(str(result["unit_tokens"]))
    hit_cost = Decimal(max(0, int(hit_tokens))) * Decimal(str(rates.get("prompt_cache_hit", 0))) / unit
    miss_cost = Decimal(max(0, int(miss_tokens))) * Decimal(str(rates.get("prompt_cache_miss", 0))) / unit
    output_cost = Decimal(max(0, int(completion_tokens))) * Decimal(str(rates.get("completion", 0))) / unit
    reasoning_cost = None
    visible_output_cost = None
    if reasoning_tokens is not None:
        normalized_reasoning = min(
            max(0, int(reasoning_tokens)),
            max(0, int(completion_tokens)),
        )
        visible_output_tokens = max(0, int(completion_tokens) - normalized_reasoning)
        reasoning_cost = (
            Decimal(normalized_reasoning)
            * Decimal(str(rates.get("completion", 0)))
            / unit
        )
        visible_output_cost = (
            Decimal(visible_output_tokens)
            * Decimal(str(rates.get("completion", 0)))
            / unit
        )
    result.update(
        {
            "hit_cost_cny": float(round(hit_cost, 9)),
            "miss_cost_cny": float(round(miss_cost, 9)),
            "output_cost_cny": float(round(output_cost, 9)),
            "reasoning_cost_cny": (
                float(round(reasoning_cost, 9))
                if reasoning_cost is not None
                else None
            ),
            "visible_output_cost_cny": (
                float(round(visible_output_cost, 9))
                if visible_output_cost is not None
                else None
            ),
            "total_cost_cny": float(round(hit_cost + miss_cost + output_cost, 9)),
        }
    )
    return result


def _response_model(response: Any) -> str:
    if isinstance(response, dict):
        return str(response.get("model", "") or "")
    try:
        return str(getattr(response, "model", "") or "")
    except Exception:
        return ""


def build_call_event(
    *,
    response: Any,
    usage: Mapping[str, Any] | None,
    caller: str,
    provider: str,
    requested_model: str,
    base_url: str,
    request_started_at: datetime | str | None,
    response_received_at: datetime | str | None,
    status: str,
    error_category: str = "",
    request_options: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    context = _tracking_context()
    if context is None or not is_deepseek_call(requested_model, base_url):
        return None
    pricing = load_pricing_config()
    timezone = ZoneInfo(str(pricing.get("timezone") or "Asia/Shanghai"))
    started = _as_datetime(request_started_at, timezone)
    received = _as_datetime(response_received_at, timezone)
    hit = int((usage or {}).get("prompt_cache_hit_tokens", 0) or 0)
    miss = int((usage or {}).get("prompt_cache_miss_tokens", 0) or 0)
    output = int((usage or {}).get("completion_tokens", 0) or 0)
    reasoning_raw = (usage or {}).get("reasoning_tokens")
    reasoning = (
        min(max(0, int(reasoning_raw)), output)
        if reasoning_raw is not None
        else None
    )
    visible_output = output - reasoning if reasoning is not None else None
    response_model = _response_model(response)
    billing_model = response_model or str(requested_model or "")
    cost = calculate_cost(
        model=billing_model,
        request_started_at=started,
        hit_tokens=hit,
        miss_tokens=miss,
        completion_tokens=output,
        reasoning_tokens=reasoning,
        pricing=pricing,
    )
    input_tokens = hit + miss
    return {
        "schema_version": 2,
        "event_id": str(uuid.uuid4()),
        "run_name": context["run_name"],
        "phase": context["phase"],
        "caller": str(caller or "llm_normal"),
        "module": str(caller or "llm_normal"),
        "status": status,
        "error_category": str(error_category or ""),
        "request_started_at": started.isoformat(timespec="milliseconds"),
        "response_received_at": received.isoformat(timespec="milliseconds"),
        "duration_ms": max(0, int(round((received - started).total_seconds() * 1000))),
        "provider": str(provider or ""),
        "requested_model": str(requested_model or ""),
        "response_model": response_model,
        "endpoint": str(base_url or ""),
        "tokens": {
            "prompt_cache_hit": hit,
            "prompt_cache_miss": miss,
            "completion": output,
            "reasoning": reasoning,
            "visible_output": visible_output,
            "total": hit + miss + output,
        },
        "input_cache_rate": round(hit / input_tokens, 6) if input_tokens else 0.0,
        "pricing": cost,
        "cost_cny": float(cost["total_cost_cny"]),
        "evaluation_id": context["evaluation_id"],
        "scale_name": context["scale_name"],
        "request_options": dict(request_options or {}),
    }


def append_event(event: Mapping[str, Any], *, experiment_data_root: str | Path | None = None) -> Path:
    run_name = _safe_run_name(event.get("run_name"))
    if not run_name:
        raise ValueError("Event has no safe run_name")
    root = Path(
        experiment_data_root
        or os.getenv(ENV_EXPERIMENT_DATA_ROOT)
        or DEFAULT_EXPERIMENT_DATA_ROOT
    ).resolve()
    calls_path, _ = _ledger_paths(run_name, root)
    calls_path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(dict(event), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    with calls_path.open("a+", encoding="utf-8") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        stream.seek(0, os.SEEK_END)
        if stream.tell() > 0:
            stream.seek(stream.tell() - 1)
            if stream.read(1) != "\n":
                stream.seek(0, os.SEEK_END)
                stream.write("\n")
        stream.seek(0, os.SEEK_END)
        stream.write(encoded + "\n")
        stream.flush()
        os.fsync(stream.fileno())
        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
    return calls_path


def record_call_event(**kwargs: Any) -> dict[str, Any] | None:
    event = build_call_event(**kwargs)
    if event is not None:
        append_event(event)
    return event


def _empty_metrics() -> dict[str, Any]:
    return {
        "call_count": 0,
        "successful_calls": 0,
        "failed_calls": 0,
        "missing_usage_calls": 0,
        "missing_reasoning_detail_calls": 0,
        "unknown_price_calls": 0,
        "prompt_cache_hit_tokens": 0,
        "prompt_cache_miss_tokens": 0,
        "completion_tokens": 0,
        "reasoning_tokens": 0,
        "visible_output_tokens": 0,
        "total_tokens": 0,
        "input_cache_rate": 0.0,
        "cost_cny": 0.0,
        "reasoning_cost_cny": 0.0,
        "visible_output_cost_cny": 0.0,
    }


def _add_metrics(metrics: dict[str, Any], event: Mapping[str, Any]) -> None:
    metrics["call_count"] += 1
    status = str(event.get("status", "") or "")
    if status == "failed":
        metrics["failed_calls"] += 1
    else:
        metrics["successful_calls"] += 1
    if status == "missing_usage":
        metrics["missing_usage_calls"] += 1
    pricing = event.get("pricing", {}) or {}
    if not bool(pricing.get("known", False)):
        metrics["unknown_price_calls"] += 1
    tokens = event.get("tokens", {}) or {}
    hit = int(tokens.get("prompt_cache_hit", 0) or 0)
    miss = int(tokens.get("prompt_cache_miss", 0) or 0)
    output = int(tokens.get("completion", 0) or 0)
    reasoning = tokens.get("reasoning")
    visible_output = tokens.get("visible_output")
    metrics["prompt_cache_hit_tokens"] += hit
    metrics["prompt_cache_miss_tokens"] += miss
    metrics["completion_tokens"] += output
    if reasoning is None or visible_output is None:
        if status not in {"failed", "missing_usage"}:
            metrics["missing_reasoning_detail_calls"] += 1
    else:
        metrics["reasoning_tokens"] += max(0, int(reasoning or 0))
        metrics["visible_output_tokens"] += max(0, int(visible_output or 0))
    metrics["total_tokens"] += hit + miss + output
    metrics["cost_cny"] += float(event.get("cost_cny", 0) or 0)
    reasoning_cost = pricing.get("reasoning_cost_cny")
    visible_output_cost = pricing.get("visible_output_cost_cny")
    if reasoning_cost is not None:
        metrics["reasoning_cost_cny"] += float(reasoning_cost or 0)
    if visible_output_cost is not None:
        metrics["visible_output_cost_cny"] += float(visible_output_cost or 0)


def _finish_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    total_input = metrics["prompt_cache_hit_tokens"] + metrics["prompt_cache_miss_tokens"]
    metrics["input_cache_rate"] = (
        round(metrics["prompt_cache_hit_tokens"] / total_input, 6) if total_input else 0.0
    )
    metrics["cost_cny"] = round(float(metrics["cost_cny"]), 9)
    metrics["reasoning_cost_cny"] = round(
        float(metrics["reasoning_cost_cny"]), 9
    )
    metrics["visible_output_cost_cny"] = round(
        float(metrics["visible_output_cost_cny"]), 9
    )
    return metrics


def _simulation_session_count(run_name: str, experiment_data_root: Path) -> int:
    results_root = experiment_data_root.parent
    candidates = [
        experiment_data_root / run_name / "traces" / "judge_conversation.json",
        results_root / "checkpoints" / run_name / "judge_traces" / "judge_conversation.json",
    ]
    for path in candidates:
        if not path.is_file():
            continue
        try:
            payload = _read_json(path)
            sessions = payload.get("sessions", [])
            if isinstance(sessions, list):
                return len(sessions)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
    return 0


def _summary_from_events(
    run_name: str,
    events: list[dict[str, Any]],
    *,
    experiment_data_root: Path,
    malformed_lines: list[int],
) -> dict[str, Any]:
    phases = {name: _empty_metrics() for name in sorted(VALID_PHASES)}
    combined = _empty_metrics()
    modules: dict[str, dict[str, Any]] = defaultdict(_empty_metrics)
    evaluation_ids: set[str] = set()
    scale_units: set[tuple[str, str]] = set()
    repeat_evaluation_modules: dict[str, set[str]] = defaultdict(set)
    for event in events:
        phase = str(event.get("phase", "") or "")
        if phase not in phases:
            continue
        _add_metrics(phases[phase], event)
        _add_metrics(combined, event)
        _add_metrics(modules[str(event.get("module", "") or "unknown")], event)
        if phase == "repeat_evaluation":
            evaluation_id = str(event.get("evaluation_id", "") or "")
            scale_name = str(event.get("scale_name", "") or "")
            if evaluation_id:
                evaluation_ids.add(evaluation_id)
                repeat_evaluation_modules[evaluation_id].add(
                    str(event.get("module", "") or "unknown")
                )
                if scale_name:
                    scale_units.add((evaluation_id, scale_name))
    for metrics in phases.values():
        _finish_metrics(metrics)
    _finish_metrics(combined)
    module_ranking = [
        {"module": module, **_finish_metrics(metrics)}
        for module, metrics in modules.items()
    ]
    module_ranking.sort(key=lambda item: (-item["cost_cny"], -item["total_tokens"], item["module"]))
    session_count = _simulation_session_count(run_name, experiment_data_root)
    repeat_cost = phases["repeat_evaluation"]["cost_cny"]
    missing_expected_modules = [
        {
            "evaluation_id": evaluation_id,
            "present_modules": sorted(module_names),
            "missing_modules": ["depression_scale_generate_chat_forced"],
        }
        for evaluation_id, module_names in sorted(repeat_evaluation_modules.items())
        if "repeat_eval_score" in module_names
        and "depression_scale_generate_chat_forced" not in module_names
    ]
    integrity_warnings = []
    if malformed_lines:
        integrity_warnings.append("calls.jsonl contains malformed lines; valid events were retained")
    if missing_expected_modules:
        integrity_warnings.append(
            "repeat evaluation score events have no recorded depression_scale_generate_chat_forced answer events"
        )
    return {
        "schema_version": 2,
        "run_name": run_name,
        "generated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds"),
        "simulation": phases["simulation"],
        "repeat_evaluation": phases["repeat_evaluation"],
        "combined": combined,
        "module_cost_ranking": module_ranking,
        "averages": {
            "cbt_session_count": session_count,
            "cost_per_cbt_session_cny": (
                round(phases["simulation"]["cost_cny"] / session_count, 9)
                if session_count else None
            ),
            "dual_scale_evaluation_count": len(evaluation_ids),
            "cost_per_dual_scale_evaluation_cny": (
                round(repeat_cost / len(evaluation_ids), 9) if evaluation_ids else None
            ),
            "scale_evaluation_count": len(scale_units),
            "cost_per_scale_evaluation_cny": (
                round(repeat_cost / len(scale_units), 9) if scale_units else None
            ),
        },
        "integrity": {
            "unique_event_count": len(events),
            "duplicate_event_count": 0,
            "malformed_line_count": len(malformed_lines),
            "malformed_line_numbers": malformed_lines,
            "missing_expected_modules": missing_expected_modules,
            "warnings": integrity_warnings,
        },
    }


def rebuild_summary(
    run_name: str,
    *,
    experiment_data_root: str | Path = DEFAULT_EXPERIMENT_DATA_ROOT,
) -> dict[str, Any]:
    root = Path(experiment_data_root).resolve()
    calls_path, summary_path = _ledger_paths(run_name, root)
    calls_path.parent.mkdir(parents=True, exist_ok=True)
    events: list[dict[str, Any]] = []
    malformed_lines: list[int] = []
    duplicate_count = 0
    seen: set[str] = set()
    with calls_path.open("a+", encoding="utf-8") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        stream.seek(0)
        for line_number, raw_line in enumerate(stream, start=1):
            if not raw_line.strip():
                continue
            try:
                event = json.loads(raw_line)
            except json.JSONDecodeError:
                malformed_lines.append(line_number)
                continue
            if not isinstance(event, dict) or str(event.get("run_name", "") or "") != run_name:
                malformed_lines.append(line_number)
                continue
            event_id = str(event.get("event_id", "") or "")
            if not event_id or event_id in seen:
                duplicate_count += 1
                continue
            seen.add(event_id)
            events.append(event)
        summary = _summary_from_events(
            run_name,
            events,
            experiment_data_root=root,
            malformed_lines=malformed_lines,
        )
        summary["integrity"]["duplicate_event_count"] = duplicate_count
        fd, temporary_name = tempfile.mkstemp(
            prefix=".summary.", suffix=".tmp", dir=str(summary_path.parent)
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as output:
                json.dump(summary, output, ensure_ascii=False, indent=2, sort_keys=True)
                output.write("\n")
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary_name, summary_path)
        except BaseException:
            try:
                os.unlink(temporary_name)
            except OSError:
                pass
            raise
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
    return summary
