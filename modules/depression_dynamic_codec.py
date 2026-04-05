"""Dynamic depression engine state codec helpers."""

import copy
from typing import Any, Dict


SCHEMA_VERSION = "1.0.0"


def dump_state(engine: Any, enabled: bool) -> Dict[str, Any]:
    """Serialize runtime state for checkpoint."""
    runtime: Dict[str, Any] = {}
    if engine is not None and hasattr(engine, "to_dict"):
        try:
            payload = engine.to_dict()
            if isinstance(payload, dict):
                runtime = payload
        except Exception:
            runtime = {}
    return {
        "schema_version": SCHEMA_VERSION,
        "enabled": bool(enabled),
        "runtime": runtime,
    }


def load_state(engine: Any, payload: Any, logger: Any = None) -> Dict[str, Any]:
    """Restore runtime state from checkpoint payload."""
    if engine is None or not hasattr(engine, "load_state"):
        return {"loaded": False, "reason": "engine_unavailable"}
    if not isinstance(payload, dict):
        return {"loaded": False, "reason": "payload_not_dict"}

    schema = str(payload.get("schema_version", "") or "")
    runtime = payload.get("runtime", {})
    if isinstance(runtime, dict) and runtime:
        runtime_payload = runtime
    elif schema == "" and payload:
        # Backward compatibility: payload itself may be runtime body.
        runtime_payload = payload
    else:
        return {"loaded": False, "reason": "runtime_empty"}

    if schema and schema != SCHEMA_VERSION and logger is not None:
        try:
            logger.warning(
                "[DEPR_DYNAMIC][LOAD] schema_version_mismatch expected={} actual={}".format(
                    SCHEMA_VERSION,
                    schema,
                )
            )
        except Exception:
            pass

    try:
        engine.load_state(copy.deepcopy(runtime_payload))
        return {"loaded": True, "reason": "ok"}
    except Exception as e:
        if logger is not None:
            try:
                logger.warning(
                    "[DEPR_DYNAMIC][LOAD] restore_failed error={}".format(e)
                )
            except Exception:
                pass
        return {"loaded": False, "reason": "restore_failed"}
