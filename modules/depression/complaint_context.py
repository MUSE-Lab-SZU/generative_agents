"""Read-only, versioned disclosure units derived from the active complaint stage."""
import hashlib
import json
import math


def normalize_units(value, generated=False):
    if not isinstance(value, list):
        raise ValueError("disclosure_units must be a list")
    result, seen = [], set()
    for raw in value:
        if not isinstance(raw, dict):
            raise ValueError("Disclosure units must be objects")
        identifier = str(raw.get("id", "")).strip()
        content = str(raw.get("content", "")).strip()
        threshold = float(raw.get("disclosure_threshold", 1.0))
        if not identifier or identifier in seen or not content:
            raise ValueError("Disclosure units require unique IDs and nonempty content")
        if not math.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("Disclosure threshold must be between 0 and 1")
        seen.add(identifier)
        result.append({"id": identifier, "content": content,
                       "disclosure_threshold": 1.0 if generated else threshold})
    return result


def stage_units(stage):
    # Explicit units replace all raw narrative fields, preventing duplicate-field bypasses.
    units = stage.get("disclosure_units")
    if units is None:
        fields = [(key, stage.get(key)) for key in ("summary", "core_belief")]
        fields += [("narrative_focus." + str(i), text)
                   for i, text in enumerate(stage.get("narrative_focus", []))]
        units = [{"id": key, "content": text, "disclosure_threshold": 1.0}
                 for key, text in fields if text]
    units = normalize_units(units, generated=stage.get("source") == "llm")
    result = []
    for unit in units:
        digest = hashlib.sha256(json.dumps(
            [stage["id"], unit], ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        result.append({"memory_id": "complaint_" + digest,
                       "content": unit["content"], "stage_id": stage["id"],
                       "unit_id": unit["id"], "disclosure_threshold": unit["disclosure_threshold"]})
    return result
