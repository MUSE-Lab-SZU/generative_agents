"""Shared PHQ-9/BDI-II parsing primitives used by evaluation CLIs."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any


CN_SCORE_VALUES = {
    "0": 0,
    "1": 1,
    "2": 2,
    "3": 3,
    "零": 0,
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
}
AMBIGUITY_MARKERS = (
    "或者",
    "不确定",
    "之间",
    "两三",
    "一两",
    "也可能",
    "选0感觉是骗人的",
)
NEGATED_SELECTION_MARKERS = ("不想选", "不是选", "不能选", "不敢选")
DIRECT_SCORE_PATTERNS = (
    r"(?:我)?\s*(?:会|想|大概|可能|应该|还是|就|其实)?\s*(?:选|选择)\s*(?:了)?\s*([0-3零一二两三])",
    r"([0-3零一二两三])\s*分",
    r"(?:评分的话|打分的话|大概|应该|可能|算是|算|就是|我觉得|我想|我会|应该是|大概是|可能是|差不多)\s*[，,。\.……\s]*([0-3零一二两三])\s*(?:吧|。|，|,|$)",
    r"(?:^|[，,。\.……\s])([0-3零一二两三])\s*吧",
    r"^[\s（\(\）\)……。,.，、嗯唔]*([0-3零一二两三])\s*(?:吧|。|，|,|$)",
)


def has_ambiguous_score_context(text: str, start: int, end: int) -> bool:
    context = text[max(0, start - 18) : min(len(text), end + 28)]
    return any(marker in context for marker in AMBIGUITY_MARKERS)


def extract_direct_answer_score(answer: str) -> tuple[int | None, str]:
    """Return ``(score, status)`` where status is direct/ambiguous/none."""

    normalized_answer = str(answer or "")
    hits: list[tuple[int, int, int]] = []
    for pattern in DIRECT_SCORE_PATTERNS:
        for match in re.finditer(pattern, normalized_answer):
            context = normalized_answer[max(0, match.start() - 6) : match.end() + 6]
            if any(marker in context for marker in NEGATED_SELECTION_MARKERS):
                continue
            score = CN_SCORE_VALUES.get(match.group(1))
            if score is not None:
                hits.append((match.start(), match.end(), score))
    if not hits:
        return None, "none"

    hits.sort(key=lambda item: (item[0], item[1]))
    first_start, first_end, first_score = hits[0]
    if has_ambiguous_score_context(normalized_answer, first_start, first_end):
        return first_score, "ambiguous"
    for start, _end, score in hits[1:]:
        if start - first_end < 35 and score != first_score:
            return first_score, "ambiguous"
    return first_score, "direct"


def extract_item_scores(
    scored_result: Mapping[str, Any] | None,
    item_key: str | None,
    *,
    expected_count: int | None = None,
) -> list[int] | None:
    if not item_key or not isinstance(scored_result, Mapping):
        return None
    items = scored_result.get(item_key)
    if not isinstance(items, list):
        return None

    scores: list[int] = []
    for item in items:
        score = item.get("score") if isinstance(item, Mapping) else None
        if not isinstance(score, int) or score < 0 or score > 3:
            return None
        scores.append(score)
    if expected_count is not None and len(scores) != expected_count:
        return None
    return scores


def render_validation_number(value: Any) -> str:
    if value is None:
        return "—"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if number.is_integer():
        return str(int(number))
    return f"{number:.1f}"
