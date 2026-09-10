"""Shared scale protocol metadata and parsing primitives used by evaluation CLIs."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any


OVERALL_DEPRESSION_SCALE = "总体抑郁水平及干扰程度量表"
OVERALL_ANXIETY_SCALE = "总体焦虑水平及干扰程度量表"

LONG_SCALE_NAMES = ("PHQ-9", "BDI-II")
SHORT_SCALE_NAMES = (OVERALL_DEPRESSION_SCALE, OVERALL_ANXIETY_SCALE)

SCALE_SPECS = {
    "PHQ-9": {
        "question_file": "PHQ-9-v2.jsonl",
        "scoring_prompt": "PHQ-9评估提示词.md",
        "item_score_key": "phq9_scores",
        "item_count": 9,
        "max_item_score": 3,
        "score_range": (0.0, 27.0),
    },
    "BDI-II": {
        "question_file": "BDI-II-v2.jsonl",
        "scoring_prompt": "BDI-II评估提示词.md",
        "item_score_key": "bdi_ii_scores",
        "item_count": 21,
        "max_item_score": 3,
        "score_range": (0.0, 63.0),
    },
    OVERALL_DEPRESSION_SCALE: {
        "question_file": f"{OVERALL_DEPRESSION_SCALE}.jsonl",
        "scoring_prompt": f"{OVERALL_DEPRESSION_SCALE}评估提示词.md",
        "item_score_key": "overall_depression_interference_scores",
        "item_count": 5,
        "max_item_score": 4,
        "score_range": (0.0, 20.0),
    },
    OVERALL_ANXIETY_SCALE: {
        "question_file": f"{OVERALL_ANXIETY_SCALE}.jsonl",
        "scoring_prompt": f"{OVERALL_ANXIETY_SCALE}评估提示词.md",
        "item_score_key": "overall_anxiety_interference_scores",
        "item_count": 5,
        "max_item_score": 4,
        "score_range": (0.0, 20.0),
    },
}


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


def extract_direct_answer_score(answer: str, *, max_score: int = 3) -> tuple[int | None, str]:
    """Return ``(score, status)`` where status is direct/ambiguous/none."""

    if max_score not in {3, 4}:
        raise ValueError("max_score must be 3 or 4")
    normalized_answer = str(answer or "")
    score_class = "0-3零一二两三" if max_score == 3 else "0-4零一二两三四"
    patterns = tuple(
        pattern.replace("0-3零一二两三", score_class)
        for pattern in DIRECT_SCORE_PATTERNS
    )
    score_values = dict(CN_SCORE_VALUES)
    if max_score == 4:
        score_values.update({"4": 4, "四": 4})
    hits: list[tuple[int, int, int]] = []
    for pattern in patterns:
        for match in re.finditer(pattern, normalized_answer):
            context = normalized_answer[max(0, match.start() - 6) : match.end() + 6]
            if any(marker in context for marker in NEGATED_SELECTION_MARKERS):
                continue
            score = score_values.get(match.group(1))
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


def extract_unambiguous_direct_answer_score(
    answer: str,
    *,
    max_score: int = 3,
) -> tuple[int | None, str]:
    """Accept a direct score only when all explicit selections agree."""

    answer_score, status = extract_direct_answer_score(answer, max_score=max_score)
    if status != "direct" or answer_score is None:
        return None, status

    score_values = dict(CN_SCORE_VALUES)
    if max_score == 4:
        score_values.update({"4": 4, "四": 4})
    score_class = "0-3零一二两三" if max_score == 3 else "0-4零一二两三四"
    explicit_tokens = re.findall(
        rf"(?:选|选择)\s*(?:了)?\s*([{score_class}])|([{score_class}])\s*分",
        str(answer or ""),
    )
    explicit_scores = {
        score_values[token]
        for groups in explicit_tokens
        for token in groups
        if token
    }
    if len(explicit_scores) > 1:
        return None, "ambiguous"
    return answer_score, status


def extract_item_scores(
    scored_result: Mapping[str, Any] | None,
    item_key: str | None,
    *,
    expected_count: int | None = None,
    max_score: int = 3,
) -> list[int] | None:
    if not item_key or not isinstance(scored_result, Mapping):
        return None
    items = scored_result.get(item_key)
    if not isinstance(items, list):
        return None

    scores: list[int] = []
    for item in items:
        score = item.get("score") if isinstance(item, Mapping) else None
        if not isinstance(score, int) or score < 0 or score > max_score:
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
