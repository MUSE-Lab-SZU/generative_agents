#!/usr/bin/env python3
"""Validate PHQ-9 / BDI-II scored scale results.

This script checks two things:
- JSON self-consistency: item counts, item score ranges, total_score arithmetic,
  severity labels, safety flags, and scale_scores.json aggregation.
- Conservative answer cross-checks: when an answered JSONL contains an explicit
  score choice, compare it with the LLM item score and flag differences.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SCALE_SPECS = {
    "PHQ-9": {
        "answered": "PHQ-9_post_answered.jsonl",
        "scored": "PHQ-9_post_scored.json",
        "items_key": "phq9_scores",
        "item_count": 9,
        "max_total": 27,
    },
    "BDI-II": {
        "answered": "BDI-II_post_answered.jsonl",
        "scored": "BDI-II_post_scored.json",
        "items_key": "bdi_ii_scores",
        "item_count": 21,
        "max_total": 63,
    },
}

CN_NUMBERS = {
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


@dataclass(frozen=True)
class CheckIssue:
    kind: str
    path: Path
    scale: str
    detail: str


@dataclass(frozen=True)
class AnswerFlag:
    path: Path
    scale: str
    item_id: int
    llm_score: Any
    extracted_score: int
    status: str
    evidence: str
    answer_excerpt: str


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def phq_severity(total: int | float) -> str:
    if total <= 4:
        return "无抑郁"
    if total <= 9:
        return "轻度抑郁"
    if total <= 14:
        return "中度抑郁"
    if total <= 19:
        return "中重度抑郁"
    return "重度抑郁"


def bdi_severity(total: int | float) -> str:
    if total <= 13:
        return "无抑郁"
    if total <= 19:
        return "轻度抑郁"
    if total <= 28:
        return "中度抑郁"
    return "重度抑郁"


def expected_severity(scale: str, total: int | float) -> str:
    return phq_severity(total) if scale == "PHQ-9" else bdi_severity(total)


def extract_score_item_scores(scored: dict[str, Any], scale: str) -> list[Any]:
    key = SCALE_SPECS[scale]["items_key"]
    items = scored.get(key)
    if not isinstance(items, list):
        return []
    return [item.get("score") if isinstance(item, dict) else None for item in items]


def validate_scored_file(path: Path, scale: str) -> tuple[list[CheckIssue], dict[str, Any]]:
    issues: list[CheckIssue] = []
    spec = SCALE_SPECS[scale]
    try:
        data = load_json(path)
    except Exception as exc:
        return [CheckIssue("load_error", path, scale, str(exc))], {}

    items = data.get(spec["items_key"])
    if not isinstance(items, list):
        issues.append(CheckIssue("missing_items", path, scale, f"missing {spec['items_key']}"))
        return issues, data

    if len(items) != spec["item_count"]:
        issues.append(
            CheckIssue(
                "wrong_item_count",
                path,
                scale,
                f"expected {spec['item_count']}, got {len(items)}",
            )
        )

    numeric_scores: list[int] = []
    for idx, item in enumerate(items, start=1):
        score = item.get("score") if isinstance(item, dict) else None
        if not isinstance(score, int) or score < 0 or score > 3:
            issues.append(CheckIssue("bad_item_score", path, scale, f"item {idx}: {score!r}"))
        else:
            numeric_scores.append(score)

    total = data.get("total_score")
    if not isinstance(total, (int, float)):
        issues.append(CheckIssue("bad_total_score", path, scale, f"total_score={total!r}"))
        return issues, data

    if len(numeric_scores) == spec["item_count"]:
        item_sum = sum(numeric_scores)
        if item_sum != total:
            corrected = expected_severity(scale, item_sum)
            issues.append(
                CheckIssue(
                    "total_mismatch",
                    path,
                    scale,
                    f"item sum={item_sum}, reported total_score={total}, corrected severity={corrected}",
                )
            )

    reported_severity = data.get("severity")
    expected = expected_severity(scale, total)
    if reported_severity != expected:
        issues.append(
            CheckIssue(
                "severity_mismatch",
                path,
                scale,
                f"expected {expected} from reported total={total}, got {reported_severity!r}",
            )
        )

    item9 = items[8].get("score") if len(items) >= 9 and isinstance(items[8], dict) else None
    risk = str(data.get("safety_risk") or "")
    if item9 == 0 and "未见" not in risk:
        issues.append(CheckIssue("safety_mismatch", path, scale, f"item9=0, safety_risk={risk!r}"))
    if isinstance(item9, int) and item9 >= 1 and "存在" not in risk:
        issues.append(CheckIssue("safety_mismatch", path, scale, f"item9={item9}, safety_risk={risk!r}"))

    return issues, data


def validate_aggregate(scales_dir: Path) -> list[CheckIssue]:
    issues: list[CheckIssue] = []
    aggregate_path = scales_dir / "scale_scores.json"
    if not aggregate_path.exists():
        return issues
    try:
        aggregate = load_json(aggregate_path)
    except Exception as exc:
        return [CheckIssue("aggregate_load_error", aggregate_path, "ALL", str(exc))]

    post = aggregate.get("post")
    if not isinstance(post, dict):
        return [CheckIssue("aggregate_bad_schema", aggregate_path, "ALL", "missing post object")]

    for scale, spec in SCALE_SPECS.items():
        scored_path = scales_dir / str(spec["scored"])
        if not scored_path.exists():
            continue
        scored = load_json(scored_path)
        if post.get(scale) != scored:
            issues.append(CheckIssue("aggregate_mismatch", aggregate_path, scale, "post entry differs from scored file"))
    return issues


def number_value(text: str) -> int:
    return CN_NUMBERS[text]


def uncertainty_near(text: str, start: int, end: int) -> bool:
    context = text[max(0, start - 18) : min(len(text), end + 28)]
    markers = [
        "或者",
        "不确定",
        "之间",
        "两三",
        "一两",
        "选0感觉是骗人的",
        "可能是",
        "也可能",
    ]
    return any(marker in context for marker in markers)


def extract_explicit_score(answer: str) -> tuple[int | None, str, str]:
    patterns = [
        r"(?:我)?\s*(?:会|想|大概|可能|应该|还是|就|其实)?\s*(?:选|选择)\s*(?:了)?\s*([0-3零一二两三])",
        r"([0-3零一二两三])\s*分",
        r"(?:大概|应该|可能|算是|算|就是|评分的话|打分的话|接近|更接近|我觉得|我想|我会|应该是|大概是|可能是|差不多)\s*[，,。\.……\s]*([0-3零一二两三])\s*(?:吧|。|，|,|$)",
        r"(?:^|[，,。\.……\s])([0-3零一二两三])\s*吧",
        r"^[\s（\(\）\)……。,.，、嗯唔]*([0-3零一二两三])\s*(?:吧|。|，|,|$)",
    ]
    hits: list[tuple[int, int, int, str]] = []
    for pattern in patterns:
        for match in re.finditer(pattern, answer):
            context = answer[max(0, match.start() - 6) : match.end() + 6]
            if any(marker in context for marker in ["不想选", "不是选", "不能选", "不敢选"]):
                continue
            hits.append((match.start(), match.end(), number_value(match.group(1)), match.group(0)))
    if not hits:
        return None, "none", ""
    hits.sort(key=lambda item: (item[0], item[1]))
    first = hits[0]
    ambiguous = uncertainty_near(answer, first[0], first[1])
    for hit in hits[1:]:
        if hit[0] - first[1] < 35 and hit[2] != first[2]:
            ambiguous = True
    return first[2], "ambiguous_explicit" if ambiguous else "direct", first[3]


def extract_phq_frequency_score(answer: str) -> tuple[int | None, str, str]:
    phrases = [
        (3, ["几乎每天", "几乎每一天", "每天都", "每天会", "每天"]),
        (2, ["超过一半", "一半以上", "大部分", "很多天", "多数天"]),
        (1, ["有几天", "几天", "有时候", "偶尔"]),
        (0, ["完全没有", "没有那样", "没有造成", "没有影响"]),
    ]
    found: list[tuple[int, int, str, str]] = []
    for score, phrase_list in phrases:
        for phrase in phrase_list:
            for match in re.finditer(re.escape(phrase), answer):
                context = answer[max(0, match.start() - 5) : match.end() + 5]
                if any(negation in context for negation in [f"不是{phrase}", f"没到{phrase}", f"不到{phrase}", f"没有{phrase}", f"不算{phrase}"]):
                    continue
                found.append((match.start(), score, phrase, context))
    if not found:
        return None, "none", ""
    if len({item[1] for item in found}) > 1:
        max_item = max(found, key=lambda item: item[1])
        return max_item[1], "ambiguous_frequency", max_item[2]
    found.sort(key=lambda item: item[0])
    return found[0][1], "frequency", found[0][2]


def extract_answer_score(answer: str, scale: str) -> tuple[int | None, str, str]:
    explicit = extract_explicit_score(answer)
    if explicit[0] is not None:
        return explicit
    if scale == "PHQ-9":
        return extract_phq_frequency_score(answer)
    return None, "none", ""


def cross_check_answers(answered_path: Path, scored_path: Path, scale: str) -> list[AnswerFlag]:
    flags: list[AnswerFlag] = []
    answers = load_jsonl(answered_path)
    scored = load_json(scored_path)
    item_scores = extract_score_item_scores(scored, scale)
    for row in answers:
        item_id = row.get("id")
        if not isinstance(item_id, int):
            continue
        if scale == "PHQ-9" and item_id == 10:
            continue
        if item_id < 1 or item_id > len(item_scores):
            continue
        answer = str(row.get("answer") or "")
        extracted, status, evidence = extract_answer_score(answer, scale)
        if extracted is None:
            continue
        llm_score = item_scores[item_id - 1]
        if extracted != llm_score:
            flags.append(
                AnswerFlag(
                    path=answered_path,
                    scale=scale,
                    item_id=item_id,
                    llm_score=llm_score,
                    extracted_score=extracted,
                    status=status,
                    evidence=evidence,
                    answer_excerpt=answer[:220].replace("\n", " / "),
                )
            )
    return flags


def find_scale_dirs(roots: list[Path]) -> list[Path]:
    dirs = set()
    for root in roots:
        for path in root.rglob("*_post_answered.jsonl"):
            dirs.add(path.parent)
        for path in root.rglob("*_post_scored.json"):
            dirs.add(path.parent)
    return sorted(dirs)


def short_path(path: Path) -> str:
    return str(path)


def render_markdown(
    roots: list[Path],
    scale_dirs: list[Path],
    missing: list[CheckIssue],
    issues: list[CheckIssue],
    answer_flags: list[AnswerFlag],
) -> str:
    scored_counts = {
        scale: sum(1 for directory in scale_dirs if (directory / str(spec["scored"])).exists())
        for scale, spec in SCALE_SPECS.items()
    }
    confident_answer_flags = [flag for flag in answer_flags if flag.status == "direct"]
    ambiguous_answer_flags = [flag for flag in answer_flags if flag.status != "direct"]
    total_mismatches = [issue for issue in issues if issue.kind == "total_mismatch"]

    lines = [
        "# PHQ-9 / BDI-II Score Validation",
        "",
        f"- Roots: {', '.join(str(root) for root in roots)}",
        f"- Scale directories found: {len(scale_dirs)}",
        f"- PHQ-9 scored files: {scored_counts['PHQ-9']}",
        f"- BDI-II scored files: {scored_counts['BDI-II']}",
        f"- Missing expected files: {len(missing)}",
        f"- JSON/self-consistency issues: {len(issues)}",
        f"- BDI-II total arithmetic mismatches: {len(total_mismatches)}",
        f"- Answer cross-check flags: {len(answer_flags)} ({len(confident_answer_flags)} confident, {len(ambiguous_answer_flags)} ambiguous)",
        "",
    ]

    if missing:
        lines.extend(["## Missing Files", "", "| Directory | Missing |", "|---|---|"])
        for issue in missing:
            lines.append(f"| `{short_path(issue.path)}` | {issue.detail} |")
        lines.append("")

    if issues:
        lines.extend(["## JSON / Arithmetic Issues", "", "| Kind | Scale | File | Detail |", "|---|---|---|---|"])
        for issue in issues:
            lines.append(f"| {issue.kind} | {issue.scale} | `{short_path(issue.path)}` | {issue.detail} |")
        lines.append("")

    if confident_answer_flags:
        lines.extend(
            [
                "## Answer Cross-Check Flags",
                "",
                "These are cases where the answer text contains a direct score cue that differs from the LLM item score.",
                "",
                "| Scale | Item | LLM | Extracted | Status | File | Evidence | Answer Excerpt |",
                "|---|---:|---:|---:|---|---|---|---|",
            ]
        )
        for flag in confident_answer_flags:
            lines.append(
                f"| {flag.scale} | {flag.item_id} | {flag.llm_score} | {flag.extracted_score} | "
                f"{flag.status} | `{short_path(flag.path)}` | {flag.evidence} | {flag.answer_excerpt} |"
            )
        lines.append("")

    if ambiguous_answer_flags:
        lines.extend(
            [
                "## Ambiguous Answer Flags",
                "",
                "These require human review; the answer mentions multiple possible scores or conflicting frequency cues.",
                "",
                "| Scale | Item | LLM | Extracted | Status | File | Evidence | Answer Excerpt |",
                "|---|---:|---:|---:|---|---|---|---|",
            ]
        )
        for flag in ambiguous_answer_flags:
            lines.append(
                f"| {flag.scale} | {flag.item_id} | {flag.llm_score} | {flag.extracted_score} | "
                f"{flag.status} | `{short_path(flag.path)}` | {flag.evidence} | {flag.answer_excerpt} |"
            )
        lines.append("")

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate PHQ-9 / BDI-II scored scale results")
    parser.add_argument("roots", nargs="+", type=Path, help="Result roots, e.g. results/0623 results/0630")
    parser.add_argument("--report", type=Path, default=None, help="Optional Markdown report path")
    args = parser.parse_args()

    scale_dirs = find_scale_dirs(args.roots)
    missing: list[CheckIssue] = []
    issues: list[CheckIssue] = []
    answer_flags: list[AnswerFlag] = []

    for directory in scale_dirs:
        for scale, spec in SCALE_SPECS.items():
            answered_path = directory / str(spec["answered"])
            scored_path = directory / str(spec["scored"])
            if not answered_path.exists():
                missing.append(CheckIssue("missing_file", directory, scale, str(spec["answered"])))
            if not scored_path.exists():
                missing.append(CheckIssue("missing_file", directory, scale, str(spec["scored"])))
                continue
            file_issues, _ = validate_scored_file(scored_path, scale)
            issues.extend(file_issues)
            if answered_path.exists():
                answer_flags.extend(cross_check_answers(answered_path, scored_path, scale))
        if not (directory / "scale_scores.json").exists():
            missing.append(CheckIssue("missing_file", directory, "ALL", "scale_scores.json"))
        issues.extend(validate_aggregate(directory))

    report = render_markdown(args.roots, scale_dirs, missing, issues, answer_flags)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
