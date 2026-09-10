#!/usr/bin/env python3
"""
评分 worker — 由 run_experiment.py 通过子进程调用。

用法:
  python runshells/run_score_worker.py \
    --answers /path/to/scale_XX_answered.jsonl \
    --scoring-prompt /path/to/评估提示词.md \
    --output /path/to/scale_XX_scored.json

用 ExpertLLM (DeepSeek API) 对量表问答进行评分。
"""
import argparse
import json
import os
import re
import sys
import time

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "customization", "depression_scale_agent"))

from ExpertLLM import ExpertLLM
from modules.model.llm_model import (
    format_call_error_details,
    safe_exception_message_for_log,
)
from modules.model.api_cost import configure_tracking, rebuild_summary
from runshells.scale_protocol import (
    SCALE_SPECS,
    SHORT_SCALE_NAMES,
    extract_unambiguous_direct_answer_score,
)


def parse_json_reply(reply):
    """Parse a JSON response, accepting explanatory text before a fenced block."""
    text = reply.strip()
    fenced_json = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    if fenced_json:
        text = fenced_json.group(1).strip()
    return json.loads(text)


def extract_short_scale_direct_scores(answers, scale_name):
    """Return every unambiguous patient-selected score for a short scale."""

    if scale_name not in SHORT_SCALE_NAMES:
        return {}
    spec = SCALE_SPECS[scale_name]
    expected_count = int(spec["item_count"])
    max_score = int(spec["max_item_score"])
    scores = {}
    for row in answers:
        item_id = row.get("id") if isinstance(row, dict) else None
        if not isinstance(item_id, int) or isinstance(item_id, bool):
            continue
        if item_id < 1 or item_id > expected_count or item_id in scores:
            continue
        score, status = extract_unambiguous_direct_answer_score(
            str(row.get("answer", "") or ""),
            max_score=max_score,
        )
        if status == "direct" and score is not None:
            scores[item_id] = int(score)
    return scores


def _direct_score_metadata(direct_scores):
    return [
        {"item_id": item_id, "score": score, "score_source": "direct_answer_regex"}
        for item_id, score in sorted(direct_scores.items())
    ]


def build_direct_short_scale_result(scale_name, direct_scores):
    """Build the existing scorer JSON shape without an ExpertLLM call."""

    spec = SCALE_SPECS[scale_name]
    item_key = str(spec["item_score_key"])
    expected_count = int(spec["item_count"])
    items = [
        {
            "item": f"{scale_name}{item_id}",
            "score": int(direct_scores[item_id]),
            "basis": "患者明确回答的正则提取",
            "score_source": "direct_answer_regex",
        }
        for item_id in range(1, expected_count + 1)
    ]
    return {
        "scale": scale_name,
        item_key: items,
        "total_score": sum(item["score"] for item in items),
        "severity": "未设置分级",
        "scoring_mode": "direct_answer_regex_only",
        "llm_fallback_used": False,
        "direct_answer_scores": _direct_score_metadata(direct_scores),
        "direct_answer_count": expected_count,
        "direct_answer_override_count": 0,
    }


def apply_direct_short_scale_scores(result, scale_name, direct_scores):
    """Keep LLM scores only for unresolved items and audit direct overrides."""

    if not isinstance(result, dict) or scale_name not in SHORT_SCALE_NAMES:
        return result
    spec = SCALE_SPECS[scale_name]
    item_key = str(spec["item_score_key"])
    expected_count = int(spec["item_count"])
    max_score = int(spec["max_item_score"])
    items = result.get(item_key)
    if not isinstance(items, list) or len(items) != expected_count:
        return result

    override_count = 0
    for item_id, direct_score in direct_scores.items():
        item = items[item_id - 1]
        if not isinstance(item, dict):
            return result
        llm_score = item.get("score")
        if llm_score != direct_score:
            override_count += 1
        item["score"] = int(direct_score)
        item["basis"] = "患者明确回答的正则提取（覆盖 LLM 评分）"
        item["score_source"] = "direct_answer_regex"

    for index, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            return result
        score = item.get("score")
        if isinstance(score, bool) or not isinstance(score, int) or score < 0 or score > max_score:
            return result
        if index not in direct_scores:
            item["score_source"] = "llm_fallback"

    result["total_score"] = sum(int(item["score"]) for item in items)
    result["scoring_mode"] = "llm_fallback_with_direct_answer_overrides"
    result["llm_fallback_used"] = True
    result["direct_answer_scores"] = _direct_score_metadata(direct_scores)
    result["direct_answer_count"] = len(direct_scores)
    result["direct_answer_override_count"] = override_count
    return result


def write_result(path, result):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--answers", required=True, help="answered.jsonl 路径")
    parser.add_argument("--scoring-prompt", required=True, help="评分提示词 md 路径")
    parser.add_argument("--output", required=True, help="输出 scored json 路径")
    parser.add_argument("--run-name", default="", help="源独立实验 run_name")
    parser.add_argument(
        "--phase",
        choices=("simulation", "repeat_evaluation"),
        default="simulation",
        help="API 成本归属阶段",
    )
    parser.add_argument("--experiment-data-root", default="", help="成本账本所在 experiment_data")
    parser.add_argument("--evaluation-id", default="", help="一次双量表复评的稳定标识")
    parser.add_argument("--scale-name", default="", help="当前量表名称")
    parser.add_argument("--model", default="", help="继承 forced_llm.model")
    parser.add_argument("--base-url", default="", help="继承 forced_llm.base_url")
    parser.add_argument("--thinking-json", default="", help="继承 forced_llm.thinking 的 JSON")
    parser.add_argument("--reasoning-effort", default="", help="继承 forced_llm.reasoning_effort")
    args = parser.parse_args()

    cost_root = args.experiment_data_root or os.getenv("GA_API_COST_EXPERIMENT_DATA_ROOT", "")
    tracked_run = args.run_name or os.getenv("GA_API_COST_RUN_NAME", "")
    if tracked_run:
        configure_tracking(
            tracked_run,
            args.phase,
            experiment_data_root=cost_root or os.path.join(BASE_DIR, "results", "experiment_data"),
            evaluation_id=args.evaluation_id,
            scale_name=args.scale_name,
            rebuild=False,
        )

    # 读取问答
    answers = []
    with open(args.answers, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                answers.append(json.loads(line))
    print(f"[INFO] {len(answers)} 条问答 from {os.path.basename(args.answers)}")

    direct_scores = extract_short_scale_direct_scores(answers, args.scale_name)
    short_scale_spec = SCALE_SPECS.get(args.scale_name, {})
    expected_short_items = int(short_scale_spec.get("item_count", 0))
    if args.scale_name in SHORT_SCALE_NAMES and len(direct_scores) == expected_short_items:
        result = build_direct_short_scale_result(args.scale_name, direct_scores)
        if tracked_run:
            rebuild_summary(
                tracked_run,
                experiment_data_root=cost_root or os.path.join(BASE_DIR, "results", "experiment_data"),
            )
        write_result(args.output, result)
        print("[INFO] 短量表全部题目均为明确回答，跳过 ExpertLLM 评分。")
        print(f"[DONE] 评分结果保存到 {args.output}")
        return

    # 读取评分 prompt
    with open(args.scoring_prompt, "r", encoding="utf-8") as f:
        system_prompt = f.read()

    # 格式化问答
    qa_lines = []
    for a in answers:
        qid = a.get("id", "?")
        q_text = a.get("question", "")
        a_text = a.get("answer", "")
        qa_lines.append(f"Q{qid}: {q_text}\n回答: {a_text}")
    user_prompt = "\n\n".join(qa_lines)

    # 调用 ExpertLLM
    print(f"[INFO] 调用 ExpertLLM...")
    llm_started_at = time.monotonic()
    try:
        expert_kwargs = {}
        if args.model:
            expert_kwargs["model"] = args.model
        if args.base_url:
            expert_kwargs["base_url"] = args.base_url
        if args.thinking_json:
            expert_kwargs["thinking"] = json.loads(args.thinking_json)
        if args.reasoning_effort:
            expert_kwargs["reasoning_effort"] = args.reasoning_effort
        llm = ExpertLLM(**expert_kwargs)
        reply = llm.generate(
            user_prompt,
            system_prompt=system_prompt,
            caller="repeat_eval_score",
        )
        if not reply:
            print(f"[ERROR] ExpertLLM 返回空")
            sys.exit(1)
    except Exception as e:
        details = format_call_error_details(
            e,
            caller="repeat_eval_score",
            stage="request",
            provider="openai",
            model=args.model or os.getenv("EXPERT_LLM_MODEL") or "deepseek-v4-flash",
            base_url=args.base_url or os.getenv("EXPERT_LLM_BASE_URL") or "https://api.deepseek.com",
            attempt=1,
            total_attempts=1,
            retrying=False,
            elapsed_ms=(time.monotonic() - llm_started_at) * 1000,
        )
        print(
            "[ERROR] ExpertLLM 调用失败: {} | [LLM_CALL_ERROR] {}".format(
                safe_exception_message_for_log(e),
                details,
            )
        )
        if tracked_run:
            rebuild_summary(
                tracked_run,
                experiment_data_root=cost_root or os.path.join(BASE_DIR, "results", "experiment_data"),
            )
        sys.exit(1)

    if tracked_run:
        rebuild_summary(
            tracked_run,
            experiment_data_root=cost_root or os.path.join(BASE_DIR, "results", "experiment_data"),
        )

    # 解析 JSON
    try:
        result = parse_json_reply(reply)
    except json.JSONDecodeError as e:
        # 保存原始回复以供调试
        result = {"raw_reply": reply, "parse_error": str(e)}
        print(f"[WARN] JSON 解析失败: {e}")

    result = apply_direct_short_scale_scores(result, args.scale_name, direct_scores)
    write_result(args.output, result)
    print(f"[DONE] 评分结果保存到 {args.output}")


if __name__ == "__main__":
    main()
