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


def parse_json_reply(reply):
    """Parse a JSON response, accepting explanatory text before a fenced block."""
    text = reply.strip()
    fenced_json = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    if fenced_json:
        text = fenced_json.group(1).strip()
    return json.loads(text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--answers", required=True, help="answered.jsonl 路径")
    parser.add_argument("--scoring-prompt", required=True, help="评分提示词 md 路径")
    parser.add_argument("--output", required=True, help="输出 scored json 路径")
    args = parser.parse_args()

    # 读取问答
    answers = []
    with open(args.answers, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                answers.append(json.loads(line))
    print(f"[INFO] {len(answers)} 条问答 from {os.path.basename(args.answers)}")

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
        llm = ExpertLLM()
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
            model=os.getenv("EXPERT_LLM_MODEL") or "deepseek-chat",
            base_url=os.getenv("EXPERT_LLM_BASE_URL") or "https://api.deepseek.com",
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
        sys.exit(1)

    # 解析 JSON
    try:
        result = parse_json_reply(reply)
    except json.JSONDecodeError as e:
        # 保存原始回复以供调试
        result = {"raw_reply": reply, "parse_error": str(e)}
        print(f"[WARN] JSON 解析失败: {e}")

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"[DONE] 评分结果保存到 {args.output}")


if __name__ == "__main__":
    main()
