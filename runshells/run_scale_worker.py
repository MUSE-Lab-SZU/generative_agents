#!/usr/bin/env python3
"""
量表评估 worker — 由 run_experiment.py 通过子进程调用。

用法:
  python runshells/run_scale_worker.py \
    --cp-name Counsel-MILD-DYN-0512-1747 \
    --snapshot simulate-001.json \
    --agent 卡布达 \
    --question-file PHQ-9.jsonl \
    --output /path/to/scale_PHQ-9_pre_answered.jsonl

加载指定 checkpoint 快照，让 agent 回答量表题目，保存 answered.jsonl。
"""
import argparse
import json
import os
import sys

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SCALE_AGENT_DIR = os.path.join(BASE_DIR, "customization", "depression_scale_agent")
SCALE_QUESTIONS_DIR = os.path.join(SCALE_AGENT_DIR, "questions", "templates")
CHECKPOINTS_ROOT = os.path.join(BASE_DIR, "results", "checkpoints")

sys.path.insert(0, SCALE_AGENT_DIR)
# mock gradio for CLI
import types
if "gradio" not in sys.modules:
    sys.modules["gradio"] = types.ModuleType("gradio")

from app import ChatSession, iter_jsonl, resolve_question


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cp-name", required=True)
    parser.add_argument("--snapshot", required=True, help="快照文件名")
    parser.add_argument("--agent", required=True, help="agent 名称")
    parser.add_argument("--question-file", required=True, help="题目文件名 (如 PHQ-9.jsonl)")
    parser.add_argument("--output", required=True, help="输出 answered.jsonl 路径")
    args = parser.parse_args()

    question_path = os.path.join(SCALE_QUESTIONS_DIR, args.question_file)
    if not os.path.exists(question_path):
        print(f"[ERROR] 题目文件不存在: {question_path}")
        sys.exit(1)

    questions = list(iter_jsonl(question_path))
    print(f"[INFO] 加载 {len(questions)} 道题目 from {args.question_file}")

    try:
        session = ChatSession(args.cp_name, snapshot_file=args.snapshot)
    except Exception as e:
        print(f"[ERROR] ChatSession 初始化失败: {e}")
        sys.exit(1)

    try:
        session.set_agent(args.agent)
    except Exception as e:
        print(f"[ERROR] set_agent 失败: {e}")
        sys.exit(1)

    answers = []
    for idx, item in enumerate(questions):
        question = resolve_question(item)
        try:
            reply = session.answer_without_memory(question)
            record = dict(item) if isinstance(item, dict) else {"id": idx + 1, "question": question}
            record["answer"] = reply
            answers.append(record)
            print(f"[Q{idx+1}/{len(questions)}] done")
        except Exception as e:
            print(f"[Q{idx+1}/{len(questions)}] ERROR: {e}")
            record = dict(item) if isinstance(item, dict) else {"id": idx + 1, "question": question}
            record["answer"] = f"[ERROR] {e}"
            answers.append(record)

    with open(args.output, "w", encoding="utf-8") as f:
        for a in answers:
            f.write(json.dumps(a, ensure_ascii=False) + "\n")
    print(f"[DONE] {len(answers)} answers saved to {args.output}")


if __name__ == "__main__":
    main()
