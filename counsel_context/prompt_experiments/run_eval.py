#!/usr/bin/env python3
"""Prompt 优化评估脚本。

对 labeled_profiles.jsonl 中的案例用指定版本的 prompt 跑量表评分，
输出结果 JSON，用于对比不同 prompt 版本的准确率。

不修改任何项目现有文件。

用法：
  python run_eval.py --scale PHQ-9 --output results/baseline_phq9.json
  python run_eval.py --scale BDI-II --prompt-dir prompts/v1_improved --output results/v1_bdi2.json
  python run_eval.py --scale SDS --limit 10  # 只跑前 10 条快速测试
"""
import argparse
import json
import os
import re
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from customization.depression_scale_agent.ExpertLLM import ExpertLLM  # noqa: E402

LABELED_PATH = os.path.join(PROJECT_ROOT, "Datasets", "analysis", "labeled_profiles.jsonl")

SCALE_CONFIG = {
    "PHQ-9": {
        "prompt_file": "PHQ-9评估提示词.md",
        "severity_map": {
            "无抑郁": "无",
            "轻度抑郁": "轻度",
            "中度抑郁": "中度",
            "中重度抑郁": "中重度",
            "重度抑郁": "重度",
        },
        "total_key": "total_score",
        "severity_key": "severity",
    },
    "BDI-II": {
        "prompt_file": "BDI-II评估提示词.md",
        "severity_map": {
            "无抑郁": "无",
            "轻度抑郁": "轻度",
            "中度抑郁": "中度",
            "重度抑郁": "重度",
        },
        "total_key": "total_score",
        "severity_key": "severity",
    },
    "SDS": {
        "prompt_file": "SDS评估提示词.md",
        "severity_map": {
            "正常": "无",
            "轻度抑郁": "轻度",
            "中度抑郁": "中度",
            "重度抑郁": "重度",
        },
        "total_key": "standard_score",
        "severity_key": "severity_by_standard_score",
    },
}


def load_cases(path, limit=None):
    cases = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            cases.append(json.loads(line))
    if limit:
        cases = cases[:limit]
    return cases


def load_prompt(prompt_dir, scale_name):
    cfg = SCALE_CONFIG[scale_name]
    prompt_path = os.path.join(prompt_dir, cfg["prompt_file"])
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read()


def extract_json(text):
    """从 LLM 输出中提取 JSON。"""
    if not text:
        return None
    text = text.strip()
    # 直接解析
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # 去掉 markdown 代码块
    m = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    # 找第一个 { 到最后一个 }
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass
    return None


def normalize_severity(raw, scale_name):
    """将 LLM 输出的严重度字符串映射到标注用的 轻度/中度/重度。"""
    cfg = SCALE_CONFIG[scale_name]
    raw = str(raw or "").strip()
    for key, val in cfg["severity_map"].items():
        if key in raw:
            return val
    if "轻度" in raw or "mild" in raw.lower():
        return "轻度"
    if "中度" in raw or "moderate" in raw.lower():
        return "中度"
    if "重度" in raw or "severe" in raw.lower():
        return "重度"
    if "无" in raw or "正常" in raw or "none" in raw.lower():
        return "无"
    return raw


def evaluate_case(llm, case, scoring_prompt, scale_name):
    """对单条案例跑评分。"""
    user_text = case.get("original_text", "")
    if not user_text.strip():
        return {"status": "empty_text", "parsed": None}

    user_prompt = f"""以下是评估模式说明：
当前输入不是结构化量表问答，而是一段患者的自述/案例描述。请从中提取与各条目相关的信息进行评分。
如果某条目在自述中没有对应信息，按0分处理。

患者自述/案例描述：
{user_text}

{scoring_prompt}"""

    try:
        raw_reply = llm.generate(user_prompt, temperature=0.2)
    except Exception as exc:
        return {"status": "llm_error", "error": str(exc), "parsed": None}

    parsed = extract_json(raw_reply)
    if parsed is None:
        return {"status": "parse_error", "raw_reply": raw_reply[:500], "parsed": None}

    cfg = SCALE_CONFIG[scale_name]
    total_score = parsed.get(cfg["total_key"])
    raw_severity = parsed.get(cfg["severity_key"], "")
    norm_severity = normalize_severity(raw_severity, scale_name)

    return {
        "status": "success",
        "total_score": total_score,
        "severity_raw": raw_severity,
        "severity_norm": norm_severity,
        "parsed": parsed,
    }


def main():
    parser = argparse.ArgumentParser(description="Prompt 优化离线评估")
    parser.add_argument("--scale", required=True, choices=["PHQ-9", "BDI-II", "SDS"], help="量表名称")
    parser.add_argument("--prompt-dir", default=None, help="prompt 版本目录（默认 prompts/v0_baseline/）")
    parser.add_argument("--output", default=None, help="输出 JSON 路径")
    parser.add_argument("--limit", type=int, default=None, help="只跑前 N 条")
    parser.add_argument("--delay", type=float, default=0.5, help="每次 API 调用间隔秒数")
    args = parser.parse_args()

    prompt_dir = args.prompt_dir or os.path.join(SCRIPT_DIR, "prompts", "v0_baseline")
    output_path = args.output or os.path.join(
        SCRIPT_DIR, "results",
        f"{os.path.basename(prompt_dir)}_{args.scale.replace('-', '')}.json"
    )

    print(f"量表: {args.scale}")
    print(f"Prompt: {prompt_dir}")
    print(f"数据: {LABELED_PATH}")
    print(f"输出: {output_path}")

    scoring_prompt = load_prompt(prompt_dir, args.scale)
    cases = load_cases(LABELED_PATH, args.limit)
    print(f"案例数: {len(cases)}")

    llm = ExpertLLM()
    results = []
    ok_count = 0
    err_count = 0

    for i, case in enumerate(cases):
        case_id = case.get("id", i)
        label = case.get("severity", "?")
        print(f"  [{i + 1}/{len(cases)}] id={case_id} label={label} ... ", end="", flush=True)

        result_entry = {
            "id": case_id,
            "label_severity": label,
            "label_sds_score": case.get("sds_score"),
        }

        eval_result = evaluate_case(llm, case, scoring_prompt, args.scale)
        result_entry["result"] = eval_result

        if eval_result["status"] == "success":
            ok_count += 1
            pred = eval_result.get("severity_norm", "?")
            total = eval_result.get("total_score", "?")
            match = "✓" if pred == label else "✗"
            print(f"pred={pred} score={total} {match}")
        else:
            err_count += 1
            print(f"ERROR: {eval_result['status']}")

        results.append(result_entry)

        if args.delay > 0:
            time.sleep(args.delay)

    # 汇总统计
    severity_correct = sum(
        1 for r in results
        if r["result"]["status"] == "success" and r["result"].get("severity_norm") == r["label_severity"]
    )
    severity_total = sum(1 for r in results if r["result"]["status"] == "success")
    accuracy = severity_correct / severity_total if severity_total > 0 else 0

    summary = {
        "scale": args.scale,
        "prompt_dir": prompt_dir,
        "total_cases": len(cases),
        "success": ok_count,
        "errors": err_count,
        "severity_accuracy": round(accuracy, 3),
        "severity_correct": severity_correct,
        "severity_total": severity_total,
    }
    print(f"\n汇总: {ok_count} 成功 / {err_count} 失败")
    print(f"严重度准确率: {severity_correct}/{severity_total} = {accuracy:.1%}")

    output = {"summary": summary, "results": results}
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"已保存: {output_path}")


if __name__ == "__main__":
    main()
