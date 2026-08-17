#!/usr/bin/env python3
"""治疗后量表补充评估脚本。

直接运行方式：
    python3 runshells/run_extra_scale_eval.py

也支持用命令行覆盖部分参数，例如：
    python3 runshells/run_extra_scale_eval.py --name sim-test --agent 卡布达 --suffix extra2

说明：
- 默认读取 results/checkpoints/<name>/ 下的最后一个快照，做治疗后量表评估。
- 输出目录与 runshells/run_one_experiment.py 保持一致：
  results/experiment_data/<name>/scales/
- 为了和 run_one_experiment.py 的默认输出区分，文件名会额外带上 suffix。
- SCALES 可自由增删，不限制量表数量；每个量表可通过 repeat 控制重复测试次数。
"""

from __future__ import annotations

import argparse
import json
import sys
import types
from dataclasses import dataclass
from pathlib import Path


# ============================================================
# ↓↓↓ 可调参数：直接修改这里即可（中文注释）↓↓↓
# ============================================================

# 实验名称；必须对应 results/checkpoints/<name>
RUN_NAME = "sim-init-0516"

# 量表评估的目标角色
SCALE_AGENT = "卡布达"

# 指定快照文件名；留空表示自动取最后一个 simulate-*.json
SNAPSHOT_NAME = ""

# 输出文件后缀，用于和 run_one_experiment.py 生成的量表文件区分
OUTPUT_SUFFIX = "仿真后测试"

# 是否只打印流程，不实际执行
DRY_RUN = False

# 量表配置：
# - question_file: 题目 json/jsonl，支持：
#   1) templates 目录下文件名（如 PHQ-9-v2.jsonl）
#   2) 仓库相对路径
#   3) 绝对路径
# - scoring_prompt: 评分提示词，支持：
#   1) scoring_prompts 目录下文件名
#   2) 仓库相对路径
#   3) 绝对路径
# - repeat: 同一量表重复测试次数
SCALES = {
    "PHQ-9": {
        "question_file": "PHQ-9-v2.jsonl",
        "scoring_prompt": "PHQ-9评估提示词.md",
        "repeat": 1,
    },
    "BDI-II": {
        "question_file": "BDI-II-v2.jsonl",
        "scoring_prompt": "BDI-II评估提示词.md",
        "repeat": 1,
    },
    "SDS": {
        "question_file": "SDS-v2.jsonl",
        "scoring_prompt": "SDS评估提示词.md",
        "repeat": 1,
    },
}

# ============================================================
# ↑↑↑ 可调参数：直接修改这里即可（中文注释）↑↑↑
# ============================================================


BASE_DIR = Path(__file__).resolve().parent.parent
CHECKPOINTS_ROOT = BASE_DIR / "results" / "checkpoints"
EXPERIMENT_DATA_ROOT = BASE_DIR / "results" / "experiment_data"
SCALE_AGENT_DIR = BASE_DIR / "customization" / "depression_scale_agent"
SCALE_QUESTIONS_DIR = SCALE_AGENT_DIR / "questions" / "templates"
SCALE_SCORING_DIR = SCALE_AGENT_DIR / "questions" / "scoring_prompts"

sys.path.insert(0, str(SCALE_AGENT_DIR))
if "gradio" not in sys.modules:
    sys.modules["gradio"] = types.ModuleType("gradio")


@dataclass
class RuntimeConfig:
    name: str
    agent: str
    snapshot: str
    suffix: str
    dry_run: bool



def load_scale_chat_components():
    from app import ChatSession, iter_jsonl, resolve_question

    return ChatSession, iter_jsonl, resolve_question



def load_expert_llm_class():
    from ExpertLLM import ExpertLLM

    return ExpertLLM



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run extra post-simulation scale evaluation")
    parser.add_argument("--name", default=None, help="覆盖脚本前面的 RUN_NAME")
    parser.add_argument("--agent", default=None, help="覆盖脚本前面的 SCALE_AGENT")
    parser.add_argument("--snapshot", default=None, help="指定快照文件名；留空则自动取最后一个快照")
    parser.add_argument("--suffix", default=None, help="输出文件后缀，用于和默认量表输出区分")
    parser.add_argument("--dry-run", action="store_true", help="只打印操作，不实际执行")
    return parser.parse_args()



def normalize_suffix(raw: str) -> str:
    suffix = raw.strip().replace(" ", "_").replace("/", "_").replace("\\", "_")
    suffix = suffix.strip("_")
    if not suffix:
        raise ValueError("OUTPUT_SUFFIX / --suffix 不能为空。")
    return suffix



def resolve_runtime_config(args: argparse.Namespace) -> RuntimeConfig:
    name = (args.name if args.name is not None else RUN_NAME).strip()
    if not name:
        raise ValueError("必须提供 RUN_NAME 或 --name。")

    agent = (args.agent if args.agent is not None else SCALE_AGENT).strip()
    if not agent:
        raise ValueError("必须提供 SCALE_AGENT 或 --agent。")

    snapshot = (args.snapshot if args.snapshot is not None else SNAPSHOT_NAME).strip()
    suffix = normalize_suffix(args.suffix if args.suffix is not None else OUTPUT_SUFFIX)

    return RuntimeConfig(
        name=name,
        agent=agent,
        snapshot=snapshot,
        suffix=suffix,
        dry_run=bool(DRY_RUN or args.dry_run),
    )



def ensure_output_dirs(name: str, *, dry_run: bool) -> Path:
    output_dir = EXPERIMENT_DATA_ROOT / name
    scales_dir = output_dir / "scales"
    if dry_run:
        print(f"[DRY-RUN] ensure dirs: {output_dir} / {scales_dir}")
        return scales_dir
    scales_dir.mkdir(parents=True, exist_ok=True)
    return scales_dir



def find_last_snapshot_name(name: str) -> str:
    checkpoint_dir = CHECKPOINTS_ROOT / name
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"checkpoint directory not found: {checkpoint_dir}")
    snapshots = sorted(
        p.name for p in checkpoint_dir.iterdir()
        if p.name.startswith("simulate-") and p.name.endswith(".json")
    )
    if not snapshots:
        raise FileNotFoundError(f"no simulate-*.json found under: {checkpoint_dir}")
    return snapshots[-1]



def resolve_scale_file(raw_path: str, *, default_dir: Path) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path

    repo_relative = BASE_DIR / path
    if repo_relative.exists():
        return repo_relative

    return default_dir / raw_path



def get_repeat_count(scale_cfg: dict) -> int:
    repeat = scale_cfg.get("repeat", 1)
    try:
        repeat = int(repeat)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid repeat for scale config: {repeat}") from exc
    return max(1, repeat)



def build_output_stem(scale_name: str, *, suffix: str, repeat_idx: int, repeat_count: int) -> str:
    parts = [scale_name, "post", suffix]
    if repeat_count > 1:
        parts.append(f"r{repeat_idx:02d}")
    return "_".join(parts)



def to_relative(path: Path) -> str:
    try:
        return str(path.relative_to(BASE_DIR))
    except ValueError:
        return str(path)



def load_questions(question_path: Path) -> list[dict]:
    _, iter_jsonl, _ = load_scale_chat_components()
    return list(iter_jsonl(str(question_path)))



def answer_scale_questions(cp_name: str, snapshot_name: str, agent_name: str, question_path: Path) -> list[dict]:
    ChatSession, _, resolve_question = load_scale_chat_components()
    questions = load_questions(question_path)
    print(f"[INFO] 加载 {len(questions)} 道题目 from {question_path}")

    try:
        session = ChatSession(cp_name, snapshot_file=snapshot_name)
    except Exception as exc:
        raise RuntimeError(f"ChatSession 初始化失败: {exc}") from exc

    try:
        session.set_agent(agent_name)
    except Exception as exc:
        raise RuntimeError(f"set_agent 失败: {exc}") from exc

    answers: list[dict] = []
    for idx, item in enumerate(questions, start=1):
        question = resolve_question(item)
        record = dict(item) if isinstance(item, dict) else {"id": idx, "question": question}
        try:
            reply = session.answer_without_memory(question)
            record["answer"] = reply
            print(f"[Q{idx}/{len(questions)}] done")
        except Exception as exc:
            record["answer"] = f"[ERROR] {exc}"
            print(f"[Q{idx}/{len(questions)}] ERROR: {exc}")
        answers.append(record)

    return answers



def write_answered_jsonl(output_path: Path, answers: list[dict]) -> None:
    with output_path.open("w", encoding="utf-8") as f:
        for answer in answers:
            f.write(json.dumps(answer, ensure_ascii=False) + "\n")
    print(f"[DONE] {len(answers)} answers saved to {output_path}")



def score_answers(answers: list[dict], scoring_prompt_path: Path) -> dict:
    ExpertLLM = load_expert_llm_class()

    with scoring_prompt_path.open("r", encoding="utf-8") as f:
        system_prompt = f.read()

    qa_lines = []
    for answer in answers:
        qid = answer.get("id", "?")
        q_text = answer.get("question", "")
        a_text = answer.get("answer", "")
        qa_lines.append(f"Q{qid}: {q_text}\n回答: {a_text}")
    user_prompt = "\n\n".join(qa_lines)

    print("[INFO] 调用 ExpertLLM...")
    llm = ExpertLLM()
    reply = llm.generate(
        user_prompt,
        system_prompt=system_prompt,
        caller="extra_scale_score",
    )
    if not reply:
        raise RuntimeError("ExpertLLM 返回空")

    text = reply.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [line for line in lines if not line.strip().startswith("```")]
        text = "\n".join(lines)

    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        print(f"[WARN] JSON 解析失败: {exc}")
        return {"raw_reply": reply, "parse_error": str(exc)}



def write_scored_json(output_path: Path, scored: dict) -> None:
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(scored, f, ensure_ascii=False, indent=2)
    print(f"[DONE] 评分结果保存到 {output_path}")



def print_effective_config(cfg: RuntimeConfig) -> None:
    print("==========================================")
    print(" 量表补充评估配置")
    print("==========================================")
    print(f"  实验名称:     {cfg.name}")
    print(f"  评估角色:     {cfg.agent}")
    print(f"  指定快照:     {cfg.snapshot or '(自动取最后快照)'}")
    print(f"  文件后缀:     {cfg.suffix}")
    print(f"  dry-run:      {cfg.dry_run}")
    print("  SCALES:")
    for scale_name, scale_cfg in SCALES.items():
        repeat = get_repeat_count(scale_cfg)
        print(
            f"    - {scale_name}: question={scale_cfg['question_file']}, "
            f"prompt={scale_cfg['scoring_prompt']}, repeat={repeat}"
        )
    print("==========================================")



def run_extra_post_scales(cfg: RuntimeConfig) -> None:
    scales_dir = ensure_output_dirs(cfg.name, dry_run=cfg.dry_run)
    snapshot_name = cfg.snapshot or ("<latest-snapshot>" if cfg.dry_run else find_last_snapshot_name(cfg.name))

    aggregated = {
        "phase": "post",
        "suffix": cfg.suffix,
        "checkpoint": cfg.name,
        "snapshot": snapshot_name,
        "agent": cfg.agent,
        "scales": {},
    }

    for scale_name, scale_cfg in SCALES.items():
        question_path = resolve_scale_file(scale_cfg["question_file"], default_dir=SCALE_QUESTIONS_DIR)
        scoring_prompt_path = resolve_scale_file(scale_cfg["scoring_prompt"], default_dir=SCALE_SCORING_DIR)
        repeat_count = get_repeat_count(scale_cfg)

        if not cfg.dry_run and not question_path.exists():
            raise FileNotFoundError(f"题目文件不存在: {question_path}")
        if not cfg.dry_run and not scoring_prompt_path.exists():
            raise FileNotFoundError(f"评分提示词不存在: {scoring_prompt_path}")

        scale_runs = []
        for repeat_idx in range(1, repeat_count + 1):
            stem = build_output_stem(
                scale_name,
                suffix=cfg.suffix,
                repeat_idx=repeat_idx,
                repeat_count=repeat_count,
            )
            answers_path = scales_dir / f"{stem}_answered.jsonl"
            scored_path = scales_dir / f"{stem}_scored.json"

            print(f"\n[RUN] {scale_name} repeat {repeat_idx}/{repeat_count}")
            print(f"  question: {question_path}")
            print(f"  prompt:   {scoring_prompt_path}")
            print(f"  answer:   {answers_path}")
            print(f"  scored:   {scored_path}")

            run_meta = {
                "repeat": repeat_idx,
                "answer_file": to_relative(answers_path),
                "scored_file": to_relative(scored_path),
            }

            if cfg.dry_run:
                scale_runs.append(run_meta)
                continue

            answers = answer_scale_questions(cfg.name, snapshot_name, cfg.agent, question_path)
            write_answered_jsonl(answers_path, answers)

            scored = score_answers(answers, scoring_prompt_path)
            write_scored_json(scored_path, scored)
            run_meta["score"] = scored
            scale_runs.append(run_meta)

        aggregated["scales"][scale_name] = {
            "question_file": to_relative(question_path),
            "scoring_prompt": to_relative(scoring_prompt_path),
            "repeat": repeat_count,
            "runs": scale_runs,
        }

    aggregated_path = scales_dir / f"scale_scores_{cfg.suffix}.json"
    print(f"\n[WRITE] {aggregated_path}")
    if not cfg.dry_run:
        with aggregated_path.open("w", encoding="utf-8") as f:
            json.dump(aggregated, f, ensure_ascii=False, indent=2)



def main() -> None:
    args = parse_args()
    cfg = resolve_runtime_config(args)
    print_effective_config(cfg)
    run_extra_post_scales(cfg)
    print("\n[Done] 量表补充评估完成")


if __name__ == "__main__":
    main()
