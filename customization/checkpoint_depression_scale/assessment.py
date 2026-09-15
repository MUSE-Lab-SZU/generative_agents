"""量表、存档选择及可独立验证的计分流程（不依赖 Gradio 或模型）。"""

from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
CHECKPOINTS_ROOT = BASE_DIR / "results" / "checkpoints"
QUESTION_FILE = Path(__file__).with_name("总体抑郁水平及干扰程度量表.jsonl")
SCALE_NAME = "总体抑郁水平及干扰程度量表"
RESPONSE_INSTRUCTION = (
    '\n请以第一人称作答，严格使用“评分：N。理由：简短说明”的格式，'
    'N 必须是你选择的一个 0、1、2、3 或 4，不要复述其他选项。'
)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def safe_child(parent, name):
    if not isinstance(name, str) or not name or name in {".", ".."} or Path(name).name != name:
        raise ValueError("无效的存档、checkpoint 或角色名称。")
    child = (Path(parent) / name).resolve()
    if child.parent != Path(parent).resolve():
        raise ValueError("选择的路径不在允许目录内。")
    return child


def list_archives(root=CHECKPOINTS_ROOT):
    if not root.is_dir():
        return []
    return sorted(p.name for p in root.iterdir() if p.is_dir() and not p.is_symlink())


def list_checkpoints(archive, root=CHECKPOINTS_ROOT):
    directory = safe_child(root, archive)
    return sorted(p.name for p in directory.glob("simulate-*.json") if p.is_file() and not p.is_symlink())


def load_checkpoint(archive, checkpoint, root=CHECKPOINTS_ROOT):
    if checkpoint not in list_checkpoints(archive, root):
        raise ValueError(f"找不到 checkpoint：{checkpoint}")
    path = safe_child(safe_child(root, archive), checkpoint)
    config = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or not isinstance(config.get("agents"), dict):
        raise ValueError(f"checkpoint 缺少 agents：{checkpoint}")
    return config


def list_agents(archive, root=CHECKPOINTS_ROOT):
    names = set()
    for checkpoint in list_checkpoints(archive, root):
        names.update(load_checkpoint(archive, checkpoint, root)["agents"])
    return sorted(names)


def checkpoints_for_agent(archive, agent, root=CHECKPOINTS_ROOT):
    return [name for name in list_checkpoints(archive, root)
            if agent in load_checkpoint(archive, name, root)["agents"]]


def load_questions(path=QUESTION_FILE):
    questions = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if [q.get("id") for q in questions] != [1, 2, 3, 4, 5]:
        raise ValueError("量表必须按顺序包含 id 为 1–5 的五道题。")
    for question in questions:
        prompt = question.get("question")
        if not isinstance(prompt, str) or not all(f"{n}=" in prompt for n in range(5)):
            raise ValueError("每题必须包含完整的 0–4 选项。")
    return questions


def parse_score(reply):
    """只接受明确的结构化评分或单个数字，不从叙述中猜测分数。"""
    reply = str(reply or "").strip()
    if re.fullmatch(r"[0-4]", reply):
        return int(reply)
    match = re.fullmatch(r"评分\s*[:：]\s*([0-4])(?:\s*[。；;，,]\s*理由\s*[:：]\s*[^\n]+)?[。\s]*", reply)
    if match and len(re.findall(r"评分\s*[:：]", reply)) == 1:
        return int(match.group(1))
    return None


def evaluate_checkpoint(session, questions, record, progress=None):
    """逐题问答；保留所有尝试，缺失分数绝不参与总分。"""
    simulation_time = record.get("simulation_time")
    if isinstance(simulation_time, dict):
        simulation_time = simulation_time.get("start")
    time_context = f"当前模拟时间为 {simulation_time}，请以该时间为基准回顾过去一周。\n" if simulation_time else ""
    for question in questions:
        item = {"id": question["id"], "question": question["question"], "score": None, "attempts": []}
        record["items"].append(item)
        for attempt in range(2):
            if progress:
                progress(question["id"], attempt + 1)
            prompt = time_context + question["question"] + RESPONSE_INSTRUCTION
            if attempt:
                prompt = "上一条回答未能识别出唯一的有效评分，请重新回答本题。\n" + prompt
            turn = {"prompt": prompt, "reply": None, "timestamp": utc_now()}
            item["attempts"].append(turn)
            turn["reply"] = str(session.chat(prompt) or "")
            item["score"] = parse_score(turn["reply"])
            if item["score"] is not None:
                break


def run_batch(archive, agent, checkpoints, *, root=CHECKPOINTS_ROOT, session_factory=None, progress=None):
    """每完成一个 checkpoint 即落盘并 yield；单点失败后继续其余点。"""
    if not archive or not agent or not checkpoints:
        raise ValueError("请选择模拟存档、agent 和至少一个 checkpoint。")
    selected = sorted(set(checkpoints))
    configs = {name: load_checkpoint(archive, name, root) for name in selected}
    if any(agent not in config["agents"] for config in configs.values()):
        raise ValueError("所选 checkpoint 中有存档点不包含该 agent，请重新选择。")
    questions = load_questions()
    question_hash = hashlib.sha256(QUESTION_FILE.read_bytes()).hexdigest()
    if session_factory is None:
        from customization.checkpoint_depression_scale.runtime import checkpoint_session
        session_factory = checkpoint_session
    run_id = uuid.uuid4().hex
    output = safe_child(root, archive) / f"depression-scale-{run_id}.jsonl"
    with output.open("x", encoding="utf-8") as stream:
        for position, checkpoint in enumerate(selected, 1):
            config = configs[checkpoint]
            record = {
                "schema_version": 1, "type": "checkpoint_assessment", "run_id": run_id,
                "scale": SCALE_NAME, "question_file": QUESTION_FILE.name,
                "question_sha256": question_hash, "archive": archive, "agent": agent,
                "checkpoint": checkpoint, "simulation_time": config.get("time"),
                "simulation_step": config.get("step"), "started_at": utc_now(),
                "status": "running", "items": [], "total_score": None, "max_score": 20,
                "storage_mode": "isolated_copy", "errors": [],
            }
            try:
                if progress:
                    progress(position, len(selected), checkpoint, 0, 0)
                with session_factory(archive, checkpoint, agent, root=root) as session:
                    record["runtime"] = getattr(session, "metadata", {})
                    def on_question(question_id, attempt):
                        if progress:
                            progress(position, len(selected), checkpoint, question_id, attempt)
                    evaluate_checkpoint(session, questions, record, on_question)
                record["status"] = "completed" if all(item["score"] is not None for item in record["items"]) else "incomplete"
            except Exception as exc:
                record["status"] = "error"
                record["errors"].append(f"{type(exc).__name__}: {exc}")
            record["valid_items"] = sum(item["score"] is not None for item in record["items"])
            record["item_scores"] = {str(item["id"]): item["score"] for item in record["items"]}
            if record["status"] == "completed":
                record["total_score"] = sum(record["item_scores"].values())
            record["finished_at"] = utc_now()
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
            yield record, output
