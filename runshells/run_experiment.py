#!/usr/bin/env python3
"""
抑郁症治疗遍历实验自动化脚本。

遍历 3(severity) × 2(dynamic/static) = 6 个实验条件，
每个条件依次：替换配置 → 运行模拟 → 合并记录 → 收集数据。

用法：
  python runshells/run_experiment.py                    # 完整遍历
  python runshells/run_experiment.py --dry-run          # 只打印操作
  python runshells/run_experiment.py --condition Counsel-MOD-DYN  # 只跑单个条件
  python runshells/run_experiment.py --skip-simulation  # 跳过模拟，只收集已有数据
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Dict, List, Optional, Tuple

# ─── 项目根目录 ───────────────────────────────────────────────
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# ─── 需要替换的配置文件路径 ─────────────────────────────────────
AGENT_JSON = os.path.join(
    BASE_DIR, "frontend", "static", "assets", "village", "agents", "卡布达", "agent.json"
)
DEPRESSION_CONFIG = os.path.join(
    BASE_DIR, "frontend", "static", "assets", "village", "agents", "卡布达", "depression_config.json"
)
GLOBAL_CONFIG = os.path.join(BASE_DIR, "data", "config.json")

# ─── 外部脚本 ─────────────────────────────────────────────────
START_SCRIPT = os.path.join(BASE_DIR, "start.py")
MERGE_SCRIPT = os.path.join(BASE_DIR, "merge_consultation_dialogues.py")

# ─── 数据目录 ─────────────────────────────────────────────────
CHECKPOINTS_ROOT = os.path.join(BASE_DIR, "results", "checkpoints")
EXPERIMENT_DATA_ROOT = os.path.join(BASE_DIR, "results", "experiment_data")

# ─── 备份目录 ─────────────────────────────────────────────────
BACKUP_DIR = os.path.join(tempfile.gettempdir(), "generative_agents_config_backup")

# ─── Case 文件路径（三种严重程度） ──────────────────────────────
CASE_FILES = {
    "mild": os.path.join(BASE_DIR, "data", "depression_cases", "mild-depression-case.json"),
    "moderate": os.path.join(BASE_DIR, "data", "depression_cases", "moderate-depression-case.json"),
    "severe": os.path.join(BASE_DIR, "data", "depression_cases", "severe-depression-case.json"),
}

# ─── 动态配置按严重程度的参数映射 ────────────────────────────────
DYNAMIC_SEVERITY_PARAMS = {
    "mild": {
        "initial_state": "mild_episode",
        "initial_stage": {
            "label": "将低落和疲惫归因于环境和疲劳",
            "description": "你倾向于将情绪低落和兴趣减退解释为'最近太忙了'或'没休息好'，虽然偶尔承认可能有心理层面的问题，但主要还是用外部原因来合理化自己的状态。",
            "distress_level": 0.4,
            "openness_level": 0.55,
            "hopefulness_level": 0.5,
            "terminal_recovery": False,
        },
    },
    "moderate": {
        "initial_state": "moderate_episode",
        "initial_stage": {
            "label": "把学业受挫和低落感当成人生失败",
            "description": "你很容易把休学、效率下降和持续疲惫解释成'我这个人本来就不行'，既害怕让人失望，又不相信自己真的能好起来。",
            "distress_level": 0.75,
            "openness_level": 0.25,
            "hopefulness_level": 0.15,
            "terminal_recovery": False,
        },
    },
    "severe": {
        "initial_state": "severe_episode",
        "initial_stage": {
            "label": "深陷绝望与自我否定的泥沼",
            "description": "你感到彻底的绝望和无价值感，认为自己的存在只会给他人带来痛苦。对未来完全没有期待，甚至觉得死亡是一种解脱。你对治疗持怀疑态度，认为没有什么能帮到你。",
            "distress_level": 0.95,
            "openness_level": 0.08,
            "hopefulness_level": 0.03,
            "terminal_recovery": False,
        },
    },
}

# ─── 实验参数 ─────────────────────────────────────────────────
SEVERITIES = ["mild", "moderate", "severe"]
PERSONAS = ["dynamic", "static"]
SIM_STEP = 60
SIM_STRIDE = 360

# 需要备份的文件列表
BACKUP_FILES = {
    "agent.json": AGENT_JSON,
    "depression_config.json": DEPRESSION_CONFIG,
    "config.json": GLOBAL_CONFIG,
}

# ─── 实验条件列表 ─────────────────────────────────────────────
ALL_CONDITIONS: List[Tuple[str, str, str]] = []
for _sev in SEVERITIES:
    for _per in PERSONAS:
        _sev_short = {"mild": "MILD", "moderate": "MOD", "severe": "SEV"}[_sev]
        _per_short = {"dynamic": "DYN", "static": "STA"}[_per]
        _name = f"Counsel-{_sev_short}-{_per_short}"
        ALL_CONDITIONS.append((_name, _sev, _per))


# ─── 旧报告恢复数据（session_eval only）─────────────────────
RECOVERED_SESSION_EVAL = {
    "Counsel-MILD-DYN": {
        "scores": [85, 85, 85, 90, 95, 75, 65, 85, 65, 65, 82, 65, 75, 65],
        "session_ends": 7,
        "dialogue_turns": 368,
    },
    "Counsel-MILD-STA": {
        "scores": [90, 85, 90, 85, 85, 85, 85, 92, 90, 95, 95],
        "session_ends": 11,
        "dialogue_turns": 166,
    },
    "Counsel-MOD-DYN": {
        "scores": [75, 85, 85, 90, 85, 72, 82, 85, 85, 75, 85, 85, 75, 90],
        "session_ends": 11,
        "dialogue_turns": 328,
    },
    "Counsel-MOD-STA": {
        "scores": [90, 90, 92, 92, 92, 90, 85, 85, 90, 85, 65, 95],
        "session_ends": 11,
        "dialogue_turns": 236,
    },
    "Counsel-SEV-DYN": {
        "scores": [65, 65, 85, 85, 85, 65, 65, 85, 92, 90, 55, 65, 65, 75],
        "session_ends": 5,
        "dialogue_turns": 370,
    },
    "Counsel-SEV-STA": {
        "scores": [85, 90, 85, 90, 95, 85, 65, 65, 65, 72, 65, 68, 85, 65],
        "session_ends": 6,
        "dialogue_turns": 402,
    },
}


# ─── 三量表配置 ───────────────────────────────────────────────
SCALES: Dict[str, dict] = {
    "PHQ-9": {
        "question_file": "PHQ-9.jsonl",
        "scoring_prompt": "PHQ-9评估提示词.md",
        "scoring_prompt_dir": os.path.join(BASE_DIR, "customization", "depression_scale_agent", "questions", "scoring_prompts"),
        "items": 9,
    },
    "BDI-II": {
        "question_file": "BDI-II.jsonl",
        "scoring_prompt": "BDI-II评估提示词.md",
        "scoring_prompt_dir": os.path.join(BASE_DIR, "customization", "depression_scale_agent", "questions", "scoring_prompts"),
        "items": 21,
    },
    "SDS": {
        "question_file": "SDS.jsonl",
        "scoring_prompt": "SDS评估提示词.md",
        "scoring_prompt_dir": os.path.join(BASE_DIR, "customization", "depression_scale_agent", "questions", "scoring_prompts"),
        "items": 20,
    },
}


# ═══════════════════════════════════════════════════════════════
# 配置文件替换接口（预留，后续逐个实现）
# ═══════════════════════════════════════════════════════════════

def prepare_agent_config(severity: str, dry_run: bool = False) -> None:
    """根据严重程度替换 agent.json 的 depression_profile 部分。

    从 {severity}-depression-case.json 读取 case_config、render_order、
    dialogue_protocol、important_notice、current_event，写入 agent.json。
    同时更新 currently 和 scratch（三种严重程度统一）。
    """
    case_file = CASE_FILES.get(severity)
    if not case_file or not os.path.exists(case_file):
        print(f"  [ERROR] case 文件不存在: {case_file}")
        return

    with open(case_file, "r", encoding="utf-8") as f:
        case_data = json.load(f)

    # case 文件以 severity 为顶层 key
    severity_data = case_data.get(severity, case_data)

    if dry_run:
        print(f"  [DRY-RUN] prepare_agent_config(severity={severity})")
        print(f"    case_config keys: {list(severity_data.get('case_config', {}).keys())}")
        print(f"    render_order keys: {list(severity_data.get('render_order', {}).keys())}")
        print(f"    currently: {severity_data.get('currently', '(unchanged)')}")
        return

    with open(AGENT_JSON, "r", encoding="utf-8") as f:
        agent = json.load(f)

    # depression_profile
    dp = agent.setdefault("depression_profile", {})
    dp["severity"] = severity
    dp["case_config"] = severity_data["case_config"]
    dp["render_order"] = severity_data["render_order"]
    dp["dialogue_protocol"] = severity_data.get("dialogue_protocol", [])
    dp["important_notice"] = severity_data.get(
        "important_notice",
        dp.get("important_notice", ""),
    )

    # current_event → runtime.current_event
    ce = severity_data.get("current_event")
    if ce:
        dp.setdefault("runtime", {})["current_event"] = ce

    # currently + scratch（统一更新）
    if "currently" in severity_data:
        agent["currently"] = severity_data["currently"]
    if "scratch" in severity_data:
        agent["scratch"] = severity_data["scratch"]

    with open(AGENT_JSON, "w", encoding="utf-8") as f:
        json.dump(agent, f, ensure_ascii=False, indent=2)

    print(f"  [OK] agent.json depression_profile → severity={severity}")


def prepare_depression_config(severity: str, persona: str, dry_run: bool = False) -> None:
    """根据严重程度和人设类型替换 depression_config.json。

    - dynamic: 替换为对应严重程度的完整配置（depression_simulation.enabled: true）
    - static:  替换为 {"depression_simulation": {"enabled": false}}

    后续确认每种 severity 的动态配置具体内容后实现 dynamic 部分。
    """
    if persona == "static":
        static_config = {"depression_simulation": {"enabled": False}}
        if not dry_run:
            with open(DEPRESSION_CONFIG, "w", encoding="utf-8") as f:
                json.dump(static_config, f, ensure_ascii=False, indent=2)
            print(f"  [OK] 写入静态 depression_config.json (enabled=false)")
        else:
            print(f"  [DRY-RUN] 写入静态 depression_config.json: {static_config}")
        return

    # dynamic: 基于当前 depression_config.json 结构，替换 complaint_roadmap
    params = DYNAMIC_SEVERITY_PARAMS.get(severity)
    if not params:
        print(f"  [ERROR] 未知的 severity: {severity}")
        return

    if dry_run:
        print(f"  [DRY-RUN] prepare_depression_config(severity={severity}, persona=dynamic)")
        print(f"    initial_state: {params['initial_state']}")
        print(f"    distress: {params['initial_stage']['distress_level']}, "
              f"openness: {params['initial_stage']['openness_level']}, "
              f"hope: {params['initial_stage']['hopefulness_level']}")
        return

    with open(DEPRESSION_CONFIG, "r", encoding="utf-8") as f:
        config = json.load(f)

    sim = config.setdefault("depression_simulation", {})
    sim["enabled"] = True

    roadmap = sim.setdefault("complaint_roadmap", {})
    roadmap["initial_state"] = params["initial_state"]
    roadmap["initial_stage"] = params["initial_stage"]

    with open(DEPRESSION_CONFIG, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    print(f"  [OK] depression_config.json → severity={severity}, enabled=true")


def prepare_global_config(persona: str, dry_run: bool = False, local_llm: bool = False) -> None:
    """根据人设类型修改 data/config.json 的全局开关。

    - dynamic: depression_dynamic.enabled=true, depression_update.enabled=false
    - static:  depression_dynamic.enabled=false, depression_update.enabled=false
    """
    if persona == "dynamic":
        dynamic_enabled = True
    else:
        dynamic_enabled = False

    if dry_run:
        print(f"  [DRY-RUN] prepare_global_config(persona={persona})")
        print(f"    depression_dynamic.enabled → {dynamic_enabled}")
        print(f"    depression_update.enabled → false")
        if local_llm:
            print("    forced_llm → Ollama qwen3:32b")
        return

    with open(GLOBAL_CONFIG, "r", encoding="utf-8") as f:
        config = json.load(f)

    intervention = config.setdefault("intervention", {})
    intervention.setdefault("depression_dynamic", {})["enabled"] = dynamic_enabled
    intervention.setdefault("depression_update", {})["enabled"] = False

    if local_llm:
        forced_llm = intervention.setdefault("forced_llm", {})
        forced_llm.update({
            "provider": "ollama",
            "model": "qwen3:32b",
            "base_url": "http://127.0.0.1:11434/v1",
        })
        forced_llm.pop("api_key_env", None)

    with open(GLOBAL_CONFIG, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    print(f"  [OK] config.json → depression_dynamic.enabled={dynamic_enabled}")
    if local_llm:
        print("  [OK] config.json → forced_llm = Ollama qwen3:32b")


# ═══════════════════════════════════════════════════════════════
# 配置备份 / 恢复
# ═══════════════════════════════════════════════════════════════

def backup_configs(dry_run: bool = False) -> None:
    """备份当前配置文件到临时目录。"""
    if os.path.exists(BACKUP_DIR):
        shutil.rmtree(BACKUP_DIR)
    os.makedirs(BACKUP_DIR, exist_ok=True)
    for label, path in BACKUP_FILES.items():
        if os.path.exists(path):
            dest = os.path.join(BACKUP_DIR, label)
            shutil.copy2(path, dest)
            if not dry_run:
                print(f"  [BACKUP] {label} → {dest}")
        else:
            print(f"  [WARN] 备份源文件不存在: {path}")


def restore_configs(dry_run: bool = False) -> None:
    """从临时目录恢复原始配置文件。"""
    for label, path in BACKUP_FILES.items():
        src = os.path.join(BACKUP_DIR, label)
        if os.path.exists(src):
            shutil.copy2(src, path)
            if not dry_run:
                print(f"  [RESTORE] {label} → {path}")
        else:
            print(f"  [WARN] 备份文件不存在: {src}")


# ═══════════════════════════════════════════════════════════════
# 模拟运行
# ═══════════════════════════════════════════════════════════════

def run_simulation(trial_name: str, dry_run: bool = False, sim_step: int = SIM_STEP, local_llm: bool = False) -> bool:
    """调用 start.py 运行模拟。自动添加时间后缀避免名称冲突。"""
    run_name = f"{trial_name}-{time.strftime('%m%d-%H%M')}"
    cmd = [
        sys.executable, START_SCRIPT,
        "--name", run_name,
        "--step", str(sim_step),
        "--stride", str(SIM_STRIDE),
    ]
    print(f"  [RUN] {' '.join(cmd)}")
    if dry_run:
        return True, run_name

    try:
        env = os.environ.copy()
        if local_llm:
            env["DEEPSEEK_API_KEY"] = "ollama"
        result = subprocess.run(cmd, cwd=BASE_DIR, timeout=3600, env=env)
        if result.returncode != 0:
            print(f"  [ERROR] 模拟返回非零退出码: {result.returncode}")
            return False, run_name
        return True, run_name
    except subprocess.TimeoutExpired:
        print(f"  [ERROR] 模拟超时 (1小时)")
        return False, run_name
    except Exception as e:
        print(f"  [ERROR] 模拟异常: {e}")
        return False, run_name


def merge_dialogues(trial_name: str, dry_run: bool = False) -> bool:
    """调用 merge_consultation_dialogues.py 合并咨询记录。"""
    if not os.path.exists(MERGE_SCRIPT):
        print(f"  [SKIP] merge script not found: {MERGE_SCRIPT}")
        return True

    cmd = [sys.executable, MERGE_SCRIPT, trial_name]
    print(f"  [RUN] {' '.join(cmd)}")
    if dry_run:
        return True

    try:
        result = subprocess.run(cmd, cwd=BASE_DIR, timeout=300)
        if result.returncode != 0:
            print(f"  [WARN] 合并返回非零退出码: {result.returncode}")
        return True
    except Exception as e:
        print(f"  [WARN] 合并异常: {e}")
        return True


# ═══════════════════════════════════════════════════════════════
# 数据收集
# ═══════════════════════════════════════════════════════════════

def _analyze_session_eval(judge_file: str, trial_name: str) -> None:
    """解析 judge_conversation.json，提取并打印 session_eval 摘要。"""
    scores, session_ends = _extract_session_eval_scores(judge_file)

    if scores:
        avg = sum(scores) / len(scores)
        print(f"  [SESSION-EVAL] {trial_name}: {len(scores)} 次评估")
        print(f"    efficacy_score: 平均={avg:.1f}, 最低={min(scores)}, 最高={max(scores)}")
        print(f"    session_end 次数: {session_ends}")
        print(f"    分数序列: {scores}")
    else:
        print(f"  [SESSION-EVAL] {trial_name}: 未找到 efficacy_score 数据")


def collect_trial_data(trial_name: str, severity: str, persona: str, dry_run: bool = False, round_idx: int = 1) -> None:
    """从 checkpoint 收集试验数据到 experiment_data 目录。"""
    checkpoint_dir = os.path.join(CHECKPOINTS_ROOT, trial_name)
    output_dir = os.path.join(EXPERIMENT_DATA_ROOT, trial_name)

    if not os.path.exists(checkpoint_dir):
        print(f"  [SKIP] checkpoint 目录不存在: {checkpoint_dir}")
        return

    if dry_run:
        print(f"  [DRY-RUN] 收集数据: {checkpoint_dir} → {output_dir}")
        return

    os.makedirs(output_dir, exist_ok=True)

    # 创建子目录
    configs_dir = os.path.join(output_dir, "configs")
    traces_dir = os.path.join(output_dir, "traces")
    scales_dir = os.path.join(output_dir, "scales")
    expert_dir = os.path.join(output_dir, "scales", "expert")
    for d in [configs_dir, traces_dir, scales_dir, expert_dir]:
        os.makedirs(d, exist_ok=True)

    # 复制配置快照 → configs/
    for label, path in BACKUP_FILES.items():
        if os.path.exists(path):
            shutil.copy2(path, os.path.join(configs_dir, f"original_{label}"))

    # 复制当前使用的配置（已在 restore 前调用，所以是试验时的配置）
    for label, src_path in [
        ("agent.json", AGENT_JSON),
        ("depression_config.json", DEPRESSION_CONFIG),
        ("config.json", GLOBAL_CONFIG),
    ]:
        backup_src = os.path.join(BACKUP_DIR, label)
        if os.path.exists(backup_src):
            pass  # 原始配置已在 restore 前 copy 过了

    # 写入试验元数据
    meta = {
        "trial_name": trial_name,
        "severity": severity,
        "persona": persona,
        "therapy": "CBT",
        "sim_step": SIM_STEP,
        "sim_stride": SIM_STRIDE,
        "round": round_idx,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(os.path.join(output_dir, "trial_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # 提取 judge_conversation → traces/
    judge_file = os.path.join(checkpoint_dir, "judge_traces", "judge_conversation.json")
    if os.path.exists(judge_file):
        shutil.copy2(judge_file, os.path.join(traces_dir, "judge_conversation.json"))
        print(f"  [COLLECT] judge_conversation.json → traces/")
    else:
        print(f"  [WARN] judge_conversation.json 不存在")

    # 复制 forced_prompt_traces → traces/
    fpt_dir = os.path.join(checkpoint_dir, "forced_prompt_traces")
    if os.path.isdir(fpt_dir):
        dst_fpt = os.path.join(traces_dir, "forced_prompt_traces")
        if os.path.exists(dst_fpt):
            shutil.rmtree(dst_fpt)
        shutil.copytree(fpt_dir, dst_fpt)
        print(f"  [COLLECT] forced_prompt_traces/ → traces/")

    # 复制 merge_consultation_dialogues → traces/
    merge_dir = os.path.join(checkpoint_dir, "merge_consultation_dialogues")
    if os.path.isdir(merge_dir):
        merge_file = os.path.join(merge_dir, "merge_consultation_dialogues.json")
        if os.path.exists(merge_file):
            shutil.copy2(merge_file, os.path.join(traces_dir, "merge_consultation_dialogues.json"))
            print(f"  [COLLECT] merge_consultation_dialogues.json → traces/")

    print(f"  [OK] 数据已收集到 {output_dir}")


# ═══════════════════════════════════════════════════════════════
# 分析报告生成
# ═══════════════════════════════════════════════════════════════

def _extract_session_eval_scores(judge_file: str) -> list:
    """从 judge_conversation.json 提取 efficacy_score 序列。

    支持两种格式:
    - {"sessions": [{"session_eval": {"efficacy_score": 75, ...}}, ...]}
    - [{...}, {...}]  扁平列表格式
    """
    try:
        with open(judge_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return [], 0

    # 格式1: {"sessions": [...]} — 从每个 session 的 session_eval 字段提取
    if isinstance(data, dict) and "sessions" in data:
        entries = [s.get("session_eval", {}) for s in data["sessions"]]
    elif isinstance(data, list):
        entries = data
    else:
        entries = [data]

    scores = []
    session_ends = 0
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        if "efficacy_score" in entry:
            scores.append(entry["efficacy_score"])
        if entry.get("session_end") is True:
            session_ends += 1

    return scores, session_ends


def _load_scale_answers(answered_file: str) -> list:
    """从量表回答 JSONL 文件加载问答记录。"""
    answers = []
    if not os.path.exists(answered_file):
        return answers
    with open(answered_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                answers.append(json.loads(line))
    return answers


def _count_conversation_turns(conv: dict | list) -> dict:
    """统计对话轮次和对话时段数。

    conversation.json 格式:
    {timestamp: [{对话key: [[speaker, text], ...]}], ...}
    """
    result = {"sessions": 0, "turns": 0}
    if isinstance(conv, list):
        result["turns"] = len(conv)
        return result

    if isinstance(conv, dict):
        for ts, convs in conv.items():
            result["sessions"] += 1
            for c in convs:
                for k, v in c.items():
                    if isinstance(v, list):
                        result["turns"] += len(v)
                    else:
                        result["turns"] += 1
    return result


def _load_expert_scores(score_file: str) -> dict:
    """从专家评分文件加载评分结果。"""
    if not os.path.exists(score_file):
        return {}
    try:
        with open(score_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def generate_report(dry_run: bool = False) -> None:
    """遍历完成后，生成综合分析报告。

    报告 5 个 Section:
    1. 实验概况
    2. 治疗前后量表对比（PHQ-9 / BDI-II / SDS）
    3. 逐次咨询效果轨迹
    4. 治疗效果汇总表
    5. 跨条件分析（严重程度、人设、交互效应、量表一致性、多轮稳定性）
    """
    print(f"\n{'='*60}")
    print("生成分析报告")
    print(f"{'='*60}")

    if dry_run:
        print("  [DRY-RUN] 跳过报告生成")
        return

    report_path = os.path.join(EXPERIMENT_DATA_ROOT, "reports", "analysis_report.md")
    os.makedirs(EXPERIMENT_DATA_ROOT, exist_ok=True)

    # ── 收集所有试验数据 ──
    all_dirs = [d for d in os.listdir(EXPERIMENT_DATA_ROOT)
                if os.path.isdir(os.path.join(EXPERIMENT_DATA_ROOT, d))
                and d.startswith("Counsel-")]

    def _find_trial_dirs(trial_name: str) -> List[str]:
        """找到匹配的所有试验目录（按时间戳排序，支持多轮）。

        只返回带时间戳后缀的目录（正式实验数据），排除旧试跑数据。
        带时间戳格式: NAME-MMDD-HHMM
        """
        # 只匹配以 trial_name- 开头的目录（带时间戳后缀）
        ts_dirs = sorted(
            [d for d in all_dirs if d.startswith(trial_name + "-")],
        )
        if ts_dirs:
            return [os.path.join(EXPERIMENT_DATA_ROOT, d) for d in ts_dirs]
        # 回退：如果没有时间戳目录，检查精确匹配（兼容旧数据）
        if trial_name in all_dirs:
            return [os.path.join(EXPERIMENT_DATA_ROOT, trial_name)]
        return []

    def _find_single_trial_dir(trial_name: str) -> Optional[str]:
        """找到最新匹配的试验目录（兼容旧逻辑）。"""
        dirs = _find_trial_dirs(trial_name)
        return dirs[-1] if dirs else None

    # 按条件名收集数据，支持多轮
    condition_data = {}  # key = trial_name, value = list of round dicts
    for trial_name, severity, persona in ALL_CONDITIONS:
        trial_dirs = _find_trial_dirs(trial_name)
        if not trial_dirs:
            print(f"  [SKIP] 试验目录不存在: {trial_name}")
            continue

        rounds_data = []
        for td in trial_dirs:
            rd = {
                "dir": td,
                "dir_name": os.path.basename(td),
                "severity": severity,
                "persona": persona,
                "trial_name": trial_name,
                "session_eval": None,
                "conversation": None,
                "scale_answers": {},
                "scale_scores": {},
                "meta": {},
                "round": 1,
            }

            # 1. 读取 trial 元数据
            meta_file = os.path.join(td, "trial_meta.json")
            if os.path.exists(meta_file):
                try:
                    with open(meta_file, "r", encoding="utf-8") as f:
                        rd["meta"] = json.load(f)
                        rd["round"] = rd["meta"].get("round", 1)
                except (json.JSONDecodeError, OSError):
                    pass

            # 2. 提取 session_eval scores
            judge_file = os.path.join(td, "traces", "judge_conversation.json")
            if not os.path.exists(judge_file):
                judge_file = os.path.join(td, "judge_conversation.json")  # 兼容旧目录
            if os.path.exists(judge_file):
                scores, session_ends = _extract_session_eval_scores(judge_file)
                rd["session_eval"] = {
                    "scores": scores,
                    "count": len(scores),
                    "avg": sum(scores) / len(scores) if scores else 0,
                    "min": min(scores) if scores else None,
                    "max": max(scores) if scores else None,
                    "session_ends": session_ends,
                }

            # 3. 统计对话轮次（从 checkpoints 读取）
            checkpoint_dir = os.path.join(CHECKPOINTS_ROOT, os.path.basename(td))
            conv_file = os.path.join(td, "conversation.json")
            if not os.path.exists(conv_file) and os.path.isdir(checkpoint_dir):
                conv_file = os.path.join(checkpoint_dir, "conversation.json")
            if os.path.exists(conv_file):
                try:
                    with open(conv_file, "r", encoding="utf-8") as f:
                        conv = json.load(f)
                    rd["conversation"] = _count_conversation_turns(conv)
                except (json.JSONDecodeError, OSError):
                    pass

            # 4. 读取量表问答（优先从 scales/ 子目录读取）
            scales_dir = os.path.join(td, "scales")
            if os.path.isdir(scales_dir):
                for sf in os.listdir(scales_dir):
                    if sf.endswith("_answered.jsonl"):
                        scale_name = sf.replace("_answered.jsonl", "")
                        if "_" in scale_name:
                            answers = _load_scale_answers(os.path.join(scales_dir, sf))
                            if answers:
                                rd["scale_answers"][scale_name] = answers
            else:
                # 兼容旧目录结构
                for sf in os.listdir(td):
                    if sf.startswith("scale_") and sf.endswith("_answered.jsonl"):
                        scale_name = sf.replace("scale_", "").replace("_answered.jsonl", "")
                        answers = _load_scale_answers(os.path.join(td, sf))
                        if answers:
                            rd["scale_answers"][scale_name] = answers

            # 5. 读取量表评分
            scores_file = os.path.join(td, "scales", "scale_scores.json")
            if not os.path.exists(scores_file):
                scores_file = os.path.join(td, "scale_scores.json")  # 兼容旧目录
            if os.path.exists(scores_file):
                try:
                    with open(scores_file, "r", encoding="utf-8") as f:
                        rd["scale_scores"] = json.load(f)
                except (json.JSONDecodeError, OSError):
                    pass

            # 6. 读取 30Q 综合评分
            rd["scale_30q"] = {}
            for phase in ["pre", "post"]:
                scored_path = os.path.join(td, "scales", f"30Q_{phase}_scored.json")
                if not os.path.exists(scored_path):
                    scored_path = os.path.join(td, f"scale_30Q_{phase}_scored.json")  # 兼容旧目录
                if os.path.exists(scored_path):
                    try:
                        with open(scored_path, "r", encoding="utf-8") as f:
                            rd["scale_30q"][phase] = json.load(f)
                    except (json.JSONDecodeError, OSError):
                        pass

            rounds_data.append(rd)

        condition_data[trial_name] = rounds_data

    if not condition_data:
        print("  [WARN] 没有找到任何试验数据，跳过报告生成")
        return

    total_trials = sum(len(v) for v in condition_data.values())
    max_rounds = max(max((rd["round"] for rd in rds), default=1) for rds in condition_data.values())

    # ── 辅助: 计算均值和标准差 ──
    def _mean(vals):
        vals = [v for v in vals if v is not None]
        return sum(vals) / len(vals) if vals else None

    def _std(vals):
        vals = [v for v in vals if v is not None]
        if len(vals) < 2:
            return 0.0
        m = sum(vals) / len(vals)
        return math.sqrt(sum((x - m) ** 2 for x in vals) / (len(vals) - 1))

    def _fmt_mean_sd(vals, fmt=".1f"):
        m = _mean(vals)
        if m is None:
            return "—"
        if len(vals) <= 1:
            return f"{m:{fmt}}"
        s = _std(vals)
        return f"{m:{fmt}}±{s:{fmt}}"

    # ── 生成 Markdown 报告 ──
    lines = []
    lines.append("# 抑郁症 AI 智能体治疗遍历实验 — 分析报告\n")
    lines.append(f"\n> 数据目录：`results/experiment_data/` | 报告目录：`results/experiment_data/reports/`\n")
    lines.append(f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    # ═══════════════════════════════════════════════════════════
    # Section 1: 实验概况
    # ═══════════════════════════════════════════════════════════
    lines.append("## 1. 实验概况\n")

    lines.append("### 1.1 实验设计\n")
    lines.append("基于 generative agents 框架的抑郁症 CBT 治疗仿真实验。")
    lines.append("系统包含两个核心角色：")
    lines.append("- **卡布达**（患者）：23 岁，因长期宠物狗去世引发抑郁症状的年轻女性")
    lines.append("- **蜻蜓队长**（医生）：CBT 认知行为治疗师\n")
    lines.append(f"采用 **3(severity) x 2(persona) 因子设计**，共 {len(condition_data)} 个实验条件：")
    lines.append("")
    lines.append("| 因子 | 水平 | 说明 |")
    lines.append("|------|------|------|")
    lines.append("| 抑郁严重程度 | mild | 轻度：日常功能基本保留，情绪有正向反应性，社交退缩轻微 |")
    lines.append("| | moderate | 中度：基本活动需努力维持，明显疲乏/睡眠/食欲障碍，快感缺失但未完全丧失 |")
    lines.append("| | severe | 重度：极端功能损害，完全社交退缩，存在被动自杀意念，基本自理困难 |")
    lines.append("| 抑郁人设类型 | dynamic | 动态人设：抑郁状态随对话/事件实时演化（depression_dynamic 引擎） |")
    lines.append("| | static | 静态人设：抑郁状态固定不变，仅通过 prompt 注入静态描述 |")
    lines.append("")
    lines.append(f"共 {len(condition_data)} 个条件：")
    for tn, sev, per in ALL_CONDITIONS:
        lines.append(f"- `{tn}`：{sev} + {per}")
    lines.append("")

    lines.append("### 1.2 仿真参数\n")
    total_hours = SIM_STEP * SIM_STRIDE // 60
    total_days = total_hours // 24
    lines.append(f"- **模拟步数** (step)：{SIM_STEP} 步")
    lines.append(f"- **时间步长** (stride)：{SIM_STRIDE} 分钟/步")
    lines.append(f"- **总仿真时长**：{total_hours} 小时 ≈ {total_days} 天")
    lines.append(f"- **咨询调度**：每 4 步触发一次 CBT 咨询会议，预计约 {SIM_STEP // 4} 次")
    lines.append("- **CBT session 序列**（11 阶段）：")
    lines.append("  信息采集 → 认知标签 → 条件规则 → 核心信念发现 → 法庭辩论 → 行为实验 → 回顾(成功/回避) → 工具盘点 → 心理急救包 → 压力测试 → 毕业仪式")
    lines.append(f"- **每次咨询最大对话轮次**：18 轮，终止窗口为最后 6 轮")
    lines.append(f"- **循环轮数**：{max_rounds}")
    lines.append(f"- **总试验次数**：{total_trials}")
    lines.append("")

    lines.append("### 1.3 评估指标说明\n")
    lines.append("**指标 1：session_eval（逐次咨询效果评估）**\n")
    lines.append("每次 CBT 咨询结束后，由 DeepSeek API 评估本轮咨询对该阶段 session 目标的推进效果。")
    lines.append("")
    lines.append("| 字段 | 类型 | 范围 | 含义 |")
    lines.append("|------|------|------|------|")
    lines.append("| efficacy_score | int | 0-100 | 本轮咨询对该阶段 session **总体目标的累计推进效果** |")
    lines.append("| session_end | bool | true/false | 当前 CBT 阶段是否可以结束，推进到下一阶段 |")
    lines.append("")
    lines.append("- efficacy_score 评估的是治疗推进程度，不是对话语言质量")
    lines.append("- 高分（≥80）= 该阶段目标基本达成；低分（<70）= 关键步骤仍有缺失")
    lines.append("- session_end=true 需满足：核心目标已完成且继续停留收益低")
    lines.append("")
    lines.append("**指标 2：量表评估（治疗前后对比）**\n")
    lines.append("在仿真开始前（第一个快照）和结束后（最后一个快照），由 agent 分别回答三个标准量表题目，再由 DeepSeek API 评分。")
    lines.append("")
    lines.append("| 量表 | 题数 | 计分 | 总分范围 | 无/正常 | 轻度 | 中度 | 中重度 | 重度 | 时间范围 | 评估维度 |")
    lines.append("|------|------|------|----------|--------|------|------|--------|------|----------|----------|")
    lines.append("| PHQ-9 | 9 题 | 0-3 分/题 | 0-27 | 0-4 | 5-9 | 10-14 | 15-19 | 20-27 | 近 2 周 | 抑郁症状出现频率 |")
    lines.append("| BDI-II | 21 题 | 0-3 分/题 | 0-63 | 0-13 | 14-19 | 20-28 | — | 29-63 | 近 2 周 | 抑郁症状严重程度 |")
    lines.append("| SDS | 20 题 | 1-4 分/题 | 20-80（标准分 ×1.25） | <50 | 50-59 | 60-69 | — | ≥70 | 近 1 周 | 抑郁症状出现频度 |")
    lines.append("")
    lines.append("- 治疗前 = 加载第一个仿真快照，agent 处于初始抑郁状态")
    lines.append("- 治疗后 = 加载最后一个仿真快照，agent 经历完整 CBT 治疗流程")
    lines.append("- 分数下降 = 症状减轻（好转），分数上升 = 症状加重")
    lines.append("")

    # 检查哪些条件有量表数据
    has_scale = any(
        any(rd["scale_scores"].get("pre") or rd["scale_scores"].get("post") for rd in rds)
        for rds in condition_data.values()
    )
    lines.append("### 1.4 数据状态\n")
    lines.append(f"- 前后量表数据: {'已生成' if has_scale else '未生成（需运行 --pre-post-scale）'}")
    has_recovered = any(
        rd["meta"].get("data_source") == "recovered_from_report"
        for rds in condition_data.values() for rd in rds
    )
    if has_recovered:
        lines.append("- **部分轮次数据从旧报告恢复**（标记为 recovered）：仅含 session_eval 评分，无量表和原始对话数据")
    lines.append("")

    # ═══════════════════════════════════════════════════════════
    # Section 2: 治疗前后量表对比
    # ═══════════════════════════════════════════════════════════
    lines.append("## 2. 治疗前后量表对比\n")

    if has_scale:
        # 2.1 汇总表
        lines.append("### 2.1 汇总表\n")
        lines.append("| 条件 | 量表 | 治疗前 | 治疗前程度 | 治疗后 | 治疗后程度 | 变化 | 变化率 |")
        lines.append("|------|------|--------|-----------|--------|-----------|------|--------|")

        for trial_name, rds in condition_data.items():
            severity = rds[0]["severity"]
            persona = rds[0]["persona"]
            for scale_name in SCALES:
                pre_totals = []
                post_totals = []
                pre_severities = []
                post_severities = []
                for rd in rds:
                    pre_scored = rd["scale_scores"].get("pre", {}).get(scale_name)
                    post_scored = rd["scale_scores"].get("post", {}).get(scale_name)
                    pt = _extract_scale_total(pre_scored) if isinstance(pre_scored, dict) else None
                    pot = _extract_scale_total(post_scored) if isinstance(post_scored, dict) else None
                    if pt is not None:
                        pre_totals.append(pt)
                    if pot is not None:
                        post_totals.append(pot)
                    if isinstance(pre_scored, dict):
                        pre_sev = _get_severity_label(pre_scored)
                        if pre_sev != "—":
                            pre_severities.append(pre_sev)
                    if isinstance(post_scored, dict):
                        post_sev = _get_severity_label(post_scored)
                        if post_sev != "—":
                            post_severities.append(post_sev)

                pre_str = _fmt_mean_sd(pre_totals) if pre_totals else "—"
                post_str = _fmt_mean_sd(post_totals) if post_totals else "—"
                pre_sev_str = pre_severities[0] if pre_severities else "—"
                post_sev_str = post_severities[0] if post_severities else "—"

                if pre_totals and post_totals and len(pre_totals) == len(post_totals):
                    deltas = [post_totals[i] - pre_totals[i] for i in range(len(pre_totals))]
                    pcts = [d / p * 100 if p != 0 else 0 for d, p in zip(deltas, pre_totals)]
                    delta_str = _fmt_mean_sd(deltas)
                    pct_str = f"{_mean(pcts):.1f}%"
                elif pre_totals and post_totals:
                    pre_m = _mean(pre_totals)
                    post_m = _mean(post_totals)
                    delta_str = f"{post_m - pre_m:.1f}"
                    pct_str = f"{(post_m - pre_m) / pre_m * 100:.1f}%" if pre_m else "—"
                else:
                    delta_str = "—"
                    pct_str = "—"

                lines.append(f"| {trial_name} | {scale_name} | {pre_str} | {pre_sev_str} | {post_str} | {post_sev_str} | {delta_str} | {pct_str} |")
        lines.append("")

        # 2.2 多轮明细（仅多轮时显示）
        if max_rounds > 1:
            lines.append("### 2.2 逐轮明细\n")
            for trial_name, rds in condition_data.items():
                lines.append(f"**{trial_name}**\n")
                lines.append("| 轮次 | " + " | ".join(SCALES.keys()) + " |")
                lines.append("|------|" + "|".join(["--------"] * len(SCALES)) + "|")
                for rd in sorted(rds, key=lambda x: x["round"]):
                    row = f"| R{rd['round']} |"
                    for scale_name in SCALES:
                        pre_total = _extract_scale_total(rd["scale_scores"].get("pre", {}).get(scale_name, {}))
                        post_total = _extract_scale_total(rd["scale_scores"].get("post", {}).get(scale_name, {}))
                        if pre_total is not None and post_total is not None:
                            delta = post_total - pre_total
                            row += f" {pre_total:.0f}→{post_total:.0f} ({delta:+.0f}) |"
                        else:
                            row += " — |"
                    lines.append(row)
                lines.append("")

        # 2.3 评分详情链接（逐条打分已移至附录）
        lines.append("### 2.3 各量表评分详情\n")
        lines.append("> 逐条目评分详情已移至 [附录 A3](analysis_report_appendix.md#a3-各量表逐条目评分详情)。\n")
    else:
        lines.append("> 前后量表数据尚未生成。")
        lines.append("> 运行 `python runshells/run_experiment.py --pre-post-scale` 或在模拟时自动生成。\n")

    # ── 生成附录：问答原始记录 ──
    appendix_path = os.path.join(EXPERIMENT_DATA_ROOT, "reports", "analysis_report_appendix.md")
    appendix_lines = []
    appendix_lines.append("# 实验报告附录 — 量表问答原始记录\n")
    appendix_lines.append(f"\n> 数据目录：`results/experiment_data/` | 报告目录：`results/experiment_data/reports/`\n")
    appendix_lines.append(f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    appendix_lines.append("> 本附录包含所有量表评估的原始问答记录（agent 回答文本），与主报告的评分结果对应。\n")

    has_appendix = False

    # A1: 治疗前后量表问答对比
    if has_scale:
        appendix_lines.append("## A1. 治疗前后量表问答对比\n")
        for trial_name, rds in condition_data.items():
            for rd in rds:
                if rd["round"] != rds[0]["round"]:
                    continue
                for scale_name in SCALES:
                    pre_answers = rd["scale_answers"].get(f"{scale_name}_pre")
                    post_answers = rd["scale_answers"].get(f"{scale_name}_post")
                    if not pre_answers and not post_answers:
                        continue
                    has_appendix = True
                    appendix_lines.append(f"### {trial_name} — {scale_name} ({rd['dir_name']})\n")
                    if pre_answers and post_answers:
                        appendix_lines.append("| 题号 | 问题 | 治疗前 | 治疗后 |")
                        appendix_lines.append("|------|------|--------|--------|")
                        for pre_a, post_a in zip(pre_answers, post_answers):
                            qid = pre_a.get("id", "?")
                            q_text = str(pre_a.get("question", ""))
                            pre_text = str(pre_a.get("answer", ""))
                            post_text = str(post_a.get("answer", ""))
                            appendix_lines.append(f"| Q{qid} | {q_text} | {pre_text} | {post_text} |")
                    elif pre_answers:
                        appendix_lines.append("| 题号 | 问题 | 治疗前回答 |")
                        appendix_lines.append("|------|------|-----------|")
                        for a in pre_answers:
                            qid = a.get("id", "?")
                            q_text = str(a.get("question", ""))
                            a_text = str(a.get("answer", ""))
                            appendix_lines.append(f"| Q{qid} | {q_text} | {a_text} |")
                    appendix_lines.append("")

    # A2: 原始 app.py 量表回答（30Q 等）
    for trial_name, rds in condition_data.items():
        for rd in rds:
            if rd["round"] != rds[0]["round"]:
                continue
            original_scales = {
                k: v for k, v in rd["scale_answers"].items()
                if not k.endswith("_pre") and not k.endswith("_post")
            }
            if not original_scales:
                continue
            if not has_appendix:
                appendix_lines.append("## A2. 原始量表问答记录（app.py ChatSession）\n")
            has_appendix = True
            appendix_lines.append(f"### {trial_name} ({rd['dir_name']})\n")
            for scale_name, answers in original_scales.items():
                appendix_lines.append(f"**量表: {scale_name}** ({len(answers)} 题)")
                appendix_lines.append("| 题号 | 问题 | 回答 |")
                appendix_lines.append("|------|------|------|")
                for a in answers:
                    qid = a.get("id", "?")
                    q_text = str(a.get("question", ""))
                    a_text = str(a.get("answer", ""))
                    appendix_lines.append(f"| Q{qid} | {q_text} | {a_text} |")
                appendix_lines.append("")

    # A3: 各量表逐条目评分详情（从 Section 2.3 移入）
    if has_scale:
        appendix_lines.append("## A3. 各量表逐条目评分详情\n")
        for trial_name, rds in condition_data.items():
            for rd in rds:
                if rd["round"] != rds[0]["round"]:
                    continue
                for phase in ["pre", "post"]:
                    phase_label = "治疗前" if phase == "pre" else "治疗后"
                    phase_data = rd["scale_scores"].get(phase, {})
                    if not phase_data:
                        continue
                    appendix_lines.append(f"### {trial_name} — {phase_label} ({rd['dir_name']})\n")
                    for scale_name in SCALES:
                        scored = phase_data.get(scale_name)
                        if not scored or not isinstance(scored, dict):
                            continue
                        items = scored.get("sds_scores") or scored.get("bdi_ii_scores") or scored.get("phq9_scores") or []
                        if items:
                            appendix_lines.append(f"**{scale_name}**")
                            appendix_lines.append("| 条目 | 得分 | 依据 |")
                            appendix_lines.append("|------|------|------|")
                            for item in items:
                                item_name = item.get("item", item.get("name", "?"))
                                score = item.get("score", "—")
                                basis = item.get("basis", "")
                                appendix_lines.append(f"| {item_name} | {score} | {basis} |")
                            appendix_lines.append("")

                        total = scored.get("total_score") or scored.get("total_score_raw") or scored.get("standard_score", "—")
                        sev = scored.get("severity") or scored.get("severity_by_index") or scored.get("severity_by_standard_score", "—")
                        appendix_lines.append(f"  **总分: {total} | 严重程度: {sev}**\n")
                    appendix_lines.append("")

    if has_scale or has_appendix:
        has_appendix = True
        with open(appendix_path, "w", encoding="utf-8") as f:
            f.write("\n".join(appendix_lines))
        lines.append(f"> 量表问答原始记录见 [附录](analysis_report_appendix.md)\n")

    # ═══════════════════════════════════════════════════════════
    # Section 3: 逐次咨询效果轨迹
    # ═══════════════════════════════════════════════════════════
    lines.append("## 3. 逐次咨询效果轨迹\n")
    lines.append("> 注：checkpoint 中的 `depression_dynamic_state`（distress_level / openness_level / hopefulness_level）")
    lines.append("> 在仿真过程中保持初始种子值不变，无法反映逐次变化。以下轨迹基于 session_eval 的 efficacy_score。\n")

    for trial_name, rds in condition_data.items():
        lines.append(f"### {trial_name}\n")

        if max_rounds > 1 and len(rds) > 1:
            # 多轮叠加
            # 对齐各轮的 scores 列表
            max_len = max((len(rd["session_eval"]["scores"]) for rd in rds if rd["session_eval"]), default=0)
            if max_len == 0:
                lines.append("> 无 session_eval 数据\n")
                continue

            header = "| 咨询序号 | " + " | ".join(f"R{rd['round']}" for rd in sorted(rds, key=lambda x: x["round"])) + " | 均值±SD | 趋势 |"
            sep = "|---------|" + "|".join(["--------"] * len(rds)) + "|----------|------|"
            lines.append(header)
            lines.append(sep)

            for i in range(max_len):
                round_vals = []
                for rd in sorted(rds, key=lambda x: x["round"]):
                    if rd["session_eval"] and i < len(rd["session_eval"]["scores"]):
                        round_vals.append(rd["session_eval"]["scores"][i])
                    else:
                        round_vals.append(None)

                row = f"| {i+1} |"
                for v in round_vals:
                    row += f" {v if v is not None else '—'} |"

                valid_vals = [v for v in round_vals if v is not None]
                mean_str = _fmt_mean_sd(valid_vals)
                # 趋势箭头
                if i > 0 and valid_vals:
                    prev_vals = []
                    for rd in sorted(rds, key=lambda x: x["round"]):
                        if rd["session_eval"] and i - 1 < len(rd["session_eval"]["scores"]):
                            prev_vals.append(rd["session_eval"]["scores"][i - 1])
                    if prev_vals:
                        prev_m = _mean(prev_vals)
                        curr_m = _mean(valid_vals)
                        if prev_m is not None and curr_m is not None:
                            if curr_m > prev_m + 1:
                                trend = "↑"
                            elif curr_m < prev_m - 1:
                                trend = "↓"
                            else:
                                trend = "→"
                        else:
                            trend = "—"
                    else:
                        trend = "—"
                else:
                    trend = "—"
                row += f" {mean_str} | {trend} |"
                lines.append(row)
            lines.append("")
        else:
            # 单轮
            rd = rds[0]
            se = rd["session_eval"]
            if se:
                lines.append("| 咨询序号 | efficacy_score | 趋势 |")
                lines.append("|---------|---------------|------|")
                for i, s in enumerate(se["scores"]):
                    if i == 0:
                        trend = "—"
                    elif s > se["scores"][i - 1] + 1:
                        trend = "↑"
                    elif s < se["scores"][i - 1] - 1:
                        trend = "↓"
                    else:
                        trend = "→"
                    lines.append(f"| {i+1} | {s} | {trend} |")
                lines.append(f"\n- 评估次数: {se['count']}")
                lines.append(f"- 平均分: {se['avg']:.1f}")
                lines.append(f"- 最低/最高: {se['min']} / {se['max']}")
                lines.append(f"- session_end 次数: {se['session_ends']}")
            else:
                lines.append("> 无 session_eval 数据")
            lines.append("")

    # ═══════════════════════════════════════════════════════════
    # Section 4: 治疗效果汇总表
    # ═══════════════════════════════════════════════════════════
    lines.append("## 4. 治疗效果汇总表\n")
    lines.append("> **分别评估**：PHQ-9/BDI-II/SDS 分别问答 + ExpertLLM 逐题评分。")
    lines.append("> **30Q 合并评估**：30 道筛查问题一次性问答 + ExpertLLM 综合三量表评分。\n")

    # 检查是否有 30Q 数据
    has_30q = any(
        any(rd["scale_30q"].get("pre") or rd["scale_30q"].get("post") for rd in rds)
        for rds in condition_data.values()
    )

    def _extract_30q_total(scored_30q: dict, scale_key: str) -> Optional[float]:
        """从 30Q 评分结果中提取指定量表的总分。"""
        if not scored_30q:
            return None
        scale_data = scored_30q.get(scale_key, {})
        if not scale_data:
            return None
        if scale_key == "sds":
            for k in ["standard_score", "total_score_raw"]:
                v = scale_data.get(k)
                if v is not None:
                    try: return float(v)
                    except: pass
        else:
            v = scale_data.get("total_score")
            if v is not None:
                try: return float(v)
                except: pass
        return None

    def _get_30q_severity(scored_30q: dict, scale_key: str) -> str:
        """从 30Q 评分结果中提取指定量表的严重程度。"""
        if not scored_30q:
            return "—"
        scale_data = scored_30q.get(scale_key, {})
        if not scale_data:
            return "—"
        for k in ["severity", "severity_by_standard_score", "severity_by_index"]:
            v = scale_data.get(k)
            if v:
                return str(v)
        return "—"

    # 4.1 分别评估汇总
    lines.append("### 4.1 三量表分别评估\n")
    header = "| 条件 | 严重程度 | 人设 | 平均efficacy |"
    sep = "|------|----------|------|-------------|"
    for scale_name in SCALES:
        header += f" {scale_name} 前→后(Δ) | {scale_name} 程度 |"
        sep += "-------------------|-------------|"
    header += " 咨询次数 | 对话轮次 | 轮次 |"
    sep += "---------|---------|------|"
    lines.append(header)
    lines.append(sep)

    for trial_name, rds in condition_data.items():
        severity = rds[0]["severity"]
        persona = rds[0]["persona"]

        avg_scores = [rd["session_eval"]["avg"] for rd in rds if rd["session_eval"]]
        conv_turns = [rd["conversation"]["turns"] for rd in rds if rd["conversation"]]
        conv_sessions = [rd["conversation"]["sessions"] for rd in rds if rd["conversation"]]
        if not conv_sessions:
            conv_sessions = [rd["session_eval"]["count"] for rd in rds if rd["session_eval"]]
        if not conv_turns:
            for rd in rds:
                dt = rd["meta"].get("dialogue_turns")
                if dt:
                    conv_turns.append(dt)
        n_rounds = len(rds)

        efficacy_str = _fmt_mean_sd(avg_scores)

        scale_cols = ""
        for scale_name in SCALES:
            pre_totals = []
            post_totals = []
            pre_sevs = []
            post_sevs = []
            for rd in rds:
                pre_scored = rd["scale_scores"].get("pre", {}).get(scale_name)
                post_scored = rd["scale_scores"].get("post", {}).get(scale_name)
                pt = _extract_scale_total(pre_scored) if isinstance(pre_scored, dict) else None
                pot = _extract_scale_total(post_scored) if isinstance(post_scored, dict) else None
                if pt is not None:
                    pre_totals.append(pt)
                if pot is not None:
                    post_totals.append(pot)
                if isinstance(pre_scored, dict):
                    s = _get_severity_label(pre_scored)
                    if s != "—": pre_sevs.append(s)
                if isinstance(post_scored, dict):
                    s = _get_severity_label(post_scored)
                    if s != "—": post_sevs.append(s)

            if pre_totals and post_totals:
                pre_m = _mean(pre_totals)
                post_m = _mean(post_totals)
                delta = post_m - pre_m
                score_col = f"{pre_m:.0f}→{post_m:.0f}({delta:+.0f})"
            elif pre_totals:
                score_col = f"{_mean(pre_totals):.0f}→—"
            elif post_totals:
                score_col = f"—→{_mean(post_totals):.0f}"
            else:
                score_col = "—"

            sev_pre = pre_sevs[0] if pre_sevs else "—"
            sev_post = post_sevs[0] if post_sevs else "—"
            if sev_pre != "—" and sev_post != "—":
                sev_col = f"{sev_pre}→{sev_post}"
            elif sev_pre != "—":
                sev_col = f"{sev_pre}→—"
            elif sev_post != "—":
                sev_col = f"—→{sev_post}"
            else:
                sev_col = "—"

            scale_cols += f"| {score_col} | {sev_col} "

        sessions_str = _fmt_mean_sd(conv_sessions, fmt=".0f") if conv_sessions else "—"
        turns_str = _fmt_mean_sd(conv_turns, fmt=".0f") if conv_turns else "—"
        has_recovered = any(rd["meta"].get("data_source") == "recovered_from_report" for rd in rds)
        rounds_str = str(n_rounds)
        if has_recovered:
            rounds_str += " (含recovered)"

        lines.append(
            f"| {trial_name} | {severity} | {persona} "
            f"| {efficacy_str} {scale_cols}| "
            f"{sessions_str} | {turns_str} | {rounds_str} |"
        )
    lines.append("")

    # 4.2 30Q 合并评估（如果有数据）
    if has_30q:
        lines.append("### 4.2 30Q 合并评估\n")
        _30q_scale_keys = {"PHQ-9": "phq9", "BDI-II": "bdi_ii", "SDS": "sds"}

        header_30q = "| 条件 |"
        sep_30q = "|------|"
        for scale_name in SCALES:
            header_30q += f" {scale_name} 30Q前→后(Δ) | {scale_name} 30Q程度 |"
            sep_30q += "---------------------|---------------|"
        lines.append(header_30q)
        lines.append(sep_30q)

        for trial_name, rds in condition_data.items():
            scale_cols_30q = ""
            for scale_name in SCALES:
                sk = _30q_scale_keys[scale_name]
                pre_totals = []
                post_totals = []
                pre_sevs = []
                post_sevs = []
                for rd in rds:
                    pre_30q = rd["scale_30q"].get("pre")
                    post_30q = rd["scale_30q"].get("post")
                    pt = _extract_30q_total(pre_30q, sk)
                    pot = _extract_30q_total(post_30q, sk)
                    if pt is not None: pre_totals.append(pt)
                    if pot is not None: post_totals.append(pot)
                    s = _get_30q_severity(pre_30q, sk)
                    if s != "—": pre_sevs.append(s)
                    s = _get_30q_severity(post_30q, sk)
                    if s != "—": post_sevs.append(s)

                if pre_totals and post_totals:
                    pre_m = _mean(pre_totals)
                    post_m = _mean(post_totals)
                    delta = post_m - pre_m
                    score_col = f"{pre_m:.0f}→{post_m:.0f}({delta:+.0f})"
                elif pre_totals:
                    score_col = f"{_mean(pre_totals):.0f}→—"
                elif post_totals:
                    score_col = f"—→{_mean(post_totals):.0f}"
                else:
                    score_col = "—"

                sev_pre = pre_sevs[0] if pre_sevs else "—"
                sev_post = post_sevs[0] if post_sevs else "—"
                if sev_pre != "—" and sev_post != "—":
                    sev_col = f"{sev_pre}→{sev_post}"
                elif sev_pre != "—":
                    sev_col = f"{sev_pre}→—"
                else:
                    sev_col = "—"

                scale_cols_30q += f"| {score_col} | {sev_col} "

            lines.append(f"| {trial_name} {scale_cols_30q}|")
        lines.append("")

    # 4.3 分别评估 vs 30Q 对比（如果有 30Q 数据）
    if has_30q:
        lines.append("### 4.3 分别评估 vs 30Q 对比\n")
        _30q_scale_keys = {"PHQ-9": "phq9", "BDI-II": "bdi_ii", "SDS": "sds"}

        for scale_name in SCALES:
            sk = _30q_scale_keys[scale_name]
            lines.append(f"**{scale_name}**\n")
            lines.append(f"| 条件 | 分别评估前 | 30Q评估前 | 分别评估后 | 30Q评估后 | 分别程度→ | 30Q程度→ |")
            lines.append("|------|-----------|----------|-----------|----------|----------|---------|")

            for trial_name, rds in condition_data.items():
                sep_pre, sep_post, sep_pre_sev, sep_post_sev = "—", "—", "—", "—"
                q30_pre, q30_post, q30_pre_sev, q30_post_sev = "—", "—", "—", "—"
                for rd in rds:
                    pre_scored = rd["scale_scores"].get("pre", {}).get(scale_name)
                    post_scored = rd["scale_scores"].get("post", {}).get(scale_name)
                    pt = _extract_scale_total(pre_scored) if isinstance(pre_scored, dict) else None
                    pot = _extract_scale_total(post_scored) if isinstance(post_scored, dict) else None
                    if pt is not None: sep_pre = f"{pt:.0f}"
                    if pot is not None: sep_post = f"{pot:.0f}"
                    ps = _get_severity_label(pre_scored) if isinstance(pre_scored, dict) else "—"
                    pos = _get_severity_label(post_scored) if isinstance(post_scored, dict) else "—"
                    if ps != "—": sep_pre_sev = ps
                    if pos != "—": sep_post_sev = pos

                    pre_30q = rd["scale_30q"].get("pre")
                    post_30q = rd["scale_30q"].get("post")
                    pt30 = _extract_30q_total(pre_30q, sk)
                    pot30 = _extract_30q_total(post_30q, sk)
                    if pt30 is not None: q30_pre = f"{pt30:.0f}"
                    if pot30 is not None: q30_post = f"{pot30:.0f}"
                    ps30 = _get_30q_severity(pre_30q, sk)
                    pos30 = _get_30q_severity(post_30q, sk)
                    if ps30 != "—": q30_pre_sev = ps30
                    if pos30 != "—": q30_post_sev = pos30

                lines.append(
                    f"| {trial_name} | {sep_pre} | {q30_pre} | {sep_post} | {q30_post} "
                    f"| {sep_pre_sev}→{sep_post_sev} | {q30_pre_sev}→{q30_post_sev} |"
                )
            lines.append("")
    else:
        lines.append("### 4.2 30Q 合并评估\n")
        lines.append("> 30Q 合并评估数据尚未生成。运行 `python runshells/run_experiment.py --30q` 生成。\n")

    # ═══════════════════════════════════════════════════════════
    # Section 5: 跨条件分析
    # ═══════════════════════════════════════════════════════════
    lines.append("## 5. 跨条件分析\n")

    # 5.1 严重程度效应
    lines.append("### 5.1 严重程度效应\n")
    lines.append("| 严重程度 | 平均 efficacy | 条件数 |")
    lines.append("|----------|-------------|--------|")
    for sev in SEVERITIES:
        group_avgs = []
        for tn, rds in condition_data.items():
            if rds[0]["severity"] == sev:
                for rd in rds:
                    if rd["session_eval"]:
                        group_avgs.append(rd["session_eval"]["avg"])
        if group_avgs:
            lines.append(f"| {sev} | {_fmt_mean_sd(group_avgs)} | {len(group_avgs)} |")
        else:
            lines.append(f"| {sev} | — | 0 |")
    lines.append("")

    # 5.2 人设类型效应
    lines.append("### 5.2 人设类型效应 (dynamic vs static)\n")
    lines.append("| 人设类型 | 平均 efficacy | 条件数 |")
    lines.append("|----------|-------------|--------|")
    for per in PERSONAS:
        group_avgs = []
        for tn, rds in condition_data.items():
            if rds[0]["persona"] == per:
                for rd in rds:
                    if rd["session_eval"]:
                        group_avgs.append(rd["session_eval"]["avg"])
        if group_avgs:
            lines.append(f"| {per} | {_fmt_mean_sd(group_avgs)} | {len(group_avgs)} |")
        else:
            lines.append(f"| {per} | — | 0 |")
    lines.append("")

    # 5.3 交互效应矩阵
    lines.append("### 5.3 交互效应 (严重程度 × 人设类型)\n")
    lines.append("| | dynamic | static |")
    lines.append("|---|---------|--------|")
    for sev in SEVERITIES:
        row = f"| **{sev}** |"
        for per in PERSONAS:
            avgs = []
            for tn, rds in condition_data.items():
                if rds[0]["severity"] == sev and rds[0]["persona"] == per:
                    for rd in rds:
                        if rd["session_eval"]:
                            avgs.append(rd["session_eval"]["avg"])
            if avgs:
                row += f" {_fmt_mean_sd(avgs)} |"
            else:
                row += " — |"
        lines.append(row)
    lines.append("")

    # 5.4 量表与 session_eval 一致性
    lines.append("### 5.4 量表与 session_eval 一致性\n")
    consistency_data = []
    for tn, rds in condition_data.items():
        for rd in rds:
            if not rd["session_eval"]:
                continue
            for scale_name in SCALES:
                pre_scored = rd["scale_scores"].get("pre", {}).get(scale_name)
                post_scored = rd["scale_scores"].get("post", {}).get(scale_name)
                if isinstance(pre_scored, dict) and isinstance(post_scored, dict):
                    pre_t = _extract_scale_total(pre_scored)
                    post_t = _extract_scale_total(post_scored)
                    if pre_t is not None and post_t is not None:
                        consistency_data.append({
                            "trial": tn,
                            "round": rd["round"],
                            "scale": scale_name,
                            "scale_change": post_t - pre_t,
                            "efficacy_avg": rd["session_eval"]["avg"],
                        })

    if consistency_data:
        lines.append("| 条件 | 量表 | 轮次 | 量表变化 | 平均 efficacy | 一致性 |")
        lines.append("|------|------|------|---------|-------------|--------|")
        for cd in consistency_data:
            # 一致性：量表分数下降（变好）对应 efficacy 上升（变好）
            scale_down = cd["scale_change"] < 0  # 量表分下降 = 症状减轻
            efficacy_high = cd["efficacy_avg"] >= 80  # 高 efficacy = 效果好
            consistent = "✓" if (scale_down and efficacy_high) or (not scale_down and not efficacy_high) else "✗"
            lines.append(
                f"| {cd['trial']} | {cd['scale']} | R{cd['round']} "
                f"| {cd['scale_change']:+.0f} | {cd['efficacy_avg']:.1f} | {consistent} |"
            )
        lines.append("")
    else:
        lines.append("> 量表评分数据不足，无法进行一致性分析。\n")

    # 5.5 多轮稳定性分析（仅多轮时显示）
    if max_rounds > 1:
        lines.append("### 5.5 多轮稳定性分析\n")
        lines.append("| 条件 | 轮间 efficacy 均值 | SD | CV | 稳定性 |")
        lines.append("|------|------------------|-----|-----|--------|")
        for tn, rds in condition_data.items():
            round_avgs = [rd["session_eval"]["avg"] for rd in rds if rd["session_eval"]]
            if len(round_avgs) < 2:
                lines.append(f"| {tn} | {_fmt_mean_sd(round_avgs)} | — | — | 数据不足 |")
                continue
            m = _mean(round_avgs)
            s = _std(round_avgs)
            cv = s / m if m and m != 0 else 0
            stability = "稳定" if cv < 0.15 else ("波动" if cv < 0.30 else "高波动")
            lines.append(f"| {tn} | {m:.1f} | {s:.1f} | {cv:.2f} | {stability} |")
        lines.append("")

    # ── 写入报告 ──
    report_content = "\n".join(lines)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)

    print(f"  [OK] 报告已生成: {report_path}")
    print(f"  包含 {len(condition_data)} 个条件 × {total_trials} 次试验的数据")


# ═══════════════════════════════════════════════════════════════
# 量表评估（基于已有 checkpoint）
# ═══════════════════════════════════════════════════════════════

SCALE_AGENT_DIR = os.path.join(BASE_DIR, "customization", "depression_scale_agent")
SCALE_QUESTIONS_DIR = os.path.join(SCALE_AGENT_DIR, "questions", "templates")

# 量表评估需要完整的依赖环境（llama_index 等），通过 conda 环境子进程调用
SCALE_WORKER_SCRIPT = os.path.join(BASE_DIR, "runshells", "run_scale_worker.py")
CONDA_PYTHON = "/mnt/nvme1/zxou/miniconda3/envs/generative_agents_py310/bin/python"


def _find_checkpoint_dir(trial_name: str) -> Optional[str]:
    """找到匹配的 checkpoint 目录（带时间戳后缀的最新目录）。"""
    if not os.path.isdir(CHECKPOINTS_ROOT):
        return None
    all_dirs = [
        d for d in os.listdir(CHECKPOINTS_ROOT)
        if os.path.isdir(os.path.join(CHECKPOINTS_ROOT, d)) and d.startswith("Counsel-")
    ]
    candidates = sorted(
        [d for d in all_dirs if d == trial_name or d.startswith(trial_name + "-")],
        reverse=True,
    )
    if not candidates:
        return None
    with_ts = [c for c in candidates if c != trial_name]
    if with_ts:
        return os.path.join(CHECKPOINTS_ROOT, with_ts[0])
    return os.path.join(CHECKPOINTS_ROOT, candidates[0])


def run_scale_evaluation(
    scale_file: str = "30Q.jsonl",
    agent_name: str = "卡布达",
    dry_run: bool = False,
    condition_filter: Optional[str] = None,
) -> None:
    """对每个实验条件的 checkpoint 运行量表评估（旧接口，兼容 --scale-only）。

    通过子进程调用 run_scale_worker.py。
    """
    question_path = os.path.join(SCALE_QUESTIONS_DIR, scale_file)
    if not os.path.exists(question_path):
        print(f"  [ERROR] 量表文件不存在: {question_path}")
        return

    conditions = ALL_CONDITIONS
    if condition_filter:
        conditions = [c for c in conditions if c[0] == condition_filter]
        if not conditions:
            print(f"  [ERROR] 未找到条件: {condition_filter}")
            return

    print(f"  量表: {scale_file}")
    print(f"  Agent: {agent_name}")
    print(f"  条件数: {len(conditions)}\n")

    for trial_name, severity, persona in conditions:
        print(f"--- {trial_name} ({severity}/{persona}) ---")

        cp_dir = _find_checkpoint_dir(trial_name)
        if not cp_dir:
            print(f"  [SKIP] checkpoint 不存在: {trial_name}")
            continue

        cp_name = os.path.basename(cp_dir)
        last_snapshot = _resolve_last_snapshot(cp_name)
        if not last_snapshot:
            print(f"  [SKIP] 无快照: {cp_name}")
            continue

        print(f"  [CHECKPOINT] {cp_name} → {last_snapshot}")

        if dry_run:
            print(f"  [DRY-RUN] 跳过量表评估")
            continue

        data_dir = _find_trial_dir_in_experiment_data(trial_name)
        if not data_dir:
            print(f"  [WARN] experiment_data 目录不存在，跳过保存")
            continue

        output_path = os.path.join(data_dir, "scales", scale_file.replace('.jsonl', '_answered.jsonl'))
        ok = _run_scale_against_snapshot(cp_name, last_snapshot, agent_name, scale_file, output_path)
        if ok:
            print(f"  [OK] 量表结果已保存: {output_path}")
        else:
            print(f"  [ERROR] 量表评估失败")

    print(f"\n{'='*60}")
    print("量表评估完成")


def _find_trial_dir_in_experiment_data(trial_name: str) -> Optional[str]:
    """在 experiment_data 中查找条件的输出目录。"""
    if not os.path.isdir(EXPERIMENT_DATA_ROOT):
        return None
    all_dirs = [
        d for d in os.listdir(EXPERIMENT_DATA_ROOT)
        if os.path.isdir(os.path.join(EXPERIMENT_DATA_ROOT, d)) and d.startswith("Counsel-")
    ]
    candidates = sorted(
        [d for d in all_dirs if d == trial_name or d.startswith(trial_name + "-")],
        reverse=True,
    )
    if not candidates:
        return None
    with_ts = [c for c in candidates if c != trial_name]
    if with_ts:
        return os.path.join(EXPERIMENT_DATA_ROOT, with_ts[0])
    return os.path.join(EXPERIMENT_DATA_ROOT, candidates[0])


# ═══════════════════════════════════════════════════════════════
# 前后量表评估辅助函数
# ═══════════════════════════════════════════════════════════════

def _find_first_snapshot(cp_dir: str) -> Optional[str]:
    """找到 checkpoint 目录下最早的快照文件（= 治疗前基线）。

    对比 app.py:resolve_snapshot_file() 默认返回最后快照（治疗后）。
    """
    if not os.path.isdir(cp_dir):
        return None
    files = sorted(
        f for f in os.listdir(cp_dir)
        if f.startswith("simulate-") and f.endswith(".json")
    )
    return files[0] if files else None


def _run_scale_against_snapshot(
    cp_name: str,
    snapshot_file: str,
    agent_name: str,
    question_file: str,
    output_path: str,
) -> bool:
    """通过子进程调用 run_scale_worker.py 让 agent 回答量表题目。

    返回 True 表示成功，结果保存到 output_path。
    """
    cmd = [
        CONDA_PYTHON, SCALE_WORKER_SCRIPT,
        "--cp-name", cp_name,
        "--snapshot", snapshot_file,
        "--agent", agent_name,
        "--question-file", question_file,
        "--output", output_path,
    ]
    print(f"    [RUN] {' '.join(cmd)}")
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=1800,  # 30 分钟超时
        )
        if result.stdout:
            for line in result.stdout.strip().split("\n"):
                print(f"      {line}")
        if result.returncode != 0:
            print(f"    [ERROR] worker 返回码: {result.returncode}")
            if result.stderr:
                for line in result.stderr.strip().split("\n")[:5]:
                    print(f"      {line}")
            return False
        return os.path.exists(output_path)
    except subprocess.TimeoutExpired:
        print(f"    [ERROR] worker 超时（30 分钟）")
        return False
    except Exception as e:
        print(f"    [ERROR] worker 调用失败: {e}")
        return False


def _score_single_scale(answers_path: str, scoring_prompt_path: str, output_path: str) -> Optional[dict]:
    """通过子进程调用 run_score_worker.py 用 ExpertLLM (DeepSeek API) 评分。

    返回解析后的 JSON 评分结果，失败时返回 None。
    """
    if not os.path.exists(scoring_prompt_path):
        print(f"    [WARN] 评分 prompt 不存在: {scoring_prompt_path}")
        return None

    cmd = [
        CONDA_PYTHON,
        os.path.join(BASE_DIR, "runshells", "run_score_worker.py"),
        "--answers", answers_path,
        "--scoring-prompt", scoring_prompt_path,
        "--output", output_path,
    ]
    print(f"    [SCORE] {' '.join(cmd[-4:])}")
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5 分钟超时
        )
        if result.stdout:
            for line in result.stdout.strip().split("\n"):
                print(f"      {line}")
        if result.returncode != 0:
            print(f"    [ERROR] score worker 返回码: {result.returncode}")
            if result.stderr:
                for line in result.stderr.strip().split("\n")[:3]:
                    print(f"      {line}")
            return None

        if os.path.exists(output_path):
            with open(output_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return None
    except subprocess.TimeoutExpired:
        print(f"    [ERROR] score worker 超时")
        return None
    except Exception as e:
        print(f"    [ERROR] score worker 失败: {e}")
        return None


def _build_session_timeline(scores: list, session_ends: int) -> str:
    """生成 session_eval 时间线的 Markdown 文本。"""
    if not scores:
        return "> 无 session_eval 数据"

    lines = []
    for i, s in enumerate(scores):
        if i == 0:
            trend = "—"
        elif s > scores[i - 1]:
            trend = "↑"
        elif s < scores[i - 1]:
            trend = "↓"
        else:
            trend = "→"
        lines.append(f"  {i+1} | {s} | {trend}")

    return "\n".join(lines)


def _resolve_last_snapshot(cp_name: str) -> Optional[str]:
    """获取 checkpoint 目录的最后一个快照文件名（不依赖 app.py）。"""
    cp_dir = os.path.join(CHECKPOINTS_ROOT, cp_name)
    if not os.path.isdir(cp_dir):
        return None
    files = sorted(
        f for f in os.listdir(cp_dir)
        if f.startswith("simulate-") and f.endswith(".json")
    )
    return files[-1] if files else None


def _load_answered_jsonl(path: str) -> List[dict]:
    """加载 answered.jsonl 文件。"""
    if not os.path.exists(path):
        return []
    answers = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                answers.append(json.loads(line))
    return answers


def run_pre_post_scale(
    run_name: str,
    agent_name: str = "卡布达",
    dry_run: bool = False,
) -> None:
    """对已有 checkpoint 运行治疗前/后量表评估。

    通过子进程调用 run_scale_worker.py（使用正确的 conda 环境）。
    """
    print(f"\n  --- 前后量表评估 ---")

    cp_dir = _find_checkpoint_dir(run_name)
    if not cp_dir:
        print(f"    [SKIP] checkpoint 不存在: {run_name}")
        return

    cp_name = os.path.basename(cp_dir)

    # 找最早快照（治疗前）和最晚快照（治疗后）
    first_snapshot = _find_first_snapshot(cp_dir)
    last_snapshot = _resolve_last_snapshot(cp_name)
    if not first_snapshot or not last_snapshot:
        print(f"    [SKIP] checkpoint 无快照文件: {cp_dir}")
        return

    print(f"    治疗前快照: {first_snapshot}")
    print(f"    治疗后快照: {last_snapshot}")

    if first_snapshot == last_snapshot:
        print(f"    [WARN] 只有一个快照，前后评估将使用同一快照")

    # 找到 experiment_data 输出目录
    data_dir = _find_trial_dir_in_experiment_data(run_name)
    if not data_dir:
        print(f"    [WARN] experiment_data 目录不存在，跳过保存")
        return

    if dry_run:
        print(f"    [DRY-RUN] 跳过量表评估")
        return

    scale_results = {"pre": {}, "post": {}, "change": {}}

    for scale_name, scale_cfg in SCALES.items():
        q_file = scale_cfg["question_file"]
        s_prompt = os.path.join(scale_cfg["scoring_prompt_dir"], scale_cfg["scoring_prompt"])
        q_path = os.path.join(SCALE_QUESTIONS_DIR, q_file)

        if not os.path.exists(q_path):
            print(f"    [SKIP] 量表题目文件不存在: {q_file} (预期 {scale_cfg['items']} 题)")
            continue

        print(f"\n    === 量表: {scale_name} ({scale_cfg['items']} 题) ===")

        # 治疗前
        print(f"    [PRE] 治疗前评估 ({first_snapshot})")
        pre_path = os.path.join(data_dir, "scales", f"{scale_name}_pre_answered.jsonl")
        pre_score_path = os.path.join(data_dir, "scales", f"{scale_name}_pre_scored.json")
        ok = _run_scale_against_snapshot(cp_name, first_snapshot, agent_name, q_file, pre_path)
        if ok:
            pre_answers = _load_answered_jsonl(pre_path)
            print(f"    [SAVE] {pre_path} ({len(pre_answers)} 题)")
            pre_scored = _score_single_scale(pre_path, s_prompt, pre_score_path)
            if pre_scored:
                scale_results["pre"][scale_name] = pre_scored
        else:
            print(f"    [WARN] 治疗前量表评估失败")

        # 治疗后
        print(f"    [POST] 治疗后评估 ({last_snapshot})")
        post_path = os.path.join(data_dir, "scales", f"{scale_name}_post_answered.jsonl")
        post_score_path = os.path.join(data_dir, "scales", f"{scale_name}_post_scored.json")
        ok = _run_scale_against_snapshot(cp_name, last_snapshot, agent_name, q_file, post_path)
        if ok:
            post_answers = _load_answered_jsonl(post_path)
            print(f"    [SAVE] {post_path} ({len(post_answers)} 题)")
            post_scored = _score_single_scale(post_path, s_prompt, post_score_path)
            if post_scored:
                scale_results["post"][scale_name] = post_scored
        else:
            print(f"    [WARN] 治疗后量表评估失败")

        # 计算变化
        pre_total = _extract_scale_total(scale_results["pre"].get(scale_name, {}))
        post_total = _extract_scale_total(scale_results["post"].get(scale_name, {}))
        if pre_total is not None and post_total is not None:
            delta = post_total - pre_total
            pct = round(delta / pre_total * 100, 1) if pre_total != 0 else 0
            scale_results["change"][scale_name] = {"delta": delta, "pct": pct}
            print(f"    [CHANGE] {scale_name}: {pre_total} → {post_total} (Δ={delta}, {pct}%)")

    # 保存聚合评分
    scores_path = os.path.join(data_dir, "scales", "scale_scores.json")
    with open(scores_path, "w", encoding="utf-8") as f:
        json.dump(scale_results, f, ensure_ascii=False, indent=2)
    print(f"\n    [SAVE] {scores_path}")


def run_30q_evaluation(
    agent_name: str = "卡布达",
    dry_run: bool = False,
    condition_filter: Optional[str] = None,
) -> None:
    """对每个实验条件的 checkpoint 运行 30Q 合并评估（治疗前/后各一次）。

    30Q = 30 道抑郁筛查问题一次性问答，然后用 30Q综合评估提示词.md 一次性评分
    三个量表（PHQ-9/BDI-II/SDS）。
    结果保存为 scale_30Q_pre/post_scored.json。
    """
    scoring_prompt = os.path.join(BASE_DIR, "30Q综合评估提示词.md")
    if not os.path.exists(scoring_prompt):
        print(f"  [ERROR] 30Q 评分提示词不存在: {scoring_prompt}")
        return

    conditions = ALL_CONDITIONS
    if condition_filter:
        conditions = [c for c in conditions if c[0] == condition_filter]

    print(f"{'='*60}")
    print("30Q 合并评估")
    print(f"  Agent: {agent_name}")
    print(f"  条件数: {len(conditions)}")
    print(f"{'='*60}")

    for trial_name, severity, persona in conditions:
        print(f"\n--- {trial_name} ({severity}/{persona}) ---")

        cp_dir = _find_checkpoint_dir(trial_name)
        if not cp_dir:
            print(f"  [SKIP] checkpoint 不存在: {trial_name}")
            continue

        cp_name = os.path.basename(cp_dir)
        first_snapshot = _find_first_snapshot(cp_dir)
        last_snapshot = _resolve_last_snapshot(cp_name)
        if not first_snapshot or not last_snapshot:
            print(f"  [SKIP] 无快照: {cp_dir}")
            continue

        data_dir = _find_trial_dir_in_experiment_data(trial_name)
        if not data_dir:
            print(f"  [WARN] experiment_data 目录不存在，跳过")
            continue

        for phase, snapshot in [("pre", first_snapshot), ("post", last_snapshot)]:
            phase_label = "治疗前" if phase == "pre" else "治疗后"
            answered_path = os.path.join(data_dir, "scales", f"30Q_{phase}_answered.jsonl")
            scored_path = os.path.join(data_dir, "scales", f"30Q_{phase}_scored.json")

            # 如果已有评分结果则跳过
            if os.path.exists(scored_path):
                print(f"  [{phase.upper()}] {phase_label}: 已有评分，跳过 ({scored_path})")
                continue

            print(f"  [{phase.upper()}] {phase_label} ({snapshot})")

            if dry_run:
                print(f"    [DRY-RUN] 跳过 30Q 问答")
                continue

            # 1. 问答
            if not os.path.exists(answered_path):
                ok = _run_scale_against_snapshot(cp_name, snapshot, agent_name, "30Q.jsonl", answered_path)
                if not ok:
                    print(f"    [ERROR] 30Q 问答失败")
                    continue
            else:
                print(f"    [SKIP] 已有问答: {answered_path}")

            # 2. 评分
            print(f"    [SCORE] 30Q 综合评分...")
            scored = _score_single_scale(answered_path, scoring_prompt, scored_path)
            if scored:
                print(f"    [OK] 评分完成: {scored_path}")
            else:
                print(f"    [WARN] 评分失败")

    print(f"\n{'='*60}")
    print("30Q 合并评估完成")


def _extract_scale_total(scored_result: dict) -> Optional[float]:
    """从评分结果中提取总分（适配不同量表的 JSON 结构）。"""
    if not scored_result:
        return None
    # SDS: standard_score 或 total_score_raw
    # PHQ-9: total_score
    # BDI-II: total_score
    for key in ["total_score", "total_score_raw", "standard_score"]:
        val = scored_result.get(key)
        if val is not None:
            try:
                return float(val)
            except (TypeError, ValueError):
                pass
    return None


def _get_severity_label(scored_result: dict) -> str:
    """从评分结果中提取严重程度标签。"""
    if not scored_result:
        return "—"
    for key in ["severity", "severity_by_index", "severity_by_standard_score"]:
        val = scored_result.get(key)
        if val:
            return str(val)
    return "—"


# ═══════════════════════════════════════════════════════════════
# 单次试验
# ═══════════════════════════════════════════════════════════════

def run_single_trial(
    trial_name: str,
    severity: str,
    persona: str,
    dry_run: bool = False,
    skip_simulation: bool = False,
    sim_step: int = SIM_STEP,
    round_idx: int = 1,
    local_llm: bool = False,
) -> None:
    """执行一次完整试验。"""
    print(f"\n{'='*60}")
    print(f"Trial: {trial_name}  |  severity={severity}  |  persona={persona}  |  round={round_idx}")
    print(f"{'='*60}")

    # 1. 替换配置
    print(f"\n--- 配置替换 ---")
    prepare_agent_config(severity, dry_run=dry_run)
    prepare_depression_config(severity, persona, dry_run=dry_run)
    prepare_global_config(persona, dry_run=dry_run, local_llm=local_llm)

    # 2. 运行模拟
    run_name = trial_name  # fallback
    if not skip_simulation:
        print(f"\n--- 运行模拟 ---")
        success, run_name = run_simulation(trial_name, dry_run=dry_run, sim_step=sim_step, local_llm=local_llm)
        if not success and not dry_run:
            print(f"  [WARN] 模拟未成功完成，继续后续步骤...")

        # 3. 合并咨询记录
        print(f"\n--- 合并咨询记录 ---")
        merge_dialogues(run_name, dry_run=dry_run)
    else:
        print(f"\n--- 跳过模拟 ---")

    # 4. 收集数据（在 restore 之前，这样当前试验的配置还没被覆盖）
    print(f"\n--- 收集数据 ---")
    collect_trial_data(run_name, severity, persona, dry_run=dry_run, round_idx=round_idx)

    # 5. 前后量表评估
    if not skip_simulation:
        run_pre_post_scale(run_name, agent_name="卡布达", dry_run=dry_run)
    else:
        print(f"\n--- 跳过量表评估（skip-simulation 模式）---")

    # 6. 恢复原始配置
    print(f"\n--- 恢复配置 ---")
    restore_configs(dry_run=dry_run)

    print(f"\n--- Trial {trial_name} (Round {round_idx}) 完成 ---")


# ═══════════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════════

def recover_old_trial_data():
    """从旧报告数据恢复试验目录（仅 session_eval）。"""
    from datetime import datetime

    print("恢复旧报告试验数据...")
    recovered_count = 0

    for trial_name, data in RECOVERED_SESSION_EVAL.items():
        severity = next((s for t, s, p in ALL_CONDITIONS if t == trial_name), "unknown")
        persona = next((p for t, s, p in ALL_CONDITIONS if t == trial_name), "unknown")

        dir_name = f"{trial_name}-0512-recovered"
        data_dir = os.path.join(EXPERIMENT_DATA_ROOT, dir_name)

        if os.path.exists(data_dir):
            print(f"  [SKIP] {dir_name} 已存在")
            continue

        os.makedirs(data_dir, exist_ok=True)

        meta = {
            "trial_name": trial_name,
            "severity": severity,
            "persona": persona,
            "round": 1,
            "data_source": "recovered_from_report",
            "recovered_at": datetime.now().isoformat(),
            "note": "session_eval data recovered from previous report; no conversation/scale data",
            "dialogue_turns": data["dialogue_turns"],
        }
        with open(os.path.join(data_dir, "trial_meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        sessions = []
        for i, score in enumerate(data["scores"]):
            sessions.append({
                "meeting": f"m_recovered_{i+1}",
                "turns": [],
                "session_eval": {
                    "efficacy_score": score,
                    "session_end": None,
                    "note": "recovered from report",
                },
            })
        judge_data = {"sessions": sessions}
        with open(os.path.join(data_dir, "judge_conversation.json"), "w", encoding="utf-8") as f:
            json.dump(judge_data, f, ensure_ascii=False, indent=2)

        print(f"  [OK] {dir_name}: {len(data['scores'])} sessions, avg={sum(data['scores'])/len(data['scores']):.1f}")
        recovered_count += 1

    print(f"\n恢复完成: {recovered_count} 个条件")


def main():
    parser = argparse.ArgumentParser(description="抑郁症治疗遍历实验")
    parser.add_argument(
        "--condition", type=str, default=None,
        help="只跑指定条件（如 Counsel-MOD-DYN）",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="只打印操作，不实际执行",
    )
    parser.add_argument(
        "--skip-simulation", action="store_true",
        help="跳过模拟运行，只收集已有数据",
    )
    parser.add_argument(
        "--step", type=int, default=None,
        help=f"模拟步数（默认: {SIM_STEP}）",
    )
    parser.add_argument(
        "--report-only", action="store_true",
        help="只生成分析报告（基于已有数据）",
    )
    parser.add_argument(
        "--scale-only", action="store_true",
        help="对已有 checkpoint 运行量表评估（不重新模拟）",
    )
    parser.add_argument(
        "--scale-file", type=str, default="30Q.jsonl",
        help="量表题目文件名（默认: 30Q.jsonl）",
    )
    parser.add_argument(
        "--scale-agent", type=str, default="卡布达",
        help="量表评估的目标 agent（默认: 卡布达）",
    )
    parser.add_argument(
        "--rounds", type=int, default=1,
        help="循环轮数（默认: 1，正式实验用 3）",
    )
    parser.add_argument(
        "--start-round", type=int, default=1,
        help="从第几轮开始（用于跳过已完成的轮次）",
    )
    parser.add_argument(
        "--pre-post-scale", action="store_true",
        help="对已有 checkpoint 运行前后量表评估（不重新模拟）",
    )
    parser.add_argument(
        "--30q", "--run-30q", action="store_true", dest="run_30q",
        help="对已有 checkpoint 运行 30Q 合并评估（治疗前/后各一次 30Q 问答 + 综合评分）",
    )
    parser.add_argument(
        "--recover-old-data", action="store_true",
        help="从旧报告恢复试验数据（仅 session_eval）",
    )
    parser.add_argument(
        "--local-llm", action="store_true",
        help="使用本地 Ollama qwen3:32b 替代 DeepSeek API",
    )
    args = parser.parse_args()

    # 恢复旧数据模式
    if args.recover_old_data:
        recover_old_trial_data()
        return

    # 仅生成报告模式
    if args.report_only:
        generate_report(dry_run=args.dry_run)
        return

    # 仅量表评估模式（旧接口，保持兼容）
    if args.scale_only:
        run_scale_evaluation(
            scale_file=args.scale_file,
            agent_name=args.scale_agent,
            dry_run=args.dry_run,
            condition_filter=args.condition,
        )
        return

    # 前后量表评估模式（对已有 checkpoint，不重新模拟）
    if args.pre_post_scale:
        conditions = ALL_CONDITIONS
        if args.condition:
            conditions = [c for c in conditions if c[0] == args.condition]
        for trial_name, severity, persona in conditions:
            cp_dir = _find_checkpoint_dir(trial_name)
            if not cp_dir:
                print(f"[SKIP] checkpoint 不存在: {trial_name}")
                continue
            cp_name = os.path.basename(cp_dir)
            run_pre_post_scale(cp_name, agent_name=args.scale_agent, dry_run=args.dry_run)
        return

    # 30Q 合并评估模式
    if getattr(args, "run_30q", False):
        run_30q_evaluation(
            agent_name=args.scale_agent,
            dry_run=args.dry_run,
            condition_filter=args.condition,
        )
        return

    # 步数覆盖
    sim_step = SIM_STEP
    if args.step is not None:
        sim_step = args.step

    # 筛选条件
    conditions = ALL_CONDITIONS
    if args.condition:
        matched = [c for c in conditions if c[0] == args.condition]
        if not matched:
            print(f"错误: 未找到条件 '{args.condition}'")
            print(f"可用条件: {[c[0] for c in ALL_CONDITIONS]}")
            sys.exit(1)
        conditions = matched

    rounds = max(1, args.rounds)
    start_round = max(1, args.start_round)
    if start_round > rounds:
        print(f"错误: --start-round ({start_round}) > --rounds ({rounds})")
        sys.exit(1)
    total_trials = len(conditions) * (rounds - start_round + 1)
    print(f"实验计划: {len(conditions)} 个条件 × {rounds - start_round + 1} 轮 (R{start_round}-R{rounds}) = {total_trials} 次试验")
    print(f"条件列表: {[c[0] for c in conditions]}")
    print(f"轮数: {rounds}")
    print(f"Dry run: {args.dry_run}")
    print(f"Skip simulation: {args.skip_simulation}")

    # 备份原始配置
    print(f"\n--- 备份原始配置 ---")
    backup_configs(dry_run=args.dry_run)

    # 多轮循环
    completed = []
    failed = []
    try:
        for round_idx in range(start_round, rounds + 1):
            if rounds > 1:
                print(f"\n{'='*60}")
                print(f"= Round {round_idx}/{rounds}")
                print(f"{'='*60}")

            for i, (trial_name, severity, persona) in enumerate(conditions):
                print(f"\n{'#'*60}")
                print(f"# [R{round_idx} {i+1}/{len(conditions)}] {trial_name}")
                print(f"{'#'*60}")
                try:
                    run_single_trial(
                        trial_name, severity, persona,
                        dry_run=args.dry_run,
                        skip_simulation=args.skip_simulation,
                        sim_step=sim_step,
                        round_idx=round_idx,
                        local_llm=args.local_llm,
                    )
                    completed.append((trial_name, round_idx))
                except Exception as e:
                    print(f"  [ERROR] Trial {trial_name} R{round_idx} 异常: {e}")
                    failed.append((trial_name, round_idx))
                    restore_configs(dry_run=args.dry_run)
    finally:
        print(f"\n--- 最终恢复配置 ---")
        restore_configs(dry_run=args.dry_run)

    # 总结
    print(f"\n{'='*60}")
    print(f"实验完成: {len(completed)} 成功 / {len(failed)} 失败")
    if rounds > 1:
        for r in range(1, rounds + 1):
            r_ok = [c[0] for c in completed if c[1] == r]
            r_fail = [c[0] for c in failed if c[1] == r]
            print(f"  Round {r}: {len(r_ok)} 成功 {r_ok}")
            if r_fail:
                print(f"          {len(r_fail)} 失败 {r_fail}")
    else:
        print(f"  成功: {[c[0] for c in completed]}")
        if failed:
            print(f"  失败: {[c[0] for c in failed]}")
    print(f"{'='*60}")

    # 生成分析报告
    if completed:
        generate_report(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
