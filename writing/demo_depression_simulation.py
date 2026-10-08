#!/usr/bin/env python3
"""
==========================================================================
Demo: 基于生成式智能体的抑郁症模拟、治疗与评估系统
==========================================================================

本脚本展示本项目核心 pipeline 的完整运行流程:
  1. 初始化抑郁症患者智能体（卡布达）—— 含抑郁状态引擎
  2. 初始化心理医生智能体（蜻蜓队长）—— 含 CBT/PST 治疗模块
  3. 执行一次 CBT 结构化治疗对话（Session 1: 信息收集）
  4. 使用标准化量表（PHQ-9）对患者进行阶段性评估

运行前置条件:
  - 已启动 vLLM 服务 (Qwen3-8B @ port 18000, BGE-M3 @ port 18001)
    bash runshells/vllm_services.sh start
  - 已配置 .env 文件 (DEEPSEEK_API_KEY=sk-xxxx) 用于专家评估模型

用法:
  python writing/demo_depression_simulation.py [--verbose] [--steps 3]

参考文档:
  - writing/基于AI对抑郁症进行模拟、治疗与评估 - 研究进展（附示例）.pdf
  - writing/NMH_depression_paper_analysis.md

==========================================================================
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime

# ── 确保项目根目录在 sys.path 中 ──────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)


# ══════════════════════════════════════════════════════════════════════════
#  辅助函数
# ══════════════════════════════════════════════════════════════════════════

def separator(title: str, width: int = 72):
    """打印分节标题"""
    print(f"\n{'═' * width}")
    print(f"  {title}")
    print(f"{'═' * width}\n")


def load_json(path: str) -> dict:
    """加载 JSON 文件"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: str) -> list:
    """加载 JSONL 文件"""
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def print_dialogue(speaker: str, text: str, color: str = ""):
    """格式化打印对话"""
    colors = {
        "patient": "\033[36m",    # 青色 - 患者
        "doctor":  "\033[33m",    # 黄色 - 医生
        "eval":    "\033[35m",    # 紫色 - 评估
        "system":  "\033[90m",    # 灰色 - 系统
        "result":  "\033[32m",    # 绿色 - 结果
        "reset":   "\033[0m",
    }
    c = colors.get(color, "")
    r = colors["reset"]
    print(f"  {c}[{speaker}]{r}  {text}")


# ══════════════════════════════════════════════════════════════════════════
#  Phase 1: 系统配置加载与展示
# ══════════════════════════════════════════════════════════════════════════

def phase1_show_system_config():
    """展示系统核心配置"""
    separator("Phase 1: 系统配置总览")

    config = load_json("data/config.json")

    # ── LLM 后端 ─────────────────────────────────────────────────────────
    llm_cfg = config.get("agent", {}).get("think", {}).get("llm", {})
    emb_cfg = config.get("agent", {}).get("associate", {}).get("embedding", {})

    print("【LLM 后端配置】")
    print(f"  智能体认知模型:  {llm_cfg.get('model', 'N/A')}")
    print(f"  智能体认知端点:  {llm_cfg.get('base_url', 'N/A')}")
    print(f"  嵌入模型:        {emb_cfg.get('model', 'N/A')}")
    print(f"  嵌入模型端点:    {emb_cfg.get('base_url', 'N/A')}")

    # ── 干预系统 ─────────────────────────────────────────────────────────
    interv_cfg = config.get("intervention", {})
    forced_llm = interv_cfg.get("forced_llm", {})

    print("\n【干预系统配置】")
    print(f"  干预系统启用:    {interv_cfg.get('enabled', False)}")
    print(f"  心理医生:        {interv_cfg.get('doctor', 'N/A')}")
    print(f"  患者列表:        {interv_cfg.get('patients', [])}")
    print(f"  强制对话模型:    {forced_llm.get('model', 'N/A')}")
    print(f"  治疗方案:        {interv_cfg.get('session_prompt_injection', {}).get('namespace', 'N/A')}")

    # ── 抑郁引擎 ─────────────────────────────────────────────────────────
    print("\n【抑郁仿真引擎】")
    dep_config_path = "frontend/static/assets/village/agents/卡布达/depression_config.json"
    if os.path.exists(dep_config_path):
        dep_cfg = load_json(dep_config_path)
        profile = dep_cfg.get("profile", {})
        print(f"  角色:            {profile.get('role', 'N/A')}")
        print(f"  抑郁严重度:      {profile.get('depression_severity', 'N/A')}")
        print(f"  病例摘要:        {profile.get('case_summary', 'N/A')[:60]}...")

        # 展示 complaint chain 的状态列表
        chain = dep_cfg.get("complaint_chain", {})
        stages = chain.get("stages", [])
        initial = chain.get("initial_stage_id", "N/A")
        print(f"  投诉链状态数:    {len(stages)} 个状态")
        print(f"  初始状态:        {initial}")

        # 展示情绪推断配置
        emotion_cfg = dep_cfg.get("emotion", {})
        volatility = emotion_cfg.get("volatility_limit", 0.12) if isinstance(emotion_cfg, dict) else 0.12
        print(f"  情绪波动限幅:    {volatility}")

    return config


# ══════════════════════════════════════════════════════════════════════════
#  Phase 2: 智能体角色展示
# ══════════════════════════════════════════════════════════════════════════

def phase2_show_agent_profiles():
    """展示患者和医生的配置"""
    separator("Phase 2: 智能体角色档案")

    # ── 患者智能体: 卡布达 ────────────────────────────────────────────────
    patient = load_json("frontend/static/assets/village/agents/卡布达/agent.json")
    print("【患者智能体: 卡布达】")
    print(f"  年龄:            {patient['scratch']['age']} 岁")
    print(f"  天生特质:        {patient['scratch']['innate']}")
    print(f"  职业:            {patient.get('profile', {}).get('occupation', 'N/A')}")
    print(f"  学历:            {patient.get('profile', {}).get('education', 'N/A')}")
    print(f"  说话习惯:        {patient.get('profile', {}).get('speaking_habit', 'N/A')}")
    print(f"  当前状态:        {patient.get('currently', 'N/A')[:80]}...")

    # ── 医生智能体: 蜻蜓队长 ──────────────────────────────────────────────
    doctor = load_json("frontend/static/assets/village/agents/蜻蜓队长/agent.json")
    print("\n【心理医生智能体: 蜻蜓队长】")
    print(f"  年龄:            {doctor['scratch']['age']} 岁")
    print(f"  天生特质:        {doctor['scratch']['innate']}")
    print(f"  当前状态:        {doctor.get('currently', 'N/A')[:80]}...")

    # ── 社会关系网络 ─────────────────────────────────────────────────────
    print("\n【虚拟社区关系网络】")
    agents_dir = "frontend/static/assets/village/agents"
    for agent_dir in sorted(Path(agents_dir).iterdir()):
        cfg_path = agent_dir / "agent.json"
        if cfg_path.exists():
            cfg = load_json(str(cfg_path))
            role = "患者" if cfg["name"] == "卡布达" else \
                   "医生" if cfg["name"] == "蜻蜓队长" else "居民"
            print(f"  {cfg['name']:　<6s} [{role}]  — 特质: {cfg['scratch']['innate']}")


# ══════════════════════════════════════════════════════════════════════════
#  Phase 3: 抑郁引擎状态展示
# ══════════════════════════════════════════════════════════════════════════

def phase3_show_depression_engine():
    """展示抑郁仿真引擎的核心机制"""
    separator("Phase 3: 抑郁仿真引擎")

    dep_cfg = load_json("frontend/static/assets/village/agents/卡布达/depression_config.json")

    # ── 投诉链状态机 ─────────────────────────────────────────────────────
    chain = dep_cfg.get("complaint_chain", {})
    initial = chain.get("initial_stage_id", "N/A")
    stages = chain.get("stages", [])

    print("【投诉链 (Complaint Chain) 状态机】")
    print(f"  初始状态: {initial}\n")

    # 展示前3个状态的详细信息
    for shown, stage_data in enumerate(stages[:3]):
        stage_id = stage_data.get("id", "N/A")
        label = stage_data.get("label", "N/A")
        print(f"  状态 [{shown+1}]: {stage_id}")
        print(f"    标签:      {label}")
        print(f"    核心信念:  {stage_data.get('core_belief', 'N/A')[:60]}")

        # 叙事焦点
        nf = stage_data.get("narrative_focus", [])
        if isinstance(nf, list):
            print(f"    叙事焦点:  {', '.join(str(x) for x in nf[:4])}")

        # 说话风格
        style = stage_data.get("speaking_style", {})
        if isinstance(style, dict):
            print(f"    说话风格:  tempo={style.get('tempo', 'N/A')}, "
                  f"tone={style.get('tone', 'N/A')}")

        # 情绪向量
        emo = stage_data.get("emotion_vector", {})
        if isinstance(emo, dict):
            print(f"    情绪向量:  valence={emo.get('valence', 'N/A')}, "
                  f"arousal={emo.get('arousal', 'N/A')}, "
                  f"defensiveness={emo.get('defensiveness', 'N/A')}")

        # 说话风格参数
        speaking = stage_data.get("speaking_style", {})
        if isinstance(speaking, dict):
            tempo = speaking.get("tempo", "N/A")
            disclosure = speaking.get("disclosure", "N/A")
            print(f"    说话风格:  tempo={tempo}, disclosure={disclosure}")

        # 推进/保持信号
        advance = stage_data.get("advance_signals", [])
        hold = stage_data.get("hold_signals", [])
        if advance:
            print(f"    推进信号:  {advance[0][:50]}")
        if hold:
            print(f"    保持信号:  {hold[0][:50]}")

        # 关系修正
        rel_mod = stage_data.get("relation_modifiers", {})
        if isinstance(rel_mod, dict):
            roles = list(rel_mod.keys())
            print(f"    关系修正:  {', '.join(roles)}")

        # 状态转移
        nexts = stage_data.get("next_candidates", [])
        terminal = stage_data.get("is_terminal_stage", False)
        print(f"    可转移到:  {nexts if nexts else '(暂无, 终止状态)' if terminal else '(待配置)'}")
        print()

    remaining = len(stages) - min(3, len(stages))
    if remaining > 0:
        print(f"  ... (还有 {remaining} 个状态)")

    # ── 引擎子模块总览 ───────────────────────────────────────────────────
    print("【引擎五大子模块】")
    modules = [
        ("SessionContextBuilder",    "从对话中提取情境上下文（地点、时间、关系）"),
        ("ComplaintChainManager",    "投诉链状态机管理，决定状态推进/保持"),
        ("EmotionInferencer",       "LLM 驱动的逐轮情绪推断（波动限幅约束）"),
        ("EmotionInferencer",        "逐轮推断瞬时情绪向量（效价/唤醒/防御性等）"),
        ("DynamicPromptBuilder",     "组装最终的抑郁人设提示词注入 LLM"),
    ]
    for name, desc in modules:
        print(f"  {name:<28s} → {desc}")


# ══════════════════════════════════════════════════════════════════════════
#  Phase 4: CBT 治疗对话演示
# ══════════════════════════════════════════════════════════════════════════

def phase4_show_cbt_session():
    """展示 CBT 治疗对话的结构化流程"""
    separator("Phase 4: CBT 结构化治疗对话 (Session 1)")

    prompts = load_json("data/prompts/intervention_prompts.json")
    cbt = prompts.get("CBT", {})

    print("【CBT 治疗方案完整会话结构】\n")
    session_order = [
        "session1", "session_loop",
        "session2.1", "session2.2", "session2.3",
        "session3.1", "session3.2", "session3.3-A", "session3.3-B",
        "session4.1", "session4.2", "session4.3", "session4.4",
    ]
    stage_labels = {
        "session1": "信息收集", "session_loop": "循环采样",
        "session2.1": "认知概念化-标记扭曲", "session2.2": "条件规则识别",
        "session2.3": "核心信念挖掘",
        "session3.1": "认知重构-法庭练习", "session3.2": "行为实验设计",
        "session3.3-A": "成功路径回顾", "session3.3-B": "回避路径处理",
        "session4.1": "旅程回顾", "session4.2": "心理急救包",
        "session4.3": "压力测试复盘", "session4.4": "毕业仪式",
    }

    for i, s in enumerate(session_order):
        label = stage_labels.get(s, "")
        prompt_text = cbt.get(s, "")
        snippet = prompt_text[:80].replace("\n", " ") + "..." if prompt_text else "(未找到)"
        print(f"  {i+1:2d}. {s:<16s} [{label}]")
        print(f"      {snippet}\n")

    # ── 模拟一轮 CBT Session 1 对话 ──────────────────────────────────────
    print("【模拟 CBT Session 1 对话示例】\n")
    dialogue = [
        ("doctor", "蜻蜓队长",
         "你好，卡布达。谢谢你愿意来。这里是一个安全的、不会被评判的空间。"
         "在开始之前，我想先了解一下——最近是什么让你决定来这里的？"),
        ("patient", "卡布达",
         "...其实也没什么特别大的事。就是最近...工作丢了之后，就觉得什么都提不起劲。"
         "朋友叫我出去我也懒得去，晚上也睡不好..."),
        ("doctor", "蜻蜓队长",
         "听起来这段时间你承受了不少。你提到'提不起劲'和'睡不好'——"
         "如果让你回忆最近一次有这种感觉的具体场景，你能描述一下当时发生了什么吗？"
         "当时你心里在想什么？"),
        ("patient", "卡布达",
         "就...前天吧。金龟次郎说有个设计比赛让我试试，我看了下要求就觉得——"
         "'反正我这种连工作都保不住的人，参加也是丢脸'...然后就直接关掉了网页。"
         "那晚翻来覆去到三四点才睡着。"),
        ("doctor", "蜻蜓队长",
         "很好，你描述得很清楚。我们一起来看看这个场景——\n"
         "  情境: 看到设计比赛通知\n"
         "  情绪: 羞耻、无力\n"
         "  想法: '我这种连工作都保不住的人，参加也是丢脸'\n"
         "你注意到没有，你的想法直接影响了你的情绪和后续行为（关掉网页）。"
         "这就是我们在 CBT 中说的'认知三角'——想法、情绪和行为之间的联系。"),
        ("patient", "卡布达",
         "...嗯，好像确实是这样。但我控制不了自己这么想啊..."),
        ("doctor", "蜻蜓队长",
         "这很正常，这些想法常常是自动出现的，我们叫它'自动思维'。"
         "好消息是，当我们开始识别它们之后，就有机会去审视它们。"
         "这是我们接下来几次要一起练习的。今天我们先到这里，下次我们继续。"),
    ]

    for role, speaker, text in dialogue:
        print_dialogue(speaker, text, color=role)
        print()

    # ── 展示 session 后记忆注入 ─────────────────────────────────────────
    print("【记忆注入规则】")
    mem_injections = load_json("data/intervention/memory_injections.json")
    for entry in mem_injections.get("entries", []):
        rule = entry.get("rule", "N/A")
        items = entry.get("items", [])
        print(f"  规则: {rule}")
        for item in items[:2]:
            s = item.get("subject", "")
            p = item.get("predicate", "")
            o = item.get("object", "")
            print(f"    → {s} {p} {o}")
        if len(items) > 2:
            print(f"    ... (共 {len(items)} 条)")
        print()


# ══════════════════════════════════════════════════════════════════════════
#  Phase 5: 阶段性评估演示
# ══════════════════════════════════════════════════════════════════════════

def phase5_show_evaluation():
    """展示标准化量表评估流程"""
    separator("Phase 5: 标准化量表评估 (PHQ-9)")

    # ── 量表题目展示 ─────────────────────────────────────────────────────
    phq9_path = "customization/depression_scale_agent/questions/templates/PHQ-9.jsonl"
    if os.path.exists(phq9_path):
        questions = load_jsonl(phq9_path)
        print(f"【PHQ-9 量表】 ({len(questions)} 个条目)\n")

        for i, item in enumerate(questions):
            q_text = item.get("question", item.get("prompt", item.get("text", str(item))))
            print(f"  Q{i+1}. {q_text[:80]}")
    else:
        print("  (PHQ-9 模板文件未找到, 展示标准 PHQ-9 条目)\n")
        standard_phq9 = [
            "做事时提不起劲或没有兴趣",
            "感到心情低落、沮丧或绝望",
            "入睡困难、睡不安稳或睡眠过多",
            "感觉疲倦或没有活力",
            "食欲不振或吃得太多",
            "觉得自己很糟，或觉得自己是个失败者",
            "对事物专注有困难（例如阅读或看电视时）",
            "动作或说话速度缓慢，或正好相反——烦躁不安",
            "有不如死掉或用某种方式伤害自己的念头",
        ]
        for i, q in enumerate(standard_phq9):
            print(f"  Q{i+1}. {q}")

    # ── 模拟评估对话 ─────────────────────────────────────────────────────
    print("\n【模拟评估对话 (患者回答 PHQ-9)】\n")

    eval_dialogue = [
        ("eval",   "评估系统",  "请回答以下问题：在过去两周内，你是否对做事提不起劲或没有兴趣？"),
        ("patient", "卡布达",   "是的，几乎每天都有。之前喜欢的设计现在完全不想碰。"),
        ("eval",   "评估系统",  "在过去两周内，你是否感到心情低落、沮丧或绝望？"),
        ("patient", "卡布达",   "大部分时间都是这样。觉得什么都不重要了。"),
        ("eval",   "评估系统",  "你是否入睡困难、睡不安稳或睡眠过多？"),
        ("patient", "卡布达",   "经常失眠，躺在床上脑子里全是乱七八糟的想法，要很晚才能睡着。"),
    ]

    for role, speaker, text in eval_dialogue:
        print_dialogue(speaker, text, color=role)
        print()

    # ── 展示评估时间线 ───────────────────────────────────────────────────
    print("【阶段性评估时间线 (Staged Evaluation)】\n")
    timeline = [
        ("T0 (基线)",   "仿真开始前，评估患者初始抑郁严重度"),
        ("Session 4",   "每完成 4 次治疗对话后触发一次评估"),
        ("Session 8",   "中期评估，检测治疗响应趋势"),
        ("Session 12",  "后期评估，评估治疗效果"),
        ("T4 (随访)",   "全部治疗结束后 8 个仿真步骤后进行随访评估"),
    ]
    for label, desc in timeline:
        print(f"  {label:<14s}  {desc}")

    # ── 展示可用量表 ─────────────────────────────────────────────────────
    print("\n【可用评估量表】\n")
    scales = [
        ("PHQ-9",  "患者健康问卷-9", "自评", "9 条目"),
        ("BDI-II", "贝克抑郁量表-II", "自评", "21 条目"),
        ("SDS",    "抑郁自评量表",     "自评", "20 条目"),
    ]
    print(f"  {'量表':<8s} {'名称':<18s} {'类型':<6s} {'条目数':<8s}")
    print(f"  {'─' * 44}")
    for abbr, name, typ, n in scales:
        print(f"  {abbr:<8s} {name:<18s} {typ:<6s} {n:<8s}")


# ══════════════════════════════════════════════════════════════════════════
#  Phase 6: 完整 Pipeline 可视化
# ══════════════════════════════════════════════════════════════════════════

def phase6_show_pipeline():
    """展示完整 Pipeline 架构"""
    separator("Phase 6: 完整 Pipeline 架构")

    pipeline = """
    ┌──────────────────────────────────────────────────────────────────────┐
    │                  基于生成式智能体的抑郁症仿真系统                     │
    ├──────────────────────────────────────────────────────────────────────┤
    │                                                                      │
    │  ┌─────────────┐    ┌──────────────┐    ┌─────────────────────┐     │
    │  │ 抑郁症患者   │    │  心理医生     │    │  评估 LLM           │     │
    │  │ (卡布达)     │    │ (蜻蜓队长)   │    │  (量表评估)         │     │
    │  └──────┬──────┘    └──────┬───────┘    └──────────┬──────────┘     │
    │         │                  │                       │                 │
    │  ┌──────┴──────┐    ┌─────┴────────┐    ┌────────┴─────────┐      │
    │  │ 抑郁引擎    │    │ CBT/PST      │    │ PHQ-9 / BDI-II  │      │
    │  │ · 投诉链    │    │ 结构化会话   │    │ / SDS 量表      │      │
    │  │ · 情绪推断  │    │ · 信息收集   │    │                  │      │
    │  │ · 情绪推断  │    │ · 概念化     │    │ 自动评估时间线:  │      │
    │  │ · 动态提示  │    │ · 重构/实验  │    │ T0 → S4 → S8 →  │      │
    │  │ · 记忆注入  │    │ · 防复发     │    │ S12 → T4        │      │
    │  └──────┬──────┘    └─────┬────────┘    └────────┬─────────┘      │
    │         │                 │                       │                 │
    │         └────────┬────────┘───────────┬───────────┘                 │
    │                  │                    │                              │
    │         ┌────────┴────────┐  ┌────────┴──────────┐                 │
    │         │ 对话记录 &      │  │ 对话法官 (Judge)  │                 │
    │         │ 咨询笔记        │  │ · 监控对话质量    │                 │
    │         │ · 治疗连续性    │  │ · 建议继续/终止  │                 │
    │         └─────────────────┘  └───────────────────┘                 │
    │                                                                      │
    ├──────────────────────────────────────────────────────────────────────┤
    │  LLM 后端:  Qwen3-8B (本地 vLLM) ← 智能体认知                      │
    │             DeepSeek API           ← 强制对话 / 专家评估             │
    │             BGE-M3 (本地 vLLM)     ← 记忆检索嵌入                   │
    └──────────────────────────────────────────────────────────────────────┘
    """
    print(pipeline)

    # ── 实验组设计 ───────────────────────────────────────────────────────
    print("【实验组设计】\n")
    groups = [
        ("G1", "医生干预组",   "完整 CBT/PST 治疗，含抑郁引擎 + 对话法官 + 量表评估"),
        ("G2", "无干预对照组", "无治疗干预，抑郁引擎仍然活跃"),
        ("G3", "普通社交组",   "与其他居民进行随机中性社交对话（替代医生治疗）"),
        ("G4", "咨询室组",     "与 G1 相同但限定在咨询室环境中"),
        ("G5", "负面社交组",   "与其他居民进行负面/有害对话（伤害性对照）"),
    ]
    for gid, name, desc in groups:
        print(f"  {gid}: {name:<12s} — {desc}")


# ══════════════════════════════════════════════════════════════════════════
#  Phase 7: 快速运行接口（需要 vLLM 服务）
# ══════════════════════════════════════════════════════════════════════════

def phase7_interactive_demo(steps: int = 3, verbose: bool = False):
    """交互式运行真实仿真（需要 vLLM 服务已启动）"""
    separator("Phase 7: 交互式仿真运行")

    print("正在尝试连接本地 LLM 服务...\n")

    try:
        from modules.model.llm_model import create_llm_model
        config = load_json("data/config.json")
        llm_cfg = config["agent"]["think"]["llm"]
        llm = create_llm_model(llm_cfg)

        # 测试 LLM 连接
        test_prompt = "请用一句话介绍你自己。"
        response = llm.completion(test_prompt, retry=2)
        print(f"  LLM 连接成功! 模型: {llm_cfg['model']}")
        print(f"  测试响应: {response[:100]}...\n")

    except Exception as e:
        print(f"  ⚠ LLM 服务未启动或连接失败: {e}")
        print(f"  请先运行: bash runshells/vllm_services.sh start")
        print(f"  跳过交互式仿真部分。\n")
        return

    # ── 尝试初始化仿真 ───────────────────────────────────────────────────
    try:
        from start import SimulateServer
        print("正在初始化仿真环境...\n")

        # 使用最小参数启动
        sim_name = f"demo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        server = SimulateServer(
            name=sim_name,
            step=steps,
            stride=60,
            start="20260425-09:30",
            verbose="info",
        )

        print(f"  仿真初始化成功!")
        print(f"  仿真名称: {sim_name}")
        print(f"  仿真步数: {steps}")
        print(f"  步长: 60 分钟/步\n")

        print("正在运行仿真...\n")
        for i in range(steps):
            server.run_step()
            print(f"  Step {i+1}/{steps} 完成")

        print(f"\n  仿真完成! 结果保存在: results/{sim_name}/")

    except Exception as e:
        print(f"  仿真运行出错: {e}")
        print(f"  建议检查配置文件和 LLM 服务状态。\n")


# ══════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Demo: 基于生成式智能体的抑郁症模拟、治疗与评估系统"
    )
    parser.add_argument("--verbose", action="store_true", help="显示详细输出")
    parser.add_argument("--steps", type=int, default=3, help="交互式仿真的步数 (默认 3)")
    parser.add_argument("--no-interactive", action="store_true", help="跳过交互式仿真 (仅展示)")
    args = parser.parse_args()

    print()
    print("╔═══════════════════════════════════════════════════════════════════╗")
    print("║  Demo: 基于生成式智能体的抑郁症模拟、治疗与评估                  ║")
    print("║  AI-Powered Depression Simulation, Treatment & Assessment       ║")
    print("╚═══════════════════════════════════════════════════════════════════╝")
    print(f"\n  项目路径: {PROJECT_ROOT}")
    print(f"  运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Phase 1-6: 展示模式（无需 LLM 服务）
    phase1_show_system_config()
    phase2_show_agent_profiles()
    phase3_show_depression_engine()
    phase4_show_cbt_session()
    phase5_show_evaluation()
    phase6_show_pipeline()

    # Phase 7: 交互式仿真（需要 LLM 服务）
    if not args.no_interactive:
        phase7_interactive_demo(steps=args.steps, verbose=args.verbose)
    else:
        print("\n  (--no-interactive 已跳过交互式仿真)")

    # ── 总结 ─────────────────────────────────────────────────────────────
    separator("Demo 完成")
    print("  本 Demo 展示了系统核心 Pipeline 的六个阶段:")
    print("    1. 系统配置总览        — LLM 后端 / 干预系统 / 抑郁引擎配置")
    print("    2. 智能体角色档案      — 患者/医生/居民的详细人设")
    print("    3. 抑郁仿真引擎        — 投诉图状态机 / LLM情绪推断 / 四层动态Prompt")
    print("    4. CBT 治疗对话        — 结构化会话流程 / 示例对话")
    print("    5. 标准化量表评估      — PHQ-9 / BDI-II / SDS 量表与时间线")
    print("    6. 完整 Pipeline 架构  — 系统全景图 / 实验组设计")
    print("    7. 交互式仿真*         — 实际运行仿真（需 vLLM 服务）")
    print()
    print("  运行完整仿真:")
    print("    python start.py --name demo --step 48 --stride 60 --start 20260425-09:30")
    print()
    print("  运行批量实验:")
    print("    python3 runshells/run_batch_experiment.py")
    print()
    print("  启动评估界面:")
    print("    python customization/depression_scale_agent/app.py")
    print()


if __name__ == "__main__":
    main()
