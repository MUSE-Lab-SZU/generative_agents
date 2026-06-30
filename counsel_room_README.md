# G4 心理咨询室实验

> **分支**: `counsel_room` — 在隔离的咨询室环境中测试抑郁症干预效果，与完整小镇环境形成对照实验。

## 目录

- [实验目标](#实验目标)
- [实验组设计](#实验组设计)
- [地图区分机制](#地图区分机制)
- [快速开始](#快速开始)
- [`--counsel-room` 详解](#--counsel-room-详解)
- [运行指令](#运行指令)
- [配置体系](#配置体系)
- [输出结构](#输出结构)

## 实验目标

本分支在 `doctor-agent` 分支基础上，增加一个 **G4 心理咨询室组**，用于对比：

| 组 | 环境 | Agent 数 | 地图 | 目的 |
|----|------|---------|------|------|
| **G1** | 完整小镇 | 6 人 | 50×50 | 有医生干预基线 |
| **G2** | 完整小镇 | 6 人 | 50×50 | 无干预对照 |
| **G3** | 完整小镇 | 6 人 | 50×50 | 随机居民聊天对照 |
| **G4** | 心理咨询室 | **2 人** | **7×6** | **隔离环境干预** |
| **G5** | 完整小镇 | 6 人 | 50×50 | 负向居民聊天对照 |

G4 的**核心作用**是做 ablation study：

- 去掉小镇中 4 个无关居民（金龟次郎、田德莉娜、呱呱蛙、蟑螂恶霸）和 6 个多余区域
- 创建纯粹的 1 对 1 咨询环境（患者卡布达 + 医生蜻蜓队长）
- 与 G1 对比：如果 G4 疗效 ≥ G1 → 环境干扰确实拖累治疗效果
- 与 G2/G3/G5 对比：在无干扰环境下 baseline 的差异

## 地图区分机制

### 两张地图，两套配置

| | 村庄实验 (G1/G2/G3/G5) | 咨询室实验 (G4) |
|---|---|---|
| **配置入口** | `data/config.json` (v1 格式: `agent` 字段) | `data/config_counsel_room.json`（v2 格式: `agent_base` 字段） |
| **group overlay** | 叠加合并 `experiments/config/groups/g{1,2,3,5}_*.json` | **无 overlay**（干预配置已内嵌在模板自身） |
| **地图来源** | `assets/village/maze.json` | `assets/counsel_room/maze.json` |
| **Agent 来源** | `assets/village/agents/{name}/agent.json` | `assets/counsel_room/agents/{name}/agent.json` |
| **Depression 来源** | `assets/village/agents/{name}/depression_config_{sev}.json` | `assets/village/agents/{name}/depression_config_{sev}.json`（复用村庄文件） |

### 地图参数对比

```
村庄地图 (50×50 = 2,500 tiles)         咨询室地图 (7×6 = 42 tiles)
┌──────────────────────────────────┐    ┌──────────────────┐
│  公园 (384 tiles)                 │    │ 聊天桌聊天桌聊天桌 │
│  卡布达与金龟次郎的家 (110)         │    │ 聊天桌卡布达聊天桌 │
│  呱呱哇的家 (100)                  │    │ 聊天桌蜻蜓队长聊 │
│  田德莉娜的家 (100)                 │    │ 聊天桌聊天桌聊天桌 │
│  蟑螂恶霸的家 (100)                 │    └──────────────────┘
│  蜻蜓队长的心理咨询室 (66) ←仅2.6%  │    仅 1 个 arena:
│  杂货店 (42)                       │    蜻蜓队长的心理咨询室 > 聊天桌
│  7 sector / 32 arena              │    2 agents 固定坐定
│  6 agents 可自由移动               │    纯 1 对 1 咨询
└──────────────────────────────────┘
```

### 运行时路径区分（代码层面）

`run_batch_experiment.py` 的 `build_condition_runtime_config_payload` 函数根据 `counsel_room` 标志走不同分支：

```python
# 村庄模式（默认）
template = load_json_file(runtime_config_template_path())   # data/config.json
overlay = load_json_file(GROUP_OVERLAY_FILES[condition.group])
template = deep_merge_dict(template, overlay)
assets_root = "assets/village"
agents_list = list(PERSONAS)          # 6 个角色
agent_base_source_key = "agent"       # v1 格式

# 咨询室模式（--counsel-room）
template = load_json_file(COUNSEL_ROOM_TEMPLATE)            # data/config_counsel_room.json
assets_root = "assets/counsel_room"
agents_list = ["卡布达", "蜻蜓队长"]   # 仅 2 个角色
agent_base_source_key = "agent_base"  # v2 格式
depression_assets_subdir = "village"  # depression config 复用村庄
```

### 路径解析

所有路径通过 `Game.load_static()` 解析，拼接 `static_root`（`frontend/static/`）+ 配置中的相对路径：

```
static_root = "frontend/static"
config["maze"]["path"] = "assets/counsel_room/maze.json"
→ 实际文件: frontend/static/assets/counsel_room/maze.json ✅

config["agents"]["卡布达"]["config_path"] = "assets/counsel_room/agents/卡布达/agent.json"
→ 实际文件: frontend/static/assets/counsel_room/agents/卡布达/agent.json ✅
```

### `.gitignore` 隔离

咨询室专用文件通过 `.gitignore` 与 doctor-agent 分支隔离，不会提交到上游：

```
# counsel_room 分支专用配置和地图
frontend/static/assets/counsel_room/
data/config_counsel_room.json
```

## 快速开始

### 前置条件

- Python 3.10+
- 启动 Qwen3-8B LLM 服务（用于 agent 思考）：`http://127.0.0.1:18000/v1`
- 启动 BGE-M3 embedding 服务：`http://127.0.0.1:18001/v1`（和 `18002/v1`）
- 卡布达外置记忆服务端口 `8031` 可用
- DeepSeek API Key（用于干预系统的 forced LLM）

### 环境变量

```bash
export DEEPSEEK_API_KEY="your-key-here"
```

## `--counsel-room` 详解

`--counsel-room` 是一个**复合开关**，在 `start.py` 和 `run_batch_experiment.py` 两个层面共同生效，共涉及 **7 项变更**：

### start.py 层面（1 项）

| 代码 | 作用 |
|------|------|
| `runtime_config = args.runtime_config or ("data/config_counsel_room.json" if args.counsel_room else "")` | 将运行时配置指向咨询室专用文件 |

`--counsel-room` 等价于 `--runtime-config data/config_counsel_room.json`。

### run_batch_experiment.py 层面（6 项）

| # | 代码位置 | 村庄模式（默认） | `--counsel-room` 后 |
|---|---------|----------------|-------------------|
| **①** | `ALL_CONDITIONS[:]` | 30+ 条件（6 variant × 5 group × 3 severity） | **仅 3 个**（kbd1 × G4 × MILD/MOD/SEV） |
| **②** | `GROUP_SELECTOR_ALIASES` | `{"G1":"g1","G2":"g2","G3":"g3","G5":"g5"}` | **注入 `"G4":"g4"`** |
| **③** | 配置模板来源 | `data/config.json` + 合并 group overlay | **直接加载** `data/config_counsel_room.json`（无 overlay） |
| **④** | `assets_root` | `"assets/village"` | **`"assets/counsel_room"`** |
| **⑤** | `agents_list` | 6 个角色 | **仅 2 个**（卡布达 + 蜻蜓队长） |
| **⑥** | `variant_kwargs` | `{}` | `assets_subdir="counsel_room"`（spatial tree 用咨询室版）<br>`depression_assets_subdir="village"`（depression config 复用村庄） |

### config_counsel_room.json 的静态差异

与 `data/config.json` + G4 overlay 合并结果相比，咨询室配置文件本身还包含以下差异：

| 差异项 | 村庄 (config.json + overlay) | 咨询室 (config_counsel_room.json) |
|-------|---------------------------|----------------------------------|
| **配置格式** | v1（`agent` 字段） | **v2**（`agent_base` 字段，直接作为顶层） |
| **地图路径** | `assets/village/maze.json`（50×50） | **`assets/counsel_room/maze.json`**（7×6） |
| **患者列表** | `["卡布达", "金龟次郎"]` | **`["卡布达"]`** |
| **会诊规则** | 含 `fast_every_2step_jgcl`（金龟次郎专属） | **已移除**金龟次郎规则 |
| **chat_controls** | 含金龟次郎的 `agent_overrides` | **仅保留**卡布达 + 蜻蜓队长 |
| **memory_policy** | patients 含金龟次郎 | **仅**卡布达 |
| **consult_record** | 按 overlay 配置 | **`enabled: false`** |
| **resident_chat** | 可能有排期 rule | **无**（`enabled: false, rules: []`） |

### 一句话总结

```
--counsel-room =
  (1) 条件池缩到 G4×3 个            ← 脚本层面
  (2) 注册 G4 选择器别名             ← 脚本层面
  (3) 切到自包含的 v2 配置文件       ← 配置层面
  (4) 地图从 50×50 → 7×6          ← 配置层面
  (5) Agent 从 6 人 → 2 人         ← 配置层面
  (6) 移除金龟次郎所有引用           ← 配置层面
  (7) 患者 spatial tree 用咨询室版   ← 运行时层面
```

## 运行指令

### 完整批次（3 个严重程度 × 1 个 variant）

```bash
# 跑全部 3 个条件（mild → moderate → severe，串行执行）
python runshells/run_batch_experiment.py --counsel-room
```

### 单个条件

```bash
# 方式 1：精确条件名匹配
python runshells/run_batch_experiment.py --counsel-room --condition Counsel-KBD1-G4-MILD

# 方式 2：3 段选择器（variant 默认 kbd1，推荐写法）
python runshells/run_batch_experiment.py --counsel-room --condition Counsel-G4-MILD
python runshells/run_batch_experiment.py --counsel-room --condition Counsel-G4-MOD
python runshells/run_batch_experiment.py --counsel-room --condition Counsel-G4-SEV

# 方式 3：按严重度批量（3 段选择器 + ALL 通配）
python runshells/run_batch_experiment.py --counsel-room --condition Counsel-G4-ALL
# 等价于不带 --condition，跑全部 3 个
```

### 多条件并行

```bash
# 需先调大 MAX_PARALLEL（可在 run_batch_experiment.py 顶部修改）
python runshells/run_batch_experiment.py --counsel-room \
  --condition Counsel-G4-MILD --condition Counsel-G4-SEV --max-parallel 2
```

### 干跑验证（不执行仿真）

```bash
python runshells/run_batch_experiment.py --counsel-room --dry-run
```

### 与村庄实验的指令对比

| 用途 | 村庄 (G1/G2/G3/G5) | 咨询室 (G4) |
|------|-------------------|------------|
| 全部条件 | `python runshells/run_batch_experiment.py` | `python runshells/run_batch_experiment.py --counsel-room` |
| 单条件 | `--condition Counsel-G1-MILD` | `--counsel-room --condition Counsel-G4-MILD` |
| 按严重度批量 | `--condition G1-ALL` | `--counsel-room --condition G4-ALL` |
| 干跑 | `--dry-run` | `--counsel-room --dry-run` |
| 续跑失败 | `--resume-batch <name>` | `--counsel-room --resume-batch <name>` |
| 跳过后处理 | `--skip-merge --skip-post-scale` | `--counsel-room --skip-merge --skip-post-scale` |

> ⚠️ **`--counsel-room` 必须始终放在前面**。它会将 `ALL_CONDITIONS` 替换为仅 3 个 G4 条件，并在 `GROUP_SELECTOR_ALIASES` 注入 `G4` 映射。如果不加 `--counsel-room`，`--condition G4-MILD` 会找不到 G4 映射而报错。

### 村庄实验与咨询室实验的指令路线区别

```bash
# ── 村庄实验（G1/G2/G3/G5）：从 data/config.json + group overlay 合成 ──
# ALL_CONDITIONS 包含所有 30+ 条件（6 variant × 5 group × 3 severity）
# 直接用 --condition 选择 group
python runshells/run_batch_experiment.py                              # 全部条件
python runshells/run_batch_experiment.py --condition G1-ALL           # 全部 G1
python runshells/run_batch_experiment.py --condition Counsel-G1-MILD  # G1-MILD

# ── 咨询室实验（G4）：直接加载 data/config_counsel_room.json ──
# ALL_CONDITIONS 被替换为仅 3 个（kbd1 × G4 × 3 severity）
# 必须先加 --counsel-room 才能识别 G4
python runshells/run_batch_experiment.py --counsel-room                              # 全部 3 个
python runshells/run_batch_experiment.py --counsel-room --condition G4-ALL           # 同上
python runshells/run_batch_experiment.py --counsel-room --condition Counsel-G4-MILD  # 仅 MILD
```

### 启动参数速查

| 参数 | 默认值 | 说明 |
|------|-------|------|
| `--counsel-room` | — | 启用咨询室模式 |
| `--condition` | 全部条件 | 筛选要跑的条件（可重复） |
| `--step` | 120 | 仿真步数 |
| `--stride` | 720 | 每步推进分钟数（= 12h/步） |
| `--start` | `20260614-09:30` | 仿真起始时间 |
| `--max-parallel` | 1 (咨询室模式) | 并行条件数 |
| `--dry-run` | — | 仅打印不执行 |
| `--resume-batch` | — | 续跑指定批次 |
| `--skip-merge` | — | 跳过对话合并 |
| `--skip-post-scale` | — | 跳过后测评估 |
| `--skip-compress` | — | 跳过回放压缩 |
| `--name` | 自动生成 | 自定义批次名 |

## 配置体系

咨询室模式由 **3 层配置叠加**构成（自上而下优先级递减）：

```
① 批量脚本参数（run_batch_experiment.py 顶部常量 + CLI flags）
   步数、时间、并行度等运行参数
       ↓
② 咨询室基础配置（data/config_counsel_room.json）
   agent_base、地图、agent 列表、intervention 完整定义
       ↓
③ Variant 运行时（kabuda_variant_runtime.py 生成）
   agent.json（spatial tree）+ depression_config_{severity}.json
```

### 会诊排期

当前启用的唯一会诊规则：

| 规则 | 类型 | 间隔 | 时长 | 启用 |
|------|------|------|------|------|
| `fast_every_4step` | interval_steps | 每 6 步 (phase 1) | 2881 min | ✅ |
| `weekly_followup` | weekly | 每周一 14:00 | 30 min | ❌ |
| `every_7_days_followup` | interval_days | 每 7 天 | 30 min | ❌ |
| `fast_every_6h` | interval_minutes | 每 360 min | 30 min | ❌ |
| `prod_weekly_monday_1400` | weekly | 每周一 14:00 | 30 min | ❌ |

### 其余已启用干预模块

chat_controls、order_extract、session_prompt_injection、forced_llm、dialog_judge、session_eval、consult_history、depression_dynamic、memory_injection、memory_policy

## 输出结构

```
results/
├── checkpoints/
│   └── {run_name}/                          ← 仿真 checkpoint
│       ├── simulate-{step}.json              ← 快照
│       ├── staged_eval/                      ← 分阶段评估
│       └── storage/                          ← agent 存储
├── experiment_data/
│   ├── batch_state/{batch_name}/             ← 批次状态
│   │   ├── runtime_configs/                  ← 生成的 runtime config
│   │   ├── runtime_personas/                 ← variant 运行时文件
│   │   ├── timings/                          ← 时序记录
│   │   └── {condition_name}.json             ← 各条件状态
│   ├── {run_name}/                           ← 后处理输出
│   │   ├── trial_meta.json                   ← 实验元信息
│   │   ├── traces/                           ← 对话轨迹
│   │   ├── scales/                           ← PHQ-9 / BDI-II 评分
│   │   └── visualizations/                   ← 记忆可视化
│   └── reports/                              ← 汇总报告
```

## 文件清单

| 文件 | 用途 |
|------|------|
| `data/config_counsel_room.json` | 咨询室完整配置（v2 格式） |
| `frontend/static/assets/counsel_room/maze.json` | 7×6 心理咨询室地图 |
| `frontend/static/assets/counsel_room/agents/卡布达/agent.json` | 卡布达咨询室配置 |
| `frontend/static/assets/counsel_room/agents/蜻蜓队长/agent.json` | 蜻蜓队长咨询室配置 |
| `runshells/kabuda_variant_runtime.py` | Variant 运行时生成器（支持 `assets_subdir`） |
| `runshells/run_batch_experiment.py` | 批量实验入口（支持 `--counsel-room`） |
