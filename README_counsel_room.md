# 咨询室模式（G4 Counsel Room）使用指南

**适用代码**：当前仓库中的咨询室运行时与实验脚本

**用途**：在 6×7 心理咨询室场景下跑抑郁症 CBT 干预实验 —— 卡布达（患者）+ 蜻蜓队长（医生）双人物、kbd1-9 可替换患者变体、非会诊步 lite 加速，并支持安全恢复后继续复评。

> 本指南面向「跑咨询室实验」的操作者：读完应能用一条命令跑起任意 kbd×severity 组合。

---

## 1. 这是什么

咨询室模式是村庄抑郁症仿真在「密闭诊室」场景下的精简变体，用于 G4 干预组实验：

| 维度 | 村庄模式（默认） | 咨询室模式（`--counsel-room`） |
|------|------------------|-------------------------------|
| 地图 | `assets/village/maze.json`（50×50） | `assets/counsel_room/maze.json`（**6×7**，42 个 tile） |
| 人物 | 6 人花名册 | **仅 2 人**：卡布达（患者）+ 蜻蜓队长（医生） |
| 活动范围 | 全村漫游 | 锁定在 `蜻蜓队长的心理咨询室/聊天桌` |
| 配置 overlay | `experiments/config/groups/g{1..9}_*.json` | `data/config_counsel_room.json`（最小 overlay） |
| 患者变体 | kbd1-9 | kbd1-9（机制不同，见 §6） |
| 入口脚本 | `run_one_experiment.py` / `run_batch_experiment.py` | **仅** `run_batch_experiment.py --counsel-room` |

其余配置（CBT 会诊管线、`forced_llm`、sparse checkpoint、staged_eval、周期反思节奏）**全部继承自 `data/config.json`**，咨询室 overlay 只覆盖差异点。

---

## 2. 实现范围

当前仓库已经包含咨询室资源、最小 overlay、`--counsel-room`/`--assets-root` CLI、kbd1-9 变体适配、lite 加速和 post-scale 重试。本文只描述当前工作树中的可用行为，不依赖特定远程分支状态。

---

## 3. 配置与 LLM 路由

咨询室运行配置 = 主分支 `data/config.json`（base）+ `data/config_counsel_room.json`（overlay）深度合并。overlay 只动三处：把干预患者限定为「卡布达」、开启非会诊步 lite 加速、显式本地 vLLM 端点（与 base 已等价）。其余一律继承 base。

**两类 LLM，别混**：

- **`think_llm`** = 本地 vLLM（`qwen3-8b @ 18000`、`bge-m3 @ 18001`）。用于 agent 自身认知：percept/poignancy、schedule、周期反思、自发对话。
- **`forced_llm`** = 外部 DeepSeek API。用于会诊管线：CBT 会话对话、`session_eval`、`dialog_judge`、`environment_model`（医嘱作业）。继承自 base，overlay 不动。
- 路由原则：**大多数干预环节走外部 DeepSeek，只有 agent 自身认知走本地 vLLM**。

> `DEEPSEEK_API_KEY` 通过 `set -a && source .env && set +a` 注入环境（`.env` 不入库）。跑实验前务必确认。

---

## 4. 资源文件

```
frontend/static/assets/counsel_room/
├── maze.json                 # 6×7 地图
└── agents/
    ├── 卡布达/agent.json       # 患者：坐标锁定「蜻蜓队长的心理咨询室/聊天桌」
    └── 蜻蜓队长/agent.json     # 医生：同样锁定在诊室
```

两个 agent 的活动空间只含咨询室这一条路径 —— 这就是「密闭诊室」的几何约束：角色无法走出咨询室，所有交互发生在聊天桌。立绘复用村庄资源（`assets/village/agents/卡布达/portrait.png`）。

---

## 5. 运行命令示例

### 5.1 前置：起 vLLM + 注入 DeepSeek key

```bash
# 1) 确认本地 vLLM 已起（think/embedding）
curl -s http://127.0.0.1:18000/v1/models | head
curl -s http://127.0.0.1:18001/v1/models | head

# 2) 注入 DeepSeek key（forced_llm 用）
cd /home/zyli/generative_agents-2
set -a && source .env && set +a
echo "DEEPSEEK_API_KEY 长度: ${#DEEPSEEK_API_KEY}"
```

### 5.2 dry-run 验证（推荐每次先跑）

```bash
# 只打印将要执行的命令 + 生成的 runtime config，不真跑
python3 runshells/run_batch_experiment.py \
  --counsel-room --condition Counsel-KBD2-G4-MILD --dry-run
```

确认输出的 runtime config 用 `assets/counsel_room/maze.json`、`agents` 只有卡布达+蜻蜓队长、`lite_non_consult: true`。

### 5.3 单条件实跑

```bash
python3 runshells/run_batch_experiment.py \
  --counsel-room --condition Counsel-KBD2-G4-MILD \
  --max-parallel 1
```

### 5.4 条件选择器（`--condition` 可重复传入）

| 选择器 | 展开为 |
|--------|--------|
| `Counsel-G4-MILD` / `MOD` / `SEV` / `ALL` | **kbd1** × 对应严重度（3-token 旧格式，向后兼容） |
| `Counsel-KBD2-G4-MILD` | 指定变体 kbd2 × MILD |
| `Counsel-KBD2-G4-ALL` | kbd2 × 3 严重度 |
| `Counsel-ALL-G4-MILD` | 全 9 变体 × MILD |
| `Counsel-ALL-G4-ALL` | 全 9 变体 × 3 严重度 = **27 条** |
| `ALL` / `*` 或不传 `--condition` | 默认 kbd1 × 3 严重度 |

变体 token：`kbd1..kbd9` / `KBD1..KBD9` / `KABUDA1..KABUDA9`（`ORIGINAL`/`BASE` 是 kbd1 别名）。
严重度 token：`mild`/`moderate`/`severe` 或 `MILD`/`MOD`/`MODERATE`/`SEV`/`SEVERE`。

```bash
# 跑 kbd2 全严重度，2 路并行
python3 runshells/run_batch_experiment.py \
  --counsel-room --condition Counsel-KBD2-G4-ALL \
  --max-parallel 2
```

### 5.5 后台 nohup 脱机运行（断开 SSH 不死）

```bash
nohup python3 runshells/run_batch_experiment.py \
  --counsel-room --condition Counsel-KBD2-G4-ALL --max-parallel 2 \
  > results/experiment_data/counsel-kbd2-all.nohup 2>&1 &
disown
echo "PID=$!"
```

### 5.6 续跑（崩溃后从 checkpoint 继续）

```bash
# 续跑整个 batch 中所有可续的失败项
python3 runshells/run_batch_experiment.py \
  --counsel-room --name <原batch名> --resume-batch

# 只续跑某个条件
python3 runshells/run_batch_experiment.py \
  --counsel-room --name <原batch名> \
  --resume-condition Counsel-KBD2-G4-MILD
```

> 已有 batch 需要安全恢复并补复评时，使用 §5.7 的恢复流水线。

### 5.7 恢复/续跑 + 重复复评（`run_resume_batch_then_repeat_eval.sh`）

`run_resume_batch_then_repeat_eval.sh` 把「稀疏 checkpoint 安全恢复 + 补完仿真 + 重复量表复评」串成一道流水线。**默认村庄模式**，咨询室加 `--counsel-room`（自动 `GROUP=G4` + 透传）。它是 **resume-only**：针对已存在的 batch（`batch_state/<batch>/<cond>.json`）按状态判断恢复/续跑/跳过，再补复评；不会从零起跑。

> 从零起跑请先用 §5.3 的 `run_batch_experiment.py --counsel-room` 建好 batch，再用本脚本恢复/补复评。多变体可外部循环 KBD 多次调用（见 §5.7.3）。

#### 5.7.1 顶部参数（改脚本 `↓↓↓ 恢复实验参数 ↓↓↓` 块）

| 参数 | 默认 | 说明 |
|------|------|------|
| `EXP_DATE` | `0718` | batch 名日期段（`batch-${EXP_DATE}-NN`）、复评名后缀 |
| `COUNSEL_ROOM` | `false` | `true`=咨询室；**一般用 `--counsel-room` 开，不在顶部改** |
| `GROUP` | `G9` | 村庄组（`g1/g2/g3/g5/g6/g7/g9`）；`--counsel-room` 自动覆盖为 `G4` |
| `KBD` | `KBD6` | 单个患者变体（多变体见 §5.7.3 外部循环） |
| `SEVERITY` | `SEV` | `MILD`/`MOD`/`SEV` |
| `SIM_TARGET_STEP` | `120` | 目标总步数 |
| `SIM_STRIDE` | `720` | 每步推进分钟数（720=12h） |
| `SIM_MAX_PARALLEL` | `1` | 单 batch 内 condition 并行 |
| `EVAL_LABELS` | `T0,session_4,…,POST` | 复评时间点 |
| `EVAL_REPEAT` | `10` | 每点重复复评次数（取稳态 + 95% CI） |
| `EVAL_MAX_PARALLEL` | `3` | 复评 worker 并行度 |
| `REPEAT_COUNT` | `2` | 恢复 batch-01 至 batch-NN |
| `MAX_PARALLEL_REPEATS` | `""`(=`REPEAT_COUNT`) | 同时恢复几个 batch |

> 另有 `SIM_NAME`/`SIM_CONDITION`/`EVAL_CONDITION`/`EVAL_NAME`/`SIM_LOG`/`EVAL_LOG` 等派生变量，由上面输入自动求值，一般不改。
>
> `EXP_DATE`/`GROUP`/`KBD`/`SEVERITY` 这 4 个**支持环境变量覆盖**，不必改脚本：
> ```bash
> SEVERITY=MILD KBD=KBD4 bash runshells/run_resume_batch_then_repeat_eval.sh --counsel-room -n 2 -j 2
> ```
> （`--counsel-room` 会把 GROUP 重设为 G4，其余用环境传入值。）

#### 5.7.2 CLI 参数

| 参数 | 说明 |
|------|------|
| `-n`/`--repeat-count N` | 恢复 batch-01 至 batch-N |
| `-r`/`--repeat-index N` | 只恢复指定的单个 batch 编号 |
| `-j`/`--max-parallel N` | 同时恢复的 batch 数 |
| `--counsel-room` | **开咨询室**（`GROUP=G4` + 给 run_batch_experiment 透传 `--counsel-room`） |
| `--dry-run` | 只校验 + 显示命令，不起跑、不动文件 |
| `-h`/`--help` | 帮助 |

#### 5.7.3 用法示例

```bash
# 村庄：dry-run 看恢复锚点与命令
bash runshells/run_resume_batch_then_repeat_eval.sh --dry-run

# 村庄：恢复 batch-0718-01..02，1 路并行
bash runshells/run_resume_batch_then_repeat_eval.sh -n 2 -j 1

# 咨询室：恢复 + 补复评（需先用 run_batch_experiment 建好对应 batch）
bash runshells/run_resume_batch_then_repeat_eval.sh --counsel-room -n 2 -j 2

# 咨询室：只检查 batch-0718-02 的恢复计划
KBD=KBD2 bash runshells/run_resume_batch_then_repeat_eval.sh \
  --counsel-room --repeat-index 2 --dry-run

# 咨询室多变体：外部循环 KBD（每次一个变体 × REPEAT_COUNT）
for k in KBD2 KBD4 KBD6; do
  KBD=$k bash runshells/run_resume_batch_then_repeat_eval.sh --counsel-room -n 2 -j 2
done
```

> 单条 sim 跑完只有终点分（T0→session_20 轨迹空，见 §9.1），轨迹靠这里的复评阶段补。命名：`batch-${EXP_DATE}-${suffix}` / 复评 `repeat-${KBD}-${GROUP}-${SEVERITY}-${EXP_DATE}-${suffix}`。

---

## 6. 患者变体（kbd1-9）

运行前会为每个条件生成**归一化的运行时 persona**（运行时统一叫「卡布达」）：

| 变体 | agent 人设来源 | 空间模板 |
|------|----------------|----------|
| **kbd1** | 咨询室专用 `agents/卡布达/agent.json`（手工版） | 已内含（诊室锁定） |
| **kbd2-9** | 咨询室专用 `agents/卡布达N/agent.json`（2026-07-22 起） | 已内含（诊室锁定） |

要点：
- **kbd1-9 均为咨询室专用文件**：kbd2-9 的文件由「村庄 persona + 咨询室空间模板 + 无村庄地点的 daily_plan」生成后固化（见下方修订说明），运行时直读、不做叠加。
- 抑郁配置**始终来自村庄三件套**（`depression_config_mild/moderate/severe.json`），与变体解耦。
- 记忆注入按变体切换：kbd1→`memory_injections.json`，kbdN→`memory_injections_kabudaN.json`。
- **修订（2026-07-22）**：kbd2-9 原靠运行时叠加生成（村庄 persona + 空间模板），因村庄 `daily_plan` 泄漏（生成「今天下午在杂货店」类现在式村庄事件）改为固化文件；改某个变体的咨询室 persona 直接编辑对应文件，村庄源文件改动不再自动传导。新变体若无咨询室文件，运行时仍兜底叠加（daily_plan 用模板干净版）。

---

## 7. 加速与鲁棒性

### 7.1 lite 模式：非会诊步精简

lite 模式在不破坏结构一致性的前提下，让**非会诊步**跳过 percept 和自发对话、保留 schedule 和周期反思；**会诊步不受影响**（照常 full think + forced CBT 对话）。

为什么安全：抑郁状态机是事件驱动的（只在 chat/reflection 里变化），周期反思是会诊间抑郁演化的**唯一通道**，lite 保留了它，所以 PHQ-9 不会冻结。村庄模式默认关闭 lite，不受影响。

咨询室 overlay 默认开启 lite。实际耗时取决于本地 vLLM、外部 API 延迟、重试次数与并行度，应以当次日志为准。

### 7.2 post-scale 重试

DeepSeek 偶发瞬时抖动会让评分 worker 退出，从而整个 batch 崩。现已给**治疗后量表的作答/评分调用**包了重试（最多 3 次、线性退避），瞬时 API 抖动不再让 batch 崩。非 LLM 步骤（compress/merge 等）仍是一次性。

---

## 8. 默认实验参数（咨询室 batch）

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `STEP` | 120 | 仿真步数 |
| `STRIDE` | 720（分钟） | 每步推进 12 小时 → 120 步 = **60 sim-day** |
| `START_TIME` | `20260614-09:30` | 仿真起始 |
| `SCALE_AGENT` | 卡布达 | 量表评估对象 |
| `MAX_PARALLEL` | 1 | 并行条件数 |
| 会诊节奏 | interval_steps=6, step_phase=1 | 会诊发生在 step 1/7/13/…（≈20 次会诊） |
| checkpoint | `consult_sparse` | 仅在会诊/staged_eval/每 56 步存档 → **存档数 ≠ 步数** |
| `RUN_POST_SCALE` | True | 跑治疗后 PHQ-9 / BDI-II |
| `RUN_COMPRESS` | True | 生成回放资源 |

> ⚠️ **进度度量陷阱**：`results/checkpoints/<run>/simulate-*.json` 的文件数是 **sparse 存档数**（如 16），不是步数。真实步数看日志的 `step` 标记。

---

## 9. 输出与产物

```
results/
├── checkpoints/<run_name>/              # 仿真产物
│   ├── simulate-*.json                  # sparse 存档
│   ├── conversation.json                # 全量对话
│   ├── staged_eval/{T0,session_4,…}/    # 分阶段评估快照（capture_only）
│   └── run_batch_experiment.log
├── experiment_data/<run_name>/          # 后处理产物
│   ├── traces/、scales/                 # merge 结果、post 评分
│   └── trial_meta.json                  # variant/severity/overlay 等元信息
├── experiment_data/batch_state/<batch>/ # batch 调度状态（条件状态机、runtime 配置、persona）
└── experiment_data/reports/             # 汇总
    └── <batch>-<condition>_summary.{json,md}
```

`trial_meta.json` 记录该次运行的 variant、severity、overlay、persona 来源 —— 复现实验先看它。

### 9.1 评估解耦：capture_only 与延迟复评（重要）

主分支起 `staged_eval` 默认 `capture_only`：**仿真期间只拍记忆快照、不做答**。所以 sim 跑完的 `<batch>-<condition>_summary.md` 里 T0→session_20 中途轨迹是空的，只有 POST 终点分。这不是 bug，是「仿真/评估解耦」设计 —— 快照带校验，事后复评分对应的是 sim 当时冻结的那个状态，可验证、可换模型重跑。

补轨迹靠**重复复评**：对每个时间点的快照重新做答+评分，重复 N 次取均值 + 95% CI。两条路：

1. **单独补某条已完成的 run**（最常用）：
   ```bash
   pip install scipy  # 复评依赖
   python3 runshells/run_archived_repeat_scale_eval.py \
     --archive-results-root results \
     --condition Counsel-KBD2-G4-SEV \
     --original-summary results/experiment_data/reports/<batch>-<condition>_summary.json \
     --labels T0,session_4,session_8,session_12,session_16,session_20,POST \
     --repeat 10 --name repeat-kbd2-G4-SEV --max-parallel 3 --resume-partial
   ```
   产出 `reports/repeat-<name>_summary.{json,md}`（带轨迹 + CI）。建议先 `--dry-run` 校验快照。

2. **恢复已有 sim + 复评**：用 §5.7 的 `run_resume_batch_then_repeat_eval.sh --counsel-room`。该脚本是 resume-only，不会从零创建 batch。

> 前提：vLLM 在线（做答走 forced_llm=DeepSeek、embedding 走本地 bge-m3）、`scipy` 已装、`DEEPSEEK_API_KEY` 已注入。复评期间别再碰该 run 的 checkpoint。

---

## 10. 规范与约定

1. **咨询室入口唯一**：用 `run_batch_experiment.py --counsel-room`。`run_one_experiment.py` 没有 `--counsel-room`，跑不出咨询室。
2. **overlay 最小化**：新增咨询室差异优先加到 `data/config_counsel_room.json`，不要复制整个 `config.json`。能继承的（forced_llm、CBT、sparse、反思节奏）一律不动。
3. **变体双资产**：新患者变体需两处放置——村庄 `agents/卡布达N/`（persona + 抑郁三件套，村庄实验用）+ 咨询室 `counsel_room/agents/卡布达N/agent.json`（咨询室实验直读，daily_plan 等不得含村庄地点）。缺咨询室文件时运行时兜底叠加（daily_plan 用模板干净版），但正式实验应先建文件。
4. **跑前必做**：(a) 确认 vLLM 18000/18001 在线；(b) `set -a && source .env && set +a` 注入 DeepSeek key；(c) `--dry-run` 过一遍。
5. **长跑用 nohup**：后台 + `disown`，PPID 归 1，断开 SSH 不死。
6. **崩溃先看状态机**：`batch_state/<batch>/<cond>.json` 的 `status` 字段（`failed_after_checkpoint` = 可续跑；`failed` = 无 checkpoint，从头）。

---

## 11. 已知问题

- **行为实验降门槛倾向待复核**：同事提交的说明提到 forced_llm 在行为实验 session 可能持续下调阈值，但本次提交未附对应 findings 或实验记录。修改 prompt 前应先结合实际会谈 trace 复核。
- **（可选）简化 overlay**：去掉冗余的 `agent.think.llm` / `agent.associate.embedding`，只留 `patients` + `lite_non_consult`。

---

## 12. 资源与文件位置

| 文件 | 作用 |
|------|------|
| [data/config_counsel_room.json](data/config_counsel_room.json) | 咨询室 overlay（patients + lite_non_consult） |
| [frontend/static/assets/counsel_room/maze.json](frontend/static/assets/counsel_room/maze.json) | 6×7 诊室地图 |
| [frontend/static/assets/counsel_room/agents/卡布达/agent.json](frontend/static/assets/counsel_room/agents/卡布达/agent.json) | 患者诊室配置 kbd1（兼作新变体兜底的空间/daily_plan 模板） |
| [frontend/static/assets/counsel_room/agents/卡布达2-9/agent.json](frontend/static/assets/counsel_room/agents/) | 患者诊室配置 kbd2-9（2026-07-22 起直读） |
| [frontend/static/assets/counsel_room/agents/蜻蜓队长/agent.json](frontend/static/assets/counsel_room/agents/蜻蜓队长/agent.json) | 医生诊室配置 |
| [runshells/run_batch_experiment.py](runshells/run_batch_experiment.py) | 批量入口（`--counsel-room`、条件选择器、base+overlay 合并） |
| [runshells/run_one_experiment.py](runshells/run_one_experiment.py) | 单次入口（无 `--counsel-room`） |
| [runshells/run_resume_batch_then_repeat_eval.sh](runshells/run_resume_batch_then_repeat_eval.sh) | 恢复已有 sim 并复评（默认村庄，`--counsel-room` 开咨询室，§5.7） |
| [runshells/run_archived_repeat_scale_eval.py](runshells/run_archived_repeat_scale_eval.py) | 重复量表复评引擎（补 capture_only 轨迹，§9.1） |
