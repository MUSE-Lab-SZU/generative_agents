# Nature Mental Health 抑郁症相关论文分析报告


## 一、期刊概览与投稿格式要求

### 1.1 期刊基本信息

| 属性 | 详情 |
|------|------|
| 全称 | Nature Mental Health |
| 出版商 | Springer Nature (Nature Portfolio) |
| 创刊 | 2023年1月 (月刊, 纯在线) |
| ISSN | 2731-6076 |
| DOI前缀 | s44220 |
| 文章类型 | Article, Analysis, Review, Perspective, Comment, News & Views |
| scope | 全光谱心理健康研究，强调跨学科与转化价值 |

### 1.2 Article 格式要求

| 项目 | 要求 |
|------|------|
| 正文 | ~3,000 words (不含方法、参考文献、图注) |
| 摘要 | ≤150 words, 无小标题, 无引用 |
| 展示项 | 图 + 表共 ≤6 项 |
| 参考文献 | ~30–50 篇 (建议精炼, 每条引用需有明确功能) |
| 方法部分 | 可放正文末尾或作为单独 Supplementary Information |
| 数据可用性 | 必须声明 Data Availability |
| 伦理声明 | 涉及人类/动物的研究必须提供伦理审批信息 |
| AI使用声明 | 如使用AI工具辅助写作须在 Acknowledgements 中声明 |

### 1.3 与本项目相关的文章类型选择

**推荐: Article (研究论文)**

理由：
- 本项目有完整的实验结果（仿真数据 + 评估指标）
- 有明确的 methodological contribution（生成式智能体认知架构 + 抑郁症建模）
- 有可展示的 pipeline/system design

**备选: Analysis / Perspective**（如果结果尚不充分，可先投方法学视角的分析论文）

---

## 二、NMH 抑郁症相关论文的写作风格分析

### 2.1 整体叙事结构：Nature 系列的 Funnel Pattern

NMH 论文遵循 Nature 家族经典的漏斗式结构：

```
领域级背景 → 现有瓶颈 → 前人尝试 → 未解决的核心空白 → 本研究
```

**具体展开**（以 Zavlis et al. 2025, s44220-025-00465-9 为范本）：

1. **第一段**：确立领域的宏观重要性（如全球抑郁症负担、发病率趋势、社会影响）
2. **第二段**：聚焦到具体瓶颈（现有治疗方案的局限性、评估手段的主观性、机制理解的不足）
3. **第三段**：综述前人工作，**综合**而非罗列（"Several methodologies have been employed..."）
4. **第四段**：明确指出 gap（"However, these methods often rely on assumptions that may not hold..."）
5. **第五段**：引出本研究（"Here, we..." / "In this study, we..."）

### 2.2 语言特征与动词校准 (Verb Calibration)

从已分析的 NMH 论文中总结出的动词使用模式：

| 语境 | 推荐动词 | 避免使用 |
|------|----------|----------|
| 报告结果 | detected, increased, showed, enabled, achieved | proved, confirmed (过于绝对) |
| 讨论机制 | suggests, is consistent with, may reflect, indicates | demonstrates causally (除非有因果证据) |
| 方法描述 | implemented, employed, utilized, applied | leveraged, utilized vaguely |
| 比较声明 | outperformed, exceeded, was higher than | was better (无量化) |

**关键原则**：
- **Evidence-first**：先给证据，再做声明。避免"空洞的重要性声明"（如 "depression is a major public health challenge" — 如果后面没有紧跟具体数字）
- **Calibrated hedging**：根据证据强度调节确定性程度。有 RCT 支持用 "demonstrates"，有观察性数据用 "suggests"，有相关性用 "is associated with"
- **不夸大**：NMH 论文审稿人对 overclaiming 极其敏感。避免 "first", "novel", "groundbreaking" 等词除非有充分依据

### 2.3 摘要结构（Nature Mental Health 版）

NMH 摘要遵循标准 6 要素模式（150 words 内）：

```
Context (领域背景) → Gap (现有不足) → Approach (本文方法) → Key Result (核心结果) → Implication (意义) → Boundary (边界/局限)
```

**范例分析**（Zavlis et al. 2025 摘要结构）：
- *Context*: "Computational modeling approaches are increasingly applied to mental health..."
- *Gap*: "However, their clinical utility remains unclear..."
- *Approach*: "We conducted a systematic review..."
- *Key Result*: "We identified [N] studies across [domains]..."
- *Implication*: "Computational models show promise for..."
- *Boundary*: "...but further validation is needed"

### 2.4 图表设计特征

从 NMH 抑郁症论文中观察到的常见图表类型：

| 图表类型 | 功能 | 出现频率 |
|----------|------|----------|
| Pipeline/System overview figure | 展示方法学整体架构 | 极高 (几乎所有计算类论文) |
| Network/Connectivity diagram | 展示症状网络、脑区连接 | 高 |
| Trajectory plots | 展示纵向变化、治疗响应曲线 | 高 |
| Heatmaps/Confusion matrices | 分类/预测性能可视化 | 中 |
| Forest plots | Meta分析效应量汇总 | 中 |
| Box/Violin plots | 组间比较分布 | 高 |

---

## 三、方法学部分：需要突出的关键点

### 3.1 Nature Mental Health 的方法学期望

NMH 审稿人期望方法部分满足：

1. **可复现性** (Reproducibility)：读者能根据描述复现核心实验
2. **理论动机** (Theoretical motivation)：每个模块有明确的理论/实证基础
3. **边界声明** (Boundary conditions)：清楚说明方法的适用范围和局限

### 3.2 生成式智能体论文的方法学关键要素

基于 Kambeitz & Meyer-Lindenberg (2025, npj Digital Medicine) 和本项目特点，方法部分应突出：

#### (A) 智能体认知架构 (Agent Cognitive Architecture)

**必须描述的核心组件**：

| 组件 | 需回答的问题 | 参考范例 |
|------|-------------|----------|
| 记忆系统 | 长期记忆如何存储？检索机制基于什么（recency, importance, relevance）？ | Park et al. 2023 的 Memory Stream |
| 反思机制 | 智能体如何从经验中提取洞察？反思触发的条件是什么？ | Park et al. 2023 的 Reflection |
| 人格建模 | 人格特质如何影响行为生成？基于什么理论框架（如 Big Five）？ | Kambeitz et al. 2025 |
| LLM 后端 | 使用哪个模型？prompt 设计如何确保行为一致性？ | 明确说明模型版本和参数 |

**写作模板**（三要素模式：Motivation → Mechanism → Evidence/Role）：

> **[模块名]** — *Motivation*: 为什么需要这个模块？不加这个模块会怎样？  
> *Mechanism*: 具体实现到同行可复现的程度  
> *Evidence/Role*: 这个模块对整体结果的贡献是什么（ablation hook）

#### (B) 抑郁症建模策略

**需重点论述**：

1. **症状建模基础**：基于什么临床量表/诊断标准？
   - PHQ-9 (Kroenke et al. 2001) — 抑郁严重度
   - GAD-7 (Spitzer et al. 2006) — 焦虑共病
   - BDI-II, HDRS 等
   - **关键**：解释为什么选择某个量表，它如何映射到智能体的内部状态

2. **治疗干预建模**：
   - 心理治疗（CBT, 支持性治疗）如何被编码为智能体交互事件？
   - 药物治疗如何影响智能体的情绪参数？
   - 治疗响应的个体差异如何建模？

3. **环境与社会因素**：
   - 智能体所处的社会环境（家庭、工作、社区）如何影响抑郁轨迹？
   - 负性生活事件（adverse life events）如何触发症状变化？
   - 保护性因素（社会支持、运动）如何建模？

#### (C) 验证与效度论证

**NMH 审稿人最关注的问题**：仿真结果是否可信？

| 验证层次 | 具体做法 | 论文中的表述 |
|----------|----------|-------------|
| 内部效度 | 复现已知的心理学发现（如高神经质 → 更高抑郁风险） | "replicating established psychological findings" |
| 外部效度 | 与真实世界流行病学数据对比 | "compared with empirical findings from longitudinal studies" |
| 构念效度 | 标准化量表评分与临床预期一致 | "agents completed standardized mental health assessments" |
| 生态效度 | 仿真场景与真实临床场景的对应关系 | "ecological validity of the simulated environment" |

### 3.3 方法部分禁止使用的模糊表述

| 禁止 | 替代 |
|------|------|
| "under standard conditions" | 写明具体参数和配置 |
| "using routine methods" | 引用具体方法名称和出处 |
| "data were analyzed statistically" | 写明统计检验名称、假设、显著性水平 |
| "the method was validated" | 描述验证方案、数据集、指标 |
| "samples were randomly assigned" | 说明随机化方法（如 seed, stratification） |

---

## 四、实验设计与评估指标（详细版）

### 4.1 仿真参数设定

基于项目当前配置（`run_batch_experiment.py` + `data/config.json`）：

| 参数 | 当前值 | 说明 | 论文建议 |
|------|--------|------|----------|
| 仿真步数 (STEP) | 60 | 每次仿真运行 60 步 | 保持 48–60 步（覆盖完整 CBT 4 阶段） |
| 步长 (STRIDE) | 360 分钟 (6 小时) | 每步代表 6 小时虚拟时间 | 保持 360，总计约 15 天虚拟时间 |
| 起始时间 | 20260607-09:30 | 虚拟社区起始时刻 | 在论文中说明为工作日上午 |
| 治疗间隔 | 每 4 步一次 | `fast_every_4step` 规则 | 即每 24 小时虚拟时间一次治疗 |
| 对话轮次上限 | 18 轮/次 | `forced_chat_iter` | 论文中需说明为什么选择 18 轮 |
| 阶段性评估间隔 | 每 4 次完成对话 | `session_interval: 4` | 在治疗第 4、8 次后各评估一次 |
| T4 随访窗口 | 治疗结束后 8 步 | `t4_after_steps: 8` | 即治疗后 48 小时虚拟时间 |
| 重复次数 | 待定 | 目前 batch 脚本仅 1 次 | **建议每组 ≥3 次重复**（见 4.5） |

### 4.2 实验组设计（五组对照）

> 以下基于 `experiments/config/groups/` 中的实际配置文件

#### 组别总览

| 组别 | 配置文件 | 核心干预 | 抑郁引擎 | 对话法官 | 治疗提示 | 评估 | 临床对应 |
|------|----------|----------|----------|----------|----------|------|----------|
| **G1** | `g1_doctor_intervention.json` | CBT 结构化治疗（蜻蜓队长→卡布达） | ✅ 全部启用 | ✅ | ✅ 10 session CBT | ✅ 全套 | 实验组（标准 CBT） |
| **G2** | `g2_no_intervention.json` | 无任何干预 | ✅ 自然病程 | ❌ | ❌ | ✅ 全套 | 空白对照组（自然病程） |
| **G3** | `g3_random_resident_chat.json` | 与田德莉娜/呱呱蛙/蟑螂恶霸的随机中性社交 | ✅ 自然病程 | ❌ | ❌ 中性社交提示 | ✅ 全套 | 安慰剂对照（非治疗性社交） |
| **G4** | `g4_counseling_room.json` | 同 G1，限定在咨询室环境 | ✅ 全部启用 | ✅ | ✅ 10 session CBT | ✅ 全套 | 环境控制（是否咨询室有额外效应） |
| **G5** | `g5_negative_resident_chat.json` | 与同样居民的负面社交 | ✅ 自然病程 | ❌ | ❌ 负面支持提示 | ✅ 全套 | 伤害性对照（有害社交的恶化效应） |

#### 各组关键配置差异

**G1 — 医生干预组（主实验组）**
- 医生: `蜻蜓队长`; 患者: `卡布达`, `金龟次郎`
- 会面规则: 每 4 步强制治疗对话（`fast_every_4step`, phase 2 触发）
- CBT 会话序列: session1 → session_loop → 2.1 → 2.2 → 2.3 → 3.1 → 3.2 → 3.3-A → 4.1 → 4.2（共 10 个有序 session）
- 记忆注入: 初始压力注入 + session 完成后关键洞察注入（session 1, 2.1, 4.4 后触发）
- 咨询历史: 启用，top-k=3 检索（保持治疗连续性）
- 记忆策略: 角色加权 0.4（医生对话权重更高）

**G2 — 无干预对照组**
- 所有干预子系统显式关闭
- 抑郁引擎仍活跃 → 模拟抑郁症自然病程
- 记忆注入仅保留初始压力事件（不注入治疗相关记忆）
- **目的**: 建立"不治疗会怎样"的基线

**G3 — 随机中性社交组**
- 无医生角色，改为与 3 位居民的随机社交
- 候选居民: 田德莉娜（朋友）、呱呱蛙（邻居）、蟑螂恶霸（欺凌者）
- 策略: `random_no_repeat`（每次随机选不同居民）
- 对话提示: `resident_chat_neutral_social.txt`（中性社交，非治疗性）
- **目的**: 控制单纯"有人跟你说话"的安慰剂效应

**G4 — 咨询室环境组**
- 与 G1 完全相同的治疗配置
- 差异: 环境限定在"蜻蜓队长的心理咨询室"内
- **目的**: 排除物理环境对治疗效果的额外贡献

**G5 — 负面社交组**
- 结构与 G3 相同，但对话提示改为 `resident_chat_negative_support.txt`
- 同样与 3 位居民对话，但引导为负面/有害支持
- **目的**: 验证有害社交会加剧抑郁症状（伤害性对照）

### 4.3 抑郁严重度维度

项目提供三套抑郁配置（`frontend/static/assets/village/agents/卡布达/`）：

| 配置文件 | 严重度 | 对应实验条件 | 论文中的角色 |
|----------|--------|-------------|-------------|
| `depression_config_mild.json` | 轻度 | `Counsel-G4-MILD` | 亚组分析 |
| `depression_config_moderate.json` | 中度 | `Counsel-G4-MOD` | 主要分析 |
| `depression_config_severe.json` | 重度 | `Counsel-G4-SEV` | 亚组分析 |

**建议分析策略**：
- **主分析**: 以中度（moderate）为默认严重度，这是 CBT 临床试验最常见的入组标准
- **调节效应分析**: 严重度作为调节变量 (moderator)，检验"基线越重→治疗效果越大/小"的假设
- **论文表述**: "We examined whether treatment response varied as a function of baseline depression severity (mild vs. moderate vs. severe)."

### 4.4 具体评估指标与数据采集方案

#### (A) 主要终点指标 (Primary Endpoints)

| 指标 | 采集方式 | 采集时间点 | 数据来源 | 临床意义 |
|------|----------|-----------|----------|----------|
| **PHQ-9 总分** | 评估 LLM 逐题问答 → ExpertLLM 评分 | T0, S4, S8, T4 | `staged_eval_worker.py` 输出 | 主要疗效指标 |
| **PHQ-9 变化量 (ΔPHQ-9)** | 计算得: T4 − T0 | T4 vs T0 | 由 PHQ-9 总分计算 | 治疗效应量 |
| **治疗响应率** | ΔPHQ-9 ≤ −50% 的比例 | T4 | 由 ΔPHQ-9 判定 | 临床响应标准 |
| **缓解率** | T4 时 PHQ-9 < 5 的比例 | T4 | 由 T4 PHQ-9 判定 | 临床治愈标准 |

#### (B) 次要终点指标 (Secondary Endpoints)

| 指标 | 采集方式 | 采集时间点 | 数据来源 | 论文用途 |
|------|----------|-----------|----------|----------|
| **BDI-II 总分** | 同 PHQ-9 流程 | T0, S4, S8, T4 | staged_eval | 交叉验证（与 PHQ-9 一致性） |
| **SDS 总分** | 同 PHQ-9 流程 | T0, S4, S8, T4 | staged_eval | 交叉验证（自评量表补充） |
| **抑郁严重度分级** | ExpertLLM 判定: 正常/轻度/中度/重度 | 每次评估 | staged_eval | 分类转归分析 |
| **PHQ-9 条目级变化** | 单题分析 | T0 vs T4 | staged_eval | 症状维度分析（哪个症状改善最大） |

#### (C) 过程指标 (Process Measures)

| 指标 | 采集方式 | 数据来源 | 论文用途 |
|------|----------|----------|----------|
| **投诉链状态转移** | 记录每次状态变更: from_stage → to_stage | `depression engine` 日志 | 展示抑郁认知的动态演变（→ Fig 5 热力图） |
| **情绪向量轨迹** | 记录每轮对话的 6 维情绪值 | `EmotionInferencer` 输出 | 治疗过程中情绪动态变化 |
| **认知偏置激活** | 记录每轮注入的偏置类型和强度 | `ComplaintBiasInjector` 日志 | 消融实验的数据基础 |
| **对话轮次数** | 统计每次治疗对话的实际轮数 | `forced_chat` 日志 | 治疗剂量（dose）的代理指标 |
| **治疗完成度** | 记录完成的 session 编号 | `session_prompt_injection` 日志 | 区分"完成全部 CBT" vs "部分完成" |
| **记忆注入事件** | 记录何时注入了什么记忆 | `memory_injections` 日志 | 治疗关键事件的时序分析 |

#### (D) 行为指标 (Behavioral Measures)

| 指标 | 采集方式 | 数据来源 | 论文用途 |
|------|----------|----------|----------|
| **社交互动频率** | 统计非治疗性社交的次数 | `chat_events` 日志 | 社交退缩的客观指标 |
| **日常活动多样性** | 统计访问不同地点的数目 | `agent movement` 日志 | 功能损害程度 |
| **对话情感倾向** | 对患者所有非治疗对话做 sentiment analysis | 对话记录 | 情绪改善的外部验证 |

### 4.5 样本量与重复方案

#### 核心问题：LLM 的随机性

LLM 生成存在随机性（temperature > 0），同一配置运行多次会产生不同结果。**必须多次重复取平均**。

| 因素 | 水平数 | 值 |
|------|--------|-----|
| 实验组 (Group) | 5 | G1, G2, G3, G4, G5 |
| 抑郁严重度 (Severity) | 3 | mild, moderate, severe |
| 重复次数 (Repeats) | **≥3**（建议 5） | 每个条件独立运行 3–5 次 |

**推荐实验矩阵**：

```
总条件数 = 5 groups × 3 severities × 3 repeats = 45 次仿真
（如果 repeats=5 则为 75 次）

每次仿真: 60 steps × ~10 min/step ≈ 10 小时 (GPU)
总估算: 45 × 10h = 450 GPU-hours ≈ 19 GPU-days
```

**如果计算资源有限，优先保留的矩阵**：

| 优先级 | 组合 | 条件数 | 理由 |
|--------|------|--------|------|
| **必须** | G1 + G2, moderate × 3 repeats | 6 | 主结果的核心对比 |
| **必须** | G1 + G2 + G3 + G5, moderate × 3 repeats | 12 | 四组对照（去掉 G4） |
| **建议** | G1-G5, moderate × 3 repeats | 15 | 完整五组 |
| **加分** | G1-G5, 3 severities × 3 repeats | 45 | 全矩阵 + 严重度调节效应 |

### 4.6 消融实验设计

消融实验是证明系统各模块必要性的关键，建议分两轮：

#### 第一轮：模块级消融

| 条件名 | 保留 | 移除 | 预期结果 |
|--------|------|------|----------|
| **Full** | 所有模块 | — | 基准治疗响应 |
| **NoBias** | 除认知偏置外的全部 | `ComplaintBiasInjector` | 治疗响应减弱（患者不再有认知扭曲，治疗无靶点） |
| **NoEmotion** | 除情绪推断外的全部 | `EmotionInferencer` | 对话情感单一，真实性下降 |
| **NoChain** | 除投诉链外的全部 | `ComplaintChainManager` | 抑郁状态不随治疗动态变化 |
| **NoMemory** | 除治疗记忆外的全部 | 治疗后 `memory_injection` | 跨次治疗失去连续性 |

#### 第二轮：架构级消融（可选，Supplementary Information）

| 条件名 | 对比 | 预期结果 |
|--------|------|----------|
| **SimplePrompt** | 用简单人格描述替代完整抑郁引擎 | 行为"像正常人"而非抑郁患者 |
| **NoForcedLLM** | 治疗对话改用本地 Qwen3（不用 DeepSeek） | 治疗质量下降（模型能力差异） |
| **NoDialogJudge** | 移除对话法官 | 可能出现无限循环或低质量对话 |

**消融实验的数据需求**：每组至少 3 次重复，severity 固定为 moderate。

### 4.7 统计分析方案

#### (A) 主要分析

| 分析 | 方法 | 因变量 | 自变量 | 模型 |
|------|------|--------|--------|------|
| **治疗效应** | 线性混合效应模型 (LMM) | ΔPHQ-9 | Group (G1 vs G2) | `lmer(ΔPHQ-9 ~ Group + (1|Repeat))` |
| **组间差异** | 单因素方差分析 + post-hoc | T4 PHQ-9 | Group (5 levels) | one-way ANOVA + Tukey HSD |
| **纵向轨迹** | 重复测量 LMM | PHQ-9 (T0/S4/S8/T4) | Group × Time | `lmer(PHQ-9 ~ Group*Time + (1|Repeat))` |
| **严重度调节** | 两因素 LMM | ΔPHQ-9 | Group × Severity | `lmer(ΔPHQ-9 ~ Group*Severity + (1|Repeat))` |

#### (B) 报告要求

| 统计指标 | 要求 | 示例 |
|----------|------|------|
| 效应量 | 所有两两比较 | Cohen's d = 0.85, 95% CI [0.42, 1.28] |
| p 值 | 主要假设检验 | p = 0.003 (Bonferroni-corrected) |
| 置信区间 | 主要终点 | ΔPHQ-9 = −6.2 ± 1.8, 95% CI [−8.0, −4.4] |
| 轨迹图 | 纵向数据 | 均值 ± SEM shaded band |
| 组间比较 | 箱线图 + 效应量标注 | median, IQR, pairwise d |

#### (C) NMH 审稿人特别关注的统计问题

1. **多重比较校正**: 5 组两两比较 = 10 对，必须 Bonferroni 或 FDR 校正
2. **纵向分析选择**: 推荐混合效应模型 (LMM) 而非重复测量 ANOVA（更灵活处理缺失数据）
3. **效应量优先于 p 值**: NMH 倾向于报告效应量和置信区间，p 值作为补充
4. **明确统计假设**: 每个检验前声明零假设和备择假设

### 4.8 数据采集 Checklist

每次仿真运行后，确保以下数据已收集：

#### 必须收集（主论文用）

- [ ] PHQ-9 总分 × 4 时间点 (T0, S4, S8, T4)
- [ ] BDI-II 总分 × 4 时间点
- [ ] SDS 总分 × 4 时间点
- [ ] 抑郁严重度分级 × 4 时间点
- [ ] 完成的 session 编号列表
- [ ] 每次治疗对话的轮次数

#### 建议收集（Supplementary Information 用）

- [ ] 每轮对话的情绪向量 (6 维)
- [ ] 每轮对话的激活认知偏置类型
- [ ] 投诉链状态转移序列
- [ ] 记忆注入事件列表
- [ ] 非治疗性社交互动日志
- [ ] LLM 调用的 token consumption

#### 可选收集（未来工作用）

- [ ] 条目级 PHQ-9 评分（单题分析）
- [ ] 对话文本情感分析结果
- [ ] 智能体移动轨迹
- [ ] 记忆检索日志

### 4.9 预期结果与论文展示建议

| Figure | 数据来源 | 建议可视化 | 对应 Claim |
|--------|----------|-----------|------------|
| **Fig 2** | PHQ-9 × 4 时间点 × 5 组 | 纵向折线图 (mean ± SEM) | CBT 显著改善 PHQ-9 |
| **Fig 3** | T4 PHQ-9 × 5 组 | 箱线图 + pairwise Cohen's d | G1 > G3 > G2 > G5 |
| **Fig 4** | 消融实验 ΔPHQ-9 | 条形图 (mean ± SD) | 认知偏置模块关键 |
| **Fig 5** | 投诉链状态转移 × 2 时段 | 热力图 (before vs after) | 治疗改变认知模式 |
| **Table 1** | 所有基线/终点指标 | 均值 ± SD 表 | 全组概览 |
| **SI Table** | 条目级 PHQ-9 变化 | 热力图 | 哪些症状改善最大 |

### 4.10 常见实验部分的失败模式

| 失败模式 | 表现 | 修复方法 |
|----------|------|----------|
| 混合观察与解释 | Results 段出现 "suggests", "may reflect" | 将解释移到 Discussion |
| 模糊对比 | "higher than control" 无具体数字 | 添加效应量、样本量、检验方法 |
| 结果藏入补充材料 | 重要结果仅见于 SI | 放回正文 |
| 缺乏统计检验 | 描述趋势但无统计支持 | 添加适当的统计检验 |
| 无假设的统计 | 列出大量 p 值但无预设假设 | 每个检验前声明假设 |
| 重复不足 | 仅 1 次仿真就下结论 | ≥3 次重复，报告均值和方差 |
| 忽略 LLM 随机性 | 不报告 temperature/seed | 明确声明 LLM 参数和随机化方式 |

### 4.11 患者人设多样性设计

> **当前局限**: 项目仅有 2 个患者人设——卡布达（23 岁，男，设计系休学学生）和金龟次郎（50 岁，男，杂货店主），且仅卡布达拥有完整的 v2.0 抑郁引擎（含 3 种严重度变体）。系统目前无人设模板或批量生成机制，所有人设均为手工编写的单例。
>
> **Nature 审稿人的典型质疑**: "The simulation used only one patient profile. How can the authors claim generalizability?"
>
> 以下建议基于中国抑郁流行病学数据 (GBD 2021, [PMC11720356](https://pmc.ncbi.nlm.nih.gov/articles/PMC11720356/)) 和 CBT 临床试验的亚组分析惯例 ([Gergov 2024](https://link.springer.com/article/10.1007/s40894-023-00228-6))。

#### (A) 年龄层设计

**流行病学依据**（中国抑郁障碍年龄分布）：

| 年龄段 | 流行病学特征 | 风险因素 | 建议 |
|--------|-------------|----------|------|
| **18–25** (青年) | 发病率最高的一级峰值；学业/就业转型期压力 | 学业失败、求职受挫、社交焦虑、身份认同 | ✅ **必须纳入** — 已有卡布达 (23M) |
| **26–40** (壮年) | 患病率最高的年龄段 (Statista 2022: 35-44 岁风险最高)；工作/家庭双重压力 | 职业倦怠、婚姻冲突、经济压力、育儿负担 | ✅ **必须纳入** — 目前缺失 |
| **41–55** (中年) | 常被忽视但疾病负担高；空巢、健康退化 | 失业/裁员、更年期、慢性病共病、父母养老 | ✅ **建议纳入** — 已有金龟次郎 (50M) 但仅 v1.2 schema |
| **56–65** (中老年) | 中国老龄化背景下增长最快的群体；农村 (29.2%) > 城市 (20.5%) | 空巢、退休适应、丧偶、慢性疼痛 | ⚠️ 加分项 |

**推荐方案**：至少覆盖 **3 个年龄段**，每个年龄段至少 1 个人设。

```
青年 (18-25): 卡布达 23M ✅ 已有
壮年 (26-40): 新建人设 × 2（见下文）
中年 (41-55): 金龟次郎 50M（升级到 v2.0 schema）
```

#### (B) 性别分布

**流行病学依据**：
- 中国女性抑郁患病率 (11.7%) 约为男性 (6.7%) 的 1.7 倍 ([ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0165032707001747))
- 老年群体中差距更大：女性 24.2% vs 男性 19.4%

**推荐方案**：

| 方案 | 男 | 女 | 比例 | 适用场景 |
|------|-----|-----|------|----------|
| **最小可行** | 1 | 1 | 50:50 | 仅展示性别差异存在，不做统计推断 |
| **推荐** | 2 | 3 | 40:60 | 接近流行病学比例，可做探索性亚组分析 |
| **理想** | 3 | 4 | 43:57 | 接近流行病学比例，亚组分析有统计功效 |

> **注意**: 当前 2 个患者均为男性。若只新增 1 个人设，建议首选女性。

#### (C) 职业/生活情境覆盖

每种职业背景对应不同的抑郁触发事件和认知模式，直接影响投诉链的设计：

| 编号 | 人设代号 | 年龄 | 性别 | 职业/身份 | 触发事件 | 投诉链类型 | 对应认知偏置侧重 |
|------|---------|------|------|----------|----------|-----------|----------------|
| P1 | **卡布达** (已有) | 23 | 男 | 设计系休学学生 | 毕业设计失败→休学 | 职业自我怀疑链 | mental_filter, personalization |
| P2 | **新建-青年女** | 25 | 女 | 互联网公司初级运营 | 试用期被辞退→求职焦虑 | 职业否定链 | catastrophizing, fortune_telling |
| P3 | **新建-壮年男** | 35 | 男 | 中年程序员/工程师 | 项目失败→被裁员+房贷压力 | 经济-家庭双重崩溃链 | all_or_nothing, should_statements |
| P4 | **新建-壮年女** | 38 | 女 | 全职妈妈（前教师） | 孩子教育失败+自我价值丧失 | 角色迷失链 | personalization, emotional_reasoning |
| P5 | **金龟次郎** (升级) | 50 | 男 | 小镇杂货店主 | 经营困难→被迫关店 | 失去控制感链 | all_or_nothing, should_statements |
| P6 | **新建-中老年女** (加分) | 55 | 女 | 退休护士/社区工作者 | 退休+空巢+配偶慢性病 | 失去意义感链 | mental_filter, fortune_telling |

#### (D) 严重度 × 人设矩阵

**推荐最小矩阵**（论文主分析用）：

```
人设 × 严重度 × 重复次数

          mild    moderate    severe
P1 卡布达    3次      3次         3次      ← 已有完整 v2.0 配置
P2 青年女    3次      3次         3次      ← 新建
P3 壮年男    3次      3次         3次      ← 新建
P4 壮年女    3次      3次         3次      ← 新建
P5 金龟次郎  3次      3次         3次      ← 升级到 v2.0

小计: 5 人设 × 3 严重度 × 3 重复 = 45 次仿真
加上对照组 (G2/G3/G5, 不需要多人设): 3 组 × 3 严重度 × 3 重复 = 27 次
总计: 72 次仿真 × ~10h ≈ 720 GPU-hours ≈ 30 GPU-days
```

**如果资源受限，优先缩减方案**：

| 方案 | 人设数 | 严重度 | 重复 | 总次数 | GPU-days | 论文可写性 |
|------|--------|--------|------|--------|----------|-----------|
| **A. 完整** | 5 | 3 | 3 | 45+27=72 | ~30 | 完整亚组分析 |
| **B. 推荐** | 4 (P1-P4) | 3 | 3 | 36+27=63 | ~26 | 年龄×性别×严重度 |
| **C. 经济** | 3 (P1,P2,P3) | 2 (mod, sev) | 3 | 18+18=36 | ~15 | 年龄+性别，仅 2 种严重度 |
| **D. 最小** | 2 (P1,P2) | 1 (mod) | 5 | 10+5=15 | ~6 | 仅证明多样性存在，无统计功效 |

#### (E) 每个人设需创建的文件

项目当前无人设模板系统。每新增一个患者人设，需要手工创建以下文件：

| 文件 | 路径模板 | 内容 | 工作量 |
|------|---------|------|--------|
| `agent.json` | `frontend/static/assets/village/agents/{姓名}/agent.json` | 姓名、年龄、性别、性格、说话习惯、当前状态 | 中（参照卡布达模板修改） |
| `depression_config.json` | 同上目录 | 默认抑郁配置（建议 = moderate） | 高（需设计投诉链 stages） |
| `depression_config_mild.json` | 同上目录 | 轻度变体：核心信念较弱、偏置较少、情绪温和 | 中（从 moderate 放缩） |
| `depression_config_moderate.json` | 同上目录 | 中度变体 | 高（参照卡布达 moderate 设计） |
| `depression_config_severe.json` | 同上目录 | 重度变体：核心信念极端、偏置多重、情绪封闭 | 中（从 moderate 放缩） |

**关键设计要素**（每个人设都必须差异化）：

```
1. core_belief — 必须与触发事件匹配
   P2: "我什么都做不好，连最简单的工作都保不住"
   P3: "我是个失败者，养不了家，算什么男人"
   P4: "我把一切都给了家庭，结果什么都不是"

2. 投诉链 stages — 必须反映不同的人生困境
   P2: 面试失败 → 自我否定 → 社交回避 → 求职恐惧
   P3: 裁员冲击 → 家庭愧疚 → 经济焦虑 → 自我孤立
   P4: 角色空虚 → 育儿自责 → 身份迷失 → 价值崩塌

3. bias_profile — 不同人设侧重不同认知偏置
   P2: catastrophizing(0.7), fortune_telling(0.6)  — 灾难化未来
   P3: all_or_nothing(0.7), should_statements(0.6) — 完美主义崩塌
   P4: personalization(0.7), emotional_reasoning(0.6) — 过度内归因

4. emotion_vector baseline — 不同人设情绪起点不同
   P2: high arousal(焦虑驱动), moderate shame
   P3: low valence + high hopelessness(经济压力)
   P4: high shame + low trust(社交退缩)
```

#### (F) 统计学论证：为什么需要多个人设

**问题本质**：在 LLM 仿真中，"N" 有两层含义——

| 层次 | 含义 | 对应的变异来源 | 控制方法 |
|------|------|--------------|----------|
| **Layer 1: 同一人设多次运行** | LLM 随机性 | temperature, 采样随机性 | 重复 ≥3 次，报告均值±SD |
| **Layer 2: 不同人设** | 人口学多样性 | 年龄、性别、职业、认知模式 | 系统化人设设计，分层分析 |

**仅用 1 个人设 (N=1) 的后果**：
- 审稿人质疑：结果是否为该人设的特异性反应？
- 无法区分"CBT 对这个患者有效" vs "CBT 对这类患者有效"
- 无法进行年龄×治疗效果、性别×治疗效果的亚组分析

**多人设的统计分析计划**：

| 分析 | 模型 | 目的 |
|------|------|------|
| 主效应 | `lmer(ΔPHQ-9 ~ Group + (1|Persona) + (1|Repeat))` | 控制人设变异后的治疗效应 |
| 年龄调节 | `lmer(ΔPHQ-9 ~ Group*AgeGroup + (1|Repeat))` | 年龄是否调节治疗效果 |
| 性别调节 | `lmer(ΔPHQ-9 ~ Group*Gender + (1|Repeat))` | 性别是否调节治疗效果 |
| 人设异质性 | `ICCs (intraclass correlation)` | 人设间变异 vs 人设内变异的比例 |

> **建议在论文中的表述**: "To examine the generalizability of treatment effects across diverse patient profiles, we created four additional patient personas spanning ages 25–50, both genders, and distinct occupational backgrounds (Extended Data Table X). Treatment response was modeled using linear mixed-effects models with random intercepts for persona and repeat, controlling for between-persona variability."

#### (G) 金龟次郎的升级路径

金龟次郎目前使用旧的 v1.2 inline `depression_profile`，需要升级到 v2.0 独立配置：

| 升级项 | 当前状态 (v1.2) | 目标状态 (v2.0) |
|--------|----------------|----------------|
| 配置位置 | `agent.json` 内嵌 | 独立 `depression_config*.json` 文件 |
| 投诉链 | 无（仅有 static case_config） | 完整的 `complaint_chain` + `stages` |
| 认知偏置 | 无（仅有 cognitive_pattern 文本描述） | `bias_profile` + 7 类偏置权重 |
| 情绪向量 | 无（仅有 emotion_experience 文本） | 6 维 `emotion_vector` + 波动约束 |
| 严重度变体 | 仅 mild | mild + moderate + severe 三套 |

#### (H) 论文中如何呈现人设多样性

| 位置 | 内容 | 篇幅 |
|------|------|------|
| **Methods 4.2** | 概述人设设计原则（流行病学抽样、年龄段覆盖、性别比例） | 50-80 words |
| **Extended Data Table 2** | 全部人设的详细特征表（年龄、性别、职业、触发事件、核心信念、偏置侧重） | 1 full page |
| **SI Section 2** | 每个人设的完整投诉链设计 + 配置文件 | 不限 |
| **Results 2.1** | 一句声明多人设 + ICC 值 | 20-30 words |
| **SI Figure** | 各人设的基线 PHQ-9 分布（验证不同人设确实产生不同基线） | 1 figure |

---

## 五、生成式智能体 + 抑郁症研究领域的定位与差异化

### 5.1 当前领域生态

| 研究方向 | 代表工作 | 发表期刊 | 与本项目关系 |
|----------|----------|----------|-------------|
| 生成式智能体 + 社会环境因素 | Kambeitz & Meyer-Lindenberg 2025 | npj Digital Medicine | 最直接的先驱论文（Perspective/Review） |
| 计算建模综述 | Zavlis et al. 2025 | **Nat. Ment. Health** | 方法学框架参考 |
| 症状网络温度 | Epskamp et al. 2025 | **Nat. Ment. Health** | 抑郁症状动态建模参考 |
| 多模态抑郁检测 | Zhang et al. 2026 | **Nat. Ment. Health** | 技术路线对比 |
| 脑结构-免疫代谢 | Mamalakis et al. 2023 | **Nat. Ment. Health** | 生物机制层面参考 |
| 身体活动作为跨诊断标记 | Benedyk et al. 2024 | **Nat. Ment. Health** | 行为层面参考 |
| 暴露组-抑郁 | Bizzozzero et al. 2023 | **Nat. Ment. Health** | 环境因素参考 |

### 5.2 本项目的差异化优势

**关键定位策略**：

1. **从 Perspective 到实证**：Kambeitz (2025) 提出了概念框架，本项目提供**首次实证验证**
2. **聚焦治疗过程**：现有工作多关注社会环境因素对心理健康的影响，本项目聚焦**治疗干预过程的仿真**
3. **临床场景模拟**：不仅模拟日常生活，还模拟**心理咨询/治疗场景**
4. **评估体系创新**：引入标准化的临床量表评估智能体的"心理健康状态"

### 5.3 推荐引用的核心文献

| 文献 | 引用场景 | DOI |
|------|----------|-----|
| Park et al. 2023 (Generative Agents) | 智能体架构基础 | arXiv:2304.03442 |
| Kambeitz & Meyer-Lindenberg 2025 | 生成式智能体 + 心理健康领域定位 | 10.1038/s41746-024-01422-z |
| Zavlis et al. 2025 | 计算精神病学方法学综述 | 10.1038/s44220-025-00465-9 |
| Hauser et al. 2022 | 计算精神病学理论框架 (Lancet Digital Health) | 10.1016/S2589-7500(22)00152-2 |
| Friston 2023 | 计算精神病学 (Mol Psychiatry) | 10.1038/s41380-022-01743-z |
| Kroenke et al. 2001 | PHQ-9 量表 | 10.1046/j.1525-1497.2001.016009606.x |
| Beck 1967 | 认知模型 | 经典引用 |
| Volkmer et al. 2024 | LLM在精神科的应用与挑战 | 10.1016/j.psychres.2024.116026 |
| Shanahan et al. 2023 | LLM 角色扮演 (Nature) | 10.1038/s41586-023-06647-8 |

---

## 六、写作工作流建议

### 6.1 推荐的 8 步写作流程

```
1. 论点句 (Argument sentence)      → 每节用一句话概括核心论点
2. 术语表 (Terminology ledger)     → 统一关键术语，全文一致
3. 章节架构 (Section architecture) → 确定各节长度和内容分配
4. 段落映射 (Paragraph mapping)    → 每段一个任务，列出段间逻辑
5. 证据驱动起草 (Draft from evidence) → 从数据/结果出发写，不空谈
6. 动词校准 (Calibrate verbs)      → 根据证据强度调整动词确定性
7. 去除无支撑声明 (Remove unsupported claims) → 每个声明都有对应证据
8. 段落流畅度检查 (Paragraph flow check) → 段间过渡是否自然
```

### 6.2 各节篇幅分配建议（3000 words 总量）

| 章节 | 建议词数 | 占比 | 备注 |
|------|----------|------|------|
| Introduction | 600–700 | ~20% | 4段经典漏斗 |
| Results | 1000–1200 | ~35% | 证据阶梯，claim-first |
| Discussion | 600–700 | ~20% | 意义 + 局限 + 未来方向 |
| Methods | 500–600 | ~18% | 可部分移入 SI |
| Abstract | ≤150 | — | 最后写 |

### 6.3 中英文写作转换要点

由于本项目团队以中文为主，写作时需注意：

| 中文写作常见模式 | 英文修复策略 |
|------------------|-------------|
| 背景和方法混在一句话 | 先拆分为 claim / evidence / condition，按英文章节顺序重组 |
| "显著提高" 无基线 | 添加比较对象或弱化动词 |
| "首次" 无范围限定 | 用 bounded novelty claim (如 "to our knowledge, the first to...in the context of...") |
| 结果和讨论混合 | 观察放 Results，解释放 Discussion |
| 逗号连接的长句 | 拆分或添加显式连接词 |
| 重复主语名词 | 使用代词或省略 |

---

## 七、关键参考文献列表

### Nature Mental Health 抑郁症核心论文

1. **Zavlis et al. (2025)** — "Computational modelling approaches in mental health research: a systematic review"  
   Nat. Ment. Health. DOI: 10.1038/s44220-025-00465-9

2. **Network temperature paper (2025)** — "Network temperature as a marker of depression symptom dynamics"  
   Nat. Ment. Health. DOI: 10.1038/s44220-025-00415-5

3. **Zhang et al. (2026)** — "Multimodal deep learning for depression detection"  
   Nat. Ment. Health. DOI: 10.1038/s44220-026-00632-6

4. **Mamalakis et al. (2023)** — "Brain structure and immunometabolic mechanisms in depression"  
   Nat. Ment. Health. DOI: 10.1038/s44220-023-00120-1

5. **Benedyk et al. (2024)** — "Physical activity as a transdiagnostic marker"  
   Nat. Ment. Health. DOI: 10.1038/s44220-024-00204-6

6. **Bizzozzero et al. (2023)** — "Exposome-wide depression study"  
   Nat. Ment. Health. DOI: 10.1038/s44220-023-00124-x

### 生成式智能体 + 心理健康

7. **Kambeitz & Meyer-Lindenberg (2025)** — "Modelling the impact of environmental and social determinants on mental health using generative agents"  
   npj Digit. Med. 8, 36. DOI: 10.1038/s41746-024-01422-z

8. **Park et al. (2023)** — "Generative agents: Interactive simulacra of human behavior"  
   arXiv:2304.03442

9. **Shanahan et al. (2023)** — "Role play with large language models"  
   Nature 623, 493–498. DOI: 10.1038/s41586-023-06647-8

### 计算精神病学理论

10. **Hauser et al. (2022)** — "The promise of a model-based psychiatry"  
    Lancet Digit. Health 4, e816–e828. DOI: 10.1016/S2589-7500(22)00152-2

11. **Friston (2023)** — "Computational psychiatry: from synapses to sentience"  
    Mol. Psychiatry 28, 256–268. DOI: 10.1038/s41380-022-01743-z

### 临床评估工具

12. **Kroenke et al. (2001)** — PHQ-9  
    J. Gen. Intern. Med. 16, 606–613. DOI: 10.1046/j.1525-1497.2001.016009606.x

13. **Spitzer et al. (2006)** — GAD-7  
    Arch. Intern. Med. 166, 1092–1097. DOI: 10.1001/archinte.166.10.1092

---

## 八、投稿前的 Checklist

### 科学内容
- [ ] 每个主要声明都有对应的对比/消融/压力测试证据
- [ ] 方法部分每个模块都有 Motivation → Mechanism → Evidence/Role 三要素
- [ ] 仿真结果与真实世界临床数据有对比验证
- [ ] 局限性讨论充分且诚实

### 格式规范
- [ ] 正文 ≤3,000 words
- [ ] 摘要 ≤150 words, 无小标题
- [ ] 展示项 ≤6 个 (figures + tables)
- [ ] 参考文献 ~30–50 篇
- [ ] Data Availability 声明
- [ ] Ethics 声明（如涉及真实患者数据对比）
- [ ] AI 使用声明（如使用 LLM 辅助写作）

### 语言质量
- [ ] 无 "first" / "novel" 的无范围限定使用
- [ ] 动词确定性校准（evidence-first）
- [ ] Results 无 Discussion 语法混入
- [ ] 统计报告完整（效应量 + CI + 检验方法）
- [ ] 每个模糊表述已替换为可复现信息

---

## 九、系统 Demo：基于生成式智能体的抑郁症模拟、治疗与评估

> **Demo 脚本**: [demo_depression_simulation.py](demo_depression_simulation.py)  
> **参考文档**: 基于AI对抑郁症进行模拟、治疗与评估 - 研究进展（附示例）.pdf

### 9.1 系统总览

本系统由三个核心角色组成，形成一个完整的**模拟 → 干预 → 评估**闭环：

```
抑郁症智能体 (卡布达)  ←→  心理医生智能体 (蜻蜓队长)  ←→  评估 LLM
     患者角色                    治疗师角色                   量表评分
```

| 角色 | 名称 | 功能 | LLM 后端 |
|------|------|------|----------|
| 抑郁症 Agent | 卡布达 | 模拟抑郁症患者情绪/认知/行为 | Qwen3-8B (本地 vLLM) |
| 心理医生 | 蜻蜓队长 | 实施 CBT/PST 结构化治疗 | DeepSeek API |
| 评估 LLM | — | PHQ-9/BDI-II/SDS 量表评分 | DeepSeek API (ExpertLLM) |

### 9.2 抑郁症智能体的认知架构

患者智能体（卡布达）的抑郁仿真由五个子模块协同驱动：

| 子模块 | 功能 | 输出 |
|--------|------|------|
| **ComplaintChainManager** | 投诉链状态机，管理抑郁状态阶段转移 | 当前阶段标签 + 转移信号 |
| **ComplaintBiasInjector** | 选择并注入认知扭曲（如心理过滤、个人化） | 认知偏置提示词片段 |
| **EmotionInferencer** | 逐轮推断情绪向量（效价/唤醒/防御性/羞耻/绝望/信任） | 6维情绪向量 |
| **SessionContextBuilder** | 从对话中提取情境信息（地点/时间/关系） | 结构化上下文 |
| **DynamicPromptBuilder** | 组装最终抑郁人设提示词 | 完整 prompt 注入 LLM |

**投诉链 (Complaint Chain) 示例**（以轻度抑郁为例）：

```
mild_job_loss_self_doubt
  ↓ (对话中检测到 advance_signals)
mild_rumination_deepening
  ↓
mild_social_withdrawal
  ↓
mild_residual_or_recovery  ← 治疗有效时可跳转至此
```

每个状态包含：核心信念、叙事焦点、说话风格、情绪向量、认知偏置配置、与其他角色（家人/朋友/邻居/欺凌者）的关系修正。

### 9.3 CBT 治疗流程

系统实现了完整的 CBT 四阶段结构化治疗方案：

| 阶段 | 会话 | 临床目标 |
|------|------|----------|
| **Stage 1** 信息收集 | session1 → session_loop | 建立关系，识别核心压力源，收集 S-E-T（情境-情绪-想法）样本 |
| **Stage 2** 认知概念化 | session 2.1–2.3 | 标记认知扭曲，识别条件规则，用向下箭头技术发现核心信念 |
| **Stage 3** 认知重构 | session 3.1–3.3B | 法庭练习（支持/反对证据），设计行为实验，成功/回避路径分支 |
| **Stage 4** 防复发 | session 4.1–4.4 | 旅程回顾，心理急救包，压力测试复盘，毕业仪式 |

**Session 1 对话示例**（摘自 Demo）：

> **蜻蜓队长**: 你好，卡布达。谢谢你愿意来。这里是一个安全的、不会被评判的空间。最近是什么让你决定来这里的？
>
> **卡布达**: ...其实也没什么特别大的事。就是最近...工作丢了之后，就觉得什么都提不起劲。朋友叫我出去我也懒得去，晚上也睡不好...
>
> **蜻蜓队长**: 听起来这段时间你承受了不少。你提到'提不起劲'和'睡不好'——如果让你回忆最近一次有这种感觉的具体场景，你能描述一下当时发生了什么吗？当时你心里在想什么？
>
> **卡布达**: 就...前天吧。金龟次郎说有个设计比赛让我试试，我看了下要求就觉得——'反正我这种连工作都保不住的人，参加也是丢脸'...然后就直接关掉了网页。那晚翻来覆去到三四点才睡着。
>
> **蜻蜓队长**: 很好，你描述得很清楚。我们一起来看看——情境: 看到设计比赛通知 / 情绪: 羞耻、无力 / 想法: '我这种连工作都保不住的人，参加也是丢脸'。你注意到没有，你的想法直接影响了你的情绪和后续行为。这就是 CBT 中的'认知三角'。

每次治疗对话后，系统自动执行：
1. **对话法官 (Dialog Judge)** 评估对话质量并建议继续/终止
2. **会话评估 (Session Eval)** 评定当次疗效
3. **记忆注入 (Memory Injection)** 将关键洞察写入患者的长期记忆
4. **抑郁状态更新** 根据治疗效果更新投诉链状态

### 9.4 标准化量表评估

**评估量表体系**：

| 量表 | 类型 | 条目数 | 评估内容 |
|------|------|--------|----------|
| PHQ-9 | 自评 | 9 | 抑郁症状严重度（过去两周） |
| BDI-II | 自评 | 21 | 贝克抑郁认知/躯体/情感症状 |
| SDS | 自评 | 20 | Zung 抑郁自评量表 |

**阶段性评估时间线**（Staged Evaluation）：

```
T0 (基线)  →  Session 4  →  Session 8  →  Session 12  →  T4 (随访)
 评估起点     首次中期      二次中期        后期评估       治疗结束后
              评估          评估                         随访评估
```

评估流程：患者智能体逐条回答量表题目 → ExpertLLM (DeepSeek) 根据标准化评分提示词判定抑郁等级 → 输出正常/轻度/中度/重度。

### 9.5 实验组设计

| 组别 | 名称 | 干预内容 | 目的 |
|------|------|----------|------|
| G1 | 医生干预组 | 完整 CBT 治疗（抑郁引擎 + 对话法官 + 量表评估） | 主实验组 |
| G2 | 无干预对照组 | 无治疗干预，抑郁引擎仍活跃 | 自然病程基线 |
| G3 | 普通社交组 | 与其他居民随机中性社交对话 | 排除"对话本身"的安慰效应 |
| G4 | 咨询室组 | 同 G1 但限定在咨询室环境 | 环境效应控制 |
| G5 | 负面社交组 | 与其他居民进行负面/有害对话 | 伤害性对照 |

### 9.6 Demo 运行方法

```bash
# 1. 启动 vLLM 服务
bash runshells/vllm_services.sh start

# 2. 展示模式（无需 LLM 服务，仅展示系统配置和流程）
python writing/demo_depression_simulation.py --no-interactive

# 3. 交互式运行（需要 vLLM 服务 + .env 配置）
python writing/demo_depression_simulation.py --steps 3 --verbose

# 4. 运行完整仿真
python start.py --name demo --step 48 --stride 60 --start 20260425-09:30

# 5. 运行批量实验
python3 runshells/run_batch_experiment.py

# 6. 启动评估界面 (Gradio)
python customization/depression_scale_agent/app.py
```

### 9.7 Demo 输出示例

Demo 运行后依次展示六个阶段：

| Phase | 内容 | 需要LLM |
|-------|------|---------|
| 1 | 系统配置总览（LLM后端/干预系统/抑郁引擎） | ❌ |
| 2 | 智能体角色档案（患者/医生/居民人设） | ❌ |
| 3 | 抑郁仿真引擎（投诉链/认知偏置/情绪推断） | ❌ |
| 4 | CBT 治疗对话（结构化会话流程 + 示例对话） | ❌ |
| 5 | 标准化量表评估（PHQ-9 条目 + 评估时间线） | ❌ |
| 6 | 完整 Pipeline 架构（系统全景图 + 实验组） | ❌ |
| 7 | 交互式仿真（实际运行仿真） | ✅ |

---

*本分析基于 2023–2026 年 Nature Mental Health 已发表论文的系统性调研，结合 Kambeitz & Meyer-Lindenberg (2025) 和 Zavlis et al. (2025) 的详细文本分析，以及 Nature-writing 写作技能框架整理而成。Demo 基于项目实际代码架构设计，引用了真实的智能体配置、抑郁引擎模块和 CBT/PST 治疗方案。*
