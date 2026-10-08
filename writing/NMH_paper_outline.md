# Paper Outline: [ModelName] — 基于 LLM 生成式智能体的抑郁症治疗仿真


## 📑 目录

- [拟定标题](#拟定标题-3-个备选)
- [Abstract](#abstract-150-words-无小标题)
- [正文大纲](#正文大纲)
  - [1. Introduction](#1-introduction-600-words-4-paragraphs)
  - [2. Results](#2-results-1000-words-4-subsections)
  - [3. Discussion](#3-discussion-600-words-4-paragraphs)
  - [4. Methods](#4-methods-500-words-正文精简版-详细版见-supplementary-information)
- [展示项规划](#展示项规划-6)
- [参考文献预算](#参考文献预算-40-篇)
- [写作顺序建议](#写作顺序建议)
- [下一步行动](#下一步行动)

---

## 拟定标题 (3 个备选)

1. **LLM-powered generative agents simulate depression treatment trajectories in a virtual community**
2. **Simulating cognitive behavioural therapy for depression using LLM-based generative agents**
3. **Generative agents with depression modelling enable in-silico evaluation of psychotherapy interventions**

---

## Abstract (~150 words, 无小标题)

> **[Context]** 抑郁症是最常见的精神障碍之一，但心理治疗的大规模随机对照试验受限于成本、伦理和时间。**[Gap]** 现有计算模型难以捕捉治疗对话中的动态认知-情绪交互。**[Approach]** 我们提出 [ModelName]——一个基于大语言模型的生成式智能体系统，通过投诉链状态机、认知偏置注入和动态情绪推断，在虚拟社区中模拟抑郁症患者对认知行为疗法（CBT）的治疗响应。**[Key Result]** 在 7 组对照实验中（N=×3 重复），接受结构化 CBT 的患者智能体 PHQ-9 评分较基线下降 XX%（Cohen's d = X.XX），显著优于无干预对照组（p < 0.01）。支持性咨询组（G6）的治疗效应显著弱于 CBT 组，表明认知重构技术是不可替代的活性成分。记忆移除消融组（G7）的 T4 随访回弹率显著高于完整 CBT 组，揭示记忆巩固对疗效持久性的因果贡献。**[Implication]** [ModelName] 为心理治疗的机制研究和方案优化提供了一种低成本、可重复的 in-silico 实验平台。**[Boundary]** 仿真结果仍需与真实临床队列数据交叉验证。

---

## 正文大纲

### 1. Introduction (~600 words, 4 paragraphs)

**Para 1 — Field stake (领域重要性)**
- 全球抑郁症负担：3.2 亿患者，WHO 2030 年最大疾病负担
- 心理治疗（CBT）是一线推荐方案，但疗效异质性大（response rate 40-60%）
- RCT 是金标准但成本高昂（单中心 RCT 均值 >$50 万）、周期长（12-24 月）、伦理约束多

**Para 2 — Bottleneck (现有瓶颈)**
- 治疗机制研究依赖事后回溯，难以实时捕捉认知-情绪的动态交互
- 传统计算模型（agent-based models, Bayesian models）行为 repertoire 有限，难以模拟治疗对话的丰富语义
- 数字表型（digital phenotyping）可被动采集但无法模拟干预

**Para 3 — Prior work → Gap**
- Park et al. (2023) 提出生成式智能体框架，在虚拟社区中实现可信的社交行为
- Kambeitz & Meyer-Lindenberg (2025, npj Digital Medicine) 提出将生成式智能体用于心理健康研究的概念框架
- **Gap**: 尚无研究在生成式智能体中实现临床级别的抑郁症建模并实证评估心理治疗干预

**Para 4 — This study (本文贡献)**
- 我们开发了 [ModelName] 系统，首次将抑郁症的认知行为模型（Beck 认知模型）与 LLM 生成式智能体结合
- 在虚拟社区中对认知行为疗法（CBT）进行 in-silico 评估
- 7 组对照实验（含支持性咨询对照 G6、记忆移除消融 G7）+ 标准化量表（PHQ-9, BDI-II, SDS）纵向评估 + 治疗后回访观察

---

### 2. Results (~1,000 words, 4 subsections)

> 每节遵循 claim-first → evidence → comparison 结构

**2.1 [ModelName] 系统概览与内部效度验证**
- **Claim**: [ModelName] 能模拟具有临床一致性的抑郁症患者行为
- **Evidence**: 卡布达（轻度抑郁智能体）在 PHQ-9 基线评估中得分 X/27，与轻度抑郁临床预期一致
- **Validation**: 高神经质特质智能体的基线抑郁评分显著高于低神经质智能体（复现神经质-抑郁关联）
- → 对应 **Figure 1**: 系统架构图（Pipeline overview）

**2.2 CBT 治疗响应的主结果**
- **Claim**: 接受结构化 CBT 的患者智能体在治疗结束时 PHQ-9 评分显著下降
- **Quantitative**: G1 (CBT) ΔPHQ-9 = -X.X ± X.X vs G2 (无干预) ΔPHQ-9 = -X.X ± X.X; Cohen's d = X.XX, p < 0.01
- **Trajectory**: 治疗响应呈非线性，Session 4-8 间出现最大改善（认知重构阶段）
- → 对应 **Figure 2**: 各组 PHQ-9 评分纵向轨迹图

**2.3 七组对照实验比较**
- **Claim**: CBT 治疗效果优于普通社交接触、负面社交和支持性咨询，且记忆巩固是疗效持久性的必要条件
- G1 (CBT) vs G2 (无干预): 确立总体治疗效应
- G1 (CBT) vs G3 (随机社交): 证明治疗效果非源于单纯社交互动
- G1 (CBT) vs G5 (负面社交): 负面社交导致症状恶化
- G1 (CBT) vs G6 (支持性咨询): **关键对照**——隔离 CBT 技术特异性效应 vs. 非特异性治疗因素（治疗联盟、定期关注）
- G1 (CBT) vs G7 (记忆移除): **消融对照**——隔离记忆巩固在治疗响应中的因果角色
- G4 (咨询室环境) ≈ G1: 排除环境特异性
- → 对应 **Figure 3**: 七组终点 PHQ-9 评分箱线图 + pairwise effect sizes（重点标注 G1 vs. G6 / G1 vs. G7）

**2.4 消融分析：CBT 特异性成分与记忆巩固的贡献**
- **Claim**: CBT 结构化技术（非仅支持性接触）和记忆巩固是治疗仿真的关键组件
- G1 vs. G6 (移除 CBT 结构): 仅保留支持性接触时治疗响应下降 XX%；证明认知重构等技术是不可替代的活性成分
- G1 vs. G7 (移除记忆写入): 阻断患者形成新记忆后，治疗即时效果可能保留但持久性显著下降（T4 回弹 XX%）
- Full model vs No-emotion (可选): 移除情绪推断后对话真实性下降、投诉图阶段-情绪一致性受损
- → 对应 **Figure 4**: 消融实验结果条形图（G1/G6/G7 为主；NoEmotion/NoGraph 可选放入 SI）

---

### 3. Discussion (~600 words, 4 paragraphs)

**Para 1 — 核心发现总结**
- [ModelName] 首次实现了在生成式智能体中的临床级抑郁症建模与治疗评估
- CBT 在仿真中表现出与真实临床文献一致的治疗效应量
- G6（支持性咨询）与 G1（CBT）的对比表明结构化认知技术是治疗效果的特异性活性成分，不可被一般性支持接触替代
- G7（记忆移除）的消融结果揭示记忆巩固在治疗响应持久性中的因果角色
- 治疗后回访模式使系统能够观察治疗终止后的自然轨迹

**Para 2 — 与现有工作的关系**
- 与 Kambeitz & Meyer-Lindenberg (2025) 的概念框架对接：从建议到实证
- 与传统 agent-based models (Bonabeau 2002) 的区别：语义丰富的对话而非简化行为规则
- 与计算精神病学建模 (Hauser et al. 2022, Friston 2023) 的互补：前者侧重机制，我们侧重交互

**Para 3 — 局限性**
- LLM 训练数据的文化偏差可能影响跨文化泛化
- 患者画像主要围绕青年男性（卡布达系列），虽通过 9 个 variant 覆盖不同压力源，但年龄/性别多样性有限（金龟次郎为中年男性，女性人设待扩展）
- 仿真结果尚未与真实 RCT 数据进行定量校准
- LLM 的随机性需通过多次重复（≥3 次）和条目级稳定性分析来控制
- G7 记忆移除是极端消融条件（完全阻断记忆写入），需与真实世界中记忆减弱（如睡眠剥夺、药物影响）的非极端对照区别讨论

**Para 4 — 展望**
- 扩展到多种抑郁亚型（共病焦虑、难治性抑郁）
- 引入药物治疗建模 + 联合治疗仿真
- 与真实临床队列数据进行定量校准
- 探索个性化治疗方案优化（precision psychiatry）

---

### 4. Methods (~500 words, 正文精简版; 详细版见 Supplementary Information)

**4.1 系统架构**
- 基于生成式智能体框架 (Park et al. 2023)，扩展五模块抑郁仿真引擎
- LLM 后端: Qwen3-8B (本地 vLLM) 用于智能体认知; DeepSeek API 用于治疗对话生成
- 嵌入模型: BGE-M3 用于长期记忆检索

**4.2 抑郁症智能体建模**
- **Complaint Graph (投诉图)**: 受 Beck 认知模型启发的 LLM 驱动状态机，管理核心信念→叙事焦点→情绪向量的阶段转移；图推进和转移由专用 LLM prompt 判断（`graph_planner.txt` / `graph_transition.txt`），支持树形分支结构
- **Dynamic Emotion Inference (动态情绪推断)**: LLM 驱动的逐轮瞬时情绪推断，输出 label（情绪标签）/ style（情绪风格）/ intensity（强度）/ disclosure_level（暴露程度）/ defensiveness（防御水平）/ volatility_note（波动描述），带波动性限幅约束以防止不现实的突然变化
- **Session Context Builder (会话上下文构建)**: 从对话中提取结构化情境信息（地点/时间/对话对象关系/互动类型/主诉主题）
- **Trauma Memory System (创伤记忆系统)**: 与投诉图节点关联的创伤记忆存储，可在表达中作为背景叙事浮现
- **Dynamic Prompt Builder (动态 Prompt 组装)**: 四层 Prompt 组装（基础人格 → 投诉节点 → 会话上下文 → 瞬时情绪），稳定人设在前、轮级波动在后
- 人格建模: 自由文本人格描述（如"敏感、内省、善良"）结合抑郁严重度参数（mild/moderate/severe），通过 prompt 注入影响行为生成

**4.3 治疗干预模块**
- CBT 四阶段结构化方案 (12 个 session + 自适应循环): 信息收集 → 认知概念化 → 认知重构 → 防复发；每阶段含自适应 session_loop，未达标时自动重复
- 治疗后回访模式 (Post-Treatment Follow-up): 完成全部 CBT session 后自动切换为回访提示词，聚焦状态监测、作业跟进和风险筛查，不再推进新 session——便于观察治疗终止后的自然轨迹
- 支持性咨询对照 (G6): 医生以相同频率会见患者但仅使用非指导性、验证式沟通——不进行认知重构、不布置行为实验、无结构化 session 推进——用于隔离 CBT 技术特异性效应
- 对话法官 (Dialog Judge): LLM 驱动的对话质量监控，实时评估并建议继续/终止；advice 被约束为策略性指导（如"探索患者的'应该'陈述"），禁止生成可照抄的完整医生话术
- 会话评估 (Session Eval): 独立 LLM 判定单次治疗是否达标
- 环境任务模型 (Environment Model): 医生布置的会后作业经由 LLM 模拟为具体生活事件（完成/部分完成/卡住），写入患者及相关居民的记忆流，使后续对话能自然跟进作业执行情况
- 会话后记忆注入: 治疗关键洞察以 (subject, predicate, object) 三元组形式写入患者长期记忆，确保跨次治疗连续性

**4.4 评估体系**
- 三项标准化自评量表: PHQ-9, BDI-II, SDS
- 阶段性评估时间线: T0 (基线) → Session 4 (中期) → Session 8 (后期) → T4 (随访，治疗结束后约 48 虚拟小时)；每 4 次完成对话触发一次
- 对于无结构化 session 的对照组（G2/G3/G5/G6），通过 step-interval fallback 生成虚拟评估点，确保组间可比
- Expert LLM 评分: 独立的 DeepSeek-Chat 模型（区别于治疗对话使用的 DeepSeek-V4-Flash）基于专用评分提示词进行严重度判定，输出正常/轻度/中度/重度四级
- 评估质量保障: 支持重复评估 + 条目级稳定性分析 + 自适应补跑不稳定条目 + 条目分与 LLM 总分交叉校验

**4.5 实验设计**
- **7 组对照** × 3 种严重度 (mild/moderate/severe) × N 次重复
  - G1: CBT 结构化治疗（全模块）
  - G2: 无干预（自然病程基线）
  - G3: 随机中性社交（安慰剂对照——排除"有人说话"的安慰效应）
  - G4: 咨询室环境限定（排除环境特异性）
  - G5: 负面社交（伤害性对照——有害社交的恶化效应）
  - **G6: 支持性心理咨询（新增——仅非指导性支持，无 CBT 结构/判断/评估——隔离 CBT 技术特异性效应）**
  - **G7: 记忆移除（新增——同 G1 但阻断患者 event/thought/chat 记忆写入——消融记忆巩固的因果角色）**
- 患者人设: 9 个卡布达 variant（共享人口学身份，差异化压力源/投诉图轨迹）+ 金龟次郎（50M，跨年龄验证）
- 每次仿真 48–120 步 (每步 = 360 分钟/6 小时虚拟时间)
- 抑郁严重度作为调节变量纳入分析
- 统计方法: 线性混合效应模型 (LMM)，随机截距纳入人设和重复；效应量报告 Cohen's d；多重比较 Bonferroni 校正

---

## 展示项规划 (≤6)

| # | 类型 | 内容 | 对应章节 |
|---|------|------|----------|
| Fig 1 | 架构图 | [ModelName] 系统总览（5 层：虚拟社区→智能体认知→抑郁引擎→治疗干预→评估与实验设计） | Results 2.1 |
| Fig 2 | 折线图 | 各组 PHQ-9 评分纵向轨迹 (T0→S4→S8→T4)，7 条折线 + 95% CI shaded band | Results 2.2 |
| Fig 3 | 箱线图 | 七组终点 PHQ-9 评分分布 + pairwise Cohen's d（重点高亮 G1 vs. G6 / G1 vs. G7） | Results 2.3 |
| Fig 4 | 条形图 | 消融实验结果：G1 vs. G6（移除 CBT 结构）vs. G7（移除记忆写入），可选附加 NoEmotion/NoGraph | Results 2.4 |
| Fig 5 | 热力图 | 投诉图状态转移矩阵 (治疗前 vs 治疗后) | Results 2.2 or SI |
| Table 1 | 表格 | 七组基线特征与终点指标汇总 (demographics + PHQ-9/BDI/SDS + T4 随访) | Results 2.3 |

---

## 参考文献预算 (~40 篇)

| 类别 | 数量 | 核心文献 |
|------|------|----------|
| 抑郁症临床 | 6-8 | WHO 报告, CBT meta-analysis, treatment response rates |
| 生成式智能体 | 4-5 | Park 2023, Shanahan 2023, Kambeitz 2025, Wang 2023 |
| 计算精神病学 | 4-5 | Hauser 2022, Friston 2023, Montague 2012, Zavlis 2025 |
| LLM 与心理健康 | 4-5 | Volkmer 2024, Stade 2024, Hodson 2024 |
| 认知行为理论 | 3-4 | Beck 1967, Beck 2011 (CBT manual), Kroenke 2001 (PHQ-9) |
| 方法/技术 | 4-5 | Bonabeau 2002, Tracy 2018, embedding, vLLM |
| 统计/报告 | 2-3 | Cohen 1988, NHM author guidelines |

