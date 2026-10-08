# Nature Mental Health 论文大纲（Nature 格式整合版）

> **生成日期**: 2026-08-11
> **对齐格式**: Nature Mental Health Article（正文 ~3,000 words；摘要 ≤150 words 无小标题；展示项 ≤6；参考文献 30–50）
> **写作规范来源**: `NMH_depression_paper_analysis.md`（Nature-writing 框架内化版：漏斗结构 + 动词校准 + 6 要素摘要 + 方法学 Motivation→Mechanism→Evidence/Role 三要素）
> **整合输入**: `NMH_paper_outline.md`（骨架）、`nature_paper_draft.md`（英文草稿）、`methodology.md` v2.0（方法学详述）、`2026-08-07-论文初稿_小镇仿真与实验资料大纲.md`（代码实际状态）、`2026-08-07-动态抑郁状态建模与交互生成模块_章节编排.md`（抑郁模块技术编排）
>
> **核心判断**: 现有 `nature_paper_draft.md` 已是按 Nature 格式写就的较完整草稿，但其建立在"7 组 × 3 严重度 × 9 人设"完整矩阵的假设上；最新资料大纲表明实际进度为"跨人设已做、跨严重度未做、G4/G6/G7 结果待核验"。本大纲对齐"已做 / 未做 / 待核验"的真实状态，并保留 Nature 格式骨架。

---

## 0. 元信息与标题

- **文章类型**: Article（实证研究论文）
- **拟定标题**（3 备选，推荐第 1）:
  1. LLM-powered generative agents simulate depression treatment trajectories in a virtual community（推荐）
  2. Simulating cognitive behavioural therapy for depression using LLM-based generative agents
  3. Generative agents with depression modelling enable in-silico evaluation of psychotherapy interventions
- **一句话定位**: 用 LLM 驱动的生成式智能体在虚拟社区中模拟抑郁症患者的认知-情绪动态，并对认知行为疗法（CBT）进行 in-silico 治疗评估。

---

## 1. Abstract（≤150 words，无小标题，6 要素串联）

**一句话概括**: 在 RCT 受成本/伦理/时间约束、现有计算模型难以捕捉治疗对话中动态认知-情绪交互的背景下，我们构建嵌入 Beck 认知模型的生成式智能体平台 [ModelName]，通过多组对照与 PHQ-9/BDI-II/SDS 三量表纵向评估，证明结构化 CBT 显著降低抑郁评分且优于支持性咨询，记忆写入消融揭示疗效持久性的因果机制，为心理治疗机制研究提供可重复的 in-silico 范式但尚需临床队列验证。

---

## 2. Main / Introduction（~600 words，4 段漏斗）

- **Para 1 — Field stake（领域重要性）**: 抑郁症全球负担沉重（约 2.8 亿患者），CBT 是一线推荐方案但其响应率仅 40–60% 且疗效异质性大、机制不明。
- **Para 2 — Bottleneck（现有瓶颈）**: RCT 受成本（单中心 >50 万美元）、伦理（不能随机分配有害社交）与时间（12–24 月）约束，机制研究依赖事后回溯、时序分辨率不足。
- **Para 3 — Prior work → Gap（前人工作与空白）**: 生成式智能体框架（Park et al. 2023）与心理健康应用概念框架（Kambeitz & Meyer-Lindenberg 2025）已提出，但尚无研究在生成式智能体中实现临床级抑郁建模并实证评估结构化心理治疗。
- **Para 4 — This study（本研究）**: 我们提出 [ModelName]，将 Beck 认知模型嵌入 Stanford Town 框架，以五模块抑郁认知架构 + 四阶段 CBT + 多组对照 + 三量表纵向评估，对心理治疗进行 in-silico 机制拆解。

---

## 3. Results（~1,000 words，4 小节，claim-first）

- **2.1 系统概览与内部效度验证**: 患者智能体基线 PHQ-9 与严重度配置一致，高神经质人设基线显著更高（复现神经质-抑郁关联），重复 run 的 ICC 显示可接受的组内一致性，对应 **Fig 1**（系统架构）。
- **2.2 CBT 治疗响应主结果**: G1 组 PHQ-9 显著下降（Cohen's d = X.XX），G2 自然病程恶化，Session 4–8 认知重构期改善最大，BDI-II/SDS 收敛，响应率与缓解率均优于 G2，对应 **Fig 2**（纵向轨迹）。
- **2.3 多组对照比较**: CBT 优于中性社交（G3）/无干预（G2）/负面社交（G5），G4≈G1 排除环境效应，G1>G6 隔离 CBT 技术特异性效应，G7 急性效应保留但 T4 反弹揭示记忆巩固对疗效持久性的因果角色，对应 **Fig 3 + Table 1**。
- **2.4 消融分析**: 逐级移除 CBT 结构（G6）、记忆写入（G7）、情绪推断、投诉图后治疗响应呈梯度下降，证明整合认知架构各模块的必要性，对应 **Fig 4**（可选附 NoEmotion/NoGraph 于 SI）。

---

## 4. Discussion（~600 words，4 段）

- **Para 1 — 核心发现总结**: 本平台实现生成式智能体中的临床级抑郁建模与差异化治疗响应，CBT 效应量落在真实 meta 分析区间，G1–G6 与 G7 分别隔离了 CBT 技术特异性与记忆巩固机制。
- **Para 2 — 与现有工作的关系**: 将生成式智能体从社会模拟拓展至临床心理健康研究，从 Kambeitz 概念框架走向实证，与传统 ABM（语义丰富性差异）及计算精神病学（机制 vs 交互互补）形成互补。
- **Para 3 — 局限性**: LLM 训练语料文化偏差影响跨文化泛化；人设多样性有限（青年男性为主，女性/跨年龄段待扩展）；尚未与真实 RCT 数据定量校准；LLM 随机性需多次重复控制；G7 为极端消融、无直接临床对应。
- **Para 4 — 展望**: 扩展人设与人口学多样性、引入药物联合建模、与临床队列定量校准、探索个性化治疗方案优化（precision psychiatry in silico）。

---

## 5. Methods（~500 words 正文精简版；详版见 Supplementary Information）

- **4.1 系统架构**: 基于 Stanford Town 生成式智能体框架扩展，配备本地 LlamaIndex（BGE-M3）+ 外部 EC-Doll 双记忆系统，与 Qwen3-8B（认知）/ DeepSeek（治疗与评分）分工的 LLM 后端。
- **4.2 抑郁认知架构**: 由投诉图状态机（LLM 驱动阶段转移）+ 动态情绪推断（波动限幅 0.12）+ 会话上下文构建 + 创伤记忆系统 + 四层动态 prompt 组装（base→stage→context→emotion）五模块协同驱动。
- **4.3 治疗干预模块**: 四阶段 10-session CBT + 自适应循环 + 对话法官（实时终止/策略建议）+ 会话评估 + 咨询历史检索（gate→top-k→摘要）+ 环境任务模型（作业模拟写入记忆）+ 治疗后随访模式。
- **4.4 评估体系**: PHQ-9/BDI-II/SDS 三量表在 T0/S4/S8/T4 四点施测；**冻结 checkpoint 后延迟复评**（量表不实时作答、避免改变轨迹）；ExpertLLM 独立评分并做回答-评分一致性校验。
- **4.5 实验设计**: 多组对照 × 严重度 × 重复的析因设计；严格区分独立 run / 冻结 checkpoint / checkpoint 内复评三层统计单位；采用线性混合效应模型（LMM，随机截距纳入人设与重复）+ Cohen's d + Bonferroni 校正。

---

## 6. 展示项规划（≤6）

| # | 类型 | 一句话内容 | 对应章节 |
|---|------|-----------|----------|
| Fig 1 | 架构图 | 系统五层总览：虚拟社区→认知架构→抑郁引擎→治疗管线→评估与实验设计 | Results 2.1 |
| Fig 2 | 折线图 | 各组 PHQ-9 纵向轨迹（T0→S4→S8→T4）含 95% CI 带 | Results 2.2 |
| Fig 3 | 箱线图 | 终点 PHQ-9 分布 + 两两 Cohen's d（高亮 G1 vs G6 / G1 vs G7） | Results 2.3 |
| Fig 4 | 条形图 | 消融实验：G1 vs G6（移除 CBT 结构）vs G7（移除记忆写入），可选 NoEmotion/NoGraph | Results 2.4 |
| Fig 5 | 热力图 | 投诉图状态转移矩阵（治疗前 vs 治疗后） | Results 2.2 / SI |
| Table 1 | 表格 | 各组基线特征与终点指标（demographics + PHQ-9/BDI-II/SDS + T4 随访）汇总 | Results 2.3 |

---

## 7. 定稿前必须解决的"待调研"项（来自最新资料大纲的审慎提醒）

> 以下条目反映代码支持与实际报告样本之间的差距，定稿前须逐项核验。

- **实际报告样本**: 目标批次的 condition / KBD 人设 / 严重度 / 外层重复数须由 runtime config 与 `trial_meta.json` 定稿，不能直接把默认配置或代码支持的全部组别当作已报告样本。
- **跨严重度尚未开展**: 跨人设比较已做，但跨严重度（mild/moderate/severe）比较尚未开展——初稿**不应**写成已验证严重度间稳健性。
- **重复复评与过程指标**: 重复复评可靠性（ICC / 加权 Cohen's Kappa）与过程指标（会谈剂量、阶段推进、情绪轨迹、记忆写入）是否纳入正文仍待调研定稿，不能仅因代码支持就写入正式方法。
- **G4/G6/G7 可报告性**: 咨询室环境（G4）、支持性咨询（G6）、记忆消融（G7）的实际结果完整性须核验目标批次后才能纳入正文。
- **controller 版本**: 当前默认 legacy controller（固定顺序线性推进）；若目标批次使用 progressive 控制器，须单独说明其 Progressive D 状态追踪与子目标评估机制。
- **G9 取舍**: 代码支持 G9（积极居民聊天），定稿时决定是否纳入正文（构成 G3/G5/G9 效价三连对照）或仅留 SI。
- **G4 定义修正**: G4 为"仅加载患者与医生、不含其他居民"的咨询室环境条件，非简单的"地点限定"，须按 `data/config_counsel_room.json` 与 `README_counsel_room.md` 准确表述。

---

## 8. 与现有 `nature_paper_draft.md` 的差异与修订建议

| 维度 | 现有草稿假设 | 实际/最新状态 | 修订建议 |
|------|-------------|--------------|----------|
| 实验矩阵 | 7 组 × 3 严重度 × 9 人设完整矩阵 | 跨人设已做、跨严重度未做、多组待核验 | 正文按"已验证"范围声明，未做部分移入 Limitations/Future work |
| 评估表述 | 偏"四时间点实时评估" | 冻结 checkpoint + 延迟复评 | Methods 4.4 必须改为"仿真完成后从冻结状态恢复评估" |
| 统计单位 | 未显式区分三层单位 | 须区分 run / checkpoint / 复评 | Methods 4.5 显式声明独立 run 才是效应量单位 |
| 组别范围 | G1–G7 | 代码支持 G1–G9 | 决定 G9 纳入正文或 SI |
| G4 定义 | "环境限定" | 仅患者+医生、无居民 | 按 counsel_room 配置准确表述 |
| 消融表述 | G6/G7 + NoEmotion/NoGraph | NoEmotion/NoGraph 为可选/待做 | 标注为 supplementary 或 future work |

---

## 9. 写作执行顺序建议（8 步流程，来自 analysis 第 6.1 节）

1. **论点句**: 每节先写一句核心论点（本文已给出）。
2. **术语表**: 统一 [ModelName]、complaint graph、dialog judge、staged evaluation 等关键术语全文一致。
3. **章节架构**: 按 3000 words 配额分配（Intro ~600 / Results ~1100 / Discussion ~600 / Methods ~500 / Abstract 最后写）。
4. **段落映射**: 每段一个任务，列出段间逻辑连接。
5. **证据驱动起草**: 从数据/结果出发写，不空谈（XX 占位符待真实数据填入）。
6. **动词校准**: 按证据强度调节（demonstrates/suggests/is associated with）。
7. **去无支撑声明**: 每个 claim 都有对应图/表/统计支撑。
8. **段落流畅度检查**: 段间过渡自然，Results 无 Discussion 语法混入。
