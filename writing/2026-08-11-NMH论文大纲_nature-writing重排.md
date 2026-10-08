# Nature Mental Health 论文大纲（nature-writing skill 重排版）

> **生成日期**: 2026-08-11
> **Skill 执行状态**: `nature-writing` agent 定义（`.claude/agents/nature-writing.md`）已读取；其指向的 governing workflow 文件 `~/ai-skills/nature-skills/skills/nature-writing/SKILL.md` 在本机缺失（`ai-skills` 目录未 clone）。以 `writing/NMH_depression_paper_analysis.md`（自述"基于 Nature-writing 写作技能框架整理而成"）作为等效 governing workflow。
> **严格遵循的 skill 规范**: ① 漏斗式 Introduction ② claim-first Results ③ 6 要素 Abstract ④ 方法学 Motivation→Mechanism→Evidence/Role ⑤ 动词校准 ⑥ **每个 claim 必须映射到 evidence**（本版的增量：Evidence Map + Claim 强度校准表）。
> **对齐原则**: 区分"已做 / 未做 / 待核验"，不把代码支持等同于已报告样本。

---

## 标题

- **主标题**: LLM-powered generative agents simulate depression treatment trajectories in a virtual community
- **一句话**: 用 LLM 驱动的生成式智能体在虚拟社区中模拟抑郁症患者的认知-情绪动态，并对 CBT 进行 in-silico 治疗评估。

---

## Abstract（≤150 words，无小标题，6 要素，一句话串联）

抑郁负担重而心理治疗 RCT 受成本/伦理/时间约束、现有计算模型难以捕捉治疗对话的动态认知-情绪交互，我们构建嵌入 Beck 认知模型的生成式智能体平台 [ModelName]，通过多组对照与 PHQ-9/BDI-II/SDS 三量表纵向评估，证明结构化 CBT 显著降低抑郁评分且优于支持性咨询，记忆写入消融揭示疗效持久性的因果机制，为心理治疗机制研究提供可重复 in-silico 范式但尚需临床队列验证。

---

## Introduction（~600 words，4 段漏斗，每段一句话）

- **P1 Field stake**: 抑郁症全球负担沉重（约 2.8 亿患者），CBT 是一线推荐但响应率仅 40–60% 且疗效异质性大、机制不明。
- **P2 Bottleneck**: RCT 受成本（>50 万美元）、伦理（不能随机分配有害社交）、时间（12–24 月）约束，机制研究依赖事后回溯、时序分辨率不足。
- **P3 Prior work → Gap**: 生成式智能体（Park 2023）与心理健康概念框架（Kambeitz 2025）已提出，但尚无研究在生成式智能体中实现临床级抑郁建模并实证评估结构化心理治疗。
- **P4 This study**: 提出 [ModelName]，将 Beck 认知模型嵌入 Stanford Town 框架，以五模块抑郁认知架构 + 四阶段 CBT + 多组对照 + 三量表纵向评估，对心理治疗进行 in-silico 机制拆解。

---

## Results（~1,000 words，4 小节，claim-first，每节一句话 + evidence 指向）

- **2.1 系统概览与内部效度**: 患者智能体基线 PHQ-9 与严重度配置一致且高神经质人设基线更高（复现神经质-抑郁关联），重复 run 的 ICC 显示可接受的组内一致性 → **Fig 1 / Extended Data Table 1**。
- **2.2 CBT 治疗响应主结果**: G1 组 PHQ-9 显著下降而 G2 自然病程恶化，Session 4–8 认知重构期改善最大，BDI-II/SDS 收敛 → **Fig 2 / Table 1**。
- **2.3 多组对照比较**: CBT 优于中性社交/无干预/负面社交，G4≈G1 排除环境效应，G1>G6 隔离 CBT 技术特异性，G7 急性效应保留但 T4 反弹揭示记忆巩固对疗效持久性的因果角色 → **Fig 3 / Table 1**。
- **2.4 消融分析**: 逐级移除 CBT 结构（G6）、记忆写入（G7）、情绪推断、投诉图后治疗响应呈梯度下降，证明整合认知架构各模块的必要性 → **Fig 4**（NoEmotion/NoGraph 可选入 SI）。

---

## Discussion（~600 words，4 段，每段一句话）

- **P1 核心发现**: 本平台实现生成式智能体中的临床级抑郁建模与差异化治疗响应，CBT 效应量落在真实 meta 分析区间，G1–G6 与 G7 分别隔离了 CBT 技术特异性与记忆巩固机制。
- **P2 与现有工作关系**: 将生成式智能体从社会模拟拓展至临床心理健康研究，从 Kambeitz 概念框架走向实证，与 ABM（语义丰富性差异）及计算精神病学（机制 vs 交互互补）形成互补。
- **P3 局限性**: LLM 训练语料文化偏差影响跨文化泛化；人设多样性有限（青年男性为主）；尚未与真实 RCT 数据定量校准；LLM 随机性需多次重复控制；G7 为极端消融、无直接临床对应。
- **P4 展望**: 扩展人设与人口学多样性、引入药物联合建模、与临床队列定量校准、探索个性化治疗方案优化。

---

## Methods（~500 words 正文 + SI 详版，5 小节，每节一句话，Motivation→Mechanism→Evidence/Role）

- **4.1 系统架构**: 基于 Stanford Town 生成式智能体框架扩展，配备本地 LlamaIndex（BGE-M3）+ 外部 EC-Doll 双记忆系统与 Qwen3-8B（认知）/ DeepSeek（治疗与评分）分工的 LLM 后端。
- **4.2 抑郁认知架构**: 由投诉图状态机（LLM 驱动阶段转移）+ 动态情绪推断（波动限幅 0.12）+ 会话上下文构建 + 创伤记忆系统 + 四层动态 prompt 组装（base→stage→context→emotion）五模块协同驱动。
- **4.3 治疗干预模块**: 四阶段 10-session CBT + 自适应循环 + 对话法官（实时终止/策略建议）+ 会话评估 + 咨询历史检索 + 环境任务模型 + 治疗后随访模式。
- **4.4 评估体系**: PHQ-9/BDI-II/SDS 三量表在 T0/S4/S8/T4 四点施测；**冻结 checkpoint 后延迟复评**（量表不实时作答、避免改变轨迹）；ExpertLLM 独立评分并做回答-评分一致性校验。
- **4.5 实验设计**: 多组对照 × 严重度 × 重复的析因设计；严格区分独立 run / 冻结 checkpoint / checkpoint 内复评三层统计单位；LMM（随机截距纳入人设与重复）+ Cohen's d + Bonferroni 校正。

---

## Evidence Map（claim → 展示项 → 数据源 → 状态）— skill 规范核心增量

| # | Claim | 展示项 | 数据源 | 代码/样本状态 |
|---|-------|--------|--------|--------------|
| C1 | 基线 PHQ-9 与严重度配置一致 | Fig 1 / EDT 1 | `staged_eval` T0 | 跨人设已做；**跨严重度未做** |
| C2 | 高神经质人设 → 更高基线（复现神经质-抑郁关联） | EDT 1 | `staged_eval` T0 | 待核验人设神经质标注是否区分 |
| C3 | G1 ΔPHQ-9 显著下降 vs G2 | Fig 2 / Table 1 | `staged_eval` T0→T4 | 待核验目标批次 G1/G2 完整性 |
| C4 | Session 4–8（认知重构期）改善最大 | Fig 2 | `staged_eval` S4/S8 | 依赖 session 计数定义（legacy/progressive） |
| C5 | BDI-II/SDS 与 PHQ-9 收敛 | Table 1 | `staged_eval` 三量表 | 待核验三量表是否齐备 |
| C6 | G1 > G6（CBT 技术特异性） | Fig 3 | 终点 PHQ-9 | **G6 结果待核验** |
| C7 | G4 ≈ G1（排除环境效应） | Fig 3 | 终点 PHQ-9 | **G4 结果待核验**（G4=仅患者+医生） |
| C8 | G7 急性效应保留、T4 反弹（记忆巩固因果） | Fig 3 | T4 PHQ-9 | **G7 结果待核验** |
| C9 | 消融梯度下降（结构/记忆/情绪/图） | Fig 4 | ΔPHQ-9 | G6/G7 待核验；NoEmotion/NoGraph 可选/待做 |
| C10 | 投诉图状态转移随治疗改变 | Fig 5 | `complaint_graph` 日志 | 过程指标，**纳入正文待定稿** |
| C11 | 组内一致性可接受（ICC） | SI | 重复 run | 重复复评可靠性待定稿 |

---

## Claim 强度与动词校准表（skill 规范：evidence-first hedging）

| Claim 类别 | 证据强度 | 推荐动词 | 避免使用 |
|-----------|---------|---------|---------|
| 基线一致 / 神经质关联（C1,C2） | 观察性 | is consistent with, suggests | proves, confirms |
| 主治疗效应 G1 vs G2（C3） | LMM 统计支持 | showed, achieved, produced | proved |
| 优于支持性咨询 G1 vs G6（C6） | 组间比较 | outperformed, exceeded | was better（无量化） |
| 记忆巩固因果（C8） | 消融证据但极端操纵 | suggests a causal role | demonstrates causally |
| 消融梯度（C9） | 组间比较 | reduced, attenuated | completely abolished |
| 整体定位 | — | to our knowledge, the first to … in the context of … | first / novel（无范围限定） |

---

## 展示项规划（≤6）

| # | 类型 | 一句话内容 | 对应 claim |
|---|------|-----------|-----------|
| Fig 1 | 架构图 | 系统五层总览：虚拟社区→认知架构→抑郁引擎→治疗管线→评估与实验设计 | C1 |
| Fig 2 | 折线图 | 各组 PHQ-9 纵向轨迹（T0→S4→S8→T4）含 95% CI 带 | C3, C4, C5 |
| Fig 3 | 箱线图 | 终点 PHQ-9 分布 + 两两 Cohen's d（高亮 G1 vs G6 / G1 vs G7） | C6, C7, C8 |
| Fig 4 | 条形图 | 消融实验：G1 vs G6 vs G7，可选 NoEmotion/NoGraph | C9 |
| Fig 5 | 热力图 | 投诉图状态转移矩阵（治疗前 vs 治疗后） | C10 |
| Table 1 | 表格 | 各组基线特征与终点指标（demographics + PHQ-9/BDI-II/SDS + T4）汇总 | C3, C5 |

---

## 已做 / 未做 / 待核验 对齐表（对齐代码实际进度）

| 类别 | 条目 | 处理 |
|------|------|------|
| **已做** | 跨人设比较（KBD1–9） | 可入正文 |
| **已做** | G1 结构化 CBT 流程 + 对话法官 + 会话评估 | 可入 Methods |
| **已做** | 冻结 checkpoint + 延迟复评机制 | 可入 Methods 4.4 |
| **未做** | 跨严重度（mild/moderate/severe）比较 | **不可声称已验证严重度稳健性**；移入 Limitations/Future |
| **未做** | NoEmotion / NoGraph 消融 | 标注 supplementary 或 future work |
| **待核验** | G2/G4/G6/G7/G9 实际结果完整性 | 核验 `trial_meta.json` 与 batch state 后再纳入正文 |
| **待核验** | 重复复评可靠性（ICC / 加权 Kappa） | 决定是否入正文或仅 SI |
| **待核验** | 过程指标（会谈剂量/阶段推进/情绪轨迹）是否入正文 | 调研定稿 |
| **待核验** | controller 版本（legacy vs progressive） | 若用 progressive 须单独说明其状态追踪 |
| **待核验** | G9（积极社交）取舍 | 纳入正文（G3/G5/G9 效价三连）或仅 SI |

---

## 与上一版（`2026-08-11-NMH论文大纲_Nature格式整合.md`）的增量

1. **Evidence Map**：每个 claim 显式映射到展示项、数据源、代码/样本状态（skill 规范"Claims require evidence"的直接落地）。
2. **Claim 强度与动词校准表**：按证据强度分级标注推荐动词与禁用词。
3. **已做/未做/待核验对齐表**：把"待调研"项结构化为表格，明确哪些可入正文、哪些须降级处理。
4. Methods 每节补标 Motivation→Mechanism→Evidence/Role 三要素的落点。
