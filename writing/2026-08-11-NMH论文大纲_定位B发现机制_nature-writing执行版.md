# Nature Mental Health 论文大纲（定位 B：发现/机制论文 · nature-writing skill 执行版）

> **生成日期**: 2026-08-11
> **Skill 执行**: nature-writing skill 已正确加载（`~/ai-skills/nature-skills/` 已 clone，manifest + always_load + 匹配片段已读取）
> **轴值**: task=manuscript · paper_type=research · section=full · language=zh-to-en · journal=nature-family（用 Nature Mental Health 实际指令：~3000 words、≤150 摘要、≤6 展示项、30–50 ref）
> **定位**: B — 发现/机制论文。核心 claim = CBT 认知重构不可替代（G1>G6）+ 记忆巩固因果维系疗效持久性（G7 T4 反弹）；平台是手段而非贡献
> **Introduction variant**: technical-challenge（围绕"治疗机制难捕捉、非特异性因素难分离"的技术瓶颈）
> **命名**: `[ModelName]` 占位符
> **状态**: outline（depth dial 第一阶段），待批准后展开完整英文 prose

---

## ⚠️ 验证检查点（投入完整 prose 前必须满足）

定位 B 的两个 load-bearing claim 依赖待核验结果。以下条件任一不满足，须回退定位 A：

| Claim | 依赖证据 | 当前状态 | 不满足时的回退 |
|-------|---------|---------|--------------|
| C-tech: G1 > G6（CBT 技术特异性） | G6 终点 PHQ-9 显著低于 G1 | **待核验** | 降级为"平台可拆解技术 vs 非特异性因素"，claim 动词改 suggest；或回退 A |
| C-mech: G7 急性保留 + T4 反弹（记忆巩固因果） | G7 T4 PHQ-9 显著高于 G1 **且 T4 为停止干预后独立随访轨迹** | **待核验** | 降级为"平台可操纵记忆写入以检验持久性"；或回退 A |

**修正（来自文件2 §3.2.3）**: C-mech 的因果论断依赖 T4 是"停止干预后的自然轨迹"。文件2 指出当前默认 T4 仍在 follow-up prompt 强制医患会谈下，**非真正停止治疗后轨迹**。若目标批次未采用"停止干预后独立 post-simulation follow-up"协议，则 G7 的"T4 反弹"反映的是"仍在强制随访下的差异"，C-mech 的"记忆巩固对疗效持久性的因果贡献"论断**不成立**，须降级为描述性差异或回退定位 A。

**建议**: 先核验 G6/G7 目标批次结果（`trial_meta.json` + batch state）+ **确认 T4 follow-up 协议是否为"停止干预后独立随访"**，再决定是否展开 B 的完整 prose。

---

## Terminology Ledger（drafting 前锁定，skill 强制）

| Canonical term | First-use definition | Variants in source | Decision |
|---|---|---|---|
| `[ModelName]` | (待命名，占位符) | "the platform", "the system" | 占位，最终命名后全文替换 |
| generative agent | LLM-powered simulacra of human behaviour | 生成式智能体 | en prose: generative agent |
| **complaint graph** | LLM-driven directed graph of depressive cognitive stages | "投诉图", "主诉图", "complaint chain" | **锁定 complaint graph** |
| CBT | cognitive behavioural therapy | 认知行为疗法 | spell out once, then CBT |
| supportive counselling | non-directive, validation-focused doctor contact (G6) | 支持性咨询 | en: supportive counselling |
| memory consolidation | writing treatment-relevant experience into memory stream | 记忆巩固, 记忆写入 | en: memory consolidation |
| cognitive restructuring | CBT technique: distortion identification + evidence examination + behavioural experiments | 认知重构 | en: cognitive restructuring |
| dialog judge | real-time LLM conversation monitor | 对话法官 | en: dialog judge |
| staged evaluation | checkpoint-frozen delayed scale assessment | 阶段性评估, 冻结复评 | 锁定 staged evaluation |
| **PHQ-9 / BDI-II** | two self-report scales (主线) | 三量表, PHQ-9/BDI-II/SDS | **修正（文件2 §2.6.2 + "不建议写入"清单）: 主线两量表，SDS 未实际启用，移入 future work 或注明未启用** |
| SDS | Zung Self-Rating Depression Scale | — | **未实际启用**，不入正文 Methods/Results，仅 future work 提及 |
| ExpertLLM | independent scoring model (DeepSeek-Chat) | 评估 LLM | ExpertLLM |

---

## Title（按 title fragment：最强结果 + scope 派生）

**主标题（机制导向，非平台导向）**:
Cognitive restructuring and memory consolidation drive simulated depression treatment response in LLM-based generative agents

**备选**:
- In silico isolation of cognitive restructuring and memory consolidation in simulated CBT for depression
- Generative agents reveal cognitive restructuring and memory consolidation as active ingredients of simulated CBT

---

## Abstract（≤150 words，6 要素，最后写）

抑郁负担重而 RCT 难分离 CBT 的特异性技术贡献与非特异性治疗因素、亦难伦理地操纵记忆巩固；我们构建嵌入 Beck 认知模型的生成式智能体平台 [ModelName]，以七组对照设计（含支持性咨询 G6 与记忆写入消融 G7 两个关键对照）对 CBT 进行 in-silico 机制分离；结构化 CBT 的 PHQ-9 降幅显著大于支持性咨询（Cohen's d = [X.XX]），表明认知重构是不可被一般性支持替代的活性成分；记忆写入消融保留急性治疗响应但 T4 随访显著反弹（[XX]% vs [XX]%），揭示记忆巩固对疗效持久性的因果贡献；该平台将心理治疗的特异性与非特异性成分在 in silico 中分离，但需临床队列验证。

---

## Introduction（~600 words，4 段，technical-challenge variant，每段一句话）

- **P1 field stake**: 抑郁症全球负担沉重，CBT 是一线推荐但响应率仅 40–60% 且其起效机制——尤其是特异性技术 vs 非特异性治疗因素的相对贡献——仍不明。
- **P2 bottleneck（技术挑战）**: RCT 无法分离 CBT 特异性技术与治疗联盟/定期关注等非特异性因素（因伦理不能设置"仅支持无技术"对照），亦不能伦理地操纵记忆巩固以检验其对疗效持久性的因果角色，机制研究依赖事后回溯、时序分辨率不足。
- **P3 prior work → gap**: 生成式智能体框架（Park 2023）与心理健康应用概念框架（Kambeitz 2025）已提出，但尚无研究在受控 in-silico 设计中分离 CBT 技术特异性或因果检验记忆巩固对治疗持久性的贡献。
- **P4 present study**: 我们提出 [ModelName]，以七组对照设计——关键为 G6（仅支持性咨询，隔离技术特异性）与 G7（记忆写入消融，因果检验记忆巩固）——对 CBT 机制进行 in-silico 分离，preview 设计逻辑而非结果数字。

---

## Results（~1,000 words，evidence ladder，claim-first，B 重心在 2.3/2.4）

- **2.1 system validation（~200 words，compact）**: 患者智能体基线 PHQ-9 与严重度配置一致，高神经质人设基线更高复现神经质-抑郁关联，重复 run 的 ICC 显示可接受的组内一致性 → **Fig 1 / EDT 1**。
- **2.2 main treatment effect（~200 words，compact）**: G1 组 PHQ-9 显著下降而 G2 自然病程恶化，Session 4–8 认知重构期改善最大，BDI-II 与 PHQ-9 收敛 → **Fig 2 / Table 1**。
- **2.3 CBT technique specificity（~350 words，重心，load-bearing）**: G1 终点 PHQ-9 显著低于 G6（Cohen's d = [X.XX]，p < [X]），且呈 G1 > G3 > G2 > G5 的梯度，G4≈G1 排除环境效应，表明认知重构等技术贡献超越治疗联盟与定期关注等非特异性因素 → **Fig 3 / Table 1**。
- **2.4 memory consolidation causality（~350 words，重心，load-bearing）**: G7 急性期 PHQ-9 与 G1 无显著差异但 T4 随访反弹率显著更高（[XX]% vs [XX]%），急性响应保留而持久性受损的分离表明记忆巩固对疗效维持有因果贡献 → **Fig 3 / Fig 4**。

---

## Discussion（~600 words，4 段，hourglass 窄→宽，每段一句话）

- **P1 核心发现 + implication**: 本平台在 in-silico 中分离了 CBT 的特异性技术贡献（G1>G6）与记忆巩固对疗效持久性的因果角色（G7 T4 反弹），两项发现落在真实 CBT meta 分析的效应量区间。
- **P2 prior work dialogue**: 将生成式智能体从 Kambeitz 的概念框架推向实证机制检验，与 ABM（语义丰富性差异）、计算精神病学（机制精度 vs 交互互补）互补。
- **P3 limitations**: LLM 训练语料文化偏差影响跨文化泛化；人设多样性有限（青年男性为主，女性/跨年龄段待扩展）；未与真实 RCT 定量校准；LLM 随机性需多次重复控制；G7 为极端消融（完全阻断记忆写入），无直接临床对应，真实记忆衰减为梯度而非二元。
- **P4 future**: 扩展人设与人口学多样性、引入药物联合建模、与临床队列定量校准、探索个性化治疗方案优化。

---

## Methods（~500 words 正文 + SI 详版，每模块 Motivation→Mechanism→Evidence/Role）

- **4.1 task + overview**: 基于 Stanford Town 生成式智能体框架扩展，本地 LlamaIndex（BGE-M3）+ 外部 EC-Doll 双记忆，Qwen3-8B（认知）/ DeepSeek（治疗与评分）分工后端。
- **4.2 抑郁认知架构（五模块各含三要素）**: complaint graph（建模抑郁认知阶段转移，Motivation=静态人设难呈现动态抑郁；Mechanism=LLM 驱动有向图阶段转移；Evidence/Role=NoGraph 消融钩子）+ emotion inferencer（逐轮情绪推断带波动限幅）+ session context builder + trauma memory system + 四层 dynamic prompt builder（base→stage→context→emotion）。
- **4.3 治疗干预**: 四阶段 10-session CBT + 自适应循环 + dialog judge（实时终止/策略建议）+ session eval + consult history（gate→top-k→摘要）+ environment task model + post-treatment follow-up mode；**G6 supportive counselling 仅保留非指导性支持，不推进 CBT 阶段**；**G7 阻断患者 event/thought/chat 记忆写入**。
- **4.4 评估体系**: 主线使用 **PHQ-9（0–27）与 BDI-II（0–63）两量表**（SDS 未实际启用，不入正文）在 T0/S4/S8/T4 施测；**冻结 checkpoint 后延迟复评**（量表不实时作答，避免改变轨迹）；ExpertLLM 独立评分 + 回答-评分一致性校验；无结构 session 对照组用 step-interval fallback。
- **4.5 implementation + boundary**: 七组 × 严重度 × 重复析因；区分独立 run / 冻结 checkpoint / checkpoint 内复评三层统计单位；LMM（随机截距纳入人设与重复）+ Cohen's d + Bonferroni；in-silico 边界，非临床疗效声明。

---

## Claim-evidence map（B 定位核心，每条标注 status）

| Claim | Evidence | Status |
|---|---|---|
| 基线 PHQ-9 与严重度一致 | staged_eval T0 | needs evidence（跨严重度未做，仅跨人设） |
| 高神经质 → 更高基线（复现关联） | T0 + 人设神经质标注 | needs evidence（标注待核） |
| G1 降分显著 vs G2 | T0→T4 LMM | needs evidence（数值待核） |
| BDI-II 与 PHQ-9 收敛 | 两量表 T0→T4 | needs evidence |
| **C-tech: G1 > G6（技术特异性）** | 终点 PHQ-9 LMM | **needs evidence（G6 待核验）· LOAD-BEARING** |
| G1 > G3 > G2 > G5 梯度 | 终点 PHQ-9 | needs evidence |
| G4 ≈ G1（排除环境） | 终点 PHQ-9 | needs evidence（G4 待核验） |
| **C-mech: G7 急性保留 + T4 反弹** | T4 PHQ-9 / 反弹率 | **needs evidence（G7 待核验）· LOAD-BEARING** |
| 消融梯度下降（结构/记忆/情绪/图） | ΔPHQ-9 | needs evidence（NoEmotion/NoGraph 待做） |
| 效应量落真实 meta 区间 | 与 Cuijpers/Hofmann meta 对比 | inferred（需真实数值） |

---

## 展示项规划（≤6，对齐 B 重心）

| # | 类型 | 一句话内容 | 对应 claim |
|---|------|-----------|-----------|
| Fig 1 | 架构图 | 系统五层总览：虚拟社区→认知架构→抑郁引擎→治疗管线→评估与实验设计（B 定位下为方法学背景，非主贡献） | 2.1 |
| Fig 2 | 折线图 | 各组 PHQ-9 纵向轨迹（T0→S4→S8→T4）含 95% CI 带 | 2.2 |
| Fig 3 | 箱线图 | 终点 + T4 PHQ-9 分布 + 两两 Cohen's d（**高亮 G1 vs G6 与 G7 T4 反弹**） | **2.3, 2.4** |
| Fig 4 | 条形图 | 消融：G1 vs G6（移除 CBT 结构）vs G7（移除记忆写入），可选 NoEmotion/NoGraph | 2.4 |
| Fig 5 | 热力图 | complaint graph 状态转移矩阵（治疗前 vs 治疗后） | 2.2 / SI |
| Table 1 | 表格 | 各组基线特征与终点指标（demographics + PHQ-9/BDI-II + T4 随访）汇总 | 2.2, 2.3 |

---

## 与 A 定位的关键结构差异（备查）

| 维度 | A（平台，未采用） | **B（发现/机制，已采用）** |
|---|---|---|
| Title | 强调 platform/simulation | **强调 cognitive restructuring + memory consolidation** |
| Results 重心 | 2.1（~350词） | **2.3/2.4（各~350词）** |
| Fig 1 角色 | 主贡献展示项 | 方法学背景 |
| claim 动词 | 2.1 show；2.3/2.4 suggest | **2.3/2.4 须 show/demonstrate（需强证据）** |
| Discussion P1 | "平台实现临床级建模" | **"分离了技术特异性与记忆巩固机制"** |
| 风险 | 低 | **高（G6/G7 待核验）** |
