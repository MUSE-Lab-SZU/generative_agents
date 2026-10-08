# Nature 子刊 Agent 论文全景调研（NMH 之外的 6 刊）

> **日期**: 2026-09-25
> **任务**: 按前一份 NMH 调研的同一方法（全库枚举 + Crossref 逐条核实 + nature.com 全文架构抄录 + 甄别排除），覆盖 venue 地图上的全部 6 本子刊
> **调研规模**: 4 个并行 agent，枚举 npj MHR 全刊 241 篇 / npj Digit Med 全刊 3,222 篇 / NMI 与 Nat Med 28 组关键词 / NHB 与 HSSC 78 组查询；筛出核心论文约 100 篇，全部元数据经 Crossref 或 nature.com 页面核实
> **证据目录**: `D:\CarrieOu\04-Experiments\tmp_npjmhr\`、`tmp_npjdm\`、`tmp_nmi_nmagent\`、`tmp_nhb_hssc\`（含全文 HTML 快照与解析脚本，可复查）
> **配套**: 前一份报告 `2026-09-24_NMH智能体论文调研与章节架构修改建议.md`（本报告 §7 有对它的勘误）

---

## 0. 三条关键修正 + 一个决定性发现

**修正 1（引用勘误，立即执行）**：AMIE 原始论文 **Tu et al. 2025 "Towards conversational diagnostic artificial intelligence" 发表在 Nature 正刊（Nature 642:442–450, DOI 10.1038/s41586-025-08866-7），不是 Nature Medicine**。Nature Medicine 上只有其后续（多模态 AMIE, Saab et al. 2026, NM 32:1726–1736）与平行工作。此前报告 §8 与记忆中"AMIE 在 Nature Medicine"的说法已勘误。

**修正 2**：npj Mental Health Research **创刊于 2022 年 12 月**（非 2024），卷 1–5；**2026 年获得首个影响因子 9.1，进入精神病学 JIF 前 10**，年下载超 100 万次（Tsai 2026 编辑部文章披露）。

**修正 3（引用格式）**：Humanities & Social Sciences Communications 的 DOI 前缀是 **10.1057**（Palgrave 系，如 10.1057/s41599-024-03611-3），不是 10.1038——写 bib 时注意。

**决定性发现**：**Yosef et al. 2025（npj Mental Health Research 4:43, 10.1038/s44184-025-00159-1）是整个 Nature Portfolio 中与我们方法学最同构的先例**——用 LLM 驱动的"数字患者"（digital patients）与 LLM 治疗师进行文本心理治疗会话、填写人用问卷（满意度+工作同盟）作为疗效端点，**全部数据为 AI 生成**，以完整 Article 发表。它证明了"纯合成数据 + in-silico 治疗评估 + 人用量表端点"这个体裁在 Nature Portfolio 有正式通道。其局限（单会话、横断、无诊断异质性、无纵向轨迹）恰好是我们的增量空间。

---

## 1. 总览：6 刊 agent 论文密度与体裁

| 期刊 | agent 论文密度 | 主体体裁 | 代表作 | Park 式"智能体仿真"Article | 对本文适配度 |
|---|---|---|---|---|---|
| **npj Mental Health Research** | 全刊 61/241 篇涉 AI；LLM 核心 23 篇（14 篇实证 Article） | 真实用户/临床数据为主，**已收纯合成数据研究** | **Yosef 2025 数字患者**；Maples 2024 Replika；Stade 2024 | 无（空白） | ★★★ 有直接先例 |
| **npj Digital Medicine** | 高相关 10 篇 + 中相关 20 篇；2025–2026 加速 | **agent 系统实证 Article 为主体** | Luo 2025 LLMDP 虚拟患者+RCT；Chen 2025 MAC 多智能体；Makarov 2025 DT-GPT；WiseMind 2026 | 无（Kambeitz 宣言只是 Perspective） | ★★★ 体裁之家，但要求真实验证锚点 |
| **Nature 正刊 + Nature Medicine** | 少量旗舰级 | 随机盲法临床评估 | **AMIE（Nature 2025）**；Rollwage 2026 心理治疗认知层 | 无 | ✗ 不投，但 AMIE/Rollwage 是必引对标 |
| **Nature Machine Intelligence** | 方法高地，12 篇核心 | "能力宣言"式系统论文 | ChemCrow 2024；Serapio-García 2025 LLM 人格塑造 | 无 | △ 仅当重写为方法论文 |
| **Nature Human Behaviour** | 18 篇 LLM-人类对照实证 | LLM 当被试与真实数据对照 | Strachan 2024 ToM；Luo 2024 BrainBench；**Goldenberg 2026 反方** | 无 | △ 需强真实数据锚点 |
| **HSSC** | 10 篇 LLM 仿真（传统 ABM 另有大量） | 全 IMRaD 实证 | Qu & Wang 2024 硅基样本；Ji 2026 CiteAgent | 无（仅一般社会场景） | △ 体裁匹配但社科定位 |

**跨刊最重要的一条空白结论**：截至 2026-09，**六刊均未发表 Park 式"生成式智能体虚拟社区仿真临床人群 + in-silico 对照治疗试验"的 research Article**。最近的三块拼图分别在 npj MHR（Yosef：数字患者×单会话）、npj Digit Med（Luo：虚拟患者×RCT 评估训练效果；Si：模拟患者实验法）和我们手里（智能体社区×纵向七组对照×量表轨迹）。三条路径都有"体裁空白+邻近先例"的格局。

---

## 2. 分刊论文列表

### 2.1 npj Mental Health Research（路径 C；23 篇核心，14 篇实证）

**重点 4 篇（全文架构已抄录）**：

| # | 论文 | 元数据 | 对本文的意义 |
|---|---|---|---|
| 1 | **Yosef et al. 2025**, "The impact of fine-tuning LLMs on the quality of automated therapy assessed by digital patients" | 4:43, 10.1038/s44184-025-00159-1, Article | **直接先例**：LLM 数字患者评测 LLM 治疗师（动机式访谈），纯合成数据。架构：Intro 14 段 → Methods（**前置**）→ Results 3 小节 → Discussion 12 段无小节；11 图 6 表 + SI；含"全合成数据"的显式伦理论证段 |
| 2 | **Maples et al. 2024**, "Loneliness and suicide mitigation for students using GPT3-enabled chatbots" | 3:4, 10.1038/s44184-023-00047-6, Article | Replika 1006 名学生调查；90% 孤独、30 人自述阻止自杀企图。**注意其 Matters Arising 事件**（Zimmerman 质疑产业利益——该刊批判文化活跃，安全框架必须前置）。架构：Background 8 段 → Methods 前置 → Results 4 小节 → Discussion 有 h3 小节；2 图 |
| 3 | **Siddals et al. 2024**, "It happened to be the perfect thing": experiences of generative AI chatbots for mental health | 3:48, 10.1038/s44184-024-00097-4, Article | 19 名用户质性访谈；"安全护栏反而破坏信任"。架构：Intro 6 段 → Results 6 小节 → Discussion 14 段 → **Methods 殿后**；含 Reflexivity statement |
| 4 | **Stade et al. 2024**, "LLMs could change the future of behavioral healthcare" | 3:12, 10.1038/s44184-024-00056-z, Article（提案体裁）, 被引 356 | "辅助→协作→自主"三阶段框架——我们可定位为"协作级评估基建"。主题式平级章节，3 图 2 表 |

其余实证 10 篇（Mahbub SUD 笔记解码、Xu 门诊对话识别、Elyoseph WHO 指南评审、Sadeghi 多模态抑郁检测、Ryan 医生决策实验、PsyEval 基准、Verhees 提示工程、Mwangi 模拟临床视频 MSE、Lokadjaja 言语检测、Steinbrenner 论坛 NLP）+ 综述 1 + 短评 2 + Matters Arising 往来 4。

**该刊架构惯例**：单段无结构摘要；**Methods 位置自由**（14 篇中 7 篇 Methods 前置、2 篇殿后、1 篇合并式）；小节一律无编号 h3；Discussion 常无小节（12–14 段）；图表上限宽松（Yosef 11 图 6 表）；无 Extended Data，附加材料走 SI；接受全合成数据但要求显式伦理论证。

### 2.2 npj Digital Medicine（路径 B；高相关 10 篇 + 中相关 20 篇）

**重点 5 篇（全文架构已抄录）**：

| # | 论文 | 元数据 | 对本文的意义 |
|---|---|---|---|
| 1 | **Kambeitz & Meyer-Lindenberg 2025**（种子核实：体裁 = **Perspective**，非 Article） | 8:36, 10.1038/s41746-024-01422-z | 生成式智能体×心理健康官方议程：8 节 33 段 + 3 图 1 表（Fig.1 即 Park 式框架图；Table 1 micro/meso/macro 决定因素清单——**审稿人期待的可视化语言**）；无 Methods/Results |
| 2 | **Luo et al. 2025**, LLMDP 数字患者系统 | 8:502, 10.1038/s41746-025-01575-8, Article, 被引 41 | **"LLM 虚拟患者 + 随机对照试验"完整模板**：Results 5 小节（数据构建→系统开发→验证实验→**RCT 主结果**→态度收尾）；Methods 10 小节含 RCT 设计/随机盲法/统计；6 图 2 表 |
| 3 | **Chen et al. 2025**, MAC 多智能体会诊 | 8:159, 10.1038/s41746-025-00299-6, Article, 被引 120 | **Results 与 Methods 小节一一镜像**（9 对 15 小节）——该刊惯例的极端形态；8 图 7 表 |
| 4 | **Makarov et al. 2025**, DT-GPT | 8:588, 10.1038/s41746-025-01562-z, Article, 被引 48 | LLM 预测患者健康轨迹的"数字孪生"：Results 小节标题全是发现句（"DT-GPT achieved state-of-the-art…"）；Methods 10 小节流水线 |
| 5 | **Si et al. 2025**, 模拟患者实验法 | 8:574, 10.1038/s41746-025-01555-z, Article | 384 次"患者–AI"标准化问诊实验评质量/安全/公平——**与我们 in-silico 评估设计同构** |

另有：WiseMind 2026（9:575，精神科双智能体，当前收稿热点）、CARE-AD 2025（纵向 EHR 多智能体）、Mehandru 2024（Comment，**ABM + AI-SCE 评估框架**——in-silico 评估的话语工具）、Wedlund 2021（Comment，in-silico 试验合法性）、Goodell 2025（工具调用智能体）、以及中相关 20 篇（数字孪生 RCT、CBT 型对话智能体荟萃 Hang 2026、虚拟精神科问诊 Philip 2020、PHQ-9×虚拟人 Marin-Morales 2026 等）。

**该刊架构惯例**：固定 Intro → Results → Discussion → **Methods 殿后**；Results 3–9 个 H3 且与 Methods 镜像；Discussion 纯段落 8–16 段；Methods 重头 10–15 小节；正文 5,000–7,500 词；5–10 图 + 2–7 表；Code+Data availability 双声明标配。**风险**：同类文全部带真实验证锚点（RCT/临床医生评分/真实数据基准），纯 in-silico 需显式效度论证。

### 2.3 Nature 正刊 + Nature Medicine（对标引用库，非投稿目标）

| # | 论文 | 元数据 | 对本文的意义 |
|---|---|---|---|
| 1 | **Tu et al. 2025, AMIE**（勘误后） | **Nature** 642:442–450, 10.1038/s41586-025-08866-7, Article, OpenAlex 被引 326 | 对话诊断智能体的评估圣经：**场景包（带 ground truth 与可接受答案集）+ 随机化/反平衡/双盲 + 双视角评分（patient-actor 问卷 + 33 名专科医生三重复评）+ bootstrap/Wilcoxon + FDR**；评分 rubric 全部放 Extended Data 表。Results 组块：诊断准确率（分层）→ 信息获取效率 → 对话质量（双视角） |
| 2 | **Rollwage et al. 2026**, 心理治疗认知层 | NM 32:1717–1725, 10.1038/s41591-026-04278-w, Article | **必须对标差异化**：随机双盲，227 名真实参与者会谈 + 22 名专家按 **CTRS（认知治疗评分量表）** 等评分，LLM+认知层 > 裸 LLM > 人类治疗师；再以真实部署 19,674 份会谈验证。**他们评"AI 当治疗师"，我们评"AI 模拟患者 + in-silico 试验"**——引言须一句话划清。其 CTRS/治疗联盟量表组可作为我们评估指标库 |
| 3 | **Saab et al. 2026**, 多模态 AMIE | NM 32:1726–1736, 10.1038/s41591-026-04371-0, Article | 105 例模拟会诊、18 名专科医生、29/32 轴胜出；含 **LLM-as-a-judge 自动评估与人工评分校准小节** + 场景扰动鲁棒性小节——我们评估流程可对标 |
| 4 | Johri et al. 2025, CRAFT-MD | NM 31:77–86, 10.1038/s41591-024-03328-5, Article | 用模拟 AI 患者与被测临床 LLM 多轮对话的评估框架 |
| 5–8 | SPARK 2026（病理多智能体）、HemaGuide 2026（专家盲法+11 层消融）、Freyer 2025（监管 Comment）、Kather 2024 | NM 各卷 | Nat Med 2026 年新接纳"agent 框架"亚型；伦理/监管限定语引用点 |

### 2.4 Nature Machine Intelligence（方法高地，12 篇核心）

| # | 论文 | 元数据 | 对本文的意义 |
|---|---|---|---|
| 1 | **Serapio-García et al. 2025**, LLM 人格心理测量框架 | 7:1954–1968, 10.1038/s42256-025-01115-6, Article（DeepMind） | **对我们人设模块的直接方法学依据**：对 18 个 LLM 施测大五人格量表（含 Neuroticism）并可"塑造"目标人格剖面——支撑"虚拟抑郁患者人格设定"的合法性 |
| 2 | Bran et al. 2024, ChemCrow | 6:525–535, 10.1038/s42256-024-00832-8, Article, 被引 738 | NMI agent 论文体裁样板："Results and discussion"合并 + **风险专节（Risk-mitigation / Unintended risks 双节）** + 工具逐个小节的 Methods |
| 3 | Sharma et al. 2023, 人机共情对话 | 5:46–57, 10.1038/s42256-022-00593-2, Article, 被引 493 | 真实心理支持平台 47 万对话——引言必引 |
| 4 | Kim et al. 2026, 多智能体协作收益边界 | 8, 10.1038/s42256-026-01268-y, Article | **支撑我们"受控多臂而非自由多智能体"设计的论据** |
| 5–12 | ELLMER 具身（7:592）、ROS 框架（8:313）、Qiu 2024 医疗 agent 短综述（Topol, 6:1418）、信念启动（5, 2023）、越狱防御（5, 2023）、agentic science（7, 2025）、精神科领域适配 LLM（8, 2026）、多智能体透明性 Comment | | 背景与安全引用点 |

### 2.5 Nature Human Behaviour（18 篇精选；无 Park 式仿真实证）

**必引三篇 + 必答一篇**：

| # | 论文 | 元数据 | 对本文的意义 |
|---|---|---|---|
| 1 | **Strachan et al. 2024**, Testing theory of mind in LLMs and humans | 8:1285–1295, 10.1038/s41562-024-01882-z, Article, 被引 273 | "把 LLM 当被试做心理学实验"完整模板：**全部题目重写防训练集污染 + 似然对照实验防捷径**——与我们记忆消融的机制验证逻辑同构 |
| 2 | **Luo et al. 2024**, BrainBench | 9:305–315, 10.1038/s41562-024-02046-9, Article, 被引 142 | LLM 前瞻预测真实神经科学结果（81.4% vs 专家 63.4%）；**记忆化排查（zlib 熵+困惑度比）+ 消融**——防泄漏三件套的样板 |
| 3 | **Wright et al. 2026**, LLM 零样本人格评分 | 10:541–555, 10.1038/s41562-025-02389-x, Article | **与我们的"LLM 模拟量表作答+ExpertLLM 评分"环节最同构**的方法学先例 |
| 4 | **Goldenberg et al. 2026**, "Large language models do not have emotions" | 10.1038/s41562-026-02558-6（在线） | **必须正面回应的反方**：引言需以它为靶——我们模拟的是抑郁相关的言语/行为轨迹与量表响应，不宣称 LLM 具有情绪体验 |
| 5 | **Spens & Burgess 2024**, 记忆巩固的生成式模型 | 8:526–543, 10.1038/s41562-023-01799-z, Article | **G7 记忆消融的理论对话对象**（记忆回放驱动图式建构/失真） |
| 6 | Feuerriegel et al. 2026, LLM 行为科学报告清单 | 10:1038/s41562-026-02492-7, Comment, 81 位作者 | **写 Methods 时逐条对照的报告规范** |
| 7–18 | Akata 2025 重复博弈（9:1380）、Lu 2025 文化倾向（9:2360）、Salvi 2025 说服力 RCT（9:1645）、Johnson 2025 模拟利他、Vaccaro 2024 元分析（被引 549）、Burton 2024 集体智能 Perspective、Glickman 2024 人-AI 反馈环等 | | 背景与支撑引用 |

### 2.6 HSSC（10 篇 LLM 仿真；全 OA；DOI 前缀 10.1057）

| # | 论文 | 元数据 | 对本文的意义 |
|---|---|---|---|
| 1 | **Qu & Wang 2024**, 舆论仿真的性能与偏差 | 11:1095, 10.1057/s41599-024-03609-x, Article, 被引 73 | **"硅基样本 vs 真实调查"验证写法范文**：以 algorithmic fidelity 为框架，Cohen's Kappa + Cramér's V 分层对照 WVS 六国数据，诚实报告失效亚组（日本/南非 Kappa≈0）；每受访者仿真 100 次取分布 |
| 2 | **Ji et al. 2026**, CiteAgent 引用网络仿真 | 13:127, 10.1057/s41599-025-06193-w, Article | **"先复制已知规律、再跑反事实"两段式结构**——生成网络先复现幂律/优先连接/引用扭曲（KS 统计量对照真实 CiteSeer/Cora），通过后才做反事实实验。这是仿真论文最有说服力的论证结构，直接可借 |
| 3 | **Gao et al. 2024**, LLM-ABM 综述（清华 FIB Lab） | 11:1259, 10.1057/s41599-024-03611-3, Article, 被引 274 | LLM 智能体仿真的术语体系与四环节（画像/交互/决策/校准）——术语引用锚点 |
| 4–10 | Chen 2026 AI 行为科学（Review）、Haase 2026 多智能体六级框架（Review）、Zuo 2026 内卷 SABM、Shin 2026 句法加工、Liu 2025 政治价值、Li 2026 推荐 agent、Dehghani 2025 法律 LLM | | 背景与定位引用 |

---

## 3. 章节架构模板库（按用途取用）

| 我们的写作需求 | 最佳模板 | 关键可借点 |
|---|---|---|
| 纯合成数据的合法性论证 | **Yosef 2025**（npj MHR） | Discussion 中显式的"全合成数据"伦理论证段（隐私、真实数据稀缺、审慎结论）；Methods 前置 + Results 镜像 |
| 系统 + 对照评估整体结构 | **Luo 2025 LLMDP**（npj DM） | Results 5 小节：数据→系统→验证→主实验→态度收尾；Methods 含随机/盲法/统计专节 |
| Results/Methods 镜像 + 全发现句标题 | **Chen 2025 MAC** + **Makarov 2025 DT-GPT**（npj DM） | Results 小节标题 = "DT-GPT achieved state-of-the-art…"式发现句 |
| 盲法评估三件套 | **AMIE / Tu 2025**（Nature） | 场景包 ground truth + 随机反平衡双盲 + 双视角评分（参与者问卷 + 专家盲评）+ FDR 统计 + rubric 放 Extended Data |
| 自动评分与人工校准 | **Saab 2026**（NM） | "LLM-as-a-judge"小节 + 场景扰动鲁棒性小节 |
| "复制已知事实→反事实"两段式 | **Ji 2026**（HSSC） | 效度检查=复现公认 stylized facts，通过后才做干预实验——我们可表述为"先复现自然病程/神经质关联等已知规律，再做七组对照" |
| 保真度统计写法 | **Qu & Wang 2024**（HSSC） | algorithmic fidelity 框架 + Kappa 分层 + 重复采样分布 + 诚实报告失效亚组 |
| 防泄漏/防捷径 | **Strachan 2024 + Luo 2024**（NHB） | 新颖题目改写、记忆化统计、消融排除捷径——G7 记忆消融可直接包装为机制验证 |
| 人设模块方法学依据 | **Serapio-García 2025**（NMI） | LLM 人格可测量可塑造（含 Neuroticism） |
| 框架图表语言 | **Kambeitz 2025**（npj DM Perspective） | Fig.1 Park 式框架图 + micro/meso/macro Table 1 清单 |

---

## 4. "仿真 vs 真实数据"验证写法九条惯例（跨刊汇总，对本文最有价值）

1. **给效度一个理论名号**：Qu & Wang 用 algorithmic fidelity；我们可定义 clinical/construct fidelity（仿真量表轨迹与真实抑郁人群轨迹的一致性）
2. **一致性统计量分层报告**：效应量对照真实荟萃分析（d 值区间）+ 轨迹分布一致性 + 重复 run 的 ICC/一致性区间
3. **重复采样处理随机性**：Qu & Wang 每受访者 100 次；我们每个条件多 run 报告分布（已在 A–L 调整 D 中，保持）
4. **"先复制已知、再反事实"两段式**（Ji）：先复现自然病程/神经质-抑郁关联，再上七组对照
5. **防泄漏/防捷径三件套**（Strachan/Luo）：新颖人设叙事、记忆化排查、消融即机制验证（G7 正好是）
6. **诚实报告失效边界**：Qu & Wang 公开报告失效亚组——我们的 Limitations 应写明哪些人设/严重度未验证
7. **校准与置信度**（Luo）：不止报均值，报"预测置信与实际的一致性"——量表复评的可信度分析可借
8. **人类/真实数据锚点是高刊门槛**：NHB/npj DM 全部带真实锚点；我们的真实锚点 = CBT 荟萃分析效应量对照（P0-2 的依据从"建议"升级为"必需"）
9. **报告规范合规**：Feuerriegel 2026 清单逐条对照 Methods；AMIE 式 rubric 入 Extended Data；纯仿真声明"不涉及人类被试"（Ji 先例）

---

## 5. 各刊 Article 架构惯例对照

| 维度 | NMH（前报告） | npj MHR | npj Digit Med | NHB |
|---|---|---|---|---|
| 章节顺序 | 无标题引言→Results→Discussion→（Conclusions）→**Methods 殿后** | **Methods 位置自由**（多数派前置） | Intro→Results→Discussion→Methods 殿后 | Main→Results→Discussion→Methods 殿后 |
| Results 小节 | 发现句式为主流 | 3–6 个 h3 | 3–9 个 h3，**与 Methods 镜像** | 3–4 个发现句式 |
| Discussion 小节 | 不允许 | 常无小节 | 纯段落 | 纯段落 |
| 图表上限 | 主项≤6 + ED≤10 | **宽松（11 图 6 表实例）** | 5–10 图 + 2–7 表 | ~5 图 |
| 字数 | 正文≤3,000（弹性） | 宽松 | 5,000–7,500 | ~4,000 |
| Extended Data | 有（≤10） | 无（走 SI） | 无 | 有 |
| 纯合成数据 | 无先例 | **有先例（Yosef）** | 有邻近（Luo/Si，但需真实锚点） | 无先例 |

---

## 6. 对本文的直接行动项

### 6.1 新增必引清单（按论证功能，供扩文献 P2-7 使用）

- **先例与合法性**：Yosef 2025（npj MHR，纯合成 in-silico 治疗评估先例）；Kambeitz & Meyer-Lindenberg 2025（npj DM，生成式智能体宣言）；Gao 2024（HSSC，LLM-ABM 综述）
- **评估方法对标**：AMIE/Tu 2025（**注意是 Nature 正刊**）；Saab 2026；CRAFT-MD/Johri 2025；Qu & Wang 2024（保真度统计）；Ji 2026（复制-反事实结构）
- **人设/量表方法学**：Serapio-García 2025（人格塑造）；Wright 2026（LLM 量表评分）
- **差异化对标（引言划界）**：Rollwage 2026（AI 当治疗师 vs 我们模拟患者做 in-silico 试验）
- **必须回应**：Goldenberg 2026（LLM 无情绪——我们不宣称 LLM 有情绪体验，模拟的是外显轨迹）
- **机制对话**：Spens & Burgess 2024（记忆巩固生成式模型，G7 对话对象）
- **报告规范**：Feuerriegel 2026（LLM 行为科学清单）；统计组合借 AMIE（bootstrap/Wilcoxon + FDR）
- **背景**：Sharma 2023、Maples 2024、Siddals 2024、Stade 2024、Kim 2026、Hua 2025、Hang 2026（CBT 对话智能体荟萃）

### 6.2 写作结构升级（叠加在前报告 P0 之上）

1. **引言新增"划界句"**：一句话区分我们与 Rollwage（AI-as-therapist）和 Yosef（单会话数字患者测治疗师）——我们是"智能体社区纵向 in-silico 对照试验"
2. **效度段落理论化**：2.1 小节开头定义 clinical fidelity 框架（借 algorithmic fidelity 话语），把"五维真实性+量表基线+神经质关联"统一为其下的证据
3. **"复制已知→反事实"结构显式化**：2.1（复现已知规律）→ 2.2–2.4（干预反事实）的递进在段落衔接中点明
4. **统计口径升级**：bootstrap/Wilcoxon + FDR 校正替代/并列 Bonferroni（AMIE 口径）；评分 rubric 移入 Extended Data
5. **伦理论证段**（Yosef 先例）：Discussion 增加显式段落论证"为何纯 in-silico 先行"（记忆消融不能在真人身上做 + 七组对照伦理不可行）——这同时是投稿任何一刊的护城河

### 6.3 投稿路径更新评估（证据增强，不替代决策）

| 路径 | 本轮新增证据 | 评估变化 |
|---|---|---|
| A: NMH | 仍无先例；但 Yosef 在姊妹刊 npj MHR 的存在让"Nature Portfolio 接受此体裁"有了实证 | 维持——最高声望+临床读者，风险不变 |
| B: npj Digital Medicine | Luo LLMDP/Si/WiseMind 证明 agent 系统是收稿热点；但**全部带真实验证锚点**，纯 in-silico 需补效度论证 | 微降——体裁之家但锚点要求最高 |
| C: npj Mental Health Research | **Yosef 直接先例 + 体裁空白（无纵向、无社区仿真）+ IF 9.1 + Methods 自由 + 图表宽松 + 心理健康读者** | **显著升温——体裁匹配度当前最优** |

建议向导师汇报时把 C 列为与 A 并列的候选：A 赌"首创+临床读者"（配合 P0 改造），C 赌"有先例的稳妥通道"（改造压力更小）。若走 C，前报告的 P0-1/P0-2 仍然适用（发现句标题+外部锚定），但 Methods 位置、图表量、字数的自由度都更大。

---

## 7. 对前一份报告的勘误

1. §8 venue 表中 "Nature Medicine（AMIE Tu et al. 2025…）" → 更正为 "Nature 正刊 642:442–450"；Nature Medicine 条目改为多模态 AMIE（Saab 2026）等
2. §8 提到 HSSC 综述"GitHub 配套"确认无误；补充 DOI 前缀 10.1057 说明
3. npj MHR "2024 创刊" → 更正为 2022 创刊、2026 首个 IF 9.1

---

*报告结束。四份 agent 原始调研记录见会话历史；证据快照在四个 tmp 目录中可复查。*
