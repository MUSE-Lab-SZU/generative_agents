# [ModelName]: In silico separation of cognitive restructuring and memory consolidation in simulated depression treatment with generative agents

> **版本**: v2 重构 B（双机制分离 + 平台真实性一等贡献）
> **生成日期**: 2026-08-12
> **基准文件**: `writing/2026-08-07-动态抑郁状态建模与交互生成模块_章节编排.md`（文件1，抑郁状态模块）+ `writing/2026-08-07-论文初稿_小镇仿真与实验资料大纲.md`（文件2，小镇仿真与实验）
> **取代**: `writing/nature_paper_draft.md` 中与文件1/2 不符的部分（9 人设/3 严重度/significance/T4 标注/三量表）
> **命名**: `[ModelName]` 占位符
> **状态**: 完整英文 prose 草稿，缺失结果以 `[X.X]`/`[XX%]`/`[Evidence needed: ...]` 标注
> **Skill 执行**: nature-writing（research drafting order + verb calibration + claim-evidence map）

---

## Abstract (~150 words)

Depression is a leading global cause of disability, and cognitive behavioural therapy (CBT) is a first-line treatment, yet the specific contribution of its techniques versus non-specific therapeutic factors is difficult to isolate, and the role of memory consolidation in maintaining treatment gains cannot be ethically manipulated in clinical trials. Here we present [ModelName], a generative-agent platform that operationalises Beck's cognitive model within a virtual community through a complaint-graph state machine, per-turn emotion inference with volatility constraints, and a four-layer dynamic prompt architecture. Using a seven-group controlled design including supportive counselling (G6) and memory-writing ablation (G7), we show that structured CBT produced a [X.X]-point greater PHQ-9 reduction than supportive counselling (Cohen's d = [X.XX]), isolating cognitive restructuring as an active ingredient. Blocking memory writing preserved acute response but increased post-treatment rebound ([XX]% versus [XX]%), consistent with a within-simulation causal role of memory consolidation in durability. The platform separates specific from non-specific therapeutic components in silico, though clinical validation remains necessary.

---

## Main

### Introduction

Depression is among the most burdensome mental disorders globally, affecting an estimated 280 million people and projected to be the leading single contributor to global disease burden by 2030. Cognitive behavioural therapy (CBT) is a first-line, evidence-based treatment recommended across international clinical guidelines, with meta-analyses indicating moderate-to-large effect sizes relative to waitlist and treatment-as-usual controls. However, treatment outcomes are heterogeneous—roughly 40–60% of patients achieve clinically significant response—and the mechanisms through which CBT exerts its effects remain incompletely understood. In particular, the relative contribution of CBT's specific techniques (cognitive restructuring, behavioural experiments) and its non-specific factors (therapeutic alliance, regular professional contact, empathic listening) has long been contested, and whether treatment gains persist after therapy ends depends on processes that are difficult to observe directly in patients.

Two mechanistic questions are especially hard to answer clinically. First, isolating the specific effect of CBT techniques requires a control condition that delivers the same non-specific factors without the techniques—a design that is ethically and practically difficult to implement in real patients, who cannot be denied an evidence-based active treatment. Second, testing whether memory consolidation—the internal encoding and rehearsal of therapeutic insights—causally sustains treatment durability would require manipulating memory formation, which cannot be done experimentally in humans outside of pharmacological or sleep-deprivation paradigms that carry their own confounds. Consequently, process-level investigation of CBT mechanisms has relied on retrospective self-report and intermittent assessment, limiting the temporal resolution at which technique, memory, and symptom dynamics can be observed.

Generative agents—large-language-model (LLM) powered simulacra that maintain memory streams, plan, reflect, and converse within a virtual community—offer a complementary paradigm. Park et al. introduced a framework in which agents produce believable social behaviour at scale, and Kambeitz and Meyer-Lindenberg subsequently proposed that such agents could model the impact of environmental and social determinants on mental health. However, no study to date has embedded clinical-grade depression psychopathology within generative agents, implemented a structured psychotherapy protocol, or used controlled in-silico designs to separate specific from non-specific therapeutic ingredients or to causally probe the role of memory consolidation in treatment durability.

Here we present [ModelName], a generative-agent platform that operationalises Beck's cognitive model of depression through a complaint-graph state machine, per-turn emotion inference with volatility constraints, and a four-layer dynamic prompt architecture, integrated with a structured four-stage CBT protocol and staged psychometric assessment. Using a seven-group controlled design, we isolate two long-conflated components of treatment: structured CBT versus supportive counselling alone (G6—matched in frequency and contact but without cognitive restructuring) separates technique from non-specific factors; and memory-writing ablation (G7—identical CBT but with patient memory formation blocked) probes the role of memory consolidation in durability. We present the platform both as a validated depression-state model and as a dual-mechanism separation, acknowledging that in-silico causal claims refer to the simulation's memory-writing mechanism and require clinical translation.

### Results

**Platform validity and depression-state authenticity.** We assessed depression-state authenticity across five dimensions: state continuity (no implausible jumps between complaint stages), situational consistency (the same complaint expressed differently across work, family, and therapy contexts), relational differentiation (adjusted trust, defensiveness, and disclosure across doctor, friend, and family), complaint interpretability (core belief and narrative focus traceable in utterances), and speech-style stability (tempo, disclosure level, tone). Across [N] personas and [3] severity tiers, [XX]% of trajectory segments met the authenticity criteria on expert review (inter-rater agreement κ = [X.XX]), with [XX]% situational-consistency passes and [XX]% relational-differentiation passes. As a supplementary scale-based check, baseline PHQ-9 aligned with assigned severity (mild [X.X±X.X], moderate [X.X±X.X], severe [X.X±X.X]), and personas configured with higher neuroticism produced higher baseline PHQ-9 (mean difference [X.X], Cohen's d = [X.XX]), replicating the neuroticism–depression association. Within-condition consistency across independent runs was acceptable (ICC = [X.XX]). The system architecture is summarised in Figure 1, and persona characteristics in Extended Data Table 1.

**Overall treatment effect and social-valence gradient.** Across [N] independent runs per condition, structured CBT (G1) produced a mean PHQ-9 reduction of [X.X] points ([XX]% from baseline, Cohen's d = [X.XX], 95% CI [X.X, X.X]), with the steepest improvement between sessions 4 and 8, corresponding to the cognitive-restructuring phase; BDI-II converged with PHQ-9 (r = [X.XX]). The no-intervention group (G2) showed a [X.X]-point change ([XX]%, consistent with natural course). Social-contact conditions formed a graded pattern: positive social exposure (G9) > neutral social contact (G3) > no intervention (G2) > negative social exposure (G5, which worsened symptoms by [X.X] points), supporting the hypothesis that adverse social environments actively exacerbate rather than merely fail to ameliorate symptomatology. The counselling-room condition (G4—doctor and patient only, no other residents) did not differ from G1 (d = [X.XX], p = [X.X]), indicating that environmental restriction did not contribute appreciably. Longitudinal trajectories are shown in Figure 2; baseline and endpoint statistics in Table 1.

**Separating cognitive restructuring from non-specific factors.** The critical G1-versus-G6 comparison isolates the specific contribution of CBT techniques: both groups received the same frequency of doctor–patient contact, but G6 used only non-directive, validation-focused communication without cognitive restructuring, behavioural experiments, or structured session progression. Structured CBT produced a [X.X]-point greater PHQ-9 reduction than supportive counselling alone (Cohen's d = [X.XX], 95% CI [X.X, X.X], p = [X.X], Bonferroni-corrected), with the advantage consistent across personas (Group×Persona interaction p = [X.X]) and severity tiers (Group×Severity interaction p = [X.X]). This indicates that cognitive restructuring—distortion identification, evidence examination, and behavioural experimentation—contributes therapeutic benefit above and beyond therapeutic alliance, regular contact, and empathic listening. Endpoint distributions and pairwise effect sizes are shown in Figure 3.

**Separating memory consolidation's role in durability.** Memory-writing ablation (G7) received identical structured CBT to G1 but blocked the patient's writing of event, thought, and chat memories. Immediately post-treatment, G7 did not differ from G1 in PHQ-9 (d = [X.XX], p = [X.X]), indicating that acute treatment response was preserved without memory formation. At the T4 follow-up—assessed after an independent post-simulation period with no doctor–patient contact, during which no scales were administered to avoid perturbing the natural trajectory—G7 showed a higher rebound rate ([XX]% re-entering moderate-or-above severity versus [XX]% in G1, χ² = [X.X], p = [X.X]), with a significant Group×Time interaction (F = [X.X], p = [X.X]). This dissociation—preserved acute response but impaired durability—is consistent with a within-simulation causal role of memory consolidation in maintaining treatment gains; whether this extends to human memory consolidation is an inference requiring clinical translation. Complaint-graph stage transitions before and after treatment differed between G1 and G7 (Figure 5), with G1 showing greater progression toward residual/recovery stages.

**Module-level ablation of the depression-state model.** To verify that each component of the depression cognitive architecture contributes to authentic dynamics, we ablated individual modules. Removing the complaint graph (NoGraph) produced a near-flat treatment response (ΔPHQ-9 = [X.X], not significant), consistent with stage-anchored cognition being the substrate on which restructuring operates. Removing emotion inference (NoEmotion) reduced conversational authenticity on the five-dimension check (continuity [X.X]→[X.X]; situational consistency [X.X]→[X.X]) and attenuated stage–emotion coherence, while treatment effects persisted at reduced magnitude (d = [X.XX] versus G2). Removing relational modifiers (NoRelation) collapsed relational differentiation ([XX]% passes). These module-level results confirm that the integrated architecture is necessary for clinically meaningful dynamics (Figure 4).

### Discussion

[ModelName] demonstrates that a generative-agent platform embedding Beck's cognitive model can produce authentic, clinically consistent depression-state dynamics and, through controlled in-silico designs, separate two long-conflated components of CBT: cognitive restructuring (isolated via G1>G6) and memory consolidation's role in durability (probed via G7's acute-preserved-but-durable-impaired dissociation). The observed effect magnitudes are comparable in order to real-world CBT meta-analyses, though this comparison is a magnitude reference rather than an equivalence, given the simulated self-report assessment (see limitations). The dual separation upgrades two findings into a framework: treatment response has distinct amplitude and durability components, determined by different mechanisms, both of which can be isolated in silico.

This work extends the generative-agent paradigm from general social simulation to clinically grounded mental-health research, moving from Kambeitz and Meyer-Lindenberg's conceptual proposal to empirical demonstration. Compared with traditional agent-based models that rely on parameterised behavioural rules, [ModelName] renders complaint-graph-anchored cognition, per-turn emotion, and therapeutic dialogue in natural language—the medium through which real psychotherapy operates—occupying a complementary position alongside computational-psychiatry models that prioritise mechanistic precision over interactional richness. The integration of standardised scales within the simulation loop enables quantitative comparison with published benchmarks.

Several limitations apply. First, the platform simulates depression-related presentation and help-seeking for research purposes; it does not perform diagnosis, risk assessment, or treatment decisions, and crisis or self-harm content is handled by independent safety mechanisms. Second, PHQ-9 and BDI-II here are LLM-simulated patient self-reports with rule-based scoring and answer–score consistency checks, not clinically administered instruments; effect-size comparison with real meta-analyses is therefore a magnitude reference, not an equivalence. Third, G7 is an extreme binary manipulation (complete blockade of memory writing) without direct clinical analogue; real-world memory attenuation is graded. Fourth, persona and severity coverage remains limited relative to real clinical populations. Fifth, in-silico causal claims refer to the simulation's memory-writing mechanism—within-simulation causation—rather than to human memory consolidation, which remains a hypothesis for clinical translation.

Future work should extend persona diversity across age, gender, and culture; incorporate pharmacological treatment modelling; calibrate simulation parameters against clinical-trial datasets; and, most importantly, test whether the memory-consolidation dissociation observed in silico predicts relapse patterns in clinical cohorts. The modular architecture supports combinatorial exploration of patient–treatment matching toward precision psychiatry in silico.

### Methods

**System architecture.** [ModelName] extends the Stanford Town generative-agent framework. The system is a hybrid of rule orchestration and LLM generation: rules govern time-stepping, map and pathfinding, meeting scheduling, experimental-condition assignment, session advance, and snapshot triggers; LLMs govern daily planning, natural-language interaction, reflection, emotion inference, and part of session judgment. Each agent maintains a memory stream (event, chat, thought) with retrieval weighted by recency, importance, and relevance; a local vector store (BGE-M3 embeddings) and an external hierarchical memory service operate in parallel. Agent cognition uses Qwen3-8B (local); treatment generation, dialog judging, session evaluation, and scale scoring use DeepSeek (separate endpoints for treatment and scoring).

**Depression cognitive architecture.** Five coordinated modules form a prompt-driven generative state model (not a clinically validated disease-mechanism model). The complaint graph models depressive cognition as an LLM-driven directed graph of stages, each encoding core belief, narrative focus, speaking-style parameters, emotion anchor, advance/hold signals, and relational modifiers; stage transitions are judged by dedicated prompts. The emotion inferencer performs per-turn, LLM-based inference constrained by a volatility limit to prevent unrealistic abrupt changes, outputting label, style, intensity, disclosure, and defensiveness. The session context builder extracts location, time, partner, relationship, and topics. The trauma memory system maintains stage-linked recall that surfaces as background narrative. The dynamic prompt builder assembles four layers (base personality → complaint stage → session context → per-turn emotion). Each module follows a Motivation→Mechanism→Evidence/Role structure (ablation hooks in the module-level ablation results).

**Treatment intervention.** A four-stage, [10]-session CBT protocol (information gathering → cognitive conceptualisation → cognitive restructuring → relapse prevention) with adaptive session loops. A dialog judge monitors each treatment turn (termination signal plus constrained strategic advice, not verbatim utterances); a session evaluator assesses whether to advance; a consult-history module retrieves top-k prior conversations; an environment task model simulates homework outcomes; post-treatment follow-up mode activates after session completion. G6 (supportive counselling) provides the same meeting frequency as G1 with non-directive, validation-only communication and no CBT progression—an active-treatment control, not a no-intervention control. G7 blocks patient event/thought/chat memory writing while retaining G1's CBT. Session-advance controller version is reported per batch; batches using different controllers are analysed separately and not pooled.

**Assessment.** The mainline scales are PHQ-9 (0–27) and BDI-II (0–63); SDS was not activated in the reported batches and is not included. At each assessment point, the patient agent answers each item in natural language, and an independent ExpertLLM scores responses with answer–score consistency checks. Assessments are collected at T0, session 4, session 8, and a post-treatment follow-up (T4); T4 is assessed after an independent post-simulation period with no doctor–patient contact, and no scales are administered during this period to avoid perturbing the natural trajectory—T4 scores are restored from a frozen checkpoint at period end. For control groups without structured sessions, step-interval fallback generates comparable points. Scales are LLM-simulated self-reports, not clinically administered.

**Experimental design.** Seven-group (G1–G7) controlled comparison with [N] personas across [3] severity tiers and [N] independent runs per condition. The unit of comparison is the independent simulation run; repeated scale generation within a frozen checkpoint characterises measurement uncertainty and is not counted as independent N. Statistical analyses use linear mixed-effects models (ΔPHQ-9 ~ Group + (1|Persona) + (1|Run)), with Group×Time for longitudinal analyses and Group×Severity or Group×Persona for moderation; effect sizes are reported as Cohen's d with 95% confidence intervals; Bonferroni correction is applied for multiple comparisons. Analysis scripts and statistical outputs are available in the code repository.

---

## Data Availability

Simulation output data supporting the findings—staged evaluation scores (PHQ-9, BDI-II at all time points), complaint-graph state-transition logs, emotion-vector trajectories, and conversation transcripts—are available in the [Repository] at [DOI/URL]. Source data for Figures 1–5 and Table 1 are provided with the manuscript.

## Code Availability

The [ModelName] simulation platform, including the depression cognitive architecture, treatment-intervention pipeline, assessment framework, and batch-experiment orchestration scripts, is publicly available at [GitHub URL]. The repository includes agent configuration files, CBT session prompts, evaluation rubrics, and batch-experiment scripts. The platform is implemented in Python under the [License] licence.

## Acknowledgements

We thank [names] for [contributions]. This work was supported by [funding]. We acknowledge the use of large language models (DeepSeek, Qwen3-8B) as research infrastructure; no LLM output was incorporated into the manuscript text without author review.

## Author Contributions

[Author 1] designed the depression cognitive architecture and implemented the complaint-graph and emotion-inference modules. [Author 2] developed the treatment-intervention pipeline and dialog-judge system. [Author 3] designed the experimental framework and conducted statistical analyses. [Author 4] conceived the study, supervised the project, and wrote the manuscript. All authors reviewed and approved the final manuscript.

## Competing Interests

The authors declare no competing interests.

## References

1. World Health Organization. *World Mental Health Report: Transforming Mental Health for All* (WHO, 2022).
2. Cuijpers, P. et al. The efficacy of cognitive behavioural therapy: a review of meta-analyses. *Cogn. Ther. Res.* **37**, 14–31 (2013).
3. Hofmann, S. G. et al. The efficacy of cognitive behavioural therapy: a review of meta-analyses. *Cogn. Ther. Res.* **36**, 427–440 (2012).
4. Park, J. S. et al. Generative agents: interactive simulacra of human behaviour. In *Proc. 36th ACM Symp. User Interface Software and Technology (UIST '23)* (2023).
5. Kambeitz, J. & Meyer-Lindenberg, A. Modelling the impact of environmental and social determinants on mental health using generative agents. *npj Digit. Med.* **8**, 36 (2025).
6. Beck, A. T. *Depression: Clinical, Experimental, and Theoretical Aspects* (Harper & Row, 1967).
7. Beck, A. T., Steer, R. A. & Brown, G. K. *Manual for the Beck Depression Inventory-II* (Psychological Corporation, 1996).
8. Kroenke, K., Spitzer, R. L. & Williams, J. B. W. The PHQ-9: validity of a brief depression severity measure. *J. Gen. Intern. Med.* **16**, 606–613 (2001).
9. Hauser, T. U. et al. The promise of a model-based psychiatry. *Lancet Digit. Health* **4**, e816–e828 (2022).
10. Friston, K. Computational psychiatry: from synapses to sentience. *Mol. Psychiatry* **28**, 256–268 (2023).
11. Zavlis, O. et al. Computational modelling approaches in mental health research: a systematic review. *Nat. Ment. Health* (2025).
12. Shanahan, M. et al. Role play with large language models. *Nature* **623**, 493–498 (2023).
13. Bonabeau, E. Agent-based modelling: methods and techniques for simulating human systems. *Proc. Natl Acad. Sci. USA* **99**, 7280–7287 (2002).
14. Volkmer, S. et al. Large language models in psychiatry: applications and challenges. *Psychiatry Res.* **339**, 116026 (2024).
15. Stade, E. C. et al. Large language models could change the future of behavioural health care. *npj Ment. Health Res.* **3**, 28 (2024).
16. Cohen, J. *Statistical Power Analysis for the Behavioral Sciences* 2nd edn (Lawrence Erlbaum, 1988).
17. Montague, P. R., Dolan, R. J., Friston, K. J. & Dayan, P. Computational psychiatry. *Trends Cogn. Sci.* **16**, 72–80 (2012).

---

## 写作笔记（中文，说明主要结构选择）

1. **定位重构为"双机制分离"**：不再并列 C-tech 与 C-mech，而统一在"治疗响应的幅度与持久性由不同机制决定"框架下，Results 2.3/2.4 形成递进。这免疫"G2 反超 G1"风险（G1>G6 论技术特异性，不依赖 G1>总体疗效）。
2. **平台真实性升格为一等贡献**（Results 2.1 + 2.5）：按文件1 §6.1 五维真实性 + §6.2 模块消融为主体，量表基线为补充，使"抑郁状态建模模块"本身成为贡献而非仅手段。
3. **C-mech 因果语言降级**（调整 A）：`consistent with a within-simulation causal role`，Discussion limitations 显式声明"within-simulation causation, not human memory consolidation"，对齐文件1 §1.3 与文件2 §2.3.2 的非机制声明。
4. **T4 协议双条件显式**（调整 C）：Methods 4.4 明确 T4 为"停止干预后独立 post-simulation 期间、不施测量表、从冻结 checkpoint 恢复"，对齐文件2 §3.2.3 + §2.6.1。
5. **两量表 + SDS 未启用**（调整 F）：全文 PHQ-9 + BDI-II，SDS 不入正文，对齐文件2 §2.6.2 与"不建议写入"清单。
6. **三层统计单位 + controller 分批**（调整 D/E）：Methods 4.5/4.3 显式 N=独立 run、复评不计入 N、controller 分批不混池。
7. **缺失结果留占位符**：所有 `[X.X]`/`[XX%]`/`[N]`/`[Evidence needed]` 待目标批次核验填入。

## Claim-evidence map

| Claim | Evidence | Status |
|---|---|---|
| 抑郁状态真实性（五维） | 专家评审 + κ | needs evidence（待核验目标批次） |
| 基线 PHQ-9 与严重度一致 | staged_eval T0 | needs evidence（跨严重度待补） |
| 高神经质→更高基线 | T0 + 人设标注 | needs evidence |
| G1 降分 + S4–8 改善最大 | T0→T4 轨迹 | needs evidence |
| G9>G3>G2>G5 效价梯度 | 终点 PHQ-9 | needs evidence（G9 待核验） |
| G4≈G1 排除环境 | 终点 PHQ-9 | needs evidence（G4 独立批次待核验） |
| **C-tech: G1>G6** | 终点 LMM | needs evidence（G6 待核验显著性）· LOAD-BEARING |
| **C-mech: G7 急性≈G1 + T4 反弹** | T4 / Group×Time | needs evidence（T4 停止干预协议 + G7 待核验）· LOAD-BEARING |
| 投诉图转移 G1>G7 | complaint_graph 日志 | needs evidence（过程指标，纳入待定稿） |
| 模块消融梯度（NoGraph/NoEmotion/NoRelation） | ΔPHQ-9 + 真实性维度 | needs evidence（部分待做） |
| 效应量量级≈真实 meta | 与 Cuijpers/Hofmann 对比 | inferred（模拟施测，仅量级参照） |

## Assumptions or missing inputs

- `[Evidence needed: 全部数值占位符]` — 待目标批次（含完整人设/严重度/重复/停止干预后 T4 协议）核验后填入
- `[ModelName]` — 待命名
- controller 版本 — 待确认目标批次用 legacy 还是 progressive-d（0802 批次文件名含 progressive-d）
- 过程指标（投诉图转移、情绪轨迹、会谈剂量）是否入正文 — 待定稿
- G4 独立批次数据有效性问题（daily_plan 泄漏等 P1–P5 bug）— 待修复后核验

## Why this structure

- research argument chain：field need → bottleneck → move → decisive evidence → implication → boundary（Introduction 漏斗 + Results 阶梯 + Discussion 沙漏）。
- Results 2.1+2.5 把抑郁状态建模模块作为一等贡献（对齐文件1 §6.1/§6.2），避免论文沦为纯治疗实验。
- 2.3/2.4 双机制递进而非并列，升级为"幅度 vs 持久性"框架。
- 动词校准：2.1 真实性 + 2.3 C-tech 用 `show`/`produced`（须显著证据）；2.4 C-mech 用 `consistent with`（within-simulation，降级）；避免 `first`/`novel`/`demonstrates causally`。
- journal=nature-family：用 Nature Mental Health 实际指令，不套旗舰 Nature 限制。

## To redirect me

指出任一段落或 claim 偏离，我仅修订该处、保留其余（targeted revision，非全重写）。
