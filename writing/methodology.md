# Methodology: LLM-Powered Generative Agents for Depression Treatment Simulation

> **Version**: 2.0 (2026-07-08)
> **Corresponding to**: `doctor-agent` branch, Nature Mental Health submission preparation

---

## 1. System Overview

We present a generative-agent-based simulation platform that models depression psychopathology and evaluates psychotherapy interventions within a virtual community. The system extends the Stanford Town multi-agent framework ([Park et al., 2023](#references)) with five purpose-built modules forming a **depression cognitive architecture**—Complaint Graph Manager, Emotion Inferencer, Session Context Builder, Trauma Memory System, and Dynamic Prompt Builder—integrated with a structured **CBT treatment pipeline** and a **staged psychometric assessment framework**. Seven-group controlled experiments compare cognitive behavioural therapy (CBT) against multiple baselines across three depression severity levels, with longitudinal PHQ-9, BDI-II, and SDS assessments.

**Figure 1** provides the architectural overview.

---

## 2. Virtual Community & Base Agent Architecture

### 2.1 Stanford Town Foundation

The simulation inherits the generative agent framework proposed by Park et al. (2023): each agent maintains a **memory stream** recording daily events, thoughts, and conversations, with retrieval weighted by recency, importance, and relevance. Agents perceive their environment, generate plans, execute actions, and engage in natural-language conversations. The virtual village provides a bounded social context including residential, commercial, and recreational locations.

### 2.2 Agent Roles

| Role | Name | Function | LLM Backend |
|------|------|----------|-------------|
| **Patient** | 卡布达 (Kabuda) | Simulates depression psychopathology (cognitive, emotional, behavioural) | Qwen3-8B (local vLLM) |
| **Doctor** | 蜻蜓队长 (Captain Dragonfly) | Delivers structured CBT / supportive counseling | DeepSeek API |
| **Residents** | 田德莉娜, 呱呱蛙, 蟑螂恶霸 | Provide neutral or negative social interaction controls | Qwen3-8B / DeepSeek |
| **Evaluator** | ExpertLLM | Scores standardized depression scales | DeepSeek API |

### 2.3 Cognitive Loop

Each agent step follows: **perceive** (sense environment and other agents' actions) → **retrieve** (query memory stream) → **plan** (decompose daily schedule) → **reflect** (synthesize high-level insights from recent experiences) → **act** (execute movement, object interaction, or conversation). Patient agents additionally receive depression-specific prompt injections at each conversation turn.

---

## 3. Depression Cognitive Architecture

The patient agent's depression simulation is driven by five coordinated sub-modules, collectively producing a dynamic, clinically-informed depressive phenotype.

### 3.1 Complaint Graph Manager

Inspired by Beck's cognitive model of depression, the **Complaint Graph Manager** (`modules/depression/state_machine.py`) models the progression of depressive cognition as an LLM-driven directed graph of psychological states (complaint stages). Each stage encodes:

- **Core belief**: the dominant dysfunctional schema (e.g., "I am fundamentally inadequate")
- **Narrative focus**: the life domain currently under cognitive distortion (e.g., career failure)
- **Speaking style**: linguistic markers of depressive communication (tempo, disclosure level, tone, repair patterns)
- **Emotion vector anchor**: the expected emotional configuration for this stage
- **Advance / hold signals**: LLM-detected conversational cues that trigger or block stage progression
- **Relational modifiers**: how the patient perceives and responds to other agents

Stage transitions are driven by an **LLM-based graph planner** (`graph_planner.txt`) and **graph transition** prompt (`graph_transition.txt`), which evaluate the current conversation against the active stage's advance/hold signals and determine whether to remain, advance to a candidate next stage, or regress. The graph supports branching (multiple candidate next stages) to capture individual differences in depression trajectories.

**Example chain** (mild depression, job-loss variant):
```
mild_job_loss_self_doubt → mild_rumination_deepening → mild_social_withdrawal → mild_residual_or_recovery
```

Nine patient variants (卡布达 1–9) implement distinct complaint graphs mapped to different psychosocial triggers: job loss, academic failure, social rejection, family conflict, and accumulated micro-stressors.

### 3.2 Dynamic Emotion Inference

The **Emotion Inferencer** (`modules/depression/emotion_inferencer.py`) performs per-turn, LLM-based emotion inference constrained to small, reversible fluctuations around the current complaint stage's anchor. It receives the current complaint stage, graph window snapshot, session context, prior-turn emotion, and conversation content, and outputs:

| Field | Range | Description |
|-------|-------|-------------|
| **label** | 2–12 chars | Instantaneous emotional state label (e.g., "压抑的愧疚", "试探性的松动") |
| **style** | 15–120 chars | Natural-language description of emotional expression style |
| **intensity** | 0.0–1.0 | Overall emotional intensity |
| **disclosure_level** | 0.0–1.0 | Willingness to reveal authentic feelings (low → deflection; high → tentative openness) |
| **defensiveness** | 0.0–1.0 | Resistance to probing (high → short, guarded responses) |
| **volatility_note** | 10–80 chars | Frame-to-frame change note (must be small, reversible, coherent with current stage) |

A **volatility limit** (`volatility_limit`, default 0.12) constrains the maximum frame-to-frame emotion shift, preventing unrealistic abrupt changes (sudden recovery or collapse). The emotion output is embedded into the patient's generation prompt as the fourth (topmost) layer of the dynamic prompt assembly, directly shaping the linguistic expression of the current emotional state.

### 3.3 Session Context Builder

The **Session Context Builder** (`modules/depression/context_analyzer.py`) extracts structured situational information from the ongoing conversation: location, time of day, conversation partner identity and relationship, interaction type, detected complaint-relevant topics, speech act categories, and conversational stance. This context forms the third layer of the dynamic prompt, grounding the patient's response in the immediate social situation.

### 3.4 Trauma Memory System

The **Trauma Memory System** (`modules/depression/memory_system.py`) maintains a separate store of trauma-related memories linked to complaint graph stages. During prompted recall, relevant traumatic memories associated with the current stage can surface as background narrative texture in the patient's expression, without overriding the active complaint stage's primary narrative focus.

### 3.5 Dynamic Prompt Assembly

The **Dynamic Prompt Builder** (`modules/depression/prompt_builder.py`) assembles a four-layer prompt (`dynamic_prompt_layers.txt`) injected into the patient's conversation generation context:

1. **Base personality layer**: free-text personality description from `agent.json`
2. **Complaint stage layer**: current stage label, summary, core belief, narrative focus, speaking style parameters, and the graph window showing candidate next stages
3. **Session context layer**: location, time, conversation partner, relationship, detected topics
4. **Emotion layer**: instantaneous emotion label, style, intensity, disclosure level, defensiveness, and volatility note from the Emotion Inferencer

This layered architecture ensures that stable personality traits anchor the patient's behaviour, while per-turn emotional dynamics modulate expression within clinically plausible bounds defined by the current complaint stage.

### 3.6 Personality & Severity Calibration

Patient agents are configured via two complementary mechanisms:

1. **Free-text personality descriptions** in `agent.json` (e.g., "sensitive, introspective, kind-hearted but self-critical"), which form the base layer of the dynamic prompt.

2. **Depression severity tiers** (`depression_config_mild/moderate/severe.json`) that calibrate the complaint graph's starting stage, emotion vector baselines, core-belief intensity, and speaking style parameters. This enables within-persona severity manipulation for stratified analysis.

---

## 4. Treatment Intervention System

### 4.1 Structured CBT Protocol

The system implements a **4-stage, 10+ session CBT protocol** adapted from Beck's cognitive therapy manual:

| Stage | Sessions | Clinical Objective | Key Techniques |
|-------|----------|-------------------|----------------|
| **1. Information Gathering** | Session 1 → session_loop | Alliance building, S-E-T (Situation-Emotion-Thought) sample collection | Active listening, validation, symptom mapping |
| **2. Cognitive Conceptualization** | 2.1 → 2.2 → 2.3 | Identify cognitive distortions, conditional rules, and core beliefs | Downward arrow, distortion labeling, rule extraction |
| **3. Cognitive Restructuring** | 3.1 → 3.2 → 3.3-A/B | Challenge dysfunctional beliefs; design behavioural experiments | Courtroom exercise, evidence for/against, success/avoidance branching |
| **4. Relapse Prevention** | 4.1 → 4.2 | Consolidate gains; build resilience toolkit | Journey review, psychological first-aid kit, graduation ritual |

Each stage includes an **adaptive session loop**: if the session evaluator (Section 4.3) determines a session did not meet its objectives, the system repeats the current stage before advancing. Session prompts are injected at each doctor turn via `data/prompts/intervention_prompts.json`, providing stage-specific therapeutic guidance.

### 4.2 Dialog Judge

The **Dialog Judge** (`data/prompts/intervention/dialog_judge.txt`) is an LLM-based real-time conversation monitor that operates at each turn during treatment dialogues. It receives:

- **Patient state summary**: a compressed representation of the patient's current emotional and cognitive state (produced by a separate `think.llm` call before each judge invocation)
- **Current conversation history**: the full dialogue since the meeting began
- **Current session prompt**: the CBT stage-specific instructions
- **Prior session evaluation**: the outcome of the previous session (if any)

It outputs:
1. **`terminate`**: a boolean signal to end the current conversation
2. **`advice`**: a strategic suggestion for the doctor's next response, constrained to tactical guidance (e.g., "explore the patient's use of 'should' statements") rather than verbatim text, preventing the judge from ghostwriting the doctor's utterances

### 4.3 Session Evaluation

After each treatment conversation concludes, the **Session Evaluator** (`data/prompts/intervention/session_eval.txt`) independently assesses therapeutic progress. Input includes the full conversation transcript, the session prompt, and historical evaluation records. It outputs:

- **Efficacy score**: qualitative rating of the session's therapeutic value
- **Session-end flag**: whether the current CBT stage objective has been met
- **Rationale**: natural-language justification for clinical auditing

### 4.4 Post-Treatment Follow-up

When all scheduled CBT sessions are completed, the system transitions to **post-treatment follow-up mode** (`data/prompts/intervention/post_treatment_followup.txt`). Instead of advancing the session counter, follow-up prompts guide the doctor to:

1. Assess recent mood, sleep, energy, and symptom fluctuations
2. Inquire about homework adherence without blame
3. Monitor risk signals (hopelessness, self-harm ideation)
4. Help the patient identify one small, realistic next step

Follow-up conversations do not trigger session evaluations, allowing naturalistic observation of post-treatment trajectories without artificial session-count pressure.

### 4.5 Supportive Counseling (G6 Control)

A **supportive counseling** variant (`data/prompts/intervention/supportive_counseling_doctor.txt`) provides an active-treatment control: the doctor meets the patient at the same frequency as CBT but uses only non-directive, validation-focused communication—no cognitive restructuring, no behavioural experiments, no structured session progression. This isolates the **specific effect of CBT technique** from the **non-specific effect of regular professional contact**.

### 4.6 Consultation History & Continuity

The **Consult History** module maintains a dedicated memory store for doctor-patient dialogues. Before each doctor turn, a **gate LLM** determines whether historical retrieval is warranted. If so, the top-*k* most relevant prior conversations are retrieved (via embedding similarity) and summarized into a **consult history memory** injected into the doctor's generation context. This ensures therapeutic continuity across sessions, analogous to reviewing clinical notes.

### 4.7 Environment Task Model

When the doctor assigns between-session homework (e.g., behavioural experiments, activity scheduling), the **Environment Model** (`data/prompts/intervention/environment_task_outcome.txt`) simulates how these tasks might play out in the patient's daily life. Given the task description, the patient's current state, and the list of known agents, it generates concrete event descriptions (e.g., "卡布达 attempted to go for a 15-minute walk but stopped after 5 minutes, feeling self-conscious when he saw neighbours"). These events are injected into the patient's (and, if social, other agents') memory stream, enabling subsequent reflection and realistic follow-up discussion.

### 4.8 Meeting Scheduling

Treatment sessions are triggered by configurable **meeting rules** (`data/config.json → intervention.meeting_rules`):

- **Interval-based**: every *N* simulation steps (e.g., every 6 steps ≈ every 36 virtual hours)
- **Phase-gated**: only active during specified simulation phase windows
- **Duration-capped**: each meeting has a maximum virtual-time duration

Non-treatment social interactions (G3, G5) are scheduled through the **Resident Chat Scheduler** (`modules/resident_chat_scheduler.py`), with configurable interaction partners, randomization strategies, and prompt injections (neutral vs. negative).

---

## 5. Assessment Framework

### 5.1 Standardized Scales

Three validated self-report instruments are administered:

| Scale | Items | Construct | Reference |
|-------|-------|-----------|-----------|
| **PHQ-9** | 9 | Depression symptom severity (past 2 weeks) | Kroenke et al. (2001) |
| **BDI-II** | 21 | Cognitive, affective, and somatic depression symptoms | Beck et al. (1996) |
| **SDS** | 20 | Zung Self-Rating Depression Scale | Zung (1965) |

Each scale is administered via structured LLM-mediated interview: the patient agent answers each item in natural language, then an independent **ExpertLLM** (DeepSeek-Chat, separate from the treatment model) scores responses using standardized rubrics, outputting per-item scores and a severity classification (normal / mild / moderate / severe).

### 5.2 Staged Evaluation Timeline

Assessments are collected at four longitudinal time points:

```
T0 (Baseline)  →  Session 4 (Mid-treatment)  →  Session 8 (Late-treatment)  →  T4 (Follow-up)
Pre-intervention    After ~4 completed           After ~8 completed              ~48 virtual hours
                    treatment sessions            treatment sessions              post-treatment
```

The **Staged Evaluation Manager** (`modules/staged_eval_manager.py`) triggers assessments automatically during simulation without interrupting the agent loop. Trigger conditions are based on **completed meeting counts** (not simulation steps), ensuring assessments align with actual treatment exposure regardless of scheduling variability.

For control groups without structured sessions (G2, G3, G5), a **step-interval fallback** (`staged_eval.step_interval`) generates virtual assessment points at fixed simulation intervals, enabling comparable longitudinal data.

### 5.3 Assessment Quality Assurance

To address LLM output stochasticity, the system supports:

- **Repeat evaluation**: each assessment can be run multiple times (`runshells/run_t0_repeat_eval.py`)
- **Item-level stability analysis**: items with inconsistent scores across repeats are flagged
- **Adaptive re-sampling**: unstable items are automatically re-queried until convergence
- **Score reconciliation**: item-level scores are validated against LLM-reported total scores, with discrepancies flagged for review (`runshells/validate_scale_score_results.py`)

---

## 6. Experimental Design

### 6.1 Seven-Group Controlled Comparison

| Group | Code | Intervention | Depression Engine | Dialog Judge | Session Eval | CBT Prompts | Purpose |
|-------|------|-------------|-------------------|--------------|-------------|-------------|---------|
| **G1** | Doctor Intervention | Structured CBT (10 sessions) | ✅ Full | ✅ | ✅ | ✅ | Primary treatment group |
| **G2** | No Intervention | None (natural course) | ✅ Active | ❌ | ❌ | ❌ | Natural-history baseline |
| **G3** | Neutral Social | Random neutral chats with residents | ✅ Active | ❌ | ❌ | ❌ | Placebo control (non-therapeutic social contact) |
| **G4** | Counseling Room | Same as G1, environment-restricted | ✅ Full | ✅ | ✅ | ✅ | Environment specificity control |
| **G5** | Negative Social | Negative/harmful chats with residents | ✅ Active | ❌ | ❌ | ❌ | Iatrogenic-harm control |
| **G6** | Supportive Counseling | Doctor meetings with non-directive support only | ✅ Full | ❌ | ❌ | ❌ | Active-treatment control (isolates CBT-specific effect) |
| **G7** | Memory Removed | Same as G1, patient memory writing blocked | ✅ Full | ✅ | ✅ | ✅ | Ablation control (isolates memory consolidation role) |

**Design logic**:
- **G1 vs. G2**: establishes the overall treatment effect
- **G1 vs. G3**: rules out the "any conversation helps" confound
- **G1 vs. G4**: rules out environmental confounds
- **G1 vs. G5**: quantifies harm from negative social environments
- **G1 vs. G6**: isolates CBT technique from non-specific therapeutic factors (therapeutic alliance, regular contact, professional attention)
- **G1 vs. G7**: isolates the role of memory consolidation in treatment response

### 6.2 Severity Stratification

Each patient persona is instantiated at three depression severity levels:

| Severity | PHQ-9 Range | Clinical Profile | Complaint Graph Features |
|----------|-------------|------------------|--------------------------|
| **Mild** | 5–9 | Subthreshold-to-mild symptomatology | Earlier graph stages, moderate core-belief intensity |
| **Moderate** | 10–14 | Clinically significant, typical CBT candidate | Deeper graph stages, entrenched core belief |
| **Severe** | 15–19 | Severe symptomatology, higher risk | Advanced graph stages, pervasive hopelessness, low disclosure |

Severity is operationalized through the depression config files, which calibrate: starting complaint-graph stage, emotion-vector baselines, core-belief intensity, and speaking style parameters. Analysis treats severity as a **moderator variable** in linear mixed-effects models.

### 6.3 Persona Diversity

Nine patient persona variants (卡布达 1–9) implement distinct psychosocial profiles—varying in trigger events (job loss, academic failure, social rejection, family conflict), complaint graph trajectories, and speech patterns—while sharing the same demographic identity. This design enables **within-identity replication** (controlling for demographic confounds while varying psychological parameters) as a complement to the cross-identity diversity recommended for clinical generalizability.

Two additional patient identities (金龟次郎, 50M, small-business owner; and planned female personas) provide cross-demographic validation.

### 6.4 Ablation Components

The modular architecture enables systematic ablation (primarily through G6 and G7 group-level comparisons, with additional targeted studies):

| Ablation | Module Removed | Hypothesis |
|----------|---------------|------------|
| **No CBT Structure** | `session_prompt_injection` + `dialog_judge` + `session_eval` (G6) | Supportive contact alone is insufficient for cognitive restructuring |
| **No Memory Consolidation** | Patient event/thought/chat memory writing (G7) | Memory formation is necessary for durable treatment gains |
| **No Emotion Dynamics** (optional) | `EmotionInferencer` | Flat affect undermines conversational authenticity and complaint-graph coherence |
| **No Complaint Graph** (optional) | `ComplaintGraphManager` | Static depression cannot model treatment response trajectories or stage-appropriate emotion expression |

---

## 7. Implementation

### 7.1 LLM Backend Configuration

| Component | Model | Deployment | Role |
|-----------|-------|-----------|------|
| **Agent cognition** | Qwen3-8B (4-bit quantized) | Local vLLM (GPU 0–1) | Patient/resident daily behaviour, reflection, planning |
| **Treatment generation** | DeepSeek-V4-Flash | Cloud API | Doctor utterances, requiring clinical nuance |
| **Dialog judging** | DeepSeek-V4-Flash | Cloud API | Real-time conversation quality monitoring |
| **Session evaluation** | DeepSeek-V4-Flash | Cloud API | Post-session therapeutic assessment |
| **Scale scoring** | DeepSeek-Chat | Cloud API | ExpertLLM psychometric rating (separate model from treatment LLM) |
| **Memory embedding** | BGE-M3 | Local vLLM (GPU 2–3) | Semantic retrieval for local & external memory |

### 7.2 Memory Systems

Two memory systems operate in parallel:

- **Local memory** (LlamaIndex-based vector store): stores agent events, thoughts, and conversations with embedding-based retrieval (BGE-M3). Chat memories include metadata flags (`forced`, `meeting_id`, `retrieval_scope`) for conversation-type-aware retrieval.

- **External memory** (EC-Doll service): a dedicated remote memory service providing hierarchical memory organization (short-term → L1 → L2 milestones), used for long-term persistence and cross-simulation continuity.

### 7.3 Experiment Pipeline

```
run_batch_experiment.py
  ├── Generates per-condition runtime configs from experiment group templates
  ├── Launches start.py for each condition (supports parallel execution)
  ├── Collects: staged_eval results, judge traces, conversation logs, scale scores
  ├── Post-processes: memory visualization, external memory audit, dialogue merging
  └── Outputs: summary JSON + Markdown reports with score validation
```

Key automation scripts:
- `runshells/run_batch_experiment.py`: batch experiment orchestration
- `runshells/run_one_experiment.py`: single-condition end-to-end run
- `runshells/run_staged_eval_worker.py`: background scale evaluation worker
- `runshells/run_archived_repeat_scale_eval.py`: post-hoc repeat evaluation with stability analysis
- `runshells/run_t0_repeat_eval.py`: baseline assessment reliability testing
- `runshells/validate_scale_score_results.py`: score consistency verification

### 7.4 Simulation Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Simulation steps | 48–120 | Covers complete 4-stage CBT + follow-up |
| Step stride | 360 min (6 h) | Each step ≈ one wakeful segment of a day |
| Treatment interval | Every 6 steps | ≈ every 36 virtual hours (~2.5 treatments/week) |
| Forced chat turns | 18 max | Sufficient for therapeutic dialogue depth |
| Scale evaluation | Every 4 completed meetings | Mid- and late-treatment assessment points |
| T4 follow-up | 8 steps post-treatment | ≈ 48 virtual hours after last session |

---

## Figure 1: System Architecture Overview

> **Figure 1** provides a panoramic view of the generative-agent-based depression treatment simulation platform. The figure is designed according to Nature Mental Health conventions: clean visual hierarchy, colour-coded functional layers, minimal text labels, and deliberate white space to accommodate subsequent detailed figures (Figures 2–5).

### Visual Layout (Top-to-Bottom, Five Horizontally-Stacked Layers)

**Layer 1 — Virtual Community** (Top, pale blue background)
- A stylized village map with labeled locations (home, counselling room, park, shop)
- Agent icons distributed across locations: patient (卡布达, highlighted with orange border), doctor (蜻蜓队长, blue), residents (田德莉娜/呱呱蛙/蟑螂恶霸, grey)
- Curved arrows showing daily movement trajectories and social interactions
- **Visual weight**: light; establishes ecological context

**Layer 2 — Agent Cognitive Architecture** (Second row, pale green background)
- Central node: "Memory Stream" with three retrieval beams labeled *Recency · Importance · Relevance*
- Three downstream modules in horizontal sequence:
  - **Perceive → Plan** (daily schedule decomposition)
  - **Retrieve → Reflect** (insight extraction from memory)
  - **Act** (movement, object interaction, conversation initiation)
- **Visual weight**: moderate; shows the base architecture inherited from Park et al. (2023)

**Layer 3 — Depression Cognitive Engine** (Third row, pale orange background)
- Five interlocking modules arranged horizontally, connected by directional arrows forming a feedback loop:
  1. **Complaint Graph** (state machine icon) → "LLM-driven stage transitions (graph_planner + graph_transition)"
  2. **Emotion Inferencer** (LLM icon + emotion tags) → "Per-turn: label · style · intensity · disclosure · defensiveness · volatility"
  3. **Session Context** (structured data icon) → "Location · Time · Partner · Relationship"
  4. **Trauma Memory** (database icon) → "Stage-linked trauma recall"
  5. **Dynamic Prompt Builder** (layered document icon) → "4-layer assembly: base → stage → context → emotion"
- A return arrow from Emotion Inferencer to Complaint Graph labeled "emotion → stage feedback"
- **Visual weight**: heavy; this is the primary methodological contribution

**Layer 4 — Treatment Intervention Pipeline** (Fourth row, pale purple background)
- Horizontal timeline: **Session 1** → **Session 2.1–2.3** → **Session 3.1–3.3** → **Session 4.1–4.2** → **Follow-up**
- Three parallel LLM modules shown as vertical annotations beneath the timeline:
  - **Dialog Judge** (gavel icon): "Real-time termination + advice"
  - **Consult History** (database icon): "Gate → Retrieve top-k → Summarize"
  - **Session Eval** (clipboard icon): "Post-session efficacy scoring"
- Environment Model shown as a side branch: "Doctor orders → Task simulation → Memory injection"
- **Visual weight**: heavy; shows the treatment delivery mechanism

**Layer 5 — Assessment & Experiment Design** (Bottom, pale grey background)
- Left panel: **Staged Evaluation Timeline**
  - Four assessment points: `T0 (Baseline)` → `S4 (Mid)` → `S8 (Late)` → `T4 (Follow-up)`
  - Three scale icons beneath each point: PHQ-9 · BDI-II · SDS
- Right panel: **Seven-Group Design** (compact matrix)
  - Rows: G1–G7; Columns: intervention type, depression engine, CBT structure, purpose
  - Colour coding: G1 (green, primary treatment), G2 (grey, no treatment), G3/G5 (yellow, social control), G6 (blue, active control), G7 (red, ablation)
  - Minimal text: group codes + one-line purpose labels
- **Visual weight**: moderate; summary-level, with details deferred to Figures 2–5 and Table 1

### Design Specifications

| Aspect | Specification |
|--------|--------------|
| **Colour palette** | Nature-brand-compatible: muted blues, oranges, purples, greens on white background; 3–5 colours maximum per layer |
| **Typography** | Sans-serif (Helvetica/Arial); 8–10 pt labels; bold for module names, regular for descriptions |
| **Connectors** | Curved Bézier arrows (not straight lines) for flow; dashed lines for feedback loops; distinct arrowhead styles for data flow vs. control flow |
| **Spatial allocation** | Layers 3 + 4 occupy ~40% of vertical space (primary contributions); Layers 1 + 5 ~15% each (context); Layer 2 ~15% (foundation) |
| **White space** | Generous padding between layers; figure designed to be read at full-page width (~180 mm), with internal elements legible at 50% reduction |
| **Deferred detail** | Complaint chain stages (→ Fig 5 heatmap), PHQ-9 trajectories (→ Fig 2), group comparisons (→ Figs 3–4), ablation results (→ Fig 4 / SI) |
| **Output format** | Vector (SVG/PDF) for manuscript; raster (PNG at 300 dpi) for submission |

### Relationship to Other Figures

| Figure | Focus | Relationship to Fig 1 |
|--------|-------|----------------------|
| **Fig 1** | System architecture (panoramic) | — |
| **Fig 2** | PHQ-9 longitudinal trajectories × 7 groups | Zooms into Layer 5 assessment data |
| **Fig 3** | Endpoint PHQ-9 distributions + pairwise effect sizes | Zooms into Layer 5 group comparison |
| **Fig 4** | Ablation results (G1 vs. G6 vs. G7) | Zooms into Layers 3–4 component contributions |
| **Fig 5** | Complaint graph stage transition heatmap (before vs. after treatment) | Zooms into Layer 3 complaint graph dynamics |
| **Table 1** | Baseline characteristics and endpoint summary | Complements Layer 5 with numerical detail |

---

## References

1. Park, J. S. et al. (2023). Generative agents: Interactive simulacra of human behavior. *UIST 2023*.
2. Beck, A. T. (1967). *Depression: Clinical, experimental, and theoretical aspects*. Harper & Row.
3. Beck, A. T., Steer, R. A., & Brown, G. K. (1996). *Manual for the Beck Depression Inventory-II*. Psychological Corporation.
4. Kroenke, K., Spitzer, R. L., & Williams, J. B. W. (2001). The PHQ-9: Validity of a brief depression severity measure. *Journal of General Internal Medicine*, 16(9), 606–613.
5. Zung, W. W. K. (1965). A self-rating depression scale. *Archives of General Psychiatry*, 12(1), 63–70.
6. Kambeitz, J. & Meyer-Lindenberg, A. (2025). Modelling the impact of environmental and social determinants on mental health using generative agents. *npj Digital Medicine*, 8, 36.
7. Hauser, T. U. et al. (2022). The promise of a model-based psychiatry. *The Lancet Digital Health*, 4(12), e816–e828.
