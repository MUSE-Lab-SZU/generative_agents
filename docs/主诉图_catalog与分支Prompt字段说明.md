# 主诉图 `stage_catalog` 与分支 Prompt 字段说明

## 目的与边界

`ComplaintGraphManager.stage_catalog` 是主诉图的**运行态权威索引**，类型为
`Dict[str, Dict[str, Any]]`：键是节点 ID，值是完整的主诉节点对象。它由初始
`complaint_graph.stages` 配置创建，并在 LLM 规划出合法新分支后追加运行时节点。
程序的推进校验、ID 去重、图边校验、回放与持久化都以它为准。

它不应为了控制上下文而删除节点。传给 LLM 的是临时的
`prompt_graph_snapshot`：只包含当前任务需要的局部图、全量短 ID 索引和有界的
语义参照，不包含完整的 `stage_catalog`。

## `stage_catalog` 中单个节点的字段

每个节点都会经 `_sanitize_stage` 清洗为以下结构。

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `id` | `str` | 节点唯一标识。它是 `stage_catalog` 的键，也是 `next_candidates` 和路径中引用节点的方式。最大 80 字符。 |
| `label` | `str` | 面向人阅读的简短节点名称，概括该节点的主诉/自我认识。 |
| `summary` | `str` | 节点概述，说明患者在此处对痛苦、模式或处境形成的更具体认识。最大 220 字符。 |
| `core_belief` | `str` | 此节点下的核心信念或自我解释，不是诊断结论。最大 220 字符。 |
| `narrative_focus` | `List[str]` | 角色叙述时会反复围绕的主诉焦点，最多 12 项。 |
| `speaking_style` | `dict` | 角色表达方式，包含 `tempo`（语速/节奏）、`disclosure`（袒露程度）、`tone`（语气）和 `repair_pattern`（犹豫、修正或收回表达的常见方式）。 |
| `emotion_vector` | `dict[str, float]` | 六个 0～1 的情绪维度：`valence`（情绪正负向）、`arousal`（唤起/激活）、`defensiveness`（防御）、`shame`（羞耻）、`hopelessness`（无望）、`trust`（信任）。 |
| `advance_signals` | `List[str]` | 说明角色已形成下一层自我认识、可考虑推进的语言或行为信号，最多 12 项。 |
| `hold_signals` | `List[str]` | 说明角色仍停留在当前节点的语言或行为信号，最多 12 项。 |
| `relation_modifiers` | `dict` | 特定关系/互动对象下对状态或表达的修饰规则；具体内部键由配置定义。 |
| `next_candidates` | `List[str]` | 当前节点可直接进入的子节点 ID 列表，而非完整节点对象。清洗后最多 8 项，实际规划窗口通常由 `window_size` 限制。 |
| `is_terminal_stage` | `bool` | 是否终止节点。为 `true` 时不再调用分支规划器补子分支。 |
| `source` | `str` | 节点来源，通常为 `config`、`llm` 或 `fallback`。 |

## 共享的 Prompt 图快照字段

`branch_seed`、`branch_detail` 与 `branch_repair` 都先生成以下共享字段。

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `snapshot_version` | `int` | 快照格式版本，当前为 `1`，用于未来兼容解析和日志排查。 |
| `mode` | `str` | 当前任务模式：`branch_seed`、`branch_detail` 或 `branch_repair`。 |
| `current_stage` | 完整节点对象 | 当前待规划子分支的父节点。它保留完整字段，因为模型需要理解此处的主诉、信念和表达状态。 |
| `local_graph.recent_path` | 节点摘要数组 | 已实际走过路径的最近节点，按路径顺序保留；每项仅含 `id`、`label`、`summary`。不含未走到的前瞻节点。 |
| `local_graph.existing_children` | 完整节点数组 | `current_stage.next_candidates` 中当前合法的直接子节点。数量受 `window_size` 限制。 |
| `local_graph.semantic_guard` | 节点摘要数组 | 有界的近期节点语义参照，用于减少生成同义/重复分支；每项仅含 `id`、`label`、`summary`。 |
| `known_stage_ids` | `List[str]` | 已在 catalog 中占用的短 ID 索引，供模型避免重复命名。程序端仍会再次校验，因此它不是唯一性保障。 |

节点摘要统一为：

```json
{ "id": "stage_id", "label": "节点名称", "summary": "节点概述" }
```

## `branch_seed` 字段

`branch_seed` 的任务是先生成若干简略的直接子分支。除共享字段外，payload 还包含：

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `existing_next_candidates` | `List[str]` | 当前父节点已存在且合法的直接候选 ID。 |
| `needed_count` | `int` | 还需生成的候选数量，等于 `window_size - 已有合法候选数`。 |
| `window_size` | `int` | 当前节点最多保留多少直接候选分支。 |
| `candidate_seed` | `{}` | 此模式没有待补全种子，固定为空对象。 |
| `session_context` | `dict` | 当前会话场景、参与者和运行时事件等上下文。 |
| `conversation_content` | `str` | 当前相关对话内容；长度受 LLM 配置 `max_text_length` 限制。 |

模型返回的 `children` 中每项只需提供 `id`、`label`、`summary`。程序会检查 ID 是否已存在、是否重复，并可能调用 repair 补齐。

## `branch_detail` 字段

`branch_detail` 的任务是补全一个已经通过初步校验的候选。它共享 `branch_seed` 的图快照与
`existing_next_candidates`、`needed_count`、`window_size`、`session_context`、`conversation_content`，但有以下差异：

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `mode` | 固定为 `branch_detail` | 指示模型只补全单个节点，而不再创建一组新分支。 |
| `candidate_seed` | 节点摘要 | 要补全的候选，含 `id`、`label`、`summary`。返回的完整节点必须沿用该 `id`。 |
| `needed_count` | `int` | 父节点理论上仍缺少的数量；detail 调用不应据此额外生成节点。 |

模型返回 `stage` 完整节点对象。程序会强制使用 `candidate_seed.id`，再经过标准清洗；新节点的
`next_candidates` 在写入 catalog 时会被置为空数组，避免模型凭空创建未知图边。

## `branch_repair` 字段

`branch_repair` 的任务是在 seed 候选被拒绝或数量不足时补足合法分支。除共享字段与
`existing_next_candidates`、`needed_count`、`window_size`、`session_context`、`conversation_content` 外，还有：

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `mode` | 固定为 `branch_repair` | 指示模型生成补充分支。 |
| `accepted_branch_candidates` | 节点摘要数组 | 当前已录用的父节点子分支，以及本轮已通过校验的 seed；避免新分支与它们重复。 |
| `rejected_branch_candidates` | 带原因的摘要数组 | 之前被程序拒绝的候选。每项为 `id`、`label`、`summary`、`reason`；`reason` 常见值为 `duplicate_or_existing_stage_id`、`empty_stage_id` 或 `not_an_object`。 |

repair 模板中的“补分支上下文”只重复报告拒绝数、录用数和仍需数量；候选详细内容只在 payload
中出现一次，避免同一 JSON 在 prompt 中重复占用上下文。

## 快照预算配置

下列值放在 `complaint_graph.planner` 中；未配置时使用默认值。

| 配置 | 默认值 | 范围 | 用途 |
| --- | ---: | --- | --- |
| `prompt_recent_path_limit` | `4` | 1～12 | `local_graph.recent_path` 最多保留的路径节点数。 |
| `prompt_semantic_guard_limit` | `20` | 0～80 | `local_graph.semantic_guard` 最多保留的摘要数量。 |
| `prompt_known_stage_id_limit` | `512` | 50～5000 | `known_stage_ids` 最多保留的 ID 数。超过此上限时，由程序端唯一性校验和 repair 机制兜底。 |

该设计使完整节点对象数量主要受局部窗口限制，而不是随 `stage_catalog` 总节点数线性增长。即使
catalog 有 76 个或更多节点，LLM 也不会再接收 76 个完整节点对象。
