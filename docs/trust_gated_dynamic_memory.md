# 信任门控动态记忆

## 已实现的运行流程

只在 `depression_config.json` 的顶层 `enabled` 与 `memory.enabled` 同时为 true 时启用。普通居民不加载动态库。当前已为 **卡布达默认配置**开启；其 mild/moderate/severe 配置共用同一份初始记忆文件，保持原有启用开关。其他居民按各自配置运行。

1. Agent 给本轮生成稳定 `turn_id`，传入对方当前话语及最近上下文。
2. `DynamicMemorySystem.prepare()` 生成查询向量，在当前居民的独立记录中做余弦相似度检索。
3. 相似度达到 `min_similarity` 后，以 `0.9 * similarity + 0.1 * importance` 排序，可披露和受阻结果分别限额。当前采用精确向量扫描，复杂度 O(Nd)，适合小镇规模；未引入新向量数据库或标签匹配兜底。
4. 只有信任达到阈值的正文进入动态提示；受阻条目只产生通用不适信号。实际曾向该对象说过的原话可作为已知事实回忆，不能因此解锁相关深层记忆。
5. 小镇聊天循环通过复读检查、接受回复后才提交信任、记录实际披露、写入本轮原话或反思；被丢弃的候选回复不写回。重复 `turn_id` 不重复推进主诉图、增加信任或写记忆。

预览不更改持久状态、访问次数和信任；只有可重建向量缓存和最多 32 条待提交决策会在内存中变化。直接调用引擎的客户端应在 preview 的 `turn_id` 与 commit 的 `metadata.turn_id` 中传同一 ID。无显式 ID 的直接提交按来源、对象、实际内容、模拟时间去重。

## 存储和普通记忆隔离

```text
<居民 storage_root>/
├── associate/                     # 原有联想记忆
└── depression_memory/
    ├── state.json                 # 权威快照：正文、信任、披露、提交ID、公开节点ID
    └── index/vectors.json         # 可重建 embedding 缓存
```

采用单一 `state.json` 原子替换，避免正文与信任分文件写入造成版本不一致。checkpoint 同时内嵌完整动态状态，恢复时它优先于磁盘上的较新状态；向量缓存由模型配置和检索文本的哈希标识。

底层复用 `Associate.index.embedding_model` 所绑定的 embedding 实例，调用其 `get_text_embedding()` / `get_query_embedding()`；不共享向量节点、正文或信任。未配置动态记忆的居民没有这些开销。

- 新聊天摘要、对话事件和反思正文进入动态库，普通库只保存中性描述。摘要/旧记录缺少可靠的逐条披露标注时默认阈值为 1.0，避免摘要降低来源限制。
- 模拟中的已发生聊天原话以 `source_type=chat` 保存，反思标为主观解释，不生成新的传记事实。
- 初次启用时，已有普通节点复制为 `legacy_associate` 记录，阈值为 1.0。原始普通索引不删除；对该患者启用专属检索过滤，只放行启用后登记的公开节点。过滤覆盖事件、想法、聚焦检索和聊天检索。
- 非抑郁居民只记住实际听到或观察到的内容，不读取患者隐藏记忆。患者对话行动在地图上使用中性摘要。
- 向量检索失败时返回空上下文并记录告警，不回退到未门控正文。

## 提示词边界

启用后 `memory.public_persona` 替代基础描述中的私人人设；不设置时使用只有名字的公开描述。请勿在公开描述中重复敏感事实。

主诉图仍在内部推进。回复提示的基础状态只获得数值情绪和有限表达风格枚举；当前节点的主诉文本通过独立的「本轮可表达的当前主诉」类别按信任门控后提供。未来分支、节点标签和自由文本修正模式不直接注入。反思简版提示维持原来的可见范围。前一轮情绪只传递数值字段，避免另一对象的敏感内容通过情绪说明串入本轮。

信任是持久的逐对象状态，瞬时情绪可读取它，但不能改变门控阈值。受阻信号会轻微提高防御并限制表达程度。没有相关敏感记忆时不会仅因低信任产生受阻信号。

## 当前主诉内容单元

每个 `complaint_graph.stages` 节点可配置 `disclosure_units`。它是该节点对话文本的完整披露视图，**替代**原始 summary/core_belief/narrative_focus 的自动注入，不是追加到这些字段后面。可编写低敏感度的表层表达，但同一事实的改写应采用一致阈值，并与初始经历记忆的阈值对齐。

```json
"disclosure_units": [
  {"id": "surface", "content": "最近工作上的事让我有些难受。", "disclosure_threshold": 0.25},
  {"id": "core_belief", "content": "我总把工作上的失败看成自己的价值不足。", "disclosure_threshold": 0.7}
]
```

- 只从当前节点产生候选；对象明确且 `trust_for(other) >= disclosure_threshold` 才可进入提示。节点情绪中的 trust 不参与授权。无须向量相似度命中；向量服务不可用时，此独立通路仍然可用。
- 单元是当前主观解释，不能据此补写传记事实。提示要求按话题自然表达，无关时不展开。初版不对主诉单元额外执行相关性检索，也不因存在隐藏单元而产生不适信号。
- 按配置顺序选择完整单元，不截断。`memory.complaint_max_items` 默认 3，`memory.complaint_max_context_chars` 默认 600，均允许设为 0。主诉先占用这部分预算，经历记忆使用剩余额度，两类正文总量受现有 `max_context_chars`（默认 2400）约束；此限制不计算 JSON 包装和其他提示层。完全相同的已选正文只注入一次。
- 单元 ID 在节点内唯一，阈值必须是有限的 `[0,1]` 数值。配置非法时明确报错；没有此字段或值为 null 的旧节点，从 summary/core_belief/narrative_focus 派生阈值为 1.0 的单元；显式空数组表示不提供主诉文本。
- 动态生成节点的单元阈值由程序强制为 1.0，防止模型自行降低来源限制；缺少单元时同样采用旧节点规则，非法单元列表整体拒绝为空。此策略保守，可能减少中高信任阶段的动态分支表达，后续应通过带来源证据的敏感度策略和真实对话评测优化。
- 同一 turn_id 固定预览时的节点内容版本、信任和可见单元。实际回复接受后才更新信任、记录披露；信任变化和节点推进在下一轮生效。直接提交而未预览的事件不会被追溯认定为使用了主诉单元。
- 单元引用 ID 是节点 ID、单元正文和阈值的哈希。正文保留在主诉图，不复制进普通向量记忆。提交时 `complaint_sources` 只保存引用对应的节点、单元、阈值和实际披露对象；引用参与聊天或后续派生记忆的阈值继承。为保守起见，本轮所有可见来源均参与继承。
- 切换节点后，旧节点不再从主诉通路进入提示。对该对象说过的实际原话仍可作为聊天记忆召回，不能据此解锁旧节点或新节点的完整信念。
- checkpoint 保存显式主诉单元及其披露来源记录；恢复时已保存的节点策略优先于配置变更。旧 checkpoint 没有新记录时按空记录兼容，旧节点采用上述默认规则；配置节点原先未存入 checkpoint 的情况仍沿用项目已有的配置恢复方式。

卡布达的默认、mild、moderate、severe 四份配置已提供表层主诉、回避/不安和核心信念单元，阈值分别为 0.25、0.45、0.7；现有启用开关保持不变。这里的阈值是工程配置，尚未通过真实 LLM 对话校准。

新增回归测试不依赖 pytest，可执行：

```bash
python -m unittest discover -s tests -p test_complaint_context.py -v
```

## 配置

```json
{
  "memory": {
    "enabled": true,
    "public_persona": "角色公开身份、爱好和表达习惯",
    "initial_trust": 0.3,
    "initial_trust_by_partner": {"蜻蜓队长": 0.4},
    "min_similarity": 0.5,
    "max_items": 4,
    "max_context_chars": 2400,
    "runtime_threshold": 0.8,
    "initial_memories_file": "initial_dynamic_memories.json"
  }
}
```

初始内容单独维护在该 agent 目录下的 `initial_dynamic_memories.json`：

```json
{
  "schema_version": 1,
  "owner_id": "卡布达",
  "items": [
    {
      "memory_id": "unique_fact_id",
      "content": "一条可独立披露的经历或感受",
      "retrieval_text": "该经历的语义概述与相关表达",
      "kind": "subjective_belief",
      "disclosure_threshold": 0.7,
      "importance": 0.8,
      "generates_discomfort": true,
      "source_type": "persona_seed",
      "provenance": "authored_extension",
      "source_ids": ["原始人设字段路径"]
    }
  ]
}
```

- `initial_memories_file` 相对 agent 目录解析，与启动命令的工作目录无关；也支持绝对路径。普通居民或 `memory.enabled=false` 时不读取文件。
- 文件必须声明 `schema_version: 1`、与居民名字一致的 `owner_id` 和 `items` 列表，每条记忆须有唯一、非空的 `memory_id`。缺文件、格式错误或归属不符时明确报错。
- 保留内联 `memory.items` 以兼容旧配置；不得与外部文件同时提供非空条目，新人设应使用独立文件。
- 初始文件只供新模拟初始化，运行中的访问、信任和新增记忆只写入 `storage_root/depression_memory/`。已有 checkpoint 或磁盘运行快照优先，编辑初始文件不会追溯修改旧模拟；新增条目在新运行中生效。
- 卡布达目前有 20 条初始记忆，覆盖工作与自我价值、日常习惯、回避模式、求职担忧和支持偏好。`provenance=existing_persona` 表示整理既有设定，`authored_extension` 表示依据现有人设补充的虚构细节。它们不冒充模拟中实际发生的事件。
- `retrieval_text` 用于 embedding，不做关键词集合匹配；缺省使用正文。
- 阈值与重要性限定在 `[0,1]`。相似度阈值需要在实际 embedding 模型上校准。
- 初始经历默认不过期；可给运行态条目设置 ISO 时间 `expires_at`。使用模拟时钟，未来或过期记录不召回。
- 当前不增加时间衰减，以免久远人生经历被近期聊天系统性挤出。先以相关性为主建立评测基线。
- 记忆正文变更应使用新 ID，并保留来源。相同 ID 对应不同正文会报错。
- 对象键目前使用小镇居民唯一名字，配置和运行时需保持一致；尚不支持重命名后的关系迁移或多人会话。

信任评估复用动态模块的 LLM 调用：仅允许五种变化，步长依次为 `+0.08/+0.04/0/-0.08/-0.16`。非零变化必须附上能在对方本轮原话中找到的证据。解析失败、调用失败或缺少证据时保持信任。反思不更新对他人的信任。

披露评估只接受本轮已通过门控的记忆 ID；LLM 判断是否实际表达仍有语义判断误差，需用真实会话评测校准。代码保证门控与存储边界，但不能把生成模型的自由输出视为绝对无幻觉。

## 验证

```bash
/home/zbzhao/miniconda3/envs/generative_agents_cn/bin/python -m pytest -q tests/test_trust_gated_memory.py
```

测试覆盖向量同义召回、无关内容不召回、逐对象/逐居民隔离、受阻正文不注入、预览只读、提交幂等、旧 checkpoint 优先、过期、检索失败、普通检索过滤与 Agent 接线。测试用可控 embedding 和评估响应，不调用收费服务；仍需在实际 embedding/LLM 服务下检查披露自然度并校准参数。

原有 `tests/test_depression_dynamic_events.py` 在修改前存在 7 个失败：测试仍调用引擎已不存在的 `llm_transition_signal` 参数。本次记录这些基线失败，不通过改变主诉图接口掩盖它们。

2026-09-23 在 `generative_agents_cn` conda 环境中联合运行 `test_complaint_context.py`、`test_trust_gated_memory.py` 和 `test_depression_dynamic_events.py`：70 passed、7 failed，另有 4 个 subtests 通过。新增 16 项主诉上下文测试和现有动态记忆测试全部通过。将改动前 HEAD 的代码导出到临时目录、使用相同动态事件测试复测，结果为 54 passed、相同的 7 failed，确认本次改动未增加这三组测试中的失败项。测试使用可控模型响应，未调用真实 LLM/embedding 服务。
