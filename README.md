# 基于斯坦福小镇的抑郁症干预仿真系统 GenerativeAgentsCN

> 更新时间：2026-04-25  
> 目标：快速看懂项目、跑通链路、定位关键配置与日志。

## 更新日志（近期）

以下为 README 内维护的近期更新摘要：

- 2026-04-25：量表评估更新，对齐外置记忆系统
  - 相关文件：`customization\depression_scale_agent\app.py`

- 2026-04-24：强制干预地址语义正常化，修复写入记忆时的地址错误问题
  - 相关文件：`modules/agent.py`

- 2026-04-23：更新外置记忆系统对接、新增可视化脚本
  - 相关文件：`modules/external_memory_bridge.py`、`modules\ec_doll_memory_service_client.py`、`visualize_external_memory_audit.py`

- 2026-04-21：融入赵学长的动态抑郁人设系统、修复兼容bug
  - 相关文件：`analyze_depression_dynamic_state.py`、`modules/depression_dynamic_adapter.py`、`modules/depression/state_machine.py`、`modules/depression/context_analyzer.py`、`modules/depression/engine.py`、`modules/agent.py`、`data/config.json`、`frontend/static/assets/village/agents/卡布达/depression_config.json`

- 2026-04-20：新增会话中判断LLM、会话后评估LLM
  - 相关文件：`modules/intervention_manager.py`、`data/config.json`等

## 1. 项目简介

本项目是一个多智能体仿真系统，在常规行为仿真基础上，扩展了医患场景中的干预会话能力，重点包含：

- 会诊调度与强制会话（Intervention）
- 会话判定与终止检测（Dialog Judge）
- 会后评估与咨询记录（Session Eval / Consult Record）
- 本地记忆与外置记忆协同（External Memory）
- 抑郁动态链路（Depression Dynamic）

注意：该项目用于技术与研究场景，不构成医疗建议。

## 2. 代码目录结构

| 路径          | 说明                                                             |
| ------------- | ---------------------------------------------------------------- |
| `start.py`    | 仿真主入口（按 step 推进，写 checkpoint 与对话日志）             |
| `compress.py` | 将 checkpoint 压缩为回放数据（`movement.json`、`simulation.md`） |
| `replay.py`   | 回放 Web 服务入口（Flask）                                       |
| `modules/`    | 核心逻辑模块（agent、intervention、memory、depression 等）       |
| `data/`       | 配置与提示词（`data/config.json`、`data/prompts/...`）           |
| `frontend/`   | 回放前端资源与静态资产、动态抑郁人设`depression_config.json`     |
| `results/`    | 仿真输出目录（`checkpoints/`、`compressed/`）                    |

## 3. 快速使用说明

## 3.1 环境准备

- 建议 Python 3.10+。
- 根目录当前未维护统一 `requirements.txt`；请使用项目现有可运行环境。
- 若新环境首次运行，按报错安装依赖（常见：`flask`、`python-dotenv`、`requests`）。

## 3.2 启动仿真

```bash
python start.py --name sim-xxx --step 10 --stride 60 --verbose info --log sim-xxx.log
# 例如
python start.py --name sim-test-0425 --step 48 --stride 60 --start 20260425-09:30 --verbose info --log sim-test-0425.log
# 断链重跑
python start.py --name sim-test-0425 --resume --step 12 --stride 60 --verbose info --log sim-test-0425-resume-1.log
```

常用参数：

- `--name`：本次仿真名称（用于结果目录命名）
- `--step`：本次推进步数
- `--stride`：每步对应的分钟数
- `--resume`：从已有同名 checkpoint 继续跑
- `--verbose`：日志级别（`info`、`debug`），新功能增加时没留意这个`debug`模式，不知道有没有用
- `--log`：日志文件名称，在`results/<name>/`目录下，默认`results/<name>/sim-xxx.log`

## 3.3 生成回放数据

```bash
python compress.py --name sim-xxx
```

## 3.4 启动回放页面

```bash
python replay.py
```

浏览器访问：

```text
http://127.0.0.1:5051/?name=sim-xxx
```

## 3.5 结果目录

- `results/checkpoints/<name>/`：每步快照、对话记录等
- `results/checkpoints/<name>/sim-xxx.log`：日志输出，方便debug
- `results/checkpoints/<name>/judge_traces/judge_conversation.json`：[患者-判断LLM-医生] + 评估LLM 的输出结果
- `results\checkpoints\<name>\memory_visualization\`：记忆可视化结果（需运行`visualize_agent_memory.py`）
- `results\external_memory_audit\<name>\`：外置记忆审核结果（需运行`visualize_external_memory_audit.py`）
- `results/checkpoints/<name>/merge_consultation_dialogues/merge_consultation_dialogues.json`：存放对话以及中间产物结果：{[患者-判断LLM-医生] + 评估LLM + 咨询记录} \* 强制干预对话次数（需运行`merge_consultation_dialogues.py`）
- `results/compressed/<name>/`：回放资源（`movement.json`、`simulation.md`）
- `customization\depression_scale_agent\questions`：存放量表评估结果，需在服务器里运行`customization\depression_scale_agent\app.py`。新增可视化**病人完整Prompt注入**、外置记忆系统查询结果

## 4. data/config.json 模块配置

下面给出快速索引，详细值以 `data/config.json` 为准。

## 4.1 agent 配置

- `think.llm`,`associate.embedding`：模型配置
- `agent.think.poignancy_max`：积累到该分数是产生一次反思，值越小反思地越多
- `agent.external_memory`：外置记忆系统配置
  - `read_mode`：仅在强制干预对话中进行外置记忆系统的检索，当前只实现了这个
- `chat_history`/`chat_memory`等：与旧记忆系统有关，不必理会

## 4.2 intervention 配置

- `enabled`：开启定期强制干预对话功能
- `doctor/patients`：医患角色定义
- `meeting_rules`：会诊触发规则（基本上设置为"fast_every_2step"与"fast_every_2step_jgcl"，其他规则未实验过）
- `order_extract`：医嘱提取（感觉该功能暂无用处）
- `session_prompt_injection`：治疗Prompt的注入
- `chat_controls`：强制会话轮次与终止控制
  - `forced_chat_iter`：对话轮数上限（单角色回复次数上限）
- `forced_llm`：**强制干预对话模型设置**，需要在`.env`里配置deepseek api key
- `dialog_judge`：会话中判断LLM
- `session_eval`：会话后评估LLM
  - `history_recent_n`：注入历史reason的数量
- `consult_record`：咨询记录生成与注入
- `depression_update`：抑郁运行态更新（**请设置`enabled`为`false`**，已弃用）
- `depression_dynamic`：动态抑郁人设（**请设置`enabled`为`true`**，赵学长的动态抑郁人设配置）
  - `target_hints`：触发环节
  - 注：请在`frontend\static\assets\village\agents\卡布达\depression_config.json`做出抑郁人设配置
- `memory_injection`：记忆注入规则（当前有初始化注入、某会话完成后注入2种规则）
- `memory_policy`：会话检索权重策略（与旧记忆系统相关，不必理会）

## 4.3 配置注意事项

- `doctor`、`patients`、`meeting_rules`、`chat_controls.agent_overrides` 的角色名必须一致。
- `session_prompt_injection.order` 需与提示词文件中的 session id 对齐。
- 若启用外置记忆，请确认 `agent.external_memory.base_url` 可访问，且服务端接口就绪。可运行`test\live_ec_doll_memory_service_health_ready.py`检查。

## 5. 医患对话主要链路

核心链路可理解为“调度 -> 锁定 -> 对话 -> 判定 -> 收尾 -> 会后处理”。

1. 步进入口  
   `start.py` 每个 step 调用 `InterventionManager.on_step_start(...)`。

2. 会诊调度  
   在 `on_step_start` 内根据 `meeting_rules` 判断触发时机，调用 `_trigger_meeting(...)` 建立会诊任务。

3. 对话前锁定与目标重写  
   `before_agent_think(...)` 根据 lock 重写行动目标。

4. 对话主循环  
   `Agent._chat_with(...)` 执行多轮对话生成，按`是否启用外置记忆系统`选择 `generate_chat.txt` 或 `generate_chat_external.txt`。（只是Prompt模板不一样）

5. 强制判定与终止  
   强制会话下由`会话中判断LLM`和轮次上限控制结束时机。

6. 会后收尾  
   `InterventionManager.after_chat(...)` 清理 lock、更新队列、记录会后状态。

7. 会后扩展处理  
   按配置触发：会话后评估LLM`session_eval`、咨询记录生成`consult_record`、注入外置记忆系统`memory_injection`、更新动态抑郁人设`depression_dynamic` 等链路。

配置与链路映射速查：

- 调度阶段：`meeting_rules`
- 对话阶段：`chat_controls` / `forced_llm` / `dialog_judge`
- 会后阶段：`session_eval` / `consult_record` / `memory_injection` / `depression_dynamic`

## 6. 运行结果与回放

## 6.1 关键输出文件

- `results/checkpoints/<name>/`：每步快照、对话记录等
- `results/checkpoints/<name>/sim-xxx.log`：日志输出，方便debug
- `results/checkpoints/<name>/judge_traces/judge_conversation.json`：[患者-判断LLM-医生] + 评估LLM 的输出结果
- `results\checkpoints\<name>\memory_visualization\`：记忆可视化结果（需运行`visualize_agent_memory.py`）
- `results\external_memory_audit\<name>\`：外置记忆审核结果（需运行`visualize_external_memory_audit.py`）
- `results/checkpoints/<name>/merge_consultation_dialogues/merge_consultation_dialogues.json`：存放对话以及中间产物结果：{[患者-判断LLM-医生] + 评估LLM + 咨询记录} \* 强制干预对话次数（需运行`merge_consultation_dialogues.py`）
- `results/compressed/<name>/`：回放资源（`movement.json`、`simulation.md`）

## 6.2 conversation.json 说明

- 记录每个仿真时刻发生的会话内容。
- 常用于排查“某一步是否发生对话、对话文本是什么、谁与谁在说话”。

## 6.3 judge_conversation.json 说明

- 位于 `judge_traces/` 目录，由仿真过程自动写出
- 用于审计强制会话中`会话中判断LLM`以及`会话后评估LLM`的输出

## 6.4 visualize_agent_memory.py 作用

该脚本用于离线可视化某个角色的记忆数据（`event/thought/chat`）：

- 读取来源：
  - 快照文件（`simulate-*.json`）中的 memory 引用
  - `storage/<agent>/associate/docstore.json`
  - `storage/<agent>/associate/default__vector_store.json`
- 输出产物：
  - `memory_visualization/*.json`
  - `memory_visualization/*.csv`
  - `memory_visualization/*.html`

基本使用方式：

1. 编辑 `visualize_agent_memory.py` 顶部配置区（`CHECKPOINT_DIR`、`SNAPSHOT_FILE`、`AGENT_NAME`）。
2. 运行：

```bash
python visualize_agent_memory.py
```

## 6.5 visualize_external_memory_audit.py 作用

该脚本用于审计外置记忆系统的调用情况、可视化某个角色的记忆数据

- 输出产物：
  - `memories.csv`
  - `relations.csv`
  - `report.html`
  - `report.json`
  - `timeline.csv`
    基本使用方式：基本同上

## 6.6 merge_consultation_dialogues.py 作用

该脚本用于将强制会话相关的对话以及中间产物进行合并，输出到一个文件中，方便查看和分析。

- 输出产物：
  - `merge_consultation_dialogues.json`
- 基本使用方式：基本同上
-

## 6.7 analyze_depression_dynamic_state.py 作用

该脚本用于分析动态抑郁人设系统的状态变化情况，输出每个step的状态数据，方便查看和分析。(具体由赵学长更新)

## 7. 常见问题与排障

## 7.1 仿真名称冲突 / 恢复失败

- 新建仿真时名称**已存在**会被要求重输。
- resume模式下若找不到对应目录，会提示目录不存在。
- 建议先检查 `results/checkpoints/<name>/` 是否存在。

## 7.2 回放提示缺少 movement.json

- 说明还未执行压缩步骤。
- 先运行 `python compress.py --name <name>`，再访问 replay 页面。

## 7.3 强制会诊未触发

重点检查：

- `intervention.enabled` 是否开启
- `meeting_rules` 是否有 `enabled=true` 且触发条件匹配当前 step/time
- `doctor/patient` 名称是否与角色配置一致

## 7.4 外置记忆是否生效

重点检查：

- `agent.external_memory.enabled=true`
- `base_url` 可访问
- 日志中是否出现 `[EXT_MEMORY_RETRIEVE]` 与 `[EXT_MEMORY_CHAT_ROUTE] route=external`
- 运行`test\live_ec_doll_memory_service_health_ready.py`检查

## 7.5 日志定位建议

- 对话内容：`conversation.json`
- 判定轨迹：`judge_traces/judge_conversation.json`
- 全链路审计：`results/checkpoints/<name>/<name>.log`
- 外置记忆审核：`results/external_memory_audit/<name>/`
- 强制干预会话对话记录及中间产物：`results/checkpoints/<name>/merge_consultation_dialogues/merge_consultation_dialogues.json`

## 8. 维护约定（建议）

- 每次改动 `intervention` 或 `config`，同步更新 README 的“更新日志（近期）”。
- 新增配置项时，在 README 中补“作用 + 关键参数 + 风险提示”。
- 不在 README 示例中写真实密钥、令牌、私有地址。
