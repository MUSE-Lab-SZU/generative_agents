# 动态抑郁模块操作指南（独立双路径状态转换）

## 1. 改造目标

本次改造将状态转换逻辑从“规则触发主导、LLM 仅辅助重加权”升级为“规则路径与 LLM 路径独立建模，再进行概率融合”。

目标是解决以下问题：

1. 规则触发词未命中时，原模型无法利用高质量语义信号推动状态变化。
2. 规则信号与 LLM 信号耦合过紧，难以单独调参与诊断。
3. 状态转换过程缺少明确的路径拆分，代码可读性与可维护性不足。

## 2. 新的转换算法

对每个可达目标状态 `next_state`，分别计算：

1. `P_rule`：规则路径概率（依赖触发词命中与累积强度）。
2. `P_llm`：LLM 路径概率（依赖 `positive_score/negative_score` 与 `signal_weight`）。

最终融合概率为：

`P_final = 1 - (1 - P_rule) * (1 - P_llm)`

该形式可解释为两条独立证据路径并联后的联合命中概率，满足“路径独立、共同影响”的设计要求。

## 3. 代码落点

核心实现位于 `modules/depression/state_machine.py`：

1. `TransitionCandidate`：统一候选状态的数据结构。
2. `update_state(...)`：主流程改为“构建候选 -> 选择最大 `P_final` -> 随机采样判定”。
3. `_build_transition_candidates(...)`：对每个目标状态分别计算 `P_rule`、`P_llm`、`P_final`。
4. `_estimate_rule_path_probability(...)`：规则路径概率。
5. `_estimate_llm_path_probability(...)`：LLM 独立路径概率。
6. `_combine_independent_probabilities(...)`：独立路径概率融合函数。

## 4. 关键参数说明

状态机参数（`SymptomStateMachine` 类常量）：

1. `transition_sensitivity`（实例字段）：规则路径整体敏感度。
2. `RECOVERY_BOOST`：恢复方向转移增强因子。
3. `LLM_SIGNAL_WEIGHT`：LLM 默认权重。
4. `LLM_INDEPENDENT_PATH_SCALE`：LLM 独立路径整体增益。
5. `LLM_INDEPENDENT_PATH_BASELINE`：LLM 路径基础下限，避免弱基线状态下完全失效。
6. `MAX_ADJUSTED_PROBABILITY`：候选概率上限。

## 5. LLM API 配置

### 5.1 优先使用 forced_llm（影响 `llm_transition_signal` 生成）

`llm_transition_judge` 采用“forced_llm 优先，_llm 回退”的调用策略。

编辑 `data/config.json` 的 `intervention.forced_llm`：

```json
"forced_llm": {
  "enabled": true,
  "provider": "openai",
  "model": "deepseek-chat",
  "base_url": "https://api.deepseek.com",
  "api_key_env": "DEEPSEEK_API_KEY",
  "retry": 2,
  "temperature": 0.5
}
```

说明：
- 优先读取 `forced_llm` 运行时配置并创建 `agent._forced_llm`。
- 若 `forced_llm` 配置缺失或不可用，自动回退到 `agent._llm`，不阻断链路。

### 5.2 开启 LLM 转换判定

编辑目标 agent 配置：
`frontend/static/assets/village/agents/<agent_name>/depression_config.json`

在 `depression_simulation` 下加入：

```json
"llm_transition_judge": {
  "enabled": true,
  "min_confidence": 0.60,
  "max_text_length": 1600,
  "signal_weight": 0.25,
  "allowed_event_keys": ["chat_event"]
}
```

动态抑郁模块建议仅保留 `chat_event`，只关注心理语义事件。

## 6. 启用步骤

1. 在 `data/config.json` 打开 `intervention.depression_dynamic.enabled=true`。
2. 在 agent 的 `depression_config.json` 打开 `depression_simulation.enabled=true`。
3. 按 5.1 配好 `forced_llm`（建议 DeepSeek）。
4. 按 5.2 开启 `llm_transition_judge`。
5. 运行：

```bash
python start.py --name sim_dual_path --step 30 --stride 10 --verbose debug
```

## 7. 验证清单

查看日志关键字：

1. `[DEPR_DYNAMIC][INIT] ... enabled=true`
2. `[DEPR_DYNAMIC][LLM_TRANSITION] ... accepted=true`
3. `[DEPR_DYNAMIC][EVENT] ... llm_signal=true`

判断独立路径生效的经验标准：

1. 某些轮次无规则触发词命中仍出现状态转换。
2. 状态转换触发记录中可见 `llm_signal` 或 LLM 匹配触发词。
3. 在同样规则输入下，仅调整 `signal_weight` 时，转移率可观察变化。

## 8. 调参建议

1. `min_confidence`：
   - 过低会引入噪声。
   - 过高会让 LLM 路径近乎失效。
   - 建议区间：`0.55 ~ 0.75`。
2. `signal_weight`：
   - 建议从 `0.20 ~ 0.35` 起步。
3. `LLM_INDEPENDENT_PATH_SCALE`：
   - 想增强 LLM 影响可从 `2.0` 提到 `2.5`。
   - 想更保守可降到 `1.2 ~ 1.8`。
4. `transition_sensitivity`：
   - 提高会同步增强规则与 LLM 路径基线。

## 9. 风险与边界

1. 模型输出不稳定时，LLM 路径会带来额外波动。
2. 如果对话文本很短或信息密度低，LLM 路径贡献可能长期偏弱。
3. `llm_transition_judge` 已移除 `timeout_ms` 参数，当前不再维护该配置项。

## 10. 建议的实验方案

1. 固定随机种子与初始状态，运行 A/B 两组：
   - A 组：`llm_transition_judge.enabled=false`
   - B 组：`llm_transition_judge.enabled=true`
2. 统计 100+ 轮内：
   - 总转移次数
   - 恶化/恢复方向比例
   - 危机态停留时长
3. 用同一输入重复三次，观察方差并回收调参。
