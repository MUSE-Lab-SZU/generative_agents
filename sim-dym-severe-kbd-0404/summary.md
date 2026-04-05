# 动态抑郁人设变化量化摘要

- 存档：`sim-dym-severe-kbd-0404`
- 角色：`卡布达`
- 医生：`蜻蜓队长`
- 配对键：`蜻蜓队长::卡布达`
- 快照范围：`simulate-20260404-0930.json` -> `simulate-20260406-0830.json`
- 步数范围：`1` -> `48`

## 零、直观评分（优先看这里）
- wellbeing_index：`35.25` -> `29.298`（0-100，越高越好）
- risk_index：`64.75` -> `70.702`（0-100，越高风险越大）
- improvement_vs_start：`-5.952`（>0 改善，<0 恶化）
- 状态判定：`轻度恶化`

## 一、动态抑郁状态变化
- dynamic_interaction_count：`0` -> `15`（增量 `15`）
- dynamic_state（出现过）：`severe_episode`
- dynamic_state 分布：severe_episode:48
- 状态严重度索引：`3` -> `3`（严重度未变化）
- dynamic_overall_stress 统计：min=0.700, max=0.750, mean=0.716
- 环境压力值（末值）：stress=`0.7` / comfort=`0.30000000000000004` / trigger_potential=`0.5`

## 二、depression_dynamic_state 数值核心（accumulated_triggers）
- support 总量：`0.000` -> `11.000`
- 负向触发总量：`0.000` -> `31.000`
- support-负向 平衡值：`0.000` -> `-20.000`
- 分项变化：
- `stress`：`0.000` -> `11.000`（增量 `11.000`）
- `support`：`0.000` -> `11.000`（增量 `11.000`）
- `学业失败`：`0.000` -> `3.000`（增量 `3.000`）
- `未来焦虑`：`0.000` -> `6.000`（增量 `6.000`）
- `社交压力`：`0.000` -> `1.000`（增量 `1.000`）
- `自我价值`：`0.000` -> `6.000`（增量 `6.000`）
- `自杀意念`：`0.000` -> `4.000`（增量 `4.000`）

## 三、强制干预对话效果（可观测）
- session_current_index：`-1` -> `12`
- session_current_session：`session1` -> `session6.3`
- session_completed：`False` -> `True`
- consult_pair_records_total：`0` -> `14`
- intervention_lock_enabled：`False` -> `False`

## 四、任务与积压
- forced_tasks_total：`0` -> `12`
- forced_tasks_pending（末值）：`5`
- forced_tasks_scheduled（末值）：`7`
- forced_tasks_overdue_open（末值）：`12`

## 五、关键事件计数
- dynamic_interaction_increment:15, session_index_advanced:12, session_completed:1, lock_released:1

## 六、建议重点查看
- `dynamic_state_timeline.csv`：专门看 `depression_dynamic_state` 数值变化（包含每个 trigger 的累计值和单步增量）。
- `timeline.csv`：全链路时间线（动态抑郁 + 强制干预）。
- `events.csv`：关键跳变点（会话推进/完成、锁释放、动态互动增长）。
- `depression_state_deep_dive.png`：动态抑郁状态深度图（状态数值变化主图）。
- `persona_effect_plots.png`：原四联图（干预链路效果主图）。
