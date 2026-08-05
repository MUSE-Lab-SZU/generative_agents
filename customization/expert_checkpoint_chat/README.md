# 专家治疗师访谈（Checkpoint 沙箱）

启动：

```bash
python customization/expert_checkpoint_chat/app.py
```

页面按“模拟存档 → checkpoint → Agent”选择加载源。专家身份固定为“治疗师”。

- `results/checkpoints` 只读；每次网页会话会把 `storage` 复制到 `sessions/`，供联想记忆和动态抑郁状态在会话内演化。
- 动态抑郁 Agent 仍会执行 `reset()`，因此会补全主诉图窗口。
- 每次新会话在 `records/` 创建 JSONL；发送首条消息后可用“下载本次 JSONL 记录”按钮下载。

`sessions/` 和 `records/` 是运行时数据，不应提交到版本库。
