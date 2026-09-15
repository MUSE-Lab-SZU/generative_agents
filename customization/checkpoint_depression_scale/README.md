# Checkpoint 总体抑郁量表测试

在项目已有的 Python 环境（包含 Gradio、llama-index 及模型依赖）中，从项目根目录启动：

```bash
python customization/checkpoint_depression_scale/app.py
```

打开终端输出的 Gradio 地址，选择 `results/checkpoints` 下的模拟存档（例如 `sim0805`）、agent，以及一个或多个 `simulate-*.json` 存档点，点击“开始量表测试”。也可以一次选择该 agent 的全部存档点。页面显示逐题进度、各存档点的单题分数、总分和原始回答，并提供结果下载。

问卷位于本模块的 `总体抑郁水平及干扰程度量表.jsonl`，从 `docs/总体抑郁水平及干扰程度量表.pdf` 转录，沿用 `id` / `question` 格式，保留 5 题及每题的全部 0–4 分选项。每个 prompt 以 checkpoint 的模拟时间为基准询问过去一周的体验。

## 结果与计分

每次测试创建一个新文件，不覆盖已有结果：

```text
results/checkpoints/<模拟存档>/depression-scale-<run_id>.jsonl
```

每行对应一个存档点，完成或失败后立即写入并刷新磁盘。记录包含模拟存档、agent、checkpoint 文件名、模拟时间与步数、量表文件 SHA-256、各题 prompt、全部原始回答及重试、单题分数、总分、运行时间和错误信息。

- 接受明确的 `评分：2。理由：……` 或单个 `2` 等回答，合法整数范围为 0–4。
- 无法识别时重新询问本题一次；不通过另一个模型猜测答案分数。
- 5 题全部有效时，`status` 为 `completed`，`total_score` 为五题之和（0–20）。
- 重试后仍有无效回答时，`status` 为 `incomplete`；加载或模型调用异常时为 `error`。两者的 `total_score` 均为 `null`，已经得到的答案仍保留。一个点失败后继续其他点。
- 结果只报告题目分数及总分，不添加问卷未给出的严重程度阈值。

## 存档恢复与隔离

每个 checkpoint 都重新创建运行时，复制该 agent 的 `storage` 到临时目录，结束时清理；原始 checkpoint、记忆存储和 `conversation.json` 不会被本模块修改，模拟目录只增加测试结果 JSONL。网页会话串行调用模型，避免进程全局模拟时钟和游戏状态混用。

加载快照中的角色状态、日程、记忆引用和动态抑郁状态，模型与抑郁配置路径按当前项目重新定位。调用现有 `agent.reset()` 初始化模型及主诉图窗口，并沿用现有 `generate_chat` 对话逻辑。五题及格式重试属于同一场访谈，动态抑郁状态会在该场访谈内演化，后续存档点不会继承本次测试状态。新增回答只留在本场访谈上下文，不另行写入长期记忆。

现有存档使用模拟目录下共享的 `storage`，没有每个 checkpoint 独立的完整向量库备份。加载时原有记忆逻辑会按模拟时钟过滤未来及过期节点；快照引用中缺失的节点会列入 `runtime.missing_memory_references`，供判断恢复完整性。若有记忆引用却完全缺少 agent 存储目录，则该点直接报错。为避免读取未来对话，不加载覆盖整个模拟的 `conversation.json`，历史对话通过 checkpoint 的记忆恢复。

需要可用的模型及 embedding 服务，沿用项目现有配置。

## 验证

```bash
python -m unittest discover -s customization/checkpoint_depression_scale -p 'test_*.py' -v
```

测试使用模拟回答验证重试、计分、失败继续、结果落盘和临时存储隔离，不调用真实模型。
