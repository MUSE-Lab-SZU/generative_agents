# experiment_eval

用于生成式 Agent 实验的统一统计与绘图包。它替代原 `kbd_repeat/`，把“重复量表画图工具”扩展为：

- 冻结快照 K 次量表复评的严格校验、总分/条目长表、总分 ICC 和条目 Weighted Kappa；
- 独立外层仿真 run 的轨迹、AUC、nadir/rebound、PHQ/BDI 收敛性；
- 外层 run 的时间点差异、终点 contrast、Hedges’ g 与 waterfall；
- 会谈 turns/字符剂量、CBT prompt 完成位置、阶段停留、环境任务；
- PHQ/BDI item 9 与恶化阈值等自动安全代理；
- 默认输出 300 DPI PNG 图，并同时生成 CSV、JSON、Markdown 和数据字典。

绘图代码集中在 `experiment_eval/visualization/`：

- `scale_plots.py`：单 run 轨迹、测量 CI/SD、条目可靠性与附录图；
- `core_plots.py`：外层 contrast、PHQ/BDI 收敛、ICC、waterfall、剂量和 CBT 阶段图。
- `weighted_kappa_plots.py`：repeat-pair Kappa 热图、条目 forest，以及覆盖充分时的分层热图。
- `cross_persona_plots.py`：persona × scale 轨迹、persona-specific forest、结果/过程热图、LOPO 和描述性方差分解。

`analysis_api.py` 为历史 0712/0716 定制分析脚本提供小型公共接口，避免继续依赖已经移除的单体绘图脚本。

`cross_persona.py` 在同一 `ExperimentRecord` 层级上增加跨人设汇总。它始终以独立 outer run 为效应量和区间单位；K 次量表生成只进入 frozen-snapshot 均值和测量噪声分量。

## 统计单位

- `outer_run_id`：一次完整独立仿真实验，是组间描述和效应量的单位。
- `snapshot_id`：某次仿真在某时间点冻结的状态。
- `measurement_repeat_id`：同一 snapshot 的重复量表生成；K 不能当独立样本量。

Weighted Kappa 只用于 0–3 有序条目：对象按同一 `snapshot_id × scale × item`
严格配对，measurement repeat 视为评分者。主结果使用 quadratic weights，linear
weights 为敏感性结果；总分继续使用 ICC。单一类别导致期望分歧为 0 时输出 NA，且
所有 Kappa 结果都应结合类别分布解释。

## 时间点口径

`session_20` 只表示第 20 次咨询后的观察点，不自动等于 POST。CBT session prompt 完成后的固定回访提示词仍会触发强制医患咨询，因此：

```text
true_delayed_followup = false
```

只有未来在停止强制干预后继续仿真并冻结新时间点，才能分析真正的维持/延迟反弹。

## 入口

```bash
python docs/analysis/run_experiment_evaluation.py --help
```

0727 KBD2 G1/G4：

```bash
python docs/analysis/run_experiment_evaluation.py \
  --reports-dir results/experiment_data-0727-KBD2/experiment_data/reports \
  --experiment-data-root results/experiment_data-0727-KBD2 \
  --checkpoints-root results/checkpoints-0727-KBD2 \
  --out-dir docs/experiment_evaluation_0727/0727_kbd2_g1_g4 \
  --dataset-label 0727-KBD2-progressive-D \
  --group G1 --group G4 \
  --batch-mode compare-all --report-mode presentation \
  --outer-summary mean-ci --y-axis adaptive
```

0718 其他条件：

```bash
python docs/analysis/run_experiment_evaluation.py \
  --reports-dir results/experiment_data/reports \
  --out-dir docs/experiment_evaluation_0727/0718_treatment_context_controls \
  --dataset-label 0718-treatment-context-controls \
  --kbd KBD2 --kbd KBD4 --kbd KBD6 \
  --group G1 --group G4 --group G6 --group G7 \
  --batch-mode compare-all --report-mode core
```

跨人设论文图：

```bash
python docs/analysis/run_cross_persona_analysis.py
```

也可直接运行模块并覆盖输入/输出目录：

```bash
python -m experiment_eval.cross_persona_cli --help
```

默认复用 0727 KBD2/KBD4/KBD6 的 G1/G4 reports 和 0718 的 G1/G3/G4/G5/G6/G7/G9 reports，输出 3×2 轨迹图、G1−G9 森林图、persona × group 结果/过程热图、LOPO、简化方差分解及对应 CSV。0718 每个 persona × group 只有 2 个 outer run，因此除轨迹展示外的异质性结果均按探索性输出。

## 完成实验的安全空间清理

`prune_archive.py` 会把已经完成的存档精简为 `experiment_eval` 统计和绘图所需文件。它直接调用当前版本的严格 report loader 与 process/stage extractor 验证临时副本，因此后续指标代码增加新的必需输入时，应同步更新该脚本和测试。

如果经常处理同一套目录，可以直接编辑 `prune_archive.py` 顶部的“用户默认配置区”：

```python
RESULTS_ROOT = PROJECT_ROOT / "results"
RESULTS_ARCHIVE = ""
CHECKPOINTS_ARCHIVE = "checkpoints-0727-KBD4"
EXPERIMENT_DATA_ARCHIVE = "experiment_data-0727-KBD4"
OPERATION_MODE = "prune"
BACKUP_OUTPUT_DIR = ""
BACKUP_COMPLETED_ONLY = False
BACKUP_INCLUDE_RUN_PREFIXES = ()
APPLY_CHANGES = False
KEEP_BACKUP = False
STAGING_PARENT = ""
```

配置后直接运行即可：

```bash
python -m experiment_eval.prune_archive
```

`OPERATION_MODE` 有两种取值：

- `"prune"`：默认模式，用精简副本替换源存档；校验成功后按 `KEEP_BACKUP` 决定是否删除旧完整目录。
- `"backup"`：非破坏模式，源存档完全不变，把精简后的必要内容复制到一个尚不存在的 `BACKUP_OUTPUT_DIR`。

如果当前只需要统计和绘制已经生成 complete repeat summary 的实验，可在 backup
模式启用：

```python
BACKUP_COMPLETED_ONLY = True
```

或在命令行增加 `--completed-only`。此时只复制 complete summary、对应 run 的最终
checkpoint、会谈 trace 和必要 provenance；不会完整救援 incomplete/unreported run，
也不会复制 `*_answered.jsonl`、`*_scored.json`、`*_trace.json` 等量表原始审计文件。
该选项禁止用于 `prune`，避免从源存档误删尚未完成的实验。

同一个 `run_name` 若同时存在旧 `_incomplete.json` 和后来生成的 complete summary，
complete summary 优先；旧 incomplete 文件会显示为 superseded，不再触发完整救援。

只提取已完成的回访 repeat 时，可再按 `run_name` 前缀筛选：

```python
BACKUP_COMPLETED_ONLY = True
BACKUP_INCLUDE_RUN_PREFIXES = ("followup-",)
```

对应命令行参数是 `--include-run-prefix followup-`。该参数可以重复传入以允许多个
前缀，并且只能和 `--mode backup --completed-only` 一起使用。筛选以 summary 内的
`run_name` 为准；不匹配的普通 complete run 及其 summary 不会进入新备份。如果同一
summary 文件同时包含匹配与不匹配的 run，脚本会拒绝拆分该报告，避免静默改写报告。

远程智算平台建议先使用 `backup`。例如：

```python
OPERATION_MODE = "backup"
BACKUP_OUTPUT_DIR = "retained-backups/KBD4-0727"
APPLY_CHANGES = False
```

相对 `BACKUP_OUTPUT_DIR` 从 `RESULTS_ROOT` 解析。首次仍保持 `APPLY_CHANGES = False`
检查计划，确认无误后临时传 `--apply`，或再将它设为 `True`。目标目录必须尚不存在，
且不能位于源存档内部或与源 checkpoints/experiment_data 目录重叠。

命令行参数优先于顶部配置。例如顶部设为 `APPLY_CHANGES = True` 时，仍可用 `--no-apply` 临时强制 dry-run；`--no-keep-backup` 同理。建议顶部长期保留 `APPLY_CHANGES = False`，每次先检查计划，再用 `--apply` 正式执行。

不修改顶部配置也可以直接执行非破坏备份：

```bash
python -m experiment_eval.prune_archive \
  --results-root results \
  --checkpoints-archive checkpoints-0727-KBD4 \
  --experiment-data-archive experiment_data-0727-KBD4 \
  --mode backup \
  --backup-output retained-backups/KBD4-0727

# 确认 dry-run 后再追加 --apply
```

若只要已有 complete repeat summary 的画图数据，在上述命令末尾增加
`--completed-only`。

备份根目录下会保留两个源目标目录的名称及内部相对结构。例如上面的输出为
`retained-backups/KBD4-0727/checkpoints-0727-KBD4/` 和
`retained-backups/KBD4-0727/experiment_data-0727-KBD4/`。因此仍可把这两个目录
作为分离存档传回本工具或其他评估命令。若输入是传统合并存档，则输出根目录下
直接是 `checkpoints/` 和 `experiment_data/`。

始终先执行只读 dry-run：

```bash
python -m experiment_eval.prune_archive \
  --results-root results \
  --checkpoints-archive checkpoints-0727-KBD4 \
  --experiment-data-archive experiment_data-0727-KBD4
```

这里的三类路径是：

- `--results-root`：存档名称的解析基准，默认是仓库的 `results/`；
- `--checkpoints-archive`：checkpoints 存档名或直接路径；
- `--experiment-data-archive`：experiment_data 存档名或直接路径。

相对存档名从 `--results-root` 下解析；绝对路径可以直接传入。对于本身同时包含 `checkpoints/` 和 `experiment_data/` 的传统合并存档，改用：

```bash
python -m experiment_eval.prune_archive \
  --results-root results \
  --results-archive path/to/archived-results
```

确认 dry-run 中的 report 数、run 数、incomplete rescue 和预计保留体积后，增加 `--apply`：

```bash
python -m experiment_eval.prune_archive \
  --results-root results \
  --checkpoints-archive checkpoints-0727-KBD4 \
  --experiment-data-archive experiment_data-0727-KBD4 \
  --apply
```

正式清理采用以下顺序：

1. 在临时目录构建精简副本；
2. 对每个文件做大小和 SHA-256 比对；
3. 重新加载全部 complete repeat summary；
4. 比较精简前后的 outcome、dose/process 和 progressive-stage 输入；
5. 将原目录重命名为备份，安装精简副本并再次校验；
6. 校验通过后删除旧完整目录；使用 `--keep-backup` 可暂不删除。

默认保留：

- `experiment_data/reports/` 全部报告；
- 每个 complete run 的最后一个 `simulate-*.json`；
- checkpoint 与 experiment_data 中存在的 `judge_conversation.json`；
- `cbt_condition_manifest.json`、`trial_meta.json` 和 `batch_state`；
- `*_answered.jsonl`、`*_scored.json` 及同目录 `*_trace.json`；
- wrapper 目录之外的存档级元数据。

任何 incomplete repeat 对应的 run 都会完整保留 checkpoints 和 experiment_data，以便补跑。存在 checkpoint/run 但没有 repeat report 的目录也会作为 unreported rescue 完整保留，避免误删仍在运行或尚未汇总的实验。

只有成功执行 `--apply` 且未使用 `--keep-backup` 时，旧的完整 snapshot、storage、staged job、sidecar、forced prompt trace 和日志才会被永久删除。精简后仍能重画现有图，但 complete run 不再支持换模型重新复评历史冻结状态。

报告中的所有结果均为 in-silico Agent 指标，不代表真人临床疗效。
