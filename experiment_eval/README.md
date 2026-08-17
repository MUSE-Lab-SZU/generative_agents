# experiment_eval

生成式 Agent 实验的统一统计、报告与论文画图包。当前结构遵循“一种图一个目录”：实际绘图实现统一位于 `charts/<family>/<chart_id>/plot.py`，公共 Matplotlib 基础设施位于 `charts/shared/`。旧的 `experiment_eval.visualization.*` import 由 `compat/visualization/` 提供兼容。

> 统计单位是独立 outer simulation run。同一冻结快照的 K 次量表生成只用于测量稳定性，不是 K 个独立样本。所有输出均为 in-silico Agent 指标，不代表真人临床疗效。

## 默认出图政策：面向汇报的精简图集

除非用户明确要求“全量诊断图”“逐 run 质控”或指定某个旧图族，后续实验画图必须先做汇报价值筛选，不得直接运行会展开全部历史兼容图的组合（尤其不要使用 `--batch-mode all --report-mode all`）。默认目标是 **10–20 张、最多 30 张 canonical PNG**；同内容的 PDF/SVG 不计入张数，但也只在论文或排版确有需要时导出。

默认规则如下：

1. **禁止自动生成 `by_repeat/`。** 单个 outer repeat 不能支持实验条件或人设比较；`--batch-mode by-repeat/all` 只允许用于用户明确要求的 run 级质控，不进入常规汇报图集。
2. **禁止按单个 outer run 批量展开旧图族。** 不把“单个实验条件的单个重复”作为主要纵轴、分面或一整套子目录；不为每个 group/persona/repeat 重复渲染相同 chart family。
3. **历史兼容附录和 presentation 图默认关闭。** `01_*`–`07_*`、`presentation_00_*`–`presentation_04_*` 以及其他只因旧 pipeline 存在而衍生的变体，除非回答了新的、明确的研究问题，否则不生成。单 run waterfall、单 run 热图和纯个体轨迹仅在异常检查、安全审计或用户点名时生成。
4. **优先保留跨实验条件、跨人设的聚合图。** 默认候选包括：persona × scale 联合轨迹、条件/人设终点变化及不确定性、必要的批次一致性图、具有明确问题的过程或条目图，以及 parent-linked followup 的维持/反弹图。每张图必须回答不同问题；箱线图、柱图、forest、heatmap 若信息重复，只保留最适合汇报的一种。
5. **合并重复时以 pooled cell 为主要单位。** 图中标明每格真实 outer-run `n`，可用淡线/散点展示个体 runs，但不得把每个 repeat 拆成独立“比较对象”，也不得用 measurement repeats 或 followup branches 扩大样本量。
6. **先列选择，后渲染。** 在 manifest/README 中记录拟生成的 canonical 图及其研究问题；批量渲染前删除重复作用的图型。若预计超过 30 张，必须进一步筛选，或先向用户说明为什么确有必要。
7. **交付时必须列出未画图。** 最终报告和回复都要单列“未生成的图及原因”，至少覆盖：因汇报价值低而主动省略、与已选图重复、样本量/设计不满足、数据字段缺失四类；不能只汇报已经生成的图。
8. **历史文件名不代表本次数据范围。** 例如 `figure_01_0727_persona_scale_trajectory.png` 的 `0727` 是兼容旧 API 的文件名，不说明输入来自 0727。manifest、标题或报告必须写明实际数据集和 `n`；新工作流优先使用不含旧日期的 canonical 别名，避免误读。

推荐的默认汇报图集顺序是：研究设计/覆盖 1 张、主要结局 2–4 张、跨人设 1–3 张、批次敏感性 1–2 张（仅跨批次合并时）、followup 2–4 张（仅 lineage 完整时）、过程/条目/可靠性各 0–2 张。旧图仍保留为可调用能力，而不是默认批量产物。

## 从哪里运行

主入口：

```bash
python -m experiment_eval --help
```

常用命令：

```bash
# 默认 reports 目录，只生成适合汇报的 core 图与报告
python -m experiment_eval

# 指定输入、过程数据与输出
python -m experiment_eval \
  --reports-dir results/experiment_data/reports \
  --experiment-data-root results/experiment_data \
  --checkpoints-root results/checkpoints \
  --out-dir docs/experiment_evaluation \
  --batch-mode compare-all --report-mode all

# 显式选择简洁轨迹 / 核心图 / 历史附录图
python -m experiment_eval --report-mode lines-only
python -m experiment_eval --report-mode core
python -m experiment_eval --report-mode appendix
```

历史入口 `python docs/analysis/run_experiment_evaluation.py` 与 `python -m experiment_eval.cli` 仍然兼容，但新脚本应使用 `python -m experiment_eval`。

专用入口：

```bash
python -m experiment_eval.cross_persona_cli --help
python -m experiment_eval.interpretive_analysis --help
python -m experiment_eval.stratified_interpretive --help
python -m experiment_eval.scale_credibility --help

# 同一输出目录分别生成 KBD2 跨条件与 G1 跨人设的可信度图
python -m experiment_eval.scale_credibility --reports-dir <kbd2-reports> \
  --out-dir <output> --dimension condition --filename-suffix kbd2_by_condition
python -m experiment_eval.scale_credibility --reports-dir <g1-reports> \
  --out-dir <output> --dimension persona --filename-suffix g1_by_persona
```

## 输入与输出

主入口自动发现 `--reports-dir` 中名称包含 `repeat-`、以 `_summary.json` 结尾的 repeat summary。也可重复传入 `--report-file` 精确选择文件。

可选输入：

- `--experiment-data-root`：conversation、judge、simulation event 等过程数据；
- `--checkpoints-root`：CBT stage、prompt 完成、主诉评估节点等 checkpoint 数据；
- `--long-data`：逐 run、逐时间点、逐量表条目的 long/tidy CSV/JSON/JSONL；
- `--feature-data`：一行一个独立 run 的 persona/baseline/behavior 特征宽表；
- `--persona-profile-data`：persona × group × timepoint × scale 的 run-level long 表。

### 从 results 存档构建规范数据接口

历史存档先用独立入口整理，再交给画图/统计入口。该入口会校验严格 repeat summary、区分根 outer run 与 follow-up branch，并且只在存档有明确 `parent_run_name + parent_label` 时把随访接回根 run：

```bash
python -m experiment_eval.archive_data \
  --archive-root 'results/0808实验存档示例' \
  --out-dir docs/experiment_evaluation/0808_archive_interfaces \
  --expected-condition G1 \
  --expected-condition G2
```

输出接口：

| 文件 | 统计单位 / 用途 |
| --- | --- |
| `archive_run_catalog.csv` | 一行一个独立根 outer run；保存 persona、实验条件、协议/模型 provenance、QC 证据和原始过程数据覆盖。 |
| `archive_outcomes_long.csv` | 根 run × timepoint × scale；可直接作为 `--persona-profile-data`，包含真实 `sim_time`、elapsed day、评分来源和 follow-up lineage。 |
| `archive_items_long.csv` | 根 run × timepoint × scale × item；可直接作为 `--long-data`。10 次 frozen-snapshot 测量已折叠为条目均值。 |
| `archive_baseline_features.csv` | 一行一个根 run；可直接作为 `--feature-data`，当前只导出无泄漏的 baseline 总分/条目和 persona one-hot。 |
| `archive_followup_lineage.csv` | 一行一个 follow-up branch；区分已链接、有无 repeat 评分及 parent timepoint。 |
| `archive_engagement_events.csv` | 一行一次已观察 conversation；含目标角色参与、双方 turn/字符数、interaction/session/rule/source 和可审计 initiator。 |
| `archive_engagement_coverage.csv` | 一行一个根 run；区分有效 0 与原始日志 unavailable。 |
| `archive_condition_coverage.csv` | persona × condition × scale × timepoint 的覆盖表；输出根 run/outcome/item 的实际 n 和缺失状态。完全没有出现的条件须通过 `--expected-condition` 声明。 |
| `archive_data_availability.json` | 字段覆盖、不可恢复项、分析可行性和样本量闸门。 |
| `archive_interface_manifest.json` | schema、统计单位、分组维度和上述文件索引。 |

这里的默认分组维度只有 `persona` 和 `condition`。`SEV` 是本存档所有 run 共有的历史条件 token，仅保留为 `archived_severity_label` 供审计，不作为严重程度分组或 study/setting。主入口的自动 report discovery 也会排除有明确 provenance 的 follow-up repeat summary，避免把随访分支误算成新的 outer run；需要纵向随访时使用上述 archive 接口。

默认输出目录为 `docs/experiment_evaluation/`。每个 batch 可包含：

- 300 DPI PNG；分面箱线图还支持 PDF/SVG；
- 图对应的规范化 CSV 和统计 CSV；
- `statistics.json`、Markdown 报告、数据字典、图表索引；
- `batch_manifest.json` 或图类型自己的 manifest/feasibility audit。

图片文件名、统计口径和 CLI 参数保持兼容；旧的 `charts.*` 与 `visualization.*` 绘图 import 继续映射到新目录。已经确认无仓库内调用的根目录数据/统计 re-export 和旧分面图 CLI 已移除。

## 目录结构

```text
experiment_eval/
├── __main__.py                 # python -m experiment_eval
├── cli.py / cli_config.py      # 主入口与参数
├── pipeline.py                 # 数据、统计、图表批次编排
├── loader.py / process.py      # 公共数据读取和过程数据提取
├── statistics.py / reports.py  # 公共统计与报告
├── data/                       # 新图的数据读取、规范化与提取实现
├── analysis/                   # 新图的统计、效应量与模型实现
├── charts/
│   ├── shared/                 # 跨图公共绘图组件
│   ├── outcomes/
│   │   ├── trajectory_lines/
│   │   │   ├── __init__.py
│   │   │   └── plot.py
│   │   └── change_ci/
│   ├── persona/                # persona 轨迹、forest、profile、SHAP
│   ├── symptoms/               # symptom trajectory/forest/network
│   ├── interpretive/           # item、主诉与分层解释
│   ├── reliability/            # ICC 与测量可靠性
│   ├── process/                # 干预剂量与 engagement
│   ├── appendix/               # 历史附录图
│   ├── kappa/                  # weighted-kappa 图
│   └── faceted/                # Figure 2 / Figure 4 箱线图
├── compat/visualization/       # 按图族归类的历史 import 兼容层
├── visualization/              # 仅保留别名注册与主 figure registry
└── workflows/                  # 独立端到端工作流
```

## 为什么以前看起来有重复代码

原命名使用同一图族前缀表示不同层，例如：

- `change_ci_data.py`：从 record 提取规范数据；
- `change_ci_statistics.py`：计算估计量、CI 和 contrast；
- `visualization/change_ci_plots.py`：历史绘图 import；
- `charts/outcomes/change_ci/plot.py`：真正的绘图实现。

它们不是四份相同算法，但平铺在根目录和 `visualization/` 时很难区分。本次整理后：

- 真正的数据实现进入 `data/`；
- 真正的统计/模型实现进入 `analysis/`；
- 真正的图实现进入 `charts/<family>/<chart_id>/`；
- 历史 API 进入 `compat/visualization/<family>/`；
- 根目录中无内部调用的 `*_data.py / *_statistics.py` 兼容导出已经删除；新代码直接使用 `data.*` 与 `analysis.*`；
- `visualization/` 仅负责把旧模块名映射到 compat，并保留主图注册器。

数据提取、指标计算和画图分层：

```text
repeat summary / checkpoints / long data
                 │
                 ▼
        loader / data/ / process
                 │
                 ▼
       statistics / analysis/
                 │
                 ▼
  charts/<family>/<chart_id>/plot.py
                 │
                 ▼
           PNG + CSV + manifest
```

## 逐图清单

以下“输入”是绘图函数接收的已整理对象，不表示 `plot.py` 会自行扫描文件。文件发现和解析必须在 loader/pipeline 层完成。

### 常规轨迹、结果、可靠性与过程图

| 图目录 | 作用 | 绘图输入 | 图片输出 |
| --- | --- | --- | --- |
| `charts/outcomes/trajectory_lines/` | 各组/独立 run 的纯折线量表轨迹 | `ExperimentRecord[]`、timepoint labels、scale | `presentation_00_<scale>_trajectory_lines_only.png` |
| `charts/outcomes/trajectory_ci/` | 轨迹均值、测量 CI/SD、可选 outer-run 汇总 | records、labels、scale、误差配置 | `presentation_01_<scale>_trajectory_with_ci.png` |
| `charts/outcomes/endpoint_change_ci/` | baseline 到观测终点的变化及差值 CI | records、labels、scales | `presentation_02_endpoint_delta_with_ci.png` |
| `charts/outcomes/change_rebound/` | baseline→nadir、rebound、终点变化 | records、labels、scales | `presentation_03_best_change_and_rebound.png` |
| `charts/outcomes/group_time_contrasts/` | 组间 baseline-referenced 时间变化对比 | records、labels、scales | `core_00_group_time_contrasts.png` |
| `charts/outcomes/outer_contrast_forest/` | baseline-adjusted 终点差和 Hedges' g | records、labels、scales | `core_01_outer_run_contrast_forest.png` |
| `charts/outcomes/cross_scale_convergence/` | 同期 PHQ-9/BDI-II 收敛效度；统一 baseline/endpoint 的变化一致性；可按 condition/persona 着色并拟合组内描述线 | records、labels | `core_02a_cross_scale_concurrent_validity[_<dimension>].png`、`core_02b_cross_scale_change_agreement[_<dimension>].png` |
| `charts/reliability/measurement_icc/` | frozen snapshot 总分 ICC(A,1)/ICC(A,K)；支持左侧分层标签、中央 forest、右侧数值列；SEM/MDC95 与 repeat SD 分布 | records、labels、scales | `core_03_measurement_icc[_<dimension>].png`、`core_04_measurement_error.png` |
| `charts/outcomes/endpoint_waterfall/` | 每个 outer run 的终点变化瀑布图 | records、labels、scales | `core_04_<scale>_endpoint_waterfall.png` |
| `charts/reliability/measurement_heatmap/` | repeat SD、CI width、category flip rate 热图 | records、labels、scales | `presentation_04_<scale>_<metric>_measurement_reliability_heatmap.png` |
| `charts/process/delivery/` | prompt 完成、干预剂量、CBT stage 停留 | process rows、stage rows | `core_05_*`、`core_06_*`、`core_07_*` |

### 兼容附录图

| 图目录 | 作用 | 绘图输入 | 图片输出 |
| --- | --- | --- | --- |
| `charts/appendix/final_delta/` | 历史兼容：按已有严重度字段展示终点变化柱图；当前 0808 数据不适用 | records、labels、scales | `01_final_delta_bars_by_severity.png` |
| `charts/appendix/trajectory/` | 单量表、按严重度的附录轨迹 | records、labels、scale | `<nn>_<scale>_trajectory_by_severity.png` |
| `charts/appendix/delta_trajectory/` | 相对 baseline 的变化轨迹 | records、labels、scales | `<nn>_delta_from_baseline_by_severity.png` |
| `charts/appendix/volatility/` | 轨迹波动与最大向上变化 | records、labels、scales | `<nn>_stability_rebound_by_severity.png` |
| `charts/appendix/endpoint_uncertainty/` | 终点评分及 snapshot 内 SD | records、labels、scales | `<nn>_endpoint_repeat_uncertainty_by_severity.png` |
| `charts/appendix/delta_heatmap/` | 各 outer run 的终点变化热图 | records、labels、scales | `<nn>_final_delta_heatmap.png` |

### 跨 persona 图

| 图目录 | 作用 | 绘图输入 | 图片输出 |
| --- | --- | --- | --- |
| `charts/persona/scale_trajectories/` | persona × scale 纵向轨迹 | cross-persona records、labels、scales | `figure_01_0727_persona_scale_trajectory.png` |
| `charts/persona/contrast_forest/` | 各 persona 的组间 adjusted difference 与 g | persona contrast rows | `figure_02_0718_g1_vs_g9_persona_forest.png` |
| `charts/persona/outcome_heatmap/` | persona × group outcome 汇总 | persona/group summary rows | `figure_03a_0718_persona_group_outcome_heatmap.png` |
| `charts/persona/process_heatmap/` | persona × group 过程数据覆盖和剂量 | persona/group process rows | `figure_03b_0718_persona_group_process_heatmap.png` |
| `charts/persona/leave_one_out/` | leave-one-persona-out 敏感性 | LOPO rows | `figure_s01_0718_g1_vs_g9_leave_one_persona_out.png` |
| `charts/persona/variance_partition/` | group/persona/run/time/measurement 方差描述 | variance rows | `figure_s02_0727_variance_decomposition.png` |

### Item、主诉与分层解释图

| 图目录 | 作用 | 绘图输入 | 图片输出 |
| --- | --- | --- | --- |
| `charts/interpretive/item_life_state_change/` | T0→终点的逐条目变化 forest | life-state summary rows | `life_state_01_item_change_forest.png` |
| `charts/interpretive/life_state_symptom_trajectory/` | 等权症状域的纵向轨迹 | symptom rows、labels | `life_state_02_symptom_trajectory.png` |
| `charts/interpretive/group_symptom_heatmap/` | group × symptom change 热图 | life-state group summary | `life_state_03_group_symptom_change_heatmap.png` |
| `charts/interpretive/complaint_evaluation_changes/` | 主诉评估节点之间的阶段变化和推进率 | complaint interval rows、evaluation labels | `complaint_01_evaluation_node_changes.png` |
| `charts/interpretive/stratified_run_symptom_heatmap/` | 单 run 症状变化分层热图 | run symptom rows | `life_state_01_single_run_symptom_heatmap.png` |
| `charts/interpretive/stratified_replicate_agreement/` | R01/R02 变化复现性 | agreement rows | `life_state_02_r01_r02_reproducibility.png` |
| `charts/interpretive/stratified_item_heatmap/` | 单 run 的 PHQ/BDI item change | item rows、scale | `life_state_03/04_*_single_run_items.png` |
| `charts/interpretive/complaint_interval_alignment/` | 主诉推进与量表变化的 interval 对齐 | alignment rows | `complaint_01_interval_progress_outcome_alignment.png` |
| `charts/interpretive/complaint_run_alignment/` | run-level 主诉推进与结局对齐 | run alignment rows | `complaint_02_run_progress_outcome_alignment.png` |
| `charts/kappa/entity_heatmap/` | persona/group × scale 的 Kappa | entity Kappa rows | `weighted_kappa_03_entity_scale_heatmap.png` |

### Weighted Kappa 图

| 图目录 | 作用 | 绘图输入 | 图片输出 |
| --- | --- | --- | --- |
| `charts/kappa/repeat_pair_heatmap/` | measurement repeat pair 的 Kappa 热图 | weighted-kappa payload | `weighted_kappa_01_repeat_pair_heatmap.png` |
| `charts/kappa/item_forest/` | 各量表条目的紧凑 Kappa heatmap；可生成 PHQ-9/BDI-II 双子图并按 condition/persona 分列（forest 保留为兼容调用） | weighted-kappa payload | `weighted_kappa_02_item_heatmap[_<dimension>].png` |
| `charts/kappa/stratum_heatmap/` | 分层 Kappa 热图 | weighted-kappa payload | `weighted_kappa_03_<stratum>_scale_heatmap.png` |

### 新增论文图

| 图目录 | 作用 | 绘图输入 | 图片输出 |
| --- | --- | --- | --- |
| `charts/outcomes/change_ci/` | 各 timepoint/group change + CI、显著性 bracket | change estimates、contrasts | `paper_figure_02_<outcome>_change_ci.png` |
| `charts/process/engagement/` | 每日 interaction 与 24h 时间分布 | daily rows、time-bin rows | `paper_figure_03_engagement_process_<group>.png` |
| `charts/persona/treatment_response_profile/` | persona × condition 的标准化治疗反应 profile | profile summary DataFrame | `persona_treatment_response_profile_<method>.png` |
| `charts/persona/shap/` | persona/baseline/behavior predictor 的 SHAP | predictor analysis dict | `persona_predictor_shap.png` |
| `charts/symptoms/trajectory_small_multiples/` | PHQ/BDI 逐条目 small multiples | symptom summary DataFrame | `symptom_trajectory_<scale>.png` |
| `charts/symptoms/effect_forest/` | model effect 与 time-specific Cohen's d forest | model rows、effect rows | `symptom_effect_forest_<scale>.png` |
| `charts/symptoms/trajectory_item_change/` | A：两量表、两组总分轨迹；B/C：两个量表逐 run 条目前后变化热图 | total trajectory rows、run-level item changes | `symptom_trajectory_item_change_composite.png` |
| `charts/symptoms/network/` | longitudinal 或 group-comparison symptom network | network panel payloads | `symptom_network_<scale>_<panel>.png` |
| `charts/faceted/figure2/` | 1×2 outcome × group/time 箱线图 | canonical tidy DataFrame、PlotOptions | 默认 `figure2_outcome_group_boxplots.<format>` |
| `charts/faceted/figure4/` | 2×2 outcome × study/category/time 箱线图；category 可为真实 persona 类别，当前 0808 无 severity facet | canonical tidy DataFrame、PlotOptions | 默认 `figure4_outcome_study_severity_boxplots.<format>` |

分面箱线图直接调用统一 chart API：

```python
from pathlib import Path

from experiment_eval.charts.faceted.figure2 import plot_figure2
from experiment_eval.charts.shared.faceted import PlotOptions
from experiment_eval.data.faceted import load_long_data

frame = load_long_data([Path("input.csv")])
plot_figure2(
    frame,
    Path("output/figure2_outcome_group_boxplots"),
    PlotOptions(formats=["png"]),
)
```

## 图与数据/统计模块的对应关系

| 图族 | 数据整理 | 指标计算/建模 | 编排 |
| --- | --- | --- | --- |
| 常规量表与 core | `loader.py`、`process.py` | `statistics.py`、`weighted_kappa.py` | `pipeline.py`、`reports.py` |
| Change + CI | `data/change_ci.py` | `analysis/change_ci.py` | `pipeline.py` |
| Engagement | `data/engagement.py` | `analysis/engagement.py` | `pipeline.py` |
| Persona profile | `data/persona_profile.py` | `analysis/persona_profile.py` | `persona_profile.py` |
| SHAP/症状长表图 | `data/longitudinal.py` | `analysis/persona_predictor.py`、`analysis/symptom_*.py` | `paper_longitudinal.py` |
| 跨 persona | `loader.py`、`process.py` | `cross_persona.py` | `cross_persona_cli.py` |
| life-state/主诉 | `life_state.py`、`complaint_nodes.py` | `weighted_kappa.py` | interpretive 两个入口 |
| 分面箱线图 | `data/faceted.py` | canonical tidy 的分组分布 | 直接调用 `charts/faceted/figure2` 或 `figure4` |

## 新增图表必须遵守的代码规则

1. 一图一目录：新图必须建立 `charts/<family>/<chart_id>/`，至少包含 `__init__.py` 和 `plot.py`。不要把实现追加到 `visualization/` 或 `compat/`。
2. 单一职责：`plot.py` 只负责把已经整理好的 records/rows/DataFrame 画成一张图或一个定义明确的多 panel figure；不扫描 `results/`，不解析 checkpoint，不计算复杂模型。
3. 数据与统计分离：文件发现、schema 校验放 loader/data 模块；CI、效应量、模型和网络估计放 statistics/model 模块。多个图共用时放稳定公共层，不复制。
4. 公共绘图复用：必须从 `charts.shared.plotting` 使用 `plt`、`save_figure`、字体、配色和布局工具。禁止每张图重复设置 backend、cache、字体和默认 DPI。
5. 明确函数契约：公开函数命名 `plot_<chart>()`；类型标注输入和 `out_dir/output_base`；返回 `Path | None` 或 `list[Path]`。无可画数据时返回空结果或 `None`，不得静默生成误导性空图。
6. 输出稳定：已有图不得随意更改文件名、默认 DPI、轴含义和统计单位。新增图文件名应包含稳定 chart id；同时把路径写入 manifest/chart index。
7. 统计单位正确：outer run 才是组间样本；measurement repeats 不能扩充 n。配对、follow-up、resume 必须有可审计 lineage 才能合并。
8. 注册集中：主入口需要调用的新图只在 `visualization/registry.py` 或相应专用 orchestrator 注册一次，不在 CLI 的多个 batch 分支重复调用。
9. 谨慎保留兼容 API：只有仓库内仍有调用或明确对外使用的旧模块才保留兼容映射。删除兼容路径前必须先检索仓库调用方并同步测试和文档。
10. import 无副作用：导入 chart 不得读取数据、创建输出目录、调用 API 或显示 GUI。Matplotlib 必须使用 shared 中配置的 Agg backend。
11. 可测试：至少补充一次最小输入出图测试，断言输出文件存在；涉及统计的图还要测试样本闸门、缺失输入和独立样本口径。
12. 文档同步：在本 README 的逐图清单补充位置、作用、函数输入、图片输出和运行入口；如有数据不足，只在“可能数据缺口”章节另行分析。
13. 保留领域语义：不要无说明改写中文角色名、group/persona 标签、时间点和历史实验命名。
14. 禁止伪造数据：缺失字段保持 unavailable/NA；不得从总分拆条目、从相邻时点插值、用 measurement repeats 冒充 outer runs，或为了出图重新生成历史标签。

推荐模板：

```python
# experiment_eval/charts/example_family/example_chart/plot.py
from pathlib import Path
from typing import Any

from ...shared.plotting import plt, save_figure


def plot_example(rows: list[dict[str, Any]], out_dir: Path) -> Path | None:
    if not rows:
        return None
    fig, ax = plt.subplots(constrained_layout=True)
    # 只画图；rows 应由 data/statistics 层提前整理
    return save_figure(fig, out_dir / "example_chart.png")
```

```python
# experiment_eval/charts/example_family/example_chart/__init__.py
from .plot import plot_example

__all__ = ["plot_example"]
```

## 兼容路径说明

以下旧 import 仍可调用，但实际兼容模块已按图族存放在 `compat/visualization/`：

- `compat/visualization/core/`：trajectory、outcome、reliability、process；
- `compat/visualization/persona/`：cross-persona、profile、SHAP；
- `compat/visualization/paper/`：change-CI、engagement、symptom paper figures；
- `compat/visualization/interpretive/`：item、complaint、stratified、Kappa。

例如旧调用：

```python
from experiment_eval.visualization.change_ci_plots import plot_change_ci_comparison
```

与新实现路径：

```python
from experiment_eval.charts.outcomes.change_ci import plot_change_ci_comparison
```

功能等价。重构前的直接图路径（如 `experiment_eval.charts.change_ci`）也会映射到新的分组目录。现有项目代码可以继续使用旧路径；新增代码应使用 `experiment_eval.charts.<family>.<chart_id>`。

## 存档精简（非绘图入口）

`prune_archive.py` 是与 `experiment_eval` 校验逻辑耦合的存档整理入口，不是画图入口。默认只做 dry-run；`backup` 不修改源目录，`prune --apply` 会替换源存档。默认分析型策略保留 reports、所有普通 simulation checkpoints、manifest、conversation/events、咨询与 prompt trace、staged scale 的答案/评分/metadata、每个 run 的 `visualizations/` 与 `merge_consultation_dialogues.json`，并保留每个纳入 run 的完整 `storage/` 和 dynamic LLM 审计 trace；省略 `_tmp`、`snapshot_storage`、内嵌 snapshot 的 `job.json` 和重复的大型量表 trace。可用 `--no-include-merged-consultation-dialogues` 或 `--no-include-dynamic-llm-traces` 关闭两类 trace；确需全部量表 trace 时显式传 `--include-raw-scale-traces`。旧参数 `--include-one-dynamic-llm-trace` 作为兼容别名保留，但现在同样表示保留每个纳入 run 的 dynamic trace。

```bash
python -m experiment_eval.prune_archive --help
```

建议暂时保留此路径以兼容现有命令和测试。若未来把存档生命周期工具集中迁到 `tools/archive/`，可将这里保留为薄兼容入口；当前它会调用 report loader 和过程指标校验，因此直接移出包并不会降低耦合度。



## 可能数据缺口

本节只保留一张总表。候选数据只能从 `results/` 历史存档中提取或可靠整理；没有记录的字段保持 unavailable，不补默认值、不重跑历史回答、不把 frozen-snapshot 的量表重复、resume 或 follow-up branch 计成新的独立样本。当前真实分组只有 persona 与实验条件，`SEV` 只是历史命名 token，不是严重程度分组。

状态含义：**已实现**表示已有可直接复用的接口；**部分实现**表示当前存档可提取一部分，但仍缺跨 archive 规则或明确元数据；**未实现**表示尚无统一接口/分析；**数据不足**表示代码即使存在，当前存档也不满足有效分析条件。

| 类别 | 数据缺口 / 目标接口 | 涉及分析 | 状态 | 当前可恢复内容与 0808 结论 | 后续实现或判定条件 |
| --- | --- | --- | --- | --- | --- |
| 设计覆盖 | persona × condition × scale × timepoint 的缺失单元格和实际 n | 全部组间图、箱线图、轨迹 | **已实现** | `archive_condition_coverage.csv` 输出 root/outcome/item n、缺失数和状态；0808 的已观察设计为 6 persona × G1 × 3 根 runs。 | 整个条件完全缺席时无法从文件名反推，调用 `archive_data --expected-condition G2` 等声明预期设计。 |
| 设计覆盖 | 多个 results 存档的统一合并、root-run 去重和冲突报告 | 所有跨存档分析 | **未实现** | 单个 archive 已能导出规范接口并保持 follow-up lineage。 | 新增 merge/validate 入口；仅合并 protocol、group overlay、量表/评分版本、endpoint 语义兼容且 root ID 不重复的数据。 |
| 实验身份 | 明确的 study / main experiment / replication role | 分面箱线图、层级模型、敏感性分析 | **数据不可用** | batch、日期、KBD、R01/R02 均可读，但不能自动等同于 study 或 replication。 | 只有 trial/batch manifest 明确记录关系时才映射；否则 availability 保持 unavailable。 |
| 实验条件语义 | G 编号到 CBT/supportive/control/comparator 的可审计映射 | Change+CI、effect forest、CBT-relative target | **部分实现** | condition 与 controller/group-overlay provenance 可提取；不按 G 编号猜治疗语义。0808 只有 G1。 | 从各 archive 的 overlay/config 建立版本化 contrast manifest；无法确认 comparator 时不做治疗效应归因。 |
| 协议一致性 | controller、prompt、模型、量表、评分器和 group overlay 的跨 archive 等价性 | 全部跨条件/跨批次分析 | **部分实现** | run catalog 已保存可恢复的 controller/model/hash provenance。 | 实现兼容性审计与冲突表；缺版本信息时拆分分析，不静默合并。 |
| 独立性/QC | 显式 simulation run QC、失败原因、repair/recovery provenance | 全部推断分析 | **部分实现** | strict repeat-summary 的 measurement QC 已验证；simulation behavior QC 多数未显式记录。 | 从 batch state、failure/repair report、checkpoint 完整性提取统一 QC 状态；没有声明时不默认通过。 |
| 时间语义 | baseline、session、POST/NOW、delayed follow-up 的共同 endpoint 角色 | Change+CI、轨迹、箱线图、forest、network | **部分实现** | 保存原 timepoint、`sim_time`、elapsed day、session count；T0 与显式 follow-up 已分类。 | 建立按 protocol 的 endpoint map；不同 archive 的“最后一点”不能无标记冒充共同 post。 |
| 随访谱系 | root、resume、follow-up branch 的 parent linkage 与去重 | 纵向轨迹、forest、network | **已实现** | 0808 的 10 个 follow-up branch 均可由 `parent_run_name + parent_label` 接回根 run，且不增加独立 n。 | 无显式 parent 的分支只报告 coverage，不并入配对分析。 |
| 测量缺失 | item/timepoint/scale 的失败、缺条目、complete-case/available-case 清单 | 条目轨迹、forest、network | **部分实现** | condition coverage 能发现 outcome/item 的缺失 cell；strict summary 能读取 reviewed item scores。 | 仍需统一失败原因和评分 repair provenance；不从总分拆条目、不插值。 |
| 评分口径 | raw/reviewed/repeat mean/modal item 与 total 的版本化 provenance | 轨迹、profile、forest、network 敏感性 | **部分实现** | 主接口保存 aggregation、primary score、score source 和 measurement repeat n；raw answers/scores 可由新存档策略保留。 | 增加 feature dictionary/评分版本兼容检查；同一 snapshot 的多个评分版本只作敏感性，不增加 n。 |
| 样本量 | 每个 persona × condition cell 足够的独立根 runs | Change+CI、boxplot、mixed model | **数据不足** | 0808 每个 persona × G1 只有 3 个根 runs；0718 的历史 group cell 也很小。 | 继续接入同协议存档；图中标真实 n。不能用 10 次量表重复、checkpoint 或 branch 扩大样本量。 |
| Engagement | 结构化 conversation/event、target involvement、turn/字符数和有效 0/unavailable | Engagement vs outcome、dose-response | **部分实现** | events 与 coverage 接口已实现；0808 只有实际复制 raw 目录的 run 能恢复过程数据。 | 统一 session/consultation 分母、planned dose 与 completion state；分母不可恢复时不算完成率。 |
| Engagement | 真实对话 elapsed duration、token/cost、会话中断 | Engagement 扩展 | **数据不可用/未实现** | 当前可得仿真时间戳、turn 和字符数，不等于真实耗时。 | 仅在日志本身记录 start/end/token/cost 时接入；不由字符数伪造时长。 |
| Persona profile | 跨条件 profile、baseline-adjusted profile 与稳定性区间 | Persona profile | **部分实现** | 同一条件内可做探索性 persona 描述，baseline 总分/条目和 persona one-hot 已导出。 | 需要至少两个语义明确的条件、共同 endpoint 和更多独立 runs；只使用干预前协变量。 |
| Persona 静态特征 | 历史 persona/depression/stressor/complaint 配置的结构化副本 | Persona profile、predictor/SHAP | **部分实现** | 多数 run 只能安全恢复 persona ID；不能回到当前资产为历史 run 补配置。 | 从 runtime_personas/config/checkpoint 提取并记录来源/hash；覆盖不一致时拆列或保持缺失。 |
| Baseline 行为 | 干预前固定窗口的社交、伙伴、活动、记忆、主诉特征 | Predictor/SHAP、异质性 | **未实现且覆盖不足** | conversation/events 可提取，但尚无跨 run 一致的 pre-treatment cutoff。 | 先定义无泄漏窗口和特征字典；治疗期行为只能命名为 early-response/process predictor。 |
| 分面箱线图 | 明确 study/category/facet 与每格稳定分布 | Figure 4 类图 | **部分实现/数据不足** | persona 和 condition 可直接分面；没有 severity 分组，也没有 well-being outcome。 | 仅在每格独立 n 足够且 study 语义明确时增加 facet；缺失格保持缺失，不填 0。 |
| 症状条目轨迹 | 同协议完整 PHQ-9/BDI-II 条目与共同观察窗口 | Symptom trajectory small multiples | **部分实现** | item long 表已实现，0808 可恢复多时间点及已评分 follow-up。 | 继续核查条目版本、共同 timepoint、完整向量与评分 provenance；缺条目不反推。 |
| 症状效应 forest | 明确两臂、共同 baseline/endpoint、每臂足够 n 和预先定义检验族 | Mixed model、Cohen's d、BH-FDR | **未达到数据条件** | 计算与绘图代码可用，但 0808 只有 G1，不能形成 treatment/comparator contrast。 | 至少两个语义明确且协议可比的条件；每臂 n<2 不算 d，极小 n 只标 exploratory。 |
| 症状网络 | 每个 group × timepoint 的完整 item vector、方差和稳定性样本 | PHQ-9/BDI-II network | **数据不足** | 条目字段并非主要缺口；0808 最大共同 cell 远低于 PHQ-9=45、BDI-II=105 的默认闸门。 | 达到独立 run 门槛后再做 bootstrap/case-dropping；零方差节点如实报告，不加人工噪声。 |
| Treatment/group network node | 真正 mixed graphical model 所需的跨组混合数据与方法 | Treatment–symptom edges | **未实现** | 当前网络刻意不含 group node。 | 只有样本量和方法均支持 mixed network 时实现；不把普通相关或 one-hot 近似冒充 treatment edge。 |
| Predictor/SHAP | 稳定 outcome、同 persona 对照、无泄漏 baseline 特征和足够独立 runs | Persona predictor / SHAP | **数据不足且部分字段未实现** | baseline symptom + persona one-hot 已导出；0808 18 个根 runs且仅 G1，低于至少 `max(40, 5×特征数)` 的门槛。 | 明确 CBT/control、补足同协议 root runs、加入缺失率/provenance audit 后再拟合；batch/protocol 完全混杂时停止解释。 |
| 额外结局 | well-being、生活质量或其他并未实际测量的 outcome | 双 outcome panel、扩展 forest | **数据不可用** | 当前可靠 outcome 为存档中真实量表（主要 PHQ-9/BDI-II）。 | 只接入存档真实测量；不能由症状分数或对话文本生成替代 outcome。 |

0808 样例的机器可读结论以 `archive_data_availability.json` 和 `archive_condition_coverage.csv` 为准。新增远程 G1 人设存档或本地 KBD2 条件组后，应先分别导出接口和 coverage，再做协议兼容、root-run 去重与 condition 语义审计。
