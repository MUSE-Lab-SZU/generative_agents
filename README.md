# 基于斯坦福小镇的抑郁症干预仿真系统 GenerativeAgentsCN

> 更新时间：2026-08-25

## 关键测试结果速查

某存档所有结果在`results/experiment_data/xxx`和`results/checkpoints/xxx`目录下，关键测试结果如下：

1. **Prompt使用情况（调用LLM时的Prompt可视化）**：`results/experiment_data/xxx/traces/forced_prompt_traces/`文件夹可以看到所有对话的所有Prompt使用情况，包括患者、判断LLM、医生、评估LLM的Prompt。
   - 作用：最直观观察患者状态变化、各种Prompt使用是否合理。
   - 生成：使用 `runshells/run_one_experiment.py` 或 `runshells/run_batch_experiment.py` 运行仿真后自动收集；Legacy G1/G2 复现入口 `runshells/run_experiment.py` 也会收集。
2. **仿真内阶段评估（staged_eval）**：`results/checkpoints/xxx/staged_eval/`保存原始阶段评估结果；`results/experiment_data/xxx/scales/staged/`保存收集后的结果与评分文件。
   - 作用：在仿真过程中自动触发 PHQ-9 / BDI-II / SDS 评估，适合做基线、阶段点、结束后一段时间（T4）的纵向对比。
   - 生成：`start.py` 在仿真过程中自动调用 `modules/staged_eval_manager.py` 生成；使用 `runshells/run_one_experiment.py` / `runshells/run_batch_experiment.py` 时会自动收集。
3. app.py评估脚本：`results/experiment_data/xxx/scales`存放了app.py评估脚本的输出结果，包括量表回答、单量表评估和汇总（**scale_scores.json**）。
   - 作用：使用三种自评量表评估患者抑郁程度，包括SDS、BDI-II、PHQ-9。因为上次心理医生反馈说最好就使用自评量表，所以没使用联合量表。
   - 生成：使用`runshells/run_one_experiment.py`运行仿真实验后自动生成
4. 医患对话、判断LLM、会后评估LLM：`results/experiment_data/xxx/traces/merge_consultation_dialogues.json`文件。
   - 作用：可视化所有医患干预对话，包括患者、判断LLM、医生、评估LLM的输出。适合给心理专业人员查看、分析。若 `consult_record.enabled=false`，这里不会额外附带 SOAP 咨询记录。
   - 生成：使用`runshells/run_one_experiment.py`运行仿真实验后自动生成
5. 记忆可视化：`results/experiment_data/xxx/visualizations/agent_memory/memory_view_卡布达_xxx.csv`可以查看某角色的所有记忆
   - 作用：可视化某角色的所有记忆，可以筛选chat记忆查看对话情况、筛选thought记忆查看反思结果、筛选event记忆查看遇到的事件。
   - 生成：使用`runshells/run_one_experiment.py`运行仿真实验后自动生成
6. 外置记忆系统存储情况：`results\external_memory_audit\xxx\report.html`文件。
   - 作用：可视化某存档的所有agent在外置记忆系统中的存储情况，包括记忆数量、记忆层级情况等。
   - 生成：使用`runshells/run_one_experiment.py`运行仿真实验后自动生成

## 更新日志（近期）

以下为 README 内维护的近期更新摘要：
- 2026-08-25：主诉图 `core_belief` 改为初始化时锁定的病例级稳定核心信念；planner 不再生成或修改该字段，患者对信念的强化、怀疑或松动统一写入节点 `label/summary`，程序在 candidate、commit 与 checkpoint 恢复时回填固定值，并兼容旧 checkpoint 和既有 stage schema。
- 2026-08-25：同步重整九个卡布达人设的静态生活背景与各严重度主诉配置，移除 `agent.json` 中预置的症状进展、治疗尝试和近期事件，将动态抑郁表现交由主诉图与运行态生成；病例级核心信念按人设/严重度保持一致的稳定基线。
- 2026-08-25：按需重绘新增 0824 KBD2 五条件轨迹图，支持完整 T0–session_20 时间轴及按批次配置的不同 outer-run 数量（G5 为 2 次、其余为 3 次），同时让既有短程图继续沿用 T0–S12 节点。
- 2026-08-24（已由 2026-08-25 的病例级固定策略替代）：主诉图 branch detail 当时会保留有效的 LLM `core_belief` 输出，不再依赖关键词启发式门槛；仅在缺失、空白或类型非法时回退继承父节点，其余稳定运行态字段仍保持精确继承。
- 2026-08-24：批量仿真默认延长至 120 step，完整复评覆盖 T0、session_4 至 session_20，follow-up 默认从 session_20 启动；按需重绘新增 0823 KBD2 七条件、三次 outer run 轨迹图。
- 2026-08-23：新增 `docs/town_method/` 方法与实验框架文档集，系统梳理小镇仿真的研究边界、实验条件与重复结构、动态抑郁人设、CBT 控制流程、冻结量表评估、结果图表口径及代码/产物证据地图；明确区分当前实现、可选机制、历史兼容项和仍待确认的方法假设，并纳入版本管理。
- 2026-08-23：为九个患者及各严重度配置新增客观的长期病例背景锚点；锚点在初始化后固定并随 checkpoint 恢复，独立于当前主诉节点，注入患者动态 prompt 与 CBT judge/医生指导以维持长期治疗脉络，同时明确不限制新主题、推进、好转或反复，也不进入共享动态状态摘要。
- 2026-08-23：细化 Progressive D 控制评估的患者侧证据口径：允许重度抑郁下低强度但有意义的 partial，要求 complete 具有独立的患者新增材料；单纯附和、重复医生措辞、意向或对自身复述的怀疑不再被误判为完成，并可在确有联盟/自主性阻塞时旁移承接。
- 2026-08-23：按需重绘新增 0822 KBD2 六条件轨迹图，并支持按实验条件校验不均衡的 outer-run 数量（G4 为 2 次、其余为 3 次）；图内逐组标注实际样本数，避免将不同重复数误写为统一 n。
- 2026-08-23：智算中心版配置将 Qwen3 与 BGE-M3 的本地负载均衡端口池各扩展至 3 个副本（18000/18002/18003 与 18001/18004/18005）。
- 2026-08-22：新增全局 `intervention.depression_dynamic.domain_state.enabled` 开关；默认配置关闭七维状态的提示词注入、窗口裁决和持久化写入，旧配置或 checkpoint 未声明该字段时仍按兼容策略启用。
- 2026-08-22：主诉图规划改为每轮最多三个完整候选后再交由 transition 匹配提交：候选保留父节点主轴与本轮最小变化，完整语义（含核心信念与叙事焦点）参与匹配；程序自动为重复技术 ID 加后缀，避免将整个已有节点 ID 集合暴露给 LLM。
- 2026-08-22：收紧主诉变化提取对“但仍然”等转折、行为/功能与身体感受混淆的约束；会谈审计脚本改为同时展示 LLM 建议与真实运行态 commit，防止将未提交的 advance 误报为节点推进。
- 2026-08-22：冻结 checkpoint 量表上下文消融扩展为任意已归档 KBD2 组的 T0/S4/S8/S12 重放，按每个时间点相对 T0 配对汇总条件内变化和诊断 contrasts，且测量重复不会计入 outer simulation run 样本量。
- 2026-08-22：消融入口新增 report-only 与按量表汇总能力，可从缺少 worker metadata 的已完成评分恢复 PHQ-9 结果；`--cleanup-artifacts` 仅在 report-only 下删除可再生 trace/job/worker 文件，保留回答、评分及汇总，并标识部分量表产物。
- 2026-08-22：新增 PHQ-9 消融恢复与可视化工具：可由已保存回答补跑缺失 expert 评分、生成节点级 QC/汇总表，并绘制 0821 全条件中 full 与 full_no_state 的描述性轨迹（±1 SD、缺失节点显式保留）；按需重绘同步支持 0821 KBD2 轨迹图。
- 2026-08-21：领域窗口裁决改为读取带说话者标签的完整会谈，以患者原话或已发生行为作为唯一非零更新证据；基线同时提供数值和定性描述，新增 comparison_anchor 与更新幅度的一致性校验、引用轻微改写回填和验证 trace，主诉图 detector 则恢复为仅处理图变化。
- 2026-08-21：主诉图推进改为先生成最小 branch seed 供 transition 匹配，选中后才补全并写入节点；说话风格、情绪向量和关系修饰等稳定字段精确继承父节点，核心信念仅在有明确相对自我信念证据时更新。证据引用缺失、轻微改写或重复改为可审计告警，不再单独阻断有效推进。
- 2026-08-21：minimal CBT 控制器将不合规 evidence turn、过短或附和式证据降为告警，保留风险和阶段守卫；按需重绘新增 0820 KBD2 七条件、三次 outer run 轨迹图，并更新智算中心 vLLM 默认模型目录。
- 2026-08-20：七维持续症状状态改为会谈窗口级更新：逐轮 detector 仅收集带逐字引文的领域证据候选，不再直接改分；完整对话结束后由独立 updater 对七个领域逐一保守裁决（no_update/±5/±10），按会谈 ID 幂等写入，并将待处理窗口和裁决结果随 checkpoint 保存恢复。
- 2026-08-20：新增 0819 冻结 checkpoint 的量表上下文消融入口，可对 G5/G6/G9/G11 的 T0、S4 在完整上下文、仅持续症状状态、状态加静态主诉及隐藏状态四种条件下只读重放 PHQ-9/BDI-II；输出完整 prompt/路由/评分审计和配对 contrasts，且不推进仿真或写回源 checkpoint。
- 2026-08-20：按需重绘新增 0819 KBD2 七条件、三次 outer run 的轨迹图；主诉图审计脚本可对齐 detector 原始输出、程序校验后的 verified_change 与 planner 调用，复评分析亦可从早期快照关联配置中的初始节点；同步清理过时的 legacy/refactor 文档。
- 2026-08-18：为动态抑郁状态新增七维持续症状轨道（情绪/兴趣、睡眠、食欲、精力、注意力、功能、回避）：各卡布达配置提供版本化初始值，变化检测可在主诉图推进之外独立识别有逐字证据的领域改善、恶化或不变，并以固定步长更新、限幅、去重后随 checkpoint 保存恢复。
- 2026-08-18：患者动态 prompt 新增症状状态的定性分档描述，不暴露原始分数或证据；收紧领域变化的名称、方向、逐字证据和冲突校验，并补充图状态独立性、持久化、prompt 脱敏及全 persona 配置的回归测试。
- 2026-08-18：主诉图改为 evidence-first 的按需提交流程：变化检测通过逐字证据校验后，规划器才围绕 verified_change 生成少量临时候选；只有 transition 明确选中的候选才会写入运行态图并推进，hold、无变化或无目标时均不改写图，也不再预生成未来窗口。
- 2026-08-18：收紧动态抑郁图的路径隔离与提交校验：情绪推断不再接收 future candidate 图信息，候选上限按已验证证据计算，禁止回退到默认首个分支；补充覆盖临时候选、无变化保持、非法目标和无未来扩展的回归测试。
- 2026-08-18：新增智算中心版运行配置与 vLLM 服务脚本：默认启动 3 个 Qwen3 与 3 个 BGE-M3 副本并通过端口池负载均衡；支持按 GPU/端口覆盖、状态管理和共享 GPU 的 BGE 显存预算自适应，便于多卡部署。
- 2026-08-18：按需重绘新增 0818 实验的 KBD2 跨条件量表轨迹图（G1/G2/G4/G5/G6/G11），将 outer run 数改为按实验配置校验与标注，支持 3 次独立运行的均值和置信区间；同时忽略本地 depression-analysis 目录并修正 Docker 构建指令。
- 2026-08-18：为 DeepSeek/OpenAI 兼容请求新增 prompt cache 命中/未命中 token 观测：按调用来源记录单次及进程累计用量，日志仅保留脱敏 endpoint 与模型元数据、不写入 prompt 内容；普通模型、会谈 judge/eval、量表评分、语义分析和专家工具均传递可追溯 caller。
- 2026-08-18：重排对话生成、终止判定、会谈历史摘要及 legacy judge/session-eval 提示词，将稳定任务规则与输出格式置于运行态角色、记忆和对话内容之前；新增回归测试锁定 DeepSeek 的静态前缀顺序。
- 2026-08-17：主诉图推进改为两阶段 LLM 判定：先在不提供候选节点的条件下提取患者已发生的变化，再仅以该变化匹配候选分支；无变化或无法精确匹配时保持当前节点。候选规划与补分支不再接收原始证据或暗示变化已发生，患者动态发言 prompt 亦移除未来候选分支，避免预设路径泄漏和候选反向诱导。
- 2026-08-17：重写动态抑郁主诉图的规划与推进口径：候选分支和推进均须由患者自身的新状态、认知松动、行为尝试、功能/关系变化或有证据的恶化支持，不再因表述更抽象、复杂或病理化而自动推进；反思判定更保守，并以近期对话和行为事件构造受限证据上下文。
- 2026-08-17：主诉图 planner/transition 调用现写入可对齐的动态 LLM trace；新增脚本可从 checkpoint 和对话记录导出患者动态 prompt、会谈中的最终节点变化及其前置规划器调用，以及对话/复评主诉图分析，便于审计实际 prompt、LLM 输出和状态迁移。
- 2026-08-17：归档 repeat scale eval 支持更细粒度的 interrupted-job 恢复，并可在成功后清理逐题 trace、job 与阶段快照副本以节省空间。`run_batch_then_repeat_eval.sh` 默认仿真目标改为 72 步、评估至 `session_12`，follow-up 起点改为 `session_12`；可用 `--keep-raw-artifacts` 保留复评底稿，G10/G11/G12 自动使用 legacy controller。
- 2026-08-17：完善自然负向居民聊天提示词：仅在患者明确表达痛苦或寻求理解时呈现低回应、失配和撤回投入，普通生活聊天保持自然，且禁止将负向互动伪装为有效建议、鼓励或危机干预。
- 2026-08-13：新增 G10/G11/G12 自然居民聊天效价组。三组沿用 G2 的无强制会谈与 step 定期量表配置，不启用随机居民聊天调度；每次实际对话生成居民发言时，仅当对方为“卡布达”才分别注入中性、负向、积极专有前缀，卡布达侧及无卡布达参与的对话不注入。
- 2026-08-12：`experiment_eval` 确立面向汇报的精简出图政策，主 CLI 默认改为 `--report-mode core`，不再默认展开 `by-repeat` 或历史全图集；manifest/README 需列出 canonical 图及因重复、样本量/设计或字段缺失而未生成的图，完整规则见 `experiment_eval/README.md`。
- 2026-08-12：新增 `python -m experiment_eval.scale_credibility` 量表可信度图集：输出 ICC(A,1)/ICC(A,K)、SEM/MDC95 与 repeat SD、逐条目 quadratic Weighted Kappa 热图，以及 PHQ-9/BDI-II 同期收敛效度和统一基线/终点变化一致性；支持按实验条件或人设分层，并写出可追溯 CSV。
- 2026-08-12：论文长表分析新增 `--paper-figure symptom-composite`，在显式两组、两量表及共同人设/时间点条件下生成总分轨迹和逐 outer run 条目前后变化热图；过程 engagement 图同步改为正式 CBT 会谈 turn 剂量与相对 T0 量表变化，排除偶发医患对话。
- 2026-08-12：补充按需重绘脚本与回归测试，用于批次/量表可信度/症状复合图的精选重绘；归档复评和批量实验恢复改为容忍被磁盘写满截断的 JSON，并以原子方式写入状态文件。
- 2026-08-11：新增 `python -m experiment_eval.pooled_followup_analysis`，针对归档接口导出的 outcome 长表生成 0802/0808 批次一致性、合并分布与分面箱线图诊断，并对 0802 的无干预 follow-up 输出轨迹、维持/反弹、个体变化和实际间隔图；随访仅按 parent lineage 回接根 outer run，不增加独立样本量。
- 2026-08-11：主评估、跨人设、解释性分析和 `archive_data` 新增可重复的 `--repeat-alias PATH_MATCH=REPEAT_ID`，支持按报告路径重映射 outer repeat；归档报告解析兼容旧 follow-up 命名和已迁移路径，`prune_archive` 同步保留被 repeat summary 引用的原始报告，默认额外保留合并咨询对话及一份完整 dynamic LLM trace。
- 2026-08-11：跨人设/分层解释报告与热图改为根据实际 outer run、repeat 和设计覆盖动态写入样本量，并可通过 `--subset` 单独续跑分层子分析；评分 worker 现可解析含说明文字的 fenced JSON 回复，Docker 依赖安装切换至南京大学 PyTorch 镜像。
- 2026-08-11：新增 `python -m experiment_eval.archive_data`，可从可迁移的 `results` 存档导出以根 outer run 为统计单位的 run catalog、量表/条目长表、基线特征、过程事件、条件覆盖与可用性审计；回访分支仅通过 provenance 关联到原始 run，主入口的自动报告发现默认排除回访 repeat summary，避免将其误计为独立样本。
- 2026-08-11：重构 `experiment_eval.prune_archive` 的分析型归档策略：保留报告、仿真 checkpoint、配置/回访 provenance、对话事件、咨询与 prompt trace、staged 量表元数据和可视化产物，并识别已链接回访与仅报告 run；大型量表 trace、内嵌 snapshot/work 副本默认不保留，可通过 `--include-raw-scale-traces` 显式纳入。
- 2026-08-11：过程 engagement 提取现在从归档的 condition manifest 识别目标患者与医生，输出目标参与、双方 turn/字符数、会谈规则/来源和发起者等可审计字段；分面箱线图与 persona profile 同步修正输入别名及 study/setting 语义，避免把严重程度误作研究设置。
- 2026-08-11：`run_resume_batch_then_repeat_eval.sh --refresh-model-routing` 现会在 strict partial resume 的无干预回访启动前刷新已有回访 checkpoint 的模型路由，便于中断后切换端点继续运行。
- 2026-08-10：整理 `experiment_eval` 绘图目录：图实现按 outcomes/persona/symptoms 等图族归入 `charts/<family>/<chart_id>/`，数据、统计与工作流分别归入 `data/`、`analysis/`、`workflows/`；删除无仓库内调用的根目录 re-export 和旧 `faceted_boxplots` CLI，统一从 `python -m experiment_eval` 或正式模块路径调用。
- 2026-08-10：扩展 `experiment_eval` 的论文图表流水线：统一入口新增 Change+CI（含 complete-case/QC/复现敏感性与可选 ANCOVA）和 interaction engagement/process 图，并将提取、统计、绘图与数据可用性审计分层输出；新增 CSV 统计产物可追溯显著性校正、覆盖窗口及缺失/解析诊断。
- 2026-08-10：新增规范条目长表驱动的逐症状轨迹、混合模型/FDR effect forest、EBIC graphical-lasso symptom network 与 persona/SHAP 图表族，并加入独立 outer run 推断、样本量与可行性闸门；同时提供 persona Z-score 治疗反应 profile 和 `experiment_eval.charts.faceted_boxplot_*` 分面箱线图，均支持规范数据或已严格校验的 repeat summary。
- 2026-08-10：重构 `experiment_eval` 的主评估入口与绘图编排：新增 `python -m experiment_eval` 模块入口、集中 CLI 默认值和可复用 batch pipeline，并将核心/展示/附录、轨迹、结局、可靠性、过程及跨人设图拆分为独立模块和 registry 调度；评估 README 同步补充输入输出、统计单位与图表职责说明。
- 2026-08-10：恢复批处理新增可选 `--refresh-model-routing`（可用 `--routing-config` 指定来源），仅将 LLM/BGE 的 `base_url` 与负载均衡配置同步到最新恢复快照，并在 `results/recovery_backups/` 留存备份；默认端点池与智算平台 vLLM 服务相应调整为 2 个 Qwen、2 个 BGE 实例（3 卡），同时下调默认显存占用。
- 2026-08-09：无干预回访支持 strict partial resume：`run_post_sim_followup.py` 会校验同一请求、连续且完整的 `followup_step_*` bundle 与对应 root snapshot，从最后一个有效节点恢复运行；中断后的 root snapshot、storage 与 conversation 会先备份至 `results/checkpoints/_partial_followup_resume_backups/`。`run_resume_batch_then_repeat_eval.sh --followup` 默认启用，可用 `--no-followup-resume-partial` 关闭；回访后复评继续通过独立报告和 `--resume-partial` 补齐。
- 2026-08-08：调整 `runshells/run_batch_then_repeat_eval.sh --followup` 流程：无干预回访与原仿真 repeat eval 并行执行，回访完成后再以独立名称、报告和日志运行回访节点复评；dry-run 输出同步展示并行关系，避免两阶段产物互相覆盖。
- 2026-08-08：增强 `runshells/run_resume_batch_then_repeat_eval.sh` 的恢复能力：原仿真和回访复评均使用独立源 summary 与标签，已完整报告自动跳过，未完成报告通过 `--resume-partial` 续跑；脚本会等待并校验回访 summary，复评失败时终止关联回访任务并保留对应错误日志。
- 2026-08-08：为 OpenAI 兼容的对话与 embedding 模型新增进程内 round-robin 端点池。`load_balancing.ports` 复用既有 `base_url` 的协议、主机和 API 路径，仅轮换端口；对话请求会选择独立 OpenAI client，LlamaIndex embedding 同步/异步请求亦通过独立 client 池轮换，避免并发请求改写彼此端点。默认配置现启用 3 个 Qwen 与 3 个 BGE 端口。
- 2026-08-08：新增 `runshells/vllm_services_plat.sh`，支持远程智算平台按 GPU/端口列表启动、停止和查看多实例 vLLM 服务；可在 3 张卡各运行一个 Qwen，并在第 4 张卡上运行 3 个 BGE 副本且自动降低同卡显存占用。`docs/RUNBOOK.md` 补充对应启动命令与六端点预检流程。
- 2026-08-08：新增默认启用、可配置的 `simulation_events.jsonl` 仿真可观测日志（schema `simulation_events_v1`）：每步记录各 Agent 的位置、活动/行动、会谈标识和规划路径长度，并在对话完成后记录双方、物理地点、时间及强制会谈元数据。记录采用只追加、best-effort 写入，故障仅告警而不参与 LLM 调用或仿真控制；事件日志会随单次实验收集，并由 `experiment_eval.prune_archive` 保留在评估归档中。
- 2026-08-08：收紧 Progressive D 会后小目标完成审计的默认达标比例，由 60% 提升至 80%；同时将对应评估 Prompt 的角色名称泛化为“会后小目标评估器”。
- 2026-08-06：扩展量表可靠性评估：`experiment_eval` 在既有总分 ICC 之外新增 PHQ-9/BDI-II 0–3 有序条目的 repeat-pair Weighted Cohen’s Kappa（quadratic 为主、linear 为敏感性分析），严格以 `snapshot_id × scale × item` 配对；输出可追溯条目长表、总体/分层/逐条目 CSV、统计说明和 Kappa 热图/forest 图，并在加载阶段校验各 repeat 的条目数一致。
- 2026-08-06：新增解释性分析与 checkpoint 专家访谈工具：独立的 life-state、主诉评估节点与分层报告按条目将两量表归并为九类生活状态，只在精确评估节点读取 judge trace 的主诉阶段，并报告其与量表结局的对齐及重复测量一致性。`python customization/expert_checkpoint_chat/app.py` 提供专家治疗师访谈页面，只读加载存档 checkpoint，优先校验并复制严格快照，必要时按 node/time 上界作明确标记的回退；会话 JSONL 与沙箱存储分离至本地运行目录，`Game/create_game` 现可显式指定 `storage_root`。
- 2026-08-05：增强实验归档与过程评估：`python -m experiment_eval.prune_archive` 新增非破坏 `backup` 模式，源存档不变而将评估所需内容复制至新目录；`--completed-only` 可只保留已有 complete repeat summary 的画图输入，`--include-run-prefix` 可进一步筛选如 `followup-` 回访 run，并拒绝拆分混合报告或覆盖/嵌套目标。已有 complete summary 会覆盖同 run 的旧 incomplete 记录。过程指标现区分“有 checkpoint”与“有 judge trace”，图表在跨人设时标出 KBD 并按人设聚合，避免同组不同人设混淆。
- 2026-08-05：新增 `python runshells/watch_followup_then_repeat_eval.py`，可轮询严格完成的无干预回访并自动启动独立命名的 repeat eval；它校验回访 manifest、最终 full snapshot、所有相对步数 bundle、controller manifest 及可选的 completed batch_state，对完整报告跳过、失败延迟重试并使用文件锁避免重复执行。主跑与续跑流水线同步使回访复评使用独立报告/日志，默认复评内部并行数提高到 6；续跑默认目标恢复为 120 步并补回至 `session_20` 的评估标签。
- 2026-08-02：checkpointing 默认启用 `rolling_resume`：每次完成新的会谈后，在 `results/checkpoints/<run>/recovery_checkpoint/latest/` 原子替换完整 runtime/conversation/storage 恢复包，staged-eval 快照仍独立保留。仿真会为该包确保写出完整 snapshot；稀疏恢复工具现同时校验滚动包和 staged bundle、优先选取最新一致锚点并记录 artifact kind，旧 checkpoint 恢复时补齐新的 checkpointing 默认项。新增原子替换及优先从滚动 session checkpoint 恢复的回归测试。
- 2026-08-02：调整 Progressive D 强制会谈 Judge：Prompt 改为读取本场完整对话、会话开始前的小目标进度快照、患者状态与当前轮次，不再截取为最近对话或混入本场前的控制进展；终止规则明确在满足当前提纲 Exit Criteria、无高风险且无明确拒绝时，由医生下一句作不再提问的结束性总结，并把第 12 轮作为收束软提示。管理器仅在 Progressive D 注入完整对话和 `TURN_NO`，其余 controller 保持原有占位符契约；新增完整对话、进度快照与轮次注入回归测试。
- 2026-08-02：新增仿真后无干预回访流水线 `python runshells/run_post_sim_followup.py`：仅接受通过严格 bundle 校验的 staged snapshot，源 checkpoint 保持只读，复制为独立分支后禁用会谈、居民聊天、医嘱、session prompt/judge/eval 与咨询历史，保留动态抑郁、记忆及既有历史；回访以源快照为锚点，按相对步数冻结 `followup_step_<N>` 节点并生成可供重复量表复评的 summary/manifest。`run_batch_then_repeat_eval.sh --followup` 可选串联该阶段（默认关闭；默认 120 步、每 30 步快照），批量汇总按数值顺序排列回访节点；新增严格来源、隔离分支、清理控制状态、幂等完成与节点一致性回归测试。
- 2026-08-01：Progressive 新实验入口收口为 D-only，使用 `--cbt-controller progressive` 即自动选择 D；`--progressive-stage D` 只作为旧命令兼容参数，A/B/C 的入口、运行分支、配置键和专用 Prompt 已删除。Stage D 改用 `intervention.progressive_d` 原生 policy，不再把 A/B/C 旗标作为运行时门禁；新 condition manifest 升级为 schema v2 capabilities。0731 KBD2 G1/G4 的 schema v1 manifest、旧四旗标 D runtime config/checkpoint 继续只读解析和恢复，A/B/C 旧身份明确拒绝。
- 2026-08-01：完成 legacy 清理：移除无调用的动态抑郁 runtime policy、会谈/聊天兼容 getter，以及 EC-Doll emotion/cognitive 旧 API；`data/config.json` 不再声明对应四个 `depr_*` 键，但旧 runtime config/checkpoint 仍可宽松解析并被忽略。实验评估输出升级为 `experiment_eval_metrics_v2`，移除 `temporal_sd`、`temporal_sd_deprecated` 和 `legacy_aliases`，统一使用 `trajectory_volatility`；弃用文档及兼容/移除回归测试同步更新，`context_analyzer` 仍保留观察。
- 2026-08-01：`experiment_eval` 图表输出统一为默认 300 DPI PNG，不再自动生成 SVG/PDF；跨人设分析的 manifest、终端提示、图表索引和 `experiment_eval/README.md` 已同步改为 PNG 产物说明，集成测试覆盖仅生成 PNG 与 CSV 的行为。
- 2026-08-01：优化动态抑郁主诉图的分支规划 Prompt：不再向 LLM 传入完整 `stage_catalog`，改为包含当前节点、直接子节点、近期实际路径和有限语义防重摘要的有界 `local_graph`，并以 `known_stage_ids` 保留全局 ID 去重提示；分支补齐 Prompt 同步压缩重复上下文，程序端既有唯一性校验保持不变。新增的三个 Prompt 快照上限参数已同步到 KBD1–9 的默认/轻度/中度/重度共 36 份 `depression_config`，确保村庄与咨询室批量实验显式使用一致配置，并新增主诉图 Prompt 有界性、事件溯源与模拟时钟时间戳回归测试。
- 2026-07-31：抽取并统一存档、回放与评估的共享协议：新增 `customization/snapshot_ui_service.py`，供普通聊天、抑郁量表和专家私聊 Gradio 工具复用只读存档加载、角色枚举与游戏初始化，同时保留各工具原有的角色配置路径策略；新增 `simulation_roster.py`/`replay_protocol.py`，压缩与回放不再导入启动入口，回放改从压缩数据动态读取角色花名册并兼容咨询室存档。量表条目/直接作答解析、工件 SHA-256 摘要和实验配置深合并亦归并为共享实现，staged-eval、存档复评、稀疏续跑及批量/T0 评估继续兼容既有产物；新增跨入口协议与只读行为测试。
- 2026-07-31：完成安全清理阶段 3 的 legacy 边界标注：明确 `cbt_controller_mode=legacy` 与 Progressive D 的 `legacy_stage_transition_adapter` 仍在使用；将 `run_experiment.py` / `.sh` 标为仅供 Legacy G1/G2 历史复现，删除已确认无需的 `tmp_repeat_0712_second_eval.py`，并为旧 runtime policy/public getter 增加 deprecation warning。同时修正批量串联入口使用 legacy/minimal 时误传 Progressive Stage D 的问题。详见 `docs/LEGACY_AND_DEPRECATION.md`。
- 2026-07-31：重构按存档管理外置记忆用户的命令行工具：新增 `memory_user_admin_utils.py` 统一快照角色发现、配置定位、`user_id+存档名` 目标生成、报告写入等共用逻辑，`inspect_memory_user_stats_by_save.py` 与 `cleanup_memory_users_by_save.py` 改为复用该模块，并新增覆盖快照目标解析和报告写入的测试。同步移除会谈/重复量表/实验脚本与动态抑郁界面中的无效参数、未使用包装函数和冗余代码，将 Scratch 的正则模式改为 raw string；既有行为与实验默认参数保持不变。
- 2026-07-31：清理多处未使用的导入、局部变量及字典遍历键绑定，涵盖动态抑郁量表界面、实验评估/报告绘图、聊天与干预控制、回放、staged-eval 恢复、重复量表复评、实验启动及外置记忆审计工具；不改变既有运行逻辑、输出或默认实验参数。
- 2026-07-31：新增跨人设探索性分析入口 `python docs/analysis/run_cross_persona_analysis.py`（或 `python -m experiment_eval.cross_persona_cli`）：以独立 outer run 为统计单位、仅将 K 次量表生成用于 frozen-snapshot 均值和测量噪声，输出 persona × scale 轨迹、G1−G9 人设特异 forest、结果/过程热图、leave-one-persona-out、描述性方差分解及 CSV；Hedges’ g 同步补充小样本 noncentral-t 95% CI。LLM 回调现可正确保留 `False` 结果且空响应仍回退 failsafe；vLLM 默认改为 Qwen 使用 GPU 0、embedding 使用 GPU 1（显存利用率 0.40）。
- 2026-07-30：合并学长更新，修复 stage_index 问题
- 2026-07-30：新增完成实验存档的安全空间清理工具 `python -m experiment_eval.prune_archive`，现支持不改动源存档、仅把必要内容复制到新目录的 `backup` 模式。完整重复量表报告、评估原始产物、最终 checkpoint、会谈判定 trace 和必要 manifest 会保留，而 incomplete 或尚无 repeat report 的 run 一律完整救援。
- 2026-07-29：原 `kbd_repeat/` 已重构为统一实验指标包 `experiment_eval/`，绘图代码集中到 `experiment_eval/visualization/`，统一入口为 `python docs/analysis/run_experiment_evaluation.py`；新增规范化测量长表、外层 run 轨迹/AUC/nadir/rebound、PHQ/BDI 收敛、ICC、时间点和终点 contrast、Hedges’ g、waterfall、安全代理、会谈剂量、CBT prompt 完成与阶段停留分析。
- 2026-07-28：动态抑郁模块新增独立 LLM 调用 trace：在启用 `depression_dynamic.log_enabled` 时，仿真会将主诉图初始化、对话预览及 chat/reflection 提交期间的调用追加写入 `results/checkpoints/<实验名>/judge_traces/depression_dynamic_llm_trace.jsonl`。每条记录包含仿真时间、角色、调用类型（主诉图推进/规划、情绪推断或未知）、结构化交互上下文、完整 prompt/response 与成功标记；聊天调用同时记录患者与对方发言。trace 写入失败只告警一次且不影响仿真，续跑会继续追加既有文件。
- 2026-07-28：Progressive Stage D 升级为 `progressive_d_v3_1_calibrated_batch_control`：会后批量评估改用中性的分级进度规则，明确区分患者侧 complete、partial 与 none，避免仅凭治疗师材料、一般性参与或语义重叠判定全部完成；`side_step / soft_step_back / hold` 现在优先于本地完成线并阻止本场推进。常规完成线仍为至少一个 complete 且累计得分达到 60%；满 3 场兜底收紧为至少一个 complete 且得分达到 40%。新版本只用于新实验，旧 `progressive_d_v3_batch_control` checkpoint 不混用新规则续跑。
- 2026-07-28：新增持续 forced-LLM 失败的只读检测与可选自动恢复：`detect_forced_llm_rollback.py` 以 trace sidecar 为主、日志为补充，区分仿真污染回溯、仅坏重复复评重算、正常续跑和无安全锚点；`prepare_sparse_checkpoint_resume.py` 支持指定已校验锚点，回溯时按锚点 sidecar 过滤恢复 `consult_history`、隔离关联派生产物并备份较新内容，批量续跑脚本可通过 `--auto-detect-forced-llm-rollback` 自动执行上述分流。与此同时，关系/聊天摘要固定使用本地模型、CBT State Tracker 改走 think-LLM，避免无关调用计入 forced 路由；医嘱仅在正式 `doctor_consult` 结束后抽取，并以会谈开始时间稳定记录任务与时间戳。
- 2026-07-27：会话判定 trace 新增按患者发言记录的动态主诉图 transition（含 hold/advance、节点/指针、原因与实际变化标记），并在 session 汇总会谈前后及会后反思状态；新增 `backfill_judge_complaint_graph.py`，可对历史 checkpoint 先 dry-run、后在备份与一致性校验保护下原子回填主 trace、sidecar 和 experiment-data 副本。`run_batch_then_repeat_eval.sh` 现默认使用 `progressive` 控制器与 Stage D。
- 2026-07-27：新增可选的 CBT 动态控制器并保持 `legacy` 为默认模式：`minimal` 模式将患者发言依次交给六字段 State Tracker、确定性 Strategy Router 和候选受限 Judge，医生侧只接收单一主策略、微技能与本轮目标；会后进展与阶段推进均由代码执行保守校验，风险不明/曾出现高风险、被动认可、医生侧证据、组件异常和重复会谈不会误触发推进。当前链路进一步用 `P1/P2…` 患者发言编号代替复制原话作为证据，并新增中英 CBT 术语表：Prompt 向模型展示中文短名与说明，JSON 和程序校验继续使用稳定英文 ID。
- 2026-07-27：完成 Progressive Stage A–D 渐进式控制链路：在既有固定会谈提纲之上增加静态阶段/子目标 shadow tracking、持久化压缩进度与 Judge 引导，并逐步启用受限的 `stay / advance_subgoal / side_step / advance_stage / soft_step_back / hold` 转换；Stage D 当时升级为 `progressive_d_v3_batch_control`：State Tracker 仅根据患者内部状态输出五字段控制状态，Judge 同时读取压缩内部材料和全部子目标进度，会后单次批量评估按固定顺序覆盖全部子目标并单调合并，且不再保存患者引文或发言编号。当时版本的固定提纲完成采用本地宽松审计（至少一个完成且进度分达 60%，或满 3 场且存在实质进展）；该规则已由 2026-07-28 的 v3.1 分级校准规则替代。高风险会阻止推进；`possible` 风险完成一次安全核对后可恢复常规 CBT 路由，而曾出现 `high` 风险会持续锁定安全优先。Stage D 使用专用三类 Prompt，入口会校验对应资源和控制器版本；会后控制仍是唯一推进来源并保持失败保守、checkpoint 兼容及重放幂等。
- 2026-07-27（历史记录）：当时为批量实验、重复量表复评和稀疏 checkpoint 续跑补齐 controller-aware 入口，曾支持 Progressive A/B/C/D；自 2026-08-01 起 `--cbt-controller progressive` 自动表示唯一的 D，A/B/C 已移除。
- 2026-07-24：实验脚本跳过POST评估
- 2026-07-22：新增咨询室轻量仿真模式：`runshells/run_batch_experiment.py --counsel-room` 会加载 6×7 咨询室地图和“卡布达＋蜻蜓队长”双角色最小配置，支持 `Counsel-G4-MILD` 与 `Counsel-KBD2-G4-MOD`/`Counsel-ALL-G4-ALL` 等 KBD1–9 × 三种严重度选择器；变体保留原有人设和抑郁配置，同时叠加咨询室空间状态，非会诊步可跳过感知及自发对话以降低开销、会诊锁定期间仍走完整流程。批量汇总会按单一 G4 维度输出，`compress.py` 和单次流水线可从存档自动识别地图与角色花名册；目前回放前端仍仅有村庄底图模板。操作指南见 [`README_counsel_room.md`](README_counsel_room.md)。
- 2026-07-22：新增归档重复量表报告工具（现已迁入 `experiment_eval/`）：它会严格校验固定 K 次完整复评、聚合字段及重复编号，从重复量表 summary JSON 生成轨迹、终点变化、最佳改善/反弹和测量可靠性图，并输出 PNG/SVG/PDF、统计明细 CSV、条目统计、Markdown 报告、图表索引和批次清单。
- 2026-07-22：增强实验后处理与恢复韧性：`run_one_experiment.py` 的治疗后量表回答与评分子进程失败时会最多重试 3 次并线性退避，压缩阶段会传入自动识别的资源目录；稀疏 checkpoint 恢复脚本新增 `--repeat-index` 以只处理指定外层重复，并在复评失败时保留退出码、写入明确错误日志和失败汇总。
- 2026-07-19：新增稀疏 checkpoint 安全恢复流水线：通过精确 batch state 定位原 run，从最新通过 runtime/storage bundle、SHA-256、文件集合和 trace sidecar 校验的 staged-eval 时间点回退，恢复前检查活跃进程、LLM/embedding 服务与必需密钥，并将较新 live 产物备份到 `results/recovery_backups/`；批量续跑现按目标总步数扣除已完成步数，达标时直接跳过，且可用新脚本 dry-run、并行恢复多个重复实验，再衔接原 summary 后处理和支持断点续跑的存档重复量表复评。
- 2026-07-19：补强实验运行可观测性与本机服务默认配置：`depression_dynamic` 内部 LLM 正常调用会在 debug 日志中记录完整 prompt/response，便于追溯动态主诉链路。
- 2026-07-19：统一增强 LLM、Ollama、embedding 与重复量表评分 worker 的失败诊断：错误日志新增调用方、阶段、模型、脱敏 endpoint、重试次数、耗时、异常分类、HTTP 状态/request ID 和有限 cause chain，区分限流/并发、鉴权、超时、连接、服务端及输出解析错误；日志会移除 URL 凭据、查询参数、API key、Authorization 和提示词/回答正文，同时保持原有重试、failsafe、最终抛错或空检索结果语义，并新增对应回归测试。
- 2026-07-19：新增仅管理 Qwen chat 的多卡张量并行脚本 `runshells/vllm_chat_tp.sh`，可在不影响既有 embedding 服务的情况下执行 `start/stop/restart/status/logs/print-config`，支持 GPU、并行度、显存利用率、上下文长度等环境变量覆盖，并提供逐卡显存预检、PID 归属校验、端口冲突检测、API 健康检查和启动超时诊断；`docs/RUNBOOK.md`同步补充部署示例与显存说明。
- 2026-07-19：修正批量评估轨迹的基线语义：`delta_from_baseline`只使用具有有效量表分数的 T0，T0 仅快照而未评分时不再把后续首个评估点误当基线；同时将“批量仿真→重复复评”脚本的日期、KBD、居民聊天组和严重程度提取为统一参数并据此生成 condition、任务名和日志名，减少切换实验组合时的漏改风险。
- 2026-07-18：新增严格的分阶段评估时间点快照：`staged_eval.execution_mode`默认改为`capture_only`，在 T0、session 等触发点冻结当时的完整本地 storage、runtime config 与对话上下文并生成带 SHA-256、文件数和大小校验的 snapshot bundle，不再在仿真过程中启动量表 worker；存档重复复评会校验 bundle 路径、清单、内容摘要及文件集合，拒绝损坏快照、外置记忆读取和默认复用最终 storage，旧存档仅可通过`--allow-legacy-final-storage`显式降级并在 JSON/Markdown 报告中标记为未验证，同时批量汇总会区分“仅快照”与“已执行评估”。
- 2026-07-18：重构存档与即时重复量表复评：对每个 agent×评估点固定执行 K 次完整 PHQ-9 / BDI-II（默认 K=10），仅在 K 次结果齐全时生成主结果；每次评估区分“评分 LLM 自报总分”“LLM 条目分之和”和“患者明确且无歧义的 0-3 分回答覆盖冲突条目后的复核总分”，主分采用 K 次复核总分均值。评分 JSON 会严格拒绝非整数、重复、缺失、错序或错号条目；报告继续提供样本 SD、t 分布 95% CI、类别分布/翻转、首轮与中位数等敏感性分析以及 ICC(1,1)/ICC(1,K)，不恢复不稳定条目自适应补跑及投票重构主分逻辑。
- 2026-07-16：新增 G9 积极居民聊天组：复用 G3 的随机居民聊天调度，新增非治疗性、低施压的积极支持前缀并限定只注入居民侧；批量实验 condition、流水线示例、项目文档与回归测试同步支持 `G9`，用于和 G3 中性聊天、G5 负面支持进行话语效价对照。
- 2026-07-13：增强存档重复量表复评的断点续跑能力：`runshells/run_archived_repeat_scale_eval.py`默认启用`--resume-partial`，同一`--name`重跑时会校验并复用完整的 answered/scored，只补跑缺失或损坏阶段；损坏的 job/metadata 不会被误判为完成。只要任一 condition×评估点×量表缺少固定 K 次有效结果，就只写`*_incomplete.json/.md`诊断文件并以非零状态退出，不生成正式`*_summary.json/.md`。
- 2026-07-12：优化强制医患对话的判断LLM提示词：改为围绕当前阶段唯一缺口进行“承接后最小推进”，明确防重复探索、阶段适配与约18个医生轮次的收束节奏；`advice`统一为包含当前缺口、承接、动作、提问意图和避免重复项的单行结构，并使终止判断与回复策略保持一致。
- 2026-07-12：修正并强化居民聊天干预：`ResidentChatScheduler`默认将居民侧设为`prompt_target=doctor`，G3/G5组显式指定该目标并新增回归测试，确保中性社交和负面支持提示词只注入被选中居民；同时收紧中性聊天的安慰/实际帮助边界，规范负面支持的伤害强度、篇幅及快速收束方式。
- 2026-07-12：`runshells/run_batch_then_repeat_eval.sh`支持通过`--repeat-count`和`--max-parallel`运行多轮独立的“批量仿真→重复量表复评”流水线；每轮自动使用`-01`、`-02`等独立名称、summary和日志，并提供进度、失败轮次汇总及中断时的子进程清理。
- 2026-07-12：修复回访对话未正确计入staged_eval计数问题。
- 2026-07-10：合并学长更新
- 2026-07-08：增强 Docker 镜像对 Ollama/Clash 一体化运行的支持：`docker/Dockerfile.txt`默认镜像标签改为`generative-agents-cn-vllm-ollama`，支持 Linux host proxy 构建、Miniconda/conda 清华镜像下载、可选安装 Mihomo/Clash 和 Ollama，并新增`/workspace/ollama-models`、`/workspace/clash`、`.ollama/.clash`日志目录及`7890/9090/11434`端口暴露；`docker/Dockerfile-0708.txt`保留原vLLM镜像模板快照便于回退对照。
- 2026-07-08：新增 Ollama 与 Clash/Mihomo 服务脚本：`runshells/ollama_services.sh`可从`data/config.json`读取chat/embedding模型和端口，支持`start/stop/restart/status/print-config`、GPU绑定、模型预加载和本地OpenAI兼容端点提示；`runshells/download_ollama_models.sh`可按配置拉取Ollama模型并支持`PROXY_URL`；`runshells/clash_services.sh`可启动代理、下载订阅配置、输出代理环境变量，并串联`download-ollama`下载模型。
- 2026-07-08：新增未完成实验的即时重复量表复评入口：`runshells/run_immediate_repeat_scale_eval.py`可基于指定checkpoint的最新`simulate-*.json`和`storage/`即时生成`NOW`评估点，刷新runtime_config中的本地记忆索引，自动合成缺失的原始summary，并委托`run_archived_repeat_scale_eval.py`执行重复量表、稳定性补跑和报告汇总；支持`--labels auto`、`--include-now`、`--report-only`、`--reset-target-depression-state`、`--output-group`和`--use-vllm-models`。
- 2026-07-07：新增 CBT prompt 完成后的固定回访提示词会谈：`session_prompt_injection.post_treatment_followup` 默认启用并使用 `data/prompts/intervention/post_treatment_followup.txt`；这些会谈仍由强制医患咨询触发，只是不再推进新的 session prompt，因此属于治疗后固定提示词观察阶段，不是真正的无干预延迟随访。
- 2026-07-07：补齐会谈完成计数状态：`modules/intervention_manager.py`新增`completed_meeting_state`，在强制医患会谈闭环时按医患pair记录完成次数、meeting_id和结束时间；`modules/staged_eval_manager.py`新增`intervention_completed_meeting`计数来源，使阶段评估在缺少咨询历史或跳过部分会后产物时仍能基于真实完成会谈数触发。
- 2026-07-07：调整当前实验默认开关：`data/config.json`默认开启`intervention.order_extract`和`intervention.environment_model`，并在G1医生干预组启用环境模型；同时默认关闭`staged_eval.t4_enabled`和`auto_stop_after_t4_done`，避免T4完成后自动收尾影响后续回访/复评流程。
- 2026-07-07：增强批量实验与复评流水线：`runshells/run_batch_experiment.py`会在`memory_write_control`屏蔽目标agent全部`event/thought/chat`本地记忆时跳过角色记忆可视化；新增`runshells/run_batch_then_repeat_eval.sh`串联“批量仿真 -> archived repeat scale eval”；`runshells/run_archived_repeat_scale_eval.py`对初始不稳定且无严格多数的条目会先等补跑轮次完成，再按定稿规则处理并更新稳定性提示文案。
- 2026-07-06：接入“医生作业抽取 -> Environment Model -> 会后任务结果记忆”链路：`intervention.order_extract`改为只抽取医患双方已确认的`describe/date`任务，`intervention.environment_model`可通过`forced_llm`/`think_llm`/`custom_llm`生成作业落地结果，并经`MemoryInjectionManager.inject_many`写入患者及参与居民的`event`记忆；新增`data/prompts/intervention/environment_task_outcome.txt`、`environment_task_state.audit/reflected_batches`、`ENV_TASK_*`日志和任务结果后的可选反思触发，默认配置仍保持关闭。
- 2026-07-06：新增可控反思触发策略：`data/config.json`的`agent.think.reflection_policy`支持旧`poignancy_max`阈值、每6步定期反思、会谈结束后条件反思和环境任务结果后的条件反思；`modules/agent.py`统一`reflect/trigger_reflection`入口并记录触发来源、会话上下文和反思去重状态，`modules/intervention_manager.py`按`meeting_kind`/`target_role`/`once_per_meeting`在医患会诊或居民聊天闭环后触发目标角色反思。
- 2026-07-06：扩展强制会谈Prompt全链路追踪：`modules/agent.py`的`completion`支持显式`_forced_prompt_trace`参数，`modules/intervention_manager.py`可把`order_extract_llm`和`environment_model_llm`纳入`forced_prompt_trace_state`并在Markdown报告中统计对应调用次数，便于同时审查患者/医生发言、判断LLM、会后评估、医嘱抽取和环境模型输出。
- 2026-07-06：收紧医嘱抽取、对话结束判断和普通居民聊天Prompt：`data/prompts/extract_doctor_order.txt`只保留双方已达成共识的会后任务，不再输出时间、地点、时长和置信度；`data/prompts/decide_chat_terminate.txt`明确普通居民聊天场景，并把道别、离开、去睡、让对方去忙、退缩式收束等纳入结束信号；中性/负面居民聊天Prompt限制治疗性安慰、情绪练习和行为激活式建议，`terminate_detection_tail_window_turns`从9扩到12。
- 2026-07-06：新增G6支持性心理咨询实验组：`experiments/config/groups/g6_supportive_counseling.json`启用定期医患会谈但关闭CBT session prompt、判断LLM、会后评估和咨询记录，新增`data/prompts/intervention/supportive_counseling_doctor.txt`作为仅医生侧注入的支持性咨询提示词；`runshells/run_batch_experiment.py`同步把`g6`纳入批量condition选择。
- 2026-07-06：新增G7记忆移除实验组与运行时记忆写入控制：`experiments/config/groups/g7_memory_removed.json`通过`agent.memory_write_control`屏蔽卡布达的`event`/`thought`/`chat`写入；`modules/agent.py`新增全局+角色局部配置合并、目标agent/节点类型匹配和`[MEMORY_WRITE_BLOCKED]`日志，默认配置在`data/config.json`中保持关闭，便于做“有干预但患者不形成新记忆”的消融对照。
- 2026-07-06：扩展强制会谈Prompt注入粒度：`data/config.json`的`meeting_rules`新增`prompt_target`默认值`all`，`modules/intervention_manager.py`在排队会谈与规则触发时携带该字段，并支持`all`/`doctor`/`patient`三种注入目标，避免G6等条件把医生专用提示词误注入患者侧。
- 2026-07-06：更新CBT与判断LLM提示词：`data/prompts/intervention_prompts.json`补齐并重写第三、第四阶段流程，突出“负面信念松动”“防失败行为实验”“未执行复盘”“心理急救包”和最终毕业会谈等低门槛推进策略；`data/prompts/intervention/dialog_judge.txt`进一步限制`advice`只输出医生下一句的回复策略/提问意图，禁止生成可直接照抄的完整医生话术
- 2026-07-06：增强存档重复量表复评的定稿与报告逻辑：`runshells/run_archived_repeat_scale_eval.py`默认开启条目稳定性补跑，新增`--reset-target-depression-state`和`--output-group`，可把复评结果映射到新group并可重置目标患者动态抑郁状态；报告主分数改为“评分复核后的条目最终分之和”，无严格多数条目按最高票并列分中的最低分定稿，同时保留重复均值、低分定稿条目、来源condition/group和扩展后的Group×Severity矩阵。
- 2026-07-06：调整本地运行与仓库引用默认项：`data/config.json`默认LLM/embedding从vLLM OpenAI兼容端点切回本机Ollama（`qwen3:8b-q4_K_M`、`bge-m3:latest`）
- 2026-07-05：增强`staged_eval`触发能力：`modules/staged_eval_manager.py`新增`staged_eval.step_interval`步数间隔触发，可按固定step生成虚拟`session_N`评估点并保留触发元数据；`data/config.json`默认关闭该模式，`experiments/config/groups/g2_no_intervention.json`为无干预组开启每24步、虚拟4个session间隔的阶段评估，便于无会诊完成事件时仍可得到纵向量表点。
- 2026-07-05：关闭并移除卡布达系列“会诊完成后注入正向/恢复记忆”规则：`data/config.json`将`session_completed_memory_rule_kbd.enabled`改为`false`，`data/intervention/memory_injections*.json`删除KBD1到KBD9的`session_completed_memory_rule_kbd`条目，避免会后额外正向事件/认知记忆影响干预组与对照组的结果可比性。
- 2026-07-05：改进批量实验量表汇总：`runshells/run_batch_experiment.py`新增PHQ-9/BDI-II预期条目数和程度分级函数，汇总时优先使用条目分合计计算`total_score`与`severity`，同时保留LLM报告的`reported_total_score`和`score_source`，减少总分字段与条目分不一致造成的报告偏差。
- 2026-07-05：大幅增强存档重复量表复评：`runshells/run_archived_repeat_scale_eval.py`新增脚本顶部可调参数、`session_16`默认评估点、缺失原始summary时从`batch_state`/checkpoint自动合成摘要、按condition筛选、条目稳定性分析、仅对不稳定条目自适应补跑、校正总分参考和Markdown“条目稳定性与补跑”报告。
- 2026-07-05：支持量表条目级补跑链路：`runshells/run_staged_eval_worker.py`新增`scale_item_ids`参数，可只回答/评分指定量表条目；配合归档复评脚本用于不稳定条目的追加抽样，而不必整份量表全部重跑。
- 2026-06-30：合并学长更新
- 2026-06-30：增强T0重复量表评估报告：`runshells/run_t0_repeat_eval.py`新增`--report-only`和`--results-dir`，可只读取已有`t0-repeat-*`结果重建`t0_repeat_results.json`与`t0_repeat_report.md`；报告新增“评分结果校验附录”和“复核后对比附录”，对比条目分合计、LLM总分、回答文本中的明确0-3分线索，并输出复核后总分稳定性。
- 2026-06-30：新增存档重复量表复评脚本：`runshells/run_archived_repeat_scale_eval.py`可在不覆盖原始存档的情况下，对已有实验的`T0`、`session_4`、`session_8`、`session_12`、`T4`、`POST`等评估点重复答题和评分；重复结果写入`experiment_data/repeat_scale_eval/<name>/`，并在`experiment_data/reports/`生成类batch summary报告，包含轨迹、最终变化、原报告差异和评分校验。
- 2026-06-30：调整G2无干预组配置：`experiments/config/groups/g2_no_intervention.json`将`intervention.forced_llm.enabled`改为`true`，其余会话队列、判断LLM、会后评估和咨询历史仍保持关闭，便于该组在保留动态抑郁、记忆注入和记忆策略时复用统一LLM通道。
- 2026-06-30：扩展卡布达动态抑郁人设矩阵：新增`卡布达4`到`卡布达9`的完整角色资产与`agent.json`、默认/轻度/中度/重度`depression_config`，补齐`data/intervention/memory_injections_kabuda4.json`到`memory_injections_kabuda9.json`；同时同步调整`卡布达`、`卡布达2`、`卡布达3`已有动态抑郁配置，使压力源、躯体化细节、自动想法和短对话节奏更一致。
- 2026-06-30：扩展卡布达 variant 运行时选择能力：`runshells/kabuda_variant_runtime.py`支持`KBD1`到`KBD9`/`KABUDA1`到`KABUDA9`别名、对应源角色目录和记忆注入文件；`runshells/run_batch_experiment.py`的条件示例同步更新到`Counsel-KBD9-ALL-MOD`等九人设批量实验格式。
- 2026-06-30：增强批量实验汇总的量表评分复核：`runshells/run_batch_experiment.py`在summary payload和Markdown报告中新增“评分结果校验附录”，自动检查PHQ-9/BDI-II条目分合计与LLM总分差异，并从明确回答中抽取0-3分做条目级异常对比。
- 2026-06-30：新增量表稳定性与评分结果校验工具：`runshells/run_t0_repeat_eval.py`可按卡布达变体×严重程度并行重复执行T0 PHQ-9/BDI-II评估并输出聚合报告；`runshells/validate_scale_score_results.py`可扫描结果目录，校验post评分文件、`scale_scores.json`聚合一致性、安全风险字段和回答文本中的显式分数线索。
- 2026-06-29：合并学长更新
- 2026-06-28：将`data/prompts/intervention/dialog_judge.txt`纳入版本管理，并重写判断LLM提示词结构：显式区分“先判断医生下一句、再判断是否结束”，要求advice回应患者核心压力源，必要时用低门槛问题做压力源桥接；若不宜追问也必须说明治疗节奏原因，减少对话长期停留在泛低落/泛自责层面的情况。
- 2026-06-28：优化对话Prompt基础人设注入：`generate_chat.txt`和`generate_chat_external_memory.txt`改用`${base_desc_block}`，`modules/prompt/scratch.py`仅在没有动态抑郁对话块时注入基础描述，避免基础人设与抑郁动态Prompt重复堆叠。
- 2026-06-28：收紧医患历史检索触发条件：`modules/intervention_manager.py`仅在强制干预对话中评估咨询历史，并在写入医患历史前校验`meeting_id`非空，避免普通聊天或缺少会话ID时产生无效历史记录。
- 2026-06-28：调整实验默认参数与压力源记忆：`data/config.json`把`poignancy_max`提高到50、医患历史读取条数降到2，并默认关闭`order_extract`；`data/intervention/memory_injections.json`强化卡布达“休学+试用期被辞退”压力链条及“我什么都做不好”的自动思维。
- 2026-06-28：同步下调卡布达2/卡布达3运行时聊天轮数：`frontend/static/assets/village/agents/卡布达2/agent.json`和`卡布达3/agent.json`的`chat_iter`从18改为4，和当前较短对话实验节奏保持一致。
- 2026-06-27：修复判断LLM读取“上次会话后评估结论”时跨阶段串用的问题：`modules/intervention_manager.py`会记录并校验`current_session`，只有同一阶段才复用历史评估结论；新阶段会显式注入“当前阶段刚开始/第一次对话”的状态说明，避免上一阶段结论被误判为当前阶段已完成步骤。
- 2026-06-27：调整批量实验收尾逻辑：`runshells/run_batch_experiment.py`默认并行数改回1；`POST`评估排序固定放到最后；外置记忆审计失败时不再中断整个condition，而是记录`external_memory_audit_optional_failed`状态并继续保留前面实验结果。
- 2026-06-27：更新容器/数据盘同步脚本：`runshells/sync_project_from_disk.sh`默认不删除目标目录额外文件，只有设置`PROJECT_SYNC_DELETE=1`才做严格镜像；新增`runshells/sync_results_to_disk_loop.sh`，可按固定间隔把容器内`results/`持续同步到数据盘，支持`RESULTS_SYNC_INTERVAL`、`RESULTS_SYNC_ONCE`、`RESULTS_SYNC_DRY_RUN`和`RESULTS_SYNC_DELETE`。
- 2026-06-27：调整vLLM默认资源配置：`runshells/vllm_services.sh`默认将embedding服务放到GPU 1，Qwen和embedding的显存利用率默认均为0.90，并将Qwen默认上下文长度提升到32768；同时新增`data/prompts/intervention/dialog_judge-0627备份.txt`保存判断LLM提示词备份。
- 2026-06-22：新增卡布达 variant 批量替换能力：`runshells/run_batch_experiment.py`的condition扩展为`Counsel-KBD1/KBD2/KBD3-Gx-SEV`格式，并支持`Counsel-ALL-G1-MILD`等通配选择；新增`runshells/kabuda_variant_runtime.py`按condition生成归一化运行时人设文件，运行时目标名仍保持`卡布达`，同时新增`memory_injections_kabuda2.json`和`memory_injections_kabuda3.json`匹配“被比较”“社交误会”两套压力源，并修复resume/post-scale读取checkpoint时覆盖自定义`config_path`的问题。
- 2026-06-22：调整实验默认节奏与检索开销：`data/config.json`中将`chat_iter`降为3，chat记忆检索改为`direct`模式且`similarity_top_k`降为3，医患历史读取和focus检索数量下调；定期医患对话间隔从14步改为6步、单次持续时间改为2881分钟，普通居民聊天最小间隔改为2880分钟。
- 2026-06-22：同步调整随机居民聊天实验组配置：`g3_random_resident_chat.json`和`g5_negative_resident_chat.json`的触发间隔从14步改为6步，便于在较短实验步数内完成更多对话触发与阶段评估观察。
- 2026-06-22：增强批量实验脚本`runshells/run_batch_experiment.py`：默认仿真步数改为120，`--condition`支持重复传入并自动去重，可配合`--max-parallel`并行运行多个实验条件；新增`BATCH_EMBEDDING_BASE_URLS`配置，运行时按并行槽位把不同condition的embedding请求分流到多个BGE endpoint。
- 2026-06-22：升级分阶段vLLM启动脚本`runshells/vllm_services_staged.sh`：默认将Qwen部署在GPU 0-1，将两个BGE-M3 embedding endpoint分别部署在GPU 2和GPU 3，新增`EMBED2_*`、`ENABLE_SECOND_EMBED`等配置，并补齐双embedding服务的启动、等待、状态、日志和停止逻辑。
- 2026-06-22：优化本地记忆检索与对话控制：`modules/memory/associate.py`在无event/thought节点时直接返回，并按`retrieve_max`限制候选召回规模，避免每次全量检索；`modules/agent.py`修复聊天反思证据收集未记录已处理对象的问题，并让强制对话轮数预算至少为1，减少异常配置造成的轮数边界问题。
- 2026-06-17：合并学长关于动态抑郁人设的更新
- 2026-06-17：新增`runshells/recover_staged_eval_scores.py`脚本，作用是给中断实验的staged_eval补上score评分以及report，避免之前的实验浪费。
- 2026-06-17：调整vllm启动参数和config.json相关参数，避免超出上下文限制
- 2026-06-16：新建Dockerfile相关文件，vllm启动脚本新增多卡模式
- 2026-06-15：改造本地医患对话记忆检索：聊天时用对方最近发言作为`retrieve_focus`和历史对话检索query，无对方发言时回退到`other.name + relation`；`retrieve_chats`新增`query`、`prefer_forced`、`force_direct`参数，间隔判断保持对象过滤+近期排序，prompt历史注入先限定对话对象再语义检索并优先召回forced chat；本地chat metadata新增`forced`、`meeting_id`、`expire_days`、`retrieval_scope`，并将`forced_chat_expire_days`设为`-1`以保留完整实验周期的医患咨询记忆。
- 2026-06-14：调整对话触发时间，避免落到夜间睡眠时间。去掉SDS评估。
- 2026-06-13：修复event记忆重复产生的问题
- 2026-06-12：昨天做实验发现16h只跑了7个session对话，太长了。优化`modules/prompt/scratch.py`，使其可以根据`stride`调整计划decompose间隔。调整step=280、stride=720、每14步session对话1次（模拟现实的一周一次）
- 2026-06-11：拉大step值（60->560）、把每4step/定期对话改成每28step/定期对话，模拟现实频率。治疗结束前只保留某些json快照，避免太多了不好审查。
- 2026-06-10：把staged_eval从串行改成并发、把`run_batch_experiment.py`的所有条件实验从串行改成并发
- 2026-06-09：合并学长更新的动态抑郁人设
- 2026-06-09：新增实验脚本`runshells/run_batch_experiment.py`功能：显示运行时间和防磁盘空间不足。修改`config.json`的"max_completed_sessions"配置值，避免staged_eval只有3个的情况。
- 2026-06-08：修复`staged_eval`在G1组触发失败的bug，触发原因是0603的判断条件修改
- 2026-06-07：优化`experiments/config/groups`配置、修复`runshells/run_batch_experiment.py`中断时可能导致`config.json`文件错乱问题
- 2026-06-07：新增vllm部署相关脚本，使用`bash runshells/vllm_services.sh start`命令启动vllm，使用`bash runshells/stop_vllm_services.sh`命令关闭vllm，同时修改了`config.json`和相关llm调用函数，新增`test/live_vllm_preflight.py`运行前vllm健康检查脚本
- 2026-06-05：新增运行前ollama健康检查脚本`test/live_ollama_preflight.py`。新增embedding请求超时机制，之前只设置了chat-llm请求超时。
- 2026-06-04：修改`staged_eval`的实施逻辑，从仿真内改成外挂子进程，避免影响仿真实验的内部状态
- 2026-06-03：修改`staged_eval`的判断条件，统一为对话次数。
- 2026-06-03：同步 README，补充`consult_history`（医患历史摘要）与`staged_eval`说明，更新流程图并移除主流程里对`consult_record`的默认启用描述。
- 2026-06-03：瘦身json快照，把`forced_prompt_traces`和`dialog_judge_trace`放到外面json文件了。
- 2026-06-02：新增`staged_eval`的判定条件，走“定期随机居民对话”时按照对话次数累计判断条件，从而触发阶段性评估
- 2026-06-02：修复“定期医患对话”只触发1次的bug，新增“定期随机居民聊天”也能触发staged_eval
- 2026-06-01：优化llm请求机制、实验脚本增加重试机制，当ollama请求卡住时间过长时中断并重新运行。
- 2026-05-31：新增正常/负面聊天提示词前缀（`data/prompts/intervention/resident_chat_neutral_social.txt`和`data/prompts/intervention/resident_chat_negative_support.txt`），结合“定期随机居民聊天”模块使用。
- 2026-05-31：新增定期随机居民聊天功能，复用“定期医患对话”相关功能实现，作为对照组。具体逻辑在`modules/resident_chat_scheduler.py`里。
- 2026-05-30：新增`runshells/run_batch_experiment.py`脚本，用于批量实验。新增`experiment`文件夹存放不同实验组的配置文件。
- 2026-05-30：仿真过程中增加评估环节（与app.py同功能），为了方便实现计划中的“阶段评估”，不需要人工打断再resume。具体逻辑在`modules/staged_eval_manager.py`里
- 2026-05-29：改善聊天没聊到“压力源”问题：新增卡布达初始化记忆注入条数，修改session1的Prompt，删除session_loop配置，给judge-llm总结患者状态时的Prompt中增加压力源约束，给judge-llm提示词增加软约束，**调整患者提示词**`data/prompts/depression/dynamic_prompt_layers.txt`
  - 关于`dynamic_prompt_layers.txt`改造：如果想再收干净一点，下一步可以把 modules/depression/prompt_builder.py:129-163 这段 chain_window_section 的构建也裁掉，但这不是必须项。
- 2026-05-29：修复<患者状态>总结给judge-llm时，因患者Prompt结构更新而引起空缺的问题。增加若干初始记忆注入`data/intervention/memory_injections.json`
- 2026-05-28：合并动态抑郁人设的更新。
- 2026-05-27：修复`customization/depression_scale_agent/ExpertLLM.py `读取api-key的问题。
- 2026-05-26：新增可视化，把“医患对话历史调用模块”中gate_llm和summary_llm输出融进Prompt可视化里
- 2026-05-25：外置记忆服务中，拉长ingest后立马升级L2/里程碑的重试时间，避免服务解析时间太长而超时
- 2026-05-25：修复chat记忆和某些thought记忆重复写入的问题。
- 2026-05-23：修复对比脚本`runshells/run_extra_scale_compare.py`不兼容问题，现在可以指定2个存档的`scales/scale_scores.json`进行量表差异分析对比。
- 2026-05-22：新增医患对话历史调用模块，功能主要是：1. 每次医患对话结束时都会把完整对话记录存到【医患历史对话记忆库】里，并把node方式的chat_summary作为检索向量。2. 医生/患者发言前由gate-llm判断是否需要检索该记忆库，若需要则检索并返回若干个相关的完整对话记录。3. 若需要检索并返回成功，则调用llm进行总结，形成“记忆摘要”。
- 2026-05-22：修复抑郁主诉链只有4个的问题，主要触因是主诉链更新Prompt以及一致性检查函数去除未知id。实验后发现有一些问题，做了二次改造，主要是允许新增多个后续主诉节点（next_chain）的树形结构而非只有1个的线性结构。
- 2026-05-20：修复合并失误的问题，`modules/depression`的代码没问题，但是之前的`agent.py`里没有完全合并学长分支内容。
- 2026-05-19：优化一些Prompt，并且新建`data/prompts/intervention/consultation_memory_summary.txt`专门用于干预对话的“摘要”。
- 2026-05-18：修复`merge_consultation_dialogues.py`脚本依赖`consult_record`配置开启的问题
- 2026-05-17：`config.json`新增配置`agent.associate.recent_dedup_limit`，用于控制event/chat记忆描述相似去重的最近条数范围，设为0时不去重。修复chat记忆的`poignancy`值异常不计入累加而导致thought触发频率下降的问题
- 2026-05-17：根据记忆服务更新，新增2个脚本并优化`visualize_external_memory_audit.py`。新脚本`recheck_memory_embedding_service.py`作用是“调用外置记忆服务的 embedding 重探活接口”，检索服务重启时使用。`inspect_memory_user_stats_by_save.py`作用是查看某存档所有角色记忆总体情况，目前看下来所有记忆在运行期间都是"raw_fallback"状态，后续问问怎么回事。
- 2026-05-16：仿真实验脚本`runshells/run_one_experiment.py`新增可视化本地记忆、外置记忆系统的环节（相当于自动执行`visualize_agent_memory.py`和`visualize_external_memory_audit.py`）
- 2026-05-16：完整删除旧人设系统，现在只有学长的动态抑郁人设
- 2026-05-15：`config.json`新增配置"reflect_focus_topk"和"reflect_insights_topk"，用于控制反思输出数量。同时降低"poignancy_max"数值，避免反思触发间隔太大
- 2026-05-15：新增仿真后量表批量测试、重复测试脚本`runshells/run_extra_scale_eval.py`。设置好量表以及重复次数后运行，在`results/experiment_data/xxx/scales`可以看到单独以及汇总结果。
- 2026-05-15：合并学长更新的动态抑郁人设，并且修复人设读取不成功的潜在问题。
- 2026-05-14：新增测试脚本`test/live_external_memory_retrieve_dump.py`，用于测试外置记忆系统的记忆召回功能。发现问题：无论`query`是什么，外置记忆系统召回的内容完全一样，准备反馈给那边的同学。
- 2026-05-14：合并欧博的脚本功能，并修正为适用于本工作区的版本。另外参照欧博脚本新增了自动化跑一次实验并评估的脚本`runshells/run_one_experiment.py`。（**特殊作用**：可以单独跑一个`step = 1`的结果，用与评估初始化状态的抑郁程度）
- 2026-05-13：重组实验结果目录结构。experiment_data 条件目录增加 configs/traces/scales 子目录；questions/ 拆分为 templates/scoring_prompts/adhoc；experiment_logs 合并到 experiment_data/logs。详见 `counsel_context/实验结果目录重组说明.md`。（来自欧博）
- 2026-05-13：在`readme.md`第五章增加流程图，方便理解流程
- 2026-05-13：新增功能：使用think.llm压缩患者状态信息后再注入给judge-llm，避免信息冗余
- 2026-05-12：新增`scripts\intervention_prompt_txt_sync.py`脚本，方便查看和修改 session Prompt。
- 2026-05-12：根据心理医生反馈调整 CBT session Prompt、judge-llm Prompt 和 session-eval Prompt，主要降低患者表达能力的要求、让session-eval和judge-llm输出更合理。session-eval的Prompt里新增了当前session停留情况，避免长时间停留某个session里
- 2026-05-11：新增3个量表的单独提示词，在app.py中复制到专家模型提示词里。新增3个量表的v2版本，放到`customization\depression_scale_agent\questions`文件夹里。
- 2026-05-10：将"memory_policy"配置落实到app.py中。config.json新增记忆排序权重调整项并落地到快照里。（关于系统自带的记忆系统，与外置记忆系统无关）
- 2026-05-07：可视化Prompt使用后，优化以前 Prompt 注入中的一些问题。
- 2026-05-07：新增可视化强制干预对话时LLM调用的Prompt，包括患者、判断LLM、医生、评估LLM。可视化md文档在`result/checkpoints/xxx/force_prompt_traces`里。注：md文档命名的时间不一定是发生对话的时间。
- 2026-05-07：合并学长更新的主诉链分支。
- 2026-05-06：修复emotion模块在app.py以及仿真链路中未能正常显示的问题，因为emotion模块绑定在了旧人设链路中，待学长修复后再进行二次更新。
- 2026-04-28：手动升级session对话记忆、注入记忆的层级。（0429发现bug，已修复）
  - 相关文件：`data/config.json`、`modules/external_memory_bridge.py`、`modules/memory_injection_manager.py`、`modules/intervention_manager.py`
-
- 2026-04-28：新增删除外置记忆系统中某存档所有agent接口、脚本。【待实验】
  - 相关文件：`cleanup_memory_users_by_save.py`、`modules/ec_doll_memory_service_client.py`、

- 2026-04-26：截断外置记忆系统的【近期原文（本服务入库）】记忆数量（"short_term_recent_n"配置项）、去掉 [2026-04-25Txx:xx:xx] 前缀
  - 相关文件：`modules/external_memory_bridge.py`、`data/config.json`

- 2026-04-26：修复启用外置记忆系统时未能正确注入动态抑郁人设问题（0428发现新bug，已修复）
  - 相关文件：
    - `data/config.json`
    - `data/prompts/generate_chat_external_memory.txt`
    - `modules/agent.py`
    - `modules/external_memory_bridge.py`
    - `modules/prompt/scratch.py`
    - `customization/depression_scale_agent/app.py`

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

## config.json 推荐配置（当前主线实验）

以下推荐面向当前主线链路：**外置记忆 + 强制干预 + 动态抑郁人设**。真正生效仍以 `data/config.json` 为准；若要排查行为异常，建议优先看这里，再顺着 `start.py -> modules/game.py -> modules/agent.py / modules/intervention_manager.py` 往下追。

```text
data/config.json
├── agent
│   ├── think
│   │   ├── llm -> provider=ollama / model=qwen3:8b-q4_K_M
│   │   ├── poignancy_max -> 30（推荐先保持 20~40；太大反思太少，太小容易过密）
│   │   ├── reflect_focus_topk -> 4（推荐 3~5）
│   │   └── reflect_insights_topk -> 1（推荐先保持 1）
│   ├── associate
│   │   ├── embedding -> provider=ollama / model=bge-m3:lates
│   │   ├── recent_dedup_limit -> 0（0=不去重；控制event/chat记忆描述去重）
│   └── external_memory
│       ├── enabled -> true（开启外置记忆系统）
│       ├── short_term_recent_n -> 20（控制近期记忆数量）
│       └── updata_to_milestone_apply_scenes -> ["session2.3_completed", "session3.3-A_completed", "session4.2_completed"]（哪些session对话记忆升级L2）
├── intervention
    ├── enabled -> true
    ├── doctor -> 蜻蜓队长
    ├── patients -> [卡布达, 金龟次郎]（后者暂时无设置人设）
    ├── meeting_rules
    │   ├── 推荐：同一轮实验只启用 1 条规则，避免会诊频率叠加
    │   └── 当前启用：fast_every_4step（卡布达）；其他规则按需单独切换
    ├── order_extract.enabled -> true（后续可能删除医嘱提取功能，没什么用感觉）
    ├── session_prompt_injection
    │   ├── enabled -> true
    │   └── namespace -> CBT
    ├── chat_controls
    │   ├── enabled -> true
    │   └── agent_overrides[*]
    │       ├── forced_chat_iter -> 18（控制发言次数上限）
    │       └── forced_chat_min_turns -> 2
    ├── forced_llm
    │   ├── enabled -> true
    │   ├── model -> deepseek-v4-flash
    │   ├── api_key_env -> DEEPSEEK_API_KEY
    │   └── temperature -> 0.5（推荐 0.3~0.7）
    ├── dialog_judge
    │   ├── enabled -> true
    │   ├── force_forced_llm -> true
    │   └── patient_state_summary_retry -> 2
    ├── session_eval
    │   ├── enabled -> true
    │   ├── route -> forced_llm
    │   └── history_recent_n -> 3
    ├── consult_history
    │   ├── enabled -> true（当前主线医患对话建议开启）
    │   ├── retrieve_top_k -> 3
    │   └── summary_route -> forced_llm
    ├── consult_record.enabled -> false（咨询记录模块，感觉没什么用暂时关闭了）
    ├── depression_dynamic
    │   ├── enabled -> true（当前主线建议开启）
    │   └── target_hints -> （详情看config.json，人设使用范围）
    ├── memory_injection.enabled -> true（记忆注入功能）
    └── memory_policy.enabled -> true（使用了小镇自带记忆系统，不用理会）
└── staged_eval
    ├── enabled -> true（仿真内自动量表评估）
    ├── include_t0 -> true（初始基线）
    ├── session_interval -> 2（每完成 2 次对话触发一次阶段评估）
    ├── t4_enabled -> true（治疗结束后延迟观察点）
    └── max_completed_sessions -> 10（阶段评估最多统计到第 10 次完成对话）
```

补充说明：

- 若你本地分支仍保留 `intervention.depression_update` 旧配置，建议继续保持 `enabled=false`，不要与 `depression_dynamic` 同时启用。
- `doctor`、`patients`、`meeting_rules[*].doctor/patient`、`chat_controls.agent_overrides` 里的角色名必须完全一致。
- 推荐先确认 `external_memory`、`forced_llm`、`dialog_judge`、`consult_history`、`session_eval`、`depression_dynamic` 这 6 条链路都通；若要做评估实验，再额外确认 `staged_eval` 触发是否符合预期。

## 1. 项目简介

本项目是一个多智能体仿真系统，在常规行为仿真基础上，扩展了医患场景中的干预会话能力，重点包含：

- 会诊调度与强制会话（Intervention）
- 会话判定与终止检测（Dialog Judge）
- 医患历史检索与摘要（Consult History）
- 会后评估与阶段评估（Session Eval / Staged Eval）
- 本地记忆与外置记忆协同（External Memory）
- 抑郁动态链路（Depression Dynamic）

注意：该项目用于技术与研究场景，不构成医疗建议。

## 2. 代码目录结构

| 路径          | 说明                                                              |
| ------------- | ----------------------------------------------------------------- |
| `start.py`    | 仿真主入口（按 step 推进，写 checkpoint 与对话日志）              |
| `compress.py` | 将 checkpoint 压缩为回放数据（`movement.json`、`simulation.md`）  |
| `replay.py`   | 回放 Web 服务入口（Flask）                                        |
| `runshells/`  | 单次实验、批量实验、量表补跑与结果后处理脚本                      |
| `experiments/`| 分组实验 overlay 配置（如 g1/g2/g3/g5/g6/g7/g9/g10/g11/g12）      |
| `modules/`    | 核心逻辑模块（agent、intervention、memory、depression 等）        |
| `data/`       | 配置与提示词（`data/config.json`、`data/prompts/...`）            |
| `frontend/`   | 回放前端资源与静态资产、动态抑郁人设`depression_config.json`      |
| `results/`    | 仿真输出目录（`checkpoints/`、`experiment_data/`、`compressed/`） |

## 3. 快速使用说明

## 3.1 环境准备

- 建议 Python 3.10+。
- 根目录当前未维护统一 `requirements.txt`；请使用项目现有可运行环境。
- 若新环境首次运行，按报错安装依赖（常见：`flask`、`python-dotenv`、`requests`）。
- 在当前目录下新建`.env`文件，配置deepseek api key，格式为`DEEPSEEK_API_KEY=sk-xxxx`

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

## 3.5 抑郁量表评估（app.py）

使用 Gradio 界面让仿真中的患者 agent 回答量表题目，再由专家模型进行评分评估。

### 启动

```bash
conda activate generative_agents_py310
python -u customization/depression_scale_agent/app.py
```

启动后在终端输出中找到地址（如 `http://127.0.0.1:7860/`），在浏览器中打开。首次加载存档较慢，请耐心等待。

### 操作流程

1. **选择存档与快照**：在界面顶部选择要评估的模拟存档和配置快照（默认选最后一个快照，即治疗后状态）
2. **选择角色**：选择需要评估的患者角色
3. **选择量表问卷**：在"选择 jsonl 文件"下拉框中选择量表文件，可用的量表包括：
   - `PHQ-9.jsonl` / `PHQ-9-v2.jsonl`
   - `BDI-II.jsonl` / `BDI-II-v2.jsonl`
   - `SDS.jsonl` / `SDS-v2.jsonl`
4. **批量问答**：点击"开始批量问答"按钮，等待逐题完成。完成后可下载 `*_answered.jsonl` 文件，同时生成 `*_prompt_trace.json` 和 `*_prompt_trace.md`（患者完整注入 Prompt 追踪）
5. **专家模型评估**：
   - 打开 `customization/depression_scale_agent/questions/scoring_prompts/` 目录下对应量表的评估提示词（如 `PHQ-9评估提示词.md`、`BDI-II评估提示词.md`、`SDS评估提示词.md`）
   - 将评估提示词复制到界面的"System Prompt"输入框中
   - 点击"调用专家模型"，等待输出评估结果
   - 将输出复制到新建文件中保存，作为专家评估分析

### 输出文件

所有输出位于 `customization/depression_scale_agent/questions/adhoc/` 目录：

| 文件                           | 说明                                       |
| ------------------------------ | ------------------------------------------ |
| `*_answered.jsonl`             | 量表问答结果（每题问题 + 回答）            |
| `*_answered_prompt_trace.json` | 完整 Prompt 注入追踪（JSON）               |
| `*_answered_prompt_trace.md`   | 完整 Prompt 注入追踪（Markdown，可读性好） |

这条链路适合做“手动指定存档 / 快照”的单次评估；如果要看仿真过程中的自动阶段评估，见下方 `staged_eval`。

## 3.6 批量实验与仿真内自动评估

### `runshells/run_batch_experiment.py` 在做什么

这个脚本面向“按实验组批量跑仿真并自动汇总评估结果”的场景，主流程是：

1. 为每个 `group × severity` 条件生成独立的 runtime config，不再覆盖共享 `data/config.json`。
2. 按 condition 调度 `start.py` 跑仿真；可通过 `--max-parallel` 并行运行多个条件。
3. 收集 `judge_conversation.json`、`forced_prompt_traces/`、`conversation.json`、`staged_eval/` 等核心产物。
4. 对 `scales/staged/` 下的阶段评估结果自动补齐评分文件。
5. 汇总到 `results/experiment_data/reports/*_summary.json` 和 `*_summary.md`。

直接运行：

```bash
python3 runshells/run_batch_experiment.py
```

常见筛选方式：

```bash
python3 runshells/run_batch_experiment.py --condition Counsel-G1-MILD
python3 runshells/run_batch_experiment.py --condition Counsel-G1-ALL
python3 runshells/run_batch_experiment.py --condition Counsel-ALL-MOD
```

常用控制参数：

```bash
python3 runshells/run_batch_experiment.py --max-parallel 2
python3 runshells/run_batch_experiment.py --name my-batch --resume-batch
python3 runshells/run_batch_experiment.py --name my-batch --resume-condition Counsel-G1-MILD
```

批量状态会写到：

- `results/experiment_data/batch_state/<batch_name>/`
- 每个 condition 一份状态文件，记录 `run_name`、`status`、`resume_allowed`、`last_completed_phase`
- `runtime_configs/`：每个 condition 的运行时配置
- `timings/`：每个 condition 的分片 timing 记录，批次结束后会合并成 `reports/<batch>_timings.jsonl`

脚本顶部常量可以直接改当前批次的 `RUN_NAME`、`STEP`、`STRIDE`、`SCALE_AGENT`，以及是否执行 `merge`、`post_scale`、`compress`、记忆可视化、外置记忆审计。

### `staged_eval` 是怎么触发的

- `start.py` 在首个 step 内先调用 `StagedEvalManager.maybe_run_t0(...)`，用于生成初始基线评估（`T0`）。
- 每个 step 写完快照与对话后，再调用 `maybe_run_post_step_eval(...)` 检查是否触发阶段评估。
- 触发后会先把评估任务排队，再由后台 worker 异步执行；仿真主循环不会同步等待 worker 完成。
- 单个 run 当前默认同一时刻最多只有 1 个后台 `staged_eval` worker；仿真结束前会自动 drain，避免收集结果时漏 trigger。
- 当完成对话次数命中 `session_interval` 倍数时，会生成 `session_2`、`session_4` 这类阶段点。
- `session_N` 仍然沿用历史目录命名，但这里的 `N` 现在表示“第 N 次完成对话”。
- 当目标医患 session 全部完成后，若开启 `t4_enabled`，会在 `t4_after_steps` 个仿真步之后再补一个 `T4` 观察点。

阶段评估原始文件位于：

- `results/checkpoints/<name>/staged_eval/<trigger>/`

收集后的实验目录位于：

- `results/experiment_data/<name>/scales/staged/<trigger>/`

每个 trigger 目录下通常包含：

- `{SCALE}_answered.jsonl`：量表逐题回答
- `{SCALE}_trace.json`：逐题回答时的 trace
- `metadata.json`：触发标签、完成 session 数、快照名、仿真时间等元信息

## 3.7 结果目录

### results/checkpoints/（仿真原始产物）

start.py 直接输出的仿真数据：

- `results/checkpoints/<name>/simulate-*.json`：每步快照
- `results/checkpoints/<name>/conversation.json`：所有对话记录
- `results/checkpoints/<name>/consult_history/`：医患历史对话记忆库与检索索引
- `results/checkpoints/<name>/staged_eval/`：仿真内自动阶段评估原始结果
- `results/checkpoints/<name>/storage/`：agent 向量存储

### results/experiment_data/（实验分析产物）

由 `runshells/run_one_experiment.py` / `runshells/run_batch_experiment.py` 收集和分析；Legacy G1/G2 复现时也可由 `runshells/run_experiment.py` 生成：

- `results/experiment_data/<name>/configs/`：实验输入配置（`original_agent.json`、`original_config.json`、`original_depression_config.json`）
- `results/experiment_data/<name>/traces/`：仿真后处理产物
  - `judge_conversation.json`：[患者-判断LLM-医生] + 评估LLM 输出
  - `forced_prompt_traces/`：每次干预的完整 Prompt 追踪；若开启 `consult_history`，其中会包含 gate / summary 的提示词与输出
  - `merge_consultation_dialogues.json`：合并对话结果；仅在 `consult_record.enabled=true` 时额外附带咨询记录
- `results/experiment_data/<name>/scales/`：量表评估数据
  - `staged/<trigger>/`：仿真内自动阶段评估（如 `T0`、`session_2`、`T4`）
  - `{SCALE}_{phase}_answered.jsonl`：量表问答结果
  - `{SCALE}_{phase}_scored.json`：量表评分结果
  - `scale_scores.json`：汇总评分
  - `expert/`：专家 LLM 评估分析
- `results/experiment_data/<name>/trial_meta.json`：试验元数据
- `results/experiment_data/reports/`：实验报告
  - `analysis_report.md`：综合分析报告
  - `analysis_report_appendix.md`：报告附录（量表问答原始记录）
- `results/experiment_data/logs/`：实验运行日志
- `results/experiment_data/batch_state/<batch_name>/`：批量脚本状态
  - `<condition>.json`：condition 生命周期状态与续跑信息
  - `runtime_configs/`：每个 condition 的 runtime config
  - `timings/`：每个 condition 的 timing 分片

### results/compressed/

- `results/compressed/<name>/`：回放资源（`movement.json`、`simulation.md`）

### customization/depression_scale_agent/questions/

- `questions/templates/`：量表题目定义（`.jsonl`，只读输入）
- `questions/scoring_prompts/`：评分提示词（`*评估提示词.md`，只读输入）
- `questions/adhoc/`：app.py 临时评估输出（`*_answered.jsonl`、`*_prompt_trace.*`）

## 4. data/config.json 模块配置

下面给出与上方“config.json 推荐配置（当前主线实验）”一致的快速索引；更细的推荐值和当前启用状态，优先以上面的配置树与 `data/config.json` 为准。

## 4.1 agent 配置

- `think.llm`、`associate.embedding`：当前主线实验默认都走本地 `ollama`
- `agent.think.poignancy_max`：反思触发阈值；当前为 `30`，一般建议先在 `20~40` 内调
- `agent.think.reflect_focus_topk` / `reflect_insights_topk`：控制反思输出数量；当前分别为 `4` / `1`
- `agent.associate.recent_dedup_limit`：event/chat 记忆近期相似去重范围；`0` 表示不去重，当前为 `32`
- `agent.associate.chat_retrieve.similarity_top_k`：本地语义检索条数；当前为 `6`
- `agent.external_memory`：当前主线建议开启
  - `enabled=true`
  - `read_mode=chat_only`：只在强制干预对话中检索外置记忆
  - `fallback_to_local=true`：外置记忆服务异常时回退本地记忆
  - `short_term_recent_n=20`：近期原文注入数量，建议先在 `10~30` 内调
- `chat_history` / `chat_memory`：仍保留在旧链路里，但通常不是当前主线实验的首要调参点

## 4.2 intervention 配置

- `enabled`：开启强制干预主链路
- `doctor` / `patients`：医患角色定义
- `meeting_queue.enabled`：建议保持开启，避免同一医生的会诊调度混乱
- `meeting_rules`：会诊触发规则
  - 推荐一次实验只启用 **1 条** 规则，避免频率叠加
  - 当前主线启用的是 `fast_every_4step`（卡布达）；其他规则默认关闭，按需单独切换
- `order_extract.enabled`：当前为 `true`；若本轮不分析医嘱，可临时关闭
- `session_prompt_injection`：治疗 Prompt 注入链路，当前主线保持开启，`namespace=CBT`
- `chat_controls`：强制对话轮次控制
  - 当前采用 `defaults.enabled=false + agent_overrides.enabled=true` 的方式
  - `forced_chat_iter=18`：当前每个相关角色都按 override 单独控制
- `forced_llm`：强制干预对话模型；当前为 `deepseek-v4-flash`，需在 `.env` 配置 `DEEPSEEK_API_KEY`
- `dialog_judge`：会话中判断 LLM；当前保持 `enabled=true` 且 `force_forced_llm=true`
- `session_eval`：会话后评估 LLM；当前 `route=forced_llm`，`history_recent_n=3`
- `consult_history`：医患历史摘要模块；通过 gate 判断是否检索历史对话，再把检索命中的完整对话总结成 `consult_history_memory` 注入当前回复
- `consult_record`：咨询记录生成；当前主线为 `enabled=false`，要看 SOAP 记录时再开启
- `depression_dynamic`：动态抑郁人设主链路；当前主线建议保持 `enabled=true`
  - `normal_chain_enabled=true`
  - `forced_chain_enabled=true`
  - `prompt_injection_enabled=true`
  - `event_commit_enabled=true`
  - 角色侧配置见 `frontend/static/assets/village/agents/卡布达/depression_config.json`
- `memory_injection`：记忆注入规则，当前启用了初始化注入和指定 session 完成后的注入
- `memory_policy`：强制对话阶段的检索权重策略；当前主线保持开启

## 4.3 配置注意事项

- `doctor`、`patients`、`meeting_rules[*].doctor/patient`、`chat_controls.agent_overrides` 里的角色名必须完全一致。
- `session_prompt_injection.order` 必须与提示词文件中的 session id 对齐。
- 若本地分支仍保留 `intervention.depression_update` 旧配置，建议继续保持 `enabled=false`，不要与 `depression_dynamic` 同时启用。
- 若启用外置记忆，请确认 `agent.external_memory.base_url` 可访问，且服务端接口就绪；可运行 `test/live_ec_doll_memory_service_health_ready.py` 检查。
- `staged_eval` 的阶段触发现在统一按“完成对话次数”累计；`session_N` 只是历史命名，不再表示 CBT session 数。
- 建议优先确认 `external_memory`、`forced_llm`、`dialog_judge`、`consult_history`、`session_eval`、`depression_dynamic` 这 6 条链路都正常，再去微调 `meeting_rules`、`chat_controls`、`recent_dedup_limit` 等细项。

## 4.4 staged_eval 配置

- `enabled`：是否开启仿真内自动阶段评估
- `target_agent`：被评估的患者角色
- `include_t0`：是否在首个 step 生成基线评估 `T0`
- `session_interval`：每完成多少次对话，触发一次阶段评估
- `max_completed_sessions`：阶段评估最多统计到多少次完成对话数
- `t4_enabled`：是否在治疗完成后再补一个延迟观察点 `T4`
- `t4_after_steps`：治疗完成后再等待多少个 step 触发 `T4`
- `scales`：当前要跑的量表，默认是 `PHQ-9`、`BDI-II`、`SDS`

## 5. 运行流程图总览

下面用“时间推进 -> 整体主循环 -> 强制干预对话中的 LLM 协作 -> 仿真内自动评估”四个视角说明当前工作区的核心流程。

### 5.1 step / stride 的时间流逝机制

- `step`：本次 `simulate()` 要执行多少个“离散仿真步”。
- `stride`：每个 step 结束后，仿真时间统一向前推进多少分钟。
- 一个 step 内，所有 agent 都共享同一个仿真时刻 `T`；不会在 agent 之间单独推进时间。
- 每个 step 的顺序是：**在当前时刻 `T` 完成调度与所有 agent 行为 -> 写 checkpoint / conversation / judge trace -> 最后再 `forward(stride)`**。
- 因此，相邻两个 checkpoint 的时间差通常就是 `stride`。例如：`step=48, stride=60` 表示连续跑 48 个仿真步，每步代表 60 分钟仿真时间。
- `resume` 时，会读取最近一次 checkpoint 的时间，并在此基础上再加一个 `stride` 作为下一步起点。

```mermaid
flowchart LR
    A["起始仿真时间<br/>例如 2026-04-25 09:30"] --> B["step 1 在当前时刻 T 执行<br/>on_step_start + 所有 agent.think"]
    B --> C["写出 checkpoint / conversation / trace<br/>时间仍然记为 T"]
    C --> D["timer.forward(stride)<br/>例如 +60 分钟"]
    D --> E["step 2 在下一时刻 T+stride 执行"]
    E --> F["重复直到跑完 step 个仿真步"]
```

### 5.2 整体主流程

```mermaid
flowchart TD
    A["start.py 读取配置<br/>创建 Game / InterventionManager"] --> B["simulate(step, stride)"]
    B --> C["当前仿真时刻 T"]
    C --> D["InterventionManager.on_step_start(game, T)<br/>检查 meeting_rules / 更新 meeting queue"]
    D --> E["遍历所有 agent"]
    E --> F["before_agent_think(agent, T)<br/>如命中 lock 则重写目标地址"]
    F --> G["agent.think()<br/>move / schedule / percept / plan / reflect"]
    G --> H{"本轮是否进入对话?"}
    H -- 否 --> I["更新 agent 状态 / 坐标"]
    H -- 是 --> J["Agent._chat_with(...)"]
    J --> I
    I --> K{"是否还有下一个 agent?"}
    K -- 是 --> E
    K -- 否 --> L["写 simulate-*.json / conversation.json<br/>judge_traces / forced_prompt_traces"]
    L --> M{"stride > 0 ?"}
    M -- 是 --> N["timer.forward(stride)"]
    M -- 否 --> O["保持当前仿真时间"]
    N --> P{"是否还有剩余 step?"}
    O --> P
    P -- 是 --> C
    P -- 否 --> Q["仿真结束"]
```

可以把它理解为：**每个 step 先做完整轮业务，再统一推进一次仿真时间**。

### 5.3 强制干预对话期间，哪些 LLM / 模块在工作

这里的“强制干预对话”是指：命中 `intervention lock` 后，`before_agent_think(...)` 会把医生当前行动目标重写为 `<persona, patient>`，随后进入 `Agent._chat_with(..., forced=True)`。

```mermaid
flowchart TD
    A["命中 intervention lock"] --> B["before_agent_think 重写医生目标<br/>&lt;persona, patient&gt;"]
    B --> C["Agent._chat_with(..., forced=True)"]
    C --> D["可选：ExternalMemoryBridge.retrieve_chat_context<br/>输出 external_memory_context"]
    C --> E["think.llm：患者状态摘要<br/>输入：患者最近一次 generate_chat Prompt 缓存<br/>输出：patient_state_summary"]
    E --> F["judge_llm ：会话中判断<br/>输入：patient_state_summary + conversation + session_prompt + prev_session_eval_reason<br/>输出：terminate / advice"]
    C --> G["consult_history：gate -> 检索 -> 摘要<br/>输出：consult_history_memory"]
    D --> H["医生 utterance 生成<br/>优先 forced_llm，失败时回退 think.llm"]
    F --> H
    G --> H
    H --> I["患者 utterance 生成<br/>优先 forced_llm，失败时回退 think.llm"]
    G --> I
    I --> J{"是否结束?"}
    J -- 否 --> F
    J -- 是 --> K["after_chat 收尾：清 lock / 更新队列"]
    K --> L["session_eval LLM<br/>输出：efficacy_score / session_end / reason"]
```

#### 5.3.1 强制对话里的 LLM 分工速查

| 环节                                  | 主要模型路由                                                      | 主要输入                                                                                                             | 主要输出 / 作用                           |
| ------------------------------------- | ----------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------- | ----------------------------------------- |
| 患者状态摘要                          | 医生侧 `think.llm`                                                | 患者最近一次 `generate_chat` Prompt 缓存中的动态状态文本                                                             | 给 judge 用的 `patient_state_summary`     |
| 会话中判断 `dialog_judge`             | `forced_llm`                                                      | `patient_state_summary`、当前对话历史、当前 session prompt、上一次 session eval reason                               | `terminate`、`advice`                     |
| 医患历史摘要 `consult_history`        | gate 默认走 `forced_llm`；summary 由 `summary_route` 决定         | 最新一轮对方发言、当前对话历史、命中的历史完整对话记录                                                                | `consult_history_memory`                  |
| 医生回复生成 `generate_chat`          | 强制链路里**优先** `forced_llm`，失败则回退医生自己的 `think.llm` | relation、chats、医生 session prompt、consult record 注入、judge advice、depression block、`external_memory_context`、`consult_history_memory` | 医生自然语言回复                          |
| 患者回复生成 `generate_chat`          | 强制链路里**优先** `forced_llm`，失败则回退患者自己的 `think.llm` | relation、chats、depression block、`external_memory_context`、`consult_history_memory`                              | 患者自然语言回复                          |
| 复读检测 `generate_chat_check_repeat` | 与 `Agent.completion(...)` 相同的强制路由规则                     | 当前对话历史、当前轮回复                                                                                             | 是否出现复读，用于提前结束                |
| 终止检测 `decide_chat_terminate`      | 仅在未启用 `dialog_judge` 时参与；同样优先 `forced_llm`           | 当前对话历史                                                                                                         | 是否结束对话                              |
| 会后评估 `session_eval`               | 由 `session_eval.route` 决定：`forced_llm` 或 `think_llm`         | session prompt、历史 eval reason、usage log、完整对话                                                                | `efficacy_score`、`session_end`、`reason` |

#### 5.3.2 关于模型路由，最容易混淆的点

- `Agent.completion(...)` 在 `forced=True` 的上下文里，不只是 `generate_chat`，很多 agent 侧 Prompt（例如 `summarize_relation`、`generate_chat_check_repeat`、`summarize_chats`，以及在关闭 `dialog_judge` 时的 `decide_chat_terminate`）都会**先尝试走 `intervention.forced_llm`**。
- `consult_history` 不是每轮必跑：当前要求模块开启、角色能解析成医患对、`turn_no > 1`，并且 gate 判断 `need_retrieval=true` 后才会继续检索和摘要。
- 是否真的走 `forced_llm`，要同时满足：
  - `intervention.forced_llm.enabled = true`
  - 当前确实是医生-患者配对
  - 双方 lock 都有效，且 `meeting_id` 一致
  - 对应 API key 已配置
- 如果强制路由不可用或调用失败，agent 侧生成会回退到该角色自己的 `think.llm`。
- `ExternalMemoryBridge.retrieve_chat_context(...)` 本身不是这里的一次 LLM 调用，但它会在回复生成前产出 `external_memory_context`，直接影响后续 `generate_chat` 的输入。

配置与链路映射速查：

- 调度阶段：`meeting_rules`
- 强制对话阶段：`chat_controls` / `forced_llm` / `dialog_judge` / `consult_history`
- 会后阶段：`session_eval` / `memory_injection` / `depression_dynamic` / `consult_record`（可选，默认关闭）

### 5.4 仿真内自动评估（staged_eval）

`staged_eval` 不依赖人工暂停仿真。`start.py` 会在仿真过程中自动检查触发条件，并把每个评估点单独落盘。

```mermaid
flowchart TD
    A["start.py 进入首个 step"] --> B{"include_t0 ?"}
    B -- 是 --> C["执行 T0 评估<br/>写 checkpoints/&lt;name&gt;/staged_eval/T0/"]
    B -- 否 --> D["继续正常仿真"]
    C --> D
    D --> E["每个 step 正常写快照 / conversation"]
    E --> F["maybe_run_post_step_eval(...)"]
    F --> G{"完成对话数是否命中 session_interval ?"}
    G -- 是 --> H["写 session_N 评估目录"]
    G -- 否 --> I{"目标 session 是否已完成且满足 T4 ?"}
    H --> I
    I -- 是 --> J["写 T4 评估目录"]
    I -- 否 --> K["进入下一步仿真"]
    J --> K
```

常见 trigger 含义：

- `T0`：初始基线
- `session_2` / `session_4`：阶段性评估点（历史命名，对应第 2 / 4 次完成对话）
- `T4`：治疗结束后一段时间的延迟观察点

## 6. 运行结果与回放

## 6.1 关键输出文件

- `results/checkpoints/<name>/`：仿真原始产物（快照、对话记录、storage）
- `results/checkpoints/<name>/consult_history/`：医患历史对话记忆库与索引
- `results/checkpoints/<name>/staged_eval/`：仿真内自动阶段评估原始结果
- `results/experiment_data/<name>/configs/`：实验输入配置
- `results/experiment_data/<name>/traces/judge_conversation.json`：[患者-判断LLM-医生] + 评估LLM 的输出结果
- `results/experiment_data/<name>/traces/forced_prompt_traces/`：每次干预的完整 Prompt 追踪
- `results/experiment_data/<name>/traces/merge_consultation_dialogues.json`：合并对话及咨询记录（需运行 `merge_consultation_dialogues.py`）
- `results/experiment_data/<name>/scales/`：量表评估问答、评分、专家分析；其中 `scales/staged/` 是仿真内自动评估
- `results/experiment_data/reports/analysis_report.md`：综合分析报告
- `results/experiment_data/reports/*_summary.{json,md}`：批量实验的汇总结果
- `results/compressed/<name>/`：回放资源（`movement.json`、`simulation.md`）

## 6.2 conversation.json 说明

- 记录每个仿真时刻发生的会话内容。
- 常用于排查“某一步是否发生对话、对话文本是什么、谁与谁在说话”。

## 6.3 judge_conversation.json 说明

- 位于 `judge_traces/` 目录，由仿真过程自动写出
- 用于审计强制会话中`会话中判断LLM`以及`会话后评估LLM`的输出
- 每个 turn 的 `complaint_graph` 记录患者本轮对应的动态主诉图
  `hold/advance`、前后节点、图指针、判定原因和 `changed`；其中
  `changed` 只表示节点或指针实际发生变化
- 每个 session 的 `complaint_graph` 汇总 `before_chat`、`after_chat`、
  `after_reflection` 和患者 chat transition 数量；会后反思的 transition
  单独保存在 `reflection`
- 历史 checkpoint 可在不重新仿真的情况下先 dry-run、再回填：

```bash
python runshells/backfill_judge_complaint_graph.py \
  results/checkpoints/<name> \
  --dry-run

python runshells/backfill_judge_complaint_graph.py \
  results/checkpoints/<name> \
  --write
```

写入模式会先备份到 `results/recovery_backups/`，并同步 checkpoint 主
trace、dialog-judge sidecar 和内容一致的 experiment-data trace；若患者
输出或主诉图 history 无法无歧义对齐，则在写入前终止。

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
- 补充说明：
  - 默认主线里 `consult_record=false`，因此合并结果主要包含患者、判断LLM、医生、会后评估理由。
  - 若显式开启 `consult_record`，则会额外合并 SOAP 结构化咨询记录。
- 基本使用方式：基本同上
-

## 6.7 analyze_depression_dynamic_state.py 作用

该脚本用于分析动态抑郁人设系统的状态变化情况，输出每个step的状态数据，方便查看和分析。(具体由赵学长更新)

## 6.8 cleanup_memory_users_by_save.py作用

该脚本用于删除外置记忆系统中某个存档下的所有agent数据，避免外置记忆系统的数据冗余。

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
- 强制干预会话对话记录及中间产物：`results/experiment_data/<name>/traces/merge_consultation_dialogues.json`

## 8. 维护约定（建议）

- 每次改动 `intervention` 或 `config`，同步更新 README 的“更新日志（近期）”。
- 新增配置项时，在 README 中补“作用 + 关键参数 + 风险提示”。
- 不在 README 示例中写真实密钥、令牌、私有地址。
