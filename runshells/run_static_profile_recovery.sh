#!/usr/bin/env bash
# ==================== 常用配置（修改这里即可） ====================
# 使用已安装项目依赖的 Python；也可通过环境变量指定解释器路径。
PYTHON_BIN="${PYTHON_BIN:-python}"
# 相对路径均以项目根目录为基准。模型和端口读取 agent.think.llm。
CONFIG="data/config.json"
# 填写一个或多个已结束实验的 checkpoint run 目录；命令行目录可替换此数组。
# 示例：RUN_DIRS=("results/checkpoints/你的run名称")
RUN_DIRS=()
AGENTS_DIR="frontend/static/assets/village/agents"  # 历史实验可指定对应版本的人设资产
PROFILE_MAP=""             # 非标准 run 名的 GT 映射 JSON；空字符串表示自动匹配
MODE="both"                # both / all-session / single-session
SEED=42                    # 固定候选抽样和顺序；续跑不可改变
WORKERS=3                  # 并发请求数；2–3 个 vLLM 可先设为部署数量
# config 的 load_balancing.enabled=true 时按 ports 轮换，否则并发访问 base_url。
ATTEMPTS=3                 # 每样本本次最多尝试次数（含第一次）
RETRY_DELAY=2              # 请求/解析/证据校验失败后指数退避，单位秒，上限60秒
RESUME=true                # 跳过成功样本，补做中断任务；首次运行也可开启
RETRY_ERRORS=true          # 续跑时重试已耗尽次数的错误；不重算合法拒答或成功结果
DRY_RUN=false              # true：只检查输入，不调用模型、不写结果
SUMMARY_OUTPUT=""          # 可选跨 run 汇总 JSON 路径；续跑时重建，空表示不生成
# 逐 run 输出：humanlike/static_profile_recovery/；不修改原始存档。
# 旧版无 identity 的输出不能续跑，请先移走旧输出目录。
# =================================================================
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if (( $# > 0 )); then RUN_DIRS=("$@"); fi
if (( ${#RUN_DIRS[@]} == 0 )); then
    echo '请填写顶部 RUN_DIRS，或：bash runshells/run_static_profile_recovery.sh results/checkpoints/<run>' >&2
    exit 2
fi
cd "$PROJECT_ROOT"
args=(--config "$CONFIG" --agents-dir "$AGENTS_DIR" --mode "$MODE" --seed "$SEED"
      --workers "$WORKERS" --attempts "$ATTEMPTS" --retry-delay "$RETRY_DELAY")
if [[ "$RESUME" == true ]]; then
    args+=(--resume)
    if [[ "$RETRY_ERRORS" == true ]]; then args+=(--retry-errors); fi
fi
if [[ "$DRY_RUN" == true ]]; then args+=(--dry-run); fi
if [[ -n "$PROFILE_MAP" ]]; then args+=(--profile-map "$PROFILE_MAP"); fi
if [[ -n "$SUMMARY_OUTPUT" ]]; then args+=(--summary-output "$SUMMARY_OUTPUT"); fi
exec "$PYTHON_BIN" runshells/run_static_profile_recovery.py "${args[@]}" "${RUN_DIRS[@]}"
