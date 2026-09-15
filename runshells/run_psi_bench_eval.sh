#!/usr/bin/env bash
# ==================== 常用配置（修改这里即可） ====================
# Python 需来自已安装项目依赖的环境；也可 PYTHON_BIN=/path/to/python bash 本脚本。
PYTHON_BIN="${PYTHON_BIN:-python}"
# 相对路径以项目根目录为基准。端点/模型/轮换端口在此 JSON 的 agent.think.llm 中配置。
CONFIG="data/config智算中心版.json"
# 可以填写多个已结束实验的 checkpoint 目录。空数组时必须通过命令行传入目录。
# 示例：RUN_DIRS=("results/checkpoints/你的run名称")
RUN_DIRS=()
# 同时在途的分类请求数。2–3 个 vLLM 可先用 3，按显存和吞吐调整。
# load_balancing.enabled=true 时按 ports 轮换；false 时全部请求发送到 base_url。
WORKERS=3
# 每个分类器本次运行最多尝试次数（含第一次），请求失败/JSON解析失败均重试。
ATTEMPTS=3
RETRY_DELAY=2                # 重试等待基数（秒），指数递增，上限 60 秒
PTC_HISTORY=6               # 当前回复之前最近几条医患消息，不是患者轮数
EMOTION_HISTORY=4
RESUME=true                 # 相同输入/模型/Prompt 下跳过成功结果，补做中断任务
RETRY_ERRORS=true           # 续跑时重新尝试已耗尽重试的 error；不重算成功标签
DRY_RUN=false               # true：仅验证输入提取，不调用模型、不写结果
# 输出固定在每个 run 的 humanlike/PSI-Bench-style/，不修改原存档。
# =================================================================
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
# 命令行可替代顶部 RUN_DIRS；相对路径同样以项目根目录为基准。
if (( $# > 0 )); then
    RUN_DIRS=("$@")
fi
if (( ${#RUN_DIRS[@]} == 0 )); then
    echo '请填写脚本顶部 RUN_DIRS，或：bash runshells/run_psi_bench_eval.sh results/checkpoints/<run>' >&2
    exit 2
fi
cd "$PROJECT_ROOT"
args=(--config "$CONFIG" --workers "$WORKERS" --attempts "$ATTEMPTS"
      --retry-delay "$RETRY_DELAY" --ptc-history "$PTC_HISTORY" --emotion-history "$EMOTION_HISTORY")
if [[ "$RESUME" == true ]]; then
    args+=(--resume)
    if [[ "$RETRY_ERRORS" == true ]]; then args+=(--retry-errors); fi
fi
if [[ "$DRY_RUN" == true ]]; then args+=(--dry-run); fi
exec "$PYTHON_BIN" runshells/run_psi_bench_eval.py "${args[@]}" "${RUN_DIRS[@]}"
