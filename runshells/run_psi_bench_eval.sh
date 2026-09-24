#!/usr/bin/env bash
# ==================== 类人 Judge 配置（统一修改这里） ====================
PYTHON_BIN="${PYTHON_BIN:-python}"
JUDGE_BACKEND="deepseek_api"            # local_vllm / deepseek_api
LOCAL_MODEL="qwen3.5-9b-vllm"
LOCAL_BASE_URL="http://127.0.0.1:18000/v1"
LOCAL_PORTS="18000"                   # 单端口 18000；多副本 18000,18002,18003
LOCAL_API_KEY="EMPTY"
DEEPSEEK_MODEL="deepseek-flash"
DEEPSEEK_BASE_URL="https://api.deepseek.com"
DEEPSEEK_API_KEY_ENV="DEEPSEEK_API_KEY" # 密钥本身只放环境变量
JUDGE_TIMEOUT_SECONDS=120

humanlike_judge_args() {
    HUMANLIKE_JUDGE_ARGS=(
        --backend "$JUDGE_BACKEND"
        --local-model "$LOCAL_MODEL" --local-base-url "$LOCAL_BASE_URL"
        --local-ports "$LOCAL_PORTS" --local-api-key "$LOCAL_API_KEY"
        --deepseek-model "$DEEPSEEK_MODEL" --deepseek-base-url "$DEEPSEEK_BASE_URL"
        --deepseek-api-key-env "$DEEPSEEK_API_KEY_ENV"
        --timeout-seconds "$JUDGE_TIMEOUT_SECONDS"
    )
}
# Recovery shell 只读取以上配置，不执行 PTC/Emotion。
if [[ ${HUMANLIKE_JUDGE_SETTINGS_ONLY:-false} == true ]]; then return 0; fi
# ================================================================

# ==================== PTC/Emotion 运行配置 ====================
RUN_DATE="0922"                      # 只选 batch-0922-*；设为空关闭筛选
CHECKPOINTS_ROOT="results/checkpoints"
OUTPUT_ROOT="humanlike_outputs/0922_deepseek"
RUN_DIRS=()                          # 空时按 RUN_DATE 自动发现；命令行可覆盖
WORKERS=3
ATTEMPTS=3
RETRY_DELAY=2
PTC_HISTORY=6
EMOTION_HISTORY=4
RESUME=true
RETRY_ERRORS=true
DRY_RUN=false
SHOW_PROMPTS=0                      # >0 时只读打印前 N 条患者回复的完整 Prompt
# 输出：OUTPUT_ROOT/<run>/PSI-Bench-style-v2-<backend>/
# ================================================================
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if (( $# > 0 )); then RUN_DIRS=("$@"); fi
if [[ -z "$RUN_DATE" && ${#RUN_DIRS[@]} -eq 0 ]]; then
    echo '请填写 RUN_DIRS，或设置 RUN_DATE=0922 自动发现 results/checkpoints 中的存档' >&2
    exit 2
fi
cd "$PROJECT_ROOT"
humanlike_judge_args
args=("${HUMANLIKE_JUDGE_ARGS[@]}" --checkpoints-root "$CHECKPOINTS_ROOT"
      --workers "$WORKERS" --attempts "$ATTEMPTS" --retry-delay "$RETRY_DELAY"
      --ptc-history "$PTC_HISTORY" --emotion-history "$EMOTION_HISTORY")
if [[ -n "$RUN_DATE" ]]; then args+=(--run-date "$RUN_DATE"); fi
if [[ -n "$OUTPUT_ROOT" ]]; then args+=(--output-root "$OUTPUT_ROOT"); fi
if [[ "$RESUME" == true ]]; then
    args+=(--resume)
    if [[ "$RETRY_ERRORS" == true ]]; then args+=(--retry-errors); fi
fi
if [[ "$DRY_RUN" == true ]]; then args+=(--dry-run); fi
if (( SHOW_PROMPTS > 0 )); then args+=(--show-prompts "$SHOW_PROMPTS"); fi
exec "$PYTHON_BIN" runshells/run_psi_bench_eval.py "${args[@]}" "${RUN_DIRS[@]}"
