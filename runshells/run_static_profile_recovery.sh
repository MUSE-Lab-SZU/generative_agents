#!/usr/bin/env bash
# 统一 Judge 配置位于 run_psi_bench_eval.sh 最顶端。
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
HUMANLIKE_JUDGE_SETTINGS_ONLY=true
source "$PROJECT_ROOT/runshells/run_psi_bench_eval.sh"
unset HUMANLIKE_JUDGE_SETTINGS_ONLY

# ==================== Recovery 运行配置 ====================
RUN_DATE="0922"                      # 只选 batch-0922-*；设为空关闭筛选
CHECKPOINTS_ROOT="results/checkpoints"
OUTPUT_ROOT="humanlike_outputs/0922_deepseek"
RUN_DIRS=()                          # 空时按 RUN_DATE 自动发现；命令行可覆盖
SOURCE="consult"                    # consult / resident
CHAT_KIND="all"                     # resident 来源：all / forced / spontaneous
PATIENT_NAME=""                     # 归档角色缺失时可指定
AGENTS_DIR="frontend/static/assets/village/agents"
PROFILE_MAP=""
MODE="single-session"              # both / all-session / single-session
SEED=42
WORKERS=3
ATTEMPTS=3
RETRY_DELAY=2
RESUME=true
RETRY_ERRORS=true
DRY_RUN=false
SHOW_PROMPTS=0
SUMMARY_OUTPUT=""
# 输出按 source/chat-kind/backend 分目录；旧版结果不会被覆盖。
# ================================================================
if (( $# > 0 )); then RUN_DIRS=("$@"); fi
if [[ -z "$RUN_DATE" && ${#RUN_DIRS[@]} -eq 0 ]]; then
    echo '请填写 RUN_DIRS，或设置 RUN_DATE=0922 自动发现 results/checkpoints 中的存档' >&2
    exit 2
fi
cd "$PROJECT_ROOT"
humanlike_judge_args
args=("${HUMANLIKE_JUDGE_ARGS[@]}" --checkpoints-root "$CHECKPOINTS_ROOT"
      --source "$SOURCE" --chat-kind "$CHAT_KIND" --agents-dir "$AGENTS_DIR"
      --mode "$MODE" --seed "$SEED" --workers "$WORKERS"
      --attempts "$ATTEMPTS" --retry-delay "$RETRY_DELAY")
if [[ -n "$RUN_DATE" ]]; then args+=(--run-date "$RUN_DATE"); fi
if [[ -n "$OUTPUT_ROOT" ]]; then args+=(--output-root "$OUTPUT_ROOT"); fi
if [[ "$RESUME" == true ]]; then
    args+=(--resume)
    if [[ "$RETRY_ERRORS" == true ]]; then args+=(--retry-errors); fi
fi
if [[ "$DRY_RUN" == true ]]; then args+=(--dry-run); fi
if (( SHOW_PROMPTS > 0 )); then args+=(--show-prompts "$SHOW_PROMPTS"); fi
if [[ -n "$PATIENT_NAME" ]]; then args+=(--patient-name "$PATIENT_NAME"); fi
if [[ -n "$PROFILE_MAP" ]]; then args+=(--profile-map "$PROFILE_MAP"); fi
if [[ -n "$SUMMARY_OUTPUT" ]]; then args+=(--summary-output "$SUMMARY_OUTPUT"); fi
exec "$PYTHON_BIN" runshells/run_static_profile_recovery.py "${args[@]}" "${RUN_DIRS[@]}"
