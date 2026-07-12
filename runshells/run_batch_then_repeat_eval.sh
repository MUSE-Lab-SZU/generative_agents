#!/bin/bash
# ============================================================
# 批量仿真 + 重复评估量表（支持并行重复）
#
# 用法:
#   bash runshells/run_batch_then_repeat_eval.sh
#   bash runshells/run_batch_then_repeat_eval.sh --repeat-count 3
#   bash runshells/run_batch_then_repeat_eval.sh --repeat-count 6 --max-parallel 2
#
# 说明:
#   每轮先运行仿真实验；仿真成功后，再运行 archived repeat scale eval。
#   每轮的存档、报告和日志均以 -01、-02 … 后缀区分。
# ============================================================

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

# ============================================================
# ↓↓↓ 实验参数在此修改 ↓↓↓
# ============================================================

SIM_NAME="batch-0710"
SIM_CONDITION="Counsel-KBD2-G3-SEV"
SIM_MAX_PARALLEL=1
SIM_EMBEDDING_BASE_URLS="${BATCH_EMBEDDING_BASE_URLS:-http://127.0.0.1:11434}"
SIM_LOG="results/batch-0710-KBD2-G3-SEV_run.log"

EVAL_ARCHIVE_RESULTS_ROOT="results"
EVAL_CONDITION="Counsel-KBD2-G3-SEV"
EVAL_LABELS="T0,session_4,session_8,session_12,session_16,session_20,POST"
EVAL_REPEAT=3
EVAL_NAME="repeat-KBD2-G3-SEV-0710"
EVAL_MAX_PARALLEL=3
EVAL_LOG="results/repeat-KBD2-G3-SEV-0710.log"

# ============================================================
# ↑↑↑ 实验参数在此修改 ↑↑↑
# ============================================================

REPEAT_COUNT=6
MAX_PARALLEL_REPEATS=""

usage() {
  cat <<'EOF'
用法:
  bash runshells/run_batch_then_repeat_eval.sh [--repeat-count 次数] [--max-parallel 并行数]

参数:
  -n, --repeat-count N  要运行的独立实验轮数，默认 1
  -j, --max-parallel N  同时运行的实验轮数，默认等于重复次数（全部并行）
  -h, --help            显示本帮助

示例:
  bash runshells/run_batch_then_repeat_eval.sh --repeat-count 3
  bash runshells/run_batch_then_repeat_eval.sh -n 6 -j 2
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -n|--repeat-count)
      [[ $# -ge 2 ]] || { echo "错误: $1 需要一个整数。" >&2; exit 2; }
      REPEAT_COUNT="$2"
      shift 2
      ;;
    -j|--max-parallel)
      [[ $# -ge 2 ]] || { echo "错误: $1 需要一个整数。" >&2; exit 2; }
      MAX_PARALLEL_REPEATS="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "错误: 未知参数: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

[[ "$REPEAT_COUNT" =~ ^[1-9][0-9]*$ ]] || { echo "错误: --repeat-count 必须是正整数。" >&2; exit 2; }
if [[ -z "$MAX_PARALLEL_REPEATS" ]]; then
  MAX_PARALLEL_REPEATS="$REPEAT_COUNT"
fi
[[ "$MAX_PARALLEL_REPEATS" =~ ^[1-9][0-9]*$ ]] || { echo "错误: --max-parallel 必须是正整数。" >&2; exit 2; }
if (( MAX_PARALLEL_REPEATS > REPEAT_COUNT )); then
  MAX_PARALLEL_REPEATS="$REPEAT_COUNT"
fi
if [[ "$SIM_CONDITION" != "$EVAL_CONDITION" ]]; then
  echo "错误: 并行重复模式要求 SIM_CONDITION 与 EVAL_CONDITION 相同。" >&2
  exit 2
fi

mkdir -p results

suffix_path() {
  local path="$1"
  local suffix="$2"
  if [[ "$path" == *.* ]]; then
    printf '%s-%s.%s\n' "${path%.*}" "$suffix" "${path##*.}"
  else
    printf '%s-%s\n' "$path" "$suffix"
  fi
}

run_one_repeat() {
  local repeat_index="$1"
  local suffix
  suffix=$(printf '%02d' "$repeat_index")

  local run_sim_name="${SIM_NAME}-${suffix}"
  local run_eval_name="${EVAL_NAME}-${suffix}"
  local run_sim_log
  local run_eval_log
  local run_summary
  run_sim_log=$(suffix_path "$SIM_LOG" "$suffix")
  run_eval_log=$(suffix_path "$EVAL_LOG" "$suffix")
  run_summary="results/experiment_data/reports/${run_sim_name}-${SIM_CONDITION}_summary.json"

  {
    echo "=========================================="
    echo " 重复实验 ${suffix}：仿真开始 $(date '+%F %T')"
    echo " 仿真名称: $run_sim_name"
  } > "$run_sim_log"
  BATCH_EMBEDDING_BASE_URLS="$SIM_EMBEDDING_BASE_URLS" \
  python runshells/run_batch_experiment.py \
    --name "$run_sim_name" \
    --condition "$SIM_CONDITION" \
    --max-parallel "$SIM_MAX_PARALLEL" \
    >> "$run_sim_log" 2>&1 || return 1

  if [[ ! -f "$run_summary" ]]; then
    echo "错误: 重复实验 ${suffix} 未生成 summary: $run_summary" >> "$run_sim_log"
    return 1
  fi

  {
    echo "=========================================="
    echo " 重复实验 ${suffix}：复评开始 $(date '+%F %T')"
    echo " 复评名称: $run_eval_name"
    echo " 原始 summary: $run_summary"
  } > "$run_eval_log"
  python runshells/run_archived_repeat_scale_eval.py \
    --archive-results-root "$EVAL_ARCHIVE_RESULTS_ROOT" \
    --condition "$EVAL_CONDITION" \
    --original-summary "$run_summary" \
    --labels "$EVAL_LABELS" \
    --repeat "$EVAL_REPEAT" \
    --name "$run_eval_name" \
    --max-parallel "$EVAL_MAX_PARALLEL" \
    >> "$run_eval_log" 2>&1 || return 1
  echo "重复实验 ${suffix}：复评完成 $(date '+%F %T')" >> "$run_eval_log"
}

render_progress() {
  local completed="$1"
  local total="$2"
  local started_at="$3"
  local width=30
  local percent=$(( completed * 100 / total ))
  local filled=$(( completed * width / total ))
  local empty=$(( width - filled ))
  local elapsed=$(( $(date +%s) - started_at ))
  local bar spaces
  printf -v bar '%*s' "$filled" ''
  bar=${bar// /#}
  printf -v spaces '%*s' "$empty" ''
  printf '\r进度 [%-*s] %3d%% (%d/%d，已用时 %02d:%02d)' \
    "$width" "${bar}${spaces}" "$percent" "$completed" "$total" \
    $(( elapsed / 60 )) $(( elapsed % 60 ))
}

echo "=========================================="
echo " 批量仿真 + 重复评估量表（并行重复）"
echo "=========================================="
echo "  仿真条件:      $SIM_CONDITION"
echo "  评估条件:      $EVAL_CONDITION"
echo "  重复次数:      $REPEAT_COUNT"
echo "  外层并行数:    $MAX_PARALLEL_REPEATS"
echo "  命名后缀:      -01 至 -$(printf '%02d' "$REPEAT_COUNT")"
echo "  仿真日志格式:  $(suffix_path "$SIM_LOG" 'xx')"
echo "  复评日志格式:  $(suffix_path "$EVAL_LOG" 'xx')"
echo "=========================================="
echo ""

STATUS_DIR=$(mktemp -d "${TMPDIR:-/tmp}/batch-repeat-status.XXXXXX")
declare -A PID_TO_REPEAT=()
declare -a FAILED_REPEATS=()
started_at=$(date +%s)
next_repeat=1
completed=0

cleanup_children() {
  local pid
  for pid in "${!PID_TO_REPEAT[@]}"; do
    kill "$pid" 2>/dev/null || true
  done
  rm -rf "$STATUS_DIR"
}
trap cleanup_children INT TERM

start_repeat() {
  local repeat_index="$1"
  (
    set +e
    run_one_repeat "$repeat_index"
    local exit_code=$?
    printf '%s\n' "$exit_code" > "$STATUS_DIR/${repeat_index}.status"
    exit "$exit_code"
  ) &
  PID_TO_REPEAT[$!]="$repeat_index"
}

render_progress "$completed" "$REPEAT_COUNT" "$started_at"
while (( completed < REPEAT_COUNT )); do
  while (( next_repeat <= REPEAT_COUNT && ${#PID_TO_REPEAT[@]} < MAX_PARALLEL_REPEATS )); do
    start_repeat "$next_repeat"
    ((next_repeat += 1))
  done

  for pid in "${!PID_TO_REPEAT[@]}"; do
    repeat_index="${PID_TO_REPEAT[$pid]}"
    status_file="$STATUS_DIR/${repeat_index}.status"
    [[ -f "$status_file" ]] || continue

    wait "$pid" || true
    exit_code=$(<"$status_file")
    if [[ "$exit_code" != "0" ]]; then
      FAILED_REPEATS+=("$(printf '%02d' "$repeat_index")")
    fi
    unset 'PID_TO_REPEAT[$pid]'
    ((completed += 1))
    render_progress "$completed" "$REPEAT_COUNT" "$started_at"
  done
  render_progress "$completed" "$REPEAT_COUNT" "$started_at"
  (( completed < REPEAT_COUNT )) && sleep 1
done
echo ""
rm -rf "$STATUS_DIR"
trap - INT TERM

echo "=========================================="
if (( ${#FAILED_REPEATS[@]} == 0 )); then
  echo " 全部完成"
else
  echo " 已完成，但以下重复实验失败: ${FAILED_REPEATS[*]}"
  echo " 请检查对应日志，例如: $(suffix_path "$SIM_LOG" '01')"
fi
echo "=========================================="

(( ${#FAILED_REPEATS[@]} == 0 ))
