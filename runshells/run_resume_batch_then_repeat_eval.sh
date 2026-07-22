#!/bin/bash
# ============================================================
# 稀疏 checkpoint 安全恢复 + 补完仿真 + 重复量表复评
#
# 运行前请先启动并确认 Qwen/BGE 服务；本脚本只检查，不自动启动服务。
# 每个重复通过 batch_state 精确锁定原 run_name，并继续使用原 summary/复评名称。
# 默认村庄模式；咨询室模式加 --counsel-room（自动使用 G4 并透传模式参数）。
# ============================================================

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

# ============================================================
# ↓↓↓ 恢复实验参数在此修改 ↓↓↓
# ============================================================

EXP_DATE="${EXP_DATE:-0718}"
GROUP="${GROUP:-G9}"
KBD="${KBD:-KBD6}"
SEVERITY="${SEVERITY:-SEV}"
COUNSEL_ROOM=false

SIM_NAME="batch-${EXP_DATE}"
SIM_CONDITION="Counsel-${KBD}-${GROUP}-${SEVERITY}"
SIM_TARGET_STEP=120
SIM_STRIDE=720
SIM_MAX_PARALLEL=1
SIM_EMBEDDING_BASE_URLS="${BATCH_EMBEDDING_BASE_URLS:-http://127.0.0.1:18001/v1}"
SIM_LOG="results/resume-batch-${EXP_DATE}-${KBD}-${GROUP}-${SEVERITY}_run.log"
SIM_CHECKPOINT_LOG="run_batch_experiment-resume.log"

EVAL_ARCHIVE_RESULTS_ROOT="results"
EVAL_CONDITION="Counsel-${KBD}-${GROUP}-${SEVERITY}"
EVAL_LABELS="T0,session_4,session_8,session_12,session_16,session_20,POST"
EVAL_REPEAT=10
EVAL_NAME="repeat-${KBD}-${GROUP}-${SEVERITY}-${EXP_DATE}"
EVAL_MAX_PARALLEL=3
EVAL_LOG="results/resume-repeat-${KBD}-${GROUP}-${SEVERITY}-${EXP_DATE}.log"

REPEAT_COUNT=2
# 留空时处理 01 至 REPEAT_COUNT；设置为正整数时只处理该重复编号。
REPEAT_INDEX=""
MAX_PARALLEL_REPEATS=""

# ============================================================
# ↑↑↑ 恢复实验参数在此修改 ↑↑↑
# ============================================================

DRY_RUN=false

usage() {
  cat <<'EOF'
用法:
  bash runshells/run_resume_batch_then_repeat_eval.sh [参数]

参数:
  -n, --repeat-count N  恢复 batch-日期-01 至 batch-日期-N，默认使用脚本顶部配置
  -r, --repeat-index N  只恢复 batch-日期-N；设置后不遍历 repeat-count 范围
  -j, --max-parallel N  同时恢复的独立实验数，默认等于本次选中的重复数
      --counsel-room    使用咨询室模式（固定 G4，并透传给批量实验脚本）
      --dry-run         完整校验并显示恢复锚点/命令，不移动文件、不启动任务
  -h, --help            显示帮助

示例:
  bash runshells/run_resume_batch_then_repeat_eval.sh --dry-run
  bash runshells/run_resume_batch_then_repeat_eval.sh -n 2 -j 1
  bash runshells/run_resume_batch_then_repeat_eval.sh --repeat-index 2 --dry-run
  bash runshells/run_resume_batch_then_repeat_eval.sh --repeat-index 2
  bash runshells/run_resume_batch_then_repeat_eval.sh --counsel-room -n 2 -j 2
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -n|--repeat-count)
      [[ $# -ge 2 ]] || { echo "错误: $1 需要一个整数。" >&2; exit 2; }
      REPEAT_COUNT="$2"
      shift 2
      ;;
    -r|--repeat-index)
      [[ $# -ge 2 ]] || { echo "错误: $1 需要一个整数。" >&2; exit 2; }
      REPEAT_INDEX="$2"
      shift 2
      ;;
    -j|--max-parallel)
      [[ $# -ge 2 ]] || { echo "错误: $1 需要一个整数。" >&2; exit 2; }
      MAX_PARALLEL_REPEATS="$2"
      shift 2
      ;;
    --counsel-room)
      COUNSEL_ROOM=true
      shift
      ;;
    --dry-run)
      DRY_RUN=true
      shift
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

if [[ "$COUNSEL_ROOM" == true ]]; then
  GROUP="G4"
  SIM_CONDITION="Counsel-${KBD}-${GROUP}-${SEVERITY}"
  EVAL_CONDITION="$SIM_CONDITION"
  SIM_LOG="results/resume-batch-${EXP_DATE}-${KBD}-${GROUP}-${SEVERITY}_run.log"
  EVAL_NAME="repeat-${KBD}-${GROUP}-${SEVERITY}-${EXP_DATE}"
  EVAL_LOG="results/resume-repeat-${KBD}-${GROUP}-${SEVERITY}-${EXP_DATE}.log"
fi

[[ "$REPEAT_COUNT" =~ ^[1-9][0-9]*$ ]] || { echo "错误: --repeat-count 必须是正整数。" >&2; exit 2; }
if [[ -n "$REPEAT_INDEX" ]]; then
  [[ "$REPEAT_INDEX" =~ ^[1-9][0-9]*$ ]] || { echo "错误: --repeat-index 必须是正整数。" >&2; exit 2; }
  SELECTED_REPEAT_START="$REPEAT_INDEX"
  SELECTED_REPEAT_END="$REPEAT_INDEX"
  SELECTED_REPEAT_COUNT=1
else
  SELECTED_REPEAT_START=1
  SELECTED_REPEAT_END="$REPEAT_COUNT"
  SELECTED_REPEAT_COUNT="$REPEAT_COUNT"
fi
[[ "$SIM_TARGET_STEP" =~ ^[1-9][0-9]*$ ]] || { echo "错误: SIM_TARGET_STEP 必须是正整数。" >&2; exit 2; }
if [[ -z "$MAX_PARALLEL_REPEATS" ]]; then
  MAX_PARALLEL_REPEATS="$SELECTED_REPEAT_COUNT"
fi
[[ "$MAX_PARALLEL_REPEATS" =~ ^[1-9][0-9]*$ ]] || { echo "错误: --max-parallel 必须是正整数。" >&2; exit 2; }
if (( MAX_PARALLEL_REPEATS > SELECTED_REPEAT_COUNT )); then
  MAX_PARALLEL_REPEATS="$SELECTED_REPEAT_COUNT"
fi
if [[ "$SIM_CONDITION" != "$EVAL_CONDITION" ]]; then
  echo "错误: SIM_CONDITION 与 EVAL_CONDITION 必须相同。" >&2
  exit 2
fi
command -v jq >/dev/null 2>&1 || { echo "错误: 恢复脚本需要 jq。" >&2; exit 2; }

suffix_path() {
  local path="$1"
  local suffix="$2"
  if [[ "$path" == *.* ]]; then
    printf '%s-%s.%s\n' "${path%.*}" "$suffix" "${path##*.}"
  else
    printf '%s-%s\n' "$path" "$suffix"
  fi
}

print_command() {
  printf '[DRY-RUN]'
  printf ' %q' "$@"
  printf '\n'
}

repeat_report_complete() {
  local report_path="$1"
  local repeat_name="$2"
  [[ -f "$report_path" ]] || return 1
  jq -e \
    --arg batch_name "$repeat_name" \
    --arg condition "$EVAL_CONDITION" \
    --argjson repeat "$EVAL_REPEAT" \
    '.batch_name == $batch_name
      and .repeat == $repeat
      and .completion.ready_for_final_report == true
      and any(.conditions[]?; .condition_name == $condition)' \
    "$report_path" >/dev/null
}

run_one_repeat() {
  local repeat_index="$1"
  local suffix batch_name state_path run_name state_status
  suffix=$(printf '%02d' "$repeat_index")
  batch_name="${SIM_NAME}-${suffix}"
  state_path="results/experiment_data/batch_state/${batch_name}/${SIM_CONDITION}.json"

  if [[ ! -f "$state_path" ]]; then
    echo "错误: 未找到精确 batch state: $state_path" >&2
    return 1
  fi
  if ! jq -e --arg batch "$batch_name" --arg condition "$SIM_CONDITION" \
    '.batch_name == $batch and .condition_name == $condition and (.run_name | type == "string" and length > 0)' \
    "$state_path" >/dev/null; then
    echo "错误: batch state 内容与选择条件不匹配: $state_path" >&2
    return 1
  fi
  run_name=$(jq -r '.run_name' "$state_path")
  state_status=$(jq -r '.status // ""' "$state_path")

  local run_summary run_eval_name run_eval_summary run_sim_log run_eval_log
  run_summary="results/experiment_data/reports/${batch_name}-${SIM_CONDITION}_summary.json"
  run_eval_name="${EVAL_NAME}-${suffix}"
  run_eval_summary="results/experiment_data/reports/${run_eval_name}_summary.json"
  run_sim_log=$(suffix_path "$SIM_LOG" "$suffix")
  run_eval_log=$(suffix_path "$EVAL_LOG" "$suffix")

  echo "=========================================="
  echo " 重复实验 ${suffix}"
  echo " batch:     ${batch_name}"
  echo " condition: ${SIM_CONDITION}"
  echo " run_name:  ${run_name}"
  echo " status:    ${state_status}"
  echo "=========================================="

  if [[ "$state_status" == "completed" ]]; then
    if [[ ! -f "$run_summary" ]]; then
      echo "错误: batch state 已完成但 summary 缺失: $run_summary" >&2
      return 1
    fi
    echo "[SKIP] 仿真与后处理已完成: $run_name"
  else
    case "$state_status" in
      simulation_done|postprocessing|postprocessing_interrupted)
        echo "[INFO] 仿真已完成或正在后处理，只续跑 batch 后处理。"
        ;;
      *)
        local -a recovery_cmd=(
          python runshells/prepare_sparse_checkpoint_resume.py
          --batch-name "$batch_name"
          --condition "$SIM_CONDITION"
          --target-step "$SIM_TARGET_STEP"
        )
        if [[ "$DRY_RUN" == true ]]; then
          recovery_cmd+=(--dry-run)
        fi
        "${recovery_cmd[@]}"
        ;;
    esac

    local -a batch_cmd=(
      python runshells/run_batch_experiment.py
      --name "$batch_name"
      --condition "$SIM_CONDITION"
      --resume-condition "$SIM_CONDITION"
      --step "$SIM_TARGET_STEP"
      --stride "$SIM_STRIDE"
      --max-parallel "$SIM_MAX_PARALLEL"
      --log "$SIM_CHECKPOINT_LOG"
    )
    if [[ "$COUNSEL_ROOM" == true ]]; then
      batch_cmd+=(--counsel-room)
    fi
    if [[ "$DRY_RUN" == true ]]; then
      batch_cmd+=(--dry-run)
      print_command env "BATCH_EMBEDDING_BASE_URLS=$SIM_EMBEDDING_BASE_URLS" "${batch_cmd[@]}"
    else
      {
        echo "=========================================="
        echo " 恢复仿真开始 $(date '+%F %T')"
        echo " run_name: $run_name"
      } >> "$run_sim_log"
      BATCH_EMBEDDING_BASE_URLS="$SIM_EMBEDDING_BASE_URLS" \
        "${batch_cmd[@]}" >> "$run_sim_log" 2>&1
      if [[ ! -f "$run_summary" ]]; then
        echo "错误: 恢复后未生成 summary: $run_summary" | tee -a "$run_sim_log" >&2
        return 1
      fi
      echo "恢复仿真与后处理完成 $(date '+%F %T')" >> "$run_sim_log"
    fi
  fi

  if repeat_report_complete "$run_eval_summary" "$run_eval_name"; then
    echo "[SKIP] 重复复评已完整: $run_eval_summary"
    return 0
  fi

  local -a eval_cmd=(
    python runshells/run_archived_repeat_scale_eval.py
    --archive-results-root "$EVAL_ARCHIVE_RESULTS_ROOT"
    --condition "$EVAL_CONDITION"
    --original-summary "$run_summary"
    --labels "$EVAL_LABELS"
    --repeat "$EVAL_REPEAT"
    --name "$run_eval_name"
    --max-parallel "$EVAL_MAX_PARALLEL"
    --resume-partial
  )
  if [[ "$DRY_RUN" == true ]]; then
    print_command "${eval_cmd[@]}"
    return 0
  fi

  {
    echo "=========================================="
    echo " 重复复评开始 $(date '+%F %T')"
    echo " 复评名称: $run_eval_name"
    echo " 原始 summary: $run_summary"
  } >> "$run_eval_log"
  local eval_exit_code
  if "${eval_cmd[@]}" >> "$run_eval_log" 2>&1; then
    echo "重复复评完成 $(date '+%F %T')" >> "$run_eval_log"
  else
    eval_exit_code=$?
    echo "错误: 重复复评失败 exit_code=${eval_exit_code} $(date '+%F %T')" \
      | tee -a "$run_eval_log" >&2
    return "$eval_exit_code"
  fi
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
  printf '\r恢复进度 [%-*s] %3d%% (%d/%d，已用时 %02d:%02d)' \
    "$width" "${bar}${spaces}" "$percent" "$completed" "$total" \
    $(( elapsed / 60 )) $(( elapsed % 60 ))
}

echo "=========================================="
echo " 稀疏 checkpoint 安全恢复流水线"
echo "=========================================="
echo " 条件:          $SIM_CONDITION"
echo " 模式:          $([[ "$COUNSEL_ROOM" == true ]] && echo 咨询室 || echo 村庄)"
if [[ -n "$REPEAT_INDEX" ]]; then
  echo " 重复存档:      ${SIM_NAME}-$(printf '%02d' "$REPEAT_INDEX")（仅此编号）"
else
  echo " 重复存档:      ${SIM_NAME}-01 至 ${SIM_NAME}-$(printf '%02d' "$REPEAT_COUNT")"
fi
echo " 目标总步数:    $SIM_TARGET_STEP"
echo " 外层并行数:    $MAX_PARALLEL_REPEATS"
echo " 量表复评次数:  $EVAL_REPEAT"
echo " dry-run:       $DRY_RUN"
echo "=========================================="

STATUS_DIR=$(mktemp -d "${TMPDIR:-/tmp}/resume-batch-repeat-status.XXXXXX")
declare -A PID_TO_REPEAT=()
declare -a FAILED_REPEATS=()
started_at=$(date +%s)
next_repeat="$SELECTED_REPEAT_START"
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

render_progress "$completed" "$SELECTED_REPEAT_COUNT" "$started_at"
while (( completed < SELECTED_REPEAT_COUNT )); do
  while (( next_repeat <= SELECTED_REPEAT_END && ${#PID_TO_REPEAT[@]} < MAX_PARALLEL_REPEATS )); do
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
    render_progress "$completed" "$SELECTED_REPEAT_COUNT" "$started_at"
  done
  (( completed < SELECTED_REPEAT_COUNT )) && sleep 1
done
echo ""
rm -rf "$STATUS_DIR"
trap - INT TERM

echo "=========================================="
if (( ${#FAILED_REPEATS[@]} == 0 )); then
  echo " 恢复流水线全部完成"
  exit 0
fi
echo " 以下重复实验恢复失败: ${FAILED_REPEATS[*]}"
echo " 请检查: $(suffix_path "$SIM_LOG" 'xx') / $(suffix_path "$EVAL_LOG" 'xx')"
echo "=========================================="
exit 1
