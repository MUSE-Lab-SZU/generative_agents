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
#   外层 --repeat-count 表示独立仿真轮数。每轮仿真成功后固定执行
#   EVAL_REPEAT 次完整 PHQ-9 / BDI-II 复评；启用回访时，原仿真复评与
#   无干预回访并行，回访完成后再做独立的回访复评。
#   每轮的存档、报告和日志均以 -01、-02 … 后缀区分。
#   默认村庄模式；咨询室模式加 --counsel-room（自动使用 G4 并透传模式参数）。
# ============================================================

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

# ============================================================
# ↓↓↓ 实验参数在此修改 ↓↓↓
# ============================================================

EXP_DATE="${EXP_DATE:-0718}"
GROUP="${GROUP:-G9}"
KBD="${KBD:-KBD6}"
SEVERITY="${SEVERITY:-SEV}"
COUNSEL_ROOM=false
CBT_CONTROLLER="progressive"
CBT_CONTROLLER_EXPLICIT=false
PROGRESSIVE_STAGE="D"
PROGRESSIVE_STAGE_EXPLICIT=false
BASE_CONFIG=""
OUTPUT_TAG=""
DRY_RUN=false
CLEANUP_COMPLETED_ARTIFACTS=true

SIM_NAME="batch-${EXP_DATE}"
SIM_CONDITION="Counsel-${KBD}-${GROUP}-${SEVERITY}"
SIM_TARGET_STEP=72
SIM_STRIDE=720
SIM_MAX_PARALLEL=1
SIM_EMBEDDING_BASE_URLS="${BATCH_EMBEDDING_BASE_URLS:-http://127.0.0.1:18001/v1}"
SIM_LOG="results/batch-${EXP_DATE}-${KBD}-${GROUP}-${SEVERITY}_run.log"

EVAL_ARCHIVE_RESULTS_ROOT="results"
EVAL_CONDITION="Counsel-${KBD}-${GROUP}-${SEVERITY}"
EVAL_LABELS="T0,session_4,session_8,session_12"
# 同一 agent × 时间点 × 量表的固定完整复评次数（不是独立患者样本数）
EVAL_REPEAT=10
EVAL_NAME="repeat-${KBD}-${GROUP}-${SEVERITY}-${EXP_DATE}"
EVAL_MAX_PARALLEL=6
EVAL_LOG="results/repeat-${KBD}-${GROUP}-${SEVERITY}-${EXP_DATE}.log"

# 仿真后回访默认关闭，显式传 --followup 后插入无干预观察期。
FOLLOWUP_ENABLED=false
FOLLOWUP_STEPS=120
FOLLOWUP_INTERVAL=30
FOLLOWUP_SOURCE_LABEL="session_12"
FOLLOWUP_MAX_PARALLEL=1

# ============================================================
# ↑↑↑ 实验参数在此修改 ↑↑↑
# ============================================================

REPEAT_COUNT=2
MAX_PARALLEL_REPEATS=""

usage() {
  cat <<'EOF'
用法:
  bash runshells/run_batch_then_repeat_eval.sh [参数]

参数:
  -n, --repeat-count N       要运行的独立实验轮数，默认使用脚本顶部配置
  -j, --max-parallel N       同时运行的实验轮数，默认等于重复次数（全部并行）
      --counsel-room         使用咨询室模式（固定 G4，并透传给批量实验脚本）
      --cbt-controller MODE  legacy|minimal|progressive；默认 progressive，G10/G11/G12 自动使用 legacy
      --progressive-stage D  兼容旧命令的可选参数；progressive 自动使用 D
      --config PATH          base config JSON；默认 data/config.json
      --output-tag NAME      controller identity 后的附加输出标签
      --followup             仿真后并行运行无干预回访与原仿真复评；随后复评回访节点
      --no-followup          不运行回访阶段（默认）
      --followup-steps N     回访继续运行步数，默认 120
      --followup-interval N  回访 snapshot 间隔，默认 30
      --followup-source-label LABEL
                             原仿真作为回访起点的 staged label，默认 session_12
      --followup-max-parallel N
                             同一轮内 follow-up condition 并行数，默认 1
      --keep-raw-artifacts   完整复评后仍保留 job、逐题 trace 和 experiment_data 阶段快照副本
      --dry-run              完成合并、校验、命名和 manifest 展示，不运行仿真/复评
  -h, --help                 显示本帮助

示例:
  bash runshells/run_batch_then_repeat_eval.sh --repeat-count 3
  bash runshells/run_batch_then_repeat_eval.sh -n 6 -j 2
  bash runshells/run_batch_then_repeat_eval.sh --counsel-room -n 2 -j 2
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
    --counsel-room)
      COUNSEL_ROOM=true
      shift
      ;;
    --cbt-controller)
      [[ $# -ge 2 ]] || { echo "错误: $1 需要 legacy|minimal|progressive。" >&2; exit 2; }
      CBT_CONTROLLER="$2"
      CBT_CONTROLLER_EXPLICIT=true
      shift 2
      ;;
    --progressive-stage)
      [[ $# -ge 2 ]] || { echo "错误: $1 需要 D。" >&2; exit 2; }
      PROGRESSIVE_STAGE="$2"
      PROGRESSIVE_STAGE_EXPLICIT=true
      shift 2
      ;;
    --config)
      [[ $# -ge 2 ]] || { echo "错误: $1 需要配置路径。" >&2; exit 2; }
      BASE_CONFIG="$2"
      shift 2
      ;;
    --output-tag)
      [[ $# -ge 2 ]] || { echo "错误: $1 需要标签。" >&2; exit 2; }
      OUTPUT_TAG="$2"
      shift 2
      ;;
    --followup)
      FOLLOWUP_ENABLED=true
      shift
      ;;
    --no-followup)
      FOLLOWUP_ENABLED=false
      shift
      ;;
    --followup-steps)
      [[ $# -ge 2 ]] || { echo "错误: $1 需要一个整数。" >&2; exit 2; }
      FOLLOWUP_STEPS="$2"
      shift 2
      ;;
    --followup-interval)
      [[ $# -ge 2 ]] || { echo "错误: $1 需要一个整数。" >&2; exit 2; }
      FOLLOWUP_INTERVAL="$2"
      shift 2
      ;;
    --followup-source-label)
      [[ $# -ge 2 ]] || { echo "错误: $1 需要一个 label。" >&2; exit 2; }
      FOLLOWUP_SOURCE_LABEL="$2"
      shift 2
      ;;
    --followup-max-parallel)
      [[ $# -ge 2 ]] || { echo "错误: $1 需要一个整数。" >&2; exit 2; }
      FOLLOWUP_MAX_PARALLEL="$2"
      shift 2
      ;;
    --keep-raw-artifacts)
      CLEANUP_COMPLETED_ARTIFACTS=false
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

# G10/G11/G12 不运行 CBT 强制会谈，overlay 会关闭 intervention.enabled，
# 因而只能使用 legacy controller identity；minimal/progressive 会被入口校验拒绝。
if [[ "$COUNSEL_ROOM" != true ]]; then
  case "$GROUP" in
    G10|G11|G12)
      if [[ "$CBT_CONTROLLER_EXPLICIT" == true && "$CBT_CONTROLLER" != "legacy" ]]; then
        echo "错误: $GROUP 不运行 CBT 强制会谈，只能使用 --cbt-controller legacy。" >&2
        exit 2
      fi
      CBT_CONTROLLER="legacy"
      ;;
  esac
fi

case "$CBT_CONTROLLER" in
  progressive)
    ;;
  legacy|minimal)
    if [[ "$PROGRESSIVE_STAGE_EXPLICIT" == true ]]; then
      echo "错误: $CBT_CONTROLLER controller 不能搭配 --progressive-stage。" >&2
      exit 2
    fi
    PROGRESSIVE_STAGE=""
    ;;
  *)
    echo "错误: --cbt-controller 需要 legacy|minimal|progressive。" >&2
    exit 2
    ;;
esac

if [[ "$COUNSEL_ROOM" == true ]]; then
  GROUP="G4"
  SIM_CONDITION="Counsel-${KBD}-${GROUP}-${SEVERITY}"
  EVAL_CONDITION="$SIM_CONDITION"
  SIM_LOG="results/batch-${EXP_DATE}-${KBD}-${GROUP}-${SEVERITY}_run.log"
  EVAL_NAME="repeat-${KBD}-${GROUP}-${SEVERITY}-${EXP_DATE}"
  EVAL_LOG="results/repeat-${KBD}-${GROUP}-${SEVERITY}-${EXP_DATE}.log"
fi

condition_key_cmd=(
  python runshells/cbt_experiment_config.py condition-key
  --condition "$SIM_CONDITION"
  --cbt-controller "$CBT_CONTROLLER"
)
if [[ -n "$PROGRESSIVE_STAGE" ]]; then
  condition_key_cmd+=(--progressive-stage "$PROGRESSIVE_STAGE")
fi
if [[ -n "$OUTPUT_TAG" ]]; then
  condition_key_cmd+=(--output-tag "$OUTPUT_TAG")
fi
CONDITION_KEY="$("${condition_key_cmd[@]}")"
SIM_LOG="results/batch-${EXP_DATE}-${CONDITION_KEY}_run.log"
EVAL_NAME="repeat-${CONDITION_KEY}-${EXP_DATE}"
EVAL_LOG="results/repeat-${CONDITION_KEY}-${EXP_DATE}.log"

[[ "$REPEAT_COUNT" =~ ^[1-9][0-9]*$ ]] || { echo "错误: --repeat-count 必须是正整数。" >&2; exit 2; }
[[ "$SIM_TARGET_STEP" =~ ^[1-9][0-9]*$ ]] || { echo "错误: SIM_TARGET_STEP 必须是正整数。" >&2; exit 2; }
[[ "$SIM_STRIDE" =~ ^[1-9][0-9]*$ ]] || { echo "错误: SIM_STRIDE 必须是正整数。" >&2; exit 2; }
[[ "$FOLLOWUP_STEPS" =~ ^[1-9][0-9]*$ ]] || { echo "错误: --followup-steps 必须是正整数。" >&2; exit 2; }
[[ "$FOLLOWUP_INTERVAL" =~ ^[1-9][0-9]*$ ]] || { echo "错误: --followup-interval 必须是正整数。" >&2; exit 2; }
[[ "$FOLLOWUP_MAX_PARALLEL" =~ ^[1-9][0-9]*$ ]] || { echo "错误: --followup-max-parallel 必须是正整数。" >&2; exit 2; }
[[ -n "$FOLLOWUP_SOURCE_LABEL" ]] || { echo "错误: --followup-source-label 不能为空。" >&2; exit 2; }
if [[ "$FOLLOWUP_ENABLED" == true ]] && (( FOLLOWUP_INTERVAL > FOLLOWUP_STEPS )); then
  echo "错误: 启用回访复评时 --followup-interval 不能大于 --followup-steps。" >&2
  exit 2
fi
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

if [[ "$DRY_RUN" != true ]]; then
  mkdir -p results
fi

suffix_path() {
  local path="$1"
  local suffix="$2"
  if [[ "$path" == *.* ]]; then
    printf '%s-%s.%s\n' "${path%.*}" "$suffix" "${path##*.}"
  else
    printf '%s-%s\n' "$path" "$suffix"
  fi
}

followup_labels() {
  local current="$FOLLOWUP_INTERVAL"
  local labels=""
  while (( current <= FOLLOWUP_STEPS )); do
    if [[ -n "$labels" ]]; then
      labels+=","
    fi
    labels+="followup_step_${current}"
    ((current += FOLLOWUP_INTERVAL))
  done
  printf '%s\n' "$labels"
}

run_repeat_eval() {
  local source_summary="$1"
  local labels="$2"
  local eval_name="$3"
  local eval_log="$4"

  {
    echo "=========================================="
    echo " 重复复评开始 $(date '+%F %T')"
    echo " 复评名称: $eval_name"
    echo " 评估源 summary: $source_summary"
  } > "$eval_log"
  local -a eval_cmd=(
    python runshells/run_archived_repeat_scale_eval.py
    --archive-results-root "$EVAL_ARCHIVE_RESULTS_ROOT"
    --condition "$EVAL_CONDITION"
    --original-summary "$source_summary"
    --labels "$labels"
    --repeat "$EVAL_REPEAT"
    --name "$eval_name"
    --max-parallel "$EVAL_MAX_PARALLEL"
    --require-controller-manifest
  )
  if [[ "$CLEANUP_COMPLETED_ARTIFACTS" == true ]]; then
    eval_cmd+=(--cleanup-completed-artifacts)
  fi
  "${eval_cmd[@]}" >> "$eval_log" 2>&1 || return 1
  echo "重复复评完成 $(date '+%F %T')" >> "$eval_log"
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
  local run_followup_name="followup-${CONDITION_KEY}-${EXP_DATE}-${suffix}"
  local run_followup_summary="results/experiment_data/reports/${run_followup_name}_summary.json"
  local run_followup_eval_name="${run_eval_name}-followup"
  local run_followup_eval_log
  run_sim_log=$(suffix_path "$SIM_LOG" "$suffix")
  run_eval_log=$(suffix_path "$EVAL_LOG" "$suffix")
  run_followup_eval_log=$(suffix_path "$run_eval_log" "followup")
  run_summary="results/experiment_data/reports/${run_sim_name}-${CONDITION_KEY}_summary.json"

  local -a batch_cmd=(
    python runshells/run_batch_experiment.py
    --name "$run_sim_name"
    --condition "$SIM_CONDITION"
    --step "$SIM_TARGET_STEP"
    --stride "$SIM_STRIDE"
    --max-parallel "$SIM_MAX_PARALLEL"
    --skip-post-scale
    --cbt-controller "$CBT_CONTROLLER"
  )
  if [[ -n "$PROGRESSIVE_STAGE" ]]; then
    batch_cmd+=(--progressive-stage "$PROGRESSIVE_STAGE")
  fi
  if [[ -n "$BASE_CONFIG" ]]; then
    batch_cmd+=(--config "$BASE_CONFIG")
  fi
  if [[ -n "$OUTPUT_TAG" ]]; then
    batch_cmd+=(--output-tag "$OUTPUT_TAG")
  fi
  if [[ "$DRY_RUN" == true ]]; then
    local cleanup_option=""
    if [[ "$CLEANUP_COMPLETED_ARTIFACTS" == true ]]; then
      cleanup_option=" --cleanup-completed-artifacts"
    fi
    batch_cmd+=(--dry-run)
  fi
  if [[ "$COUNSEL_ROOM" == true ]]; then
    batch_cmd+=(--counsel-room)
  fi
  if [[ "$DRY_RUN" == true ]]; then
    echo "=========================================="
    echo " 重复实验 ${suffix}：dry-run"
    echo " 仿真名称: $run_sim_name"
    BATCH_EMBEDDING_BASE_URLS="$SIM_EMBEDDING_BASE_URLS" "${batch_cmd[@]}"
    if [[ "$FOLLOWUP_ENABLED" == true ]]; then
      echo "[DRY-RUN] 原仿真复评与 follow-up 将并行启动；回访复评等待 follow-up 完成。"
      printf '  [parallel] python runshells/run_post_sim_followup.py --source-summary %q --source-label %q --name %q --steps %q --interval %q --max-parallel %q --dry-run\n' \
        "$run_summary" "$FOLLOWUP_SOURCE_LABEL" "$run_followup_name" \
        "$FOLLOWUP_STEPS" "$FOLLOWUP_INTERVAL" "$FOLLOWUP_MAX_PARALLEL"
    fi
    echo "[DRY-RUN] 原仿真 repeat eval 将读取 generation summary/manifest，但本次不执行:"
    printf '  [parallel] python runshells/run_archived_repeat_scale_eval.py --archive-results-root %q --condition %q --original-summary %q --labels %q --repeat %q --name %q --max-parallel %q --require-controller-manifest%s\n' \
      "$EVAL_ARCHIVE_RESULTS_ROOT" "$EVAL_CONDITION" "$run_summary" "$EVAL_LABELS" \
      "$EVAL_REPEAT" "$run_eval_name" "$EVAL_MAX_PARALLEL" "$cleanup_option"
    if [[ "$FOLLOWUP_ENABLED" == true ]]; then
      echo "[DRY-RUN] follow-up 完成后将执行独立回访复评:"
      printf '  python runshells/run_archived_repeat_scale_eval.py --archive-results-root %q --condition %q --original-summary %q --labels %q --repeat %q --name %q --max-parallel %q --require-controller-manifest%s\n' \
        "$EVAL_ARCHIVE_RESULTS_ROOT" "$EVAL_CONDITION" "$run_followup_summary" "$(followup_labels)" \
        "$EVAL_REPEAT" "$run_followup_eval_name" "$EVAL_MAX_PARALLEL" "$cleanup_option"
    fi
    return 0
  fi

  {
    echo "=========================================="
    echo " 重复实验 ${suffix}：仿真开始 $(date '+%F %T')"
    echo " 仿真名称: $run_sim_name"
  } > "$run_sim_log"
  BATCH_EMBEDDING_BASE_URLS="$SIM_EMBEDDING_BASE_URLS" \
    "${batch_cmd[@]}" >> "$run_sim_log" 2>&1 || return 1

  if [[ ! -f "$run_summary" ]]; then
    echo "错误: 重复实验 ${suffix} 未生成 summary: $run_summary" >> "$run_sim_log"
    return 1
  fi

  local followup_pid=""
  if [[ "$FOLLOWUP_ENABLED" == true ]]; then
    {
      echo "=========================================="
      echo " 重复实验 ${suffix}：无干预回访开始（与原仿真复评并行） $(date '+%F %T')"
      echo " 回访名称: $run_followup_name"
      echo " 源 summary: $run_summary"
    } >> "$run_sim_log"
    (
      python runshells/run_post_sim_followup.py \
        --source-summary "$run_summary" \
        --source-label "$FOLLOWUP_SOURCE_LABEL" \
        --name "$run_followup_name" \
        --steps "$FOLLOWUP_STEPS" \
        --interval "$FOLLOWUP_INTERVAL" \
        --max-parallel "$FOLLOWUP_MAX_PARALLEL" \
        >> "$run_sim_log" 2>&1
    ) &
    followup_pid=$!
  fi

  local eval_exit_code=0
  run_repeat_eval "$run_summary" "$EVAL_LABELS" "$run_eval_name" "$run_eval_log" || eval_exit_code=$?
  if [[ "$eval_exit_code" != "0" ]]; then
    if [[ -n "$followup_pid" ]]; then
      kill "$followup_pid" 2>/dev/null || true
      wait "$followup_pid" 2>/dev/null || true
    fi
    return "$eval_exit_code"
  fi

  if [[ "$FOLLOWUP_ENABLED" == true ]]; then
    local followup_exit_code=0
    wait "$followup_pid" || followup_exit_code=$?
    if [[ "$followup_exit_code" != "0" ]]; then
      echo "错误: 重复实验 ${suffix} 无干预回访失败 exit_code=${followup_exit_code}" >> "$run_sim_log"
      return "$followup_exit_code"
    fi
    if [[ ! -f "$run_followup_summary" ]]; then
      echo "错误: 重复实验 ${suffix} 未生成 follow-up summary: $run_followup_summary" >> "$run_sim_log"
      return 1
    fi
    run_repeat_eval \
      "$run_followup_summary" \
      "$(followup_labels)" \
      "$run_followup_eval_name" \
      "$run_followup_eval_log"
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
  printf '\r进度 [%-*s] %3d%% (%d/%d，已用时 %02d:%02d)' \
    "$width" "${bar}${spaces}" "$percent" "$completed" "$total" \
    $(( elapsed / 60 )) $(( elapsed % 60 ))
}

echo "=========================================="
echo " 批量仿真 + 重复评估量表（并行重复）"
echo "=========================================="
echo "  仿真条件:      $SIM_CONDITION"
echo "  评估条件:      $EVAL_CONDITION"
echo "  Condition key: $CONDITION_KEY"
echo "  CBT controller:$CBT_CONTROLLER"
echo "  Progressive:   ${PROGRESSIVE_STAGE:-'(none)'}"
echo "  Output tag:    ${OUTPUT_TAG:-'(none)'}"
echo "  Base config:   ${BASE_CONFIG:-data/config.json}"
echo "  Dry-run:       $DRY_RUN"
echo "  复评底稿清理:  $CLEANUP_COMPLETED_ARTIFACTS"
echo "  回访阶段:      $FOLLOWUP_ENABLED"
if [[ "$FOLLOWUP_ENABLED" == true ]]; then
  echo "  回访步数/间隔: ${FOLLOWUP_STEPS}/${FOLLOWUP_INTERVAL}"
  echo "  回访源节点:    $FOLLOWUP_SOURCE_LABEL"
  echo "  回访并行数:    $FOLLOWUP_MAX_PARALLEL"
fi
echo "  模式:          $([[ "$COUNSEL_ROOM" == true ]] && echo 咨询室 || echo 村庄)"
echo "  目标总步数:    $SIM_TARGET_STEP"
echo "  每步分钟数:    $SIM_STRIDE"
echo "  重复次数:      $REPEAT_COUNT"
echo "  外层并行数:    $MAX_PARALLEL_REPEATS"
echo "  命名后缀:      -01 至 -$(printf '%02d' "$REPEAT_COUNT")"
echo "  仿真日志格式:  $(suffix_path "$SIM_LOG" 'xx')"
echo "  复评日志格式:  $(suffix_path "$EVAL_LOG" 'xx')"
echo "=========================================="
echo ""

if [[ "$DRY_RUN" == true ]]; then
  dry_index=1
  while (( dry_index <= REPEAT_COUNT )); do
    run_one_repeat "$dry_index"
    ((dry_index += 1))
  done
  echo "=========================================="
  echo " Dry-run 完成；未运行 start.py，未执行 repeat eval。"
  echo "=========================================="
  exit 0
fi

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
