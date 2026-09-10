#!/bin/bash
# ============================================================
# 稀疏 checkpoint 安全恢复 + 补完仿真 + 并行复评/回访
#
# 运行前请先启动并确认 Qwen/BGE 服务；本脚本只检查，不自动启动服务。
# 每个重复通过 condition key + batch_state 精确锁定原 run_name，并继续使用原
# summary/复评名称；尚未启动的重复会以同一实验身份补跑。
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
KBD="${KBD^^}"
SEVERITY="${SEVERITY^^}"
COUNSEL_ROOM=false
CBT_CONTROLLER="progressive"
CBT_CONTROLLER_EXPLICIT=false
PROGRESSIVE_STAGE="D"
PROGRESSIVE_STAGE_EXPLICIT=false
BASE_CONFIG=""
OUTPUT_TAG=""
# --resume 默认使用 checkpoint 内保存的配置。显式开启后，仅将本文件中的
# LLM/BGE endpoint 路由同步到恢复锚点，便于中断后切换轮询端口。
REFRESH_MODEL_ROUTING=false
ROUTING_CONFIG="data/config.json"
ROUTING_CONFIG_EXPLICIT=false
CLEANUP_COMPLETED_ARTIFACTS=true

SIM_NAME="batch-${EXP_DATE}"
SIM_CONDITION="Counsel-${KBD}-${GROUP}-${SEVERITY}"
SIM_TARGET_STEP=96
SIM_STRIDE=720
SIM_MAX_PARALLEL=1
SIM_EMBEDDING_BASE_URLS="${BATCH_EMBEDDING_BASE_URLS:-http://127.0.0.1:18001/v1}"
SIM_LOG="results/resume-batch-${EXP_DATE}-${KBD}-${GROUP}-${SEVERITY}_run.log"
SIM_CHECKPOINT_LOG="run_batch_experiment-resume.log"

EVAL_ARCHIVE_RESULTS_ROOT="results"
EVAL_CONDITION="Counsel-${KBD}-${GROUP}-${SEVERITY}"
EVAL_LABELS="auto"
# T0 / 实际末次 PHQ-9、BDI-II 的固定完整复评次数（不是独立患者样本数）
EVAL_REPEAT=10
# 中间两个短量表的固定完整复评次数
EVAL_INTERMEDIATE_SCALE_REPEATS=5
EVAL_NAME="repeat-${KBD}-${GROUP}-${SEVERITY}-${EXP_DATE}"
EVAL_MAX_PARALLEL=6
EVAL_LOG="results/resume-repeat-${KBD}-${GROUP}-${SEVERITY}-${EXP_DATE}.log"

# 与主跑脚本一致：默认不运行回访；显式 --followup 时恢复仿真后继续回访。
FOLLOWUP_ENABLED=false
FOLLOWUP_STEPS=120
FOLLOWUP_INTERVAL=30
FOLLOWUP_SOURCE_LABEL="session_16"
FOLLOWUP_MAX_PARALLEL=1
# 回访曾被中断时，严格校验已冻结的 followup_step 节点后从最后一个完整节点续跑。
FOLLOWUP_RESUME_PARTIAL=true

REPEAT_COUNT=2
# 留空时处理 01 至 REPEAT_COUNT；设置为正整数时只处理该重复编号。
REPEAT_INDEX=""
MAX_PARALLEL_REPEATS=""

# 开启后先检查 forced_llm 持续故障，再决定恢复锚点或仅重跑坏复评。
AUTO_DETECT_FORCED_LLM_ROLLBACK="${AUTO_DETECT_FORCED_LLM_ROLLBACK:-false}"
FORCED_LLM_MIN_CALLS="${FORCED_LLM_MIN_CALLS:-6}"
FORCED_LLM_FAILURE_RATIO="${FORCED_LLM_FAILURE_RATIO:-0.5}"
FORCED_LLM_CONSECUTIVE_SIM_MEETINGS="${FORCED_LLM_CONSECUTIVE_SIM_MEETINGS:-2}"

# ============================================================
# ↑↑↑ 恢复实验参数在此修改 ↑↑↑
# ============================================================

DRY_RUN=false

usage() {
  cat <<'EOF'
用法:
  bash runshells/run_resume_batch_then_repeat_eval.sh [参数]

参数:
  -n, --repeat-count N       恢复 batch-日期-01 至 batch-日期-N，默认使用脚本顶部配置
  -r, --repeat-index N       只恢复 batch-日期-N；设置后不遍历 repeat-count 范围
  -j, --max-parallel N       同时恢复的独立实验数，默认等于本次选中的重复数
      --counsel-room         使用咨询室模式（固定 G4，并透传给批量实验脚本）
      --cbt-controller MODE  legacy|minimal|progressive；必须与原实验一致
      --progressive-stage D  兼容旧命令的可选参数；progressive 自动使用 D
      --config PATH          base config JSON；必须与原实验一致
      --output-tag NAME      controller identity 后的附加输出标签；必须与原实验一致
      --refresh-model-routing
                             使用 routing config 覆盖恢复快照中的 LLM/BGE base_url
                             与 load_balancing；默认关闭，其他运行配置保持不变
      --routing-config PATH  --refresh-model-routing 的路由来源；默认跟随 --config，
                             未指定 --config 时使用 data/config.json
      --followup             仿真恢复后并行运行无干预回访与原仿真复评；随后复评回访节点
      --no-followup          不运行回访阶段（默认）
      --followup-steps N     回访继续运行步数，默认 120
      --followup-interval N  回访 snapshot 间隔，默认 30
      --followup-source-label LABEL
                             原仿真作为回访起点的 staged label，默认 session_16
      --followup-max-parallel N
                             同一轮内 follow-up condition 并行数，默认 1
      --followup-resume-partial
                             严格校验中断回访并从最后一个完整 followup_step 节点续跑（默认）
      --no-followup-resume-partial
                             禁用回访 partial 续跑；遇到不完整同名回访时停止
      --auto-detect-forced-llm-rollback
                             检测 sustained forced_llm 失败并限制恢复锚点
      --forced-llm-min-calls N
                             单个检测单元最少相关调用数，默认 6
      --forced-llm-failure-ratio R
                             大量失败比例阈值 0..1，默认 0.5
      --forced-llm-consecutive-meetings N
                             仿真连续异常会谈数，默认 2
      --keep-raw-artifacts   完整复评后仍保留 job、逐题 trace 和 experiment_data 阶段快照副本
      --dry-run              完整校验并显示恢复锚点/命令，不移动文件、不启动任务
  -h, --help                 显示帮助

示例:
  bash runshells/run_resume_batch_then_repeat_eval.sh --dry-run
  bash runshells/run_resume_batch_then_repeat_eval.sh -n 2 -j 1
  bash runshells/run_resume_batch_then_repeat_eval.sh --repeat-index 2 --dry-run
  bash runshells/run_resume_batch_then_repeat_eval.sh --repeat-index 2
  bash runshells/run_resume_batch_then_repeat_eval.sh --counsel-room -n 2 -j 2
  bash runshells/run_resume_batch_then_repeat_eval.sh --cbt-controller progressive -n 2
  bash runshells/run_resume_batch_then_repeat_eval.sh --refresh-model-routing --dry-run
  bash runshells/run_resume_batch_then_repeat_eval.sh --refresh-model-routing
  bash runshells/run_resume_batch_then_repeat_eval.sh --auto-detect-forced-llm-rollback --dry-run
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
    --refresh-model-routing)
      REFRESH_MODEL_ROUTING=true
      shift
      ;;
    --routing-config)
      [[ $# -ge 2 ]] || { echo "错误: $1 需要配置路径。" >&2; exit 2; }
      ROUTING_CONFIG="$2"
      ROUTING_CONFIG_EXPLICIT=true
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
    --followup-resume-partial)
      FOLLOWUP_RESUME_PARTIAL=true
      shift
      ;;
    --no-followup-resume-partial)
      FOLLOWUP_RESUME_PARTIAL=false
      shift
      ;;
    --auto-detect-forced-llm-rollback)
      AUTO_DETECT_FORCED_LLM_ROLLBACK=true
      shift
      ;;
    --forced-llm-min-calls)
      [[ $# -ge 2 ]] || { echo "错误: $1 需要一个正整数。" >&2; exit 2; }
      FORCED_LLM_MIN_CALLS="$2"
      shift 2
      ;;
    --forced-llm-failure-ratio)
      [[ $# -ge 2 ]] || { echo "错误: $1 需要 0 到 1 的数字。" >&2; exit 2; }
      FORCED_LLM_FAILURE_RATIO="$2"
      shift 2
      ;;
    --forced-llm-consecutive-meetings)
      [[ $# -ge 2 ]] || { echo "错误: $1 需要一个正整数。" >&2; exit 2; }
      FORCED_LLM_CONSECUTIVE_SIM_MEETINGS="$2"
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

if [[ "$ROUTING_CONFIG_EXPLICIT" != true && -n "$BASE_CONFIG" ]]; then
  ROUTING_CONFIG="$BASE_CONFIG"
fi

case "$KBD" in
  LRN|GC|CY|TW|ZYH|SQL|ZMY|XFH)
    if [[ "$SEVERITY" != "MOD" ]]; then
      echo "错误: 新人设 $KBD 仅提供中度配置，请设置 SEVERITY=MOD。" >&2
      exit 2
    fi
    ;;
esac

# 与主跑脚本保持一致：G10/G11/G12 不运行 CBT 强制会谈，只使用 legacy identity。
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
  SIM_LOG="results/resume-batch-${EXP_DATE}-${KBD}-${GROUP}-${SEVERITY}_run.log"
  EVAL_NAME="repeat-${KBD}-${GROUP}-${SEVERITY}-${EXP_DATE}"
  EVAL_LOG="results/resume-repeat-${KBD}-${GROUP}-${SEVERITY}-${EXP_DATE}.log"
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
SIM_LOG="results/resume-batch-${EXP_DATE}-${CONDITION_KEY}_run.log"
EVAL_NAME="repeat-${CONDITION_KEY}-${EXP_DATE}"
EVAL_LOG="results/resume-repeat-${CONDITION_KEY}-${EXP_DATE}.log"

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
[[ "$SIM_STRIDE" =~ ^[1-9][0-9]*$ ]] || { echo "错误: SIM_STRIDE 必须是正整数。" >&2; exit 2; }
[[ "$FOLLOWUP_STEPS" =~ ^[1-9][0-9]*$ ]] || { echo "错误: --followup-steps 必须是正整数。" >&2; exit 2; }
[[ "$FOLLOWUP_INTERVAL" =~ ^[1-9][0-9]*$ ]] || { echo "错误: --followup-interval 必须是正整数。" >&2; exit 2; }
[[ "$FOLLOWUP_MAX_PARALLEL" =~ ^[1-9][0-9]*$ ]] || { echo "错误: --followup-max-parallel 必须是正整数。" >&2; exit 2; }
[[ -n "$FOLLOWUP_SOURCE_LABEL" ]] || { echo "错误: --followup-source-label 不能为空。" >&2; exit 2; }
if [[ "$FOLLOWUP_ENABLED" == true ]] && (( FOLLOWUP_INTERVAL > FOLLOWUP_STEPS )); then
  echo "错误: 启用回访复评时 --followup-interval 不能大于 --followup-steps。" >&2
  exit 2
fi
[[ "$FORCED_LLM_MIN_CALLS" =~ ^[1-9][0-9]*$ ]] || { echo "错误: FORCED_LLM_MIN_CALLS 必须是正整数。" >&2; exit 2; }
[[ "$FORCED_LLM_CONSECUTIVE_SIM_MEETINGS" =~ ^[1-9][0-9]*$ ]] || { echo "错误: FORCED_LLM_CONSECUTIVE_SIM_MEETINGS 必须是正整数。" >&2; exit 2; }
command -v jq >/dev/null 2>&1 || { echo "错误: 恢复脚本需要 jq。" >&2; exit 2; }
if ! jq -en --arg value "$FORCED_LLM_FAILURE_RATIO" \
  '($value | tonumber) as $number | $number >= 0 and $number <= 1' >/dev/null; then
  echo "错误: FORCED_LLM_FAILURE_RATIO 必须是 0 到 1 的数字。" >&2
  exit 2
fi
case "${AUTO_DETECT_FORCED_LLM_ROLLBACK,,}" in
  1|true|yes|on) AUTO_DETECT_FORCED_LLM_ROLLBACK=true ;;
  0|false|no|off) AUTO_DETECT_FORCED_LLM_ROLLBACK=false ;;
  *) echo "错误: AUTO_DETECT_FORCED_LLM_ROLLBACK 必须是 true/false。" >&2; exit 2 ;;
esac
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
if [[ "$REFRESH_MODEL_ROUTING" == true && ! -f "$ROUTING_CONFIG" ]]; then
  echo "错误: routing config 不存在: $ROUTING_CONFIG" >&2
  exit 2
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

print_command() {
  printf '[DRY-RUN]'
  printf ' %q' "$@"
  printf '\n'
}

repeat_report_complete() {
  local report_path="$1"
  local repeat_name="$2"
  local source_summary="$3"
  local requested_labels="$4"
  [[ -f "$report_path" ]] || return 1
  jq -e \
    --arg batch_name "$repeat_name" \
    --arg condition "$EVAL_CONDITION" \
    --arg source_summary "$source_summary" \
    --arg requested_labels "$requested_labels" \
    --argjson repeat "$EVAL_REPEAT" \
    --argjson intermediate_repeat "$EVAL_INTERMEDIATE_SCALE_REPEATS" \
    '.batch_name == $batch_name
      and .repeat == $repeat
      and ((.evaluation_protocol.intermediate_scale_repeats // -1) == $intermediate_repeat)
      and ((.source_original_summary // "") | endswith($source_summary))
      and ((.labels // []) == ($requested_labels | split(",")))
      and .completion.ready_for_final_report == true
      and any(.conditions[]?; .condition_name == $condition)' \
    "$report_path" >/dev/null
}

run_or_resume_repeat_eval() {
  local source_summary="$1"
  local requested_labels="$2"
  local eval_name="$3"
  local eval_summary="$4"
  local eval_log="$5"

  if repeat_report_complete \
    "$eval_summary" \
    "$eval_name" \
    "$source_summary" \
    "$requested_labels"; then
    echo "[SKIP] 重复复评已完整: $eval_summary"
    return 0
  fi

  local -a eval_cmd=(
    python runshells/run_archived_repeat_scale_eval.py
    --archive-results-root "$EVAL_ARCHIVE_RESULTS_ROOT"
    --condition "$EVAL_CONDITION"
    --original-summary "$source_summary"
    --labels "$requested_labels"
    --repeat "$EVAL_REPEAT"
    --intermediate-scale-repeats "$EVAL_INTERMEDIATE_SCALE_REPEATS"
    --name "$eval_name"
    --max-parallel "$EVAL_MAX_PARALLEL"
    --resume-partial
    --require-controller-manifest
  )
  if [[ "$CLEANUP_COMPLETED_ARTIFACTS" == true ]]; then
    eval_cmd+=(--cleanup-completed-artifacts)
  fi
  if [[ "$DRY_RUN" == true ]]; then
    print_command "${eval_cmd[@]}"
    return 0
  fi

  {
    echo "=========================================="
    echo " 重复复评开始 $(date '+%F %T')"
    echo " 复评名称: $eval_name"
    echo " 评估源 summary: $source_summary"
  } >> "$eval_log"
  local eval_exit_code=0
  "${eval_cmd[@]}" >> "$eval_log" 2>&1 || eval_exit_code=$?
  if [[ "$eval_exit_code" == "0" ]]; then
    echo "重复复评完成 $(date '+%F %T')" >> "$eval_log"
    return 0
  fi
  echo "错误: 重复复评失败 exit_code=${eval_exit_code} $(date '+%F %T')" \
    | tee -a "$eval_log" >&2
  return "$eval_exit_code"
}

refresh_model_routing() {
  local run_name="$1"
  local -a routing_cmd=(
    python runshells/refresh_resume_model_routing.py
    --run-name "$run_name"
    --routing-config "$ROUTING_CONFIG"
  )
  if [[ "$DRY_RUN" == true ]]; then
    routing_cmd+=(--dry-run)
  fi
  echo "[ROUTING] 刷新恢复快照的模型 endpoint 路由: $run_name"
  "${routing_cmd[@]}"
}

quarantine_related_paths() {
  local run_name="$1"
  local reason="$2"
  shift 2
  (( $# > 0 )) || return 0
  local -a quarantine_cmd=(
    python runshells/prepare_sparse_checkpoint_resume.py
    --run-name "$run_name"
    --quarantine-only
    --quarantine-reason "$reason"
  )
  local path
  for path in "$@"; do
    quarantine_cmd+=(--quarantine-path "$path")
  done
  if [[ "$DRY_RUN" == true ]]; then
    quarantine_cmd+=(--dry-run)
    print_command "${quarantine_cmd[@]}"
  fi
  "${quarantine_cmd[@]}"
}

run_one_repeat() {
  local repeat_index="$1"
  local suffix batch_name state_path run_name state_status
  local recovery_exit_code batch_exit_code
  suffix=$(printf '%02d' "$repeat_index")
  batch_name="${SIM_NAME}-${suffix}"
  state_path="results/experiment_data/batch_state/${batch_name}/${CONDITION_KEY}.json"
  run_name=""
  state_status="not_started"

  if [[ -f "$state_path" ]]; then
    if ! jq -e \
      --arg batch "$batch_name" \
      --arg condition "$SIM_CONDITION" \
      --arg condition_key "$CONDITION_KEY" \
      --arg controller "$CBT_CONTROLLER" \
      --arg progressive_stage "$PROGRESSIVE_STAGE" \
      --arg output_tag "$OUTPUT_TAG" \
      '.batch_name == $batch
        and .condition_name == $condition
        and .condition_key == $condition_key
        and .cbt_controller == $controller
        and ((.progressive_stage // "") == $progressive_stage)
        and ((.output_tag // "") == $output_tag)
        and (.run_name | type == "string" and length > 0)
        and (
          .run_name == ($batch + "-" + $condition_key)
          or (.run_name | startswith($batch + "-" + $condition_key + "-"))
        )' \
      "$state_path" >/dev/null; then
      echo "错误: batch state 与本次 controller/config identity 不匹配: $state_path" >&2
      return 1
    fi
    run_name=$(jq -r '.run_name' "$state_path")
    state_status=$(jq -r '.status // ""' "$state_path")
  else
    echo "[INFO] 重复实验 ${suffix} 尚未创建 batch state，将以相同实验身份从头启动。"
  fi

  local run_summary run_eval_name run_eval_summary run_sim_log run_eval_log
  local run_followup_name run_followup_checkpoint_name run_followup_summary run_followup_eval_name
  local run_followup_eval_summary run_followup_eval_log
  run_summary="results/experiment_data/reports/${batch_name}-${CONDITION_KEY}_summary.json"
  run_eval_name="${EVAL_NAME}-${suffix}"
  run_eval_summary="results/experiment_data/reports/${run_eval_name}_summary.json"
  run_followup_name="followup-${CONDITION_KEY}-${EXP_DATE}-${suffix}"
  run_followup_checkpoint_name="${run_followup_name}-${CONDITION_KEY}-from-${FOLLOWUP_SOURCE_LABEL}"
  run_followup_summary="results/experiment_data/reports/${run_followup_name}_summary.json"
  run_followup_eval_name="${run_eval_name}-followup"
  run_followup_eval_summary="results/experiment_data/reports/${run_followup_eval_name}_summary.json"
  run_sim_log=$(suffix_path "$SIM_LOG" "$suffix")
  run_eval_log=$(suffix_path "$EVAL_LOG" "$suffix")
  run_followup_eval_log=$(suffix_path "$run_eval_log" "followup")
  local detected_anchor_snapshot=""
  local forced_recovery_prepared=false
  local forced_rollback_dry_run=false

  echo "=========================================="
  echo " 重复实验 ${suffix}"
  echo " batch:     ${batch_name}"
  echo " condition: ${SIM_CONDITION}"
  echo " key:       ${CONDITION_KEY}"
  echo " run_name:  ${run_name:-'(not created)'}"
  echo " status:    ${state_status}"
  echo "=========================================="

  if [[ "$AUTO_DETECT_FORCED_LLM_ROLLBACK" == true && -n "$run_name" ]]; then
    local checkpoint_dir="results/checkpoints/${run_name}"
    local -a checkpoint_snapshots=("${checkpoint_dir}"/simulate-*.json)
    if [[ -e "${checkpoint_snapshots[0]}" ]]; then
      local analysis_path="$STATUS_DIR/${repeat_index}.forced-llm-analysis.json"
      local repeat_condition_dir="results/experiment_data/repeat_scale_eval/${run_eval_name}/${EVAL_CONDITION}"
      local -a detector_cmd=(
        python runshells/detect_forced_llm_rollback.py
        --checkpoint-dir "$checkpoint_dir"
        --repeat-eval-condition-dir "$repeat_condition_dir"
        --min-calls "$FORCED_LLM_MIN_CALLS"
        --failure-ratio "$FORCED_LLM_FAILURE_RATIO"
        --consecutive-sim-meetings "$FORCED_LLM_CONSECUTIVE_SIM_MEETINGS"
      )
      echo "[CHECK] forced_llm 持续故障与安全锚点: $run_name"
      if ! "${detector_cmd[@]}" > "$analysis_path"; then
        echo "错误: forced_llm 检测失败；报告保留于 $analysis_path" >&2
        [[ -s "$analysis_path" ]] && jq . "$analysis_path" >&2
        return 2
      fi

      local detector_action failure_meeting anchor_step
      detector_action=$(jq -r '.action' "$analysis_path")
      detected_anchor_snapshot=$(jq -r '.recommended_anchor.snapshot_name // ""' "$analysis_path")
      anchor_step=$(jq -r '.recommended_anchor.step_no // 0' "$analysis_path")
      failure_meeting=$(jq -r '.simulation.failure_start.meeting_id // ""' "$analysis_path")
      echo "[CHECK] action=${detector_action} anchor=${detected_anchor_snapshot:-'(none)'} step=${anchor_step} failure_start=${failure_meeting:-'(none)'}"

      if [[ "$detector_action" == "simulation_rollback" ]]; then
        local -a related_paths=()
        local candidate
        for candidate in \
          "results/experiment_data/${run_name}" \
          "results/compressed/${run_name}" \
          "results/experiment_data/batch_state/${batch_name}/timings/${CONDITION_KEY}.jsonl" \
          "results/experiment_data/reports/${batch_name}_timings.jsonl" \
          "$run_summary"; do
          [[ -e "$candidate" ]] && related_paths+=("$candidate")
        done
        while IFS= read -r candidate; do
          [[ -n "$candidate" && -e "$candidate" ]] && related_paths+=("$candidate")
        done < <(jq -r '.repeat_eval.invalid_after_anchor_dirs[]?' "$analysis_path")
        for candidate in \
          "results/experiment_data/reports/${run_eval_name}_summary.json" \
          "results/experiment_data/reports/${run_eval_name}_summary.md" \
          "results/experiment_data/reports/${run_eval_name}_incomplete.json" \
          "results/experiment_data/reports/${run_eval_name}_incomplete.md"; do
          [[ -e "$candidate" ]] && related_paths+=("$candidate")
        done

        local -a recovery_cmd=(
          python runshells/prepare_sparse_checkpoint_resume.py
          --run-name "$run_name"
          --target-step "$SIM_TARGET_STEP"
          --anchor-snapshot "$detected_anchor_snapshot"
          --batch-state-path "$state_path"
        )
        for candidate in "${related_paths[@]}"; do
          recovery_cmd+=(--quarantine-path "$candidate")
        done
        if [[ "$DRY_RUN" == true ]]; then
          recovery_cmd+=(--dry-run)
          print_command "${recovery_cmd[@]}"
          forced_rollback_dry_run=true
        fi
        "${recovery_cmd[@]}"
        recovery_exit_code=$?
        if [[ "$recovery_exit_code" != "0" ]]; then
          echo "错误: forced_llm 回溯准备失败 exit_code=${recovery_exit_code}" >&2
          return "$recovery_exit_code"
        fi
        forced_recovery_prepared=true
        state_status="failed_after_checkpoint"
      elif [[ "$detector_action" == "repeat_eval_repair" ]]; then
        local -a bad_repeat_paths=()
        while IFS= read -r candidate; do
          [[ -n "$candidate" && -e "$candidate" ]] && bad_repeat_paths+=("$candidate")
        done < <(jq -r '.repeat_eval.bad_unit_dirs[]?' "$analysis_path")
        for candidate in \
          "results/experiment_data/reports/${run_eval_name}_summary.json" \
          "results/experiment_data/reports/${run_eval_name}_summary.md" \
          "results/experiment_data/reports/${run_eval_name}_incomplete.json" \
          "results/experiment_data/reports/${run_eval_name}_incomplete.md"; do
          [[ -e "$candidate" ]] && bad_repeat_paths+=("$candidate")
        done
        quarantine_related_paths \
          "$run_name" \
          "repeat_eval_forced_llm_failure" \
          "${bad_repeat_paths[@]}"
      elif [[ "$detector_action" != "normal_resume" ]]; then
        echo "错误: forced_llm 检测器返回未知 action=$detector_action" >&2
        return 2
      fi
    else
      echo "[INFO] 尚无 snapshot，跳过 forced_llm 自动检测。"
    fi
  fi

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
      not_started)
        echo "[INFO] 该重复尚未启动，将直接运行完整 batch。"
        ;;
      *)
        local checkpoint_dir="results/checkpoints/${run_name}"
        local -a checkpoint_snapshots=("${checkpoint_dir}"/simulate-*.json)
        if [[ "$forced_recovery_prepared" == true ]]; then
          echo "[INFO] 已按 forced_llm 检测结果准备恢复锚点。"
        elif [[ -e "${checkpoint_snapshots[0]}" ]]; then
          local -a recovery_cmd=(
            python runshells/prepare_sparse_checkpoint_resume.py
            --run-name "$run_name"
            --target-step "$SIM_TARGET_STEP"
          )
          if [[ -n "$detected_anchor_snapshot" ]]; then
            recovery_cmd+=(--anchor-snapshot "$detected_anchor_snapshot")
          fi
          if [[ "$DRY_RUN" == true ]]; then
            recovery_cmd+=(--dry-run)
          fi
          "${recovery_cmd[@]}"
          recovery_exit_code=$?
          if [[ "$recovery_exit_code" != "0" ]]; then
            echo "错误: 稀疏 checkpoint 恢复准备失败 exit_code=${recovery_exit_code}" >&2
            return "$recovery_exit_code"
          fi
        else
          echo "[INFO] 当前状态尚无可用快照，将沿用原 run_name 从头启动。"
        fi
        if [[ "$REFRESH_MODEL_ROUTING" == true && -n "$run_name" && -e "${checkpoint_snapshots[0]}" ]]; then
          refresh_model_routing "$run_name"
        elif [[ "$REFRESH_MODEL_ROUTING" == true ]]; then
          echo "[INFO] 当前重复没有可恢复 snapshot；新实验会直接使用 $ROUTING_CONFIG。"
        fi
        ;;
    esac

    local -a batch_cmd=(
      python runshells/run_batch_experiment.py
      --name "$batch_name"
      --condition "$SIM_CONDITION"
      --step "$SIM_TARGET_STEP"
      --stride "$SIM_STRIDE"
      --max-parallel "$SIM_MAX_PARALLEL"
      --log "$SIM_CHECKPOINT_LOG"
      --skip-post-scale
      --cbt-controller "$CBT_CONTROLLER"
    )
    if [[ "$state_status" != "not_started" ]]; then
      batch_cmd+=(--resume-condition "$SIM_CONDITION")
    fi
    if [[ -n "$PROGRESSIVE_STAGE" ]]; then
      batch_cmd+=(--progressive-stage "$PROGRESSIVE_STAGE")
    fi
    if [[ -n "$BASE_CONFIG" ]]; then
      batch_cmd+=(--config "$BASE_CONFIG")
    fi
    if [[ -n "$OUTPUT_TAG" ]]; then
      batch_cmd+=(--output-tag "$OUTPUT_TAG")
    fi
    if [[ "$COUNSEL_ROOM" == true ]]; then
      batch_cmd+=(--counsel-room)
    fi
    if [[ "$DRY_RUN" == true ]]; then
      batch_cmd+=(--dry-run)
      print_command env "BATCH_EMBEDDING_BASE_URLS=$SIM_EMBEDDING_BASE_URLS" "${batch_cmd[@]}"
      if [[ "$forced_rollback_dry_run" == true ]]; then
        echo "[DRY-RUN] batch state 将在实际回溯时改为 failed_after_checkpoint；此处不执行会误读旧 completed 状态的 batch dry-run。"
      else
        BATCH_EMBEDDING_BASE_URLS="$SIM_EMBEDDING_BASE_URLS" "${batch_cmd[@]}"
        batch_exit_code=$?
        if [[ "$batch_exit_code" != "0" ]]; then
          return "$batch_exit_code"
        fi
      fi
    else
      {
        echo "=========================================="
        echo " 恢复仿真开始 $(date '+%F %T')"
        echo " run_name: $run_name"
      } >> "$run_sim_log"
      BATCH_EMBEDDING_BASE_URLS="$SIM_EMBEDDING_BASE_URLS" \
        "${batch_cmd[@]}" >> "$run_sim_log" 2>&1
      batch_exit_code=$?
      if [[ "$batch_exit_code" != "0" ]]; then
        echo "错误: 恢复 batch 失败 exit_code=${batch_exit_code} $(date '+%F %T')" \
          | tee -a "$run_sim_log" >&2
        return "$batch_exit_code"
      fi
      if [[ ! -f "$run_summary" ]]; then
        echo "错误: 恢复后未生成 summary: $run_summary" | tee -a "$run_sim_log" >&2
        return 1
      fi
      echo "恢复仿真与后处理完成 $(date '+%F %T')" >> "$run_sim_log"
    fi
  fi

  local followup_pid=""
  if [[ "$FOLLOWUP_ENABLED" == true ]]; then
    if [[ "$REFRESH_MODEL_ROUTING" == true
      && "$FOLLOWUP_RESUME_PARTIAL" == true
      && ! -f "$run_followup_summary" ]]; then
      local followup_checkpoint_dir="results/checkpoints/${run_followup_checkpoint_name}"
      local -a followup_checkpoint_snapshots=("${followup_checkpoint_dir}"/simulate-*.json)
      if [[ -e "${followup_checkpoint_snapshots[0]}" ]]; then
        refresh_model_routing "$run_followup_checkpoint_name"
      fi
    fi
    local -a followup_cmd=(
      python runshells/run_post_sim_followup.py
      --source-summary "$run_summary"
      --source-label "$FOLLOWUP_SOURCE_LABEL"
      --name "$run_followup_name"
      --steps "$FOLLOWUP_STEPS"
      --interval "$FOLLOWUP_INTERVAL"
      --max-parallel "$FOLLOWUP_MAX_PARALLEL"
    )
    if [[ "$FOLLOWUP_RESUME_PARTIAL" == true ]]; then
      followup_cmd+=(--resume-partial)
    else
      followup_cmd+=(--no-resume-partial)
    fi
    if [[ "$DRY_RUN" == true ]]; then
      followup_cmd+=(--dry-run)
      echo "[DRY-RUN] 原仿真复评与 follow-up 将并行启动；回访复评等待 follow-up 完成。"
      print_command "${followup_cmd[@]}"
    else
      {
        echo "=========================================="
        echo " 恢复流水线：无干预回访开始（与原仿真复评并行） $(date '+%F %T')"
        echo " 回访名称: $run_followup_name"
        echo " 源 summary: $run_summary"
      } >> "$run_sim_log"
      "${followup_cmd[@]}" >> "$run_sim_log" 2>&1 &
      followup_pid=$!
    fi
  fi

  local original_eval_exit_code=0
  run_or_resume_repeat_eval \
    "$run_summary" \
    "$EVAL_LABELS" \
    "$run_eval_name" \
    "$run_eval_summary" \
    "$run_eval_log" || original_eval_exit_code=$?
  if [[ "$original_eval_exit_code" != "0" ]]; then
    if [[ -n "$followup_pid" ]]; then
      kill "$followup_pid" 2>/dev/null || true
      wait "$followup_pid" 2>/dev/null || true
    fi
    return "$original_eval_exit_code"
  fi

  if [[ "$FOLLOWUP_ENABLED" == true ]]; then
    if [[ "$DRY_RUN" != true ]]; then
      local followup_exit_code=0
      wait "$followup_pid" || followup_exit_code=$?
      if [[ "$followup_exit_code" != "0" ]]; then
        echo "错误: 回访失败 exit_code=${followup_exit_code}" | tee -a "$run_sim_log" >&2
        return "$followup_exit_code"
      fi
      if [[ ! -f "$run_followup_summary" ]]; then
        echo "错误: 回访未生成 summary: $run_followup_summary" | tee -a "$run_sim_log" >&2
        return 1
      fi
    fi
    run_or_resume_repeat_eval \
      "$run_followup_summary" \
      "$(followup_labels)" \
      "$run_followup_eval_name" \
      "$run_followup_eval_summary" \
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
  printf '\r恢复进度 [%-*s] %3d%% (%d/%d，已用时 %02d:%02d)' \
    "$width" "${bar}${spaces}" "$percent" "$completed" "$total" \
    $(( elapsed / 60 )) $(( elapsed % 60 ))
}

echo "=========================================="
echo " 稀疏 checkpoint 安全恢复流水线"
echo "=========================================="
echo " 条件:          $SIM_CONDITION"
echo " Condition key: $CONDITION_KEY"
echo " CBT controller:$CBT_CONTROLLER"
echo " Progressive:   ${PROGRESSIVE_STAGE:-'(none)'}"
echo " Output tag:    ${OUTPUT_TAG:-'(none)'}"
echo " Base config:   ${BASE_CONFIG:-data/config.json}"
echo " 复评底稿清理:  $CLEANUP_COMPLETED_ARTIFACTS"
echo " 刷新模型路由:  $REFRESH_MODEL_ROUTING"
if [[ "$REFRESH_MODEL_ROUTING" == true ]]; then
  echo " 路由来源:      $ROUTING_CONFIG"
fi
echo " 模式:          $([[ "$COUNSEL_ROOM" == true ]] && echo 咨询室 || echo 村庄)"
echo " 回访阶段:      $FOLLOWUP_ENABLED"
if [[ "$FOLLOWUP_ENABLED" == true ]]; then
  echo " 回访步数/间隔: ${FOLLOWUP_STEPS}/${FOLLOWUP_INTERVAL}"
  echo " 回访源节点:    $FOLLOWUP_SOURCE_LABEL"
  echo " 回访并行数:    $FOLLOWUP_MAX_PARALLEL"
  echo " 回访 partial 续跑: $FOLLOWUP_RESUME_PARTIAL"
fi
if [[ -n "$REPEAT_INDEX" ]]; then
  echo " 重复存档:      ${SIM_NAME}-$(printf '%02d' "$REPEAT_INDEX")（仅此编号）"
else
  echo " 重复存档:      ${SIM_NAME}-01 至 ${SIM_NAME}-$(printf '%02d' "$REPEAT_COUNT")"
fi
echo " 目标总步数:    $SIM_TARGET_STEP"
echo " 外层并行数:    $MAX_PARALLEL_REPEATS"
echo " 长量表复评次数:  $EVAL_REPEAT"
echo " 中间短量表次数:  $EVAL_INTERMEDIATE_SCALE_REPEATS"
echo " forced_llm检测:$AUTO_DETECT_FORCED_LLM_ROLLBACK"
if [[ "$AUTO_DETECT_FORCED_LLM_ROLLBACK" == true ]]; then
  echo " 检测阈值:      min_calls=${FORCED_LLM_MIN_CALLS}, failure_ratio=${FORCED_LLM_FAILURE_RATIO}, consecutive=${FORCED_LLM_CONSECUTIVE_SIM_MEETINGS}"
fi
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
