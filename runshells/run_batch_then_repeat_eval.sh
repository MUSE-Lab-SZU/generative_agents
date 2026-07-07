#!/bin/bash
# ============================================================
# 批量仿真 + 重复评估量表
#
# 用法:
#   bash runshells/run_batch_then_repeat_eval.sh
#
# 说明:
#   先运行仿真实验；仿真成功后，再运行 archived repeat scale eval。
#   如需改实验组/日志名，优先修改下方参数区。
# ============================================================

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

# ============================================================
# ↓↓↓ 实验参数在此修改 ↓↓↓
# ============================================================

SIM_NAME="batch-0707"
SIM_CONDITION="Counsel-KBD2-G1-SEV"
SIM_MAX_PARALLEL=1
SIM_EMBEDDING_BASE_URLS="${BATCH_EMBEDDING_BASE_URLS:-http://127.0.0.1:11434}"
SIM_LOG="results/batch-0707-KBD2-G1-SEV_run.log"

EVAL_ARCHIVE_RESULTS_ROOT="results"
EVAL_CONDITION="Counsel-KBD2-G1-SEV"
EVAL_ORIGINAL_SUMMARY="results/experiment_data/reports/batch-0707-Counsel-KBD2-G1-SEV_summary.json"
EVAL_LABELS="T0,session_4,session_8,session_12,session_16,session_20,POST"
EVAL_REPEAT=3
EVAL_NAME="repeat-KBD2-G1-SEV-0707"
EVAL_MAX_PARALLEL=3
EVAL_LOG="results/repeat-KBD2-G1-SEV-0707.log"

# ============================================================
# ↑↑↑ 实验参数在此修改 ↑↑↑
# ============================================================

mkdir -p results

echo "=========================================="
echo " 批量仿真 + 重复评估量表"
echo "=========================================="
echo "  仿真条件:      $SIM_CONDITION"
echo "  仿真日志:      $SIM_LOG"
echo "  评估条件:      $EVAL_CONDITION"
echo "  评估 summary:  $EVAL_ORIGINAL_SUMMARY"
echo "  评估日志:      $EVAL_LOG"
echo "=========================================="
echo ""

echo "[1/2] 运行仿真实验..."
BATCH_EMBEDDING_BASE_URLS="$SIM_EMBEDDING_BASE_URLS" \
python runshells/run_batch_experiment.py \
  --name "$SIM_NAME" \
  --condition "$SIM_CONDITION" \
  --max-parallel "$SIM_MAX_PARALLEL" \
  > "$SIM_LOG" 2>&1

echo "  仿真实验完成，日志: $SIM_LOG"
echo ""

echo "[2/2] 运行重复评估量表..."
python runshells/run_archived_repeat_scale_eval.py \
  --archive-results-root "$EVAL_ARCHIVE_RESULTS_ROOT" \
  --condition "$EVAL_CONDITION" \
  --original-summary "$EVAL_ORIGINAL_SUMMARY" \
  --labels "$EVAL_LABELS" \
  --repeat "$EVAL_REPEAT" \
  --name "$EVAL_NAME" \
  --max-parallel "$EVAL_MAX_PARALLEL" \
  > "$EVAL_LOG" 2>&1

echo "  重复评估量表完成，日志: $EVAL_LOG"
echo ""
echo "=========================================="
echo " 全部完成"
echo "=========================================="
