#!/usr/bin/env bash

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
STATE_DIR="${STATE_DIR:-$PROJECT_DIR/.vllm}"
LOG_DIR="${LOG_DIR:-$STATE_DIR/logs}"

MODEL_ROOT="${MODEL_ROOT:-/mnt/nvme1/zyli/MODEL}"
QWEN_MODEL_DIR="${QWEN_MODEL_DIR:-$MODEL_ROOT/Qwen3-8B}"
EMBED_MODEL_DIR="${EMBED_MODEL_DIR:-$MODEL_ROOT/bge-m3}"

QWEN_GPUS="${QWEN_GPUS:-${QWEN_GPU:-3}}"
EMBED_GPUS="${EMBED_GPUS:-${EMBED_GPU:-0}}"
QWEN_PORT="${QWEN_PORT:-18000}"
EMBED_PORT="${EMBED_PORT:-18001}"

QWEN_NAME="${QWEN_NAME:-qwen3-8b-vllm}"
EMBED_NAME="${EMBED_NAME:-bge-m3-vllm}"
HOST="${HOST:-0.0.0.0}"
VLLM_BIN="${VLLM_BIN:-vllm}"
PYTHON_BIN="${PYTHON_BIN:-python}"

QWEN_DTYPE="${QWEN_DTYPE:-half}"
EMBED_DTYPE="${EMBED_DTYPE:-half}"
QWEN_GPU_MEMORY_UTILIZATION="${QWEN_GPU_MEMORY_UTILIZATION:-0.90}"
EMBED_GPU_MEMORY_UTILIZATION="${EMBED_GPU_MEMORY_UTILIZATION:-0.25}"
QWEN_MAX_MODEL_LEN="${QWEN_MAX_MODEL_LEN:-32768}"
EMBED_MAX_MODEL_LEN="${EMBED_MAX_MODEL_LEN:-8192}"
QWEN_TENSOR_PARALLEL_SIZE="${QWEN_TENSOR_PARALLEL_SIZE:-}"
EMBED_TENSOR_PARALLEL_SIZE="${EMBED_TENSOR_PARALLEL_SIZE:-}"

QWEN_PID_FILE="$STATE_DIR/qwen3.pid"
EMBED_PID_FILE="$STATE_DIR/bge_m3.pid"
QWEN_LOG_FILE="$LOG_DIR/qwen3.log"
EMBED_LOG_FILE="$LOG_DIR/bge_m3.log"

ACTION="${1:-start}"

mkdir -p "$STATE_DIR" "$LOG_DIR"

VLLM_CMD=()

require_command() {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "未找到命令: $1" >&2
        exit 1
    fi
}

check_model_dir() {
    local model_dir="$1"
    if [ ! -d "$model_dir" ]; then
        echo "模型目录不存在: $model_dir" >&2
        exit 1
    fi
}

count_gpu_ids() {
    local gpu_ids
    gpu_ids="${1//[[:space:]]/}"

    if [ -z "$gpu_ids" ]; then
        echo 1
        return 0
    fi

    local comma_count
    comma_count="${gpu_ids//[^,]/}"
    echo $((${#comma_count} + 1))
}

resolve_tensor_parallel_size() {
    local explicit_value="$1"
    local gpu_ids="$2"

    if [ -n "$explicit_value" ]; then
        echo "$explicit_value"
        return 0
    fi

    count_gpu_ids "$gpu_ids"
}

is_pid_running() {
    local pid_file="$1"
    if [ ! -f "$pid_file" ]; then
        return 1
    fi

    local pid
    pid="$(cat "$pid_file")"
    if [ -z "$pid" ]; then
        return 1
    fi

    kill -0 "$pid" >/dev/null 2>&1
}

print_install_hint() {
    cat >&2 <<EOF
未找到 vLLM 可执行入口。

请先在当前 Conda 环境安装运行依赖，例如：
  conda activate llm-depression
  bash runshells/install_vllm_runtime.sh

如果你想手动安装，也可以执行：
  python -m pip install -U vllm modelscope

安装后建议验证：
  python -m pip show vllm
  vllm --help
EOF
}

resolve_vllm_cmd() {
    if command -v "$VLLM_BIN" >/dev/null 2>&1; then
        VLLM_CMD=("$VLLM_BIN")
        return 0
    fi

    if command -v "$PYTHON_BIN" >/dev/null 2>&1 && "$PYTHON_BIN" -m pip show vllm >/dev/null 2>&1; then
        VLLM_CMD=("$PYTHON_BIN" -m vllm)
        return 0
    fi

    print_install_hint
    exit 1
}

start_service() {
    local service_name="$1"
    local model_dir="$2"
    local gpu_ids="$3"
    local port="$4"
    local served_name="$5"
    local dtype="$6"
    local gpu_memory_utilization="$7"
    local max_model_len="$8"
    local tensor_parallel_size="$9"
    local runner_extra="${10}"
    local pid_file="${11}"
    local log_file="${12}"
    local extra_args=()

    if is_pid_running "$pid_file"; then
        echo "$service_name 已在运行，PID=$(cat "$pid_file")"
        return 0
    fi

    check_model_dir "$model_dir"

    echo "启动 $service_name"
    echo "  GPUs:   $gpu_ids"
    echo "  TP:     $tensor_parallel_size"
    echo "  Port:   $port"
    echo "  Model:  $model_dir"
    echo "  Log:    $log_file"

    if [ -n "$runner_extra" ]; then
        # runner_extra comes from a trusted constant in this script.
        # shellcheck disable=SC2206
        extra_args=($runner_extra)
    fi

    # Keep options conservative by default.  Set QWEN_GPUS/EMBED_GPUS to a
    # comma-separated list such as 0,1,2,3 to enable tensor parallel inference.
    CUDA_VISIBLE_DEVICES="$gpu_ids" nohup "${VLLM_CMD[@]}" serve "$model_dir" \
        --host "$HOST" \
        --port "$port" \
        --served-model-name "$served_name" \
        --dtype "$dtype" \
        --tensor-parallel-size "$tensor_parallel_size" \
        --gpu-memory-utilization "$gpu_memory_utilization" \
        --max-model-len "$max_model_len" \
        "${extra_args[@]}" \
        >"$log_file" 2>&1 &

    echo $! >"$pid_file"
    echo "$service_name 启动完成，PID=$(cat "$pid_file")"
}

stop_service() {
    local service_name="$1"
    local pid_file="$2"

    if ! is_pid_running "$pid_file"; then
        echo "$service_name 未运行"
        rm -f "$pid_file"
        return 0
    fi

    local pid
    pid="$(cat "$pid_file")"
    kill "$pid"
    rm -f "$pid_file"
    echo "$service_name 已停止，PID=$pid"
}

print_status() {
    local qwen_tp
    local embed_tp
    qwen_tp="$(resolve_tensor_parallel_size "$QWEN_TENSOR_PARALLEL_SIZE" "$QWEN_GPUS")"
    embed_tp="$(resolve_tensor_parallel_size "$EMBED_TENSOR_PARALLEL_SIZE" "$EMBED_GPUS")"

    echo "vLLM 服务状态"
    echo "  Qwen3 service : $(if is_pid_running "$QWEN_PID_FILE"; then echo "RUNNING pid=$(cat "$QWEN_PID_FILE")"; else echo "STOPPED"; fi)"
    echo "  BGE-M3 service: $(if is_pid_running "$EMBED_PID_FILE"; then echo "RUNNING pid=$(cat "$EMBED_PID_FILE")"; else echo "STOPPED"; fi)"
    echo "  Qwen GPUs     : $QWEN_GPUS (tensor_parallel_size=$qwen_tp)"
    echo "  Embed GPUs    : $EMBED_GPUS (tensor_parallel_size=$embed_tp)"
    echo "  Qwen URL      : http://127.0.0.1:$QWEN_PORT/v1/chat/completions"
    echo "  Embed URL     : http://127.0.0.1:$EMBED_PORT/v1/embeddings"
    echo "  Qwen Log      : $QWEN_LOG_FILE"
    echo "  Embed Log     : $EMBED_LOG_FILE"
}

print_config() {
    cat <<EOF
推荐写入 data/config.json 的关键字段：

agent.think.llm
  provider: openai
  model: $QWEN_NAME
  base_url: http://127.0.0.1:$QWEN_PORT/v1
  api_key: EMPTY

agent.associate.embedding
  provider: openai
  model: $EMBED_NAME
  base_url: http://127.0.0.1:$EMBED_PORT/v1
  api_key: EMPTY
EOF
}

print_usage() {
    cat <<EOF
用法: $0 [start|stop|restart|status|print-config]

默认规划：
  Qwen3  -> GPUs $QWEN_GPUS, port $QWEN_PORT, model $QWEN_MODEL_DIR
  BGE-M3 -> GPUs $EMBED_GPUS, port $EMBED_PORT, model $EMBED_MODEL_DIR

可覆盖环境变量：
  MODEL_ROOT, QWEN_MODEL_DIR, EMBED_MODEL_DIR
  QWEN_GPUS, EMBED_GPUS, QWEN_TENSOR_PARALLEL_SIZE, EMBED_TENSOR_PARALLEL_SIZE
  QWEN_GPU, EMBED_GPU, QWEN_PORT, EMBED_PORT
  QWEN_NAME, EMBED_NAME, VLLM_BIN, PYTHON_BIN

多卡示例：
  1 卡: QWEN_GPUS=0 EMBED_GPUS=0 bash runshells/vllm_services.sh start
  2 卡: QWEN_GPUS=0 EMBED_GPUS=1 bash runshells/vllm_services.sh start
  4 卡: QWEN_GPUS=0,1,2 EMBED_GPUS=3 bash runshells/vllm_services.sh start
  8 卡: QWEN_GPUS=0,1,2,3,4,5,6 EMBED_GPUS=7 bash runshells/vllm_services.sh start
EOF
}

case "$ACTION" in
    start)
        resolve_vllm_cmd
        QWEN_RESOLVED_TP="$(resolve_tensor_parallel_size "$QWEN_TENSOR_PARALLEL_SIZE" "$QWEN_GPUS")"
        EMBED_RESOLVED_TP="$(resolve_tensor_parallel_size "$EMBED_TENSOR_PARALLEL_SIZE" "$EMBED_GPUS")"
        start_service "Qwen3" "$QWEN_MODEL_DIR" "$QWEN_GPUS" "$QWEN_PORT" "$QWEN_NAME" \
            "$QWEN_DTYPE" "$QWEN_GPU_MEMORY_UTILIZATION" "$QWEN_MAX_MODEL_LEN" "$QWEN_RESOLVED_TP" "" \
            "$QWEN_PID_FILE" "$QWEN_LOG_FILE"
        start_service "BGE-M3" "$EMBED_MODEL_DIR" "$EMBED_GPUS" "$EMBED_PORT" "$EMBED_NAME" \
            "$EMBED_DTYPE" "$EMBED_GPU_MEMORY_UTILIZATION" "$EMBED_MAX_MODEL_LEN" "$EMBED_RESOLVED_TP" "--runner pooling --convert embed" \
            "$EMBED_PID_FILE" "$EMBED_LOG_FILE"
        print_status
        ;;
    stop)
        stop_service "Qwen3" "$QWEN_PID_FILE"
        stop_service "BGE-M3" "$EMBED_PID_FILE"
        ;;
    restart)
        resolve_vllm_cmd
        "$0" stop
        "$0" start
        ;;
    status)
        print_status
        ;;
    print-config)
        print_config
        ;;
    help|-h|--help)
        print_usage
        ;;
    *)
        echo "未知动作: $ACTION" >&2
        print_usage
        exit 1
        ;;
esac
