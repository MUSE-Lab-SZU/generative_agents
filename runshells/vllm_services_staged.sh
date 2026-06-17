#!/usr/bin/env bash

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
STATE_DIR="${STATE_DIR:-$PROJECT_DIR/.vllm}"
LOG_DIR="${LOG_DIR:-$STATE_DIR/logs}"

MODEL_ROOT="${MODEL_ROOT:-/share/home/tm866039793920000/a874457430/MODEL}"
QWEN_MODEL_DIR="${QWEN_MODEL_DIR:-$MODEL_ROOT/Qwen3-8B}"
EMBED_MODEL_DIR="${EMBED_MODEL_DIR:-$MODEL_ROOT/bge-m3}"

QWEN_GPUS="${QWEN_GPUS:-${QWEN_GPU:-0,1,2,3}}"
EMBED_GPUS="${EMBED_GPUS:-${EMBED_GPU:-3}}"
QWEN_PORT="${QWEN_PORT:-18000}"
EMBED_PORT="${EMBED_PORT:-18001}"

QWEN_NAME="${QWEN_NAME:-qwen3-8b-vllm}"
EMBED_NAME="${EMBED_NAME:-bge-m3-vllm}"
HOST="${HOST:-0.0.0.0}"
VLLM_BIN="${VLLM_BIN:-vllm}"
PYTHON_BIN="${PYTHON_BIN:-python}"

QWEN_DTYPE="${QWEN_DTYPE:-half}"
EMBED_DTYPE="${EMBED_DTYPE:-half}"

# This staged script is tuned for the 4-GPU experiment host. Qwen still leaves
# room on the shared embedding GPU, but uses a wider context than the 1-GPU run.
QWEN_GPU_MEMORY_UTILIZATION="${QWEN_GPU_MEMORY_UTILIZATION:-0.76}"
EMBED_GPU_MEMORY_UTILIZATION="${EMBED_GPU_MEMORY_UTILIZATION:-0.18}"

QWEN_MAX_MODEL_LEN="${QWEN_MAX_MODEL_LEN:-32768}"
EMBED_MAX_MODEL_LEN="${EMBED_MAX_MODEL_LEN:-8192}"
QWEN_TENSOR_PARALLEL_SIZE="${QWEN_TENSOR_PARALLEL_SIZE:-}"
EMBED_TENSOR_PARALLEL_SIZE="${EMBED_TENSOR_PARALLEL_SIZE:-}"

WAIT_TIMEOUT_SECONDS="${WAIT_TIMEOUT_SECONDS:-900}"
WAIT_INTERVAL_SECONDS="${WAIT_INTERVAL_SECONDS:-5}"

QWEN_PID_FILE="$STATE_DIR/qwen3.pid"
EMBED_PID_FILE="$STATE_DIR/bge_m3.pid"
QWEN_LOG_FILE="$LOG_DIR/qwen3.log"
EMBED_LOG_FILE="$LOG_DIR/bge_m3.log"

ACTION="${1:-start}"

mkdir -p "$STATE_DIR" "$LOG_DIR"

VLLM_CMD=()

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

resolve_vllm_cmd() {
    if command -v "$VLLM_BIN" >/dev/null 2>&1; then
        VLLM_CMD=("$VLLM_BIN")
        return 0
    fi

    if command -v "$PYTHON_BIN" >/dev/null 2>&1 && "$PYTHON_BIN" -m pip show vllm >/dev/null 2>&1; then
        VLLM_CMD=("$PYTHON_BIN" -m vllm)
        return 0
    fi

    echo "未找到 vLLM 可执行入口: $VLLM_BIN" >&2
    echo "请确认当前环境已安装 vllm，或设置 VLLM_BIN/PYTHON_BIN。" >&2
    exit 1
}

wait_for_model() {
    local service_name="$1"
    local base_url="$2"
    local model_name="$3"
    local pid_file="$4"
    local log_file="$5"
    local started
    local now
    local elapsed

    started="$(date +%s)"
    echo "等待 $service_name 就绪: $base_url, model=$model_name, timeout=${WAIT_TIMEOUT_SECONDS}s"

    while true; do
        if ! is_pid_running "$pid_file"; then
            echo "$service_name 进程已退出，最近日志如下：" >&2
            tail -80 "$log_file" >&2 || true
            exit 1
        fi

        if "$PYTHON_BIN" - "$base_url" "$model_name" >/dev/null 2>&1 <<'PY'
import json
import sys
import urllib.error
import urllib.request

base_url, model_name = sys.argv[1:3]
base_url = base_url.rstrip("/")

def get_json(path):
    with urllib.request.urlopen(base_url + path, timeout=5) as resp:
        body = resp.read().decode("utf-8", errors="replace")
    if not body.strip():
        return {}
    return json.loads(body)

try:
    urllib.request.urlopen(base_url + "/health", timeout=5).read()
    payload = get_json("/v1/models")
except Exception:
    sys.exit(1)

models = payload.get("data", [])
ids = {str(item.get("id", "")) for item in models if isinstance(item, dict)}
sys.exit(0 if model_name in ids else 1)
PY
        then
            echo "$service_name 已就绪"
            return 0
        fi

        now="$(date +%s)"
        elapsed=$((now - started))
        if [ "$elapsed" -ge "$WAIT_TIMEOUT_SECONDS" ]; then
            echo "$service_name 等待超时，最近日志如下：" >&2
            tail -120 "$log_file" >&2 || true
            exit 1
        fi

        sleep "$WAIT_INTERVAL_SECONDS"
    done
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
    echo "  Mem:    $gpu_memory_utilization"
    echo "  Log:    $log_file"

    if [ -n "$runner_extra" ]; then
        # runner_extra comes from a trusted constant in this script.
        # shellcheck disable=SC2206
        extra_args=($runner_extra)
    fi

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
    echo "$service_name 启动命令已提交，PID=$(cat "$pid_file")"
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

start_qwen() {
    local qwen_tp
    qwen_tp="$(resolve_tensor_parallel_size "$QWEN_TENSOR_PARALLEL_SIZE" "$QWEN_GPUS")"
    start_service "Qwen3" "$QWEN_MODEL_DIR" "$QWEN_GPUS" "$QWEN_PORT" "$QWEN_NAME" \
        "$QWEN_DTYPE" "$QWEN_GPU_MEMORY_UTILIZATION" "$QWEN_MAX_MODEL_LEN" "$qwen_tp" "" \
        "$QWEN_PID_FILE" "$QWEN_LOG_FILE"
}

start_embed() {
    local embed_tp
    embed_tp="$(resolve_tensor_parallel_size "$EMBED_TENSOR_PARALLEL_SIZE" "$EMBED_GPUS")"
    start_service "BGE-M3" "$EMBED_MODEL_DIR" "$EMBED_GPUS" "$EMBED_PORT" "$EMBED_NAME" \
        "$EMBED_DTYPE" "$EMBED_GPU_MEMORY_UTILIZATION" "$EMBED_MAX_MODEL_LEN" "$embed_tp" "--runner pooling --convert embed" \
        "$EMBED_PID_FILE" "$EMBED_LOG_FILE"
}

print_status() {
    echo "vLLM 分阶段服务状态"
    echo "  Qwen3 service : $(if is_pid_running "$QWEN_PID_FILE"; then echo "RUNNING pid=$(cat "$QWEN_PID_FILE")"; else echo "STOPPED"; fi)"
    echo "  BGE-M3 service: $(if is_pid_running "$EMBED_PID_FILE"; then echo "RUNNING pid=$(cat "$EMBED_PID_FILE")"; else echo "STOPPED"; fi)"
    echo "  Qwen GPUs     : $QWEN_GPUS (mem=$QWEN_GPU_MEMORY_UTILIZATION)"
    echo "  Embed GPUs    : $EMBED_GPUS (mem=$EMBED_GPU_MEMORY_UTILIZATION)"
    echo "  Qwen URL      : http://127.0.0.1:$QWEN_PORT/v1/chat/completions"
    echo "  Embed URL     : http://127.0.0.1:$EMBED_PORT/v1/embeddings"
    echo "  Qwen Log      : $QWEN_LOG_FILE"
    echo "  Embed Log     : $EMBED_LOG_FILE"
}

print_usage() {
    cat <<EOF
用法: $0 [start|start-qwen|start-embed|stop|restart|status|help]

start 会先启动 Qwen，确认 Qwen 服务就绪后，再启动 BGE-M3 embedding。

共享 GPU 示例：
  QWEN_GPUS=0,1,2,3 EMBED_GPUS=3 bash runshells/vllm_services_staged.sh start

如果 embedding 仍然 OOM，可以继续降低：
  QWEN_GPU_MEMORY_UTILIZATION=0.60 EMBED_GPU_MEMORY_UTILIZATION=0.15
EOF
}

case "$ACTION" in
    start)
        resolve_vllm_cmd
        start_qwen
        wait_for_model "Qwen3" "http://127.0.0.1:$QWEN_PORT" "$QWEN_NAME" "$QWEN_PID_FILE" "$QWEN_LOG_FILE"
        start_embed
        wait_for_model "BGE-M3" "http://127.0.0.1:$EMBED_PORT" "$EMBED_NAME" "$EMBED_PID_FILE" "$EMBED_LOG_FILE"
        print_status
        ;;
    start-qwen)
        resolve_vllm_cmd
        start_qwen
        wait_for_model "Qwen3" "http://127.0.0.1:$QWEN_PORT" "$QWEN_NAME" "$QWEN_PID_FILE" "$QWEN_LOG_FILE"
        print_status
        ;;
    start-embed)
        resolve_vllm_cmd
        start_embed
        wait_for_model "BGE-M3" "http://127.0.0.1:$EMBED_PORT" "$EMBED_NAME" "$EMBED_PID_FILE" "$EMBED_LOG_FILE"
        print_status
        ;;
    stop)
        stop_service "BGE-M3" "$EMBED_PID_FILE"
        stop_service "Qwen3" "$QWEN_PID_FILE"
        ;;
    restart)
        "$0" stop
        "$0" start
        ;;
    status)
        print_status
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
