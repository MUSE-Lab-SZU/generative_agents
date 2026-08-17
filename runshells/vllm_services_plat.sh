#!/usr/bin/env bash

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
STATE_DIR="${STATE_DIR:-$PROJECT_DIR/.vllm}"
LOG_DIR="${LOG_DIR:-$STATE_DIR/logs}"

MODEL_ROOT="${MODEL_ROOT:-/mnt/nvme1/zyli/MODEL}"
QWEN_MODEL_DIR="${QWEN_MODEL_DIR:-$MODEL_ROOT/Qwen3-8B}"
EMBED_MODEL_DIR="${EMBED_MODEL_DIR:-$MODEL_ROOT/bge-m3}"

QWEN_GPUS="${QWEN_GPUS:-${QWEN_GPU:-1}}"
EMBED_GPUS="${EMBED_GPUS:-${EMBED_GPU:-0}}"
QWEN_PORT="${QWEN_PORT:-18000}"
EMBED_PORT="${EMBED_PORT:-18001}"
# When set, these comma-separated values create one vLLM process per item.
# They are deliberately separate from QWEN_GPUS/EMBED_GPUS, whose comma list
# continues to mean tensor parallelism for the legacy single-service mode.
QWEN_SERVICE_GPUS="${QWEN_SERVICE_GPUS:-0,1}"
EMBED_SERVICE_GPUS="${EMBED_SERVICE_GPUS:-2,2}"
QWEN_PORTS="${QWEN_PORTS:-18000,18002}"
EMBED_PORTS="${EMBED_PORTS:-18001,18004}"

QWEN_NAME="${QWEN_NAME:-qwen3-8b-vllm}"
EMBED_NAME="${EMBED_NAME:-bge-m3-vllm}"
HOST="${HOST:-0.0.0.0}"
VLLM_BIN="${VLLM_BIN:-vllm}"
PYTHON_BIN="${PYTHON_BIN:-python}"

QWEN_DTYPE="${QWEN_DTYPE:-half}"
EMBED_DTYPE="${EMBED_DTYPE:-half}"
QWEN_GPU_MEMORY_UTILIZATION="${QWEN_GPU_MEMORY_UTILIZATION:-0.90}"
# If several BGE replicas are placed on one GPU, the default is set after the
# service list is parsed so their KV-cache reservations can coexist.
EMBED_GPU_MEMORY_UTILIZATION="${EMBED_GPU_MEMORY_UTILIZATION:-0.20}"
QWEN_MAX_MODEL_LEN="${QWEN_MAX_MODEL_LEN:-32768}"
EMBED_MAX_MODEL_LEN="${EMBED_MAX_MODEL_LEN:-8192}"
QWEN_TENSOR_PARALLEL_SIZE="${QWEN_TENSOR_PARALLEL_SIZE:-}"
EMBED_TENSOR_PARALLEL_SIZE="${EMBED_TENSOR_PARALLEL_SIZE:-}"

QWEN_PID_FILE="$STATE_DIR/qwen3.pid"
EMBED_PID_FILE="$STATE_DIR/bge_m3.pid"
QWEN_LOG_FILE="$LOG_DIR/qwen3.log"
EMBED_LOG_FILE="$LOG_DIR/bge_m3.log"

declare -a QWEN_SERVICE_GPU_LIST QWEN_SERVICE_PORT_LIST
declare -a EMBED_SERVICE_GPU_LIST EMBED_SERVICE_PORT_LIST

ACTION="${1:-start}"

mkdir -p "$STATE_DIR" "$LOG_DIR"

VLLM_CMD=()

split_csv() {
    local value="$1"
    local -n result="$2"
    local item
    IFS=',' read -r -a result <<< "$value"
    for item in "${result[@]}"; do
        if [ -z "${item//[[:space:]]/}" ]; then
            echo "服务 GPU/端口列表中不能有空项: $value" >&2
            exit 1
        fi
    done
}

initialize_service_pools() {
    if [ -n "$QWEN_SERVICE_GPUS" ] || [ -n "$QWEN_PORTS" ]; then
        if [ -z "$QWEN_SERVICE_GPUS" ] || [ -z "$QWEN_PORTS" ]; then
            echo "多服务 Qwen 必须同时设置 QWEN_SERVICE_GPUS 和 QWEN_PORTS" >&2
            exit 1
        fi
        split_csv "$QWEN_SERVICE_GPUS" QWEN_SERVICE_GPU_LIST
        split_csv "$QWEN_PORTS" QWEN_SERVICE_PORT_LIST
    else
        QWEN_SERVICE_GPU_LIST=("$QWEN_GPUS")
        QWEN_SERVICE_PORT_LIST=("$QWEN_PORT")
    fi

    if [ -n "$EMBED_SERVICE_GPUS" ] || [ -n "$EMBED_PORTS" ]; then
        if [ -z "$EMBED_SERVICE_GPUS" ] || [ -z "$EMBED_PORTS" ]; then
            echo "多服务 BGE 必须同时设置 EMBED_SERVICE_GPUS 和 EMBED_PORTS" >&2
            exit 1
        fi
        split_csv "$EMBED_SERVICE_GPUS" EMBED_SERVICE_GPU_LIST
        split_csv "$EMBED_PORTS" EMBED_SERVICE_PORT_LIST
    else
        EMBED_SERVICE_GPU_LIST=("$EMBED_GPUS")
        EMBED_SERVICE_PORT_LIST=("$EMBED_PORT")
    fi

    if [ "${#QWEN_SERVICE_GPU_LIST[@]}" -ne "${#QWEN_SERVICE_PORT_LIST[@]}" ] || \
       [ "${#EMBED_SERVICE_GPU_LIST[@]}" -ne "${#EMBED_SERVICE_PORT_LIST[@]}" ]; then
        echo "每个服务 GPU 必须对应一个端口" >&2
        exit 1
    fi

    if [ -z "$EMBED_GPU_MEMORY_UTILIZATION" ]; then
        local -A seen_embed_gpus=()
        local gpu
        for gpu in "${EMBED_SERVICE_GPU_LIST[@]}"; do
            if [ -n "${seen_embed_gpus[$gpu]+x}" ]; then
                EMBED_GPU_MEMORY_UTILIZATION="0.30"
                return 0
            fi
            seen_embed_gpus["$gpu"]=1
        done
        EMBED_GPU_MEMORY_UTILIZATION="0.95"
    fi
}

pool_pid_file() {
    local prefix="$1"
    local index="$2"
    if [ "$index" -eq 0 ]; then
        echo "$STATE_DIR/$prefix.pid"
    else
        echo "$STATE_DIR/${prefix}_${index}.pid"
    fi
}

pool_log_file() {
    local prefix="$1"
    local index="$2"
    if [ "$index" -eq 0 ]; then
        echo "$LOG_DIR/$prefix.log"
    else
        echo "$LOG_DIR/${prefix}_${index}.log"
    fi
}

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
    local index gpu port pid_file log_file tp

    echo "vLLM 服务状态"
    echo "  BGE 显存利用率: $EMBED_GPU_MEMORY_UTILIZATION（每个 BGE 服务）"
    for index in "${!QWEN_SERVICE_GPU_LIST[@]}"; do
        gpu="${QWEN_SERVICE_GPU_LIST[$index]}"
        port="${QWEN_SERVICE_PORT_LIST[$index]}"
        pid_file="$(pool_pid_file qwen3 "$index")"
        log_file="$(pool_log_file qwen3 "$index")"
        tp="$(resolve_tensor_parallel_size "$QWEN_TENSOR_PARALLEL_SIZE" "$gpu")"
        echo "  Qwen3[$index] : $(if is_pid_running "$pid_file"; then echo "RUNNING pid=$(cat "$pid_file")"; else echo "STOPPED"; fi), GPU=$gpu, TP=$tp, URL=http://127.0.0.1:$port/v1, log=$log_file"
    done
    for index in "${!EMBED_SERVICE_GPU_LIST[@]}"; do
        gpu="${EMBED_SERVICE_GPU_LIST[$index]}"
        port="${EMBED_SERVICE_PORT_LIST[$index]}"
        pid_file="$(pool_pid_file bge_m3 "$index")"
        log_file="$(pool_log_file bge_m3 "$index")"
        tp="$(resolve_tensor_parallel_size "$EMBED_TENSOR_PARALLEL_SIZE" "$gpu")"
        echo "  BGE-M3[$index]: $(if is_pid_running "$pid_file"; then echo "RUNNING pid=$(cat "$pid_file")"; else echo "STOPPED"; fi), GPU=$gpu, TP=$tp, URL=http://127.0.0.1:$port/v1, log=$log_file"
    done
}

print_config() {
    cat <<EOF
推荐写入 data/config.json 的关键字段：

agent.think.llm
  provider: openai
  model: $QWEN_NAME
  base_url: http://127.0.0.1:${QWEN_SERVICE_PORT_LIST[0]}/v1
  api_key: EMPTY
  load_balancing:
    enabled: true
    ports: [$(IFS=,; echo "${QWEN_SERVICE_PORT_LIST[*]}")]

agent.associate.embedding
  provider: openai
  model: $EMBED_NAME
  base_url: http://127.0.0.1:${EMBED_SERVICE_PORT_LIST[0]}/v1
  api_key: EMPTY
  load_balancing:
    enabled: true
    ports: [$(IFS=,; echo "${EMBED_SERVICE_PORT_LIST[*]}")]
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
  QWEN_SERVICE_GPUS, EMBED_SERVICE_GPUS, QWEN_PORTS, EMBED_PORTS
  QWEN_NAME, EMBED_NAME, QWEN_GPU_MEMORY_UTILIZATION, EMBED_GPU_MEMORY_UTILIZATION
  VLLM_BIN, PYTHON_BIN

多卡示例：
  1 卡: QWEN_GPUS=0 EMBED_GPUS=0 bash runshells/vllm_services_plat.sh start
  2 卡: QWEN_GPUS=0 EMBED_GPUS=1 bash runshells/vllm_services_plat.sh start
  3 卡（2 个 Qwen、2 个 BGE）:
    QWEN_SERVICE_GPUS=0,1 QWEN_PORTS=18000,18002 \\
    EMBED_SERVICE_GPUS=2,2 EMBED_PORTS=18001,18004 \\
    bash runshells/vllm_services_plat.sh start
  8 卡: QWEN_GPUS=0,1,2,3,4,5,6 EMBED_GPUS=7 bash runshells/vllm_services_plat.sh start

同一张卡运行多个 BGE 时，未显式设置 EMBED_GPU_MEMORY_UTILIZATION 会自动使用 0.30；
请按实际显存与最大上下文长度调节。
EOF
}

initialize_service_pools

case "$ACTION" in
    start)
        resolve_vllm_cmd
        for index in "${!QWEN_SERVICE_GPU_LIST[@]}"; do
            gpu="${QWEN_SERVICE_GPU_LIST[$index]}"
            start_service "Qwen3[$index]" "$QWEN_MODEL_DIR" "$gpu" "${QWEN_SERVICE_PORT_LIST[$index]}" "$QWEN_NAME" \
                "$QWEN_DTYPE" "$QWEN_GPU_MEMORY_UTILIZATION" "$QWEN_MAX_MODEL_LEN" "$(resolve_tensor_parallel_size "$QWEN_TENSOR_PARALLEL_SIZE" "$gpu")" "" \
                "$(pool_pid_file qwen3 "$index")" "$(pool_log_file qwen3 "$index")"
        done
        for index in "${!EMBED_SERVICE_GPU_LIST[@]}"; do
            gpu="${EMBED_SERVICE_GPU_LIST[$index]}"
            start_service "BGE-M3[$index]" "$EMBED_MODEL_DIR" "$gpu" "${EMBED_SERVICE_PORT_LIST[$index]}" "$EMBED_NAME" \
                "$EMBED_DTYPE" "$EMBED_GPU_MEMORY_UTILIZATION" "$EMBED_MAX_MODEL_LEN" "$(resolve_tensor_parallel_size "$EMBED_TENSOR_PARALLEL_SIZE" "$gpu")" "--runner pooling --convert embed" \
                "$(pool_pid_file bge_m3 "$index")" "$(pool_log_file bge_m3 "$index")"
        done
        print_status
        ;;
    stop)
        for index in "${!QWEN_SERVICE_GPU_LIST[@]}"; do
            stop_service "Qwen3[$index]" "$(pool_pid_file qwen3 "$index")"
        done
        for index in "${!EMBED_SERVICE_GPU_LIST[@]}"; do
            stop_service "BGE-M3[$index]" "$(pool_pid_file bge_m3 "$index")"
        done
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
