#!/usr/bin/env bash

# 在已有 embedding 服务不变的前提下，用多张 GPU 仅启动 Qwen chat 服务。
# 默认值针对 2/3 号 RTX 4090 均无 20 GiB 空闲显存的场景做了保守配置。

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
STATE_DIR="${STATE_DIR:-$PROJECT_DIR/.vllm}"
LOG_DIR="${LOG_DIR:-$STATE_DIR/logs}"

MODEL_ROOT="${MODEL_ROOT:-/mnt/nvme1/zyli/MODEL}"
QWEN_MODEL_DIR="${QWEN_MODEL_DIR:-$MODEL_ROOT/Qwen3-8B}"
QWEN_GPUS="${QWEN_GPUS:-2,3}"
QWEN_TENSOR_PARALLEL_SIZE="${QWEN_TENSOR_PARALLEL_SIZE:-2}"
QWEN_PORT="${QWEN_PORT:-18000}"
QWEN_NAME="${QWEN_NAME:-qwen3-8b-vllm}"
QWEN_DTYPE="${QWEN_DTYPE:-half}"

# vLLM 的利用率是“每张卡总显存”的比例，并不是“当前空闲显存”的比例。
# 0.40 在 24 GiB 4090 上约为 9.8 GiB/卡，适合当前 GPU 2/3 的余量。
QWEN_GPU_MEMORY_UTILIZATION="${QWEN_GPU_MEMORY_UTILIZATION:-0.40}"
QWEN_MAX_MODEL_LEN="${QWEN_MAX_MODEL_LEN:-16384}"
QWEN_MAX_NUM_SEQS="${QWEN_MAX_NUM_SEQS:-16}"
QWEN_ENFORCE_EAGER="${QWEN_ENFORCE_EAGER:-1}"
GPU_FREE_HEADROOM_MIB="${GPU_FREE_HEADROOM_MIB:-512}"
STARTUP_TIMEOUT_SECONDS="${STARTUP_TIMEOUT_SECONDS:-180}"

HOST="${HOST:-0.0.0.0}"
VLLM_BIN="${VLLM_BIN:-vllm}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CURL_BIN="${CURL_BIN:-curl}"

PID_FILE="$STATE_DIR/qwen3.pid"
LOG_FILE="$LOG_DIR/qwen3.log"
ACTION="${1:-start}"
VLLM_CMD=()

mkdir -p "$STATE_DIR" "$LOG_DIR"

die() {
    echo "错误: $*" >&2
    exit 1
}

is_positive_integer() {
    [[ "$1" =~ ^[1-9][0-9]*$ ]]
}

is_nonnegative_integer() {
    [[ "$1" =~ ^[0-9]+$ ]]
}

is_pid_running() {
    [[ -f "$PID_FILE" ]] || return 1
    local pid cmdline
    pid="$(<"$PID_FILE")"
    [[ "$pid" =~ ^[1-9][0-9]*$ ]] || return 1
    kill -0 "$pid" >/dev/null 2>&1 || return 1

    # 避免旧 PID 文件中的数字已被系统复用于无关进程时误报或误杀。
    if [[ -r "/proc/$pid/cmdline" ]]; then
        cmdline="$(tr '\0' ' ' <"/proc/$pid/cmdline")"
        [[ "$cmdline" == *"vllm"* && "$cmdline" == *"$QWEN_MODEL_DIR"* ]] || return 1
    fi
}

api_is_ready() {
    command -v "$CURL_BIN" >/dev/null 2>&1 || return 1
    "$CURL_BIN" --fail --silent --show-error --max-time 2 \
        "http://127.0.0.1:$QWEN_PORT/v1/models" >/dev/null 2>&1
}

port_is_listening() {
    command -v ss >/dev/null 2>&1 || return 1
    ss -H -ltn "sport = :$QWEN_PORT" 2>/dev/null | grep -q .
}

resolve_vllm_cmd() {
    if command -v "$VLLM_BIN" >/dev/null 2>&1; then
        VLLM_CMD=("$VLLM_BIN")
        return
    fi
    if command -v "$PYTHON_BIN" >/dev/null 2>&1 \
        && "$PYTHON_BIN" -m pip show vllm >/dev/null 2>&1; then
        VLLM_CMD=("$PYTHON_BIN" -m vllm)
        return
    fi
    die "未找到 vLLM。请先激活已安装 vLLM 的 Conda 环境，或设置 VLLM_BIN/PYTHON_BIN。"
}

validate_config() {
    [[ -d "$QWEN_MODEL_DIR" ]] || die "模型目录不存在: $QWEN_MODEL_DIR"
    is_positive_integer "$QWEN_TENSOR_PARALLEL_SIZE" \
        || die "QWEN_TENSOR_PARALLEL_SIZE 必须是正整数"
    is_positive_integer "$QWEN_MAX_MODEL_LEN" || die "QWEN_MAX_MODEL_LEN 必须是正整数"
    is_positive_integer "$QWEN_MAX_NUM_SEQS" || die "QWEN_MAX_NUM_SEQS 必须是正整数"
    is_nonnegative_integer "$GPU_FREE_HEADROOM_MIB" || die "GPU_FREE_HEADROOM_MIB 必须是非负整数"
    is_positive_integer "$STARTUP_TIMEOUT_SECONDS" || die "STARTUP_TIMEOUT_SECONDS 必须是正整数"
    [[ "$QWEN_ENFORCE_EAGER" == "0" || "$QWEN_ENFORCE_EAGER" == "1" ]] \
        || die "QWEN_ENFORCE_EAGER 只能是 0 或 1"

    QWEN_GPUS="${QWEN_GPUS//[[:space:]]/}"
    [[ "$QWEN_GPUS" =~ ^[0-9]+(,[0-9]+)*$ ]] || die "QWEN_GPUS 格式应类似 2,3"

    local gpu_list=()
    IFS=',' read -r -a gpu_list <<<"$QWEN_GPUS"
    [[ "${#gpu_list[@]}" -eq "$QWEN_TENSOR_PARALLEL_SIZE" ]] \
        || die "QWEN_GPUS 有 ${#gpu_list[@]} 张卡，但 tensor parallel size 是 $QWEN_TENSOR_PARALLEL_SIZE"

    awk -v value="$QWEN_GPU_MEMORY_UTILIZATION" \
        'BEGIN { exit !(value > 0 && value <= 1) }' \
        || die "QWEN_GPU_MEMORY_UTILIZATION 必须在 (0, 1] 范围内"
}

check_gpu_memory() {
    command -v nvidia-smi >/dev/null 2>&1 || die "未找到 nvidia-smi，无法检查 GPU"

    local gpu_list=()
    local gpu
    IFS=',' read -r -a gpu_list <<<"$QWEN_GPUS"

    echo "GPU 显存预检（单位 MiB）:"
    for gpu in "${gpu_list[@]}"; do
        local info total_mib free_mib budget_mib required_mib
        info="$(nvidia-smi --id="$gpu" \
            --query-gpu=memory.total,memory.free \
            --format=csv,noheader,nounits 2>/dev/null)" \
            || die "无法读取 GPU $gpu；请检查 GPU 编号和驱动"
        IFS=',' read -r total_mib free_mib <<<"$info"
        total_mib="${total_mib//[[:space:]]/}"
        free_mib="${free_mib//[[:space:]]/}"
        [[ "$total_mib" =~ ^[0-9]+$ && "$free_mib" =~ ^[0-9]+$ ]] \
            || die "无法解析 GPU $gpu 的显存信息: $info"

        budget_mib="$(awk -v total="$total_mib" -v ratio="$QWEN_GPU_MEMORY_UTILIZATION" \
            'BEGIN { printf "%d", total * ratio }')"
        required_mib=$((budget_mib + GPU_FREE_HEADROOM_MIB))
        printf '  GPU %s: total=%s, free=%s, vLLM预算=%s, 额外余量=%s\n' \
            "$gpu" "$total_mib" "$free_mib" "$budget_mib" "$GPU_FREE_HEADROOM_MIB"
        if (( free_mib < required_mib )); then
            die "GPU $gpu 至少需要 ${required_mib} MiB 当前空闲显存，但只有 ${free_mib} MiB。请释放显存、换卡，或谨慎调低 QWEN_GPU_MEMORY_UTILIZATION。"
        fi
    done
}

print_config() {
    cat <<EOF
Qwen chat 分卡部署配置
  GPUs              : $QWEN_GPUS
  tensor parallel   : $QWEN_TENSOR_PARALLEL_SIZE
  model             : $QWEN_MODEL_DIR
  served model name : $QWEN_NAME
  endpoint          : http://127.0.0.1:$QWEN_PORT/v1
  memory utilization: $QWEN_GPU_MEMORY_UTILIZATION（每张卡总显存占比）
  max model length  : $QWEN_MAX_MODEL_LEN
  max sequences     : $QWEN_MAX_NUM_SEQS
  enforce eager     : $QWEN_ENFORCE_EAGER
  log               : $LOG_FILE

该脚本只管理 Qwen chat，不会启动、重启或停止 embedding 服务。
EOF
}

start_service() {
    validate_config
    resolve_vllm_cmd
    command -v "$CURL_BIN" >/dev/null 2>&1 || die "未找到 curl，无法执行健康检查"

    if api_is_ready; then
        echo "Qwen chat API 已可用: http://127.0.0.1:$QWEN_PORT/v1"
        return
    fi
    if is_pid_running; then
        echo "Qwen 进程仍在运行（PID=$(<"$PID_FILE")），但 API 尚未就绪。"
        echo "请查看日志: tail -f $LOG_FILE"
        return
    fi
    rm -f "$PID_FILE"
    if port_is_listening; then
        die "端口 $QWEN_PORT 已被其他服务占用，但 /v1/models 健康检查未通过"
    fi

    check_gpu_memory
    print_config

    local eager_args=()
    if [[ "$QWEN_ENFORCE_EAGER" == "1" ]]; then
        eager_args=(--enforce-eager)
    fi

    CUDA_VISIBLE_DEVICES="$QWEN_GPUS" nohup "${VLLM_CMD[@]}" serve "$QWEN_MODEL_DIR" \
        --host "$HOST" \
        --port "$QWEN_PORT" \
        --served-model-name "$QWEN_NAME" \
        --dtype "$QWEN_DTYPE" \
        --tensor-parallel-size "$QWEN_TENSOR_PARALLEL_SIZE" \
        --gpu-memory-utilization "$QWEN_GPU_MEMORY_UTILIZATION" \
        --max-model-len "$QWEN_MAX_MODEL_LEN" \
        --max-num-seqs "$QWEN_MAX_NUM_SEQS" \
        "${eager_args[@]}" \
        >"$LOG_FILE" 2>&1 &

    local pid=$!
    printf '%s\n' "$pid" >"$PID_FILE"
    echo "Qwen chat 已提交启动，PID=$pid；等待 API 就绪……"

    local waited=0
    while (( waited < STARTUP_TIMEOUT_SECONDS )); do
        if api_is_ready; then
            echo "Qwen chat API 已就绪: http://127.0.0.1:$QWEN_PORT/v1"
            return
        fi
        if ! kill -0 "$pid" >/dev/null 2>&1; then
            echo "Qwen chat 启动失败，最近日志如下：" >&2
            tail -n 40 "$LOG_FILE" >&2 || true
            rm -f "$PID_FILE"
            exit 1
        fi
        sleep 2
        waited=$((waited + 2))
    done

    echo "等待 ${STARTUP_TIMEOUT_SECONDS}s 后 API 仍未就绪；进程仍保留，便于继续观察。" >&2
    echo "请运行: bash $0 logs" >&2
    exit 1
}

stop_service() {
    if ! is_pid_running; then
        rm -f "$PID_FILE"
        echo "未找到由项目脚本记录的 Qwen 运行进程；embedding 未受影响。"
        return
    fi

    local pid
    pid="$(<"$PID_FILE")"
    kill "$pid"
    local waited=0
    while kill -0 "$pid" >/dev/null 2>&1 && (( waited < 30 )); do
        sleep 1
        waited=$((waited + 1))
    done
    if kill -0 "$pid" >/dev/null 2>&1; then
        die "PID $pid 在 30 秒内未退出；未强制终止，请人工检查"
    fi
    rm -f "$PID_FILE"
    echo "Qwen chat 已停止（PID=$pid）；embedding 未受影响。"
}

show_status() {
    print_config
    if api_is_ready; then
        echo "状态: READY"
    elif is_pid_running; then
        echo "状态: STARTING/UNHEALTHY（PID=$(<"$PID_FILE")）"
    else
        echo "状态: STOPPED"
    fi
}

show_logs() {
    [[ -f "$LOG_FILE" ]] || die "日志文件不存在: $LOG_FILE"
    tail -n 80 -f "$LOG_FILE"
}

print_usage() {
    cat <<EOF
用法: bash $0 [start|stop|restart|status|logs|print-config]

默认按当前机器显存分布部署：
  GPU 2 + GPU 3，tensor parallel size=2，约 9.8 GiB/卡，最大上下文 32768。

常用覆盖示例：
  QWEN_GPUS=0,2 QWEN_TENSOR_PARALLEL_SIZE=2 bash $0 start
  QWEN_GPU_MEMORY_UTILIZATION=0.38 QWEN_MAX_MODEL_LEN=4096 bash $0 start

注意：多卡张量并行会把模型近似等分到每张卡；可用显存不能跨卡简单相加，
所选每张卡都必须单独通过启动前的显存检查。
EOF
}

case "$ACTION" in
    start)
        start_service
        ;;
    stop)
        stop_service
        ;;
    restart)
        stop_service
        start_service
        ;;
    status)
        show_status
        ;;
    logs)
        show_logs
        ;;
    print-config)
        print_config
        ;;
    help|-h|--help)
        print_usage
        ;;
    *)
        echo "未知动作: $ACTION" >&2
        print_usage >&2
        exit 2
        ;;
esac
