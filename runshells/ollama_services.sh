#!/usr/bin/env bash

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
CONFIG_FILE="${CONFIG_FILE:-$PROJECT_DIR/data/config.json}"
STATE_DIR="${STATE_DIR:-$PROJECT_DIR/.ollama}"
LOG_DIR="${LOG_DIR:-$STATE_DIR/logs}"
MODEL_ROOT="${MODEL_ROOT:-/share/home/tm866039793920000/a874457430/MODEL}"
OLLAMA_MODELS="${OLLAMA_MODELS:-$MODEL_ROOT/ollama}"
OLLAMA_BIN="${OLLAMA_BIN:-ollama}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CURL_BIN="${CURL_BIN:-curl}"
OLLAMA_GPUS="${OLLAMA_GPUS:-${OLLAMA_GPU:-0}}"
OLLAMA_KEEP_ALIVE="${OLLAMA_KEEP_ALIVE:--1}"
OLLAMA_MAX_LOADED_MODELS="${OLLAMA_MAX_LOADED_MODELS:-2}"
OLLAMA_NUM_PARALLEL="${OLLAMA_NUM_PARALLEL:-1}"
OLLAMA_FLASH_ATTENTION="${OLLAMA_FLASH_ATTENTION:-1}"
OLLAMA_CONTEXT_LENGTH="${OLLAMA_CONTEXT_LENGTH:-8192}"
HOST_BIND="${HOST_BIND:-0.0.0.0}"
WAIT_TIMEOUT_SECONDS="${WAIT_TIMEOUT_SECONDS:-180}"
WAIT_INTERVAL_SECONDS="${WAIT_INTERVAL_SECONDS:-2}"

ACTION="${1:-start}"

mkdir -p "$STATE_DIR" "$LOG_DIR"

PID_FILE="$STATE_DIR/ollama.pid"
LOG_FILE="$LOG_DIR/ollama.log"

require_command() {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "未找到命令: $1" >&2
        exit 1
    fi
}

read_config() {
    readarray -t CONFIG_VALUES < <("$PYTHON_BIN" - "$CONFIG_FILE" <<'PY'
import json
import sys
from urllib.parse import urlparse

path = sys.argv[1]
with open(path, "r", encoding="utf-8") as f:
    cfg = json.load(f)

llm = cfg["agent"]["think"]["llm"]
embedding = cfg["agent"]["associate"]["embedding"]

def port_from_base_url(base_url):
    parsed = urlparse(str(base_url or "http://127.0.0.1:11434"))
    return parsed.port or 11434

print(llm["model"])
print(embedding["model"])
print(port_from_base_url(llm.get("base_url") or embedding.get("base_url")))
PY
)

    CHAT_MODEL="${OLLAMA_CHAT_MODEL:-${CONFIG_VALUES[0]}}"
    EMBED_MODEL="${OLLAMA_EMBED_MODEL:-${CONFIG_VALUES[1]}}"
    OLLAMA_PORT="${OLLAMA_PORT:-${CONFIG_VALUES[2]}}"
    OLLAMA_HOST="${OLLAMA_HOST:-$HOST_BIND:$OLLAMA_PORT}"
    CLIENT_OLLAMA_HOST="127.0.0.1:$OLLAMA_PORT"
    LOCAL_BASE_URL="http://127.0.0.1:$OLLAMA_PORT"
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

is_ollama_ready() {
    "$CURL_BIN" -fsS "$LOCAL_BASE_URL/api/tags" >/dev/null 2>&1
}

wait_for_ollama() {
    local started
    local now
    local elapsed
    started="$(date +%s)"
    echo "等待 Ollama 就绪: $LOCAL_BASE_URL, timeout=${WAIT_TIMEOUT_SECONDS}s"

    while true; do
        if ! is_pid_running "$PID_FILE"; then
            echo "Ollama 进程已退出，最近日志如下：" >&2
            tail -120 "$LOG_FILE" >&2 || true
            exit 1
        fi

        if is_ollama_ready; then
            echo "Ollama 已就绪"
            return 0
        fi

        now="$(date +%s)"
        elapsed=$((now - started))
        if [ "$elapsed" -ge "$WAIT_TIMEOUT_SECONDS" ]; then
            echo "Ollama 等待超时，最近日志如下：" >&2
            tail -120 "$LOG_FILE" >&2 || true
            exit 1
        fi

        sleep "$WAIT_INTERVAL_SECONDS"
    done
}

ensure_model_available() {
    local model_name="$1"
    if ! OLLAMA_HOST="$CLIENT_OLLAMA_HOST" OLLAMA_MODELS="$OLLAMA_MODELS" "$OLLAMA_BIN" show "$model_name" >/dev/null 2>&1; then
        echo "模型未下载: $model_name" >&2
        echo "请先执行: bash runshells/download_ollama_models.sh" >&2
        exit 1
    fi
}

preload_chat_model() {
    echo "预加载 chat 模型: $CHAT_MODEL"
    "$CURL_BIN" -fsS "$LOCAL_BASE_URL/api/chat" \
        -H "Content-Type: application/json" \
        -d "{\"model\":\"$CHAT_MODEL\",\"keep_alive\":$OLLAMA_KEEP_ALIVE}" \
        >/dev/null
}

preload_embed_model() {
    echo "预加载 embedding 模型: $EMBED_MODEL"
    "$CURL_BIN" -fsS "$LOCAL_BASE_URL/api/embed" \
        -H "Content-Type: application/json" \
        -d "{\"model\":\"$EMBED_MODEL\",\"input\":\"health check\",\"keep_alive\":$OLLAMA_KEEP_ALIVE}" \
        >/dev/null
}

start_service() {
    require_command "$OLLAMA_BIN"
    require_command "$PYTHON_BIN"
    require_command "$CURL_BIN"
    read_config
    mkdir -p "$OLLAMA_MODELS"

    if is_pid_running "$PID_FILE"; then
        echo "Ollama 已在运行，PID=$(cat "$PID_FILE")"
    else
        echo "启动 Ollama"
        echo "  GPUs:       $OLLAMA_GPUS"
        echo "  Host:       $OLLAMA_HOST"
        echo "  Local URL:  $LOCAL_BASE_URL"
        echo "  Model dir:  $OLLAMA_MODELS"
        echo "  Log:        $LOG_FILE"

        CUDA_VISIBLE_DEVICES="$OLLAMA_GPUS" \
        OLLAMA_HOST="$OLLAMA_HOST" \
        OLLAMA_MODELS="$OLLAMA_MODELS" \
        OLLAMA_KEEP_ALIVE="$OLLAMA_KEEP_ALIVE" \
        OLLAMA_MAX_LOADED_MODELS="$OLLAMA_MAX_LOADED_MODELS" \
        OLLAMA_NUM_PARALLEL="$OLLAMA_NUM_PARALLEL" \
        OLLAMA_FLASH_ATTENTION="$OLLAMA_FLASH_ATTENTION" \
        OLLAMA_CONTEXT_LENGTH="$OLLAMA_CONTEXT_LENGTH" \
            nohup "$OLLAMA_BIN" serve >"$LOG_FILE" 2>&1 &

        echo $! >"$PID_FILE"
    fi

    wait_for_ollama
    ensure_model_available "$CHAT_MODEL"
    ensure_model_available "$EMBED_MODEL"
    preload_chat_model
    preload_embed_model
    print_status
}

stop_service() {
    read_config
    if is_pid_running "$PID_FILE"; then
        local pid
        pid="$(cat "$PID_FILE")"
        kill "$pid"
        rm -f "$PID_FILE"
        echo "Ollama 已停止，PID=$pid"
    else
        rm -f "$PID_FILE"
        echo "Ollama 未运行"
    fi
}

print_status() {
    read_config
    echo "Ollama 服务状态"
    echo "  Service      : $(if is_pid_running "$PID_FILE"; then echo "RUNNING pid=$(cat "$PID_FILE")"; else echo "STOPPED"; fi)"
    echo "  GPUs         : $OLLAMA_GPUS"
    echo "  URL          : $LOCAL_BASE_URL"
    echo "  OpenAI chat  : $LOCAL_BASE_URL/v1/chat/completions"
    echo "  Embedding    : $LOCAL_BASE_URL/api/embed"
    echo "  Chat model   : $CHAT_MODEL"
    echo "  Embed model  : $EMBED_MODEL"
    echo "  Model dir    : $OLLAMA_MODELS"
    echo "  Log          : $LOG_FILE"
}

print_config() {
    read_config
    cat <<EOF
当前 data/config.json 对应字段：

agent.think.llm
  provider: ollama
  model: $CHAT_MODEL
  base_url: $LOCAL_BASE_URL/v1
  api_key: EMPTY

agent.associate.embedding
  provider: ollama
  model: $EMBED_MODEL
  base_url: $LOCAL_BASE_URL
  api_key: EMPTY
EOF
}

print_usage() {
    read_config
    cat <<EOF
用法: $0 [start|stop|restart|status|print-config|help]

默认会在 config.json 的端口 $OLLAMA_PORT 启动一个 Ollama 服务，并预加载：
  Chat      -> $CHAT_MODEL
  Embedding -> $EMBED_MODEL

一张卡启动两个模型：
  OLLAMA_GPUS=0 bash runshells/ollama_services.sh start

两张卡启动两个模型：
  OLLAMA_GPUS=0,1 bash runshells/ollama_services.sh start

可覆盖环境变量：
  CONFIG_FILE, MODEL_ROOT, OLLAMA_MODELS, OLLAMA_BIN
  OLLAMA_CHAT_MODEL, OLLAMA_EMBED_MODEL, OLLAMA_PORT
  OLLAMA_GPUS, HOST_BIND, OLLAMA_KEEP_ALIVE
  OLLAMA_MAX_LOADED_MODELS, OLLAMA_NUM_PARALLEL
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
        read_config
        print_status
        ;;
    print-config)
        require_command "$PYTHON_BIN"
        print_config
        ;;
    help|--help|-h)
        require_command "$PYTHON_BIN"
        print_usage
        ;;
    *)
        echo "未知操作: $ACTION" >&2
        print_usage >&2
        exit 1
        ;;
esac
