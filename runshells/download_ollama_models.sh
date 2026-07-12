#!/usr/bin/env bash

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
CONFIG_FILE="${CONFIG_FILE:-$PROJECT_DIR/data/config.json}"
MODEL_ROOT="${MODEL_ROOT:-/share/home/tm866039793920000/a874457430/MODEL}"
OLLAMA_MODELS="${OLLAMA_MODELS:-$MODEL_ROOT/ollama}"
OLLAMA_BIN="${OLLAMA_BIN:-ollama}"
PYTHON_BIN="${PYTHON_BIN:-python}"
STATE_DIR="${STATE_DIR:-$PROJECT_DIR/.ollama}"
LOG_DIR="${LOG_DIR:-$STATE_DIR/logs}"
LOG_FILE="$LOG_DIR/download_ollama_models.log"
PID_FILE="$STATE_DIR/download_ollama_models.pid"
WAIT_TIMEOUT_SECONDS="${WAIT_TIMEOUT_SECONDS:-120}"
WAIT_INTERVAL_SECONDS="${WAIT_INTERVAL_SECONDS:-2}"
PROXY_URL="${PROXY_URL:-}"

if ! command -v "$OLLAMA_BIN" >/dev/null 2>&1; then
    echo "未找到 $OLLAMA_BIN，请先安装 Ollama。" >&2
    echo "Docker 镜像内会预装；本机可参考: curl -fsSL https://ollama.com/install.sh | sh" >&2
    exit 1
fi

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "未找到 $PYTHON_BIN，无法读取 $CONFIG_FILE。" >&2
    exit 1
fi

if ! command -v curl >/dev/null 2>&1; then
    echo "未找到 curl，无法检查 Ollama 服务。" >&2
    exit 1
fi

if [ -n "$PROXY_URL" ]; then
    export HTTP_PROXY="$PROXY_URL"
    export HTTPS_PROXY="$PROXY_URL"
    export ALL_PROXY="$PROXY_URL"
    export http_proxy="$PROXY_URL"
    export https_proxy="$PROXY_URL"
    export all_proxy="$PROXY_URL"
    echo "使用代理下载 Ollama 模型: $PROXY_URL"
fi

readarray -t CONFIG_VALUES < <("$PYTHON_BIN" - "$CONFIG_FILE" <<'PY'
import json
import sys
from urllib.parse import urlparse

path = sys.argv[1]
with open(path, "r", encoding="utf-8") as f:
    cfg = json.load(f)

llm = cfg["agent"]["think"]["llm"]
embedding = cfg["agent"]["associate"]["embedding"]

def ollama_host_from_base_url(base_url):
    parsed = urlparse(str(base_url or "http://127.0.0.1:11434"))
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 11434
    return f"{host}:{port}"

print(llm["model"])
print(embedding["model"])
print(ollama_host_from_base_url(llm.get("base_url") or embedding.get("base_url")))
PY
)

CHAT_MODEL="${OLLAMA_CHAT_MODEL:-${CONFIG_VALUES[0]}}"
EMBED_MODEL="${OLLAMA_EMBED_MODEL:-${CONFIG_VALUES[1]}}"
OLLAMA_HOST="${OLLAMA_HOST:-${CONFIG_VALUES[2]}}"
OLLAMA_HOST_WITHOUT_SCHEME="${OLLAMA_HOST#http://}"
OLLAMA_HOST_WITHOUT_SCHEME="${OLLAMA_HOST_WITHOUT_SCHEME#https://}"
OLLAMA_HOST_WITHOUT_PATH="${OLLAMA_HOST_WITHOUT_SCHEME%%/*}"
OLLAMA_PORT="${OLLAMA_HOST_WITHOUT_PATH##*:}"
if [ "$OLLAMA_PORT" = "$OLLAMA_HOST_WITHOUT_PATH" ]; then
    OLLAMA_PORT="11434"
fi
OLLAMA_BASE_URL="http://127.0.0.1:$OLLAMA_PORT"
STARTED_SERVER=0

mkdir -p "$OLLAMA_MODELS" "$STATE_DIR" "$LOG_DIR"

is_ollama_ready() {
    curl -fsS "$OLLAMA_BASE_URL/api/tags" >/dev/null 2>&1
}

wait_for_ollama() {
    local started
    local now
    local elapsed
    started="$(date +%s)"

    while true; do
        if is_ollama_ready; then
            return 0
        fi

        now="$(date +%s)"
        elapsed=$((now - started))
        if [ "$elapsed" -ge "$WAIT_TIMEOUT_SECONDS" ]; then
            echo "Ollama 服务等待超时，最近日志如下：" >&2
            tail -120 "$LOG_FILE" >&2 || true
            exit 1
        fi

        sleep "$WAIT_INTERVAL_SECONDS"
    done
}

cleanup() {
    if [ "$STARTED_SERVER" = "1" ] && [ -f "$PID_FILE" ]; then
        local pid
        pid="$(cat "$PID_FILE")"
        if [ -n "$pid" ] && kill -0 "$pid" >/dev/null 2>&1; then
            kill "$pid" >/dev/null 2>&1 || true
        fi
        rm -f "$PID_FILE"
    fi
}
trap cleanup EXIT

if is_ollama_ready; then
    echo "Ollama 已在运行: $OLLAMA_BASE_URL"
    if [ -n "$PROXY_URL" ]; then
        echo "提示：已有 Ollama 服务若不是在该代理环境下启动，服务端拉取模型可能仍不会走代理。"
    fi
else
    echo "启动临时 Ollama 服务: $OLLAMA_BASE_URL"
    OLLAMA_HOST="$OLLAMA_HOST" OLLAMA_MODELS="$OLLAMA_MODELS" nohup "$OLLAMA_BIN" serve >"$LOG_FILE" 2>&1 &
    echo $! >"$PID_FILE"
    STARTED_SERVER=1
    wait_for_ollama
fi

echo "Ollama 模型目录: $OLLAMA_MODELS"
echo "下载 chat 模型: $CHAT_MODEL"
OLLAMA_HOST="127.0.0.1:$OLLAMA_PORT" OLLAMA_MODELS="$OLLAMA_MODELS" "$OLLAMA_BIN" pull "$CHAT_MODEL"

echo ""
echo "下载 embedding 模型: $EMBED_MODEL"
OLLAMA_HOST="127.0.0.1:$OLLAMA_PORT" OLLAMA_MODELS="$OLLAMA_MODELS" "$OLLAMA_BIN" pull "$EMBED_MODEL"

echo ""
echo "下载完成。"
echo "Chat model : $CHAT_MODEL"
echo "Embed model: $EMBED_MODEL"
echo "Model dir  : $OLLAMA_MODELS"
