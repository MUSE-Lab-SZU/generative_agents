#!/usr/bin/env bash

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
STATE_DIR="${STATE_DIR:-$PROJECT_DIR/.clash}"
CLASH_DIR="${CLASH_DIR:-$STATE_DIR}"
LOG_DIR="${LOG_DIR:-$STATE_DIR/logs}"
CLASH_BIN="${CLASH_BIN:-mihomo}"
CLASH_CONFIG_URL="${CLASH_CONFIG_URL:-${CLASH_SUBSCRIBE_URL:-}}"
CLASH_CONFIG_FILE="${CLASH_CONFIG_FILE:-$CLASH_DIR/config.yaml}"
CLASH_HTTP_PROXY="${CLASH_HTTP_PROXY:-http://127.0.0.1:7890}"
CLASH_BOOTSTRAP_PROXY="${CLASH_BOOTSTRAP_PROXY:-}"
UPDATE_CONFIG="${UPDATE_CONFIG:-0}"
WAIT_TIMEOUT_SECONDS="${WAIT_TIMEOUT_SECONDS:-30}"
WAIT_INTERVAL_SECONDS="${WAIT_INTERVAL_SECONDS:-1}"
ACTION="${1:-start}"

PID_FILE="$STATE_DIR/clash.pid"
LOG_FILE="$LOG_DIR/clash.log"

mkdir -p "$STATE_DIR" "$CLASH_DIR" "$LOG_DIR"

require_command() {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "未找到命令: $1" >&2
        echo "Docker 构建时可加: --build-arg INSTALL_CLASH=1" >&2
        exit 1
    fi
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

download_config() {
    if [ -z "$CLASH_CONFIG_URL" ] && [ ! -f "$CLASH_CONFIG_FILE" ]; then
        echo "缺少 Clash/Mihomo 配置。" >&2
        echo "请设置 CLASH_CONFIG_URL/CLASH_SUBSCRIBE_URL，或把配置放到: $CLASH_CONFIG_FILE" >&2
        exit 1
    fi

    if [ -n "$CLASH_CONFIG_URL" ] && { [ "$UPDATE_CONFIG" = "1" ] || [ ! -f "$CLASH_CONFIG_FILE" ]; }; then
        echo "下载 Clash/Mihomo 配置到: $CLASH_CONFIG_FILE"
        if [ -n "$CLASH_BOOTSTRAP_PROXY" ]; then
            HTTPS_PROXY="$CLASH_BOOTSTRAP_PROXY" HTTP_PROXY="$CLASH_BOOTSTRAP_PROXY" ALL_PROXY="$CLASH_BOOTSTRAP_PROXY" \
                curl -fL --retry 3 --connect-timeout 30 "$CLASH_CONFIG_URL" -o "$CLASH_CONFIG_FILE"
        else
            curl -fL --retry 3 --connect-timeout 30 "$CLASH_CONFIG_URL" -o "$CLASH_CONFIG_FILE"
        fi
    fi
}

wait_for_proxy() {
    local started
    local now
    local elapsed
    started="$(date +%s)"

    while true; do
        if ! is_pid_running "$PID_FILE"; then
            echo "Clash/Mihomo 进程已退出，最近日志如下：" >&2
            tail -120 "$LOG_FILE" >&2 || true
            exit 1
        fi

        if curl -fsS --proxy "$CLASH_HTTP_PROXY" --connect-timeout 5 https://www.gstatic.com/generate_204 >/dev/null 2>&1; then
            echo "Clash/Mihomo 代理已就绪: $CLASH_HTTP_PROXY"
            return 0
        fi

        now="$(date +%s)"
        elapsed=$((now - started))
        if [ "$elapsed" -ge "$WAIT_TIMEOUT_SECONDS" ]; then
            echo "Clash/Mihomo 等待超时，最近日志如下：" >&2
            tail -120 "$LOG_FILE" >&2 || true
            exit 1
        fi

        sleep "$WAIT_INTERVAL_SECONDS"
    done
}

start_service() {
    require_command "$CLASH_BIN"
    require_command curl
    download_config

    if is_pid_running "$PID_FILE"; then
        echo "Clash/Mihomo 已在运行，PID=$(cat "$PID_FILE")"
    else
        echo "启动 Clash/Mihomo"
        echo "  Bin:    $CLASH_BIN"
        echo "  Dir:    $CLASH_DIR"
        echo "  Config: $CLASH_CONFIG_FILE"
        echo "  Log:    $LOG_FILE"
        nohup "$CLASH_BIN" -d "$CLASH_DIR" -f "$CLASH_CONFIG_FILE" >"$LOG_FILE" 2>&1 &
        echo $! >"$PID_FILE"
    fi

    wait_for_proxy
    print_env
}

stop_service() {
    if is_pid_running "$PID_FILE"; then
        local pid
        pid="$(cat "$PID_FILE")"
        kill "$pid"
        rm -f "$PID_FILE"
        echo "Clash/Mihomo 已停止，PID=$pid"
    else
        rm -f "$PID_FILE"
        echo "Clash/Mihomo 未运行"
    fi
}

print_status() {
    echo "Clash/Mihomo 服务状态"
    echo "  Service: $(if is_pid_running "$PID_FILE"; then echo "RUNNING pid=$(cat "$PID_FILE")"; else echo "STOPPED"; fi)"
    echo "  Proxy  : $CLASH_HTTP_PROXY"
    echo "  Config : $CLASH_CONFIG_FILE"
    echo "  Log    : $LOG_FILE"
}

print_env() {
    cat <<EOF
可用于当前 shell 的代理变量：
  export HTTP_PROXY=$CLASH_HTTP_PROXY
  export HTTPS_PROXY=$CLASH_HTTP_PROXY
  export ALL_PROXY=$CLASH_HTTP_PROXY
  export PROXY_URL=$CLASH_HTTP_PROXY
EOF
}

download_ollama_models() {
    start_service
    PROXY_URL="$CLASH_HTTP_PROXY" bash "$PROJECT_DIR/runshells/download_ollama_models.sh"
}

print_usage() {
    cat <<EOF
用法: $0 [start|stop|restart|status|env|download-ollama|help]

订阅 URL 运行示例：
  CLASH_CONFIG_URL='你的订阅URL' bash runshells/clash_services.sh start

通过 Clash/Mihomo 下载 config.json 里的 Ollama 模型：
  CLASH_CONFIG_URL='你的订阅URL' bash runshells/clash_services.sh download-ollama

可覆盖环境变量：
  CLASH_BIN, CLASH_DIR, CLASH_CONFIG_URL, CLASH_SUBSCRIBE_URL
  CLASH_CONFIG_FILE, CLASH_HTTP_PROXY, CLASH_BOOTSTRAP_PROXY
  UPDATE_CONFIG, WAIT_TIMEOUT_SECONDS
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
        print_status
        ;;
    env)
        print_env
        ;;
    download-ollama)
        download_ollama_models
        ;;
    help|--help|-h)
        print_usage
        ;;
    *)
        echo "未知操作: $ACTION" >&2
        print_usage >&2
        exit 1
        ;;
esac
