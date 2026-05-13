#!/bin/bash
# ============================================================
# 抑郁症治疗遍历实验 — 后台运行脚本
#
# 用法:
#   ./runshells/run_experiment.sh                          # 完整遍历（后台）
#   ./runshells/run_experiment.sh --dry-run                # 干跑（前台）
#   ./runshells/run_experiment.sh -- --condition Counsel-MOD-DYN    # 单条件
#   ./runshells/run_experiment.sh --report-only            # 只生成报告
#   ./runshells/run_experiment.sh --attach                 # 接入后台会话
#   ./runshells/run_experiment.sh --status                 # 查看后台任务状态
#   ./runshells/run_experiment.sh --stop                   # 停止后台任务
#
# SSH 断开后任务继续运行（基于 tmux 会话）。
# ============================================================

set -e

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

# ---------- 常量 ----------
CONDA_ENV="generative_agents_py310"
TMUX_SESSION="experiment-traversal"
LOG_DIR="$PROJECT_DIR/results/experiment_data/logs"
LOG_FILE="$LOG_DIR/experiment_$(date +%Y%m%d_%H%M%S).log"

# ---------- 帮助 ----------
show_help() {
    echo "抑郁症治疗遍历实验"
    echo ""
    echo "用法: $0 [选项] [-- 传给 run_experiment.py 的参数]"
    echo ""
    echo "本脚本选项:"
    echo "  --attach    接入后台 tmux 会话（查看实时输出）"
    echo "  --status    查看后台任务和日志状态"
    echo "  --stop      停止后台 tmux 会话"
    echo "  --dry-run   干跑模式（前台执行，不进入 tmux）"
    echo "  --help      显示帮助"
    echo ""
    echo "传给 run_experiment.py 的参数（加在 -- 后面）:"
    echo "  --condition NAME   只跑指定条件"
    echo "  --skip-simulation  跳过模拟"
    echo "  --report-only      只生成报告"
    echo "  --dry-run          干跑（也可直接用，不加 --）"
    echo ""
    echo "示例:"
    echo "  $0                                      # 完整遍历（后台）"
    echo "  $0 --dry-run                            # 干跑（前台）"
    echo "  $0 -- --condition Counsel-MOD-DYN       # 单条件（后台）"
    echo "  $0 -- --report-only                     # 只生成报告（后台）"
    echo "  $0 --attach                             # 接入后台会话"
    echo "  $0 --status                             # 查看状态"
    echo "  $0 --stop                               # 停止后台任务"
}

# ============================================================
# 环境检查与初始化
# ============================================================

ensure_ollama() {
    echo "[环境] 检查 Ollama 服务..."
    if pgrep -x "ollama" > /dev/null 2>&1; then
        echo "  Ollama 已在运行 (PID: $(pgrep -x ollama | head -1))"
    else
        echo "  启动 Ollama 服务..."
        nohup ollama serve > /tmp/ollama.log 2>&1 &
        local PID=$!
        echo "  Ollama 已启动 (PID: $PID), 日志: /tmp/ollama.log"
        sleep 2
        if ! pgrep -x "ollama" > /dev/null 2>&1; then
            echo "  [ERROR] Ollama 启动失败，请检查 /tmp/ollama.log"
            exit 1
        fi
    fi
}

ensure_conda() {
    echo "[环境] 检查 Conda 环境: $CONDA_ENV ..."
    eval "$(conda shell.bash hook)"
    conda activate "$CONDA_ENV"
    local PY_VERSION=$(python --version 2>&1)
    echo "  Conda 环境已激活: $PY_VERSION"
}

# ============================================================
# 后台任务管理
# ============================================================

tmux_running() {
    tmux has-session -t "$TMUX_SESSION" 2>/dev/null
}

do_status() {
    echo "=========================================="
    echo " 实验任务状态"
    echo "=========================================="

    if tmux_running; then
        echo "  tmux 会话 [$TMUX_SESSION]: 运行中"
        echo "  接入命令: tmux attach -t $TMUX_SESSION"
    else
        echo "  tmux 会话 [$TMUX_SESSION]: 未运行"
    fi

    echo ""
    echo "  日志目录: $LOG_DIR"
    if [ -d "$LOG_DIR" ]; then
        local LATEST=$(ls -t "$LOG_DIR"/*.log 2>/dev/null | head -1)
        if [ -n "$LATEST" ]; then
            echo "  最新日志: $LATEST"
            echo "  查看命令: tail -f $LATEST"
            echo ""
            echo "  --- 最新日志末尾 ---"
            tail -20 "$LATEST"
            echo "  --- END ---"
        else
            echo "  (无日志文件)"
        fi
    else
        echo "  (日志目录不存在)"
    fi
}

do_stop() {
    if tmux_running; then
        echo "停止 tmux 会话 [$TMUX_SESSION]..."
        tmux kill-session -t "$TMUX_SESSION"
        echo "已停止。"
    else
        echo "tmux 会话 [$TMUX_SESSION] 未运行，无需停止。"
    fi
}

do_attach() {
    if tmux_running; then
        echo "接入 tmux 会话 [$TMUX_SESSION]... (Ctrl+B, D 退出但不停止任务)"
        tmux attach -t "$TMUX_SESSION"
    else
        echo "tmux 会话 [$TMUX_SESSION] 未运行。"
        echo "使用 $0 启动实验。"
        exit 1
    fi
}

# ============================================================
# 前台执行（dry-run 模式）
# ============================================================

run_foreground() {
    ensure_ollama
    ensure_conda

    echo ""
    echo "=========================================="
    echo " 前台执行 (dry-run)"
    echo "=========================================="
    echo "  参数: $@"
    echo ""

    python runshells/run_experiment.py "$@"
}

# ============================================================
# 后台执行（tmux 会话）
# ============================================================

run_background() {
    # 检查是否已在运行
    if tmux_running; then
        echo "[WARN] tmux 会话 [$TMUX_SESSION] 已在运行。"
        echo "  查看状态: $0 --status"
        echo "  接入会话: $0 --attach"
        echo "  停止会话: $0 --stop"
        exit 1
    fi

    ensure_ollama

    # 准备日志目录
    mkdir -p "$LOG_DIR"

    # 解析传给 Python 的参数
    local PY_ARGS="$*"

    echo ""
    echo "=========================================="
    echo " 后台执行"
    echo "=========================================="
    echo "  tmux 会话: $TMUX_SESSION"
    echo "  日志文件:  $LOG_FILE"
    echo "  Python 参数: $PY_ARGS"
    echo ""
    echo "  接入会话: $0 --attach"
    echo "  查看日志: tail -f $LOG_FILE"
    echo "  查看状态: $0 --status"
    echo "  停止任务: $0 --stop"
    echo "=========================================="

    # 创建 tmux 会话，在会话内激活 conda + 运行脚本
    tmux new-session -d -s "$TMUX_SESSION" \
        "eval \"\$(conda shell.bash hook)\" && \
         conda activate $CONDA_ENV && \
         cd $PROJECT_DIR && \
         python runshells/run_experiment.py $PY_ARGS 2>&1 | tee $LOG_FILE ; \
         echo '' ; \
         echo '========================================' ; \
         echo ' 实验已完成。按 Enter 关闭会话。' ; \
         echo '========================================' ; \
         read"

    echo ""
    echo "已启动后台 tmux 会话。SSH 断开后任务继续运行。"
}

# ============================================================
# 主入口
# ============================================================

# 分离本脚本的选项和传给 Python 的参数
SCRIPT_ARGS=()
PY_ARGS=()
SEPARATOR_SEEN=false

for arg in "$@"; do
    if [ "$SEPARATOR_SEEN" = true ]; then
        PY_ARGS+=("$arg")
    elif [ "$arg" = "--" ]; then
        SEPARATOR_SEEN=true
    else
        SCRIPT_ARGS+=("$arg")
    fi
done

# 处理脚本选项
if [ ${#SCRIPT_ARGS[@]} -eq 0 ]; then
    run_background "${PY_ARGS[@]}"
    exit 0
fi

case "${SCRIPT_ARGS[0]}" in
    --attach)
        do_attach
        ;;
    --status)
        do_status
        ;;
    --stop)
        do_stop
        ;;
    --dry-run)
        run_foreground --dry-run "${PY_ARGS[@]}"
        ;;
    --help|-h)
        show_help
        ;;
    *)
        echo "未知选项: ${SCRIPT_ARGS[0]}"
        echo "使用 --help 查看帮助"
        exit 1
        ;;
esac
