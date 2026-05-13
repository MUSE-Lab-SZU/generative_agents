#!/bin/bash
# ============================================================
# 回放脚本（封装数据 + 启动回放）
# 用法: ./sim_replay.sh [sim-name]
# 示例: ./sim_replay.sh sim-test-0213-0930
# ============================================================

set -e

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

# ---------- 检查回放是否已运行 ----------
if pgrep -f "python replay_v2.py" > /dev/null; then
    echo "回放服务已在运行"
    if [ -n "$1" ]; then
        echo "请在浏览器打开: http://127.0.0.1:5051/?name=$1"
    else
        echo "请在浏览器打开: http://127.0.0.1:5051/?name=sim-xxx"
    fi
    echo "(端口号以实际输出为准)"
    exit 0
fi

# ---------- 选择仿真 ----------
if [ -n "$1" ]; then
    SIM_NAME="$1"
else
    echo "最近 10 个仿真:"
    echo "---"
    ls -d results/compressed/sim-test-* 2>/dev/null \
        | sed 's|results/compressed/||' \
        | tail -10 \
        | nl -v 0 -w 2 \
        || echo "  (无)"
    echo "---"
    echo ""
    echo -n "请输入仿真名称: "
    read SIM_NAME
fi

if [ -z "$SIM_NAME" ]; then
    echo "未指定仿真名称"
    exit 1
fi

# ---------- 激活 Conda 环境 ----------
eval "$(conda shell.bash hook)"
conda activate generative_agents_py310

# ---------- 封装数据 ----------
echo "[1/2] 封装数据: $SIM_NAME ..."
python compress.py --name "$SIM_NAME"
echo "  数据封装完成!"

# ---------- 启动回放 ----------
python replay_v2.py &
REPLAY_PID=$!
sleep 2

echo "[2/2] 回放已启动..."
echo ""
echo "请在浏览器打开: http://127.0.0.1:5051/?name=$SIM_NAME"
echo "(端口号以实际输出为准)"
echo ""

wait $REPLAY_PID
