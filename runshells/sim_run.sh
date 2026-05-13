#!/bin/bash
# ============================================================
# 仿真运行脚本（启动 + 汇总对话）
# 用法: ./sim_run.sh [--name NAME] [--step N] [--stride N]
# 示例: ./sim_run.sh --step 20 --stride 360
# ============================================================

set -e

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

# ---------- 默认参数 ----------
STEP=20
STRIDE=360
NAME=""
LOCAL_LLM=false

# ---------- 解析参数 ----------
while [[ $# -gt 0 ]]; do
    case $1 in
        --name)
            NAME="$2"; shift 2 ;;
        --step)
            STEP="$2"; shift 2 ;;
        --stride)
            STRIDE="$2"; shift 2 ;;
        --local-llm)
            LOCAL_LLM=true; shift ;;
        --help|-h)
            echo "用法: $0 [选项]"
            echo ""
            echo "选项:"
            echo "  --name    仿真名称 (默认自动生成, 如 sim-test-0213-0930)"
            echo "  --step    迭代步数  (默认: 20)"
            echo "  --stride  每步间隔分钟 (默认: 360)"
            echo "  --local-llm  使用本地 Ollama qwen3 替代 DeepSeek"
            echo "  --help    显示帮助"
            exit 0
            ;;
        *)
            echo "未知参数: $1"; exit 1 ;;
    esac
done

# ---------- 自动生成参数 ----------
# 仿真日期取今天，时刻统一为 09:00
START_TIME="$(date +%Y%m%d)-09:00"

if [ -z "$NAME" ]; then
    # 自动生成名称: sim-test-月日-时分
    NAME="sim-test-$(date +%m%d-%H%M)"
    echo "自动生成仿真名称: $NAME"
fi

echo "仿真起始时间: $START_TIME"


# ---------- 打印配置 ----------
echo ""
echo "=========================================="
echo " 仿真测试配置"
echo "=========================================="
echo "  环境:      $CONDA_ENV"
echo "  名称:      $NAME"
echo "  起始时间:  $START_TIME"
echo "  迭代步数:  $STEP"
echo "  步间隔:    ${STRIDE}min"
echo "=========================================="
echo ""

# ---------- Conda 环境 ----------
CONDA_ENV="generative_agents_py310"

# ---------- 启动 Ollama ----------
echo "[1/2] 检查 Ollama 服务..."
if pgrep -x "ollama" > /dev/null; then
    echo "  Ollama 已在运行"
else
    echo "  启动 Ollama 服务..."
    nohup ollama serve > /tmp/ollama.log 2>&1 &
    echo "  Ollama 已启动 (PID: $!), 日志: /tmp/ollama.log"
    sleep 2
fi

# ---------- 激活 Conda 并运行仿真 ----------
echo "[2/2] 运行仿真: $NAME ..."
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"

# ---------- forced_llm 补丁（--local-llm 时生效） ----------
CONFIG_FILE="$PROJECT_DIR/data/config.json"
CONFIG_BACKUP="$PROJECT_DIR/data/config.json.bak"
trap 'if [ -f "$CONFIG_BACKUP" ]; then mv "$CONFIG_BACKUP" "$CONFIG_FILE"; echo "  config.json 已恢复"; fi' EXIT

if [ "$LOCAL_LLM" = true ]; then
    export DEEPSEEK_API_KEY=ollama
    if command -v jq &>/dev/null; then
        cp "$CONFIG_FILE" "$CONFIG_BACKUP"
        jq '.intervention.forced_llm.provider = "ollama" |
            .intervention.forced_llm.model = "qwen3:32b" |
            .intervention.forced_llm.base_url = "http://127.0.0.1:11434/v1" |
            del(.intervention.forced_llm.api_key_env)' "$CONFIG_FILE" > "${CONFIG_FILE}.tmp" \
            && mv "${CONFIG_FILE}.tmp" "$CONFIG_FILE"
        echo "  forced_llm → Ollama qwen3:32b (本地模型)"
    else
        echo "  [警告] jq 未安装，跳过本地模型切换"
    fi
fi

python start.py \
    --name "$NAME" \
    --start "$START_TIME" \
    --step "$STEP" \
    --stride "$STRIDE"

echo "  仿真完成!"

# ---------- 恢复原始 config ----------
if [ -f "$CONFIG_BACKUP" ]; then
    mv "$CONFIG_BACKUP" "$CONFIG_FILE"
    echo "  config.json 已恢复为 DeepSeek 配置"
fi

# ---------- 汇总对话 ----------
echo ""
echo "汇总对话..."
python merge_consultation_dialogues.py "$NAME"
echo "对话汇总完成!"

# ---------- 完成 ----------
echo ""
echo "=========================================="
echo " 测试完成! 后续操作:"
echo "=========================================="
echo "  位置移动: $PROJECT_DIR/runshells/sim_status.sh $NAME"
echo "  回放模拟: $PROJECT_DIR/runshells/sim_replay.sh $NAME"
echo "" ; echo -n "请选择后续操作 (1:位置 2:回放 其他:跳过): " ; read REPLY ; if [ "$REPLY" = "1" ]; then bash "$PROJECT_DIR/runshells/sim_status.sh" "$NAME"; elif [ "$REPLY" = "2" ]; then bash "$PROJECT_DIR/runshells/sim_replay.sh" "$NAME"; else echo "跳过"; fi
echo "=========================================="
