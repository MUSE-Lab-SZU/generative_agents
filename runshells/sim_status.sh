#!/bin/bash
# ============================================================
# Agent 位置状态查看脚本
# 用法: ./sim_status.sh [sim-name]
# 示例: ./sim_status.sh sim-test-0213-0930
# ============================================================

set -e

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

# ---------- 选择仿真 ----------
if [ -n "$1" ]; then
    SIM_NAME="$1"
else
    echo "最近 10 个仿真:"
    echo "---"
    find results/checkpoints -maxdepth 1 -mindepth 1 -type d 2>/dev/null \
        | sed 's|results/checkpoints/||' \
        | sort \
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

CHECKPOINTS="$PROJECT_DIR/results/checkpoints/$SIM_NAME"
if [ ! -d "$CHECKPOINTS" ]; then
    echo "不存在该仿真的存档: $CHECKPOINTS"
    exit 1
fi

echo ""

python3 - "$CHECKPOINTS" <<'PYEOF'
import os, sys, json

checkpoints_dir = sys.argv[1]
sim_name = os.path.basename(checkpoints_dir)

files = sorted(f for f in os.listdir(checkpoints_dir)
               if f.endswith(".json") and f != "conversation.json")

if not files:
    print(f"{sim_name} 中没有存档文件")
    sys.exit(1)

print(f"{'='*50}")
print(f" 仿真状态: {sim_name}")
print(f"{'='*50}")

for fname in files:
    with open(os.path.join(checkpoints_dir, fname), encoding="utf-8") as f:
        data = json.load(f)

    step = data["step"]
    time_str = data["time"]
    agents = data.get("agents", {})

    print(f"\nStep {step} - {time_str}")

    for agent_name, agent_data in agents.items():
        coord = agent_data.get("coord", [0, 0])
        action_event = agent_data.get("action", {}).get("event", {})
        address = action_event.get("address", [])
        describe = action_event.get("describe", "")

        # 地址去第一级("the Ville")
        location = "，".join(address[1:]) if len(address) > 1 else str(address)
        action = describe if describe else "睡觉"

        print(f"  {agent_name:<10}  {location:<20} → {action}")

print(f"\n{'='*50}")
print(f" 共 {len(files)} 个存档")
print(f"{'='*50}")
PYEOF
