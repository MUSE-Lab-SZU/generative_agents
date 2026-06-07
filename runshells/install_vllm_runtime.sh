#!/usr/bin/env bash

set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
PIP_ARGS="${PIP_ARGS:-}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "未找到 Python 命令: $PYTHON_BIN" >&2
    exit 1
fi

echo "使用 Python: $("$PYTHON_BIN" -c 'import sys; print(sys.executable)')"

if [ -n "${CONDA_DEFAULT_ENV:-}" ]; then
    echo "当前 Conda 环境: $CONDA_DEFAULT_ENV"
else
    echo "警告: 当前未检测到激活的 Conda 环境。"
    echo "如果你希望安装到 llm-depression，请先执行: conda activate llm-depression"
fi

echo ""
echo "开始安装 vLLM 运行依赖..."

# PIP_ARGS is intentionally split to allow passing options like -i <mirror>.
# shellcheck disable=SC2206
extra_pip_args=($PIP_ARGS)
"$PYTHON_BIN" -m pip install --no-cache-dir -U "${extra_pip_args[@]}" vllm modelscope

echo ""
echo "安装完成，验证如下："
"$PYTHON_BIN" -m pip show vllm
echo ""
"$PYTHON_BIN" -m pip show modelscope
echo ""
echo "下一步可执行："
echo "  bash runshells/download_vllm_models.sh"
echo "  bash runshells/vllm_services.sh start"
