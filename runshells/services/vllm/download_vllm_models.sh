#!/usr/bin/env bash

set -euo pipefail

MODEL_ROOT="${MODEL_ROOT:-/share/home/tm866039793920000/a874457430/MODEL}"
MODELSCOPE_BIN="${MODELSCOPE_BIN:-modelscope}"
QWEN_REPO="${QWEN_REPO:-Qwen/Qwen3-8B}"
EMBED_REPO="${EMBED_REPO:-BAAI/bge-m3}"
QWEN_DIR="${QWEN_DIR:-$MODEL_ROOT/Qwen3-8B}"
EMBED_DIR="${EMBED_DIR:-$MODEL_ROOT/bge-m3}"

if ! command -v "$MODELSCOPE_BIN" >/dev/null 2>&1; then
    echo "未找到 $MODELSCOPE_BIN，请先安装 modelscope。" >&2
    echo "例如：pip install -U modelscope" >&2
    exit 1
fi

mkdir -p "$MODEL_ROOT"

echo "下载 Qwen 模型到: $QWEN_DIR"
mkdir -p "$QWEN_DIR"
"$MODELSCOPE_BIN" download --model "$QWEN_REPO" --local_dir "$QWEN_DIR"

echo ""
echo "下载 embedding 模型到: $EMBED_DIR"
mkdir -p "$EMBED_DIR"
"$MODELSCOPE_BIN" download --model "$EMBED_REPO" --local_dir "$EMBED_DIR"

echo ""
echo "下载完成。"
echo "Qwen repo : $QWEN_REPO"
echo "Embed repo: $EMBED_REPO"
