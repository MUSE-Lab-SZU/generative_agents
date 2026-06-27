#!/usr/bin/env bash

set -euo pipefail

DISK_PROJECT_DIR="${1:-${RESULTS_SYNC_DST_PROJECT:-}}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CONTAINER_PROJECT_DIR="${2:-${RESULTS_SYNC_SRC_PROJECT:-${PROJECT_DIR:-$DEFAULT_PROJECT_DIR}}}"
RESULTS_SYNC_INTERVAL="${RESULTS_SYNC_INTERVAL:-60}"
RESULTS_SYNC_DELETE="${RESULTS_SYNC_DELETE:-0}"
RESULTS_SYNC_DRY_RUN="${RESULTS_SYNC_DRY_RUN:-0}"
RESULTS_SYNC_ONCE="${RESULTS_SYNC_ONCE:-0}"

print_usage() {
    cat <<EOF
用法:
  bash runshells/sync_results_to_disk_loop.sh <数据盘工作区路径> [容器项目路径]

示例:
  bash runshells/sync_results_to_disk_loop.sh /mnt/data/generative_agents-2
  RESULTS_SYNC_INTERVAL=30 bash runshells/sync_results_to_disk_loop.sh /mnt/data/generative_agents-2
  RESULTS_SYNC_ONCE=1 bash runshells/sync_results_to_disk_loop.sh /mnt/data/generative_agents-2
  RESULTS_SYNC_DRY_RUN=1 bash runshells/sync_results_to_disk_loop.sh /mnt/data/generative_agents-2

同步方向:
  From: [容器项目路径]/results/
  To:   <数据盘工作区路径>/results/

默认每 ${RESULTS_SYNC_INTERVAL} 秒同步一次，只覆盖更新过的结果文件，不会删除数据盘已有文件。
如确实要让数据盘 results 和容器 results 严格一致:
  RESULTS_SYNC_DELETE=1 bash runshells/sync_results_to_disk_loop.sh <数据盘工作区路径>
EOF
}

if [ -z "$DISK_PROJECT_DIR" ]; then
    print_usage >&2
    exit 2
fi

if [ "$DISK_PROJECT_DIR" = "-h" ] || [ "$DISK_PROJECT_DIR" = "--help" ]; then
    print_usage
    exit 0
fi

SOURCE_RESULTS_DIR="$CONTAINER_PROJECT_DIR/results"
TARGET_RESULTS_DIR="$DISK_PROJECT_DIR/results"

if [ ! -d "$CONTAINER_PROJECT_DIR" ]; then
    echo "容器项目目录不存在: $CONTAINER_PROJECT_DIR" >&2
    exit 1
fi

if [ ! -d "$SOURCE_RESULTS_DIR" ]; then
    echo "容器 results 目录不存在: $SOURCE_RESULTS_DIR" >&2
    exit 1
fi

if ! command -v rsync >/dev/null 2>&1; then
    echo "未找到 rsync。请先安装 rsync，或使用包含 rsync 的新版镜像。" >&2
    exit 1
fi

mkdir -p "$TARGET_RESULTS_DIR"

rsync_args=(
    -a
    --partial
    --human-readable
    --info=stats2
    --exclude=__pycache__/
    --exclude=*.pyc
    --exclude=*.pyo
    --exclude=*.pid
    --exclude=*.tmp
    --exclude=*.bak
)

if [ "$RESULTS_SYNC_DELETE" = "1" ]; then
    rsync_args+=(--delete)
fi

if [ "$RESULTS_SYNC_DRY_RUN" = "1" ]; then
    rsync_args+=(--dry-run)
fi

sync_once() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 同步 results"
    echo "  From: $SOURCE_RESULTS_DIR/"
    echo "  To:   $TARGET_RESULTS_DIR/"
    echo "  Delete missing files: $RESULTS_SYNC_DELETE"
    echo "  Dry run: $RESULTS_SYNC_DRY_RUN"
    rsync "${rsync_args[@]}" "$SOURCE_RESULTS_DIR"/ "$TARGET_RESULTS_DIR"/
}

if [ "$RESULTS_SYNC_ONCE" = "1" ]; then
    sync_once
    echo "同步完成。"
    exit 0
fi

echo "开始持续同步 results。按 Ctrl+C 停止。"
echo "同步间隔: ${RESULTS_SYNC_INTERVAL}s"

while true; do
    sync_once
    sleep "$RESULTS_SYNC_INTERVAL"
done
