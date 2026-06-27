#!/usr/bin/env bash

set -euo pipefail

SOURCE_DIR="${1:-${PROJECT_SYNC_SRC:-}}"
TARGET_DIR="${2:-${PROJECT_SYNC_DST:-${PROJECT_DIR:-/workspace/project}}}"
PROJECT_SYNC_DELETE="${PROJECT_SYNC_DELETE:-0}"
PROJECT_SYNC_DRY_RUN="${PROJECT_SYNC_DRY_RUN:-0}"
PROJECT_SYNC_INCLUDE_ENV="${PROJECT_SYNC_INCLUDE_ENV:-0}"

print_usage() {
    cat <<EOF
用法:
  bash runshells/sync_project_from_disk.sh <数据盘工作区路径> [容器项目路径]

示例:
  bash runshells/sync_project_from_disk.sh /mnt/data/generative_agents-2
  PROJECT_SYNC_DRY_RUN=1 bash runshells/sync_project_from_disk.sh /mnt/data/generative_agents-2
  PROJECT_SYNC_DELETE=1 bash runshells/sync_project_from_disk.sh /mnt/data/generative_agents-2

默认目标路径:
  ${TARGET_DIR}

默认只覆盖源目录中存在且有更新的文件，不会删除目标目录中的其它文件。
如确实要让目标目录和源目录严格一致:
  PROJECT_SYNC_DELETE=1 bash runshells/sync_project_from_disk.sh /mnt/data/generative_agents-2

默认不会同步:
  .env, results/, docs/, plans/, 模型目录, 缓存, 日志, Python cache

如确实要同步 .env:
  PROJECT_SYNC_INCLUDE_ENV=1 bash runshells/sync_project_from_disk.sh <数据盘工作区路径>
EOF
}

if [ -z "$SOURCE_DIR" ]; then
    print_usage >&2
    exit 2
fi

if [ "$SOURCE_DIR" = "-h" ] || [ "$SOURCE_DIR" = "--help" ]; then
    print_usage
    exit 0
fi

if [ ! -d "$SOURCE_DIR" ]; then
    echo "源目录不存在: $SOURCE_DIR" >&2
    exit 1
fi

if ! command -v rsync >/dev/null 2>&1; then
    echo "未找到 rsync。请先安装 rsync，或使用包含 rsync 的新版镜像。" >&2
    exit 1
fi

mkdir -p "$TARGET_DIR"

rsync_args=(
    -a
    --info=stats2,progress2
    --human-readable
    --exclude=/.git/
    --exclude=/.agents/
    --exclude=/.codex/
    --exclude=/.claude/
    --exclude=/.vscode/
    --exclude=/docs/
    --exclude=/plans/
    --exclude=/result/
    --exclude=/results/
    --exclude=/checkpoint/
    --exclude=/checkpoints/
    --exclude=/.vllm/
    --exclude=/.pytest_cache/
    --exclude=__pycache__/
    --exclude=*.pyc
    --exclude=*.pyo
    --exclude=*.log
    --exclude=*.pid
    --exclude=*.tmp
    --exclude=*.bak
    --exclude=/models/
    --exclude=/MODEL/
    --exclude=/model/
    --exclude=/hf-cache/
    --exclude=/modelscope-cache/
    --exclude=/.cache/
)

if [ "$PROJECT_SYNC_DELETE" = "1" ]; then
    rsync_args+=(--delete)
fi

if [ "$PROJECT_SYNC_DRY_RUN" = "1" ]; then
    rsync_args+=(--dry-run)
fi

if [ "$PROJECT_SYNC_INCLUDE_ENV" != "1" ]; then
    rsync_args+=(--exclude=/.env)
fi

echo "同步项目工作区"
echo "  From: $SOURCE_DIR/"
echo "  To:   $TARGET_DIR/"
echo "  Delete missing files: $PROJECT_SYNC_DELETE"
echo "  Dry run: $PROJECT_SYNC_DRY_RUN"
echo "  Include .env: $PROJECT_SYNC_INCLUDE_ENV"

rsync "${rsync_args[@]}" "$SOURCE_DIR"/ "$TARGET_DIR"/

if [ "$PROJECT_SYNC_DRY_RUN" != "1" ] && [ -d "$TARGET_DIR/runshells" ]; then
    chmod +x "$TARGET_DIR"/runshells/*.sh 2>/dev/null || true
fi

echo "同步完成。"
