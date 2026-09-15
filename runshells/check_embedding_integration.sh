#!/usr/bin/env bash
# Optional: six tiny batches against three already running embedding replicas.
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
exec "${PYTHON_BIN:-python}" "$PROJECT_DIR/runshells/qwen_quick_probe.py" embedding "$@"
