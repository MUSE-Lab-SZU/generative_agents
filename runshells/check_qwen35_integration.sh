#!/usr/bin/env bash
# Probe an already running endpoint (three short requests).
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
exec "${PYTHON_BIN:-python}" "$PROJECT_DIR/runshells/qwen_quick_probe.py" check "$@"
