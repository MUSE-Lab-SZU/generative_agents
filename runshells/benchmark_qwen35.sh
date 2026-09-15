#!/usr/bin/env bash
# Sequential, same-GPU A/B test; never stops existing services.
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
exec "${PYTHON_BIN:-python}" "$PROJECT_DIR/runshells/qwen_quick_probe.py" compare "$@"
