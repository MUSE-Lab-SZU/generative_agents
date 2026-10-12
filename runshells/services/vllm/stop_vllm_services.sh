#!/usr/bin/env bash

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"

exec bash "$PROJECT_DIR/runshells/services/vllm/vllm_services.sh" stop
