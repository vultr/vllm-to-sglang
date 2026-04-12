#!/bin/bash
set -euo pipefail

# ============================================================
# vLLM -> SGLang shim
# This script replaces the vllm binary. The k8s production stack
# calls `vllm serve <model> [flags]`, and we intercept everything.
# ============================================================

echo ""
echo "=========================================="
echo "  vLLM -> SGLang Shim"
echo "=========================================="
echo "  Invoked as: vllm $*"
echo ""
echo "  All arguments received:"
i=1
for arg in "$@"; do
  echo "    [$i] $arg"
  i=$((i + 1))
done
echo "=========================================="
echo ""

# Defaults
HOST="0.0.0.0"
PORT="8000"

# Parse host and port from whatever the stack sends
while [[ $# -gt 0 ]]; do
  case "$1" in
    serve)        shift ;;  # skip the 'serve' subcommand
    --host)       HOST="$2"; shift 2 ;;
    --host=*)     HOST="${1#*=}"; shift ;;
    --port)       PORT="$2"; shift 2 ;;
    --port=*)     PORT="${1#*=}"; shift ;;
    *)            shift ;;  # ignore everything else
  esac
done

echo "Launching SGLang on ${HOST}:${PORT}"
echo ""

exec python -m sglang.launch_server \
  --model-path mistralai/Devstral-2-123B-Instruct-2512 \
  --host "$HOST" \
  --port "$PORT" \
  --tp 8 \
  --tool-call-parser mistral