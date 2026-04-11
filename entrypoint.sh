#!/bin/bash
set -euo pipefail

# Defaults matching vLLM production stack defaults
HOST="0.0.0.0"
PORT="8000"

# Save original args before parsing eats them
ALL_ARGS="$*"

# Parse only host and port from whatever args the vLLM stack sends.
# Everything else is ignored.
while [[ $# -gt 0 ]]; do
  case "$1" in
    --host)       HOST="$2"; shift 2 ;;
    --host=*)     HOST="${1#*=}"; shift ;;
    --port)       PORT="$2"; shift 2 ;;
    --port=*)     PORT="${1#*=}"; shift ;;
    *)            shift ;;  # ignore everything else
  esac
done

echo "=== vLLM production stack args received ==="
echo "Raw args: $ALL_ARGS"
echo ""
i=1
for arg in $ALL_ARGS; do
  echo "  [$i] $arg"
  i=$((i + 1))
done
echo "============================================"
echo ""
echo "=== SGLang shim ==="
echo "Ignoring vLLM args. Launching SGLang on ${HOST}:${PORT}"
echo "==================="

exec python -m sglang.launch_server \
  --model-path mistralai/Devstral-2-123B-Instruct-2512 \
  --host "$HOST" \
  --port "$PORT" \
  --tp 8 \
  --tool-call-parser mistral