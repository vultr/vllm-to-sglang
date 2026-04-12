#!/bin/bash
set -euo pipefail

# ============================================================
# vLLM -> SGLang shim
# This script replaces the vllm binary. The k8s production stack
# calls `vllm serve <model> [flags]`, and we intercept everything.
#
# Architecture:
#   haproxy on the vLLM port (front door)
#     /metrics → 200 empty (stub)
#     /health  → 200 if SGLang backend is up, 503 if not (instant)
#     /*       → proxy to SGLang on port+1
#   SGLang on port+1 (internal)
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

# Log to file
LOG_PATH="${VLLM_SHIM_LOG:-/tmp/vllm-shim.log}"
{
  echo "$(date -Iseconds) vLLM -> SGLang Shim (shell)"
  echo "  Invoked as: vllm $*"
  echo "  All arguments received:"
  i=1
  for arg in "$@"; do
    echo "    [$i] $arg"
    i=$((i + 1))
  done
  echo ""
} >> "$LOG_PATH"

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

# SGLang runs one port higher; haproxy binds the original port
SGLANG_PORT=$((PORT + 1))

echo "Launching SGLang on ${HOST}:${SGLANG_PORT} (internal)"
echo "Launching haproxy on ${HOST}:${PORT} (front door, /metrics + /health stub)"
echo ""

# Write haproxy config
HAPROXY_CFG="/tmp/haproxy-shim.cfg"
cat > "$HAPROXY_CFG" <<EOF
global
  log /dev/log local0
  maxconn 4096

defaults
  mode http
  timeout connect 5s
  timeout client 300s
  timeout server 300s

frontend proxy
  bind ${HOST}:${PORT}

  # /metrics stub — instant 200 empty (vLLm stack expects this)
  http-request return status 200 content-type text/plain "" if { path /metrics }

  # /health — instant response based on SGLang backend state
  # haproxy health-checks SGLang in the background; this avoids
  # the 1s k8s probe timeout racing SGLang's ~1.001s /health response
  acl sglang_up nbsrv(sglang) gt 0
  http-request return status 200 content-type text/plain "" if { path /health } sglang_up
  http-request return status 503 content-type text/plain "SGLang not ready" if { path /health }

  default_backend sglang

backend sglang
  option httpchk GET /health
  http-check expect status 200
  server s1 127.0.0.1:${SGLANG_PORT} check inter 5s fall 3 rise 2 timeout check 3s
EOF

echo "haproxy config written to ${HAPROXY_CFG}" >> "$LOG_PATH"

# Start SGLang in the background
python -m sglang.launch_server \
  --model-path mistralai/Devstral-2-123B-Instruct-2512 \
  --host "$HOST" \
  --port "$SGLANG_PORT" \
  --tp 8 \
  --tool-call-parser mistral &

SGLANG_PID=$!

# Give SGLang a moment to start before haproxy starts routing
sleep 2

# Start haproxy in the foreground (this is now PID 1 for the container)
haproxy -f "$HAPROXY_CFG" &

HAPROXY_PID=$!

echo "SGLang PID: ${SGLANG_PID}, haproxy PID: ${HAPROXY_PID}" >> "$LOG_PATH"

# Wait for whichever dies first — if either goes, we go
wait -n "$SGLANG_PID" "$HAPROXY_PID"
EXIT_CODE=$?
echo "A process exited (code ${EXIT_CODE}), shutting down" >> "$LOG_PATH"
kill "$SGLANG_PID" "$HAPROXY_PID" 2>/dev/null || true
exit $EXIT_CODE
