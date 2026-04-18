#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# vLLM -> SGLang shim (shell version)
# This script replaces the vllm binary. The k8s production stack
# calls `vllm serve <model> [flags]`, and we intercept everything.
#
# Dynamically translates vLLM args to SGLang equivalents.
# No hardcoded model or tensor-parallel size.
#
# Architecture:
#   haproxy on the vLLM port (front door)
#     /metrics → 200 empty (stub)
#     /health  → 200 if SGLang backend is up, 503 if not
#     /*       → proxy to middleware on port+2
#   middleware on port+2 (strips vLLM-only params, fixes schemas)
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

# ── Parse vLLM args → extract model, host, port, translate the rest ──

MODEL=""
HOST="0.0.0.0"
PORT="8000"
SGLANG_ARGS=()
SKIPPED_ARGS=()

# Default tool-call-parser; override with SGLANG_TOOL_CALL_PARSER env var
TOOL_CALL_PARSER="${SGLANG_TOOL_CALL_PARSER:-mistral}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    # Skip 'serve' subcommand
    serve)        shift ;;

    # ── Extracted for infrastructure (not passed to SGLang) ──
    --host)       HOST="$2"; shift 2 ;;
    --host=*)     HOST="${1#*=}"; shift ;;
    --port)       PORT="$2"; shift 2 ;;
    --port=*)     PORT="${1#*=}"; shift ;;

    # ── Positional model name ──
    --model|--model-name)
      MODEL="$2"; shift 2 ;;
    --model=*|--model-name=*)
      MODEL="${1#*=}"; shift ;;

    # ── Direct renames (vLLM → SGLang) ──
    --tensor-parallel-size)
      SGLANG_ARGS+=("--tp" "$2"); shift 2 ;;
    --tensor-parallel-size=*)
      SGLANG_ARGS+=("--tp" "${1#*=}"); shift ;;
    --gpu_memory_utilization)
      SGLANG_ARGS+=("--mem-fraction-static" "$2"); shift 2 ;;
    --gpu_memory_utilization=*)
      SGLANG_ARGS+=("--mem-fraction-static" "${1#*=}"); shift ;;
    --trust_remote_code|--trust-remote-code)
      SGLANG_ARGS+=("--trust-remote-code"); shift ;;

    # ── vLLM flags with no SGLang equivalent → skip ──
    --no-enable-prefix-caching|--enable-prefix-caching)
      SKIPPED_ARGS+=("$1"); shift ;;
    --enable-chunked-prefill|--no-enable-chunked-prefill)
      SKIPPED_ARGS+=("$1"); shift ;;
    --disable-log-requests|--disable-log-stats)
      SKIPPED_ARGS+=("$1"); shift ;;
    --swap-space|--block-size|--max-num-seqs|--max-num-batched-tokens)
      SKIPPED_ARGS+=("$1" "$2"); shift 2 ;;
    --swap-space=*|--block-size=*|--max-num-seqs=*|--max-num-batched-tokens=*)
      SKIPPED_ARGS+=("$1"); shift ;;
    --distributed-executor-backend|--pipeline-parallel-size|--data-parallel-size)
      SKIPPED_ARGS+=("$1" "$2"); shift 2 ;;
    --quantization|--dtype|--revision|--tokenizer-revision|--tokenizer-mode)
      SKIPPED_ARGS+=("$1" "$2"); shift 2 ;;
    --quantization=*|--dtype=*|--revision=*|--tokenizer-revision=*|--tokenizer-mode=*)
      SKIPPED_ARGS+=("$1"); shift ;;

    # ── Pass through to SGLang as-is ──
    --tool-call-parser)
      TOOL_CALL_PARSER="$2"; shift 2 ;;
    --tool-call-parser=*)
      TOOL_CALL_PARSER="${1#*=}"; shift ;;
    *)
      # Positional arg = model name (first non-flag)
      if [[ ! "$1" =~ ^- ]] && [[ -z "$MODEL" ]]; then
        MODEL="$1"; shift
      else
        # Unknown — pass through, might be valid for SGLang
        SGLANG_ARGS+=("$1"); shift
      fi ;;
  esac
done

if [[ -z "$MODEL" ]]; then
  echo "ERROR: No model specified in vLLM args!"
  exit 1
fi

# ── Port scheme: haproxy=original, SGLang=+1, middleware=+2 ──
SGLANG_PORT=$((PORT + 1))
MIDDLEWARE_PORT=$((PORT + 2))

echo "Model: ${MODEL}"
echo "SGLang:  ${HOST}:${SGLANG_PORT}"
echo "Middleware: ${HOST}:${MIDDLEWARE_PORT}"
echo "haproxy: ${HOST}:${PORT}"
if [[ ${#SGLANG_ARGS[@]} -gt 0 ]]; then
  echo "Translated args: ${SGLANG_ARGS[*]}"
fi
if [[ ${#SKIPPED_ARGS[@]} -gt 0 ]]; then
  echo "Skipped (no SGLang equivalent): ${SKIPPED_ARGS[*]}"
fi
echo ""

# ── haproxy setup ───────────────────────────────────────────

mkdir -p /tmp/haproxy-errors
printf "HTTP/1.0 503 Service Unavailable\r\nContent-Length: 16\r\nConnection: close\r\nContent-Type: text/plain\r\n\r\nSGLang not ready" > /tmp/haproxy-errors/503-sglang.http

HAPROXY_CFG="/tmp/haproxy-shim.cfg"
cat > "$HAPROXY_CFG" <<EOF
global
  maxconn 4096

defaults
  mode http
  timeout connect 5s
  timeout client 300s
  timeout server 300s

frontend proxy
  bind ${HOST}:${PORT}

  acl is_health path /health
  acl sglang_up nbsrv(sglang) gt 0
  http-request deny deny_status 200 if is_health sglang_up
  http-request deny deny_status 503 if is_health
  errorfile 503 /tmp/haproxy-errors/503-sglang.http

  default_backend sglang

backend sglang
  option httpchk GET /health
  http-check expect status 200
  server s1 127.0.0.1:${MIDDLEWARE_PORT} check inter 5s fall 3 rise 2
EOF

# ── Build and launch SGLang ─────────────────────────────────

SGLANG_CMD=(
  python -m sglang.launch_server
  --model-path "$MODEL"
  --host "$HOST"
  --port "$SGLANG_PORT"
  --enable-metrics
)
if [[ -n "$TOOL_CALL_PARSER" ]]; then
  SGLANG_CMD+=(--tool-call-parser "$TOOL_CALL_PARSER")
fi
SGLANG_CMD+=("${SGLANG_ARGS[@]}")

echo "SGLang command: ${SGLANG_CMD[*]}"
echo ""

{
  echo "haproxy config written to ${HAPROXY_CFG}"
  echo "Model: ${MODEL}, SGLang port: ${SGLANG_PORT}, middleware port: ${MIDDLEWARE_PORT}, haproxy port: ${PORT}"
  echo "SGLang command: ${SGLANG_CMD[*]}"
  if [[ ${#SKIPPED_ARGS[@]} -gt 0 ]]; then
    echo "Skipped vLLM args: ${SKIPPED_ARGS[*]}"
  fi
} >> "$LOG_PATH"

# Launch SGLang
"${SGLANG_CMD[@]}" &
SGLANG_PID=$!

# Launch middleware
SGLANG_HOST="$HOST" SGLANG_PORT="$SGLANG_PORT" MIDDLEWARE_PORT="$MIDDLEWARE_PORT" \
  python /opt/vllm-shim/vllm_middleware.py &
MIDDLEWARE_PID=$!

sleep 2

# Launch haproxy (front door on the original port)
haproxy -f "$HAPROXY_CFG" &
HAPROXY_PID=$!

echo "SGLang PID: ${SGLANG_PID}, middleware PID: ${MIDDLEWARE_PID}, haproxy PID: ${HAPROXY_PID}" >> "$LOG_PATH"

# Wait for whichever dies first
wait -n "$SGLANG_PID" "$MIDDLEWARE_PID" "$HAPROXY_PID"
EXIT_CODE=$?
echo "A process exited (code ${EXIT_CODE}), shutting down" >> "$LOG_PATH"
kill "$SGLANG_PID" "$MIDDLEWARE_PID" "$HAPROXY_PID" 2>/dev/null || true
exit $EXIT_CODE
