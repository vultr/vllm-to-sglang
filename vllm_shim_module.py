#!/usr/bin/env python3
"""
vLLM -> SGLang Python shim.
Catches `python -m vllm.entrypoints.openai.api_server` (and similar)
and launches SGLang behind haproxy + middleware instead.

Dynamically translates vLLM CLI args to SGLang equivalents.
No hardcoded model name or tensor-parallel size.

Architecture:
  haproxy on the vLLM port (front door)
    /health  → 200 if SGLang backend is up, 503 if not (instant)
    /*       → proxy to middleware on port+2 (including /metrics)
  middleware on port+2 (strips vLLM-only params, fixes tool schemas, translates metrics)
  SGLang on port+1 (internal, also serves /metrics when --enable-metrics is passed)
"""

import os
import sys
import subprocess
import time
import datetime

# ── vLLM → SGLang argument mapping ──────────────────────────
# Key = vLLM flag, value = (sglang_flag, has_value)
# has_value=True means the flag takes an argument (e.g. --port 8000)
# has_value=False means it's a boolean flag (e.g. --no-enable-prefix-caching)
ARG_MAP = {
    # Direct renames (vLLM name → SGLang name)
    "--tensor-parallel-size": ("--tp", True),
    "--gpu_memory_utilization": ("--mem-fraction-static", True),
    "--max_model_len": ("--max-running-requests", True),  # approximate
    "--max-model-len": ("--max-running-requests", True),  # kebab variant
    "--enforce_eager": (
        "--enable-torch-compile",
        False,
    ),  # opposite intent, skip by default
    "--trust_remote_code": ("--trust-remote-code", False),
    "--trust-remote-code": ("--trust-remote-code", False),
    # vLLM flags with no SGLang equivalent → skip
    "--no-enable-prefix-caching": (None, False),
    "--enable-prefix-caching": (None, False),
    "--enable-chunked-prefill": (None, False),
    "--no-enable-chunked-prefill": (None, False),
    "--disable-log-requests": (None, False),
    "--disable-log-stats": (None, False),
    "--swap-space": (None, True),
    "--block-size": (None, True),
    "--num-gpu-blocks-override": (None, True),
    "--num-cpu-blocks-override": (None, True),
    "--max-num-seqs": (None, True),
    "--max-num-batched-tokens": (None, True),
    "--distributed-executor-backend": (None, True),
    "--pipeline-parallel-size": (None, True),
    "--data-parallel-size": (None, True),
    "--revision": (None, True),
    "--code-revision": (None, True),
    "--tokenizer-revision": (None, True),
    "--tokenizer-mode": (None, True),
    "--quantization": (None, True),
    "--dtype": (None, True),
    "--max-seq-len-to-capture": (None, True),
    "--enable-lora": (None, False),
    "--max-lora-rank": (None, True),
    "--max-cpu-loras": (None, True),
    "--lora-dtype": (None, True),
    "--enable-prompt-adapter": (None, False),
    "--scheduler-delay-factor": (None, True),
    "--enable-multi-modal": (None, False),
    "--limit-mm-per-prompt": (None, True),
}

# Default tool-call-parser; override with SGLANG_TOOL_CALL_PARSER env var
DEFAULT_TOOL_CALL_PARSER = "qwen3_coder"


def parse_vllm_args(args):
    """
    Parse vLLM CLI args and extract model, host, port,
    plus any args we should translate to SGLang.
    Returns (model, host, port, sglang_extra_args, skipped_args).
    """
    model = None
    host = "0.0.0.0"
    port = "8000"
    sglang_extra = []  # translated args for SGLang
    skipped = []  # vLLM args we're ignoring

    i = 0
    while i < len(args):
        arg = args[i]

        # 'serve' subcommand — skip
        if arg == "serve":
            i += 1
            continue

        # Positional model argument (first non-flag after serve, or standalone)
        if not arg.startswith("-") and model is None:
            model = arg
            i += 1
            continue

        # --flag=value form
        if "=" in arg and arg.startswith("--"):
            flag, val = arg.split("=", 1)
            if flag == "--host":
                host = val
            elif flag == "--port":
                port = val
            elif flag in ARG_MAP:
                sglang_flag, has_val = ARG_MAP[flag]
                if sglang_flag is None:
                    skipped.append(arg)
                elif has_val:
                    sglang_extra.extend([sglang_flag, val])
                else:
                    # boolean flag with =value (unusual but valid)
                    sglang_extra.append(sglang_flag)
            else:
                # Unknown flag — pass through as-is (might be a SGLang flag too)
                sglang_extra.append(arg)
            i += 1
            continue

        # --flag value form
        if arg in ("--host",):
            if i + 1 < len(args):
                host = args[i + 1]
            i += 2
            continue
        if arg in ("--port",):
            if i + 1 < len(args):
                port = args[i + 1]
            i += 2
            continue

        if arg in ARG_MAP:
            sglang_flag, has_val = ARG_MAP[arg]
            if sglang_flag is None:
                skipped.append(arg)
                if has_val and i + 1 < len(args) and not args[i + 1].startswith("-"):
                    skipped.append(args[i + 1])
                    i += 2
                else:
                    i += 1
            elif has_val:
                if i + 1 < len(args):
                    sglang_extra.extend([sglang_flag, args[i + 1]])
                    i += 2
                else:
                    i += 1
            else:
                sglang_extra.append(sglang_flag)
                i += 1
            continue

        # --tool-call-parser — pass through to SGLang
        if arg == "--tool-call-parser":
            if i + 1 < len(args):
                sglang_extra.extend(["--tool-call-parser", args[i + 1]])
                i += 2
            else:
                i += 1
            continue

        # Unknown flag — pass through if it takes a value, might be valid for SGLang
        if (
            arg.startswith("--")
            and i + 1 < len(args)
            and not args[i + 1].startswith("-")
        ):
            sglang_extra.extend([arg, args[i + 1]])
            i += 2
        elif arg.startswith("--"):
            sglang_extra.append(arg)
            i += 1
        else:
            # Unknown positional — probably the model if we don't have it yet
            if model is None:
                model = arg
            i += 1

    return model, host, port, sglang_extra, skipped


def main():
    args = sys.argv[1:]

    log_path = os.environ.get("VLLM_SHIM_LOG", "/tmp/vllm-shim.log")
    with open(log_path, "a") as f:
        f.write(
            f"\n{datetime.datetime.now().isoformat()} vLLM -> SGLang Shim (Python module)\n"
        )
        f.write(f"  Invoked as: python -m {__name__} {' '.join(args)}\n")
        f.write("  All arguments received:\n")
        for i, arg in enumerate(args, 1):
            f.write(f"    [{i}] {arg}\n")
        f.write("\n")

    print()
    print("==========================================")
    print("  vLLM -> SGLang Shim (Python module)")
    print("==========================================")
    print(f"  Invoked as: python -m {__name__} {' '.join(args)}")
    print()
    print("  All arguments received:")
    for i, arg in enumerate(args, 1):
        print(f"    [{i}] {arg}")
    print("==========================================")
    print()

    model, host, port, sglang_extra, skipped = parse_vllm_args(args)

    if not model:
        print("ERROR: No model specified in vLLM args!")
        os._exit(1)

    # SGLang port scheme: original+1 = SGLang, original+2 = middleware
    sglang_port = str(int(port) + 1)
    middleware_port = str(int(port) + 2)

    # Build SGLang command
    sglang_cmd = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        model,
        "--host",
        host,
        "--port",
        sglang_port,
        "--enable-metrics",
    ]

    # Add tool-call-parser (env override or default)
    tcp = os.environ.get("SGLANG_TOOL_CALL_PARSER", DEFAULT_TOOL_CALL_PARSER)
    if tcp:
        sglang_cmd.extend(["--tool-call-parser", tcp])

    # Add translated/forwarded args
    sglang_cmd.extend(sglang_extra)

    print(f"Model: {model}")
    print(f"SGLang host: {host}:{sglang_port}")
    print(f"Middleware:  {host}:{middleware_port}")
    print(f"haproxy:    {host}:{port}")
    if sglang_extra:
        print(f"Translated args: {' '.join(sglang_extra)}")
    if skipped:
        print(f"Skipped (no SGLang equivalent): {' '.join(skipped)}")
    print()
    print(f"SGLang command: {' '.join(sglang_cmd)}")
    print()

    # ── haproxy setup ────────────────────────────────────────

    os.makedirs("/tmp/haproxy-errors", exist_ok=True)
    with open("/tmp/haproxy-errors/503-sglang.http", "w") as f:
        f.write(
            "HTTP/1.0 503 Service Unavailable\r\nContent-Length: 16\r\nConnection: close\r\nContent-Type: text/plain\r\n\r\nSGLang not ready"
        )

    haproxy_cfg = "/tmp/haproxy-shim.cfg"
    with open(haproxy_cfg, "w") as f:
        f.write(f"""global
  maxconn 4096

defaults
  mode http
  timeout connect 5s
  timeout client 300s
  timeout server 300s

frontend proxy
  bind {host}:{port}

  acl is_health path /health
  acl sglang_up nbsrv(sglang) gt 0
  http-request deny deny_status 200 if is_health sglang_up
  http-request deny deny_status 503 if is_health
  errorfile 503 /tmp/haproxy-errors/503-sglang.http

  default_backend sglang

backend sglang
  option httpchk GET /health
  http-check expect status 200
  server s1 127.0.0.1:{middleware_port} check inter 5s fall 3 rise 2
""")

    with open(log_path, "a") as f:
        f.write(f"haproxy config written to {haproxy_cfg}\n")
        f.write(
            f"Model: {model}, SGLang port: {sglang_port}, middleware port: {middleware_port}, haproxy port: {port}\n"
        )
        f.write(f"SGLang command: {' '.join(sglang_cmd)}\n")
        if skipped:
            f.write(f"Skipped vLLM args: {' '.join(skipped)}\n")

    # ── Launch processes ─────────────────────────────────────

    sglang_proc = subprocess.Popen(sglang_cmd)

    middleware_env = os.environ.copy()
    middleware_env["SGLANG_HOST"] = host
    middleware_env["SGLANG_PORT"] = sglang_port
    middleware_env["MIDDLEWARE_PORT"] = middleware_port
    middleware_proc = subprocess.Popen(
        [sys.executable, "/opt/vllm-shim/vllm_middleware.py"],
        env=middleware_env,
    )

    time.sleep(2)

    haproxy_proc = subprocess.Popen(["haproxy", "-f", haproxy_cfg])

    with open(log_path, "a") as f:
        f.write(
            f"SGLang PID: {sglang_proc.pid}, middleware PID: {middleware_proc.pid}, haproxy PID: {haproxy_proc.pid}\n"
        )

    # Wait for whichever dies first
    while True:
        sglang_ret = sglang_proc.poll()
        middleware_ret = middleware_proc.poll()
        haproxy_ret = haproxy_proc.poll()
        if sglang_ret is not None:
            print(f"SGLang exited (code {sglang_ret}), shutting down")
            middleware_proc.terminate()
            haproxy_proc.terminate()
            os._exit(sglang_ret)
        if middleware_ret is not None:
            print(f"Middleware exited (code {middleware_ret}), shutting down")
            sglang_proc.terminate()
            haproxy_proc.terminate()
            os._exit(middleware_ret)
        if haproxy_ret is not None:
            print(f"haproxy exited (code {haproxy_ret}), shutting down")
            sglang_proc.terminate()
            middleware_proc.terminate()
            os._exit(haproxy_ret)
        time.sleep(1)


if __name__ == "__main__":
    main()

# Also run if imported as a module (some invocation paths just import the file)
main()
