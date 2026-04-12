"""
vLLM -> SGLang Python shim.
Catches `python -m vllm.entrypoints.openai.api_server` (and similar)
and launches SGLang behind haproxy instead.

Architecture:
  haproxy on the vLLM port (front door)
    /metrics → 200 empty (stub)
    /health  → 200 if SGLang backend is up, 503 if not (instant)
    /*       → proxy to SGLang on port+1
  SGLang on port+1 (internal)

haproxy 2.4 compat: uses errorfile for stub responses instead
of http-request return (which needs 2.8+ for payload syntax).
"""
import os
import sys
import subprocess
import time

def main():
    args = sys.argv[1:]

    log_path = os.environ.get("VLLM_SHIM_LOG", "/tmp/vllm-shim.log")
    import datetime
    with open(log_path, "a") as f:
        f.write(f"\n{datetime.datetime.now().isoformat()} vLLM -> SGLang Shim (Python module)\n")
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

    host = "0.0.0.0"
    port = "8000"

    i = 0
    while i < len(args):
        if args[i] == "--host" and i + 1 < len(args):
            host = args[i + 1]
            i += 2
        elif args[i].startswith("--host="):
            host = args[i].split("=", 1)[1]
            i += 1
        elif args[i] == "--port" and i + 1 < len(args):
            port = args[i + 1]
            i += 2
        elif args[i].startswith("--port="):
            port = args[i].split("=", 1)[1]
            i += 1
        else:
            i += 1

    # SGLang runs one port higher; middleware two ports higher
    sglang_port = str(int(port) + 1)
    middleware_port = str(int(port) + 2)

    print(f"Launching SGLang on {host}:{sglang_port} (internal)")
    print(f"Launching middleware on {host}:{middleware_port} (strips logprobs)")
    print(f"Launching haproxy on {host}:{port} (front door, /metrics + /health stub)")
    print()

    # Prepare error files for haproxy stub responses
    # haproxy errorfile format: HTTP/1.x status_code reason\r\nheaders\r\n\r\nbody
    os.makedirs("/tmp/haproxy-errors", exist_ok=True)
    with open("/tmp/haproxy-errors/200-empty.http", "w") as f:
        f.write("HTTP/1.0 200 OK\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
    with open("/tmp/haproxy-errors/503-sglang.http", "w") as f:
        f.write("HTTP/1.0 503 Service Unavailable\r\nContent-Length: 16\r\nConnection: close\r\nContent-Type: text/plain\r\n\r\nSGLang not ready")

    # Write haproxy config (compatible with haproxy 2.4)
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

  # /metrics stub — instant 200 empty (vLLM stack expects this)
  acl is_metrics path /metrics
  http-request deny deny_status 200 if is_metrics
  errorfile 200 /tmp/haproxy-errors/200-empty.http

  # /health — instant response based on SGLang backend state
  # haproxy health-checks SGLang in the background; this avoids
  # the 1s k8s probe timeout racing SGLang's ~1.001s /health response
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
        f.write(f"SGLang port: {sglang_port}, middleware port: {middleware_port}, haproxy port: {port}\n")

    # Start SGLang in the background
    sglang_proc = subprocess.Popen(
        [
            sys.executable, "-m", "sglang.launch_server",
            "--model-path", "mistralai/Devstral-2-123B-Instruct-2512",
            "--host", host,
            "--port", sglang_port,
            "--tp", "8",
            "--tool-call-parser", "mistral",
        ],
    )

    # Start the middleware (strips vLLM-only params like logprobs)
    middleware_env = os.environ.copy()
    middleware_env["SGLANG_PORT"] = sglang_port
    middleware_env["MIDDLEWARE_PORT"] = middleware_port
    middleware_proc = subprocess.Popen(
        [sys.executable, "/opt/vllm-shim/vllm_middleware.py"],
        env=middleware_env,
    )

    # Give SGLang a moment before haproxy starts routing
    time.sleep(2)

    # Start haproxy in the background
    haproxy_proc = subprocess.Popen(["haproxy", "-f", haproxy_cfg])

    with open(log_path, "a") as f:
        f.write(f"SGLang PID: {sglang_proc.pid}, middleware PID: {middleware_proc.pid}, haproxy PID: {haproxy_proc.pid}\n")

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
