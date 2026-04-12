"""
vLLM -> SGLang Python shim.
Catches `python -m vllm.entrypoints.openai.api_server` (and similar)
and launches SGLang instead.
"""
import os
import sys
import subprocess

def main():
    args = sys.argv[1:]

    log_path = os.environ.get("VLLM_SHIM_LOG", "/tmp/vllm-shim.log")
    with open(log_path, "a") as f:
        import datetime
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

    print(f"Launching SGLang on {host}:{port}")
    print()

    os.execvp(
        sys.executable,
        [
            sys.executable, "-m", "sglang.launch_server",
            "--model-path", "mistralai/Devstral-2-123B-Instruct-2512",
            "--host", host,
            "--port", port,
            "--tp", "8",
            "--tool-call-parser", "mistral",
        ],
    )

if __name__ == "__main__":
    main()

# Also run if imported as a module (some invocation paths just import the file)
main()