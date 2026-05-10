"""CLI entry point: parses vLLM args, spawns haproxy + middleware + backend, runs the supervisor."""

import os
import subprocess
import sys
from pathlib import Path

from vllm_shim.backend import registry
from vllm_shim.cli.haproxy import HAProxyConfig, write_error_file
from vllm_shim.cli.haproxy import launch as launch_haproxy
from vllm_shim.cli.parser import ArgParser
from vllm_shim.cli.supervisor import ManagedProcess, Supervisor
from vllm_shim.values.port_allocation import PortAllocation
from vllm_shim.values.service_address import ServiceAddress

HAPROXY_CONFIG_PATH = Path("/tmp/haproxy-shim.cfg")


def main() -> int:
    """Spawn the three child processes and block in the supervisor until one exits."""
    parsed = ArgParser().parse(sys.argv[1:])
    backend = registry.select()

    backend_args, _dropped = backend.args.translate(parsed.passthrough)
    ports = PortAllocation.from_listen(parsed.port)
    backend_addr = ServiceAddress(parsed.host, ports.backend)
    middleware_addr = ServiceAddress("127.0.0.1", ports.middleware)
    listen_addr = ServiceAddress(parsed.host, ports.frontend)

    backend_cmd = backend.launcher.build_command(parsed.model, backend_addr, backend_args)
    # Auto-translate selected vLLM env vars into the backend's namespace so
    # operators with vLLM-style k8s env: blocks don't lose those settings on
    # the SGLang/TRT-LLM side. See the backend's env.ENV_MAP for what's
    # in scope; vLLM-side names stay in place and are ignored by the backend.
    backend_env = backend.env.translate(os.environ)
    backend_proc = subprocess.Popen(backend_cmd, env=backend_env)

    # Middleware reads its config from env, not argv, so it can be launched independently
    # (e.g. via the vllm-shim-middleware console script in tests).
    middleware_env = os.environ.copy()
    middleware_env["VLLM_SHIM_BACKEND_HOST"] = backend_addr.host
    middleware_env["VLLM_SHIM_BACKEND_PORT"] = str(backend_addr.port)
    middleware_env["VLLM_SHIM_MIDDLEWARE_PORT"] = str(middleware_addr.port)
    middleware_proc = subprocess.Popen(
        [sys.executable, "-m", "vllm_shim.middleware"],
        env=middleware_env,
    )

    write_error_file()
    HAProxyConfig(listen=listen_addr, upstream=middleware_addr).write_to(HAPROXY_CONFIG_PATH)
    haproxy_proc = launch_haproxy(HAPROXY_CONFIG_PATH)

    # Order matters for shutdown drain quality (haproxy first, backend last); see Supervisor.
    return Supervisor(
        [
            ManagedProcess("haproxy", haproxy_proc),
            ManagedProcess("middleware", middleware_proc),
            ManagedProcess("backend", backend_proc),
        ]
    ).run()


if __name__ == "__main__":
    raise SystemExit(main())
