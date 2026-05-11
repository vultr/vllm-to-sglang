"""CLI entry point: parses vLLM args, spawns haproxy + middleware + backend, runs the supervisor."""

import os
import subprocess
import sys
from pathlib import Path

from vllm_shim.aiter.capture import build_callback, plan_capture, resolve_hf_home
from vllm_shim.aiter.restore import plan_restore, restore_configs
from vllm_shim.aiter.shape_store import ShapeStore
from vllm_shim.aiter.stream_tee import StreamTee
from vllm_shim.backend import registry
from vllm_shim.cli import info
from vllm_shim.cli.haproxy import HAProxyConfig, write_error_file
from vllm_shim.cli.haproxy import launch as launch_haproxy
from vllm_shim.cli.model_resolver import resolve_model
from vllm_shim.cli.parser import ArgParser
from vllm_shim.cli.rocm_probe import probe as probe_rocm
from vllm_shim.cli.supervisor import ManagedProcess, Supervisor
from vllm_shim.values.port_allocation import PortAllocation
from vllm_shim.values.service_address import ServiceAddress

HAPROXY_CONFIG_PATH = Path("/tmp/haproxy-shim.cfg")


def _pin_served_model_name(
    passthrough: tuple[str, ...], original: str, resolved: str
) -> tuple[str, ...]:
    """When the model arg has been rewritten to a snapshot directory, the
    user's original arg is what their clients still expect to see in
    /v1/models. Inject --served-model-name to preserve that, unless the
    caller already supplied one (in which case their choice wins)."""
    if resolved == original:
        return passthrough
    for a in passthrough:
        if a == "--served-model-name" or a.startswith("--served-model-name="):
            return passthrough
    return (*passthrough, "--served-model-name", original)


def main() -> int:
    """Spawn the three child processes and block in the supervisor until one exits."""
    original_argv = tuple(sys.argv[1:])
    parsed = ArgParser().parse(list(original_argv))
    backend = registry.select()

    ports = PortAllocation.from_listen(parsed.port)
    backend_addr = ServiceAddress(parsed.host, ports.backend)
    middleware_addr = ServiceAddress("127.0.0.1", ports.middleware)
    listen_addr = ServiceAddress(parsed.host, ports.frontend)

    # Bring up the public listener and middleware before resolving the model.
    # resolve_model can block for minutes on a cold HF cache; haproxy must
    # already be accepting connections so k8s liveness probes see the static
    # 503 errorfile (via nbsrv(sglang) gt 0) instead of connection refused.
    write_error_file()
    HAProxyConfig(listen=listen_addr, upstream=middleware_addr).write_to(HAPROXY_CONFIG_PATH)
    haproxy_proc = launch_haproxy(HAPROXY_CONFIG_PATH)

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

    resolved_model = resolve_model(parsed.model, revision=parsed.revision)
    passthrough = _pin_served_model_name(parsed.passthrough, parsed.model, resolved_model)
    backend_args, dropped = backend.args.translate(passthrough)

    backend_cmd = backend.launcher.build_command(resolved_model, backend_addr, backend_args)
    # Auto-translate selected vLLM env vars into the backend's namespace so
    # operators with vLLM-style k8s env: blocks don't lose those settings on
    # the SGLang/TRT-LLM side. See the backend's env.ENV_MAP for what's
    # in scope; vLLM-side names stay in place and are ignored by the backend.
    backend_env = backend.env.translate(os.environ)

    # AITER integration: probe ROCm + resolve HF cache once, then drive both
    # the restore (point AITER's AITER_CONFIG_* env vars at previously tuned
    # CSVs on the PV) and capture (record shape misses for the next tuner
    # run) decisions off the same shared environment. Restore overrides land
    # in backend_env before spawn so AITER picks them up at import time.
    hf_home = resolve_hf_home()
    gpu = probe_rocm()
    restore_plan = plan_restore(hf_home=hf_home, gpu=gpu)
    # Operator-set AITER_CONFIG_* env vars win over our restore. Same
    # principle as translate_env_with_map: if the operator wrote it
    # into the pod spec, they meant it. The launch-info dump shows
    # only the overrides that actually took effect.
    restored = {
        k: v for k, v in restore_configs(restore_plan).items() if k not in backend_env
    }
    backend_env.update(restored)
    capture_plan = plan_capture(
        hf_home=hf_home,
        gpu=gpu,
        model=parsed.model,
        parallelism=backend.parallelism.extract(backend_args),
    )

    # Snapshot every translation/resolution decision to disk + stderr so
    # vllm-shim-info can echo it from a pod shell and pod logs show the
    # final backend invocation without grepping the supervisor's output.
    launch_info = info.collect(
        original_argv=original_argv,
        backend_name=backend.name,
        model_original=parsed.model,
        model_resolved=resolved_model,
        revision=parsed.revision,
        listen=listen_addr,
        ports=ports,
        backend_argv=backend_cmd,
        dropped_args=dropped,
        parent_env=os.environ,
        backend_env=backend_env,
        aiter_capture=capture_plan,
        aiter_restore=restore_plan,
        aiter_restored=restored,
    )
    info.write(launch_info)
    info.print_summary(launch_info)

    if capture_plan.enabled and capture_plan.root is not None:
        backend_proc = subprocess.Popen(
            backend_cmd, env=backend_env, stderr=subprocess.PIPE
        )
        # The tee is a daemon thread; backend stderr is the only stream it
        # reads, and it exits on EOF when the backend closes its end. We
        # don't track it past the entrypoint - the supervisor's shutdown
        # path already owns the backend's lifecycle.
        assert backend_proc.stderr is not None
        StreamTee(
            source=backend_proc.stderr,
            sink=sys.stderr.buffer,
            callback=build_callback(ShapeStore(capture_plan.root)),
        ).start()
    else:
        backend_proc = subprocess.Popen(backend_cmd, env=backend_env)

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
