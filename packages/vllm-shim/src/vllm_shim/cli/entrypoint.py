"""CLI entry point: parses vLLM args, spawns haproxy + middleware + backend, runs the supervisor."""

import os
import subprocess
import sys
from collections.abc import Callable, Mapping
from pathlib import Path

from vllm_shim.aiter.capture import build_callback, plan_capture, resolve_shim_home
from vllm_shim.aiter.hip_online_tuning import (
    apply_hip_online_tuning,
    plan_hip_online_tuning,
)
from vllm_shim.aiter.restore import plan_restore, restore_configs
from vllm_shim.aiter.shape_store import ShapeStore
from vllm_shim.aiter.stream_tee import StreamTee
from vllm_shim.backend import registry
from vllm_shim.cli import info
from vllm_shim.cli.haproxy import HAProxyConfig, write_error_file
from vllm_shim.cli.haproxy import launch as launch_haproxy
from vllm_shim.cli.model_resolver import resolve_model
from vllm_shim.cli.parser import ArgParser
from vllm_shim.cli.rocm_perf import rocm_perf_defaults
from vllm_shim.cli.rocm_probe import GpuAgent
from vllm_shim.cli.rocm_probe import bucket as bucket_for_gpu
from vllm_shim.cli.rocm_probe import probe as probe_rocm
from vllm_shim.cli.supervisor import ManagedProcess, Supervisor
from vllm_shim.values.port_allocation import PortAllocation
from vllm_shim.values.service_address import ServiceAddress

HAPROXY_CONFIG_PATH = Path("/tmp/haproxy-shim.cfg")


def _parse_tune_budget(env_value: str | None) -> int:
    """Read VLLM_SHIM_TUNE_AT_STARTUP_SECONDS as a non-negative int.

    Unset, empty, "0", or anything that doesn't parse cleanly means
    "tuning at startup is off". A positive integer opts in with that
    many seconds as the hard wall-clock budget.
    """
    if not env_value:
        return 0
    try:
        n = int(env_value)
    except ValueError:
        return 0
    return n if n > 0 else 0


def _parse_tune_hot(env_value: str | None) -> int | None:
    """Read VLLM_SHIM_TUNE_AT_STARTUP_HOT as an optional positive int.

    Unset, empty, "0", or anything that doesn't parse cleanly means
    "no --hot flag is passed to vllm-shim-tune" (tune every captured
    shape). A positive integer opts into hot-shape filtering with
    that many shapes per target.
    """
    if not env_value:
        return None
    try:
        n = int(env_value)
    except ValueError:
        return None
    return n if n > 0 else None


def _maybe_run_startup_tune(
    *,
    shim_home: Path | None,
    gpu: GpuAgent | None,
    budget_seconds: int,
    hot: int | None = None,
    env: Mapping[str, str] | None = None,
    run: Callable[[list[str], int, Mapping[str, str] | None], int] | None = None,
) -> None:
    """Best-effort vllm-shim-tune subprocess before the backend launches.

    Gated on ``budget_seconds > 0`` plus the same prerequisites as
    AITER restore (ROCm GPU + resolvable shim home). The tuner inherits
    stderr so its progress lands in pod logs. Any failure - timeout,
    crash, missing console script - is logged and swallowed; tuning is
    never allowed to block the backend from launching.

    Hard wall-clock cap matters: without it, a fresh-from-empty tune
    of a large MoE on a single GPU could spend an hour benchmarking,
    blowing through k8s ``progressDeadlineSeconds`` and CrashLoopBackOff
    the pod. Partial tunes are safe - AITER writes tuned rows
    incrementally, so a killed-mid-shape subprocess leaves a valid
    (partial) tuned CSV that the next restart picks up where it left
    off.

    ``hot`` (driven by ``VLLM_SHIM_TUNE_AT_STARTUP_HOT``) opts the
    subprocess into per-target hot-shape filtering. The wall-clock
    budget is still a hard ceiling; ``--hot N`` reduces the input set
    so the budget is far less likely to truncate mid-shape, which is
    the common reason a startup tune ships an under-covered config.

    ``--no-flydsl`` is always passed: FlyDSL candidates JIT-compile
    per kernel during benchmarking and dominate per-shape wall-time
    on the two targets that include them (bf16_tuned_gemm,
    a8w8_bpreshuffle_tuned_gemm). Excluding them keeps the candidate
    pool small enough that more shapes fit inside the startup budget.
    Operator-driven ``vllm-shim-tune`` runs don't get this flag by
    default; they keep the full candidate set.

    ``env`` is the dict the supervisor will later hand to the backend
    (i.e. ``backend_env`` from main()). Forwarding it to the tune
    subprocess matters: rocm_perf_defaults injects ``AITER_JIT_DIR``
    into backend_env, and AITER's JIT loader only uses the patched
    source path when that var is set (``get_module_custom_op`` in
    ``aiter/jit/core.py``). Without it, the tuner inherits a plain
    ``os.environ``, the bare import falls back to ``aiter.jit.<md>``,
    and the loader picks up the base-image-shipped .so from
    site-packages instead of the patched build.

    ``HIP_ONLINE_TUNING`` is stripped here even when forwarded in
    ``env``: gradlib's ``hipblasLtMatmul_sol_wrapper`` reads the var
    at every matmul and, when set, *overwrites* the caller's
    explicit ``solution_index`` with a CSV lookup and falls back to
    its own internal benchmark sweep
    (``repos/aiter/gradlib/csrc/hipbsolgemm.cu``, the
    ``if (online_tuning && n <= decode_max_n)`` blocks). That sweep
    fights the tuner's own per-candidate benchmarking, hits
    crash-prone algos that the tuner would have ranked or skipped,
    and crashes the worker. The crash respawns the worker, the new
    PID isn't in ``mp_tuner``'s ``gpu_map``, and every subsequent
    task on that worker raises ``KeyError`` until the tuner exits.
    Online tuning during offline tuning is incoherent anyway:
    serving-path online tuning is the runtime fallback when there's
    no tuned CSV; the offline tuner is the thing that *produces*
    the tuned CSV that makes online tuning unnecessary.
    """
    if budget_seconds <= 0 or gpu is None or shim_home is None:
        return
    cmd = [
        "vllm-shim-tune",
        "--shim-home",
        str(shim_home),
        "--bucket",
        bucket_for_gpu(gpu),
        "--no-flydsl",
    ]
    if hot is not None:
        cmd.extend(["--hot", str(hot)])
    hot_note = f", --hot {hot}" if hot is not None else ""
    sys.stderr.write(
        f"vllm-shim startup tune: running ({budget_seconds}s budget{hot_note})\n"
    )
    tune_env = (
        None
        if env is None
        else {k: v for k, v in env.items() if k != "HIP_ONLINE_TUNING"}
    )
    runner = run if run is not None else _default_tune_runner
    try:
        rc = runner(cmd, budget_seconds, tune_env)
        sys.stderr.write(f"vllm-shim startup tune: exit {rc}\n")
    except subprocess.TimeoutExpired:
        sys.stderr.write(
            f"vllm-shim startup tune: exceeded {budget_seconds}s budget; "
            "continuing with whatever was tuned so far\n"
        )
    except Exception as e:
        # Broad on purpose: missing console script, AITER blowups,
        # permission errors all map to "skip and continue". Tuning is
        # never allowed to block the backend from launching.
        sys.stderr.write(
            f"vllm-shim startup tune: failed ({e}); continuing\n"
        )


def _default_tune_runner(
    cmd: list[str], timeout: int, env: Mapping[str, str] | None
) -> int:
    # env=None preserves subprocess.run's default (inherit os.environ).
    # A populated env is the merged backend_env (parent + translations
    # + rocm_perf_defaults), which is what AITER's JIT loader expects.
    return subprocess.run(
        cmd, timeout=timeout, check=False, env=dict(env) if env is not None else None
    ).returncode


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

    # AITER integration: probe ROCm + resolve the shim's persistent home once,
    # then drive both the restore (point AITER's AITER_CONFIG_* env vars at
    # previously tuned CSVs on the PV) and capture (record shape misses for
    # the next tuner run) decisions off the same shared environment. Restore
    # overrides land in backend_env before spawn so AITER picks them up at
    # import time.
    shim_home = resolve_shim_home()
    gpu = probe_rocm()
    restore_plan = plan_restore(shim_home=shim_home, gpu=gpu)
    # Operator-set AITER_CONFIG_* env vars win over our restore. Same
    # principle as translate_env_with_map: if the operator wrote it
    # into the pod spec, they meant it. The launch-info dump shows
    # only the overrides that actually took effect.
    restored = {
        k: v for k, v in restore_configs(restore_plan).items() if k not in backend_env
    }
    backend_env.update(restored)
    capture_plan = plan_capture(
        shim_home=shim_home,
        gpu=gpu,
        model=parsed.model,
        parallelism=backend.parallelism.extract(backend_args),
    )
    # The shape-not-found line is emitted at INFO by AITER's logger
    # (tuned_gemm.py:192). If an operator raises log level globally,
    # capture goes silent. Pin INFO when capture is enabled, via
    # setdefault so an explicit AITER_LOG_LEVEL in the pod spec still
    # wins (e.g. someone deliberately quieting AITER for debugging).
    if capture_plan.enabled:
        backend_env.setdefault("AITER_LOG_LEVEL", "INFO")

    # Opinionated ROCm perf defaults (MIOpen cache on PV, hipBLASLt
    # preference, MI300X-specific HW-queue/RCCL knobs). Same setdefault
    # discipline as restore: only fill in what the operator left blank.
    # See cli/rocm_perf.py for source links and rationale.
    rocm_perf_applied = {
        k: v
        for k, v in rocm_perf_defaults(gpu, shim_home).items()
        if k not in backend_env
    }
    backend_env.update(rocm_perf_applied)

    # When the operator opted into AITER's HIP_ONLINE_TUNING, anchor
    # ./hip_online_tuning_res.csv on the PV via symlink. The path is
    # hardcoded relative to the backend's CWD in gradlib's C++ source,
    # so a symlink is the only way to persist accumulated tuning data
    # across pod restarts. See vllm_shim/aiter/hip_online_tuning.py.
    hip_tuning_plan = plan_hip_online_tuning(os.environ, gpu, shim_home, Path.cwd())
    apply_hip_online_tuning(hip_tuning_plan)

    # Opt-in startup tune: turns a pod restart into a tune+serve cycle
    # on single-GPU deployments where running a separate tuning job
    # isn't possible. Off by default; operator sets
    # VLLM_SHIM_TUNE_AT_STARTUP_SECONDS to a positive integer to enable
    # with that wall-clock budget. Runs after restore so the tuner sees
    # what's already tuned (incremental by default), and before the
    # backend spawn so the freshly tuned configs land in AITER's first
    # import. See docs/aiter.md.
    _maybe_run_startup_tune(
        shim_home=shim_home,
        gpu=gpu,
        budget_seconds=_parse_tune_budget(
            os.environ.get("VLLM_SHIM_TUNE_AT_STARTUP_SECONDS")
        ),
        hot=_parse_tune_hot(os.environ.get("VLLM_SHIM_TUNE_AT_STARTUP_HOT")),
        env=backend_env,
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
        rocm_perf=rocm_perf_applied,
        hip_online_tuning=hip_tuning_plan,
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
