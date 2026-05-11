"""AITER shape-capture orchestration.

Glues the existing pieces (rocm probe, ``ShapeStore``, ``parse_line``,
``StreamTee``) into a decision + setup the entrypoint can call once.

Capture is opt-in by environment: it lights up only when a ROCm GPU is
detected (so AITER is actually involved) AND the shim's persistent
home directory can be resolved (so the captured CSVs survive pod
restarts). Either prerequisite missing yields
``CapturePlan(enabled=False, reason=...)`` so the launch-info dump can
surface why.

The module exposes pure decision logic (``plan_capture``), persistent
home resolution (``resolve_shim_home``), and a tiny callback builder
(``build_callback``); the actual ``StreamTee`` wiring happens in the
entrypoint where the backend subprocess is spawned.
"""

import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from vllm_shim.aiter.log_parser import parse_line
from vllm_shim.aiter.path import shape_capture_root
from vllm_shim.aiter.shape_store import ShapeStore
from vllm_shim.cli.rocm_probe import GpuAgent, bucket
from vllm_shim.values.parallelism import Parallelism

REASON_NO_GPU = "no ROCm GPU detected"
REASON_NO_SHIM_HOME = "could not resolve VLLM_SHIM_HOME"
REASON_ENABLED = "enabled"


def resolve_shim_home() -> Path | None:
    """Resolve the shim's persistent home directory.

    Order of preference:

    1. ``$VLLM_SHIM_HOME`` if set. Production deployments set this to
       the persistent volume mount (typically ``/data/vllm-shim``) so
       captured shapes and tuned configs survive pod restarts.
    2. ``~/.vllm-shim`` as the default for dev hosts and operators who
       haven't picked a mount path yet.

    Returns ``None`` only when home expansion itself fails (e.g. some
    container images run with no home directory at all), so the caller
    can surface that as a disabled-capture reason.
    """
    env = os.environ.get("VLLM_SHIM_HOME")
    if env:
        return Path(env).expanduser()
    try:
        home = Path.home()
    except (RuntimeError, OSError):
        return None
    return home / ".vllm-shim"


@dataclass(frozen=True, slots=True)
class CapturePlan:
    """Output of ``plan_capture``: did we light up capture, and where to."""

    enabled: bool
    root: Path | None
    reason: str


def plan_capture(
    *,
    shim_home: Path | None,
    gpu: GpuAgent | None,
    model: str,
    parallelism: Parallelism,
) -> CapturePlan:
    """Decide whether AITER shape capture is feasible for this launch.

    Pure function. ``shim_home`` is the already-resolved shim home dir
    (use ``resolve_shim_home`` to compute it), or None when even home
    expansion failed. ``gpu`` is the first GPU agent from
    ``rocm_probe.probe()`` (or None on CUDA hosts and dev boxes). Both
    prerequisites must be present; otherwise the returned plan is
    disabled with a human-readable ``reason`` for the launch-info dump.
    """
    if gpu is None:
        return CapturePlan(enabled=False, root=None, reason=REASON_NO_GPU)
    if shim_home is None:
        return CapturePlan(enabled=False, root=None, reason=REASON_NO_SHIM_HOME)
    root = shape_capture_root(
        shim_home=shim_home,
        bucket=bucket(gpu),
        model=model,
        parallelism=parallelism,
    )
    return CapturePlan(enabled=True, root=root, reason=REASON_ENABLED)


def build_callback(store: ShapeStore) -> Callable[[str], None]:
    """Wrap ``parse_line + store.add`` for ``StreamTee.callback``.

    Lines that don't match the AITER shape-not-found pattern are
    ignored (``parse_line`` returns None). The store handles its own
    dedup so the callback can be called for every stderr line without
    paying for repeat writes.
    """

    def on_line(line: str) -> None:
        shape = parse_line(line)
        if shape is not None:
            store.add(shape)

    return on_line
