"""AITER shape-capture orchestration.

Glues the existing pieces (rocm probe, ``ShapeStore``, ``parse_line``,
``StreamTee``) into a decision + setup the entrypoint can call once.

Capture is opt-in by environment: it lights up only when a ROCm GPU is
detected (so AITER is actually involved) AND an HF cache directory can
be resolved (so the captured CSVs land on the persistent volume that
survives pod restarts). Either prerequisite missing yields
``CapturePlan(enabled=False, reason=...)`` so the launch-info dump can
surface why.

The module exposes pure decision logic (``plan_capture``), HF cache
resolution (``resolve_hf_home``), and a tiny callback builder
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
REASON_NO_HF_HOME = "could not resolve HF cache directory"
REASON_ENABLED = "enabled"


def resolve_hf_home() -> Path | None:
    """Best-effort resolution of the HF cache root, matching HF's own rules.

    Order of preference:

    1. ``huggingface_hub.constants.HF_HOME`` — already handles the
       ``HF_HOME`` env var, ``XDG_CACHE_HOME`` fallback, and ``~``
       expansion. This is the same value HF itself uses for reads and
       writes, so captured shapes share the persistent volume with the
       model snapshots.
    2. Direct env var fallback if the import fails for any reason
       (paranoia: ``huggingface_hub`` is a required dep, but a broken
       install shouldn't take the whole launch with it).
    3. ``~/.cache/huggingface`` as a last resort, matching HF's documented
       default.

    Returns ``None`` only when home expansion itself fails (e.g. no home
    directory at all in some container images), so the caller can surface
    that as a disabled-capture reason.
    """
    try:
        from huggingface_hub.constants import HF_HOME as HF_HUB_HF_HOME
    except ImportError:
        HF_HUB_HF_HOME = ""
    if HF_HUB_HF_HOME:
        return Path(HF_HUB_HF_HOME)
    env = os.environ.get("HF_HOME")
    if env:
        return Path(env).expanduser()
    try:
        home = Path.home()
    except (RuntimeError, OSError):
        return None
    return home / ".cache" / "huggingface"


@dataclass(frozen=True, slots=True)
class CapturePlan:
    """Output of ``plan_capture``: did we light up capture, and where to."""

    enabled: bool
    root: Path | None
    reason: str


def plan_capture(
    *,
    hf_home: Path | None,
    gpu: GpuAgent | None,
    model: str,
    parallelism: Parallelism,
) -> CapturePlan:
    """Decide whether AITER shape capture is feasible for this launch.

    Pure function. ``hf_home`` is the already-resolved HF cache root
    (use ``resolve_hf_home`` to compute it), or None when no cache
    location could be determined. ``gpu`` is the first GPU agent from
    ``rocm_probe.probe()`` (or None on CUDA hosts and dev boxes). Both
    prerequisites must be present; otherwise the returned plan is
    disabled with a human-readable ``reason`` for the launch-info dump.
    """
    if gpu is None:
        return CapturePlan(enabled=False, root=None, reason=REASON_NO_GPU)
    if hf_home is None:
        return CapturePlan(enabled=False, root=None, reason=REASON_NO_HF_HOME)
    root = shape_capture_root(
        hf_home=hf_home,
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
