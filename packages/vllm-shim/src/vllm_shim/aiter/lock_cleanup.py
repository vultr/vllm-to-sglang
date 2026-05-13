"""Clear stale AITER JIT lock files before launching the backend.

AITER uses file-existence-based locks (see
``repos/aiter/aiter/jit/utils/file_baton.py``) to serialise JIT kernel
compilation across the TP/EP ranks of a single pod. The first rank to
``try_acquire`` an exclusive lock file does the build; the others spin
in ``wait()`` checking ``os.path.exists`` every 200 ms. The release
path is ``os.remove`` of the lock file. There is no timeout, no
liveness check on the acquirer, no PID written into the lock body.

This is fine when every rank exits normally. It is catastrophic when a
rank crashes mid-build: the lock file survives, and on the next pod
start every rank sees the file, every rank goes into ``wait()``, and
no rank ever clears it. Because ``$VLLM_SHIM_HOME/aiter/jit/<sha>``
lives on the persistent volume by design (so compiled .so files
survive pod restarts), the wedge follows the PV across pod restarts
until somebody shells into the volume and removes the file by hand.

This module clears those locks at entrypoint startup. It runs in the
single-threaded entrypoint process before any AITER-touching child
(the optional startup tune subprocess, the backend) has spawned, so
any lock file present at this moment is by definition orphaned from a
previous pod's crash and safe to remove.

Lock file naming patterns AITER uses (verified in
``repos/aiter/aiter/jit/core.py`` lines 309/716/744 and
``repos/aiter/csrc/cpp_itfs/utils.py`` lines 160/379):

- ``lock`` (exact basename, e.g. ``<build_dir>/lock``)
- ``lock_*`` prefix (e.g. ``lock_module_custom_all_reduce``,
  ``lock_3rdparty_clone_<third_party>``)
- ``*.lock`` suffix (e.g. ``<func_name>.lock``)

We match all three explicitly rather than a single ``lock*`` glob, so
arbitrary user-named files containing ``lock`` in their name are not
swept up by accident.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from vllm_shim.aiter.capture import REASON_ENABLED, REASON_NO_GPU, REASON_NO_SHIM_HOME
from vllm_shim.cli.rocm_perf import aiter_cache_key
from vllm_shim.cli.rocm_probe import GpuAgent


@dataclass(frozen=True, slots=True)
class CleanupPlan:
    """Output of ``plan_cleanup``: did we light up cleanup, and where to scan."""

    enabled: bool
    root: Path | None
    reason: str


def plan_cleanup(
    *,
    shim_home: Path | None,
    gpu: GpuAgent | None,
    env: Mapping[str, str] | None = None,
) -> CleanupPlan:
    """Decide whether to clean and where to scan.

    Pure function. Same prerequisites as ``plan_restore``: ROCm GPU
    (else there is no AITER to wedge) and a resolved shim home (else
    no JIT dir to scan).

    The scan root follows the operator's ``AITER_JIT_DIR`` if they set
    one in the parent environment, mirroring the precedence
    ``rocm_perf_defaults`` uses to set the same env var; otherwise it
    is the shim's default JIT path
    ``$VLLM_SHIM_HOME/aiter/jit/<aiter-cache-key>``. The two paths
    must stay aligned with what ``rocm_perf.aiter_cache_key`` returns
    or cleanup scans the wrong directory and silently no-ops.
    """
    if gpu is None:
        return CleanupPlan(enabled=False, root=None, reason=REASON_NO_GPU)
    if shim_home is None:
        return CleanupPlan(enabled=False, root=None, reason=REASON_NO_SHIM_HOME)
    if env is not None and (operator := env.get("AITER_JIT_DIR")):
        return CleanupPlan(enabled=True, root=Path(operator), reason=REASON_ENABLED)
    return CleanupPlan(
        enabled=True,
        root=shim_home / "aiter" / "jit" / aiter_cache_key(),
        reason=REASON_ENABLED,
    )


def _is_lock_file(name: str) -> bool:
    """Match the three lock-file naming patterns AITER uses.

    See module docstring for the source-line references that ground
    each pattern.
    """
    return name == "lock" or name.startswith("lock_") or name.endswith(".lock")


def cleanup_locks(plan: CleanupPlan) -> list[Path]:
    """Delete every stale AITER lock under ``plan.root``.

    Best-effort: a permission error or filesystem race on an
    individual lock file is swallowed so one un-removable file cannot
    block pod startup. The expected failure modes (permission, race
    with concurrent removal, transient FS quirk) all map cleanly to
    "skip and let AITER handle it downstream"; the only way AITER
    handles it downstream is to wait forever on the same file, but
    that is the pre-cleanup baseline so we are no worse off.

    Returns the absolute paths actually deleted, so callers can log a
    count and surface the list in the launch-info dump if desired.
    """
    if not plan.enabled or plan.root is None or not plan.root.is_dir():
        return []
    cleaned: list[Path] = []
    for f in plan.root.rglob("*"):
        if not f.is_file():
            continue
        if not _is_lock_file(f.name):
            continue
        try:
            f.unlink()
            cleaned.append(f)
        except OSError:
            continue
    return cleaned
