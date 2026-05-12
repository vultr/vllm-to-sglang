"""Anchor the HIP online-tuning CSV on the PV via a CWD symlink.

When ``HIP_ONLINE_TUNING=1`` is set, AITER's gradlib reads and writes
``./hip_online_tuning_res.csv`` relative to the backend's current
working directory (see ``repos/aiter/gradlib/csrc/hipbsolgemm.cu``,
``get_algoIdx_hip_tuning_csv`` and ``append_hip_tuning_csv``). The
path is hardcoded into the C++ source, so no env var can redirect it.

Default container layout puts the CWD on an ephemeral layer, which
means the several-minute first-call tune cost gets re-paid by every
fresh pod. To anchor the data on the PV the shim already manages we
keep the canonical file at ``$VLLM_SHIM_HOME/hip_online_tuning_res.csv``
and symlink the CWD path onto it; AITER's relative-path read/write
transparently goes through the symlink.

Symlinking (rather than a copy-on-shutdown handler) is load-bearing:
gradlib appends per-shape and a pod that exits abnormally would lose
any rows tuned since the last save.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from vllm_shim.cli.rocm_probe import GpuAgent

FILENAME = "hip_online_tuning_res.csv"

REASON_DISABLED = "env not set"
REASON_NO_GPU = "no ROCm GPU"
REASON_NO_SHIM_HOME = "no shim home"
REASON_WILL_LINK = "will link"
REASON_ALREADY_LINKED = "already linked"
REASON_BLOCKED = "non-symlink file at target"


@dataclass(frozen=True, slots=True)
class HipTuningPlan:
    """What the shim will do about HIP_ONLINE_TUNING for this launch.

    ``enabled`` means the operator opted in via the env var AND the
    shim has a PV to point at. The ``reason`` field distinguishes
    sub-states (will-link vs. already-linked vs. blocked) and is the
    same string the launch-info dump surfaces.
    """

    enabled: bool
    storage: Path | None
    target: Path | None
    reason: str


def plan_hip_online_tuning(
    env: Mapping[str, str],
    gpu: GpuAgent | None,
    shim_home: Path | None,
    cwd: Path,
) -> HipTuningPlan:
    """Decide whether and how to anchor the tuning CSV on the PV.

    Pure-ish: reads filesystem state at ``cwd / FILENAME`` to detect
    an already-correct symlink so a restart can no-op, but does no
    writes. The caller passes the result to ``apply`` to materialize
    the symlink.

    Gated on a ROCm GPU + a resolvable shim home, matching restore and
    capture. ``HIP_ONLINE_TUNING`` only has meaning when AITER's
    gradlib is loaded (ROCm-only); we don't want to leave a symlink
    behind on a CUDA host where the operator misset the env.
    """
    if env.get("HIP_ONLINE_TUNING") not in ("1", "true"):
        return HipTuningPlan(False, None, None, REASON_DISABLED)
    if gpu is None:
        return HipTuningPlan(False, None, None, REASON_NO_GPU)
    if shim_home is None:
        return HipTuningPlan(False, None, None, REASON_NO_SHIM_HOME)
    storage = shim_home / FILENAME
    target = cwd / FILENAME
    if target.is_symlink():
        # ``readlink`` returns the exact bytes the symlink stores. We
        # always create symlinks with absolute storage paths, so a
        # bit-equal comparison is enough; a wrong-pointing or relative
        # symlink (operator artifact, image upgrade) falls through to
        # WILL_LINK and gets replaced.
        try:
            if target.readlink() == storage:
                return HipTuningPlan(True, storage, target, REASON_ALREADY_LINKED)
        except OSError:
            pass
        return HipTuningPlan(True, storage, target, REASON_WILL_LINK)
    if target.exists():
        # A regular file already lives at the target path. This is
        # almost always operator data from a pre-shim run; refuse to
        # clobber it and surface the reason so the operator can move
        # it onto the PV by hand.
        return HipTuningPlan(True, storage, target, REASON_BLOCKED)
    return HipTuningPlan(True, storage, target, REASON_WILL_LINK)


def apply_hip_online_tuning(plan: HipTuningPlan) -> None:
    """Materialize the plan: touch the PV file and (re)create the symlink.

    Idempotent. ``REASON_BLOCKED`` and disabled plans are no-ops; the
    operator sees the reason in the launch info and acts manually.
    """
    if not plan.enabled or plan.storage is None or plan.target is None:
        return
    if plan.reason == REASON_BLOCKED:
        return
    plan.storage.parent.mkdir(parents=True, exist_ok=True)
    plan.storage.touch(exist_ok=True)
    if plan.reason == REASON_ALREADY_LINKED:
        return
    if plan.target.is_symlink():
        plan.target.unlink()
    plan.target.symlink_to(plan.storage)
