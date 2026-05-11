"""Restore tuned AITER configs from the persistent volume.

AITER looks up tuned configs at a hardcoded location: ``/tmp/aiter_configs/<target>.csv``.
On a fresh pod that directory is empty, so AITER falls back to default
configs (which is what produces the shape-not-found warnings we capture).

If a previous tuning run produced configs and stored them under HF_HOME
(``$HF_HOME/vllm-shim/aiter-configs/<bucket>/``), this module symlinks
each into ``/tmp/aiter_configs/`` at startup so AITER picks them up.

Restore is the mirror of capture: same prerequisites (ROCm GPU + HF cache
resolvable), but partitioned by GPU SKU only (no model / parallelism),
since tuned configs are keyed by shape dimensions and reusable across
models that happen to hit those shapes.
"""

import contextlib
from dataclasses import dataclass
from pathlib import Path

from vllm_shim.aiter.capture import REASON_ENABLED, REASON_NO_GPU, REASON_NO_HF_HOME
from vllm_shim.cli.rocm_probe import GpuAgent, bucket

# AITER's hardcoded read location. Not configurable AITER-side, so we
# don't expose it as a knob here either.
AITER_CONFIG_DIR = Path("/tmp/aiter_configs")


@dataclass(frozen=True, slots=True)
class RestorePlan:
    """Output of ``plan_restore``: did we light up restore, from where to where."""

    enabled: bool
    source: Path | None
    target: Path
    reason: str


def plan_restore(
    *,
    hf_home: Path | None,
    gpu: GpuAgent | None,
    target: Path = AITER_CONFIG_DIR,
) -> RestorePlan:
    """Decide whether to restore tuned configs for this launch.

    Pure function. Same prerequisites as ``plan_capture``: ROCm GPU
    (else there's no point) and a resolved HF cache (else no source).
    ``target`` is the directory AITER reads from; the default is the
    location AITER hardcodes, parameterised here for tests.
    """
    if gpu is None:
        return RestorePlan(enabled=False, source=None, target=target, reason=REASON_NO_GPU)
    if hf_home is None:
        return RestorePlan(
            enabled=False, source=None, target=target, reason=REASON_NO_HF_HOME
        )
    source = hf_home / "vllm-shim" / "aiter-configs" / bucket(gpu)
    return RestorePlan(enabled=True, source=source, target=target, reason=REASON_ENABLED)


def restore_configs(plan: RestorePlan) -> list[str]:
    """Symlink every tuned-config file from ``plan.source`` into ``plan.target``.

    Returns the basenames that were newly symlinked, in stable order, so
    the launch-info dump can show exactly what AITER will pick up.
    Idempotent: a destination that already exists (regular file or
    symlink) is left alone so re-launches in the same ``/tmp`` don't
    fight over it. Per-file OS errors (permissions, ENOSPC, broken
    source) are swallowed - the shim must not refuse to launch the
    backend because a restore step misbehaved.
    """
    if not plan.enabled or plan.source is None:
        return []
    plan.target.mkdir(parents=True, exist_ok=True)
    if not plan.source.is_dir():
        return []
    restored: list[str] = []
    for f in sorted(plan.source.iterdir()):
        if not f.is_file():
            continue
        dest = plan.target / f.name
        if dest.exists() or dest.is_symlink():
            continue
        with contextlib.suppress(OSError):
            dest.symlink_to(f.resolve())
            restored.append(f.name)
    return restored
