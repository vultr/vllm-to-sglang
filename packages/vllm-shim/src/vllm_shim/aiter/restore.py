"""Restore tuned AITER configs by env-var override, not by symlink.

AITER picks its tuned-config CSV path off env vars at import time (see
``repos/aiter/aiter/jit/core.py``). Each kernel target has its own
override: ``AITER_CONFIG_GEMM_BF16`` for ``bf16_tuned_gemm.csv``,
``AITER_CONFIG_GEMM_A8W8`` for ``a8w8_tuned_gemm.csv``, and so on.
Defaults point inside the AITER install directory, which is ephemeral
in a container.

Restore therefore lives in env-translation: for each known target we
find a matching CSV under ``$HF_HOME/vllm-shim/aiter-configs/<bucket>/``,
we set the corresponding env var in the backend's environment. AITER
then reads directly from the persistent volume on first lookup. No
symlinks, no writes into ``/tmp``, multiple pods on the same PV share
the same files read-only for free.

Restore is the mirror of capture: same prerequisites (ROCm GPU + HF
cache resolvable), partitioned by GPU SKU only (no model / parallelism)
because tuned configs are keyed by shape dimensions and reusable across
models that hit the same shapes.
"""

from dataclasses import dataclass
from pathlib import Path

from vllm_shim.aiter.capture import REASON_ENABLED, REASON_NO_GPU, REASON_NO_HF_HOME
from vllm_shim.cli.rocm_probe import GpuAgent, bucket

# Mapping from AITER tuned-config basename (the ``target`` field of an
# ``AiterShape``) to the env var that overrides where AITER reads it.
# Sourced from ``repos/aiter/aiter/jit/core.py``; keep in sync if AITER
# adds new tuned-config targets.
_TARGET_ENV: dict[str, str] = {
    "bf16_tuned_gemm": "AITER_CONFIG_GEMM_BF16",
    "a4w4_blockscale_tuned_gemm": "AITER_CONFIG_GEMM_A4W4",
    "a8w8_tuned_gemm": "AITER_CONFIG_GEMM_A8W8",
    "a8w8_bpreshuffle_tuned_gemm": "AITER_CONFIG_GEMM_A8W8_BPRESHUFFLE",
    "a8w8_blockscale_tuned_gemm": "AITER_CONFIG_GEMM_A8W8_BLOCKSCALE",
    "a8w8_blockscale_bpreshuffle_tuned_gemm": (
        "AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE"
    ),
    "bf16_tuned_batched_gemm": "AITER_CONFIG_BF16_BATCHED_GEMM",
    "a8w8_tuned_batched_gemm": "AITER_CONFIG_A8W8_BATCHED_GEMM",
    "tuned_fmoe": "AITER_CONFIG_FMOE",
}


@dataclass(frozen=True, slots=True)
class RestorePlan:
    """Output of ``plan_restore``: did we light up restore, and from where."""

    enabled: bool
    source: Path | None
    reason: str


def plan_restore(
    *,
    hf_home: Path | None,
    gpu: GpuAgent | None,
) -> RestorePlan:
    """Decide whether to restore tuned configs for this launch.

    Pure function. Same prerequisites as ``plan_capture``: ROCm GPU
    (else there's no AITER to feed) and a resolved HF cache (else no
    source).
    """
    if gpu is None:
        return RestorePlan(enabled=False, source=None, reason=REASON_NO_GPU)
    if hf_home is None:
        return RestorePlan(enabled=False, source=None, reason=REASON_NO_HF_HOME)
    source = hf_home / "vllm-shim" / "aiter-configs" / bucket(gpu)
    return RestorePlan(enabled=True, source=source, reason=REASON_ENABLED)


def restore_configs(plan: RestorePlan) -> dict[str, str]:
    """Build the env-var overrides that point AITER at our tuned configs.

    Returns a mapping of ``AITER_CONFIG_*`` env var to the absolute CSV
    path under ``plan.source``. The caller merges this into the backend
    environment before spawning, and AITER picks it up at import time.

    A target file present in ``plan.source`` but not in our known
    mapping is skipped (likely a new tuned-config kind that the shim
    doesn't recognise yet); the operator can drop it in with a manual
    env var if needed.

    Idempotent and side-effect free: the function does no filesystem
    writes, only reads ``plan.source`` for a directory listing.
    """
    if not plan.enabled or plan.source is None or not plan.source.is_dir():
        return {}
    overrides: dict[str, str] = {}
    for f in sorted(plan.source.iterdir()):
        if not f.is_file():
            continue
        env_var = _TARGET_ENV.get(f.stem)
        if env_var is None:
            continue
        overrides[env_var] = str(f)
    return overrides
