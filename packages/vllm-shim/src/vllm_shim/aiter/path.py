"""Filesystem layout for captured AITER shapes.

The captured-shape directory is keyed by every dimension that can change
which tuned configs are valid: GPU SKU bucket, model, parallel topology.
Concretely:

    <hf_home>/vllm-shim/aiter-shapes/<bucket>/<model>/<parallelism>/<target>.csv

Where ``<bucket>`` comes from ``cli.rocm_probe.bucket`` (``gfx942-304cu``),
``<parallelism>`` from ``Parallelism.path_segment`` (``tp8``, ``tp8-ep8``),
and ``<target>`` is the AITER CSV stem (``bf16_gemm``, ``fp8_blockscale_gemm``).
``ShapeStore`` writes the per-target file inside the returned root.

The layout sits under ``$HF_HOME`` so it survives across pod restarts on
the same persistent volume used for the model cache; a future
``vllm-shim-tune`` subcommand can walk the tree and feed each leaf into
the AITER tuner.
"""

from pathlib import Path

from vllm_shim.values.parallelism import Parallelism

_ROOT_SUBDIR = ("vllm-shim", "aiter-shapes")


def sanitize_model(model: str) -> str:
    """Filesystem-safe segment for a model identifier.

    Two input shapes need to round-trip cleanly:

    - HF repo IDs like ``moonshotai/Kimi-K2.6``: collapse ``/`` to ``--``
      (matches the HF cache directory convention sans the ``models--``
      prefix). Produces ``moonshotai--Kimi-K2.6``.
    - Local paths like ``/data/models/foo`` or ``./foo``: use the
      basename. The full path is operator-specific and not useful as a
      stable bucket key.

    Falls back to ``unknown-model`` only when both strategies produce an
    empty string (e.g. the input was just ``/``); a missing identifier
    earlier in the pipeline is a bug, not something to silently mask.
    """
    # Treat any leading ``/`` as a POSIX-style absolute path regardless of
    # host OS. ``Path.is_absolute`` is False on Windows for ``/data/...``,
    # but our deployments are POSIX containers and the typical operator
    # input is ``/data/models/foo``; we don't want test behaviour to
    # diverge between dev hosts.
    p = Path(model)
    if p.is_absolute() or model.startswith(("/", "./", "../")):
        name = p.name
    else:
        name = model.strip("/").replace("/", "--")
    return name or "unknown-model"


def shape_capture_root(
    hf_home: Path,
    bucket: str,
    model: str,
    parallelism: Parallelism,
) -> Path:
    """Directory where ``ShapeStore`` will land per-target CSVs."""
    return (
        hf_home
        / _ROOT_SUBDIR[0]
        / _ROOT_SUBDIR[1]
        / bucket
        / sanitize_model(model)
        / parallelism.path_segment()
    )
