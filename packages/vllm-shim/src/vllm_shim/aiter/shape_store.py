"""Persist captured AITER shapes to CSV with dedup.

One CSV per AITER target (the basename AITER itself looks up, e.g.
``bf16_tuned_gemm.csv``). The directory the store writes into is
supplied by the caller; it encodes hardware bucket + model +
parallelism in its path so files can't cross-pollinate between
incompatible setups.

Column order and naming match AITER's own ``save_shapes`` writer in
``repos/aiter/aiter/tuned_gemm.py`` so the captured CSV is a drop-in
``--untune_file`` for the per-target tune scripts (``M, N, K, bias,
dtype, outdtype, scaleAB, bpreshuffle``). The shape-key columns the
tuner actually cares about are ``M, N, K`` (and ``B`` for batched
variants we don't currently produce); the rest are descriptive and
identify the kernel family.

Dedup is double-bucketed: an in-memory set per target prevents repeat
writes during this process, and on first write to a target the existing
CSV (if any) is loaded into that set so restarts converge on a deduped
file. Concurrent writers on a shared persistent volume could in theory
produce duplicate rows; we accept that and document it.
"""

import csv
from dataclasses import dataclass
from pathlib import Path

from vllm_shim.aiter.log_parser import AiterShape

# Column order is the contract with the AITER tuner. ``outdtype`` (not
# ``otype``) is intentional - that's what AITER's CSV schema uses, even
# though the runtime log line spells the same value as ``otype=``.
_HEADER: tuple[str, ...] = (
    "M",
    "N",
    "K",
    "bias",
    "dtype",
    "outdtype",
    "scaleAB",
    "bpreshuffle",
)

_ShapeKey = tuple[int, int, int, bool, str, str, bool, bool]


def padded_m_gl0(m: int) -> int:
    """Bucket ``m`` onto AITER's fine-grained tuning grid.

    Ports ``getPaddedM(M, N, K, gl=0)`` from
    ``repos/aiter/csrc/py_itfs_cu/gemm_common.cu``. AITER's runtime
    pads M into a discrete bucket before its tuned-config lookup:

      M <= 256:    round up to nearest multiple of 16
      M <= 1024:   round up to nearest multiple of 32
      M <= 4096:   round up to nearest multiple of 64
      M >  4096:   round up to nearest multiple of 128

    Continuous batching produces a different raw M on essentially
    every step; capturing the raw values would bloat the shape CSVs
    without giving the tuner anything new to do, because AITER's
    runtime won't query the tuned config at an off-grid M. Bucketing
    at capture time collapses thousands of raw values into ~180
    grid values per (N, K) pair.

    Keep in lockstep with AITER's C source: if AMD changes the
    padding regime in a future ROCm release, the comment + the
    constants below need to follow.
    """
    if m <= 0:
        return m
    if m <= 256:
        return (m + 15) // 16 * 16
    if m <= 1024:
        return (m + 31) // 32 * 32
    if m <= 4096:
        return (m + 63) // 64 * 64
    return (m + 127) // 128 * 128


def _key(shape: AiterShape) -> _ShapeKey:
    return (
        padded_m_gl0(shape.m),
        shape.n,
        shape.k,
        shape.bias,
        shape.dtype,
        shape.outdtype,
        shape.scale_ab,
        shape.bpreshuffle,
    )


def _row(shape: AiterShape) -> list[str]:
    return [
        str(padded_m_gl0(shape.m)),
        str(shape.n),
        str(shape.k),
        str(shape.bias),
        shape.dtype,
        shape.outdtype,
        str(shape.scale_ab),
        str(shape.bpreshuffle),
    ]


def _row_to_key(row: dict[str, str]) -> _ShapeKey:
    # ``padded_m_gl0`` here makes the in-memory dedup set consistent
    # with the bucketing rule even when the on-disk CSV still carries
    # raw M values from a pre-bucketing capture run. New writes go
    # through ``_key``/``_row`` which also pad, so disk converges to
    # canonical as new shapes accumulate; the formal cleanup of any
    # remaining raw M rows happens in ``tune.canonicalize_shape_files``.
    return (
        padded_m_gl0(int(row["M"])),
        int(row["N"]),
        int(row["K"]),
        row["bias"] == "True",
        row["dtype"],
        row["outdtype"],
        row["scaleAB"] == "True",
        row["bpreshuffle"] == "True",
    )


@dataclass(slots=True)
class ShapeStore:
    """Writer that appends a deduped CSV row per AITER shape."""

    root: Path
    _seen: dict[str, set[_ShapeKey]] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self._seen = {}

    def add(self, shape: AiterShape) -> bool:
        """Append ``shape`` to its target CSV if unseen.

        Returns True when a new row was written, False when the shape
        was already on file (in memory or on disk).
        """
        path = self.root / f"{shape.target}.csv"
        seen = self._seen_for(path, shape.target)
        key = _key(shape)
        if key in seen:
            return False
        seen.add(key)
        self._append(path, shape)
        return True

    def _seen_for(self, path: Path, target: str) -> set[_ShapeKey]:
        cached = self._seen.get(target)
        if cached is not None:
            return cached
        existing: set[_ShapeKey] = set()
        if path.exists():
            with path.open(newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        existing.add(_row_to_key(row))
                    except (KeyError, ValueError):
                        # Corrupt row in an existing file shouldn't
                        # stop dedup from working for the rest.
                        continue
        self._seen[target] = existing
        return existing

    def _append(self, path: Path, shape: AiterShape) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not path.exists()
        with path.open("a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(_HEADER)
            writer.writerow(_row(shape))
