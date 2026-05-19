"""Persist captured AITER shapes to CSV with per-shape hit counts.

One CSV per AITER target (the basename AITER itself looks up, e.g.
``bf16_tuned_gemm.csv``). The directory the store writes into is
supplied by the caller; it encodes hardware bucket + model +
parallelism in its path so files can't cross-pollinate between
incompatible setups.

Column order is ``M, N, K, bias, dtype, outdtype, scaleAB, bpreshuffle,
hits``. The first eight columns mirror AITER's own ``save_shapes``
writer in ``repos/aiter/aiter/tuned_gemm.py`` so the downstream
``vllm-shim-tune`` step can read these files by column name and produce
a strict-schema AITER tuner input. The trailing ``hits`` column counts
how many times AITER logged a miss for the shape; it drives the
hot-shape ranking that ``tune.merge_shape_files`` applies via ``--hot``.

Persistence model:

- First sight of a key: increment to 1 and flush the per-target CSV
  immediately, so a crash before the next flush still records the
  shape's existence.
- Repeat sight: increment in memory; flush only when
  ``flush_interval_seconds`` has elapsed since the last write. A hot
  shape that fires thousands of times per second during decode would
  otherwise rewrite the CSV every line; this throttles to roughly one
  rewrite per target per window.

Hit counts read from disk default to 1 when the column is missing or
malformed, so existing CSVs written before the column was introduced
read back as "seen at least once" and round-trip into the new schema
on the next flush.

Concurrent writers on a shared persistent volume could in theory
double-count or lose increments; we accept that and document it.
"""

import csv
import time
from dataclasses import dataclass
from pathlib import Path

from vllm_shim.aiter.log_parser import AiterShape

# Column order is the contract with downstream consumers (tune's merge
# step and the AITER tuner it shells out to). ``outdtype`` (not
# ``otype``) is intentional - that's what AITER's CSV schema uses, even
# though the runtime log line spells the same value as ``otype=``.
# ``hits`` is the shim-side addition that ranks shapes for ``--hot``.
_HEADER: tuple[str, ...] = (
    "M",
    "N",
    "K",
    "bias",
    "dtype",
    "outdtype",
    "scaleAB",
    "bpreshuffle",
    "hits",
)

# Columns that identify a unique shape. ``hits`` is value, not identity.
SHAPE_COLUMNS: tuple[str, ...] = _HEADER[:-1]
HITS_COLUMN: str = "hits"

# Default flush cadence. Picked to bound recent-count loss on crash
# without burning I/O on every increment when a hot shape fires
# thousands of times per second.
DEFAULT_FLUSH_INTERVAL_SECONDS: float = 30.0

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


def _row_from_key(key: _ShapeKey, hits: int) -> list[str]:
    m, n, k, bias, dtype, outdtype, scale_ab, bpreshuffle = key
    return [
        str(m),
        str(n),
        str(k),
        str(bias),
        dtype,
        outdtype,
        str(scale_ab),
        str(bpreshuffle),
        str(hits),
    ]


def parse_hits(value: str | None) -> int:
    """Read a ``hits`` cell with the shim's default-to-1 semantics.

    Missing, empty, non-integer, or non-positive values all map to 1.
    Rationale: a row exists in the CSV iff the shape was seen at least
    once, so 1 is the most conservative non-noisy default and matches
    the value a fresh first-sight write would produce.
    """
    if value is None or value == "":
        return 1
    try:
        n = int(value)
    except ValueError:
        return 1
    return max(1, n)


def _row_to_state(row: dict[str, str]) -> tuple[_ShapeKey, int]:
    """Decode a persisted CSV row into ``(key, hits)``.

    ``padded_m_gl0`` is applied so that any pre-bucketing raw M still
    on disk converges to the canonical bucket on read. Two raw rows
    that collapse onto the same padded bucket sum their hits in the
    caller.
    """
    key: _ShapeKey = (
        padded_m_gl0(int(row["M"])),
        int(row["N"]),
        int(row["K"]),
        row["bias"] == "True",
        row["dtype"],
        row["outdtype"],
        row["scaleAB"] == "True",
        row["bpreshuffle"] == "True",
    )
    return key, parse_hits(row.get(HITS_COLUMN))


@dataclass(slots=True)
class ShapeStore:
    """Append-or-increment writer for one capture directory.

    ``root`` is the per-(bucket, model, parallelism) directory; one
    CSV per AITER target lands inside it. ``flush_interval_seconds``
    throttles the increment-driven rewrites; first-sight writes
    bypass the throttle so a new shape is persisted immediately.
    """

    root: Path
    flush_interval_seconds: float = DEFAULT_FLUSH_INTERVAL_SECONDS
    _state: dict[str, dict[_ShapeKey, int]] = None  # type: ignore[assignment]
    _dirty: set[str] = None  # type: ignore[assignment]
    _last_flush: dict[str, float] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self._state = {}
        self._dirty = set()
        self._last_flush = {}

    def add(self, shape: AiterShape) -> bool:
        """Increment the hit count for ``shape``.

        Returns True when this was the first time the shape was seen
        (in memory or on disk), False on repeat. First-sight writes
        flush immediately; repeat sights flush only when the throttle
        window has elapsed.
        """
        path = self.root / f"{shape.target}.csv"
        counts = self._load_target(path, shape.target)
        key = _key(shape)
        first_sight = key not in counts
        counts[key] = counts.get(key, 0) + 1
        self._dirty.add(shape.target)
        if first_sight or self._should_flush(shape.target):
            self._flush(path, shape.target)
        return first_sight

    def flush(self) -> None:
        """Force-flush every target with pending increments.

        Useful for tests and for any future caller that wants to drain
        pending counts on a clean shutdown signal.
        """
        for target in list(self._dirty):
            path = self.root / f"{target}.csv"
            self._flush(path, target)

    def _load_target(self, path: Path, target: str) -> dict[_ShapeKey, int]:
        cached = self._state.get(target)
        if cached is not None:
            return cached
        counts: dict[_ShapeKey, int] = {}
        if path.exists():
            with path.open(newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        key, hits = _row_to_state(row)
                    except (KeyError, ValueError):
                        # Corrupt row in an existing file shouldn't
                        # stop dedup from working for the rest. The
                        # next flush drops these rows entirely; we
                        # don't preserve unparseable bytes.
                        continue
                    counts[key] = counts.get(key, 0) + hits
        self._state[target] = counts
        return counts

    def _should_flush(self, target: str) -> bool:
        last = self._last_flush.get(target, 0.0)
        return time.monotonic() - last >= self.flush_interval_seconds

    def _flush(self, path: Path, target: str) -> None:
        counts = self._state.get(target)
        if counts is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(_HEADER)
            for key, hits in counts.items():
                writer.writerow(_row_from_key(key, hits))
        tmp.replace(path)
        self._dirty.discard(target)
        self._last_flush[target] = time.monotonic()
