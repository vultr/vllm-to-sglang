"""Parse AITER's "not found tuned config" stderr lines.

AITER (the AMD kernel library SGLang uses on ROCm) emits a line for
every GEMM shape it can't find a tuned config for. The canonical form
comes from ``repos/aiter/aiter/tuned_gemm.py``::

    shape is M:1024, N:7168, K:512 dtype='torch.bfloat16' \
otype='torch.bfloat16' bias=False, scaleAB=False, bpreshuffle=False, \
not found tuned config in <path-to>/bf16_tuned_gemm.csv, will use \
default config!

The optional ``[aiter]`` prefix depends on whichever logging formatter
sits in front of AITER's ``logging.getLogger("aiter")``; we match the
inner anchor so the parser works regardless.

Quoting gotcha: AITER builds this line via Python's ``f"{dtype=}"``
syntax, which is ``f"dtype={dtype!r}"`` under the hood. When ``dtype``
is a string (as it is in current AITER), repr wraps it in quotes
(``dtype='torch.bfloat16'``). Older AITER versions passed a real
``torch.dtype`` and produced unquoted output. The parser strips
leading/trailing ASCII quotes from dtype/outdtype so both forms map
to the same canonical CSV value. Persisting the literal quoted form
poisoned the tuner downstream, which interpreted ``'torch.bfloat16'``
as a device string.

Field naming gotcha: AITER's log line uses ``otype=`` but its CSV
schema (and the tuner) uses ``outdtype`` for the same value. We follow
the CSV / tuner spelling internally so captured shapes feed directly
into the tuner without column renaming.

This module captures one shape per matching line so a later tuning run
can generate the missing entries. Writing is the next layer's problem.
"""

import re
from dataclasses import dataclass
from pathlib import PurePosixPath

# Both halves of the message must be present for this to be the
# shape-not-found event we care about; other lines from the aiter
# logger (e.g. init banners) should not match. We intentionally don't
# require the shape fields to appear in any particular order in the
# anchor; the per-field regexes below catch a truncated line by
# returning None for any missing field.
_ANCHOR_RE = re.compile(
    r"shape is .* not found tuned config in (?P<path>\S+\.csv)"
)

# Each field is extracted independently so reordering or new fields in
# future AITER versions doesn't break the parser. The log line spells
# the output dtype as ``otype=`` even though AITER's CSV writes it as
# ``outdtype``; we parse it into the CSV name to stay aligned with the
# tuner.
_FIELDS_RE: dict[str, re.Pattern[str]] = {
    "m": re.compile(r"\bM:(\d+)"),
    "n": re.compile(r"\bN:(\d+)"),
    "k": re.compile(r"\bK:(\d+)"),
    "dtype": re.compile(r"\bdtype=(\S+?)[\s,]"),
    "outdtype": re.compile(r"\botype=(\S+?)[\s,]"),
    "bias": re.compile(r"\bbias=(True|False)\b"),
    "scale_ab": re.compile(r"\bscaleAB=(True|False)\b"),
    "bpreshuffle": re.compile(r"\bbpreshuffle=(True|False)\b"),
}


def _unquote(value: str) -> str:
    """Strip a single layer of matching ASCII quotes around a captured value.

    AITER's f-string interpolation wraps string values in repr quotes
    (``dtype='torch.bfloat16'``). Storing the literal quoted form into
    the CSV poisons the tuner, which then interprets the quoted string
    as a malformed dtype/device. We strip one matched pair only; values
    without quotes pass through unchanged.
    """
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        return value[1:-1]
    return value


@dataclass(frozen=True, slots=True)
class AiterShape:
    """A single GEMM shape AITER couldn't find a tuned config for.

    ``target`` is the bare CSV stem (e.g. ``bf16_tuned_gemm``), derived
    from whatever path the log line names. The directory part is
    operator-configurable via ``AITER_CONFIG_*`` env vars; only the
    basename is meaningful as a target identifier.
    """

    m: int
    n: int
    k: int
    dtype: str
    outdtype: str
    bias: bool
    scale_ab: bool
    bpreshuffle: bool
    target: str


def parse_line(line: str) -> AiterShape | None:
    """Return the shape if ``line`` is an AITER shape-not-found message.

    Returns None for any other line, and also for AITER lines where the
    anchor matches but one or more shape fields are missing (defensive
    against a future AITER that changes the message format).
    """
    anchor = _ANCHOR_RE.search(line)
    if not anchor:
        return None
    fields: dict[str, str] = {}
    for name, pat in _FIELDS_RE.items():
        match = pat.search(line)
        if not match:
            return None
        fields[name] = match.group(1)
    return AiterShape(
        m=int(fields["m"]),
        n=int(fields["n"]),
        k=int(fields["k"]),
        dtype=_unquote(fields["dtype"]),
        outdtype=_unquote(fields["outdtype"]),
        bias=fields["bias"] == "True",
        scale_ab=fields["scale_ab"] == "True",
        bpreshuffle=fields["bpreshuffle"] == "True",
        target=PurePosixPath(anchor.group("path")).stem,
    )
