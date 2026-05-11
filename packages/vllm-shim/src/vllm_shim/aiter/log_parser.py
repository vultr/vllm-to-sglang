"""Parse AITER's "not found tuned config" stderr lines.

AITER (the AMD kernel library SGLang uses on ROCm) emits a line for
every GEMM shape it can't find a tuned config for:

    [aiter] shape is M:1024, N:7168, K:512 dtype=torch.bfloat16 \
otype=torch.bfloat16 bias=False scaleAB=False bpreshuffle=False, \
not found tuned config in /tmp/aiter_configs/bf16_gemm.csv, will use \
default config!

This module captures one shape per matching line so a later tuning run
can generate the missing entries. Writing is the next layer's problem.
"""

import re
from dataclasses import dataclass
from pathlib import PurePosixPath

# Both halves of the message must be present for this to be the
# shape-not-found event we care about; other lines starting with
# "[aiter]" should not match.
_ANCHOR_RE = re.compile(
    r"\[aiter\] shape is .* not found tuned config in (?P<path>\S+\.csv)"
)

# Each field is extracted independently so reordering or new fields in
# future AITER versions doesn't break the parser.
_FIELDS_RE: dict[str, re.Pattern[str]] = {
    "m": re.compile(r"\bM:(\d+)"),
    "n": re.compile(r"\bN:(\d+)"),
    "k": re.compile(r"\bK:(\d+)"),
    "dtype": re.compile(r"\bdtype=(\S+?)[\s,]"),
    "otype": re.compile(r"\botype=(\S+?)[\s,]"),
    "bias": re.compile(r"\bbias=(True|False)\b"),
    "scale_ab": re.compile(r"\bscaleAB=(True|False)\b"),
    "bpreshuffle": re.compile(r"\bbpreshuffle=(True|False)\b"),
}


@dataclass(frozen=True, slots=True)
class AiterShape:
    """A single GEMM shape AITER couldn't find a tuned config for.

    ``target`` is the bare CSV stem (e.g. ``bf16_gemm``), not the full
    ``/tmp/aiter_configs/...`` path. Callers pick the directory.
    """

    m: int
    n: int
    k: int
    dtype: str
    otype: str
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
        dtype=fields["dtype"],
        otype=fields["otype"],
        bias=fields["bias"] == "True",
        scale_ab=fields["scale_ab"] == "True",
        bpreshuffle=fields["bpreshuffle"] == "True",
        target=PurePosixPath(anchor.group("path")).stem,
    )
