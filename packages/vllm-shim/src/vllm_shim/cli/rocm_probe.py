"""Hardware probing via ``rocminfo``.

AITER tuned configs are GPU-SKU specific: same gfx target can ship with
different CU counts, and a kernel tuned for one CU count will mis-perform
on another. The bucket key must capture both.

The shim's venv is Python 3.12 with no torch installed (SGLang's ROCm
venv is a separate 3.10 environment), so we shell out to ``rocminfo``
instead of importing a Python device-properties API.
"""

import re
import subprocess
from dataclasses import dataclass

# Agent blocks are delimited by lines like "Agent 1", "Agent 2", ...
# surrounded by "*******" rules. Splitting on the "Agent N" header is
# sufficient; the leading rule line lands at the end of the previous
# chunk and is ignored.
_AGENT_HEADER_RE = re.compile(r"^Agent \d+\s*$", re.MULTILINE)


@dataclass(frozen=True, slots=True)
class GpuAgent:
    """A single GPU entry parsed out of ``rocminfo``."""

    gfx_target: str
    compute_units: int
    marketing_name: str


def parse_rocminfo(text: str) -> list[GpuAgent]:
    """Extract every GPU agent from a ``rocminfo`` capture.

    CPU agents (Intel/AMD host processors) are dropped: they also have a
    ``Compute Unit:`` field, so the GPU check must be on ``Device Type:``.
    """
    agents: list[GpuAgent] = []
    for block in _AGENT_HEADER_RE.split(text)[1:]:
        if not re.search(r"^\s*Device Type:\s+GPU\b", block, re.MULTILINE):
            continue
        name_m = re.search(r"^\s*Name:\s+(\S+)", block, re.MULTILINE)
        cu_m = re.search(r"^\s*Compute Unit:\s+(\d+)", block, re.MULTILINE)
        mkt_m = re.search(r"^\s*Marketing Name:\s+(.+?)\s*$", block, re.MULTILINE)
        if not (name_m and cu_m):
            continue
        agents.append(
            GpuAgent(
                gfx_target=name_m.group(1),
                compute_units=int(cu_m.group(1)),
                marketing_name=mkt_m.group(1) if mkt_m else "",
            )
        )
    return agents


def probe() -> GpuAgent | None:
    """Run ``rocminfo`` and return the first GPU agent, or None on failure.

    Returns None when ``rocminfo`` is not on PATH (CUDA hosts, dev boxes)
    or when the command exits non-zero. Callers should treat None as
    "hardware bucket unknown" and fall back to a generic path.
    """
    try:
        result = subprocess.run(
            ["rocminfo"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (subprocess.SubprocessError, FileNotFoundError):
        return None
    agents = parse_rocminfo(result.stdout)
    return agents[0] if agents else None


def bucket(agent: GpuAgent) -> str:
    """Return the path segment identifying this GPU SKU, e.g. ``gfx942-304cu``."""
    return f"{agent.gfx_target}-{agent.compute_units}cu"
