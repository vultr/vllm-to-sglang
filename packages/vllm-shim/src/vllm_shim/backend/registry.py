"""Backend registry: maps env-var names to Backend classes."""

import os

from vllm_shim.backend.base.backend import Backend
from vllm_shim.backend.sglang.backend import SGLangBackend

_BACKENDS: dict[str, type[Backend]] = {
    "sglang": SGLangBackend,
}


def select() -> Backend:
    """Returns the Backend named by VLLM_SHIM_BACKEND (default 'sglang')."""
    name = os.environ.get("VLLM_SHIM_BACKEND", "sglang")
    cls = _BACKENDS.get(name)
    if cls is None:
        raise ValueError(
            f"Unknown backend: {name!r}. Known: {sorted(_BACKENDS)}"
        )
    return cls()
