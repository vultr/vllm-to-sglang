"""Launch-time info dump.

Captures everything the entrypoint decides at startup (translated argv,
dropped args, env renames, port allocation, model resolution) and writes
it as JSON to a known path so operators can ``vllm-shim-info`` from a
shell inside the pod. Also prints a short summary to stderr so the same
data is visible in pod logs without exec'ing in.
"""

import json
import sys
from collections.abc import Mapping, Sequence
from importlib import metadata
from pathlib import Path
from typing import Any

from vllm_shim.values.port_allocation import PortAllocation
from vllm_shim.values.service_address import ServiceAddress

INFO_PATH = Path("/tmp/vllm-shim-info.json")

# Knobs surfaced explicitly in the JSON. Mirrors the Configuration surface
# table in CLAUDE.md; when adding a new env knob there, add it here too so
# operators see it in the launch dump.
_SHIM_CONFIG_KEYS = (
    "VLLM_SHIM_BACKEND",
    "VLLM_SHIM_LOG",
    "SGLANG_TOOL_CALL_PARSER",
    "TRTLLM_BACKEND",
    "TRTLLM_TOOL_PARSER",
    "TRTLLM_REASONING_PARSER",
)

# Hugging Face cache state, useful when a snapshot_download misbehaves.
_HF_KEYS = ("HF_HOME", "HF_HUB_OFFLINE", "HF_HUB_CACHE", "TRANSFORMERS_CACHE")


def shim_version() -> str:
    """Return the installed vllm-shim distribution version, or 'unknown'."""
    try:
        return metadata.version("vllm-shim")
    except metadata.PackageNotFoundError:
        return "unknown"


def collect(
    *,
    original_argv: Sequence[str],
    backend_name: str,
    model_original: str,
    model_resolved: str,
    revision: str | None,
    listen: ServiceAddress,
    ports: PortAllocation,
    backend_argv: Sequence[str],
    dropped_args: Sequence[str],
    parent_env: Mapping[str, str],
    backend_env: Mapping[str, str],
) -> dict[str, Any]:
    """Assemble the info dict from already-decided launch state."""
    env_translation = {k: v for k, v in backend_env.items() if k not in parent_env}
    shim_config = {k: parent_env[k] for k in _SHIM_CONFIG_KEYS if k in parent_env}
    hf_cache = {k: parent_env[k] for k in _HF_KEYS if k in parent_env}
    return {
        "shim_version": shim_version(),
        "backend": backend_name,
        "original_argv": list(original_argv),
        "model": {
            "original": model_original,
            "resolved": model_resolved,
            "revision": revision,
        },
        "listen": str(listen),
        "ports": {
            "frontend": ports.frontend,
            "backend": ports.backend,
            "middleware": ports.middleware,
        },
        "backend_argv": list(backend_argv),
        "dropped_args": list(dropped_args),
        "env_translation": env_translation,
        "shim_config": shim_config,
        "hf_cache": hf_cache,
    }


def write(info: dict[str, Any], path: Path = INFO_PATH) -> None:
    """Serialize the info dict as pretty JSON with a trailing newline."""
    path.write_text(json.dumps(info, indent=2, sort_keys=True) + "\n")


def print_summary(info: dict[str, Any]) -> None:
    """Write a multi-line summary of the launch info to stderr."""
    model = info["model"]
    rev = f"@{model['revision']}" if model["revision"] else ""
    sys.stderr.write(
        f"vllm-shim {info['shim_version']} -> {info['backend']} "
        f"listening on {info['listen']}\n"
        f"  model: {model['original']}{rev} -> {model['resolved']}\n"
        f"  backend argv: {' '.join(info['backend_argv'])}\n"
    )
    if info["dropped_args"]:
        sys.stderr.write(f"  dropped: {' '.join(info['dropped_args'])}\n")
    if info["env_translation"]:
        renames = ", ".join(f"{k}={v}" for k, v in info["env_translation"].items())
        sys.stderr.write(f"  env renames: {renames}\n")


def main() -> int:
    """vllm-shim-info console script: print the JSON written by the entrypoint."""
    if not INFO_PATH.exists():
        sys.stderr.write(
            f"No launch info at {INFO_PATH}; the shim may not have started yet.\n"
        )
        return 1
    sys.stdout.write(INFO_PATH.read_text())
    return 0
