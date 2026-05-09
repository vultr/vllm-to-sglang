"""TRTLLMLauncher: builds the `trtllm-serve` argv."""

import os
from collections.abc import Sequence

from vllm_shim.backend.base.launcher import Launcher
from vllm_shim.values.service_address import ServiceAddress

DEFAULT_BACKEND = "pytorch"
DEFAULT_TOOL_PARSER = "qwen3_coder"


class TRTLLMLauncher(Launcher):
    """Builds the trtllm-serve command line; honors TRTLLM_BACKEND, TRTLLM_TOOL_PARSER,
    and TRTLLM_REASONING_PARSER overrides."""

    def build_command(
        self,
        model: str,
        address: ServiceAddress,
        extra_args: Sequence[str],
    ) -> list[str]:
        backend = os.environ.get("TRTLLM_BACKEND", DEFAULT_BACKEND)
        tool_parser = os.environ.get("TRTLLM_TOOL_PARSER", DEFAULT_TOOL_PARSER)
        reasoning_parser = os.environ.get("TRTLLM_REASONING_PARSER")

        cmd = [
            "trtllm-serve",
            model,
            "--host", address.host,
            "--port", str(address.port),
            "--backend", backend,
            "--tool_parser", tool_parser,
        ]
        if reasoning_parser:
            cmd.extend(["--reasoning_parser", reasoning_parser])
        cmd.extend(list(extra_args))
        return cmd
