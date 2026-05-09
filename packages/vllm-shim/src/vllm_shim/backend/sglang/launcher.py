import os
import sys
from collections.abc import Sequence

from vllm_shim.backend.base.launcher import Launcher
from vllm_shim.values.service_address import ServiceAddress

DEFAULT_TOOL_CALL_PARSER = "qwen3_coder"


class SGLangLauncher(Launcher):
    def build_command(
        self,
        model: str,
        address: ServiceAddress,
        extra_args: Sequence[str],
    ) -> list[str]:
        parser = os.environ.get("SGLANG_TOOL_CALL_PARSER", DEFAULT_TOOL_CALL_PARSER)
        return [
            sys.executable,
            "-m",
            "sglang.launch_server",
            "--model-path", model,
            "--host", address.host,
            "--port", str(address.port),
            "--enable-metrics",
            "--tool-call-parser", parser,
            *extra_args,
        ]
