"""SGLangLauncher: builds the SGLang server argv with a `python -m` fallback."""

import os
import shutil
import sys
from collections.abc import Sequence

from vllm_shim.backend.base.launcher import Launcher
from vllm_shim.values.service_address import ServiceAddress

DEFAULT_TOOL_CALL_PARSER = "qwen3_coder"


class SGLangLauncher(Launcher):
    """Builds the SGLang server command line; honors SGLANG_TOOL_CALL_PARSER override."""

    def build_command(
        self,
        model: str,
        address: ServiceAddress,
        extra_args: Sequence[str],
    ) -> list[str]:
        parser = os.environ.get("SGLANG_TOOL_CALL_PARSER", DEFAULT_TOOL_CALL_PARSER)
        return [
            *self._invocation(),
            "--model-path", model,
            "--host", address.host,
            "--port", str(address.port),
            "--enable-metrics",
            "--tool-call-parser", parser,
            *extra_args,
        ]

    @staticmethod
    def _invocation() -> list[str]:
        # Prefer the `sglang` console script when present. The CUDA SGLang
        # image installs via legacy `setup.py develop`, which writes
        # entry-point metadata but no wrapper script; in that case fall back
        # to the module form. The fallback assumes vllm-shim and SGLang share
        # an interpreter (true on the CUDA image, where both live in
        # /usr/bin/python); the ROCm image ships the wrapper at
        # /opt/venv/bin/sglang, whose shebang routes execution back into
        # SGLang's own venv regardless of what sys.executable is here.
        if shutil.which("sglang"):
            return ["sglang", "serve"]
        return [sys.executable, "-m", "sglang.launch_server"]
