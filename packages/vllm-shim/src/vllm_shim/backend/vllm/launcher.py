"""VLLMLauncher: builds the vLLM API server argv via ``python -m``.

The shim's ``vllm`` console script and the ``vllm-entrypoints`` namespace stub
both occupy the surface real vLLM would normally claim. Two consequences:

- ``shutil.which("vllm")`` would resolve to the shim's hijack script, so we
  cannot use the wrapper-script fallback that SGLang's launcher uses.
- Invoking ``python -m vllm.entrypoints.openai.api_server`` from the shim's
  own interpreter would hit the ``vllm-entrypoints`` stub and bounce back
  into the shim entrypoint, looping forever.

The Dockerfiles solve this by installing vllm-shim into its own ``/opt/shim``
venv. Real vLLM lives in the base image's Python (``/usr/bin/python3`` on
both ``vllm/vllm-openai`` and ``rocm/vllm-dev:base``), so we launch via that
interpreter by absolute path. ``VLLM_BACKEND_PYTHON`` lets operators retarget
when the upstream image moves its interpreter.
"""

import os
from collections.abc import Sequence

from vllm_shim.backend.base.launcher import Launcher
from vllm_shim.values.service_address import ServiceAddress

DEFAULT_TOOL_CALL_PARSER = "qwen3_coder"
DEFAULT_BACKEND_PYTHON = "/usr/bin/python3"


class VLLMLauncher(Launcher):
    """Builds the vLLM API server command line.

    Honors VLLM_BACKEND_PYTHON (interpreter path) and VLLM_TOOL_CALL_PARSER overrides.
    """

    def build_command(
        self,
        model: str,
        address: ServiceAddress,
        extra_args: Sequence[str],
    ) -> list[str]:
        python = os.environ.get("VLLM_BACKEND_PYTHON", DEFAULT_BACKEND_PYTHON)
        parser = os.environ.get("VLLM_TOOL_CALL_PARSER", DEFAULT_TOOL_CALL_PARSER)
        return [
            python,
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            model,
            "--host",
            address.host,
            "--port",
            str(address.port),
            "--tool-call-parser",
            parser,
            *extra_args,
        ]
