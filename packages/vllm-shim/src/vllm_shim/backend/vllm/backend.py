"""VLLMBackend: identity-shaped wiring for the vLLM backend.

Every component is the identity over the shim's input contract (args, env,
metrics) or trivially shaped (empty filter tuple, native parallel flags),
because vLLM *is* the input CLI's target. Health and metrics paths inherit
the ABC defaults (``/health``, ``/metrics``).
"""

from typing import ClassVar

from vllm_shim.backend.base.backend import Backend
from vllm_shim.backend.vllm.args import VLLMArgTranslator
from vllm_shim.backend.vllm.env import VLLMEnvTranslator
from vllm_shim.backend.vllm.launcher import VLLMLauncher
from vllm_shim.backend.vllm.metrics import VLLMMetricsTranslator
from vllm_shim.backend.vllm.parallelism import VLLMParallelismExtractor


class VLLMBackend(Backend):
    """The vLLM implementation of the Backend contract."""

    name: ClassVar[str] = "vllm"

    def __init__(self) -> None:
        self.args = VLLMArgTranslator()
        self.env = VLLMEnvTranslator()
        self.metrics = VLLMMetricsTranslator()
        self.launcher = VLLMLauncher()
        self.filters = ()
        self.parallelism = VLLMParallelismExtractor()
