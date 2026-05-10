"""TRTLLMBackend: wires together the TRT-LLM-specific args, launcher, and metrics."""

from typing import ClassVar

from vllm_shim.backend.base.backend import Backend
from vllm_shim.backend.trtllm.args import TRTLLMArgTranslator
from vllm_shim.backend.trtllm.env import TRTLLMEnvTranslator
from vllm_shim.backend.trtllm.launcher import TRTLLMLauncher
from vllm_shim.backend.trtllm.metrics import TRTLLMMetricsTranslator


class TRTLLMBackend(Backend):
    """The TensorRT-LLM implementation of the Backend contract."""

    name: ClassVar[str] = "trtllm"
    health_path: ClassVar[str] = "/health"
    metrics_path: ClassVar[str] = "/prometheus/metrics"

    def __init__(self) -> None:
        self.args = TRTLLMArgTranslator()
        self.env = TRTLLMEnvTranslator()
        self.metrics = TRTLLMMetricsTranslator()
        self.launcher = TRTLLMLauncher()
        self.filters = ()
