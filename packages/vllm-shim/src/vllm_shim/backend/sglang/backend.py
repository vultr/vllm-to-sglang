from typing import ClassVar

from vllm_shim.backend.base.backend import Backend
from vllm_shim.backend.sglang.args import SGLangArgTranslator
from vllm_shim.backend.sglang.filter.fix_schema import FixToolSchemas
from vllm_shim.backend.sglang.filter.strip_params import StripVLLMParams
from vllm_shim.backend.sglang.launcher import SGLangLauncher
from vllm_shim.backend.sglang.metrics import SGLangMetricsTranslator


class SGLangBackend(Backend):
    name: ClassVar[str] = "sglang"

    def __init__(self) -> None:
        self.args = SGLangArgTranslator()
        self.metrics = SGLangMetricsTranslator()
        self.launcher = SGLangLauncher()
        self.filters = (StripVLLMParams(), FixToolSchemas())
