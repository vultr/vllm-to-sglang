"""SGLangBackend: wires together the SGLang-specific args, launcher, metrics, and filters."""

from typing import ClassVar

from vllm_shim.backend.base.backend import Backend
from vllm_shim.backend.sglang.args import SGLangArgTranslator
from vllm_shim.backend.sglang.filter.fix_schema import FixToolSchemas
from vllm_shim.backend.sglang.filter.strip_params import StripVLLMParams
from vllm_shim.backend.sglang.launcher import SGLangLauncher
from vllm_shim.backend.sglang.metrics import SGLangMetricsTranslator


class SGLangBackend(Backend):
    """The SGLang implementation of the Backend contract."""

    name: ClassVar[str] = "sglang"

    def __init__(self) -> None:
        self.args = SGLangArgTranslator()
        self.metrics = SGLangMetricsTranslator()
        self.launcher = SGLangLauncher()
        # Strip first, then fix schemas: stripping removes keys the schema fixer would walk.
        self.filters = (StripVLLMParams(), FixToolSchemas())
