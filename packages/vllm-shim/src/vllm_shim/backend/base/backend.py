"""Backend ABC: the five-component contract every concrete backend implements."""

from abc import ABC, abstractmethod
from typing import ClassVar

from vllm_shim.backend.base.args import ArgTranslator
from vllm_shim.backend.base.env import EnvTranslator
from vllm_shim.backend.base.filter import RequestFilter
from vllm_shim.backend.base.launcher import Launcher
from vllm_shim.backend.base.metrics import MetricsTranslator
from vllm_shim.backend.base.parallelism import ParallelismExtractor


class Backend(ABC):
    """Contract for a serving backend. Concrete subclasses set the six
    component attributes in __init__ and override the ClassVars."""

    name: ClassVar[str]
    health_path: ClassVar[str] = "/health"
    metrics_path: ClassVar[str] = "/metrics"

    args: ArgTranslator
    env: EnvTranslator
    metrics: MetricsTranslator
    launcher: Launcher
    filters: tuple[RequestFilter, ...]
    parallelism: ParallelismExtractor

    @abstractmethod
    def __init__(self) -> None: ...
