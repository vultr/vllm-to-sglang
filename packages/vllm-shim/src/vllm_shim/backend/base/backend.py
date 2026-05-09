"""Backend ABC: the four-component contract every concrete backend implements."""

from abc import ABC, abstractmethod
from typing import ClassVar

from vllm_shim.backend.base.args import ArgTranslator
from vllm_shim.backend.base.filter import RequestFilter
from vllm_shim.backend.base.launcher import Launcher
from vllm_shim.backend.base.metrics import MetricsTranslator


class Backend(ABC):
    """Contract for a serving backend. Concrete subclasses set the four
    component attributes in __init__ and override the ClassVars."""

    name: ClassVar[str]
    health_path: ClassVar[str] = "/health"
    metrics_path: ClassVar[str] = "/metrics"

    args: ArgTranslator
    metrics: MetricsTranslator
    launcher: Launcher
    filters: tuple[RequestFilter, ...]

    @abstractmethod
    def __init__(self) -> None: ...
