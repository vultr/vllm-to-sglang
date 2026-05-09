"""MetricsTranslator ABC: rewrite a backend's Prometheus exposition into vLLM-named series."""

from abc import ABC, abstractmethod


class MetricsTranslator(ABC):
    """Translates the backend's Prometheus exposition format to vLLM-named metrics."""

    @abstractmethod
    def translate(self, prom_text: str) -> str: ...
