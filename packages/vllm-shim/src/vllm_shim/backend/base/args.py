"""ArgTranslator ABC: rewrite vLLM CLI flags into a backend's CLI."""

from abc import ABC, abstractmethod
from collections.abc import Sequence


class ArgTranslator(ABC):
    """Translates vLLM CLI arguments into the backend's CLI arguments."""

    @abstractmethod
    def translate(self, vllm_args: Sequence[str]) -> tuple[list[str], list[str]]:
        """Returns (backend_args, dropped_args). Pure function, no I/O."""
