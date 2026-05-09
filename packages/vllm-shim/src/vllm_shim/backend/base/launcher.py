"""Launcher ABC: build the subprocess argv that spawns a backend."""

from abc import ABC, abstractmethod
from collections.abc import Sequence

from vllm_shim.values.service_address import ServiceAddress


class Launcher(ABC):
    """Builds the subprocess argv for spawning the backend."""

    @abstractmethod
    def build_command(
        self,
        model: str,
        address: ServiceAddress,
        extra_args: Sequence[str],
    ) -> list[str]: ...
