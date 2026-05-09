from abc import ABC, abstractmethod


class RequestFilter(ABC):
    """Mutates a forwarded request body before it reaches the backend."""

    @abstractmethod
    def applies_to(self, method: str, path: str) -> bool: ...

    @abstractmethod
    def transform(self, body: bytes) -> bytes: ...
