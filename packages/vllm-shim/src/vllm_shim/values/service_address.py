"""ServiceAddress: a (host, port) pair with formatting helpers."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ServiceAddress:
    """Immutable host/port pair. Use `str(addr)` for `host:port`, `addr.url()` for full URL."""

    host: str
    port: int

    def __str__(self) -> str:
        return f"{self.host}:{self.port}"

    def url(self, scheme: str = "http") -> str:
        """Format as a full URL (default scheme `http`)."""
        return f"{scheme}://{self.host}:{self.port}"
