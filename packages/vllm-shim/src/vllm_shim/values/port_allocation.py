"""Port allocation: derive backend and middleware ports from the public listen port."""

from dataclasses import dataclass
from typing import Self


@dataclass(frozen=True, slots=True)
class PortAllocation:
    """Three-port layout: public listener (N), backend (N+1), middleware (N+2)."""

    frontend: int
    backend: int
    middleware: int

    @classmethod
    def from_listen(cls, port: int) -> Self:
        """Allocate backend and middleware ports as offsets from the public listener."""
        return cls(frontend=port, backend=port + 1, middleware=port + 2)
