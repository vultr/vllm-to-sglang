from dataclasses import dataclass
from typing import Self


@dataclass(frozen=True, slots=True)
class PortAllocation:
    frontend: int
    backend: int
    middleware: int

    @classmethod
    def from_listen(cls, port: int) -> Self:
        return cls(frontend=port, backend=port + 1, middleware=port + 2)
