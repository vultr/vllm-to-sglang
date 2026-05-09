from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ServiceAddress:
    host: str
    port: int

    def __str__(self) -> str:
        return f"{self.host}:{self.port}"

    def url(self, scheme: str = "http") -> str:
        return f"{scheme}://{self.host}:{self.port}"
