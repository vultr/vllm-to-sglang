"""ParsedArgs: structured output of vLLM CLI parsing."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ParsedArgs:
    """Output of vLLM CLI parsing. `passthrough` carries every flag and value
    the parser did not consume internally, in original order, ready to feed
    into a backend's ArgTranslator."""

    model: str
    host: str
    port: int
    passthrough: tuple[str, ...]
