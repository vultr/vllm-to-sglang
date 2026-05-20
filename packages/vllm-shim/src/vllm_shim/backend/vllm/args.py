"""vLLM ArgTranslator: identity pass-through.

The shim's input CLI *is* the vLLM CLI, so every passthrough flag is already
a vLLM flag and should reach vLLM unchanged. The empty ``ARG_MAP`` keeps the
``(translated, dropped)`` return shape consistent with the SGLang and TRT-LLM
translators, so callers (entrypoint, info dump) need no special casing for
the identity backend.
"""

from collections.abc import Sequence

from vllm_shim.backend._shared import translate_with_arg_map
from vllm_shim.backend.base.args import ArgTranslator

ARG_MAP: dict[str, tuple[str | None, bool]] = {}


class VLLMArgTranslator(ArgTranslator):
    """Identity translator: every flag passes through verbatim."""

    def translate(self, vllm_args: Sequence[str]) -> tuple[list[str], list[str]]:
        return translate_with_arg_map(list(vllm_args), ARG_MAP)
