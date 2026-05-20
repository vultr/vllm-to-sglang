"""vLLM EnvTranslator: identity pass-through.

``VLLM_*`` env vars are exactly what vLLM consumes natively, so there is
nothing to rename. The empty ``ENV_MAP`` keeps the shape symmetric with
SGLang and TRT-LLM, both of which use the same helper.
"""

from collections.abc import Mapping

from vllm_shim.backend._shared import translate_env_with_map
from vllm_shim.backend.base.env import EnvTranslator

ENV_MAP: dict[str, str] = {}


class VLLMEnvTranslator(EnvTranslator):
    """Identity translator: parent env is returned unchanged."""

    def translate(self, parent_env: Mapping[str, str]) -> dict[str, str]:
        return translate_env_with_map(parent_env, ENV_MAP)
