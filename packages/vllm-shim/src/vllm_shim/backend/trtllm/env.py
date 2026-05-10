"""TRT-LLM EnvTranslator: rename a small set of vLLM env vars to their TRT-LLM counterparts.

The set is deliberately conservative: only env vars where the underlying
concept is identical and the names are clearly aligned (cross-checked against
``TLLM_*`` / ``TRTLLM_*`` env names referenced in repos/TensorRT-LLM/). The
remaining vLLM env vars are inherited by the backend subprocess as-is and
ignored - trtllm-serve doesn't trip over names it doesn't recognise.

If the user has already set the TRT-LLM-side name explicitly, this translator
does NOT overwrite it (operator intent wins over auto-translation).

ROCm note: TRT-LLM has zero ROCm/AITER env vars (CUDA-only), so no ROCm
translations apply here.
"""

from collections.abc import Mapping

from vllm_shim.backend._shared import translate_env_with_map
from vllm_shim.backend.base.env import EnvTranslator

# vLLM env var name => TRT-LLM env var name.
ENV_MAP: dict[str, str] = {
    "VLLM_ALLOW_LONG_MAX_MODEL_LEN": "TLLM_ALLOW_LONG_MAX_MODEL_LEN",
    "VLLM_NO_USAGE_STATS": "TRTLLM_NO_USAGE_STATS",
    "VLLM_RAY_BUNDLE_INDICES": "TRTLLM_RAY_BUNDLE_INDICES",
    "VLLM_RAY_PER_WORKER_GPUS": "TRTLLM_RAY_PER_WORKER_GPUS",
    "VLLM_USAGE_STATS_SERVER": "TRTLLM_USAGE_STATS_SERVER",
}


class TRTLLMEnvTranslator(EnvTranslator):
    """Renames vLLM env vars into TRT-LLM-side names per ENV_MAP."""

    def translate(self, parent_env: Mapping[str, str]) -> dict[str, str]:
        return translate_env_with_map(parent_env, ENV_MAP)
