"""SGLang EnvTranslator: rename a small set of vLLM env vars to their SGLang counterparts.

The set is deliberately conservative: only env vars where the suffix is
identical and the underlying concept is the same in both engines (cross-checked
against repos/sglang/python/sglang/srt/environ.py). Anything else the user sets
on a ``VLLM_*`` env var is simply inherited by the backend subprocess and
ignored - SGLang doesn't trip over env vars it doesn't recognise.

If the user has already set the SGLang-side name explicitly, this translator
does NOT overwrite it (operator intent wins over auto-translation).
"""

from collections.abc import Mapping

from vllm_shim.backend._shared import translate_env_with_map
from vllm_shim.backend.base.env import EnvTranslator

# vLLM env var name => SGLang env var name. See module docstring.
ENV_MAP: dict[str, str] = {
    # Concept-identical, suffix-identical pairs:
    "VLLM_CPU_OMP_THREADS_BIND": "SGLANG_CPU_OMP_THREADS_BIND",
    "VLLM_DP_RANK": "SGLANG_DP_RANK",
    "VLLM_HOST_IP": "SGLANG_HOST_IP",
    "VLLM_LOGGING_CONFIG_PATH": "SGLANG_LOGGING_CONFIG_PATH",
    "VLLM_NCCL_SO_PATH": "SGLANG_NCCL_SO_PATH",
    # Both engines use this as a base port for internal service-port allocation
    # (vllm/envs.py:get_vllm_port + the :578 comment; sglang/srt/utils/network.py:get_open_port).
    # NOT the listen port; that comes from --port on the CLI in both engines.
    "VLLM_PORT": "SGLANG_PORT",
    "VLLM_PP_LAYER_PARTITION": "SGLANG_PP_LAYER_PARTITION",
    "VLLM_RINGBUFFER_WARNING_INTERVAL": "SGLANG_RINGBUFFER_WARNING_INTERVAL",
    "VLLM_SKIP_P2P_CHECK": "SGLANG_SKIP_P2P_CHECK",
    "VLLM_USE_MODELSCOPE": "SGLANG_USE_MODELSCOPE",

    # ROCm-side equivalents. The master AITER toggle has the same semantics in
    # both engines (off by default, set=1 to engage AMD's AITER kernels). The
    # three QUICK_REDUCE_* names are interesting: vLLM exposes them under a
    # VLLM_ROCM_* alias, but SGLang reads the un-prefixed upstream names that
    # AMD's quickreduce library defines. Translating to the un-prefixed form
    # lands the value where SGLang actually looks for it.
    "VLLM_ROCM_USE_AITER": "SGLANG_USE_AITER",
    "VLLM_ROCM_QUICK_REDUCE_CAST_BF16_TO_FP16": "ROCM_QUICK_REDUCE_CAST_BF16_TO_FP16",
    "VLLM_ROCM_QUICK_REDUCE_MAX_SIZE_BYTES_MB": "ROCM_QUICK_REDUCE_MAX_SIZE_BYTES_MB",
    "VLLM_ROCM_QUICK_REDUCE_QUANTIZATION": "ROCM_QUICK_REDUCE_QUANTIZATION",
}

# Notable non-translations and why:
# - Per-feature VLLM_ROCM_USE_AITER_* toggles (MLA, MOE, MHA, RMSNORM, ...):
#   SGLang's per-feature AITER granularity is much coarser and the semantics
#   don't line up cleanly (e.g. VLLM_ROCM_USE_AITER_MLA = "engage AITER for
#   MLA path", SGLANG_AITER_MLA_PERSIST = "use persistent MLA buffers"). The
#   master VLLM_ROCM_USE_AITER toggle is the one that translates safely.


class SGLangEnvTranslator(EnvTranslator):
    """Renames vLLM env vars into SGLang-side names per ENV_MAP."""

    def translate(self, parent_env: Mapping[str, str]) -> dict[str, str]:
        return translate_env_with_map(parent_env, ENV_MAP)
