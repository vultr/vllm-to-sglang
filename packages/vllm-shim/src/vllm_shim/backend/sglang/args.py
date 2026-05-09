"""SGLang ArgTranslator: rewrites vLLM flags using ARG_MAP."""

from collections.abc import Sequence

from vllm_shim.backend._shared import translate_with_arg_map
from vllm_shim.backend.base.args import ArgTranslator

# vLLM flag => (sglang_flag_or_None, has_value)
# None means "drop this flag (and its value if has_value)".
# Anything not listed here passes through verbatim.
ARG_MAP: dict[str, tuple[str | None, bool]] = {
    # Renames with value
    "--tensor-parallel-size": ("--tp", True),
    "--gpu-memory-utilization": ("--mem-fraction-static", True),
    "--gpu_memory_utilization": ("--mem-fraction-static", True),
    "--max-model-len": ("--context-length", True),
    "--max_model_len": ("--context-length", True),
    "--max-num-seqs": ("--max-running-requests", True),
    "--max-num-batched-tokens": ("--chunked-prefill-size", True),
    "--seed": ("--random-seed", True),
    "--distributed-timeout-seconds": ("--dist-timeout", True),
    "--lora-modules": ("--lora-paths", True),
    # Renames without value
    "--enable-multi-modal": ("--enable-multimodal", False),
    "--enforce-eager": ("--disable-cuda-graph", False),
    "--enforce_eager": ("--disable-cuda-graph", False),
    "--no-enable-prefix-caching": ("--disable-radix-cache", False),
    "--trust_remote_code": ("--trust-remote-code", False),
    "--trust-remote-code": ("--trust-remote-code", False),
    # vLLM defaults already match SGLang behavior, drop silently
    "--enable-prefix-caching": (None, False),
    "--enable-chunked-prefill": (None, False),
    "--no-enable-chunked-prefill": (None, False),
    "--disable-log-requests": (None, False),
    "--disable-log-stats": (None, False),
    # No SGLang equivalent, drop
    "--swap-space": (None, True),
    "--block-size": (None, True),
    "--num-gpu-blocks-override": (None, True),
    "--num-cpu-blocks-override": (None, True),
    "--distributed-executor-backend": (None, True),
    "--code-revision": (None, True),
    "--tokenizer-revision": (None, True),
    "--max-seq-len-to-capture": (None, True),
    "--max-cpu-loras": (None, True),
    "--lora-dtype": (None, True),
    "--enable-prompt-adapter": (None, False),
    "--scheduler-delay-factor": (None, True),
    "--limit-mm-per-prompt": (None, True),
}


class SGLangArgTranslator(ArgTranslator):
    """Pure function over argv: rename, drop, or pass through each token per ARG_MAP."""

    def translate(self, vllm_args: Sequence[str]) -> tuple[list[str], list[str]]:
        return translate_with_arg_map(vllm_args, ARG_MAP)
