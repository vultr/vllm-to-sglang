"""TRT-LLM ArgTranslator: rewrites vLLM flags using ARG_MAP."""

from collections.abc import Sequence

from vllm_shim.backend._shared import translate_with_arg_map
from vllm_shim.backend.base.args import ArgTranslator

# vLLM flag => (trtllm_flag_or_None, has_value)
# None means "drop this flag (and its value if has_value)".
# Anything not listed here passes through verbatim.
ARG_MAP: dict[str, tuple[str | None, bool]] = {
    # Renames with value
    "--tensor-parallel-size": ("--tp_size", True),
    "--tensor_parallel_size": ("--tp_size", True),
    "--pipeline-parallel-size": ("--pp_size", True),
    "--pipeline_parallel_size": ("--pp_size", True),
    "--max-model-len": ("--max_seq_len", True),
    "--max_model_len": ("--max_seq_len", True),
    "--max-num-seqs": ("--max_batch_size", True),
    "--max_num_seqs": ("--max_batch_size", True),
    "--max-num-batched-tokens": ("--max_num_tokens", True),
    "--gpu-memory-utilization": ("--kv_cache_free_gpu_memory_fraction", True),
    "--gpu_memory_utilization": ("--kv_cache_free_gpu_memory_fraction", True),
    "--trust-remote-code": ("--trust_remote_code", True),
    "--trust_remote_code": ("--trust_remote_code", True),
    "--served-model-name": ("--served_model_name", True),
    "--chat-template": ("--chat_template", True),
    "--revision": ("--revision", True),
    "--hf-revision": ("--revision", True),
    "--tokenizer": ("--tokenizer", True),
    # Renames boolean
    "--enable-chunked-prefill": ("--enable_chunked_prefill", False),
    # No TRT-LLM equivalent or default already matches: drop with their value if any
    "--swap-space": (None, True),
    "--block-size": (None, True),
    "--num-gpu-blocks-override": (None, True),
    "--num-cpu-blocks-override": (None, True),
    "--enforce-eager": (None, False),
    "--enforce_eager": (None, False),
    "--enable-prefix-caching": (None, False),
    "--no-enable-prefix-caching": (None, False),
    "--no-enable-chunked-prefill": (None, False),
    "--quantization": (None, True),
    "--dtype": (None, True),
    "--lora-modules": (None, True),
    "--code-revision": (None, True),
    "--tokenizer-revision": (None, True),
    "--max-seq-len-to-capture": (None, True),
    "--max-cpu-loras": (None, True),
    "--lora-dtype": (None, True),
    "--enable-prompt-adapter": (None, False),
    "--scheduler-delay-factor": (None, True),
    "--limit-mm-per-prompt": (None, True),
    "--distributed-executor-backend": (None, True),
    "--distributed-timeout-seconds": (None, True),
    "--seed": (None, True),
    "--disable-log-requests": (None, False),
    "--disable-log-stats": (None, False),
}


class TRTLLMArgTranslator(ArgTranslator):
    """Pure function over argv: rename, drop, or pass through each token per ARG_MAP."""

    def translate(self, vllm_args: Sequence[str]) -> tuple[list[str], list[str]]:
        return translate_with_arg_map(vllm_args, ARG_MAP)
