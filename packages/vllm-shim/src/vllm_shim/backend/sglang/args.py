from collections.abc import Sequence

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
    def translate(self, vllm_args: Sequence[str]) -> tuple[list[str], list[str]]:
        out: list[str] = []
        dropped: list[str] = []
        i = 0
        args = list(vllm_args)
        while i < len(args):
            arg = args[i]

            # --flag=value form
            if arg.startswith("--") and "=" in arg:
                flag, val = arg.split("=", 1)
                mapping = ARG_MAP.get(flag)
                if mapping is None:
                    out.append(arg)
                else:
                    sglang_flag, has_val = mapping
                    if sglang_flag is None:
                        dropped.append(arg)
                    elif has_val:
                        out.extend([sglang_flag, val])
                    else:
                        out.append(sglang_flag)
                i += 1
                continue

            # --flag (with possible separate value)
            mapping = ARG_MAP.get(arg)
            if mapping is None:
                # Pass through. If next token looks like a value, take it too.
                if (
                    i + 1 < len(args)
                    and not args[i + 1].startswith("-")
                    and arg.startswith("--")
                ):
                    out.extend([arg, args[i + 1]])
                    i += 2
                else:
                    out.append(arg)
                    i += 1
                continue

            sglang_flag, has_val = mapping
            if sglang_flag is None:
                dropped.append(arg)
                if has_val and i + 1 < len(args):
                    dropped.append(args[i + 1])
                    i += 2
                else:
                    i += 1
            elif has_val:
                if i + 1 < len(args):
                    out.extend([sglang_flag, args[i + 1]])
                    i += 2
                else:
                    i += 1
            else:
                out.append(sglang_flag)
                i += 1
        return out, dropped
