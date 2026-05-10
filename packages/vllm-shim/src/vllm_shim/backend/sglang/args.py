"""SGLang ArgTranslator: rewrites vLLM flags using ARG_MAP.

Coverage: every vLLM `serve` flag from repos/vllm/vllm/engine/arg_utils.py
(EngineArgs.add_cli_args / AsyncEngineArgs) and
repos/vllm/vllm/entrypoints/openai/cli_args.py (FrontendArgs / BaseFrontendArgs
/ make_arg_parser) has an explicit decision below: rename, drop, or fall through
because SGLang accepts the same flag name natively (see the canonical SGLang
flag list at repos/sglang/python/sglang/srt/server_args.py). vLLM short aliases
(-tp, -pp, -dp, -q, -n, -r, -asc, -sc, -ac, -cc, -ep, -dcp, -pcp, -dpa..-dpr,
and -O / -O<level>) are normalised here too, since SGLang's parser does not
recognise any of them.
"""

import json
from collections.abc import Sequence

from vllm_shim.backend._shared import strip_optimization_level, translate_with_arg_map
from vllm_shim.backend.base.args import ArgTranslator

# vLLM flag => (sglang_flag_or_None, has_value)
# None means "drop this flag (and its value if has_value)".
# Anything not listed here passes through verbatim, on the assumption that
# either (a) SGLang has the same flag name (most common case for engine-level
# knobs - --dtype, --kv-cache-dtype, --pipeline-parallel-size, etc.) or
# (b) the user passed a SGLang-only flag deliberately.
ARG_MAP: dict[str, tuple[str | None, bool]] = {
    # === Renames: vLLM concept → SGLang concept, value forwarded unchanged ===
    "--tensor-parallel-size": ("--tp", True),
    "--gpu-memory-utilization": ("--mem-fraction-static", True),
    "--gpu_memory_utilization": ("--mem-fraction-static", True),
    "--max-model-len": ("--context-length", True),
    "--max_model_len": ("--context-length", True),
    "--max-num-seqs": ("--max-running-requests", True),
    "--max-num-batched-tokens": ("--chunked-prefill-size", True),
    "--max-loras": ("--max-loras-per-batch", True),
    "--seed": ("--random-seed", True),
    "--distributed-timeout-seconds": ("--dist-timeout", True),
    "--lora-modules": ("--lora-paths", True),
    "--limit-mm-per-prompt": ("--limit-mm-data-per-request", True),
    "--tokenizer": ("--tokenizer-path", True),
    "--root-path": ("--fastapi-root-path", True),
    "--language-model-only": ("--language-only", False),
    "--mm-encoder-only": ("--encoder-only", False),
    "--enable-multi-modal": ("--enable-multimodal", False),
    "--enable-log-requests": ("--log-requests", False),
    "--enable-layerwise-nvtx-tracing": ("--enable-layerwise-nvtx-marker", False),
    "--enforce-eager": ("--disable-cuda-graph", False),
    "--enforce_eager": ("--disable-cuda-graph", False),
    "--no-enable-prefix-caching": ("--disable-radix-cache", False),
    "--trust-remote-code": ("--trust-remote-code", False),
    "--trust_remote_code": ("--trust-remote-code", False),

    # === Short aliases: rewrite to long form (SGLang accepts no short flags) ===
    "-tp": ("--tp", True),
    "-pp": ("--pipeline-parallel-size", True),
    "-dp": ("--data-parallel-size", True),
    "-q": ("--quantization", True),
    "-n": ("--nnodes", True),
    "-r": ("--node-rank", True),

    # === Default-already-matches: drop silently (no behavior change) ===
    "--enable-prefix-caching": (None, False),
    "--enable-chunked-prefill": (None, False),
    "--no-enable-chunked-prefill": (None, False),
    "--disable-log-requests": (None, False),
    "--disable-log-stats": (None, False),
    # SGLang default is autotune-enabled; --disable-flashinfer-autotune is the opt-out.
    "--enable-flashinfer-autotune": (None, False),

    # === Engine knobs with no SGLang equivalent: drop (with value) ===
    "--additional-config": (None, True),
    "--all2all-backend": (None, True),  # values differ from SGLang's --moe-a2a-backend choices
    "--allowed-local-media-path": (None, True),
    "--allowed-media-domains": (None, True),
    "--api-server-count": (None, True),
    "-asc": (None, True),
    "--attention-config": (None, True),
    "-ac": (None, True),
    "--block-size": (None, True),
    "--code-revision": (None, True),
    "--collect-detailed-traces": (None, True),
    "--compilation-config": (None, True),
    "-cc": (None, True),
    "--config-format": (None, True),
    "--convert": (None, True),
    "--cp-kv-cache-interleave-size": (None, True),
    # value format diverges from SGLang's --cuda-graph-bs (JSON list vs nargs="+")
    "--cudagraph-capture-sizes": (None, True),
    "--data-parallel-address": (None, True),
    "-dpa": (None, True),
    "--data-parallel-backend": (None, True),
    "-dpb": (None, True),
    "--data-parallel-rank": (None, True),
    "-dpn": (None, True),
    "--data-parallel-rpc-port": (None, True),
    "-dpp": (None, True),
    "--data-parallel-size-local": (None, True),
    "-dpl": (None, True),
    "--data-parallel-start-rank": (None, True),
    "-dpr": (None, True),
    "--dbo-decode-token-threshold": (None, True),
    "--dbo-prefill-token-threshold": (None, True),
    "--dcp-comm-backend": (None, True),
    "--dcp-kv-cache-interleave-size": (None, True),
    "--decode-context-parallel-size": (None, True),
    "-dcp": (None, True),
    "--default-mm-loras": (None, True),
    "--distributed-executor-backend": (None, True),
    "--ec-transfer-config": (None, True),
    "--eplb-config": (None, True),  # vLLM JSON; SGLang exposes flat --eplb-* flags
    "--expert-placement-strategy": (None, True),
    "--gdn-prefill-backend": (None, True),
    "--generation-config": (None, True),
    "--hf-config-path": (None, True),
    "--hf-overrides": (None, True),
    "--hf-token": (None, True),
    "--ignore-patterns": (None, True),
    "--io-processor-plugin": (None, True),
    "--ir-op-priority": (None, True),
    "--kernel-config": (None, True),
    "--kv-cache-dtype-skip-layers": (None, True),
    "--kv-cache-memory-bytes": (None, True),
    "--kv-cache-metrics-sample": (None, True),
    "--kv-offloading-backend": (None, True),
    "--kv-offloading-size": (None, True),
    "--kv-transfer-config": (None, True),
    "--logits-processors": (None, True),
    "--logprobs-mode": (None, True),
    "--long-prefill-token-threshold": (None, True),
    "--lora-dtype": (None, True),
    "--mamba-block-size": (None, True),
    "--mamba-cache-dtype": (None, True),
    "--mamba-cache-mode": (None, True),
    "--mamba-cache-philox-rounds": (None, True),
    "--mamba-ssm-cache-dtype": (None, True),  # vLLM permits "auto"; SGLang choices don't include it
    "--master-addr": (None, True),
    "--master-port": (None, True),
    "--max-cpu-loras": (None, True),
    "--max-cudagraph-capture-size": (None, True),
    "--max-logprobs": (None, True),
    "--max-long-partial-prefills": (None, True),
    "--max-num-partial-prefills": (None, True),
    "--max-parallel-loading-workers": (None, True),
    "--max-seq-len-to-capture": (None, True),
    "--media-io-kwargs": (None, True),
    "--mm-encoder-attn-backend": (None, True),  # values differ from SGLang's --mm-attention-backend
    "--mm-encoder-tp-mode": (None, True),
    "--mm-processor-cache-gb": (None, True),
    "--mm-processor-cache-type": (None, True),
    "--mm-processor-kwargs": (None, True),
    "--mm-shm-cache-max-object-size-mb": (None, True),
    "--moe-backend": (None, True),  # values differ from SGLang's --moe-runner-backend
    "--num-cpu-blocks-override": (None, True),
    "--num-gpu-blocks-override": (None, True),
    "--numa-bind-cpus": (None, True),
    "--numa-bind-nodes": (None, True),
    "--offload-backend": (None, True),
    "--optimization-level": (None, True),  # -O / -O<level> handled by _strip_optimization_level
    "--override-attention-dtype": (None, True),
    "--override-generation-config": (None, True),
    "--performance-mode": (None, True),
    "--pooler-config": (None, True),
    "--prefill-context-parallel-size": (None, True),
    "-pcp": (None, True),
    "--prefix-caching-hash-algo": (None, True),
    "--profiler-config": (None, True),
    "--pt-load-map-location": (None, True),
    "--reasoning-config": (None, True),
    "--reasoning-parser-plugin": (None, True),
    "--renderer-num-workers": (None, True),
    "--runner": (None, True),
    "--safetensors-load-strategy": (None, True),
    "--scheduler-cls": (None, True),
    "--scheduler-delay-factor": (None, True),
    "--scheduling-policy": (None, True),  # vLLM and SGLang policy enums diverge
    "--show-hidden-metrics-for-version": (None, True),
    "--shutdown-timeout": (None, True),
    "--structured-outputs-config": (None, True),
    "--swap-space": (None, True),  # historical vLLM flag, retained for older callers
    "--tokenizer-revision": (None, True),
    "--ubatch-size": (None, True),
    "--video-pruning-rate": (None, True),
    "--weight-transfer-config": (None, True),
    "--worker-cls": (None, True),
    "--worker-extension-cls": (None, True),

    # === Engine knobs with no SGLang equivalent: drop (boolean) ===
    "--aggregate-engine-logging": (None, False),
    "--allow-deprecated-quantization": (None, False),
    "--async-scheduling": (None, False),
    "--calculate-kv-scales": (None, False),
    "--cpu-offload-params": (None, False),
    "--cudagraph-metrics": (None, False),
    "--data-parallel-external-lb": (None, False),
    "-dpe": (None, False),
    "--data-parallel-hybrid-lb": (None, False),
    "-dph": (None, False),
    "--disable-cascade-attn": (None, False),
    "--disable-chunked-mm-input": (None, False),
    "--disable-hybrid-kv-cache-manager": (None, False),
    "--disable-nccl-for-dp-synchronization": (None, False),
    "--disable-sliding-window": (None, False),
    "--enable-dbo": (None, False),
    "--enable-elastic-ep": (None, False),
    "--enable-ep-weight-filter": (None, False),
    "--enable-expert-parallel": (None, False),
    "-ep": (None, False),
    "--enable-mamba-cache-stochastic-rounding": (None, False),
    "--enable-mm-embeds": (None, False),
    "--enable-prompt-adapter": (None, False),
    "--enable-prompt-embeds": (None, False),
    "--enable-sleep-mode": (None, False),
    "--enable-tower-connector-lora": (None, False),
    "--fail-on-environ-validation": (None, False),
    "--fully-sharded-loras": (None, False),
    "--grpc": (None, False),
    "--headless": (None, False),
    "--interleave-mm-strings": (None, False),
    "--kv-cache-metrics": (None, False),
    "--kv-sharing-fast-prefill": (None, False),
    "--mm-tensor-ipc": (None, False),
    "--numa-bind": (None, False),
    "--offload-params": (None, False),
    "--ray-workers-use-nsight": (None, False),
    "--scheduler-reserve-full-isl": (None, False),
    "--skip-mm-profiling": (None, False),
    "--specialize-active-lora": (None, False),
    "--use-tqdm-on-load": (None, False),

    # === Frontend HTTP server flags handled by the shim's haproxy/middleware,
    # not the engine. SGLang has no equivalent for any of these, so dropping is
    # the right move. (api-key, ssl-keyfile/certfile/ca-certs, enable-ssl-refresh,
    # host, port are common to both and pass through.) ===
    "--allow-credentials": (None, False),
    "--allowed-headers": (None, True),
    "--allowed-methods": (None, True),
    "--allowed-origins": (None, True),
    "--chat-template-content-format": (None, True),
    "--default-chat-template-kwargs": (None, True),
    "--disable-access-log-for-endpoints": (None, True),
    "--disable-fastapi-docs": (None, False),
    "--disable-uvicorn-access-log": (None, False),
    "--enable-flash-late-interaction": (None, False),
    "--enable-force-include-usage": (None, False),
    "--enable-log-deltas": (None, False),
    "--enable-log-outputs": (None, False),
    "--enable-logging-iteration-details": (None, False),
    "--enable-offline-docs": (None, False),
    "--enable-prompt-tokens-details": (None, False),
    "--enable-request-id-headers": (None, False),
    "--enable-server-load-tracking": (None, False),
    "--enable-tokenizer-info-endpoint": (None, False),
    "--exclude-tools-when-tool-choice-none": (None, False),
    "--h11-max-header-count": (None, True),
    "--h11-max-incomplete-event-size": (None, True),
    "--log-config-file": (None, True),
    "--log-error-stack": (None, False),
    "--max-log-len": (None, True),
    "--middleware": (None, True),
    "--response-role": (None, True),
    "--return-tokens-as-token-ids": (None, False),
    "--ssl-cert-reqs": (None, True),
    "--ssl-ciphers": (None, True),
    "--tokens-only": (None, False),
    "--tool-parser-plugin": (None, True),
    "--trust-request-chat-template": (None, False),
    "--uds": (None, True),
    "--uvicorn-log-level": (None, True),
    "--enable-auto-tool-choice": (None, False),
}


def _translate_speculative_config(json_value: str) -> list[str] | None:
    """Translate a vLLM ``--speculative-config`` JSON blob to SGLang flat flags.

    Returns the SGLang flag list, or None when the config can't be translated
    (unknown method, malformed JSON, missing num_speculative_tokens). Callers
    should drop the original flag in the None case.

    Only ``method=mtp`` is supported, mapped per the SGLang-recommended MTP
    recipe in repos/sglang/docs/basic_usage/deepseek_v32.md and the
    repos/sglang/test/registered/8-gpu-models/test_deepseek_v3_mtp.py fixture:

        --speculative-algorithm EAGLE
        --speculative-num-steps N
        --speculative-eagle-topk 1
        --speculative-num-draft-tokens N+1
    """
    try:
        cfg = json.loads(json_value)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
    if not isinstance(cfg, dict):
        return None
    if cfg.get("method") != "mtp":
        return None
    n = cfg.get("num_speculative_tokens")
    if not isinstance(n, int) or n <= 0:
        return None
    return [
        "--speculative-algorithm", "EAGLE",
        "--speculative-num-steps", str(n),
        "--speculative-eagle-topk", "1",
        "--speculative-num-draft-tokens", str(n + 1),
    ]


def _split_speculative_config(args: list[str]) -> tuple[list[str], list[str], list[str]]:
    """Pull ``--speculative-config`` / ``-sc`` (and ``=``-form) out of args.

    Returns ``(remaining_args, translated_flags, dropped)``. ``remaining_args``
    is the input with every occurrence removed, ready to feed to
    ``translate_with_arg_map``. ``translated_flags`` is the concatenation of
    SGLang flags emitted by ``_translate_speculative_config`` for each
    recognised occurrence. ``dropped`` collects the raw tokens for occurrences
    we couldn't translate.
    """
    remaining: list[str] = []
    translated: list[str] = []
    dropped: list[str] = []
    i = 0
    while i < len(args):
        arg = args[i]
        if arg in ("--speculative-config", "-sc"):
            value = args[i + 1] if i + 1 < len(args) else ""
            flags = _translate_speculative_config(value)
            if flags is not None:
                translated.extend(flags)
            else:
                dropped.append(arg)
                if i + 1 < len(args):
                    dropped.append(value)
            i += 2 if i + 1 < len(args) else 1
            continue
        if arg.startswith(("--speculative-config=", "-sc=")):
            _, value = arg.split("=", 1)
            flags = _translate_speculative_config(value)
            if flags is not None:
                translated.extend(flags)
            else:
                dropped.append(arg)
            i += 1
            continue
        remaining.append(arg)
        i += 1
    return remaining, translated, dropped


class SGLangArgTranslator(ArgTranslator):
    """Pure function over argv: rename, drop, or pass through each token per ARG_MAP."""

    def translate(self, vllm_args: Sequence[str]) -> tuple[list[str], list[str]]:
        a1, d1 = strip_optimization_level(list(vllm_args))
        a2, spec_flags, d2 = _split_speculative_config(a1)
        translated, d3 = translate_with_arg_map(a2, ARG_MAP)
        return translated + spec_flags, d1 + d2 + d3
