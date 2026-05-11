"""Tests for SGLangArgTranslator (vLLM-to-SGLang flag rewriting)."""

import pytest
from vllm_shim.backend.sglang.args import SGLangArgTranslator


@pytest.fixture
def translator() -> SGLangArgTranslator:
    return SGLangArgTranslator()


# === Renames =====================================================================

def test_rename_tensor_parallel_size(translator: SGLangArgTranslator) -> None:
    out, dropped = translator.translate(["--tensor-parallel-size", "8"])
    assert out == ["--tp", "8"]
    assert dropped == []


def test_rename_tensor_parallel_size_underscore(translator: SGLangArgTranslator) -> None:
    # vLLM users often write the underscore form; SGLang's argparse only
    # registers the dashed names, so we have to normalise.
    out, _ = translator.translate(["--tensor_parallel_size", "8"])
    assert out == ["--tp", "8"]


def test_rename_pipeline_parallel_size_underscore(translator: SGLangArgTranslator) -> None:
    out, _ = translator.translate(["--pipeline_parallel_size", "2"])
    assert out == ["--pipeline-parallel-size", "2"]


def test_rename_ep_size_underscore(translator: SGLangArgTranslator) -> None:
    out, _ = translator.translate(["--ep_size", "8"])
    assert out == ["--ep-size", "8"]


def test_rename_expert_parallel_size_underscore(translator: SGLangArgTranslator) -> None:
    out, _ = translator.translate(["--expert_parallel_size", "8"])
    assert out == ["--expert-parallel-size", "8"]


def test_rename_max_model_len_to_context_length(translator: SGLangArgTranslator) -> None:
    out, _ = translator.translate(["--max-model-len", "32768"])
    assert out == ["--context-length", "32768"]


def test_rename_enforce_eager_to_disable_cuda_graph(translator: SGLangArgTranslator) -> None:
    out, _ = translator.translate(["--enforce-eager"])
    assert out == ["--disable-cuda-graph"]


def test_rename_no_enable_prefix_caching_to_disable_radix_cache(
    translator: SGLangArgTranslator,
) -> None:
    out, _ = translator.translate(["--no-enable-prefix-caching"])
    assert out == ["--disable-radix-cache"]


def test_rename_max_loras_to_max_loras_per_batch(translator: SGLangArgTranslator) -> None:
    out, _ = translator.translate(["--max-loras", "4"])
    assert out == ["--max-loras-per-batch", "4"]


def test_rename_limit_mm_per_prompt(translator: SGLangArgTranslator) -> None:
    out, _ = translator.translate(["--limit-mm-per-prompt", '{"image":2}'])
    assert out == ["--limit-mm-data-per-request", '{"image":2}']


def test_rename_tokenizer_to_tokenizer_path(translator: SGLangArgTranslator) -> None:
    out, _ = translator.translate(["--tokenizer", "/models/tk"])
    assert out == ["--tokenizer-path", "/models/tk"]


def test_rename_root_path_to_fastapi_root_path(translator: SGLangArgTranslator) -> None:
    out, _ = translator.translate(["--root-path", "/v1"])
    assert out == ["--fastapi-root-path", "/v1"]


def test_rename_language_model_only_to_language_only(translator: SGLangArgTranslator) -> None:
    out, _ = translator.translate(["--language-model-only"])
    assert out == ["--language-only"]


def test_rename_mm_encoder_only_to_encoder_only(translator: SGLangArgTranslator) -> None:
    out, _ = translator.translate(["--mm-encoder-only"])
    assert out == ["--encoder-only"]


def test_rename_enable_log_requests(translator: SGLangArgTranslator) -> None:
    out, _ = translator.translate(["--enable-log-requests"])
    assert out == ["--log-requests"]


def test_rename_enable_layerwise_nvtx_tracing(translator: SGLangArgTranslator) -> None:
    out, _ = translator.translate(["--enable-layerwise-nvtx-tracing"])
    assert out == ["--enable-layerwise-nvtx-marker"]


def test_seed_renamed_to_random_seed(translator: SGLangArgTranslator) -> None:
    out, _ = translator.translate(["--seed", "42"])
    assert out == ["--random-seed", "42"]


def test_lora_modules_renamed(translator: SGLangArgTranslator) -> None:
    out, _ = translator.translate(["--lora-modules", "alice=/p/a"])
    assert out == ["--lora-paths", "alice=/p/a"]


# === Short alias normalisation ===================================================

def test_short_alias_tp(translator: SGLangArgTranslator) -> None:
    out, _ = translator.translate(["-tp", "4"])
    assert out == ["--tp", "4"]


def test_short_alias_pp(translator: SGLangArgTranslator) -> None:
    out, _ = translator.translate(["-pp", "2"])
    assert out == ["--pipeline-parallel-size", "2"]


def test_short_alias_dp(translator: SGLangArgTranslator) -> None:
    out, _ = translator.translate(["-dp", "8"])
    assert out == ["--data-parallel-size", "8"]


def test_short_alias_q_quantization(translator: SGLangArgTranslator) -> None:
    out, _ = translator.translate(["-q", "fp8"])
    assert out == ["--quantization", "fp8"]


def test_short_alias_n_nnodes(translator: SGLangArgTranslator) -> None:
    out, _ = translator.translate(["-n", "2"])
    assert out == ["--nnodes", "2"]


def test_short_alias_r_node_rank(translator: SGLangArgTranslator) -> None:
    out, _ = translator.translate(["-r", "0"])
    assert out == ["--node-rank", "0"]


def test_short_alias_drops(translator: SGLangArgTranslator) -> None:
    # All short aliases for vLLM-only flags must drop with their values.
    cases = [
        (["-asc", "4"], ["-asc", "4"]),
        (["-ac", "{}"], ["-ac", "{}"]),
        (["-cc", "{}"], ["-cc", "{}"]),
        (["-dcp", "2"], ["-dcp", "2"]),
        (["-pcp", "2"], ["-pcp", "2"]),
        (["-ep"], ["-ep"]),
    ]
    for argv, expected_dropped in cases:
        out, dropped = translator.translate(argv)
        assert out == [], f"expected empty out for {argv}, got {out}"
        assert dropped == expected_dropped, f"for {argv}"


# === Default-already-matches drops ==============================================

def test_drop_enable_prefix_caching_silently(translator: SGLangArgTranslator) -> None:
    out, dropped = translator.translate(["--enable-prefix-caching"])
    assert out == []
    assert dropped == ["--enable-prefix-caching"]


def test_drop_enable_chunked_prefill(translator: SGLangArgTranslator) -> None:
    out, dropped = translator.translate(["--enable-chunked-prefill"])
    assert out == []
    assert dropped == ["--enable-chunked-prefill"]


def test_drop_enable_flashinfer_autotune(translator: SGLangArgTranslator) -> None:
    out, _ = translator.translate(["--enable-flashinfer-autotune"])
    assert out == []


# === No-equivalent drops (representative sample) ================================

def test_drop_block_size(translator: SGLangArgTranslator) -> None:
    out, dropped = translator.translate(["--block-size", "16"])
    assert out == []
    assert dropped == ["--block-size", "16"]


def test_drop_optimization_level_long_form(translator: SGLangArgTranslator) -> None:
    out, dropped = translator.translate(["--optimization-level", "3"])
    assert out == []
    assert dropped == ["--optimization-level", "3"]


def test_drop_O3_short(translator: SGLangArgTranslator) -> None:
    out, dropped = translator.translate(["-O3"])
    assert out == []
    assert dropped == ["-O3"]


def test_drop_O_with_separate_value(translator: SGLangArgTranslator) -> None:
    out, dropped = translator.translate(["-O", "3"])
    assert out == []
    assert dropped == ["-O", "3"]


def test_drop_O_equals_form(translator: SGLangArgTranslator) -> None:
    out, dropped = translator.translate(["-O=3"])
    assert out == []
    assert dropped == ["-O=3"]


def test_drop_enable_auto_tool_choice(translator: SGLangArgTranslator) -> None:
    out, dropped = translator.translate(["--enable-auto-tool-choice"])
    assert out == []
    assert dropped == ["--enable-auto-tool-choice"]


def test_drop_mm_encoder_tp_mode(translator: SGLangArgTranslator) -> None:
    out, dropped = translator.translate(["--mm-encoder-tp-mode", "data"])
    assert out == []
    assert dropped == ["--mm-encoder-tp-mode", "data"]


def test_drop_mm_processor_cache_type_equals(translator: SGLangArgTranslator) -> None:
    out, dropped = translator.translate(["--mm-processor-cache-type=shm"])
    assert out == []
    assert dropped == ["--mm-processor-cache-type=shm"]


def test_drop_headless(translator: SGLangArgTranslator) -> None:
    out, dropped = translator.translate(["--headless"])
    assert out == []
    assert dropped == ["--headless"]


def test_drop_grpc(translator: SGLangArgTranslator) -> None:
    out, dropped = translator.translate(["--grpc"])
    assert out == []
    assert dropped == ["--grpc"]


# === Frontend HTTP flag drops ====================================================

def test_drop_allowed_origins(translator: SGLangArgTranslator) -> None:
    out, dropped = translator.translate(["--allowed-origins", '["https://a.com"]'])
    assert out == []
    assert dropped == ["--allowed-origins", '["https://a.com"]']


def test_drop_uvicorn_log_level(translator: SGLangArgTranslator) -> None:
    out, _ = translator.translate(["--uvicorn-log-level", "debug"])
    assert out == []


def test_drop_middleware(translator: SGLangArgTranslator) -> None:
    out, _ = translator.translate(["--middleware", "my.mod:fn"])
    assert out == []


# === Pass-through (same name in both) ===========================================

def test_passthrough_unknown_flag(translator: SGLangArgTranslator) -> None:
    out, _ = translator.translate(["--quantization", "fp8"])
    assert out == ["--quantization", "fp8"]


def test_passthrough_dtype(translator: SGLangArgTranslator) -> None:
    out, _ = translator.translate(["--dtype", "bfloat16"])
    assert out == ["--dtype", "bfloat16"]


def test_passthrough_kv_cache_dtype(translator: SGLangArgTranslator) -> None:
    out, _ = translator.translate(["--kv-cache-dtype", "fp8"])
    assert out == ["--kv-cache-dtype", "fp8"]


def test_equals_form_split(translator: SGLangArgTranslator) -> None:
    out, _ = translator.translate(["--max-model-len=4096"])
    assert out == ["--context-length", "4096"]


def test_underscore_variant(translator: SGLangArgTranslator) -> None:
    out, _ = translator.translate(["--gpu_memory_utilization", "0.85"])
    assert out == ["--mem-fraction-static", "0.85"]


# === Speculative-config translation =============================================

def test_speculative_config_mtp_space_form(translator: SGLangArgTranslator) -> None:
    out, _ = translator.translate([
        "--speculative-config",
        '{"method":"mtp","num_speculative_tokens":2}',
    ])
    assert out == [
        "--speculative-algorithm", "EAGLE",
        "--speculative-num-steps", "2",
        "--speculative-eagle-topk", "1",
        "--speculative-num-draft-tokens", "3",
    ]


def test_speculative_config_mtp_equals_form(translator: SGLangArgTranslator) -> None:
    out, _ = translator.translate([
        '--speculative-config={"method":"mtp","num_speculative_tokens":3}',
    ])
    assert out == [
        "--speculative-algorithm", "EAGLE",
        "--speculative-num-steps", "3",
        "--speculative-eagle-topk", "1",
        "--speculative-num-draft-tokens", "4",
    ]


def test_speculative_config_short_alias_sc(translator: SGLangArgTranslator) -> None:
    out, _ = translator.translate([
        "-sc",
        '{"method":"mtp","num_speculative_tokens":1}',
    ])
    assert out == [
        "--speculative-algorithm", "EAGLE",
        "--speculative-num-steps", "1",
        "--speculative-eagle-topk", "1",
        "--speculative-num-draft-tokens", "2",
    ]


def test_speculative_config_unknown_method_dropped(translator: SGLangArgTranslator) -> None:
    out, dropped = translator.translate([
        "--speculative-config",
        '{"method":"eagle","num_speculative_tokens":2}',
    ])
    assert out == []
    assert dropped == [
        "--speculative-config",
        '{"method":"eagle","num_speculative_tokens":2}',
    ]


def test_speculative_config_invalid_json_dropped(translator: SGLangArgTranslator) -> None:
    out, dropped = translator.translate([
        "--speculative-config",
        "not-json",
    ])
    assert out == []
    assert dropped == ["--speculative-config", "not-json"]


def test_speculative_config_missing_n_dropped(translator: SGLangArgTranslator) -> None:
    out, dropped = translator.translate([
        "--speculative-config",
        '{"method":"mtp"}',
    ])
    assert out == []
    assert "--speculative-config" in dropped


# === Mixed input scenarios ======================================================

def test_mixed_input_preserves_order(translator: SGLangArgTranslator) -> None:
    out, dropped = translator.translate([
        "--trust-remote-code",
        "--max-num-seqs", "256",
        "--swap-space", "4",
        "--quantization", "fp8",
    ])
    assert out == [
        "--trust-remote-code",
        "--max-running-requests", "256",
        "--quantization", "fp8",
    ]
    assert dropped == ["--swap-space", "4"]


def test_full_failing_invocation(translator: SGLangArgTranslator) -> None:
    """The complete vLLM invocation that originally surfaced this bug."""
    out, _ = translator.translate([
        "-O3",
        "--enable-auto-tool-choice",
        "--mm-encoder-tp-mode=data",
        "--mm-processor-cache-type=shm",
        "--speculative-config",
        '{"method":"mtp","num_speculative_tokens":2}',
        "--tensor-parallel-size", "8",
        "--max-model-len", "16384",
    ])
    assert out == [
        "--tp", "8",
        "--context-length", "16384",
        "--speculative-algorithm", "EAGLE",
        "--speculative-num-steps", "2",
        "--speculative-eagle-topk", "1",
        "--speculative-num-draft-tokens", "3",
    ]
