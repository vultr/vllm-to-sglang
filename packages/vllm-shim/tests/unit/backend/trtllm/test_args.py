"""Tests for TRTLLMArgTranslator (vLLM-to-TRT-LLM flag rewriting)."""

import pytest
from vllm_shim.backend.trtllm.args import TRTLLMArgTranslator


@pytest.fixture
def translator() -> TRTLLMArgTranslator:
    return TRTLLMArgTranslator()


def test_rename_tensor_parallel_size(translator: TRTLLMArgTranslator) -> None:
    out, dropped = translator.translate(["--tensor-parallel-size", "8"])
    assert out == ["--tp_size", "8"]
    assert dropped == []


def test_rename_pipeline_parallel_size(translator: TRTLLMArgTranslator) -> None:
    out, _ = translator.translate(["--pipeline-parallel-size", "2"])
    assert out == ["--pp_size", "2"]


def test_rename_max_model_len_to_max_seq_len(translator: TRTLLMArgTranslator) -> None:
    out, _ = translator.translate(["--max-model-len", "32768"])
    assert out == ["--max_seq_len", "32768"]


def test_rename_max_num_seqs_to_max_batch_size(translator: TRTLLMArgTranslator) -> None:
    out, _ = translator.translate(["--max-num-seqs", "256"])
    assert out == ["--max_batch_size", "256"]


def test_rename_max_num_batched_tokens(translator: TRTLLMArgTranslator) -> None:
    out, _ = translator.translate(["--max-num-batched-tokens", "8192"])
    assert out == ["--max_num_tokens", "8192"]


def test_rename_gpu_memory_utilization(translator: TRTLLMArgTranslator) -> None:
    out, _ = translator.translate(["--gpu-memory-utilization", "0.85"])
    assert out == ["--kv_cache_free_gpu_memory_fraction", "0.85"]


def test_rename_trust_remote_code_takes_value(translator: TRTLLMArgTranslator) -> None:
    out, _ = translator.translate(["--trust-remote-code", "true"])
    assert out == ["--trust_remote_code", "true"]


def test_rename_served_model_name(translator: TRTLLMArgTranslator) -> None:
    out, _ = translator.translate(["--served-model-name", "alias"])
    assert out == ["--served_model_name", "alias"]


def test_rename_chat_template(translator: TRTLLMArgTranslator) -> None:
    out, _ = translator.translate(["--chat-template", "/tmp/t.jinja"])
    assert out == ["--chat_template", "/tmp/t.jinja"]


def test_rename_revision(translator: TRTLLMArgTranslator) -> None:
    out, _ = translator.translate(["--revision", "abc123"])
    assert out == ["--hf_revision", "abc123"]


def test_rename_hf_revision(translator: TRTLLMArgTranslator) -> None:
    out, _ = translator.translate(["--hf-revision", "abc123"])
    assert out == ["--hf_revision", "abc123"]


def test_rename_enable_chunked_prefill(translator: TRTLLMArgTranslator) -> None:
    out, _ = translator.translate(["--enable-chunked-prefill"])
    assert out == ["--enable_chunked_prefill"]


def test_drop_swap_space(translator: TRTLLMArgTranslator) -> None:
    out, dropped = translator.translate(["--swap-space", "4"])
    assert out == []
    assert dropped == ["--swap-space", "4"]


def test_drop_block_size(translator: TRTLLMArgTranslator) -> None:
    out, _ = translator.translate(["--block-size", "16"])
    assert out == []


def test_drop_enforce_eager(translator: TRTLLMArgTranslator) -> None:
    out, _ = translator.translate(["--enforce-eager"])
    assert out == []


def test_drop_enable_prefix_caching(translator: TRTLLMArgTranslator) -> None:
    out, _ = translator.translate(["--enable-prefix-caching"])
    assert out == []


def test_drop_no_enable_prefix_caching(translator: TRTLLMArgTranslator) -> None:
    out, _ = translator.translate(["--no-enable-prefix-caching"])
    assert out == []


def test_drop_seed(translator: TRTLLMArgTranslator) -> None:
    out, dropped = translator.translate(["--seed", "42"])
    assert out == []
    assert dropped == ["--seed", "42"]


def test_drop_quantization(translator: TRTLLMArgTranslator) -> None:
    out, _ = translator.translate(["--quantization", "fp8"])
    assert out == []


def test_drop_dtype(translator: TRTLLMArgTranslator) -> None:
    out, _ = translator.translate(["--dtype", "bfloat16"])
    assert out == []


def test_drop_lora_modules(translator: TRTLLMArgTranslator) -> None:
    out, _ = translator.translate(["--lora-modules", "alice=/p/a"])
    assert out == []


def test_drop_log_requests(translator: TRTLLMArgTranslator) -> None:
    out, _ = translator.translate(["--disable-log-requests"])
    assert out == []


def test_passthrough_unknown_flag(translator: TRTLLMArgTranslator) -> None:
    out, _ = translator.translate(["--config", "/tmp/cfg.yaml"])
    assert out == ["--config", "/tmp/cfg.yaml"]


def test_passthrough_native_trtllm_flag(translator: TRTLLMArgTranslator) -> None:
    out, _ = translator.translate(["--reasoning_parser", "nano-v3"])
    assert out == ["--reasoning_parser", "nano-v3"]


def test_equals_form_split(translator: TRTLLMArgTranslator) -> None:
    out, _ = translator.translate(["--max-model-len=4096"])
    assert out == ["--max_seq_len", "4096"]


def test_underscore_variant(translator: TRTLLMArgTranslator) -> None:
    out, _ = translator.translate(["--gpu_memory_utilization", "0.85"])
    assert out == ["--kv_cache_free_gpu_memory_fraction", "0.85"]


def test_underscore_tensor_parallel(translator: TRTLLMArgTranslator) -> None:
    out, _ = translator.translate(["--tensor_parallel_size", "4"])
    assert out == ["--tp_size", "4"]


def test_mixed_input_preserves_order(translator: TRTLLMArgTranslator) -> None:
    out, dropped = translator.translate([
        "--trust-remote-code", "true",
        "--max-num-seqs", "256",
        "--swap-space", "4",
        "--config", "/tmp/cfg.yaml",
    ])
    assert out == [
        "--trust_remote_code", "true",
        "--max_batch_size", "256",
        "--config", "/tmp/cfg.yaml",
    ]
    assert dropped == ["--swap-space", "4"]


# === Short alias normalisation ==================================================

def test_short_alias_tp(translator: TRTLLMArgTranslator) -> None:
    out, _ = translator.translate(["-tp", "8"])
    assert out == ["--tp_size", "8"]


def test_short_alias_pp(translator: TRTLLMArgTranslator) -> None:
    out, _ = translator.translate(["-pp", "2"])
    assert out == ["--pp_size", "2"]


def test_short_alias_drops(translator: TRTLLMArgTranslator) -> None:
    cases = [
        (["-asc", "4"], ["-asc", "4"]),
        (["-q", "fp8"], ["-q", "fp8"]),
        (["-n", "2"], ["-n", "2"]),
        (["-r", "0"], ["-r", "0"]),
        (["-dp", "8"], ["-dp", "8"]),
        (["-dcp", "2"], ["-dcp", "2"]),
        (["-pcp", "2"], ["-pcp", "2"]),
        (["-ep"], ["-ep"]),
        (["-sc", "{}"], ["-sc", "{}"]),
    ]
    for argv, expected_dropped in cases:
        out, dropped = translator.translate(argv)
        assert out == [], f"expected empty out for {argv}, got {out}"
        assert dropped == expected_dropped, f"for {argv}"


# === -O family ==================================================================

def test_drop_optimization_level_long_form(translator: TRTLLMArgTranslator) -> None:
    out, dropped = translator.translate(["--optimization-level", "3"])
    assert out == []
    assert dropped == ["--optimization-level", "3"]


def test_drop_O3_short(translator: TRTLLMArgTranslator) -> None:
    out, dropped = translator.translate(["-O3"])
    assert out == []
    assert dropped == ["-O3"]


def test_drop_O_with_separate_value(translator: TRTLLMArgTranslator) -> None:
    out, dropped = translator.translate(["-O", "3"])
    assert out == []
    assert dropped == ["-O", "3"]


# === Speculative-config: drop entirely (no TRT-LLM CLI flag) ====================

def test_drop_speculative_config(translator: TRTLLMArgTranslator) -> None:
    out, dropped = translator.translate([
        "--speculative-config", '{"method":"mtp","num_speculative_tokens":2}',
    ])
    assert out == []
    assert dropped == [
        "--speculative-config", '{"method":"mtp","num_speculative_tokens":2}',
    ]


def test_drop_speculative_config_equals_form(translator: TRTLLMArgTranslator) -> None:
    out, dropped = translator.translate(['--speculative-config={"method":"mtp"}'])
    assert out == []
    assert dropped == ['--speculative-config={"method":"mtp"}']


# === Newly-classified flag drops (representative sample) ========================

def test_drop_enable_auto_tool_choice(translator: TRTLLMArgTranslator) -> None:
    out, dropped = translator.translate(["--enable-auto-tool-choice"])
    assert out == []
    assert dropped == ["--enable-auto-tool-choice"]


def test_drop_mm_encoder_tp_mode_equals(translator: TRTLLMArgTranslator) -> None:
    out, dropped = translator.translate(["--mm-encoder-tp-mode=data"])
    assert out == []
    assert dropped == ["--mm-encoder-tp-mode=data"]


def test_drop_headless(translator: TRTLLMArgTranslator) -> None:
    out, _ = translator.translate(["--headless"])
    assert out == []


def test_drop_allowed_origins(translator: TRTLLMArgTranslator) -> None:
    out, _ = translator.translate(["--allowed-origins", '["*"]'])
    assert out == []


# === Renames added in this pass =================================================

def test_rename_kv_cache_dtype(translator: TRTLLMArgTranslator) -> None:
    out, _ = translator.translate(["--kv-cache-dtype", "fp8"])
    assert out == ["--kv_cache_dtype", "fp8"]


def test_rename_reasoning_parser(translator: TRTLLMArgTranslator) -> None:
    out, _ = translator.translate(["--reasoning-parser", "deepseek-v3"])
    assert out == ["--reasoning_parser", "deepseek-v3"]


def test_rename_tool_call_parser_to_tool_parser(translator: TRTLLMArgTranslator) -> None:
    out, _ = translator.translate(["--tool-call-parser", "qwen3_coder"])
    assert out == ["--tool_parser", "qwen3_coder"]


def test_rename_otlp_traces_endpoint(translator: TRTLLMArgTranslator) -> None:
    out, _ = translator.translate(["--otlp-traces-endpoint", "http://x:4317"])
    assert out == ["--otlp_traces_endpoint", "http://x:4317"]


def test_rename_video_pruning_rate(translator: TRTLLMArgTranslator) -> None:
    out, _ = translator.translate(["--video-pruning-rate", "0.5"])
    assert out == ["--video_pruning_rate", "0.5"]
