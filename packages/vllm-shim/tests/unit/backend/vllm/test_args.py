"""Tests for VLLMArgTranslator (identity pass-through)."""

import pytest
from vllm_shim.backend.vllm.args import VLLMArgTranslator


@pytest.fixture
def translator() -> VLLMArgTranslator:
    return VLLMArgTranslator()


# === Pass-through (every flag a vLLM CLI sends is already a vLLM flag) ==========


def test_tensor_parallel_size_passes_through(translator: VLLMArgTranslator) -> None:
    out, dropped = translator.translate(["--tensor-parallel-size", "8"])
    assert out == ["--tensor-parallel-size", "8"]
    assert dropped == []


def test_underscore_form_passes_through(translator: VLLMArgTranslator) -> None:
    out, _ = translator.translate(["--tensor_parallel_size", "8"])
    assert out == ["--tensor_parallel_size", "8"]


def test_short_alias_tp_passes_through(translator: VLLMArgTranslator) -> None:
    # Unlike the SGLang translator, we do not rewrite vLLM's short aliases:
    # vLLM accepts them natively.
    out, _ = translator.translate(["-tp", "4"])
    # The shared translator only auto-captures values for flags starting with
    # `--`; short flags get treated as standalone tokens, so the value lands
    # in the next pass-through. We assert both tokens reach the output.
    assert "-tp" in out
    assert "4" in out


def test_optimization_level_short_form_passes_through(translator: VLLMArgTranslator) -> None:
    # vLLM accepts -O3 natively (FlexibleArgumentParser expands it); the
    # vLLM backend does NOT call strip_optimization_level.
    out, dropped = translator.translate(["-O3"])
    assert out == ["-O3"]
    assert dropped == []


def test_speculative_config_passes_through(translator: VLLMArgTranslator) -> None:
    # The SGLang translator splits this into flat flags. The vLLM translator
    # leaves it intact because vLLM consumes the JSON form directly.
    out, dropped = translator.translate(
        [
            "--speculative-config",
            '{"method":"mtp","num_speculative_tokens":2}',
        ]
    )
    assert out == [
        "--speculative-config",
        '{"method":"mtp","num_speculative_tokens":2}',
    ]
    assert dropped == []


def test_equals_form_passes_through(translator: VLLMArgTranslator) -> None:
    out, _ = translator.translate(["--max-model-len=4096"])
    assert out == ["--max-model-len=4096"]


def test_boolean_flag_passes_through(translator: VLLMArgTranslator) -> None:
    out, dropped = translator.translate(["--trust-remote-code"])
    assert out == ["--trust-remote-code"]
    assert dropped == []


def test_multiple_flags_preserve_order(translator: VLLMArgTranslator) -> None:
    argv = [
        "--trust-remote-code",
        "--max-num-seqs",
        "256",
        "--quantization",
        "fp8",
        "--enable-auto-tool-choice",
    ]
    out, dropped = translator.translate(argv)
    assert out == argv
    assert dropped == []


def test_empty_input_is_empty_output(translator: VLLMArgTranslator) -> None:
    out, dropped = translator.translate([])
    assert out == []
    assert dropped == []
