import pytest
from vllm_shim.backend.sglang.args import SGLangArgTranslator


@pytest.fixture
def translator() -> SGLangArgTranslator:
    return SGLangArgTranslator()


def test_rename_tensor_parallel_size(translator: SGLangArgTranslator) -> None:
    out, dropped = translator.translate(["--tensor-parallel-size", "8"])
    assert out == ["--tp", "8"]
    assert dropped == []


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


def test_drop_swap_space(translator: SGLangArgTranslator) -> None:
    out, dropped = translator.translate(["--swap-space", "4"])
    assert out == []
    assert dropped == ["--swap-space", "4"]


def test_drop_block_size(translator: SGLangArgTranslator) -> None:
    out, dropped = translator.translate(["--block-size", "16"])
    assert out == []
    assert dropped == ["--block-size", "16"]


def test_drop_enable_prefix_caching_silently(translator: SGLangArgTranslator) -> None:
    out, dropped = translator.translate(["--enable-prefix-caching"])
    assert out == []
    assert dropped == ["--enable-prefix-caching"]


def test_passthrough_unknown_flag(translator: SGLangArgTranslator) -> None:
    out, _ = translator.translate(["--quantization", "fp8"])
    assert out == ["--quantization", "fp8"]


def test_equals_form_split(translator: SGLangArgTranslator) -> None:
    out, _ = translator.translate(["--max-model-len=4096"])
    assert out == ["--context-length", "4096"]


def test_underscore_variant(translator: SGLangArgTranslator) -> None:
    out, _ = translator.translate(["--gpu_memory_utilization", "0.85"])
    assert out == ["--mem-fraction-static", "0.85"]


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


def test_seed_renamed_to_random_seed(translator: SGLangArgTranslator) -> None:
    out, _ = translator.translate(["--seed", "42"])
    assert out == ["--random-seed", "42"]


def test_lora_modules_renamed(translator: SGLangArgTranslator) -> None:
    out, _ = translator.translate(["--lora-modules", "alice=/p/a"])
    assert out == ["--lora-paths", "alice=/p/a"]
