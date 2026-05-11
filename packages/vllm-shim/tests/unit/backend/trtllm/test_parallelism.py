"""Tests for the TRT-LLM ParallelismExtractor.

Every flag spelling exercised here is one a post-translation trtllm-serve
argv can actually contain: either emitted by ``TRTLLMArgTranslator`` (e.g.
``--tp_size``, ``--pp_size``) or passed through verbatim by an operator
using TRT-LLM-native names (``--ep_size``, ``--moe_expert_parallel_size``).
"""

from vllm_shim.backend.trtllm.parallelism import TRTLLMParallelismExtractor
from vllm_shim.values.parallelism import Parallelism


def _extract(*args: str) -> Parallelism:
    return TRTLLMParallelismExtractor().extract(args)


def test_defaults_to_all_ones_for_empty_args() -> None:
    assert _extract() == Parallelism(tp=1, ep=1, pp=1)


def test_reads_tp_size_emitted_by_translator() -> None:
    # The TRT-LLM ArgTranslator emits `--tp_size N` for vLLM's
    # `-tp` / `--tensor-parallel-size`.
    assert _extract("--tp_size", "8") == Parallelism(tp=8)


def test_reads_pp_size_emitted_by_translator() -> None:
    assert _extract("--pp_size", "2") == Parallelism(pp=2)


def test_reads_ep_size_alias() -> None:
    # trtllm-serve accepts `--ep_size` as an alias for the canonical
    # `--moe_expert_parallel_size`. Operators usually pass the shorter form.
    assert _extract("--ep_size", "8") == Parallelism(ep=8)


def test_reads_moe_expert_parallel_size_canonical() -> None:
    assert _extract("--moe_expert_parallel_size", "4") == Parallelism(ep=4)


def test_reads_equals_form() -> None:
    assert _extract("--tp_size=8", "--ep_size=8") == Parallelism(tp=8, ep=8)


def test_reads_all_three_together() -> None:
    args = (
        "--tp_size", "4",
        "--ep_size", "4",
        "--pp_size", "2",
    )
    assert _extract(*args) == Parallelism(tp=4, ep=4, pp=2)


def test_ignores_unrelated_flags() -> None:
    args = (
        "/data/m",
        "--port", "8000",
        "--tp_size", "8",
        "--kv_cache_free_gpu_memory_fraction", "0.9",
    )
    assert _extract(*args) == Parallelism(tp=8)


def test_rejects_sglang_style_flags() -> None:
    # An operator using SGLang spellings against a TRT-LLM backend is a
    # misconfiguration; the extractor must not silently accept dashed
    # flags as if they were the underscore variants.
    assert _extract("--tp-size", "8") == Parallelism()
    assert _extract("--pipeline-parallel-size", "2") == Parallelism()


def test_last_occurrence_wins() -> None:
    assert _extract("--tp_size", "2", "--tp_size", "8") == Parallelism(tp=8)
