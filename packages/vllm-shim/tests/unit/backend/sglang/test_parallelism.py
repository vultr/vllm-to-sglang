"""Tests for the SGLang ParallelismExtractor.

Every flag spelling exercised here is one a post-translation SGLang argv
can actually contain: either emitted by ``SGLangArgTranslator`` (e.g.
``--tp``, ``--pipeline-parallel-size``) or passed through verbatim by an
operator using SGLang-native names (``--tp-size``, ``--ep-size``).
"""

from vllm_shim.backend.sglang.parallelism import SGLangParallelismExtractor
from vllm_shim.values.parallelism import Parallelism


def _extract(*args: str) -> Parallelism:
    return SGLangParallelismExtractor().extract(args)


def test_defaults_to_all_ones_for_empty_args() -> None:
    assert _extract() == Parallelism(tp=1, ep=1, pp=1)


def test_reads_tp_short_form_emitted_by_translator() -> None:
    # The SGLang ArgTranslator emits `--tp N` for vLLM's `-tp`/`--tensor-parallel-size`.
    assert _extract("--tp", "8") == Parallelism(tp=8)


def test_reads_tp_size_native_alias() -> None:
    # Operators passing SGLang-native flags via passthrough.
    assert _extract("--tp-size", "8") == Parallelism(tp=8)


def test_reads_tensor_parallel_size_native() -> None:
    assert _extract("--tensor-parallel-size", "4") == Parallelism(tp=4)


def test_reads_pipeline_parallel_size() -> None:
    # Long form is what the translator emits and what SGLang accepts canonically.
    assert _extract("--pipeline-parallel-size", "2") == Parallelism(pp=2)


def test_reads_pp_size_native_alias() -> None:
    assert _extract("--pp-size", "2") == Parallelism(pp=2)


def test_reads_ep_size() -> None:
    assert _extract("--ep-size", "8") == Parallelism(ep=8)


def test_reads_expert_parallel_size_native() -> None:
    assert _extract("--expert-parallel-size", "4") == Parallelism(ep=4)


def test_reads_ep_short_alias() -> None:
    # SGLang's argparse defines `--ep` as an alias for `--ep-size`.
    assert _extract("--ep", "8") == Parallelism(ep=8)


def test_reads_equals_form() -> None:
    assert _extract("--tp=8", "--ep-size=8") == Parallelism(tp=8, ep=8)


def test_reads_all_three_together() -> None:
    args = (
        "--tp", "4",
        "--ep-size", "4",
        "--pipeline-parallel-size", "2",
    )
    assert _extract(*args) == Parallelism(tp=4, ep=4, pp=2)


def test_ignores_unrelated_flags() -> None:
    args = (
        "--model", "/data/m",
        "--port", "8000",
        "--tp", "8",
        "--mem-fraction-static", "0.9",
    )
    assert _extract(*args) == Parallelism(tp=8)


def test_last_occurrence_wins() -> None:
    # If the translator emits one form and the user also passed another,
    # last-wins matches the argparse behaviour SGLang will apply.
    assert _extract("--tp", "2", "--tp-size", "8") == Parallelism(tp=8)
