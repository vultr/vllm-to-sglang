"""Tests for VLLMParallelismExtractor (reads tp/ep/pp from vLLM-native argv)."""

from vllm_shim.backend.vllm.parallelism import VLLMParallelismExtractor
from vllm_shim.values.parallelism import Parallelism


def _extract(*args: str) -> Parallelism:
    return VLLMParallelismExtractor().extract(args)


def test_defaults_to_all_ones_for_empty_args() -> None:
    assert _extract() == Parallelism(tp=1, ep=1, pp=1)


def test_reads_tensor_parallel_size() -> None:
    assert _extract("--tensor-parallel-size", "8") == Parallelism(tp=8)


def test_reads_tensor_parallel_size_underscore() -> None:
    assert _extract("--tensor_parallel_size", "4") == Parallelism(tp=4)


def test_reads_tp_short_alias() -> None:
    assert _extract("-tp", "8") == Parallelism(tp=8)


def test_reads_pipeline_parallel_size() -> None:
    assert _extract("--pipeline-parallel-size", "2") == Parallelism(pp=2)


def test_reads_pipeline_parallel_size_underscore() -> None:
    assert _extract("--pipeline_parallel_size", "2") == Parallelism(pp=2)


def test_reads_pp_short_alias() -> None:
    assert _extract("-pp", "2") == Parallelism(pp=2)


def test_reads_expert_parallel_size() -> None:
    assert _extract("--expert-parallel-size", "8") == Parallelism(ep=8)


def test_reads_expert_parallel_size_underscore() -> None:
    assert _extract("--expert_parallel_size", "4") == Parallelism(ep=4)


def test_reads_equals_form() -> None:
    assert _extract("--tensor-parallel-size=8", "--expert-parallel-size=4") == Parallelism(
        tp=8, ep=4
    )


def test_reads_all_three_together() -> None:
    args = (
        "--tensor-parallel-size",
        "4",
        "--expert-parallel-size",
        "4",
        "--pipeline-parallel-size",
        "2",
    )
    assert _extract(*args) == Parallelism(tp=4, ep=4, pp=2)


def test_ignores_unrelated_flags() -> None:
    args = (
        "--model",
        "/data/m",
        "--port",
        "8000",
        "-tp",
        "8",
        "--gpu-memory-utilization",
        "0.9",
    )
    assert _extract(*args) == Parallelism(tp=8)


def test_last_occurrence_wins() -> None:
    assert _extract("--tensor-parallel-size", "2", "-tp", "8") == Parallelism(tp=8)
