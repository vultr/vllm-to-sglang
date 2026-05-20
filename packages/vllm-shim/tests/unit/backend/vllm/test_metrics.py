"""Tests for VLLMMetricsTranslator (identity pass-through)."""

from vllm_shim.backend.vllm.metrics import VLLMMetricsTranslator


def test_passes_through_unchanged() -> None:
    raw = (
        "# HELP vllm:num_requests_running running requests\n"
        "# TYPE vllm:num_requests_running gauge\n"
        'vllm:num_requests_running{model="x"} 5\n'
    )
    out = VLLMMetricsTranslator().translate(raw)
    assert out == raw


def test_empty_input_is_empty_output() -> None:
    assert VLLMMetricsTranslator().translate("") == ""


def test_does_not_append_synthesized_tail() -> None:
    # vLLM emits its own vllm:num_requests_swapped; we must not re-emit it.
    out = VLLMMetricsTranslator().translate("")
    assert "vllm:healthy_pods_total" not in out
    assert "vllm:num_requests_swapped" not in out


def test_native_vllm_swapped_series_passes_through() -> None:
    # The series exists in modern vLLM; verify it survives translation as-is.
    raw = '# TYPE vllm:num_requests_swapped gauge\nvllm:num_requests_swapped{model="x"} 3\n'
    out = VLLMMetricsTranslator().translate(raw)
    assert out == raw
    # And the value is not flattened to zero by us.
    assert 'vllm:num_requests_swapped{model="x"} 3' in out
