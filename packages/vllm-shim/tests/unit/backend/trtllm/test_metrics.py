"""Tests for the TRT-LLM Prometheus exposition translator."""

from vllm_shim.backend.trtllm.metrics import TRTLLMMetricsTranslator


def test_renames_request_success_total() -> None:
    raw = (
        "# HELP trtllm_request_success_total successful requests\n"
        "# TYPE trtllm_request_success_total counter\n"
        'trtllm_request_success_total{model="x"} 10\n'
    )
    out = TRTLLMMetricsTranslator().translate(raw)
    assert "vllm:request_success_total" in out
    assert "trtllm_request_success_total" not in out


def test_renames_kv_cache_hit_rate() -> None:
    raw = 'trtllm_kv_cache_hit_rate{model="x"} 0.42\n'
    out = TRTLLMMetricsTranslator().translate(raw)
    assert "vllm:gpu_prefix_cache_hit_rate" in out
    assert "trtllm_kv_cache_hit_rate" not in out


def test_renames_e2e_latency_histogram_suffixes() -> None:
    raw = (
        "# HELP trtllm_e2e_request_latency_seconds latency\n"
        "# TYPE trtllm_e2e_request_latency_seconds histogram\n"
        'trtllm_e2e_request_latency_seconds_bucket{le="0.5"} 1\n'
        "trtllm_e2e_request_latency_seconds_sum 1.5\n"
        "trtllm_e2e_request_latency_seconds_count 3\n"
    )
    out = TRTLLMMetricsTranslator().translate(raw)
    assert "vllm:e2e_request_latency_seconds_bucket" in out
    assert "vllm:e2e_request_latency_seconds_sum" in out
    assert "vllm:e2e_request_latency_seconds_count" in out


def test_renames_time_to_first_token_seconds() -> None:
    raw = "trtllm_time_to_first_token_seconds_sum 2.0\n"
    out = TRTLLMMetricsTranslator().translate(raw)
    assert "vllm:time_to_first_token_seconds_sum" in out


def test_renames_request_queue_time_seconds() -> None:
    raw = "trtllm_request_queue_time_seconds_count 5\n"
    out = TRTLLMMetricsTranslator().translate(raw)
    assert "vllm:request_queue_time_seconds_count" in out


def test_passes_unknown_metric_through() -> None:
    raw = 'trtllm_kv_cache_reused_blocks_total{model="x"} 42\n'
    out = TRTLLMMetricsTranslator().translate(raw)
    assert "trtllm_kv_cache_reused_blocks_total" in out


def test_synthesizes_kv_cache_usage_perc_from_utilization() -> None:
    raw = 'trtllm_kv_cache_utilization{model="x"} 0.25\n'
    out = TRTLLMMetricsTranslator().translate(raw)
    assert "vllm:kv_cache_usage_perc" in out
    # Ratio 0.25 -> 25.0%
    assert "25.0000" in out


def test_kv_cache_usage_perc_skipped_when_source_absent() -> None:
    out = TRTLLMMetricsTranslator().translate("")
    assert "vllm:kv_cache_usage_perc" not in out


def test_emits_healthy_pods_total() -> None:
    out = TRTLLMMetricsTranslator().translate("")
    assert "vllm:healthy_pods_total" in out


def test_emits_num_requests_swapped_zero() -> None:
    out = TRTLLMMetricsTranslator().translate("")
    assert "vllm:num_requests_swapped 0" in out


def test_help_and_type_lines_renamed() -> None:
    raw = (
        "# HELP trtllm_request_success_total help text\n"
        "# TYPE trtllm_request_success_total counter\n"
    )
    out = TRTLLMMetricsTranslator().translate(raw)
    assert "# HELP vllm:request_success_total" in out
    assert "# TYPE vllm:request_success_total" in out
