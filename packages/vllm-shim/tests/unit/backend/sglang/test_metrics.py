"""Tests for the SGLang Prometheus exposition translator."""

from vllm_shim.backend.sglang.metrics import SGLangMetricsTranslator


def test_renames_known_gauge() -> None:
    raw = (
        "# HELP sglang:num_running_reqs running requests\n"
        "# TYPE sglang:num_running_reqs gauge\n"
        'sglang:num_running_reqs{model="x"} 5\n'
    )
    out = SGLangMetricsTranslator().translate(raw)
    assert "vllm:num_requests_running" in out
    assert "sglang:num_running_reqs" not in out


def test_passes_unknown_metric_through() -> None:
    raw = 'sglang:gen_throughput{model="x"} 42\n'
    out = SGLangMetricsTranslator().translate(raw)
    assert "sglang:gen_throughput" in out


def test_renames_histogram_suffixes() -> None:
    raw = (
        "# HELP sglang:e2e_request_latency_seconds latency\n"
        "# TYPE sglang:e2e_request_latency_seconds histogram\n"
        'sglang:e2e_request_latency_seconds_bucket{le="0.5"} 1\n'
        "sglang:e2e_request_latency_seconds_sum 1.5\n"
        "sglang:e2e_request_latency_seconds_count 3\n"
    )
    out = SGLangMetricsTranslator().translate(raw)
    assert "vllm:e2e_request_latency_seconds_bucket" in out
    assert "vllm:e2e_request_latency_seconds_sum" in out
    assert "vllm:e2e_request_latency_seconds_count" in out


def test_synthesizes_kv_cache_usage_perc() -> None:
    raw = (
        'sglang:num_used_tokens{model="x"} 50\n'
        'sglang:max_total_num_tokens{model="x"} 200\n'
    )
    out = SGLangMetricsTranslator().translate(raw)
    assert "vllm:kv_cache_usage_perc" in out
    assert "25.0000" in out


def test_emits_healthy_pods_total() -> None:
    out = SGLangMetricsTranslator().translate("")
    assert "vllm:healthy_pods_total" in out


def test_emits_num_requests_swapped_zero() -> None:
    out = SGLangMetricsTranslator().translate("")
    assert "vllm:num_requests_swapped 0" in out


def test_success_and_aborted_emit_distinct_series() -> None:
    raw = (
        "# HELP sglang:num_requests_total total requests\n"
        "# TYPE sglang:num_requests_total counter\n"
        'sglang:num_requests_total{model="x"} 10\n'
        "# HELP sglang:num_aborted_requests_total aborted requests\n"
        "# TYPE sglang:num_aborted_requests_total counter\n"
        'sglang:num_aborted_requests_total{model="x"} 2\n'
    )
    out = SGLangMetricsTranslator().translate(raw)
    assert out.count("# TYPE vllm:request_success_total") == 1
    assert "vllm:request_success_total" in out
    assert "vllm:request_aborted_total" in out
