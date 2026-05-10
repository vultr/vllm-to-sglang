"""Unit tests for the backend-shared helpers (translate_with_arg_map,
translate_prom_line, vllm_synthesized_tail)."""

from vllm_shim.backend._shared import (
    translate_env_with_map,
    translate_prom_line,
    translate_with_arg_map,
    vllm_synthesized_tail,
)

# ---------- translate_with_arg_map ----------

_BASIC_MAP: dict[str, tuple[str | None, bool]] = {
    "--rename-with-value": ("--new-with-value", True),
    "--rename-bool": ("--new-bool", False),
    "--drop-with-value": (None, True),
    "--drop-bool": (None, False),
}


def test_translate_rename_with_value() -> None:
    out, dropped = translate_with_arg_map(["--rename-with-value", "X"], _BASIC_MAP)
    assert out == ["--new-with-value", "X"]
    assert dropped == []


def test_translate_rename_boolean() -> None:
    out, _ = translate_with_arg_map(["--rename-bool"], _BASIC_MAP)
    assert out == ["--new-bool"]


def test_translate_drop_with_value() -> None:
    out, dropped = translate_with_arg_map(["--drop-with-value", "X"], _BASIC_MAP)
    assert out == []
    assert dropped == ["--drop-with-value", "X"]


def test_translate_drop_boolean() -> None:
    out, dropped = translate_with_arg_map(["--drop-bool"], _BASIC_MAP)
    assert out == []
    assert dropped == ["--drop-bool"]


def test_translate_passthrough_unknown_flag_with_value() -> None:
    out, _ = translate_with_arg_map(["--unknown", "X"], _BASIC_MAP)
    assert out == ["--unknown", "X"]


def test_translate_passthrough_unknown_boolean_when_followed_by_flag() -> None:
    out, _ = translate_with_arg_map(["--unknown-bool", "--rename-bool"], _BASIC_MAP)
    assert out == ["--unknown-bool", "--new-bool"]


def test_translate_equals_form_split() -> None:
    out, _ = translate_with_arg_map(["--rename-with-value=X"], _BASIC_MAP)
    assert out == ["--new-with-value", "X"]


def test_translate_equals_form_drop() -> None:
    out, dropped = translate_with_arg_map(["--drop-with-value=X"], _BASIC_MAP)
    assert out == []
    assert dropped == ["--drop-with-value=X"]


def test_translate_preserves_order() -> None:
    out, _ = translate_with_arg_map(
        ["--rename-bool", "--unknown", "X", "--rename-with-value", "Y"],
        _BASIC_MAP,
    )
    assert out == ["--new-bool", "--unknown", "X", "--new-with-value", "Y"]

# ---------- translate_prom_line ----------

_NAME_MAP: dict[str, str] = {
    "src_metric_total": "vllm:request_success_total",
    "src_latency_seconds": "vllm:e2e_request_latency_seconds",
}


def test_prom_line_renames_known_sample() -> None:
    out = translate_prom_line('src_metric_total{model="x"} 5', _NAME_MAP)
    assert out == ['vllm:request_success_total{model="x"} 5']


def test_prom_line_renames_help_line() -> None:
    out = translate_prom_line("# HELP src_metric_total help text", _NAME_MAP)
    assert out == ["# HELP vllm:request_success_total help text"]


def test_prom_line_renames_type_line() -> None:
    out = translate_prom_line("# TYPE src_metric_total counter", _NAME_MAP)
    assert out == ["# TYPE vllm:request_success_total counter"]


def test_prom_line_renames_histogram_suffixes() -> None:
    assert translate_prom_line(
        'src_latency_seconds_bucket{le="0.5"} 1', _NAME_MAP
    ) == ['vllm:e2e_request_latency_seconds_bucket{le="0.5"} 1']
    assert translate_prom_line("src_latency_seconds_sum 1.5", _NAME_MAP) == [
        "vllm:e2e_request_latency_seconds_sum 1.5"
    ]
    assert translate_prom_line("src_latency_seconds_count 3", _NAME_MAP) == [
        "vllm:e2e_request_latency_seconds_count 3"
    ]


def test_prom_line_passes_unknown_through() -> None:
    out = translate_prom_line('unknown_metric{x="y"} 9', _NAME_MAP)
    assert out == ['unknown_metric{x="y"} 9']


def test_prom_line_returns_unmatched_input_unchanged() -> None:
    # Comment that is neither HELP nor TYPE
    out = translate_prom_line("# random comment", _NAME_MAP)
    assert out == ["# random comment"]


# ---------- vllm_synthesized_tail ----------


def test_synthesized_tail_lines() -> None:
    tail = vllm_synthesized_tail()
    assert "# HELP vllm:healthy_pods_total Number of healthy vLLM pods" in tail
    assert "# TYPE vllm:healthy_pods_total gauge" in tail
    assert 'vllm:healthy_pods_total{endpoint="default"} 1' in tail
    assert "# HELP vllm:num_requests_swapped Number of swapped requests" in tail
    assert "# TYPE vllm:num_requests_swapped gauge" in tail
    assert "vllm:num_requests_swapped 0" in tail


def test_synthesized_tail_length() -> None:
    assert len(vllm_synthesized_tail()) == 6


# ---------- translate_env_with_map ----------


def test_translate_env_renames_when_target_absent() -> None:
    out = translate_env_with_map({"FOO": "1"}, {"FOO": "BAR"})
    assert out == {"FOO": "1", "BAR": "1"}


def test_translate_env_preserves_existing_target() -> None:
    out = translate_env_with_map(
        {"FOO": "1", "BAR": "0"},
        {"FOO": "BAR"},
    )
    assert out["BAR"] == "0"
    assert out["FOO"] == "1"


def test_translate_env_skips_when_source_absent() -> None:
    out = translate_env_with_map({"OTHER": "x"}, {"FOO": "BAR"})
    assert "BAR" not in out
    assert out == {"OTHER": "x"}


def test_translate_env_does_not_mutate_input() -> None:
    parent = {"FOO": "1"}
    snapshot = dict(parent)
    translate_env_with_map(parent, {"FOO": "BAR"})
    assert parent == snapshot
