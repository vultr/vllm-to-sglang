"""Tests for the SGLangBackend wiring (component types and filter order)."""

from vllm_shim.backend.base.args import ArgTranslator
from vllm_shim.backend.base.filter import RequestFilter
from vllm_shim.backend.base.launcher import Launcher
from vllm_shim.backend.base.metrics import MetricsTranslator
from vllm_shim.backend.sglang.backend import SGLangBackend


def test_name_is_sglang() -> None:
    assert SGLangBackend.name == "sglang"


def test_health_path_default() -> None:
    assert SGLangBackend.health_path == "/health"


def test_components_match_protocols() -> None:
    b = SGLangBackend()
    assert isinstance(b.args, ArgTranslator)
    assert isinstance(b.metrics, MetricsTranslator)
    assert isinstance(b.launcher, Launcher)
    assert isinstance(b.filters, tuple)
    assert all(isinstance(f, RequestFilter) for f in b.filters)


def test_filters_in_documented_order() -> None:
    from vllm_shim.backend.sglang.filter.fix_schema import FixToolSchemas
    from vllm_shim.backend.sglang.filter.strip_params import StripVLLMParams

    b = SGLangBackend()
    assert isinstance(b.filters[0], StripVLLMParams)
    assert isinstance(b.filters[1], FixToolSchemas)
    assert len(b.filters) == 2
