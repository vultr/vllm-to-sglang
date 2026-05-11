"""Tests for the TRTLLMBackend wiring (component types and class-level constants)."""

from vllm_shim.backend.base.args import ArgTranslator
from vllm_shim.backend.base.env import EnvTranslator
from vllm_shim.backend.base.launcher import Launcher
from vllm_shim.backend.base.metrics import MetricsTranslator
from vllm_shim.backend.base.parallelism import ParallelismExtractor
from vllm_shim.backend.trtllm.backend import TRTLLMBackend


def test_name_is_trtllm() -> None:
    assert TRTLLMBackend.name == "trtllm"


def test_health_path_default() -> None:
    assert TRTLLMBackend.health_path == "/health"


def test_metrics_path_override() -> None:
    assert TRTLLMBackend.metrics_path == "/prometheus/metrics"


def test_components_match_protocols() -> None:
    b = TRTLLMBackend()
    assert isinstance(b.args, ArgTranslator)
    assert isinstance(b.env, EnvTranslator)
    assert isinstance(b.metrics, MetricsTranslator)
    assert isinstance(b.launcher, Launcher)
    assert isinstance(b.filters, tuple)
    assert isinstance(b.parallelism, ParallelismExtractor)


def test_filters_empty_in_v1() -> None:
    b = TRTLLMBackend()
    assert b.filters == ()
