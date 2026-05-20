"""Tests for the VLLMBackend wiring (component types and identity contract)."""

from vllm_shim.backend.base.args import ArgTranslator
from vllm_shim.backend.base.env import EnvTranslator
from vllm_shim.backend.base.launcher import Launcher
from vllm_shim.backend.base.metrics import MetricsTranslator
from vllm_shim.backend.base.parallelism import ParallelismExtractor
from vllm_shim.backend.vllm.backend import VLLMBackend


def test_name_is_vllm() -> None:
    assert VLLMBackend.name == "vllm"


def test_health_path_default() -> None:
    assert VLLMBackend.health_path == "/health"


def test_metrics_path_default() -> None:
    assert VLLMBackend.metrics_path == "/metrics"


def test_components_match_protocols() -> None:
    b = VLLMBackend()
    assert isinstance(b.args, ArgTranslator)
    assert isinstance(b.env, EnvTranslator)
    assert isinstance(b.metrics, MetricsTranslator)
    assert isinstance(b.launcher, Launcher)
    assert isinstance(b.filters, tuple)
    assert isinstance(b.parallelism, ParallelismExtractor)


def test_filters_tuple_is_empty() -> None:
    # vLLM accepts vLLM-shaped request bodies natively; no filters needed.
    assert VLLMBackend().filters == ()
