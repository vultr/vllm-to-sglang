"""Tests for the backend base ABCs (instantiation guard)."""

import pytest
from vllm_shim.backend.base.args import ArgTranslator
from vllm_shim.backend.base.backend import Backend
from vllm_shim.backend.base.env import EnvTranslator
from vllm_shim.backend.base.filter import RequestFilter
from vllm_shim.backend.base.launcher import Launcher
from vllm_shim.backend.base.metrics import MetricsTranslator


@pytest.mark.parametrize(
    "cls",
    [ArgTranslator, EnvTranslator, MetricsTranslator, RequestFilter, Launcher, Backend],
)
def test_abc_cannot_be_instantiated(cls: type) -> None:
    with pytest.raises(TypeError):
        cls()
