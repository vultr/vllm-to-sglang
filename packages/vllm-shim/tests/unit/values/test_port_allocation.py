"""Tests for the PortAllocation value object."""

from vllm_shim.values.port_allocation import PortAllocation


def test_from_listen_assigns_offsets() -> None:
    p = PortAllocation.from_listen(8000)
    assert p.frontend == 8000
    assert p.backend == 8001
    assert p.middleware == 8002


def test_equality() -> None:
    assert PortAllocation.from_listen(7000) == PortAllocation.from_listen(7000)
    assert PortAllocation.from_listen(7000) != PortAllocation.from_listen(8000)
