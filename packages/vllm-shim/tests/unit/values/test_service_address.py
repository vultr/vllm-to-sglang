"""Tests for the ServiceAddress value object."""

import pytest
from vllm_shim.values.service_address import ServiceAddress


def test_str_format() -> None:
    assert str(ServiceAddress("0.0.0.0", 8000)) == "0.0.0.0:8000"


def test_url_default_scheme() -> None:
    assert ServiceAddress("127.0.0.1", 8001).url() == "http://127.0.0.1:8001"


def test_url_custom_scheme() -> None:
    assert ServiceAddress("api.example", 443).url("https") == "https://api.example:443"


def test_equality() -> None:
    assert ServiceAddress("h", 1) == ServiceAddress("h", 1)
    assert ServiceAddress("h", 1) != ServiceAddress("h", 2)


def test_immutable() -> None:
    addr = ServiceAddress("h", 1)
    with pytest.raises(AttributeError):
        addr.host = "x"  # type: ignore[misc]
