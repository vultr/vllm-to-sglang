import pytest
from vllm_shim.backend import registry
from vllm_shim.backend.sglang.backend import SGLangBackend


def test_default_is_sglang(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.delenv("VLLM_SHIM_BACKEND", raising=False)
    backend = registry.select()
    assert isinstance(backend, SGLangBackend)


def test_explicit_sglang_env(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("VLLM_SHIM_BACKEND", "sglang")
    assert isinstance(registry.select(), SGLangBackend)


def test_unknown_raises(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("VLLM_SHIM_BACKEND", "tensorrt")
    with pytest.raises(ValueError, match="Unknown backend"):
        registry.select()
