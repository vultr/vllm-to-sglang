"""Tests for TRTLLMEnvTranslator (vLLM-to-TRT-LLM env var rewriting)."""

import pytest
from vllm_shim.backend.trtllm.env import ENV_MAP, TRTLLMEnvTranslator


@pytest.fixture
def translator() -> TRTLLMEnvTranslator:
    return TRTLLMEnvTranslator()


@pytest.mark.parametrize("vllm_name,trtllm_name", list(ENV_MAP.items()))
def test_every_entry_renames(
    translator: TRTLLMEnvTranslator,
    vllm_name: str,
    trtllm_name: str,
) -> None:
    out = translator.translate({vllm_name: "value"})
    assert out[trtllm_name] == "value"
    assert out[vllm_name] == "value"


def test_no_usage_stats(translator: TRTLLMEnvTranslator) -> None:
    out = translator.translate({"VLLM_NO_USAGE_STATS": "1"})
    assert out["TRTLLM_NO_USAGE_STATS"] == "1"


def test_allow_long_max_model_len(translator: TRTLLMEnvTranslator) -> None:
    out = translator.translate({"VLLM_ALLOW_LONG_MAX_MODEL_LEN": "1"})
    assert out["TLLM_ALLOW_LONG_MAX_MODEL_LEN"] == "1"


def test_existing_target_is_not_overwritten(translator: TRTLLMEnvTranslator) -> None:
    out = translator.translate({
        "VLLM_NO_USAGE_STATS": "1",
        "TRTLLM_NO_USAGE_STATS": "0",
    })
    assert out["TRTLLM_NO_USAGE_STATS"] == "0"


def test_unrelated_env_is_preserved(translator: TRTLLMEnvTranslator) -> None:
    parent = {
        "PATH": "/usr/bin",
        "CUDA_VISIBLE_DEVICES": "0,1",
        "VLLM_USE_DEEP_GEMM": "1",
    }
    out = translator.translate(parent)
    for k, v in parent.items():
        assert out[k] == v


def test_pure_function_does_not_mutate_input(translator: TRTLLMEnvTranslator) -> None:
    parent = {"VLLM_NO_USAGE_STATS": "1"}
    snapshot = dict(parent)
    translator.translate(parent)
    assert parent == snapshot


def test_rocm_env_vars_are_not_translated(translator: TRTLLMEnvTranslator) -> None:
    """TRT-LLM is CUDA-only; ROCm vars have no equivalent and stay as-is."""
    out = translator.translate({"VLLM_ROCM_USE_AITER": "1"})
    assert out["VLLM_ROCM_USE_AITER"] == "1"
    assert "TRTLLM_USE_AITER" not in out
    assert "TLLM_USE_AITER" not in out
