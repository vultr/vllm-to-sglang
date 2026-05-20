"""Tests for VLLMEnvTranslator (identity pass-through)."""

import pytest
from vllm_shim.backend.vllm.env import ENV_MAP, VLLMEnvTranslator


@pytest.fixture
def translator() -> VLLMEnvTranslator:
    return VLLMEnvTranslator()


def test_env_map_is_empty() -> None:
    # vLLM consumes VLLM_* vars natively; no renames needed.
    assert ENV_MAP == {}


def test_vllm_env_var_passes_through_unchanged(translator: VLLMEnvTranslator) -> None:
    parent = {"VLLM_USE_MODELSCOPE": "1", "VLLM_ROCM_USE_AITER": "1"}
    out = translator.translate(parent)
    assert out["VLLM_USE_MODELSCOPE"] == "1"
    assert out["VLLM_ROCM_USE_AITER"] == "1"


def test_no_invented_targets(translator: VLLMEnvTranslator) -> None:
    out = translator.translate({"VLLM_USE_MODELSCOPE": "1"})
    # No SGLANG_* or other engine-side renames synthesised.
    assert all(not k.startswith("SGLANG_") for k in out)


def test_unrelated_env_is_preserved(translator: VLLMEnvTranslator) -> None:
    parent = {
        "PATH": "/usr/bin",
        "HF_HOME": "/cache/hf",
        "CUDA_VISIBLE_DEVICES": "0,1",
    }
    out = translator.translate(parent)
    for k, v in parent.items():
        assert out[k] == v


def test_pure_function_does_not_mutate_input(translator: VLLMEnvTranslator) -> None:
    parent = {"VLLM_USE_MODELSCOPE": "1"}
    snapshot = dict(parent)
    translator.translate(parent)
    assert parent == snapshot


def test_returns_fresh_dict(translator: VLLMEnvTranslator) -> None:
    parent = {"FOO": "bar"}
    out = translator.translate(parent)
    out["NEW"] = "added"
    assert "NEW" not in parent
