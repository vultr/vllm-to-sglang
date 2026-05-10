"""Tests for SGLangEnvTranslator (vLLM-to-SGLang env var rewriting)."""

import pytest
from vllm_shim.backend.sglang.env import ENV_MAP, SGLangEnvTranslator


@pytest.fixture
def translator() -> SGLangEnvTranslator:
    return SGLangEnvTranslator()


# === Renames take effect ========================================================

@pytest.mark.parametrize("vllm_name,sglang_name", list(ENV_MAP.items()))
def test_every_entry_renames(
    translator: SGLangEnvTranslator,
    vllm_name: str,
    sglang_name: str,
) -> None:
    out = translator.translate({vllm_name: "value"})
    assert out[sglang_name] == "value"
    # vLLM-side name is left in place; SGLang ignores names it doesn't know.
    assert out[vllm_name] == "value"


def test_use_modelscope(translator: SGLangEnvTranslator) -> None:
    out = translator.translate({"VLLM_USE_MODELSCOPE": "1"})
    assert out["SGLANG_USE_MODELSCOPE"] == "1"


def test_rocm_use_aiter(translator: SGLangEnvTranslator) -> None:
    out = translator.translate({"VLLM_ROCM_USE_AITER": "1"})
    assert out["SGLANG_USE_AITER"] == "1"


def test_rocm_quick_reduce_targets_unprefixed_upstream_name(
    translator: SGLangEnvTranslator,
) -> None:
    """SGLang reads the un-prefixed ROCM_QUICK_REDUCE_* names directly."""
    out = translator.translate({
        "VLLM_ROCM_QUICK_REDUCE_QUANTIZATION": "fp8",
        "VLLM_ROCM_QUICK_REDUCE_MAX_SIZE_BYTES_MB": "2048",
        "VLLM_ROCM_QUICK_REDUCE_CAST_BF16_TO_FP16": "1",
    })
    assert out["ROCM_QUICK_REDUCE_QUANTIZATION"] == "fp8"
    assert out["ROCM_QUICK_REDUCE_MAX_SIZE_BYTES_MB"] == "2048"
    assert out["ROCM_QUICK_REDUCE_CAST_BF16_TO_FP16"] == "1"


# === Operator overrides ========================================================

def test_existing_target_is_not_overwritten(translator: SGLangEnvTranslator) -> None:
    """If the user has already set the SGLang-side name, keep it."""
    out = translator.translate({
        "VLLM_USE_MODELSCOPE": "1",
        "SGLANG_USE_MODELSCOPE": "0",  # operator override
    })
    assert out["SGLANG_USE_MODELSCOPE"] == "0"


def test_target_set_with_no_source_passes_through(
    translator: SGLangEnvTranslator,
) -> None:
    """Operator-set SGLang-side names with no vLLM equivalent flow through."""
    out = translator.translate({"SGLANG_USE_AITER": "0"})
    assert out["SGLANG_USE_AITER"] == "0"
    assert "VLLM_ROCM_USE_AITER" not in out


# === Unrelated env preserved ===================================================

def test_unrelated_env_is_preserved(translator: SGLangEnvTranslator) -> None:
    parent = {
        "PATH": "/usr/bin",
        "HF_HOME": "/cache/hf",
        "CUDA_VISIBLE_DEVICES": "0,1",
        "VLLM_USE_DEEP_GEMM": "1",  # not in ENV_MAP, stays as-is
    }
    out = translator.translate(parent)
    for k, v in parent.items():
        assert out[k] == v


def test_pure_function_does_not_mutate_input(translator: SGLangEnvTranslator) -> None:
    parent = {"VLLM_USE_MODELSCOPE": "1"}
    snapshot = dict(parent)
    translator.translate(parent)
    assert parent == snapshot


# === Notable non-translations ==================================================

def test_vllm_port_translates_to_sglang_port(translator: SGLangEnvTranslator) -> None:
    """Both engines use this as a base port for internal service-port
    allocation (NOT the listen port, which comes from --port). The shim
    used to squat on SGLANG_PORT for its own supervisor->middleware IPC,
    which is what previously made this translation unsafe; the IPC env
    var has since been renamed to VLLM_SHIM_BACKEND_PORT."""
    out = translator.translate({"VLLM_PORT": "9999"})
    assert out["SGLANG_PORT"] == "9999"


def test_per_feature_aiter_toggles_not_translated(
    translator: SGLangEnvTranslator,
) -> None:
    """Per-feature AITER toggles don't have 1:1 SGLang equivalents."""
    out = translator.translate({
        "VLLM_ROCM_USE_AITER_MLA": "1",
        "VLLM_ROCM_USE_AITER_MOE": "1",
        "VLLM_ROCM_USE_AITER_RMSNORM": "1",
    })
    # Source preserved (backend ignores).
    assert out["VLLM_ROCM_USE_AITER_MLA"] == "1"
    # No invented SGLang-side targets.
    assert "SGLANG_USE_AITER_MLA" not in out
    assert "SGLANG_USE_AITER_MOE" not in out
    assert "SGLANG_USE_AITER_RMSNORM" not in out
