"""Tests for the ROCm perf env-var defaults."""

from pathlib import Path

from vllm_shim.cli.rocm_perf import rocm_perf_defaults
from vllm_shim.cli.rocm_probe import GpuAgent

_MI300X = GpuAgent(gfx_target="gfx942", compute_units=304, marketing_name="MI300X")
_MI250X = GpuAgent(gfx_target="gfx90a", compute_units=104, marketing_name="MI250X")


def test_no_gpu_returns_empty() -> None:
    # CUDA hosts and dev boxes: no ROCm GPU detected. The shim must
    # not push ROCm-specific env vars onto an environment that won't
    # understand them.
    assert rocm_perf_defaults(None, Path("/data/vllm-shim")) == {}


def test_no_shim_home_returns_empty() -> None:
    # MIOPEN_* paths need a stable home; without one we'd have nowhere
    # to point them. Returning empty also avoids fragmenting the
    # decision logic in the entrypoint.
    assert rocm_perf_defaults(_MI300X, None) == {}


def test_generic_defaults_apply_to_any_rocm_gpu(tmp_path: Path) -> None:
    # MI250X (gfx90a): generic ROCm defaults yes, MI300-specific no.
    out = rocm_perf_defaults(_MI250X, tmp_path)
    assert out["HF_HUB_ENABLE_HF_TRANSFER"] == "1"
    assert out["SAFETENSORS_FAST_GPU"] == "1"
    assert out["TORCH_BLAS_PREFER_HIPBLASLT"] == "1"
    assert "GPU_MAX_HW_QUEUES" not in out
    assert "TORCH_NCCL_HIGH_PRIORITY" not in out


def test_jit_cache_paths_anchor_under_shim_home(tmp_path: Path) -> None:
    # Triton + TorchInductor + SGLang caches all share the PV so JIT
    # artifacts survive pod restarts. Without this, first request on
    # every restart re-pays the compile cost.
    out = rocm_perf_defaults(_MI300X, tmp_path)
    assert out["TRITON_CACHE_DIR"] == str(tmp_path / "triton")
    assert out["TORCHINDUCTOR_CACHE_DIR"] == str(tmp_path / "torchinductor")
    assert out["SGLANG_CACHE_DIR"] == str(tmp_path / "sglang")


def test_miopen_paths_route_under_shim_home(tmp_path: Path) -> None:
    # Both vars point at the same dir so MIOpen finds its db where it
    # caches its custom kernels. Routing under shim_home means the
    # persistent volume that already holds AITER configs holds the
    # MIOpen cache too (one PV, one survival story).
    out = rocm_perf_defaults(_MI300X, tmp_path)
    expected = str(tmp_path / "miopen")
    assert out["MIOPEN_USER_DB_PATH"] == expected
    assert out["MIOPEN_CUSTOM_CACHE_DIR"] == expected


def test_mi300x_adds_sku_specific_vars(tmp_path: Path) -> None:
    # gfx942 lights up the recent AMD always-on recs. The values
    # (2 / 1) are MI300-class specific; future SKUs need their own
    # audit before reusing them.
    out = rocm_perf_defaults(_MI300X, tmp_path)
    assert out["GPU_MAX_HW_QUEUES"] == "2"
    assert out["TORCH_NCCL_HIGH_PRIORITY"] == "1"


def test_keys_are_a_known_set(tmp_path: Path) -> None:
    # Pin the surface: anything in this set lands in the operator's
    # backend env via setdefault. Adding a new default should be a
    # deliberate edit here too so reviewers see the change.
    out = rocm_perf_defaults(_MI300X, tmp_path)
    assert set(out.keys()) == {
        "HF_HUB_ENABLE_HF_TRANSFER",
        "SAFETENSORS_FAST_GPU",
        "MIOPEN_USER_DB_PATH",
        "MIOPEN_CUSTOM_CACHE_DIR",
        "TORCH_BLAS_PREFER_HIPBLASLT",
        "TRITON_CACHE_DIR",
        "TORCHINDUCTOR_CACHE_DIR",
        "SGLANG_CACHE_DIR",
        "GPU_MAX_HW_QUEUES",
        "TORCH_NCCL_HIGH_PRIORITY",
    }


def test_pure_function_no_filesystem_writes(tmp_path: Path) -> None:
    # Cache dirs (miopen/triton/torchinductor/sglang) do not need to
    # exist at the time we set the env vars; the consumers create
    # them on first write. Asserting we don't pre-create catches
    # accidental mkdir creep.
    rocm_perf_defaults(_MI300X, tmp_path)
    for name in ("miopen", "triton", "torchinductor", "sglang"):
        assert not (tmp_path / name).exists()
