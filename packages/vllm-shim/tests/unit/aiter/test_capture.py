"""Tests for the AITER shape-capture orchestration."""

from pathlib import Path

import pytest
from vllm_shim.aiter.capture import (
    REASON_ENABLED,
    REASON_NO_GPU,
    REASON_NO_SHIM_HOME,
    build_callback,
    plan_capture,
    resolve_shim_home,
)
from vllm_shim.aiter.shape_store import ShapeStore
from vllm_shim.cli.rocm_probe import GpuAgent
from vllm_shim.values.parallelism import Parallelism

_GPU = GpuAgent(gfx_target="gfx942", compute_units=304, marketing_name="MI300X")
_CANONICAL_LINE = (
    "shape is M:1024, N:7168, K:512 "
    "dtype=torch.bfloat16 otype=torch.bfloat16 "
    "bias=False, scaleAB=False, bpreshuffle=False, "
    "not found tuned config in /opt/aiter/aiter/configs/bf16_tuned_gemm.csv, "
    "will use default config!"
)


# ---------- plan_capture ----------


def test_plan_disabled_when_no_gpu() -> None:
    plan = plan_capture(
        shim_home=Path("/data/vllm-shim"),
        gpu=None,
        model="gpt2",
        parallelism=Parallelism(),
    )
    assert plan.enabled is False
    assert plan.root is None
    assert plan.reason == REASON_NO_GPU


def test_plan_disabled_when_shim_home_unresolvable() -> None:
    # ``shim_home=None`` is what ``resolve_shim_home`` returns only when
    # even ``Path.home()`` failed - rare, but the plan must still produce
    # a clean disabled state with a stable reason.
    plan = plan_capture(
        shim_home=None, gpu=_GPU, model="gpt2", parallelism=Parallelism()
    )
    assert plan.enabled is False
    assert plan.root is None
    assert plan.reason == REASON_NO_SHIM_HOME


def test_plan_enabled_with_full_environment() -> None:
    plan = plan_capture(
        shim_home=Path("/data/vllm-shim"),
        gpu=_GPU,
        model="moonshotai/Kimi-K2.6",
        parallelism=Parallelism(tp=8, ep=8),
    )
    assert plan.enabled is True
    assert plan.reason == REASON_ENABLED
    assert plan.root == Path(
        "/data/vllm-shim/aiter/shapes/gfx942-304cu/moonshotai--Kimi-K2.6/tp8-ep8"
    )


def test_plan_no_gpu_takes_priority_over_no_shim_home() -> None:
    # When both are missing, the GPU check is the more fundamental
    # prerequisite (AITER capture is meaningless without ROCm); pick a
    # stable reason order so the launch-info dump is predictable.
    plan = plan_capture(
        shim_home=None, gpu=None, model="gpt2", parallelism=Parallelism()
    )
    assert plan.reason == REASON_NO_GPU


# ---------- resolve_shim_home ----------


def test_resolve_shim_home_uses_env_var_when_set(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Operator-deployed pods set VLLM_SHIM_HOME to a PV mount path; the
    # resolver must honour that exactly so capture/restore land there.
    monkeypatch.setenv("VLLM_SHIM_HOME", str(tmp_path))
    assert resolve_shim_home() == tmp_path


def test_resolve_shim_home_expands_tilde(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VLLM_SHIM_HOME", "~/custom-shim")
    resolved = resolve_shim_home()
    assert resolved is not None
    assert "~" not in str(resolved)
    assert resolved.name == "custom-shim"


def test_resolve_shim_home_defaults_to_dot_vllm_shim_under_home(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # No env: fall back to ~/.vllm-shim. Dev hosts and forgetful operators
    # both land somewhere coherent without crashing the shim.
    monkeypatch.delenv("VLLM_SHIM_HOME", raising=False)
    resolved = resolve_shim_home()
    assert resolved is not None
    assert resolved.name == ".vllm-shim"
    assert resolved.parent == Path.home()


def test_resolve_shim_home_returns_path_object() -> None:
    resolved = resolve_shim_home()
    assert resolved is None or isinstance(resolved, Path)


# ---------- build_callback ----------


def test_callback_parses_aiter_line_and_stores_shape(tmp_path: Path) -> None:
    store = ShapeStore(tmp_path)
    cb = build_callback(store)
    cb(_CANONICAL_LINE + "\n")
    assert (tmp_path / "bf16_tuned_gemm.csv").exists()


def test_callback_ignores_non_aiter_lines(tmp_path: Path) -> None:
    # The tee fires the callback for every stderr line; the vast
    # majority will be SGLang's normal logging, not AITER shape misses.
    store = ShapeStore(tmp_path)
    cb = build_callback(store)
    cb("INFO 11-05 12:00:00 sglang ready on :8001\n")
    cb("\n")
    cb("[aiter] initializing rocm runtime\n")
    assert not any(tmp_path.iterdir())


def test_callback_dedups_repeated_lines(tmp_path: Path) -> None:
    # AITER emits the same shape miss every time the kernel is hit;
    # the store's dedup must keep the CSV from growing unbounded.
    store = ShapeStore(tmp_path)
    cb = build_callback(store)
    for _ in range(5):
        cb(_CANONICAL_LINE + "\n")
    csv_path = tmp_path / "bf16_tuned_gemm.csv"
    # Header + one data row.
    assert len(csv_path.read_text().splitlines()) == 2
