"""Tests for the AITER shape-capture orchestration."""

from pathlib import Path

import pytest
from vllm_shim.aiter.capture import (
    REASON_ENABLED,
    REASON_NO_GPU,
    REASON_NO_HF_HOME,
    build_callback,
    plan_capture,
    resolve_hf_home,
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
        hf_home=Path("/data/hf"), gpu=None, model="gpt2", parallelism=Parallelism()
    )
    assert plan.enabled is False
    assert plan.root is None
    assert plan.reason == REASON_NO_GPU


def test_plan_disabled_when_hf_home_unresolvable() -> None:
    # ``hf_home=None`` is what ``resolve_hf_home`` returns only when even
    # the home directory can't be expanded - rare, but the plan must
    # still produce a clean disabled state with a stable reason.
    plan = plan_capture(
        hf_home=None, gpu=_GPU, model="gpt2", parallelism=Parallelism()
    )
    assert plan.enabled is False
    assert plan.root is None
    assert plan.reason == REASON_NO_HF_HOME


def test_plan_enabled_with_full_environment() -> None:
    plan = plan_capture(
        hf_home=Path("/data/hf"),
        gpu=_GPU,
        model="moonshotai/Kimi-K2.6",
        parallelism=Parallelism(tp=8, ep=8),
    )
    assert plan.enabled is True
    assert plan.reason == REASON_ENABLED
    assert plan.root == Path(
        "/data/hf/vllm-shim/aiter-shapes/gfx942-304cu/"
        "moonshotai--Kimi-K2.6/tp8-ep8"
    )


def test_plan_no_gpu_takes_priority_over_no_hf_home() -> None:
    # When both are missing, the GPU check is the more fundamental
    # prerequisite (AITER capture is meaningless without ROCm); pick a
    # stable reason order so the launch-info dump is predictable.
    plan = plan_capture(
        hf_home=None, gpu=None, model="gpt2", parallelism=Parallelism()
    )
    assert plan.reason == REASON_NO_GPU


# ---------- resolve_hf_home ----------


def test_resolve_hf_home_uses_env_var_when_set(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # huggingface_hub.constants.HF_HOME is computed at import time, but
    # both the hub and our resolver honour HF_HOME if set. We round-trip
    # via the resolver to confirm the env var wins over the default.
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    # Force re-import path: clear any cached module so HF_HOME is recomputed.
    import importlib

    import huggingface_hub.constants
    importlib.reload(huggingface_hub.constants)
    resolved = resolve_hf_home()
    assert resolved == tmp_path


def test_resolve_hf_home_returns_path_object() -> None:
    # Sanity: the resolver always returns a Path (or None), never str.
    resolved = resolve_hf_home()
    assert resolved is None or isinstance(resolved, Path)


def test_resolve_hf_home_falls_back_without_hf_hub(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # If huggingface_hub import fails (paranoid path), we should still
    # find the env var.
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    import sys

    monkeypatch.setitem(sys.modules, "huggingface_hub.constants", None)
    resolved = resolve_hf_home()
    assert resolved == tmp_path


def test_resolve_hf_home_default_is_dot_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # With no env var and no huggingface_hub, fall back to the documented
    # HF default at ~/.cache/huggingface.
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    import sys

    monkeypatch.setitem(sys.modules, "huggingface_hub.constants", None)
    resolved = resolve_hf_home()
    assert resolved is not None
    assert resolved.parts[-2:] == (".cache", "huggingface")


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
