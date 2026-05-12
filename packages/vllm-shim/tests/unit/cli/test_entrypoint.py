"""Tests for entrypoint helpers (orchestration is exercised in integration)."""

import subprocess
from collections.abc import Mapping
from pathlib import Path

import pytest
from vllm_shim.cli.entrypoint import (
    _maybe_run_startup_tune,
    _parse_tune_budget,
    _parse_tune_hot,
    _pin_served_model_name,
)
from vllm_shim.cli.rocm_probe import GpuAgent

_MI300X = GpuAgent(gfx_target="gfx942", compute_units=304, marketing_name="MI300X")


def test_no_pinning_when_path_unchanged() -> None:
    out = _pin_served_model_name(("--trust-remote-code",), "/data/models/m", "/data/models/m")
    assert out == ("--trust-remote-code",)


def test_pins_original_when_resolved_to_snapshot_directory() -> None:
    out = _pin_served_model_name(
        ("--trust-remote-code",),
        "moonshotai/Kimi-K2.6",
        "/data/hub/models--moonshotai--Kimi-K2.6/snapshots/abc",
    )
    assert out == (
        "--trust-remote-code",
        "--served-model-name",
        "moonshotai/Kimi-K2.6",
    )


def test_respects_existing_served_model_name_space_form() -> None:
    out = _pin_served_model_name(
        ("--served-model-name", "alias", "--trust-remote-code"),
        "moonshotai/Kimi-K2.6",
        "/data/hub/models--moonshotai--Kimi-K2.6/snapshots/abc",
    )
    assert out == ("--served-model-name", "alias", "--trust-remote-code")


def test_respects_existing_served_model_name_equals_form() -> None:
    out = _pin_served_model_name(
        ("--served-model-name=alias",),
        "moonshotai/Kimi-K2.6",
        "/data/hub/models--moonshotai--Kimi-K2.6/snapshots/abc",
    )
    assert out == ("--served-model-name=alias",)


# ---------- _parse_tune_budget ----------


def test_parse_tune_budget_unset_means_off() -> None:
    assert _parse_tune_budget(None) == 0
    assert _parse_tune_budget("") == 0


def test_parse_tune_budget_zero_means_off() -> None:
    assert _parse_tune_budget("0") == 0


def test_parse_tune_budget_negative_means_off() -> None:
    # Negative values are meaningless as a wall-clock budget; treat as
    # off rather than raising so a pod with a typo doesn't crashloop.
    assert _parse_tune_budget("-5") == 0


def test_parse_tune_budget_non_numeric_means_off() -> None:
    # An operator typing "true" expecting a boolean shouldn't accidentally
    # opt in with an undefined budget. Strict numeric parse, fall back to 0.
    assert _parse_tune_budget("true") == 0
    assert _parse_tune_budget("yes") == 0


def test_parse_tune_budget_positive_int_is_returned_verbatim() -> None:
    assert _parse_tune_budget("900") == 900
    assert _parse_tune_budget("3600") == 3600


# ---------- _parse_tune_hot ----------


def test_parse_tune_hot_unset_means_no_filter() -> None:
    # None == "tune all captured shapes"; default off.
    assert _parse_tune_hot(None) is None
    assert _parse_tune_hot("") is None


def test_parse_tune_hot_zero_means_no_filter() -> None:
    # "0" reads as "off" so it composes cleanly with helm chart
    # patterns that fill an unset env with "0".
    assert _parse_tune_hot("0") is None


def test_parse_tune_hot_negative_means_no_filter() -> None:
    assert _parse_tune_hot("-1") is None
    assert _parse_tune_hot("-100") is None


def test_parse_tune_hot_non_numeric_means_no_filter() -> None:
    assert _parse_tune_hot("hot") is None
    assert _parse_tune_hot("true") is None


def test_parse_tune_hot_positive_int_is_returned_verbatim() -> None:
    assert _parse_tune_hot("100") == 100
    assert _parse_tune_hot("500") == 500


# ---------- _maybe_run_startup_tune ----------


def test_tune_skipped_when_budget_zero() -> None:
    calls: list[list[str]] = []

    def runner(cmd: list[str], _timeout: int, _env: Mapping[str, str] | None) -> int:
        calls.append(cmd)
        return 0

    _maybe_run_startup_tune(
        shim_home=Path("/data/vllm-shim"),
        gpu=_MI300X,
        budget_seconds=0,
        run=runner,
    )
    assert calls == []


def test_tune_skipped_when_no_gpu() -> None:
    # CUDA host / dev box: same gate as AITER capture/restore. Without
    # a ROCm GPU there's nothing to tune.
    calls: list[list[str]] = []

    def runner(cmd: list[str], _timeout: int, _env: Mapping[str, str] | None) -> int:
        calls.append(cmd)
        return 0

    _maybe_run_startup_tune(
        shim_home=Path("/data/vllm-shim"),
        gpu=None,
        budget_seconds=900,
        run=runner,
    )
    assert calls == []


def test_tune_skipped_when_no_shim_home() -> None:
    calls: list[list[str]] = []

    def runner(cmd: list[str], _timeout: int, _env: Mapping[str, str] | None) -> int:
        calls.append(cmd)
        return 0

    _maybe_run_startup_tune(
        shim_home=None,
        gpu=_MI300X,
        budget_seconds=900,
        run=runner,
    )
    assert calls == []


def test_tune_invokes_runner_with_bucket_and_budget() -> None:
    received: list[tuple[list[str], int]] = []

    def runner(cmd: list[str], timeout: int, _env: Mapping[str, str] | None) -> int:
        received.append((cmd, timeout))
        return 0

    _maybe_run_startup_tune(
        shim_home=Path("/data/vllm-shim"),
        gpu=_MI300X,
        budget_seconds=900,
        run=runner,
    )
    assert len(received) == 1
    cmd, timeout = received[0]
    # Bucket key must match what restore/capture use so the tuner reads
    # the right shapes/configs subtree.
    assert "--bucket" in cmd
    assert cmd[cmd.index("--bucket") + 1] == "gfx942-304cu"
    assert "--shim-home" in cmd
    assert timeout == 900
    # Without VLLM_SHIM_TUNE_AT_STARTUP_HOT set, the subprocess gets
    # no --hot flag and tunes every captured shape (existing behavior).
    assert "--hot" not in cmd


def test_tune_passes_hot_flag_when_set(
    capsys: pytest.CaptureFixture[str],
) -> None:
    received: list[list[str]] = []

    def runner(cmd: list[str], _timeout: int, _env: Mapping[str, str] | None) -> int:
        received.append(cmd)
        return 0

    _maybe_run_startup_tune(
        shim_home=Path("/data/vllm-shim"),
        gpu=_MI300X,
        budget_seconds=900,
        hot=100,
        run=runner,
    )
    cmd = received[0]
    assert "--hot" in cmd
    assert cmd[cmd.index("--hot") + 1] == "100"
    # Pod-log line surfaces both knobs so an operator scanning the
    # startup banner can confirm both took effect.
    err = capsys.readouterr().err
    assert "900s budget" in err
    assert "--hot 100" in err


def test_tune_forwards_env_to_runner() -> None:
    # backend_env carries AITER_JIT_DIR (from rocm_perf_defaults) and
    # any AITER_CONFIG_* overrides from the restore step. Without
    # forwarding, AITER's JIT loader falls back to its package-relative
    # site-packages .so (the unset-AITER_JIT_DIR branch in
    # aiter/jit/core.py:get_module_custom_op) and the patched kernel
    # source is never compiled. The base image's pre-built .so wins
    # and shim-local AITER patches are silently masked.
    received: list[Mapping[str, str] | None] = []

    def runner(_cmd: list[str], _timeout: int, env: Mapping[str, str] | None) -> int:
        received.append(env)
        return 0

    backend_env = {
        "AITER_JIT_DIR": "/data/vllm-shim/aiter/jit-abc123",
        "AITER_CONFIG_BF16_TUNED_GEMM": "/data/vllm-shim/aiter/configs/x.csv",
        "PATH": "/opt/shim/bin:/opt/venv/bin",
    }
    _maybe_run_startup_tune(
        shim_home=Path("/data/vllm-shim"),
        gpu=_MI300X,
        budget_seconds=900,
        env=backend_env,
        run=runner,
    )
    assert received == [backend_env]


def test_tune_swallows_timeout(capsys: pytest.CaptureFixture[str]) -> None:
    # Hard cap exists precisely to keep one bad tune from crashlooping
    # the pod via k8s progressDeadline. The exception must NOT propagate.
    def runner(_cmd: list[str], _timeout: int, _env: Mapping[str, str] | None) -> int:
        raise subprocess.TimeoutExpired(cmd="vllm-shim-tune", timeout=10)

    _maybe_run_startup_tune(
        shim_home=Path("/data/vllm-shim"),
        gpu=_MI300X,
        budget_seconds=10,
        run=runner,
    )
    err = capsys.readouterr().err
    assert "exceeded 10s budget" in err


def test_tune_swallows_other_failures(capsys: pytest.CaptureFixture[str]) -> None:
    # File-not-found (vllm-shim-tune not on PATH), permission errors,
    # AITER blowups - all best-effort. Backend still launches.
    def runner(_cmd: list[str], _timeout: int, _env: Mapping[str, str] | None) -> int:
        raise FileNotFoundError("vllm-shim-tune not found")

    _maybe_run_startup_tune(
        shim_home=Path("/data/vllm-shim"),
        gpu=_MI300X,
        budget_seconds=900,
        run=runner,
    )
    err = capsys.readouterr().err
    assert "failed" in err
    assert "continuing" in err
