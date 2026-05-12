"""Tests for entrypoint helpers (orchestration is exercised in integration)."""

import subprocess
from pathlib import Path

import pytest
from vllm_shim.cli.entrypoint import (
    _maybe_run_startup_tune,
    _parse_tune_budget,
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


# ---------- _maybe_run_startup_tune ----------


def test_tune_skipped_when_budget_zero() -> None:
    calls: list[list[str]] = []

    def runner(cmd: list[str], _timeout: int) -> int:
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

    def runner(cmd: list[str], _timeout: int) -> int:
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

    def runner(cmd: list[str], _timeout: int) -> int:
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

    def runner(cmd: list[str], timeout: int) -> int:
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


def test_tune_swallows_timeout(capsys: pytest.CaptureFixture[str]) -> None:
    # Hard cap exists precisely to keep one bad tune from crashlooping
    # the pod via k8s progressDeadline. The exception must NOT propagate.
    def runner(_cmd: list[str], _timeout: int) -> int:
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
    def runner(_cmd: list[str], _timeout: int) -> int:
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
