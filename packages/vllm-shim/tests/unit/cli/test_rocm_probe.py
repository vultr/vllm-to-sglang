"""Tests for the rocminfo parser and probe."""

import subprocess
from pathlib import Path
from typing import Any

import pytest
from vllm_shim.cli import rocm_probe
from vllm_shim.cli.rocm_probe import GpuAgent

FIXTURE = Path(__file__).parent / "fixtures" / "rocminfo_mi300x.txt"


def test_parse_finds_all_eight_mi300x_gpus() -> None:
    agents = rocm_probe.parse_rocminfo(FIXTURE.read_text())
    assert len(agents) == 8
    assert all(a.gfx_target == "gfx942" for a in agents)
    assert all(a.compute_units == 304 for a in agents)
    assert all(a.marketing_name == "AMD Instinct MI300X" for a in agents)


def test_parse_filters_out_cpu_agents() -> None:
    # The fixture has 2 Intel Xeon CPU agents (Compute Unit: 96) followed
    # by 8 GPUs. If the parser keyed off Compute Unit alone instead of
    # Device Type, we'd see 10 entries here.
    agents = rocm_probe.parse_rocminfo(FIXTURE.read_text())
    assert all(a.gfx_target.startswith("gfx") for a in agents)
    assert not any("Xeon" in a.marketing_name for a in agents)


def test_parse_empty_input_returns_empty_list() -> None:
    assert rocm_probe.parse_rocminfo("") == []


def test_parse_header_only_input_returns_empty_list() -> None:
    text = "HSA System Attributes\nRuntime Version: 1.18\n"
    assert rocm_probe.parse_rocminfo(text) == []


def test_parse_skips_block_missing_required_fields() -> None:
    # Block declares itself a GPU but is missing Name/Compute Unit;
    # the parser must skip it rather than raise.
    text = "Agent 1\n  Device Type: GPU\n  Marketing Name: Mystery Card\n"
    assert rocm_probe.parse_rocminfo(text) == []


def test_bucket_formats_gfx_target_and_cu_count() -> None:
    agent = GpuAgent(gfx_target="gfx942", compute_units=304, marketing_name="MI300X")
    assert rocm_probe.bucket(agent) == "gfx942-304cu"


def test_bucket_distinguishes_skus_with_same_gfx() -> None:
    # Hypothetical: MI300A would share gfx942 but have a different CU count.
    a = GpuAgent(gfx_target="gfx942", compute_units=304, marketing_name="MI300X")
    b = GpuAgent(gfx_target="gfx942", compute_units=228, marketing_name="MI300A")
    assert rocm_probe.bucket(a) != rocm_probe.bucket(b)


def test_probe_returns_first_gpu_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = FIXTURE.read_text()

    def fake_run(*_args: Any, **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args=["rocminfo"], returncode=0, stdout=captured, stderr=""
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    agent = rocm_probe.probe()
    assert agent == GpuAgent(
        gfx_target="gfx942", compute_units=304, marketing_name="AMD Instinct MI300X"
    )


def test_probe_returns_none_when_rocminfo_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    def raise_fnf(*_args: Any, **_kwargs: Any) -> Any:
        raise FileNotFoundError("rocminfo")

    monkeypatch.setattr(subprocess, "run", raise_fnf)
    assert rocm_probe.probe() is None


def test_probe_returns_none_when_rocminfo_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    def raise_called(*_args: Any, **_kwargs: Any) -> Any:
        raise subprocess.CalledProcessError(returncode=1, cmd=["rocminfo"])

    monkeypatch.setattr(subprocess, "run", raise_called)
    assert rocm_probe.probe() is None


def test_probe_returns_none_when_no_gpu_agents(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(*_args: Any, **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args=["rocminfo"], returncode=0, stdout="", stderr=""
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert rocm_probe.probe() is None
