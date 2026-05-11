"""Tests for the launch-info collection, write, summary, and CLI entry."""

import json
from pathlib import Path

import pytest
from vllm_shim.cli import info
from vllm_shim.values.port_allocation import PortAllocation
from vllm_shim.values.service_address import ServiceAddress


def _sample_args() -> dict[str, object]:
    return {
        "original_argv": ("serve", "moonshotai/Kimi-K2.6", "--tensor-parallel-size", "8"),
        "backend_name": "sglang",
        "model_original": "moonshotai/Kimi-K2.6",
        "model_resolved": "/data/hub/models--moonshotai--Kimi-K2.6/snapshots/abc",
        "revision": None,
        "listen": ServiceAddress("0.0.0.0", 8000),
        "ports": PortAllocation.from_listen(8000),
        "backend_argv": ("python", "-m", "sglang.launch_server", "--port", "8001"),
        "dropped_args": ("--enable-chunked-prefill",),
    }


def test_collect_assembles_expected_shape() -> None:
    out = info.collect(
        **_sample_args(),  # type: ignore[arg-type]
        parent_env={"VLLM_SHIM_BACKEND": "sglang"},
        backend_env={"VLLM_SHIM_BACKEND": "sglang"},
    )
    assert out["backend"] == "sglang"
    assert out["model"] == {
        "original": "moonshotai/Kimi-K2.6",
        "resolved": "/data/hub/models--moonshotai--Kimi-K2.6/snapshots/abc",
        "revision": None,
    }
    assert out["listen"] == "0.0.0.0:8000"
    assert out["ports"] == {"frontend": 8000, "backend": 8001, "middleware": 8002}
    assert out["dropped_args"] == ["--enable-chunked-prefill"]


def test_collect_env_translation_shows_only_added_keys() -> None:
    out = info.collect(
        **_sample_args(),  # type: ignore[arg-type]
        parent_env={"VLLM_HOST_IP": "1.2.3.4", "PATH": "/usr/bin"},
        backend_env={
            "VLLM_HOST_IP": "1.2.3.4",
            "PATH": "/usr/bin",
            "SGLANG_HOST_IP": "1.2.3.4",
        },
    )
    assert out["env_translation"] == {"SGLANG_HOST_IP": "1.2.3.4"}


def test_collect_filters_shim_config_to_known_keys() -> None:
    out = info.collect(
        **_sample_args(),  # type: ignore[arg-type]
        parent_env={
            "VLLM_SHIM_BACKEND": "sglang",
            "SGLANG_TOOL_CALL_PARSER": "qwen3_coder",
            "UNRELATED_VAR": "ignore-me",
        },
        backend_env={},
    )
    assert out["shim_config"] == {
        "VLLM_SHIM_BACKEND": "sglang",
        "SGLANG_TOOL_CALL_PARSER": "qwen3_coder",
    }


def test_collect_picks_up_hf_cache_vars_when_present() -> None:
    out = info.collect(
        **_sample_args(),  # type: ignore[arg-type]
        parent_env={"HF_HOME": "/data", "HF_HUB_OFFLINE": "0"},
        backend_env={},
    )
    assert out["hf_cache"] == {"HF_HOME": "/data", "HF_HUB_OFFLINE": "0"}


def test_write_produces_pretty_json_with_trailing_newline(tmp_path: Path) -> None:
    target = tmp_path / "info.json"
    info.write({"a": 1, "b": [2, 3]}, path=target)
    raw = target.read_text()
    assert raw.endswith("\n")
    assert json.loads(raw) == {"a": 1, "b": [2, 3]}
    assert "\n" in raw.strip()  # multi-line, not a one-liner


def test_print_summary_writes_dropped_and_renames(
    capsys: pytest.CaptureFixture[str],
) -> None:
    info.print_summary(
        {
            "shim_version": "0.0.1",
            "backend": "sglang",
            "listen": "0.0.0.0:8000",
            "model": {
                "original": "org/m",
                "resolved": "/cache/m/snapshots/abc",
                "revision": "abc",
            },
            "backend_argv": ["python", "-m", "sglang.launch_server"],
            "dropped_args": ["--enable-chunked-prefill"],
            "env_translation": {"SGLANG_HOST_IP": "1.2.3.4"},
        }
    )
    err = capsys.readouterr().err
    assert "vllm-shim 0.0.1 -> sglang listening on 0.0.0.0:8000" in err
    assert "org/m@abc -> /cache/m/snapshots/abc" in err
    assert "dropped: --enable-chunked-prefill" in err
    assert "SGLANG_HOST_IP=1.2.3.4" in err


def test_print_summary_omits_optional_sections_when_empty(
    capsys: pytest.CaptureFixture[str],
) -> None:
    info.print_summary(
        {
            "shim_version": "0.0.1",
            "backend": "sglang",
            "listen": "0.0.0.0:8000",
            "model": {"original": "m", "resolved": "m", "revision": None},
            "backend_argv": ["python", "-m", "sglang.launch_server"],
            "dropped_args": [],
            "env_translation": {},
        }
    )
    err = capsys.readouterr().err
    assert "dropped:" not in err
    assert "env renames:" not in err


def test_main_prints_file_when_present(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    target = tmp_path / "info.json"
    target.write_text('{"shim_version": "0.0.1"}\n')
    monkeypatch.setattr(info, "INFO_PATH", target)
    assert info.main() == 0
    assert capsys.readouterr().out == '{"shim_version": "0.0.1"}\n'


def test_main_returns_1_and_warns_when_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(info, "INFO_PATH", tmp_path / "missing.json")
    assert info.main() == 1
    assert "No launch info" in capsys.readouterr().err
