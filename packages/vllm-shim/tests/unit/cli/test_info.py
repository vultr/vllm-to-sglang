"""Tests for the launch-info collection, write, summary, and CLI entry."""

import json
from pathlib import Path

import pytest
from vllm_shim.aiter.capture import REASON_ENABLED, REASON_NO_GPU, CapturePlan
from vllm_shim.aiter.restore import RestorePlan
from vllm_shim.cli import info
from vllm_shim.values.port_allocation import PortAllocation
from vllm_shim.values.service_address import ServiceAddress


def _disabled_capture() -> CapturePlan:
    return CapturePlan(enabled=False, root=None, reason=REASON_NO_GPU)


def _disabled_restore() -> RestorePlan:
    return RestorePlan(enabled=False, source=None, reason=REASON_NO_GPU)


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
        "aiter_capture": _disabled_capture(),
        "aiter_restore": _disabled_restore(),
        "aiter_restored": {},
        "rocm_perf": {},
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


def test_collect_aiter_capture_disabled_shape() -> None:
    out = info.collect(
        **_sample_args(),  # type: ignore[arg-type]
        parent_env={},
        backend_env={},
    )
    # Disabled plans must surface the reason but null the root so the JSON
    # dump is unambiguous (no stale path from a previous run).
    assert out["aiter_capture"] == {
        "enabled": False,
        "root": None,
        "reason": REASON_NO_GPU,
    }


def test_collect_aiter_capture_enabled_shape() -> None:
    args = _sample_args()
    args["aiter_capture"] = CapturePlan(
        enabled=True,
        root=Path("/data/vllm-shim/aiter/shapes/gfx942-304cu/m/tp8"),
        reason=REASON_ENABLED,
    )
    out = info.collect(
        **args,  # type: ignore[arg-type]
        parent_env={},
        backend_env={},
    )
    # Path objects get stringified so the dict is JSON-serialisable as is.
    assert out["aiter_capture"]["enabled"] is True
    assert out["aiter_capture"]["root"] == str(
        Path("/data/vllm-shim/aiter/shapes/gfx942-304cu/m/tp8")
    )
    assert out["aiter_capture"]["reason"] == REASON_ENABLED


def test_collect_aiter_restore_enabled_shape() -> None:
    args = _sample_args()
    args["aiter_restore"] = RestorePlan(
        enabled=True,
        source=Path("/data/vllm-shim/aiter/configs/gfx942-304cu"),
        reason=REASON_ENABLED,
    )
    args["aiter_restored"] = {
        "AITER_CONFIG_GEMM_BF16": "/data/vllm-shim/.../bf16_tuned_gemm.csv",
        "AITER_CONFIG_GEMM_A8W8": "/data/vllm-shim/.../a8w8_tuned_gemm.csv",
    }
    out = info.collect(
        **args,  # type: ignore[arg-type]
        parent_env={},
        backend_env={},
    )
    assert out["aiter_restore"]["enabled"] is True
    assert out["aiter_restore"]["source"] == str(
        Path("/data/vllm-shim/aiter/configs/gfx942-304cu")
    )
    # The serialized overrides are the env-var -> path mapping AITER
    # will read at import time; tests assert on the dict directly so
    # any divergence between the in-memory and dumped shapes is loud.
    assert out["aiter_restore"]["overrides"] == {
        "AITER_CONFIG_GEMM_BF16": "/data/vllm-shim/.../bf16_tuned_gemm.csv",
        "AITER_CONFIG_GEMM_A8W8": "/data/vllm-shim/.../a8w8_tuned_gemm.csv",
    }


def test_collect_aiter_restore_disabled_shape() -> None:
    out = info.collect(
        **_sample_args(),  # type: ignore[arg-type]
        parent_env={},
        backend_env={},
    )
    assert out["aiter_restore"] == {
        "enabled": False,
        "source": None,
        "reason": REASON_NO_GPU,
        "overrides": {},
    }


def test_collect_rocm_perf_carries_applied_defaults() -> None:
    args = _sample_args()
    args["rocm_perf"] = {
        "GPU_MAX_HW_QUEUES": "2",
        "MIOPEN_USER_DB_PATH": "/data/vllm-shim/miopen",
    }
    out = info.collect(
        **args,  # type: ignore[arg-type]
        parent_env={},
        backend_env={},
    )
    # Surfaces the applied defaults verbatim so operators can see
    # *exactly* what the shim injected (vs. what the base image or
    # their own pod spec set).
    assert out["rocm_perf"] == {
        "GPU_MAX_HW_QUEUES": "2",
        "MIOPEN_USER_DB_PATH": "/data/vllm-shim/miopen",
    }


def test_collect_rocm_perf_empty_when_disabled() -> None:
    out = info.collect(
        **_sample_args(),  # type: ignore[arg-type]
        parent_env={},
        backend_env={},
    )
    assert out["rocm_perf"] == {}


def test_write_produces_pretty_json_with_trailing_newline(tmp_path: Path) -> None:
    target = tmp_path / "info.json"
    info.write({"a": 1, "b": [2, 3]}, path=target)
    raw = target.read_text()
    assert raw.endswith("\n")
    assert json.loads(raw) == {"a": 1, "b": [2, 3]}
    assert "\n" in raw.strip()  # multi-line, not a one-liner


_DISABLED_CAPTURE_DICT: dict[str, object] = {
    "enabled": False,
    "root": None,
    "reason": REASON_NO_GPU,
}
_DISABLED_RESTORE_DICT: dict[str, object] = {
    "enabled": False,
    "source": None,
    "reason": REASON_NO_GPU,
    "overrides": {},
}


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
            "aiter_capture": _DISABLED_CAPTURE_DICT,
            "aiter_restore": _DISABLED_RESTORE_DICT,
        }
    )
    err = capsys.readouterr().err
    assert "vllm-shim 0.0.1 -> sglang listening on 0.0.0.0:8000" in err
    assert "org/m@abc -> /cache/m/snapshots/abc" in err
    assert "dropped: --enable-chunked-prefill" in err
    assert "SGLANG_HOST_IP=1.2.3.4" in err
    assert f"aiter capture: disabled ({REASON_NO_GPU})" in err
    assert f"aiter restore: disabled ({REASON_NO_GPU})" in err


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
            "aiter_capture": _DISABLED_CAPTURE_DICT,
            "aiter_restore": _DISABLED_RESTORE_DICT,
        }
    )
    err = capsys.readouterr().err
    assert "dropped:" not in err
    assert "env renames:" not in err


def test_print_summary_shows_capture_path_when_enabled(
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
            "aiter_capture": {
                "enabled": True,
                "root": "/data/vllm-shim/aiter/shapes/gfx942-304cu/m/tp8",
                "reason": REASON_ENABLED,
            },
            "aiter_restore": _DISABLED_RESTORE_DICT,
        }
    )
    err = capsys.readouterr().err
    # Operators reading pod logs should see *where* shapes will land so
    # they can tail the directory while loading; the reason string adds
    # nothing when enabled.
    assert (
        "aiter capture: enabled -> /data/vllm-shim/aiter/shapes/gfx942-304cu/m/tp8"
        in err
    )


def test_print_summary_shows_restore_count_and_env_vars_when_enabled(
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
            "aiter_capture": _DISABLED_CAPTURE_DICT,
            "aiter_restore": {
                "enabled": True,
                "source": "/data/vllm-shim/aiter/configs/gfx942-304cu",
                "reason": REASON_ENABLED,
                "overrides": {
                    "AITER_CONFIG_GEMM_BF16": "/data/vllm-shim/.../bf16_tuned_gemm.csv",
                    "AITER_CONFIG_GEMM_A8W8": "/data/vllm-shim/.../a8w8_tuned_gemm.csv",
                },
            },
        }
    )
    err = capsys.readouterr().err
    # The count + env-var names tell the operator which AITER targets
    # were picked up. Paths are in the JSON dump if anyone wants them.
    assert "aiter restore: 2 configs from" in err
    assert "AITER_CONFIG_GEMM_BF16" in err
    assert "AITER_CONFIG_GEMM_A8W8" in err


def test_print_summary_shows_rocm_perf_when_applied(
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
            "aiter_capture": _DISABLED_CAPTURE_DICT,
            "aiter_restore": _DISABLED_RESTORE_DICT,
            "rocm_perf": {
                "GPU_MAX_HW_QUEUES": "2",
                "TORCH_BLAS_PREFER_HIPBLASLT": "1",
            },
        }
    )
    err = capsys.readouterr().err
    # Operator-facing line: count + sorted env var names. Paths and
    # values live in the JSON dump; the stderr summary stays compact.
    assert "rocm perf: 2 defaults" in err
    assert "GPU_MAX_HW_QUEUES" in err
    assert "TORCH_BLAS_PREFER_HIPBLASLT" in err


def test_print_summary_omits_rocm_perf_when_empty(
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
            "aiter_capture": _DISABLED_CAPTURE_DICT,
            "aiter_restore": _DISABLED_RESTORE_DICT,
            "rocm_perf": {},
        }
    )
    err = capsys.readouterr().err
    # No GPU / disabled path: don't print a useless line.
    assert "rocm perf" not in err


def test_print_summary_says_nothing_to_restore_when_source_empty(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # First-ever run: the source dir exists (or doesn't) but has no
    # configs. The summary should say so explicitly rather than print
    # an empty list, so the operator understands restore RAN but found
    # nothing - distinct from restore being disabled.
    info.print_summary(
        {
            "shim_version": "0.0.1",
            "backend": "sglang",
            "listen": "0.0.0.0:8000",
            "model": {"original": "m", "resolved": "m", "revision": None},
            "backend_argv": ["python", "-m", "sglang.launch_server"],
            "dropped_args": [],
            "env_translation": {},
            "aiter_capture": _DISABLED_CAPTURE_DICT,
            "aiter_restore": {
                "enabled": True,
                "source": "/data/vllm-shim/aiter/configs/gfx942-304cu",
                "reason": REASON_ENABLED,
                "overrides": {},
            },
        }
    )
    err = capsys.readouterr().err
    assert "aiter restore: enabled, nothing to restore from" in err


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
