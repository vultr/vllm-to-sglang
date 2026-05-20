"""Tests for VLLMLauncher.build_command and its env-var overrides."""

from vllm_shim.backend.vllm.launcher import (
    DEFAULT_BACKEND_PYTHON,
    DEFAULT_TOOL_CALL_PARSER,
    VLLMLauncher,
)
from vllm_shim.values.service_address import ServiceAddress


def test_invokes_module_form_with_default_python(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.delenv("VLLM_BACKEND_PYTHON", raising=False)
    cmd = VLLMLauncher().build_command("m", ServiceAddress("h", 1), [])
    assert cmd[:4] == [
        DEFAULT_BACKEND_PYTHON,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
    ]


def test_backend_python_env_override(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("VLLM_BACKEND_PYTHON", "/opt/venv/bin/python")
    cmd = VLLMLauncher().build_command("m", ServiceAddress("h", 1), [])
    assert cmd[0] == "/opt/venv/bin/python"
    assert cmd[1:3] == ["-m", "vllm.entrypoints.openai.api_server"]


def test_command_includes_model_host_port() -> None:
    cmd = VLLMLauncher().build_command("org/m", ServiceAddress("0.0.0.0", 8001), [])
    idx_model = cmd.index("--model")
    assert cmd[idx_model + 1] == "org/m"
    idx_host = cmd.index("--host")
    assert cmd[idx_host + 1] == "0.0.0.0"
    idx_port = cmd.index("--port")
    assert cmd[idx_port + 1] == "8001"


def test_default_tool_call_parser_qwen3_coder(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.delenv("VLLM_TOOL_CALL_PARSER", raising=False)
    cmd = VLLMLauncher().build_command("m", ServiceAddress("h", 1), [])
    idx = cmd.index("--tool-call-parser")
    assert cmd[idx + 1] == DEFAULT_TOOL_CALL_PARSER == "qwen3_coder"


def test_tool_call_parser_env_override(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("VLLM_TOOL_CALL_PARSER", "mistral")
    cmd = VLLMLauncher().build_command("m", ServiceAddress("h", 1), [])
    idx = cmd.index("--tool-call-parser")
    assert cmd[idx + 1] == "mistral"


def test_extra_args_appended_at_tail() -> None:
    cmd = VLLMLauncher().build_command("m", ServiceAddress("h", 1), ["--tensor-parallel-size", "8"])
    assert cmd[-2:] == ["--tensor-parallel-size", "8"]


def test_extra_args_after_tool_call_parser() -> None:
    cmd = VLLMLauncher().build_command("m", ServiceAddress("h", 1), ["--dtype", "bfloat16"])
    parser_idx = cmd.index("--tool-call-parser")
    dtype_idx = cmd.index("--dtype")
    assert dtype_idx > parser_idx
