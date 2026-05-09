"""Tests for TRTLLMLauncher.build_command and its env-var knobs."""

from vllm_shim.backend.trtllm.launcher import TRTLLMLauncher
from vllm_shim.values.service_address import ServiceAddress


def test_command_starts_with_trtllm_serve() -> None:
    cmd = TRTLLMLauncher().build_command("org/m", ServiceAddress("0.0.0.0", 8001), [])
    assert cmd[0] == "trtllm-serve"


def test_model_is_positional_second_token() -> None:
    cmd = TRTLLMLauncher().build_command("org/m", ServiceAddress("0.0.0.0", 8001), [])
    assert cmd[1] == "org/m"


def test_command_includes_host_and_port() -> None:
    cmd = TRTLLMLauncher().build_command("m", ServiceAddress("h", 1), [])
    assert "--host" in cmd and "h" in cmd
    assert "--port" in cmd and "1" in cmd


def test_default_backend_is_pytorch(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.delenv("TRTLLM_BACKEND", raising=False)
    cmd = TRTLLMLauncher().build_command("m", ServiceAddress("h", 1), [])
    idx = cmd.index("--backend")
    assert cmd[idx + 1] == "pytorch"


def test_backend_env_override(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("TRTLLM_BACKEND", "tensorrt")
    cmd = TRTLLMLauncher().build_command("m", ServiceAddress("h", 1), [])
    idx = cmd.index("--backend")
    assert cmd[idx + 1] == "tensorrt"


def test_default_tool_parser_qwen3_coder(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.delenv("TRTLLM_TOOL_PARSER", raising=False)
    cmd = TRTLLMLauncher().build_command("m", ServiceAddress("h", 1), [])
    idx = cmd.index("--tool_parser")
    assert cmd[idx + 1] == "qwen3_coder"


def test_tool_parser_env_override(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("TRTLLM_TOOL_PARSER", "mistral")
    cmd = TRTLLMLauncher().build_command("m", ServiceAddress("h", 1), [])
    idx = cmd.index("--tool_parser")
    assert cmd[idx + 1] == "mistral"


def test_reasoning_parser_absent_by_default(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.delenv("TRTLLM_REASONING_PARSER", raising=False)
    cmd = TRTLLMLauncher().build_command("m", ServiceAddress("h", 1), [])
    assert "--reasoning_parser" not in cmd


def test_reasoning_parser_env_appends_flag(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("TRTLLM_REASONING_PARSER", "nano-v3")
    cmd = TRTLLMLauncher().build_command("m", ServiceAddress("h", 1), [])
    idx = cmd.index("--reasoning_parser")
    assert cmd[idx + 1] == "nano-v3"


def test_extra_args_appended_at_end() -> None:
    cmd = TRTLLMLauncher().build_command("m", ServiceAddress("h", 1), ["--tp_size", "8"])
    idx = cmd.index("--tp_size")
    assert cmd[idx + 1] == "8"
    # Extra args come after the launcher's own args.
    assert idx > cmd.index("--tool_parser")
