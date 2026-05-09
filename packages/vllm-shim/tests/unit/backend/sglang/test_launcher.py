import sys

from vllm_shim.backend.sglang.launcher import SGLangLauncher
from vllm_shim.values.service_address import ServiceAddress


def test_command_starts_with_python_module() -> None:
    cmd = SGLangLauncher().build_command("org/m", ServiceAddress("0.0.0.0", 8001), [])
    assert cmd[:3] == [sys.executable, "-m", "sglang.launch_server"]


def test_command_includes_model_path_host_port() -> None:
    cmd = SGLangLauncher().build_command("org/m", ServiceAddress("0.0.0.0", 8001), [])
    assert "--model-path" in cmd and "org/m" in cmd
    assert "--host" in cmd and "0.0.0.0" in cmd
    assert "--port" in cmd and "8001" in cmd


def test_command_includes_enable_metrics() -> None:
    cmd = SGLangLauncher().build_command("m", ServiceAddress("h", 1), [])
    assert "--enable-metrics" in cmd


def test_command_appends_extra_args() -> None:
    cmd = SGLangLauncher().build_command("m", ServiceAddress("h", 1), ["--tp", "8"])
    idx = cmd.index("--tp")
    assert cmd[idx + 1] == "8"


def test_default_tool_call_parser_qwen3_coder(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.delenv("SGLANG_TOOL_CALL_PARSER", raising=False)
    cmd = SGLangLauncher().build_command("m", ServiceAddress("h", 1), [])
    idx = cmd.index("--tool-call-parser")
    assert cmd[idx + 1] == "qwen3_coder"


def test_tool_call_parser_env_override(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("SGLANG_TOOL_CALL_PARSER", "mistral")
    cmd = SGLangLauncher().build_command("m", ServiceAddress("h", 1), [])
    idx = cmd.index("--tool-call-parser")
    assert cmd[idx + 1] == "mistral"
