import pytest
from vllm_shim.cli.parser import ArgParser


def test_strips_serve_subcommand() -> None:
    p = ArgParser().parse(["serve", "org/model"])
    assert p.model == "org/model"


def test_extracts_host_port() -> None:
    p = ArgParser().parse(["serve", "m", "--host", "1.2.3.4", "--port", "9000"])
    assert p.host == "1.2.3.4"
    assert p.port == 9000


def test_default_host_and_port() -> None:
    p = ArgParser().parse(["serve", "m"])
    assert p.host == "0.0.0.0"
    assert p.port == 8000


def test_equals_form_for_host_port() -> None:
    p = ArgParser().parse(["serve", "m", "--host=h", "--port=7000"])
    assert p.host == "h"
    assert p.port == 7000


def test_passthrough_preserves_remaining_argv() -> None:
    p = ArgParser().parse(
        ["serve", "m", "--tensor-parallel-size", "8", "--trust-remote-code"]
    )
    assert p.passthrough == ("--tensor-parallel-size", "8", "--trust-remote-code")


def test_model_via_named_flag() -> None:
    p = ArgParser().parse(["serve", "--model", "org/m"])
    assert p.model == "org/m"


def test_missing_model_raises() -> None:
    with pytest.raises(ValueError, match="No model"):
        ArgParser().parse(["serve", "--host", "h"])
