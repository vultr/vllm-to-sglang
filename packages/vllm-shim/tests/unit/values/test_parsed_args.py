"""Tests for the ParsedArgs value object."""

from vllm_shim.values.parsed_args import ParsedArgs


def test_construction() -> None:
    args = ParsedArgs(
        model="org/model",
        host="0.0.0.0",
        port=8000,
        passthrough=("--tensor-parallel-size", "8"),
    )
    assert args.model == "org/model"
    assert args.passthrough == ("--tensor-parallel-size", "8")


def test_equality_and_hash() -> None:
    a = ParsedArgs("m", "h", 1, ("x",))
    b = ParsedArgs("m", "h", 1, ("x",))
    assert a == b
    assert hash(a) == hash(b)
