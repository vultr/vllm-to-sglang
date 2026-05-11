"""Tests for entrypoint helpers (orchestration is exercised in integration)."""

from vllm_shim.cli.entrypoint import _pin_served_model_name


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
