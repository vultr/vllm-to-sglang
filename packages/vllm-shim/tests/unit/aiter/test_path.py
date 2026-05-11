"""Tests for the AITER shape-capture path layout."""

from pathlib import Path

from vllm_shim.aiter.path import sanitize_model, shape_capture_root
from vllm_shim.values.parallelism import Parallelism

# ---------- sanitize_model ----------


def test_repo_id_collapses_slashes_to_double_dash() -> None:
    # Matches HF's cache directory convention (sans the ``models--`` prefix);
    # makes shape paths visually parsable against the HF cache layout.
    assert sanitize_model("moonshotai/Kimi-K2.6") == "moonshotai--Kimi-K2.6"


def test_repo_id_with_no_slash_is_unchanged() -> None:
    assert sanitize_model("gpt2") == "gpt2"


def test_repo_id_strips_leading_or_trailing_slashes() -> None:
    # Defensive: HF tooling would normalise these, but argv comes from
    # humans and operators sometimes drop a trailing slash on the CLI.
    assert sanitize_model("moonshotai/Kimi-K2.6/") == "moonshotai--Kimi-K2.6"


def test_absolute_path_uses_basename() -> None:
    # Production Stack mounts models at well-known paths; the full prefix
    # carries no information about which model it is.
    assert sanitize_model("/data/models/Kimi-K2.6") == "Kimi-K2.6"


def test_absolute_path_with_trailing_slash_uses_basename() -> None:
    assert sanitize_model("/data/models/Kimi-K2.6/") == "Kimi-K2.6"


def test_relative_path_uses_basename() -> None:
    assert sanitize_model("./local-model") == "local-model"
    assert sanitize_model("../local-model") == "local-model"


def test_fallback_when_basename_would_be_empty() -> None:
    # ``/`` is the only realistic input that survives the absolute-path
    # branch with an empty basename; don't crash, but emit something a
    # human can recognise as "we didn't have anything to work with".
    assert sanitize_model("/") == "unknown-model"


# ---------- shape_capture_root ----------


def test_shape_capture_root_full_layout(tmp_path: Path) -> None:
    root = shape_capture_root(
        shim_home=tmp_path,
        bucket="gfx942-304cu",
        model="moonshotai/Kimi-K2.6",
        parallelism=Parallelism(tp=8, ep=8),
    )
    assert root == (
        tmp_path
        / "aiter"
        / "shapes"
        / "gfx942-304cu"
        / "moonshotai--Kimi-K2.6"
        / "tp8-ep8"
    )


def test_shape_capture_root_single_gpu_topology(tmp_path: Path) -> None:
    # tp=1/ep=1/pp=1 must still produce a well-formed leaf; the segment
    # falls back to ``tp1`` so the layout has no empty components.
    root = shape_capture_root(
        shim_home=tmp_path,
        bucket="gfx942-304cu",
        model="gpt2",
        parallelism=Parallelism(),
    )
    assert root.name == "tp1"
    assert "gpt2" in root.parts


def test_shape_capture_root_lives_under_shim_home(tmp_path: Path) -> None:
    # The whole point of putting this under VLLM_SHIM_HOME is so the
    # captured shapes share a persistent volume and survive pod
    # restarts. Verify the leaf is genuinely a descendant.
    root = shape_capture_root(
        shim_home=tmp_path,
        bucket="gfx942-304cu",
        model="gpt2",
        parallelism=Parallelism(tp=4, pp=2),
    )
    assert root.is_relative_to(tmp_path)


def test_shape_capture_root_local_path_model(tmp_path: Path) -> None:
    # A pod served from a baked-in local model path still gets a stable
    # bucket key (the basename) rather than the full prefix path.
    root = shape_capture_root(
        shim_home=tmp_path,
        bucket="gfx942-304cu",
        model="/data/models/my-model",
        parallelism=Parallelism(tp=8),
    )
    assert "my-model" in root.parts
    assert "data" not in root.parts
