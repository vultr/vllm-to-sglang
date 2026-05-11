"""Tests for resolve_model (HF repo ID -> local snapshot directory)."""

from pathlib import Path

import pytest
from vllm_shim.cli import model_resolver
from vllm_shim.cli.model_resolver import resolve_model


def test_local_directory_passes_through(tmp_path: Path) -> None:
    assert resolve_model(str(tmp_path)) == str(tmp_path)


def test_repo_id_calls_snapshot_download(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, str | None]] = []

    def fake_snapshot_download(repo_id: str, revision: str | None = None) -> str:
        calls.append((repo_id, revision))
        return f"/cache/{repo_id.replace('/', '--')}/snapshots/abc"

    # Resolver imports huggingface_hub lazily inside the function. Patching
    # the attribute on the already-imported module is what the resolver sees.
    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)

    out = resolve_model("moonshotai/Kimi-K2.6", revision="abc123")

    assert out == "/cache/moonshotai--Kimi-K2.6/snapshots/abc"
    assert calls == [("moonshotai/Kimi-K2.6", "abc123")]


def test_module_does_not_eagerly_import_huggingface_hub() -> None:
    # Belt-and-braces guard against accidentally hoisting the import to module
    # scope; the resolver should stay cheap to import.
    assert "huggingface_hub" not in vars(model_resolver)
