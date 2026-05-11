"""Resolve a vLLM-style model identifier to a local directory path.

Production Stack mounts HF_HOME on a persistent volume, so first launch
downloads into that cache and subsequent launches are a no-op metadata check.
Resolution sidesteps trust_remote_code tokenizers that hardcode
os.path.join(model_arg, "...") and only work when the caller passes a real
directory (e.g. Kimi K2.6's fast tokenizer).
"""

import os


def resolve_model(model: str, revision: str | None = None) -> str:
    """Return a local directory path. `model` passes through unchanged if it
    is already an existing directory; otherwise it is treated as an HF repo
    ID and resolved via snapshot_download, which respects HF_HOME and
    HF_HUB_OFFLINE and is a metadata-only operation when the snapshot is
    already cached."""
    if os.path.isdir(model):
        return model
    from huggingface_hub import snapshot_download

    return snapshot_download(model, revision=revision)
