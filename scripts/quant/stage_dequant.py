"""Pre-dequantize MiMo-V2.5-Pro FP8 source to a bf16 safetensors snapshot.

Stage 1 of the recipe documented in `docs/quark-mimo-v2.md`. Reads the upstream
FineGrainedFP8 release from the HF cache, dequants every `*.weight` against its
`*.weight_scale_inv` companion, drops the scale tensors, and writes a snapshot
with identical layout but plain bf16 weights and a stripped `quantization_config`.

The critical detail is the per-rank scale layout autodetector in `dequant()`.
Fused `qkv_proj.weight_scale_inv` is stored as 8 concatenated per-rank chunks
of (27, 48), not a single padded global block scale of (216, 48). Misreading
the layout inflates Q weights by 5-10x rank-dependently while preserving the
global amax, so the bug is invisible to amax-based sanity checks. See the
docs for the discriminator and the end-to-end failure mode.

Memory: per-shard streaming, peak GPU is bounded by the largest input shard
(<30 GB bf16). Single MI300X is enough; we free the cache between shards.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

SRC_DIR = Path(
    "/root/.cache/huggingface/hub/models--XiaomiMiMo--MiMo-V2.5-Pro/"
    "snapshots/f5953f1b993e90367f3d87fed59d099f2c527763"
)
OUT_ROOT = Path("/root/.cache/huggingface/hub/models--XiaomiMiMo--MiMo-V2.5-Pro-BF16")
SNAPSHOT = "bf16"
OUT_DIR = OUT_ROOT / "snapshots" / SNAPSHOT

BLOCK = 128
TP_SIZE = 8  # MiMo Pro is published as TP=8-interleaved; per-rank scales assume this.
DEQUANT_DEVICE = "cuda:0"

SKIP_COPY_SUFFIXES = (".safetensors",)
SKIP_COPY_NAMES = frozenset({"model.safetensors.index.json", "config.json"})


def dequant(weight_fp8: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """FineGrainedFP8 dequant with per-rank or global block scale, autodetected.

    Discriminator: if `scale.shape[0] != ceil(weight.shape[0] / BLOCK)`, the
    scale is laid out as `TP_SIZE` concatenated per-rank chunks along dim 0.
    That holds for fused `qkv_proj` where the TP-rank weight chunks are not
    multiples of `BLOCK`, so the global-block interpretation would let one
    scale block straddle a rank boundary. All non-fused projections have
    `out_features` divisible by both `TP_SIZE` and `BLOCK`, making per-rank
    and global mathematically equivalent; we return the global path for those.
    """
    w = weight_fp8.to(DEQUANT_DEVICE).to(torch.float32)
    out, inn = w.shape
    blocks_out = (out + BLOCK - 1) // BLOCK
    blocks_in = (inn + BLOCK - 1) // BLOCK
    s = scale.to(DEQUANT_DEVICE).to(torch.float32)

    if s.shape[0] != blocks_out:
        assert out % TP_SIZE == 0, (
            f"per-rank dequant requires out_features divisible by TP_SIZE; "
            f"got {out} % {TP_SIZE} = {out % TP_SIZE}"
        )
        rows_per_rank = out // TP_SIZE
        blocks_per_rank = s.shape[0] // TP_SIZE
        assert s.shape[0] == TP_SIZE * blocks_per_rank, (
            f"scale rows {s.shape[0]} not divisible by TP_SIZE {TP_SIZE}"
        )
        parts = []
        for r in range(TP_SIZE):
            w_chunk = w[r * rows_per_rank : (r + 1) * rows_per_rank]
            s_chunk = s[r * blocks_per_rank : (r + 1) * blocks_per_rank, :blocks_in]
            expanded = s_chunk.repeat_interleave(BLOCK, dim=0)[:rows_per_rank].repeat_interleave(
                BLOCK, dim=1
            )[:, :inn]
            parts.append(w_chunk * expanded)
        return torch.cat(parts, dim=0).to(torch.bfloat16).cpu()

    s = s[:blocks_out, :blocks_in]
    expanded = s.repeat_interleave(BLOCK, dim=0)[:out].repeat_interleave(BLOCK, dim=1)[:, :inn]
    return (w * expanded).to(torch.bfloat16).cpu()


def main() -> None:
    with (SRC_DIR / "model.safetensors.index.json").open() as f:
        idx = json.load(f)
    wm: dict[str, str] = idx["weight_map"]
    shards = sorted(set(wm.values()))
    print(f"source: {SRC_DIR}")
    print(f"output: {OUT_DIR}")
    print(f"shards: {len(shards)}, tensors: {len(wm)}")

    if OUT_DIR.exists():
        print(f"removing existing {OUT_DIR}")
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True)

    new_wm: dict[str, str] = {}

    for shard_i, shard in enumerate(shards):
        src_path = SRC_DIR / shard
        out_path = OUT_DIR / shard

        out_tensors: dict[str, torch.Tensor] = {}
        n_dequant = 0
        n_passthrough = 0
        n_dropped_scale = 0

        with safe_open(src_path, framework="pt") as f:
            keys = list(f.keys())
            local_scales = {k for k in keys if k.endswith(".weight_scale_inv")}
            for k in keys:
                if k.endswith(".weight_scale_inv"):
                    n_dropped_scale += 1
                    continue
                t = f.get_tensor(k)
                sk_full = k.replace(".weight", ".weight_scale_inv")
                if k.endswith(".weight") and sk_full in local_scales:
                    s = f.get_tensor(sk_full)
                    out_tensors[k] = dequant(t, s)
                    n_dequant += 1
                else:
                    out_tensors[k] = t
                    n_passthrough += 1
                new_wm[k] = shard

        save_file(out_tensors, str(out_path))
        size_gb = out_path.stat().st_size / 1e9
        print(
            f"[{shard_i + 1:>2}/{len(shards)}] {shard}: "
            f"dequant={n_dequant} passthrough={n_passthrough} "
            f"dropped={n_dropped_scale} -> {size_gb:.2f} GB"
        )

        torch.cuda.empty_cache()
        del out_tensors

    new_idx = {"metadata": idx.get("metadata", {}), "weight_map": new_wm}
    with (OUT_DIR / "model.safetensors.index.json").open("w") as f:
        json.dump(new_idx, f, indent=2)

    with (SRC_DIR / "config.json").open() as f:
        cfg = json.load(f)
    cfg.pop("quantization_config", None)
    cfg["torch_dtype"] = "bfloat16"
    with (OUT_DIR / "config.json").open("w") as f:
        json.dump(cfg, f, indent=2)

    # Copy tokenizer / modeling / chat-template artefacts. Anything in SRC_DIR
    # that isn't a safetensors shard, the index, or the config we just rewrote
    # gets a shallow copy across so the output is self-contained for HF loaders.
    for entry in SRC_DIR.iterdir():
        if entry.is_dir():
            continue
        if entry.name in SKIP_COPY_NAMES:
            continue
        if entry.name.endswith(SKIP_COPY_SUFFIXES):
            continue
        dst = OUT_DIR / entry.name
        if dst.exists():
            continue
        shutil.copy(entry, dst)

    total = sum(p.stat().st_size for p in OUT_DIR.glob("*.safetensors"))
    print()
    print(f"done: {OUT_DIR}")
    print(f"total safetensors: {total / 1e12:.2f} TB ({total / 1e9:.1f} GB)")


if __name__ == "__main__":
    main()
