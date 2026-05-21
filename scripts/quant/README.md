# Quark quantization recipes

Three scripts that reproduce the FP8 W8A8 per-channel and MXFP4 W4A4 snapshots
of MiMo-V2.5-Pro served by this image. The recipes, the rationale, and the
validation evidence live in `docs/quark-mimo-v2.md`; this directory is the
runnable form.

| File | Stage |
|---|---|
| `stage_dequant.py`     | 1. Pre-dequant upstream FineGrainedFP8 to a bf16 staging snapshot, with per-rank handling for fused `qkv_proj`. |
| `file2file_fp8.py`     | 2a. Quark file2file W8A8 FP8 (per-channel) from the bf16 snapshot. |
| `file2file_mxfp4.py`   | 2b. Quark file2file MXFP4 (W4A4) from the bf16 snapshot. |
| `Dockerfile`           | Builds an image with `amd-quark` + `torch` (ROCm 6.4) layered on the shim image; bakes the recipes in at `/opt/quark-recipes/`. |
| `compose.yaml`         | One service per recipe; reads / writes the operator's HF cache. |

Each script is self-contained: source and output paths are top-of-file
constants, no CLI flags. Adapt the constants if you need to point at a
different upstream snapshot or a different output name. The recipes assume
the source `XiaomiMiMo/MiMo-V2.5-Pro` snapshot is already in the operator's HF
cache (gated repo, download separately).

Build the image once from the repo root:

```bash
docker compose -f scripts/quant/compose.yaml build
```

Run the stages in order:

```bash
docker compose -f scripts/quant/compose.yaml run --rm stage-dequant
docker compose -f scripts/quant/compose.yaml run --rm file2file-mxfp4
# or file2file-fp8
```

Stage 1 wall-clock is single-digit minutes; stage 2 is roughly 35 minutes on
one MI300X. The other ranks sit idle during quant; the per-shard streaming
keeps peak GPU memory bounded by the largest input shard.
