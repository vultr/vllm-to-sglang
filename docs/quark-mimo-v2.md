# Quark FP8 and MXFP4 recipe for MiMo-V2.5-Pro on MI300X

A reference recipe for producing Quark-quantized MiMo-V2.5-Pro snapshots (FP8 W8A8 per-channel and MXFP4 W4A4) that serve coherently through this image (`atl.vultrcr.com/vllm/vllm-shim:dev-sglang-rocm`) at TP=8 on 8x MI300X, including out to the model's full 1,048,575-token context.

The recipe is MiMo-specific in its scale-layout details (the published FP8 source uses a per-rank fused `qkv_proj` scale layout that no other model in our matrix has), but the lessons about file2file Quark usage, fused-QKV exclude rules, and serving validation transfer to any DeepSeek-style block-FP8 source.

The shim itself does not quantize anything. This doc lives here because the shim is the runtime that has to load the snapshots, and the loader's tolerances pin several recipe constraints (flag values, exclude scope, attention backend choice).

## Why quantize at all on this hardware

MI300X (`gfx942`) has no native FP4 MFMA pipes. Quark MXFP4 weights on SGLang go through Triton kernels that unpack to BF16 before dispatching to the existing BF16 MFMA pipes (see the patched MXFP4 kernels under `patches/sglang/` and `patches/aiter/`).

The trade is **memory for compute**, not compute for compute:

| Stage | FP8 baseline | MXFP4 |
|---|---|---|
| Weight footprint | 1.0 byte/param | ~0.53 byte/param (FP4 + E8M0 scale) |
| Prefill (compute-bound) | BF16 MFMA | BF16 MFMA + unpack overhead. Neutral to slightly slower. |
| Decode (HBM-bandwidth-bound) | BF16 MFMA, full weight bytes | Half the weight bytes; unpack uses idle compute. Faster. |

For MiMo-V2.5-Pro the headline result is that MXFP4 leaves enough HBM at TP=8 to serve the full 1M-token context without falling back to FP8 KV or other tradeoffs documented elsewhere in this directory.

## Source format and the layered bugs

The upstream `XiaomiMiMo/MiMo-V2.5-Pro` HF release is FineGrainedFP8: each `*.weight` is FP8 with a co-located `*.weight_scale_inv` in 128x128 blocks. Quark's file2file path can read this format directly (`_recover_fp8_weights` does Triton dequant on the fly), so the natural workflow is one step from FP8 to MXFP4.

In practice that one-step path failed for MiMo, and the failures were two stacking bugs that masked each other. Both must be understood before the recipe makes sense.

### Bug 1: fused `qkv_proj.weight_scale_inv` uses a per-rank layout, not global

Upstream MiMo stores `model.layers.<L>.self_attn.qkv_proj.weight_scale_inv` with shape `(216, 48)`. The natural-looking interpretation is one padded global block scale: `weight.shape[0] = 213`, padded to 216 with 128x128 blocks. That is wrong. The actual layout is **8 concatenated per-rank chunks of `(27, 48)`**, to be applied per-rank: each rank's weight chunk `[r * 3392 : (r+1) * 3392]` dequants under scale chunk `[r * 27 : (r+1) * 27]`.

Discriminator: compare `scale.shape[0]` against `ceil(weight.shape[0] / BLOCK)`. For MiMo Pro at TP=8 these are 216 vs 213, and 216 = 8 * 27 confirms per-rank. Confirmed against SGLang's `BlockQuantScaleParameter` allocation in `python/sglang/srt/layers/quantization/fp8.py`, which allocates per-rank `(output_size_per_partition + block_n - 1) // block_n` rows sharded along `output_dim=0`.

For every non-fused projection in MiMo (`o_proj`, dense MLP, MoE expert `gate/up/down`) the per-rank chunk size is a clean multiple of 128, so global and per-rank are mathematically equivalent. **Only fused `qkv_proj` requires the special case.** A naive dequant tool that always assumes the global layout inflates Q weights by 5x to 10x rank-dependently, while preserving the global amax. That makes the bug invisible to "compare amaxes" sanity checks and breaks the model end-to-end without a single tensor-level red flag.

The fix lives in `scripts/quant/stage_dequant.py` in the `dequant()` function, which autodetects the layout and walks per-rank when needed.

### Bug 2: Quark per-channel FP8 is too lossy for fused `qkv_proj`

Per-channel FP8 derives one scale per output row as `row.abs().max() / 224`. For Q rows in MiMo, a few outliers and many small values yield an effective step size of `row_max / 256`, and any value below half-step rounds to zero. The Q section's amax is much larger than K's or V's (asymmetric K/V head dims, K=192, V=128, plus Q has its own scaling), so Q rows get the worst step size for the small values that actually dominate the L2 norms.

Net effect: per-channel FP8 truncates most of Q's weight mass, the Q matmul output drops ~8x at layer 0, and the residual stream goes off-distribution from layer 0 onward. The fix is to **keep fused `qkv_proj` in BF16** (exclude from Quark) and rely on the corrected BF16 staging from Bug 1.

Block FP8 (128x128) does not have this problem because each block has homogeneous magnitudes and gets its own fine scale. SGLang's Quark dispatch does not accept `per_group` qscheme for the FP8 path (only `per_tensor` and `per_channel`), so we cannot pick block FP8 inside Quark for this stack. The exclude-list workaround is mandatory, not a stylistic choice.

MoE expert `gate/up/down` projections have more uniform row distributions and tolerate per-channel FP8 fine.

## Working three-stage pipeline

```
Upstream FP8 HF release
        │
        ▼
1. scripts/quant/stage_dequant.py       -> BF16 staging snapshot
   per-rank dequant for fused qkv_proj, global for everything else;
   strips quantization_config from config.json

        │
        ▼
2. scripts/quant/file2file_fp8.py       -> Quark per-channel FP8 snapshot
   or                                      (mlp.experts.N.{gate,up,down}_proj only)
   scripts/quant/file2file_mxfp4.py     -> Quark MXFP4 snapshot

        │
        ▼
3. SGLang serve with TP=8 --attention-backend=triton --context-length=1048575
   under this image, stock Quark dispatch (no bind-mounted patches needed
   for the quantization path itself)
```

The recipes are bundled into an image by `scripts/quant/Dockerfile` (the shim
image with `amd-quark` + ROCm 6.4 `torch` layered on top) and driven by
`scripts/quant/compose.yaml`, one service per stage. See
`scripts/quant/README.md` for the operator-facing commands.

Stage 2 produces snapshots under the local HF cache:

| Recipe | Output snapshot |
|---|---|
| FP8 W8A8 per-channel | `models--XiaomiMiMo--MiMo-V2.5-Pro-QuarkFP8/snapshots/fp8-perchannel-bf16qkv` |
| MXFP4 W4A4 | `models--XiaomiMiMo--MiMo-V2.5-Pro-MXFP4/snapshots/mxfp4-bf16qkv` |

Both stages 1 and 2 are bounded by the largest safetensors shard (~67 GB BF16 mid-pipeline, single-GPU fits easily; the other ranks sit idle). Wall clock is single-digit minutes for stage 1 and roughly 35 minutes for stage 2 on one MI300X.

### Quark API choice: `direct_quantize_checkpoint`, not `quantize_model`

For any FP8-block HF release fed into Quark, use `ModelQuantizer.direct_quantize_checkpoint(pretrained_model_path=..., save_path=..., device="cuda:0")`. It delegates to `quark.torch.quantization.file2file_quantization.quantize_model_per_safetensor`, reads safetensors directly, dispatches on `quant_method == "fp8"` for the source, runs the fake-quant observer per linear weight, and writes per-shard outputs with the same filenames.

The **wrong** API is `ModelQuantizer.quantize_model(model, dataloader)` on a transformers-loaded model. `AutoModelForCausalLM.from_pretrained` on an FP8 release runs `replace_with_fp8_linear`, leaving the model decorated with `transformers.integrations.finegrained_fp8.FP8Linear` modules. Quark's `process_model_transformation` gates replacement on `type(module) in LAYER_TO_QUANT_LAYER_MAP`, an **exact-type check, not `isinstance`**. FP8Linear fails the exact-type check, replacement count stays at 0, every downstream step harmlessly no-ops, and the export writes the FP8 input back out unchanged. We have hit this on MiMo: 79,561 FP8Linear leaves silently skipped, 56-minute pipeline produced 960 GB of unmodified FP8.

### Spec flag requirements

SGLang's Quark dispatch in `python/sglang/srt/layers/quantization/quark/quark.py::QuarkConfig` gates on specific flag combinations. Get them wrong and the layer falls through to `UnquantizedLinearMethod`, which itself fails because the weight is packed. Result: the checkpoint refuses to load with `No quark compatible scheme was found`.

| Recipe | Weight spec | Input spec |
|---|---|---|
| FP8 W8A8 per-channel | `FP8E4M3PerChannelSpec(ch_axis=0, is_dynamic=False)` | `FP8E4M3PerTensorSpec(is_dynamic=True)` |
| MXFP4 W4A4 | `OCP_MXFP4Spec(ch_axis=-1, is_dynamic=False, scale_calculation_mode="even")` | `OCP_MXFP4Spec(ch_axis=-1, is_dynamic=True, scale_calculation_mode="even")` |

The common mistake is to set `is_dynamic=True` on the weight spec under the assumption that dynamic per-batch scales are "more accurate." For MXFP4, dynamic weight scales mean Quark never persists static E8M0 scales at quant time; the checkpoint has no `weight_scale` tensor, SGLang's `_is_mx_fp4` returns False, and the load fails. Use the table above verbatim and run a 1-rank dry load before committing to a long quant run on a different model.

### Exclude list

Both FP8 and MXFP4 recipes use the same exclude list. The list is structured to match AMD's published Kimi MXFP4 recipe with the namespace remapped onto MiMo:

```python
exclude=[
    "lm_head",
    "model.embed_tokens",
    "*.self_attn.*",        # fused qkv_proj AND o_proj stay BF16; covers Bug 2
    "*.mlp.gate",           # MoE router stays BF16
    "*.mlp.gate_proj",      # matches layer-0 dense MLP only (longer suffix
    "*.mlp.up_proj",        # at mlp.experts.N.gate_proj does NOT match)
    "*.mlp.down_proj",
    "*.mtp.*",              # MTP head stays BF16, symmetric with attention/router
    "model.mtp.*",
]
```

Net effect: only `mlp.experts.N.{gate,up,down}_proj` weights become quantized. The dense MLP at layer 0 stays BF16 because the upstream model's layer 0 is a single dense MLP, not an MoE block, and `*.mlp.gate_proj` matches it by suffix. The MoE-layer projections live at `mlp.experts.N.gate_proj` and have a longer suffix, so they are not matched by these globs and do get quantized.

If you reuse this list on a non-MiMo model, verify the suffix-vs-prefix logic against the actual layer names. The patterns are fnmatch globs, not regex, and the relative suffix lengths matter.

## Serving config

The compose used in `/workspace/compose.yaml` reflects the validated MXFP4 path:

```yaml
command:
  - vllm
  - serve
  - /root/.cache/huggingface/hub/models--XiaomiMiMo--MiMo-V2.5-Pro-MXFP4/snapshots/mxfp4-bf16qkv
  - --served-model-name=XiaomiMiMo/MiMo-V2.5-Pro-MXFP4
  - --tp=8
  - --trust-remote-code
  - --attention-backend=triton
  - --context-length=1048576
  - --mem-fraction-static=0.85
  - --cuda-graph-max-bs=10
  - --max-running-requests=10
  - --chunked-prefill-size=65536
  - --kv-cache-dtype=auto
  - --reasoning-parser=mimo
  - --model-loader-extra-config={"enable_multithread_load":true}
  - --skip-server-warmup
  - --watchdog-timeout=3600
```

One of these is non-negotiable for MiMo on this hardware, documented in an adjacent project note:

- `--attention-backend=triton`: `aiter` MHA asserts at runtime for MiMo's asymmetric K/V head dims (K=192, V=128). Must stay on Triton until aiter MHA is patched.

A second flag was historically required and is no longer:

- `--context-length=1048576` (the full 2^20) plus cuda graphs enabled: validated coherent on both FP8 and MXFP4 in the current SGLang + aiter build (2026-05-21). A prior bisect on the 2026-05-19 build had captured graphs producing garbage when `max_context_len >= 2^20`, which we worked around by `--context-length=1048575` and `--disable-cuda-graph`. The current build no longer reproduces the cliff; both flags can be dropped. See the perf section below for the 3.35x decode speedup that comes from re-enabling graphs.

The `--watchdog-timeout=3600` is SGLang's internal watchdog only; it does not affect the HTTP-bridge timeout discussed below.

## Validation evidence

The MXFP4 snapshot at `mxfp4-bf16qkv` was validated on 2026-05-21 against the following coherence battery on the live `vllm-shim` stack:

| Test | Prompt tokens | Expected | Got |
|---|---|---|---|
| Short factual | <30 | "Paris" / "12" / "4" content | Coherent prose for all three |
| 12K-token needle | 12,032 | `BANANA-7491` | `BANANA-7491` |
| 52K-token needle | 52,033 | `1729` | `1729` |
| 1M-token needle | 1,040,038 | `OBSIDIAN-42` | `OBSIDIAN-42` + coherent meta-description of the filler structure |

The 12K and 52K tests run in seconds. The 1M test prefilled over ~6 minutes (5340 tok/s on the first 65K chunk decaying to 1605 tok/s on the last) and then decoded with zero recall errors.

The same battery was run on the per-channel-FP8 recipe earlier in the same session with equivalent results at the smaller context lengths; the 1M test was not re-run for FP8 because the recipe difference (`*.self_attn.*` exclude) was already shared and the 12K/52K validation isolates the recipe correctness.

## Performance: MXFP4 vs upstream block-FP8

Benchmarked on the same 8x MI300X stack on 2026-05-21 using `bench/bench.py` against `/v1/completions` with `stream=true`, `ignore_eos=true`, `temperature=0`, and the now-final serving config (TP=8, `--attention-backend=triton`, cuda graphs on at `--cuda-graph-max-bs=10`, `--context-length=1048576`, `--max-running-requests=10`). Both stacks ran a JIT warmup pass before the measured pass.

**Decode** (concurrency 1 to 8, 512 output tokens):

| scenario | MXFP4 tok/s | FP8 tok/s | MXFP4 / FP8 |
|---|---|---|---|
| single stream (conc=1, 64-in / 512-out) | 31.05 | 62.02 | 0.50x |
| batch 8 per-request | 28.20 | 33.72 | 0.84x |
| batch 8 aggregate | 225.52 | 269.64 | 0.84x |
| 32K context, 256 output | 30.05 | 57.98 | 0.52x |

**Prefill TTFT** (concurrency 1, 32 output tokens, warm kernels):

| context tokens | MXFP4 TTFT | FP8 TTFT | MXFP4 / FP8 |
|---|---|---|---|
| 2K  | 270 ms | 218 ms | 1.24x |
| 8K  | 285 ms | 245 ms | 1.16x |
| 32K | 351 ms | 311 ms | 1.13x |

With cuda graphs on, FP8 wins decode by ~2x at single-stream and ~16% at batch 8, and is also 13-24% faster at prefill. This is the **opposite** of the earlier `--disable-cuda-graph` reading (which showed MXFP4 7% ahead) and the opposite of what the prior memory note "MI300X has no FP4 MFMA, decode is bandwidth-positive" predicted. The reading now is:

- With graphs **off**, both stacks bottleneck on Python+ROCm kernel-launch overhead at ~9 tok/s. The bandwidth advantage of MXFP4's halved weight bytes is the only differentiator, and it shows as a small win.
- With graphs **on**, launch overhead disappears and compute becomes the bottleneck. FP8 dispatches to native FP8 MFMA on gfx942; MXFP4 has to unpack to BF16 before dispatching to BF16 MFMA. The unpack overhead plus the BF16 dispatch is measurably slower than native FP8 at our concurrency.
- At higher concurrency more tokens flow per layer and HBM pressure starts to matter; we ran a sweep up to concurrency 64 (with `--max-running-requests=64` and `--cuda-graph-max-bs=64`) to look for the crossover. None exists in the measured range: FP8 wins at every point and the ratio asymptotes to ~0.69x.

**Concurrency sweep** (64-in / 512-out, warm, aggregate decode tok/s):

| concurrency | MXFP4 agg | FP8 agg | MXFP4 / FP8 |
|---|---|---|---|
|  1 |    31.10 |    61.94 | 0.50x |
|  8 |   224.00 |   284.69 | 0.79x |
| 16 |   377.37 |   505.25 | 0.75x |
| 24 |   508.48 |   730.67 | 0.70x |
| 32 |   646.01 |   952.13 | 0.68x |
| 48 |   834.71 |  1220.07 | 0.68x |
| 64 |  1050.86 |  1518.03 | 0.69x |

MXFP4 closes about half the single-stream gap as concurrency grows (0.50x to 0.69x), then plateaus. Per-request decode drops faster for MXFP4 (31.10 to 16.55 tok/s, 0.53x) than for FP8 (61.94 to 23.90 tok/s, 0.39x) over the sweep, but FP8 starts so far ahead that the aggregate stays ahead at every point. The theoretical crossover where MXFP4's halved weight bandwidth would dominate over native FP8 MFMA may exist at much higher concurrency or much longer per-token attention work, but it is not in the operating envelope of this server config.

Memory is where MXFP4 actually pays off, and the bench cannot show that directly. Weight footprint is ~50% of FP8's, which is what makes the full 2^20-token context fit at TP=8 on MI300X under the same `--mem-fraction-static=0.85` envelope where upstream FP8 cannot. The choice between MXFP4 and FP8 on this hardware is therefore "more KV pool / longer max context" vs "faster per-token decode," not a free win for either side.

Measured at boot under identical flags (`--mem-fraction-static=0.85`, `--kv-cache-dtype=auto` resolving to bf16, TP=8):

| stack | full-attention pool | SWA pool | `max_total_num_tokens` | post-alloc headroom / rank |
|---|---|---|---|---|
| upstream FP8       |   760,391 tokens | 950,489 tokens   | **950,489**   | 27.16 GB |
| Quark MXFP4        | 1,991,700 tokens | 2,489,625 tokens | **2,489,625** | 27.12 GB |

MXFP4 buys +1.54M tokens of pool (2.62x larger) at the same memory budget. The post-allocation headroom is essentially unchanged across the two stacks, which confirms the savings went into KV rather than into wasted slack: each rank's full-attention KV grew from 27.20 GB to 71.23 GB and SWA KV from 5.67 GB to 14.84 GB, accounting for ~53 GB / rank that lines up with the weight-byte delta on disk (962 GB FP8 vs 538 GB MXFP4, ~424 GB / 8 ranks ≈ 53 GB / rank).

### Stacking with fp8 KV

Switching `--kv-cache-dtype=fp8_e4m3` doubles the KV pool on both stacks (each KV cell halves from bf16 to fp8). MiMo ships without per-channel KV scaling factors, so SGLang logs `Using FP8 KV cache but no scaling factors provided. Defaulting to scaling factors of 1.0` once per rank and falls back to scale=1.0. The empirical effect on the same coherence battery:

| stack | KV dtype | `max_total_num_tokens` | short prompts | 12K needle | 52K needle |
|---|---|---|---|---|---|
| FP8   | bf16     |   950,489 | clean         | ✓ | ✓ |
| FP8   | fp8_e4m3 | 1,900,979 (2.00x) | clean         | ✓ | ✓ |
| MXFP4 | bf16     | 2,489,625 | clean         | ✓ | ✓ |
| MXFP4 | fp8_e4m3 | 4,979,250 (2.00x) | "2+2=" loops  | ✓ | ✓ |

For FP8 weights the doubling is clean across this battery. For MXFP4 weights the same flag stacks two lossy quantizations and surfaces a visible regression on at least one short-prompt case (the "2+2=" continuation degenerates into "4, 2+2=4, 2+2=4, 2+2") while the needle tests still pass. The accuracy floor at adversarial long context is not measured for either stack; for production accuracy work, a Quark or SGLang-tooling KV-scale calibration pass is recommended rather than relying on scale=1.0 silently working.

Decode throughput cost of fp8 KV (warm, conc 1 / 8 / 32, aggregate tok/s):

| stack | bf16 KV | fp8 KV | fp8 KV / bf16 KV |
|---|---|---|---|
| MXFP4 c=1  |  30.75 |  29.63 | 0.96x |
| MXFP4 c=8  | 224.00 | 209.35 | 0.93x |
| MXFP4 c=32 | 646.01 | 598.43 | 0.93x |
| FP8 c=1    |  60.87 |  57.29 | 0.94x |
| FP8 c=8    | 284.69 | 267.26 | 0.94x |
| FP8 c=32   | 952.13 | 879.08 | 0.92x |

fp8 KV costs ~5-8% decode throughput in exchange for 2x pool. The slowdown is the fp8-to-bf16 dequant on every KV read; the bandwidth saving from halved KV bytes doesn't quite cover it at our concurrency.

The bench harness is the one at `/workspace/bench/bench.py` in the dev environment (not in this repo); raw per-run JSON is written to its `results/` directory. The `*-warm.json` outputs are the source for the tables above; cold-pass files exist alongside for the curious.

### EAGLE / MTP speculative decoding

MiMoV2 ships an MTP head. Enabling it through SGLang takes three flag changes plus one patch:

1. `--speculative-algorithm=EAGLE` on the MiMoV2 path auto-flips `enable_multi_layer_eagle=True` (in `python/sglang/srt/server_args.py`); the MTP block is loaded as a single-layer EAGLE draft via the `MiMoV2MTP` adapter in `python/sglang/srt/models/mimo_v2_nextn.py`.
2. `--speculative-num-steps=1 --speculative-eagle-topk=1 --speculative-num-draft-tokens=2` is the right starting config; longer chains (`num_steps=3`) crater acceptance because the chained probability collapses.
3. `--speculative-attention-mode=decode` is set on the production compose; the alternative (`prefill`) was not retested after the SWA fix below and may also work.

The patch landed at `patches/sglang/rocm/0004-fix-triton-backend-populate-SWA-buffers-for-EAGLE-dr.patch`: SGLang's Triton attention backend never populated `forward_metadata.window_kv_*` buffers in any of its three draft-extend init paths. Pure-full-attention and pure-SWA models work fine, but **MiMoV2 is hybrid-SWA** and the `MiMoV2MTPLayer` constructs its attention with `sliding_window_size=config.sliding_window_size`. So the moment the EAGLE draft worker forwards through the MTP block, `forward_extend` reads `forward_metadata.window_kv_indices=None`, and the Triton compile fails to resolve `None.type` against a `tl.load(kv_indices + ...)` call. The patch mirrors the `target_verify` SWA setup into all three `is_draft_extend` branches.

To verify the deployed image actually has the patch baked in, grep the running container:

```bash
docker exec <container> grep -c "Hybrid-SWA models" \
  /sgl-workspace/sglang/python/sglang/srt/layers/attention/triton_backend.py
```

Expect `3` (one comment per modified site: non-graph init, cuda-graph capture, cuda-graph replay). If it returns `0`, the image predates this patch and EAGLE-on-MiMo will crash with `AttributeError("'NoneType' object has no attribute 'type'")` during draft-extend CUDA graph capture, regardless of which `--speculative-*` flags are set.

Measured impact on FP8 (cuda graphs ON, ctx=1048576, max-running=64, warm; aggregate decode tok/s):

| concurrency | no MTP | + MTP (n=1) | speedup |
|---|---|---|---|
|  1 |    62.02 |    70.93 | 1.14x |
|  8 |   284.69 |   451.24 | **1.59x** |
| 32 |   952.13 |  1303.02 | **1.37x** |

Spec accept rate is 0.996 with `accept_length=1.99` out of 2, essentially every draft accepted, so the bonus token is reliable. The `num_steps=3, num_draft_tokens=4` configuration we tried first dropped acceptance to 0.15 and actively hurt aggregate throughput at concurrency 32 (725 vs 952 tok/s baseline), confirming that the MTP head is well-trained for 1-step lookahead and not for longer chains.

MTP on the **MXFP4** snapshot does not yet work: the current Quark recipe excludes `*.mtp.*` and `model.mtp.*` from quantization, so the MTP block's internal weights are bf16 on disk. SGLang's draft loader instantiates the MTP module against the global `quark` config, which expects MXFP4-packed Linear params; the shape mismatch trips the loader assertion. The fix is to re-run `file2file_mxfp4.py` with the MTP excludes removed so the MTP block becomes MXFP4 like the rest of the MoE experts; not done yet.

## Operational gotcha: vllm-shim HTTP-bridge timeout on huge prompts

The shim runs `client -> haproxy -> middleware (FastAPI) -> SGLang` with the haproxy and httpx defaults documented in `docs/middleware.md` and `docs/haproxy.md`. On a 1M-token completion request with `stream=false`, the upstream connection sits idle (no bytes returned) for the entire prefill window. We hit a **504 Gateway Time-out** at ~12 minutes on the non-streaming retry, well before SGLang itself was done.

SGLang did not abort immediately; it kept prefilling for several more chunks and only released the slot once chunked-prefill drained. Importantly, the KV cache it built was preserved in SGLang's prefix cache. The follow-up request with `stream=true` matched the cached prefix (`cached_token=1,040,035` in the log), skipped prefill entirely, and went straight to decode.

Two operational rules follow:

1. **For prompts that prefill longer than the HTTP-bridge tolerates, set `stream=true`.** The completion endpoint emits SSE keepalives during prefill, which keeps the connection alive end-to-end. This is the recommended path for any near-context-limit request through the shim.
2. **A 504 is not always a wasted prefill.** If the client times out but the SGLang prefill ran to completion, the prefix cache will let the next compatible request elide prefill entirely. Useful as a fallback if the streaming retry was forgotten.

We did not adjust the haproxy or middleware timeouts to "fix" this; the streaming path is the supported one and adjusting timeouts to accommodate idle non-streaming requests at 1M tokens would weaken the liveness-probe semantics documented in `docs/haproxy.md`.

## Where the supporting code lives

| Concern | Path |
|---|---|
| Per-rank dequant for fused qkv_proj | `scripts/quant/stage_dequant.py` (`dequant()`) |
| FP8 W8A8 per-channel recipe | `scripts/quant/file2file_fp8.py` |
| MXFP4 W4A4 recipe | `scripts/quant/file2file_mxfp4.py` |
| Recipe image and one-shot harness | `scripts/quant/Dockerfile`, `scripts/quant/compose.yaml` |
| SGLang Quark dispatch gates | `python/sglang/srt/layers/quantization/quark/quark.py` (`QuarkConfig._is_mx_fp4`, `_is_fp8_w8a8`) |
| Block-FP8 per-rank scale allocation | `python/sglang/srt/layers/quantization/fp8.py` (`BlockQuantScaleParameter`) |
| MiMo model adapter (TP, MoE wiring, layer probes) | `python/sglang/srt/models/mimo_v2.py` |
| MXFP4 Triton emulation patches | `patches/sglang/`, `patches/aiter/` |

## Cross-references

- `docs/rocm-perf.md` for the ROCm env defaults applied around the backend launch.
- `docs/middleware.md` and `docs/haproxy.md` for the HTTP bridge whose timeouts force `stream=true` at very long contexts.
- `docs/aiter.md` for shape capture and tuned config restore, which is orthogonal to quantization but shares the same `$VLLM_SHIM_HOME` PV.
- Adjacent project notes (in memory rather than in this tree) cover MiMo's CUDA-graph 2^20 cliff, aiter MHA asymmetric K/V block, and FP8 KV without scaling factors.
