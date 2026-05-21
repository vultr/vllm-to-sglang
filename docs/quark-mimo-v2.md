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

The fix lives in `/workspace/quant/stage_dequant.py` in the `dequant()` function, which autodetects the layout and walks per-rank when needed.

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
1. /workspace/quant/stage_dequant.py    -> BF16 staging snapshot
   per-rank dequant for fused qkv_proj, global for everything else;
   strips quantization_config from config.json

        │
        ▼
2. /workspace/quant/file2file_fp8.py    -> Quark per-channel FP8 snapshot
   or                                      (mlp.experts.N.{gate,up,down}_proj only)
   /workspace/quant/file2file_bf16.py   -> Quark MXFP4 snapshot
   (script is misnamed; it emits OCP_MXFP4 not bf16)

        │
        ▼
3. SGLang serve with TP=8 --attention-backend=triton --context-length=1048575
   under this image, stock Quark dispatch (no bind-mounted patches needed
   for the quantization path itself)
```

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
  - --context-length=1048575
  - --mem-fraction-static=0.85
  - --disable-cuda-graph
  - --max-running-requests=10
  - --chunked-prefill-size=65536
  - --kv-cache-dtype=auto
  - --reasoning-parser=mimo
  - --model-loader-extra-config={"enable_multithread_load":true}
  - --skip-server-warmup
  - --watchdog-timeout=3600
```

Three of these are non-negotiable for MiMo on this hardware, documented in adjacent project notes:

- `--attention-backend=triton`: `aiter` MHA asserts at runtime for MiMo's asymmetric K/V head dims (K=192, V=128). Must stay on Triton until aiter MHA is patched.
- `--context-length=1048575`, not 1048576: a CUDA-graph code path poisons something at the 20-bit boundary. Use exactly 1048575.
- `--disable-cuda-graph`: required pending the same 2^20 boundary fix, and currently also required because `--cuda-graph-max-bs=10` is set but the graph compile path is otherwise unstable in this build.

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
| Per-rank dequant for fused qkv_proj | `/workspace/quant/stage_dequant.py` (`dequant()`) |
| FP8 W8A8 per-channel recipe | `/workspace/quant/file2file_fp8.py` |
| MXFP4 W4A4 recipe (misnamed file) | `/workspace/quant/file2file_bf16.py` |
| SGLang Quark dispatch gates | `python/sglang/srt/layers/quantization/quark/quark.py` (`QuarkConfig._is_mx_fp4`, `_is_fp8_w8a8`) |
| Block-FP8 per-rank scale allocation | `python/sglang/srt/layers/quantization/fp8.py` (`BlockQuantScaleParameter`) |
| MiMo model adapter (TP, MoE wiring, layer probes) | `python/sglang/srt/models/mimo_v2.py` |
| MXFP4 Triton emulation patches | `patches/sglang/`, `patches/aiter/` |

## Cross-references

- `docs/rocm-perf.md` for the ROCm env defaults applied around the backend launch.
- `docs/middleware.md` and `docs/haproxy.md` for the HTTP bridge whose timeouts force `stream=true` at very long contexts.
- `docs/aiter.md` for shape capture and tuned config restore, which is orthogonal to quantization but shares the same `$VLLM_SHIM_HOME` PV.
- Adjacent project notes (in memory rather than in this tree) cover MiMo's CUDA-graph 2^20 cliff, aiter MHA asymmetric K/V block, and FP8 KV without scaling factors.
