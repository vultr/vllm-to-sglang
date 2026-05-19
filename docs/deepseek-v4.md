# DeepSeek-V4-Flash on ROCm gfx94*

DSV4-Flash is a 284B-parameter MoE (13B activated, mixed FP4/FP8 quantization, sparse top-K attention, hierarchical CSA + HCA compression) authored against NVIDIA Hopper/Blackwell with 32-lane warps. SGLang's `dsv4` attention backend ships custom JIT C++ kernels (`python/sglang/jit_kernel/csrc/deepseek_v4/*.cuh`) that bake `device::kWarpThreads = 32u` into block-size, tile-size, reduction, and shared-memory layout constants. On AMD CDNA3 (MI300X, gfx942) the wavefront is 64 lanes, so every one of these kernels misbehaves in a kernel-specific way: HSA faults from intra-wavefront divergence on `__syncwarp`, OOB reads when `lane_id` exceeds the array the kernel was sized against, or silently-wrong outputs when warp-cooperative loads/reductions span the wrong subset of lanes.

This page documents the kernel-by-kernel port to make the model run on gfx94*. It is the live status of `patches/sglang/rocm/` and the routing in `repos/sglang/python/sglang/jit_kernel/deepseek_v4.py`. The companion patches for MoE (Quark MXFP4 fallback), distributed init (PyNCCL `_ATTN_TP`), and weight loading on gfx94* are described separately in `docs/patches.md`; this document focuses on the attention/compression path.

## How the kernels go wrong on 64-lane wavefronts

`sgl_kernel/utils.cuh` declares `inline constexpr auto kWarpThreads = 32u;` unconditionally. The dsv4 kernels use this constant to derive:

```cpp
constexpr uint32_t kBlockSize = 128;
constexpr uint32_t kNumWarps  = kBlockSize / device::kWarpThreads;   // 4 (CUDA) | 2 (HIP)
constexpr int64_t  kTileDim   = kTileElements * kWarpThreads;        // 128 | 256
constexpr int64_t  kRopeSize  = kRopeDim / kVecSize;
static_assert(kRopeSize <= kWarpThreads);  // tight on CUDA, slack on HIP
```

When the launch dispatches `kBlockSize = 128` threads, CUDA gets 4 hardware warps and HIP gets 2 hardware wavefronts. The kernel's logical `warp_id = threadIdx.x / 32` still ranges 0..3, but on HIP logical warps 0/1 share wavefront 0 and warps 2/3 share wavefront 1. This produces three distinct failure modes:

1. **Divergence-then-`__syncwarp` deadlock.** When `num_works % 4 == 3`, logical warp 3 early-returns (`if (work_id >= num_works) return;`) while warp 2 continues. Since both are in the same wavefront, the surviving 32 lanes hit `__syncwarp()` later in the kernel and wait for the absent 32 lanes; the HSA exception manager aborts the queue with `HSA_STATUS_ERROR_EXCEPTION: An HSAIL operation resulted in a hardware exception. code: 0x1016`. This is the crash that ended every prompt longer than ~7 words before the first fix landed.
2. **Out-of-bounds reads on the dead lanes.** A kernel like `rope.cuh` reads `freq_ptr[lane_id]` with no bounds check, relying on `static_assert(kRopeSize <= kWarpThreads)` to keep `lane_id < kRopeSize`. On HIP `lane_id` ranges 0..63 against a 32-entry table; lanes 32..63 issue OOB loads and store garbage back into the trailing half of the input slice. No fault, just systematically wrong values written every layer.
3. **Wavefront-wide reductions polluting per-warp sums.** `warp::reduce_sum` lowers to a `__shfl_xor_sync` ladder. On HIP that ladder reduces across all 64 active lanes of the wavefront, but the kernel logic assumed each 32-lane logical warp owned a distinct row of work. Result: every row's reduction includes data from the next row, so RMSNorm denominators and softmax max-values are wrong.

A global swap of `kWarpThreads` to 64 does not fix this because many derived constants are hand-tuned for 32 (e.g. `kRopeVecSize = kRopeDim / (kWarpThreads * 2)` would become 0 and a `static_assert(kRopeVecSize == 1)` would fail). Each kernel needs targeted attention.

## Diagnostic methodology: per-stage `torch.cuda.synchronize` bisect

The HSA fault surfaces in Python at whatever line happens to be the next host-device sync (typically `seq_lens[i].item()` deep in a torch fallback) and is therefore unattributable without help. `HIP_LAUNCH_BLOCKING=1` would attribute it, but disabling async launch is forbidden under the workspace policy (see `feedback_aiter_tuning_minor` in user memory for why we never run with it).

Instead the patches add an env-gated `torch.cuda.synchronize()` between each suspected kernel:

```python
def _sync(stage: str) -> None:
    if os.environ.get("VLLM_SHIM_INDEXER_BISECT") == "1":
        try:
            torch.cuda.synchronize()
        except Exception as e:
            sys.stderr.write(f"INDEXER_BISECT layer={lid} stage={stage} EXC={e!r}\n")
            raise

_sync("entry")
weights = c4_indexer.compute_weights(x, skip_scale=True)
_sync("after_compute_weights")
q_fp8, weights = c4_indexer.compute_q(q_lora, positions, weights)
_sync("after_compute_q")
self.forward_indexer_compressor(...)
_sync("after_indexer_compressor")
```

When a kernel HSA-faults, the `synchronize()` immediately after it aborts the process; the python traceback then points at the `_sync(stage_name)` line. SIGABRT can't be caught by `try/except`, but the `Fatal Python error: Aborted` traceback still pinpoints the stage. Setting `VLLM_SHIM_INDEXER_BISECT=1` enables the syncs; unset, they are no-ops and the production hot path is unaffected.

The instrumentation lives in `indexer.py` (`_forward_prepare_normal`) and `compressor.py` (`forward_indexer_compressor`, `forward_compress`, `Compressor.forward`). Patches `0008..0010` add them.

## Kernel-by-kernel fixes

### `fused_norm_rope.cuh` — HSA fault on `num_compress_tokens = 3`

The compressor's fused RMSNorm + RoPE kernel launches `kBlockSize = 128` threads per block with each logical warp handling one work item. The `if (work_id >= num_works) return;` at the top of the kernel diverges within wavefront 1 when `num_works = 3` (warp 2 continues, warp 3 returns). The subsequent `__syncwarp()` after the RMSNorm reduction (line 148 of `fused_norm_rope.cuh`) HSA-aborts.

Threshold confirmed by probe sweep:

| prompt | input tokens | num_compress_tokens | result |
|---|---|---|---|
| "Once upon a time" | 4 | 1 | OK (decode runs) |
| "Once upon a time, in a" | 7 | 2 | OK |
| "Once upon a time, in a faraway" | 10 | 3 | **HSA fault** |

Fix: `_compress_fused_norm_rope_triton` in `jit_kernel/deepseek_v4.py` (patch `0012`). One triton program per work item, so the divergence problem cannot arise. Uses `tl.sum` for the RMSNorm reduction (correct across both wavefronts of a 128-thread block) and lane-mask gather to extract the rope-dim slice without warp shuffles. Selected via `_COMPRESS_FUSED_NORM_ROPE_TORCH = torch.version.hip is not None`. An earlier pure-torch fallback (patch `0011`) preceded the triton version; it was correct but ~60× slower than the kernel and the model could not sustain itself end-to-end.

### `c4.cuh` — silent wrong outputs in compress_forward

`flash_c4_prefill` and `flash_c4_decode` use warp-cooperative `tile::Memory<>::warp()` loads sized against `kTileElements * kWarpThreads`. On HIP this still uses `kWarpThreads = 32` arithmetically, so half the wavefront's lanes read indices outside the intended head_dim slice. There is no `__syncwarp` to deadlock, so the kernel runs to completion but the compressed-KV outputs are wrong. The c4 compressor feeds the c4 indexer cache, so this corrupts the indexer scores used to pick sparse-attention top-K pages, which corrupts every downstream attention step.

Fix: `_compress_forward_c4_torch` in `jit_kernel/deepseek_v4.py` (patch `0013`). Replays the kernel logic in pure torch: writes the input rows into `kv_score_buffer[page, slot]` per the write plan, then for each compress-plan entry gathers 8 history slots (4 overlap from the previous page, 4 normal from the current page in Page4Align mode; ring positions in RingBuffer mode), softmaxes `(score + ape_bias)` along the history axis, and reduces `kv * softmax(score+ape)` element-wise into the output. Handles both prefill and decode dispatches. The fallback uses small per-token Python loops (one per plan entry) which is acceptable because the work-item count is small (≤ ceil(extend_len/4)).

### `rope.cuh` — OOB reads from lanes 32..63

`deepseek_rope_kernel` does `const auto [f_real, f_imag] = freq_ptr[lane_id]` with `freq_ptr` sized for `kRopeDim / kVecSize = 32` entries. On HIP, lanes 32..63 of the wavefront load past the end of the freq table and write garbage back to the trailing 32 elements of every rope-dim slice. Applied in the inverse-RoPE step at the end of every attention layer.

Fix: `_fused_rope_torch` in `jit_kernel/deepseek_v4.py` (patch `0014`). Torch complex-multiply on `freqs_cis[positions]` against the trailing rope_dim slice, in-place writeback. Selected via `_FUSED_ROPE_TORCH = torch.version.hip is not None`.

### Main attention `main_norm_rope.cuh` — three kernels routed to torch

`FusedQNormRopeKernel`, `FusedKNormRopeFlashMLAKernel`, and `FusedQIndexerRopeHadamardQuantKernel` all share the same 32-lane warp assumption. Each has a torch fallback gated on `torch.version.hip is not None`:

- **Q norm-rope** (`_fused_q_norm_rope_torch`): RMSNorm-self over head_dim followed by complex-multiply RoPE on the trailing rope_dim. No weight tensor.
- **K norm-rope FlashMLA write** (`_fused_k_norm_rope_flashmla_torch`): RMSNorm with weight, RoPE on rope_dim, FP8 quant on the nope dim, vectorized scatter into the paged FlashMLA cache (`{nope bytes, rope bytes, UE8M0 scales}` per token, 584 bytes total laid out as 576 value + 8 scale).
- **Q indexer rope hadamard quant** (`_fused_q_indexer_rope_hadamard_quant_torch`): RoPE on rope_dim, 128-point Walsh-Hadamard, per-token amax → FP8 quant, scaled weight tensor.

Patch `0005` switched the Q-norm-rope path. The K and indexer paths use the same routing pattern; they were in place earlier in the session before this document was started.

### `topk_transform_512_pytorch_vectorized` — `torch.gather` OOB on small seq_lens

The torch fallback for the topk transform gathers physical page indices from `page_tables` using the top-K-derived `page_idx`. When `seq_len < page_size` the "needs_sequential" branch overrides `raw_indices` with `0..TOPK-1`, producing `page_idx = raw_indices >> log2(page_size)` values up to 7. For prompts that allocate only 1-2 entries in `page_tables.shape[1]`, the gather along dim 1 with `index >= page_tables.shape[1]` is OOB.

On CUDA `torch.gather` silently returns junk in this case; on ROCm the OOB read can fault. The fix (patch `0007`) clamps `page_idx` to `[0, max_pages - 1]` before gather:

```python
max_pages = page_tables.shape[1]
page_idx_clamped = torch.clamp(page_idx, min=0, max=max_pages - 1)
physical_pages = torch.gather(page_tables, dim=1, index=page_idx_clamped.long())
```

`valid_topk` masks the invalid positions back to -1 later, so the clamped value is never used in real attention.

## FP8 format mismatch: e4m3fn vs e4m3fnuz

NVIDIA and AMD use different FP8 e4m3 variants:

| | exp bias | max | negative zero | NaN encoding |
|---|---|---|---|---|
| `float8_e4m3fn` (NVIDIA) | 7 | 448 | yes | only `0xff` / `0x7f` |
| `float8_e4m3fnuz` (AMD gfx94*) | 8 | 240 | no | only `0x80` |

The same bit pattern decodes to different values under the two formats. **Producers and consumers must agree on the format**, and the dsv4 stack mixes them:

- **Indexer cache** (c4 compressed KV, `DeepSeekV4IndexerPool`): write side `act_quant` triton kernel, read side `fp8_paged_mqa_logits_torch`. Both treat the bytes as `e4m3fnuz` on ROCm. Consistent.
- **FlashMLA cache** (main attention dense KV, `swa_kv_pool`): write side `_fused_k_norm_rope_flashmla_torch` torch fallback, read side triton sparse-MLA decode kernel (cherry-picked from upstream PR #24930). The read side uses `tl.float8e4nv` which is `e4m3fn` — **not** `e4m3fnuz`. Patch `0015` switches the write side to `torch.float8_e4m3fn` with `fp8_max=448` to match. Without this fix, every read from the FlashMLA cache scales magnitudes wrong (a stored byte representing 100 in fn semantics decodes as ~50 in fnuz semantics; the systematic 2x magnitude error compounds across 64 attention layers).

Patch `0006` had earlier set `fp8_max = 240` in two places to fix fnuz saturation in the indexer-cache path. That fix is correct for the indexer cache but wrong for the FlashMLA cache; patch `0015` reverts only the FlashMLA write side to fn semantics. The two caches have different round-trip dtypes by design and must be kept distinct.

## Compose env knobs that route around still-buggy kernels

The runtime config in `/workspace/compose.yaml` sets several `SGLANG_OPT_USE_*` toggles to route the path through fallbacks the kernels we couldn't fix in time:

| env var | value | reason |
|---|---|---|
| `SGLANG_OPT_USE_COMPRESSOR_V2` | `0` | v2 compressor's `fused_norm_rope_v2` HSA-faults on gfx94* with the same root cause; v1 path + this doc's torch/triton fallbacks is the working route. |
| `SGLANG_OPT_USE_TILELANG_INDEXER` | `false` | tilelang sparse-MLA indexer hasn't been validated on gfx942. |
| `SGLANG_FP8_PAGED_MQA_LOGITS_TORCH` | `1` | torch reference for `fp8_paged_mqa_logits` (deep_gemm is CUDA-only). |
| `SGLANG_OPT_USE_TOPK_V2` | `0` | `topk_v2.cuh` includes `<cuda/ptx>` (Hopper TMA), unbuildable on ROCm. |
| `SGLANG_TOPK_TRANSFORM_512_TORCH` | `1` | route the topk transform through the torch fallback patched in `0007`. |
| `SGLANG_HACK_FLASHMLA_BACKEND` | `triton` | use the triton sparse-MLA decode kernels (PR #24930) instead of CUDA FlashMLA. |
| `SGLANG_OPT_FP8_WO_A_GEMM` | `0` | `_setup_fp8_wo_a_scales` unconditionally imports `deep_gemm` on ROCm and crashes; bypass to BF16 wo_a path. |
| `SGLANG_DSV4_FP4_EXPERTS` | `true` | enables the FP4 expert path (Quark MXFP4 + AITER triton MoE, see `0001`). |
| `SGLANG_OPT_USE_FUSED_STORE_CACHE` | `false` | `fused_store_cache` C kernel uses warp primitives unsafe on 64-lane wavefronts; route through `act_quant` + `set_index_k_scale_buffer`. |
| `SGLANG_OPT_USE_OVERLAP_STORE_CACHE` | `false` | same root cause as above. |

These knobs are the "operator-side" half of the port: the patches change kernel routing, the env vars opt out of optional paths that would re-introduce the same bugs. Removing a knob is only safe if the corresponding kernel has been patched to be 64-lane-aware.

## Patch index

All patches sit on `patched/rocm` in `repos/sglang` and are rebuilt into `patches/sglang/rocm/` by `scripts/rebuild-patches.sh`:

| patch | summary |
|---|---|
| `0001` | route MXFP4 MoE through AITER triton kernels on gfx942 (existing). |
| `0002` | disable MLP allreduce fusion under DP attention (existing). |
| `0003` | route `_ATTN_TP` via PyNCCL so DP attention captures cleanly on ROCm (existing). |
| `0004` | DSV4-Flash weight-loader fix on gfx94* (existing). |
| `0005` | route `fused_q_norm_rope` through torch fallback. |
| `0006` | `fp8_max = 240` for `e4m3fnuz` round-trip in the indexer cache. |
| `0007` | clamp `page_idx` upper bound in topk transform torch fallback. |
| `0008` | env-gated per-stage sync in indexer `_forward_prepare_normal`. |
| `0009` | env-gated per-step sync in `forward_indexer_compressor`. |
| `0010` | env-gated finer-grained syncs in `forward_compress`. |
| `0011` | torch fallback for `compress_fused_norm_rope_inplace`. |
| `0012` | triton kernel replacing the slow torch fallback from `0011`. |
| `0013` | torch fallback for `c4.cuh` `compress_forward`. |
| `0014` | torch fallback for `fused_rope` (OOB reads on 64-lane wavefronts). |
| `0015` | write FlashMLA cache as `e4m3fn` to match the `tl.float8e4nv` reader. |

## Verifying the stack

```bash
docker compose -f /workspace/compose.yaml up -d
# wait for haproxy to flip the backend to UP
until curl -fsS http://172.18.0.2:8000/health > /dev/null; do sleep 5; done
curl -s http://172.18.0.2:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-ai/DeepSeek-V4-Flash","prompt":"The Eiffel Tower is located in","max_tokens":8,"temperature":0}'
```

Probe set used during development (mix of short factual, multi-word, code-like, length-edge-case):

```python
prompts = [
    "The Eiffel Tower is located in",            # short factual
    "The capital of France is",                  # short factual
    "Q: What is 17 * 23?\nA:",                   # arithmetic
    "The three primary colors of light are",     # list completion
    "Once upon a time, in a faraway land,",      # historical HSA-fault threshold
    "Hello, world!",                             # short generic
]
```

Run with `temperature=0` and without `ignore_eos` so each prompt stops naturally on EOS; that makes regressions easy to spot.

## Known remaining gaps (post-`0015`)

Confirmed working on at least one probe after `0015`: correct factual retrieval ("Eiffel Tower in" → "the city of Paris"), correct list completion ("primary colors" → "red, green, and blue"), arithmetic prompts return coherent token sequences. Still observed: some prompts produce wrong factual answers ("Capital of France is" → "Bordeaux"-like outputs in some runs) and long ignore-eos generations drift into repetition. Likely remaining suspects:

- `c128.cuh` (compression ratio 128) — same `kWarpThreads = 32` issue, only triggers on sequences >= 128 tokens so it doesn't affect short-prompt probes but corrupts longer context. No fallback yet; the kernel is similar in shape to `c4.cuh` so a torch fallback in the same style is the natural next patch.
- `rmsnorm.cuh`, `silu_and_mul_masked_post_quant.cuh`, `mega_moe_pre_dispatch.cuh` — also use `kWarpThreads = 32`. Whether they're in the active path depends on whether the runtime config routes through them or through the triton/AITER replacements that are wired up elsewhere; verifying call-site coverage is open work.
- `paged_mqa_metadata.cuh` — host-side metadata kernel, less likely to be a correctness culprit but worth checking with the same sync-bisect harness if regressions appear after fixing the larger kernels.

## Performance

After `0012` (triton compress_fused_norm_rope, replacing the per-token torch fallback) decode runs at ~3 tok/s on 8x MI300X for short prompts (single request, no concurrency). The c4 torch fallback in `0013` and the fused_rope torch fallback in `0014` add Python-side work that lower steady-state throughput by a few tens of percent; replacing them with triton kernels is the next perf step. The dominant cost otherwise is the AITER triton MXFP4 MoE (64 layers × per-token kernel launches); switching to a fused multi-layer MoE kernel is out of scope for this port and tracked separately.
