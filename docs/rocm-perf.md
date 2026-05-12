# ROCm performance defaults

On ROCm hosts, the entrypoint applies a small opinionated set of `setdefault` env vars before spawning the backend. The defaults augment the SGLang-ROCm base image's own `ENV` block (which already pins the high-impact MI300X knobs like `HIP_FORCE_DEV_KERNARG`, `HSA_NO_SCRATCH_RECLAIM`, `NCCL_MIN_NCHANNELS=112`, `SGLANG_USE_AITER`, the `TORCHINDUCTOR_MAX_AUTOTUNE*` pair). The shim is responsible only for what the base image doesn't already cover.

All defaults follow the same precedence rule used elsewhere in the entrypoint: operator-set values (pod spec, container `ENV`, container CLI) win. The shim never overwrites. The launch-info dump surfaces only the defaults that actually took effect (the ones the operator left blank), so a missing key in `rocm_perf` is the signal that the operator's setting beat ours.

## Generic ROCm defaults

Applied whenever `rocm_probe` returns a GPU and `resolve_shim_home` resolves a path.

| Var | Value | Why |
|---|---|---|
| `HF_HUB_ENABLE_HF_TRANSFER` | `1` | Routes `huggingface_hub.snapshot_download` through the Rust transfer client. The shim's `resolve_model` is the consumer; faster cold-start. |
| `SAFETENSORS_FAST_GPU` | `1` | Direct-to-GPU weight loads via safetensors' fast path. |
| `MIOPEN_USER_DB_PATH` | `$VLLM_SHIM_HOME/miopen` | MIOpen kernel-finder DB. Without persistence, MIOpen re-runs its find phase on every pod restart and the first batch eats the latency cost. SGLang's own CI scripts set the same two vars. |
| `MIOPEN_CUSTOM_CACHE_DIR` | `$VLLM_SHIM_HOME/miopen` | Companion to the above. Same path. |
| `TORCH_BLAS_PREFER_HIPBLASLT` | `1` | PyTorch BLAS dispatch picks hipBLASLt over rocBLAS. PyTorch falls back to rocBLAS automatically when hipBLASLt has no kernel for a shape, so worst case is a no-op. |

## MI300-class additions

Applied when `gpu.gfx_target == "gfx942"` (MI300X / MI300A / MI325X). Values are CDNA3-architecture specific.

| Var | Value | Why |
|---|---|---|
| `GPU_MAX_HW_QUEUES` | `2` | AMD's MI300X workload optimization guide lists this as always-recommended. Limits HIP streams to align compute and RCCL operations. |
| `TORCH_NCCL_HIGH_PRIORITY` | `1` | RCCL stream priority on multi-GPU FSDP / TP. No-op on single-GPU. |

Other gfx targets (gfx90a / MI250X, future gfx95x / MI4xx) need their own audit before reusing these values. The function returns the generic set without these additions when the SKU isn't recognised.

## Deliberately skipped

Operators can set these by hand if their workload benefits; the shim doesn't ship them as defaults.

- `PYTORCH_TUNABLEOP_ENABLED=1` -- AMD lists it as always-on, but it's a two-phase tune-then-run that adds significant first-request latency on a live serving path. The shim already has a separate offline tuning loop via `vllm-shim-tune` for the AITER side.
- `MIOPEN_FIND_ENFORCE`, `MIOPEN_FIND_MODE`, `PYTORCH_MIOPEN_SUGGEST_NHWC` -- conv-heavy workload knobs that don't move the needle for transformer LLM serving.
- `VLLM_USE_TRITON_FLASH_ATTN=0` -- vLLM-side env name; SGLang selects its attention implementation through its own flags. No equivalent passthrough.
- `RCCL_MSCCL_ENABLE`, `NCCL_IB_*` -- topology- and fabric-specific. Operators with multi-node deployments will know their own values.

## Operator surface

The stderr summary prints one line when defaults were applied:

```
  rocm perf: 5 defaults (HF_HUB_ENABLE_HF_TRANSFER, MIOPEN_CUSTOM_CACHE_DIR, MIOPEN_USER_DB_PATH, SAFETENSORS_FAST_GPU, TORCH_BLAS_PREFER_HIPBLASLT)
```

The full env-var-to-value mapping is in the JSON dump under the `rocm_perf` key. Disabled (no GPU, no shim home, or operator set every var explicitly) means no line in the summary.

## References

- AMD MI300X workload optimization guide: <https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html>
- vLLM blog: Serving LLMs on AMD MI300X: <https://vllm.ai/blog/vllm-serving-amd>
- SGLang ROCm Dockerfile (the base image's `ENV` floor): `repos/sglang/docker/rocm.Dockerfile`
- SGLang AMD CI startup script (MIOpen cache layout): `repos/sglang/scripts/ci/amd/amd_ci_start_container.sh`

## Module

`vllm_shim.cli.rocm_perf.rocm_perf_defaults(gpu, shim_home) -> dict[str, str]`. Pure function; the entrypoint owns the `setdefault` merge.
