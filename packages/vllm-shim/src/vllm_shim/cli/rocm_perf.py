"""Opinionated ROCm performance env-var defaults, gated on the probed GPU.

The SGLang-ROCm base image (`lmsysorg/sglang-rocm:...`) already pins the
high-impact MI300X knobs via its own `ENV` block (HIP_FORCE_DEV_KERNARG,
HSA_NO_SCRATCH_RECLAIM, SGLANG_USE_AITER, NCCL_MIN_NCHANNELS=112, the
two TORCHINDUCTOR_MAX_AUTOTUNE flags, ...). The shim's contribution is
two narrower categories the base image doesn't cover:

1. **Persistence vars** that complement the AITER restore path: route
   MIOpen's kernel-finder DB under ``$VLLM_SHIM_HOME`` so kernel
   selection survives pod restarts on the same PV that holds tuned
   AITER configs. Same survival semantics, same volume.

2. **Recent AMD-recommended always-on vars** that landed in the MI300X
   workload optimization guide after the base image was cut
   (``GPU_MAX_HW_QUEUES=2``, ``TORCH_NCCL_HIGH_PRIORITY=1``). Both are
   gated on ``gfx942`` because the values are MI300-class specific;
   future SKUs (MI4xx, gfx9xx successors) will likely want different
   numbers and we don't want to drag old recommendations forward.

All defaults are merged into the backend's environment via
``setdefault`` semantics in the entrypoint, so an explicit operator
setting in the pod spec or container ``ENV`` always wins.

Skipped on purpose, with rationale (operators can still set these by
hand if their workload benefits):

- ``PYTORCH_TUNABLEOP_ENABLED=1`` -- AMD lists it as always-on, but
  it's a two-phase tune-then-run that adds significant first-request
  latency on a live serving path. We already have a separate tuning
  loop via ``vllm-shim-tune`` for the AITER side.
- ``MIOPEN_FIND_*``, ``PYTORCH_MIOPEN_SUGGEST_NHWC`` -- conv-heavy
  workload knobs that don't move the needle for LLM serving.
- ``VLLM_USE_TRITON_FLASH_ATTN=0`` -- vLLM-side env name; SGLang
  selects its attention impl through its own flags, no equivalent
  passthrough.

References:
- AMD MI300X workload optimization guide (latest):
  https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html
- vLLM serving on AMD MI300X blog:
  https://vllm.ai/blog/vllm-serving-amd
- SGLang CI MIOpen cache configuration:
  repos/sglang/scripts/ci/amd/amd_ci_start_container.sh
"""

from pathlib import Path

from vllm_shim.cli.rocm_probe import GpuAgent


def rocm_perf_defaults(
    gpu: GpuAgent | None, shim_home: Path | None
) -> dict[str, str]:
    """Return the env-var defaults to merge into the backend environment.

    Empty dict when there's no ROCm GPU (CUDA hosts, dev boxes) or no
    resolvable shim home (MIOpen paths would have nowhere to live).
    Pure function; caller is responsible for the ``setdefault`` merge.
    """
    if gpu is None or shim_home is None:
        return {}

    miopen_cache = str(shim_home / "miopen")
    defaults: dict[str, str] = {
        # Faster HF snapshot_download (the shim's resolve_model path
        # uses huggingface_hub; this flips it onto the Rust transfer
        # client when the wheel is installed).
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        # Direct-to-GPU weight loads via safetensors' fast path.
        "SAFETENSORS_FAST_GPU": "1",
        # Persist MIOpen kernel-finder DB on the PV. Without this,
        # MIOpen re-runs its find phase on every pod restart and the
        # first batch eats the latency cost. SGLang's own CI scripts
        # set the same two vars for the same reason.
        "MIOPEN_USER_DB_PATH": miopen_cache,
        "MIOPEN_CUSTOM_CACHE_DIR": miopen_cache,
        # PyTorch matmul/BLAS path: prefer hipBLASLt over rocBLAS.
        # PyTorch falls back to rocBLAS automatically when hipBLASLt
        # has no kernel for a shape, so worst case is a no-op.
        "TORCH_BLAS_PREFER_HIPBLASLT": "1",
    }

    # MI300-class (gfx942) specific. The values here are tied to the
    # CDNA3 architecture's HW-queue layout and RCCL channel topology;
    # other gfx targets need their own audit before reusing them.
    if gpu.gfx_target == "gfx942":
        defaults["GPU_MAX_HW_QUEUES"] = "2"
        defaults["TORCH_NCCL_HIGH_PRIORITY"] = "1"

    return defaults
