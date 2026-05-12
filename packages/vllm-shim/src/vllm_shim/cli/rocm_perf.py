"""Opinionated ROCm performance env-var defaults, gated on the probed GPU.

The SGLang-ROCm base image (`lmsysorg/sglang-rocm:...`) already pins the
high-impact MI300X knobs via its own `ENV` block (HIP_FORCE_DEV_KERNARG,
HSA_NO_SCRATCH_RECLAIM, SGLANG_USE_AITER, NCCL_MIN_NCHANNELS=112, the
two TORCHINDUCTOR_MAX_AUTOTUNE flags, ...). The shim's contribution is
three narrower categories the base image doesn't cover:

1. **Persistence vars** that complement the AITER restore path: route
   MIOpen's kernel-finder DB, Triton's compiled-kernel cache,
   TorchInductor's compile-artifact cache, SGLang's torch.compile
   cache, and AITER's own JIT build dir under ``$VLLM_SHIM_HOME`` so
   compiled kernels survive pod restarts on the same PV that holds
   tuned AITER configs. Same survival semantics, same volume. Without
   this, every pod restart pays the full Triton/Inductor/AITER
   compile cost on first request even though the resulting binaries
   are byte-for-byte identical to the previous pod's.

2. **PyTorch BLAS dispatch.** ``TORCH_BLAS_PREFER_HIPBLASLT=1`` so
   matmuls land on hipBLASLt by default; PyTorch falls back to
   rocBLAS automatically for shapes hipBLASLt can't serve.

3. **Recent AMD-recommended always-on vars** that landed in the MI300X
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
- ``FLYDSL_RUNTIME_CACHE_DIR`` -- AITER already manages this itself
  in ``aiter/__init__.py``, pointing at its install-bundled AOT
  FlyDSL cache when present. Overriding would *disable* the AOT
  cache and force runtime re-JIT, which is the opposite of what we
  want.
- ``AMD_COMGR_CACHE`` -- HIPRTC compilation cache; default enables
  it. Setting ``=0`` would disable.

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

# Image-baked key that namespaces the AITER JIT .so cache by AITER
# version. The Dockerfile writes ``git rev-parse --short=12 HEAD`` from
# /sgl-workspace/aiter here, so bumping AITER's pinned tag rotates the
# cache namespace and AITER recompiles from the new source on first
# call instead of silently loading a stale .so from the PV.
# Note: local-patch edits to docker/sglang/patches/ do NOT rotate the
# key; if you change a patch without bumping AITER, clear
# $VLLM_SHIM_HOME/aiter/jit/<sha>/ on the PV manually. See
# docs/aiter.md "JIT cache namespacing by AITER commit" for rationale.
_AITER_CACHE_KEY_FILE = Path("/etc/vllm-shim/aiter-cache-key")


def _aiter_cache_key() -> str:
    """Return the image-baked AITER cache key, or 'default' if unset.

    The file is written by the Dockerfile at image-build time. On dev
    boxes, CUDA images, and any environment without the file the cache
    namespace collapses to ``jit/default``, which is fine because those
    paths share the same unpatched AITER bytes anyway.
    """
    try:
        key = _AITER_CACHE_KEY_FILE.read_text().strip()
    except OSError:
        return "default"
    return key or "default"


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
        # JIT compilation caches: route Triton and TorchInductor onto
        # the PV so pod restarts don't re-pay the compile cost on the
        # first request. AITER's Triton kernels go through this path
        # too. Without ``--enable-torch-compile`` these settings stick
        # verbatim; with torch.compile enabled, SGLang's compilation
        # manager overrides them with paths derived from
        # ``SGLANG_CACHE_DIR`` (also set below), which is on the same
        # PV, so either way we land on the persistent volume.
        "TRITON_CACHE_DIR": str(shim_home / "triton"),
        "TORCHINDUCTOR_CACHE_DIR": str(shim_home / "torchinductor"),
        # SGLang's own cache root. Defaults to ``~/.cache/sglang``;
        # we anchor it on the PV so SGLang's torch.compile artifacts
        # and any future SGLang-owned cache content lands there too.
        "SGLANG_CACHE_DIR": str(shim_home / "sglang"),
        # AITER's own JIT build dir. AITER builds HIP kernels on
        # first call via ``aiter/jit/core.py``; ``bd_dir`` derives as
        # ``$AITER_JIT_DIR/build``. Default fallback is
        # ``~/.aiter/jit/build``, which is lost on pod restart.
        # Anchoring under ``$VLLM_SHIM_HOME/aiter/jit/<sha>`` puts
        # the .so cache alongside the existing ``configs/`` and
        # ``shapes/`` subdirs the shim already manages, namespaced
        # by AITER's commit SHA (see ``_aiter_cache_key``). Bumping
        # the AITER pin rotates the namespace and AITER recompiles
        # instead of loading a stale .so from the PV.
        "AITER_JIT_DIR": str(shim_home / "aiter" / "jit" / _aiter_cache_key()),
    }

    # MI300-class (gfx942) specific. The values here are tied to the
    # CDNA3 architecture's HW-queue layout and RCCL channel topology;
    # other gfx targets need their own audit before reusing them.
    if gpu.gfx_target == "gfx942":
        defaults["GPU_MAX_HW_QUEUES"] = "2"
        defaults["TORCH_NCCL_HIGH_PRIORITY"] = "1"

    return defaults
