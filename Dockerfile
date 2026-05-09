FROM lmsysorg/sglang-rocm:v0.5.10.post1-rocm720-mi30x-20260505

# ---------------------------------------------------------------
# haproxy: front door to middleware (including /metrics)
# ---------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends haproxy \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------
# Install uv (small static binary)
# ---------------------------------------------------------------
RUN pip install --no-cache-dir uv

# ---------------------------------------------------------------
# Install vllm-shim and vllm-entrypoints from the workspace.
# After install:
#   - `vllm` is a console script in /usr/local/bin replacing the bash shim
#   - `python -m vllm.entrypoints.openai.api_server` resolves to the stub
#     in the vllm-entrypoints wheel via site-packages
# ---------------------------------------------------------------
COPY pyproject.toml uv.lock /src/
COPY packages /src/packages
RUN uv pip install --system --no-cache /src/packages/vllm-shim /src/packages/vllm-entrypoints

# ---------------------------------------------------------------
# PYTHONPATH: only the source-built aiter override remains.
# ---------------------------------------------------------------
ENV PYTHONPATH="/sgl-workspace/aiter:${PYTHONPATH}"

# ---------------------------------------------------------------
# MI300X tuning
# ---------------------------------------------------------------
ENV HIP_FORCE_DEV_KERNARG=1
ENV SGLANG_USE_AITER=1

ENV PYTORCH_ROCM_ARCH=gfx942
ENV AITER_ROCM_ARCH=gfx942
ENV GPU_ARCHS=gfx942
