FROM lmsysorg/sglang-rocm:v0.5.10.post1-rocm700-mi30x-20260427

# ---------------------------------------------------------------
# haproxy: proxies everything to middleware (including /metrics)
# ---------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends haproxy \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------
# Replace the vllm binary with our shim
# ---------------------------------------------------------------
COPY vllm-shim.sh /usr/local/bin/vllm
RUN chmod +x /usr/local/bin/vllm

# Shadow `python -m vllm.*` invocations
RUN mkdir -p /opt/vllm-shim/vllm/entrypoints/openai \
             /opt/vllm-shim/vllm/entrypoints/cli
COPY vllm_shim_module.py /opt/vllm-shim/vllm/__main__.py
COPY vllm_shim_module.py /opt/vllm-shim/vllm/entrypoints/openai/api_server.py
COPY vllm_shim_module.py /opt/vllm-shim/vllm/entrypoints/cli/main.py
COPY vllm_middleware.py /opt/vllm-shim/vllm_middleware.py
RUN touch /opt/vllm-shim/vllm/__init__.py \
          /opt/vllm-shim/vllm/entrypoints/__init__.py \
          /opt/vllm-shim/vllm/entrypoints/openai/__init__.py \
          /opt/vllm-shim/vllm/entrypoints/cli/__init__.py

# ---------------------------------------------------------------
# PYTHONPATH: two fixes in one
#   1. /sgl-workspace/aiter — use the source-built aiter instead
#      of the broken pip version in site-packages
#   2. /opt/vllm-shim — shadow vllm for python -m invocations
# ---------------------------------------------------------------
ENV PYTHONPATH="/sgl-workspace/aiter:/opt/vllm-shim:${PYTHONPATH}"

# ---------------------------------------------------------------
# MI300X tuning
# ---------------------------------------------------------------
ENV HIP_FORCE_DEV_KERNARG=1
#ENV NCCL_MIN_NCHANNELS=112
#ENV GPU_MAX_HW_QUEUES=2
ENV SGLANG_USE_AITER=1

ENV PYTORCH_ROCM_ARCH=gfx942
ENV AITER_ROCM_ARCH=gfx942
ENV GPU_ARCHS=gfx942
#ENV VLLM_ROCM_USE_AITER=1

# --- Upgrade xgrammar to bleeding edge for tool-call constrained decoding ---
# Kimi K2 drops optional tool-call params with older xgrammar; upgrading fixes
# the grammar matcher so it doesn't prematurely terminate optional fields.
#
# IMPORTANT: --no-deps prevents pip from nuking the ROCm torch build and
# other vLLM-pinned dependencies. xgrammar's only runtime deps that matter
# (torch, numpy, etc.) are already in the image. Build from git main for
# nightly; pin to a release (e.g. xgrammar==0.1.33) if preferred.
#RUN pip install --no-cache-dir apache-tvm-ffi && \
#    pip install --no-cache-dir --force-reinstall --no-deps \
#    'xgrammar @ git+https://github.com/mlc-ai/xgrammar.git@main'

# --- Upgrade transformers to latest for newest model support ---
#RUN pip install --no-cache-dir --upgrade transformers