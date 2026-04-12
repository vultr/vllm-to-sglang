FROM lmsysorg/sglang-rocm:v0.5.10rc0-rocm700-mi30x-20260411

# ---------------------------------------------------------------
# haproxy: routes /metrics stub, proxies everything else to SGLang
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
ENV NCCL_MIN_NCHANNELS=112
ENV GPU_MAX_HW_QUEUES=2
ENV SGLANG_USE_AITER=1