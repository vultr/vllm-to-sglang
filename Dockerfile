FROM lmsysorg/sglang-rocm:v0.5.10rc0-rocm700-mi30x-20260411

# ---------------------------------------------------------------
# Fix aiter — try pip upgrade first, fall back to source build
# ---------------------------------------------------------------
RUN pip install --upgrade aiter || \
    ( \
      echo "=== pip upgrade failed, building aiter from source ===" && \
      pip uninstall -y aiter || true && \
      pip install psutil pybind11 flydsl==0.0.1.dev95158637 && \
      git clone --recursive https://github.com/ROCm/aiter.git /tmp/aiter && \
      cd /tmp/aiter && \
      git checkout v0.1.11.post1 && \
      git submodule sync && \
      git submodule update --init --recursive && \
      PREBUILD_KERNELS=1 GPU_ARCHS=gfx942 python3 setup.py install && \
      cd / && rm -rf /tmp/aiter \
    )

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
RUN touch /opt/vllm-shim/vllm/__init__.py \
          /opt/vllm-shim/vllm/entrypoints/__init__.py \
          /opt/vllm-shim/vllm/entrypoints/openai/__init__.py \
          /opt/vllm-shim/vllm/entrypoints/cli/__init__.py

ENV PYTHONPATH="/opt/vllm-shim:${PYTHONPATH}"

# ---------------------------------------------------------------
# MI300X tuning
# ---------------------------------------------------------------
ENV HIP_FORCE_DEV_KERNARG=1
ENV NCCL_MIN_NCHANNELS=112
ENV GPU_MAX_HW_QUEUES=2
ENV SGLANG_USE_AITER=1