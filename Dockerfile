FROM lmsysorg/sglang-rocm:v0.5.10rc0-rocm700-mi30x-20260411

# Replace the vllm binary with our shim so no matter how the
# production stack invokes vllm, we intercept it
COPY vllm-shim.sh /usr/local/bin/vllm
RUN chmod +x /usr/local/bin/vllm

# Also handle `python -m vllm.entrypoints.openai.api_server` and
# `python -m vllm.entrypoints.cli.main` by shadowing the vllm package
RUN mkdir -p /opt/vllm-shim/vllm/entrypoints/openai \
             /opt/vllm-shim/vllm/entrypoints/cli
COPY vllm_shim_module.py /opt/vllm-shim/vllm/__main__.py
COPY vllm_shim_module.py /opt/vllm-shim/vllm/entrypoints/openai/api_server.py
COPY vllm_shim_module.py /opt/vllm-shim/vllm/entrypoints/cli/main.py
RUN touch /opt/vllm-shim/vllm/__init__.py \
          /opt/vllm-shim/vllm/entrypoints/__init__.py \
          /opt/vllm-shim/vllm/entrypoints/openai/__init__.py \
          /opt/vllm-shim/vllm/entrypoints/cli/__init__.py

# Prepend shim to PYTHONPATH so it shadows any real vllm install
ENV PYTHONPATH="/opt/vllm-shim:${PYTHONPATH}"

ENV HIP_FORCE_DEV_KERNARG=1
ENV NCCL_MIN_NCHANNELS=112
ENV GPU_MAX_HW_QUEUES=2