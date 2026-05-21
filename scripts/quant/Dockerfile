FROM atl.vultrcr.com/vllm/vllm-shim:dev-sglang-rocm

# Quark recipes run in the shim image's Python 3.12 venv (/opt/shim).
# The shim image's /opt/shim ships only vllm-shim itself, so we install the
# rest of the quantization toolchain here.
#
# Torch is pinned to ROCm 6.4 (NOT 7.2) because amd-quark 0.11.x imports
# `torch.ao.quantization.pt2e` and `torch.onnx._internal.jit_utils`, both
# stripped from PyPI's rocm7.2 torch wheel. ROCm 6.4 torch is binary-compatible
# with the host's ROCm 7.x runtime/driver. Quant outputs are safetensors,
# loaded by the serving image (rocm7.2) independently.
RUN uv pip install --no-cache --python /opt/shim/bin/python \
        --index-url https://download.pytorch.org/whl/rocm6.4 \
        torch
RUN uv pip install --no-cache --python /opt/shim/bin/python \
        amd-quark \
        accelerate \
        transformers \
        datasets \
        pyarrow

# Bake the recipes into the image so the runtime needs no source bind-mount.
COPY scripts/quant/*.py /opt/quark-recipes/

WORKDIR /opt/quark-recipes
ENTRYPOINT ["/opt/shim/bin/python"]
