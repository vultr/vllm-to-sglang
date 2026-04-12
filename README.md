# vLLM → SGLang Shim

Drop-in replacement that makes a vLLM production stack (e.g. the [k8s operator](https://github.com/vllm-project/production-stack)) actually run [SGLang](https://github.com/sgl-project/sglang) instead.

## Why?

The vLLM production stack handles model lifecycle, scaling, and routing — but some models work better (or only work) on SGLang. Rather than rewriting your deployment infra, this shim intercepts every vLLM invocation and launches SGLang with equivalent arguments.

## How It Works

Two interception paths:

| What the stack calls | What happens |
|---|---|
| `vllm serve <model> [flags]` | Shell shim (`vllm-shim.sh`) parses args, execs `python -m sglang.launch_server` |
| `python -m vllm.entrypoints.openai.api_server` | Python shim (shadow module on `PYTHONPATH`) does the same |

Both extract `--host` and `--port` from whatever the stack sends and forward them to SGLang. Everything else is currently hardcoded for the target model.

## Current State

**PoC — hardcoded for `mistralai/Devstral-2-123B-Instruct-2512` on 8× MI300X.**

- Model path, `--tp 8`, and `--tool-call-parser mistral` are baked into both shims
- The Dockerfile builds on `lmsysorg/sglang-rocm` and patches a broken `aiter` build from the base image
- MI300X tuning env vars are set (`HIP_FORCE_DEV_KERNARG`, `NCCL_MIN_NCHANNELS`, etc.)

## Building

```bash
docker build -t vllm-to-sglang .
```

Then use this image anywhere the vLLM stack expects its server image.

## Making It Work For Other Models

Right now the model config is hardcoded in three places:

- `vllm-shim.sh` — the `exec python -m sglang.launch_server` line
- `vllm_shim_module.py` — the `os.execvp()` call
- `Dockerfile` — base image and ROCm-specific patches

To adapt for a different model, change `--model-path`, `--tp`, and `--tool-call-parser` in both shim files. A future pass will make this configurable via env vars or args so you don't have to edit source.

## Files

| File | Purpose |
|---|---|
| `Dockerfile` | Builds the image: ROCm SGLang base + aiter fix + shims + MI300X env |
| `vllm-shim.sh` | Shell shim — replaces the `vllm` binary |
| `vllm_shim_module.py` | Python shim — shadows `vllm.*` module imports |
