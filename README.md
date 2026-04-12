# vLLM → SGLang Shim

Drop-in replacement that makes a vLLM production stack (e.g. the [k8s operator](https://github.com/vllm-project/production-stack)) actually run [SGLang](https://github.com/sgl-project/sglang) instead.

## Why?

The vLLM production stack handles model lifecycle, scaling, and routing — but some models work better (or only work) on SGLang. Rather than rewriting your deployment infra, this shim intercepts every vLLM invocation and launches SGLang with equivalent arguments.

## How It Works

### Invocation interception

Two interception paths catch however the vLLM stack tries to start the server:

| What the stack calls | What happens |
|---|---|
| `vllm serve <model> [flags]` | Shell shim (`vllm-shim.sh`) replaces the `vllm` binary |
| `python -m vllm.entrypoints.openai.api_server` | Python shim (shadow module on `PYTHONPATH`) intercepts the import |

Both extract `--host` and `--port` from whatever the stack sends.

### haproxy proxy layer

Rather than launching SGLang directly on the vLLM port, the shim runs **haproxy** on the original port and **SGLang on port+1**. This solves two critical problems:

1. **`/metrics` stub** — The vLLM stack expects a Prometheus metrics endpoint at `/metrics`. SGLang doesn't serve one. haproxy intercepts `/metrics` and returns an empty 200 response instantly.

2. **`/health` probe timing** — SGLang's `/health` endpoint takes ~1.001s to respond, which races the 1s k8s probe timeout and causes repeated `Startup probe failed: context deadline exceeded`. haproxy health-checks SGLang in the background (every 5s, with a 3s timeout) and responds to `/health` probes **instantly** — 200 if the backend is up, 503 if it's not. No more timeout roulette.

### middleware layer

A Python middleware (FastAPI) sits between haproxy and SGLang on **port+2**. It strips vLLM-only request parameters that SGLang rejects with 422 errors:

- **`logprobs`** / **`top_logprobs`** — vLLM accepts these on chat completion requests; SGLang's Mistral tool-call parser rejects them. OpenClaw and other vLLM clients send them by default.

The middleware only touches `POST /v1/chat/completions` request bodies and passes everything else through unchanged. To strip additional params, add them to the `STRIP_PARAMS` set in `vllm_middleware.py`.

```
┌─────────────────────────────────────────────┐
│  k8s probes / vLLM stack                    │
│         │                                   │
│         ▼                                   │
│  haproxy (port 8000)                        │
│    /metrics ──► 200 empty (stub)            │
│    /health  ──► 200/503 instant (backend    │
│                 health-checked in bg)        │
│    /*       ──► proxy to middleware          │
│                       │                     │
│                       ▼                     │
│  middleware (port 8002)                      │
│    strips logprobs/top_logprobs             │
│    forwards to SGLang                       │
│                       │                     │
│                       ▼                     │
│              SGLang (port 8001)             │
└─────────────────────────────────────────────┘
```

haproxy 2.4 compat: uses `errorfile` + `http-request deny deny_status` for stub responses (the `http-request return` payload syntax requires haproxy 2.8+).

## Current State

**Running in production — `mistralai/Devstral-2-123B-Instruct-2512` on 8× MI300X.**

- Model path, `--tp 8`, and `--tool-call-parser mistral` are baked into both shims
- The Dockerfile builds on `lmsysorg/sglang-rocm` and patches a broken `aiter` build from the base image
- MI300X tuning env vars are set (`HIP_FORCE_DEV_KERNARG`, `NCCL_MIN_NCHANNELS`, etc.)
- All received args are logged to `/tmp/vllm-shim.log` (configurable via `VLLM_SHIM_LOG` env var)

## Building

```bash
docker build -t vllm-to-sglang .
```

Or use the Jenkins pipeline:

```bash
curl -X POST "https://jenkins.sweetapi.com/job/vllm-to-sglang/buildWithParameters" \
  -u "${JENKINS_USER}:${JENKINS_PASS}" \
  -d "BRANCH=metrics" \
  -d "TAG=nightly3"
```

Then use this image anywhere the vLLM stack expects its server image.

## Making It Work For Other Models

Right now the model config is hardcoded in three places:

- `vllm-shim.sh` — the `python -m sglang.launch_server` line
- `vllm_shim_module.py` — the `subprocess.Popen()` call
- `Dockerfile` — base image and ROCm-specific patches

To adapt for a different model, change `--model-path`, `--tp`, and `--tool-call-parser` in both shim files. A future pass will make this configurable via env vars or args so you don't have to edit source.

## Files

| File | Purpose |
|---|---|
| `Dockerfile` | Builds the image: ROCm SGLang base + haproxy + shims + MI300X env |
| `vllm-shim.sh` | Shell shim — replaces the `vllm` binary, launches SGLang + middleware + haproxy |
| `vllm_shim_module.py` | Python shim — shadows `vllm.*` module imports, launches SGLang + middleware + haproxy |
| `vllm_middleware.py` | FastAPI middleware — strips vLLM-only params (logprobs) before forwarding to SGLang |
