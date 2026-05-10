# vllm-shim

Drop-in replacement for the vLLM serving container that runs [SGLang](https://github.com/sgl-project/sglang) or [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) underneath.

The intended deployment target is the [vLLM Production Stack](https://github.com/vllm-project/production-stack) (the official k8s operator) and similar setups built around vLLM's CLI, `/health`, `/metrics`, and OpenAI-compatible API contract. Production Stack expects to launch `vllm serve <model> [flags]` and probe a vLLM-shaped pod; the shim satisfies that contract while delegating actual inference to SGLang or TensorRT-LLM. Existing manifests, dashboards, and alerts continue to work unchanged.

## What it does

When the production stack runs `vllm serve <model> [flags]`, the shim:

- Translates the vLLM CLI into the backend's CLI (renames, drops, pass-through). See `docs/argument-translation.md`.
- Launches three cooperating processes inside the container: haproxy, a FastAPI middleware, and the backend (SGLang or TRT-LLM).
- Exposes `/health` that reflects the backend's actual readiness, even before it finishes loading weights. See `docs/haproxy.md`.
- Exposes `/metrics` translated from the backend's native Prometheus exposition into vLLM-named series, so existing dashboards work unchanged. See `docs/metrics.md`.
- Rewrites OpenAI request bodies to strip vLLM-only parameters and repair JSON schemas the backend's strict parser would reject. See `docs/middleware.md`.

What it does *not* do: serve the vLLM Python library API. The shim intercepts the *server launch*, not library imports. See `docs/entrypoints.md`.

## Architecture

```
client
  │  :N (the port the caller passed via --port)
  ▼
┌─────────────┐
│  haproxy    │  L7 reverse proxy + health gate
└──────┬──────┘
       │  :N+2 (loopback)
       ▼
┌─────────────┐
│ middleware  │  FastAPI: /metrics, /health, body-rewriting proxy
└──────┬──────┘
       │  :N+1 (loopback)
       ▼
┌─────────────┐
│   backend   │  SGLang or TRT-LLM, plus native /metrics
└─────────────┘
```

Three processes, one container, one supervisor. haproxy is the only public listener; the middleware and the backend are loopback-only. See `docs/architecture.md` for the full picture (port allocation, request lifecycle, shutdown ordering).

## Quickstart

Build an image:

```bash
# CUDA + SGLang
docker build -f docker/sglang/Dockerfile.cuda -t vllm-shim:sglang-cuda .

# ROCm + SGLang
docker build -f docker/sglang/Dockerfile.rocm -t vllm-shim:sglang-rocm .

# CUDA + TensorRT-LLM
docker build -f docker/trtllm/Dockerfile.cuda -t vllm-shim:trtllm-cuda .
```

Run it the way Production Stack would:

```bash
docker run --rm --gpus all -p 8000:8000 vllm-shim:sglang-cuda \
    vllm serve mistralai/Mistral-7B-Instruct-v0.3 \
    --tensor-parallel-size 1 --max-model-len 4096
```

For local dev there are convenience compose files under `docker/<backend>/`. They build the image, request a GPU, and pre-configure the HF cache mount:

```bash
docker compose -f docker/sglang/compose.cuda.yaml up
docker compose -f docker/trtllm/compose.cuda.yaml up
docker compose -f docker/sglang/compose.rocm.yaml up
```

Override the model with the `MODEL` env var (defaults to `Qwen/Qwen2.5-0.5B-Instruct`, small enough to smoke-test on a single consumer GPU).

The `vllm` CLI inside the container is the shim, not real vLLM. Drop-in with no changes to the calling stack.

## Documentation

Topic docs under `docs/`:

- [`architecture.md`](docs/architecture.md): three-process design, port allocation, request lifecycle, shutdown.
- [`argument-translation.md`](docs/argument-translation.md): vLLM-to-SGLang flag translation, the `ARG_MAP`, edge cases.
- [`metrics.md`](docs/metrics.md): Prometheus exposition translation, derived series, caching.
- [`middleware.md`](docs/middleware.md): FastAPI app, request-body filters, streaming, error dumping.
- [`haproxy.md`](docs/haproxy.md): the front-door proxy, the `nbsrv()` health-gate trick, 503 errorfile.
- [`supervisor.md`](docs/supervisor.md): process supervision, signal handling, shutdown ordering.
- [`backends.md`](docs/backends.md): the `Backend` ABC, the registry, how to add a new backend.
- [`entrypoints.md`](docs/entrypoints.md): the `vllm-entrypoints` namespace stub package.
- [`configuration.md`](docs/configuration.md): every environment variable.
- [`build-and-deploy.md`](docs/build-and-deploy.md): per-`(backend, platform)` Dockerfile layout, Jenkins matrix, registry tagging.
- [`development.md`](docs/development.md): uv workspace setup, tests, type checking, lint.

## Repo layout

| Path | Purpose |
|------|---------|
| `packages/vllm-shim/` | The implementation (`vllm_shim`): CLI, supervisor, middleware, backend abstraction, SGLang and TRT-LLM backends. |
| `packages/vllm-entrypoints/` | Stub `vllm/` namespace so `python -m vllm.X` invocations route through the shim. |
| `docker/<backend>/Dockerfile.<platform>` | Per-backend, per-platform image build (e.g. `docker/sglang/Dockerfile.rocm`). |
| `docs/` | Topic documentation. |
| `Jenkinsfile` | CI/CD: builds and pushes to the Vultr container registry. |

## Development

```bash
uv sync
uv run pytest
uv run mypy
uv run ruff check .
```

The repo is a uv workspace with two member packages. See `docs/development.md`.

## Deploy

Local image build:

```bash
docker build -f docker/sglang/Dockerfile.cuda -t vllm-shim:sglang-cuda .
```

Or via Jenkins (default `BACKEND=sglang`, `PLATFORM=rocm`; final tag is `${TAG}-${BACKEND}-${PLATFORM}`):

```bash
curl -X POST "https://jenkins.sweetapi.com/job/vllm-to-sglang/buildWithParameters" \
  -d TAG=nightly \
  -d BACKEND=sglang \
  -d PLATFORM=rocm
```

See `docs/build-and-deploy.md` for the full matrix and the Jenkins pipeline.
