# vllm-to-sglang

Drop-in replacement that makes a vLLM production stack (e.g. the [k8s operator](https://github.com/vllm-project/production-stack)) actually run [SGLang](https://github.com/sgl-project/sglang) instead.

## How it works

The k8s vLLM production stack calls `vllm serve <model> [flags]`. This project intercepts that call and instead launches SGLang behind haproxy + a middleware layer.

```
k8s vLLM stack
  │
  │  vllm serve mistralai/Devstral-2-123B-Instruct-2512 \
  │    --host 0.0.0.0 --port 8000 --tensor-parallel-size 8 ...
  │
  ▼
┌─────────────────────────────────────────────────────────┐
│  vllm-shim.sh (replaces the `vllm` binary)             │
│  or vllm_shim_module.py (shadows python -m vllm.*)     │
│                                                         │
│  Parses vLLM args, translates to SGLang equivalents,   │
│  then launches three processes:                         │
│                                                         │
│  ┌──────────────────────────────────────────────────┐  │
│  │ haproxy :8000 (front door)                       │  │
│  │   /health  → 200/503 based on backend state      │  │
│  │   /*       → proxy to middleware :8002            │  │
│  └──────────────────────────────────────────────────┘  │
│                        │                                │
│                        ▼                                │
│  ┌──────────────────────────────────────────────────┐  │
│  │ middleware :8002 (FastAPI)                        │  │
│  │   /metrics → fetches SGLang metrics, translates   │  │
│  │              to vLLM-compatible names             │  │
│  │   Strips vLLM-only params from request bodies    │  │
│  │   Recursively fixes tool JSON schemas            │  │
│  │   Forwards to SGLang :8001                       │  │
│  └──────────────────────────────────────────────────┘  │
│                        │                                │
│                        ▼                                │
│  ┌──────────────────────────────────────────────────┐  │
│  │ SGLang :8001 (inference + /metrics)              │  │
│  └──────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

## Argument translation

The shim dynamically translates vLLM CLI args to SGLang equivalents — no hardcoded model names or tensor-parallel sizes.

### Infrastructure (extracted, not forwarded)

| vLLM flag | Behavior |
|-----------|----------|
| `serve` | Subcommand, dropped |
| `<model>` (positional) | Becomes `--model-path <model>` |
| `--host` | Used for haproxy, middleware, and SGLang |
| `--port` | haproxy binds this; SGLang gets +1, middleware +2 |

### Renames

| vLLM flag | SGLang equivalent |
|-----------|-------------------|
| `--tensor-parallel-size` | `--tp` |
| `--gpu-memory-utilization` | `--mem-fraction-static` |
| `--max-model-len` | `--context-length` |
| `--max-num-seqs` | `--max-running-requests` |
| `--max-num-batched-tokens` | `--chunked-prefill-size` |
| `--seed` | `--random-seed` |
| `--distributed-timeout-seconds` | `--dist-timeout` |
| `--lora-modules` | `--lora-paths` |
| `--enable-multi-modal` | `--enable-multimodal` |
| `--enforce-eager` | `--disable-cuda-graph` |
| `--no-enable-prefix-caching` | `--disable-radix-cache` |

### Pass-through (same name on both sides)

These vLLM flags are forwarded verbatim — SGLang accepts the same name and semantics:

`--trust-remote-code`, `--dtype`, `--quantization`, `--kv-cache-dtype`, `--revision`, `--load-format`, `--model-impl`, `--tokenizer-mode`, `--skip-tokenizer-init`, `--download-dir`, `--pipeline-parallel-size`, `--data-parallel-size`, `--nnodes`, `--node-rank`, `--enable-lora`, `--max-lora-rank`, `--lora-target-modules`, `--attention-backend`, `--served-model-name`, `--chat-template`, `--api-key`, `--reasoning-parser`, `--tool-call-parser`, `--tool-server`, `--otlp-traces-endpoint`, `--kv-events-config`, `--enable-mfu-metrics`, `--enable-eplb`, `--cpu-offload-gb`, `--ssl-keyfile`, `--ssl-certfile`, `--ssl-ca-certs`, `--enable-ssl-refresh`, …

Any other unknown flag is also passed through — it may be a valid SGLang flag.

### Dropped (no SGLang equivalent, or vLLM-default behavior already matches)

`--enable-prefix-caching`, `--enable-chunked-prefill`, `--no-enable-chunked-prefill`, `--disable-log-requests`, `--disable-log-stats`, `--swap-space`, `--block-size`, `--num-gpu-blocks-override`, `--num-cpu-blocks-override`, `--distributed-executor-backend`, `--code-revision`, `--tokenizer-revision`, `--max-seq-len-to-capture`, `--max-cpu-loras`, `--lora-dtype`, `--enable-prompt-adapter`, `--scheduler-delay-factor`, `--limit-mm-per-prompt`

### Environment variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `SGLANG_TOOL_CALL_PARSER` | `mistral` | Override the tool-call-parser |
| `VLLM_SHIM_LOG` | `/tmp/vllm-shim.log` | Log file path |

## Metrics translation

The middleware exposes a `/metrics` endpoint that fetches SGLang's native Prometheus metrics and translates them to vLLM-compatible names. This allows existing Grafana dashboards and Prometheus alerts built for vLLM to work without modification.

### Metric mapping

| vLLM metric | SGLang source | Type | Notes |
|-------------|---------------|------|-------|
| `vllm:healthy_pods_total` | Synthesized | Gauge | 1 when SGLang is ready, 0 otherwise |
| `vllm:num_requests_running` | `sglang:num_running_reqs` | Gauge | |
| `vllm:num_requests_waiting` | `sglang:num_queue_reqs` | Gauge | |
| `vllm:kv_cache_usage_perc` | `sglang:num_used_tokens / max_total_num_tokens * 100` | Gauge | Derived from two SGLang gauges |
| `vllm:gpu_prefix_cache_hit_rate` | `sglang:cache_hit_rate` | Gauge | Value is 0.0-1.0 |
| `vllm:e2e_request_latency_seconds` | `sglang:e2e_request_latency_seconds` | Histogram | |
| `vllm:request_time_per_output_token_seconds` | `sglang:inter_token_latency_seconds` | Histogram | |
| `vllm:time_to_first_token_seconds` | `sglang:time_to_first_token_seconds` | Histogram | |
| `vllm:prompt_tokens_total` | `sglang:prompt_tokens_total` | Counter | |
| `vllm:generation_tokens_total` | `sglang:generation_tokens_total` | Counter | |
| `vllm:num_requests_swapped` | N/A | Gauge | Always 0 (SGLang has no swap) |
| `vllm:request_success_total` | `sglang:num_requests_total` | Counter | |
| `vllm:prompt_tokens_cached_total` | `sglang:cached_tokens_total` | Counter | |

Unmapped SGLang metrics (e.g. `sglang:spec_accept_rate`, `sglang:gen_throughput`) are passed through with their original `sglang:` prefix.

The metrics endpoint has a 1-second cache to avoid hammering the SGLang metrics server on concurrent scrapes.

## Middleware: request body fixes

SGLang rejects certain parameters and schemas that vLLM (and OpenClaw) send. The middleware fixes these automatically:

### Stripped parameters

These vLLM-only parameters are removed from request bodies before forwarding to SGLang:

- `logprobs` / `top_logprobs` — SGLang's Mistral tool-call parser rejects these
- `chat_template_kwargs` — OpenClaw sends this for reasoning models; SGLang doesn't support it
- `guided_json` / `guided_regex` — vLLM-only guided decoding params

### Schema fixes

OpenClaw (and some vLLM configurations) send tool schemas with `properties: []` instead of `properties: {}`. SGLang requires `properties` to be an object at **every level** of the schema, including nested `items` and sub-objects.

The middleware recursively walks the entire JSON Schema tree and fixes:
- `properties: []` → `properties: {}` (at any depth)
- `required: <non-list>` → removed
- `parameters: <non-object>` → `{"type": "object", "properties": {}}`

## Layout

| Path | Purpose |
|------|---------|
| `packages/vllm-shim/` | Main package (`vllm_shim`): CLI, supervisor, middleware, backend abstraction, SGLang implementation |
| `packages/vllm-entrypoints/` | Stub package that ships a top-level `vllm/` so `python -m vllm.X` invocations dispatch to the shim |
| `docker/<backend>/Dockerfile.<platform>` | Per-backend, per-platform image build (e.g. `docker/sglang/Dockerfile.rocm`) |
| `Jenkinsfile` | CI/CD: builds and pushes to Vultr container registry |

## Development

```bash
uv sync
uv run pytest
uv run mypy
uv run ruff check .
```

The `vllm` console script replaces the previous bash binary. `python -m vllm.entrypoints.openai.api_server` and similar invocations are intercepted by the `vllm-entrypoints` stubs.

## Deploy

Each `(backend, platform)` pair is its own Dockerfile under `docker/`. Build from the repo root so the build context sees `pyproject.toml`, `uv.lock`, and `packages/`:

```bash
# ROCm + SGLang
docker build -f docker/sglang/Dockerfile.rocm -t vllm-to-sglang:sglang-rocm .

# CUDA + SGLang
docker build -f docker/sglang/Dockerfile.cuda -t vllm-to-sglang:sglang-cuda .
```

Or via Jenkins (default `BACKEND=sglang`, `PLATFORM=rocm`; final image tag is `${TAG}-${BACKEND}-${PLATFORM}`):

```bash
curl -X POST "https://jenkins.sweetapi.com/job/vllm-to-sglang/buildWithParameters" \
  -d TAG=nightly \
  -d BACKEND=sglang \
  -d PLATFORM=rocm
```
