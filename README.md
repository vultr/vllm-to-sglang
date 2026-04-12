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
│  │   /metrics → 200 empty (stub)                    │  │
│  │   /health  → 200/503 based on backend state      │  │
│  │   /*       → proxy to middleware :8002            │  │
│  └──────────────────────────────────────────────────┘  │
│                        │                                │
│                        ▼                                │
│  ┌──────────────────────────────────────────────────┐  │
│  │ middleware :8002 (FastAPI)                        │  │
│  │   Strips vLLM-only params from request bodies    │  │
│  │   Recursively fixes tool JSON schemas            │  │
│  │   Forwards to SGLang :8001                       │  │
│  └──────────────────────────────────────────────────┘  │
│                        │                                │
│                        ▼                                │
│  ┌──────────────────────────────────────────────────┐  │
│  │ SGLang :8001 (internal)                          │  │
│  │   The actual inference server                    │  │
│  └──────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

## Argument translation

The shim dynamically translates vLLM CLI args to SGLang equivalents — no hardcoded model names or tensor-parallel sizes.

| vLLM flag | SGLang equivalent | Notes |
|-----------|-------------------|-------|
| `serve` | *(skipped)* | Subcommand only |
| `<model>` (positional) | `--model-path <model>` | |
| `--host` | Used for all three processes | |
| `--port` | haproxy binds this port | SGLang gets +1, middleware +2 |
| `--tensor-parallel-size` | `--tp` | |
| `--gpu_memory_utilization` | `--mem-fraction-static` | |
| `--trust-remote-code` | `--trust-remote-code` | |
| `--no-enable-prefix-caching` | *(skipped)* | No SGLang equivalent |
| `--enable-chunked-prefill` | *(skipped)* | No SGLang equivalent |
| `--tool-call-parser` | `--tool-call-parser` | Defaults to `mistral` |

Unknown flags are passed through as-is — they may be valid SGLang args.

### Environment variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `SGLANG_TOOL_CALL_PARSER` | `mistral` | Override the tool-call-parser |
| `VLLM_SHIM_LOG` | `/tmp/vllm-shim.log` | Log file path |

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

## Files

| File | Purpose |
|------|---------|
| `Dockerfile` | Builds on `lmsysorg/sglang-rocm`, installs haproxy, copies shim files |
| `Jenkinsfile` | CI/CD: builds and pushes to Vultr container registry |
| `vllm-shim.sh` | Shell shim — replaces the `vllm` binary, translates args |
| `vllm_shim_module.py` | Python shim — shadows `vllm.*` module imports, translates args |
| `vllm_middleware.py` | FastAPI middleware — strips bad params, fixes tool schemas |
| `README.md` | This file |

## Deploy

```bash
docker build -t vllm-to-sglang .
```

Or via Jenkins:

```bash
curl -X POST "https://jenkins.sweetapi.com/job/vllm-to-sglang/buildWithParameters" \
  -d TAG=nightly
```
