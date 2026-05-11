# Architecture

vllm-shim runs three cooperating processes inside a single container, all spawned by one `vllm serve …` invocation. The shim's job is to make that invocation look identical to a stock vLLM container from the outside while internally driving SGLang or TensorRT-LLM (selected via `VLLM_SHIM_BACKEND`; see `docs/backends.md`).

## The three processes

```
client
  │
  ▼  :8000 (the port the caller passed via --port)
┌─────────────┐
│  haproxy    │  L7 reverse proxy + health gate
└──────┬──────┘
       │  :8002 (loopback)
       ▼
┌─────────────┐
│ middleware  │  FastAPI: /metrics, /health, body-rewriting proxy
└──────┬──────┘
       │  :8001
       ▼
┌─────────────┐
│   backend   │  SGLang or TRT-LLM, plus native /metrics
└─────────────┘
```

| Process | Role | Why it exists |
|---|---|---|
| **haproxy** | Front-door L7 proxy | Turns backend-down into a clean 503 instead of a connect failure; gives k8s a `/health` it can probe before the backend is ready. See `docs/haproxy.md`. |
| **middleware** | FastAPI app on 127.0.0.1 | Rewrites request bodies, translates the backend's `/metrics` into vLLM-named series, forwards everything else. See `docs/middleware.md`. |
| **backend** | `sglang serve` or `trtllm-serve` | The actual inference engine. Pluggable via the `Backend` ABC. See `docs/backends.md`. |

All three are children of the shim entrypoint (`vllm_shim.cli.entrypoint.main`). When any one dies, the supervisor tears down the others. See `docs/supervisor.md`.

## Port allocation

The caller picks a single port (`--port`, default 8000). The shim derives the other two by offset:

| Port | Role |
|---|---|
| `N`     | haproxy (the public listener; what `--port` named) |
| `N + 1` | backend |
| `N + 2` | middleware |

This is implemented in `PortAllocation.from_listen` (`packages/vllm-shim/src/vllm_shim/values/port_allocation.py`). The two derived ports are loopback-only; only the public port reaches the network.

The middleware listens on `0.0.0.0` for portability, but haproxy connects to it via `127.0.0.1` because the supervisor passes the loopback address explicitly when it spawns the middleware.

## Request lifecycle

A typical inference request flows like this:

1. **Client → haproxy `:N`.** haproxy checks its `nbsrv(sglang)` ACL (the haproxy backend pool is named `sglang` regardless of which engine is actually running; that's a fixed string in the haproxy template, not a backend selector). If the backend's TCP-level health check has failed three times in a row, haproxy returns the static 503 errorfile without ever opening a backend connection. Otherwise it forwards to the middleware.
2. **haproxy → middleware `:N+2`.** FastAPI routes the request: `/health` and `/metrics` go to dedicated handlers, everything else falls through to the catch-all proxy handler.
3. **Middleware filter chain.** For `POST /v1/chat/completions`, every `RequestFilter` whose `applies_to(method, path)` returns `True` runs over the body in order. The chain is backend-specific: `SGLangBackend.__init__` declares `StripVLLMParams` then `FixToolSchemas`; `TRTLLMBackend.__init__` ships an empty tuple (no body rewriting needed). Order is part of each backend's contract.
4. **Middleware → backend `:N+1`.** The middleware uses `httpx.AsyncClient`. If the body indicated `stream: true`, it opens a streaming response and pipes chunks back unchanged.
5. **Response.** Hop-by-hop headers (per RFC 7230 plus `content-length`) are stripped from the backend response before forwarding. A 4xx/5xx triggers `dump_error` to append a structured block to the shim error log.

Health and metrics requests bypass the filter chain and the upstream filtering entirely; they're served by their own handlers (`HealthHandler`, `MetricsHandler`).

## Health gating

The shim has two distinct notions of "healthy":

- **haproxy → backend.** haproxy probes the backend's `/health` directly with `httpchk GET /health` (`fall 3 rise 2`). If three checks in a row fail, the backend is "down" and `nbsrv(sglang) gt 0` becomes false, so haproxy serves the 503 errorfile.
- **k8s liveness/readiness → haproxy.** k8s probes `/health` on the public port. haproxy's own health rule turns that into 200 when the backend is up, 503 when it isn't, without ever forwarding the request.

This means k8s sees a clean `200`/`503` boundary even during backend startup or crash recovery, which is the property the vLLM production stack expects.

## Shutdown

The supervisor sends `SIGTERM` to all three children simultaneously, then `wait()`s on them in declared order (haproxy, middleware, backend) against a shared 25-second deadline. Any child still alive at the deadline gets `SIGKILL`. The "send first, wait second" split (rather than terminate-one-and-wait-one) is what bounds total shutdown time to the grace period regardless of how many children there are; see `Supervisor._terminate_all` in `packages/vllm-shim/src/vllm_shim/cli/supervisor.py` and `docs/supervisor.md` for the full rationale.

The declared order matters for *drain quality* even though the SIGTERMs go out simultaneously: haproxy exits in milliseconds, so waiting on it first lets in-flight requests finish flowing through the middleware before the backend dies.

## Module layout

| Module | Role |
|---|---|
| `vllm_shim.cli.entrypoint` | `main()`: parses args, builds command lines, spawns the three processes, runs the supervisor. |
| `vllm_shim.cli.parser` | `ArgParser`: extracts the bits the supervisor needs (model, host, port) and routes everything else into `passthrough`. |
| `vllm_shim.cli.haproxy` | haproxy config templating, error file, and launch helper. |
| `vllm_shim.cli.supervisor` | `Supervisor` and `ManagedProcess`: process lifecycle, signal handling. |
| `vllm_shim.cli.rocm_probe` | Shells out to `rocminfo` to identify the GPU SKU bucket; gates AITER capture/restore. |
| `vllm_shim.cli.info` | Launch-time info dump (JSON + stderr summary), `vllm-shim-info` console script. |
| `vllm_shim.values` | Frozen dataclasses (`ParsedArgs`, `PortAllocation`, `ServiceAddress`, `Parallelism`) shared across layers. |
| `vllm_shim.backend.base` | Backend ABCs: `Backend`, `ArgTranslator`, `EnvTranslator`, `Launcher`, `MetricsTranslator`, `RequestFilter`, `ParallelismExtractor`. |
| `vllm_shim.backend.sglang` | The concrete SGLang backend (args, env, launcher, metrics, two filters, parallelism). |
| `vllm_shim.backend.trtllm` | The concrete TensorRT-LLM backend (args, env, launcher, metrics, no filters, parallelism). |
| `vllm_shim.backend.registry` | `select()`: env-driven backend dispatch (`VLLM_SHIM_BACKEND`). |
| `vllm_shim.aiter` | AITER shape capture (stderr tee + dedup CSV) and config restore (symlink seeding). ROCm only; no-op elsewhere. See `docs/aiter.md`. |
| `vllm_shim.middleware` | FastAPI app, three handlers (health, metrics, proxy), shared httpx client, error-dump helper. |
| `vllm-entrypoints` package | Top-level `vllm/` namespace whose `__main__.py` files redirect `python -m vllm.X` invocations to the shim. See `docs/entrypoints.md`. |

## Why this shape

The three-process design isn't a quirk; each layer carries a property the others can't:

- A pure subprocess wrapper around the backend couldn't translate metrics or rewrite request bodies.
- A pure FastAPI middleware in front of the backend couldn't give k8s a clean `/health` during backend startup (FastAPI is up while the backend is still loading weights, which is exactly when the production stack will get angry).
- haproxy is the only layer that can hold open a TCP listener at the public port while the inference layer is still booting and turn that into a coherent HTTP-level 503.

The middleware handles the OpenAI-protocol-shaped fixes (body rewrites, metric renames). haproxy handles the network-shaped concerns (listener, health gating, errorfile). The backend handles the tokens.
