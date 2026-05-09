# Configuration

The shim has no config file. Everything is driven by either CLI arguments (the `vllm serve` invocation) or environment variables. This page lists the env vars; CLI handling is covered in `docs/argument-translation.md`.

There are two reasons something is an env var rather than a flag:

1. **It's an internal handoff** between the supervisor process and the middleware process (the supervisor doesn't pass argv to the middleware; it passes env).
2. **It's a runtime knob** that operators want to set in a k8s `env:` block without rewriting the launch command.

## Env vars

### `VLLM_SHIM_BACKEND`

| | |
|---|---|
| Default | `sglang` |
| Read by | `vllm_shim.backend.registry.select` (called from supervisor and middleware) |
| Effect | Selects which `Backend` subclass to instantiate. |

Valid values are `sglang` and `trtllm`; anything else raises `ValueError("Unknown backend: ...")`. See `docs/backends.md`.

### `SGLANG_TOOL_CALL_PARSER`

| | |
|---|---|
| Default | `qwen3_coder` |
| Read by | `vllm_shim.backend.sglang.launcher.SGLangLauncher.build_command` |
| Effect | Sets SGLang's `--tool-call-parser`. |

The default is `qwen3_coder` because the Qwen3 family is the current target deployment. Override to `mistral`, `llama3`, etc. when serving a different model family. See SGLang's own docs for the full list of supported parsers.

### `SGLANG_HOST`

| | |
|---|---|
| Default | `127.0.0.1` |
| Read by | `vllm_shim.middleware.app.run` |
| Effect | The host the middleware connects to when forwarding to the backend. |

Set automatically by the supervisor when it spawns the middleware (the supervisor passes the `--host` value the caller provided). Not normally set by hand; it exists to keep the middleware as a stand-alone process that can read its config from the environment.

### `SGLANG_PORT`

| | |
|---|---|
| Default | `8001` |
| Read by | `vllm_shim.middleware.app.run` |
| Effect | The port the middleware connects to when forwarding to the backend. |

Set automatically by the supervisor (computed as `--port + 1`). Same role as `SGLANG_HOST`: the middleware doesn't have to know about port allocation.

### `MIDDLEWARE_PORT`

| | |
|---|---|
| Default | `8002` |
| Read by | `vllm_shim.middleware.app.run` |
| Effect | The port the middleware listens on. |

Set automatically by the supervisor (computed as `--port + 2`). The middleware binds `0.0.0.0:{MIDDLEWARE_PORT}`; haproxy connects via `127.0.0.1:{MIDDLEWARE_PORT}`.

### TRT-LLM env vars

### `TRTLLM_BACKEND`

| | |
|---|---|
| Default | `pytorch` |
| Read by | `vllm_shim.backend.trtllm.launcher.TRTLLMLauncher.build_command` |
| Effect | Sets `--backend` on `trtllm-serve`. Override to `tensorrt` when running pre-compiled engines, or `_autodeploy` for the autodeploy backend. |

### `TRTLLM_TOOL_PARSER`

| | |
|---|---|
| Default | `qwen3_coder` |
| Read by | `vllm_shim.backend.trtllm.launcher.TRTLLMLauncher.build_command` |
| Effect | Sets `--tool_parser`. The default matches `SGLANG_TOOL_CALL_PARSER` for cross-backend consistency on Qwen3 deployments. |

### `TRTLLM_REASONING_PARSER`

| | |
|---|---|
| Default | unset |
| Read by | `vllm_shim.backend.trtllm.launcher.TRTLLMLauncher.build_command` |
| Effect | If set, appends `--reasoning_parser <value>` to the launcher argv. TRT-LLM-only feature; SGLang has no equivalent. |

### `VLLM_SHIM_LOG`

| | |
|---|---|
| Default | `/tmp/vllm-shim.log` |
| Read by | `vllm_shim.middleware.handler.proxy` (module-level constant `ERROR_LOG_PATH`) |
| Effect | Where backend error dumps are appended. |

The middleware writes a structured block to this file every time SGLang returns 4xx/5xx (request body, status, response body, path, timestamp). See "Error dumping" in `docs/middleware.md`.

Notes:
- The path is read once at module import; changing it at runtime requires a middleware restart.
- The directory must exist and be writable. The default `/tmp` is fine for ephemeral container logs; override to a mounted volume for durable capture.
- Best-effort: if writing fails (full disk, read-only fs), the request still succeeds; the dump is silently skipped.

## Env vars set by the supervisor

For completeness, here's what the supervisor injects into the middleware subprocess in `entrypoint.main`:

```python
middleware_env = os.environ.copy()
middleware_env["SGLANG_HOST"] = backend_addr.host
middleware_env["SGLANG_PORT"] = str(backend_addr.port)
middleware_env["MIDDLEWARE_PORT"] = str(middleware_addr.port)
```

The middleware inherits the rest of the parent environment (so `VLLM_SHIM_BACKEND`, `VLLM_SHIM_LOG`, etc., flow through naturally).

## What's not configurable

Some constants have purposeful homes in code rather than env. If you need to change them, edit the source. That's the intended path:

- **Port offsets** (`+1` for backend, `+2` for middleware): `vllm_shim.values.port_allocation.PortAllocation.from_listen`. Changing this means coordinating with the haproxy template and the env vars above.
- **Shutdown grace** (25 seconds): `Supervisor.__init__` default. See `docs/supervisor.md`.
- **Metrics cache TTL** (1 second): `MetricsHandler.CACHE_SECONDS`.
- **httpx timeouts** (300s read, 10s connect): `vllm_shim.middleware.http_client.get_client`.
- **haproxy timeouts**: `_TEMPLATE` in `vllm_shim.cli.haproxy`.
- **Error dump truncation** (8000/4000 bytes): `_REQ_TRUNC` / `_RESP_TRUNC` in `vllm_shim.middleware.error_dump`.
- **Strip-params list**: `STRIP_PARAMS` in `vllm_shim.backend.sglang.filter.strip_params`.

If you find yourself wanting any of these to be runtime-configurable, that's a design decision worth discussing. Promoting an internal constant to an env var commits the project to maintaining backwards compatibility on its name and semantics.
