# Middleware

The middleware is a FastAPI app that sits between haproxy and SGLang. It does three things: serves `/health` and `/metrics` directly, runs request-body filters before forwarding, and dumps full request/response context on backend errors.

It runs as its own process (spawned by `vllm_shim.cli.entrypoint`) so SGLang restarts don't take it with them. The entry point is `vllm_shim.middleware.app.run` (also exposed as the `vllm-shim-middleware` console script).

## Routes

| Method | Path | Handler |
|---|---|---|
| `GET` | `/health` | `HealthHandler`: proxies `GET /health` to SGLang with a 5s timeout. On `ConnectError`/`TimeoutException`, returns `503 backend not ready`. |
| `GET` | `/metrics` | `MetricsHandler`: see `docs/metrics.md`. |
| `*` | `/{path:path}` | `ProxyHandler`: catch-all for everything else. |

The catch-all binds `GET POST PUT DELETE PATCH OPTIONS`. `HEAD` is not registered (FastAPI's catch-all path doesn't include it by default), but `HEAD /health` works because FastAPI auto-generates `HEAD` from `GET` for explicitly registered routes.

## The proxy handler

`ProxyHandler.handle` (`packages/vllm-shim/src/vllm_shim/middleware/handler/proxy.py`) is the workhorse. The flow:

1. **Read the body.** `await request.body()` materializes the full payload. Streaming uploads are not used here because the filter chain needs random access.
2. **Detect streaming response intent.** `_is_streaming` returns true only when the request is `POST … completions …` and the JSON body has `stream: true`. Other paths use the buffered code path.
3. **Run the filter chain.** Iterate `backend.filters` in declared order; each filter that `applies_to(method, path)` mutates the body via `transform(body)`. See "Filters" below.
4. **Build forwarded headers.** Drop `host`, `content-length`, and `transfer-encoding` from the inbound headers (Starlette and httpx will recompute them). Set `content-length` to the (possibly mutated) body length.
5. **Construct the upstream URL.** `http://{backend.host}:{backend.port}/{path}`, with the original query string appended verbatim.
6. **Send.** Streaming and non-streaming take different paths (below). On `ConnectError`/`TimeoutException`, return a 503 with a JSON `{"error": {"message": ..., "type": "backend_error"}}`.
7. **Strip hop-by-hop headers** from the backend response before returning it (`_HOP_BY_HOP_HEADERS`).
8. **Dump on error.** Any 4xx/5xx response triggers `dump_error` (see below).

### Streaming path

When `_is_streaming` is true:

```python
req  = client.build_request(...)
resp = await client.send(req, stream=True)
```

If the backend returns >= 400, the middleware reads the full body (no point streaming an error), dumps it, and returns it as a normal `Response`. Otherwise it returns a `StreamingResponse` whose async generator yields chunks from `resp.aiter_bytes()` and closes the upstream response in a `finally` block to guarantee connection cleanup.

### Hop-by-hop headers

Per RFC 7230 plus content-length:

```
connection, keep-alive, proxy-authenticate, proxy-authorization,
te, trailers, transfer-encoding, upgrade, content-length
```

`content-length` is included because Starlette recomputes it for the response, and forwarding the upstream value would be wrong if the body is re-encoded.

## Filters

Filters implement the `RequestFilter` ABC (`packages/vllm-shim/src/vllm_shim/backend/base/filter.py`):

```python
class RequestFilter(ABC):
    def applies_to(self, method: str, path: str) -> bool: ...
    def transform(self, body: bytes) -> bytes: ...
```

A backend declares its filter chain as `Backend.filters: tuple[RequestFilter, ...]`. They run in tuple order. SGLang ships two:

### `StripVLLMParams`

Removes JSON keys that vLLM clients send but SGLang's parsers reject. Currently:

```
logprobs, top_logprobs, chat_template_kwargs, guided_json, guided_regex
```

Applies to `POST` requests whose path contains `chat/completions`. Each is stripped independently; if none are present, the body is returned unchanged (no re-serialization).

Why: SGLang's Mistral tool-call parser rejects `logprobs`/`top_logprobs` outright, OpenClaw sends `chat_template_kwargs` for reasoning models that SGLang doesn't support, and `guided_json`/`guided_regex` are vLLM's structured-decoding flavor (SGLang has its own).

### `FixToolSchemas`

Repairs JSON-Schema fragments inside `tools[*].function.parameters` that SGLang's strict parser rejects:

- `properties: []` → `properties: {}` (at any depth; schema is walked recursively).
- `required: <non-list>` → key removed.
- `parameters: <non-object>` → replaced with `{"type": "object", "properties": {}}`.

The recursion walks `properties` values, `items`, `anyOf`/`allOf`/`oneOf` arrays, and `additionalProperties`. Only the parameters subtree is touched; everything outside `tools[*].function.parameters` is untouched.

Why: OpenClaw and some vLLM configurations emit `properties: []` (an array, not an object) when there are no parameters, or send malformed `required` fields. vLLM tolerates this; SGLang's xgrammar-backed validator does not.

### Adding a filter

Subclass `RequestFilter`, implement the two methods, and add an instance to the backend's `filters` tuple in declared run order. Order matters when filters edit overlapping shapes (e.g., a stripper that runs before a schema fixer). For SGLang, `StripVLLMParams` runs first because it removes keys outright; running a schema walker before it would waste work.

## Error dumping

Any 4xx/5xx from SGLang triggers `dump_error` (`packages/vllm-shim/src/vllm_shim/middleware/error_dump.py`). The dump is a structured block appended to the log file:

```
============================================================
[2026-05-09T14:23:01.012345] ERROR DUMP: SGLang returned HTTP 422
Path: /v1/chat/completions
--- Request Body ---
{
  "model": "...",
  ...
}
--- Response (HTTP 422) ---
{ ... }
============================================================
```

Bodies are JSON-pretty-printed when parseable, otherwise written as decoded text. Truncation: 8000 bytes for requests, 4000 bytes for responses. The dump function is best-effort; any exception inside it is swallowed so a write failure can't bring the request down.

The path is configurable via `VLLM_SHIM_LOG` (default `/tmp/vllm-shim.log`). See `docs/configuration.md`.

## httpx client lifecycle

A single process-global `httpx.AsyncClient` is reused across all requests (`packages/vllm-shim/src/vllm_shim/middleware/http_client.py`). It's created in the FastAPI lifespan handler and closed during shutdown. Timeouts: 300s read, 10s connect.

Outside a running lifespan (e.g., in tests using `ASGITransport`), `get_client` lazily creates a default client so callers don't need to manage the lifespan explicitly. This is what `tests/integration/test_app.py` relies on.

## What's deliberately not in the middleware

- **No retry logic.** SGLang either responds or it doesn't; haproxy's health-gate handles outages, and vLLM clients handle their own retries.
- **No request rewriting beyond filters.** URL paths and query strings are forwarded verbatim. Anything path-shaped goes in haproxy or in a new filter, not in the proxy handler itself.
- **No response body filters.** All filters operate on requests. SGLang's response shape matches vLLM's well enough that no rewriting has been needed; if that changes, a `ResponseFilter` ABC parallel to `RequestFilter` is the natural extension.
