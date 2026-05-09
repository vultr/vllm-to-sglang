# HAProxy frontend

haproxy is the public-facing layer in the shim. Everything entering the container hits haproxy first; the middleware and SGLang are loopback-only.

The config is templated and written by `vllm_shim.cli.haproxy` at startup. The launch helper just `Popen`s `haproxy -f <config>`.

## Why haproxy at all

The shim could in principle expose the FastAPI middleware directly on the public port. It doesn't, because the middleware can't give the production stack the health behavior it expects.

The vLLM production stack and most k8s deployments probe `/health` on the inference pod and use the result to gate traffic, restart counters, and dashboard "healthy pods" tiles. There are two transient states the middleware alone can't represent:

1. **SGLang still loading weights.** The middleware is up and serving FastAPI, but `localhost:8001` refuses connections. Without haproxy, `/health` would either hang on the connect timeout or return whatever FastAPI synthesizes; neither matches the contract.
2. **SGLang crashed and is restarting.** Same shape: middleware is fine, backend isn't.

haproxy turns these into a clean HTTP-level 503 because TCP-level health checks can drive HTTP-level routing decisions. That's the property the rest of the layering relies on.

## Config shape

The full template lives in `_TEMPLATE` in `packages/vllm-shim/src/vllm_shim/cli/haproxy.py`. The shape:

```haproxy
global
  maxconn 4096

defaults
  mode http
  timeout connect 5s
  timeout client 300s
  timeout server 300s

frontend proxy
  bind {listen_host}:{listen_port}

  acl is_health path /health
  acl sglang_up nbsrv(sglang) gt 0
  http-request deny deny_status 200 if is_health sglang_up
  http-request deny deny_status 503 if is_health
  errorfile 503 {error_path}

  default_backend sglang

backend sglang
  option httpchk GET /health
  http-check expect status 200
  server s1 {upstream_host}:{upstream_port} check inter 5s fall 3 rise 2
```

Three things are doing real work here.

### The `nbsrv()` health trick

`nbsrv(<backend>)` is a haproxy fetch that returns the count of healthy servers in a backend. It's normally used inside ACLs to make load-balancing decisions; here it's repurposed as a health signal:

```haproxy
acl sglang_up nbsrv(sglang) gt 0
http-request deny deny_status 200 if is_health sglang_up
http-request deny deny_status 503 if is_health
```

Read top-down:

- `is_health` is true when the path is `/health`.
- `sglang_up` is true when the SGLang backend has at least one healthy server.
- The first `http-request deny` rule denies `/health` requests with **status 200** when SGLang is up. It's a `deny` because we don't want to forward `/health` to the backend (SGLang's `/health` returns its own body and HTTP version), we just want the status code from the gate itself.
- The second rule denies `/health` requests with **status 503** when the first rule didn't match (SGLang is not up).

The clever bit is that "deny with status 200" is a haproxy idiom for "synthesize this response without forwarding upstream." It gives the shim a `/health` endpoint that's purely a function of haproxy's view of SGLang, with no involvement from FastAPI or SGLang itself.

For non-`/health` paths, the rules don't fire and the request flows through `default_backend sglang` (which actually points at the middleware; the backend stanza is named `sglang` for legacy reasons). When SGLang is down, that path returns the 503 errorfile because the backend has no healthy server to send to.

### The `httpchk` probe

```haproxy
backend sglang
  option httpchk GET /health
  http-check expect status 200
  server s1 {upstream_host}:{upstream_port} check inter 5s fall 3 rise 2
```

This is haproxy's view of SGLang's health, used by `nbsrv()` above. It actively probes `GET /health` on the upstream (which in the running config is the middleware, and the middleware proxies to SGLang's real `/health`).

- `inter 5s`: probe every 5 seconds.
- `fall 3`: 3 consecutive failures take the server out of rotation.
- `rise 2`: 2 consecutive successes bring it back.

The `fall 3` window means SGLang has to be down for ~15 seconds before the gate flips. That's intentional; short blips during startup or token-cache warmup don't oscillate the readiness state.

### The 503 errorfile

```haproxy
errorfile 503 /tmp/haproxy-errors/503-sglang.http
```

When haproxy needs to emit a 503 (either via the `is_health` rule above, or because the backend has no healthy server), it serves a static file rather than haproxy's default ASCII error page. The file is a complete HTTP response:

```
HTTP/1.0 503 Service Unavailable
Content-Length: 16
Connection: close
Content-Type: text/plain

SGLang not ready
```

Written by `write_error_file()` at startup; the path is hardcoded as `/tmp/haproxy-errors/503-sglang.http` (and the parent dir is created if missing).

A static errorfile is short, predictable, and doesn't leak haproxy version info. It also has the right content-length set up front, so haproxy doesn't have to compute one for synthesized errors.

## Timeouts

```
timeout connect 5s
timeout client 300s
timeout server 300s
```

5s connect is fine because haproxy is talking to a loopback service. 300s client/server is generous for streaming completions; long generations (large `max_tokens` on a slow model) can run several minutes, and we don't want haproxy to drop the connection mid-stream.

## What haproxy does not do

- **Auth.** No bearer-token check, no rate limiting, nothing. The vLLM `--api-key` flag passes through to SGLang; auth is the inference engine's problem.
- **TLS.** No `ssl crt …` lines. TLS termination is handled upstream (k8s ingress, service mesh, or the `--ssl-keyfile`/`--ssl-certfile` flags forwarded to SGLang).
- **Load balancing across replicas.** The shim is single-replica per pod by definition; haproxy's load-balancing config is left at defaults because there's only one upstream server.

## Modifying the config

Don't write to `/tmp/haproxy-shim.cfg` directly; it's regenerated at startup. Edit `_TEMPLATE` in `vllm_shim.cli.haproxy` and add a unit test in `packages/vllm-shim/tests/unit/cli/test_haproxy.py` asserting the rendered output contains the new directive.
