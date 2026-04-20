"""
vLLM → SGLang request middleware.

Sits between haproxy and SGLang to strip vLLM-only parameters
that cause SGLang to return 422/400 errors.

Currently strips: logprobs, top_logprobs
(SGLang's Mistral tool-call parser rejects these; vLLM accepts them.)

Architecture:
  haproxy (port N) → middleware (port N+2) → SGLang (port N+1)

haproxy still handles /metrics stub and /health instant responses.
This middleware only touches the proxied request bodies.
"""

import json
import os
import re
import time
import asyncio
import httpx
from datetime import datetime
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, Response
import uvicorn

SGLANG_HOST = os.environ.get("SGLANG_HOST", "127.0.0.1")
SGLANG_PORT = int(os.environ.get("SGLANG_PORT", "8001"))
LISTEN_PORT = int(os.environ.get("MIDDLEWARE_PORT", "8002"))
METRICS_CACHE_SECONDS = 1.0

SGLANG_TO_VLLM = {
    "sglang:num_running_reqs": "vllm:num_requests_running",
    "sglang:num_queue_reqs": "vllm:num_requests_waiting",
    "sglang:cache_hit_rate": "vllm:gpu_prefix_cache_hit_rate",
    "sglang:e2e_request_latency_seconds": "vllm:e2e_request_latency_seconds",
    "sglang:inter_token_latency_seconds": "vllm:request_time_per_output_token_seconds",
    "sglang:time_to_first_token_seconds": "vllm:time_to_first_token_seconds",
    "sglang:prompt_tokens_total": "vllm:prompt_tokens_total",
    "sglang:generation_tokens_total": "vllm:generation_tokens_total",
    "sglang:num_requests_total": "vllm:request_success_total",
    "sglang:num_aborted_requests_total": "vllm:request_success_total",
    "sglang:cached_tokens_total": "vllm:prompt_tokens_cached_total",
}

_RE_METRIC_LINE = re.compile(r"^(#\s+(?:HELP|TYPE)\s+)?(\w[\w:]*)(.*)")
_RE_SAMPLE_LINE = re.compile(r"^(\w[\w:]*)(\{[^}]*\})?\s+(.+)$")

# Params that vLLM accepts but SGLang rejects.
# Extend this set as more incompatibilities are discovered.
STRIP_PARAMS = {"logprobs", "top_logprobs", "chat_template_kwargs", "guided_json", "guided_regex"}

client: httpx.AsyncClient | None = None
_sglang_ready = False


async def _lifespan(app_instance):
    global client
    client = httpx.AsyncClient(
        timeout=httpx.Timeout(300.0, connect=10.0),
    )
    # Background task: wait for SGLang to become available
    asyncio.create_task(_wait_for_sglang())
    yield
    await client.aclose()


async def _wait_for_sglang():
    """Poll SGLang until it's accepting connections, then mark ready."""
    global _sglang_ready
    while True:
        try:
            resp = await client.get(
                f"http://{SGLANG_HOST}:{SGLANG_PORT}/health",
                timeout=httpx.Timeout(5.0, connect=2.0),
            )
            if resp.status_code == 200:
                _sglang_ready = True
                print(f"Middleware: SGLang is ready at {SGLANG_HOST}:{SGLANG_PORT}")
                return
        except (httpx.ConnectError, httpx.TimeoutException):
            pass
        await asyncio.sleep(2)


app = FastAPI(lifespan=_lifespan)


@app.get("/health")
async def health():
    """Health check — haproxy polls this. Returns 200 only if SGLang is up."""
    global _sglang_ready
    if not _sglang_ready:
        return Response(content="SGLang not ready", status_code=503)
    try:
        resp = await client.get(
            f"http://{SGLANG_HOST}:{SGLANG_PORT}/health",
            timeout=httpx.Timeout(5.0, connect=2.0),
        )
        return Response(content=resp.content, status_code=resp.status_code,
                        media_type=resp.headers.get("content-type"))
    except (httpx.ConnectError, httpx.TimeoutException):
        _sglang_ready = False
        # Re-trigger background wait
        asyncio.create_task(_wait_for_sglang())
        return Response(content="SGLang not ready", status_code=503)


ERROR_LOG = os.environ.get("VLLM_SHIM_LOG", "/tmp/vllm-shim.log")


def _fix_schema(schema: dict) -> bool:
    """Recursively fix a JSON Schema dict: properties must be object, required must be list of strings."""
    fixed = False
    # Fix 'properties' — must be dict, not array/null
    if "properties" in schema and not isinstance(schema["properties"], dict):
        schema["properties"] = {}
        fixed = True
    # Fix 'required' — must be list of strings or absent
    if "required" in schema and not isinstance(schema["required"], list):
        del schema["required"]
        fixed = True
    # Recurse into every property value
    if isinstance(schema.get("properties"), dict):
        for val in schema["properties"].values():
            if isinstance(val, dict):
                if _fix_schema(val):
                    fixed = True
    # Recurse into items (for array-of-objects)
    if isinstance(schema.get("items"), dict):
        if _fix_schema(schema["items"]):
            fixed = True
    # Recurse into anyOf, allOf, oneOf
    for key in ("anyOf", "allOf", "oneOf"):
        if isinstance(schema.get(key), list):
            for item in schema[key]:
                if isinstance(item, dict):
                    if _fix_schema(item):
                        fixed = True
    # Recurse into additionalProperties if it's a schema
    if isinstance(schema.get("additionalProperties"), dict):
        if _fix_schema(schema["additionalProperties"]):
            fixed = True
    return fixed


def _dump_error(request_body: bytes, status_code: int, resp_headers: dict, resp_body_raw: bytes, path: str = ""):
    """Log full request + response payload when SGLang returns an error (4xx/5xx)."""
    try:
        ts = datetime.now().isoformat()
        req_json = None
        try:
            req_json = json.loads(request_body)
        except (json.JSONDecodeError, UnicodeDecodeError):
            pass

        resp_text = resp_body_raw.decode("utf-8", errors="replace")[:4000]
        resp_json = None
        try:
            resp_json = json.loads(resp_text)
        except (json.JSONDecodeError, UnicodeDecodeError):
            pass

        with open(ERROR_LOG, "a") as f:
            f.write(f"\n{'='*60}\n")
            f.write(f"[{ts}] ERROR DUMP — SGLang returned HTTP {status_code}\n")
            f.write(f"Path: {path}\n")
            f.write(f"--- Request Body ---\n")
            if req_json:
                f.write(json.dumps(req_json, indent=2, ensure_ascii=False)[:8000])
            else:
                f.write(request_body.decode("utf-8", errors="replace")[:8000])
            f.write(f"\n--- Response (HTTP {status_code}) ---\n")
            if resp_json:
                f.write(json.dumps(resp_json, indent=2, ensure_ascii=False)[:4000])
            else:
                f.write(resp_text)
            f.write(f"\n{'='*60}\n")

        print(f"[{ts}] ERROR DUMP: HTTP {status_code} on {path} — full payload written to {ERROR_LOG}")
    except Exception as e:
        print(f"_dump_error failed: {e}")


_metrics_cache: tuple[float, str] | None = None


def _translate_metrics_line(line: str) -> list[str]:
    m = _RE_SAMPLE_LINE.match(line)
    if m:
        name, labels_str, value = m.group(1), m.group(2) or "", m.group(3)
        vllm_name = SGLANG_TO_VLLM.get(name)
        if vllm_name:
            return [f"{vllm_name}{labels_str} {value}"]
        for suffix in ("_bucket", "_sum", "_count"):
            if name.endswith(suffix):
                vllm_base = SGLANG_TO_VLLM.get(name[: -len(suffix)])
                if vllm_base:
                    return [f"{vllm_base}{suffix}{labels_str} {value}"]
        return [line]

    m = _RE_METRIC_LINE.match(line)
    if m:
        prefix, name, rest = m.group(1) or "", m.group(2), m.group(3)
        vllm_name = SGLANG_TO_VLLM.get(name)
        if vllm_name:
            return [f"{prefix}{vllm_name}{rest}"]
        return [line]

    return [line]


def _build_vllm_metrics(raw: str) -> str:
    gauges: dict[str, dict[str, float]] = {}
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = _RE_SAMPLE_LINE.match(line)
        if not m:
            continue
        name, labels_str, value_str = m.group(1), m.group(2) or "", m.group(3)
        if name in ("sglang:num_used_tokens", "sglang:max_total_num_tokens"):
            try:
                val = float(value_str.split()[0])
            except (ValueError, IndexError):
                continue
            gauges.setdefault(name, {})[labels_str] = val

    output_lines: list[str] = []
    for line in raw.splitlines():
        translated = _translate_metrics_line(line)
        output_lines.extend(translated)

    used = gauges.get("sglang:num_used_tokens", {})
    capacity = gauges.get("sglang:max_total_num_tokens", {})
    if used and capacity:
        output_lines.append("# HELP vllm:kv_cache_usage_perc KV cache usage percentage")
        output_lines.append("# TYPE vllm:kv_cache_usage_perc gauge")
        all_labels = set(used.keys()) | set(capacity.keys())
        for lbl in sorted(all_labels):
            u = used.get(lbl, 0.0)
            c = capacity.get(lbl, 0.0)
            pct = (u / c * 100.0) if c > 0 else 0.0
            output_lines.append(f"vllm:kv_cache_usage_perc{lbl} {pct:.4f}")

    output_lines.append("# HELP vllm:healthy_pods_total Number of healthy vLLM pods")
    output_lines.append("# TYPE vllm:healthy_pods_total gauge")
    if _sglang_ready:
        output_lines.append('vllm:healthy_pods_total{endpoint="default"} 1')
    else:
        output_lines.append('vllm:healthy_pods_total{endpoint="default"} 0')

    output_lines.append("# HELP vllm:num_requests_swapped Number of swapped requests")
    output_lines.append("# TYPE vllm:num_requests_swapped gauge")
    output_lines.append("vllm:num_requests_swapped 0")

    return "\n".join(output_lines) + "\n"


@app.get("/metrics")
async def metrics():
    global _metrics_cache
    if _metrics_cache and (time.monotonic() - _metrics_cache[0]) < METRICS_CACHE_SECONDS:
        return Response(content=_metrics_cache[1], media_type="text/plain; version=0.0.4; charset=utf-8")

    if not _sglang_ready:
        return Response(content="SGLang not ready", status_code=503)

    try:
        resp = await client.get(
            f"http://{SGLANG_HOST}:{SGLANG_PORT}/metrics",
            timeout=httpx.Timeout(10.0, connect=5.0),
        )
        if resp.status_code != 200:
            return Response(content=resp.content, status_code=resp.status_code,
                            media_type=resp.headers.get("content-type"))
        translated = _build_vllm_metrics(resp.text)
        _metrics_cache = (time.monotonic(), translated)
        return Response(content=translated, media_type="text/plain; version=0.0.4; charset=utf-8")
    except (httpx.ConnectError, httpx.TimeoutException) as e:
        return Response(
            content=f"SGLang metrics backend unavailable: {e}",
            status_code=503,
            media_type="text/plain",
        )

@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"])
async def proxy(path: str, request: Request):
    body = await request.body()
    is_streaming = False

    # Strip incompatible params from chat completion POST requests
    if request.method == "POST" and "chat/completions" in path and body:
        try:
            data = json.loads(body)
            is_streaming = data.get("stream", False)
            stripped_any = False
            for key in STRIP_PARAMS:
                if key in data:
                    del data[key]
                    stripped_any = True

            # Fix tool function parameters: recurse to fix ALL bad properties/required
            tools = data.get("tools")
            if isinstance(tools, list):
                for tool in tools:
                    func = tool.get("function") if isinstance(tool, dict) else None
                    if not isinstance(func, dict):
                        continue
                    if not isinstance(func.get("parameters"), dict):
                        func["parameters"] = {"type": "object", "properties": {}}
                        stripped_any = True
                    if _fix_schema(func["parameters"]):
                        stripped_any = True

            if stripped_any:
                body = json.dumps(data).encode()
        except (json.JSONDecodeError, UnicodeDecodeError):
            pass

    # Forward headers (skip hop-by-hop and ones we're replacing)
    fwd_headers = {
        k: v for k, v in request.headers.items()
        if k.lower() not in ("host", "content-length", "transfer-encoding")
    }
    fwd_headers["content-length"] = str(len(body))

    url = f"http://{SGLANG_HOST}:{SGLANG_PORT}/{path}"
    if request.query_params:
        url += f"?{request.query_params}"

    try:
        if is_streaming:
            req = client.build_request(request.method, url, content=body, headers=fwd_headers)
            resp = await client.send(req, stream=True)

            # Dump on error for streaming responses
            if resp.status_code >= 400:
                error_body = await resp.aread()
                _dump_error(body, resp.status_code, resp_headers=dict(resp.headers), resp_body_raw=error_body, path=path)
                await resp.aclose()
                return Response(
                    content=error_body,
                    status_code=resp.status_code,
                    media_type=resp.headers.get("content-type"),
                )

            async def stream_body():
                try:
                    async for chunk in resp.aiter_bytes():
                        yield chunk
                finally:
                    await resp.aclose()

            return StreamingResponse(
                stream_body(),
                status_code=resp.status_code,
                headers={"content-type": resp.headers.get("content-type", "text/event-stream")},
            )
        else:
            resp = await client.request(request.method, url, content=body, headers=fwd_headers)

            # Dump on error
            if resp.status_code >= 400:
                _dump_error(body, resp.status_code, resp_headers=dict(resp.headers), resp_body_raw=resp.content, path=path)

            return Response(
                content=resp.content,
                status_code=resp.status_code,
                media_type=resp.headers.get("content-type"),
            )
    except (httpx.ConnectError, httpx.TimeoutException) as e:
        return Response(
            content=json.dumps({"error": {"message": f"SGLang backend unavailable: {e}", "type": "backend_error"}}),
            status_code=503,
            media_type="application/json",
        )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=LISTEN_PORT, log_level="warning")
