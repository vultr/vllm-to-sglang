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
import asyncio
import httpx
from datetime import datetime
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, Response
import uvicorn

SGLANG_HOST = os.environ.get("SGLANG_HOST", "127.0.0.1")
SGLANG_PORT = int(os.environ.get("SGLANG_PORT", "8001"))
LISTEN_PORT = int(os.environ.get("MIDDLEWARE_PORT", "8002"))

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

            # Fix tool function parameters: must be object, not array/null/missing
            # Also fix nested: parameters.properties must be object, not array
            tools = data.get("tools")
            if isinstance(tools, list):
                for tool in tools:
                    func = tool.get("function") if isinstance(tool, dict) else None
                    if not isinstance(func, dict):
                        continue
                    params = func.get("parameters")
                    if not isinstance(params, dict):
                        func["parameters"] = {"type": "object", "properties": {}}
                        stripped_any = True
                    else:
                        # Fix nested: properties must be object, not array
                        if not isinstance(params.get("properties"), dict):
                            params["properties"] = {}
                            stripped_any = True
                        # required must be a list of strings if present
                        if "required" in params and not isinstance(params["required"], list):
                            del params["required"]
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
