"""
vLLM → SGLang request middleware.

Sits between haproxy and SGLang to strip vLLM-only parameters
that cause SGLang to return 422/400 errors.

Currently strips: logprobs, top_logprobs
(SGLang's Mistral tool-call parser rejects these; vLLM accepts them.)

Architecture:
  haproxy (original port) → middleware (port+2) → SGLang (port+1)

haproxy still handles /metrics stub and /health instant responses.
This middleware only touches the proxied request bodies.
"""

import json
import os
import httpx
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, Response
import uvicorn

SGLANG_PORT = int(os.environ.get("SGLANG_PORT", "8001"))
LISTEN_PORT = int(os.environ.get("MIDDLEWARE_PORT", "8002"))

# Params that vLLM accepts but SGLang rejects.
# Extend this set as more incompatibilities are discovered.
STRIP_PARAMS = {"logprobs", "top_logprobs"}

app = FastAPI()
client: httpx.AsyncClient | None = None


@app.on_event("startup")
async def startup():
    global client
    client = httpx.AsyncClient(
        base_url=f"http://127.0.0.1:{SGLANG_PORT}",
        timeout=httpx.Timeout(300.0),
    )


@app.on_event("shutdown")
async def shutdown():
    await client.aclose()


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

    url = f"http://127.0.0.1:{SGLANG_PORT}/{path}"
    if request.query_params:
        url += f"?{request.query_params}"

    if is_streaming:
        req = client.build_request(request.method, url, content=body, headers=fwd_headers)
        resp = await client.send(req, stream=True)

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
        return Response(
            content=resp.content,
            status_code=resp.status_code,
            media_type=resp.headers.get("content-type"),
        )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=LISTEN_PORT, log_level="warning")
