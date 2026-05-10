"""Catch-all proxy handler: runs request filters, forwards to the backend, dumps on error."""

import json
import os
from collections.abc import AsyncIterator
from pathlib import Path

import httpx
from fastapi import Request
from fastapi.responses import Response, StreamingResponse

from vllm_shim.backend.base.backend import Backend
from vllm_shim.middleware.error_dump import dump_error
from vllm_shim.middleware.http_client import get_client
from vllm_shim.values.service_address import ServiceAddress

ERROR_LOG_PATH = Path(os.environ.get("VLLM_SHIM_LOG", "/tmp/vllm-shim.log"))

# RFC 7230 hop-by-hop headers plus content-length (Starlette recomputes it).
# These must not be forwarded from the backend to the client.
_HOP_BY_HOP_HEADERS = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailers",
        "transfer-encoding",
        "upgrade",
        "content-length",
    }
)


def _forward_headers(headers: httpx.Headers) -> dict[str, str]:
    return {k: v for k, v in headers.items() if k.lower() not in _HOP_BY_HOP_HEADERS}


class ProxyHandler:
    """Forwards arbitrary HTTP requests to the backend after running the filter chain."""

    def __init__(self, backend: Backend, address: ServiceAddress) -> None:
        self._backend = backend
        self._address = address

    async def handle(self, path: str, request: Request) -> Response:
        """Run filters, forward, and stream or buffer the response based on the request shape."""
        body = await request.body()
        is_streaming = self._is_streaming(request.method, path, body)

        for f in self._backend.filters:
            if f.applies_to(request.method, path):
                body = f.transform(body)

        headers = {
            k: v
            for k, v in request.headers.items()
            if k.lower() not in ("host", "content-length", "transfer-encoding")
        }
        headers["content-length"] = str(len(body))

        url = f"{self._address.url()}/{path}"
        if request.query_params:
            url += f"?{request.query_params}"

        client = get_client()
        try:
            if is_streaming:
                req = client.build_request(request.method, url, content=body, headers=headers)
                resp = await client.send(req, stream=True)
                if resp.status_code >= 400:
                    error = await resp.aread()
                    dump_error(
                        log_path=ERROR_LOG_PATH,
                        backend_name=self._backend.name,
                        request_body=body,
                        status_code=resp.status_code,
                        response_body=error,
                        path=path,
                    )
                    await resp.aclose()
                    return Response(
                        content=error,
                        status_code=resp.status_code,
                        headers=_forward_headers(resp.headers),
                    )

                async def body_iter() -> AsyncIterator[bytes]:
                    try:
                        async for chunk in resp.aiter_bytes():
                            yield chunk
                    finally:
                        await resp.aclose()

                return StreamingResponse(
                    body_iter(),
                    status_code=resp.status_code,
                    headers=_forward_headers(resp.headers),
                )

            resp = await client.request(request.method, url, content=body, headers=headers)
            if resp.status_code >= 400:
                dump_error(
                    log_path=ERROR_LOG_PATH,
                    backend_name=self._backend.name,
                    request_body=body,
                    status_code=resp.status_code,
                    response_body=resp.content,
                    path=path,
                )
            return Response(
                content=resp.content,
                status_code=resp.status_code,
                headers=_forward_headers(resp.headers),
            )
        except (httpx.ConnectError, httpx.TimeoutException) as e:
            return Response(
                content=json.dumps(
                    {"error": {"message": f"backend unavailable: {e}", "type": "backend_error"}}
                ),
                status_code=503,
                media_type="application/json",
            )

    @staticmethod
    def _is_streaming(method: str, path: str, body: bytes) -> bool:
        if method != "POST" or "completions" not in path:
            return False
        try:
            data = json.loads(body)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return False
        return bool(isinstance(data, dict) and data.get("stream"))
