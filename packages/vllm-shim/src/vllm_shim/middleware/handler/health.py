import httpx
from fastapi import Response

from vllm_shim.backend.base.backend import Backend
from vllm_shim.middleware.http_client import get_client
from vllm_shim.values.service_address import ServiceAddress


class HealthHandler:
    def __init__(self, backend: Backend, address: ServiceAddress) -> None:
        self._backend = backend
        self._address = address

    async def handle(self) -> Response:
        try:
            resp = await get_client().get(
                self._address.url() + self._backend.health_path,
                timeout=httpx.Timeout(5.0, connect=2.0),
            )
            return Response(
                content=resp.content,
                status_code=resp.status_code,
                media_type=resp.headers.get("content-type"),
            )
        except (httpx.ConnectError, httpx.TimeoutException):
            return Response(content="backend not ready", status_code=503)
