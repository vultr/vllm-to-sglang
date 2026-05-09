"""GET /metrics handler: fetches, translates, caches; serves Prometheus exposition."""

import time

import httpx
from fastapi import Response

from vllm_shim.backend.base.backend import Backend
from vllm_shim.middleware.http_client import get_client
from vllm_shim.values.service_address import ServiceAddress

CACHE_SECONDS = 1.0
PROM_MEDIA_TYPE = "text/plain; version=0.0.4; charset=utf-8"


class MetricsHandler:
    """Scrapes the backend's /metrics, translates with the backend's MetricsTranslator,
    and caches the result for CACHE_SECONDS to absorb concurrent scrapes."""

    def __init__(self, backend: Backend, address: ServiceAddress) -> None:
        self._backend = backend
        self._address = address
        self._cache: tuple[float, str] | None = None

    async def handle(self) -> Response:
        if self._cache and (time.monotonic() - self._cache[0]) < CACHE_SECONDS:
            return Response(content=self._cache[1], media_type=PROM_MEDIA_TYPE)
        try:
            resp = await get_client().get(
                self._address.url() + self._backend.metrics_path,
                timeout=httpx.Timeout(10.0, connect=5.0),
            )
        except (httpx.ConnectError, httpx.TimeoutException) as e:
            return Response(
                content=f"backend metrics unavailable: {e}",
                status_code=503,
                media_type="text/plain",
            )
        if resp.status_code != 200:
            return Response(
                content=resp.content,
                status_code=resp.status_code,
                media_type=resp.headers.get("content-type"),
            )
        translated = self._backend.metrics.translate(resp.text)
        self._cache = (time.monotonic(), translated)
        return Response(content=translated, media_type=PROM_MEDIA_TYPE)
