"""FastAPI app factory and process entry point for the middleware."""

import os

import uvicorn
from fastapi import FastAPI, Request

from vllm_shim.backend import registry
from vllm_shim.backend.base.backend import Backend
from vllm_shim.middleware.handler.health import HealthHandler
from vllm_shim.middleware.handler.metrics import MetricsHandler
from vllm_shim.middleware.handler.proxy import ProxyHandler
from vllm_shim.middleware.http_client import make_lifespan
from vllm_shim.values.service_address import ServiceAddress


def create_app(backend: Backend, backend_address: ServiceAddress) -> FastAPI:
    """Wire the three handlers and the catch-all proxy onto a fresh FastAPI app."""
    app = FastAPI(lifespan=make_lifespan())  # type: ignore[arg-type]

    health = HealthHandler(backend, backend_address)
    metrics = MetricsHandler(backend, backend_address)
    proxy = ProxyHandler(backend, backend_address)

    app.add_api_route("/health", health.handle, methods=["GET"])
    app.add_api_route("/metrics", metrics.handle, methods=["GET"])

    async def proxy_route(path: str, request: Request) -> object:
        return await proxy.handle(path, request)

    app.add_api_route(
        "/{path:path}",
        proxy_route,
        methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],
    )
    return app


def run() -> None:
    """Entry point for `python -m vllm_shim.middleware` and the
    `vllm-shim-middleware` console script."""
    backend_addr = ServiceAddress(
        os.environ.get("VLLM_SHIM_BACKEND_HOST", "127.0.0.1"),
        int(os.environ.get("VLLM_SHIM_BACKEND_PORT", "8001")),
    )
    listen_port = int(os.environ.get("VLLM_SHIM_MIDDLEWARE_PORT", "8002"))
    app = create_app(registry.select(), backend_addr)
    uvicorn.run(app, host="0.0.0.0", port=listen_port, log_level="warning")
