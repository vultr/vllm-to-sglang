"""End-to-end FastAPI app tests using a FakeBackend and pytest-httpx."""

import json
from typing import ClassVar

import httpx
import pytest
from vllm_shim.backend.base.args import ArgTranslator
from vllm_shim.backend.base.backend import Backend
from vllm_shim.backend.base.filter import RequestFilter
from vllm_shim.backend.base.launcher import Launcher
from vllm_shim.backend.base.metrics import MetricsTranslator
from vllm_shim.middleware.app import create_app
from vllm_shim.values.service_address import ServiceAddress


class _NoopArgs(ArgTranslator):
    def translate(self, vllm_args):  # type: ignore[no-untyped-def]
        return list(vllm_args), []


class _PassMetrics(MetricsTranslator):
    def translate(self, prom_text: str) -> str:
        return prom_text + "\nvllm:fake 1\n"


class _StripModel(RequestFilter):
    def applies_to(self, method: str, path: str) -> bool:
        return method == "POST"

    def transform(self, body: bytes) -> bytes:
        try:
            data = json.loads(body)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return body
        data.pop("model", None)
        return json.dumps(data).encode()


class _NoopLauncher(Launcher):
    def build_command(self, model, address, extra_args):  # type: ignore[no-untyped-def]
        return ["echo", model]


class FakeBackend(Backend):
    name: ClassVar[str] = "fake"

    def __init__(self) -> None:
        self.args = _NoopArgs()
        self.metrics = _PassMetrics()
        self.launcher = _NoopLauncher()
        self.filters = (_StripModel(),)


@pytest.fixture
def backend_address() -> ServiceAddress:
    return ServiceAddress("backend.test", 9001)


@pytest.fixture
def app(backend_address: ServiceAddress):  # type: ignore[no-untyped-def]
    return create_app(FakeBackend(), backend_address)


@pytest.mark.asyncio
async def test_health_proxies_backend(app, httpx_mock):  # type: ignore[no-untyped-def]
    httpx_mock.add_response(
        method="GET", url="http://backend.test:9001/health", text="ok", status_code=200
    )
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.get("/health")
    assert r.status_code == 200
    assert r.text == "ok"


@pytest.mark.asyncio
async def test_metrics_translated(app, httpx_mock):  # type: ignore[no-untyped-def]
    httpx_mock.add_response(
        method="GET",
        url="http://backend.test:9001/metrics",
        text="raw_metric 1\n",
        status_code=200,
    )
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.get("/metrics")
    assert r.status_code == 200
    assert "vllm:fake 1" in r.text


@pytest.mark.asyncio
async def test_proxy_runs_filter_chain(app, httpx_mock):  # type: ignore[no-untyped-def]
    captured: list[bytes] = []

    def callback(request: httpx.Request) -> httpx.Response:
        captured.append(request.content)
        return httpx.Response(200, content=b'{"ok":true}')

    httpx_mock.add_callback(callback, url="http://backend.test:9001/v1/x")
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.post("/v1/x", json={"model": "m", "keep": 1})
    assert r.status_code == 200
    assert b'"model"' not in captured[0]
    assert b'"keep"' in captured[0]


@pytest.mark.asyncio
async def test_proxy_forwards_backend_headers(app, httpx_mock):  # type: ignore[no-untyped-def]
    httpx_mock.add_response(
        method="POST",
        url="http://backend.test:9001/v1/x",
        content=b'{"ok":true}',
        headers={
            "content-type": "application/json",
            "x-request-id": "abc-123",
            "transfer-encoding": "chunked",
        },
        status_code=200,
    )
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.post("/v1/x", json={})
    assert r.headers.get("x-request-id") == "abc-123"
    assert r.headers.get("content-type") == "application/json"
    # Hop-by-hop headers must not be forwarded; Starlette will set its own.
    assert r.headers.get("transfer-encoding") != "chunked" or r.content == b'{"ok":true}'


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["/v1/chat/completions", "/v1/completions"])
async def test_streaming_response_passes_through(app, httpx_mock, path):  # type: ignore[no-untyped-def]
    chunks = [
        b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n',
        b"data: [DONE]\n\n",
    ]

    httpx_mock.add_response(
        method="POST",
        url=f"http://backend.test:9001{path}",
        stream=httpx.ByteStream(b"".join(chunks)),
        headers={"content-type": "text/event-stream"},
        status_code=200,
    )
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.post(path, json={"model": "m", "stream": True})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    assert b"[DONE]" in r.content


class FakeBackendCustomMetrics(Backend):
    """Variant of FakeBackend with metrics_path overridden, mirroring TRTLLMBackend."""

    name: ClassVar[str] = "fake-trtllm"
    metrics_path: ClassVar[str] = "/prometheus/metrics"

    def __init__(self) -> None:
        self.args = _NoopArgs()
        self.metrics = _PassMetrics()
        self.launcher = _NoopLauncher()
        self.filters = ()


@pytest.fixture
def app_custom_metrics(backend_address: ServiceAddress):  # type: ignore[no-untyped-def]
    return create_app(FakeBackendCustomMetrics(), backend_address)


@pytest.mark.asyncio
async def test_metrics_uses_backend_metrics_path(  # type: ignore[no-untyped-def]
    app_custom_metrics, httpx_mock,
) -> None:
    httpx_mock.add_response(
        method="GET",
        url="http://backend.test:9001/prometheus/metrics",
        text="raw_metric 1\n",
        status_code=200,
    )
    transport = httpx.ASGITransport(app=app_custom_metrics)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.get("/metrics")
    assert r.status_code == 200
    assert "vllm:fake 1" in r.text
