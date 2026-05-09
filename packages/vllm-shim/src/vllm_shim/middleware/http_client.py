from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI

_client: httpx.AsyncClient | None = None


def get_client() -> httpx.AsyncClient:
    """Returns the process-global AsyncClient.

    In production the lifespan initializes the client before any request
    arrives.  Outside of a running lifespan (e.g. in tests) a default client
    is created lazily so callers do not need to manage the lifespan
    explicitly.
    """
    global _client
    if _client is None:
        _client = httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=10.0))
    return _client


def make_lifespan() -> object:
    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        global _client
        _client = httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=10.0))
        try:
            yield
        finally:
            await _client.aclose()
            _client = None

    return lifespan
