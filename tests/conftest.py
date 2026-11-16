"""Test fixtures: a controllable in-process fake embedding backend."""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

import httpx

from gateway.app.config import Settings
from gateway.app.main import create_app


def vector_for(text: str, dim: int = 4) -> list[float]:
    digest = hashlib.sha256(text.encode()).digest()
    return [round(digest[i] / 255.0, 6) for i in range(dim)]


@dataclass
class Call:
    texts: list[str]
    model: str
    start: float
    end: float | None = None
    status: int | None = None


@dataclass
class FakeBackend:
    """MockTransport handler that mimics a single-slot embedding backend."""

    delay: float = 0.0
    status: int = 200
    # ok | malformed | short | duplicate_index | bad_coalesce |
    # truncated | read_timeout | connect_error | complete_after
    mode: str = "ok"
    shuffle: bool = True
    dim: int = 4
    # For mode == "complete_after": seconds after which a late response is
    # produced by the attempt that was abandoned by the gateway timeout.
    late_response_after: float = 0.0
    calls: list[Call] = field(default_factory=list)
    active: int = 0
    max_active: int = 0
    on_call: Any = None
    slot_busy: bool = False
    slot_checks: int = 0

    def __call__(self, request: httpx.Request) -> Any:
        if request.method == "GET":
            return self._handle_get(request)
        return self._handle_post(request)

    def _handle_get(self, request: httpx.Request) -> httpx.Response:
        self.slot_checks += 1
        if self.mode == "slot_error":
            return httpx.Response(500, text="probe failed")
        return httpx.Response(200, json={"busy": self.slot_busy})

    async def _handle_post(self, request: httpx.Request) -> httpx.Response:
        loop = asyncio.get_running_loop()
        body = json.loads(request.content)
        texts = body["input"]
        call = Call(texts=list(texts), model=body.get("model", ""), start=loop.time())
        self.calls.append(call)
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        if self.on_call is not None:
            await self.on_call(len(self.calls))
        try:
            if self.mode == "connect_error":
                raise httpx.ConnectError("connection refused", request=request)
            if self.mode == "truncated":
                # Bytes reached the gateway (so the backend accepted the
                # request) but the response body is incomplete.
                raise httpx.RemoteProtocolError(
                    "peer closed connection without sending complete message body",
                    request=request,
                )
            if self.mode == "read_timeout":
                raise httpx.ReadTimeout("timed out", request=request)
            if self.mode == "complete_after":
                # The gateway's own timeout fires first; this attempt later
                # completes, producing a genuine late response.
                await asyncio.sleep(self.late_response_after)
                call.status = 200
                return self._ok_response(texts)
            if self.delay:
                await asyncio.sleep(self.delay)
            call.status = self.status
            if self.status != 200:
                return httpx.Response(
                    self.status, json={"error": {"message": "injected failure"}}
                )
            if self.mode == "malformed":
                return httpx.Response(200, json={"object": "list"})
            if self.mode == "bad_coalesce":
                return httpx.Response(200, text="<html>not json</html>")
            if self.mode == "short":
                emitted = texts[:-1]
            else:
                emitted = texts
            data = [
                {
                    "object": "embedding",
                    "index": index,
                    "embedding": vector_for(text, self.dim),
                }
                for index, text in enumerate(emitted)
            ]
            if self.mode == "duplicate_index" and len(data) > 1:
                data[1]["index"] = 0
            if self.shuffle:
                data.reverse()
            return httpx.Response(
                200, json={"object": "list", "data": data, "model": "fake"}
            )
        finally:
            call.end = loop.time()
            self.active -= 1

    def _ok_response(self, texts: list[str]) -> httpx.Response:
        data = [
            {
                "object": "embedding",
                "index": index,
                "embedding": vector_for(text, self.dim),
            }
            for index, text in enumerate(texts)
        ]
        return httpx.Response(200, json={"object": "list", "data": data})

    @property
    def batch_sizes(self) -> list[int]:
        return [len(call.texts) for call in self.calls]

    @property
    def sent_texts(self) -> list[str]:
        return [text for call in self.calls for text in call.texts]


def make_settings(**overrides: Any) -> Settings:
    defaults: dict[str, Any] = {
        "backend_url": "http://fake.test/v1/embeddings",
        "backend_model": "fake-model",
        "backend_timeout_seconds": 10.0,
        "backend_connect_timeout_seconds": 2.0,
        "backend_late_response_grace_seconds": 10.0,
        "batch_window_ms": 20,
        "max_texts_per_request": 4,
        "max_texts_per_batch": 8,
        "default_timeout_ms": 5000,
        "max_timeout_ms": 600000,
        "disconnect_poll_interval_ms": 10,
        "state_file": None,
        "backend_admin_url": None,
        "recovery_probe_timeout_seconds": 2.0,
    }
    defaults.update(overrides)
    return Settings(**defaults)


@asynccontextmanager
async def gateway_client(
    backend: FakeBackend, **settings_overrides: Any
) -> AsyncIterator[tuple[Any, httpx.AsyncClient]]:
    settings = make_settings(**settings_overrides)
    app = create_app(settings, backend_transport=httpx.MockTransport(backend))
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://gateway"
        ) as client:
            yield app, client


async def post(
    client: httpx.AsyncClient, texts: list[str], timeout_ms: int = 5000, **extra: Any
) -> httpx.Response:
    payload = {"input": texts, "timeout_ms": timeout_ms, **extra}
    return await client.post("/v1/embeddings", json=payload)
