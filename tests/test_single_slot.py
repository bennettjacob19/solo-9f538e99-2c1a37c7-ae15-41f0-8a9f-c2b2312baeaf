"""Regression: single-slot concurrency must never exceed 1, including after a
batch whose completion was never confirmed.

These tests use a manually advanced scheduler clock in the sense that the
fake backend keeps a batch "accepted and computing" after the gateway-facing
response is gone, and record `max_active` the way the real mock backend does.
The pre-fix behavior sent the next batch after the read timeout, producing
max_active == 2.
"""

from __future__ import annotations

import asyncio

import httpx
import pytest

from gateway.app.batcher import BatchScheduler
from tests.conftest import FakeBackend, gateway_client, make_settings, post


async def _fault_id(client: httpx.AsyncClient) -> str:
    body = (await client.get("/admin/fault")).json()
    assert body["active"] is True, body
    return body["fault"]["fault_id"]


@pytest.mark.asyncio
async def test_no_second_batch_while_backend_still_computes_after_timeout() -> None:
    backend = FakeBackend(mode="read_timeout")
    async with gateway_client(backend) as (_, client):
        first = await post(client, ["first"], timeout_ms=5000)
        assert first.status_code == 502
        assert len(backend.calls) == 1

        # While the slot is quarantined, many more requests arrive.
        followups = await asyncio.gather(
            *[post(client, [f"n{i}"], timeout_ms=2000) for i in range(5)]
        )
        assert all(response.status_code == 503 for response in followups)
        # No batch was sent after the unconfirmed one.
        assert len(backend.calls) == 1
        assert backend.max_active == 1

        fault_id = await _fault_id(client)
        assert (
            await client.post("/admin/fault/recover", json={"fault_id": fault_id})
        ).status_code == 200
        backend.mode = "ok"
        # Normal service resumes; concurrency still never exceeded one.
        assert (await post(client, ["after"])).status_code == 200
        assert backend.max_active == 1


@pytest.mark.asyncio
async def test_slow_backend_with_caller_timeouts_keeps_one_in_flight() -> None:
    """The original production symptom: caller deadlines expire first.

    The batch is dispatched and the gateway's backend read timeout has not
    fired, so the batch is still in flight.  More callers time out, and the
    dispatcher must still send only one backend request at a time.
    """
    backend = FakeBackend(delay=0.2)
    async with gateway_client(backend) as (_, client):
        results = await asyncio.gather(
            post(client, ["a"], timeout_ms=40),
            post(client, ["b"], timeout_ms=40),
            post(client, ["c"], timeout_ms=40),
        )
        assert all(response.status_code == 504 for response in results)
        assert len(backend.calls) == 1
        assert backend.max_active == 1

        # The abandoned batch is still being awaited by the dispatcher.
        ok = await post(client, ["d"], timeout_ms=5000)
        assert ok.status_code == 200
        assert backend.max_active == 1


@pytest.mark.asyncio
async def test_unknown_failure_while_calling_backend_is_unconfirmed() -> None:
    """An unexpected exception between dispatch and confirmation must fail
    closed instead of silently releasing the slot."""
    settings = make_settings()
    backend = FakeBackend()
    client = httpx.AsyncClient(transport=httpx.MockTransport(backend))
    scheduler = BatchScheduler(settings, client)

    async def boom(*args: object, **kwargs: object) -> None:
        raise RuntimeError("unexpected codec failure")

    scheduler._call_backend = boom  # type: ignore[method-assign]
    await scheduler.start()
    try:
        loop = asyncio.get_running_loop()
        with pytest.raises(Exception) as caught:
            await scheduler.submit(
                request_id="r",
                texts=["x"],
                model="fake-model",
                deadline=loop.time() + 5,
            )
        assert getattr(caught.value, "code", "") == "backend_error"
        info = scheduler.fault_info()
        assert info["dispatch_paused"] is True
        assert info["fault"]["reason"] == "backend_unconfirmed"
    finally:
        await scheduler.close()
        await client.aclose()
