"""Unconfirmed-batch quarantine, fault ids, recovery and restart persistence.

The central regression: a backend that accepted a batch but whose response was
lost must never be treated as an idle slot.  Before the fix the gateway
released the slot on the read timeout and sent the next batch, giving the
single-slot backend two batches in flight.
"""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from gateway.app.batcher import BatchScheduler
from tests.conftest import FakeBackend, gateway_client, make_settings, post


async def _fault(client: httpx.AsyncClient) -> dict:
    response = await client.get("/admin/fault")
    assert response.status_code == 200, response.text
    return response.json()


async def _recover(
    client: httpx.AsyncClient, fault_id: str, **extra: object
) -> httpx.Response:
    return await client.post(
        "/admin/fault/recover", json={"fault_id": fault_id, **extra}
    )


async def _wait_idle(client: httpx.AsyncClient, timeout: float = 2.0) -> None:
    loop = asyncio.get_running_loop()
    end = loop.time() + timeout
    while loop.time() < end:
        stats = (await client.get("/stats")).json()
        if stats["pending_requests"] == 0 and not stats["in_flight"]:
            return
        await asyncio.sleep(0.01)


# ------------------------------------------------------------------ quarantine
@pytest.mark.asyncio
async def test_truncated_response_quarantines_and_returns_502() -> None:
    backend = FakeBackend(mode="truncated")
    async with gateway_client(backend) as (_, client):
        response = await post(client, ["lost"], timeout_ms=5000)
        # The caller still gets the 502 backend_error contract.
        assert response.status_code == 502
        error = response.json()["error"]
        assert error["code"] == "backend_error"
        assert error["fault_id"].startswith("fault-")
        assert error["reason"] == "backend_unconfirmed"

        status = await _fault(client)
        assert status["active"] is True
        assert status["dispatch_paused"] is True
        assert status["fault"]["fault_id"] == error["fault_id"]
        # The gateway must not have sent another batch.
        assert len(backend.calls) == 1
        stats = (await client.get("/stats")).json()
        assert stats["unconfirmed_errors"] == 1
        assert stats["faults_created"] == 1
        assert stats["in_flight"] is False


@pytest.mark.asyncio
async def test_read_timeout_is_unconfirmed_not_a_slot_release() -> None:
    backend = FakeBackend(mode="read_timeout")
    async with gateway_client(backend) as (_, client):
        # The read timeout is the backend-side failure: callers of the batch
        # get the normal 502 backend_error, not a caller-deadline 504.
        response = await post(client, ["timeout"], timeout_ms=5000)
        assert response.status_code == 502
        assert response.json()["error"]["code"] == "backend_error"
        await _wait_idle(client)
        status = await _fault(client)
        assert status["dispatch_paused"] is True
        assert status["fault"]["reason"] == "backend_unconfirmed"
        # Crucially: the next batch was never dispatched.
        assert len(backend.calls) == 1


@pytest.mark.asyncio
async def test_queued_and_new_requests_get_503_with_fault_id() -> None:
    """Every waiter of the failed batch gets 502; the queue gets 503."""
    backend = FakeBackend(mode="read_timeout")
    async with gateway_client(backend) as (_, client):
        first = asyncio.create_task(post(client, ["a"], timeout_ms=5000))
        await asyncio.sleep(0.05)
        queued = asyncio.create_task(post(client, ["b"], timeout_ms=5000))
        await asyncio.sleep(0.01)
        response = await first
        assert response.status_code == 502
        fault_id = response.json()["error"]["fault_id"]

        queued_response = await queued
        assert queued_response.status_code == 503
        queued_error = queued_response.json()["error"]
        assert queued_error["code"] == "backend_unconfirmed"
        assert queued_error["fault_id"] == fault_id
        assert queued_error["reason"] == "backend_unconfirmed"
        assert queued_error["batch_id"].startswith("batch-")

        new_response = await post(client, ["c"], timeout_ms=1000)
        assert new_response.status_code == 503
        assert new_response.json()["error"]["fault_id"] == fault_id
        # Still only the one accepted batch ever reached the backend.
        assert backend.batch_sizes == [1]


@pytest.mark.asyncio
async def test_confirmed_backend_error_still_releases_the_slot() -> None:
    backend = FakeBackend(status=503)
    async with gateway_client(backend) as (_, client):
        response = await post(client, ["e"])
        assert response.status_code == 502
        assert response.json()["error"]["code"] == "backend_error"
        assert "fault_id" not in response.json()["error"]
        assert (await _fault(client))["active"] is False

        backend.status = 200
        recovered = await post(client, ["ok"])
        assert recovered.status_code == 200
        assert backend.batch_sizes == [1, 1]


@pytest.mark.asyncio
async def test_connect_error_is_confirmed_and_keeps_service_running() -> None:
    backend = FakeBackend(mode="connect_error")
    async with gateway_client(backend) as (_, client):
        response = await post(client, ["no-route"])
        assert response.status_code == 502
        assert (await _fault(client))["active"] is False
        backend.mode = "ok"
        assert (await post(client, ["after"])).status_code == 200


@pytest.mark.asyncio
async def test_invalid_but_complete_payload_is_confirmed() -> None:
    backend = FakeBackend(mode="short")
    async with gateway_client(backend) as (_, client):
        response = await post(client, ["m1", "m2"])
        assert response.status_code == 502
        assert (await _fault(client))["active"] is False
        backend.mode = "ok"
        assert (await post(client, ["after"])).status_code == 200


@pytest.mark.asyncio
async def test_success_still_releases_and_preserves_merge_order() -> None:
    backend = FakeBackend()
    async with gateway_client(backend) as (_, client):
        groups = [["a1", "a2", "a3", "a4"], ["b1", "b2", "b3", "b4"]]
        responses = await asyncio.gather(*[post(client, g) for g in groups])
        for response, texts in zip(responses, groups):
            assert response.status_code == 200
            body = response.json()
            assert [item["index"] for item in body["data"]] == list(range(4))
        assert backend.batch_sizes == [8]
        # A completed batch confirms the slot; no fault is raised.
        assert (await _fault(client))["active"] is False


# -------------------------------------------------------------------- recovery
@pytest.mark.asyncio
async def test_recovery_requires_the_active_fault_id() -> None:
    backend = FakeBackend(mode="truncated")
    async with gateway_client(backend) as (_, client):
        response = await post(client, ["lost"])
        fault_id = response.json()["error"]["fault_id"]

        stale = await _recover(client, "fault-000000000000")
        assert stale.status_code == 409
        assert stale.json()["error"]["code"] == "stale_recovery"
        assert stale.json()["error"]["current_fault_id"] == fault_id
        assert (await _fault(client))["active"] is True

        missing = await client.post("/admin/fault/recover", json={})
        assert missing.status_code == 422

        ok = await _recover(client, fault_id, operator="oncall", reason="idle")
        assert ok.status_code == 200, ok.text
        body = ok.json()
        assert body["recovered"] is True
        assert body["fault_id"] == fault_id
        assert (await _fault(client))["active"] is False

        backend.mode = "ok"
        assert (await post(client, ["works"])).status_code == 200

        # Replaying the same recovery cannot clear anything else.
        replay = await _recover(client, fault_id)
        assert replay.status_code == 409
        assert replay.json()["error"]["code"] == "no_active_fault"


@pytest.mark.asyncio
async def test_old_recovery_cannot_clear_a_newer_fault() -> None:
    backend = FakeBackend(mode="truncated")
    async with gateway_client(backend) as (_, client):
        first = await post(client, ["one"])
        old_fault = first.json()["error"]["fault_id"]

        # Recovery clears fault #1, then a new unconfirmed batch raises #2.
        assert (await _recover(client, old_fault)).status_code == 200
        backend.mode = "truncated"
        second = await post(client, ["two"])
        new_fault = second.json()["error"]["fault_id"]
        assert new_fault != old_fault

        stale = await _recover(client, old_fault)
        assert stale.status_code == 409
        assert stale.json()["error"]["code"] == "stale_recovery"
        assert (await _fault(client))["fault"]["fault_id"] == new_fault
        assert (await post(client, ["three"], timeout_ms=1000)).status_code == 503


@pytest.mark.asyncio
async def test_recovery_probe_refuses_while_backend_reports_busy() -> None:
    backend = FakeBackend(mode="truncated", slot_busy=True)
    async with gateway_client(backend, backend_admin_url="http://fake.test") as (
        _,
        client,
    ):
        response = await post(client, ["lost"])
        fault_id = response.json()["error"]["fault_id"]

        blocked = await _recover(client, fault_id)
        assert blocked.status_code == 409
        assert blocked.json()["error"]["code"] == "backend_not_confirmed"
        assert blocked.json()["error"]["reason"] == "backend_still_busy"
        assert (await _fault(client))["active"] is True

        backend.slot_busy = False
        ok = await _recover(client, fault_id)
        assert ok.status_code == 200
        assert ok.json()["backend_probe"]["checked"] is True


@pytest.mark.asyncio
async def test_recovery_probe_failure_keeps_the_fault() -> None:
    backend = FakeBackend(mode="truncated", slot_busy=False)
    async with gateway_client(backend, backend_admin_url="http://fake.test") as (
        _,
        client,
    ):
        response = await post(client, ["lost"])
        fault_id = response.json()["error"]["fault_id"]
        backend.mode = "slot_error"
        blocked = await _recover(client, fault_id)
        assert blocked.status_code == 409
        assert blocked.json()["error"]["code"] == "backend_not_confirmed"
        assert (await _fault(client))["active"] is True


# --------------------------------------------------------------- late results
@pytest.mark.asyncio
async def test_late_response_is_discarded_without_touching_the_new_slot() -> None:
    backend = FakeBackend(mode="complete_after", late_response_after=0.25)
    async with gateway_client(backend, backend_timeout_seconds=0.05) as (_, client):
        response = await post(client, ["late"], timeout_ms=5000)
        assert response.status_code == 502
        first_fault = response.json()["error"]["fault_id"]

        # Wait until the abandoned attempt has produced its real response.
        await asyncio.sleep(0.35)
        stats = (await client.get("/stats")).json()
        assert stats["late_results_dropped"] == 1
        # The late success must not have cleared the fault.
        status = await _fault(client)
        assert status["fault"]["fault_id"] == first_fault
        assert status["dispatch_paused"] is True
        assert len(backend.calls) == 1

        # Recover, then make the next batch fail unconfirmed as well.  The old
        # late callback must not release this new slot.
        assert (await _recover(client, first_fault)).status_code == 200
        backend.mode = "truncated"
        second = await post(client, ["two"])
        assert second.status_code == 502
        new_fault = second.json()["error"]["fault_id"]
        assert new_fault != first_fault
        assert (await _fault(client))["fault"]["fault_id"] == new_fault
        # Wait long enough for any stray callback from batch #1 to fire.
        await asyncio.sleep(0.4)
        assert (await _fault(client))["fault"]["fault_id"] == new_fault


@pytest.mark.asyncio
async def test_stale_late_callback_cannot_release_a_newer_batch_slot() -> None:
    """Directly exercise the generation/batch guard on the late callback."""
    backend = FakeBackend()
    settings = make_settings()
    client = httpx.AsyncClient(transport=httpx.MockTransport(backend))
    scheduler = BatchScheduler(settings, client)
    await scheduler.start()
    try:
        # Simulate a newer batch being in flight, then a callback from an
        # older batch arriving with a stale identity.
        scheduler._active_batch_id = "batch-99"
        scheduler._generation = 7
        scheduler._slot = {"batch_id": "batch-99", "confirmed": False}
        try:
            await scheduler._on_late_backend_result(
                asyncio.ensure_future(asyncio.sleep(0)), "batch-1", 6
            )
        except asyncio.CancelledError:  # pragma: no cover
            pass
        assert scheduler._active_batch_id == "batch-99"
        assert scheduler.stats()["late_results_dropped"] == 1
    finally:
        await scheduler.close()
        await client.aclose()


# ------------------------------------------------------- restart persistence
@pytest.mark.asyncio
async def test_unconfirmed_batch_survives_a_hard_restart(tmp_path) -> None:
    state_file = tmp_path / "gateway-state.json"
    backend = FakeBackend(mode="truncated")
    settings = make_settings(state_file=str(state_file))

    # First "process": dispatch a batch whose outcome is never confirmed.
    client = httpx.AsyncClient(transport=httpx.MockTransport(backend))
    scheduler = BatchScheduler(settings, client)
    await scheduler.start()
    loop = asyncio.get_running_loop()
    with pytest.raises(Exception) as caught:
        await scheduler.submit(
            request_id="r1",
            texts=["lost"],
            model="fake-model",
            deadline=loop.time() + 5,
        )
    assert getattr(caught.value, "code", "") == "backend_error"
    fault = scheduler.fault_info()["fault"]
    assert fault is not None
    await scheduler.close()
    await client.aclose()

    # Second "process": it must restore the quarantine, not assume idleness.
    backend2 = FakeBackend()
    client2 = httpx.AsyncClient(transport=httpx.MockTransport(backend2))
    scheduler2 = BatchScheduler(settings, client2)
    await scheduler2.start()
    try:
        info = scheduler2.fault_info()
        assert info["dispatch_paused"] is True
        assert info["fault"]["fault_id"] == fault["fault_id"]
        assert info["fault"]["restored_after_restart"] is True
        assert scheduler2.stats()["restored_faults"] == 1

        loop = asyncio.get_running_loop()
        with pytest.raises(Exception) as caught:
            await scheduler2.submit(
                request_id="r2",
                texts=["new"],
                model="fake-model",
                deadline=loop.time() + 5,
            )
        assert getattr(caught.value, "code", "") == "backend_unconfirmed"
        assert backend2.calls == []

        # The restored fault id is recoverable; an older/stale one is not.
        from gateway.app.errors import StaleRecovery

        with pytest.raises(StaleRecovery):
            await scheduler2.recover(fault_id="fault-ancient")
        restored = await scheduler2.recover(fault_id=fault["fault_id"])
        assert restored["recovered"] is True
        assert scheduler2.fault_info()["active"] is False
    finally:
        await scheduler2.close()
        await client2.aclose()


@pytest.mark.asyncio
async def test_restart_after_unconfirmed_batch_without_fault_record(tmp_path) -> None:
    """A crash mid-call leaves ``confirmed: false``; that alone must pause."""
    state_file = tmp_path / "state.json"
    state_file.write_text(
        json.dumps(
            {
                "version": 1,
                "slot": {
                    "batch_id": "batch-42",
                    "confirmed": False,
                    "dispatched_at": "2026-01-01T00:00:00Z",
                },
                "fault": None,
            }
        )
    )
    backend = FakeBackend()
    settings = make_settings(state_file=str(state_file))
    client = httpx.AsyncClient(transport=httpx.MockTransport(backend))
    scheduler = BatchScheduler(settings, client)
    await scheduler.start()
    try:
        info = scheduler.fault_info()
        assert info["dispatch_paused"] is True
        assert info["fault"]["reason"] == "gateway_restart_with_unconfirmed_batch"
        assert info["fault"]["batch_id"] == "batch-42"
    finally:
        await scheduler.close()
        await client.aclose()


@pytest.mark.asyncio
async def test_corrupt_state_file_fails_closed(tmp_path) -> None:
    state_file = tmp_path / "state.json"
    state_file.write_text("{not json")
    backend = FakeBackend()
    settings = make_settings(state_file=str(state_file))
    client = httpx.AsyncClient(transport=httpx.MockTransport(backend))
    scheduler = BatchScheduler(settings, client)
    await scheduler.start()
    try:
        info = scheduler.fault_info()
        assert info["dispatch_paused"] is True
        assert info["fault"]["reason"] == "state_file_unreadable"
    finally:
        await scheduler.close()
        await client.aclose()


@pytest.mark.asyncio
async def test_confirmed_slot_is_not_restored_as_a_fault(tmp_path) -> None:
    state_file = tmp_path / "state.json"
    state_file.write_text(
        json.dumps(
            {
                "version": 1,
                "slot": {"batch_id": "batch-1", "confirmed": True},
                "fault": None,
            }
        )
    )
    backend = FakeBackend()
    settings = make_settings(state_file=str(state_file))
    client = httpx.AsyncClient(transport=httpx.MockTransport(backend))
    scheduler = BatchScheduler(settings, client)
    await scheduler.start()
    try:
        assert scheduler.fault_info()["active"] is False
    finally:
        await scheduler.close()
        await client.aclose()


@pytest.mark.asyncio
async def test_state_file_is_written_before_dispatch_and_after_confirmation(
    tmp_path,
) -> None:
    state_file = tmp_path / "state.json"
    backend = FakeBackend()
    settings = make_settings(state_file=str(state_file))
    client = httpx.AsyncClient(transport=httpx.MockTransport(backend))
    scheduler = BatchScheduler(settings, client)
    await scheduler.start()
    try:
        loop = asyncio.get_running_loop()
        result = await scheduler.submit(
            request_id="r",
            texts=["x"],
            model="fake-model",
            deadline=loop.time() + 5,
        )
        assert len(result.vectors) == 1
        persisted = json.loads(state_file.read_text())
        assert persisted["slot"]["confirmed"] is True
        assert persisted["fault"] is None
    finally:
        await scheduler.close()
        await client.aclose()


@pytest.mark.asyncio
async def test_persistence_failure_refuses_to_dispatch(tmp_path) -> None:
    """If the unconfirmed record cannot be written, do not send the batch."""
    blocked = tmp_path / "state.json"
    blocked.mkdir()  # a directory at the state path makes the rename fail
    backend = FakeBackend()
    settings = make_settings(state_file=str(blocked))
    client = httpx.AsyncClient(transport=httpx.MockTransport(backend))
    scheduler = BatchScheduler(settings, client)
    await scheduler.start()
    try:
        # Startup already failed closed because the path is unreadable; clear
        # that in-memory fault so the dispatch path is exercised.
        startup = scheduler.fault_info()["fault"]
        assert startup["reason"] == "state_file_unreadable"
        await scheduler.recover(fault_id=startup["fault_id"])

        loop = asyncio.get_running_loop()
        with pytest.raises(Exception) as caught:
            await scheduler.submit(
                request_id="r",
                texts=["x"],
                model="fake-model",
                deadline=loop.time() + 5,
            )
        assert getattr(caught.value, "code", "") == "backend_error"
        # Fail closed: the batch never reached the backend, and dispatch is
        # paused with an explicit persistence fault.
        assert backend.calls == []
        info = scheduler.fault_info()
        assert info["dispatch_paused"] is True
        assert info["fault"]["reason"] == "state_not_durable"
    finally:
        await scheduler.close()
        await client.aclose()
