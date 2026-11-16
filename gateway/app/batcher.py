"""Single-flight batching scheduler.

Design notes
------------
* There is exactly one dispatcher task, so at most one backend batch is in
  flight.  A new batch is not started until the previous batch has been
  *confirmed* finished, i.e. a complete HTTP response or a definite
  pre-dispatch failure was observed.
* A telemetry timeout / lost response does **not** prove the backend finished.
  The batch may still be computing on the single slot.  Such an outcome puts
  the dispatcher into quarantine: queued and new requests end immediately with
  ``503 backend_unconfirmed`` carrying a fault id and the reason, and no new
  batch is sent until an operator confirms the backend is idle and recovers
  with that fault id.
* The quarantine and the last dispatch record are persisted atomically, so a
  gateway restart does not treat an unconfirmed batch as an idle slot.
* An explicit backend HTTP error, malformed-but-complete payload, or a
  connection that provably never reached the backend is a *confirmed* failure:
  callers get ``502 backend_error``, the slot is released, and service
  continues as before.
* Recovery and late responses carry identities (fault id, batch id, scheduler
  generation).  A stale recovery request cannot clear a newer fault, and a
  late callback from an older batch cannot release the new batch's slot.
* Pending requests are merged FIFO while the accumulated text count stays at
  or below ``max_texts_per_batch`` and the model matches; a request is never
  split.  Caller deadlines are measured from request acceptance.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import httpx

from .config import Settings
from .errors import (
    BackendError,
    BackendUncertain,
    BackendUnconfirmed,
    ClientDisconnected,
    GatewayError,
    NoActiveFault,
    RecoveryNotConfirmed,
    RequestTimeout,
    ServiceUnavailable,
    StaleRecovery,
)
from .state import StateStore, StateStoreError, now_iso

logger = logging.getLogger("embedding_gateway.batcher")


@dataclass
class PendingRequest:
    """A single gateway call waiting for (or riding along) a backend batch."""

    request_id: str
    texts: list[str]
    model: str
    deadline: float  # loop.time() monotonic deadline
    enqueued_at: float
    future: asyncio.Future[list[list[float]]] = field(
        default_factory=lambda: asyncio.get_running_loop().create_future()
    )
    # pending -> dispatched -> finished | cancelled
    state: str = "pending"
    batch_id: str | None = None
    batch_size: int | None = None


@dataclass
class BatchResult:
    """Vectors for one caller plus the batch they were served from."""

    vectors: list[list[float]]
    batch_id: str
    batch_size: int


@dataclass
class BatchRecord:
    batch_id: str
    texts: int
    requests: int
    started_at: float
    finished_at: float | None = None
    outcome: str | None = None
    error: str | None = None


class BatchScheduler:
    def __init__(self, settings: Settings, client: httpx.AsyncClient) -> None:
        self.settings = settings
        self.client = client
        self._queue: deque[PendingRequest] = deque()
        self._wake = asyncio.Event()
        self._closing = False
        self._dispatcher: asyncio.Task[None] | None = None
        self._in_flight = False
        self._batch_counter = 0
        self._stats: dict[str, int] = {
            "accepted_requests": 0,
            "completed_requests": 0,
            "timed_out_requests": 0,
            "disconnected_requests": 0,
            "failed_requests": 0,
            "batches_dispatched": 0,
            "backend_errors": 0,
            "backend_texts_sent": 0,
            "unconfirmed_errors": 0,
            "rejected_requests": 0,
            "faults_created": 0,
            "recoveries": 0,
            "restored_faults": 0,
            "late_results_dropped": 0,
        }
        self._last_batch: dict[str, Any] | None = None

        # Slot authority -------------------------------------------------
        # ``_active_batch_id`` is the batch whose outcome the dispatcher is
        # currently waiting for.  ``_slot`` is the durable record of the last
        # dispatch (confirmed or not).  ``_quarantine`` is the active fault.
        self._active_batch_id: str | None = None
        self._slot: dict[str, Any] | None = None
        self._quarantine: dict[str, Any] | None = None
        self._generation = 0
        self._store = StateStore(settings.state_file)
        # Backend attempts that outlived the gateway read timeout.  They are
        # kept only to observe (and discard) a late answer; their identity is
        # checked before touching any state.
        self._background_calls: set[asyncio.Task[Any]] = set()

    # ------------------------------------------------------------------ life
    async def start(self) -> None:
        if self._dispatcher is None or self._dispatcher.done():
            self._restore_state()
            self._dispatcher = asyncio.create_task(
                self._dispatch_loop(), name="embedding-batch-dispatcher"
            )

    async def close(self) -> None:
        self._closing = True
        self._wake.set()
        dispatcher = self._dispatcher
        self._dispatcher = None
        if dispatcher is not None:
            try:
                await dispatcher
            except asyncio.CancelledError:  # pragma: no cover - safety net
                pass
        # Outstanding late-response observers are no longer needed: after a
        # restart the persisted unconfirmed slot is the source of truth.
        for task in list(self._background_calls):
            task.cancel()
        self._background_calls.clear()
        # Anything still queued never got a chance to be sent.
        while self._queue:
            item = self._queue.popleft()
            if item.state == "pending":
                item.state = "cancelled"
            if not item.future.done():
                item.future.set_exception(ServiceUnavailable("gateway shutting down"))

    # ---------------------------------------------------------- slot state
    @property
    def dispatch_paused(self) -> bool:
        return self._quarantine is not None

    def fault_info(self) -> dict[str, Any]:
        return {
            "active": self._quarantine is not None,
            "dispatch_paused": self.dispatch_paused,
            "fault": dict(self._quarantine) if self._quarantine else None,
            "persistence_enabled": self._store.enabled,
        }

    def _persist(self, *, fail_closed: bool = False) -> None:
        """Write the slot/fault record atomically.

        A persistence failure during normal operation is logged but does not
        fail a request.  With ``fail_closed=True`` (used before dispatching) an
        unwritable state file stops the dispatch instead, because a later
        restart could otherwise mistake the batch for an idle slot.
        """
        payload = {
            "version": 1,
            "updated_at": now_iso(),
            "slot": self._slot,
            "fault": self._quarantine,
        }
        try:
            self._store.save(payload)
        except OSError as exc:
            logger.error("could not persist scheduler state: %s", exc)
            if fail_closed:
                raise

    def _restore_state(self) -> None:
        try:
            data = self._store.load()
        except StateStoreError as exc:
            # Fail closed: an unreadable file might have contained an active
            # unconfirmed-batch record.  Do not assume the slot is idle.
            logger.error("%s; pausing dispatch until operator recovery", exc)
            self._enter_quarantine(
                reason="state_file_unreadable",
                detail=str(exc),
                batch_id=None,
                source="restart",
            )
            return
        if data is None:
            return

        fault = data.get("fault")
        if isinstance(fault, dict) and fault.get("fault_id"):
            restored = dict(fault)
            restored["restored_after_restart"] = True
            restored["restored_at"] = now_iso()
            self._quarantine = restored
            self._generation += 1
            self._stats["restored_faults"] += 1
            slot = data.get("slot")
            if isinstance(slot, dict):
                self._slot = slot
            logger.warning(
                "restored active fault %s (%s); dispatch stays paused",
                restored.get("fault_id"),
                restored.get("reason"),
            )
            self._persist()
            return

        slot = data.get("slot")
        if isinstance(slot, dict) and slot.get("confirmed") is False:
            batch_id = slot.get("batch_id")
            logger.warning(
                "previous process stopped with unconfirmed batch %s; "
                "pausing dispatch after restart",
                batch_id,
            )
            self._slot = slot
            self._enter_quarantine(
                reason="gateway_restart_with_unconfirmed_batch",
                detail=(
                    "the gateway restarted while a backend batch had not been "
                    "confirmed complete"
                ),
                batch_id=batch_id if isinstance(batch_id, str) else None,
                source="restart",
            )

    def _enter_quarantine(
        self,
        *,
        reason: str,
        detail: str,
        batch_id: str | None,
        source: str,
    ) -> dict[str, Any]:
        fault_id = self._new_fault_id()
        fault = {
            "fault_id": fault_id,
            "reason": reason,
            "detail": detail,
            "batch_id": batch_id,
            "source": source,
            "created_at": now_iso(),
            "restored_after_restart": False,
        }
        self._quarantine = fault
        self._generation += 1
        self._active_batch_id = None
        self._stats["faults_created"] += 1
        if self._slot is None:
            self._slot = {"batch_id": batch_id, "confirmed": False}
        else:
            self._slot = {**self._slot, "confirmed": False}
        logger.error(
            "dispatch paused: fault %s reason=%s batch=%s: %s",
            fault_id,
            reason,
            batch_id,
            detail,
        )
        # The callers of the failed batch receive their own fan-out error
        # first (the caller of _enter_quarantine does that); everything still
        # queued is rejected here.
        self._persist()
        self._reject_queued()
        self._wake.set()
        return fault

    @staticmethod
    def _new_fault_id() -> str:
        return f"fault-{uuid.uuid4().hex[:12]}"

    def _fault_error(self) -> BackendUnconfirmed:
        fault = self._quarantine or {}
        fault_id = str(fault.get("fault_id") or "fault-unknown")
        reason = str(fault.get("reason") or "backend_outcome_unconfirmed")
        batch_id = fault.get("batch_id")
        return BackendUnconfirmed(
            (
                f"dispatching paused after an unconfirmed backend batch "
                f"({reason}); fault_id={fault_id}"
            ),
            fault_id=fault_id,
            reason=reason,
            batch_id=str(batch_id) if batch_id else None,
        )

    def _reject_queued(self) -> None:
        """End every queued waiter with the current fault (503)."""
        while self._queue:
            item = self._queue.popleft()
            if item.state != "pending":
                continue
            item.state = "cancelled"
            if not item.future.done():
                item.future.set_exception(self._fault_error())
                self._stats["rejected_requests"] += 1

    async def recover(
        self,
        *,
        fault_id: str,
        operator: str | None = None,
        reason: str | None = None,
    ) -> dict[str, Any]:
        """Clear a quarantine after the operator confirmed the backend idle.

        The caller must supply the id of the *currently active* fault.  A
        recovery request for an older fault is rejected with 409 and cannot
        release a newer fault.  When a backend admin URL is configured, the
        gateway additionally probes the backend and refuses to recover while
        it still reports work in progress.
        """
        fault = self._quarantine
        if fault is None:
            raise NoActiveFault("no active backend fault to recover")
        if fault_id != fault.get("fault_id"):
            raise StaleRecovery(
                (
                    f"fault {fault_id} is not the active fault "
                    f"({fault.get('fault_id')}); refusing stale recovery"
                ),
                current_fault_id=fault.get("fault_id"),
                current_fault=dict(fault),
            )
        generation_at_entry = self._generation
        probe = await self._probe_backend_idle(fault["fault_id"])
        # Re-check after the await: another recovery may have completed and a
        # newer fault may have been raised in the meantime.
        if self._generation != generation_at_entry or self._quarantine is not fault:
            raise StaleRecovery(
                "gateway state changed while this recovery was being checked",
                current_fault_id=(
                    self._quarantine.get("fault_id") if self._quarantine else None
                ),
                current_fault=(dict(self._quarantine) if self._quarantine else None),
            )

        previous = dict(fault)
        self._quarantine = None
        self._generation += 1
        self._slot = {
            "batch_id": None,
            "confirmed": True,
            "confirmed_at": now_iso(),
            "outcome": "operator_recovery",
        }
        self._stats["recoveries"] += 1
        self._persist()
        self._wake.set()
        logger.warning(
            "fault %s recovered by operator=%s after backend idle confirmation",
            fault_id,
            operator or "<unspecified>",
        )
        return {
            "recovered": True,
            "fault_id": fault_id,
            "previous_fault": previous,
            "backend_probe": probe,
            "operator": operator,
            "reason": reason,
            "recovered_at": now_iso(),
        }

    async def _probe_backend_idle(self, fault_id: str) -> dict[str, Any]:
        admin_url = (self.settings.backend_admin_url or "").rstrip("/")
        if not admin_url:
            return {"configured": False, "checked": False, "idle": True}
        url = f"{admin_url}/slot"
        timeout = self.settings.recovery_probe_timeout_seconds
        try:
            response = await self.client.get(url, timeout=timeout)
        except httpx.HTTPError as exc:
            raise RecoveryNotConfirmed(
                (
                    "cannot confirm the backend is idle; fault "
                    f"{fault_id} remains active: {exc}"
                ),
                fault_id=fault_id,
                reason="backend_probe_failed",
                detail=str(exc),
            ) from exc
        if response.status_code != 200:
            raise RecoveryNotConfirmed(
                (
                    "cannot confirm the backend is idle; fault "
                    f"{fault_id} remains active: probe returned "
                    f"HTTP {response.status_code}"
                ),
                fault_id=fault_id,
                reason="backend_probe_failed",
                detail=response.text[:500],
            )
        try:
            body = response.json()
        except ValueError as exc:
            raise RecoveryNotConfirmed(
                "cannot confirm the backend is idle: probe returned invalid JSON",
                fault_id=fault_id,
                reason="backend_probe_failed",
                detail="invalid JSON",
            ) from exc
        busy = body.get("busy") if isinstance(body, dict) else None
        if busy is not False:
            raise RecoveryNotConfirmed(
                (
                    "backend still reports work in progress; fault "
                    f"{fault_id} remains active"
                ),
                fault_id=fault_id,
                reason="backend_still_busy",
                detail=body if isinstance(body, dict) else None,
            )
        return {"configured": True, "checked": True, "idle": True, "backend": body}

    # --------------------------------------------------------------- submit
    async def submit(
        self,
        *,
        request_id: str,
        texts: list[str],
        model: str,
        deadline: float,
        request: Any | None = None,
    ) -> BatchResult:
        """Queue ``texts`` and wait until the batch result or a failure.

        ``deadline`` is an absolute ``loop.time()`` value computed from the
        moment the HTTP request was accepted.
        """
        loop = asyncio.get_running_loop()
        self._stats["accepted_requests"] += 1

        if self._closing:
            raise ServiceUnavailable("gateway is shutting down")
        if self.dispatch_paused:
            self._stats["rejected_requests"] += 1
            raise self._fault_error()

        remaining = deadline - loop.time()
        if remaining <= 0:
            self._stats["timed_out_requests"] += 1
            raise RequestTimeout("request deadline elapsed before queueing")

        item = PendingRequest(
            request_id=request_id,
            texts=texts,
            model=model,
            deadline=deadline,
            enqueued_at=loop.time(),
        )
        self._queue.append(item)
        self._wake.set()

        result_task = asyncio.ensure_future(item.future)
        disconnect_task: asyncio.Task[bool] | None = None
        if request is not None:
            disconnect_task = asyncio.create_task(
                self._watch_disconnect(request), name=f"disconnect-{request_id}"
            )

        waiters: set[asyncio.Future[Any]] = {result_task}
        if disconnect_task is not None:
            waiters.add(disconnect_task)

        try:
            while True:
                remaining = deadline - loop.time()
                done, _ = await asyncio.wait(
                    waiters,
                    timeout=max(0.0, remaining),
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if disconnect_task is not None and disconnect_task in done:
                    if disconnect_task.result():
                        self._abandon(item, "disconnected")
                        self._stats["disconnected_requests"] += 1
                        raise ClientDisconnected("client disconnected before result")
                    # The disconnect probe gave up without observing a real
                    # disconnect; stop waiting on it.
                    waiters.discard(disconnect_task)
                    continue
                if result_task in done:
                    if result_task.cancelled():
                        # The dispatcher cancelled the future when it noticed
                        # the deadline had passed (an expired request is
                        # never sent).  Surface it as a timeout, not as an
                        # outer cancellation.
                        self._stats["timed_out_requests"] += 1
                        raise RequestTimeout("request deadline exceeded while waiting")
                    return BatchResult(
                        vectors=result_task.result(),
                        batch_id=item.batch_id or "batch-0",
                        batch_size=item.batch_size or len(texts),
                    )
                # Neither a result nor a confirmed disconnect: deadline hit.
                self._abandon(item, "timed out")
                self._stats["timed_out_requests"] += 1
                raise RequestTimeout("request deadline exceeded while waiting")
        except asyncio.CancelledError:
            # The ASGI server cancelled the handler (client vanished or
            # shutdown).  Remove the item if it is still pending; a dispatched
            # item is marked abandoned so its late vectors are dropped.
            was_dispatched = item.state == "dispatched"
            self._abandon(item, "cancelled")
            if was_dispatched:
                self._stats["disconnected_requests"] += 1
            raise
        finally:
            for waiter in waiters:
                if not waiter.done():
                    waiter.cancel()
            if item.future.done() and not item.future.cancelled():
                # Mark a backend exception as retrieved so asyncio does not
                # log "exception was never retrieved" when the caller left
                # at the same moment the batch failed.
                item.future.exception()
            elif not item.future.done():
                item.future.cancel()

    async def _watch_disconnect(self, request: Any) -> bool:
        """Poll until the client is gone.

        Returns ``True`` for an observed disconnect and ``False`` when the
        server cannot observe disconnects at all, in which case :meth:`submit`
        stops waiting on this probe.
        """
        interval = self.settings.disconnect_poll_interval_ms / 1000.0
        while True:
            try:
                disconnected = await request.is_disconnected()
            except Exception:  # noqa: BLE001 - third-party ASGI implementations
                # This server cannot observe disconnects; stop probing rather
                # than reporting a false positive.  The deadline still bounds
                # how long the caller can wait.
                logger.debug("disconnect polling unavailable for this server")
                return False
            if disconnected:
                return True
            await asyncio.sleep(interval)

    def _abandon(self, item: PendingRequest, reason: str) -> None:
        """Best-effort, synchronous dequeue of a caller that gave up.

        No ``await`` happens in this method, so in a single-threaded event
        loop it is atomic with respect to the dispatcher's dequeue/mark logic.
        """
        if item.state == "pending":
            try:
                self._queue.remove(item)
            except ValueError:
                pass
            item.state = "cancelled"
            logger.debug(
                "request %s abandoned while pending: %s", item.request_id, reason
            )
        elif item.state == "dispatched":
            # The batch is already on the wire.  Mark it so the dispatcher
            # discards the late vectors; it must still await the response.
            item.state = "cancelled"
            logger.debug(
                "request %s abandoned while in flight: %s", item.request_id, reason
            )
        if not item.future.done():
            item.future.cancel()

    # ------------------------------------------------------------ dispatch
    async def _dispatch_loop(self) -> None:
        loop = asyncio.get_running_loop()
        while not self._closing:
            if self.dispatch_paused:
                # Quarantine: no batch may be sent even if callers arrive.
                # `_reject_queued` (on entering the fault) already ended any
                # waiter and submit() rejects new ones; park until recovery
                # or shutdown sets the event.
                if self._queue:
                    self._reject_queued()
                self._wake.clear()
                await self._wake.wait()
                continue

            if not self._queue:
                self._wake.clear()
                if not self._queue:
                    await self._wake.wait()
                continue

            batch = await self._collect_batch(loop)
            if not batch:
                # Everything in the queue expired or was abandoned between
                # wake-up and collection; yield briefly instead of spinning.
                await asyncio.sleep(0.001)
                continue
            if self.dispatch_paused:
                # A fault may have been raised while collecting (it cannot be
                # raised by this task, but recovery/fault bookkeeping is kept
                # defensive).  Never dispatch into a quarantined slot.
                continue
            await self._send_batch(batch, loop)

    async def _collect_batch(
        self, loop: asyncio.AbstractEventLoop
    ) -> list[PendingRequest]:
        batch: list[PendingRequest] = []
        total = 0
        max_batch = self.settings.max_texts_per_batch
        window_end: float | None = None
        model: str | None = None

        while True:
            # Expired/abandoned entries are removed wherever they sit so they
            # are never sent and never hold a backend slot.
            while self._queue:
                head = self._queue[0]
                if head.state == "pending" and loop.time() < head.deadline:
                    break
                self._queue.popleft()
                if head.state == "pending":
                    head.state = "cancelled"
                    if not head.future.done():
                        head.future.cancel()

            if not self._queue:
                if not batch:
                    return []
                if total >= max_batch:
                    break
                # Queue drained before the batch is full.  If the merge window
                # opened by the batch's first request is still running, wait
                # for late arrivals; a backlogged batch (window already
                # elapsed while the backend was busy) ships immediately.
                remaining = (window_end or 0.0) - loop.time()
                if remaining <= 0:
                    break
                self._wake.clear()
                if self._queue:
                    continue
                try:
                    await asyncio.wait_for(self._wake.wait(), timeout=remaining)
                except TimeoutError:
                    pass
                continue

            head = self._queue[0]
            if batch:
                if head.model != model:
                    break
                if total + len(head.texts) > max_batch:
                    # A single request is never split; leave it for the next
                    # batch instead of exceeding the backend limit.
                    break
            else:
                # The window belongs to the first request actually collected.
                model = head.model
                window_end = head.enqueued_at + (self.settings.batch_window_ms / 1000.0)

            self._queue.popleft()
            head.state = "dispatched"
            batch.append(head)
            total += len(head.texts)

            if total >= max_batch:
                break

        return batch

    async def _send_batch(
        self,
        batch: list[PendingRequest],
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        self._batch_counter += 1
        batch_id = f"batch-{self._batch_counter}"
        # Drop anything that expired or disconnected between dequeue and this
        # point: such a request was never sent, so it must not consume backend
        # capacity or appear in the payload.
        now = loop.time()
        live: list[PendingRequest] = []
        for item in batch:
            if item.state != "dispatched":
                continue
            if now >= item.deadline:
                item.state = "cancelled"
                if not item.future.done():
                    item.future.cancel()
                continue
            live.append(item)
        batch = live
        if not batch:
            return
        flat_inputs = [text for item in batch for text in item.texts]
        model = batch[0].model
        record = BatchRecord(
            batch_id=batch_id,
            texts=len(flat_inputs),
            requests=len(batch),
            started_at=loop.time(),
        )
        self._in_flight = True
        generation = self._generation
        self._active_batch_id = batch_id
        self._slot = {
            "batch_id": batch_id,
            "texts": len(flat_inputs),
            "requests": len(batch),
            "dispatched_at": now_iso(),
            "confirmed": False,
        }
        try:
            # The unconfirmed record must be durable before the batch is sent,
            # otherwise a crash could look like an idle slot after restart.
            self._persist(fail_closed=True)
        except OSError as exc:
            self._in_flight = False
            self._active_batch_id = None
            error = BackendError(
                f"cannot persist dispatch record, refusing to send batch: {exc}"
            )
            self._fail_batch(batch, error)
            self._enter_quarantine(
                reason="state_not_durable",
                detail=str(exc),
                batch_id=batch_id,
                source="dispatch",
            )
            return
        self._stats["batches_dispatched"] += 1
        self._stats["backend_texts_sent"] += len(flat_inputs)

        try:
            try:
                vectors = await self._call_backend(
                    model, flat_inputs, batch_id, generation
                )
            except BackendUncertain as exc:
                # The backend may have accepted the work; we cannot prove it
                # finished.  Callers still get 502 (the timeout contract is
                # unchanged), but the slot stays quarantined.  Any exception
                # that escapes this handling (e.g. a persistence error) is
                # deliberately treated as unconfirmed by the generic branch.
                self._stats["backend_errors"] += 1
                self._stats["unconfirmed_errors"] += 1
                fault = self._enter_quarantine(
                    reason="backend_unconfirmed",
                    detail=exc.message,
                    batch_id=batch_id,
                    source="dispatch",
                )
                error = BackendError(
                    exc.message,
                    fault_id=fault["fault_id"],
                    reason=fault["reason"],
                    batch_id=batch_id,
                )
                record.outcome = "unconfirmed"
                record.error = exc.message
                self._fail_batch(batch, error)
            except BackendError as exc:
                # A complete error response (or a failure that provably never
                # reached the backend) confirms the slot is free again.
                self._stats["backend_errors"] += 1
                record.outcome = "error"
                record.error = exc.message
                if self._batch_is_current(batch_id, generation):
                    self._fail_batch(batch, exc)
                    self._confirm_slot(batch_id, outcome="error")
            except Exception as exc:  # noqa: BLE001 - must never kill dispatcher
                # Unknown failure between dispatch and confirmation: fail
                # closed.  Never assume the single slot was released.
                message = f"backend batch failed: {exc}"
                self._stats["backend_errors"] += 1
                self._stats["unconfirmed_errors"] += 1
                fault = self._enter_quarantine(
                    reason="backend_unconfirmed",
                    detail=message,
                    batch_id=batch_id,
                    source="dispatch",
                )
                error = BackendError(
                    message,
                    fault_id=fault["fault_id"],
                    reason=fault["reason"],
                    batch_id=batch_id,
                )
                record.outcome = "unconfirmed"
                record.error = message
                self._fail_batch(batch, error)
            else:
                if not self._batch_is_current(batch_id, generation):
                    # A late callback from an older batch must not release a
                    # newer batch's slot or resolve anything.
                    self._stats["late_results_dropped"] += 1
                    record.outcome = "late_dropped"
                    logger.warning(
                        "dropping late result for batch %s (generation %s, current %s)",
                        batch_id,
                        generation,
                        self._generation,
                    )
                else:
                    try:
                        per_request = self._split_vectors(batch, vectors)
                    except BackendError as exc:
                        record.outcome = "error"
                        record.error = exc.message
                        self._stats["backend_errors"] += 1
                        self._fail_batch(batch, exc)
                        self._confirm_slot(batch_id, outcome="error")
                    else:
                        for item, item_vectors in zip(batch, per_request):
                            item.batch_id = batch_id
                            item.batch_size = len(flat_inputs)
                            if item.state == "dispatched" and not item.future.done():
                                item.state = "finished"
                                item.future.set_result(item_vectors)
                                self._stats["completed_requests"] += 1
                            elif item.state == "dispatched":
                                item.state = "cancelled"
                        record.outcome = "ok"
                        self._confirm_slot(batch_id, outcome="ok")
        finally:
            self._in_flight = False
            if self._active_batch_id == batch_id:
                self._active_batch_id = None
            record.finished_at = loop.time()
            self._last_batch = {
                "batch_id": record.batch_id,
                "texts": record.texts,
                "requests": record.requests,
                "duration_ms": round(
                    (record.finished_at - record.started_at) * 1000, 2
                ),
                "outcome": record.outcome,
                "error": record.error,
            }
            # Wake the loop in case submissions arrived during the call; the
            # dispatch loop would otherwise still be parked on the event.
            self._wake.set()

        # A caller may have timed out between the response arriving and the
        # future being resolved; that is handled by the state check above.

    def _batch_is_current(self, batch_id: str, generation: int) -> bool:
        return self._active_batch_id == batch_id and self._generation == generation

    def _confirm_slot(self, batch_id: str, *, outcome: str) -> None:
        if self._active_batch_id != batch_id:
            return
        self._active_batch_id = None
        self._slot = {
            "batch_id": batch_id,
            "confirmed": True,
            "confirmed_at": now_iso(),
            "outcome": outcome,
        }
        self._persist()

    def _fail_batch(self, batch: list[PendingRequest], error: BackendError) -> None:
        for item in batch:
            if item.state == "dispatched":
                item.state = "cancelled"
                if not item.future.done():
                    item.future.set_exception(error)
                self._stats["failed_requests"] += 1
            elif item.state == "finished":
                continue

    async def _call_backend(
        self,
        model: str,
        inputs: list[str],
        batch_id: str,
        generation: int,
    ) -> list[dict[str, Any]]:
        """POST one batch, returning its data — or raise.

        The HTTP read timeout is enforced here (``backend_timeout_seconds``)
        rather than relying only on the client timeout, so the error message
        can state that the outcome is unconfirmed.  The underlying request is
        allowed to continue in the background for a bounded grace period; if
        it eventually completes, :meth:`_on_late_backend_result` discards the
        result unless it still belongs to the current batch generation.
        """
        headers = {"content-type": "application/json"}
        if self.settings.backend_api_key:
            headers["authorization"] = f"Bearer {self.settings.backend_api_key}"
        payload = {"model": model, "input": inputs}
        timeout = httpx.Timeout(
            self.settings.backend_timeout_seconds
            + self.settings.backend_late_response_grace_seconds,
            connect=self.settings.backend_connect_timeout_seconds,
        )
        attempt = asyncio.create_task(
            self.client.post(
                self.settings.backend_url,
                json=payload,
                headers=headers,
                timeout=timeout,
            ),
            name=f"backend-{batch_id}",
        )
        try:
            response = await asyncio.wait_for(
                asyncio.shield(attempt),
                timeout=self.settings.backend_timeout_seconds,
            )
        except TimeoutError as exc:
            # The request may have reached the backend and may still be
            # computing.  Keep observing it, but do not hold the slot open on
            # the gateway's side: quarantine immediately.
            self._track_late_call(attempt, batch_id, generation)
            raise BackendUncertain(
                "backend read timeout: the batch may have been accepted but "
                "its response was not received within "
                f"{self.settings.backend_timeout_seconds:g}s"
            ) from exc
        except asyncio.CancelledError:
            attempt.cancel()
            raise
        except Exception as exc:  # noqa: BLE001 - classified below
            # The HTTP attempt failed before the gateway's timeout fired.  The
            # classification below is the same, but the message must say that
            # no complete response was received.
            self._raise_backend_failure(exc, timed_out=False)

        # ``wait_for(shield(...))`` returned, so the attempt is complete (or
        # failed); surface its error while preserving the confirmed/uncertain
        # classification.
        try:
            response = attempt.result()
        except Exception as exc:  # noqa: BLE001 - re-raised below
            self._raise_backend_failure(exc, timed_out=True)

        if response.status_code < 200 or response.status_code >= 300:
            detail = response.text[:500].replace("\n", " ")
            raise BackendError(
                f"backend returned HTTP {response.status_code}: {detail}"
            )
        try:
            body = response.json()
        except ValueError as exc:
            raise BackendError("backend returned invalid JSON") from exc

        data = body.get("data") if isinstance(body, dict) else None
        if not isinstance(data, list):
            raise BackendError("backend response is missing a 'data' list")
        return data

    @staticmethod
    def _raise_backend_failure(exc: Exception, *, timed_out: bool) -> None:
        """Classify a failed backend attempt as confirmed or unconfirmed."""
        if isinstance(
            exc, (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout)
        ):
            # Provably no complete request was accepted by the backend, so the
            # slot is free and normal 502 fan-out applies.
            raise BackendError(f"backend connection failed: {exc}") from exc
        if isinstance(exc, httpx.ReadTimeout):
            raise BackendUncertain(
                "backend read timeout: the batch may have been accepted but "
                "no complete response was received"
            ) from exc
        if isinstance(exc, httpx.HTTPError):
            # Remote-protocol errors, write timeouts and truncated/garbled
            # responses: the request may have been fully processed even though
            # the answer was lost.
            raise BackendUncertain(
                "backend response was not received completely: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        raise BackendUncertain(
            f"backend attempt failed with an unconfirmed outcome: {exc}"
        ) from exc

    def _track_late_call(
        self,
        attempt: asyncio.Task[Any],
        batch_id: str,
        generation: int,
    ) -> None:
        """Observe a timed-out attempt so a genuinely late answer is seen."""
        task = asyncio.create_task(
            self._on_late_backend_result(attempt, batch_id, generation),
            name=f"late-{batch_id}",
        )
        self._background_calls.add(task)
        task.add_done_callback(self._background_calls.discard)

    async def _on_late_backend_result(
        self,
        attempt: asyncio.Task[Any],
        batch_id: str,
        generation: int,
    ) -> None:
        try:
            await attempt
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - observation only
            logger.warning(
                "late backend attempt for %s (generation %s) ended with %s: %s",
                batch_id,
                generation,
                type(exc).__name__,
                exc,
            )
            return
        self._stats["late_results_dropped"] += 1
        if self._generation != generation or (
            self._quarantine is not None
            and self._quarantine.get("batch_id") not in (None, batch_id)
        ):
            logger.warning(
                "dropped late result for old batch %s (generation %s, "
                "current %s); it cannot release a newer slot",
                batch_id,
                generation,
                self._generation,
            )
            return
        logger.warning(
            "received a late response for timed-out batch %s; it is discarded "
            "and the fault must still be cleared by an operator",
            batch_id,
        )

    def _split_vectors(
        self, batch: list[PendingRequest], data: list[dict[str, Any]]
    ) -> list[list[list[float]]]:
        total = sum(len(item.texts) for item in batch)
        if len(data) != total:
            raise BackendError(
                f"backend returned {len(data)} embeddings for {total} inputs"
            )

        ordered: list[list[float] | None] = [None] * total
        for position, entry in enumerate(data):
            if not isinstance(entry, dict):
                raise BackendError("backend embedding entries must be objects")
            index = entry.get("index", position)
            if (
                not isinstance(index, int)
                or isinstance(index, bool)
                or not 0 <= index < total
            ):
                raise BackendError(f"backend returned invalid index {index!r}")
            if ordered[index] is not None:
                raise BackendError(f"backend returned duplicate index {index}")
            embedding = entry.get("embedding")
            if not isinstance(embedding, list) or not all(
                isinstance(value, (int, float)) and not isinstance(value, bool)
                for value in embedding
            ):
                raise BackendError("backend embedding is not a list of numbers")
            ordered[index] = [float(value) for value in embedding]

        if any(vector is None for vector in ordered):
            raise BackendError("backend response is missing embedding indices")

        result: list[list[list[float]]] = []
        cursor = 0
        for item in batch:
            count = len(item.texts)
            chunk = ordered[cursor : cursor + count]
            # ``ordered`` was validated above, so no entry is None here.
            result.append([vector for vector in chunk if vector is not None])
            cursor += count
        return result

    @staticmethod
    def _as_backend_error(exc: Exception) -> BackendError:
        if isinstance(exc, BackendError):
            return exc
        if isinstance(exc, GatewayError):
            return BackendError(exc.message)
        return BackendError(f"backend batch failed: {exc}")

    # --------------------------------------------------------------- status
    def stats(self) -> dict[str, Any]:
        snapshot: dict[str, Any] = dict(self._stats)
        snapshot["pending_requests"] = sum(
            1 for item in self._queue if item.state == "pending"
        )
        snapshot["in_flight"] = self._in_flight
        snapshot["dispatch_paused"] = self.dispatch_paused
        snapshot["fault"] = dict(self._quarantine) if self._quarantine else None
        snapshot["persistence_enabled"] = self._store.enabled
        snapshot["last_batch"] = self._last_batch
        return snapshot
