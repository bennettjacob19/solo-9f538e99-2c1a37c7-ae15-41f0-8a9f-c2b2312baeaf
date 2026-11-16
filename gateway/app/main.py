"""FastAPI entrypoint for the embedding batch gateway."""

from __future__ import annotations

import asyncio
import logging
import uuid
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from .batcher import BatchScheduler
from .config import Settings
from .errors import GatewayError, ValidationError
from .schemas import (
    EmbeddingItem,
    EmbeddingRequest,
    EmbeddingResponse,
    FaultStatusResponse,
    HealthResponse,
    RecoveryRequest,
    RecoveryResponse,
    StatsResponse,
)

logger = logging.getLogger("embedding_gateway")


class AcceptanceTimeMiddleware:
    """Stamp the receipt time before the body is parsed.

    The per-request deadline must start when the gateway accepts the call,
    not when the request reaches the scheduler.
    """

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
        if scope["type"] == "http":
            state = scope.setdefault("state", {})
            state.setdefault("accepted_at", asyncio.get_running_loop().time())
        await self.app(scope, receive, send)


def _error_response(status_code: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"error": {"code": code, "message": message}},
    )


def _gateway_error_response(exc: GatewayError) -> JSONResponse:
    error: dict[str, Any] = {"code": exc.code, "message": exc.message}
    error.update(exc.extra)
    return JSONResponse(status_code=exc.status_code, content={"error": error})


def create_app(
    settings: Settings | None = None,
    *,
    backend_transport: httpx.AsyncBaseTransport | None = None,
) -> FastAPI:
    settings = settings or Settings.from_env()
    settings.validate()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Per-call timeouts are set by the scheduler (it needs a longer client
        # timeout so a late response can still be observed and discarded).
        # The single-flight guarantee comes from the one dispatcher, not from
        # the connection limit: after a recovery a stale, still-running
        # attempt may briefly share the pool with the new batch, so allow a
        # couple of connections instead of deadlocking on one.
        client = httpx.AsyncClient(
            timeout=httpx.Timeout(
                settings.backend_timeout_seconds
                + settings.backend_late_response_grace_seconds,
                connect=settings.backend_connect_timeout_seconds,
            ),
            transport=backend_transport,
            limits=httpx.Limits(max_connections=4, max_keepalive_connections=2),
        )
        scheduler = BatchScheduler(settings, client)
        app.state.client = client
        app.state.scheduler = scheduler
        await scheduler.start()
        logger.info(
            "gateway ready: backend=%s window_ms=%s batch_max=%s state_file=%s",
            settings.backend_url,
            settings.batch_window_ms,
            settings.max_texts_per_batch,
            settings.state_file or "<in-memory>",
        )
        if scheduler.dispatch_paused:
            logger.error(
                "gateway started in quarantined state; requests will receive "
                "503 until POST /admin/fault/recover with the active fault_id"
            )
        try:
            yield
        finally:
            await scheduler.close()
            await client.aclose()

    app = FastAPI(
        title="Embedding Batch Gateway",
        version="1.1.0",
        description=(
            "Accepts 1-4 short texts per call, merges concurrent calls into "
            "single-flight backend batches of at most 8 texts, and enforces a "
            "deadline measured from request acceptance.\n\n"
            "If a backend batch's completion cannot be confirmed (read "
            "timeout, truncated response, lost connection) the gateway pauses "
            "dispatching and answers queued and new requests with "
            "`503 backend_unconfirmed`, including a `fault_id` and the "
            "reason. Inspect `GET /admin/fault`; after confirming the backend "
            "is idle, call `POST /admin/fault/recover` with that `fault_id`."
        ),
        lifespan=lifespan,
    )
    app.add_middleware(AcceptanceTimeMiddleware)
    app.state.settings = settings

    @app.exception_handler(GatewayError)
    async def gateway_error_handler(
        request: Request, exc: GatewayError
    ) -> JSONResponse:
        return _gateway_error_response(exc)

    @app.exception_handler(RequestValidationError)
    async def request_validation_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        errors = jsonable_encoder(exc.errors())
        first = errors[0] if errors else {}
        location = ".".join(str(part) for part in first.get("loc", ()))
        message = f"{location}: {first.get('msg', 'invalid request')}"
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "code": "validation_error",
                    "message": message.lstrip(": "),
                    "details": errors,
                }
            },
        )

    @app.get("/healthz", response_model=HealthResponse, tags=["ops"])
    async def healthz() -> HealthResponse:
        return HealthResponse()

    @app.get("/stats", response_model=StatsResponse, tags=["ops"])
    async def stats(request: Request) -> StatsResponse:
        return StatsResponse(**request.app.state.scheduler.stats())

    @app.get(
        "/admin/fault",
        response_model=FaultStatusResponse,
        tags=["faults"],
        summary="Inspect the slot quarantine (unconfirmed backend batch)",
    )
    async def fault_status(request: Request) -> FaultStatusResponse:
        return FaultStatusResponse(**request.app.state.scheduler.fault_info())

    @app.post(
        "/admin/fault/recover",
        response_model=RecoveryResponse,
        tags=["faults"],
        summary="Clear a fault after an operator confirmed the backend is idle",
        responses={
            409: {"description": "stale/unknown fault id or backend still busy"}
        },
    )
    async def fault_recover(
        payload: RecoveryRequest, request: Request
    ) -> RecoveryResponse:
        result = await request.app.state.scheduler.recover(
            fault_id=payload.fault_id,
            operator=payload.operator,
            reason=payload.reason,
        )
        return RecoveryResponse(**result)

    @app.post(
        "/v1/embeddings",
        response_model=EmbeddingResponse,
        tags=["embeddings"],
        summary="Embed 1-4 texts",
    )
    async def embeddings(
        payload: EmbeddingRequest, request: Request
    ) -> EmbeddingResponse:
        texts = payload.as_texts()
        if not 1 <= len(texts) <= settings.max_texts_per_request:
            raise ValidationError(
                f"input must contain between 1 and "
                f"{settings.max_texts_per_request} texts"
            )

        timeout_ms = (
            payload.timeout_ms
            if payload.timeout_ms is not None
            else settings.default_timeout_ms
        )
        if timeout_ms > settings.max_timeout_ms:
            raise ValidationError(f"timeout_ms must be <= {settings.max_timeout_ms}")

        accepted_at = request.state.accepted_at
        deadline = accepted_at + timeout_ms / 1000.0
        request_id = uuid.uuid4().hex
        model = payload.model or settings.backend_model

        result = await request.app.state.scheduler.submit(
            request_id=request_id,
            texts=texts,
            model=model,
            deadline=deadline,
            request=request,
        )
        return EmbeddingResponse(
            data=[
                EmbeddingItem(index=index, embedding=vector)
                for index, vector in enumerate(result.vectors)
            ],
            model=model,
            batch_id=result.batch_id,
            batch_size=result.batch_size,
        )

    return app


app = create_app()
