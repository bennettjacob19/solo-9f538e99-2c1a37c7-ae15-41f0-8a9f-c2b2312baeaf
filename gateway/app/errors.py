"""Errors shared by the gateway request and batching layers."""

from __future__ import annotations

from typing import Any


class GatewayError(Exception):
    """Base class for failures that can be mapped to an HTTP response."""

    status_code = 500
    code = "internal_error"

    def __init__(self, message: str, **extra: Any) -> None:
        super().__init__(message)
        self.message = message
        # Extra machine-readable fields merged into the ``error`` object.
        self.extra: dict[str, Any] = {
            key: value for key, value in extra.items() if value is not None
        }


class RequestTimeout(GatewayError):
    """The request deadline (measured from acceptance) expired."""

    status_code = 504
    code = "request_timeout"


class ClientDisconnected(GatewayError):
    """The HTTP client went away before its embedding was delivered."""

    status_code = 499
    code = "client_disconnected"


class BackendError(GatewayError):
    """The single-slot embedding backend failed or returned invalid data."""

    status_code = 502
    code = "backend_error"


class BackendUncertain(BackendError):
    """The backend may have accepted the batch but its answer was lost.

    Callers of that batch still receive ``502 backend_error`` (the deadline /
    timeout contract is unchanged).  The scheduler uses this marker to stop
    dispatching: unlike an explicit backend error response, the outcome of the
    batch is unknown, so the single slot may still be occupied.
    """


class ServiceUnavailable(GatewayError):
    """The gateway is shutting down and cannot serve this request."""

    status_code = 503
    code = "service_unavailable"


class BackendUnconfirmed(ServiceUnavailable):
    """Dispatching is paused because a previous batch was never confirmed.

    Every new or already queued request is answered with ``503`` carrying the
    fault identifier and the reason, so operators can correlate the outage
    with the backend-side work that may still be running.
    """

    code = "backend_unconfirmed"

    def __init__(
        self,
        message: str,
        *,
        fault_id: str,
        reason: str,
        batch_id: str | None = None,
    ) -> None:
        super().__init__(message, fault_id=fault_id, reason=reason, batch_id=batch_id)
        self.fault_id = fault_id
        self.reason = reason
        self.batch_id = batch_id


class FaultConflict(GatewayError):
    """A fault had to be rejected because the gateway state moved on."""

    status_code = 409
    code = "fault_conflict"


class NoActiveFault(FaultConflict):
    code = "no_active_fault"

    def __init__(self, message: str) -> None:
        super().__init__(message, active_fault=None)


class StaleRecovery(FaultConflict):
    code = "stale_recovery"

    def __init__(
        self,
        message: str,
        *,
        current_fault_id: str | None,
        current_fault: dict[str, Any] | None,
    ) -> None:
        super().__init__(
            message,
            current_fault_id=current_fault_id,
            current_fault=current_fault,
        )
        self.current_fault_id = current_fault_id
        self.current_fault = current_fault


class RecoveryNotConfirmed(FaultConflict):
    """Recovery was attempted but the backend could not be shown idle."""

    code = "backend_not_confirmed"

    def __init__(
        self,
        message: str,
        *,
        fault_id: str,
        reason: str,
        detail: Any = None,
    ) -> None:
        super().__init__(
            message,
            fault_id=fault_id,
            reason=reason,
            detail=detail,
        )
        self.fault_id = fault_id


class ValidationError(GatewayError):
    """The request is well-formed JSON but violates gateway limits."""

    status_code = 422
    code = "validation_error"
