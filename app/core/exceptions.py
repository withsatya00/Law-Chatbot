from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from fastapi import FastAPI, Request, status
    from fastapi.responses import JSONResponse
else:
    # fastapi is a declared dependency; the fallback exists so this module can
    # still be imported (for its exception classes and status codes) on a host
    # where fastapi is missing. Type checking uses the real fastapi above.
    try:
        from fastapi import FastAPI, Request, status
        from fastapi.responses import JSONResponse
    except ModuleNotFoundError:
        FastAPI = object
        Request = object
        JSONResponse = None

        class status:
            HTTP_400_BAD_REQUEST = 400
            HTTP_401_UNAUTHORIZED = 401
            HTTP_403_FORBIDDEN = 403
            HTTP_404_NOT_FOUND = 404
            HTTP_422_UNPROCESSABLE_ENTITY = 422
            HTTP_409_CONFLICT = 409
            HTTP_429_TOO_MANY_REQUESTS = 429
            HTTP_500_INTERNAL_SERVER_ERROR = 500
from pydantic import ValidationError

if TYPE_CHECKING:
    import structlog
else:
    try:
        import structlog
    except ModuleNotFoundError:
        import logging

        class _StructlogFallback:
            @staticmethod
            def get_logger(name: str):
                return logging.getLogger(name)

        structlog = _StructlogFallback()


class AppError(Exception):
    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    code = "internal_error"

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class NotFoundError(AppError):
    status_code = status.HTTP_404_NOT_FOUND
    code = "not_found"


class UnauthorizedError(AppError):
    status_code = status.HTTP_401_UNAUTHORIZED
    code = "unauthorized"


class ForbiddenError(AppError):
    status_code = status.HTTP_403_FORBIDDEN
    code = "forbidden"


class BadRequestError(AppError):
    status_code = status.HTTP_400_BAD_REQUEST
    code = "bad_request"


class RateLimitError(AppError):
    status_code = status.HTTP_429_TOO_MANY_REQUESTS
    code = "rate_limited"


class DraftLockedError(BadRequestError):
    """Raised when an edit/translate/regenerate/rollback is attempted on a
    draft whose lifecycle_state is not editable (approved/locked/exported) --
    distinct from a plain BadRequestError so callers (chat layer) can show a
    specific "say 'unlock' to make changes" reply instead of a generic
    failure message."""

    code = "draft_locked"


class DraftConflictError(AppError):
    """Raised when `LegalDraftEngine.regenerate` cannot land an edit because
    another edit committed to the same draft first, even after one re-read
    -merge-retry cycle (BUG-015: two overlapping edits to the same draft
    previously landed as a silent lost update, with BOTH callers seeing a
    false "Updated..." success). Distinct from a plain AppError so the chat
    layer can show an explicit "someone/something else just changed this
    draft -- say 'retry' to try your change again" reply instead of a
    generic failure, and so it is never confused with an actual generation
    failure (BUG-013a's retry-reminder path)."""

    status_code = status.HTTP_409_CONFLICT
    code = "draft_conflict"


class UnsupportedExportError(BadRequestError):
    """Raised when a requested export format can't be produced for the
    draft's language -- currently PDF for scripts with no usable font on
    this server (Ol Chiki/Meetei Mayek). Distinct from DraftLockedError so
    the chat layer can point the user at DOCX/TXT instead."""

    code = "unsupported_export"


def _scrub(value: Any) -> Any:
    """Phase 1 item 7: redact personal data anywhere in an error payload.

    Error messages and `details` are assembled from whatever failed, and in
    this app that is routinely the user's own text -- a validation error on a
    drafting field echoes the value ("'9876543210' is not a valid..."), and a
    parse failure can quote a whole facts paragraph containing an address and
    a transaction id. Those payloads then land in client logs, error trackers
    and support tickets, which is exactly where this data should not
    accumulate.

    Imported lazily and defensively: the exception handler is the last line of
    defence in the process, so it must not itself be able to fail. If masking
    is unavailable for any reason the original value is returned unchanged --
    an unmasked error is bad, but a crash inside the error handler is worse.
    """
    try:
        from app.utils.pii import mask_pii
    except Exception:  # noqa: BLE001 - the error handler must never raise
        return value
    try:
        if isinstance(value, str):
            return mask_pii(value)
        if isinstance(value, dict):
            return {key: _scrub(item) for key, item in value.items()}
        if isinstance(value, list):
            return [_scrub(item) for item in value]
    except Exception:  # noqa: BLE001 - same reason
        return value
    return value


def _error_payload(code: str, message: str, details: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"error": {"code": code, "message": _scrub(message), "details": _scrub(details or {})}}


def install_exception_handlers(app: FastAPI) -> None:
    if JSONResponse is None:
        raise RuntimeError("FastAPI is required to install API exception handlers.")
    log = structlog.get_logger(__name__)

    @app.exception_handler(AppError)
    async def handle_app_error(_: Request, exc: AppError) -> JSONResponse:
        log.warning("handled_app_error", code=exc.code, status_code=exc.status_code)
        # Security/correctness finding N6: `RateLimitMiddleware`'s own 429
        # (outside this handler entirely) has always told the client when to
        # retry -- a `RateLimitError` raised from inside a route (e.g. the
        # notarization signing/verification limiters in `app/api/
        # notarization.py`) previously did not, leaving a client with no
        # signal for one 429 source but not the other.
        headers = {"Retry-After": "60"} if isinstance(exc, RateLimitError) else None
        return JSONResponse(
            status_code=exc.status_code,
            content=_error_payload(exc.code, exc.message, exc.details),
            headers=headers,
        )

    @app.exception_handler(ValidationError)
    async def handle_validation_error(_: Request, exc: ValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=_error_payload("validation_error", "Request validation failed.", {"errors": exc.errors()}),
        )

    @app.exception_handler(Exception)
    async def handle_unexpected(_: Request, exc: Exception) -> JSONResponse:
        log.exception("unhandled_exception", exc_info=exc)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=_error_payload("internal_error", "An unexpected error occurred."),
        )
