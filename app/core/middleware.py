import time
import uuid
from collections.abc import Awaitable, Callable

import structlog
from fastapi import Request, Response
from redis.exceptions import RedisError
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from app.cache.redis_client import redis_client
from app.core.config import settings
from app.observability.metrics import metrics
from app.utils.request_ip import client_ip


class RequestContextMiddleware(BaseHTTPMiddleware):
    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        request_id = request.headers.get("x-request-id", str(uuid.uuid4()))
        trace_id = request.headers.get("traceparent", request_id).split("-")[1] if "-" in request.headers.get("traceparent", "") else request_id
        request.state.request_id = request_id
        structlog.contextvars.bind_contextvars(request_id=request_id, trace_id=trace_id)
        start = time.perf_counter()
        try:
            response = await call_next(request)
            elapsed_ms = (time.perf_counter() - start) * 1000
            metrics.increment(f"http.status.{response.status_code}")
            metrics.observe_ms(f"http.{request.method}.{request.url.path}", elapsed_ms)
            response.headers["x-request-id"] = request_id
            response.headers["x-trace-id"] = trace_id
            response.headers["x-response-time-ms"] = f"{elapsed_ms:.2f}"
            return response
        except Exception as exc:
            metrics.increment("http.unhandled_errors")
            structlog.get_logger(__name__).exception(
                "request_failed", method=request.method, path=request.url.path, error_type=type(exc).__name__
            )
            raise
        finally:
            structlog.contextvars.clear_contextvars()


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        response = await call_next(request)
        response.headers["x-content-type-options"] = "nosniff"
        response.headers["x-frame-options"] = "DENY"
        response.headers["referrer-policy"] = "no-referrer"
        response.headers["permissions-policy"] = "camera=(), microphone=(), geolocation=()"
        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        if request.url.path in {"/health", "/health/live", "/health/ready"}:
            return await call_next(request)
        client_host = client_ip(request)
        key = f"rate-limit:{client_host}:{int(time.time() // 60)}"
        try:
            count = await redis_client.client.incr(key)
            if count == 1:
                await redis_client.client.expire(key, 60)
            if count > settings.rate_limit_per_minute:
                # Security/correctness finding N6: this middleware sits
                # OUTSIDE FastAPI's exception handlers, so a 429 from here
                # previously used its own one-off body shape (`{"detail":
                # ...}`) -- every other error in this app, including the
                # notarization-specific rate limits raised as `RateLimitError`
                # (`app/api/notarization.py`), goes through `app.core.
                # exceptions.install_exception_handlers` and comes back as
                # `{"error": {"code", "message", "details"}}`. A client
                # written against one shape broke on the other depending on
                # which limiter happened to fire. Matched here exactly, and
                # `RateLimitError`'s own handler (`install_exception_
                # handlers` below) now also sets the same `Retry-After`
                # header this path already had, so neither behavior
                # regresses for the other.
                return JSONResponse(
                    {"error": {
                        "code": "rate_limited",
                        "message": "Too many requests. Please try again shortly.",
                        "details": {},
                    }},
                    status_code=429,
                    headers={"Retry-After": "60"},
                )
        except (RedisError, RuntimeError):
            # Rate limiting is protective infrastructure; a Redis outage
            # must not turn every otherwise healthy API request into a 500.
            pass
        return await call_next(request)
