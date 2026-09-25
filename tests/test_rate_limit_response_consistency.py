"""Regression tests for correctness finding N6 (rate-limit error responses
had two different shapes depending on which limiter fired).

`RateLimitMiddleware` (global, outside FastAPI's exception handlers) used
to return `{"detail": "..."}` with a `Retry-After` header; `RateLimitError`
(raised from inside a route, e.g. the notarization signing/verification
limiters) went through `install_exception_handlers` and came back as
`{"error": {"code", "message", "details"}}` with NO `Retry-After` header at
all. A client written against one shape broke on the other.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.cache.redis_client import redis_client
from app.core.config import settings
from app.core.exceptions import RateLimitError, install_exception_handlers
from app.core.middleware import RateLimitMiddleware


def test_middleware_rate_limit_uses_the_standard_error_envelope() -> None:
    # `redis_client` is a process-wide singleton shared with every other
    # test in this session -- a fake client left in place here (this one's
    # `incr` SUCCEEDS with an over-limit count, unlike a Redis-outage fake
    # whose `incr` raises and which the middleware's own `except
    # RedisError` already tolerates harmlessly forever after) would make
    # EVERY subsequent request through this middleware, in any other test
    # file, 429 for the rest of the process. Confirmed live: exactly that,
    # breaking unrelated response-cache/retrieval/root-route tests that
    # merely happened to run afterward.
    original_client = redis_client._client
    fake_client = SimpleNamespace(
        incr=AsyncMock(return_value=settings.rate_limit_per_minute + 1),
        expire=AsyncMock(),
    )
    redis_client._client = fake_client
    try:
        call_next = AsyncMock()
        request = type("Request", (), {
            "url": type("URL", (), {"path": "/chat"})(),
            "client": type("Client", (), {"host": "203.0.113.5"})(),
        })()

        response = asyncio.run(RateLimitMiddleware(None).dispatch(request, call_next))

        assert response.status_code == 429
        import json
        body = json.loads(bytes(response.body))
        assert body == {
            "error": {
                "code": "rate_limited",
                "message": "Too many requests. Please try again shortly.",
                "details": {},
            }
        }
        assert response.headers["retry-after"] == "60"
        call_next.assert_not_awaited()
    finally:
        redis_client._client = original_client


def test_route_raised_rate_limit_error_has_the_same_shape_and_retry_after() -> None:
    """A `RateLimitError` raised from inside a route (mirrors the
    notarization signing/verification limiters) must produce the identical
    envelope AND now carry the same `Retry-After` header the middleware
    path always had."""
    app = FastAPI()
    install_exception_handlers(app)

    @app.get("/limited")
    async def limited() -> None:
        raise RateLimitError("Too many requests. Please try again shortly.")

    response = TestClient(app).get("/limited")

    assert response.status_code == 429
    assert response.json() == {
        "error": {
            "code": "rate_limited",
            "message": "Too many requests. Please try again shortly.",
            "details": {},
        }
    }
    assert response.headers["retry-after"] == "60"


def test_a_non_rate_limit_error_does_not_get_a_retry_after_header() -> None:
    """The `Retry-After` addition must be specific to rate limiting, not a
    blanket header on every error response."""
    from app.core.exceptions import BadRequestError

    app = FastAPI()
    install_exception_handlers(app)

    @app.get("/bad")
    async def bad() -> None:
        raise BadRequestError("nope")

    response = TestClient(app).get("/bad")

    assert response.status_code == 400
    assert "retry-after" not in response.headers
