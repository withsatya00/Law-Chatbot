import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

from redis.exceptions import ConnectionError as RedisConnectionError

from app.cache.redis_client import redis_client
from app.core.middleware import RateLimitMiddleware
from app.core.security import Role, create_access_token, decode_token


def test_access_token_roundtrip() -> None:
    token = create_access_token("user-1", Role.user)
    payload = decode_token(token)
    assert payload["sub"] == "user-1"
    assert payload["role"] == "user"


def test_rate_limiter_allows_request_when_redis_is_unavailable() -> None:
    fake_client = SimpleNamespace(
        incr=AsyncMock(side_effect=RedisConnectionError("redis unavailable")),
        expire=AsyncMock(),
    )
    redis_client._client = fake_client
    call_next = AsyncMock(return_value=type("Response", (), {})())
    request = type("Request", (), {
        "url": type("URL", (), {"path": "/chat"})(),
        "client": type("Client", (), {"host": "127.0.0.1"})(),
    })()

    response = asyncio.run(RateLimitMiddleware(None).dispatch(request, call_next))

    assert response is not None
    call_next.assert_awaited_once_with(request)
