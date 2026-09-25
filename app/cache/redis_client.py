from typing import Any

import orjson
from redis.asyncio import Redis
from redis.exceptions import RedisError

from app.core.config import settings

# What a Redis round-trip may legitimately fail with when the cache tier is
# degraded or absent. Every cache read/write in this package is best-effort by
# design -- an unreachable Redis must degrade chat to "no cache", never to an
# error -- but that must not be spelled `except Exception`, which also
# swallows `TypeError`/`AttributeError`/`KeyError` raised by OUR OWN
# serialization and key-building code and silently reports them as a cache
# miss. Those are bugs and have to surface.
#
#   RedisError            -- every server/protocol/connection failure redis-py
#                            raises, including ConnectionError and TimeoutError.
#   OSError               -- socket-level failures raised before redis-py wraps
#                            them, and `asyncio.TimeoutError` (an alias of the
#                            builtin `TimeoutError`, itself an `OSError`).
#   RuntimeError          -- `RedisClient.client` when `connect()` was never
#                            called, i.e. a deployment running without Redis.
#   orjson.JSONDecodeError -- a cache entry written by an older/corrupted
#                            encoder; treat as a miss and move on.
CACHE_UNAVAILABLE_ERRORS: tuple[type[Exception], ...] = (
    RedisError,
    OSError,
    RuntimeError,
    orjson.JSONDecodeError,
)


class RedisClient:
    def __init__(self) -> None:
        self._client: Redis | None = None

    async def connect(self) -> None:
        if self._client is None:
            self._client = Redis.from_url(settings.redis_url, decode_responses=False)

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    @property
    def client(self) -> Redis:
        if self._client is None:
            raise RuntimeError("Redis client is not connected.")
        return self._client

    async def ping(self) -> bool:
        try:
            return bool(await self.client.ping())
        except CACHE_UNAVAILABLE_ERRORS:
            return False

    async def get_json(self, key: str) -> Any | None:
        value = await self.client.get(key)
        return None if value is None else orjson.loads(value)

    async def set_json(self, key: str, value: Any, ttl_seconds: int | None = None) -> None:
        await self.client.set(key, orjson.dumps(value), ex=ttl_seconds)


redis_client = RedisClient()
