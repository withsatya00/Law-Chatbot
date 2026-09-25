"""Real Redis rate-limit concurrency and network-failure tests.

Opt in with PHASE1_LIVE_TESTS=1. Uses ONLY disposable Redis at 127.0.0.1:36379.
The TCP fault proxy belongs to each test; no Redis service is stopped/flushed.
"""

import asyncio
import os
from contextlib import suppress
from types import SimpleNamespace
from uuid import uuid4

import httpx
import pytest
from fastapi import FastAPI
from redis.asyncio import Redis
from redis.backoff import NoBackoff
from redis.exceptions import RedisError
from redis.retry import Retry

from app.cache.redis_client import redis_client
from app.cache.semantic_cache import SemanticCache
from app.core import middleware
from app.core.config import settings

pytestmark = pytest.mark.skipif(
    os.environ.get("PHASE1_LIVE_TESTS") != "1", reason="requires disposable Redis on :36379"
)


class RedisFaultProxy:
    """Forward real RESP traffic, or break/blackhole the test's TCP connections."""

    def __init__(self):
        self.mode = "healthy"
        self.writers = set()
        self.tasks = set()

    async def start(self):
        self.server = await asyncio.start_server(self.accept, "127.0.0.1", 0)
        self.port = self.server.sockets[0].getsockname()[1]

    async def pipe(self, source, destination):
        while data := await source.read(65536):
            destination.write(data)
            await destination.drain()

    async def accept(self, reader, writer):
        task = asyncio.current_task()
        self.tasks.add(task)
        self.writers.add(writer)
        upstream = None
        pumps = []
        try:
            if self.mode == "disconnect":
                return
            if self.mode == "timeout":
                # Consume bytes but send no RESP response: real socket timeout.
                while await reader.read(65536):
                    pass
                return
            upstream_reader, upstream = await asyncio.open_connection("127.0.0.1", 36379)
            self.writers.add(upstream)
            pumps = [
                asyncio.create_task(self.pipe(reader, upstream)),
                asyncio.create_task(self.pipe(upstream_reader, writer)),
            ]
            await asyncio.wait(pumps, return_when=asyncio.FIRST_COMPLETED)
        except (OSError, ConnectionError):
            pass
        finally:
            for pump in pumps:
                pump.cancel()
            await asyncio.gather(*pumps, return_exceptions=True)
            for connection in (writer, upstream):
                if connection is not None:
                    connection.close()
                    with suppress(OSError):
                        await connection.wait_closed()
                    self.writers.discard(connection)
            self.tasks.discard(task)

    async def fault(self, mode):
        self.mode = mode
        for writer in list(self.writers):
            writer.close()
        tasks = list(self.tasks)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    async def close(self):
        self.server.close()
        await self.server.wait_closed()
        await self.fault("disconnect")


@pytest.fixture
async def real_rate_limit(monkeypatch):
    direct = Redis(
        host="127.0.0.1",
        port=36379,
        socket_connect_timeout=1,
        socket_timeout=1,
        retry=Retry(NoBackoff(), 0),
    )
    # Opt-in tests fail, rather than silently skip, if infrastructure is missing.
    assert await direct.ping()
    proxy = RedisFaultProxy()
    await proxy.start()
    connection = Redis(
        host="127.0.0.1",
        port=proxy.port,
        socket_connect_timeout=0.3,
        socket_timeout=0.3,
        retry=Retry(NoBackoff(), 0),
    )
    monkeypatch.setattr(redis_client, "_client", connection)
    monkeypatch.setattr(settings, "rate_limit_per_minute", 7)
    monkeypatch.setattr(settings, "trusted_proxy_ips", [])
    clock = [1800000000.0]
    monkeypatch.setattr(middleware, "time", SimpleNamespace(time=lambda: clock[0]))
    peers = ["p2-" + uuid4().hex, "p2-" + uuid4().hex]
    app = FastAPI()
    app.add_middleware(middleware.RateLimitMiddleware)
    served = []

    @app.get("/work")
    async def work():
        served.append(True)
        return {"ok": True}

    @app.get("/health/ready")
    async def health():
        return {"ok": True}

    clients = [
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app, client=(peer, 1234)), base_url="http://test"
        )
        for peer in peers
    ]
    keys = [f"rate-limit:{peer}:{minute}" for peer in peers for minute in (30000000, 30000001)]
    try:
        yield SimpleNamespace(
            redis=direct,
            connection=connection,
            proxy=proxy,
            clients=clients,
            keys=keys,
            clock=clock,
            served=served,
        )
    finally:
        for client in clients:
            await client.aclose()
        await connection.aclose()
        await proxy.close()
        await direct.delete(*keys)  # Only exact UUID-owned keys; never FLUSHDB.
        await direct.aclose()


async def test_real_concurrent_rate_limit_counts_and_window_expiry(real_rate_limit):
    env = real_rate_limit
    start = asyncio.Event()

    async def request(client):
        await start.wait()
        return await client.get("/work")

    batches = [[asyncio.create_task(request(client)) for _ in range(30)] for client in env.clients]
    start.set()
    results = [await asyncio.gather(*batch) for batch in batches]
    for batch in results:
        assert sum(response.status_code == 200 for response in batch) == 7
        assert sum(response.status_code == 429 for response in batch) == 23
        assert all(
            response.headers["retry-after"] == "60"
            for response in batch
            if response.status_code == 429
        )
    assert len(env.served) == 14
    for key in (env.keys[0], env.keys[2]):
        assert int(await env.redis.get(key)) == 30
        assert 0 < await env.redis.ttl(key) <= 60
    # Production implements a fixed minute bucket, not a sliding window.
    env.clock[0] += 60
    assert (await env.clients[0].get("/work")).status_code == 200
    assert int(await env.redis.get(env.keys[1])) == 1
    # Prove Redis expiration, without a 60-second sleep or a fake Redis clock.
    await env.redis.pexpire(env.keys[0], 40)
    await asyncio.sleep(0.08)
    assert await env.redis.get(env.keys[0]) is None


@pytest.mark.parametrize("failure", ["disconnect", "timeout"])
async def test_real_redis_outage_fails_open_then_enforces_after_recovery(real_rate_limit, failure):
    env = real_rate_limit
    client = env.clients[0]
    for _ in range(7):
        assert (await client.get("/work")).status_code == 200
    assert (await client.get("/work")).status_code == 429
    assert await env.connection.ping()

    await env.proxy.fault(failure)
    with pytest.raises(RedisError):
        await env.connection.ping()  # Actual network error, no mocked exception.
    assert not await redis_client.ping()
    responses = await asyncio.wait_for(asyncio.gather(*(client.get("/work") for _ in range(10))), 5)
    assert all(
        response.status_code == 200 and response.json() == {"ok": True} for response in responses
    )
    assert int(await env.redis.get(env.keys[0])) == 8  # Outage requests reached no Redis counter.
    assert (await client.get("/health/ready")).status_code == 200

    env.proxy.mode = "healthy"
    assert await env.connection.ping()  # Same pool reconnects; no application restart.
    assert (await client.get("/work")).status_code == 429
    assert int(await env.redis.get(env.keys[0])) == 9


async def test_real_cache_network_failure_degrades_and_recovers(real_rate_limit):
    env = real_rate_limit
    cache = SemanticCache("p2-test-" + uuid4().hex)
    key = await cache.key("synthetic question", "english")
    try:
        await cache.set("synthetic question", {"answer": "cached"}, "english", ttl_seconds=60)
        assert await cache.get("synthetic question", "english") == {"answer": "cached"}
        await env.proxy.fault("disconnect")
        assert await cache.get("synthetic question", "english") is None
        await cache.set(
            "synthetic question", {"answer": "during outage"}, "english", ttl_seconds=60
        )
        env.proxy.mode = "healthy"
        assert await cache.get("synthetic question", "english") == {"answer": "cached"}
    finally:
        await env.redis.delete(key)
