"""Shared async PostgreSQL connection pool for transactional application data."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import asyncpg
import structlog

from app.core.config import settings

log = structlog.get_logger(__name__)


class PostgreSQL:
    def __init__(self) -> None:
        self.pool: asyncpg.Pool | None = None

    async def connect(self) -> None:
        if not settings.postgresql_enabled or self.pool is not None:
            return
        self.pool = await asyncpg.create_pool(
            dsn=settings.postgresql_url,
            min_size=settings.postgresql_pool_min_size,
            max_size=settings.postgresql_pool_max_size,
            command_timeout=30,
        )
        log.info("postgresql_connected")

    async def close(self) -> None:
        if self.pool is not None:
            await self.pool.close()
            self.pool = None

    async def ping(self) -> bool:
        if not settings.postgresql_enabled:
            return True
        if self.pool is None:
            return False
        try:
            return bool(await self.pool.fetchval("SELECT 1") == 1)
        except Exception:  # noqa: BLE001 - health checks return state, never crash
            return False

    @asynccontextmanager
    async def acquire(self) -> AsyncIterator[asyncpg.Connection[Any]]:
        if self.pool is None:
            raise RuntimeError("PostgreSQL is enabled but its pool is not connected")
        async with self.pool.acquire() as connection:
            yield connection


postgresql = PostgreSQL()
