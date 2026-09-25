from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import structlog
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from pymongo.errors import PyMongoError

from app.core.config import settings

log = structlog.get_logger(__name__)


class MongoDB:
    def __init__(self) -> None:
        self._client: AsyncIOMotorClient[Any] | None = None

    async def connect(self) -> None:
        if self._client is None:
            self._client = AsyncIOMotorClient(settings.mongodb_uri, uuidRepresentation="standard")

    async def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    @property
    def db(self) -> AsyncIOMotorDatabase[Any]:
        if self._client is None:
            raise RuntimeError("MongoDB client is not connected.")
        return self._client[settings.mongodb_database]

    async def ping(self) -> bool:
        """Whether the database is reachable. Never raises -- this backs
        `/health`, which must report a degraded database rather than 500.

        Narrow to the driver's own error base plus the `RuntimeError` raised by
        `db` when `connect()` was never called: anything else coming out of a
        one-word `ping` command is a bug in this process, and reporting it as
        "database down" would send an operator to investigate the wrong host.
        """
        try:
            await self.db.command("ping")
        except (PyMongoError, OSError, RuntimeError) as exc:
            log.warning("mongodb_ping_failed", error=str(exc), error_type=type(exc).__name__)
            return False
        return True


mongodb = MongoDB()

# Collections that store raw user message text for analytics purposes and are
# therefore subject to `settings.analytics_retention_days` (Part 58 issue 25).
# Deliberately does NOT include `chats` -- that IS the user's own conversation
# history, which they read back through `/history` and delete themselves via
# `DELETE /session`; expiring it out from under them would be data loss, not
# data hygiene.
_RETENTION_COLLECTIONS = ("query_logs", "intent_events")
_RETENTION_INDEX_NAME = "created_at_retention_ttl"


async def ensure_retention_indexes() -> list[str]:
    """Applies the configured analytics retention window as a MongoDB TTL
    index on `created_at`, and returns the collections it was applied to.

    A no-op when `analytics_retention_days` is 0 (the default) -- existing
    deployments keep their current "retain everything" behaviour until an
    operator sets a policy. Changing the configured window recreates the
    index, since MongoDB will not silently alter an existing TTL.
    """
    days = settings.analytics_retention_days
    if days <= 0:
        return []
    applied: list[str] = []
    expire_seconds = days * 24 * 3600
    for name in _RETENTION_COLLECTIONS:
        collection = mongodb.db[name]
        existing = await collection.index_information()
        current = existing.get(_RETENTION_INDEX_NAME)
        if current is not None and current.get("expireAfterSeconds") == expire_seconds:
            continue
        if current is not None:
            await collection.drop_index(_RETENTION_INDEX_NAME)
        await collection.create_index(
            "created_at", name=_RETENTION_INDEX_NAME, expireAfterSeconds=expire_seconds
        )
        applied.append(name)
    return applied


@asynccontextmanager
async def mongo_session() -> AsyncIterator[AsyncIOMotorDatabase[Any]]:
    yield mongodb.db
