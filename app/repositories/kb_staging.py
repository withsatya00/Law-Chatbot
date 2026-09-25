from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pymongo.errors import DuplicateKeyError

from app.models.collections import KB_STAGING_RECORDS
from app.repositories.base import MongoRepository


class KnowledgeBaseStagingRepository(MongoRepository):
    """Audit ledger for the admin Knowledge Base ingestion pipeline.

    Separate from `document_versions`/`indexing_jobs` (used by the existing
    reindex flow) so every admin upload attempt -- including ones rejected as
    duplicates or left pending after a failed index -- has a durable record,
    without touching those collections' existing shape.

    Cross-process single-active-job rule: a row that is still in flight
    (`pending`/`processing`) carries `active_key = content_hash`; a row that
    reached a terminal state has no `active_key` at all. With the sparse
    unique index on `active_key` (see `scripts/create_indexes.py`) Mongo
    itself -- not a process-local lock -- guarantees that two workers, in two
    processes, cannot both hold an active job for identical bytes, while any
    number of historical terminal rows for that same content are still kept.
    """

    collection_name = KB_STAGING_RECORDS

    STATUSES = ("pending", "processing", "indexed", "duplicate", "failed", "needs_review")
    ACTIVE_STATUSES = ("pending", "processing")
    TERMINAL_STATUSES = ("indexed", "duplicate", "failed", "needs_review")
    # What `GET /admin/knowledge-base/staging` shows when no status is asked
    # for: everything that is waiting on the queue or on an admin.
    ACTIONABLE_STATUSES = ("pending", "processing", "failed", "needs_review")

    async def find_by_status(self, status: str) -> list[dict[str, Any]]:
        return [record async for record in self.collection.find({"status": status})]

    async def find_by_statuses(self, statuses: tuple[str, ...] | list[str]) -> list[dict[str, Any]]:
        cursor = self.collection.find({"status": {"$in": list(statuses)}})
        return [record async for record in cursor]

    async def find_by_content_hash(self, content_hash: str) -> dict[str, Any] | None:
        return await self.collection.find_one({"content_hash": content_hash})

    async def find_active_by_content_hash(self, content_hash: str) -> dict[str, Any] | None:
        return await self.collection.find_one(
            {"content_hash": content_hash, "status": {"$in": list(self.ACTIVE_STATUSES)}}
        )

    async def find_indexed_by_content_hash(self, content_hash: str) -> dict[str, Any] | None:
        return await self.collection.find_one({"content_hash": content_hash, "status": "indexed"})

    async def find_duplicate_by_content_hash(self, content_hash: str) -> dict[str, Any] | None:
        """An already-`duplicate` row for this exact content, if any -- used
        by `KnowledgeBaseIngestionService._reject_duplicate` to avoid
        archiving the same rejected bytes a second (or Nth) time (see its
        own docstring for the incident this closes: a source repeatedly
        re-discovered by automation/sync, on an interval, kept being
        archived as a "new" duplicate copy every time, accumulating 150+
        redundant ~1.3MB copies of the same official documents in real
        storage over time). The caller only needs SOME still-existing
        archived copy to reuse, not specifically the newest one, so this
        takes whatever match the driver returns first rather than
        requesting a sort.
        """
        return await self.collection.find_one({"content_hash": content_hash, "status": "duplicate"})

    async def claim_active(self, document: dict[str, Any]) -> str | None:
        """Inserts a new active (`pending`) row for this content, or returns
        `None` if another process already holds the active job for the same
        bytes. The uniqueness is enforced by Mongo's sparse unique index on
        `active_key`, so this is safe across processes and machines -- an
        `asyncio.Lock` would only have covered one event loop.
        """
        document = {**document, "active_key": document["content_hash"]}
        # Pre-check first so the common case reports the collision without
        # relying on the index existing; the DuplicateKeyError below is what
        # actually closes the race between two simultaneous inserts.
        if await self.find_active_by_content_hash(document["content_hash"]) is not None:
            return None
        try:
            return await self.insert(document)
        except DuplicateKeyError:
            return None

    async def release_active(self, item_id: str, updates: dict[str, Any]) -> bool:
        """Terminal update: applies `updates` and drops `active_key` in the
        same write, so the content becomes claimable again while this row is
        preserved as history."""
        updates = {**updates, "updated_at": datetime.now(UTC)}
        result = await self.collection.update_one(
            {"_id": item_id}, {"$set": updates, "$unset": {"active_key": ""}}
        )
        return result.modified_count > 0

    async def status_counts(self, collection: Any = None) -> dict[str, Any]:
        """Counts ledger rows per status for the admin dashboard. Accepts an
        optional collection override so this is unit-testable without Mongo."""
        coll = collection if collection is not None else self.collection
        counts = {status: await coll.count_documents({"status": status}) for status in self.STATUSES}
        counts["total"] = await coll.count_documents({})
        return counts

    async def find_path_missing(self) -> list[dict[str, Any]]:
        """Rows flagged by reconciliation as having no file behind them. They
        are ledger findings, not physical files, and they stay open until an
        admin closes or re-sources them -- nothing converts them silently."""
        cursor = self.collection.find({"path_missing": True, "resolution": {"$exists": False}})
        return [record async for record in cursor]

    async def find_stale_active(self) -> list[dict[str, Any]]:
        """Active rows whose file is not where the ledger says it is. These
        are the records that must never be left `processing` forever."""
        stale = []
        for record in await self.find_by_statuses(self.ACTIVE_STATUSES):
            path = record.get("current_path") or record.get("staged_path")
            if not path or not Path(path).exists():
                stale.append(record)
        return stale
