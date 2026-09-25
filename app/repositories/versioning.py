import hashlib
import json
from datetime import UTC, datetime
from typing import Any

from pymongo.errors import DuplicateKeyError

from app.models.collections import DOCUMENT_VERSIONS, INDEXING_JOBS
from app.repositories.base import MongoRepository


class DocumentVersionRepository(MongoRepository):
    collection_name = DOCUMENT_VERSIONS

    async def find_by_hash(self, document_hash: str) -> dict[str, Any] | None:
        return await self.collection.find_one({"document_hash": document_hash, "document_status": {"$ne": "deleted"}})

    async def latest_for_source(
        self, source_document: str, *, owner_user_id: str | None = None,
        owner_session_id: str | None = None,
    ) -> dict[str, Any] | None:
        """The currently-serving version for this source -- never a `staging`
        row (P0-2: `IndexingPipeline.index_file` inserts one for every
        in-flight reindex attempt, active or superseded or deleted regardless).
        A `staging` row is not yet real: it must never be picked up as "the
        version to increment from" (would waste a version number on an
        abandoned attempt) or "the version to supersede" (nothing has been
        activated yet, there is nothing to replace). Every version ever
        written before this field's `"staging"` value existed already has
        `document_status` in `{"active", "superseded", "deleted"}`, so this
        excludes nothing that used to be returned.
        """
        scope: dict[str, Any] = {"owner_user_id": owner_user_id}
        if not owner_user_id:
            scope["owner_session_id"] = owner_session_id
        return await self.collection.find_one(
            {"source_document": source_document, "document_status": {"$ne": "staging"}, **scope},
            sort=[("version_number", -1)],
        )

    async def claim_staging(self, document: dict[str, Any]) -> str | None:
        """Inserts a new `document_status="staging"` version row, claiming the
        cross-process "one in-flight reindex per source" lock via a sparse
        unique index on `reindex_lock_key` (see `scripts/create_indexes.py`)
        -- mirrors `KnowledgeBaseStagingRepository.claim_active`'s exact
        pattern for the KB upload ledger. Returns `None` if another process
        already holds the lock for this source; the caller must not proceed
        with chunking/embedding/writing in that case.
        """
        lock_key = document["source_document"]
        if document.get("owner_user_id") or document.get("owner_session_id"):
            identity = [document["source_document"], document.get("owner_user_id"),
                        None if document.get("owner_user_id") else document.get("owner_session_id")]
            lock_key = "private:" + hashlib.sha256(json.dumps(identity).encode()).hexdigest()
        document = {**document, "reindex_lock_key": lock_key}
        # Pre-check first so the common case reports the collision without
        # relying on the index existing; the DuplicateKeyError below is what
        # actually closes the race between two simultaneous claims.
        if await self.collection.find_one({"reindex_lock_key": lock_key}) is not None:
            return None
        try:
            return await self.insert(document)
        except DuplicateKeyError:
            return None

    async def activate(self, version_id: str, updates: dict[str, Any]) -> bool:
        """Terminal update for a staging row: applies `updates` (expected to
        set `document_status="active"`) and releases the reindex lock in the
        same write, mirroring `KnowledgeBaseStagingRepository.release_active`.
        """
        updates = {**updates, "updated_at": datetime.now(UTC)}
        result = await self.collection.update_one(
            {"_id": version_id}, {"$set": updates, "$unset": {"reindex_lock_key": ""}}
        )
        return result.modified_count > 0

    async def find_duplicate_shared_identities(self) -> list[dict[str, Any]]:
        """Security finding C4: groups of shared (non-owner-scoped),
        currently-ACTIVE rows that share BOTH `document_hash` and
        `version_number` -- the same logical document/version serving as
        "the truth" twice, under different `source_document` names, with no
        guarantee their review state agrees. Scoped to `document_status ==
        "active"` specifically, not merely "not deleted": a `superseded` or
        `staging` row sharing an identity with an `active` one is not a live
        contradiction (retrieval only ever serves `active` chunks -- see
        `MongoVectorStore`/`BM25Index`) and reconciling it is ordinary
        version-lifecycle history, not an incident; confirmed against this
        deployment's own data, which has exactly one such historical
        (active + superseded) pair and zero active-vs-active duplicates.

        `known_hashes` in `IndexingPipeline.index_file` blocks a NEW
        active-vs-active duplicate from being created going forward, but
        does not repair rows already written before that check existed (or
        by a write path that bypassed it) -- this is the read side of that
        repair: report the groups for an admin to reconcile, never delete
        anything automatically. Also what `scripts/create_indexes.py` calls
        before attempting the corresponding unique index, since MongoDB
        refuses to create a unique index over data that already violates it.
        """
        pipeline: list[dict[str, Any]] = [
            {"$match": {
                "owner_user_id": None, "owner_session_id": None,
                "document_status": "active", "document_hash": {"$ne": None},
            }},
            {"$group": {
                "_id": {"document_hash": "$document_hash", "version_number": "$version_number"},
                "count": {"$sum": 1},
                "records": {"$push": {
                    "id": "$_id", "source_document": "$source_document",
                    "document_status": "$document_status",
                }},
            }},
            {"$match": {"count": {"$gt": 1}}},
        ]
        return [doc async for doc in self.collection.aggregate(pipeline)]

    async def find_stale_staging(self) -> list[dict[str, Any]]:
        """`document_status="staging"` rows left behind by a crashed reindex
        (a raised exception during `IndexingPipeline.index_file` already
        cleans its own staging row up -- only a killed process skips that).
        Never auto-reclaimed: an admin decides whether to retry the source or
        remove the row, the same recovery shape
        `KnowledgeBaseStagingRepository.find_stale_active` uses for the KB
        upload ledger. These rows are already invisible to every retrieval
        path regardless (only `document_status="active"` chunks are ever
        served), so leaving one in place is inert, not a live-answer risk.
        """
        return [doc async for doc in self.collection.find({"document_status": "staging"})]


class IndexingJobRepository(MongoRepository):
    collection_name = INDEXING_JOBS
