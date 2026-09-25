"""Provisions all MongoDB indexes the application depends on.

Safe to re-run: every index uses `create_index`/`create_indexes`, which are no-ops
when an equivalent index already exists. Run this during deployment and whenever a
new index is introduced.

Usage:
    python scripts/create_indexes.py
"""

import asyncio

import structlog
from pymongo import ASCENDING, DESCENDING, TEXT, IndexModel
from pymongo.errors import OperationFailure

from app.core.config import settings
from app.core.logger import configure_logging
from app.database.mongodb import mongodb
from app.models.collections import (
    AUDIT_LOGS,
    BACKGROUND_JOBS,
    CASES,
    CHATS,
    CONVERSATION_MEMORY,
    DOCUMENT_VERSIONS,
    DOWNLOAD_ARTIFACTS,
    DRAFT_VERSIONS,
    EMBEDDINGS_METADATA,
    EVALUATION_RUNS,
    FEEDBACK,
    FORM_WORKFLOWS,
    INDEXING_JOBS,
    KB_AUTOMATION_JOBS,
    KB_SOURCE_ADAPTERS,
    KB_STAGING_RECORDS,
    LEGAL_DRAFTS,
    LEGAL_SOURCES,
    OBSERVABILITY_EVENTS,
    OPERATIONAL_EVENTS,
    PROMPT_LOGS,
    QUERY_LOGS,
    SESSIONS,
    SYSTEM_LOGS,
    UPLOADED_DOCUMENTS,
    USER_PREFERENCES,
    USERS,
)

log = structlog.get_logger(__name__)

# Retention windows for collections that must not grow unbounded.
LOG_RETENTION_SECONDS = 60 * 24 * 3600
SESSION_RETENTION_SECONDS = 30 * 24 * 3600
CONVERSATION_MEMORY_RETENTION_SECONDS = 90 * 24 * 3600


async def create_standard_indexes() -> None:
    db = mongodb.db

    await db[USERS].create_indexes([IndexModel([("email", ASCENDING)], unique=True, name="uniq_email")])

    await db[CHATS].create_indexes(
        [
            IndexModel([("session_id", ASCENDING), ("created_at", ASCENDING)], name="session_timeline"),
            IndexModel([("user_id", ASCENDING), ("created_at", DESCENDING)], name="user_history"),
        ]
    )

    await db[UPLOADED_DOCUMENTS].create_indexes(
        [
            IndexModel([("filename", ASCENDING)], name="filename_lookup"),
            IndexModel([("document_hash", ASCENDING)], name="hash_lookup"),
        ]
    )

    await db[EMBEDDINGS_METADATA].create_indexes(
        [
            IndexModel([("document_id", ASCENDING)], name="document_id_lookup"),
            IndexModel([("metadata.act_name", ASCENDING)], name="act_name_filter"),
            IndexModel([("metadata.section_number", ASCENDING)], name="section_number_filter"),
            IndexModel([("metadata.source_document", ASCENDING)], name="source_document_filter"),
            IndexModel([("metadata.namespace", ASCENDING)], name="namespace_filter"),
            # Phase 1 "Jurisdiction-Aware Knowledge Base": `LegalRetriever.retrieve`
            # now applies `review_status` on every shared-corpus query (see
            # `app.rag.kb_jurisdiction.shared_retrieval_filters`) -- without this,
            # every retrieval would collection-scan on that field.
            IndexModel([("metadata.review_status", ASCENDING)], name="review_status_filter"),
            # P0-2 "Failure-safe reindexing": `MongoVectorStore.search`/
            # `find_by_section_number`/`find_constitution_article`/
            # `find_named_section` and `BM25Index.search` now ALL require
            # `metadata.document_status="active"` unconditionally -- every one
            # of those queries carries this clause, so it needs its own index
            # rather than relying on a compound index elsewhere to cover it.
            IndexModel([("metadata.document_status", ASCENDING)], name="document_status_filter"),
            IndexModel([("text", TEXT)], name="text_search", default_language="none"),
        ]
    )

    await db[DOCUMENT_VERSIONS].create_indexes(
        [
            IndexModel([("source_document", ASCENDING), ("version_number", DESCENDING)], name="source_version_lookup"),
            IndexModel([("document_hash", ASCENDING)], name="hash_lookup"),
            IndexModel([("document_status", ASCENDING)], name="status_filter"),
            # P0-2 "Failure-safe reindexing": the cross-process "one in-flight
            # reindex per source" lock -- only a `document_status="staging"`
            # row carries `reindex_lock_key` (set to its own `source_document`;
            # everything else `$unset`s it), so this mirrors
            # `KB_STAGING_RECORDS`' `uniq_active_ingestion` index exactly: a
            # second concurrent claim for the same source is rejected by
            # MongoDB itself, not a process-local lock, while every historical
            # active/superseded/deleted row for that source is unaffected
            # (sparse -- the index simply has no entry for them).
            IndexModel(
                [("reindex_lock_key", ASCENDING)], unique=True, sparse=True, name="uniq_reindex_lock"
            ),
        ]
    )
    # Security finding C4: `uniq_reindex_lock` above only ever prevented two
    # CONCURRENT ingestion attempts for the SAME `source_document` name from
    # racing each other -- it says nothing about two attempts under
    # DIFFERENT names that happen to carry identical content. Application
    # code already refuses that going forward (`IndexingPipeline.
    # index_file`'s `_known_hashes()` pre-check), but that alone is a
    # best-effort guard a bug or a write path that bypasses it can still
    # slip past; this is the hard backstop. Scoped to shared, currently-
    # ACTIVE rows only (a partial filter, not a plain unique index): two
    # different users privately uploading the identical file is expected
    # (see `IndexingPipeline.index_file`'s own `existing_same_hashes =
    # set() if (owner_user_id or owner_session_id) else ...`), and a
    # `superseded`/`staging`/`deleted` row sharing an identity with the
    # `active` one is ordinary version-lifecycle history, not a live
    # contradiction -- see `DocumentVersionRepository.
    # find_duplicate_shared_identities` for the full reasoning (this
    # deployment's own data has exactly that harmless historical shape).
    # Created separately from the batch above and never allowed to abort
    # the rest of index provisioning: it fails with `OperationFailure` if
    # duplicate ACTIVE data already exists, which must be reconciled by an
    # admin, not silently ignored or auto-deleted.
    try:
        await db[DOCUMENT_VERSIONS].create_indexes([
            IndexModel(
                [("document_hash", ASCENDING), ("version_number", ASCENDING)],
                unique=True,
                partialFilterExpression={
                    "owner_user_id": None, "owner_session_id": None,
                    "document_status": "active",
                },
                name="uniq_shared_document_identity",
            )
        ])
    except OperationFailure as exc:
        from app.repositories.versioning import DocumentVersionRepository

        duplicates = await DocumentVersionRepository().find_duplicate_shared_identities()
        log.warning(
            "duplicate_document_identity_index_not_created",
            reason=str(exc),
            duplicate_group_count=len(duplicates),
            hint="Existing duplicate (document_hash, version_number) rows must be reconciled by an "
            "admin (merge or supersede one of each group) before this index can be created -- see "
            "DocumentVersionRepository.find_duplicate_shared_identities for the exact groups.",
        )

    await db[INDEXING_JOBS].create_indexes(
        [
            IndexModel([("status", ASCENDING), ("started_at", DESCENDING)], name="status_timeline"),
        ]
    )

    await db[FEEDBACK].create_indexes(
        [
            IndexModel([("session_id", ASCENDING)], name="session_lookup"),
            IndexModel([("user_id", ASCENDING)], name="user_lookup"),
        ]
    )

    await db[LEGAL_DRAFTS].create_indexes(
        [
            IndexModel([("session_id", ASCENDING), ("created_at", DESCENDING)], name="session_history"),
            IndexModel([("user_id", ASCENDING), ("created_at", DESCENDING)], name="user_history"),
            IndexModel([("draft_type", ASCENDING)], name="draft_type_filter"),
        ]
    )

    await db[DRAFT_VERSIONS].create_indexes(
        [
            IndexModel([("draft_id", ASCENDING), ("version_number", DESCENDING)], name="draft_version_lookup"),
            IndexModel([("document_status", ASCENDING)], name="status_filter"),
        ]
    )

    await db[CASES].create_indexes(
        [
            IndexModel([("owner_user_id", ASCENDING), ("updated_at", DESCENDING)], name="owner_history"),
            IndexModel([("owner_user_id", ASCENDING), ("status", ASCENDING)], name="owner_status_filter"),
            IndexModel([("owner_user_id", ASCENDING), ("next_hearing_date", ASCENDING)], name="owner_upcoming_hearings"),
            IndexModel([("case_number", ASCENDING)], name="case_number_lookup"),
            IndexModel([("owner_user_id", ASCENDING), ("tasks.due_at", ASCENDING)], name="owner_task_due"),
            IndexModel([("owner_user_id", ASCENDING), ("linked_draft_ids", ASCENDING)], name="owner_linked_drafts"),
        ]
    )

    await db[LEGAL_SOURCES].create_indexes([
        IndexModel([("verification_status", ASCENDING), ("last_verified_date", ASCENDING)], name="verification_staleness"),
        IndexModel([("act_name", ASCENDING), ("section_number", ASCENDING)], name="act_section_lookup"),
        IndexModel([("document_id", ASCENDING)], name="source_document_lookup"),
        # Phase 2 governance. `checksum` detects a verified source whose file
        # changed underneath the verification; the lineage indexes make
        # "what replaced this Act?" a lookup rather than a scan.
        IndexModel([("checksum", ASCENDING)], name="source_checksum_lookup", sparse=True),
        IndexModel([("superseded_by", ASCENDING)], name="source_superseded_by", sparse=True),
        IndexModel([("supersedes", ASCENDING)], name="source_supersedes", sparse=True),
        IndexModel([("chunk_ids", ASCENDING)], name="source_chunk_lookup", sparse=True),
        IndexModel([("status", ASCENDING), ("effective_date", ASCENDING)], name="legal_status_currency"),
    ])
    await db[BACKGROUND_JOBS].create_indexes([
        IndexModel([("status", ASCENDING), ("created_at", ASCENDING)], name="worker_claim_queue"),
        IndexModel([("owner_user_id", ASCENDING), ("created_at", DESCENDING)], name="owner_jobs"),
    ])
    await db[USER_PREFERENCES].create_indexes([
        IndexModel([("owner_user_id", ASCENDING)], unique=True, name="unique_owner_preferences"),
    ])
    await db[FORM_WORKFLOWS].create_indexes([
        IndexModel([("owner_user_id", ASCENDING), ("updated_at", DESCENDING)], name="owner_forms"),
        IndexModel([("case_id", ASCENDING)], name="case_forms"),
    ])
    await db[AUDIT_LOGS].create_indexes([
        IndexModel([("actor_user_id", ASCENDING), ("created_at", DESCENDING)], name="actor_audit_timeline"),
        IndexModel([("resource_type", ASCENDING), ("resource_id", ASCENDING)], name="resource_audit"),
    ])
    await db[OPERATIONAL_EVENTS].create_indexes([
        IndexModel([("event_type", ASCENDING), ("created_at", DESCENDING)], name="operational_event_timeline"),
        IndexModel([("alert_key", ASCENDING)], unique=True, sparse=True, name="unique_open_kb_alert"),
    ])
    await db[KB_STAGING_RECORDS].create_indexes([
        # The cross-process "one active ingestion job per content" rule. Only
        # pending/processing rows carry `active_key` (terminal rows have it
        # unset), so `sparse=True` keeps every historical duplicate/failed row
        # for the same content while still making a second *active* claim on
        # identical bytes impossible -- something a process-local lock could
        # never guarantee.
        IndexModel([("active_key", ASCENDING)], unique=True, sparse=True, name="uniq_active_ingestion"),
        IndexModel([("content_hash", ASCENDING)], name="kb_staging_content_hash"),
        IndexModel([("status", ASCENDING), ("created_at", DESCENDING)], name="kb_staging_status_timeline"),
    ])
    await db[KB_AUTOMATION_JOBS].create_indexes([
        IndexModel([("source_key", ASCENDING)], unique=True, name="unique_automation_source_version"),
        IndexModel([("status", ASCENDING), ("next_attempt_at", ASCENDING)], name="automation_due_queue"),
        IndexModel([("canonical_document_key", ASCENDING), ("created_at", DESCENDING)], name="automation_versions"),
        IndexModel([("status", ASCENDING), ("next_source_check_at", ASCENDING)], name="automation_source_checks"),
        IndexModel([("status", ASCENDING), ("next_canary_at", ASCENDING)], name="automation_canary_checks"),
    ])
    await db[KB_SOURCE_ADAPTERS].create_indexes([
        IndexModel([("name", ASCENDING)], unique=True, name="unique_kb_source_adapter_name"),
        IndexModel([("status", ASCENDING), ("jurisdiction_code", ASCENDING)],
                   name="kb_source_rollout_status"),
    ])
    await db[EVALUATION_RUNS].create_indexes([
        IndexModel([("created_at", DESCENDING)], name="evaluation_history"),
    ])
    await db[DOWNLOAD_ARTIFACTS].create_indexes([
        IndexModel([("owner_user_id", ASCENDING), ("created_at", DESCENDING)], name="owner_downloads"),
    ])

    # TTL indexes: `expireAfterSeconds` deletes documents automatically once
    # `created_at`/`updated_at` is older than the retention window, so log and session
    # collections stay bounded without a manual cleanup job.
    await db[SESSIONS].create_indexes(
        [IndexModel([("created_at", ASCENDING)], name="ttl_sessions", expireAfterSeconds=SESSION_RETENTION_SECONDS)]
    )
    await db[CONVERSATION_MEMORY].create_indexes(
        [
            IndexModel(
                [("updated_at", ASCENDING)],
                name="ttl_conversation_memory",
                expireAfterSeconds=CONVERSATION_MEMORY_RETENTION_SECONDS,
            )
        ]
    )
    for log_collection in (PROMPT_LOGS, QUERY_LOGS, SYSTEM_LOGS, OBSERVABILITY_EVENTS):
        await db[log_collection].create_indexes(
            [IndexModel([("created_at", ASCENDING)], name="ttl_logs", expireAfterSeconds=LOG_RETENTION_SECONDS)]
        )

    log.info("standard_indexes_created")


async def create_vector_search_index() -> None:
    """Creates (or widens) the Atlas Vector Search index used by `$vectorSearch`
    in production.

    Only works against an Atlas cluster with Search enabled. Against community/local
    MongoDB (e.g. the docker-compose dev stack) this call is rejected by the server —
    that's expected; the retriever falls back to an in-process cosine/BM25 search in
    that case (see `MongoVectorStore`).

    Phase 4A "Atlas Vector Filter Repair": `metadata.owner_session_id` and
    `metadata.owner_user_id` (Part 45/46 per-user ownership -- present in the
    `$or` branches on every single `_prepare_rag_context` call, never
    optional) and `metadata.section_number` (added to the filter for every
    SECTION_LOOKUP/citation-style query by `LegalRetriever.retrieve`) were
    missing from the filterable-fields list below. Atlas rejects a
    `$vectorSearch` filter that references a field the index doesn't declare
    as `type: "filter"`; `MongoVectorStore.search` catches that as a
    `PyMongoError` and silently falls back to `_local_cosine_leg` (a
    brute-force in-process scan capped at `local_scan_limit`) -- meaning,
    prior to this fix, essentially every request against an Atlas-backed
    deployment was silently skipping the index-backed vector search entirely,
    not just the rare query missing one of the four fields already listed.
    `legal_category`/`language`/`namespace` are kept even though nothing in
    the current codebase populates them into a retrieval filter today --
    removing them isn't part of this fix's scope, and Atlas index field
    changes aren't user-supplied/dynamic, so there's no cost to leaving them.

    `metadata.document_status` (confirmed live 2026-09-23 against a real
    Atlas Search deployment, `mongodb/mongodb-atlas-local`): every single
    call to `MongoVectorStore.search()` -- there is no caller that skips
    this -- unconditionally injects `document_status: "active"` via
    `_document_status_filter` before the filter ever reaches here. Without
    this field declared as filterable, Atlas rejects EVERY `$vectorSearch`
    call with "Path 'metadata.document_status' needs to be indexed as
    filter", which `MongoVectorStore.search` catches as a `PyMongoError` and
    silently falls back to `_local_cosine_leg` -- meaning an Atlas-backed
    deployment missing this field would NEVER actually use `$vectorSearch`
    at all, on any request, ever (worse than the Phase 4A gap: that one was
    conditional on ownership/section filters being present; this one fires
    unconditionally on every call).
    """
    definition = {
        "name": settings.mongodb_vector_index,
        "type": "vectorSearch",
        "definition": {
            "fields": [
                {
                    "type": "vector",
                    "path": "embedding",
                    "numDimensions": settings.embedding_dimensions,
                    "similarity": "cosine",
                },
                {"type": "filter", "path": "metadata.legal_category"},
                {"type": "filter", "path": "metadata.language"},
                {"type": "filter", "path": "metadata.act_name"},
                {"type": "filter", "path": "metadata.namespace"},
                {"type": "filter", "path": "metadata.owner_session_id"},
                {"type": "filter", "path": "metadata.owner_user_id"},
                {"type": "filter", "path": "metadata.section_number"},
                {"type": "filter", "path": "metadata.document_status"},
            ]
        },
    }
    collection = mongodb.db[EMBEDDINGS_METADATA]
    try:
        await collection.create_search_index(definition)
        log.info("atlas_vector_search_index_created", index=settings.mongodb_vector_index)
    except OperationFailure as exc:
        if "already exists" in str(exc).lower() or "duplicate" in str(exc).lower():
            # Phase 4A: previously treated as a no-op success, which meant an
            # index created before this fix (missing the three fields above)
            # stayed stale forever on every re-run. `update_search_index`
            # replaces the live definition in place (requires Atlas server
            # 7.0+; see migration notes) so re-running this script now
            # actually repairs an existing under-indexed deployment, not
            # just a brand new one.
            try:
                await collection.update_search_index(settings.mongodb_vector_index, definition["definition"])
                log.info("atlas_vector_search_index_updated", index=settings.mongodb_vector_index)
            except OperationFailure as update_exc:
                log.warning(
                    "atlas_vector_search_index_update_failed",
                    index=settings.mongodb_vector_index,
                    reason=str(update_exc),
                    hint="Existing index left as-is -- update it manually via the Atlas UI/API "
                    "(requires Atlas server 7.0+ for in-place updateSearchIndex).",
                )
        else:
            log.warning(
                "atlas_vector_search_index_not_created",
                reason=str(exc),
                hint="Requires an Atlas cluster with Search enabled. Falling back to local cosine search.",
            )
    # pymongo raises driver-specific errors for unsupported commands
    except Exception as exc:  # noqa: BLE001 - pymongo raises driver-specific errors for unsupported commands; falls back to local cosine search
        log.warning(
            "atlas_vector_search_index_unsupported",
            reason=str(exc),
            hint="This MongoDB deployment does not support Atlas Search index management.",
        )


async def main() -> None:
    configure_logging()
    await mongodb.connect()
    try:
        await create_standard_indexes()
        from app.services.law_monitoring import LawMonitoringService
        await LawMonitoringService().ensure_indexes()
        await create_vector_search_index()
    finally:
        await mongodb.close()


if __name__ == "__main__":
    asyncio.run(main())
