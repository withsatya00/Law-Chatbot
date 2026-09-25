"""Phase 1 "Jurisdiction-Aware Knowledge Base": backfill for the EXISTING
corpus, indexed before `app.rag.kb_jurisdiction` existed.

Every document/chunk indexed so far has none of the jurisdiction fields at
all (`issuing_level`, `applicability`, `verification_status`, ...) -- not
"unknown", simply absent. `LegalRetriever`/`ChatService`'s shared-retrieval
gate (`kb_jurisdiction.shared_retrieval_filters`) requires `review_status`
to equal `approved` EXACTLY -- there is no backward-compatible "absent means
visible" default (an earlier version of this gate had one; it was removed as
gap 1 of Phase 1's own review, since it silently let every un-reviewed
document keep answering shared queries). This means the entire pre-Phase-1
corpus is ALREADY excluded from shared retrieval the moment this code ships,
whether or not this script has ever been run -- private, owner-scoped
documents are unaffected (see PRIVACY below). Running this script does not
change that exclusion; it only RECORDS, on each document, exactly why it is
excluded (`applicability=unknown`, etc.) so an admin can find and fix each
one, and gives a document a route back to `review_status=approved` once its
real jurisdiction metadata is supplied (via the admin upload form, or the
`update_jurisdiction_metadata` correction endpoint -- gap 2).

What this script does, one KB document at a time:

  1. Reads whatever jurisdiction-shaped fields already sit in
     `documents.metadata` (there will usually be none -- but if a prior,
     partial fix already set some, e.g. by hand, those are preserved rather
     than clobbered: only ABSENT fields are ever defaulted).
  2. Normalizes them (`kb_jurisdiction.normalize_jurisdiction`, provenance
     `unknown` -- nothing here was asserted by a named human, and nothing
     here was extracted by a regex either; it just wasn't collected at
     ingest time). A document with nothing on record resolves to
     `issuing_level=unknown`, `applicability=unknown`,
     `verification_status=unverified`, `review_status=needs_review`.
     No State, no date, no source URL is EVER guessed from the document's
     text -- this script does not parse the document at all, deliberately
     (a regex guess stored as "backfilled metadata" would be indistinguishable
     from an admin-verified fact six months from now).
  3. Writes those fields onto `documents.metadata` AND onto every one of
     that document's chunks in `embeddings_metadata.metadata` -- directly,
     via `update_one`/`bulk_write`, never through `IndexingPipeline.index_file`.
     Going back through the normal ingestion path would (a) re-embed every
     chunk for a metadata-only change (explicitly out of scope -- "Full
     re-embedding avoid karo"), and (b) get rejected outright by
     `DocumentQualityChecker`'s duplicate-hash check, since the content
     itself is unchanged (the exact failure mode "Metadata-only changes
     existing content-hash dedup se silently discard na hon" warns against).
     Text, chunk IDs, embeddings and `document_versions` history are never
     touched.
  4. Stamps `jurisdiction_schema_version` on write. A document/chunk that
     already carries it is skipped outright on any later run -- this is what
     makes running the script twice a no-op (idempotent) rather than
     re-normalizing (and potentially re-flagging) already-migrated rows.

PRIVACY (Part 45/46): a document with a real `owner_session_id` or
`owner_user_id` (a private, per-user upload -- never a curated KB source) is
excluded from the query entirely, never read or written by this script. Two
independent reasons, both load-bearing:
  * Scope -- Part 1 objective 4's "Existing private user-document
    ownership/isolation preserve karo" means jurisdiction review is a
    KB-only concept.
  * Correctness -- `LegalRetriever`'s filters are ANDed
    (`review_status` alongside the ownership `$or`), so writing
    `review_status=needs_review` onto a private chunk would make a user's
    OWN document invisible to THEMSELVES, not just to shared search. Never
    touching private documents at all is what keeps that failure mode
    structurally impossible, not merely untested.

COVERAGE IMPACT: every KB document that has never been given jurisdiction
metadata (today, that is the entire existing corpus) is ALREADY excluded from
shared retrieval as soon as this phase's code is deployed -- see the gap 1
note above, this is not something `--apply` newly causes. Running `--apply`
does not change what is retrievable; it stamps each excluded document with
WHY (`applicability=unknown`, etc., in `review_reasons`) so an admin has a
queue to work through, and unblocks the per-document correction endpoint
(`update_jurisdiction_metadata` -- gap 2) for it, which is the only way back
to `review_status=approved` without a full re-upload. `--apply` is still
opt-in and prints (never silently applies) a per-run report of how many
documents/chunks were touched, since it is a real, bulk write to
`documents`/`embeddings_metadata.metadata`.

ROLLBACK: this script keeps `previous_metadata` (the exact fields it
overwrote, per document) in its own report/log line, not in Mongo -- reverting
means `$unset`ing the fields listed under `JURISDICTION_FIELDS`
(`kb_jurisdiction.JURISDICTION_FIELDS`) plus `jurisdiction_schema_version` and
`section_overrides`/`section_override_applied` on the affected `documents`
and `embeddings_metadata` rows named in that run's report. Nothing else
(text, embeddings, chunk IDs, `document_versions`) is ever touched, so a
rollback is always metadata-only and always safe.

Usage:
    python scripts/backfill_kb_jurisdiction.py            # dry run (default)
    python scripts/backfill_kb_jurisdiction.py --apply    # writes changes
"""

from __future__ import annotations

import argparse
import asyncio
from typing import Any

import structlog

from app.core.logger import configure_logging
from app.database.mongodb import mongodb
from app.models.collections import EMBEDDINGS_METADATA, UPLOADED_DOCUMENTS
from app.rag.kb_jurisdiction import (
    JURISDICTION_FIELDS,
    JURISDICTION_SCHEMA_VERSION,
    PROVENANCE_UNKNOWN,
    JurisdictionMetadataError,
    document_metadata_fields,
    normalize_jurisdiction,
    propagate_jurisdiction_metadata,
)

log = structlog.get_logger(__name__)

# Curated KB document: no real owner. Matches both an explicitly-`None` value
# and a document indexed before Part 45/46 added these fields at all (absent
# key) -- the same `$in: [None]` convention `MongoVectorStore._mongo_filter`
# already uses for the same reason.
_CURATED_DOCUMENT_FILTER: dict[str, Any] = {
    "owner_session_id": {"$in": [None]},
    "owner_user_id": {"$in": [None]},
    "metadata.jurisdiction_schema_version": {"$exists": False},
}


async def backfill(
    *, apply: bool = False, documents: Any = None, chunks: Any = None,
) -> dict[str, Any]:
    """Runs the backfill. Read-only unless `apply=True`. `documents`/`chunks`
    are optional collection overrides (mirrors `KnowledgeBaseStagingRepository.
    status_counts`'s pattern) so this is unit-testable without a live Mongo.
    """
    documents_collection = documents if documents is not None else mongodb.db[UPLOADED_DOCUMENTS]
    chunks_collection = chunks if chunks is not None else mongodb.db[EMBEDDINGS_METADATA]

    candidates = [doc async for doc in documents_collection.find(_CURATED_DOCUMENT_FILTER)]

    documents_needs_review = 0
    documents_approved = 0
    documents_errors: list[dict[str, str]] = []
    chunks_updated = 0
    report_rows: list[dict[str, Any]] = []

    for document in candidates:
        existing_metadata = document.get("metadata") or {}
        # Preserve anything already on record (objective: "Existing
        # trustworthy metadata preserve karo") -- only fields this document
        # already carries are fed back in; nothing is invented from the
        # document's text or filename.
        raw = {key: existing_metadata[key] for key in JURISDICTION_FIELDS if key in existing_metadata}
        if "section_overrides" in existing_metadata:
            raw["section_overrides"] = existing_metadata["section_overrides"]
        try:
            normalized = normalize_jurisdiction(raw, provenance=PROVENANCE_UNKNOWN)
        except JurisdictionMetadataError as exc:
            # Pre-existing data that fails the phase-1 shape (e.g. a stray
            # State name that doesn't match any code) is reported, not
            # silently coerced or dropped -- an admin decides what it meant.
            documents_errors.append({"document_id": str(document.get("_id")), "errors": "; ".join(exc.errors)})
            continue

        fields = document_metadata_fields(normalized)
        source_document = document.get("filename")
        if normalized.metadata["review_status"] == "needs_review":
            documents_needs_review += 1
        else:
            documents_approved += 1
        report_rows.append(
            {
                "document_id": str(document.get("_id")),
                "source_document": source_document,
                "review_status": normalized.metadata["review_status"],
                "review_reasons": normalized.metadata["review_reasons"],
                "previous_metadata": raw,
            }
        )

        if not apply:
            continue

        # Same write path as the one-shot admin jurisdiction-correction
        # endpoint (`KnowledgeBaseIngestionService.update_jurisdiction_metadata`,
        # gap 2 fix) -- one function owns "how metadata reaches a document and
        # its chunks", so the two paths can never drift apart. Idempotency
        # here comes from `_CURATED_DOCUMENT_FILTER` at the CANDIDATE-SELECTION
        # stage above (a document only appears in `candidates` once, before its
        # `jurisdiction_schema_version` is set), not from a second per-chunk
        # guard -- once this document is stamped, neither it nor its chunks are
        # selected again on a re-run.
        chunks_updated += await propagate_jurisdiction_metadata(
            documents_collection, chunks_collection,
            document_id=document["_id"], source_document=source_document,
            fields=fields, section_overrides=normalized.section_overrides,
        )

    if apply and candidates:
        # Phase 1's allowed cache exception (gap 3 fix): a document just
        # marked `needs_review` must not keep being served from cache for the
        # rest of either cache's TTL. `response_cache` and `retrieval_cache`
        # (`app.cache.semantic_cache.SemanticCache`) now share ONE redis
        # generation counter -- bumping it here invalidates the final-answer
        # cache AND every cached raw retrieval-results entry in the same
        # instant. See `SemanticCache`'s own comment for why one shared
        # counter, not two independent ones.
        from app.cache.response_cache import response_cache

        await response_cache.bump_generation()

    summary = {
        "dry_run": not apply,
        "schema_version": JURISDICTION_SCHEMA_VERSION,
        "documents_scanned": len(candidates),
        "documents_updated": documents_needs_review + documents_approved,
        "documents_marked_needs_review": documents_needs_review,
        "documents_marked_approved": documents_approved,
        "documents_with_errors": len(documents_errors),
        "chunks_updated": chunks_updated,
        "errors": documents_errors[:20],
        "sample_rows": report_rows[:20],
    }
    log.info("kb_jurisdiction_backfill_complete", **{k: v for k, v in summary.items() if k != "sample_rows"})
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply", action="store_true",
        help="Write changes. Without this flag the script only reports counts (default: dry run).",
    )
    args = parser.parse_args()
    configure_logging()

    async def _main() -> None:
        await mongodb.connect()
        result = await backfill(apply=args.apply)
        print(f"dry_run={result['dry_run']}")
        print(f"documents_scanned={result['documents_scanned']}")
        print(f"documents_updated={result['documents_updated']}")
        print(f"  -> needs_review: {result['documents_marked_needs_review']}")
        print(f"  -> approved:     {result['documents_marked_approved']}")
        print(f"chunks_updated={result['chunks_updated']}")
        print(f"documents_with_errors={result['documents_with_errors']}")
        if result["errors"]:
            print("First errors:")
            for row in result["errors"]:
                print(f"  - {row['document_id']}: {row['errors']}")
        if not args.apply:
            print("\nDry run only -- no changes were written. Re-run with --apply to write them.")

    asyncio.run(_main())
